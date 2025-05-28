import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from config.settings import (
    BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TRADING_SYMBOLS, MAX_POSITION_SIZE, MAX_LEVERAGE, POSITION_RATIO, POSITION_SIZE,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K, STOCH_D, STOCH_SLOW, EMA_SHORT, EMA_MEDIUM, EMA_LONG, ATR_PERIOD,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, MAX_DAILY_LOSS, MIN_TRADING_INTERVAL,
    TRADING_ENABLED, TEST_MODE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
    SIGNAL_CONFIRMATION_COUNT, SIGNAL_HISTORY_LIMIT, VOLUME_MA_PERIOD, VOLUME_THRESHOLD
)
from src.data.binance_client import BinanceClient
from src.data.news_collector import NewsCollector
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.signal_generator import SignalGenerator
from src.utils.telegram_bot import TelegramBot
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, symbol=None):
        """Initialize the trading bot"""
        self.symbol = symbol or TRADING_SYMBOLS[0]
        self.binance = BinanceClient(self.symbol)
        self.news_collector = NewsCollector()
        self.technical_analyzer = TechnicalAnalyzer(self.symbol)
        self.telegram = TelegramBot()
        self.current_position = None
        self.daily_pnl = 0
        self.klines_data = pd.DataFrame()
        self.last_news_impact = 0
        self.news_threshold = 0.7  # 뉴스 영향도 임계값
        self.last_position_info = None
        self.last_trade_time = 0
        self.min_trade_interval = 30  # 최소 거래 간격 (초) - 60초에서 30초로 완화
        self.signal_confirmation_count = 0  # 신호 확인 카운트
        self.required_signal_confirmation = 1  # 필요한 신호 확인 횟수 - 2에서 1로 완화
        self.last_signal = 0  # 마지막 신호
        self.signal_history = []  # (signal, score, adx)
        self.signal_history_limit = 5  # 10에서 5로 감소
        self.reversal_confirmation = 0
        # 새로운 개선 사항들
        self.account_balance = 0  # 계좌 잔고
        self.max_drawdown = 0  # 최대 손실률
        self.peak_balance = 0  # 최고 잔고
        self.win_rate = 0  # 승률
        self.total_trades = 0  # 총 거래 수
        self.winning_trades = 0  # 승리 거래 수
        self.trailing_stop_enabled = True  # 트레일링 스탑 활성화
        self.volatility_threshold = 0.02  # 변동성 임계값 (2%)
        # NaN 체크 관련 변수 추가
        self.last_nan_check_time = None  # 마지막 NaN 체크 시간
        self.nan_check_interval = 60  # NaN 체크 간격 (초)
        self.data_accumulation_complete = False  # 데이터 축적 완료 여부
        # 신호 검증 관련 변수 추가
        self.last_signal_warning_time = None  # 마지막 신호 경고 시간
        self.signal_warning_interval = 60  # 신호 경고 간격 (초)
        self.daily_loss = 0
        self.daily_loss_reset_time = time.time()
        self.position_ratio = POSITION_RATIO / 100  # Convert percentage to decimal
        self.stop_loss_percentage = STOP_LOSS_PERCENTAGE
        self.take_profit_percentage = TAKE_PROFIT_PERCENTAGE
        self.max_daily_loss = MAX_DAILY_LOSS
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    async def initialize(self):
        """Initialize all components"""
        try:
            await self.binance.initialize()
            await self.telegram.initialize()
            
            # Send startup notification
            await self.telegram.send_message(
                f"🤖 Trading Bot Started\n"
                f"Symbol: {self.symbol}\n"
                f"Max Position Size: ${POSITION_SIZE}\n"
                f"Max Leverage: {MAX_LEVERAGE}x\n"
                f"Position Ratio: {self.position_ratio * 100}%\n"
                f"Stop Loss: {self.stop_loss_percentage}%\n"
                f"Take Profit: {self.take_profit_percentage}%"
            )
            
            # 초기 과거 데이터 로드 및 전략 수립
            try:
                logger.info("📊 Loading historical data...")
                # 지표 계산에 충분한 데이터 확보 (최소 200개)
                historical_data = await self.binance.get_historical_klines(interval='1m', limit=500)
                logger.info(f"📊 Loaded {len(historical_data)} historical data points")
                
                if len(historical_data) < 50:
                    logger.warning("Insufficient 1m historical data. Trying with 5m interval.")
                    historical_data = await self.binance.get_historical_klines(interval='5m', limit=200)
                    logger.info(f"📊 Loaded {len(historical_data)} 5m historical data points")
                
                self.klines_data = historical_data
                
                # 데이터 무결성 검증
                if self.validate_data_integrity():
                    # 과거 데이터로 지표 계산 및 전략 수립
                    logger.info("📈 Calculating technical indicators...")
                    self.klines_data = self.technical_analyzer.calculate_indicators(self.klines_data)
                    
                    # NaN 값 처리
                    indicator_cols = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'macd', 'macd_signal', 
                                    'macd_diff', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'atr', 'supertrend', 'adx']
                    
                    # NaN 값을 이전 값으로 채우기 (새로운 방식)
                    for col in indicator_cols:
                        if col in self.klines_data.columns:
                            # 앞쪽 NaN을 이전 값으로 채우기
                            self.klines_data[col] = self.klines_data[col].ffill()
                            # 뒤쪽 NaN을 다음 값으로 채우기
                            self.klines_data[col] = self.klines_data[col].bfill()
                    
                    # 데이터 축적 완료 표시 (초기 데이터 로드 시)
                    if len(self.klines_data) >= 100:
                        self.data_accumulation_complete = True
                        logger.info(f"✅ Data accumulation complete with {len(self.klines_data)} data points")
                        logger.info("📊 Initial indicators calculated successfully")
                    
                    await self.telegram.send_message(f"🤖 Trading bot started with {len(self.klines_data)} historical data points loaded and strategy formulated")
                else:
                    await self.telegram.send_message("⚠️ Warning: Data integrity issues detected. Bot will wait for more data.")
                    
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
                await self.telegram.send_message("⚠️ Warning: Failed to load historical data. Bot will start with empty data.")

            if not os.path.exists('.env'):
                logger.warning('No .env file found!')
                await self.telegram.send_message('⚠️ .env 파일이 없습니다. 환경변수를 확인하세요!')
            else:
                load_dotenv()
                required_keys = ['BINANCE_API_KEY','BINANCE_API_SECRET','TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID']
                missing = [k for k in required_keys if not os.getenv(k)]
                if missing:
                    logger.warning(f'Missing env keys: {missing}')
                    await self.telegram.send_message(f'⚠️ 환경변수 누락: {missing}')
        except Exception as e:
            logger.error(f"Error in initialize: {e}")
            await self.telegram.send_error(f"Error in initialize: {e}")

    def validate_data_integrity(self):
        """데이터 무결성 검증"""
        try:
            if len(self.klines_data) < 50:
                logger.warning(f"Insufficient data: only {len(self.klines_data)} rows available")
                return False
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # 필수 컬럼 존재 확인
            for col in required_cols:
                if col not in self.klines_data.columns:
                    logger.warning(f"Missing required column: {col}")
                    return False
            
            # NaN 값 확인
            if self.klines_data[required_cols].isnull().any().any():
                logger.warning("NaN values detected in price data")
                # NaN 값을 이전 값으로 채우기
                self.klines_data[required_cols] = self.klines_data[required_cols].ffill()
                # 여전히 NaN이 있다면 다음 값으로 채우기
                self.klines_data[required_cols] = self.klines_data[required_cols].bfill()
            
            # 가격 데이터 유효성 검증
            for col in ['open', 'high', 'low', 'close']:
                if (self.klines_data[col] <= 0).any():
                    logger.warning(f"Invalid price data detected in {col}")
                    return False
            
            # High >= Low 검증
            if (self.klines_data['high'] < self.klines_data['low']).any():
                logger.warning("Invalid OHLC data: high < low detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in validate_data_integrity: {e}")
            return False

    def should_check_nan_values(self):
        """NaN 값 체크 여부 결정"""
        current_time = datetime.now()
        
        # 데이터 축적이 완료되지 않았으면 체크하지 않음
        if not self.data_accumulation_complete:
            return False
        
        # 마지막 체크 시간이 없거나 간격이 지났으면 체크
        if (self.last_nan_check_time is None or 
            (current_time - self.last_nan_check_time).total_seconds() >= self.nan_check_interval):
            self.last_nan_check_time = current_time
            return True
        
        return False

    def should_log_signal_warning(self):
        """신호 검증 경고 로그 여부 결정"""
        current_time = datetime.now()
        
        # 데이터 축적이 완료되지 않았으면 로그하지 않음
        if not self.data_accumulation_complete:
            return False
        
        # 마지막 경고 시간이 없거나 간격이 지났으면 로그
        if (self.last_signal_warning_time is None or 
            (current_time - self.last_signal_warning_time).total_seconds() >= self.signal_warning_interval):
            self.last_signal_warning_time = current_time
            return True
        
        return False

    async def update_klines(self, kline_data):
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            
            # 데이터 검증
            if not self.validate_kline_data(kline_data):
                logger.warning("Invalid kline data received, skipping update")
                return False
            
            # 'timestamp'이 없으면 생성
            if 'timestamp' not in kline_data:
                if 'open_time' in kline_data:
                    kline_data['timestamp'] = kline_data['open_time']
                else:
                    kline_data['timestamp'] = int(datetime.now().timestamp())
            
            new_data = pd.DataFrame([kline_data])
            for col in required_cols:
                if col not in new_data.columns:
                    new_data[col] = np.nan
            
            new_data = new_data[required_cols]
            
            # set_index 호출 전, 이미 인덱스인지 확인
            if 'timestamp' in new_data.columns and new_data.index.name != 'timestamp':
                new_data.set_index('timestamp', inplace=True)
            
            # 새로운 데이터 추가
            self.klines_data = pd.concat([self.klines_data, new_data])
            # 중복된 인덱스 제거 (가장 최근 데이터 유지)
            self.klines_data = self.klines_data[~self.klines_data.index.duplicated(keep='last')]
            self.klines_data = self.klines_data.tail(500)  # 최근 500개 데이터만 유지
            
            # 데이터 무결성 검증
            if not self.validate_data_integrity():
                logger.warning("Data integrity check failed after update")
                return False
            
            # 지표 재계산
            try:
                self.klines_data = self.technical_analyzer.calculate_indicators(self.klines_data)
                
                # NaN 값 처리
                indicator_cols = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'macd', 'macd_signal', 
                                'macd_diff', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'atr', 'supertrend', 'adx']
                
                # NaN 값을 이전 값으로 채우기 (새로운 방식)
                for col in indicator_cols:
                    if col in self.klines_data.columns:
                        # 앞쪽 NaN을 이전 값으로 채우기
                        self.klines_data[col] = self.klines_data[col].ffill()
                        # 뒤쪽 NaN을 다음 값으로 채우기
                        self.klines_data[col] = self.klines_data[col].bfill()
                
                # 최근 데이터의 NaN 값 확인 및 처리
                recent_data = self.klines_data.tail(10).copy()  # 명시적 복사본 생성
                nan_cols = [col for col in indicator_cols if recent_data[col].isnull().any()]
                if nan_cols:
                    logger.warning(f"NaN values detected in columns: {nan_cols}")
                    # NaN이 있는 컬럼에 대해 0으로 채우기
                    for col in nan_cols:
                        recent_data.loc[:, col] = recent_data[col].fillna(0)
                    # 원본 데이터프레임 업데이트
                    self.klines_data.update(recent_data)
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                return False
            
            # 데이터 축적 완료 표시
            if not self.data_accumulation_complete and len(self.klines_data) >= 100:
                self.data_accumulation_complete = True
                logger.info("✅ Real-time data accumulation complete. Trading signals will now be generated.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in update_klines: {e}")
            return False

    def validate_kline_data(self, kline_data):
        """개별 kline 데이터 검증"""
        try:
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            
            for field in required_fields:
                if field not in kline_data:
                    logger.warning(f"Missing field in kline data: {field}")
                    return False
                
                # 숫자 타입 확인
                try:
                    value = float(kline_data[field])
                    if value <= 0 and field != 'volume':  # volume은 0일 수 있음
                        logger.warning(f"Invalid value for {field}: {value}")
                        return False
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric value for {field}: {kline_data[field]}")
                    return False
            
            # OHLC 관계 검증
            try:
                o, h, l, c = float(kline_data['open']), float(kline_data['high']), float(kline_data['low']), float(kline_data['close'])
                if h < l or h < max(o, c) or l > min(o, c):
                    logger.warning(f"Invalid OHLC relationship: O={o}, H={h}, L={l}, C={c}")
                    return False
            except:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in validate_kline_data: {e}")
            return False

    async def handle_news_impact(self, combined_impact, detailed_impacts):
        """Handle news impact on trading decisions"""
        self.last_news_impact = combined_impact
        
        # 뉴스 영향도가 임계값을 넘으면 알림 전송
        if abs(combined_impact) > self.news_threshold:
            message = (
                f"📰 <b>News Impact Alert</b>\n\n"
                f"Combined Impact: {combined_impact:.2f}\n"
                f"Crypto Impact: {detailed_impacts['crypto']:.2f}\n"
                f"NASDAQ Impact: {detailed_impacts['nasdaq']:.2f}"
            )
            await self.telegram.send_message(f"[{self.symbol}] {message}")

            # 뉴스 영향도에 따른 포지션 조정
            if self.current_position:
                current_side = 'LONG' if float(self.current_position['positionAmt']) > 0 else 'SHORT'
                
                # 뉴스 영향도와 현재 포지션이 반대 방향이면 청산 후 반대 포지션 진입
                if (combined_impact > 0 and current_side == 'SHORT') or \
                   (combined_impact < 0 and current_side == 'LONG'):
                    await self.close_position("News impact reversal")
                    # 반대 포지션 진입
                    current_price = self.klines_data['close'].iloc[-1]
                    if combined_impact > 0:
                        await self.execute_trade(1, current_price, "Strong positive news", score=5)
                    else:
                        await self.execute_trade(-1, current_price, "Strong negative news", score=5)
            else:
                # 새로운 포지션 진입
                current_price = self.klines_data['close'].iloc[-1]
                if combined_impact > self.news_threshold:
                    await self.execute_trade(1, current_price, "Strong positive news", score=5)
                elif combined_impact < -self.news_threshold:
                    await self.execute_trade(-1, current_price, "Strong negative news", score=5)

    async def check_risk_limits(self):
        """Check if current position is within risk limits"""
        try:
            if not self.current_position:
                return True

            await self.update_account_info()
            
            # 최대 손실률 체크 (20% 이상 손실 시 거래 중단)
            if self.max_drawdown > 0.2:
                await self.close_position("Maximum drawdown exceeded (20%)")
                await self.telegram.send_message("🚨 거래 중단: 최대 손실률 20% 초과")
                return False

            position_size = float(self.current_position.get('positionAmt', 0))
            if abs(position_size) > MAX_POSITION_SIZE:
                await self.close_position("Position size limit exceeded")
                return False

            if self.daily_pnl < -MAX_DAILY_LOSS:
                await self.close_position("Daily loss limit exceeded")
                return False

            return True
        except Exception as e:
            logger.error(f"Error in check_risk_limits: {e}")
            return True

    async def calculate_position_size(self, current_price):
        """Calculate position size based on account balance and risk management"""
        try:
            # Get account balance
            account_info = await self.binance.client.futures_account()
            if not account_info:
                self.logger.error("Failed to get account info")
                return 0
            
            # Get USDT balance
            usdt_balance = float(account_info['totalWalletBalance'])
            
            # Calculate position size based on balance and position ratio
            position_size = usdt_balance * self.position_ratio
            
            # Ensure position size doesn't exceed maximum
            position_size = min(position_size, MAX_POSITION_SIZE)
            
            # Calculate quantity based on current price
            quantity = position_size / current_price
            
            # Round quantity to appropriate decimal places
            if self.symbol == 'BTCUSDT':
                quantity = round(quantity, 3)  # BTC has 3 decimal places
            elif self.symbol == 'ETHUSDT':
                quantity = round(quantity, 3)  # ETH has 3 decimal places
            else:
                quantity = round(quantity, 2)  # Default to 2 decimal places
            
            self.logger.info(f"Calculated position size: {quantity} {self.symbol} (${position_size})")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_market_volatility(self):
        """시장 변동성 계산"""
        try:
            if len(self.klines_data) < 20:
                return 0.02  # 기본 변동성
            
            # ATR 기반 변동성 계산
            atr = self.klines_data['atr'].tail(20).mean()
            current_price = self.klines_data['close'].iloc[-1]
            volatility = atr / current_price
            
            return volatility
        except Exception as e:
            logger.error(f"Error in calculate_market_volatility: {e}")
            return 0.02

    async def update_account_info(self):
        """계좌 정보 업데이트"""
        try:
            # Binance API를 통해 계좌 정보 조회
            account_info = await self.binance.client.futures_account()
            self.account_balance = float(account_info['totalWalletBalance'])
            
            # 최고 잔고 업데이트
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
            
            # 최대 손실률 계산
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
        except Exception as e:
            logger.error(f"Error in update_account_info: {e}")

    def calculate_trailing_stop(self, entry_price, current_price, side, trail_percent=0.5):
        """트레일링 스탑 계산"""
        try:
            trail_distance = current_price * (trail_percent / 100)
            
            if side == 'BUY':
                # 롱 포지션: 현재가에서 트레일 거리만큼 아래
                trailing_stop = current_price - trail_distance
                return max(trailing_stop, entry_price * 0.995)  # 최소 0.5% 손절
            else:
                # 숏 포지션: 현재가에서 트레일 거리만큼 위
                trailing_stop = current_price + trail_distance
                return min(trailing_stop, entry_price * 1.005)  # 최소 0.5% 손절
                
        except Exception as e:
            logger.error(f"Error in calculate_trailing_stop: {e}")
            return entry_price * 0.995 if side == 'BUY' else entry_price * 1.005

    async def execute_trade(self, signal, current_price, market_condition, score=0):
        """Execute trade based on signal and market conditions"""
        try:
            # Calculate position size
            quantity = await self.calculate_position_size(current_price)
            if quantity <= 0:
                self.logger.warning("Invalid position size calculated")
                return False
            
            # Set leverage (only if no position exists)
            await self.binance.set_leverage(MAX_LEVERAGE)
            
            # Determine side
            side = 'BUY' if signal > 0 else 'SELL'
            
            # Calculate stop loss and take profit prices
            stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                self.klines_data, current_price, side
            )
            
            # Place order
            order = await self.binance.place_order(
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            if order:
                # Update position info
                self.current_position = {
                    'side': side,
                    'entry_price': current_price,
                    'entry': current_price,
                    'quantity': quantity,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'size': quantity * current_price,
                    'positionAmt': quantity if side == 'BUY' else -quantity
                }
                
                # Update trade statistics
                self.total_trades += 1
                
                # Send notification
                await self.telegram.send_message(
                    f"🔄 Trade Executed\n"
                    f"Symbol: {self.symbol}\n"
                    f"Side: {side}\n"
                    f"Entry Price: ${current_price:.2f}\n"
                    f"Quantity: {quantity}\n"
                    f"Position Size: ${quantity * current_price:.2f}\n"
                    f"Leverage: {MAX_LEVERAGE}x\n"
                    f"Stop Loss: ${stop_loss_price:.2f}\n"
                    f"Take Profit: ${take_profit_price:.2f}\n"
                    f"Market Condition: {market_condition}\n"
                    f"Signal Score: {score}"
                )
                
                self.logger.info(f"Trade executed successfully: {side} {quantity} {self.symbol}")
                return True
            else:
                self.logger.error("Failed to place order")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            await self.telegram.send_error(f"Error executing trade: {str(e)}")
            return False

    def calculate_dynamic_leverage(self, score, adx):
        """Calculate dynamic leverage based on market conditions"""
        try:
            # Example logic: Adjust leverage based on ADX and score
            if adx > 25 and score > 0.8:
                return min(50, MAX_LEVERAGE * 1.5)  # Increase leverage in strong trends, max 50x
            elif adx < 20 or score < 0.5:
                return max(5, MAX_LEVERAGE * 0.5)  # Decrease leverage in weak trends
            return min(50, MAX_LEVERAGE)  # Default leverage, max 50x
        except Exception as e:
            logger.error(f"Error in calculate_dynamic_leverage: {e}")
            return min(50, MAX_LEVERAGE)  # Fallback to default leverage, max 50x

    def calculate_stop_loss_take_profit(self, df, entry_price, side, leverage=MAX_LEVERAGE, lookback=10, min_pct=0.3):
        """
        Calculate dynamic stop loss and take profit based on recent price action.
        Enforce a minimum distance (min_pct, now 0.3%) from entry price for 30x leverage.
        """
        if len(df) < lookback:
            lookback = len(df)
        if lookback < 2:
            atr = entry_price * 0.005  # fallback
        else:
            atr = df['atr'].tail(lookback).mean()
        # Adjust stop loss and take profit based on leverage
        atr_multiplier = 1.5 / leverage
        if side == 'BUY':
            stop_loss = entry_price - atr * atr_multiplier
            take_profit = entry_price + atr * 2.5 * atr_multiplier
        else:
            stop_loss = entry_price + atr * atr_multiplier
            take_profit = entry_price - atr * 2.5 * atr_multiplier
        # 최소폭 강제 (0.3%)
        min_dist = entry_price * min_pct / 100
        if side == 'BUY':
            stop_loss = min(stop_loss, entry_price - min_dist)
            take_profit = max(take_profit, entry_price + min_dist * 2)
        else:
            stop_loss = max(stop_loss, entry_price + min_dist)
            take_profit = min(take_profit, entry_price - min_dist * 2)
        return stop_loss, take_profit

    async def close_position(self, reason):
        """Close current position"""
        try:
            if self.current_position:
                side = 'SELL' if float(self.current_position['positionAmt']) > 0 else 'BUY'
                await self.binance.place_order(side, abs(float(self.current_position['positionAmt'])))
                await self.telegram.send_message(f"[{self.symbol}] Position closed: {reason}")
                self.current_position = None
            await self.update_position()
        except Exception as e:
            logger.error(f"Error in close_position: {e}")
            await self.telegram.send_error(f"Error in close_position: {e}")

    async def update_position(self):
        """Update current position information"""
        try:
            position = await self.binance.get_position()
            if position and float(position['positionAmt']) != 0:
                side = 'BUY' if float(position['positionAmt']) > 0 else 'SELL'
                position['side'] = side
                position['size'] = abs(float(position['positionAmt']))  # 포지션 크기 추가
                
                if self.current_position:
                    if 'stop_loss' in self.current_position:
                        position['stop_loss'] = self.current_position['stop_loss']
                    if 'take_profit' in self.current_position:
                        position['take_profit'] = self.current_position['take_profit']
                    if 'entry' in self.current_position:
                        position['entry'] = self.current_position['entry']
                    if 'trailing_stop' in self.current_position:
                        position['trailing_stop'] = self.current_position['trailing_stop']
                if 'entry' not in position and 'entryPrice' in position:
                    position['entry'] = float(position['entryPrice'])
                if 'stop_loss' not in position or 'take_profit' not in position:
                    stop_loss, take_profit = self.technical_analyzer.calculate_stop_loss_take_profit(
                        self.klines_data, position['entry'], side)
                    position['stop_loss'] = stop_loss
                    position['take_profit'] = take_profit
                # trailing_stop이 없으면 stop_loss로 초기화
                if 'trailing_stop' not in position:
                    position['trailing_stop'] = position['stop_loss']
                # 포지션 정보가 바뀌었을 때만 메시지 전송
                pos_info = (position['side'], position['positionAmt'], position['entry'])
                if self.last_position_info != pos_info:
                    await self.telegram.send_position_update({**position, 'symbol': self.symbol})
                    self.last_position_info = pos_info
                self.current_position = position
            else:
                self.current_position = None
                self.last_position_info = None
        except Exception as e:
            logger.error(f"Error in update_position: {e}")
            await self.telegram.send_error(f"Error in update_position: {e}")

    async def monitor_position(self, last_price):
        """Monitor open position for dynamic stop loss/take profit, robust close handling and real close"""
        try:
            if not self.current_position:
                return
                
            # 필수 필드 확인 및 초기화
            required_fields = ['side', 'stop_loss', 'take_profit', 'entry', 'size']
            for field in required_fields:
                if field not in self.current_position:
                    logger.warning(f"Missing {field} in position, initializing...")
                    if field == 'trailing_stop':
                        self.current_position[field] = self.current_position['stop_loss']
                    elif field == 'size':
                        position = await self.binance.get_position()
                        if position:
                            self.current_position[field] = abs(float(position['positionAmt']))
                        else:
                            self.current_position[field] = 0
                    else:
                        await self.update_position()
                        if not self.current_position:
                            return
                        break
                    
            side = self.current_position['side']
            stop_loss = float(self.current_position['stop_loss'])
            take_profit = float(self.current_position['take_profit'])
            entry_price = float(self.current_position['entry'])
            position_size = float(self.current_position['size'])
            
            # 트레일링 스탑 초기화 확인
            if 'trailing_stop' not in self.current_position:
                self.current_position['trailing_stop'] = stop_loss
            trailing_stop = float(self.current_position['trailing_stop'])
            
            # 트레일링 스탑 업데이트
            if self.trailing_stop_enabled:
                new_trailing_stop = self.calculate_trailing_stop(entry_price, last_price, side)
                
                if side == 'BUY':
                    # 롱 포지션: 트레일링 스탑이 상승할 때만 업데이트
                    if new_trailing_stop > trailing_stop:
                        self.current_position['trailing_stop'] = new_trailing_stop
                        trailing_stop = new_trailing_stop
                        # 트레일링 스탑 업데이트는 로그로만 기록
                        logger.info(f"[{self.symbol}] Updated trailing stop for LONG position: {trailing_stop:.2f}")
                else:
                    # 숏 포지션: 트레일링 스탑이 하락할 때만 업데이트
                    if new_trailing_stop < trailing_stop:
                        self.current_position['trailing_stop'] = new_trailing_stop
                        trailing_stop = new_trailing_stop
                        # 트레일링 스탑 업데이트는 로그로만 기록
                        logger.info(f"[{self.symbol}] Updated trailing stop for SHORT position: {trailing_stop:.2f}")
            
            # 항상 실시간 포지션 수량 조회
            position = await self.binance.get_position()
            amt = float(position['positionAmt']) if position else 0
            if abs(amt) < 1e-4:  # 최소 단위 미만이면 포지션 없음
                self.current_position = None
                return
                
            qty = abs(amt)
            closed = False
            
            # 손익 계산
            if side == 'BUY':
                pnl = (last_price - entry_price) / entry_price
                # 트레일링 스탑 또는 익절가 도달 시 청산
                if last_price <= trailing_stop or last_price >= take_profit:
                    try:
                        await self.binance.place_order('SELL', qty, order_type='MARKET', reduce_only=True)
                        # 청산 메시지는 텔레그램으로 전송
                        close_reason = '트레일링 스탑' if last_price <= trailing_stop else '익절'
                        await self.telegram.send_message(
                            f"[{self.symbol}] 🔔 포지션 청산 알림\n"
                            f"방향: 롱\n"
                            f"청산가: {last_price:.2f}\n"
                            f"수익률: {pnl:.2%}\n"
                            f"사유: {close_reason}"
                        )
                        closed = True
                        if pnl > 0:
                            self.winning_trades += 1
                    except Exception as e:
                        logger.error(f"Error closing LONG position: {e}")
                        # 청산 실패 시에만 에러 메시지 전송
                        await self.telegram.send_error(f"[{self.symbol}] ❌ 롱 포지션 청산 실패: {e}")
                        
            elif side == 'SELL':
                pnl = (entry_price - last_price) / entry_price
                # 트레일링 스탑 또는 익절가 도달 시 청산
                if last_price >= trailing_stop or last_price <= take_profit:
                    try:
                        await self.binance.place_order('BUY', qty, order_type='MARKET', reduce_only=True)
                        # 청산 메시지는 텔레그램으로 전송
                        close_reason = '트레일링 스탑' if last_price >= trailing_stop else '익절'
                        await self.telegram.send_message(
                            f"[{self.symbol}] 🔔 포지션 청산 알림\n"
                            f"방향: 숏\n"
                            f"청산가: {last_price:.2f}\n"
                            f"수익률: {pnl:.2%}\n"
                            f"사유: {close_reason}"
                        )
                        closed = True
                        if pnl > 0:
                            self.winning_trades += 1
                    except Exception as e:
                        logger.error(f"Error closing SHORT position: {e}")
                        # 청산 실패 시에만 에러 메시지 전송
                        await self.telegram.send_error(f"[{self.symbol}] ❌ 숏 포지션 청산 실패: {e}")
            
            # 승률 계산
            if self.total_trades > 0:
                self.win_rate = self.winning_trades / self.total_trades
                
            if closed:
                # 청산 확인 및 포지션 초기화
                for _ in range(10):  # 최대 10초 대기
                    await asyncio.sleep(1)
                    position = await self.binance.get_position()
                    amt = float(position['positionAmt']) if position else 0
                    if abs(amt) < 1e-4:
                        self.current_position = None
                        break
                        
        except Exception as e:
            logger.error(f"Error in monitor_position: {e}")
            # 실제 에러 발생 시에만 텔레그램으로 전송
            if "position" in str(e).lower() or "order" in str(e).lower():
                await self.telegram.send_error(f"[{self.symbol}] ⚠️ 포지션 모니터링 에러: {e}")
            # 에러 발생 시 포지션 정보 업데이트 시도
            try:
                await self.update_position()
            except:
                pass

    async def send_performance_report(self):
        """성과 보고서 전송"""
        try:
            await self.update_account_info()
            report = (
                f"📊 <b>Trading Performance Report</b>\n\n"
                f"💰 Account Balance: ${self.account_balance:.2f}\n"
                f"📈 Peak Balance: ${self.peak_balance:.2f}\n"
                f"📉 Max Drawdown: {self.max_drawdown:.2%}\n"
                f"🎯 Win Rate: {self.win_rate:.2%}\n"
                f"📊 Total Trades: {self.total_trades}\n"
                f"✅ Winning Trades: {self.winning_trades}\n"
                f"❌ Losing Trades: {self.total_trades - self.winning_trades}"
            )
            await self.telegram.send_message(f"[{self.symbol}] {report}")
        except Exception as e:
            logger.error(f"Error in send_performance_report: {e}")

    async def setup(self):
        """Initialize all components and setup the trading environment"""
        try:
            # 1. Initialize components
            await self.initialize()
            
            # 2. Start news monitoring
            asyncio.create_task(
                self.news_collector.monitor_news(self.handle_news_impact)
            )
            
            # 3. Start periodic performance report (every hour)
            async def periodic_report():
                while True:
                    await asyncio.sleep(3600)  # 1 hour
                    await self.send_performance_report()
            
            asyncio.create_task(periodic_report())
            
            # 4. Start klines streaming
            async def handle_kline(kline):
                try:
                    has_enough_data = await self.update_klines(kline)
                    if not has_enough_data:
                        return
                    
                    # 데이터가 100개 이상 쌓였을 때만 지표 컬럼 체크 및 신호 생성
                    if len(self.klines_data) < 100:
                        if not self.data_accumulation_complete:
                            logger.info(f"[{self.symbol}] 📊 Waiting for more data: {len(self.klines_data)}/100")
                        return
                    
                    # 데이터 무결성 재검증
                    if not self.validate_data_integrity():
                        logger.warning(f"[{self.symbol}] Data integrity check failed, skipping signal generation")
                        return
                    
                    # 실시간 데이터로 지표 업데이트
                    try:
                        self.klines_data = self.technical_analyzer.calculate_indicators(self.klines_data)
                    except Exception as e:
                        logger.error(f"Error calculating indicators: {e}")
                        return
                    
                    # 지표 컬럼 존재 확인
                    indicator_cols = ['ema_short','ema_medium','ema_long','rsi','macd','macd_signal','macd_diff','bb_high','bb_low','stoch_k','stoch_d','atr','supertrend','adx']
                    missing_cols = [col for col in indicator_cols if col not in self.klines_data.columns]
                    if missing_cols:
                        logger.warning(f"[{self.symbol}] Missing indicator columns: {missing_cols}, skipping signal generation.")
                        return
                    
                    # NaN 값 확인 및 처리
                    if self.should_check_nan_values():
                        nan_cols = [col for col in indicator_cols if self.klines_data[col].isnull().any()]
                        if nan_cols:
                            logger.warning(f"[{self.symbol}] NaN detected in indicators: {nan_cols}")
                            recent_data = self.klines_data.tail(10)
                            if recent_data[indicator_cols].isnull().any().any():
                                logger.warning(f"[{self.symbol}] NaN in recent indicator data, skipping signal generation.")
                                return
                    
                    # 포지션 모니터링
                    await self.monitor_position(kline['close'])
                    await self.update_position()
                    
                except Exception as e:
                    logger.error(f"Error in handle_kline: {e}")
                    await self.telegram.send_error(f"Error in handle_kline: {e}")
            
            # Start WebSocket streaming
            await self.binance.stream_klines(handle_kline)
            
        except Exception as e:
            logger.error(f"Error in setup: {e}")
            await self.telegram.send_error(f"Error in setup: {e}")
            raise  # Re-raise the exception to be caught by the run method

    async def update_market_data(self):
        """Update market data and indicators"""
        try:
            # Get latest kline data
            klines = await self.binance.get_historical_klines(interval='1m', limit=100)
            if not klines.empty:
                # Update klines data
                self.klines_data = klines
                
                # Calculate indicators
                self.klines_data = self.technical_analyzer.calculate_indicators(self.klines_data)
                
                # Validate data integrity
                if not self.validate_data_integrity():
                    logger.warning(f"[{self.symbol}] Data integrity check failed after market data update")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Error in update_market_data: {e}")
            await self.telegram.send_error(f"Error in update_market_data: {e}")
            return False

    async def run(self):
        """Main trading loop with optimized API usage"""
        self.logger.info("Starting trading bot...")
        
        # 초기 데이터 로드
        await self.load_initial_data()
        
        # WebSocket 스트리밍 시작
        await self.binance.stream_klines(self.handle_kline_data)

    async def handle_kline_data(self, kline_data):
        """Handle incoming kline data with safety mechanisms and API optimization"""
        try:
            # 연결 상태 확인
            if not self.binance.is_connected:
                self.logger.warning("WebSocket not connected, skipping data processing")
                return
            
            # 데이터 유효성 검사
            current_time = time.time()
            data_timestamp = kline_data['timestamp'].timestamp()
            
            # 1분 이상 오래된 데이터 무시
            if current_time - data_timestamp > 60:
                self.logger.warning(f"Ignoring outdated data: {current_time - data_timestamp:.1f} seconds old")
                return
            
            # Rate limit 상태 모니터링
            rate_status = self.binance.get_rate_limit_status()
            if rate_status['requests_per_minute'] > rate_status['max_requests_per_minute'] * 0.8:
                self.logger.warning(f"API usage high: {rate_status['requests_per_minute']}/{rate_status['max_requests_per_minute']} requests/min")
            
            # 현재 가격 로깅
            current_price = kline_data['close']
            self.logger.info(f"Current price: {current_price}")
            
            # 레버리지 업데이트 (필요시)
            new_leverage = await self.binance.update_leverage_if_needed()
            if new_leverage:
                self.logger.info(f"Leverage updated to {new_leverage}x")
            
            # 기존 포지션 확인 및 분석 (캐시된 데이터 사용)
            position = await self.binance.get_position()
            
            if position and abs(float(position.get('positionAmt', 0))) > 0:
                # 포지션이 있는 경우 - 분석 수행
                entry_price = float(position.get('entryPrice', 0))
                position_amt = float(position.get('positionAmt', 0))
                is_long = position_amt > 0
                
                # 수익률 계산
                if is_long:
                    pnl_percentage = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_percentage = (entry_price - current_price) / entry_price * 100
                
                self.logger.info(f"Position PnL: {pnl_percentage:.2f}%")
                
                # 손절/익절 로직
                if pnl_percentage < -5:  # 5% 손실
                    self.logger.warning("Large loss detected, closing position")
                    await self.close_position()
                    return
                elif pnl_percentage > 10:  # 10% 수익
                    self.logger.info("Large profit detected, closing position")
                    await self.close_position()
                    return
            
            # 기술적 분석 수행
            self.update_klines(kline_data)
            technical_analysis = self.technical_analyzer.analyze(self.klines_data)
            
            # 뉴스 감정 분석 (캐시된 결과 사용)
            sentiment_score = await self.news_collector.get_sentiment_score()
            
            # 신호 생성
            signal = self.signal_generator.generate_signal(
                technical_analysis, 
                sentiment_score, 
                current_price
            )
            
            # 연결 불안정 시 더 높은 신뢰도 요구
            min_confidence = 2.5 if self.binance.reconnect_attempts > 0 else 2.0
            
            if signal and signal['score'] >= min_confidence:
                # WebSocket 연결 상태 재확인
                if not self.binance.is_connected:
                    self.logger.warning("WebSocket disconnected during signal processing, skipping trade")
                    return
                
                await self.execute_trade(signal, current_price)
            else:
                if signal:
                    self.logger.info(f"Signal confidence too low: {signal['score']:.2f} < {min_confidence}")
                
            # Rate limit 상태 주기적 로깅
            if hasattr(self, 'last_rate_log_time'):
                if current_time - self.last_rate_log_time > 300:  # 5분마다
                    self.logger.info(f"API Status: {rate_status}")
                    self.last_rate_log_time = current_time
            else:
                self.last_rate_log_time = current_time
                
        except Exception as e:
            self.logger.error(f"Error handling kline data: {e}")

    async def close_position(self):
        """Close current position with optimized API calls"""
        try:
            position = await self.binance.get_position()
            if not position or abs(float(position.get('positionAmt', 0))) == 0:
                self.logger.info("No position to close")
                return
            
            position_amt = float(position.get('positionAmt', 0))
            side = 'SELL' if position_amt > 0 else 'BUY'
            quantity = abs(position_amt)
            
            # 포지션 청산
            await self.binance.place_order(
                side=side,
                quantity=quantity,
                order_type='MARKET',
                reduce_only=True
            )
            
            self.logger.info(f"Position closed: {side} {quantity}")
            
            # 텔레그램 알림
            await self.telegram.send_message(f"🔴 Position Closed\nSide: {side}\nQuantity: {quantity}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def execute_trade(self, signal, current_price):
        """Execute trade with API optimization"""
        try:
            # 계정 정보 가져오기 (캐시된 데이터 사용)
            account_info = await self.binance.get_account_info()
            available_balance = float(account_info['availableBalance'])
            
            # 포지션 크기 계산
            risk_amount = available_balance * self.risk_per_trade
            
            # 심볼 정보 가져오기 (캐시된 데이터 사용)
            symbol_info = await self.binance.get_symbol_info()
            if not symbol_info:
                self.logger.error("Could not get symbol information")
                return
            
            # 최소 수량 확인
            min_qty = float(symbol_info['filters'][1]['minQty'])
            quantity = max(risk_amount / current_price, min_qty)
            
            # 수량 정밀도 조정
            step_size = float(symbol_info['filters'][1]['stepSize'])
            quantity = round(quantity / step_size) * step_size
            
            # 주문 실행
            side = signal['action'].upper()
            order = await self.binance.place_order(
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            self.logger.info(f"Trade executed: {side} {quantity} at {current_price}")
            
            # 텔레그램 알림
            await self.telegram.send_message(
                f"🟢 Trade Executed\n"
                f"Side: {side}\n"
                f"Quantity: {quantity}\n"
                f"Price: {current_price}\n"
                f"Signal Score: {signal['score']:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            await self.telegram.send_message(f"❌ Trade execution failed: {e}")

    async def monitor_api_usage(self):
        """API 사용량 모니터링 (백그라운드 태스크)"""
        while True:
            try:
                rate_status = self.binance.get_rate_limit_status()
                
                # 사용량이 80% 이상이면 경고
                if rate_status['requests_per_minute'] > rate_status['max_requests_per_minute'] * 0.8:
                    self.logger.warning(f"High API usage: {rate_status}")
                    await self.telegram.send_message(f"⚠️ High API Usage: {rate_status['requests_per_minute']}/{rate_status['max_requests_per_minute']} req/min")
                
                # 1분마다 체크
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error monitoring API usage: {e}")
                await asyncio.sleep(60)

async def main():
    """Main function with API monitoring"""
    bot = None
    try:
        # 봇 초기화
        bot = TradingBot()
        await bot.initialize()
        
        # API 모니터링 태스크 시작
        monitor_task = asyncio.create_task(bot.monitor_api_usage())
        
        # 메인 봇 실행
        await bot.run()
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if bot and hasattr(bot, 'telegram_bot'):
            await bot.telegram_bot.send_message(f"❌ Fatal Error: {e}")
    finally:
        if bot:
            await bot.cleanup()
        if 'monitor_task' in locals():
            monitor_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 