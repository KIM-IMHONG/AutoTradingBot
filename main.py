import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from src.data.binance_client import BinanceClient
from src.data.news_collector import NewsCollector
from src.analysis.technical import TechnicalAnalyzer
from src.utils.telegram_bot import TelegramBot
from config.settings import (
    TRADING_SYMBOL, POSITION_SIZE, STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_PERCENTAGE, MAX_POSITION_SIZE, MAX_DAILY_LOSS, LEVERAGE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, symbol):
        self.symbol = symbol
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
        self.last_trade_time = None  # 마지막 거래 시간
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

    async def initialize(self):
        """Initialize all components"""
        try:
            await self.binance.initialize()
            await self.telegram.initialize()
            
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
                self.klines_data[required_cols] = self.klines_data[required_cols].fillna(method='ffill')
                # 여전히 NaN이 있다면 다음 값으로 채우기
                self.klines_data[required_cols] = self.klines_data[required_cols].fillna(method='bfill')
            
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
                    if combined_impact > 0:
                        await self.execute_trade(1, self.klines_data['close'].iloc[-1], "Strong positive news", reverse=True)
                    else:
                        await self.execute_trade(-1, self.klines_data['close'].iloc[-1], "Strong negative news", reverse=True)
            else:
                # 새로운 포지션 진입
                if combined_impact > self.news_threshold:
                    await self.execute_trade(1, self.klines_data['close'].iloc[-1], "Strong positive news")
                elif combined_impact < -self.news_threshold:
                    await self.execute_trade(-1, self.klines_data['close'].iloc[-1], "Strong negative news")

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

    def calculate_dynamic_position_size(self, signal_strength, volatility, account_balance):
        """동적 포지션 사이즈 계산 - 신호 강도와 변동성에 따라 조정"""
        try:
            # 기본 포지션 사이즈 (계좌 잔고의 1-5%)
            base_size = account_balance * 0.02  # 2% 기본
            
            # 신호 강도에 따른 조정 (0.5 ~ 1.5배)
            signal_multiplier = max(0.5, min(1.5, signal_strength))
            
            # 변동성에 따른 조정 (높은 변동성일 때 포지션 축소)
            volatility_multiplier = max(0.3, min(1.0, 1 - volatility))
            
            # 최종 포지션 사이즈
            position_size = base_size * signal_multiplier * volatility_multiplier
            
            # 최대 포지션 사이즈 제한 (계좌 잔고의 10%)
            max_size = account_balance * 0.1
            
            # 최소 주문 수량 보장 (0.001 BTC)
            min_size = 0.01 * self.klines_data['close'].iloc[-1]  # BTC 최소 수량을 USD로 변환
            
            # 최종 사이즈 계산 및 최소/최대 제한 적용
            final_size = max(min_size, min(position_size, max_size))
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error in calculate_dynamic_position_size: {e}")
            # 에러 발생 시 최소 주문 수량 반환
            return 0.01 * self.klines_data['close'].iloc[-1]

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

    async def execute_trade(self, signal, price, reason="", reverse=False, score=None, adx=None):
        """Execute trade based on signal and set dynamic stop loss/take profit"""
        try:
            if not await self.check_risk_limits():
                logger.warning("🚫 Trade blocked by risk limits")
                return
            current_time = datetime.now()
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.min_trade_interval:
                remaining_time = self.min_trade_interval - (current_time - self.last_trade_time).total_seconds()
                logger.info(f'⏰ Trade blocked by min_trade_interval. Remaining: {remaining_time:.0f}s')
                return

            # 계좌 정보 업데이트
            await self.update_account_info()
            
            # 시장 변동성 계산
            volatility = self.calculate_market_volatility()
            
            # 동적 포지션 사이즈 계산 (BTC 단위로 변환)
            dynamic_position_size = self.calculate_dynamic_position_size(score or 0.5, volatility, self.account_balance)
            dynamic_position_size = dynamic_position_size / price  # USD를 BTC로 변환
            
            # 최소 주문 수량 보장 및 정밀도 조정
            dynamic_position_size = max(0.01, round(dynamic_position_size, 3))  # 최소 0.001 BTC, 3자리까지 반올림
            
            # 신호/점수/ADX 로그
            log_msg = f"Signal: {signal}, Score: {score}, ADX: {adx}, Volatility: {volatility:.4f}, Position Size: {dynamic_position_size:.3f} BTC, Reason: {reason}"
            logger.info(log_msg)
            await self.telegram.send_message(f"[{self.symbol}] {log_msg}")

            # Dynamic leverage adjustment based on market conditions
            leverage = self.calculate_dynamic_leverage(score, adx)
            await self.binance.set_leverage(leverage)

            # 신호 확인 로직 (조건 완화)
            if not reverse:  # 리버스가 아닌 경우에만 신호 확인
                if signal != self.last_signal:
                    self.signal_confirmation_count = 1
                    self.last_signal = signal
                    logger.info(f"🔄 Signal changed to {signal}, confirmation count reset to 1")
                else:
                    self.signal_confirmation_count += 1
                    logger.info(f"🔄 Signal {signal} confirmed {self.signal_confirmation_count} times")

                # 조건 완화: 2회 → 1회 확인으로 변경
                if self.signal_confirmation_count < 1:  # required_signal_confirmation을 1로 완화
                    logger.info(f"⏳ Waiting for signal confirmation: {self.signal_confirmation_count}/1")
                    return

            if signal == 1 and (not self.current_position or reverse):  # Buy signal
                stop_loss, take_profit = self.technical_analyzer.calculate_stop_loss_take_profit(
                    self.klines_data, price, 'BUY')
                await self.binance.place_order('BUY', dynamic_position_size)
                self.current_position = {
                    'side': 'BUY', 
                    'entry': price, 
                    'stop_loss': stop_loss, 
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,
                    'size': dynamic_position_size
                }
                self.last_trade_time = current_time
                self.signal_confirmation_count = 0
                self.total_trades += 1
                await self.telegram.send_trade_signal(
                    'BUY', self.symbol, price, f"Signal: {reason}\nSL: {stop_loss:.2f}, TP: {take_profit:.2f}\nSize: {dynamic_position_size:.3f} {self.symbol.split('USDT')[0]}"
                )
            elif signal == -1 and (not self.current_position or reverse):  # Sell signal
                stop_loss, take_profit = self.technical_analyzer.calculate_stop_loss_take_profit(
                    self.klines_data, price, 'SELL')
                await self.binance.place_order('SELL', dynamic_position_size)
                self.current_position = {
                    'side': 'SELL', 
                    'entry': price, 
                    'stop_loss': stop_loss, 
                    'take_profit': take_profit,
                    'trailing_stop': stop_loss,
                    'size': dynamic_position_size
                }
                self.last_trade_time = current_time
                self.signal_confirmation_count = 0
                self.total_trades += 1
                await self.telegram.send_trade_signal(
                    'SELL', self.symbol, price, f"Signal: {reason}\nSL: {stop_loss:.2f}, TP: {take_profit:.2f}\nSize: {dynamic_position_size:.3f} {self.symbol.split('USDT')[0]}"
                )

            await self.update_position()
        except Exception as e:
            logger.error(f"Error in execute_trade: {e}")
            await self.telegram.send_error(f"Error in execute_trade: {e}")

    def calculate_dynamic_leverage(self, score, adx):
        """Calculate dynamic leverage based on market conditions"""
        try:
            # Example logic: Adjust leverage based on ADX and score
            if adx > 25 and score > 0.8:
                return min(50, LEVERAGE * 1.5)  # Increase leverage in strong trends, max 50x
            elif adx < 20 or score < 0.5:
                return max(5, LEVERAGE * 0.5)  # Decrease leverage in weak trends
            return min(50, LEVERAGE)  # Default leverage, max 50x
        except Exception as e:
            logger.error(f"Error in calculate_dynamic_leverage: {e}")
            return min(50, LEVERAGE)  # Fallback to default leverage, max 50x

    def calculate_stop_loss_take_profit(self, df, entry_price, side, leverage=LEVERAGE, lookback=10, min_pct=0.3):
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
            klines = await self.binance.get_klines(limit=100)
            if not klines:
                return
                
            # Update klines data
            self.klines_data = pd.DataFrame(klines)
            
            # Calculate indicators
            self.klines_data = self.technical_analyzer.calculate_indicators(self.klines_data)
            
            # Validate data integrity
            if not self.validate_data_integrity():
                logger.warning(f"[{self.symbol}] Data integrity check failed after market data update")
                return
                
        except Exception as e:
            logger.error(f"Error in update_market_data: {e}")
            await self.telegram.send_error(f"Error in update_market_data: {e}")

    async def run(self):
        """메인 실행 함수"""
        try:
            logger.info(f"[{self.symbol}] Starting trading bot...")
            await self.telegram.send_message(f"[{self.symbol}] 🤖 트레이딩 봇 시작")
            
            # 1. 초기 설정
            await self.setup()
            
            # 2. 기존 포지션 확인 및 처리
            position = await self.binance.get_position()
            if position and float(position['positionAmt']) != 0:
                logger.info(f"[{self.symbol}] Found existing position, initializing...")
                self.current_position = {
                    'side': 'BUY' if float(position['positionAmt']) > 0 else 'SELL',
                    'entry': position['entryPrice'],
                    'size': abs(float(position['positionAmt'])),
                    'take_profit': None,  # 동적 익절가로 업데이트될 예정
                    'stop_loss': position['stopLoss']
                }
                # 기존 포지션 발견 시 한 번만 알림
                await self.telegram.send_message(
                    f"[{self.symbol}] 🔄 기존 포지션 발견\n"
                    f"방향: {self.current_position['side']}\n"
                    f"진입가: {self.current_position['entry']}\n"
                    f"수량: {self.current_position['size']} BTC"
                )
            
            while True:
                try:
                    # 3. 기존 포지션 처리
                    if self.current_position:
                        await self.handle_existing_position()
                    
                    # 4. 시장 데이터 업데이트
                    await self.update_market_data()
                    
                    # 5. 시그널 생성
                    signal, score, adx, market_condition = self.technical_analyzer.generate_signals(self.klines_data)
                    
                    # 6. 시장 상태 로깅 (로그로만 기록)
                    logger.info(f"[{self.symbol}] Signal: {signal}, Score: {score}, ADX: {adx}, Market: {market_condition}")
                    
                    # 7. 포지션이 없는 경우에만 새로운 진입 고려
                    if not self.current_position:
                        # 8. 진입 조건 확인
                        if signal != 0 and score >= 3:
                            current_price = self.klines_data['close'].iloc[-1]
                            await self.execute_trade(signal, current_price, f"Signal: {signal}, Score: {score}")
                    
                    # 9. 대기
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    # 실제 에러 발생 시에만 텔레그램으로 전송
                    if "position" in str(e).lower() or "order" in str(e).lower():
                        await self.telegram.send_error(f"[{self.symbol}] ⚠️ 메인 루프 에러: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.telegram.send_error(f"[{self.symbol}] ❌ 치명적 에러: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        await self.binance.close()
        await self.telegram.close()

    def calculate_dynamic_take_profit(self, df, entry_price, side, current_price):
        """동적 익절 전략 계산 - 개선된 버전"""
        try:
            # 1. 추세 강도 분석 (ADX 기반)
            adx = df['adx'].iloc[-1]
            adx_ma = df['adx'].rolling(20).mean().iloc[-1]
            trend_strength = "strong" if adx > max(30, adx_ma * 1.2) else "weak"
            
            # 2. 모멘텀 분석 (RSI, Stochastic, MACD)
            rsi = df['rsi'].iloc[-1]
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # 모멘텀 강도 계산
            momentum_score = 0
            if side == 'BUY':
                if rsi > 60: momentum_score += 1
                if stoch_k > 80 and stoch_k > stoch_d: momentum_score += 1
                if macd > macd_signal: momentum_score += 1
            else:
                if rsi < 40: momentum_score += 1
                if stoch_k < 20 and stoch_k < stoch_d: momentum_score += 1
                if macd < macd_signal: momentum_score += 1
                
            momentum = "strong" if momentum_score >= 2 else "weak"
            
            # 3. 변동성 분석 (ATR, Bollinger Bands)
            atr = df['atr'].iloc[-1]
            atr_ma = df['atr'].rolling(20).mean().iloc[-1]
            bb_width = (df['bb_high'].iloc[-1] - df['bb_low'].iloc[-1]) / df['bb_mid'].iloc[-1]
            bb_width_ma = ((df['bb_high'].rolling(20).mean().iloc[-1] - 
                           df['bb_low'].rolling(20).mean().iloc[-1]) / 
                          df['bb_mid'].rolling(20).mean().iloc[-1])
            
            volatility = "high" if (atr > atr_ma * 1.5 or bb_width > bb_width_ma * 1.5) else "normal"
            
            # 4. 기본 익절가 계산 (ATR 기반)
            atr_multiplier = 2.5  # 기본 승수
            if trend_strength == "strong":
                atr_multiplier += 0.5  # 강한 추세에서는 더 높은 익절가
            if volatility == "high":
                atr_multiplier -= 0.5  # 높은 변동성에서는 더 낮은 익절가
                
            base_tp = entry_price + (atr * atr_multiplier) if side == 'BUY' else entry_price - (atr * atr_multiplier)
            
            # 5. 조건별 조정
            if trend_strength == "strong" and momentum == "strong":
                # 강한 추세와 모멘텀
                if side == 'BUY':
                    # 롱 포지션: 최소 2% 추가 상승 기대
                    min_tp = current_price * 1.02
                    # 볼린저 밴드 상단을 고려
                    bb_tp = df['bb_high'].iloc[-1] * 1.01  # 밴드 상단 + 1%
                    return max(base_tp, min_tp, bb_tp)
                else:
                    # 숏 포지션: 최소 2% 추가 하락 기대
                    max_tp = current_price * 0.98
                    # 볼린저 밴드 하단을 고려
                    bb_tp = df['bb_low'].iloc[-1] * 0.99  # 밴드 하단 - 1%
                    return min(base_tp, max_tp, bb_tp)
                    
            elif trend_strength == "weak" or volatility == "high":
                # 약한 추세나 높은 변동성
                if side == 'BUY':
                    # 롱 포지션: 더 보수적인 익절가
                    return min(base_tp, current_price * 1.01)  # 최대 1% 추가 상승
                else:
                    # 숏 포지션: 더 보수적인 익절가
                    return max(base_tp, current_price * 0.99)  # 최대 1% 추가 하락
            
            # 6. 최종 익절가 검증
            if side == 'BUY':
                # 롱 포지션: 현재가보다 낮은 익절가는 방지
                return max(base_tp, current_price * 1.005)  # 최소 0.5% 추가 상승
            else:
                # 숏 포지션: 현재가보다 높은 익절가는 방지
                return min(base_tp, current_price * 0.995)  # 최소 0.5% 추가 하락
                
        except Exception as e:
            logger.error(f"Error in calculate_dynamic_take_profit: {e}")
            # 에러 발생 시 기본 익절가 반환
            if side == 'BUY':
                return entry_price * 1.01  # 기본 1% 익절
            else:
                return entry_price * 0.99  # 기본 1% 익절

    def detect_trend_reversal(self, df):
        """추세 전환 감지"""
        try:
            # 1. 가격 패턴 분석
            last_candles = df.tail(3)
            pattern = self.analyze_price_pattern(last_candles)
            
            # 2. 지표 분석
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # 3. 볼륨 분석
            volume_increase = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
            
            # 4. 전환 신호 생성
            reversal_signal = 0
            
            if pattern == "bearish_reversal" and rsi > 70 and macd < macd_signal and volume_increase:
                reversal_signal = -1  # 매도 신호
            elif pattern == "bullish_reversal" and rsi < 30 and macd > macd_signal and volume_increase:
                reversal_signal = 1  # 매수 신호
                
            return reversal_signal
            
        except Exception as e:
            logger.error(f"Error in detect_trend_reversal: {e}")
            return 0

    def analyze_price_pattern(self, candles):
        """가격 패턴 분석"""
        try:
            if len(candles) < 3:
                return "unknown"
                
            # 마지막 3개 캔들 분석
            last_3 = candles.tail(3)
            
            # 상승 후 하락 패턴
            if (last_3['close'].iloc[0] < last_3['close'].iloc[1] and 
                last_3['close'].iloc[1] > last_3['close'].iloc[2] and
                last_3['volume'].iloc[2] > last_3['volume'].iloc[1]):
                return "bearish_reversal"
                
            # 하락 후 상승 패턴
            if (last_3['close'].iloc[0] > last_3['close'].iloc[1] and 
                last_3['close'].iloc[1] < last_3['close'].iloc[2] and
                last_3['volume'].iloc[2] > last_3['volume'].iloc[1]):
                return "bullish_reversal"
                
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error in analyze_price_pattern: {e}")
            return "unknown"

    def adjust_position_for_reversal(self, current_position, reversal_signal):
        """추세 전환 시 포지션 조정"""
        try:
            if not current_position:
                return "hold"
                
            # 1. 현재 수익률 계산
            entry_price = float(current_position['entry'])
            current_price = self.klines_data['close'].iloc[-1]
            side = current_position['side']
            
            if side == 'BUY':
                current_pnl = (current_price - entry_price) / entry_price
            else:
                current_pnl = (entry_price - current_price) / entry_price
            
            # 2. 리스크 평가
            if current_pnl > 0.02:  # 2% 이상 수익 중
                if reversal_signal != 0:
                    # 수익 중이고 전환 신호가 있으면 청산
                    return "close"
            elif current_pnl < -0.01:  # 1% 이상 손실 중
                if reversal_signal != 0:
                    # 손실 중이고 전환 신호가 있으면 반대 포지션 진입
                    return "reverse"
                    
            return "hold"
            
        except Exception as e:
            logger.error(f"Error in adjust_position_for_reversal: {e}")
            return "hold"

    async def handle_existing_position(self):
        """기존 포지션 처리"""
        try:
            if not self.current_position:
                return
                
            # 1. 현재 포지션 정보 확인
            position = await self.binance.get_position()
            if not position or float(position['positionAmt']) == 0:
                self.current_position = None
                return
                
            # 2. 추세 전환 감지
            reversal_signal = self.detect_trend_reversal(self.klines_data)
            
            # 3. 포지션 조정 결정
            action = self.adjust_position_for_reversal(self.current_position, reversal_signal)
            
            # 4. 동적 익절가 업데이트
            current_price = self.klines_data['close'].iloc[-1]
            new_take_profit = self.calculate_dynamic_take_profit(
                self.klines_data,
                float(self.current_position['entry']),
                self.current_position['side'],
                current_price
            )
            
            # 5. 포지션 조정 실행
            if action == "close":
                await self.close_position("Trend reversal detected")
            elif action == "reverse":
                await self.close_position("Trend reversal - reversing position")
                # 반대 포지션 진입
                if reversal_signal == 1:
                    await self.execute_trade(1, current_price, "Trend reversal - long")
                else:
                    await self.execute_trade(-1, current_price, "Trend reversal - short")
            else:  # hold
                # 익절가만 업데이트
                self.current_position['take_profit'] = new_take_profit
                
        except Exception as e:
            logger.error(f"Error in handle_existing_position: {e}")
            await self.telegram.send_error(f"Error in handle_existing_position: {e}")

if __name__ == "__main__":
    import asyncio
    symbols = ["BTCUSDT", "ETHUSDT"]
    bots = [TradingBot(symbol) for symbol in symbols]

    async def run_all():
        await asyncio.gather(*(bot.run() for bot in bots))

    asyncio.run(run_all()) 