import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
import logging
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOLS, MAX_LEVERAGE
import websockets
import json
import numpy as np
from binance.client import Client

class BinanceClient:
    def __init__(self, symbol=None):
        """Initialize Binance client"""
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.bm = None
        self.symbol = symbol or TRADING_SYMBOLS[0]  # 기본값으로 첫 번째 심볼 사용
        self.logger = logging.getLogger(__name__)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # 초기 재연결 대기 시간 (초)
        self.volatility_window = 24  # 24시간 데이터로 변동성 계산
        self.leverage_update_interval = 3600  # 1시간마다 레버리지 업데이트
        self.last_leverage_update = 0

    async def calculate_optimal_leverage(self):
        """Calculate optimal leverage based on market volatility"""
        try:
            # 과거 데이터 가져오기 (24시간)
            klines = await self.get_historical_klines(interval='1h', limit=self.volatility_window)
            
            # 변동성 계산
            returns = klines['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # 연간화된 변동성
            
            # ATR 계산
            high_low = klines['high'] - klines['low']
            high_close = np.abs(klines['high'] - klines['close'].shift())
            low_close = np.abs(klines['low'] - klines['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # 현재 가격
            current_price = float(klines['close'].iloc[-1])
            
            # 변동성 기반 레버리지 계산
            # 변동성이 높을수록 레버리지 감소
            volatility_factor = 1 / (1 + volatility * 100)  # 변동성이 1% 증가할 때마다 레버리지 감소
            atr_factor = 1 / (1 + (atr / current_price) * 100)  # ATR이 가격의 1% 증가할 때마다 레버리지 감소
            
            # 최종 레버리지 계산
            base_leverage = MAX_LEVERAGE // 2  # 기본값은 최대 레버리지의 절반
            optimal_leverage = int(base_leverage * volatility_factor * atr_factor)
            
            # 레버리지 범위 제한
            optimal_leverage = max(1, min(optimal_leverage, MAX_LEVERAGE))
            
            self.logger.info(f"Calculated optimal leverage: {optimal_leverage}x (Volatility: {volatility:.4f}, ATR: {atr:.2f})")
            return optimal_leverage
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal leverage: {e}")
            return MAX_LEVERAGE // 2  # 에러 발생 시 기본값 반환

    async def initialize(self):
        """Initialize Binance client and websocket manager"""
        try:
            self.client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
            self.bm = BinanceSocketManager(self.client)
            
            # 초기 레버리지 동적 설정
            initial_leverage = await self.calculate_optimal_leverage()
            await self.set_leverage(initial_leverage)
            self.last_leverage_update = asyncio.get_event_loop().time()
            
            self.logger.info(f"Binance client initialized with dynamic leverage {initial_leverage}x")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise

    async def update_leverage_if_needed(self):
        """주기적으로 레버리지 업데이트"""
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_leverage_update >= self.leverage_update_interval:
            new_leverage = await self.calculate_optimal_leverage()
            await self.set_leverage(new_leverage)
            self.last_leverage_update = current_time
            return new_leverage
        return None

    async def get_historical_klines(self, interval='1m', limit=100):
        """Get historical klines/candlestick data"""
        try:
            klines = await self.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get historical klines: {e}")
            raise

    async def stream_klines(self, callback):
        """Stream real-time klines data with robust reconnection logic"""
        while True:
            try:
                await self._stream_klines_with_reconnect(callback)
            except Exception as e:
                self.reconnect_attempts += 1
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
                    raise
                
                # 지수 백오프: 재연결 시도마다 대기 시간 증가
                delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)  # 최대 5분
                self.logger.warning(f"WebSocket connection failed (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}). Reconnecting in {delay} seconds...")
                await asyncio.sleep(delay)

    async def _stream_klines_with_reconnect(self, callback):
        """Internal method for WebSocket streaming with connection handling"""
        url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_1m"
        
        try:
            # 연결 타임아웃과 ping 설정
            async with websockets.connect(
                url,
                ping_interval=20,  # 20초마다 ping 전송
                ping_timeout=10,   # ping 응답 대기 시간
                close_timeout=10,  # 연결 종료 대기 시간
                max_size=2**20,    # 최대 메시지 크기 (1MB)
                compression=None   # 압축 비활성화로 성능 향상
            ) as ws:
                self.logger.info("WebSocket connection established successfully")
                self.reconnect_attempts = 0  # 성공적으로 연결되면 재연결 카운터 리셋
                
                while True:
                    try:
                        # 메시지 수신 타임아웃 설정 (30초)
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        msg = json.loads(msg)
                        
                        if msg.get('e') == 'kline':
                            kline = msg['k']
                            data = {
                                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                                'open': float(kline['o']),
                                'high': float(kline['h']),
                                'low': float(kline['l']),
                                'close': float(kline['c']),
                                'volume': float(kline['v'])
                            }
                            await callback(data)
                            
                    except asyncio.TimeoutError:
                        self.logger.warning("No message received for 30 seconds, checking connection...")
                        # ping을 보내서 연결 상태 확인
                        try:
                            await ws.ping()
                        except Exception:
                            self.logger.warning("Ping failed, connection may be lost")
                            raise
                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.warning(f"WebSocket connection closed: {e}")
                        raise
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON message: {e}")
                        continue  # JSON 파싱 에러는 무시하고 계속
                        
        except websockets.exceptions.InvalidURI as e:
            self.logger.error(f"Invalid WebSocket URI: {e}")
            raise
        except websockets.exceptions.WebSocketException as e:
            self.logger.error(f"WebSocket error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in kline stream: {e}")
            raise

    async def place_order(self, side, quantity, order_type='MARKET', price=None, reduce_only=False):
        """Place a futures order with optional reduceOnly flag"""
        try:
            # 수량 정밀도 조정 (BTC의 경우 3자리까지)
            quantity = round(quantity, 3)
            
            # 최소 주문 수량 확인 (BTC의 경우 0.001)
            if quantity < 0.001:
                quantity = 0.001
                
            params = {
                'symbol': self.symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            if price and order_type == 'LIMIT':
                params['price'] = price
                params['timeInForce'] = 'GTC'
            if reduce_only:
                params['reduceOnly'] = True
            order = await self.client.futures_create_order(**params)
            self.logger.info(f"Order placed successfully: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    async def get_position(self):
        """Get current position information"""
        try:
            positions = await self.client.futures_position_information(symbol=self.symbol)
            return positions[0] if positions else None
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get position: {e}")
            raise

    async def close(self):
        """Close the Binance client connection"""
        if self.client:
            await self.client.close_connection()
            self.logger.info("Binance client connection closed")

    async def set_leverage(self, leverage):
        """Set leverage for the symbol - only when no position exists"""
        try:
            # Check if there are open positions
            position = await self.get_position()
            if position and abs(float(position.get('positionAmt', 0))) > 0:
                current_leverage = int(position.get('leverage', 1))
                self.logger.info(f"Position exists with {current_leverage}x leverage. Maintaining current position and leverage.")
                return True
            
            # Try to set margin type to ISOLATED first (only when no position)
            try:
                await self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
                self.logger.info(f"Changed margin type to ISOLATED for {self.symbol}")
            except Exception as margin_error:
                # If already in ISOLATED mode, this will fail - that's okay
                if "No need to change margin type" in str(margin_error) or "-4046" in str(margin_error):
                    self.logger.info(f"Margin type already set to ISOLATED for {self.symbol}")
                else:
                    self.logger.warning(f"Could not change margin type: {margin_error}")
            
            # Now try to set leverage (only when no position)
            result = await self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            self.logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Leverage reduction is not supported in Isolated Margin Mode" in error_msg:
                self.logger.info(f"Cannot change leverage in Isolated Margin Mode with open positions. Using current leverage.")
                return True  # Continue trading with current leverage
            elif "No need to change leverage" in error_msg or "-4028" in error_msg:
                self.logger.info(f"Leverage already set to {leverage}x for {self.symbol}")
                return True
            else:
                self.logger.error(f"Failed to set leverage: {e}")
                return False

    async def analyze_existing_position(self, position, current_price, technical_data=None):
        """Analyze existing position to determine if it should be maintained"""
        try:
            if not position or abs(float(position.get('positionAmt', 0))) == 0:
                return {'action': 'none', 'reason': 'No position exists'}
            
            entry_price = float(position.get('entryPrice', 0))
            position_amt = float(position.get('positionAmt', 0))
            unrealized_pnl = float(position.get('unRealizedPnl', 0))
            percentage = float(position.get('percentage', 0))
            
            # 포지션 방향 확인
            is_long = position_amt > 0
            side = 'LONG' if is_long else 'SHORT'
            
            # 현재 수익률 계산
            if is_long:
                pnl_percentage = (current_price - entry_price) / entry_price * 100
            else:
                pnl_percentage = (entry_price - current_price) / entry_price * 100
            
            self.logger.info(f"Analyzing existing {side} position: Entry={entry_price}, Current={current_price}, PnL={pnl_percentage:.2f}%")
            
            # 포지션 유지 조건 분석
            analysis_result = {
                'action': 'maintain',
                'reason': 'Position analysis in progress',
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_percentage': pnl_percentage,
                'side': side,
                'leverage': int(position.get('leverage', 1))
            }
            
            # 1. 큰 손실 상황 (-5% 이상)
            if pnl_percentage < -5:
                analysis_result.update({
                    'action': 'close',
                    'reason': f'Large loss detected: {pnl_percentage:.2f}%'
                })
                return analysis_result
            
            # 2. 큰 수익 상황 (+10% 이상) - 일부 익절 고려
            if pnl_percentage > 10:
                analysis_result.update({
                    'action': 'partial_close',
                    'reason': f'Large profit detected: {pnl_percentage:.2f}%, consider partial profit taking'
                })
                return analysis_result
            
            # 3. 기술적 분석이 있는 경우 추가 검증
            if technical_data:
                # RSI 과매수/과매도 확인
                rsi = technical_data.get('rsi', 50)
                if is_long and rsi > 80:
                    analysis_result.update({
                        'action': 'close',
                        'reason': f'LONG position with RSI overbought: {rsi:.1f}'
                    })
                    return analysis_result
                elif not is_long and rsi < 20:
                    analysis_result.update({
                        'action': 'close',
                        'reason': f'SHORT position with RSI oversold: {rsi:.1f}'
                    })
                    return analysis_result
                
                # 추세 전환 확인
                ema_short = technical_data.get('ema_short', current_price)
                ema_long = technical_data.get('ema_long', current_price)
                
                if is_long and ema_short < ema_long:
                    analysis_result.update({
                        'action': 'close',
                        'reason': 'LONG position but trend turning bearish (EMA crossover)'
                    })
                    return analysis_result
                elif not is_long and ema_short > ema_long:
                    analysis_result.update({
                        'action': 'close',
                        'reason': 'SHORT position but trend turning bullish (EMA crossover)'
                    })
                    return analysis_result
            
            # 4. 기본적으로 유지 (손실이 크지 않고 기술적으로 문제없음)
            if -2 <= pnl_percentage <= 8:
                analysis_result.update({
                    'action': 'maintain',
                    'reason': f'Position within acceptable range: {pnl_percentage:.2f}%'
                })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing existing position: {e}")
            return {
                'action': 'maintain',
                'reason': f'Analysis error, maintaining position: {e}',
                'error': True
            }

    async def get_position_with_analysis(self, current_price, technical_data=None):
        """Get position information with analysis"""
        try:
            position = await self.get_position()
            if not position:
                return None, None
            
            analysis = await self.analyze_existing_position(position, current_price, technical_data)
            return position, analysis
            
        except Exception as e:
            self.logger.error(f"Error getting position with analysis: {e}")
            return None, None 