import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
import logging
from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_SYMBOL, LEVERAGE
import websockets
import json

class BinanceClient:
    def __init__(self):
        self.client = None
        self.bm = None
        self.symbol = TRADING_SYMBOL
        self.logger = logging.getLogger(__name__)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # 초기 재연결 대기 시간 (초)

    async def initialize(self):
        """Initialize Binance client and websocket manager, set leverage"""
        try:
            self.client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
            self.bm = BinanceSocketManager(self.client)
            await self.set_leverage(LEVERAGE)
            self.logger.info("Binance client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise

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

    async def place_order(self, side, quantity, order_type='MARKET', reduce_only=False):
        """Place an order with proper quantity precision"""
        try:
            # 수량 정밀도 조정 (BTC의 경우 3자리까지)
            quantity = round(quantity, 3)
            
            # 최소 주문 수량 확인 (BTC의 경우 0.001)
            if quantity < 0.001:
                quantity = 0.001
                
            params = {
                'symbol': TRADING_SYMBOL,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'reduceOnly': reduce_only
            }
            
            order = await self.client.futures_create_order(**params)
            self.logger.info(f"Order placed: {side} {quantity} {TRADING_SYMBOL}")
            return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None

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
        """Set leverage for the trading pair"""
        try:
            # 레버리지 값을 정수로 변환하고 범위 제한
            leverage = int(min(max(1, leverage), 50))  # 최대 레버리지는 50x
            
            params = {
                'symbol': TRADING_SYMBOL,
                'leverage': leverage
            }
            await self.client.futures_change_leverage(**params)
            self.logger.info(f"Leverage set to {leverage}x")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            return False

    async def place_order(self, side, quantity, order_type='MARKET', price=None, reduce_only=False):
        """Place a futures order with optional reduceOnly flag"""
        try:
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