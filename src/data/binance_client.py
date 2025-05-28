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
import time

class RateLimiter:
    """API 요청 제한 관리 클래스"""
    def __init__(self):
        self.request_times = []
        self.weight_usage = {}
        self.max_requests_per_minute = 1000  # 안전 마진을 둔 제한
        self.max_weight_per_minute = 1200
        self.order_count = 0
        self.max_orders_per_minute = 100  # 주문 제한
        
    async def wait_if_needed(self, weight=1, is_order=False):
        """필요시 대기"""
        current_time = time.time()
        
        # 1분 이전 요청 제거
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # 요청 수 제한 확인
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
        # Weight 제한 확인
        total_weight = sum(self.weight_usage.get(t, 0) for t in self.request_times)
        if total_weight + weight > self.max_weight_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
        # 주문 제한 확인
        if is_order and self.order_count >= self.max_orders_per_minute:
            await asyncio.sleep(60)
            self.order_count = 0
            
        # 요청 기록
        self.request_times.append(current_time)
        self.weight_usage[current_time] = weight
        if is_order:
            self.order_count += 1

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
        
        # Rate Limiter 추가
        self.rate_limiter = RateLimiter()
        
        # 캐시 시스템 추가
        self.cache = {}
        self.cache_ttl = {}
        self.default_cache_duration = 30  # 30초 캐시
        
        # 배치 요청 시스템
        self.batch_requests = []
        self.batch_interval = 5  # 5초마다 배치 처리
        self.last_batch_time = time.time()
        
        # WebSocket 안전장치 추가
        self.is_connected = False
        self.last_price = None
        self.last_data_time = None
        self.connection_lost_time = None
        self.data_buffer = []  # 재연결 시 데이터 검증용
        self.max_buffer_size = 100
        self.price_validation_threshold = 0.05  # 5% 이상 가격 변동 시 검증
        self.reconnection_callback = None  # 재연결 시 호출할 콜백
        
        # API 요청 최적화
        self.position_cache_duration = 10  # 포지션 정보 10초 캐시
        self.account_cache_duration = 30   # 계정 정보 30초 캐시
        self.last_position_request = 0
        self.last_account_request = 0

    def get_cache(self, key):
        """캐시에서 데이터 가져오기"""
        if key in self.cache and key in self.cache_ttl:
            if time.time() < self.cache_ttl[key]:
                return self.cache[key]
            else:
                # 만료된 캐시 삭제
                del self.cache[key]
                del self.cache_ttl[key]
        return None

    def set_cache(self, key, value, duration=None):
        """캐시에 데이터 저장"""
        if duration is None:
            duration = self.default_cache_duration
        self.cache[key] = value
        self.cache_ttl[key] = time.time() + duration

    async def calculate_optimal_leverage(self):
        """Calculate optimal leverage based on market volatility"""
        try:
            # 캐시 확인
            cache_key = f"optimal_leverage_{self.symbol}"
            cached_result = self.get_cache(cache_key)
            if cached_result:
                return cached_result
                
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=5)
            
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
            
            # 결과 캐시 (1시간)
            self.set_cache(cache_key, optimal_leverage, 3600)
            
            self.logger.info(f"Calculated optimal leverage: {optimal_leverage}x (Volatility: {volatility:.4f}, ATR: {atr:.2f})")
            return optimal_leverage
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal leverage: {e}")
            return MAX_LEVERAGE // 2  # 에러 발생 시 기본값 반환

    async def initialize(self):
        """Initialize Binance client and websocket manager"""
        try:
            await self.rate_limiter.wait_if_needed(weight=1)
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
        """Get historical klines/candlestick data with caching"""
        try:
            # 캐시 확인
            cache_key = f"klines_{self.symbol}_{interval}_{limit}"
            cached_result = self.get_cache(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Rate limiting 적용 (weight: 1-5 depending on limit)
            weight = 1 if limit <= 100 else 2 if limit <= 500 else 5
            await self.rate_limiter.wait_if_needed(weight=weight)
            
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
                
            # 결과 캐시 (간격에 따라 다른 캐시 시간)
            cache_duration = 60 if interval == '1m' else 300 if interval == '5m' else 900
            self.set_cache(cache_key, df, cache_duration)
            
            return df
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get historical klines: {e}")
            raise

    async def stream_klines(self, callback):
        """Stream real-time klines data with robust reconnection logic"""
        self.reconnection_callback = callback  # 콜백 저장
        
        while True:
            try:
                await self._stream_klines_with_reconnect(callback)
            except Exception as e:
                self.is_connected = False
                self.connection_lost_time = time.time()
                
                self.reconnect_attempts += 1
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
                    raise
                
                # 개선된 지수 백오프: 네트워크 상태에 따른 적응적 대기
                base_delay = 2  # 기본 2초
                max_delay = 60  # 최대 1분
                
                # 연속 실패 횟수에 따른 지수 백오프
                delay = min(base_delay * (1.5 ** (self.reconnect_attempts - 1)), max_delay)
                
                # 특정 에러 타입에 따른 대기 시간 조정
                error_msg = str(e).lower()
                if "timeout" in error_msg or "ping" in error_msg:
                    delay = min(delay, 10)  # ping 타임아웃은 빠른 재연결
                elif "connection refused" in error_msg:
                    delay = max(delay, 30)  # 서버 거부는 더 긴 대기
                
                self.logger.warning(f"WebSocket connection failed (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}). Reconnecting in {delay:.1f} seconds...")
                self.logger.info(f"Error details: {e}")
                
                # 연결 끊김 중 포지션 상태 확인
                await self._check_position_during_disconnect()
                
                await asyncio.sleep(delay)
                
                # 5번 연속 실패 시 연결 방식 변경 시도
                if self.reconnect_attempts % 5 == 0:
                    self.logger.info("Trying alternative connection approach...")
                    # 잠시 더 긴 대기로 네트워크 안정화
                    await asyncio.sleep(5)

    async def _stream_klines_with_reconnect(self, callback):
        """Internal method for WebSocket streaming with connection handling"""
        # 여러 서버 엔드포인트 시도
        endpoints = [
            f"wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_1m",
            f"wss://fstream1.binance.com/ws/{self.symbol.lower()}@kline_1m",
            f"wss://fstream2.binance.com/ws/{self.symbol.lower()}@kline_1m"
        ]
        
        # 현재 시도할 엔드포인트 선택 (라운드 로빈)
        endpoint_index = self.reconnect_attempts % len(endpoints)
        url = endpoints[endpoint_index]
        
        self.logger.info(f"Attempting connection to: {url}")
        
        try:
            # 더욱 보수적인 연결 설정
            async with websockets.connect(
                url,
                ping_interval=10,   # 10초마다 ping (더 자주)
                ping_timeout=5,     # ping 응답 대기 시간 더 단축
                close_timeout=3,    # 연결 종료 대기 시간 더 단축
                max_size=2**20,     # 최대 메시지 크기 (1MB)
                compression=None,   # 압축 비활성화
                # 추가 헤더로 연결 안정성 향상
                extra_headers={
                    'User-Agent': 'TradingBot/1.0',
                    'Connection': 'keep-alive',
                    'Cache-Control': 'no-cache'
                }
            ) as ws:
                self.logger.info(f"WebSocket connection established successfully to {url}")
                self.reconnect_attempts = 0  # 성공적으로 연결되면 재연결 카운터 리셋
                
                # 연결 상태 모니터링 변수
                last_message_time = time.time()
                heartbeat_interval = 15  # 15초마다 하트비트 체크 (더 자주)
                ping_count = 0
                max_consecutive_pings = 3  # 연속 ping 실패 허용 횟수
                
                while True:
                    try:
                        # 메시지 수신 타임아웃 설정 (15초로 더 단축)
                        msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        msg = json.loads(msg)
                        last_message_time = time.time()
                        ping_count = 0  # 메시지 수신 시 ping 카운터 리셋
                        
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
                            
                            # 연결 상태 업데이트
                            if not self.is_connected:
                                self.is_connected = True
                                self.connection_lost_time = None
                                self.logger.info("WebSocket connection restored successfully")
                                
                                # 재연결 후 데이터 검증
                                if not await self._validate_reconnection_data(data):
                                    self.logger.error("Data validation failed after reconnection, skipping this data point")
                                    continue
                            
                            # 데이터 버퍼 관리
                            self.data_buffer.append(data)
                            if len(self.data_buffer) > self.max_buffer_size:
                                self.data_buffer.pop(0)
                            
                            # 가격 정보 업데이트
                            self.last_price = data['close']
                            self.last_data_time = time.time()
                            
                            await callback(data)
                            
                    except asyncio.TimeoutError:
                        # 타임아웃 발생 시 연결 상태 확인
                        current_time = time.time()
                        time_since_last_msg = current_time - last_message_time
                        
                        if time_since_last_msg > heartbeat_interval:
                            self.logger.warning(f"No message received for {time_since_last_msg:.1f} seconds, checking connection...")
                            
                            # 연결 상태 확인을 위한 ping
                            try:
                                pong_waiter = await ws.ping()
                                await asyncio.wait_for(pong_waiter, timeout=3.0)  # 더 짧은 타임아웃
                                self.logger.info("Connection ping successful")
                                last_message_time = current_time
                                ping_count = 0
                            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                                ping_count += 1
                                self.logger.warning(f"Ping failed ({ping_count}/{max_consecutive_pings})")
                                
                                if ping_count >= max_consecutive_pings:
                                    self.logger.error("Max consecutive ping failures reached, reconnecting...")
                                    raise websockets.exceptions.ConnectionClosed(1011, "Multiple ping timeouts")
                                
                                # 짧은 대기 후 재시도
                                await asyncio.sleep(1)
                                continue
                            except Exception as ping_error:
                                self.logger.warning(f"Ping error: {ping_error}, reconnecting...")
                                raise
                        else:
                            # 아직 하트비트 간격 내라면 계속 대기
                            continue
                            
                    except websockets.exceptions.ConnectionClosed as e:
                        self.logger.warning(f"WebSocket connection closed: {e}")
                        raise
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON message: {e}")
                        continue  # JSON 파싱 에러는 무시하고 계속
                    except Exception as e:
                        self.logger.error(f"Unexpected error in message handling: {e}")
                        # 예상치 못한 에러는 재연결 시도
                        raise
                        
        except websockets.exceptions.InvalidURI as e:
            self.logger.error(f"Invalid WebSocket URI: {e}")
            raise
        except websockets.exceptions.WebSocketException as e:
            self.logger.error(f"WebSocket error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in kline stream: {e}")
            raise

    async def _check_position_during_disconnect(self):
        """연결 끊김 중 포지션 상태 확인 및 보호"""
        try:
            if self.connection_lost_time:
                disconnect_duration = time.time() - self.connection_lost_time
                
                if disconnect_duration > 30:  # 30초 이상 끊김
                    self.logger.warning(f"Connection lost for {disconnect_duration:.1f} seconds. Checking position safety...")
                    
                    # 현재 포지션 확인
                    position = await self.get_position()
                    if position and abs(float(position.get('positionAmt', 0))) > 0:
                        unrealized_pnl = float(position.get('unRealizedPnl', 0))
                        percentage = float(position.get('percentage', 0))
                        
                        # 큰 손실 발생 시 경고
                        if percentage < -3:  # 3% 이상 손실
                            self.logger.error(f"ALERT: Large loss during disconnect! PnL: {percentage:.2f}%")
                            # 필요시 여기서 긴급 청산 로직 추가 가능
                        
                        self.logger.info(f"Position status during disconnect - PnL: {percentage:.2f}%, Amount: {position.get('positionAmt')}")
                
        except Exception as e:
            self.logger.error(f"Error checking position during disconnect: {e}")

    async def _validate_reconnection_data(self, new_data):
        """재연결 후 데이터 검증"""
        try:
            if not self.last_price or not new_data:
                return True
            
            new_price = new_data.get('close', 0)
            if not new_price:
                return True
            
            # 가격 변동률 검증
            price_change = abs(new_price - self.last_price) / self.last_price
            
            if price_change > self.price_validation_threshold:
                self.logger.warning(f"Large price change detected after reconnection: {price_change:.2%}")
                
                # REST API로 현재 가격 재확인
                try:
                    ticker = await self.client.futures_symbol_ticker(symbol=self.symbol)
                    api_price = float(ticker['price'])
                    
                    # API 가격과 WebSocket 가격 비교
                    api_ws_diff = abs(api_price - new_price) / api_price
                    
                    if api_ws_diff > 0.001:  # 0.1% 이상 차이
                        self.logger.error(f"Price mismatch! WebSocket: {new_price}, API: {api_price}")
                        return False
                    else:
                        self.logger.info("Price validated successfully with REST API")
                        
                except Exception as api_error:
                    self.logger.error(f"Failed to validate price with API: {api_error}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating reconnection data: {e}")
            return True  # 검증 실패 시 데이터 허용 (보수적 접근)

    async def place_order(self, side, quantity, order_type='MARKET', price=None, reduce_only=False):
        """Place an order with rate limiting"""
        try:
            # Rate limiting 적용 (주문은 높은 weight)
            await self.rate_limiter.wait_if_needed(weight=10, is_order=True)
            
            order_params = {
                'symbol': self.symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'reduceOnly': reduce_only
            }
            
            if order_type == 'LIMIT' and price:
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'
            
            result = await self.client.futures_create_order(**order_params)
            self.logger.info(f"Order placed successfully: {result}")
            return result
        except BinanceAPIException as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    async def get_position(self):
        """Get current position with caching"""
        try:
            # 캐시 확인
            current_time = time.time()
            if current_time - self.last_position_request < self.position_cache_duration:
                cache_key = f"position_{self.symbol}"
                cached_result = self.get_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=5)
            
            positions = await self.client.futures_position_information(symbol=self.symbol)
            position = positions[0] if positions else None
            
            # 캐시 저장
            cache_key = f"position_{self.symbol}"
            self.set_cache(cache_key, position, self.position_cache_duration)
            self.last_position_request = current_time
            
            return position
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get position: {e}")
            raise

    async def close(self):
        """Close the client connection"""
        if self.client:
            await self.client.close_connection()

    async def set_leverage(self, leverage):
        """Set leverage for the symbol with rate limiting"""
        try:
            # 캐시 확인 (현재 레버리지)
            cache_key = f"leverage_{self.symbol}"
            cached_leverage = self.get_cache(cache_key)
            if cached_leverage == leverage:
                return  # 이미 같은 레버리지면 요청하지 않음
                
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=1)
            
            await self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            
            # 캐시 업데이트
            self.set_cache(cache_key, leverage, 3600)  # 1시간 캐시
            
            self.logger.info(f"Leverage set to {leverage}x for {self.symbol}")
        except BinanceAPIException as e:
            if "No need to change margin type" in str(e) or "leverage not modified" in str(e):
                self.logger.info(f"Leverage already set to {leverage}x")
                # 캐시 업데이트
                cache_key = f"leverage_{self.symbol}"
                self.set_cache(cache_key, leverage, 3600)
            else:
                self.logger.error(f"Failed to set leverage: {e}")
                raise

    async def get_account_info(self):
        """Get account information with caching"""
        try:
            # 캐시 확인
            current_time = time.time()
            if current_time - self.last_account_request < self.account_cache_duration:
                cache_key = "account_info"
                cached_result = self.get_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=5)
            
            account_info = await self.client.futures_account()
            
            # 캐시 저장
            cache_key = "account_info"
            self.set_cache(cache_key, account_info, self.account_cache_duration)
            self.last_account_request = current_time
            
            return account_info
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise

    async def get_symbol_info(self):
        """Get symbol information with caching"""
        try:
            # 캐시 확인 (심볼 정보는 자주 변하지 않으므로 긴 캐시)
            cache_key = f"symbol_info_{self.symbol}"
            cached_result = self.get_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=1)
            
            exchange_info = await self.client.futures_exchange_info()
            symbol_info = None
            
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.symbol:
                    symbol_info = symbol
                    break
            
            # 캐시 저장 (4시간)
            if symbol_info:
                self.set_cache(cache_key, symbol_info, 14400)
            
            return symbol_info
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get symbol info: {e}")
            raise

    async def get_current_price(self):
        """Get current price with caching"""
        try:
            # 캐시 확인 (가격은 짧은 캐시)
            cache_key = f"price_{self.symbol}"
            cached_result = self.get_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Rate limiting 적용
            await self.rate_limiter.wait_if_needed(weight=1)
            
            ticker = await self.client.futures_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            
            # 캐시 저장 (5초)
            self.set_cache(cache_key, price, 5)
            
            return price
        except BinanceAPIException as e:
            self.logger.error(f"Failed to get current price: {e}")
            raise

    async def batch_process_requests(self):
        """배치로 요청 처리 (향후 확장용)"""
        if not self.batch_requests:
            return
            
        current_time = time.time()
        if current_time - self.last_batch_time >= self.batch_interval:
            # 배치 요청 처리 로직
            self.logger.info(f"Processing {len(self.batch_requests)} batch requests")
            
            # 실제 배치 처리는 향후 구현
            self.batch_requests.clear()
            self.last_batch_time = current_time

    def get_rate_limit_status(self):
        """현재 rate limit 상태 반환"""
        current_time = time.time()
        recent_requests = [t for t in self.rate_limiter.request_times if current_time - t < 60]
        total_weight = sum(self.rate_limiter.weight_usage.get(t, 0) for t in recent_requests)
        
        return {
            'requests_per_minute': len(recent_requests),
            'max_requests_per_minute': self.rate_limiter.max_requests_per_minute,
            'weight_per_minute': total_weight,
            'max_weight_per_minute': self.rate_limiter.max_weight_per_minute,
            'orders_per_minute': self.rate_limiter.order_count,
            'max_orders_per_minute': self.rate_limiter.max_orders_per_minute,
            'cache_size': len(self.cache)
        } 