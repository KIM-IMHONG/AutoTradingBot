import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import StochasticOscillator
try:
    from ta.trend import SuperTrend
except ImportError:
    SuperTrend = None
from config.settings import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_SHORT, EMA_MEDIUM, EMA_LONG,
    STOCH_K, STOCH_D, STOCH_SLOW
)
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, symbol=None):
        self.symbol = symbol
        self.rsi = None
        self.macd = None
        self.ema_short = None
        self.ema_medium = None
        self.ema_long = None

    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            # Check minimum data points
            min_data_points = max(50, max(RSI_PERIOD, MACD_SLOW, EMA_LONG))
            if len(df) < min_data_points:
                logger.warning(f"Insufficient data points for indicator calculation. Need at least {min_data_points}, got {len(df)}")
                return df

            # RSI
            self.rsi = RSIIndicator(close=df['close'], window=RSI_PERIOD)
            df['rsi'] = self.rsi.rsi()

            # MACD
            self.macd = MACD(
                close=df['close'],
                window_slow=MACD_SLOW,
                window_fast=MACD_FAST,
                window_sign=MACD_SIGNAL
            )
            df['macd'] = self.macd.macd()
            df['macd_signal'] = self.macd.macd_signal()
            df['macd_diff'] = self.macd.macd_diff()

            # EMAs
            self.ema_short = EMAIndicator(close=df['close'], window=EMA_SHORT)
            self.ema_medium = EMAIndicator(close=df['close'], window=EMA_MEDIUM)
            self.ema_long = EMAIndicator(close=df['close'], window=EMA_LONG)

            df['ema_short'] = self.ema_short.ema_indicator()
            df['ema_medium'] = self.ema_medium.ema_indicator()
            df['ema_long'] = self.ema_long.ema_indicator()

            # ADX (Average Directional Index)
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed TR and DM
            period = 14
            smoothed_tr = tr.rolling(window=period).sum()
            smoothed_plus_dm = pd.Series(plus_dm).rolling(window=period).sum()
            smoothed_minus_dm = pd.Series(minus_dm).rolling(window=period).sum()
            
            # Plus and Minus DI
            plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
            minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=period).mean()
            
            # Stochastic
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=STOCH_K,
                smooth_window=STOCH_D
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            
            # ATR
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            df['atr'] = atr.average_true_range()
            
            # Advanced indicators
            df = self.calculate_advanced_indicators(df)

            # Validate and clean all indicators
            indicator_cols = [
                'ema_short', 'ema_medium', 'ema_long', 'rsi', 'macd', 
                'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'stoch_k', 
                'stoch_d', 'atr', 'supertrend', 'adx'
            ]
            
            for col in indicator_cols:
                if col in df.columns:
                    # Forward fill NaN values
                    df[col] = df[col].ffill()
                    # Backward fill remaining NaN values
                    df[col] = df[col].bfill()
                    # Fill any remaining NaN values with 0
                    df[col] = df[col].fillna(0)
                    # Replace infinite values with 0
                    df[col] = df[col].replace([np.inf, -np.inf], 0)

            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {e}")
            # 에러 발생 시 기본값으로 채우기
            df['rsi'] = 50
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_diff'] = 0
            df['ema_short'] = df['close']
            df['ema_medium'] = df['close']
            df['ema_long'] = df['close']
            df['stoch_k'] = 50
            df['stoch_d'] = 50
            df['bb_high'] = df['close']
            df['bb_low'] = df['close']
            df['bb_mid'] = df['close']
            df['atr'] = 0
            df['adx'] = 25  # ADX 기본값 추가
            return df

    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate Supertrend indicator"""
        try:
            # Calculate ATR
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.rolling(period).mean()
            
            # Calculate Supertrend
            hl2 = (high + low) / 2
            
            # Upper and Lower Bands
            upperband = hl2 + (multiplier * atr)
            lowerband = hl2 - (multiplier * atr)
            
            # Initialize Supertrend
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            # Calculate Supertrend
            for i in range(period, len(df)):
                curr, prev = i, i-1
                
                # Initialize direction
                if i == period:
                    supertrend.iloc[curr] = lowerband.iloc[curr]
                    direction.iloc[curr] = 1
                    continue
                
                # Calculate Supertrend
                if supertrend.iloc[prev] == upperband.iloc[prev]:
                    if close.iloc[curr] > upperband.iloc[curr]:
                        supertrend.iloc[curr] = upperband.iloc[curr]
                        direction.iloc[curr] = 1
                    else:
                        supertrend.iloc[curr] = lowerband.iloc[curr]
                        direction.iloc[curr] = -1
                else:
                    if close.iloc[curr] < lowerband.iloc[curr]:
                        supertrend.iloc[curr] = lowerband.iloc[curr]
                        direction.iloc[curr] = -1
                    else:
                        supertrend.iloc[curr] = upperband.iloc[curr]
                        direction.iloc[curr] = 1
            
            # Fill NaN values with forward fill
            supertrend = supertrend.ffill()
            direction = direction.ffill()
            
            return supertrend
            
        except Exception as e:
            print(f"Error calculating Supertrend: {e}")
            return pd.Series(index=df.index, dtype=float)

    def calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators including ADX"""
        try:
            # Ensure we have enough data
            if len(df) < 30:
                logger.warning(f"Not enough data for advanced indicators: {len(df)} rows")
                # Initialize with zeros
                df['bb_high'] = df['close']
                df['bb_low'] = df['close']
                df['bb_mid'] = df['close']
                df['atr'] = 0
                df['stoch_k'] = 50
                df['stoch_d'] = 50
                df['supertrend'] = df['close']
                df['adx'] = 0
                return df

            # Bollinger Bands
            try:
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                df['bb_high'] = bb.bollinger_hband().fillna(df['close'])
                df['bb_low'] = bb.bollinger_lband().fillna(df['close'])
                df['bb_mid'] = bb.bollinger_mavg().fillna(df['close'])
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")
                df['bb_high'] = df['close']
                df['bb_low'] = df['close']
                df['bb_mid'] = df['close']
            
            # ATR
            try:
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
                df['atr'] = atr.average_true_range().fillna(0)
            except Exception as e:
                logger.error(f"Error calculating ATR: {e}")
                df['atr'] = 0
            
            # Stochastic Oscillator
            try:
                stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
                df['stoch_k'] = stoch.stoch().fillna(50)
                df['stoch_d'] = stoch.stoch_signal().fillna(50)
            except Exception as e:
                logger.error(f"Error calculating Stochastic: {e}")
                df['stoch_k'] = 50
                df['stoch_d'] = 50
            
            # Supertrend
            try:
                df['supertrend'] = self.calculate_supertrend(df)
                df['supertrend'] = df['supertrend'].fillna(df['close'])
            except Exception as e:
                logger.error(f"Error calculating Supertrend: {e}")
                df['supertrend'] = df['close']
            
            # ADX Calculation - Simplified and robust
            try:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift(1))
                low_close = np.abs(df['low'] - df['close'].shift(1))
                
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                
                # Calculate Directional Movement
                up_move = df['high'] - df['high'].shift(1)
                down_move = df['low'].shift(1) - df['low']
                
                # +DM and -DM
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Convert to Series for easier handling
                tr_series = pd.Series(true_range, index=df.index)
                plus_dm_series = pd.Series(plus_dm, index=df.index)
                minus_dm_series = pd.Series(minus_dm, index=df.index)
                
                # Calculate 14-period smoothed values using simple moving average
                tr14 = tr_series.rolling(window=14, min_periods=1).mean()
                plus_dm14 = plus_dm_series.rolling(window=14, min_periods=1).mean()
                minus_dm14 = minus_dm_series.rolling(window=14, min_periods=1).mean()
                
                # Calculate +DI and -DI
                # Avoid division by zero
                tr14_safe = tr14.replace(0, 0.0001)  # Replace 0 with small number
                plus_di = 100 * (plus_dm14 / tr14_safe)
                minus_di = 100 * (minus_dm14 / tr14_safe)
                
                # Calculate DX
                di_sum = plus_di + minus_di
                di_sum_safe = di_sum.replace(0, 0.0001)  # Avoid division by zero
                dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe
                
                # Calculate ADX (14-period smoothed DX)
                adx = dx.rolling(window=14, min_periods=1).mean()
                
                # Fill any remaining NaN values
                df['adx'] = adx.fillna(0)
                
                # Ensure ADX is within reasonable bounds (0-100)
                df['adx'] = df['adx'].clip(0, 100)
                
                # Logging
                current_adx = df['adx'].iloc[-1]
                logger.info(f"Current ADX value: {current_adx:.2f}")
                
                if current_adx < 5:  # Changed threshold from 0 to 5
                    logger.warning(f"ADX is very low: {current_adx:.2f}")
                    logger.info(f"Last 5 ADX values: {df['adx'].tail().values}")
                    logger.info(f"Last 5 +DI values: {plus_di.tail().values}")
                    logger.info(f"Last 5 -DI values: {minus_di.tail().values}")
                    logger.info(f"Last 5 DX values: {dx.tail().values}")
                
            except Exception as e:
                logger.error(f"Error in ADX calculation: {e}")
                df['adx'] = 25  # Default to moderate trend strength
            
            # Final validation - ensure no NaN or infinite values
            indicator_cols = ['bb_high', 'bb_low', 'bb_mid', 'atr', 'stoch_k', 'stoch_d', 'supertrend', 'adx']
            for col in indicator_cols:
                if col in df.columns:
                    # Replace NaN values
                    if col in ['stoch_k', 'stoch_d']:
                        df[col] = df[col].fillna(50)  # Neutral value for stochastic
                    elif col == 'adx':
                        df[col] = df[col].fillna(25)  # Moderate trend strength
                    elif col == 'atr':
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(df['close'])  # Use close price as fallback
                    
                    # Replace infinite values
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
                    
                    # Ensure reasonable bounds
                    if col in ['stoch_k', 'stoch_d']:
                        df[col] = df[col].clip(0, 100)
                    elif col == 'adx':
                        df[col] = df[col].clip(0, 100)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_advanced_indicators: {e}")
            # Return with default values
            df['bb_high'] = df['close']
            df['bb_low'] = df['close']
            df['bb_mid'] = df['close']
            df['atr'] = 0
            df['stoch_k'] = 50
            df['stoch_d'] = 50
            df['supertrend'] = df['close']
            df['adx'] = 25
            return df

    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        try:
            # 1. 기본 지표 계산
            df = self.calculate_indicators(df)
            
            # 2. 시장 상태 분석
            adx = df['adx'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            
            # 3. 볼륨 분석
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma
            
            # 4. 추세 분석
            ema_short = df['ema_short'].iloc[-1]
            ema_medium = df['ema_medium'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            
            # 5. 시장 상태 분류
            market_condition = "normal"
            if adx > 25:
                if ema_short > ema_medium > ema_long:
                    market_condition = "strong_uptrend"
                elif ema_short < ema_medium < ema_long:
                    market_condition = "strong_downtrend"
            elif adx < 20:
                market_condition = "sideways"
            
            # 6. 신호 생성
            signal = 0
            score = 0
            
            # RSI 기반 신호
            if rsi < 30:
                signal += 1
                score += 1
            elif rsi > 70:
                signal -= 1
                score += 1
                
            # MACD 기반 신호
            if macd > macd_signal:
                signal += 1
                score += 1
            elif macd < macd_signal:
                signal -= 1
                score += 1
                
            # Stochastic 기반 신호
            if stoch_k < 20 and stoch_k > stoch_d:
                signal += 1
                score += 1
            elif stoch_k > 80 and stoch_k < stoch_d:
                signal -= 1
                score += 1
                
            # 볼륨 기반 신호
            if volume_ratio > 1.5:
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    signal += 1
                    score += 1
                else:
                    signal -= 1
                    score += 1
                    
            # 추세 기반 신호
            if market_condition == "strong_uptrend":
                signal += 1
                score += 2
            elif market_condition == "strong_downtrend":
                signal -= 1
                score += 2
                
            # 최종 신호 결정 - 더 유연한 조건
            final_signal = 0
            
            # 1. 기본 강한 신호 (score >= 3)
            if signal > 0 and score >= 3:
                final_signal = 1
            elif signal < 0 and score >= 3:
                final_signal = -1
            
            # 2. 중간 강도 신호 + 강한 추세 (score >= 2, ADX > 30)
            elif signal > 0 and score >= 2 and adx > 30:
                final_signal = 1
            elif signal < 0 and score >= 2 and adx > 30:
                final_signal = -1
            
            # 3. 극단적 RSI 조건 (RSI < 25 or RSI > 75)
            elif signal > 0 and rsi < 25 and score >= 1:
                final_signal = 1
            elif signal < 0 and rsi > 75 and score >= 1:
                final_signal = -1
            
            # 4. 볼륨 급증 + 방향성 일치 (volume > 2x average)
            elif signal != 0 and score >= 2 and volume_ratio > 2.0:
                final_signal = signal
                
            return final_signal, score, adx, market_condition
            
        except Exception as e:
            logger.error(f"Error in generate_signals: {e}")
            return 0, 0, 0, "error"

    def get_latest_signal(self, df):
        """Get the most recent trading signal"""
        df = self.calculate_indicators(df)
        signals = self.generate_signals(df)
        return signals[0]

    def calculate_stop_loss_take_profit(self, df, entry_price, side, lookback=10, min_pct=0.3, market_condition="normal"):
        """
        시장 상황에 맞는 동적 손절/익절 계산
        """
        if len(df) < lookback:
            lookback = len(df)
        if lookback < 2:
            atr = entry_price * 0.005  # fallback
        else:
            atr = df['atr'].tail(lookback).mean()
            
        # 시장 상황별 손절/익절 전략
        if market_condition == "crash":  # 폭락장 - 역추세 매수
            # 빠른 익절, 넓은 손절 (반등 기대)
            if side == 'BUY':
                stop_loss = entry_price - atr * 3.0  # 넓은 손절
                take_profit = entry_price + atr * 1.5  # 빠른 익절
            else:
                stop_loss = entry_price + atr * 1.5
                take_profit = entry_price - atr * 3.0
                
        elif market_condition == "pump":  # 폭등장 - 역추세 매도
            # 빠른 익절, 넓은 손절 (하락 기대)
            if side == 'BUY':
                stop_loss = entry_price - atr * 1.5
                take_profit = entry_price + atr * 3.0
            else:
                stop_loss = entry_price + atr * 3.0  # 넓은 손절
                take_profit = entry_price - atr * 1.5  # 빠른 익절
                
        elif market_condition == "sideways":  # 횡보장 - 볼린저 밴드 전략
            # 좁은 손절/익절 (빠른 회전)
            if side == 'BUY':
                stop_loss = entry_price - atr * 1.0
                take_profit = entry_price + atr * 1.5
            else:
                stop_loss = entry_price + atr * 1.0
                take_profit = entry_price - atr * 1.5
                
        elif market_condition == "strong_trend":  # 강한 추세장 - 추세 추종
            # 넓은 손절, 큰 익절 (추세 지속 기대)
            if side == 'BUY':
                stop_loss = entry_price - atr * 2.0
                take_profit = entry_price + atr * 4.0
            else:
                stop_loss = entry_price + atr * 2.0
                take_profit = entry_price - atr * 4.0
                
        else:  # 일반 시장
            # 기본 전략
            if side == 'BUY':
                stop_loss = entry_price - atr * 1.5
                take_profit = entry_price + atr * 2.5
            else:
                stop_loss = entry_price + atr * 1.5
                take_profit = entry_price - atr * 2.5
        
        # 최소폭 강제 (0.3%)
        min_dist = entry_price * min_pct / 100
        if side == 'BUY':
            stop_loss = min(stop_loss, entry_price - min_dist)
            take_profit = max(take_profit, entry_price + min_dist * 2)
        else:
            stop_loss = max(stop_loss, entry_price + min_dist)
            take_profit = min(take_profit, entry_price - min_dist * 2)
            
        return stop_loss, take_profit

    def generate_comprehensive_signal(self, df, return_details=False):
        """Generate comprehensive trading signal combining volume and all technical indicators"""
        # First calculate all indicators
        df = self.calculate_indicators(df)
        indicator_cols = ['ema_short','ema_medium','ema_long','rsi','macd','macd_signal','macd_diff','bb_high','bb_low','stoch_k','stoch_d','atr','supertrend','adx']
        if df[indicator_cols].isnull().any().any():
            return (0, 0, np.nan) if return_details else 0
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # 1. 거래량 분석
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        current_volume = df['volume'].iloc[-1]
        is_volume_spike = current_volume > (volume_ma.iloc[-1] + 2 * volume_std.iloc[-1])
        
        # 2. 가격 추세 분석
        price_trend = 0
        if df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1]:
            price_trend = 1  # 상승 추세
        elif df['ema_short'].iloc[-1] < df['ema_medium'].iloc[-1] < df['ema_long'].iloc[-1]:
            price_trend = -1  # 하락 추세
            
        # 3. 모멘텀 분석
        momentum = 0
        if df['rsi'].iloc[-1] < 30 and df['stoch_k'].iloc[-1] < 20:
            momentum = 1  # 과매도
        elif df['rsi'].iloc[-1] > 70 and df['stoch_k'].iloc[-1] > 80:
            momentum = -1  # 과매수
            
        # 4. MACD 분석
        macd_signal = 0
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd_diff'].iloc[-1] > 0:
            macd_signal = 1
        elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd_diff'].iloc[-1] < 0:
            macd_signal = -1
            
        # 5. 볼린저 밴드 분석
        bb_signal = 0
        if df['close'].iloc[-1] < df['bb_low'].iloc[-1]:
            bb_signal = 1  # 하단 돌파
        elif df['close'].iloc[-1] > df['bb_high'].iloc[-1]:
            bb_signal = -1  # 상단 돌파
            
        # 6. Supertrend 분석
        supertrend_signal = 0
        if not np.isnan(df['supertrend'].iloc[-1]):
            if df['supertrend'].iloc[-1] < df['close'].iloc[-1]:
                supertrend_signal = 1
            else:
                supertrend_signal = -1
                
        # 7. ATR 기반 변동성 분석
        atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(window=20).mean().iloc[-1]
        is_high_volatility = atr > avg_atr * 1.5
        
        # 8. ADX(추세 강도) 및 시장 상황 분석
        adx = df['adx'].iloc[-1]
        
        # 9. 급격한 가격 변동 감지 (폭락/폭등)
        price_change_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100 if len(df) >= 2 else 0
        price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else 0
        price_change_15m = (df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16] * 100 if len(df) >= 16 else 0
        
        # 시장 상황 분석 및 분류
        market_condition = "normal"
        
        # 폭락/폭등 조건 (1분간 1% 이상 또는 5분간 3% 이상 변동)
        if abs(price_change_1m) >= 1.0 or abs(price_change_5m) >= 3.0:
            if price_change_1m > 0 or price_change_5m > 0:
                market_condition = "pump"  # 폭등
            else:
                market_condition = "crash"  # 폭락
        # 횡보장 조건 (ADX < 25 AND 변동성 낮음)
        elif adx < 25 and atr < 0.02:
            market_condition = "sideways"  # 횡보
        # 강한 추세 조건
        elif adx >= 40:
            market_condition = "strong_trend"  # 강한 추세
        
        # 종합 신호 생성
        signal_score = 0
        
        # 거래량 급증이 있는 경우 가중치 증가
        if is_volume_spike:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # 거래량 급증 + 상승
                signal_score += 2
            else:  # 거래량 급증 + 하락
                signal_score -= 2
                
        # 각 지표별 점수 합산 (가중치 조정)
        signal_score += price_trend * 1.5  # 가격 추세 가중치 증가
        signal_score += momentum * 1.2     # 모멘텀 가중치 증가
        signal_score += macd_signal * 1.2  # MACD 가중치 증가
        signal_score += bb_signal * 1.5    # 볼린저 밴드 가중치 증가
        signal_score += supertrend_signal * 1.3  # Supertrend 가중치 증가
        
        # ADX 기반 점수 보정 (추세 강도 반영)
        if adx >= 40:  # 매우 강한 추세
            if price_trend == 1:  # 상승 추세
                signal_score += 2
            elif price_trend == -1:  # 하락 추세
                signal_score -= 2
        elif adx >= 30:  # 강한 추세
            if price_trend == 1:  # 상승 추세
                signal_score += 1
            elif price_trend == -1:  # 하락 추세
                signal_score -= 1
                
        # RSI 극단값 보정
        if df['rsi'].iloc[-1] < 25:  # 극단적 과매도
            signal_score += 1.5
        elif df['rsi'].iloc[-1] > 75:  # 극단적 과매수
            signal_score -= 1.5
            
        # 시장 상황별 진입 조건 조정
        if market_condition == "crash":
            # 폭락 시: 역추세 매수 기회 포착 (더 관대한 조건)
            threshold_long = 2  # 매수 임계값 완화
            threshold_short = 5  # 매도 임계값 강화 (추가 하락 방지)
            
            # 폭락 시 추가 매수 신호 (RSI 극도 과매도)
            if df['rsi'].iloc[-1] < 25:
                signal_score += 2
                
        elif market_condition == "pump":
            # 폭등 시: 역추세 매도 기회 포착 (더 관대한 조건)
            threshold_long = 5  # 매수 임계값 강화 (추가 상승 방지)
            threshold_short = 2  # 매도 임계값 완화
            
            # 폭등 시 추가 매도 신호 (RSI 극도 과매수)
            if df['rsi'].iloc[-1] > 75:
                signal_score -= 2
                
        elif market_condition == "sideways":
            # 횡보 시: 볼린저 밴드 터치 시에만 진입 (더 엄격한 조건)
            threshold_long = 3  # 기본보다 완화
            threshold_short = 3
            
            # 횡보 시에는 볼린저 밴드 신호에 가중치 추가
            if bb_signal != 0:
                signal_score += bb_signal * 2  # 볼린저 밴드 신호 강화
                
        elif market_condition == "strong_trend":
            # 강한 추세 시: 추세 추종 (기본 조건)
            threshold_long = 3  # 기본보다 완화
            threshold_short = 3
            
            # 강한 추세 시 추세 방향 신호 강화
            if adx >= 40:  # 매우 강한 추세
                if price_trend == 1:
                    signal_score += 2  # 상승 추세 강화
                elif price_trend == -1:
                    signal_score -= 2  # 하락 추세 강화
        else:
            # 일반 시장 조건
            threshold_long = 4
            threshold_short = 4
        
        # 최종 신호 결정
        if signal_score >= threshold_long:
            result = 1
        elif signal_score <= -threshold_short:
            result = -1
        else:
            result = 0
            
        # 로깅용 추가 정보
        market_info = {
            'condition': market_condition,
            'price_change_1m': price_change_1m,
            'price_change_5m': price_change_5m,
            'price_change_15m': price_change_15m,
            'threshold_long': threshold_long,
            'threshold_short': threshold_short,
            'adx': adx,
            'volatility': is_high_volatility
        }
        
        if return_details:
            return (result, signal_score, adx, market_info)
        return result 

    def analyze(self, df):
        """Analyze market conditions and generate trading signals"""
        try:
            if df.empty:
                return {'signal': 0, 'score': 0, 'trend': 'neutral'}
            
            # Calculate indicators if not present
            if 'rsi' not in df.columns:
                df = self.calculate_indicators(df)
            
            # Get latest values
            current = df.iloc[-1]
            
            # Analyze trend based on EMAs
            trend = 'neutral'
            if current['ema_short'] > current['ema_medium'] > current['ema_long']:
                trend = 'bullish'
            elif current['ema_short'] < current['ema_medium'] < current['ema_long']:
                trend = 'bearish'
            
            # Analyze volume
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current['volume'] / volume_ma if volume_ma > 0 else 1
            
            # Analyze volatility
            atr = current['atr']
            atr_ratio = atr / current['close'] if current['close'] > 0 else 0
            
            # ADX 확인 (추세 강도) - 먼저 정의
            adx = current.get('adx', 25)
            
            # Generate signal based on indicators (추세 추종 강화)
            signal_count = 0
            total_score = 0
            
            # RSI signals (하락장에서 더 민감하게)
            if current['rsi'] < 25:  # 극도 과매도
                signal_count += 1
                total_score += 0.4
            elif current['rsi'] > 75:  # 극도 과매수
                signal_count -= 1
                total_score += 0.4
            elif current['rsi'] < 35:  # 과매도
                signal_count += 0.5
                total_score += 0.2
            elif current['rsi'] > 65:  # 과매수
                signal_count -= 0.5
                total_score += 0.2
            # 하락 추세에서 RSI 중간값도 매도 신호로 활용
            elif trend == 'bearish' and current['rsi'] > 50:
                signal_count -= 0.3
                total_score += 0.15
            
            # MACD signals (크로스오버와 히스토그램 모두 확인)
            macd_diff = current['macd'] - current['macd_signal']
            if len(df) > 1:
                prev_macd_diff = df['macd'].iloc[-2] - df['macd_signal'].iloc[-2]
                # MACD 크로스오버 확인
                if macd_diff > 0 and prev_macd_diff <= 0:  # 골든 크로스
                    signal_count += 1
                    total_score += 0.3
                elif macd_diff < 0 and prev_macd_diff >= 0:  # 데드 크로스
                    signal_count -= 1
                    total_score += 0.3
                elif macd_diff > 0:  # MACD가 시그널 위에 있음
                    signal_count += 0.3
                    total_score += 0.1
                elif macd_diff < 0:  # MACD가 시그널 아래에 있음
                    signal_count -= 0.3
                    total_score += 0.1
            
            # EMA signals (3개 EMA 모두 확인) - 추세 추종 강화
            if current['ema_short'] > current['ema_medium'] > current['ema_long']:
                signal_count += 1
                total_score += 0.3
            elif current['ema_short'] < current['ema_medium'] < current['ema_long']:
                signal_count -= 1
                total_score += 0.3  # 하락 추세 신호 강화
            elif current['ema_short'] > current['ema_medium']:
                signal_count += 0.5
                total_score += 0.15
            elif current['ema_short'] < current['ema_medium']:
                signal_count -= 0.5
                total_score += 0.15
            
            # Stochastic signals (더 극단적인 값에서만)
            if current['stoch_k'] < 15 and current['stoch_d'] < 15:
                signal_count += 1
                total_score += 0.25
            elif current['stoch_k'] > 85 and current['stoch_d'] > 85:
                signal_count -= 1
                total_score += 0.25
            elif current['stoch_k'] < 25 and current['stoch_d'] < 25:
                signal_count += 0.5
                total_score += 0.15
            elif current['stoch_k'] > 75 and current['stoch_d'] > 75:
                signal_count -= 0.5
                total_score += 0.15
            
            # 강한 추세에서 추가 신호 강화
            if trend == 'bearish' and adx > 50:  # 강한 하락 추세
                signal_count -= 0.8  # 추가 매도 신호
                total_score += 0.3
            elif trend == 'bullish' and adx > 50:  # 강한 상승 추세
                signal_count += 0.8  # 추가 매수 신호
                total_score += 0.3
            
            # ADX 기반 신호 조정
            if adx < 20:  # 약한 추세 - 신호 무효화
                signal_count = 0
                total_score = 0
            elif adx < 30:  # 보통 추세 - 신호 약화
                signal_count *= 0.7
                total_score *= 0.7
            elif adx >= 40:  # 강한 추세 - 신호 강화
                signal_count *= 1.3
                total_score *= 1.3
            
            # Volume confirmation (거래량 확인)
            if volume_ratio > 1.8:  # 높은 거래량
                total_score *= 1.3
            elif volume_ratio > 1.3:  # 보통 거래량
                total_score *= 1.1
            elif volume_ratio < 0.7:  # 낮은 거래량 - 신호 약화
                total_score *= 0.7
            
            # Volatility adjustment (변동성 조정) - 추세장에서는 덜 보수적
            if atr_ratio > 0.03:  # 높은 변동성
                if adx > 40:  # 강한 추세에서는 변동성 페널티 완화
                    total_score *= 0.8
                else:
                    total_score *= 0.6
            elif atr_ratio > 0.02:  # 보통 변동성
                total_score *= 0.9
            
            # 최종 신호 결정 (추세 + 지표 확인 방식)
            logger.info(f"🔍 Technical Analysis Debug:")
            logger.info(f"   Signal Count: {signal_count:.2f}")
            logger.info(f"   Total Score: {total_score:.3f}")
            logger.info(f"   ADX: {adx:.2f}, Trend: {trend}")
            logger.info(f"   RSI: {current['rsi']:.2f}")
            logger.info(f"   MACD: {current['macd']:.2f} vs Signal: {current['macd_signal']:.2f}")
            logger.info(f"   EMA: Short({current['ema_short']:.2f}) vs Medium({current['ema_medium']:.2f})")
            logger.info(f"   Volume Ratio: {volume_ratio:.2f}")
            
            # 강한 추세에서는 조건 완화 (하지만 지표 확인은 필수)
            if adx > 50:  # 매우 강한 추세
                if trend == 'bearish' and signal_count <= -0.8 and total_score >= 0.3:
                    signal = -1  # 하락 추세 + 최소 지표 확인
                    logger.info(f"   ✅ STRONG BEARISH TREND SELL! (ADX: {adx:.2f})")
                elif trend == 'bullish' and signal_count >= 0.8 and total_score >= 0.3:
                    signal = 1   # 상승 추세 + 최소 지표 확인
                    logger.info(f"   ✅ STRONG BULLISH TREND BUY! (ADX: {adx:.2f})")
                elif signal_count >= 1.2 and total_score >= 0.4:  # 강한 추세에서 조건 완화
                    signal = 1
                    logger.info(f"   ✅ BUY Signal (Strong Trend)!")
                elif signal_count <= -1.2 and total_score >= 0.4:  # 강한 추세에서 조건 완화
                    signal = -1
                    logger.info(f"   ✅ SELL Signal (Strong Trend)!")
                else:
                    signal = 0
                    logger.info(f"   ❌ No Signal in Strong Trend: Count({signal_count:.2f}) Score({total_score:.3f})")
            else:  # 일반적인 시장 조건
                if signal_count >= 1.5 and total_score >= 0.5:  # 매수 신호
                    signal = 1
                    logger.info(f"   ✅ BUY Signal Generated!")
                elif signal_count <= -1.5 and total_score >= 0.5:  # 매도 신호
                    signal = -1
                    logger.info(f"   ✅ SELL Signal Generated!")
                else:
                    signal = 0  # 신호 없음
                    logger.info(f"   ❌ No Signal: Count({signal_count:.2f}) Score({total_score:.3f})")
            
            # 최종 점수 정규화
            score = min(total_score, 1.0)
            
            # 시장 상황 분석 및 분류
            price_change_1m = (current['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100 if len(df) >= 2 else 0
            price_change_5m = (current['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else 0
            price_change_15m = (current['close'] - df['close'].iloc[-16]) / df['close'].iloc[-16] * 100 if len(df) >= 16 else 0
            
            # 시장 상황 분류
            market_condition = "normal"
            
            # 폭락/폭등 조건 (1분간 1% 이상 또는 5분간 3% 이상 변동)
            if abs(price_change_1m) >= 1.0 or abs(price_change_5m) >= 3.0:
                if price_change_1m > 0 or price_change_5m > 0:
                    market_condition = "pump"  # 폭등
                else:
                    market_condition = "crash"  # 폭락
            # 횡보장 조건 (ADX < 25 AND 변동성 낮음)
            elif adx < 25 and atr_ratio < 0.02:
                market_condition = "sideways"  # 횡보
            # 강한 추세 조건
            elif adx >= 40:
                market_condition = "strong_trend"  # 강한 추세
            
            logger.info(f"   Market Condition: {market_condition}")
            logger.info(f"   Price Changes: 1m({price_change_1m:.2f}%) 5m({price_change_5m:.2f}%)")
            
            # 시장 상황별 신호 생성 로직
            final_signal = 0
            
            if market_condition == "crash":  # 폭락장 - 역추세 매수 기회
                logger.info(f"   🔴 CRASH Market Strategy")
                # 극도 과매도 + 볼린저 밴드 하단 터치 시 매수
                if (current['rsi'] < 25 and 
                    current['close'] < current['bb_low'] and 
                    volume_ratio > 1.5 and
                    total_score >= 0.3):
                    final_signal = 1
                    logger.info(f"   ✅ CRASH REVERSAL BUY! RSI:{current['rsi']:.1f} BB_Low:{current['bb_low']:.1f}")
                else:
                    logger.info(f"   ❌ Crash conditions not met: RSI({current['rsi']:.1f}) BB({current['close']:.1f}>{current['bb_low']:.1f})")
                    
            elif market_condition == "pump":  # 폭등장 - 역추세 매도 기회
                logger.info(f"   🟢 PUMP Market Strategy")
                # 극도 과매수 + 볼린저 밴드 상단 터치 시 매도
                if (current['rsi'] > 75 and 
                    current['close'] > current['bb_high'] and 
                    volume_ratio > 1.5 and
                    total_score >= 0.3):
                    final_signal = -1
                    logger.info(f"   ✅ PUMP REVERSAL SELL! RSI:{current['rsi']:.1f} BB_High:{current['bb_high']:.1f}")
                else:
                    logger.info(f"   ❌ Pump conditions not met: RSI({current['rsi']:.1f}) BB({current['close']:.1f}<{current['bb_high']:.1f})")
                    
            elif market_condition == "sideways":  # 횡보장 - 볼린저 밴드 전략
                logger.info(f"   ↔️ SIDEWAYS Market Strategy")
                # 볼린저 밴드 상하단 터치 + 스토캐스틱 확인
                if (current['close'] < current['bb_low'] and 
                    current['stoch_k'] < 20 and 
                    total_score >= 0.2):
                    final_signal = 1
                    logger.info(f"   ✅ SIDEWAYS BUY at BB_Low! Stoch:{current['stoch_k']:.1f}")
                elif (current['close'] > current['bb_high'] and 
                      current['stoch_k'] > 80 and 
                      total_score >= 0.2):
                    final_signal = -1
                    logger.info(f"   ✅ SIDEWAYS SELL at BB_High! Stoch:{current['stoch_k']:.1f}")
                else:
                    logger.info(f"   ❌ Sideways conditions not met: BB position, Stoch:{current['stoch_k']:.1f}")
                    
            elif market_condition == "strong_trend":  # 강한 추세장 - 추세 추종
                logger.info(f"   📈 STRONG TREND Market Strategy")
                if trend == 'bearish':
                    # 하락 추세: 풀백 후 재진입 또는 지속 하락
                    if (signal_count <= -1.0 and 
                        total_score >= 0.4 and
                        current['rsi'] > 30):  # 너무 과매도되지 않은 상태에서
                        final_signal = -1
                        logger.info(f"   ✅ BEARISH TREND SELL! Count:{signal_count:.2f}")
                    else:
                        logger.info(f"   ❌ Bearish trend conditions not met: Count({signal_count:.2f}) RSI({current['rsi']:.1f})")
                elif trend == 'bullish':
                    # 상승 추세: 풀백 후 재진입 또는 지속 상승
                    if (signal_count >= 1.0 and 
                        total_score >= 0.4 and
                        current['rsi'] < 70):  # 너무 과매수되지 않은 상태에서
                        final_signal = 1
                        logger.info(f"   ✅ BULLISH TREND BUY! Count:{signal_count:.2f}")
                    else:
                        logger.info(f"   ❌ Bullish trend conditions not met: Count({signal_count:.2f}) RSI({current['rsi']:.1f})")
                        
            else:  # 일반 시장 - 기존 로직
                logger.info(f"   📊 NORMAL Market Strategy")
                final_signal = signal  # 위에서 계산한 신호 사용
            
            # 최종 신호 확인
            if final_signal != 0:
                logger.info(f"   🎯 FINAL SIGNAL: {final_signal} in {market_condition} market")
            
            return {
                'signal': final_signal,
                'score': score,
                'trend': trend,
                'volume_ratio': volume_ratio,
                'volatility': atr_ratio,
                'adx': adx,  # ADX 값 추가
                'market_condition': market_condition  # 시장 상황 추가
            }
            
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {'signal': 0, 'score': 0, 'trend': 'neutral'} 