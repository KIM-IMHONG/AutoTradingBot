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
    EMA_SHORT, EMA_MEDIUM, EMA_LONG
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
            # 최소 데이터 포인트 확인
            min_data_points = max(50, max(RSI_PERIOD, MACD_SLOW, EMA_LONG))
            if len(df) < min_data_points:
                logger.warning(f"Insufficient data points for indicator calculation. Need at least {min_data_points}, got {len(df)}")
                return df

            # RSI
            self.rsi = RSIIndicator(
                close=df['close'],
                window=RSI_PERIOD
            )
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

            # Advanced indicators
            self.calculate_advanced_indicators(df)

            # NaN 값 처리
            indicator_cols = ['ema_short', 'ema_medium', 'ema_long', 'rsi', 'macd', 'macd_signal', 
                            'macd_diff', 'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'atr', 'supertrend', 'adx']
            
            for col in indicator_cols:
                if col in df.columns:
                    # 앞쪽 NaN을 이전 값으로 채우기
                    df[col] = df[col].ffill()
                    # 뒤쪽 NaN을 다음 값으로 채우기
                    df[col] = df[col].bfill()
                    # 여전히 NaN이 있다면 0으로 채우기
                    df[col] = df[col].fillna(0)

            return df
            
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {e}")
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
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Supertrend (custom implementation)
        df['supertrend'] = self.calculate_supertrend(df)
        
        # ADX (trend strength)
        try:
            from ta.trend import ADXIndicator
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx.adx()
        except Exception:
            df['adx'] = np.nan
            
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

    def calculate_stop_loss_take_profit(self, df, entry_price, side, lookback=10, min_pct=0.3):
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
        price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100 if len(df) >= 6 else 0
        price_change_15m = (df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16] * 100 if len(df) >= 16 else 0
        
        # 시장 상황 분류
        market_condition = "normal"
        
        # 폭락/폭등 조건 (5분간 2% 이상 또는 15분간 4% 이상 변동)
        if abs(price_change_5m) >= 2.0 or abs(price_change_15m) >= 4.0:
            if price_change_5m > 0 or price_change_15m > 0:
                market_condition = "pump"  # 폭등
            else:
                market_condition = "crash"  # 폭락
        # 횡보장 조건 (ADX < 20 AND 변동성 낮음)
        elif adx < 20 and not is_high_volatility:
            market_condition = "sideways"  # 횡보
        # 강한 추세 조건
        elif adx >= 30:
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