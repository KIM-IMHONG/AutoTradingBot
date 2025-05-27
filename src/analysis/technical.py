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
    def __init__(self):
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
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell

        # 볼린저밴드 + Stochastic + Supertrend + RSI 조합
        # 롱: 볼린저 하단 돌파 + Stoch < 20 + Supertrend 매수 + RSI < 35
        # 숏: 볼린저 상단 돌파 + Stoch > 80 + Supertrend 매도 + RSI > 65
        last = df.iloc[-1]
        long_cond = (
            last['close'] < last['bb_low'] and
            last['stoch_k'] < 20 and
            (np.isnan(last['supertrend']) or last['supertrend'] < last['close']) and
            last['rsi'] < 35
        )
        short_cond = (
            last['close'] > last['bb_high'] and
            last['stoch_k'] > 80 and
            (np.isnan(last['supertrend']) or last['supertrend'] > last['close']) and
            last['rsi'] > 65
        )
        if long_cond:
            signals.iloc[-1, signals.columns.get_loc('signal')] = 1
        elif short_cond:
            signals.iloc[-1, signals.columns.get_loc('signal')] = -1
        # 기존 신호와 결합(majority voting)
        base_signals = pd.DataFrame(index=df.index)
        base_signals['signal'] = 0
        base_signals.loc[df['rsi'] < RSI_OVERSOLD, 'signal'] = 1
        base_signals.loc[df['rsi'] > RSI_OVERBOUGHT, 'signal'] = -1
        base_signals.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        base_signals.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
        base_signals.loc[
            (df['ema_short'] > df['ema_medium']) & (df['ema_medium'] > df['ema_long']), 'signal'] = 1
        base_signals.loc[
            (df['ema_short'] < df['ema_medium']) & (df['ema_medium'] < df['ema_long']), 'signal'] = -1
        signals['base_signal'] = base_signals['signal']
        signals['final_signal'] = signals[['signal', 'base_signal']].sum(axis=1).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        return signals

    def get_latest_signal(self, df):
        """Get the most recent trading signal"""
        df = self.calculate_indicators(df)
        signals = self.generate_signals(df)
        return signals['final_signal'].iloc[-1]

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
        
        # 8. ADX(추세 강도) 필터
        adx = df['adx'].iloc[-1]
        if adx < 20:
            if return_details:
                return (0, 0, adx)
            return 0  # 횡보장에서는 신호 무시
        
        # 종합 신호 생성
        signal_score = 0
        
        # 거래량 급증이 있는 경우 가중치 증가
        if is_volume_spike:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # 거래량 급증 + 상승
                signal_score += 2
            else:  # 거래량 급증 + 하락
                signal_score -= 2
                
        # 각 지표별 점수 합산
        signal_score += price_trend
        signal_score += momentum
        signal_score += macd_signal
        signal_score += bb_signal
        signal_score += supertrend_signal
        
        # 변동성이 높은 경우 신호 임계값 상향 조정
        threshold = 4 if is_high_volatility else 4
        
        # 최종 신호 결정
        if signal_score >= threshold:
            result = 1
        elif signal_score <= -threshold:
            result = -1
        else:
            result = 0
        if return_details:
            return (result, signal_score, adx)
        return result 