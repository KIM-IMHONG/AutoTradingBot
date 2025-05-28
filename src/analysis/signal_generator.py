import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, symbol: str):
        """Initialize the signal generator"""
        self.symbol = symbol
        self.last_signal = 0
        self.signal_history = []
        self.signal_history_limit = 5
        self.required_signal_confirmation = 1
        self.signal_confirmation_count = 0

    def generate_signal(self, technical_analysis: dict, sentiment_score: float, current_price: float) -> Tuple[int, float, float]:
        """
        Generate trading signal based on technical analysis and sentiment
        
        Args:
            technical_analysis (dict): Dictionary containing technical analysis results
            sentiment_score (float): Sentiment score from news analysis (-1 to 1)
            current_price (float): Current price of the asset
            
        Returns:
            Tuple[int, float, float]: (signal, score, adx)
                signal: 1 for buy, -1 for sell, 0 for no signal
                score: signal strength (0-1)
                adx: Average Directional Index value
        """
        try:
            # Validate inputs
            if not isinstance(technical_analysis, dict):
                logger.error("Invalid technical_analysis input")
                return 0, 0, 0
            
            if not isinstance(sentiment_score, (int, float)):
                logger.error("Invalid sentiment_score input")
                sentiment_score = 0
            
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                logger.error("Invalid current_price input")
                return 0, 0, 0

            # Extract technical analysis components with defaults
            tech_signal = technical_analysis.get('signal', 0)
            tech_score = technical_analysis.get('score', 0)
            trend = technical_analysis.get('trend', 'neutral')
            volume_ratio = technical_analysis.get('volume_ratio', 1.0)
            volatility = technical_analysis.get('volatility', 0.02)
            market_condition = technical_analysis.get('market_condition', 'normal')
            
            # Get ADX from technical analysis or use default
            adx = 25  # Default moderate trend strength
            if 'adx' in technical_analysis:
                adx = technical_analysis['adx']
            
            # Ensure ADX is valid
            if not isinstance(adx, (int, float)) or np.isnan(adx) or adx < 0:
                adx = 25  # Default value
            
            # Initialize signal components
            signal = tech_signal
            score = tech_score
            
            # Validate and normalize score
            if not isinstance(score, (int, float)) or np.isnan(score):
                score = 0
            score = max(0, min(score, 1.0))  # Clamp between 0 and 1
            
            # Validate signal
            if not isinstance(signal, (int, float)):
                signal = 0
            signal = int(np.sign(signal))  # Ensure signal is -1, 0, or 1
            
            # Sentiment adjustment
            sentiment_score = max(-1, min(sentiment_score, 1))  # Clamp sentiment between -1 and 1
            
            # 시장 상황별 신호 조정
            logger.info(f"🎯 Market Condition: {market_condition}")
            
            if market_condition == "crash":  # 폭락장
                logger.info(f"   🔴 CRASH Market - Contrarian Buy Strategy")
                # 역추세 매수 전략 - 감정 점수 반대로 활용
                if signal > 0 and sentiment_score < -0.3:  # 매수 신호 + 부정적 감정
                    score *= 1.5  # 신호 강화
                    logger.info(f"   ✅ Crash reversal buy signal boosted!")
                elif signal < 0:  # 매도 신호는 약화
                    score *= 0.3
                    logger.info(f"   ⚠️ Sell signal weakened in crash")
                    
            elif market_condition == "pump":  # 폭등장
                logger.info(f"   🟢 PUMP Market - Contrarian Sell Strategy")
                # 역추세 매도 전략 - 감정 점수 반대로 활용
                if signal < 0 and sentiment_score > 0.3:  # 매도 신호 + 긍정적 감정
                    score *= 1.5  # 신호 강화
                    logger.info(f"   ✅ Pump reversal sell signal boosted!")
                elif signal > 0:  # 매수 신호는 약화
                    score *= 0.3
                    logger.info(f"   ⚠️ Buy signal weakened in pump")
                    
            elif market_condition == "sideways":  # 횡보장
                logger.info(f"   ↔️ SIDEWAYS Market - Range Trading Strategy")
                # 볼린저 밴드 전략 - 빠른 회전
                if abs(sentiment_score) < 0.2:  # 중립적 감정에서만
                    score *= 1.2  # 신호 강화
                    logger.info(f"   ✅ Range trading signal in neutral sentiment")
                else:
                    score *= 0.7  # 강한 감정에서는 약화
                    logger.info(f"   ⚠️ Signal weakened due to strong sentiment in sideways")
                    
            elif market_condition == "strong_trend":  # 강한 추세장
                logger.info(f"   📈 STRONG TREND Market - Trend Following Strategy")
                # 추세 추종 전략 - 감정과 추세 일치 시 강화
                if trend == 'bullish' and signal > 0 and sentiment_score > 0:
                    score *= 1.4  # 상승 추세 + 매수 신호 + 긍정 감정
                    logger.info(f"   ✅ Bullish trend signal boosted!")
                elif trend == 'bearish' and signal < 0 and sentiment_score < 0:
                    score *= 1.4  # 하락 추세 + 매도 신호 + 부정 감정
                    logger.info(f"   ✅ Bearish trend signal boosted!")
                elif (trend == 'bullish' and signal < 0) or (trend == 'bearish' and signal > 0):
                    score *= 0.4  # 추세 반대 신호는 크게 약화
                    logger.info(f"   ⚠️ Counter-trend signal heavily weakened")
                    
            else:  # 일반 시장
                logger.info(f"   📊 NORMAL Market - Standard Strategy")
                # 기존 감정 조정 로직
                if abs(sentiment_score) > 0.3:  # Only apply if sentiment is significant
                    if sentiment_score > 0.5:  # Strong positive sentiment
                        if signal > 0:  # Reinforce buy signal
                            score *= 1.3
                        elif signal < 0:  # Reduce sell signal
                            score *= 0.7
                    elif sentiment_score < -0.5:  # Strong negative sentiment
                        if signal < 0:  # Reinforce sell signal
                            score *= 1.3
                        elif signal > 0:  # Reduce buy signal
                            score *= 0.7
            
            # Volume confirmation
            if volume_ratio > 1.5:  # High volume
                score *= 1.1
            elif volume_ratio < 0.8:  # Low volume
                score *= 0.9
            
            # Volatility adjustment - 시장 상황별로 다르게 적용
            if market_condition in ["crash", "pump"]:
                # 폭락/폭등장에서는 변동성 페널티 완화
                if volatility > 0.05:
                    score *= 0.9  # 기존 0.8에서 완화
            else:
                # 일반/횡보/추세장에서는 기존 로직
                if volatility > 0.05:  # High volatility - be more cautious
                    score *= 0.8
                elif volatility < 0.01:  # Low volatility - less opportunity
                    score *= 0.9
            
            # Trend confirmation - 시장 상황별로 다르게 적용
            if market_condition not in ["crash", "pump"]:  # 폭락/폭등장이 아닐 때만
                if trend == 'bullish' and signal > 0:
                    score *= 1.2
                elif trend == 'bearish' and signal < 0:
                    score *= 1.2
                elif trend != 'neutral' and signal != 0:
                    # Signal against trend - reduce confidence
                    if (trend == 'bullish' and signal < 0) or (trend == 'bearish' and signal > 0):
                        score *= 0.6
            
            # ADX-based signal filtering - 시장 상황별로 조정
            if market_condition == "sideways":
                # 횡보장에서는 ADX 조건 완화
                if adx < 15:  # 매우 약한 추세
                    signal = 0
                    score = 0
                elif adx < 25:  # 약한 추세
                    score *= 0.8
            else:
                # 기존 ADX 필터링
                if adx < 20:  # Weak trend - no signal
                    signal = 0
                    score = 0
                elif adx < 30:  # Moderate trend - reduce confidence
                    score *= 0.7
                elif adx >= 40:  # Very strong trend - boost confidence
                    score *= 1.2
            
            # 시장 상황별 최종 임계값 조정
            min_score_threshold = 0.4  # 기본값
            if market_condition == "crash" and signal > 0:  # 폭락장 매수
                min_score_threshold = 0.3
            elif market_condition == "pump" and signal < 0:  # 폭등장 매도
                min_score_threshold = 0.3
            elif market_condition == "sideways":  # 횡보장
                min_score_threshold = 0.25
            elif market_condition == "strong_trend":  # 강한 추세장
                min_score_threshold = 0.35
            
            # 신호 강도가 임계값보다 낮으면 무시
            if score < min_score_threshold:
                signal = 0
                score = 0
                logger.info(f"   ❌ Signal filtered out: score({score:.3f}) < threshold({min_score_threshold})")
            
            # 감정 점수와 신호 충돌 체크 - 시장 상황별로 다르게 적용
            if market_condition not in ["crash", "pump"]:  # 폭락/폭등장이 아닐 때만
                if abs(sentiment_score) > 0.5:
                    if (signal > 0 and sentiment_score < -0.5) or (signal < 0 and sentiment_score > 0.5):
                        signal = 0
                        score = 0
                        logger.info(f"   ❌ Signal invalidated due to sentiment conflict")
            
            # Final score normalization
            score = max(0, min(score, 1.0))
            
            # Update signal history
            self.signal_history.append((signal, score, adx))
            if len(self.signal_history) > self.signal_history_limit:
                self.signal_history.pop(0)
            
            # Log signal generation details
            logger.info(f"🎯 Final Signal: {signal}, Score: {score:.3f}, ADX: {adx:.2f}")
            logger.info(f"   Market: {market_condition}, Trend: {trend}, Sentiment: {sentiment_score:.2f}")
            
            return signal, score, adx

        except Exception as e:
            logger.error(f"Error in generate_signal: {e}")
            return 0, 0, 25  # Return safe defaults

    def confirm_signal(self, signal: int, score: float, adx: float) -> bool:
        """
        Confirm if the signal is valid based on history and confirmation count
        
        Args:
            signal (int): Current signal
            score (float): Signal strength
            adx (float): ADX value
            
        Returns:
            bool: True if signal is confirmed, False otherwise
        """
        if signal == 0:
            return False

        # Check if we have enough history
        if len(self.signal_history) < self.required_signal_confirmation:
            return False

        # Count consecutive signals in the same direction
        consecutive_count = 0
        for hist_signal, _, _ in reversed(self.signal_history):
            if hist_signal == signal:
                consecutive_count += 1
            else:
                break

        # Signal is confirmed if we have enough consecutive signals
        return consecutive_count >= self.required_signal_confirmation

    def get_signal_strength(self) -> float:
        """
        Calculate the overall signal strength based on history
        
        Returns:
            float: Signal strength (0-1)
        """
        if not self.signal_history:
            return 0

        # Calculate average score from history
        scores = [score for _, score, _ in self.signal_history]
        return sum(scores) / len(scores)

    def reset(self):
        """Reset signal generator state"""
        self.last_signal = 0
        self.signal_history = []
        self.signal_confirmation_count = 0 