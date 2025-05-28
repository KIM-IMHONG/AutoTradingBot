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
            
            # Apply sentiment boost/penalty
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
            
            # Volatility adjustment
            if volatility > 0.05:  # High volatility - be more cautious
                score *= 0.8
            elif volatility < 0.01:  # Low volatility - less opportunity
                score *= 0.9
            
            # Trend confirmation
            if trend == 'bullish' and signal > 0:
                score *= 1.2
            elif trend == 'bearish' and signal < 0:
                score *= 1.2
            elif trend != 'neutral' and signal != 0:
                # Signal against trend - reduce confidence
                if (trend == 'bullish' and signal < 0) or (trend == 'bearish' and signal > 0):
                    score *= 0.6
            
            # ADX-based signal filtering (더 엄격한 조건)
            if adx < 20:  # Weak trend - no signal
                signal = 0
                score = 0
            elif adx < 30:  # Moderate trend - reduce confidence significantly
                score *= 0.6
            elif adx >= 30 and adx < 40:  # Strong trend - normal confidence
                score *= 1.0
            elif adx >= 40:  # Very strong trend - boost confidence
                score *= 1.3
            
            # 추가 보수적 필터링
            # 신호 강도가 너무 낮으면 무시
            if score < 0.4:
                signal = 0
                score = 0
            
            # 감정 점수와 신호가 크게 반대되면 신호 무효화
            if abs(sentiment_score) > 0.5:
                if (signal > 0 and sentiment_score < -0.5) or (signal < 0 and sentiment_score > 0.5):
                    signal = 0
                    score = 0
            
            # Final score normalization
            score = max(0, min(score, 1.0))
            
            # Update signal history
            self.signal_history.append((signal, score, adx))
            if len(self.signal_history) > self.signal_history_limit:
                self.signal_history.pop(0)
            
            # Log signal generation details
            logger.debug(f"Signal generated - Tech: {tech_signal}, Sentiment: {sentiment_score:.2f}, "
                        f"Final: {signal}, Score: {score:.2f}, ADX: {adx:.2f}, Trend: {trend}")
            
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