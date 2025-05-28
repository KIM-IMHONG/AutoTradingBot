import logging
import asyncio
from datetime import datetime, timedelta
import aiohttp
from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TRADING_SYMBOLS
)
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        """Initialize news collector"""
        self.logger = logging.getLogger(__name__)
        self.news_cache = {}
        self.last_update = {}
        self.update_interval = 300  # 5분마다 업데이트
        self.news_threshold = 0.7  # 뉴스 영향도 임계값
        
        # Initialize cache for each trading symbol
        for symbol in TRADING_SYMBOLS:
            self.news_cache[symbol] = {
                'news': [],
                'last_update': datetime.now()
            }
    
    async def monitor_news(self, callback):
        """Monitor news updates and notify through callback"""
        while True:
            try:
                # Update news for all symbols
                await self.update_news()
                
                # Process news impact for each symbol
                for symbol in TRADING_SYMBOLS:
                    news = self.get_news(symbol)
                    if news:
                        # Calculate combined impact
                        combined_impact = self.calculate_news_impact(news)
                        
                        # Calculate detailed impacts
                        detailed_impacts = {
                            'crypto': self.calculate_crypto_impact(news),
                            'nasdaq': self.calculate_nasdaq_impact(news)
                        }
                        
                        # Notify through callback if impact is significant
                        if abs(combined_impact) > self.news_threshold:
                            await callback(combined_impact, detailed_impacts)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor_news: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def get_news(self, symbol):
        """Get relevant news for the trading symbol"""
        try:
            current_time = datetime.now()
            
            # 캐시된 뉴스가 있고 업데이트 간격이 지나지 않았다면 캐시된 뉴스 반환
            if symbol in self.news_cache:
                cache = self.news_cache[symbol]
                if current_time - cache['last_update'] < timedelta(seconds=self.update_interval):
                    return cache['news']
            
            # 기본 뉴스 정보 반환
            news = {
                'timestamp': current_time,
                'headlines': [],
                'sentiment': 'neutral',
                'impact': 0.0,
                'title': f"Sample news for {symbol}"
            }
            
            # 뉴스 캐시 업데이트
            if symbol not in self.news_cache:
                self.news_cache[symbol] = {'news': [], 'last_update': current_time}
            
            self.news_cache[symbol]['news'].append(news)
            self.news_cache[symbol]['last_update'] = current_time
            
            # Keep only recent news (last 10 entries)
            self.news_cache[symbol]['news'] = self.news_cache[symbol]['news'][-10:]
            
            return self.news_cache[symbol]['news']
            
        except Exception as e:
            self.logger.error(f"Error getting news for {symbol}: {e}")
            return []
            
    async def update_news(self):
        """Update news for all trading symbols"""
        try:
            for symbol in TRADING_SYMBOLS:
                # Create a sample news entry (실제로는 뉴스 API를 통해 가져와야 함)
                news_entry = {
                    'title': f"Sample news for {symbol}",
                    'content': f"This is a sample news content for {symbol}",
                    'timestamp': datetime.now(),
                    'impact': np.random.uniform(-0.1, 0.1)  # Random impact between -0.1 and 0.1
                }
                
                # Update cache
                if symbol not in self.news_cache:
                    self.news_cache[symbol] = {'news': [], 'last_update': datetime.now()}
                
                self.news_cache[symbol]['news'].append(news_entry)
                self.news_cache[symbol]['last_update'] = datetime.now()
                
                # Keep only recent news (last 10 entries)
                self.news_cache[symbol]['news'] = self.news_cache[symbol]['news'][-10:]
                
        except Exception as e:
            logger.error(f"Error updating news: {e}")
            
    def calculate_news_impact(self, news):
        """Calculate combined impact of news"""
        if not news:
            return 0
        
        # Calculate weighted average impact
        total_weight = 0
        weighted_impact = 0
        
        for entry in news:
            # More recent news has higher weight
            time_diff = (datetime.now() - entry['timestamp']).total_seconds()
            weight = 1 / (1 + time_diff / 3600)  # Weight decreases with time
            
            impact = entry.get('impact', 0)
            weighted_impact += impact * weight
            total_weight += weight
        
        return weighted_impact / total_weight if total_weight > 0 else 0
    
    def calculate_crypto_impact(self, news):
        """Calculate crypto-specific news impact"""
        if not news:
            return 0
        
        # Filter crypto-related news and calculate impact
        crypto_news = [entry for entry in news if 'crypto' in entry.get('title', '').lower()]
        return self.calculate_news_impact(crypto_news)
    
    def calculate_nasdaq_impact(self, news):
        """Calculate NASDAQ-related news impact"""
        if not news:
            return 0
        
        # Filter NASDAQ-related news and calculate impact
        nasdaq_news = [entry for entry in news if 'nasdaq' in entry.get('title', '').lower()]
        return self.calculate_news_impact(nasdaq_news)
    
    def get_market_sentiment(self, symbol):
        """Get market sentiment based on news"""
        news = self.get_news(symbol)
        impact = self.calculate_news_impact(news)
        
        if impact > 0.5:
            return "bullish"
        elif impact < -0.5:
            return "bearish"
        else:
            return "neutral" 