import asyncio
import aiohttp
import tweepy
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from datetime import datetime, timedelta
import logging
from config.settings import (
    NEWS_API_KEY, TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
    NEWS_IMPACT_THRESHOLD
)

class NewsCollector:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        self.twitter_client = self._setup_twitter()
        self.logger = logging.getLogger(__name__)
        self.important_keywords = {
            'crypto': [
                'bitcoin', 'crypto', 'blockchain', 'regulation',
                'sec', 'fed', 'inflation', 'interest rate',
                'market', 'trading', 'exchange', 'hack'
            ],
            'nasdaq': [
                'nasdaq', 'stock market', 'tech stocks', 'federal reserve',
                'interest rates', 'inflation', 'economic growth',
                'earnings report', 'tech sector', 'market rally'
            ]
        }
        # Rate limiting 관련 변수 추가
        self.last_api_call_time = None
        self.api_call_interval = 600  # 10분 간격으로 API 호출 제한
        self.rate_limited = False
        self.rate_limit_reset_time = None
        self.cached_impact = {'crypto': 0, 'nasdaq': 0}  # 캐시된 영향도

    def _setup_twitter(self):
        """Setup Twitter API client"""
        try:
            auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
            auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
            return tweepy.API(auth)
        except Exception as e:
            self.logger.warning(f"Failed to setup Twitter client: {e}")
            return None

    def _can_make_api_call(self):
        """API 호출 가능 여부 확인"""
        current_time = datetime.now()
        
        # Rate limit 상태 확인
        if self.rate_limited and self.rate_limit_reset_time:
            if current_time < self.rate_limit_reset_time:
                return False
            else:
                self.rate_limited = False
                self.rate_limit_reset_time = None
        
        # 마지막 호출 시간 확인
        if self.last_api_call_time:
            time_since_last_call = (current_time - self.last_api_call_time).total_seconds()
            if time_since_last_call < self.api_call_interval:
                return False
        
        return True

    async def get_news(self, category='crypto'):
        """Get news from NewsAPI with rate limiting handling"""
        try:
            # API 호출 가능 여부 확인
            if not self._can_make_api_call():
                self.logger.info(f"Skipping news API call due to rate limiting or interval")
                return []
            
            if category == 'crypto':
                query = 'bitcoin OR cryptocurrency OR blockchain'
            else:  # nasdaq
                query = 'nasdaq OR tech stocks OR stock market'

            # API 호출 시간 기록
            self.last_api_call_time = datetime.now()
            
            news = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=5,  # 페이지 크기 줄임 (10 -> 5)
                from_param=(datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%d')  # 최근 6시간만
            )
            return news['articles']
            
        except NewsAPIException as e:
            error_data = e.get_exception()
            if error_data.get('code') == 'rateLimited':
                self.logger.warning(f"News API rate limited: {error_data.get('message')}")
                self.rate_limited = True
                # 12시간 후 재시도 (무료 계정 제한)
                self.rate_limit_reset_time = datetime.now() + timedelta(hours=12)
                return []
            else:
                self.logger.error(f"News API error: {error_data}")
                return []
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []

    async def get_tweets(self, category='crypto'):
        """Get relevant tweets (disabled due to API restrictions)"""
        # Twitter API is disabled due to access restrictions
        return []

    async def analyze_sentiment(self, text, category='crypto'):
        """Enhanced sentiment analysis based on keyword matching"""
        positive_words = {
            'crypto': ['bullish', 'growth', 'adoption', 'institutional', 'positive', 'rally', 'breakthrough'],
            'nasdaq': ['rally', 'growth', 'positive', 'earnings beat', 'bullish', 'recovery']
        }
        negative_words = {
            'crypto': ['bearish', 'crash', 'ban', 'regulation', 'hack', 'negative', 'sell-off'],
            'nasdaq': ['crash', 'sell-off', 'bearish', 'negative', 'decline', 'loss']
        }
        
        text_lower = text.lower()
        score = 0
        
        # Check for category-specific keywords
        for word in positive_words[category]:
            if word in text_lower:
                score += 1
        for word in negative_words[category]:
            if word in text_lower:
                score -= 1
                
        # Check for general market sentiment words
        general_positive = ['up', 'rise', 'gain', 'positive', 'growth']
        general_negative = ['down', 'fall', 'loss', 'negative', 'decline']
        
        for word in general_positive:
            if word in text_lower:
                score += 0.5
        for word in general_negative:
            if word in text_lower:
                score -= 0.5
                
        return score / len(text.split()) if text.split() else 0  # Normalize by text length

    async def get_market_impact(self, category='crypto'):
        """Get and analyze news impact on the market with caching"""
        try:
            news_articles = await self.get_news(category)
            
            # API 호출이 제한되었거나 데이터가 없으면 캐시된 값 사용
            if not news_articles:
                if self.rate_limited:
                    self.logger.info(f"Using cached {category} impact due to rate limiting")
                return self.cached_impact.get(category, 0)
            
            all_texts = []
            for article in news_articles:
                title = article.get('title', '')
                description = article.get('description', '')
                if title and description:
                    all_texts.append(title + ' ' + description)
            
            if not all_texts:
                return self.cached_impact.get(category, 0)
            
            impact_scores = []
            for text in all_texts:
                score = await self.analyze_sentiment(text, category)
                if abs(score) > NEWS_IMPACT_THRESHOLD:
                    impact_scores.append(score)
            
            if not impact_scores:
                impact = 0
            else:
                impact = sum(impact_scores) / len(impact_scores)
            
            # 캐시 업데이트
            self.cached_impact[category] = impact
            return impact
            
        except Exception as e:
            self.logger.error(f"Error in get_market_impact for {category}: {e}")
            return self.cached_impact.get(category, 0)

    async def monitor_news(self, callback, interval=1800):  # 30분 간격으로 증가 (300초 -> 1800초)
        """Monitor news continuously and call callback with impact"""
        self.logger.info(f"Starting news monitoring with {interval}s interval")
        
        while True:
            try:
                crypto_impact = await self.get_market_impact('crypto')
                nasdaq_impact = await self.get_market_impact('nasdaq')
                
                # Combine impacts (weighted average)
                combined_impact = (crypto_impact * 0.7 + nasdaq_impact * 0.3)
                
                # 영향도가 있을 때만 로그 출력
                if abs(combined_impact) > 0.1:
                    self.logger.info(f"News impact - Crypto: {crypto_impact:.3f}, NASDAQ: {nasdaq_impact:.3f}, Combined: {combined_impact:.3f}")
                
                await callback(combined_impact, {
                    'crypto': crypto_impact,
                    'nasdaq': nasdaq_impact
                })
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(300)  # 에러 시 5분 대기 후 재시도 