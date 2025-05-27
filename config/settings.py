import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading Parameters
TRADING_SYMBOL = 'BTCUSDT'
LEVERAGE = 30
POSITION_SIZE = 0.01  # BTC
STOP_LOSS_PERCENTAGE = 2.0
TAKE_PROFIT_PERCENTAGE = 4.0

# Technical Analysis Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50

# News Analysis Parameters
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# Risk Management
MAX_POSITION_SIZE = 0.1  # Maximum position size in BTC
MAX_DAILY_LOSS = 0.05   # Maximum daily loss percentage
NEWS_IMPACT_THRESHOLD = 0.7  # Minimum sentiment score to consider news impact 