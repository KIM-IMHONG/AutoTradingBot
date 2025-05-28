import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading Settings
TRADING_SYMBOLS = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '200'))  # USDT
MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', '50'))  # Maximum leverage
POSITION_RATIO = float(os.getenv('POSITION_RATIO', '100'))  # Percentage of balance to use
POSITION_SIZE = float(os.getenv('POSITION_SIZE', '100'))  # Default position size in USDT

# Technical Analysis Settings
# RSI Settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic Settings
STOCH_K = 14
STOCH_D = 3
STOCH_SLOW = 3

# EMA Settings
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50

# ATR Settings
ATR_PERIOD = 14

# Volume Settings
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # 평균 거래량의 1.5배 이상일 때 유의미한 거래량으로 판단

# Risk Management Settings
STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '1.0'))  # 1%
TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '2.0'))  # 2%
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '5.0'))  # 5% of balance
MIN_TRADING_INTERVAL = int(os.getenv('MIN_TRADING_INTERVAL', '300'))  # 5 minutes

# Trading Bot Settings
TRADING_ENABLED = os.getenv('TRADING_ENABLED', 'True').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'True').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Signal Settings
SIGNAL_CONFIRMATION_COUNT = 2  # 신호 확인 횟수
SIGNAL_HISTORY_LIMIT = 5  # 저장할 최대 신호 수

# Logging Settings
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def validate_settings():
    """Validate required settings and API keys"""
    required_vars = {
        'BINANCE_API_KEY': BINANCE_API_KEY,
        'BINANCE_API_SECRET': BINANCE_API_SECRET,
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Warning: Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate trading settings
    if MAX_LEVERAGE < 1 or MAX_LEVERAGE > 125:
        print(f"Warning: MAX_LEVERAGE should be between 1 and 125, current value: {MAX_LEVERAGE}")
    
    if POSITION_RATIO < 0 or POSITION_RATIO > 100:
        print(f"Warning: POSITION_RATIO should be between 0 and 100, current value: {POSITION_RATIO}")
    
    if not TRADING_SYMBOLS:
        print("Warning: No trading symbols specified")
    else:
        for symbol in TRADING_SYMBOLS:
            if not symbol.endswith('USDT'):
                print(f"Warning: Trading symbol {symbol} should end with USDT")
    
    # Validate technical analysis settings
    if EMA_SHORT >= EMA_MEDIUM or EMA_MEDIUM >= EMA_LONG:
        print("Warning: EMA periods should be in ascending order (SHORT < MEDIUM < LONG)")
    
    if RSI_OVERBOUGHT <= RSI_OVERSOLD:
        print("Warning: RSI_OVERBOUGHT should be greater than RSI_OVERSOLD")

# Validate settings on import
validate_settings() 