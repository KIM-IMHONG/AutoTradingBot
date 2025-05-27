from telegram.ext import Application
import logging
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.app = None
        self.chat_id = TELEGRAM_CHAT_ID

    async def initialize(self):
        """Initialize Telegram bot"""
        try:
            self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
            await self.app.initialize()
            await self.app.start()
            self.logger.info("Telegram bot initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            raise

    async def send_message(self, message):
        """Send message to Telegram chat"""
        try:
            if self.app:
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                )
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")

    async def send_trade_signal(self, signal_type, symbol, price, reason):
        """Send trade signal notification"""
        message = (
            f"🔔 <b>Trade Signal</b>\n\n"
            f"Type: {signal_type}\n"
            f"Symbol: {symbol}\n"
            f"Price: {price}\n"
            f"Reason: {reason}"
        )
        await self.send_message(message)

    async def send_position_update(self, position):
        """Send position update notification"""
        message = (
            f"📊 <b>Position Update</b>\n\n"
            f"Symbol: {position['symbol']}\n"
            f"Side: {position['side']}\n"
            f"Size: {position['positionAmt']}\n"
            f"Entry Price: {position['entryPrice']}\n"
            f"Unrealized PnL: {position['unRealizedProfit']}"
        )
        await self.send_message(message)

    async def send_error(self, error_message):
        """Send error notification"""
        message = f"⚠️ <b>Error Alert</b>\n\n{error_message}"
        await self.send_message(message)

    async def close(self):
        """Close Telegram bot connection"""
        if self.app:
            await self.app.stop()
            await self.app.shutdown()
            self.logger.info("Telegram bot connection closed") 