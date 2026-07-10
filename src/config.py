"""
🌙 Moon Dev's Configuration File
HyperLiquid-only private perps configuration.
"""

# 🔄 Exchange Selection — HyperLiquid only
EXCHANGE = 'hyperliquid'

# ⚡ HyperLiquid Configuration
HYPERLIQUID_SYMBOLS = ['BTC', 'ETH', 'SOL', 'LTC', 'AAVE', 'HYPE']
HYPERLIQUID_LEVERAGE = 10  # Default leverage (1-50 on HyperLiquid)

# Position sizing 🎯
# Target notional = starting_balance * TARGET_BALANCE_MULTIPLIER
# Single position cap = starting_balance * MAX_POSITION_PCT
# HyperLiquid minimum order size ($10) is enforced by the trading core.
TARGET_BALANCE_MULTIPLIER = 50
MAX_POSITION_PCT = 0.92  # Max 92% of account balance in one position

# Legacy aliases (kept for compatibility until other files are updated)
usd_size = None
max_usd_order_size = None
tx_sleep = 5  # Faster execution for perps


def get_position_size_usd(account_balance: float) -> float:
    """
    Return the USD notional size for one position.
    Target = balance * 50, but capped at 92% of balance to leave room
    for multiple positions and margin buffer.
    HyperLiquid's minimum order size ($10) is enforced later.
    """
    if account_balance <= 0:
        return 0.0
    target = account_balance * TARGET_BALANCE_MULTIPLIER
    cap = account_balance * MAX_POSITION_PCT
    return min(target, cap)


def get_active_tokens():
    """Returns HyperLiquid symbols."""
    return HYPERLIQUID_SYMBOLS


# Token to Exchange Mapping (all HyperLiquid)
TOKEN_EXCHANGE_MAP = {
    'BTC': 'hyperliquid',
    'ETH': 'hyperliquid',
    'SOL': 'hyperliquid',
    'LTC': 'hyperliquid',
    'XRP': 'hyperliquid',
    'AAVE': 'hyperliquid',
    'LINK': 'hyperliquid',
    'HYPE': 'hyperliquid',
    'FARTCOIN': 'hyperliquid',
}

# 🛡️ Risk Management Settings (Tuned for $5-$10 Account)
CASH_PERCENTAGE = 20  # Keep 20% of account as backup
MAX_POSITION_PERCENTAGE = 80  # Legacy alias
TAKE_PROFIT_PERCENT = 4.5  # Take profit at +4.5%
STOP_LOSS_PERCENT = 1.5   # Stop loss at -1.5%
STOPLOSS_PRICE = 0
BREAKOUT_PRICE = 0
SLEEP_AFTER_CLOSE = 30  # Sleep 30s after closing a trade

MAX_LOSS_GAIN_CHECK_HOURS = 12
SLEEP_BETWEEN_RUNS_MINUTES = 1  # Check markets every minute

# Max Loss/Gain Settings
USE_PERCENTAGE = False

# Percentage-based limits (used when USE_PERCENTAGE = True)
MAX_LOSS_PERCENT = 10   # 10% max loss
MAX_GAIN_PERCENT = 20   # 20% max gain

# USD-based limits (Protective Stops)
MAX_LOSS_USD = 2   # If we lose $2, stop trading
MAX_GAIN_USD = 3   # If we make $3, stop and take profit

# USD MINIMUM BALANCE RISK CONTROL
MINIMUM_BALANCE_USD = 5  # If balance drops below $5, close everything
USE_AI_CONFIRMATION = False  # Set to False for faster exits

# Transaction settings ⚡
slippage = 0.01  # 1% Slippage
orders_per_open = 1  # 1 Order is enough for small size

# Market maker settings (Simple Supply/Demand)
buy_under = 0.99  # Buy if price drops 1% below target
sell_over = 1.01  # Sell if price rises 1% above target

# Data collection settings 📈
DAYSBACK_4_DATA = 2
DATA_TIMEFRAME = '30m'
SAVE_OHLCV_DATA = False

# AI Model Settings
AI_MODEL_TYPE = 'openrouter'  # Default provider: openrouter
AI_MODEL = "nex-agi/deepseek-v3.1-nex-n1:free"  # OpenRouter default
AI_MAX_TOKENS = 8024
AI_TEMPERATURE = 0.6

# Trading Strategy Agent Settings
ENABLE_STRATEGIES = True
STRATEGY_MIN_CONFIDENCE = 0.6   # 60% confidence threshold

# ⚡ WebSocket Settings (Real-time data feeds)
USE_WEBSOCKET_FEEDS = True
WEBSOCKET_FALLBACK_TO_API = True  # If WebSocket fails, fall back to API polling

# Legacy/Solana Variables (safe defaults, ignored)
symbol = 'SOL'
tokens_to_trade = HYPERLIQUID_SYMBOLS
MONITORED_TOKENS = []
PRIORITY_FEE = 100000
sell_at_multiple = 3
USDC_SIZE = 1
limit = 49
timeframe = '15m'
stop_loss_percentage = -0.24
EXIT_ALL_POSITIONS = False
DO_NOT_TRADE_LIST = ['777']
CLOSED_POSITIONS_TXT = '777'
minimum_trades_in_last_hour = 2
MIN_TRADES_LAST_HOUR = 2
REALTIME_CLIPS_ENABLED = False
