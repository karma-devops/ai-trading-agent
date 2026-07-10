"""
🌙 Moon Dev's Configuration File
HyperLiquid-only private perps configuration.
"""

import os

# 🔄 Exchange Selection — HyperLiquid only
EXCHANGE = 'hyperliquid'

# ⚡ HyperLiquid Configuration
HYPERLIQUID_SYMBOLS = ['BTC', 'ETH', 'SOL', 'LTC', 'AAVE', 'HYPE']
HYPERLIQUID_LEVERAGE = 10  # Default leverage (1-50 on HyperLiquid)

# Position sizing 🎯
# Target notional = starting_balance * TARGET_BALANCE_MULTIPLIER
# Single position cap = starting_balance * MAX_POSITION_PCT
# HyperLiquid minimum order size ($10) is enforced by the trading core.
TARGET_BALANCE_MULTIPLIER = 5   # Target 5x balance (was 50x — too aggressive)
MAX_POSITION_PCT = 0.50         # Max 50% of balance per position (was 92%)

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
USE_PERCENTAGE = True  # Use percentage-based limits (better for any account size)

MAX_LOSS_PERCENT = 15   # 15% max loss
MAX_GAIN_PERCENT = 50   # 50% max gain

# USD-based limits (set high so percentage limits take priority)
MAX_LOSS_USD = 1000
MAX_GAIN_USD = 1000

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

# AI Model Settings (read from env vars, fallback to defaults)
AI_MODEL_TYPE = os.getenv('AI_PROVIDER', 'ollama')
AI_MODEL = os.getenv('AI_MODEL', 'kimi-k2.7-code')
AI_BASE_URL = os.getenv('AI_BASE_URL', '')
AI_API_KEY = os.getenv('AI_API_KEY', '')
AI_MAX_TOKENS = 8024
AI_TEMPERATURE = 0.6

# Alternative defaults
ALT_AI_MODEL_TYPE = 'openrouter'
ALT_AI_MODEL = "deepseek/deepseek-v4-0324:free"  # DeepSeek V4 Flash on OpenRouter
ALT_AI_BASE_URL = "https://openrouter.ai/api/v1"

# Trading Strategy Agent Settings
ENABLE_STRATEGIES = True
STRATEGY_MIN_CONFIDENCE = 0.6   # 60% confidence threshold

# ⚡ WebSocket Settings (Real-time data feeds)
USE_WEBSOCKET_FEEDS = True
WEBSOCKET_FALLBACK_TO_API = True  # If WebSocket fails, fall back to API polling

# Legacy variables (kept for import compatibility, not used)
tokens_to_trade = HYPERLIQUID_SYMBOLS
MONITORED_TOKENS = []
EXIT_ALL_POSITIONS = False

# Safety switch - true = log only, false = execute real trades
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
