"""
🌙 Moon Dev's Exchange Manager
HyperLiquid-only interface for private perps trading.
Built with love by Moon Dev 🚀
"""

import os
from termcolor import colored, cprint
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()


class ExchangeManager:
    """
    HyperLiquid-only exchange interface.
    """

    def __init__(self, exchange=None):
        """Initialize the exchange manager for HyperLiquid."""
        from src import config
        from src import nice_funcs_hyperliquid as hl
        import eth_account

        self.exchange = (exchange or config.EXCHANGE).lower()
        if self.exchange != 'hyperliquid':
            raise ValueError(f"Only 'hyperliquid' is supported. Got: {self.exchange}")

        # Read private key: secrets JSON → HYPER_LIQUID_ETH_PRIVATE_KEY env → HYPER_LIQUID_KEY legacy fallback
        from src.utils.secrets_manager import load_secrets
        secrets = load_secrets()
        trading_keys = secrets.get("trading_keys", {})
        hl_key = trading_keys.get("hyperliquid_private_key")
        if not hl_key:
            hl_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
        if not hl_key:
            hl_key = os.getenv('HYPER_LIQUID_KEY')  # legacy fallback
        if not hl_key:
            raise ValueError("HyperLiquid private key not found. Set it in Account > Secrets or via HYPER_LIQUID_ETH_PRIVATE_KEY env var.")

        # Clean the key of accidental quotes or spaces
        hl_key = hl_key.strip().replace('"', '').replace("'", "")
        self.account = eth_account.Account.from_key(hl_key)

        # Wallet address for queries: secrets JSON → ACCOUNT_ADDRESS env → derived from key
        self.wallet_address = trading_keys.get("hyperliquid_wallet_address")
        if not self.wallet_address:
            self.wallet_address = os.getenv('ACCOUNT_ADDRESS', '')
        if not self.wallet_address:
            # Derive from key — works if key is the master wallet key, not an API wallet key
            self.wallet_address = self.account.address

        self.hl = hl

        cprint(f"✅ Initialized HyperLiquid exchange manager", "green")
        cprint(f"   Account: {self.account.address[:6]}...{self.account.address[-4:]}", "cyan")
        cprint(f"   Wallet:  {self.wallet_address[:6]}...{self.wallet_address[-4:]}", "cyan")

    def market_buy(self, symbol_or_token, usd_amount):
        """Execute a market buy order."""
        return self.hl.market_buy(symbol_or_token, usd_amount, self.account)

    def market_sell(self, symbol_or_token, usd_amount_or_percent):
        """Execute a market sell order."""
        return self.hl.market_sell(symbol_or_token, usd_amount_or_percent, self.account)

    def get_position(self, symbol_or_token):
        """Get current position for a symbol."""
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long = \
            self.hl.get_position(symbol_or_token, self.account)

        if im_in_pos:
            return {
                'has_position': True,
                'size': float(pos_size),
                'symbol': pos_sym,
                'entry_price': entry_px,
                'pnl_percent': pnl_perc,
                'is_long': is_long,
                'raw_data': positions
            }
        return {
            'has_position': False,
            'size': 0,
            'symbol': symbol_or_token,
            'entry_price': 0,
            'pnl_percent': 0,
            'is_long': True,
            'raw_data': None
        }

    def get_token_balance_usd(self, symbol_or_token):
        """Get USD value of token balance/position."""
        position = self.get_position(symbol_or_token)
        if position['has_position']:
            price = self.hl.get_current_price(symbol_or_token)
            return abs(position['size']) * price
        return 0

    def close_position(self, symbol_or_token):
        """Close an open position."""
        return self.hl.kill_switch(symbol_or_token, self.account)

    def ai_entry(self, symbol_or_token, usd_amount):
        """AI-assisted entry (wrapper for market_buy)."""
        return self.market_buy(symbol_or_token, usd_amount)

    def chunk_kill(self, symbol_or_token, max_order_size=None, slippage=None):
        """Close position in chunks (HyperLiquid just closes the position)."""
        return self.hl.kill_switch(symbol_or_token, self.account)

    def get_current_price(self, symbol_or_token):
        """Get current price of symbol."""
        return self.hl.get_current_price(symbol_or_token)

    def get_account_value(self):
        """Get total account value in USD."""
        return self.hl.get_account_value(self.account)

    def get_balance(self):
        """Get available balance for trading in USD."""
        return self.hl.get_balance(self.account)

    def get_all_positions(self):
        """Get all open positions."""
        return self.hl.get_all_positions(self.account)

    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol (1-50)."""
        return self.hl.set_leverage(symbol, leverage, self.account)

    def get_data(self, symbol_or_token, days_back, timeframe):
        """Get OHLCV data for analysis."""
        cprint(f"⚠️ OHLCV data not yet implemented for HyperLiquid", "yellow")
        return pd.DataFrame()

    def fetch_wallet_holdings(self, wallet_address=None):
        """Fetch all wallet holdings as DataFrame."""
        positions = self.get_all_positions()
        if positions:
            return pd.DataFrame(positions)
        return pd.DataFrame()

    def __str__(self):
        return f"ExchangeManager(exchange={self.exchange})"

    def __repr__(self):
        return self.__str__()


def create_exchange_manager(exchange=None):
    """Create and return a HyperLiquid ExchangeManager instance."""
    return ExchangeManager(exchange)
