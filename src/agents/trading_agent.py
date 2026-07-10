"""
LLM Trading Agent

DUAL-MODE AI TRADING SYSTEM:

SINGLE MODEL MODE (Fast - ~10 seconds per token):
   - Uses one AI model for quick trading decisions
   - Best for: Fast execution, high-frequency strategies
   - Configure model in config.py: AI_MODEL_TYPE and AI_MODEL_NAME

SWARM MODE (Consensus - ~45-60 seconds per token):
   - Queries 6 AI models simultaneously for consensus voting
   - Models vote: "Buy", "Sell", or "Do Nothing"
   - Majority decision wins with confidence percentage
   - Best for: Higher confidence trades, 15-minute+ timeframes
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re


def extract_json_from_text(text):
    """Safely extract the first valid JSON object or array from AI model responses containing text."""
    if not text:
        print("⚠️ Empty AI response.")
        return None

    # Try a clean fenced code block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to brace search

    # Find the first balanced JSON object or array by scanning brace pairs
    for start_char, end_char in ("{", "}"), ("[", "]"):
        depth = 0
        start_idx = None
        for i, ch in enumerate(text):
            if ch == start_char:
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == end_char:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate = text[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        # If we get here with leftover depth, try the longest match anyway
        if start_idx is not None and depth > 0:
            candidate = text[start_idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    print("⚠️ No valid JSON object found in AI response.")
    return None


# ============================================================================
# 🚨 CRITICAL: THIS MUST BE HERE (BEFORE 'src' IMPORTS)
# ============================================================================
# Add project root to path so Python can find 'src'
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
# ============================================================================

# 👇 NOW you can import from src safely
from src.models import model_factory
from src.agents.swarm_agent import SwarmAgent
from src.data.ohlcv_collector import collect_all_tokens
from src.agents.strategy_agent import StrategyAgent

# Import shared logging utility (prevents circular import with trading_app)
try:
    from src.utils.logging_utils import add_console_log, log_position_open
except ImportError:
    # Fallback if running standalone without trading_app
    def add_console_log(message, level="info", console_file=None):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def log_position_open(symbol, side, size_usd, console_file=None):
        emoji = "📈" if side == "LONG" else "📉"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {emoji} Opened {side} {symbol} ${size_usd:.2f}")

# Import position tracker for age-based decisions
try:
    from src.utils.position_tracker import (
        record_position_entry, remove_position, get_position_age_hours,
        get_all_tracked_positions, sync_with_exchange_positions
    )
    POSITION_TRACKER_AVAILABLE = True
except ImportError:
    POSITION_TRACKER_AVAILABLE = False
    def record_position_entry(*args, **kwargs): return True
    def remove_position(*args, **kwargs): return True
    def get_position_age_hours(*args, **kwargs): return 0.0
    def get_all_tracked_positions(): return {}
    def sync_with_exchange_positions(*args, **kwargs): return (0, 0)

try:
    from src.utils.trade_logger import log_trade
    TRADE_LOGGER_AVAILABLE = True
except ImportError:
    TRADE_LOGGER_AVAILABLE = False
    def log_trade(*args, **kwargs): pass

# Import three-tier close validation system
try:
    from src.utils.close_validator import (
        validate_close_decision, CloseDecision, ValidationResult,
        format_validation_result, STOP_LOSS_THRESHOLD, PROFIT_TARGET_THRESHOLD,
        TAKE_PROFIT_THRESHOLD, MIN_CONFIDENCE_TO_CLOSE
    )
    CLOSE_VALIDATOR_AVAILABLE = True
except ImportError:
    CLOSE_VALIDATOR_AVAILABLE = False
    STOP_LOSS_THRESHOLD = -2.0
    TAKE_PROFIT_THRESHOLD = 6.0
    PROFIT_TARGET_THRESHOLD = 0.5
    MIN_CONFIDENCE_TO_CLOSE = 80

# Import unified AI gateway
try:
    from src.utils.ai_gateway import (
        analyze_with_ai, AIResponse, parse_ai_response,
        format_position_prompt, format_entry_prompt
    )
    AI_GATEWAY_AVAILABLE = True
except ImportError:
    AI_GATEWAY_AVAILABLE = False

# Import intelligence integrator for strategy and volume signals
try:
    from src.utils.intelligence_integrator import (
        collect_all_intelligence, get_volume_intel_for_token,
        get_strategy_signals, format_strategy_signals_for_ai,
        format_volume_intel_for_ai, get_volume_summary
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    def collect_all_intelligence(*args, **kwargs): return {"token": "", "combined_context": ""}
    def get_volume_intel_for_token(*args, **kwargs): return None
    def get_volume_summary(): return ""


# Load Environment Variables
load_dotenv()

# ============================================================================
# 🔧 OPTIONAL: COLOR PRINT & PANDAS SHIM (Keep your existing helpers)
# ============================================================================
try:
    from termcolor import cprint
except Exception:
    def cprint(msg, *args, **kwargs):
        print(msg)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception as e:
    pd = None
    PANDAS_AVAILABLE = False
    cprint(f"⚠️ pandas not installed: {e}. Using lightweight DataFrame shim.", "yellow")
    import types

    class SimpleDataFrame:
        def __init__(self, data=None, columns=None):
            self._data = list(data) if data else []
            if columns:
                self.columns = list(columns)
            else:
                self.columns = list(self._data[0].keys()) if self._data else []
            self.index = list(range(len(self._data)))

        def __len__(self):
            return len(self._data)

        def head(self, n=5):
            return SimpleDataFrame(self._data[:n], columns=self.columns)

        def tail(self, n=3):
            return SimpleDataFrame(self._data[-n:], columns=self.columns)

        def to_string(self):
            if not self._data:
                return "<empty DataFrame>"
            header = " | ".join(self.columns)
            lines = [header]
            for row in self._data:
                lines.append(" | ".join(str(row.get(c, "")) for c in self.columns))
            return "\n".join(lines)

        def __str__(self):
            return self.to_string()

        def to_dict(self):
            return self._data

    def _concat(dfs, ignore_index=True):
        rows = []
        cols = []
        for df in dfs:
            if isinstance(df, SimpleDataFrame):
                rows.extend(df._data)
                for c in df.columns:
                    if c not in cols:
                        cols.append(c)
            elif isinstance(df, dict):
                rows.append(df)
        return SimpleDataFrame(rows, columns=cols)

    pd = types.SimpleNamespace(DataFrame=SimpleDataFrame, concat=_concat)

# ============================================================================
# 🔧 TRADING AGENT CONFIGURATION
# ============================================================================
from eth_account import Account
from src.config import EXCHANGE as CONFIG_EXCHANGE, HYPERLIQUID_LEVERAGE, MAX_POSITION_PCT, get_position_size_usd, DRY_RUN

# 🦈 EXCHANGE SELECTION - Import from config.py
# Convert to uppercase for consistency with checks throughout this file
EXCHANGE = CONFIG_EXCHANGE.upper() if CONFIG_EXCHANGE else "HYPERLIQUID"

# 🌊 AI MODE SELECTION (Default - can be overridden by user settings)
DEFAULT_SWARM_MODE = False  # True = Swarm Mode (all Models), False = Single Model

# 🌊 SWARM CONSENSUS SETTINGS
# Minimum confidence threshold for swarm consensus to execute a trade
# If consensus confidence is below this threshold, default to NOTHING
# Recommended: 55-65% (requires clear majority, not just a tie)
MIN_SWARM_CONFIDENCE = 65  # 55% = requires at least slight majority (e.g., 3/5 models agree)

# 📈 TRADING MODE SETTINGS
LONG_ONLY = False 

# ============================================================================
# ⚙️ POSITION MANAGEMENT SETTINGS
# ============================================================================
# Minimum age before a position can be closed (in hours), Set to 0.0 to allow immediate exits
# Examples: 0.0 = immediate, 0.25 = 15 min, 0.5 = 30 min, 1.0 = 1 hour
MIN_AGE_HOURS = 0.1

# AI confidence threshold for closing positions (percentage), Lower values = more aggressive closing, Higher values = more conservative
# Recommended range: 60-80%
MIN_CLOSE_CONFIDENCE = 70

# Profit threshold for automatic position closing (percentage), Positions with profit >= this value close immediately, bypassing other checks
TP_THRESHOLD = 0.5

# SINGLE MODEL SETTINGS
AI_MODEL_TYPE = 'openrouter' 
AI_MODEL_NAME = 'x-ai/grok-4.1-fast' 
AI_TEMPERATURE = 0.6   # Official recommended "sweet spot"
AI_MAX_TOKENS = 8024   # Increased for multi-step reasoning

# 💰 POSITION SIZING & RISK MANAGEMENT
USE_PORTFOLIO_ALLOCATION = True
MAX_POSITION_PERCENTAGE = int(MAX_POSITION_PCT * 100)
LEVERAGE = HYPERLIQUID_LEVERAGE

# Stop Loss & Take Profit
STOP_LOSS_PERCENTAGE = 2.0      # SL @ -2% PnL
TAKE_PROFIT_PERCENTAGE = 5.0    # TP @ +5% PnL 
PNL_CHECK_INTERVAL = 5          # check PnL every 5 minutes          

# Legacy settings 
usd_size = 25                  
max_usd_order_size = 3           
CASH_PERCENTAGE = 10

# 📊 MARKET DATA COLLECTION (Default values - can be overridden)
DAYSBACK_4_DATA = 2              # Default: 2 days (overridable via __init__)
DATA_TIMEFRAME = '30m'           # Default: 30 minutes (overridable via __init__)
SAVE_OHLCV_DATA = False          

# ⚡ TRADING EXECUTION SETTINGS
slippage = 199                   
SLEEP_BETWEEN_RUNS_MINUTES = 5  

# 🎯 TOKEN CONFIGURATION
# Note: Account address is loaded from .env via os.getenv("ACCOUNT_ADDRESS")
# or from self.account.address in TradingAgent.__init__()

# For SOLANA exchange
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" 
SOL_ADDRESS = "So11111111111111111111111111111111111111111"    
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

MONITORED_TOKENS = []

# For ASTER/HYPERLIQUID exchanges
SYMBOLS = [
    'ETH',        # Ethereum
    'BTC',        # Bitcoin
    'SOL',        # Solana
    'AAVE',       # Aave
    'LINK',       # Chainlink
    'LTC',        # Litecoin
    'HYPE',       # Hyperliquid Exchange Token
    'FARTCOIN',   # FartCoin (for the fun)
]

# ============================================================================
# 🔌 EXCHANGE IMPORTS
# ============================================================================
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Corrected Import Logic
if EXCHANGE == "ASTER":
    try:
        from src import nice_funcs_aster as n
        cprint("🦈 Exchange: Aster DEX (Futures)", "cyan", attrs=['bold'])
    except ImportError:
        cprint("❌ Error: nice_funcs_aster not found", "red")
        
elif EXCHANGE == "HYPERLIQUID":
    try:
        from src import nice_funcs_hyperliquid as n
        cprint("🦈 Exchange: HyperLiquid (Perpetuals) - Using local nice_funcs_hyperliquid.py", "cyan", attrs=['bold'])
    except ImportError:
        try:
            import nice_funcs_hyperliquid as n
            cprint("🦈 Exchange: HyperLiquid (Perpetuals) - Using src module", "cyan", attrs=['bold'])
        except ImportError:
            cprint("❌ Error: nice_funcs_hyperliquid.py not found! Ensure it is in the same folder.", "red")
            sys.exit(1)
            
elif EXCHANGE == "SOLANA":
    try:
        from src import nice_funcs as n
        cprint("🦈 Exchange: Solana (On-chain DEX)", "cyan", attrs=['bold'])
    except ImportError:
        cprint("❌ Error: Solana functions not found", "red")

else:
    cprint(f"❌ Unknown exchange: {EXCHANGE}", "red")
    cprint("Available exchanges: ASTER, HYPERLIQUID, SOLANA", "yellow")
    sys.exit(1)

from src.data.ohlcv_collector import collect_all_tokens

# ============================================================================
# PROMPTS
# ============================================================================

TRADING_PROMPT = """
You are a renowned crypto trading expert and Trading Assistant

Analyze the provided market data, CURRENT POSITION, and STRATEGY CONTEXT signals to make a trading decision.

{position_context}

Market Data Criteria:
1. Price action relative to MA20 and MA40
2. RSI levels and trend
3. Volume patterns
4. Recent price movements

{strategy_context}

Respond in this exact format:
1. First line must be one of: BUY, SELL, or NOTHING (in caps)
2. Then explain your reasoning, always including:
   - Technical analysis
   - Strategy signals analysis (if available)
   - Risk factors
   - Market conditions
   - Confidence level (as a percentage, e.g. 75%)

Remember: 
- Always prioritizes risk management! 🛡️
- Never trade USDC or SOL directly
- Consider both technical and strategy signals
"""

ALLOCATION_PROMPT = """
You are our Portfolio Allocation Assistant 🌙

Given the total portfolio size and trading recommendations, allocate capital efficiently.
Consider:
1. Position sizing based on confidence levels
2. Risk distribution
3. Keep cash buffer as specified
4. Maximum allocation per position

Format your response as a Python dictionary:
{
    "token_address": allocated_amount,  # In USD
    ...
    "USDC_ADDRESS": remaining_cash  # Always use USDC_ADDRESS for cash
}

Remember:
- Total allocations must not exceed total_size
- Higher confidence should get larger allocations
- Never allocate more than {MAX_POSITION_PERCENTAGE}% to a single position
- Keep at least {CASH_PERCENTAGE}% in USDC as safety buffer
- Only allocate to BUY recommendations
- Cash must be stored as USDC using USDC_ADDRESS: {USDC_ADDRESS}
"""

SWARM_TRADING_PROMPT = """You are an expert cryptocurrency trading AI analyzing market data.

CRITICAL RULES:
1. Your response MUST be in this exact format: ACTION | CONFIDENCE%
2. ACTION must be one of: Buy, Sell, or Nothing
3. CONFIDENCE must be a number from 0-100 indicating how confident you are
4. Do NOT provide any explanation, reasoning, or additional text
5. Do NOT show your thinking process or internal reasoning

Analyze the market data below and decide:

- "Buy" = Strong bullish signals, recommend opening/holding position
- "Sell" = Bearish signals or major weakness, recommend closing position entirely
- "Nothing" = Unclear/neutral signals, recommend holding current state unchanged

IMPORTANT: "Nothing" means maintain current position (if we have one, keep it; if we don't, stay out)

RESPONSE FORMAT EXAMPLES:
- Buy | 85%
- Sell | 70%
- Nothing | 45%

RESPOND WITH ONLY: ACTION | CONFIDENCE%"""

SMART_ALLOCATION_PROMPT = """You are an expert portfolio allocation AI for cryptocurrency trading.

CURRENT PORTFOLIO STATE:
{portfolio_state}

AI TRADING SIGNALS:
{signals}

ACCOUNT INFO:
- Available Balance: ${available_balance:.2f}
- Leverage: {leverage}x
- Max Position %: {max_position_pct}%
- Cash Buffer: {cash_buffer_pct}%
- Minimum Order: ${min_order:.2f} notional
- Trading Cycle: Every {cycle_minutes} minutes (MINIMUM HOLD TIME)

YOUR TASK:
Analyze the current positions and AI signals to create an OPTIMAL allocation plan.

⚠️ CRITICAL: Positions will be held for AT LEAST {cycle_minutes} minutes until the next cycle.
- Factor this hold time into your risk assessment
- Higher leverage + longer hold = more exposure to volatility
- Be more conservative if cycle time is long relative to expected move

ALLOCATION RULES:
1. Higher confidence signals deserve larger allocations
2. Consider reallocating from underperforming positions (negative PnL) to stronger opportunities
3. Keep at least {cash_buffer_pct}% as cash buffer
4. Each position's margin should not exceed {max_position_pct}% of total equity
5. Account for existing positions - don't over-allocate to positions already at target size
6. Factor in position PnL when deciding to hold, reduce, or close
7. BUY signals = LONG positions, SELL signals = SHORT positions (if not LONG_ONLY)
8. Consider the {cycle_minutes}-minute minimum hold time when sizing positions

RESPOND WITH ONLY A VALID JSON OBJECT in this exact format:
{{
    "actions": [
        {{"symbol": "BTC", "action": "OPEN_LONG", "margin_usd": 50.00, "reason": "High confidence BUY signal"}},
        {{"symbol": "ETH", "action": "CLOSE", "reason": "SELL signal contradicts LONG position"}},
        {{"symbol": "SOL", "action": "REDUCE", "reduce_by_usd": 100.00, "reason": "Reallocating to higher conviction trade"}},
        {{"symbol": "LTC", "action": "HOLD", "reason": "Position aligned with signal, PnL positive"}},
        {{"symbol": "AAVE", "action": "OPEN_SHORT", "margin_usd": 25.00, "reason": "SELL signal, opening short"}}
    ],
    "cash_buffer_usd": 50.00,
    "reasoning": "Brief explanation of overall allocation strategy"
}}

ACTION TYPES:
- OPEN_LONG: Open new long position (requires margin_usd)
- OPEN_SHORT: Open new short position (requires margin_usd)
- INCREASE: Add to existing position (requires margin_usd)
- REDUCE: Reduce existing position (requires reduce_by_usd in notional)
- CLOSE: Close position entirely
- HOLD: Keep position as-is

CRITICAL: Return ONLY the JSON object, no markdown, no explanation outside the JSON."""

POSITION_ANALYSIS_PROMPT = """
You are an expert crypto trading analyst. Your task is to analyze the user's open positions based on the provided position summaries and current market data.

For EACH symbol, decide whether the user should **KEEP** the position open or **CLOSE** it.

⚠️ CRITICAL: When suggesting CLOSE, you MUST provide a confidence level (0-100%) indicating how certain you are that the position is WRONG and should be closed.

⚠️ CRITICAL OUTPUT RULES:
- You MUST respond ONLY with a valid JSON object – no commentary, no Markdown, no code fences.
- JSON must be well-formed and parseable by Python's json.loads().
- The JSON must follow exactly this structure:

{
  "BTC": {
    "action": "KEEP",
    "reasoning": "Trend remains bullish; RSI under 60",
    "confidence": 0
  },
  "ETH": {
    "action": "CLOSE",
    "reasoning": "Breakdown below MA40 with weak RSI",
    "confidence": 85
  }
}

- "confidence" is 0-100 (only used for CLOSE decisions, set to 0 for KEEP)
- Confidence represents how certain you are that closing is the RIGHT decision

Do not include ```json or any other formatting around the JSON.
Respond ONLY with the raw JSON object.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_account_balance(account=None):
    """Get account balance in USD based on exchange type"""
    try:
        if EXCHANGE in ["ASTER", "HYPERLIQUID"]:
            if EXCHANGE == "ASTER":
                balance_dict = n.get_account_balance()
                balance = balance_dict.get('available', 0) 
                cprint(f"💰 {EXCHANGE} Available Balance: ${balance:,.2f} USD", "cyan")
                
            else:  # HYPERLIQUID
                address = os.getenv("ACCOUNT_ADDRESS")
                if not address:
                    if account is None:
                        account = n._get_account_from_env()
                    address = account.address

                try:
                    if hasattr(n, 'get_available_balance'):
                        balance = n.get_available_balance(address)
                        cprint(f"💰 {EXCHANGE} Available (Free) USDC: ${balance}", "cyan")
                        
                        total_val = n.get_account_value(address)
                        cprint(f"   (Total Equity including positions: ${total_val})", "white")
                    else:
                        cprint("⚠️ Using Total Equity (Warning: Checks locked collateral)", "yellow")
                        balance = n.get_account_value(address)
                        
                except Exception as e:
                    cprint(f"❌ Error getting balance: {e}", "red")
                    balance = 0

            return float(balance)
            
        else:
            # SOLANA
            balance = n.get_token_balance_usd(USDC_ADDRESS)
            return balance
            
    except Exception as e:
        cprint(f"❌ Error getting account balance: {e}", "red")
        return 0


def calculate_position_size(account_balance):
    """Calculate position size based on account balance and MAX_POSITION_PERCENTAGE"""
    if EXCHANGE in ["ASTER", "HYPERLIQUID"]:
        margin_to_use = account_balance * (MAX_POSITION_PERCENTAGE / 100)
        notional_position = margin_to_use * LEVERAGE

        cprint(f"   📊 Position Calculation ({EXCHANGE}):", "yellow", attrs=['bold'])
        cprint(f"   💵 Account Balance: ${account_balance:,.2f}", "white")
        cprint(f"   📈 Max Position %: {MAX_POSITION_PERCENTAGE}%", "white")
        cprint(f"   💰 Margin to Use: ${margin_to_use:,.2f}", "green", attrs=['bold'])
        cprint(f"   ⚡ Leverage: {LEVERAGE}x", "white")
        cprint(f"   💎 Notional Position: ${notional_position:,.2f}", "cyan", attrs=['bold'])

        return notional_position
    else:
        # For Solana: No leverage, direct position size
        position_size = account_balance * (MAX_POSITION_PERCENTAGE / 100)

        cprint(f"   📊 Position Calculation (SOLANA):", "yellow", attrs=['bold'])
        cprint(f"   💵 USDC Balance: ${account_balance:,.2f}", "white")
        cprint(f"   📈 Max Position %: {MAX_POSITION_PERCENTAGE}%", "white")
        cprint(f"   💎 Position Size: ${position_size:,.2f}", "cyan", attrs=['bold'])

        return position_size


# ============================================================================
# TRADING AGENT CLASS
# ============================================================================

class TradingAgent:
    def __init__(self, timeframe=None, days_back=None, stop_check_callback=None,
                 symbols=None, ai_provider=None, ai_model=None,
                 ai_temperature=None, ai_max_tokens=None,
                 ai_base_url=None, ai_api_key=None,
                 swarm_mode=None, swarm_models=None, active_strategy=None):
        """
        Initialize Trading Agent with configurable settings

        Args:
            timeframe (str): Data timeframe (e.g., '5m', '30m', '1h'). Defaults to DATA_TIMEFRAME.
            days_back (int): Days of historical data to fetch. Defaults to DAYSBACK_4_DATA.
            stop_check_callback (callable): Optional callback that returns True if agent should stop.
            symbols (list): List of token symbols to analyze. Defaults to SYMBOLS.
            ai_provider (str): AI provider to use (e.g., 'gemini', 'anthropic'). Defaults to AI_MODEL_TYPE.
            ai_model (str): AI model name to use. Defaults to AI_MODEL_NAME.
            ai_temperature (float): Temperature for AI model (0.0-1.0). Defaults to AI_TEMPERATURE.
            ai_max_tokens (int): Max tokens for AI response. Defaults to AI_MAX_TOKENS.
            swarm_mode (str): 'single' or 'swarm'. Defaults to 'single'.
            swarm_models (list): List of swarm model configs for multi-agent consensus.
            active_strategy (str): 'confidence_ai' (default) or 'engine_v6_1'.
            ai_base_url (str): Optional endpoint URL for Ollama/generic_openai.
            ai_api_key (str): Optional API key for Ollama/generic_openai.
        """
        # Store symbols to analyze (use passed values or fall back to config)
        if symbols is not None:
            self.symbols = symbols
        else:
            self.symbols = SYMBOLS

        # Store configurable settings as instance variables
        self.timeframe = timeframe if timeframe is not None else DATA_TIMEFRAME
        self.days_back = days_back if days_back is not None else DAYSBACK_4_DATA
        self.stop_check_callback = stop_check_callback

        # Store AI settings (use passed values or fall back to config defaults)
        self.ai_provider = ai_provider if ai_provider is not None else AI_MODEL_TYPE
        self.ai_model_name = ai_model if ai_model is not None else AI_MODEL_NAME
        self.ai_temperature = ai_temperature if ai_temperature is not None else AI_TEMPERATURE
        self.ai_max_tokens = ai_max_tokens if ai_max_tokens is not None else AI_MAX_TOKENS
        self.ai_base_url = ai_base_url if ai_base_url is not None else getattr(sys.modules.get('src.config', None), 'AI_BASE_URL', '')
        self.ai_api_key = ai_api_key if ai_api_key is not None else getattr(sys.modules.get('src.config', None), 'AI_API_KEY', '')

        # Strategy selection
        self.active_strategy = active_strategy if active_strategy is not None else getattr(sys.modules.get('src.config', None), 'ACTIVE_STRATEGY', 'confidence_ai')
        self.strategy_engine = None
        if self.active_strategy == 'engine_v6_1':
            try:
                from src.strategies.engine_v6_1 import EngineV6_1Strategy
                self.strategy_engine = EngineV6_1Strategy()
            except Exception as e:
                cprint(f"⚠️ Could not load Engine v6.1: {e}", "yellow")
                self.active_strategy = 'confidence_ai'

        # Store swarm mode settings (use passed values or fall back to defaults)
        self.use_swarm_mode = (swarm_mode == 'swarm') if swarm_mode is not None else DEFAULT_SWARM_MODE
        self.swarm_models_config = swarm_models or []

        self.account = None
        if EXCHANGE == "HYPERLIQUID":
            cprint("🔑 Initializing Hyperliquid Account...", "cyan")
            try:
                # Standardized key lookup
                raw_key = os.getenv("HYPER_LIQUID_ETH_PRIVATE_KEY", "") or os.getenv("HYPER_LIQUID_KEY", "")
                clean_key = raw_key.strip().replace('"', '').replace("'", "")

                if not clean_key:
                    raise ValueError("Private Key not found in .env")

                self.account = Account.from_key(clean_key)
                self.address = os.getenv("ACCOUNT_ADDRESS")

                if not self.address:
                    self.address = self.account.address

                cprint(f"✅ Account loaded successfully! Address: {self.address}", "green")
            except Exception as e:
                cprint(f"❌ Error loading key: {e}", "red")
                sys.exit(1)

        # Check if using swarm mode or single model
        if self.use_swarm_mode:
            # Convert user's swarm_models format to SwarmAgent's format
            custom_models = self._build_swarm_models_config()
            num_models = len(custom_models) if custom_models else 6

            cprint(
                f"\n🌊 Initializing Trading Agent in SWARM MODE ({num_models} AI consensus)...",
                "cyan",
                attrs=["bold"]
            )

            # Initialize SwarmAgent with custom models from user settings
            if custom_models:
                self.swarm = SwarmAgent(custom_models=custom_models)
                cprint(f"✅ Swarm mode initialized with {num_models} user-configured AI models!", "green")
            else:
                self.swarm = SwarmAgent()
                cprint("✅ Swarm mode initialized with default AI models!", "green")

            cprint("💼 Initializing fast model for portfolio calculations...", "cyan")
            self.model = model_factory.get_model(
                self.ai_provider, self.ai_model_name,
                base_url=self.ai_base_url, api_key=self.ai_api_key
            )
            if self.model:
                cprint(f"✅ Allocation model ready: {self.model.model_name}", "green")
        else:
            cprint(f"\n⚙️ Initializing Trading Agent with {self.ai_provider} model...", "cyan")
            self.model = model_factory.get_model(
                self.ai_provider, self.ai_model_name,
                base_url=self.ai_base_url, api_key=self.ai_api_key
            )
            self.swarm = None

            if not self.model:
                cprint(f"❌ Failed to initialize {self.ai_provider} model!", "red")
                cprint("Available models:", "yellow")
                for model_type in model_factory._models.keys():
                    cprint(f"   - {model_type}", "yellow")
                sys.exit(1)

            cprint(f"✅ Using model: {self.model.model_name}", "green")

        self.recommendations_df = pd.DataFrame(
            columns=["token", "action", "confidence", "reasoning"]
        )

        # --- StrategyAgent (non-executing) ---
        try:
            self.strategy_agent = StrategyAgent(execute_signals=False)
            cprint("✅ StrategyAgent initialized (execute_signals=False)", "green")
        except Exception as e:
            self.strategy_agent = None
            cprint(f"⚠️ StrategyAgent failed to initialize: {e}", "yellow")

        # Simple in-memory cache for enriched strategy contexts per token
        # token -> {'data': ..., 'expires_at': datetime}
        self._strategy_context_cache = {}
        self.STRATEGY_CONTEXT_TTL = 120  # seconds (tune for your timeframe)





        # Show which tokens will be analyzed
        cprint("\n🎯 Active Tokens for Trading:", "yellow", attrs=["bold"])
        cprint(f"🦈 Exchange: {EXCHANGE}", "cyan")

        for i, token in enumerate(self.symbols, 1):
            token_display = token[:8] + "..." if len(token) > 8 else token
            cprint(f"   {i}. {token_display}", "cyan")

        cprint(
            f"\n⏱️  Estimated analysis time: ~{len(self.symbols) * 60} seconds\n",
            "yellow"
        )

        cprint(f"\n🦈 Active Exchange: {EXCHANGE}", "yellow", attrs=["bold"])
        cprint("📈 Trading Mode:", "yellow", attrs=["bold"])
        if LONG_ONLY:
            cprint("   📊 LONG ONLY - No shorting enabled", "cyan")
            cprint("   💡 SELL signals close positions, can't open shorts", "white")
        else:
            cprint("   ⚡ LONG/SHORT - Full directional trading", "green")
            cprint("   💡 SELL signals can close longs OR open shorts", "white")

        cprint("\n✅ LLM Trading Agent initialized!", "green")
        add_console_log("Agent initialized!", "success")

    def _build_swarm_models_config(self):
        """
        Convert user's swarm_models format to SwarmAgent's expected format.

        User format (from settings):
        [
            {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.3, "max_tokens": 2000},
            {"provider": "openai", "model": "gpt-4o", "temperature": 0.5, "max_tokens": 2000},
            ...
        ]

        SwarmAgent format:
        {
            "gemini_1": (True, "gemini", "gemini-2.5-flash"),
            "openai_2": (True, "openai", "gpt-4o"),
            ...
        }
        """
        if not self.swarm_models_config:
            return None

        custom_models = {}
        for i, model_config in enumerate(self.swarm_models_config, 1):
            provider = model_config.get('provider', 'openrouter')
            model_name = model_config.get('model', 'nex-agi/deepseek-v3.1-nex-n1:free')

            # Create unique key for each model (e.g., "gemini_1", "openai_2")
            model_key = f"{provider}_{i}"

            # SwarmAgent expects: (enabled, provider_type, model_name)
            custom_models[model_key] = (True, provider, model_name)

            cprint(f"   📦 Swarm Model {i}: {provider}/{model_name}", "cyan")

        return custom_models if custom_models else None

    def chat_with_ai(self, system_prompt, user_content):
        """Send prompt to AI model via model factory"""
        try:
            response = self.model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=self.ai_temperature,
                max_tokens=self.ai_max_tokens
            )

            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            error_str = str(e).lower()
            model_name = getattr(self.model, 'model_name', 'Unknown')
            provider = getattr(self.model, 'provider', 'Unknown')

            # Detect specific error types for helpful messages
            if "rate_limit" in error_str or "rate limit" in error_str:
                msg = f"Rate limit: {provider}/{model_name}"
                add_console_log(msg, "error")
            elif "invalid_api_key" in error_str or "authentication" in error_str or "401" in error_str:
                msg = f"Invalid API key: {provider}"
                add_console_log(msg, "error")
            elif "insufficient" in error_str or "quota" in error_str or "billing" in error_str:
                msg = f"Quota exceeded: {provider}"
                add_console_log(msg, "error")
            elif "timeout" in error_str or "timed out" in error_str:
                msg = f"Timeout: {provider}/{model_name}"
                add_console_log(msg, "error")
            elif "connection" in error_str or "network" in error_str:
                msg = f"Connection error: {provider}"
                add_console_log(msg, "error")
            else:
                msg = f"Model failed: {provider}/{model_name}"
                add_console_log(msg, "error")

            cprint(f"❌ {msg} - {str(e)[:80]}", "red")
            return None

    def _format_market_data_for_swarm(self, token, market_data):
        """Format market data into a clean, readable format for swarm analysis"""
        try:
            cprint(f"\n📊 MARKET DATA RECEIVED FOR {token[:8]}...", "cyan", attrs=["bold"])
            add_console_log(f"📊 MARKET DATA RECEIVED FOR {token[:8]}...", "info")

            if isinstance(market_data, pd.DataFrame):
                cprint(f"✅ DataFrame received: {len(market_data)} bars", "green")
                cprint(f"📅 Date range: {market_data.index[0]} to {market_data.index[-1]}", "yellow")
                cprint(f"🕐 Timeframe: {self.timeframe}", "yellow")

                cprint("\n📈 First 5 Bars (OHLCV):", "cyan")
                print(market_data.head().to_string())

                cprint("\n📉 Last 3 Bars (Most Recent):", "cyan")
                print(market_data.tail(3).to_string())

                formatted = f"""
TOKEN: {token}
TIMEFRAME: {self.timeframe} bars
TOTAL BARS: {len(market_data)}
DATE RANGE: {market_data.index[0]} to {market_data.index[-1]}

RECENT PRICE ACTION (Last 10 bars):
{market_data.tail(10).to_string()}

FULL DATASET:
{market_data.to_string()}
"""
            else:
                cprint(f"⚠️ Market data is not a DataFrame: {type(market_data)}", "yellow")
                formatted = f"TOKEN: {token}\nMARKET DATA:\n{str(market_data)}"

            if isinstance(market_data, dict) and "strategy_signals" in market_data:
                formatted += f"\n\nSTRATEGY SIGNALS:\n{json.dumps(market_data['strategy_signals'], indent=2)}"

            cprint("\n✅ Market data formatted and ready for analysis!\n", "green")
            return formatted

        except Exception as e:
            cprint(f"❌ Error formatting market data: {e}", "red")
            return str(market_data)

    def _calculate_swarm_consensus(self, swarm_result):
        """
        Calculate consensus from individual swarm responses with confidence weighting.

        Key features:
        1. Extracts both action AND confidence from each model
        2. Logs individual votes with confidence (e.g., "Model 1 - BUY | 85%")
        3. Calculates weighted average confidence for the majority action
        4. Shows "TIED" instead of "NOTHING" when there's an actual tie
        5. Clear frontend logging: "Swarm -> BUY | 62% sure"
        """
        try:
            # Track votes with confidence scores
            votes = {"BUY": [], "SELL": [], "NOTHING": []}  # Lists of confidence scores
            model_votes = []  # For detailed logging
            model_index = 1

            cprint("\n📊 Individual Model Votes:", "cyan", attrs=["bold"])

            for provider, data in swarm_result["responses"].items():
                if not data["success"]:
                    cprint(f"   ⚠️ Model {model_index} ({provider}): Failed (skipping)", "yellow")
                    model_votes.append(f"Model {model_index} ({provider}): FAILED")
                    model_index += 1
                    continue

                response_text = data["response"].strip() if data["response"] else ""
                response_upper = response_text.upper()

                # Parse vote AND confidence with new format
                action, confidence = self._parse_vote_from_response(response_upper)

                # Store confidence score for this action
                votes[action].append(confidence)

                # Format vote display
                vote_display = f"Model {model_index} - {action} | {confidence}%"
                model_votes.append(f"Model {model_index} ({provider}): {action} | {confidence}%")

                # Color-coded console output
                if action == "BUY":
                    cprint(f"   ✅ {vote_display}", "green")
                elif action == "SELL":
                    cprint(f"   🔴 {vote_display}", "red")
                else:
                    cprint(f"   ⏸️ {vote_display}", "cyan")

                model_index += 1

            # Count total valid votes
            total_votes = sum(len(v) for v in votes.values())
            if total_votes == 0:
                cprint("❌ No valid responses from swarm - defaulting to NOTHING", "red")
                add_console_log("Swarm -> NOTHING | 0% (no responses)", "warning")
                return "NOTHING", 0, "No valid responses from swarm"

            # Calculate vote counts and average confidence per action
            vote_counts = {action: len(confs) for action, confs in votes.items()}
            avg_confidences = {}
            for action, confs in votes.items():
                if confs:
                    avg_confidences[action] = int(sum(confs) / len(confs))
                else:
                    avg_confidences[action] = 0

            # Find majority action
            majority_action = max(vote_counts, key=vote_counts.get)
            majority_count = vote_counts[majority_action]

            # Check for ties - look at top 2 vote counts
            sorted_counts = sorted(vote_counts.values(), reverse=True)
            has_tie = len(sorted_counts) >= 2 and sorted_counts[0] == sorted_counts[1] and sorted_counts[0] > 0

            if has_tie:
                # Find which actions are tied
                tied_actions = [a for a, c in vote_counts.items() if c == majority_count]
                tie_avg_confidence = int(sum(avg_confidences[a] for a in tied_actions) / len(tied_actions))

                cprint(
                    f"\n⚠️ TIE DETECTED: {', '.join(tied_actions)} each have {majority_count} votes",
                    "yellow",
                    attrs=["bold"]
                )

                # Log to frontend with TIED status
                add_console_log(f"Swarm -> TIED | {tie_avg_confidence}% sure", "warning")

                reasoning = f"🌊 Swarm Consensus: TIED ({total_votes} models voted)\n\n"
                reasoning += "Vote Breakdown:\n"
                for action in ["BUY", "SELL", "NOTHING"]:
                    count = vote_counts[action]
                    avg = avg_confidences[action]
                    reasoning += f"   {action}: {count} votes (avg {avg}% confidence)\n"
                reasoning += f"\n⚠️ TIE between: {', '.join(tied_actions)}\n"
                reasoning += "   → Conservative approach: No action taken\n\n"
                reasoning += "Individual Votes:\n"
                reasoning += "\n".join(f"   {vote}" for vote in model_votes)

                return "NOTHING", tie_avg_confidence, reasoning

            # Calculate final confidence as weighted average of winning action's votes
            final_confidence = avg_confidences[majority_action]
            vote_percentage = int((majority_count / total_votes) * 100)

            # Check minimum confidence threshold
            if final_confidence < MIN_SWARM_CONFIDENCE and majority_action != "NOTHING":
                cprint(
                    f"\n⚠️ LOW CONFIDENCE: {final_confidence}% < {MIN_SWARM_CONFIDENCE}% threshold",
                    "yellow",
                    attrs=["bold"]
                )
                cprint(f"   → Downgrading {majority_action} to NOTHING", "yellow")

                add_console_log(f"Swarm -> NOTHING | {final_confidence}% (low confidence)", "warning")

                reasoning = f"🌊 Swarm Consensus: LOW CONFIDENCE ({total_votes} models voted)\n\n"
                reasoning += "Vote Breakdown:\n"
                for action in ["BUY", "SELL", "NOTHING"]:
                    count = vote_counts[action]
                    avg = avg_confidences[action]
                    reasoning += f"   {action}: {count} votes (avg {avg}% confidence)\n"
                reasoning += f"\n⚠️ Confidence {final_confidence}% below {MIN_SWARM_CONFIDENCE}% threshold\n"
                reasoning += f"   Original: {majority_action} | Downgraded to: NOTHING\n\n"
                reasoning += "Individual Votes:\n"
                reasoning += "\n".join(f"   {vote}" for vote in model_votes)

                return "NOTHING", final_confidence, reasoning

            # Normal case: clear majority above threshold
            action_emoji = "📈" if majority_action == "BUY" else "📉" if majority_action == "SELL" else "⏸️"

            cprint(
                f"\n🌊 Swarm Consensus: {majority_action} | {final_confidence}% sure ({majority_count}/{total_votes} votes)",
                "cyan",
                attrs=["bold"]
            )

            # Log to frontend with clear format
            add_console_log(f"Swarm -> {majority_action} | {final_confidence}% sure", "trade")

            reasoning = f"🌊 Swarm Consensus: {majority_action} ({total_votes} models voted)\n\n"
            reasoning += "Vote Breakdown:\n"
            for action in ["BUY", "SELL", "NOTHING"]:
                count = vote_counts[action]
                avg = avg_confidences[action]
                reasoning += f"   {action}: {count} votes (avg {avg}% confidence)\n"
            reasoning += f"\n{action_emoji} Final Decision: {majority_action} | {final_confidence}% confident\n"
            reasoning += f"   ({majority_count}/{total_votes} models agreed = {vote_percentage}% consensus)\n\n"
            reasoning += "Individual Votes:\n"
            reasoning += "\n".join(f"   {vote}" for vote in model_votes)

            return majority_action, final_confidence, reasoning

        except Exception as e:
            cprint(f"❌ Error calculating swarm consensus: {e}", "red")
            add_console_log(f"Swarm -> ERROR | 0%", "error")
            return "NOTHING", 0, f"Error calculating consensus: {str(e)}"

    def _parse_vote_from_response(self, response_upper):
        """
        Parse a vote from the model response with strict matching.
        Now extracts both action AND confidence score.

        Expected format: "BUY | 85%" or "SELL | 70%" or "NOTHING | 45%"
        Falls back to action-only parsing if confidence not found.

        Returns: Tuple of (action: str, confidence: int)
            - action: "BUY", "SELL", or "NOTHING"
            - confidence: 0-100 (defaults to 50 if not found)
        """
        # Clean the response (remove extra whitespace, newlines)
        response_clean = response_upper.strip().split('\n')[0].strip()

        # Default confidence if not found
        confidence = 50

        # Try to parse "ACTION | XX%" format
        if "|" in response_clean:
            parts = response_clean.split("|")
            action_part = parts[0].strip()
            confidence_part = parts[1].strip() if len(parts) > 1 else ""

            # Extract confidence number
            confidence_match = re.search(r'(\d+)', confidence_part)
            if confidence_match:
                confidence = min(100, max(0, int(confidence_match.group(1))))

            # Parse action from the first part
            response_clean = action_part

        # Parse action with priority matching
        action = "NOTHING"

        # Priority 1: Exact match
        if response_clean in ["BUY"]:
            action = "BUY"
        elif response_clean in ["SELL"]:
            action = "SELL"
        elif response_clean in ["DO NOTHING", "NOTHING", "HOLD", "WAIT"]:
            action = "NOTHING"
        # Priority 2: Starts with action word
        elif response_clean.startswith("BUY"):
            action = "BUY"
        elif response_clean.startswith("SELL"):
            action = "SELL"
        elif response_clean.startswith("DO NOTHING") or response_clean.startswith("NOTHING"):
            action = "NOTHING"
        # Priority 3: Contains action word (fallback)
        elif "SELL" in response_clean:
            action = "SELL"
        elif "BUY" in response_clean:
            action = "BUY"

        return action, confidence

    def fetch_all_open_positions(self):
        """
        Fetch ALL open positions across all symbols with age tracking.

        CRITICAL FIX: Now iterates through ALL subpositions, not just the first one.
        This ensures we detect multiple positions for the same symbol.
        """
        cprint("\n" + "=" * 60, "cyan")
        cprint("📊 FETCHING ALL OPEN POSITIONS", "white", "on_blue", attrs=["bold"])
        cprint("=" * 60, "cyan")

        all_positions = {}
        exchange_positions = {}  # For syncing with tracker
        # CRITICAL: Use self.symbols (instance variable) NOT global SYMBOLS/MONITORED_TOKENS
        # This ensures user-configured symbols from settings are respected
        check_tokens = self.symbols
        total_position_count = 0

        for symbol in check_tokens:
            try:
                # get_position returns: positions (list), im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long
                # The 'positions' list contains ALL subpositions for this symbol
                positions_list, im_in_pos, _, _, _, _, _ = n.get_position(
                    symbol, self.account
                )

                if im_in_pos and positions_list:
                    # CRITICAL FIX: Iterate through ALL positions, not just the first one
                    for pos in positions_list:
                        pos_size = float(pos.get("szi", 0))
                        entry_px = float(pos.get("entryPx", 0))
                        pnl_perc = float(pos.get("returnOnEquity", 0)) * 100
                        is_long = pos_size > 0

                        if pos_size == 0:
                            continue

                        # Get position age from tracker
                        age_hours = 0.0
                        if POSITION_TRACKER_AVAILABLE:
                            age_hours = get_position_age_hours(symbol)

                        position_data = {
                            "symbol": symbol,
                            "size": pos_size,
                            "entry_price": entry_px,
                            "pnl_percent": pnl_perc,
                            "is_long": is_long,
                            "side": "LONG" if is_long else "SHORT",
                            "age_hours": age_hours,
                        }

                        # Store for tracker sync (use combined size for all positions of this symbol)
                        if symbol not in exchange_positions:
                            exchange_positions[symbol] = {
                                "entry_price": entry_px,
                                "size": 0,
                                "is_long": is_long
                            }
                        exchange_positions[symbol]["size"] += pos_size

                        if symbol not in all_positions:
                            all_positions[symbol] = []
                        all_positions[symbol].append(position_data)
                        total_position_count += 1

                        # Include age in display
                        age_str = f"{age_hours:.1f}h" if age_hours > 0 else "NEW"
                        cprint(
                            f"   {symbol:<10} | {position_data['side']:<10} | "
                            f"Size: {pos_size:>10.4f} | Entry: ${entry_px:>10.2f} | "
                            f"PnL: {pnl_perc:>6.2f}% | Age: {age_str}",
                            "cyan",
                        )

            except Exception as e:
                cprint(f"   ❌ Error fetching {symbol}: {e}", "red")
                continue

        # Show total count (including subpositions)
        cprint(f"\n   📊 Total positions detected: {total_position_count}", "yellow", attrs=["bold"])

        # Sync tracker with actual exchange positions
        if POSITION_TRACKER_AVAILABLE and exchange_positions:
            added, removed = sync_with_exchange_positions(exchange_positions)
            if added > 0:
                cprint(f"   📍 Added {added} position(s) to tracker", "yellow")
            if removed > 0:
                cprint(f"   📍 Removed {removed} stale position(s) from tracker", "yellow")

        if not all_positions:
            cprint("   ℹ️  No open positions found", "yellow")

        cprint("=" * 60 + "\n", "cyan")
        return all_positions

    def validate_close_decision(self, symbol, pnl_percent, age_hours, ai_confidence, ai_decision="CLOSE"):
        """
        Three-Tier Position Close Validation System.

        Tier 0: Emergency Stop Loss (-2% or worse) - FORCE CLOSE
        Tier 1: Profit Target (+0.5% or better) - AI decides
        Tier 2: Age Gating (young positions < 1.5h need protection)
        Tier 3: Mature Position Analysis with loss-adjusted confidence

        Returns: (should_close: bool, reason: str)
        """
        cprint(f"\n🔍 VALIDATING CLOSE DECISION FOR {symbol}:", "yellow", attrs=["bold"])
        cprint(f"   📊 P&L: {pnl_percent:.2f}% | Age: {age_hours:.1f}h | AI Confidence: {ai_confidence}%", "cyan")

        # Use the close validator if available
        if CLOSE_VALIDATOR_AVAILABLE:
            result = validate_close_decision(
                symbol=symbol,
                pnl_percent=pnl_percent,
                age_hours=age_hours,
                ai_decision=ai_decision,
                ai_confidence=ai_confidence
            )

            # Display validation result
            tier_names = {0: "STOP LOSS", 1: "PROFIT TARGET", 2: "YOUNG POSITION", 3: "MATURE POSITION"}
            tier_name = tier_names.get(result.tier_triggered, "UNKNOWN")

            if result.decision == CloseDecision.FORCE_CLOSE:
                cprint(f"   🚨 TIER {result.tier_triggered} ({tier_name}): FORCE CLOSE", "red", attrs=["bold"])
                cprint(f"   💡 {result.reason}", "red")
                add_console_log(f"STOP LOSS: Closing {symbol} at {pnl_percent:.2f}%", "warning")
                return True, result.reason

            elif result.decision == CloseDecision.FORCE_TAKE_PROFIT:
                cprint(f"   🎯 TIER {result.tier_triggered} (TAKE PROFIT): FORCE CLOSE", "green", attrs=["bold"])
                cprint(f"   💡 {result.reason}", "green")
                add_console_log(f"TAKE PROFIT: Closing {symbol} at +{pnl_percent:.2f}%", "success")
                return True, result.reason

            elif result.decision == CloseDecision.CLOSE:
                cprint(f"   ✅ TIER {result.tier_triggered} ({tier_name}): CLOSE APPROVED", "green", attrs=["bold"])
                cprint(f"   📈 Confidence: {result.original_confidence}% → {result.adjusted_confidence}% (boost: +{result.confidence_boost}%)", "green")
                cprint(f"   💡 {result.reason}", "green")
                return True, result.reason

            elif result.decision == CloseDecision.PROTECTED:
                cprint(f"   🛡️ TIER {result.tier_triggered} ({tier_name}): PROTECTED", "cyan", attrs=["bold"])
                cprint(f"   💡 {result.reason}", "cyan")
                return False, result.reason

            else:  # KEEP
                cprint(f"   ⏸️ TIER {result.tier_triggered} ({tier_name}): KEEP", "yellow", attrs=["bold"])
                cprint(f"   📉 Confidence: {result.original_confidence}% → {result.adjusted_confidence}% (boost: +{result.confidence_boost}%)", "yellow")
                cprint(f"   💡 {result.reason}", "yellow")
                return False, result.reason

        # Fallback to simple validation if close validator not available
        else:
            # Simple fallback: Stop loss at -2%, take profit at +5%, otherwise AI decides
            if pnl_percent <= STOP_LOSS_THRESHOLD:
                cprint(f"   🚨 STOP LOSS TRIGGERED: {pnl_percent:.2f}%", "red", attrs=["bold"])
                add_console_log(f"STOP LOSS: Closing {symbol} at {pnl_percent:.2f}%", "warning")
                return True, f"Stop loss at {pnl_percent:.2f}%"

            if pnl_percent >= TAKE_PROFIT_THRESHOLD:
                cprint(f"   🎯 TAKE PROFIT TRIGGERED: +{pnl_percent:.2f}%", "green", attrs=["bold"])
                add_console_log(f"TAKE PROFIT: Closing {symbol} at +{pnl_percent:.2f}%", "success")
                return True, f"Take profit at +{pnl_percent:.2f}%"

            if pnl_percent >= PROFIT_TARGET_THRESHOLD and ai_confidence >= MIN_CONFIDENCE_TO_CLOSE:
                cprint(f"   ✅ PROFIT TARGET: {pnl_percent:.2f}%, confidence {ai_confidence}%", "green")
                return True, f"Profit target at {pnl_percent:.2f}%"

            cprint(f"   ⏸️ KEEP: P&L {pnl_percent:.2f}%, confidence {ai_confidence}%", "yellow")
            return False, f"Keep position - P&L {pnl_percent:.2f}%, confidence {ai_confidence}%"

    def analyze_open_positions_with_ai(self, positions_data, market_data):
        """Enhanced with guaranteed TP/SL enforcement"""
        if not positions_data:
            return {}

        cprint("\n" + "=" * 60, "yellow")
        cprint("📊 AI ANALYZING OPEN POSITIONS", "white", "on_magenta", attrs=["bold"])
        add_console_log("Analyzing Open Positions", "info")
        cprint("=" * 60, "yellow")

        # CRITICAL: Check TP/SL thresholds FIRST - force close regardless of AI analysis
        validated_decisions = {}
        for symbol, positions in positions_data.items():
            for pos in positions:
                pnl_percent = pos["pnl_percent"]
                
                # Force TP/SL regardless of AI analysis
                if pnl_percent >= TAKE_PROFIT_THRESHOLD:
                    validated_decisions[symbol] = {
                        "action": "CLOSE", 
                        "reasoning": f"TAKE PROFIT: {pnl_percent:.2f}% >= {TAKE_PROFIT_THRESHOLD}%",
                        "confidence": 100
                    }
                    cprint(f"🚨 {symbol}: TAKE PROFIT TRIGGERED - {pnl_percent:.2f}%", "red", attrs=["bold"])
                    add_console_log(f"TAKE PROFIT: Closing {symbol} at +{pnl_percent:.2f}%", "success")
                    continue
                elif pnl_percent <= STOP_LOSS_THRESHOLD:
                    validated_decisions[symbol] = {
                        "action": "CLOSE",
                        "reasoning": f"STOP LOSS: {pnl_percent:.2f}% <= {STOP_LOSS_THRESHOLD}%",
                        "confidence": 100
                    }
                    cprint(f"🚨 {symbol}: STOP LOSS TRIGGERED - {pnl_percent:.2f}%", "red", attrs=["bold"])
                    add_console_log(f"STOP LOSS: Closing {symbol} at {pnl_percent:.2f}%", "warning")
                    continue

        # Build position summary for remaining positions
        position_summary = []
        for symbol, positions in positions_data.items():
            if symbol in validated_decisions:
                continue  # Skip positions already handled by TP/SL
                
            for pos in positions:
                position_summary.append({
                    "symbol": symbol,
                    "side": "LONG" if pos["is_long"] else "SHORT",
                    "size": pos["size"],
                    "entry_price": pos["entry_price"],
                    "current_pnl": pos["pnl_percent"],
                    "age_hours": pos["age_hours"],
                })

        # Format market conditions
        market_summary = {}
        for symbol in positions_data.keys():
            if symbol in validated_decisions:
                continue  # Skip positions already handled
                
            if symbol in market_data:
                df = market_data[symbol]
                if not df.empty:
                    latest = df.iloc[-1]

                    # Robustly detect the correct close column
                    if "Close" in df.columns:
                        current_price = latest["Close"]
                    elif "close" in df.columns:
                        current_price = latest["close"]
                    elif "close_price" in df.columns:
                        current_price = latest["close_price"]
                    elif "c" in df.columns:
                        current_price = latest["c"]
                    elif "price" in df.columns:
                        current_price = latest["price"]
                    else:
                        cprint(f"⚠️ No close price column found for {symbol}, skipping...", "yellow")
                        continue

                    market_summary[symbol] = {
                        "current_price": current_price,
                        "ma20": latest.get("MA20", 0),
                        "ma40": latest.get("MA40", 0),
                        "rsi": latest.get("RSI", 0),
                        "trend": "Bullish" if current_price > latest.get("MA20", 0) else "Bearish",
                    }

        # Only analyze positions that weren't force-closed by TP/SL
        if position_summary:
            user_prompt = f"""Analyze these open positions:

POSITIONS:
{json.dumps(position_summary, indent=2)}

CURRENT MARKET CONDITIONS:
{json.dumps(market_summary, indent=2)}

For each position, decide KEEP or CLOSE with reasoning.
Return ONLY valid JSON with the following structure:
{{
  "SYMBOL": {{
     "action": "KEEP" or "CLOSE",
     "reasoning": "short explanation"
  }}
}}"""

            try:
                response = self.chat_with_ai(POSITION_ANALYSIS_PROMPT, user_prompt)

                # Strip Markdown fences if model wrapped response in code blocks
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                # Try safe JSON extraction first
                decisions = extract_json_from_text(response)
                if not decisions:
                    cprint("⚠️ AI response not valid JSON. Attempting text fallback...", "yellow")

                    text = response.lower()
                    decisions = {}
                    for symbol in position_summary:
                        sym = symbol["symbol"]
                        if sym.lower() in text:
                            if "close" in text or "sell" in text:
                                decisions[sym] = {
                                    "action": "CLOSE",
                                    "reasoning": "Detected CLOSE or SELL keyword in fallback parsing.",
                                    "confidence": 60  # Default confidence for fallback
                                }
                            elif "keep" in text or "hold" in text or "open" in text:
                                decisions[sym] = {
                                    "action": "KEEP",
                                    "reasoning": "Detected KEEP/HOLD keyword in fallback parsing.",
                                    "confidence": 0
                                }
                            else:
                                decisions[sym] = {
                                    "action": "KEEP",
                                    "reasoning": "No clear directive, default KEEP.",
                                    "confidence": 0
                                }
                        else:
                            decisions[sym] = {
                                "action": "KEEP",
                                "reasoning": "Symbol not mentioned, default KEEP.",
                                "confidence": 0
                            }

                    cprint(f"🧠 Fallback interpreted decisions: {decisions}", "cyan")

                if not decisions:
                    cprint("❌ Error: Could not interpret AI analysis at all.", "red")
                    cprint(f"   Raw response: {response}", "yellow")
                    return validated_decisions

                # ============================================================================
                # APPLY 3-TIER VALIDATION SYSTEM
                # ============================================================================
                cprint("\n" + "=" * 60, "magenta")
                cprint("🛡️ APPLYING 3-TIER VALIDATION SYSTEM", "white", "on_magenta", attrs=["bold"])
                cprint("=" * 60, "magenta")

                for symbol, decision in decisions.items():
                    action = decision.get("action", "KEEP")
                    reason = decision.get("reasoning", "")
                    ai_confidence = int(decision.get("confidence", 0))

                    if action.upper() == "CLOSE":
                        # Get position data for validation
                        pos_data = positions_data.get(symbol, [{}])[0]
                        pnl_percent = pos_data.get("pnl_percent", 0)
                        age_hours = pos_data.get("age_hours", 0)

                        # Run validation
                        should_close, validation_reason = self.validate_close_decision(
                            symbol, pnl_percent, age_hours, ai_confidence
                        )

                        if should_close:
                            validated_decisions[symbol] = {
                                "action": "CLOSE",
                                "reasoning": f"{reason} | Validation: {validation_reason}",
                                "confidence": ai_confidence
                            }
                            cprint(f"✅ {symbol}: CLOSE APPROVED", "green", attrs=["bold"])
                        else:
                            validated_decisions[symbol] = {
                                "action": "KEEP",
                                "reasoning": f"AI suggested CLOSE but validation BLOCKED: {validation_reason}",
                                "confidence": 0
                            }
                            cprint(f"🛡️ {symbol}: CLOSE BLOCKED → FORCING KEEP", "cyan", attrs=["bold"])
                            add_console_log(f"🛡️ {symbol} CLOSE blocked: {validation_reason}", "warning")
                    else:
                        # KEEP decision - no validation needed
                        validated_decisions[symbol] = decision

            except Exception as e:
                cprint(f"❌ Error in AI analysis: {e}", "red")
                import traceback
                traceback.print_exc()

        # Print final validated decisions
        cprint("\n" + "=" * 60, "magenta")
        cprint("🎯 FINAL VALIDATED DECISIONS:", "white", "on_magenta", attrs=["bold"])
        cprint("=" * 60, "magenta")

        for symbol, decision in validated_decisions.items():
            action = decision.get("action", "UNKNOWN")
            reason = decision.get("reasoning", "")
            confidence = decision.get("confidence", 0)
            color = "red" if action.upper() == "CLOSE" else "green"
            cprint(f"   {symbol:<10} → {action:<6} | {reason}", color)
            # Short format for dashboard: "SYMBOL -> ACTION"
            add_console_log(f"{symbol} -> {action}", "info")

            # Short format for dashboard
            if action.upper() == "CLOSE":
                add_console_log(f"{symbol} -> CLOSE ({confidence}% Sure)", "warning")
            else:
                add_console_log(f"{symbol} -> KEEP", "info")

        cprint("=" * 60 + "\n", "magenta")
        return validated_decisions

    def execute_position_closes(self, close_decisions):
        """Execute closes for positions marked by AI"""
        if not close_decisions:
            return

        cprint("\n" + "=" * 60, "red")
        cprint("🔄 EXECUTING POSITION CLOSES", "white", "on_red", attrs=["bold"])
        cprint("=" * 60, "red")

        closed_count = 0

        for symbol, decision in close_decisions.items():
            if decision["action"] == "CLOSE":
                try:
                    cprint(f"\n   📉 Closing {symbol}...", "yellow")
                    cprint(f"   💡 Reason: {decision['reasoning']}", "white")

                    close_result = n.close_complete_position(symbol, self.account)

                    if close_result:
                        # Remove from position tracker
                        if POSITION_TRACKER_AVAILABLE:
                            remove_position(symbol)
                            cprint(f"   📍 Removed {symbol} from position tracker", "cyan")

                        cprint(f"✅ {symbol} position closed successfully", "green", attrs=["bold"])
                        add_console_log(f"✅ Closed {symbol} | Reason: {decision['reasoning']}", "success")

                        closed_count += 1
                    else:
                        cprint(f"   ⚠️ Position close returned False for {symbol}", "yellow")
                        add_console_log(f"⚠️ Close may have failed for {symbol}", "warning")

                    time.sleep(2)

                except Exception as e:
                    cprint(f"   ❌ Error closing {symbol}: {e}", "red")
                    import traceback
                    traceback.print_exc()

        if closed_count > 0:
            cprint(
                f"\n✨ Successfully closed {closed_count} position(s)",
                "white",
                "on_green",
                attrs=["bold"],
            )
        else:
            cprint("\n   ℹ️  No positions needed closing", "cyan")

        cprint("=" * 60 + "\n", "red")

    # ==================================================
    # Strategy Context Helpers
    # ==================================================

    def _get_cached_strategy_context(self, token):
        try:
            now = datetime.utcnow()

            cache = self._strategy_context_cache.get(token)
            if cache and cache["expires_at"] > now:
                return cache["data"]

            if not self.strategy_agent:
                return None

            strategy_context = self.strategy_agent.get_enriched_context(token)

            self._strategy_context_cache[token] = {
                "data": strategy_context,
                "expires_at": now + timedelta(seconds=self.STRATEGY_CONTEXT_TTL),
            }

            return strategy_context

        except Exception as e:
            cprint(f"⚠️ Strategy context error: {e}", "yellow")
            return None


    def _format_strategy_context_text(self, strategy_context):
        if not strategy_context:
            return "No strategy intelligence available.", {}

        lines = []

        lines.append("STRATEGY INTELLIGENCE (JSON)")
        lines.append(json.dumps(strategy_context, indent=2))

        aggregate = strategy_context.get("aggregate", {})

        lines.append("\nSTRATEGY SUMMARY")
        lines.append(f"- Direction bias: {aggregate.get('direction_bias')}")
        lines.append(f"- Confidence: {aggregate.get('confidence')}")
        lines.append(
            f"- Suggested allocation (%): "
            f"{aggregate.get('suggested_allocation_pct')}"
        )
        lines.append(f"- Conflict level: {aggregate.get('conflict_level')}")
        lines.append(f"- Timestamp: {strategy_context.get('timestamp')}")

        return "\n".join(lines), strategy_context
   

   
    def _compute_engine_signals(self, market_data: dict) -> dict:
        """Generate Engine v6.1 signals from OHLCV market data."""
        signals = {}
        if not self.strategy_engine:
            return signals
        for token, data in market_data.items():
            try:
                df = data if isinstance(data, pd.DataFrame) else None
                if df is None or df.empty:
                    continue
                # Normalize column names
                df = df.copy()
                df.columns = [str(c).lower().strip() for c in df.columns]
                required = {"open", "high", "low", "close", "volume"}
                if not required.issubset(set(df.columns)):
                    continue
                sig = self.strategy_engine.generate_signals(df, symbol=token)
                signals[token] = sig
            except Exception as e:
                cprint(f"⚠️ Engine v6.1 failed for {token}: {e}", "yellow")
        return signals

    def analyze_market_data(self, token, market_data):
        """Analyze market data using AI model (single or swarm mode)"""
        try:
            if token in EXCLUDED_TOKENS:
                print(f"⚠️ Skipping analysis for excluded token: {token}")
                return None

            # Fetch current position context
            position_context = "CURRENT POSITION: None (You have no exposure)."

            try:
                raw_pos_data = n.get_position(token, self.account)
                _, im_in_pos, pos_size, _, entry_px, pnl_perc, is_long = raw_pos_data

                if im_in_pos:
                    side = "LONG" if is_long else "SHORT"

                    if entry_px == 0 and pnl_perc == 0:
                        position_context = (
                            f"CURRENT POSITION: ✅ Active {side} (Spot) | Size: {pos_size}"
                        )
                    else:
                        position_context = (
                            f"CURRENT POSITION: ✅ Active {side} | "
                            f"Size: {pos_size} | Entry: ${entry_px:.4f} | "
                            f"PnL: {pnl_perc:.2f}%"
                        )
            except Exception as e:
                cprint(f"⚠️ Error fetching position context: {e}", "yellow")

            cprint(f"   ℹ️  Context: {position_context}", "cyan")

            # ============================================================
            # SWARM MODE
            # ============================================================
            if self.use_swarm_mode:
                num_models = len(self.swarm.active_models) if self.swarm else 6
                cprint(
                    f"\n🌊 Analyzing {token[:8]}... with SWARM ({num_models} AI models voting)",
                    "cyan",
                    attrs=["bold"],
                )

                base_market_data = self._format_market_data_for_swarm(token, market_data)
                formatted_data = f"{position_context}\n\n{base_market_data}"

                swarm_result = self.swarm.query(
                    prompt=formatted_data, system_prompt=SWARM_TRADING_PROMPT
                )

                if not swarm_result:
                    cprint(f"❌ No response from swarm for {token}", "red")
                    return None

                action, confidence, reasoning = self._calculate_swarm_consensus(
                    swarm_result
                )

                # Store the recommendation only — no trade execution here
                self.recommendations_df = pd.concat(
                    [
                        self.recommendations_df,
                        pd.DataFrame(
                            [
                                {
                                    "token": token,
                                    "action": action,
                                    "confidence": confidence,
                                    "reasoning": reasoning,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                cprint(f"✅ Swarm analysis complete for {token[:8]}!", "green")
                add_console_log(f"✅ Swarm  {token} -> {action} | {confidence}% Sure", "success")

                # Return raw result for dashboard or debugging
                return swarm_result

            # ============================================================
            # SINGLE MODEL MODE
            # ============================================================
            else:
                # -----------------------------
                # Enriched strategy context
                # -----------------------------
                try:
                    # robust token name detection
                    if isinstance(market_data, dict):
                        token_name = market_data.get("symbol") or market_data.get("token") or token
                    else:
                        token_name = token

                    strat_obj = None
                    strategy_context_text = "No strategy intelligence available."
                    strategy_context_json = {}

                    # Attempt to get enriched context from StrategyAgent (cached)
                    try:
                        strat_obj = self._get_cached_strategy_context(token_name)
                    except Exception as e:
                        cprint(f"⚠️ Error fetching strategy context for {token_name}: {e}", "yellow")
                        strat_obj = None

                    if strat_obj:
                        strategy_context_text, strategy_context_json = self._format_strategy_context_text(strat_obj)
                        add_console_log("Strategies loaded", "success")

                    else:
                        # fallback to legacy market_data['strategy_signals'] if present
                        if isinstance(market_data, dict) and "strategy_signals" in market_data:
                            try:
                                strategy_context_text = (
                                    "Strategy Signals Available:\n" +
                                    json.dumps(market_data["strategy_signals"], indent=2)
                                )
                                strategy_context_json = {"legacy_signals": market_data["strategy_signals"]}
                            except Exception:
                                strategy_context_text = "Strategy Signals Available (unserializable)."
                                strategy_context_json = {"legacy_signals": str(market_data.get("strategy_signals"))}
                        else:
                            strategy_context_text = "No strategy intelligence available."
                            strategy_context_json = {}

                    # store last context for debug / dashboard
                    self.last_strategy_context = strategy_context_json

                except Exception as e:
                    cprint(f"⚠️ Failed to prepare strategy context: {e}", "yellow")
                    strategy_context_text = "No strategy intelligence available."

                response = self.chat_with_ai(
                    TRADING_PROMPT.format(
                        strategy_context=strategy_context_text,
                        position_context=position_context,
                    ),
                    f"Market Data to Analyze:\n{market_data}",
                )

                if not response:
                    cprint(f"❌ No response from AI for {token}", "red")
                    return None

                lines = response.split("\n")
                action = lines[0].strip() if lines else "NOTHING"

                confidence = 0
                for line in lines:
                    if "confidence" in line.lower():
                        try:
                            confidence = int("".join(filter(str.isdigit, line)))
                        except Exception:
                            confidence = 50

                reasoning = (
                    "\n".join(lines[1:]) if len(lines) > 1 else "No detailed reasoning provided"
                )

                self.recommendations_df = pd.concat(
                    [
                        self.recommendations_df,
                        pd.DataFrame(
                            [
                                {
                                    "token": token,
                                    "action": action,
                                    "confidence": confidence,
                                    "reasoning": reasoning,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                add_console_log(f"Analysis Complete for {token[:4]}...", "info")
                add_console_log(f"{token} -> {action} | {confidence}%", "success")

                return response

        except Exception as e:
            print(f"❌ Error in AI analysis: {str(e)}")
            self.recommendations_df = pd.concat(
                [
                    self.recommendations_df,
                    pd.DataFrame(
                        [
                            {
                                "token": token,
                                "action": "NOTHING",
                                "confidence": 0,
                                "reasoning": f"Error during analysis: {str(e)}",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            return None



    def allocate_portfolio(self):
        """
        AI-Driven Smart Portfolio Allocation

        Uses AI to analyze:
        1. Current open positions with P&L
        2. AI trading signals with confidence levels
        3. Available balance and risk parameters

        The AI decides optimal allocation including:
        - Which positions to close/reduce (reallocate capital)
        - Which positions to open/increase
        - Allocation amounts based on confidence

        Returns:
            list: List of action dictionaries from AI, or empty list if no actions
        """
        try:
            cprint("\n" + "=" * 60, "cyan")
            cprint("🧠 AI-DRIVEN SMART ALLOCATION", "white", "on_blue", attrs=["bold"])
            cprint("=" * 60, "cyan")

            # ================================================================
            # STEP 1: Collect Current Portfolio State
            # ================================================================
            open_positions = {}
            total_position_value = 0

            for sym in self.symbols:
                try:
                    if EXCHANGE == "HYPERLIQUID":
                        pos_data = n.get_position(sym, self.account)
                    else:
                        pos_data = n.get_position(sym)

                    _, im_in_pos, pos_size, _, entry_px, pnl_pct, is_long = pos_data

                    if im_in_pos and pos_size != 0:
                        notional = abs(float(pos_size) * float(entry_px))
                        margin = notional / LEVERAGE
                        open_positions[sym] = {
                            "direction": "LONG" if is_long else "SHORT",
                            "size": abs(float(pos_size)),
                            "entry_price": float(entry_px),
                            "notional_usd": round(notional, 2),
                            "margin_usd": round(margin, 2),
                            "pnl_percent": round(float(pnl_pct), 2),
                        }
                        total_position_value += margin
                except Exception:
                    continue

            # ================================================================
            # STEP 2: Collect AI Signals
            # ================================================================
            signals = []
            for _, row in self.recommendations_df.iterrows():
                token = row["token"]
                if token not in self.symbols:
                    continue

                # Skip SELL signals in LONG_ONLY mode (can't open shorts)
                if LONG_ONLY and row["action"] == "SELL" and token not in open_positions:
                    continue

                signals.append({
                    "symbol": token,
                    "action": row["action"],
                    "confidence": int(row["confidence"]),
                })

            if not signals:
                cprint("📊 No actionable signals. Skipping allocation.", "yellow")
                add_console_log("No actionable signals for allocation", "info")
                return []

            # ================================================================
            # STEP 3: Get Account Balance and Calculate Total Equity
            # ================================================================
            account_balance = get_account_balance(self.account)
            if account_balance <= 0:
                cprint("❌ Account balance is zero. Cannot allocate.", "red")
                return []

            # CRITICAL FIX: Calculate total equity including existing positions
            total_equity = account_balance + total_position_value
            available_balance = account_balance - total_position_value
            min_order_notional = 12.0  # HyperLiquid minimum

            cprint(f"💰 Account Balance (USDC): ${account_balance:.2f}", "cyan")
            cprint(f"💎 Total Equity (Balance + Positions): ${total_equity:.2f}", "cyan")
            cprint(f"📊 Positions Value: ${total_position_value:.2f}", "cyan")
            cprint(f"💵 Available Balance: ${available_balance:.2f}", "green")

            # ================================================================
            # STEP 4: Build Portfolio State Summary for AI
            # ================================================================
            if open_positions:
                portfolio_lines = ["OPEN POSITIONS:"]
                for sym, pos in open_positions.items():
                    pnl_emoji = "🟢" if pos["pnl_percent"] >= 0 else "🔴"
                    portfolio_lines.append(
                        f"  - {sym}: {pos['direction']} | ${pos['notional_usd']:.2f} notional | "
                        f"PnL: {pnl_emoji} {pos['pnl_percent']:+.2f}%"
                    )
                portfolio_lines.append(f"\nTotal Position Value: ${total_position_value:.2f} margin")
            else:
                portfolio_lines = ["OPEN POSITIONS: None"]

            portfolio_state = "\n".join(portfolio_lines)

            # Build signals summary
            signal_lines = []
            for sig in signals:
                emoji = "📈" if sig["action"] == "BUY" else "📉" if sig["action"] == "SELL" else "⏸️"
                in_position = "✓ IN POSITION" if sig["symbol"] in open_positions else ""
                signal_lines.append(
                    f"  - {sig['symbol']}: {emoji} {sig['action']} ({sig['confidence']}% confidence) {in_position}"
                )
            signals_text = "\n".join(signal_lines)

            cprint(f"\n📊 Portfolio State:", "cyan")
            cprint(portfolio_state, "white")
            cprint(f"\n📈 AI Signals:", "cyan")
            cprint(signals_text, "white")
            cprint(f"\n⏱️  Cycle Time: {SLEEP_BETWEEN_RUNS_MINUTES} min (minimum hold time)", "yellow")

            # ================================================================
            # STEP 5: Ask AI for Allocation Plan
            # ================================================================
            cprint("\n🧠 Consulting AI for optimal allocation...", "magenta", attrs=["bold"])
            add_console_log("Agent is analyzing allocation...", "info")

            prompt = SMART_ALLOCATION_PROMPT.format(
                portfolio_state=portfolio_state,
                signals=signals_text,
                available_balance=total_equity,  # CRITICAL: Use total equity instead of available balance
                leverage=LEVERAGE,
                max_position_pct=MAX_POSITION_PERCENTAGE,
                cash_buffer_pct=CASH_PERCENTAGE,
                min_order=min_order_notional,
                cycle_minutes=SLEEP_BETWEEN_RUNS_MINUTES
            )

            ai_response = self.chat_with_ai(
                "You are a portfolio allocation expert. Return ONLY valid JSON.",
                prompt
            )

            if not ai_response:
                cprint("❌ No response from AI. Using fallback allocation.", "red")
                return self._fallback_equal_allocation(signals, total_equity, open_positions)

            # ================================================================
            # STEP 6: Parse AI Response
            # ================================================================
            try:
                # Extract JSON from response
                allocation_plan = extract_json_from_text(ai_response)

                if not allocation_plan or "actions" not in allocation_plan:
                    cprint("⚠️ AI response missing 'actions'. Using fallback.", "yellow")
                    return self._fallback_equal_allocation(signals, total_equity, open_positions)

                actions = allocation_plan["actions"]

                # Validate and filter actions
                valid_actions = []
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    if "symbol" not in action or "action" not in action:
                        continue
                    if action["symbol"] not in self.symbols:
                        continue

                    # Skip HOLD actions - nothing to execute
                    if action["action"] == "HOLD":
                        cprint(f"   ⏸️ {action['symbol']}: HOLD - {action.get('reason', 'No change needed')}", "cyan")
                        continue

                    valid_actions.append(action)

                # ================================================================
                # STEP 7: Display AI Allocation Plan
                # ================================================================
                cprint("\n" + "=" * 60, "green")
                cprint("🎯 AI ALLOCATION PLAN:", "white", "on_green", attrs=["bold"])
                cprint("=" * 60, "green")

                for action in valid_actions:
                    action_type = action["action"]
                    symbol = action["symbol"]
                    reason = action.get("reason", "")

                    if action_type == "OPEN_LONG":
                        margin = action.get("margin_usd", 0)
                        cprint(f"   📈 {symbol}: OPEN LONG ${margin:.2f} margin - {reason}", "green")
                    elif action_type == "OPEN_SHORT":
                        margin = action.get("margin_usd", 0)
                        cprint(f"   📉 {symbol}: OPEN SHORT ${margin:.2f} margin - {reason}", "red")
                    elif action_type == "INCREASE":
                        margin = action.get("margin_usd", 0)
                        cprint(f"   ➕ {symbol}: INCREASE by ${margin:.2f} margin - {reason}", "cyan")
                    elif action_type == "REDUCE":
                        reduce_amt = action.get("reduce_by_usd", 0)
                        cprint(f"   ➖ {symbol}: REDUCE by ${reduce_amt:.2f} notional - {reason}", "yellow")
                    elif action_type == "CLOSE":
                        cprint(f"   ❌ {symbol}: CLOSE position - {reason}", "red")

                if allocation_plan.get("reasoning"):
                    cprint(f"\n   💡 Strategy: {allocation_plan['reasoning']}", "magenta")

                cprint("=" * 60 + "\n", "green")
                add_console_log(f"Planned {len(valid_actions)} actions", "info")

                return valid_actions

            except Exception as e:
                cprint(f"⚠️ Error parsing AI response: {e}", "yellow")
                cprint(f"   Response: {ai_response[:200]}...", "white")
                return self._fallback_equal_allocation(signals, total_equity, open_positions)

        except Exception as e:
            cprint(f"❌ Error in portfolio allocation: {str(e)}", "red")
            import traceback
            traceback.print_exc()
            return []

    def _fallback_equal_allocation(self, signals, available_balance, open_positions):
        """
        Fallback to equal distribution when AI allocation fails.
        Returns list of action dicts in the same format as AI.
        """
        cprint("\n📊 Using fallback equal distribution...", "yellow")

        actionable_signals = [s for s in signals if s["action"] in ["BUY", "SELL"]]
        if not actionable_signals:
            return []

        # Filter out signals where we already have aligned position
        new_signals = []
        for sig in actionable_signals:
            sym = sig["symbol"]
            if sym in open_positions:
                pos = open_positions[sym]
                # If signal aligns with position, skip (already positioned)
                if (sig["action"] == "BUY" and pos["direction"] == "LONG") or \
                   (sig["action"] == "SELL" and pos["direction"] == "SHORT"):
                    continue
            new_signals.append(sig)

        if not new_signals:
            cprint("   No new positions to open.", "cyan")
            return []

        # Calculate margin per position
        usable_margin = available_balance * (MAX_POSITION_PERCENTAGE / 100)
        cash_buffer = available_balance * (CASH_PERCENTAGE / 100)

        # Prevent division by zero
        if len(new_signals) == 0:
            cprint("   No signals after filtering.", "cyan")
            return []

        margin_per_position = (usable_margin - cash_buffer) / len(new_signals)
        min_margin = 12 / LEVERAGE

        if margin_per_position < min_margin:
            # Take only highest confidence signals
            new_signals.sort(key=lambda x: x["confidence"], reverse=True)
            max_positions = int((usable_margin - cash_buffer) / min_margin)
            new_signals = new_signals[:max(1, max_positions)]

            # Prevent division by zero after filtering
            if len(new_signals) == 0:
                cprint("   Insufficient margin for any positions.", "yellow")
                return []

            margin_per_position = (usable_margin - cash_buffer) / len(new_signals)

        actions = []
        for sig in new_signals:
            action_type = "OPEN_LONG" if sig["action"] == "BUY" else "OPEN_SHORT"
            actions.append({
                "symbol": sig["symbol"],
                "action": action_type,
                "margin_usd": round(margin_per_position, 2),
                "reason": f"Fallback: {sig['action']} signal ({sig['confidence']}% confidence)"
            })

        return actions

    def _get_account_balance(self):
        """Fetch current withdrawable USDC balance from HyperLiquid."""
        try:
            if EXCHANGE == "HYPERLIQUID" and self.account:
                return n.get_balance(self.account)
        except Exception as e:
            cprint(f"⚠️ Could not read account balance: {e}", "yellow")
        return 0.0

    def execute_allocations(self, actions_list):
        """
        Execute the AI-generated allocation plan.

        Args:
            actions_list: List of action dicts from allocate_portfolio()
                Each action has: symbol, action, margin_usd/reduce_by_usd, reason

        Action types:
            - OPEN_LONG: Open new long position
            - OPEN_SHORT: Open new short position
            - INCREASE: Add to existing position
            - REDUCE: Reduce existing position
            - CLOSE: Close position entirely
        """
        if not actions_list:
            cprint("📊 No actions to execute.", "cyan")
            return

        try:
            cprint("\n" + "=" * 60, "yellow")
            cprint("🚀 EXECUTING AI ALLOCATION PLAN", "white", "on_yellow", attrs=["bold"])
            cprint("=" * 60, "yellow")
            add_console_log(f"Executing {len(actions_list)} allocation actions", "info")

            # Sort actions: CLOSE first, then REDUCE, then OPEN/INCREASE
            # This ensures we free up capital before opening new positions
            action_priority = {"CLOSE": 0, "REDUCE": 1, "OPEN_LONG": 2, "OPEN_SHORT": 2, "INCREASE": 3}
            sorted_actions = sorted(actions_list, key=lambda x: action_priority.get(x.get("action", ""), 5))

            executed_count = 0
            failed_count = 0

            for action in sorted_actions:
                symbol = action.get("symbol")
                action_type = action.get("action")
                reason = action.get("reason", "")

                if not symbol or not action_type:
                    continue

                if symbol not in self.symbols:
                    cprint(f"⚠️ Skipping {symbol} - not in configured symbols", "yellow")
                    continue

                cprint(f"\n{'─' * 50}", "cyan")
                cprint(f"🎯 {symbol}: {action_type}", "cyan", attrs=["bold"])
                if reason:
                    cprint(f"   📝 {reason}", "white")

                try:
                    # Get current position state
                    if EXCHANGE == "HYPERLIQUID":
                        pos_data = n.get_position(symbol, self.account)
                    else:
                        pos_data = n.get_position(symbol)

                    _, im_in_pos, pos_size, _, entry_px, pnl_pct, is_long = pos_data
                    current_notional = abs(float(pos_size) * float(entry_px)) if im_in_pos else 0
                    current_dir = "LONG" if is_long else "SHORT"

                    # ============================================================
                    # CLOSE: Close entire position
                    # ============================================================
                    if action_type == "CLOSE":
                        if not im_in_pos or pos_size == 0:
                            cprint(f"   ℹ️ No position to close", "cyan")
                            continue

                        cprint(f"   📊 Closing {current_dir} position (${current_notional:.2f} notional)", "yellow")

                        close_success = False
                        if EXCHANGE == "HYPERLIQUID":
                            close_success = n.close_complete_position(symbol, self.account)
                        else:
                            n.chunk_kill(symbol, max_usd_order_size, slippage)
                            close_success = True  # chunk_kill doesn't return status

                        if close_success:
                            if POSITION_TRACKER_AVAILABLE:
                                remove_position(symbol)

                            cprint(f"   ✅ Position closed!", "green")
                            add_console_log(f"✅ Closed {symbol} {current_dir}", "success")
                            executed_count += 1
                        else:
                            cprint(f"   ⚠️ Close may have failed for {symbol}", "yellow")
                            add_console_log(f"⚠️ Close failed for {symbol}", "warning")

                    # ============================================================
                    # REDUCE: Reduce position size
                    # ============================================================
                    elif action_type == "REDUCE":
                        reduce_amount = action.get("reduce_by_usd", 0)

                        if not im_in_pos or pos_size == 0:
                            cprint(f"   ℹ️ No position to reduce", "cyan")
                            continue

                        if reduce_amount <= 0:
                            cprint(f"   ⚠️ Invalid reduce amount", "yellow")
                            continue

                        cprint(f"   📊 Current: ${current_notional:.2f} notional", "white")
                        cprint(f"   ➖ Reducing by: ${reduce_amount:.2f} notional", "yellow")

                        if hasattr(n, 'partial_close'):
                            n.partial_close(symbol, reduce_amount, account=self.account)
                            cprint(f"   ✅ Position reduced!", "green")
                            add_console_log(f"✅ Reduced {symbol} by ${reduce_amount:.2f}", "success")
                            executed_count += 1
                        else:
                            cprint(f"   ⚠️ partial_close not available", "yellow")

                    # ============================================================
                    # OPEN_LONG / INCREASE (for LONG)
                    # ============================================================
                    elif action_type in ["OPEN_LONG", "INCREASE"] and (action_type == "OPEN_LONG" or (im_in_pos and is_long)):
                        margin_usd = action.get("margin_usd", 0)
                        if margin_usd <= 0:
                            cprint(f"   ⚠️ Invalid margin amount", "yellow")
                            continue

                        notional = margin_usd * LEVERAGE

                        # Balance-aware sizing: cap at max position size
                        balance = self._get_account_balance()
                        if balance > 0:
                            max_notional = get_position_size_usd(balance)
                            if notional > max_notional:
                                cprint(f"   🛡️ Capping {symbol} notional: ${notional:.2f} → ${max_notional:.2f} (92% of ${balance:.2f} balance)", "yellow")
                                notional = max_notional
                                margin_usd = notional / LEVERAGE

                        # CRITICAL FIX: Handle position conflicts more efficiently
                        if im_in_pos and not is_long:
                            # Opposite position exists - allocate in opposite direction instead of closing first
                            cprint(f"   🔄 Opposite position detected - allocating in opposite direction", "cyan")

                            # For HyperLiquid, we can directly open opposite position which will net against existing
                            if EXCHANGE == "HYPERLIQUID":
                                cprint(f"   📈 Opening LONG to net against existing SHORT", "cyan")
                                if DRY_RUN:
                                    cprint(f"   🧪 DRY_RUN: would open LONG {symbol} ${notional:.2f}", "magenta")
                                    result = {"status": "dry_run", "symbol": symbol, "notional": notional}
                                else:
                                    result = n.ai_entry(symbol, notional, leverage=LEVERAGE, account=self.account)

                                if result:
                                    cprint(f"   ✅ LONG position opened (netting against SHORT)", "green")
                                    add_console_log(f"✅ Opened LONG {symbol} ${notional:.2f} (netting)", "success")

                                    # Update tracker to reflect net position
                                    if POSITION_TRACKER_AVAILABLE:
                                        try:
                                            record_position_entry(symbol=symbol, entry_price=0, size=notional, is_long=True)
                                        except Exception as e:
                                            cprint(f"   ⚠️ Position tracker error: {e}", "yellow")

                                    try:
                                        log_position_open(symbol, "LONG", notional)
                                    except Exception as e:
                                        cprint(f"   ⚠️ Position log error: {e}", "yellow")

                                    executed_count += 1
                                    continue  # Skip the rest of the logic for this action
                                else:
                                    cprint(f"   ❌ Failed to open LONG position", "red")
                                    add_console_log(f"❌ Failed to open LONG {symbol} ${notional:.2f}", "error")
                                    failed_count += 1
                                    continue
                            else:
                                # For other exchanges, fall back to closing first
                                cprint(f"   ⚠️ Closing SHORT before opening LONG...", "yellow")
                                close_ok = False
                                if EXCHANGE == "HYPERLIQUID":
                                    close_ok = n.close_complete_position(symbol, self.account)
                                else:
                                    n.chunk_kill(symbol, max_usd_order_size, slippage)
                                    close_ok = True
                                if not close_ok:
                                    cprint(f"   ⚠️ Failed to close SHORT, skipping LONG entry", "yellow")
                                    continue
                                time.sleep(1)

                        cprint(f"   📈 Opening LONG: ${notional:.2f} notional (${margin_usd:.2f} margin)", "green")

                        # Execute trade and verify success
                        result = None
                        if DRY_RUN:
                            cprint(f"   🧪 DRY_RUN: would open LONG {symbol} ${notional:.2f}", "magenta")
                            result = {"status": "dry_run", "symbol": symbol, "notional": notional}
                        elif EXCHANGE == "HYPERLIQUID":
                            result = n.ai_entry(symbol, notional, leverage=LEVERAGE, account=self.account)
                        elif EXCHANGE == "ASTER":
                            result = n.ai_entry(symbol, notional, leverage=LEVERAGE)
                        else:
                            result = n.ai_entry(symbol, notional)

                        # Verify trade executed successfully
                        if result:
                            cprint(f"   ✅ LONG position opened!", "green")
                            add_console_log(f"✅ Opened LONG {symbol} ${notional:.2f}", "success")

                            # Record in tracker
                            if POSITION_TRACKER_AVAILABLE:
                                try:
                                    record_position_entry(symbol=symbol, entry_price=0, size=notional, is_long=True)
                                except Exception as e:
                                    cprint(f"   ⚠️ Position tracker error: {e}", "yellow")

                            try:
                                log_position_open(symbol, "LONG", notional)
                            except Exception as e:
                                cprint(f"   ⚠️ Position log error: {e}", "yellow")

                            if TRADE_LOGGER_AVAILABLE:
                                try:
                                    log_trade(
                                        symbol=symbol,
                                        action="OPEN_LONG",
                                        side="LONG",
                                        notional_usd=notional,
                                        margin_usd=margin_usd,
                                        leverage=LEVERAGE,
                                        reason=reason,
                                        strategy=self.active_strategy
                                    )
                                except Exception as e:
                                    cprint(f"   ⚠️ Trade logger error: {e}", "yellow")

                            executed_count += 1
                        else:
                            cprint(f"   ❌ LONG position failed to open (no result returned)", "red")
                            add_console_log(f"❌ {symbol} LONG failed (no result)", "error")
                            failed_count += 1

                    # ============================================================
                    # OPEN_SHORT / INCREASE (for SHORT)
                    # ============================================================
                    elif action_type in ["OPEN_SHORT"] or (action_type == "INCREASE" and im_in_pos and not is_long):
                        margin_usd = action.get("margin_usd", 0)
                        if margin_usd <= 0:
                            cprint(f"   ⚠️ Invalid margin amount", "yellow")
                            continue

                        notional = margin_usd * LEVERAGE

                        # Balance-aware sizing: cap at max position size
                        balance = self._get_account_balance()
                        if balance > 0:
                            max_notional = get_position_size_usd(balance)
                            if notional > max_notional:
                                cprint(f"   🛡️ Capping {symbol} notional: ${notional:.2f} → ${max_notional:.2f} (92% of ${balance:.2f} balance)", "yellow")
                                notional = max_notional
                                margin_usd = notional / LEVERAGE

                        # CRITICAL FIX: Handle position conflicts more efficiently
                        if im_in_pos and is_long:
                            # Opposite position exists - allocate in opposite direction instead of closing first
                            cprint(f"   🔄 Opposite position detected - allocating in opposite direction", "cyan")

                            # For HyperLiquid, we can directly open opposite position which will net against existing
                            if EXCHANGE == "HYPERLIQUID":
                                cprint(f"   📉 Opening SHORT to net against existing LONG", "cyan")
                                if DRY_RUN:
                                    cprint(f"   🧪 DRY_RUN: would open SHORT {symbol} ${notional:.2f}", "magenta")
                                    result = {"status": "dry_run", "symbol": symbol, "notional": notional}
                                else:
                                    result = n.open_short(symbol, notional, leverage=LEVERAGE, account=self.account)

                                if result:
                                    cprint(f"   ✅ SHORT position opened (netting against LONG)", "green")
                                    add_console_log(f"✅ Opened SHORT {symbol} ${notional:.2f} (netting)", "success")

                                    # Update tracker to reflect net position
                                    if POSITION_TRACKER_AVAILABLE:
                                        try:
                                            record_position_entry(symbol=symbol, entry_price=0, size=notional, is_long=False)
                                        except Exception as e:
                                            cprint(f"   ⚠️ Position tracker error: {e}", "yellow")

                                    try:
                                        log_position_open(symbol, "SHORT", notional)
                                    except Exception as e:
                                        cprint(f"   ⚠️ Position log error: {e}", "yellow")

                                    executed_count += 1
                                    continue  # Skip the rest of the logic for this action
                                else:
                                    cprint(f"   ❌ Failed to open SHORT position", "red")
                                    add_console_log(f"❌ Failed to open SHORT {symbol} ${notional:.2f}", "error")
                                    failed_count += 1
                                    continue
                            else:
                                # For other exchanges, fall back to closing first
                                cprint(f"   ⚠️ Closing LONG before opening SHORT...", "yellow")
                                close_ok = False
                                if EXCHANGE == "HYPERLIQUID":
                                    close_ok = n.close_complete_position(symbol, self.account)
                                else:
                                    n.chunk_kill(symbol, max_usd_order_size, slippage)
                                    close_ok = True
                                if not close_ok:
                                    cprint(f"   ⚠️ Failed to close LONG, skipping SHORT entry", "yellow")
                                    continue
                                time.sleep(1)

                        if EXCHANGE == "SOLANA":
                            cprint(f"   ⚠️ SHORT not supported on SOLANA", "yellow")
                            continue

                        cprint(f"   📉 Opening SHORT: ${notional:.2f} notional (${margin_usd:.2f} margin)", "red")

                        # Execute trade and verify success
                        result = None
                        if DRY_RUN:
                            cprint(f"   🧪 DRY_RUN: would open SHORT {symbol} ${notional:.2f}", "magenta")
                            result = {"status": "dry_run", "symbol": symbol, "notional": notional}
                        elif EXCHANGE == "HYPERLIQUID":
                            result = n.open_short(symbol, notional, leverage=LEVERAGE, account=self.account)
                        elif EXCHANGE == "ASTER":
                            if hasattr(n, 'open_short'):
                                result = n.open_short(symbol, notional, leverage=LEVERAGE)
                            else:
                                cprint(f"   ⚠️ open_short not available for ASTER", "yellow")
                                failed_count += 1
                                continue

                        # Verify trade executed successfully
                        if result:
                            cprint(f"   ✅ SHORT position opened!", "green")
                            add_console_log(f"✅ Opened SHORT {symbol} ${notional:.2f}", "success")

                            # Record in tracker
                            if POSITION_TRACKER_AVAILABLE:
                                try:
                                    record_position_entry(symbol=symbol, entry_price=0, size=notional, is_long=False)
                                except Exception as e:
                                    cprint(f"   ⚠️ Position tracker error: {e}", "yellow")

                            try:
                                log_position_open(symbol, "SHORT", notional)
                            except Exception as e:
                                cprint(f"   ⚠️ Position log error: {e}", "yellow")

                            if TRADE_LOGGER_AVAILABLE:
                                try:
                                    log_trade(
                                        symbol=symbol,
                                        action="OPEN_SHORT",
                                        side="SHORT",
                                        notional_usd=notional,
                                        margin_usd=margin_usd,
                                        leverage=LEVERAGE,
                                        reason=reason,
                                        strategy=self.active_strategy
                                    )
                                except Exception as e:
                                    cprint(f"   ⚠️ Trade logger error: {e}", "yellow")

                            executed_count += 1
                        else:
                            cprint(f"   ❌ SHORT position failed to open (no result returned)", "red")
                            add_console_log(f"❌ {symbol} SHORT failed (no result)", "error")
                            failed_count += 1

                    else:
                        cprint(f"   ⚠️ Unknown action type: {action_type}", "yellow")

                except Exception as e:
                    cprint(f"   ❌ Error: {str(e)}", "red")
                    add_console_log(f"❌ {symbol} {action_type} failed: {e}", "error")
                    failed_count += 1
                    import traceback
                    traceback.print_exc()

                time.sleep(2)  # Rate limiting between trades

            # Summary
            cprint(f"\n{'=' * 60}", "green")
            cprint(f"✅ EXECUTION COMPLETE: {executed_count} succeeded, {failed_count} failed", "green", attrs=["bold"])
            cprint(f"{'=' * 60}\n", "green")
            add_console_log(f"Execution complete: {executed_count} succeeded, {failed_count} failed", "success")

        except Exception as e:
            cprint(f"❌ Error in execute_allocations: {e}", "red")
            import traceback
            traceback.print_exc()

    def handle_exits(self):
        """
        PHASE 1: CLOSE positions when signal is OPPOSITE to position direction.

        This function ONLY handles EXITS - it does NOT open new positions.
        New positions are opened in execute_allocations() after balance is recalculated.

        Logic:
        - BUY signal + LONG position → KEEP (signal confirms position)
        - BUY signal + SHORT position → CLOSE (signal contradicts position)
        - SELL signal + LONG position → CLOSE (signal contradicts position)
        - SELL signal + SHORT position → KEEP (signal confirms position)
        - NOTHING signal → KEEP any position

        Order of Operations (per dev_tasks.md 3.1):
        1. CLOSE existing positions (this function)
        2. RE-EVALUATE allocation (allocate_portfolio with fresh balance)
        3. OPEN new positions (execute_allocations)
        """
        cprint("\n🔄 PHASE 1: Checking for positions to exit...", "white", "on_blue")
        add_console_log("Evaluating positions...", "info")

        positions_closed = 0
        positions_held = 0

        for _, row in self.recommendations_df.iterrows():
            token = row["token"]
            token_short = token[:8] + "..." if len(token) > 8 else token

            if token in EXCLUDED_TOKENS:
                continue

            action = row["action"]

            # Get position with direction info
            try:
                if EXCHANGE == "HYPERLIQUID":
                    pos_data = n.get_position(token, self.account)
                else:
                    pos_data = n.get_position(token)

                _, im_in_pos, pos_size, _, entry_px, pnl_perc, is_long = pos_data

                # If position was manually closed, clean up tracker
                if not im_in_pos and POSITION_TRACKER_AVAILABLE:
                    try:
                        remove_position(token)
                    except Exception:
                        pass

            except Exception as e:
                cprint(f"⚠️ Error getting position for {token}: {e}", "yellow")
                # Try to clean up tracker on error
                if POSITION_TRACKER_AVAILABLE:
                    try:
                        remove_position(token)
                    except Exception:
                        pass
                continue

            cprint(f"\n{'=' * 60}", "cyan")
            cprint(f"🎯 Token: {token_short}", "cyan", attrs=["bold"])
            cprint(f"📊 Signal: {action} ({row['confidence']}% confidence)", "yellow", attrs=["bold"])

            if im_in_pos and pos_size != 0:
                # ============= CASE: HAVE POSITION =============
                position_dir = "LONG" if is_long else "SHORT"
                cprint(f"💼 Current Position: {position_dir} | Size: {abs(pos_size):.4f} | PnL: {pnl_perc:.2f}%", "white")
                cprint(f"{'=' * 60}", "cyan")

                # CRITICAL: Check for stop loss FIRST (overrides all other logic)
                if pnl_perc <= STOP_LOSS_THRESHOLD:
                    cprint(f"🚨 STOP LOSS TRIGGERED: {pnl_perc:.2f}% <= {STOP_LOSS_THRESHOLD}%", "white", "on_red", attrs=["bold"])
                    cprint(f"⚠️ FORCE CLOSING {position_dir} position (mandatory -2% stop loss)", "white", "on_red")

                    try:
                        close_ok = False
                        if EXCHANGE == "HYPERLIQUID":
                            close_ok = n.close_complete_position(token, self.account)
                        else:
                            n.chunk_kill(token, max_usd_order_size, slippage)
                            close_ok = True

                        if close_ok:
                            # Remove from position tracker
                            if POSITION_TRACKER_AVAILABLE:
                                remove_position(token)

                            cprint("✅ Stop loss position closed successfully!", "white", "on_green")
                            add_console_log(f"STOP LOSS: Closed {token} {position_dir} at {pnl_perc:.2f}%", "warning")
                            positions_closed += 1
                        else:
                            cprint("⚠️ Stop loss close may have failed - will retry next cycle", "white", "on_yellow")
                            add_console_log(f"⚠️ Stop loss close failed for {token}", "warning")

                    except Exception as e:
                        cprint(f"❌ Error closing stop loss position: {str(e)}", "white", "on_red")
                        add_console_log(f"❌ Failed to close stop loss position {token}: {e}", "error")

                    continue  # Skip to next token after stop loss

                # Determine if signal contradicts position direction
                signal_contradicts_position = (
                    (action == "SELL" and is_long) or      # SELL signal vs LONG position
                    (action == "BUY" and not is_long)      # BUY signal vs SHORT position
                )

                if action == "NOTHING":
                    # NOTHING = hold current position regardless of direction
                    cprint("⏸️ DO NOTHING signal - HOLDING POSITION", "white", "on_blue")
                    cprint(f"💎 Maintaining {position_dir} position", "cyan")
                    positions_held += 1

                elif signal_contradicts_position:
                    # Signal contradicts position → CLOSE (ONLY CLOSE, NO OPEN LOGIC HERE)
                    if action == "SELL" and is_long:
                        cprint("🚨 SELL signal vs LONG position - CLOSING", "white", "on_red")
                    else:  # BUY signal vs SHORT position
                        cprint("🚨 BUY signal vs SHORT position - CLOSING", "white", "on_red")

                    try:
                        close_ok = False
                        if EXCHANGE == "HYPERLIQUID":
                            close_ok = n.close_complete_position(token, self.account)
                        else:
                            n.chunk_kill(token, max_usd_order_size, slippage)
                            close_ok = True

                        if close_ok:
                            # Remove from position tracker
                            if POSITION_TRACKER_AVAILABLE:
                                remove_position(token)

                            cprint("✅ Position closed successfully!", "white", "on_green")
                            add_console_log(f"Closed {token} {position_dir} - signal contradicted direction", "warning")
                            positions_closed += 1
                        else:
                            cprint("⚠️ Position close may have failed - will retry next cycle", "yellow")
                            add_console_log(f"⚠️ Close failed for {token}", "warning")

                    except Exception as e:
                        cprint(f"❌ Error closing position: {str(e)}", "white", "on_red")

                else:
                    # Signal confirms position direction → KEEP
                    if action == "BUY" and is_long:
                        cprint("✅ BUY signal confirms LONG position - KEEPING", "white", "on_green")
                    else:  # SELL signal confirms SHORT position
                        cprint("✅ SELL signal confirms SHORT position - KEEPING", "white", "on_green")

                    cprint(f"💎 Maintaining {position_dir} position", "cyan")
                    positions_held += 1

            else:
                # ============= CASE: NO POSITION =============
                cprint(f"💼 No position", "white")
                cprint(f"{'=' * 60}", "cyan")

                # Do NOT open new positions here - that happens in execute_allocations()
                if action == "SELL":
                    if LONG_ONLY:
                        cprint("⭐ SELL signal - LONG ONLY mode, can't open SHORT", "white", "on_blue")
                    else:
                        cprint("📉 SELL signal - SHORT will be opened in allocation phase", "white", "on_yellow")

                elif action == "NOTHING":
                    cprint("⏸️ DO NOTHING signal - staying flat", "white", "on_blue")

                else:  # BUY
                    cprint("📈 BUY signal - LONG will be opened in allocation phase", "white", "on_green")

        # Summary
        cprint(f"\n{'=' * 60}", "green")
        cprint(f"✅ PHASE 1 COMPLETE: Closed {positions_closed}, Held {positions_held} positions", "green", attrs=["bold"])
        cprint(f"{'=' * 60}", "green")
        add_console_log(f"Closed {positions_closed}, Held {positions_held}", "success")

    def show_final_portfolio_report(self):
        """Display final portfolio status - NO LOOPS, just a snapshot"""
        cprint("\n" + "=" * 60, "cyan")
        cprint("📊 FINAL PORTFOLIO REPORT", "white", "on_blue", attrs=["bold"])
        cprint("=" * 60, "cyan")

        # CRITICAL: Use self.symbols (instance variable) NOT global SYMBOLS/MONITORED_TOKENS
        check_tokens = self.symbols
        active_positions = []

        # Print header
        print(f"   {'TOKEN':<10} | {'SIDE':<10} | {'SIZE':<12} | {'ENTRY':<12} | {'PNL %':<10}")
        print("   " + "-" * 65)

        for token in check_tokens:
            try:
                pos_data = n.get_position(token, self.account)
                _, im_in_pos, pos_size, _, entry_px, pnl_perc, is_long = pos_data

                if im_in_pos and pos_size != 0:
                    side_icon = "LONG" if is_long else "SHORT"
                    entry_str = f"${entry_px:.2f}" if entry_px != 0 else "-"
                    pnl_str = f"{pnl_perc:+.2f}%" if pnl_perc != 0 else "-"

                    print(
                        f"   {token:<10} | {side_icon:<10} | {pos_size:<12.4f} | "
                        f"{entry_str:<12} | {pnl_str:<10}"
                    )
                    active_positions.append(token)

            except Exception:
                pass  # Silently skip errors to keep report clean

        if not active_positions:
            cprint("   (No active positions)", "cyan")

        cprint("=" * 60 + "\n", "cyan")

    def should_stop(self):
        """Enhanced stop signal checking - only check external stop signals"""
        if self.stop_check_callback is not None:
            if self.stop_check_callback():
                return True
        
        # DO NOT check TP/SL here - that's handled within the trading cycle logic
        # TP/SL should trigger position closes, not stop the entire trading cycle
        return False

    def _check_immediate_tp_sl_actions(self):
        """Check if any positions need immediate TP/SL execution"""
        try:
            for symbol in self.symbols:
                if symbol in EXCLUDED_TOKENS:
                    continue
                    
                pos_data = n.get_position(symbol, self.account)
                _, im_in_pos, _, _, _, pnl_perc, _ = pos_data
                
                if im_in_pos and pnl_perc != 0:
                    # Check TP threshold
                    if pnl_perc >= TAKE_PROFIT_THRESHOLD:
                        cprint(f"🚨 TAKE PROFIT needed for {symbol}: {pnl_perc:.2f}%", "red")
                        return True
                    
                    # Check SL threshold  
                    if pnl_perc <= STOP_LOSS_THRESHOLD:
                        cprint(f"🚨 STOP LOSS needed for {symbol}: {pnl_perc:.2f}%", "red")
                        return True
        except Exception as e:
            cprint(f"⚠️ Error checking TP/SL: {e}", "yellow")
        
        return False

    def run(self):
        """Run the trading agent (implements BaseAgent interface)"""
        self.run_trading_cycle()

    def run_trading_cycle(self, strategy_signals=None):
        """Enhanced trading cycle with position management and intelligence integration"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprint(f"\n{'=' * 80}", "cyan")
            cprint(f"🔄 TRADING CYCLE START: {current_time}", "white", "on_green", attrs=["bold"])
            cprint(f"{'=' * 80}", "cyan")

            add_console_log(f"TRADING CYCLE STARTED", "info")

            # CRITICAL FIX: Reset recommendations_df at the start of each cycle
            self.recommendations_df = pd.DataFrame(
                columns=["token", "action", "confidence", "reasoning"]
            )
            cprint("📋 Recommendations cleared for fresh cycle", "cyan")

            # Check for stop signal
            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 0: DISPLAY VOLUME INTELLIGENCE SUMMARY (if available)
            if INTELLIGENCE_AVAILABLE:
                volume_summary = get_volume_summary()
                if volume_summary and "No volume" not in volume_summary:
                    cprint("\n📊 VOLUME INTELLIGENCE:", "white", "on_blue")
                    cprint(volume_summary, "cyan")
                    add_console_log("📊 Volume intelligence loaded", "info")

            # STEP 1: FETCH ALL OPEN POSITIONS
            add_console_log("Fetching open positions...", "info")
            open_positions = self.fetch_all_open_positions()
            add_console_log(f"Found {len(open_positions)} open position(s)", "info")

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 2: COLLECT MARKET DATA
            tokens_to_trade = self.symbols
            add_console_log(f"📊 Collecting market data for {len(tokens_to_trade)} tokens...", "info")
            cprint("📊 Collecting market data for analysis...", "white", "on_blue")

            market_data = collect_all_tokens(
                tokens=tokens_to_trade,
                days_back=self.days_back,
                timeframe=self.timeframe,
                exchange=EXCHANGE,
            )
            add_console_log(f"Market data collected for {len(market_data)} tokens", "info")

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 3: AI ANALYZES OPEN POSITIONS
            close_decisions = {}
            if open_positions:
                close_decisions = self.analyze_open_positions_with_ai(open_positions, market_data)
                if self.should_stop():
                    add_console_log("ℹ️ Stop signal received - skipping position closes", "warning")
                    return
                self.execute_position_closes(close_decisions)

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 4: REFETCH POSITIONS & MARKET DATA AFTER CLOSURES
            time.sleep(2)
            open_positions = self.fetch_all_open_positions()
            cprint("📊 Refreshing market data after position updates...", "white", "on_blue")
            market_data = collect_all_tokens(
                tokens=tokens_to_trade,
                days_back=self.days_back,
                timeframe=self.timeframe,
                exchange=EXCHANGE,
            )

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 5: ANALYZE TOKENS FOR NEW ENTRIES
            cprint("\n📈 Analyzing tokens for new entry opportunities...", "white", "on_blue")

            # Engine v6.1 path: generate signals directly, skip AI entry analysis
            if self.active_strategy == 'engine_v6_1' and self.strategy_engine:
                cprint("🚀 Engine v6.1 is active - generating technical entry signals", "magenta", attrs=["bold"])
                engine_signals = self._compute_engine_signals(market_data)
                for token, sig in engine_signals.items():
                    if sig["direction"] in ("BUY", "SELL"):
                        self.recommendations_df = pd.concat(
                            [
                                self.recommendations_df,
                                pd.DataFrame(
                                    [
                                        {
                                            "token": token,
                                            "action": sig["direction"],
                                            "confidence": int(sig["signal"] * 100),
                                            "reasoning": f"Engine v6.1 | ADX={sig['metadata']['adx']} | fast={sig['metadata']['fast_ema']} medm={sig['metadata']['medm_ema']} slow={sig['metadata']['slow_sma']}",
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                        add_console_log(f"Engine v6.1 {token} -> {sig['direction']} | {int(sig['signal'] * 100)}%", "success")
            else:
                for token, data in market_data.items():
                    if self.should_stop():
                        add_console_log(f"ℹ️ Stop signal received - stopping analysis at {token}", "warning")
                        return

                    cprint(f"\n📊 Analyzing {token}...", "white", "on_green")
                    add_console_log(f"📊 Analyzing {token}...", "info")

                    if strategy_signals and token in strategy_signals:
                        data["strategy_signals"] = strategy_signals[token]

                    analysis = self.analyze_market_data(token, data)
                    if analysis:
                        print(f"\n📈 Analysis for {token}:")
                        print(analysis)
                        print("\n" + "=" * 50 + "\n")

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - aborting cycle", "warning")
                return

            # STEP 6: SHOW RECOMMENDATIONS
            cprint("\n📊 AI TRADING RECOMMENDATIONS:", "white", "on_blue")
            summary_df = self.recommendations_df[["token", "action", "confidence"]].copy()
            print(summary_df.to_string(index=False))

            if self.should_stop():
                add_console_log("ℹ️ Stop signal received - skipping trade execution", "warning")
                return

            # ================================================================
            # 🚀 UNIFIED EXECUTION - AI-DRIVEN ALLOCATION
            # Works the same for both swarm and single mode
            # ================================================================
            try:
                mode_name = "SWARM" if self.use_swarm_mode else "SINGLE"
                cprint(f"\n{'=' * 80}", "yellow")
                cprint(f"🚀 {mode_name} MODE — AI-Driven Allocation Pipeline", "white", "on_yellow", attrs=["bold"])
                cprint(f"{'=' * 80}", "yellow")
                add_console_log(f"{mode_name} mode — starting allocations", "info")

                # Phase 1: Close contradictory positions (signals vs positions)
                cprint("\n📌 PHASE 1: Exit Contradictory Positions", "yellow", attrs=["bold"])
                self.handle_exits()

                if self.should_stop():
                    add_console_log("ℹ️ Stop signal received - skipping allocation", "warning")
                    return

                # Wait for exchange to process closes
                cprint("⏳ Waiting for exchange to process...", "cyan")
                time.sleep(3)

                # Phase 2: AI-Driven Smart Allocation
                cprint("\n📌 PHASE 2: AI Smart Allocation", "cyan", attrs=["bold"])
                allocation_actions = self.allocate_portfolio()

                if self.should_stop():
                    add_console_log("ℹ️ Stop signal received - skipping execution", "warning")
                    return

                # Phase 3: Execute the AI allocation plan
                if allocation_actions and isinstance(allocation_actions, list) and len(allocation_actions) > 0:
                    cprint(f"\n📌 PHASE 3: Execute {len(allocation_actions)} Actions", "green", attrs=["bold"])
                    self.execute_allocations(allocation_actions)
                    cprint(f"\n✅ {mode_name} mode execution complete!", "green", attrs=["bold"])
                    add_console_log(f"✅ {mode_name} execution complete", "success")
                else:
                    cprint("\nℹ️ No allocation actions to execute.", "yellow")
                    add_console_log("ℹ️ No allocation actions generated", "info")

            except Exception as exec_err:
                cprint(f"❌ Execution pipeline failed: {exec_err}", "red")
                import traceback
                traceback.print_exc()
                add_console_log(f"Execution error: {exec_err}", "error")

            # STEP 8: FINAL PORTFOLIO REPORT
            self.show_final_portfolio_report()

            try:
                if os.path.exists("temp_data"):
                    for file in os.listdir("temp_data"):
                        if file.endswith("_latest.csv"):
                            os.remove(os.path.join("temp_data", file))
            except Exception as e:
                cprint(f"⚠️ Error cleaning temp data: {e}", "yellow")

            cprint(f"\n{'=' * 80}", "cyan")
            cprint("✅ TRADING CYCLE COMPLETE", "white", "on_green", attrs=["bold"])
            add_console_log("✅ Trading cycle complete", "success")
            cprint(f"{'=' * 80}\n", "cyan")

            try:
                account_balance = get_account_balance(self.account)
            except Exception as e:
                cprint(f"⚠️ Could not retrieve account balance: {e}", "yellow")
                account_balance = 0.0

            try:
                invested_total = 0.0
                positions = self.fetch_all_open_positions()
                for symbol, pos_list in positions.items():
                    for p in pos_list:
                        size = abs(float(p.get("size", 0)))
                        entry_price = float(p.get("entry_price", 0))
                        invested_total += size * entry_price
            except Exception as e:
                cprint(f"⚠️ Could not calculate invested total: {e}", "yellow")
                invested_total = 0.0

            cprint(f"💰 Account Balance: ${account_balance:,.2f}", "cyan", attrs=["bold"])
            cprint(f"🚀 Invested Total: ${invested_total:,.2f}", "cyan", attrs=["bold"])

        except Exception as e:
            cprint(f"\n❌ Error in trading cycle: {e}", "white", "on_red")
            import traceback
            traceback.print_exc()


def main():
    """Main function - simple cycle every X minutes"""
    cprint("🚀 AI Trading System Starting Up! 🚀", "white", "on_blue")
    print("🛑 Press Ctrl+C to stop.\n")

    agent = TradingAgent()

    while True:
        try:
            # Run the complete cycle
            agent.run_trading_cycle()

            # Log next cycle time BEFORE sleeping
            next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
            cprint(f"\n⏰ Next cycle at UTC: {next_run.strftime('%d-%m-%Y %H:%M:%S')}", "white", "on_green")
            add_console_log(f"Next cycle in {SLEEP_BETWEEN_RUNS_MINUTES} minutes", "info")

            # Sleep until next cycle
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n👋 AI Agent shutting down gracefully...", "white", "on_blue")
            add_console_log("👋 Agent shutting down gracefully...", "info")
            break
        except Exception as e:
            cprint(f"\n❌ Error in main loop: {e}", "white", "on_red")
            import traceback
            traceback.print_exc()
            cprint(f"\n⏰ Retrying in {SLEEP_BETWEEN_RUNS_MINUTES} minutes...", "yellow")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)


if __name__ == "__main__":
    main()
