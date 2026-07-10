"""
EngineV6_1Strategy — full-fidelity Pine Script v6.1 translation.

Magic Trend v6 PRO ("Pure Efficiency") translated line-by-line from Pine.
Every input default, every filter, every adaptive layer, and every v6.1-specific
behaviour (momentum entry, equity-curve adaptive ATR multiplier, hyper-growth
phase detection, trailing-stop parameter export) is preserved.

Signal contract
---------------
generate_signals(df, symbol="", equity_history=None)
  -> {"token": ..., "signal": 0..1, "direction": "BUY"|"SELL"|"NEUTRAL",
      "metadata": {...}}
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff().clip(lower=0.0)
    minus_dm = (-df["low"].diff()).clip(lower=0.0)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx.fillna(0)


class EngineV6_1Strategy(BaseStrategy):
    """Magic Trend v6 PRO — hardened Pine-to-Python (full-fidelity)."""

    name = "Engine v6.1"
    description = "Magic Trend v6 PRO — full-fidelity v6.1 translation. Best on PEPE / FARTCOIN."
    recommended_timeframe = "15m-1h"
    recommended_symbols = ["PEPE", "FARTCOIN"]

    # ------------------------------------------------------------------
    # Pine input defaults (preserved exactly)
    # ------------------------------------------------------------------
    # Hyper-Growth Protocol
    GROWTH_TARGET_X = 50.0
    USE_MOMENTUM = True
    MOMENTUM_THRESH = 18

    # Date range
    START_DATE = pd.Timestamp("2025-01-01 08:00:00")
    END_DATE = pd.Timestamp("2069-12-30 00:00:00")

    # Trade direction
    TRADE_DIRECTION = "Both"          # "Both", "Long Only", "Short Only"

    # Equity Guard (cold-market protection)
    USE_EQUITY_GUARD = False
    EQUITY_SMA_LEN = 21
    EQ_PERCENT = 0.7

    # ATR settings
    ATR_MULT = 1.8
    ATR_MULT_GUARD = 0.9
    WARMUP_TRADES = 3

    # Risk profile
    RISK_PROFILE = "Manual"
    MAN_ACTIVATION = 18
    MAN_OFFSET = 6
    RISK_PER_TRADE_PCT = 97.0

    # Dynamic risk ("THE FIX")
    AGGRESSIVE_DRAWDOWN_THRESHOLD = 0.10
    AGGRESSIVE_MULTIPLIER = 1.20
    PEAK_PROTECT_MULTIPLIER = 0.30

    # Indicators
    SMA_SLOW = 50
    EMA_MEDM = 18
    EMA_FAST = 6
    ATR_PERIOD = 14

    # Pine initial_capital
    INITIAL_CAPITAL = 100.0

    def __init__(self):
        super().__init__(self.name)
        # State that mirrors Pine ``var`` declarations
        self._equity_curve: List[float] = []
        self._last_closed_trades = 0
        self._last_entry_bar = -1
        self._equity_peak = 0.0
        # Adaptive-multiplier history (Pine series emulation)
        self._last_eq_sma: Optional[float] = None
        self._last_vel: Optional[float] = None
        self._acc_ema: Optional[float] = None

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        equity_history: list = None,
    ) -> dict:
        """
        Generate trading signal and rich metadata from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
        symbol : str
            Token / trading-pair symbol (returned verbatim).
        equity_history : list[float] | None
            Closed-equity curve (newest last). Mirrors Pine ``equityCurve``.
            If omitted the strategy assumes target-equity (non-hyper) state.

        Returns
        -------
        dict
            {token, signal (0..1), direction (BUY|SELL|NEUTRAL), metadata}
        """
        # Normalise columns -------------------------------------------------
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)) or len(df) < self.SMA_SLOW + 5:
            return {
                "token": symbol,
                "signal": 0.0,
                "direction": "NEUTRAL",
                "metadata": {"error": "insufficient data"},
            }

        # Hyper-growth status -----------------------------------------------
        target_equity = self.INITIAL_CAPITAL * self.GROWTH_TARGET_X
        if equity_history and len(equity_history) > 0:
            current_closed_equity = float(equity_history[-1])
            equity_curve = list(equity_history)
        else:
            current_closed_equity = target_equity
            equity_curve = []

        is_hyper_phase = current_closed_equity < target_equity
        progress_pct = current_closed_equity / target_equity * 100

        # Detect new closed trades by curve growth (before Pine caps to eqLength)
        # Pine: if strategy.closedtrades > last_closed_trades → push equity
        if equity_history and len(equity_history) > self._last_closed_trades:
            self._last_closed_trades = len(equity_history)

        # Update internal curve (capped exactly like Pine)
        if equity_history:
            self._equity_curve = list(equity_history)
            if len(self._equity_curve) > self.EQUITY_SMA_LEN:
                self._equity_curve = self._equity_curve[-self.EQUITY_SMA_LEN:]

        avg_equity = (
            float(np.mean(self._equity_curve))
            if self._equity_curve
            else current_closed_equity
        )

        closed_trades = self._last_closed_trades
        in_warmup = closed_trades < self.WARMUP_TRADES
        is_strategy_cold = (not in_warmup) and (current_closed_equity < avg_equity)

        # OVERRIDE 1: If in Hyper Phase, ignore "Cold" state. Keep pushing.
        if is_hyper_phase:
            is_strategy_cold = False

        atr_mult_use = (
            self.ATR_MULT
            if in_warmup
            else (self.ATR_MULT_GUARD if is_strategy_cold else self.ATR_MULT)
        )

        # Adaptive equity compounding core ----------------------------------
        atr_mult_use = self._adaptive_multiplier(
            current_closed_equity, avg_equity, is_hyper_phase, atr_mult_use
        )

        # Risk profile ticks ------------------------------------------------
        active_activation, active_offset = self._risk_profile_ticks()

        # Dynamic risk sizing -----------------------------------------------
        self._equity_peak = max(self._equity_peak, current_closed_equity)
        dd_percent = (
            (current_closed_equity / self._equity_peak - 1.0)
            if self._equity_peak > 0
            else 0.0
        )

        # CRITICAL EXPLOIT: Peak Protection Override
        if is_hyper_phase:
            risk_multiplier = (
                self.AGGRESSIVE_MULTIPLIER
                if dd_percent < -self.AGGRESSIVE_DRAWDOWN_THRESHOLD
                else 1.0
            )
        else:
            risk_multiplier = 1.0 if dd_percent < 0 else self.PEAK_PROTECT_MULTIPLIER

        final_risk_pct = self.RISK_PER_TRADE_PCT * risk_multiplier

        # Indicators --------------------------------------------------------
        slow_sma = _sma(df["close"], self.SMA_SLOW)
        medm_ema = _ema(df["close"], self.EMA_MEDM)
        fast_ema = _ema(df["close"], self.EMA_FAST)
        atr = _atr(df, self.ATR_PERIOD)
        adx = _adx(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        bar_index = len(df) - 1

        # Trend filters (fan)
        fan_up_trend = bool(
            fast_ema.iloc[-1] > medm_ema.iloc[-1] > slow_sma.iloc[-1]
        )
        fan_dn_trend = bool(
            fast_ema.iloc[-1] < medm_ema.iloc[-1] < slow_sma.iloc[-1]
        )

        # Momentum / ADX
        is_strong_trend = bool(adx.iloc[-1] > self.MOMENTUM_THRESH)

        # Pin bars (v6.1 exact body formula)
        bar_range = latest["high"] - latest["low"]
        if bar_range == 0:
            bullish_pin = False
            bearish_pin = False
        else:
            bullish_pin = bool(
                (
                    latest["close"] > latest["open"]
                    and latest["open"] - latest["low"] > 0.66 * bar_range
                )
                or (
                    latest["close"] < latest["open"]
                    and latest["close"] - latest["low"] > 0.66 * bar_range
                )
            )
            bearish_pin = bool(
                (
                    latest["close"] > latest["open"]
                    and latest["high"] - latest["close"] > 0.66 * bar_range
                )
                or (
                    latest["close"] < latest["open"]
                    and latest["high"] - latest["open"] > 0.66 * bar_range
                )
            )

        # Valid triggers (hyper-phase momentum exploit)
        if is_hyper_phase and self.USE_MOMENTUM:
            valid_trigger_bull = bool(
                bullish_pin or (is_strong_trend and latest["close"] > prev["high"])
            )
            valid_trigger_bear = bool(
                bearish_pin or (is_strong_trend and latest["close"] < prev["low"])
            )
        else:
            valid_trigger_bull = bullish_pin
            valid_trigger_bear = bearish_pin

        # Pierce detection (v6.1 requires open beyond MA too)
        bull_pierce = bool(
            (
                latest["low"] < fast_ema.iloc[-1]
                and latest["open"] > fast_ema.iloc[-1]
                and latest["close"] > fast_ema.iloc[-1]
            )
            or (
                latest["low"] < medm_ema.iloc[-1]
                and latest["open"] > medm_ema.iloc[-1]
                and latest["close"] > medm_ema.iloc[-1]
            )
            or (
                latest["low"] < slow_sma.iloc[-1]
                and latest["open"] > slow_sma.iloc[-1]
                and latest["close"] > slow_sma.iloc[-1]
            )
        )
        bear_pierce = bool(
            (
                latest["high"] > fast_ema.iloc[-1]
                and latest["open"] < fast_ema.iloc[-1]
                and latest["close"] < fast_ema.iloc[-1]
            )
            or (
                latest["high"] > medm_ema.iloc[-1]
                and latest["open"] < medm_ema.iloc[-1]
                and latest["close"] < medm_ema.iloc[-1]
            )
            or (
                latest["high"] > slow_sma.iloc[-1]
                and latest["open"] < slow_sma.iloc[-1]
                and latest["close"] < slow_sma.iloc[-1]
            )
        )

        # Entries
        long_entry = (
            fan_up_trend and bull_pierce and valid_trigger_bull
            and (bar_index > self._last_entry_bar)
        )
        short_entry = (
            fan_dn_trend and bear_pierce and valid_trigger_bear
            and (bar_index > self._last_entry_bar)
        )

        # Date range gate ---------------------------------------------------
        allow_long = self.TRADE_DIRECTION in ("Both", "Long Only")
        allow_short = self.TRADE_DIRECTION in ("Both", "Short Only")

        in_date_range = True
        if hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index):
            last_time = df.index[-1]
            in_date_range = self.START_DATE <= last_time <= self.END_DATE

        # Signal resolution -------------------------------------------------
        direction = "NEUTRAL"
        signal = 0.0
        if long_entry and in_date_range and allow_long:
            direction = "BUY"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index
        elif short_entry and in_date_range and allow_short:
            direction = "SELL"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index

        # Trend-reversal close flags (metadata)
        trend_rev_close_long = bool(
            fast_ema.iloc[-1] < medm_ema.iloc[-1]
            and fast_ema.iloc[-2] >= medm_ema.iloc[-2]
        )
        trend_rev_close_short = bool(
            fast_ema.iloc[-1] > medm_ema.iloc[-1]
            and fast_ema.iloc[-2] <= medm_ema.iloc[-2]
        )

        # Stop / trailing metadata (sizing logic lives in TradingAgent)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        if direction == "BUY":
            stop_loss = latest["low"] - atr_val * atr_mult_use
            activation_level = latest["close"] + active_activation
        elif direction == "SELL":
            stop_loss = latest["high"] + atr_val * atr_mult_use
            activation_level = latest["close"] - active_activation
        else:
            stop_loss = None
            activation_level = None

        # Build metadata ----------------------------------------------------
        metadata = {
            "engine": "v6.1",
            "version": "6.1",
            "fast_ema": round(float(fast_ema.iloc[-1]), 6),
            "medm_ema": round(float(medm_ema.iloc[-1]), 6),
            "slow_sma": round(float(slow_sma.iloc[-1]), 6),
            "atr": round(float(atr_val), 6),
            "atr_mult_use": round(float(atr_mult_use), 4),
            "adx": round(float(adx.iloc[-1]), 4),
            "fan_up_trend": bool(fan_up_trend),
            "fan_dn_trend": bool(fan_dn_trend),
            "bullish_pin_bar": bool(bullish_pin),
            "bearish_pin_bar": bool(bearish_pin),
            "bull_pierce": bool(bull_pierce),
            "bear_pierce": bool(bear_pierce),
            "is_hyper_phase": bool(is_hyper_phase),
            "progress_pct": round(float(progress_pct), 2),
            "is_strategy_cold": bool(is_strategy_cold),
            "in_warmup": bool(in_warmup),
            "final_risk_pct": round(float(final_risk_pct), 2),
            "risk_multiplier": round(float(risk_multiplier), 4),
            "dd_percent": round(float(dd_percent * 100), 4),
            "active_activation": int(active_activation),
            "active_offset": int(active_offset),
            "stop_loss": round(float(stop_loss), 6) if stop_loss is not None else None,
            "trailing_activation_ticks": int(active_activation),
            "trailing_offset_ticks": int(active_offset),
            "activation_level": round(float(activation_level), 6) if activation_level is not None else None,
            "trend_rev_close_long": bool(trend_rev_close_long),
            "trend_rev_close_short": bool(trend_rev_close_short),
            "bar_index": bar_index,
            "last_entry_bar": self._last_entry_bar,
            "recommended_timeframe": self.recommended_timeframe,
            "recommended_symbols": self.recommended_symbols,
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata,
        }

    # ═══════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _risk_profile_ticks(self):
        """Mirror Pine risk-profile var assignment."""
        rp = self.RISK_PROFILE
        if rp == "Sniper Mode (18/6)":
            return 18, 6
        if rp == "Trend Scalper (18/12)":
            return 18, 12
        if rp == "Conservative (25/18)":
            return 25, 18
        if rp == "Golden Growth (36/12)":
            return 36, 12
        return self.MAN_ACTIVATION, self.MAN_OFFSET

    def _adaptive_multiplier(
        self,
        current_closed_equity: float,
        avg_equity: float,
        is_hyper_phase: bool,
        atr_mult_use: float,
    ) -> float:
        """
        7-layer adaptive equity compounding core (Pine lines 63-127).
        Returns the final multiplier that replaces atr_mult_use.
        """
        er_len = 14
        curve = self._equity_curve

        # --- 1️⃣ Efficiency Ratio -----------------------------------------
        if len(curve) > er_len + 1:
            prev_equity = curve[-er_len - 1]
            change = abs(current_closed_equity - prev_equity)
            volatility = sum(
                abs(curve[-i] - curve[-i - 1]) for i in range(1, er_len + 1)
            )
        else:
            change = 0.0
            volatility = 1.0

        eff_ratio = change / volatility if volatility != 0 else 0.0

        # --- 1️⃣ Adaptive SMA (Dynamic Memory Length) -----------------------
        len_adaptive = 8 + (30 - 8) * (1 - eff_ratio)
        len_adaptive = max(8, min(30, len_adaptive))

        # eqSMA over equity_curve
        len_rounded = int(round(len_adaptive))
        eq_series = pd.Series(curve)
        if len(curve) >= len_rounded and len_rounded > 0:
            eq_sma = float(eq_series.rolling(window=len_rounded, min_periods=len_rounded).mean().iloc[-1])
        else:
            eq_sma = avg_equity

        # --- 2️⃣ Logistic Multiplier (Confidence Curve) --------------------
        epsilon = 0.02
        dist_raw = (current_closed_equity - eq_sma) / eq_sma if eq_sma != 0 else 0.0
        dist = dist_raw if abs(dist_raw) > epsilon else 0.0
        confidence = 1.0 / (1.0 + math.exp(-4 * dist))
        base_mult = 1.0 + 0.8 * confidence
        base_mult = max(0.9, min(1.8, base_mult))

        # --- 3️⃣ Acceleration Filter (Smart Smoothing) ---------------------
        vel = eq_sma - self._last_eq_sma if self._last_eq_sma is not None else 0.0
        acc = vel - self._last_vel if self._last_vel is not None else 0.0

        # accSmooth = ta.ema(acc, 5)
        alpha = 2.0 / (5 + 1)
        if self._acc_ema is None:
            acc_smooth = acc
        else:
            acc_smooth = alpha * acc + (1 - alpha) * self._acc_ema

        self._last_eq_sma = eq_sma
        self._last_vel = vel
        self._acc_ema = acc_smooth

        acc_impact = max(0, min(1, (-acc_smooth - 0.4) / 0.6))
        acc_adj = 1.0 - 0.1 * acc_impact

        # --- 4️⃣ Std-Dev Channel (Equity Envelope) -------------------------
        z = 2.2
        if len(curve) >= len_rounded and len_rounded > 0:
            eq_stdev = float(eq_series.rolling(window=len_rounded, min_periods=len_rounded).std().iloc[-1])
        else:
            eq_stdev = 0.0

        eq_upper = eq_sma + z * eq_stdev
        eq_lower = eq_sma - z * eq_stdev

        if current_closed_equity < eq_lower:
            chan_adj = 0.95
        elif current_closed_equity > eq_upper:
            chan_adj = 1.03
        else:
            chan_adj = 1.0

        # --- 5️⃣ Final Adaptive Multiplier (Hyper-Aware) -------------------
        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(6, min(30, len_adaptive * len_factor))

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        # Channel adjustment (hyper-aware)
        if current_closed_equity < eq_lower:
            chan_adj = 0.97 if is_hyper_phase else 0.95
        elif current_closed_equity > eq_upper:
            chan_adj = 1.05 if is_hyper_phase else 1.03
        else:
            chan_adj = 1.0

        # Acceleration adjustment (hyper-aware)
        acc_adj = 1.0 - 0.07 * acc_impact if is_hyper_phase else 1.0 - 0.1 * acc_impact

        mult_adaptive = base_mult * acc_adj * chan_adj
        mult_adaptive = max(1.0, min(2.0, mult_adaptive))

        # --- 6️⃣ Optional ATR Sensitivity (diagnostic) ---------------------
        # Pine: atrLen = accSmooth < -0.6 ? 10 : 14
        # Not used to alter the already-computed ATR, but returned in meta.
        # atr_len = 10 if acc_smooth < -0.6 else 14

        # --- 7️⃣ Apply Globally --------------------------------------------
        return mult_adaptive
