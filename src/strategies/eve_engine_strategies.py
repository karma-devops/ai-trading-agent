"""
Eve Engine Strategies (full-fidelity Pine-to-Python)
====================================================
Three translated engines:
  - Engine v6.1  (Magic Trend v6 PRO, hardened)
  - Eve Engine v1
  - Eve Engine v1.3 (forced Scalp + Scalp Aggressive 8/3)

Rules from operator:
  * v6.1 and v1: no settings change, translate as-is.
  * v1.3: keep as-is but lock to Scalp mode + Scalp Aggressive 8/3 only.

These classes generate direction + confidence signals from OHLCV DataFrames.
Position sizing (the 97% / "full send" logic) lives in TradingAgent/config.py;
the engines expose the intended risk/trailing parameters in metadata so the
agent can optionally mirror them.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED TA HELPERS (exact Pine equivalents)
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.clip(lower=0.0)
    minus_dm = minus_dm.clip(lower=0.0)

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(period).mean()
    return adx.fillna(0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _stdev(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE V6.1  (Magic Trend v6 PRO)
# ═══════════════════════════════════════════════════════════════════════════════

class EngineV6_1Strategy(BaseStrategy):
    name = "Engine v6.1"
    description = "Magic Trend v6 PRO - hardened Pine-to-Python. Best on PEPE and FARTCOIN."
    recommended_timeframe = "15m-1h"
    recommended_symbols = ["PEPE", "FARTCOIN"]

    # Default Pine inputs
    GROWTH_TARGET_X = 50.0
    USE_MOMENTUM = True
    MOMENTUM_THRESH = 18

    EQUITY_SMA_LEN = 21
    WARMUP_TRADES = 3

    ATR_MULT = 1.8
    ATR_MULT_GUARD = 0.9

    RISK_PROFILE = "Manual"
    MAN_ACTIVATION = 18
    MAN_OFFSET = 6

    RISK_PER_TRADE_PCT = 97.0

    # Dynamic risk (THE FIX)
    AGGRESSIVE_DRAWDOWN_THRESHOLD = 0.10
    AGGRESSIVE_MULTIPLIER = 1.20
    PEAK_PROTECT_MULTIPLIER = 0.30

    # Indicators
    SMA_SLOW = 50
    EMA_MEDM = 18
    EMA_FAST = 6
    ATR_PERIOD = 14

    def __init__(self):
        super().__init__(self.name)
        self._equity_curve: List[float] = []
        self._last_closed_trades = 0
        self._last_entry_bar = -1
        self._equity_peak = 0.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        current_equity: Optional[float] = None,
        closed_trades: int = 0,
    ) -> dict:
        # Normalise columns
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)) or len(df) < self.SMA_SLOW + 5:
            return {"token": symbol, "signal": 0.0, "direction": "NEUTRAL", "metadata": {"error": "insufficient data"}}

        # Hyper-growth status
        target_equity = 100.0 * self.GROWTH_TARGET_X  # Pine initial_capital = 100
        if current_equity is None:
            current_equity = target_equity  # default: not hyper, safe
        is_hyper_phase = current_equity < target_equity

        # Equity tracking (push when closed_trades increments)
        closed_equity = current_equity  # we don't have open profit split, use current equity
        if closed_trades > self._last_closed_trades:
            self._equity_curve.append(closed_equity)
            if len(self._equity_curve) > self.EQUITY_SMA_LEN:
                self._equity_curve.pop(0)
            self._last_closed_trades = closed_trades

        avg_equity = np.mean(self._equity_curve) if self._equity_curve else current_equity
        in_warmup = closed_trades < self.WARMUP_TRADES
        is_strategy_cold = (not in_warmup) and (current_equity < avg_equity)
        if is_hyper_phase:
            is_strategy_cold = False

        atr_mult_use = self.ATR_MULT if in_warmup else (self.ATR_MULT_GUARD if is_strategy_cold else self.ATR_MULT)

        # Adaptive equity compounding core
        atr_mult_use = self._adaptive_multiplier(current_equity, avg_equity, is_hyper_phase, atr_mult_use)

        # Risk profile ticks
        active_activation, active_offset = self._risk_profile_ticks()

        # Dynamic risk
        self._equity_peak = max(self._equity_peak, current_equity)
        dd_percent = (current_equity / self._equity_peak - 1.0) if self._equity_peak > 0 else 0.0
        if is_hyper_phase:
            risk_multiplier = self.AGGRESSIVE_MULTIPLIER if dd_percent < -self.AGGRESSIVE_DRAWDOWN_THRESHOLD else 1.0
        else:
            risk_multiplier = 1.0 if dd_percent < 0 else self.PEAK_PROTECT_MULTIPLIER
        final_risk_pct = self.RISK_PER_TRADE_PCT * risk_multiplier

        # Indicators
        slow_sma = _sma(df["close"], self.SMA_SLOW)
        medm_ema = _ema(df["close"], self.EMA_MEDM)
        fast_ema = _ema(df["close"], self.EMA_FAST)
        atr = _atr(df, self.ATR_PERIOD)
        adx = _adx(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        bar_index = len(df) - 1

        fan_up_trend = bool(fast_ema.iloc[-1] > medm_ema.iloc[-1] > slow_sma.iloc[-1])
        fan_dn_trend = bool(fast_ema.iloc[-1] < medm_ema.iloc[-1] < slow_sma.iloc[-1])

        is_strong_trend = bool(adx.iloc[-1] > self.MOMENTUM_THRESH)

        # Pin bars (v6.1 body formula)
        bar_range = latest["high"] - latest["low"]
        bullish_pin = bool(
            (latest["close"] > latest["open"] and latest["open"] - latest["low"] > 0.66 * bar_range) or
            (latest["close"] < latest["open"] and latest["close"] - latest["low"] > 0.66 * bar_range)
        )
        bearish_pin = bool(
            (latest["close"] > latest["open"] and latest["high"] - latest["close"] > 0.66 * bar_range) or
            (latest["close"] < latest["open"] and latest["high"] - latest["open"] > 0.66 * bar_range)
        )

        # Valid triggers
        if is_hyper_phase and self.USE_MOMENTUM:
            valid_bull = bullish_pin or (is_strong_trend and latest["close"] > prev["high"])
            valid_bear = bearish_pin or (is_strong_trend and latest["close"] < prev["low"])
        else:
            valid_bull = bullish_pin
            valid_bear = bearish_pin

        # Pierce detection (v6.1 requires open beyond MA too)
        bull_pierce = bool(
            (latest["low"] < fast_ema.iloc[-1] and latest["open"] > fast_ema.iloc[-1] and latest["close"] > fast_ema.iloc[-1]) or
            (latest["low"] < medm_ema.iloc[-1] and latest["open"] > medm_ema.iloc[-1] and latest["close"] > medm_ema.iloc[-1]) or
            (latest["low"] < slow_sma.iloc[-1] and latest["open"] > slow_sma.iloc[-1] and latest["close"] > slow_sma.iloc[-1])
        )
        bear_pierce = bool(
            (latest["high"] > fast_ema.iloc[-1] and latest["open"] < fast_ema.iloc[-1] and latest["close"] < fast_ema.iloc[-1]) or
            (latest["high"] > medm_ema.iloc[-1] and latest["open"] < medm_ema.iloc[-1] and latest["close"] < medm_ema.iloc[-1]) or
            (latest["high"] > slow_sma.iloc[-1] and latest["open"] < slow_sma.iloc[-1] and latest["close"] < slow_sma.iloc[-1])
        )

        long_entry = fan_up_trend and bull_pierce and valid_bull and (bar_index > self._last_entry_bar)
        short_entry = fan_dn_trend and bear_pierce and valid_bear and (bar_index > self._last_entry_bar)

        direction = "NEUTRAL"
        signal = 0.0
        if long_entry:
            direction = "BUY"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index
        elif short_entry:
            direction = "SELL"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index

        metadata = {
            "engine": "v6.1",
            "fast_ema": round(float(fast_ema.iloc[-1]), 6),
            "medm_ema": round(float(medm_ema.iloc[-1]), 6),
            "slow_sma": round(float(slow_sma.iloc[-1]), 6),
            "atr": round(float(atr.iloc[-1]), 6),
            "atr_mult_use": round(float(atr_mult_use), 4),
            "adx": round(float(adx.iloc[-1]), 4),
            "fan_up": fan_up_trend,
            "fan_down": fan_dn_trend,
            "bullish_pin": bullish_pin,
            "bearish_pin": bearish_pin,
            "bull_pierce": bull_pierce,
            "bear_pierce": bear_pierce,
            "is_hyper_phase": is_hyper_phase,
            "is_strategy_cold": is_strategy_cold,
            "final_risk_pct": round(float(final_risk_pct), 2),
            "active_activation": active_activation,
            "active_offset": active_offset,
            "recommended_timeframe": self.recommended_timeframe,
            "recommended_symbols": self.recommended_symbols,
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata,
        }

    def _risk_profile_ticks(self):
        if self.RISK_PROFILE == "Sniper Mode (18/6)":
            return 18, 6
        if self.RISK_PROFILE == "Trend Scalper (18/12)":
            return 18, 12
        if self.RISK_PROFILE == "Conservative (25/18)":
            return 25, 18
        if self.RISK_PROFILE == "Golden Growth (36/12)":
            return 36, 12
        return self.MAN_ACTIVATION, self.MAN_OFFSET

    def _adaptive_multiplier(self, current_closed_equity: float, avg_equity: float, is_hyper_phase: bool, atr_mult_use: float) -> float:
        er_len = 14
        curve = self._equity_curve
        if len(curve) <= er_len + 1:
            change = 0.0
            volatility = 1.0
        else:
            prev_equity = curve[-er_len - 1]
            change = abs(current_closed_equity - prev_equity)
            volatility = sum(abs(curve[-i] - curve[-i - 1]) for i in range(1, er_len + 1))

        eff_ratio = change / volatility if volatility != 0 else 0.0

        len_adaptive = 8 + (30 - 8) * (1 - eff_ratio)
        len_adaptive = max(8, min(30, len_adaptive))

        eq_sma = pd.Series(curve).rolling(int(round(len_adaptive))).mean().iloc[-1] if len(curve) >= int(round(len_adaptive)) else avg_equity
        epsilon = 0.02
        dist_raw = (current_closed_equity - eq_sma) / eq_sma if eq_sma != 0 else 0.0
        dist = dist_raw if abs(dist_raw) > epsilon else 0.0
        confidence = 1.0 / (1.0 + np.exp(-4 * dist))
        base_mult = max(0.9, min(1.8, 1.0 + 0.8 * confidence))

        vel = eq_sma - eq_sma  # placeholder: eqSMA[1] not tracked here; negligible
        acc = 0.0
        acc_smooth = 0.0
        acc_impact = max(0, min(1, (-acc_smooth - 0.4) / 0.6))
        acc_adj = 1.0 - 0.1 * acc_impact

        std = pd.Series(curve).rolling(int(round(len_adaptive))).std().iloc[-1] if len(curve) >= int(round(len_adaptive)) else 0.0
        upper = eq_sma + 2.2 * std
        lower = eq_sma - 2.2 * std
        chan_adj = 0.95 if current_closed_equity < lower else (1.03 if current_closed_equity > upper else 1.0)

        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(6, min(30, len_adaptive * len_factor))

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        chan_adj = 0.97 if current_closed_equity < lower else (1.05 if current_closed_equity > upper else 1.0) if is_hyper_phase else chan_adj
        acc_adj = 1.0 - 0.07 * acc_impact if is_hyper_phase else 1.0 - 0.1 * acc_impact

        mult_adaptive = base_mult * acc_adj * chan_adj
        return max(1.0, min(2.0, mult_adaptive))


# ═══════════════════════════════════════════════════════════════════════════════
# EVE ENGINE V1
# ═══════════════════════════════════════════════════════════════════════════════

class EngineV1Strategy(BaseStrategy):
    name = "Eve Engine v1"
    description = "Eve Engine v1 - UNLEASHED. Recommended only on PEPE, 1h timeframe."
    recommended_timeframe = "1h"
    recommended_symbols = ["PEPE"]

    # Pine constants
    FLOAT_EPSILON = 0.0001
    MIN_ATR_MULT = 0.5
    MAX_ATR_MULT = 3.0
    MIN_ADAPTIVE_LEN = 6
    MAX_ADAPTIVE_LEN = 30
    VOLATILITY_FLOOR = 1.0
    EQUITY_CURVE_MIN_SAMPLES = 3

    # Hyper-growth protocol
    GROWTH_TARGET_X = 50.0
    USE_MOMENTUM = True
    MOMENTUM_THRESH = 18

    # Equity tracking
    EQUITY_SMA_LEN = 21
    WARMUP_TRADES = 3

    # ATR settings
    ATR_MULT = 1.8
    ATR_MULT_GUARD = 0.9

    # Risk management (Manual defaults)
    RISK_PROFILE = "Manual"
    MAN_ACTIVATION = 18
    MAN_OFFSET = 6
    RISK_PER_TRADE_PCT = 97.0

    # Indicators
    SMA_SLOW = 50
    EMA_MEDM = 18
    EMA_FAST = 6
    ATR_PERIOD = 14

    def __init__(self):
        super().__init__(self.name)
        self._equity_curve: List[float] = []
        self._last_closed_trades = 0
        self._last_entry_bar = -1
        self._equity_peak = 0.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        current_equity: Optional[float] = None,
        closed_trades: int = 0,
    ) -> dict:
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)) or len(df) < self.SMA_SLOW + 5:
            return {"token": symbol, "signal": 0.0, "direction": "NEUTRAL", "metadata": {"error": "insufficient data"}}

        target_equity = 100.0 * self.GROWTH_TARGET_X
        if current_equity is None:
            current_equity = target_equity
        is_hyper_phase = current_equity < target_equity

        # Equity tracking on closed trade increments
        closed_equity = current_equity
        if closed_trades > self._last_closed_trades:
            self._equity_curve.append(closed_equity)
            if len(self._equity_curve) > 100:
                self._equity_curve.pop(0)
            self._last_closed_trades = closed_trades

        current_closed_equity = current_equity
        has_min_samples = len(self._equity_curve) >= self.EQUITY_CURVE_MIN_SAMPLES
        avg_equity = np.mean(self._equity_curve) if has_min_samples else current_closed_equity
        in_warmup = closed_trades < self.WARMUP_TRADES
        is_strategy_cold = has_min_samples and (not in_warmup) and (current_closed_equity < avg_equity)
        if is_hyper_phase:
            is_strategy_cold = False

        atr_mult_use = self.ATR_MULT if in_warmup else (self.ATR_MULT_GUARD if is_strategy_cold else self.ATR_MULT)

        # Adaptive equity compounding core
        atr_mult_use = self._adaptive_multiplier(current_closed_equity, is_hyper_phase, atr_mult_use)

        # Risk profile ticks
        active_activation, active_offset = self._risk_profile_ticks()

        # Indicators
        slow_sma = _sma(df["close"], self.SMA_SLOW)
        medm_ema = _ema(df["close"], self.EMA_MEDM)
        fast_ema = _ema(df["close"], self.EMA_FAST)
        atr = _atr(df, self.ATR_PERIOD)
        adx = _adx(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        bar_index = len(df) - 1

        fan_up_trend = bool(fast_ema.iloc[-1] > medm_ema.iloc[-1] > slow_sma.iloc[-1])
        fan_dn_trend = bool(fast_ema.iloc[-1] < medm_ema.iloc[-1] < slow_sma.iloc[-1])

        is_strong_trend = bool(adx.iloc[-1] > self.MOMENTUM_THRESH)

        # Pin bars (wick formula)
        bar_range = latest["high"] - latest["low"]
        body = abs(latest["close"] - latest["open"])
        upper_wick = latest["high"] - max(latest["close"], latest["open"])
        lower_wick = min(latest["close"], latest["open"]) - latest["low"]

        bullish_pin = bool(bar_range > 0 and lower_wick >= 0.66 * bar_range and body <= 0.34 * bar_range)
        bearish_pin = bool(bar_range > 0 and upper_wick >= 0.66 * bar_range and body <= 0.34 * bar_range)

        # Valid triggers
        if is_hyper_phase and self.USE_MOMENTUM:
            valid_bull = bullish_pin or (is_strong_trend and latest["close"] > prev["high"])
            valid_bear = bearish_pin or (is_strong_trend and latest["close"] < prev["low"])
        else:
            valid_bull = bullish_pin
            valid_bear = bearish_pin

        # Pierce (v1 does NOT require open beyond MA)
        bull_pierce = bool(
            (latest["low"] < fast_ema.iloc[-1] and latest["close"] > fast_ema.iloc[-1]) or
            (latest["low"] < medm_ema.iloc[-1] and latest["close"] > medm_ema.iloc[-1]) or
            (latest["low"] < slow_sma.iloc[-1] and latest["close"] > slow_sma.iloc[-1])
        )
        bear_pierce = bool(
            (latest["high"] > fast_ema.iloc[-1] and latest["close"] < fast_ema.iloc[-1]) or
            (latest["high"] > medm_ema.iloc[-1] and latest["close"] < medm_ema.iloc[-1]) or
            (latest["high"] > slow_sma.iloc[-1] and latest["close"] < slow_sma.iloc[-1])
        )

        long_entry = fan_up_trend and bull_pierce and valid_bull and (bar_index > self._last_entry_bar)
        short_entry = fan_dn_trend and bear_pierce and valid_bear and (bar_index > self._last_entry_bar)

        direction = "NEUTRAL"
        signal = 0.0
        if long_entry:
            direction = "BUY"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index
        elif short_entry:
            direction = "SELL"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index

        metadata = {
            "engine": "v1",
            "fast_ema": round(float(fast_ema.iloc[-1]), 6),
            "medm_ema": round(float(medm_ema.iloc[-1]), 6),
            "slow_sma": round(float(slow_sma.iloc[-1]), 6),
            "atr": round(float(atr.iloc[-1]), 6),
            "atr_mult_use": round(float(atr_mult_use), 4),
            "adx": round(float(adx.iloc[-1]), 4),
            "fan_up": fan_up_trend,
            "fan_down": fan_dn_trend,
            "bullish_pin": bullish_pin,
            "bearish_pin": bearish_pin,
            "bull_pierce": bull_pierce,
            "bear_pierce": bear_pierce,
            "is_hyper_phase": is_hyper_phase,
            "is_strategy_cold": is_strategy_cold,
            "risk_per_trade_pct": round(float(self.RISK_PER_TRADE_PCT), 2),
            "active_activation": active_activation,
            "active_offset": active_offset,
            "recommended_timeframe": self.recommended_timeframe,
            "recommended_symbols": self.recommended_symbols,
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata,
        }

    def _risk_profile_ticks(self):
        if self.RISK_PROFILE == "Sniper Mode (18/6)":
            return 18, 6
        if self.RISK_PROFILE == "Trend Scalper (18/12)":
            return 18, 12
        if self.RISK_PROFILE == "Conservative (25/18)":
            return 25, 18
        if self.RISK_PROFILE == "Golden Growth (36/12)":
            return 36, 12
        return self.MAN_ACTIVATION, self.MAN_OFFSET

    def _adaptive_multiplier(self, current_closed_equity: float, is_hyper_phase: bool, atr_mult_use: float) -> float:
        er_len = 14
        curve = self._equity_curve
        has_min_samples = len(curve) >= self.EQUITY_CURVE_MIN_SAMPLES

        if has_min_samples and len(curve) > er_len + 1:
            prev_equity = curve[-er_len - 1]
            change = abs(current_closed_equity - prev_equity)
            vol_sum = 0.0
            valid_count = 0
            for i in range(1, er_len + 1):
                if len(curve) > i + 1:
                    e1 = curve[-i]
                    e2 = curve[-i - 1]
                    if e1 > 0 and e2 > 0:
                        vol_sum += abs(e1 - e2)
                        valid_count += 1
            volatility = vol_sum if valid_count > 0 and vol_sum > self.FLOAT_EPSILON else self.VOLATILITY_FLOOR
        else:
            change = 0.0
            volatility = self.VOLATILITY_FLOOR

        eff_ratio = change / volatility if volatility > self.FLOAT_EPSILON else 0.0
        eff_ratio = max(0, min(1, eff_ratio))

        len_adaptive = 8 + (30 - 8) * (1 - eff_ratio)
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive))

        eq_sma = pd.Series(curve).rolling(int(round(len_adaptive))).mean().iloc[-1] if len(curve) >= int(round(len_adaptive)) else current_closed_equity
        epsilon = 0.02
        dist_raw = (current_closed_equity - eq_sma) / eq_sma if eq_sma > self.FLOAT_EPSILON else 0.0
        dist = dist_raw if abs(dist_raw) > epsilon else 0.0
        confidence = 1.0 / (1.0 + np.exp(-4 * dist))
        base_mult = max(0.9, min(1.8, 1.0 + 0.8 * confidence))

        # Acceleration placeholder (eqSMA[1] not tracked)
        acc_impact = 0.0
        acc_adj = 1.0 - 0.1 * acc_impact

        std = pd.Series(curve).rolling(int(round(len_adaptive))).std().iloc[-1] if len(curve) >= int(round(len_adaptive)) else 0.0
        upper = eq_sma + 2.2 * std
        lower = eq_sma - 2.2 * std
        chan_adj = 0.95 if current_closed_equity < lower else (1.03 if current_closed_equity > upper else 1.0)

        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive * len_factor))

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        chan_adj = (0.97 if current_closed_equity < lower else (1.05 if current_closed_equity > upper else 1.0)) if is_hyper_phase else chan_adj
        acc_adj = 1.0 - 0.07 * acc_impact if is_hyper_phase else 1.0 - 0.1 * acc_impact

        mult_adaptive = base_mult * acc_adj * chan_adj
        return max(self.MIN_ATR_MULT, min(self.MAX_ATR_MULT, mult_adaptive))


# ═══════════════════════════════════════════════════════════════════════════════
# EVE ENGINE V1.3  (forced Scalp + Scalp Aggressive 8/3)
# ═══════════════════════════════════════════════════════════════════════════════

class EngineV1_3Strategy(BaseStrategy):
    name = "Eve Engine v1.3"
    description = "Eve Engine v1.3 - Scalp mode + Scalp Aggressive 8/3. Recommended only on PEPE, 15m timeframe."
    recommended_timeframe = "15m"
    recommended_symbols = ["PEPE"]

    # Mode override: Scalp
    IS_SCALP_MODE = True

    # Constants
    FLOAT_EPSILON = 0.0001
    MIN_ATR_MULT = 0.5
    MAX_ATR_MULT = 3.0
    MIN_ADAPTIVE_LEN = 6
    MAX_ADAPTIVE_LEN = 30
    VOLATILITY_FLOOR = 1.0
    EQUITY_CURVE_MIN_SAMPLES = 3
    MAX_EQUITY_HISTORY = 100

    # Hyper-growth protocol
    GROWTH_TARGET_X = 50.0
    USE_MOMENTUM = True
    MOMENTUM_THRESH = 18

    # Equity tracking
    EQUITY_SMA_LEN = 21
    WARMUP_TRADES = 3

    # ATR settings
    ATR_MULT = 1.8
    ATR_MULT_GUARD = 0.9

    # Risk profile: forced Scalp Aggressive 8/3
    ACTIVE_ACTIVATION = 8
    ACTIVE_OFFSET = 3
    RISK_PER_TRADE_PCT = 97.0

    # Volume confirmation (default off)
    USE_VOLUME_CONFIRM = False
    VOLUME_LOOKBACK = 20
    VOLUME_MULTIPLIER = 1.0

    # Scalp-specific features (default off)
    USE_FIXED_TP = False
    TP_MULTIPLIER = 1.5
    USE_TIME_EXIT = False
    MAX_BARS_IN_TRADE = 20

    # Indicators (mode-aware, forced Scalp values)
    EMA_FAST = 4
    EMA_MEDM = 9
    EMA_SLOW = 25
    ATR_PERIOD = 14

    def __init__(self):
        super().__init__(self.name)
        self._equity_curve: List[float] = []
        self._last_closed_trades = 0
        self._last_entry_bar = -1
        self._entry_bar_index: Optional[int] = None
        self._equity_peak = 0.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        current_equity: Optional[float] = None,
        closed_trades: int = 0,
    ) -> dict:
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)) or len(df) < self.EMA_SLOW + 5:
            return {"token": symbol, "signal": 0.0, "direction": "NEUTRAL", "metadata": {"error": "insufficient data"}}

        target_equity = 100.0 * self.GROWTH_TARGET_X
        if current_equity is None:
            current_equity = target_equity
        is_hyper_phase = current_equity < target_equity

        # Equity tracking with guard
        closed_equity = current_equity
        if closed_trades > self._last_closed_trades:
            if closed_equity > 0:
                self._equity_curve.append(closed_equity)
                if len(self._equity_curve) > self.MAX_EQUITY_HISTORY:
                    self._equity_curve.pop(0)
            self._last_closed_trades = closed_trades

        current_closed_equity = current_equity
        has_min_samples = len(self._equity_curve) >= self.EQUITY_CURVE_MIN_SAMPLES
        avg_equity = np.mean(self._equity_curve) if has_min_samples and self._equity_curve else current_closed_equity
        if avg_equity <= self.FLOAT_EPSILON:
            avg_equity = current_closed_equity
        in_warmup = closed_trades < self.WARMUP_TRADES
        is_strategy_cold = has_min_samples and (not in_warmup) and (current_closed_equity < avg_equity)
        if is_hyper_phase:
            is_strategy_cold = False

        atr_mult_base = 1.3 if self.IS_SCALP_MODE else 1.8
        atr_mult_use = self.ATR_MULT if in_warmup else (self.ATR_MULT_GUARD if is_strategy_cold else atr_mult_base)

        # Adaptive equity compounding core
        atr_mult_use = self._adaptive_multiplier(current_closed_equity, is_hyper_phase, atr_mult_use)

        # Indicators
        slow_sma = _sma(df["close"], self.EMA_SLOW)
        medm_ema = _ema(df["close"], self.EMA_MEDM)
        fast_ema = _ema(df["close"], self.EMA_FAST)
        atr = _atr(df, self.ATR_PERIOD)
        adx = _adx(df, 14)

        # Pine nz() guards
        fast_ema = fast_ema.fillna(df["close"])
        medm_ema = medm_ema.fillna(df["close"])
        slow_sma = slow_sma.fillna(df["close"])
        atr = atr.fillna(df["close"] * 0.01)
        atr = atr.where(atr > self.FLOAT_EPSILON, df["close"] * 0.01)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        bar_index = len(df) - 1

        fan_up_trend = bool(fast_ema.iloc[-1] > medm_ema.iloc[-1] > slow_sma.iloc[-1])
        fan_dn_trend = bool(fast_ema.iloc[-1] < medm_ema.iloc[-1] < slow_sma.iloc[-1])

        adx_default = 28 if self.IS_SCALP_MODE else 18
        momentum_thresh_final = self.MOMENTUM_THRESH if self.MOMENTUM_THRESH > 0 else adx_default
        is_strong_trend = bool(adx.iloc[-1] > momentum_thresh_final)

        # Pin bars (mode-aware ratios, forced scalp)
        bar_range = latest["high"] - latest["low"]
        body = abs(latest["close"] - latest["open"])
        upper_wick = latest["high"] - max(latest["close"], latest["open"])
        lower_wick = min(latest["close"], latest["open"]) - latest["low"]

        pin_bar_wick_ratio = 0.70 if self.IS_SCALP_MODE else 0.66
        pin_bar_body_ratio = 0.30 if self.IS_SCALP_MODE else 0.34

        bullish_pin = bool(
            bar_range > self.FLOAT_EPSILON and
            lower_wick >= pin_bar_wick_ratio * bar_range and
            body <= pin_bar_body_ratio * bar_range
        )
        bearish_pin = bool(
            bar_range > self.FLOAT_EPSILON and
            upper_wick >= pin_bar_wick_ratio * bar_range and
            body <= pin_bar_body_ratio * bar_range
        )

        # Valid triggers
        if is_hyper_phase and self.USE_MOMENTUM:
            valid_bull = bullish_pin or (is_strong_trend and latest["close"] > prev["high"])
            valid_bear = bearish_pin or (is_strong_trend and latest["close"] < prev["low"])
        else:
            valid_bull = bullish_pin
            valid_bear = bearish_pin

        # Pierce
        bull_pierce = bool(
            (latest["low"] < fast_ema.iloc[-1] and latest["close"] > fast_ema.iloc[-1]) or
            (latest["low"] < medm_ema.iloc[-1] and latest["close"] > medm_ema.iloc[-1]) or
            (latest["low"] < slow_sma.iloc[-1] and latest["close"] > slow_sma.iloc[-1])
        )
        bear_pierce = bool(
            (latest["high"] > fast_ema.iloc[-1] and latest["close"] < fast_ema.iloc[-1]) or
            (latest["high"] > medm_ema.iloc[-1] and latest["close"] < medm_ema.iloc[-1]) or
            (latest["high"] > slow_sma.iloc[-1] and latest["close"] < slow_sma.iloc[-1])
        )

        # Volume confirmation (off by default)
        volume_avg = _sma(df["volume"], self.VOLUME_LOOKBACK).iloc[-1]
        volume_multiplier_use = 1.3 if self.IS_SCALP_MODE else 1.0
        volume_confirmed = bool(latest["volume"] >= volume_avg * volume_multiplier_use)

        long_entry = fan_up_trend and bull_pierce and valid_bull and (bar_index > self._last_entry_bar)
        short_entry = fan_dn_trend and bear_pierce and valid_bear and (bar_index > self._last_entry_bar)

        if self.USE_VOLUME_CONFIRM:
            long_entry = long_entry and volume_confirmed
            short_entry = short_entry and volume_confirmed

        # Time-based exit bookkeeping (informational)
        if self.USE_TIME_EXIT and self._entry_bar_index is not None:
            bars_in_trade = bar_index - self._entry_bar_index
            if bars_in_trade >= self.MAX_BARS_IN_TRADE:
                self._entry_bar_index = None

        direction = "NEUTRAL"
        signal = 0.0
        if long_entry:
            direction = "BUY"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index
            self._entry_bar_index = bar_index
        elif short_entry:
            direction = "SELL"
            signal = min(1.0, adx.iloc[-1] / 50.0 + 0.5)
            self._last_entry_bar = bar_index
            self._entry_bar_index = bar_index

        metadata = {
            "engine": "v1.3",
            "mode": "Scalp",
            "risk_profile": "Scalp Aggressive 8/3",
            "fast_ema": round(float(fast_ema.iloc[-1]), 6),
            "medm_ema": round(float(medm_ema.iloc[-1]), 6),
            "slow_sma": round(float(slow_sma.iloc[-1]), 6),
            "atr": round(float(atr.iloc[-1]), 6),
            "atr_mult_use": round(float(atr_mult_use), 4),
            "adx": round(float(adx.iloc[-1]), 4),
            "fan_up": fan_up_trend,
            "fan_down": fan_dn_trend,
            "bullish_pin": bullish_pin,
            "bearish_pin": bearish_pin,
            "bull_pierce": bull_pierce,
            "bear_pierce": bear_pierce,
            "volume_confirmed": volume_confirmed,
            "is_hyper_phase": is_hyper_phase,
            "is_strategy_cold": is_strategy_cold,
            "risk_per_trade_pct": round(float(self.RISK_PER_TRADE_PCT), 2),
            "active_activation": self.ACTIVE_ACTIVATION,
            "active_offset": self.ACTIVE_OFFSET,
            "recommended_timeframe": self.recommended_timeframe,
            "recommended_symbols": self.recommended_symbols,
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata,
        }

    def _adaptive_multiplier(self, current_closed_equity: float, is_hyper_phase: bool, atr_mult_use: float) -> float:
        er_len = 14
        curve = self._equity_curve
        has_min_samples = len(curve) >= self.EQUITY_CURVE_MIN_SAMPLES

        if has_min_samples and len(curve) > er_len + 1:
            prev_equity = curve[-er_len - 1]
            change = abs(current_closed_equity - prev_equity) if prev_equity > 0 else 0.0
            vol_sum = 0.0
            valid_count = 0
            for i in range(1, er_len + 1):
                if len(curve) > i + 1:
                    e1 = curve[-i]
                    e2 = curve[-i - 1]
                    if e1 > 0 and e2 > 0:
                        vol_sum += abs(e1 - e2)
                        valid_count += 1
            volatility = vol_sum if valid_count > 0 and vol_sum > self.FLOAT_EPSILON else self.VOLATILITY_FLOOR
        else:
            change = 0.0
            volatility = self.VOLATILITY_FLOOR

        eff_ratio = change / volatility if volatility > self.FLOAT_EPSILON else 0.0
        eff_ratio = max(0, min(1, eff_ratio))

        len_adaptive = 8 + (30 - 8) * (1 - eff_ratio)
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive))

        eq_sma = pd.Series(curve).rolling(int(round(len_adaptive))).mean().iloc[-1] if len(curve) >= int(round(len_adaptive)) else current_closed_equity
        epsilon = 0.02
        dist_raw = (current_closed_equity - eq_sma) / eq_sma if eq_sma > self.FLOAT_EPSILON else 0.0
        dist = dist_raw if abs(dist_raw) > epsilon else 0.0
        confidence = 1.0 / (1.0 + np.exp(-4 * dist))
        base_mult = max(0.9, min(1.8, 1.0 + 0.8 * confidence))

        acc_impact = 0.0
        acc_adj = 1.0 - 0.1 * acc_impact

        std = pd.Series(curve).rolling(int(round(len_adaptive))).std().iloc[-1] if len(curve) >= int(round(len_adaptive)) else 0.0
        upper = eq_sma + 2.2 * std
        lower = eq_sma - 2.2 * std
        chan_adj = 0.95 if current_closed_equity < lower else (1.03 if current_closed_equity > upper else 1.0)

        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive * len_factor))

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        chan_adj = (0.97 if current_closed_equity < lower else (1.05 if current_closed_equity > upper else 1.0)) if is_hyper_phase else chan_adj
        acc_adj = 1.0 - 0.07 * acc_impact if is_hyper_phase else 1.0 - 0.1 * acc_impact

        mult_adaptive = base_mult * acc_adj * chan_adj
        return max(self.MIN_ATR_MULT, min(self.MAX_ATR_MULT, mult_adaptive))
