"""
Full-fidelity v1.3 translation of the Pine Script Eve Engine v1.3.
Scalp-only.  Hard-coded engineMode='Scalp' and risk profile Scalp Aggressive 8/3.
"""

import os
import sys

# Allow import of BaseStrategy when running this file directly for smoke-test
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
from src.strategies.base_strategy import BaseStrategy


class EngineV1_3Strategy(BaseStrategy):
    """
    Eve Engine v1.3 — full-fidelity Pine Script translation (scalp-only).

    * Forced ``engineMode = 'Scalp'``
    * Risk profile locked to ``Scalp Aggressive (8/3)``
    * Scalp EMA/SMA lengths (4 / 9 / 25) and scalp ATR multiplier base (1.3)
    * Adaptive equity compounding, DMI/ADX, pin-bar detection,
      momentum triggers, price-pierce entries, TP / time-exit levels
    """

    def __init__(self, name: str = "Eve Engine v1.3"):
        super().__init__(name)

        # ------------------------------------------------------------------
        #  Forced mode & profile
        # ------------------------------------------------------------------
        self.engine_mode = "Scalp"
        self.active_activation = 8
        self.active_offset = 3

        # ------------------------------------------------------------------
        #  Mode-aware indicator lengths  (scalp defaults)
        # ------------------------------------------------------------------
        self.ema_fast_len = 4
        self.ema_medm_len = 9
        self.ema_slow_len = 25
        self.atr_period = 14

        # ------------------------------------------------------------------
        #  ATR multiplier (input default preserved exactly)
        # ------------------------------------------------------------------
        self.atr_mult_input = 1.8      # Pine input.float default
        self.atr_mult_base = 1.3       # scalp atr_mult_base
        self.atr_mult_guard = 0.9      # cold-market guard

        # ------------------------------------------------------------------
        #  Hyper-growth protocol (input defaults preserved)
        # ------------------------------------------------------------------
        self.growth_target_x = 50.0
        self.use_momentum = True
        # tooltip says Swing=18, Scalp=28 – we force scalp-recommended value
        self.momentum_thresh = 28

        # ------------------------------------------------------------------
        #  Equity tracking (input defaults preserved)
        # ------------------------------------------------------------------
        self.equity_sma_len = 21
        self.warmup_trades = 3
        self.use_equity_guard = False
        self.eq_percent = 0.7
        self.initial_capital = 100.0

        # ------------------------------------------------------------------
        #  Risk per trade (input default preserved)
        # ------------------------------------------------------------------
        self.risk_per_trade_pct = 97.0

        # ------------------------------------------------------------------
        #  Trade direction (input default preserved)
        # ------------------------------------------------------------------
        self.trade_direction = "Both"

        # ------------------------------------------------------------------
        #  Volume confirmation (input defaults preserved)
        # ------------------------------------------------------------------
        self.use_volume_confirm = False
        self.volume_lookback = 20
        self.volume_multiplier = 1.3  # scalp volumeMultiplier_use

        # ------------------------------------------------------------------
        #  Scalp-specific features (input defaults preserved)
        # ------------------------------------------------------------------
        self.use_fixed_tp = False
        self.tp_multiplier = 1.5
        self.use_time_exit = False
        self.max_bars_in_trade = 20

        # ------------------------------------------------------------------
        #  Manual fallback ticks (input defaults preserved)
        # ------------------------------------------------------------------
        self.man_activation = 36
        self.man_offset = 12

        # ------------------------------------------------------------------
        #  Hard-coded constants from Pine
        # ------------------------------------------------------------------
        self.FLOAT_EPSILON = 0.0001
        self.MIN_ATR_MULT = 0.5
        self.MAX_ATR_MULT = 3.0
        self.MIN_ADAPTIVE_LEN = 6
        self.MAX_ADAPTIVE_LEN = 30
        self.VOLATILITY_FLOOR = 1.0
        self.EQUITY_CURVE_MIN_SAMPLES = 3
        self.MAX_EQUITY_HISTORY = 100
        self.er_len = 14
        self.z_score = 2.2

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_sma(arr: list, window: int):
        w = min(window, len(arr))
        if w <= 0:
            return float(arr[-1]) if arr else 0.0
        return float(np.mean(arr[-w:]))

    @staticmethod
    def _rolling_std(arr: list, window: int):
        w = min(window, len(arr))
        if w <= 1:
            return 0.0
        return float(np.std(arr[-w:], ddof=1))

    def _dmi_adx(self, df: pd.DataFrame, period: int = 14):
        """
        DI+, DI- and ADX via Wilder's smoothing (RMA).
        Matches Pine ``ta.dmi(14, 14)``.
        """
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]

        up = high_s.diff()
        down = -low_s.diff()

        plus_dm = pd.Series(
            np.where((up > down) & (up > 0), up, 0.0), index=df.index
        )
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0), index=df.index
        )

        prev_close = close_s.shift(1)
        tr1 = high_s - low_s
        tr2 = (high_s - prev_close).abs()
        tr3 = (low_s - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing  →  ewm(alpha=1/period, adjust=False)
        atr_w = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        plus_dm_w = plus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
        minus_dm_w = minus_dm.ewm(alpha=1.0 / period, adjust=False).mean()

        di_plus = 100.0 * plus_dm_w / atr_w
        di_minus = 100.0 * minus_dm_w / atr_w
        dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

        adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        di_p = float(di_plus.iloc[-1]) if not pd.isna(di_plus.iloc[-1]) else 0.0
        di_m = float(di_minus.iloc[-1]) if not pd.isna(di_minus.iloc[-1]) else 0.0
        return adx_val, di_p, di_m

    # ------------------------------------------------------------------
    #  main signal generator
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        equity_history: list = None,
    ) -> dict:
        """
        Generate trading signal for the *latest* bar in ``df``.

        Returns
        -------
        dict
            {
                "token": str,
                "signal": float (0..1),
                "direction": "BUY" | "SELL" | "NEUTRAL",
                "metadata": {
                    # trend state, EMA values, ADX, pin-bar flags,
                    # activation/offset, stop / TP levels, etc.
                }
            }
        """
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            return {
                "token": symbol,
                "signal": 0.0,
                "direction": "NEUTRAL",
                "metadata": {"error": f"Missing columns: {missing}"},
            }

        min_rows = max(self.ema_slow_len, self.atr_period) + 5
        if len(df) < min_rows:
            return {
                "token": symbol,
                "signal": 0.0,
                "direction": "NEUTRAL",
                "metadata": {"error": f"Insufficient rows ({len(df)} < {min_rows})"},
            }

        # -- latest bar values --
        close = float(df["close"].iloc[-1])
        open_ = float(df["open"].iloc[-1])
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        volume = float(df["volume"].iloc[-1])

        close_prev = float(df["close"].iloc[-2])
        high_prev = float(df["high"].iloc[-2])
        low_prev = float(df["low"].iloc[-2])

        # ------------------------------------------------------------------
        #  EQUITY TRACKING & ADAPTIVE LOGIC
        # ------------------------------------------------------------------
        if equity_history is None:
            equity_history = []
        equity_history = [
            float(e) for e in equity_history if e is not None and e > 0
        ]
        if len(equity_history) > self.MAX_EQUITY_HISTORY:
            equity_history = equity_history[-self.MAX_EQUITY_HISTORY :]

        current_closed_equity = (
            equity_history[-1] if equity_history else self.initial_capital
        )

        in_warmup = len(equity_history) < self.warmup_trades
        has_min_samples = len(equity_history) >= self.EQUITY_CURVE_MIN_SAMPLES

        avg_equity = current_closed_equity
        if has_min_samples and len(equity_history) > 0:
            avg_equity = np.mean(equity_history)
        if avg_equity <= self.FLOAT_EPSILON:
            avg_equity = current_closed_equity

        is_strategy_cold = (
            has_min_samples and not in_warmup and current_closed_equity < avg_equity
        )

        # Hyper-phase
        target_equity_amt = self.initial_capital * self.growth_target_x
        is_hyper_phase = current_closed_equity < target_equity_amt
        progress_pct = (
            (current_closed_equity / target_equity_amt * 100.0)
            if target_equity_amt > 0
            else 0.0
        )

        # OVERRIDE: Hyper Phase ignores cold state
        if is_hyper_phase:
            is_strategy_cold = False

        # ATR multiplier selection (matches Pine atr_mult_use)
        atr_mult_use = (
            self.atr_mult_input
            if in_warmup
            else (self.atr_mult_guard if is_strategy_cold else self.atr_mult_base)
        )

        # ------------------------------------------------------------------
        #  ADAPTIVE EQUITY COMPOUNDING CORE
        # ------------------------------------------------------------------
        change = 0.0
        volatility = self.VOLATILITY_FLOOR

        if has_min_samples and len(equity_history) > self.er_len + 1:
            prev_equity = equity_history[-self.er_len - 1]
            if prev_equity is not None and prev_equity > 0:
                change = abs(current_closed_equity - prev_equity)

            vol_sum = 0.0
            valid_count = 0
            for i in range(1, self.er_len + 1):
                if len(equity_history) > i + 1:
                    e1 = equity_history[-i]
                    e2 = equity_history[-i - 1]
                    if e1 is not None and e2 is not None:
                        vol_sum += abs(e1 - e2)
                        valid_count += 1
            volatility = (
                vol_sum
                if (valid_count > 0 and vol_sum > self.FLOAT_EPSILON)
                else self.VOLATILITY_FLOOR
            )
        else:
            change = 0.0
            volatility = self.VOLATILITY_FLOOR

        eff_ratio = (
            (change / volatility) if volatility > self.FLOAT_EPSILON else 0.0
        )
        eff_ratio = max(0.0, min(1.0, eff_ratio))

        len_adaptive = 8.0 + (30.0 - 8.0) * (1.0 - eff_ratio)
        len_adaptive = max(
            self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive)
        )
        len_rounded = int(round(len_adaptive))

        # 1) Adaptive SMA (Dynamic Memory Length)
        eq_sma = self._rolling_sma(equity_history, len_rounded)
        eq_sma_prev = (
            self._rolling_sma(equity_history[:-1], len_rounded)
            if len(equity_history) > 1
            else eq_sma
        )

        # 2) Logistic Multiplier (Confidence Curve)
        epsilon = 0.02
        dist_raw = (
            ((current_closed_equity - eq_sma) / eq_sma)
            if eq_sma > self.FLOAT_EPSILON
            else 0.0
        )
        dist = dist_raw if abs(dist_raw) > epsilon else 0.0
        confidence = 1.0 / (1.0 + np.exp(-4.0 * dist))
        base_mult = 1.0 + 0.8 * confidence
        base_mult = max(0.9, min(1.8, base_mult))

        # 3) Acceleration Filter (Smart Smoothing)
        vel = eq_sma - eq_sma_prev
        if len(equity_history) > 2:
            eq_sma_prev2 = self._rolling_sma(equity_history[:-2], len_rounded)
            vel_prev = eq_sma_prev - eq_sma_prev2
        else:
            vel_prev = vel

        acc = vel - vel_prev

        # EMA of acc over 5 — best-effort with available history
        acc_smooth = acc
        if len(equity_history) > 3:
            acc_series = []
            max_off = min(10, len(equity_history) - 2)
            for off in range(max_off):
                sm = (
                    self._rolling_sma(equity_history[: -(off + 1)], len_rounded)
                    if off + 1 < len(equity_history)
                    else eq_sma
                )
                sm_prev = (
                    self._rolling_sma(equity_history[: -(off + 2)], len_rounded)
                    if off + 2 < len(equity_history)
                    else sm
                )
                sm_prev2 = (
                    self._rolling_sma(equity_history[: -(off + 3)], len_rounded)
                    if off + 3 < len(equity_history)
                    else sm_prev
                )
                v = sm - sm_prev
                v_prev = sm_prev - sm_prev2
                acc_series.append(v - v_prev)
            if acc_series:
                acc_s = pd.Series(list(reversed(acc_series)))
                acc_smooth = float(
                    acc_s.ewm(alpha=1.0 / 5.0, adjust=False).mean().iloc[-1]
                )

        acc_impact = max(0.0, min(1.0, (-acc_smooth - 0.4) / 0.6))
        acc_adj = 1.0 - 0.1 * acc_impact

        # 4) Std-Dev Channel (Equity Envelope)
        eq_stdev = self._rolling_std(equity_history, len_rounded)
        eq_upper = eq_sma + self.z_score * eq_stdev
        eq_lower = eq_sma - self.z_score * eq_stdev

        chan_adj = 1.0
        if current_closed_equity < eq_lower:
            chan_adj = 0.95
        elif current_closed_equity > eq_upper:
            chan_adj = 1.03

        # 5) Final Adaptive Multiplier (Mode-Aware overrides)
        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(
            self.MIN_ADAPTIVE_LEN,
            min(self.MAX_ADAPTIVE_LEN, len_adaptive * len_factor),
        )

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        if current_closed_equity < eq_lower:
            chan_adj = 0.97 if is_hyper_phase else 0.95
        elif current_closed_equity > eq_upper:
            chan_adj = 1.05 if is_hyper_phase else 1.03
        acc_adj = (
            1.0 - 0.07 * acc_impact
            if is_hyper_phase
            else 1.0 - 0.1 * acc_impact
        )

        mult_adaptive = base_mult * acc_adj * chan_adj
        mult_adaptive = max(
            self.MIN_ATR_MULT, min(self.MAX_ATR_MULT, mult_adaptive)
        )

        atr_mult_use = mult_adaptive

        # ------------------------------------------------------------------
        #  INDICATORS (MODE-AWARE  →  scalp)
        # ------------------------------------------------------------------
        fast_ema_s = df["close"].ewm(span=self.ema_fast_len, adjust=False).mean()
        medm_ema_s = df["close"].ewm(span=self.ema_medm_len, adjust=False).mean()
        slow_sma_s = df["close"].rolling(window=self.ema_slow_len).mean()

        fast_ema = float(fast_ema_s.iloc[-1])
        medm_ema = float(medm_ema_s.iloc[-1])
        slow_sma = float(slow_sma_s.iloc[-1])

        # Guards against NA
        if pd.isna(fast_ema):
            fast_ema = close
        if pd.isna(medm_ema):
            medm_ema = close
        if pd.isna(slow_sma):
            slow_sma = close

        # ATR
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]
        prev_close_s = close_s.shift(1)
        tr1 = high_s - low_s
        tr2 = (high_s - prev_close_s).abs()
        tr3 = (low_s - prev_close_s).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_s = tr.rolling(window=self.atr_period).mean()
        atr = float(atr_s.iloc[-1])
        if pd.isna(atr) or atr <= self.FLOAT_EPSILON:
            atr = close * 0.01

        # EMA Fan Trend Detection
        fan_up_trend = (fast_ema > medm_ema) and (medm_ema > slow_sma)
        fan_dn_trend = (fast_ema < medm_ema) and (medm_ema < slow_sma)

        # DMI / ADX
        adx_value, di_plus, di_minus = self._dmi_adx(df, period=self.atr_period)

        momentum_thresh_final = (
            self.momentum_thresh if self.momentum_thresh > 0 else 28
        )
        is_strong_trend = adx_value > momentum_thresh_final

        # Pin Bar Detection (hardened)
        bar_range = high - low
        body = abs(close - open_)
        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low

        pin_bar_wick_ratio = 0.70   # scalp
        pin_bar_body_ratio = 0.30   # scalp

        bullish_pin_bar = False
        bearish_pin_bar = False
        if bar_range > self.FLOAT_EPSILON:
            bullish_pin_bar = (
                lower_wick >= pin_bar_wick_ratio * bar_range
                and body <= pin_bar_body_ratio * bar_range
            )
            bearish_pin_bar = (
                upper_wick >= pin_bar_wick_ratio * bar_range
                and body <= pin_bar_body_ratio * bar_range
            )

        # Valid Triggers (mode-aware)
        if is_hyper_phase and self.use_momentum:
            valid_trigger_bull = bullish_pin_bar or (
                is_strong_trend and close > high_prev
            )
            valid_trigger_bear = bearish_pin_bar or (
                is_strong_trend and close < low_prev
            )
        else:
            valid_trigger_bull = bullish_pin_bar
            valid_trigger_bear = bearish_pin_bar

        # Price Pierce Detection
        bull_pierce = (
            ((low < fast_ema) and (close > fast_ema))
            or ((low < medm_ema) and (close > medm_ema))
            or ((low < slow_sma) and (close > slow_sma))
        )
        bear_pierce = (
            ((high > fast_ema) and (close < fast_ema))
            or ((high > medm_ema) and (close < medm_ema))
            or ((high > slow_sma) and (close < slow_sma))
        )

        # Volume Confirmation
        volume_avg_s = df["volume"].rolling(window=self.volume_lookback).mean()
        volume_avg = float(volume_avg_s.iloc[-1])
        volume_confirmed = (
            volume >= (volume_avg * self.volume_multiplier)
            if not pd.isna(volume_avg)
            else False
        )

        # ------------------------------------------------------------------
        #  FINAL ENTRY SIGNALS
        # ------------------------------------------------------------------
        allow_long = self.trade_direction in ("Both", "Long Only")
        allow_short = self.trade_direction in ("Both", "Short Only")

        long_entry = fan_up_trend and bull_pierce and valid_trigger_bull
        short_entry = fan_dn_trend and bear_pierce and valid_trigger_bear

        if self.use_volume_confirm:
            long_entry = long_entry and volume_confirmed
            short_entry = short_entry and volume_confirmed

        # ------------------------------------------------------------------
        #  POSITION SIZING & LEVELS (for metadata)
        # ------------------------------------------------------------------
        atr_prev = (
            float(atr_s.iloc[-2])
            if len(atr_s) > 1 and not pd.isna(atr_s.iloc[-2])
            else atr
        )
        low_prev2 = float(df["low"].iloc[-2])
        high_prev2 = float(df["high"].iloc[-2])

        # Long levels
        stop_dist_long = atr_prev * atr_mult_use
        stop_price_long = low_prev2 - stop_dist_long
        risk_per_share_long = abs(close - stop_price_long)
        # approximate syminfo.mintick * 10  →  close * 0.001  (crypto tick-size proxy)
        risk_per_share_long = max(risk_per_share_long, close * 0.001)
        if risk_per_share_long <= self.FLOAT_EPSILON:
            risk_per_share_long = close * 0.001

        # Short levels
        stop_dist_short = atr_prev * atr_mult_use
        stop_price_short = high_prev2 + stop_dist_short
        risk_per_share_short = abs(close - stop_price_short)
        risk_per_share_short = max(risk_per_share_short, close * 0.001)
        if risk_per_share_short <= self.FLOAT_EPSILON:
            risk_per_share_short = close * 0.001

        available_equity = max(current_closed_equity, self.initial_capital)
        risk_amount = available_equity * (self.risk_per_trade_pct / 100.0)

        qty_risk_long = (
            risk_amount / risk_per_share_long if risk_per_share_long > 0 else 0
        )
        qty_risk_short = (
            risk_amount / risk_per_share_short if risk_per_share_short > 0 else 0
        )

        efficiency = 0.97
        buffer_mult = 0.0
        denom = close + atr_prev * buffer_mult
        qty_max = int(np.floor(available_equity * efficiency / denom)) if denom > 0 else 0

        final_qty_long = min(qty_risk_long, qty_max)
        final_qty_short = min(qty_risk_short, qty_max)

        # Take-profit levels
        tp_long = float(close + (atr_prev * self.tp_multiplier))
        tp_short = float(close - (atr_prev * self.tp_multiplier))

        # ------------------------------------------------------------------
        #  SIGNAL RESOLUTION
        # ------------------------------------------------------------------
        direction = "NEUTRAL"
        signal = 0.0
        if long_entry and allow_long:
            direction = "BUY"
            signal = 1.0
        elif short_entry and allow_short:
            direction = "SELL"
            signal = 1.0

        metadata = {
            "engine_mode": self.engine_mode,
            "risk_profile": "Scalp Aggressive (8/3)",
            "active_activation": self.active_activation,
            "active_offset": self.active_offset,
            "fast_ema": float(round(fast_ema, 8)),
            "medm_ema": float(round(medm_ema, 8)),
            "slow_sma": float(round(slow_sma, 8)),
            "atr": float(round(atr, 8)),
            "atr_mult_use": float(round(atr_mult_use, 6)),
            "mult_adaptive": float(round(mult_adaptive, 6)),
            "adx": float(round(adx_value, 4)),
            "di_plus": float(round(di_plus, 4)),
            "di_minus": float(round(di_minus, 4)),
            "is_strong_trend": is_strong_trend,
            "momentum_thresh_final": momentum_thresh_final,
            "fan_up_trend": fan_up_trend,
            "fan_dn_trend": fan_dn_trend,
            "bullish_pin_bar": bullish_pin_bar,
            "bearish_pin_bar": bearish_pin_bar,
            "valid_trigger_bull": valid_trigger_bull,
            "valid_trigger_bear": valid_trigger_bear,
            "bull_pierce": bull_pierce,
            "bear_pierce": bear_pierce,
            "volume_confirmed": volume_confirmed,
            "is_hyper_phase": is_hyper_phase,
            "progress_pct": float(round(progress_pct, 2)),
            "is_strategy_cold": is_strategy_cold,
            "in_warmup": in_warmup,
            "current_equity": float(round(current_closed_equity, 6)),
            "target_equity": float(round(target_equity_amt, 2)),
            "stop_loss_long": float(round(stop_price_long, 8)),
            "stop_loss_short": float(round(stop_price_short, 8)),
            "take_profit_long": float(round(tp_long, 8)),
            "take_profit_short": float(round(tp_short, 8)),
            "time_exit_bars": self.max_bars_in_trade if self.use_time_exit else None,
            "use_fixed_tp": self.use_fixed_tp,
            "qty_long": float(round(final_qty_long, 6)),
            "qty_short": float(round(final_qty_short, 6)),
            "len_adaptive": float(round(len_adaptive, 2)),
            "eff_ratio": float(round(eff_ratio, 6)),
            "base_mult": float(round(base_mult, 6)),
            "acc_adj": float(round(acc_adj, 6)),
            "chan_adj": float(round(chan_adj, 6)),
        }

        return {
            "token": symbol,
            "signal": signal,
            "direction": direction,
            "metadata": metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")

    base = 0.000012
    noise = np.random.normal(0, 0.000001, n)
    trend = np.cumsum(np.random.normal(0, 0.0000002, n))
    close = base + trend + noise
    close = np.maximum(close, base * 0.5)
    open_ = close + np.random.normal(0, 0.0000003, n)
    high = np.maximum(open_, close) + np.abs(np.random.exponential(0.0000005, n))
    low = np.minimum(open_, close) - np.abs(np.random.exponential(0.0000005, n))
    volume = np.random.uniform(1e6, 5e6, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )

    # equity curve with >3 samples so adaptive logic runs
    equity_hist = [
        100.0, 102.0, 101.0, 105.0, 108.0, 110.0, 115.0, 112.0, 118.0, 120.0,
        125.0, 130.0, 128.0, 135.0, 140.0, 138.0, 145.0, 150.0, 148.0, 155.0,
        160.0, 158.0, 165.0, 170.0, 168.0,
    ]

    engine = EngineV1_3Strategy()
    result = engine.generate_signals(df, symbol="PEPE-USD", equity_history=equity_hist)
    print(result)
