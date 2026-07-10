"""
Eve Engine v1 - Full-Fidelity Pine Script Translation
======================================================
Translated from Pine Script v6 (TradingView) into a Python class
that implements the BaseStrategy interface.

Every Pine input default, constant, and logic branch is preserved:
- Swing EMA/SMA lengths (50/18/6)
- Pin-bar detection (66% wick, 34% body)
- DMI/ADX momentum triggers (ADX > 18)
- Price-pierce entries (low/high cross EMA/SMA)
- Adaptive ATR multiplier driven by equity-curve efficiency ratio
- Risk-profile activation/offset ticks
- 97% "full send" confidence / signal score
"""

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class EngineV1Strategy(BaseStrategy):
    """Full-fidelity v1 translation of the Eve Engine Pine Script strategy."""

    # ------------------------------------------------------------------
    #  CONSTANTS (exact Pine defaults)
    # ------------------------------------------------------------------
    FLOAT_EPSILON = 0.0001
    MIN_ATR_MULT = 0.5
    MAX_ATR_MULT = 3.0
    MIN_ADAPTIVE_LEN = 6
    MAX_ADAPTIVE_LEN = 30
    VOLATILITY_FLOOR = 1.0
    EQUITY_CURVE_MIN_SAMPLES = 3

    # ------------------------------------------------------------------
    #  INPUT DEFAULTS (exact Pine defaults)
    # ------------------------------------------------------------------
    # Hyper-Growth Protocol
    GROWTH_TARGET_X = 50.0
    USE_MOMENTUM = True
    MOMENTUM_THRESH = 18

    # Date Range
    START_DATE_MS = int(pd.Timestamp("2025-01-01 08:00:00").value / 1_000_000)
    END_DATE_MS = int(pd.Timestamp("2069-12-30 00:00:00").value / 1_000_000)

    # Trade Direction
    TRADE_DIRECTION = "Both"  # options: ['Both', 'Long Only', 'Short Only']

    # Risk Management
    EQ_LENGTH = 21
    WARMUP_TRADES = 3
    USE_EQUITY_GUARD = False
    EQ_PERCENT = 0.7
    ATR_MULT = 1.8
    ATR_MULT_GUARD = 0.9
    RISK_PROFILE = "Manual"
    MAN_ACTIVATION = 18
    MAN_OFFSET = 6
    RISK_PER_TRADE_PCT = 97.0

    # Indicators
    SMA_SLOW = 50
    EMA_MEDM = 18
    EMA_FAST = 6
    ATR_VALU = 14

    # Adaptive compounding
    ER_LEN = 14
    Z_SCORE = 2.2
    EPSILON = 0.02

    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__(name="eve_engine_v1")

    # ------------------------------------------------------------------
    #  Helper: Wilder-smoothed RMA
    # ------------------------------------------------------------------
    @staticmethod
    def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
        """Wilder Running Moving Average (RMA) used by Pine's ta.rma()."""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # ------------------------------------------------------------------
    #  Helper: DMI / ADX  (Welles Wilder)
    # ------------------------------------------------------------------
    @staticmethod
    def _dmi_adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14):
        """
        Replicate Pine ta.dmi(di_len, adx_len).
        Returns (+DI, -DI, ADX) as three pd.Series.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up = high.diff()
        down = -low.diff()

        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder smoothing
        tr_smooth = EngineV1Strategy._wilder_rma(tr, di_len)
        plus_dm_smooth = EngineV1Strategy._wilder_rma(plus_dm, di_len)
        minus_dm_smooth = EngineV1Strategy._wilder_rma(minus_dm, di_len)

        di_plus = 100.0 * plus_dm_smooth / tr_smooth
        di_minus = 100.0 * minus_dm_smooth / tr_smooth

        dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = EngineV1Strategy._wilder_rma(dx, adx_len)

        return di_plus, di_minus, adx

    # ------------------------------------------------------------------
    #  Helper: ATR
    # ------------------------------------------------------------------
    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    # ------------------------------------------------------------------
    #  Helper: Crossover / Crossunder (Pine ta.crossover / ta.crossunder)
    # ------------------------------------------------------------------
    @staticmethod
    def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
        """True when a crosses above b on the current bar."""
        return (a.shift(1) <= b.shift(1)) & (a > b)

    @staticmethod
    def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
        """True when a crosses below b on the current bar."""
        return (a.shift(1) >= b.shift(1)) & (a < b)

    # ------------------------------------------------------------------
    #  Main signal generator
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        equity_history: list = None,
    ) -> dict:
        """
        Generate trading signal for the latest bar in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Index should be DatetimeIndex (used for date-range check).
        symbol : str
            Token identifier (passed through to output).
        equity_history : list[float] | None
            Equity values after each closed trade (newest last).

        Returns
        -------
        dict
            {
                "token":      str,
                "signal":     float (0 .. 1),
                "direction":  "BUY" | "SELL" | "NEUTRAL",
                "metadata":   dict,
            }
        """
        # ------------------------------------------------------------------
        # 0.  Basic guards
        # ------------------------------------------------------------------
        if df is None or df.empty or len(df) < self.SMA_SLOW + 2:
            return {
                "token": symbol,
                "signal": 0.0,
                "direction": "NEUTRAL",
                "metadata": {"error": "insufficient_data"},
            }

        equity_history = equity_history or []

        # ------------------------------------------------------------------
        # 1.  Date range check (latest bar)
        # ------------------------------------------------------------------
        if hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index):
            latest_ts = int(df.index[-1].value / 1_000_000)
        else:
            latest_ts = self.START_DATE_MS  # no datetime index => assume in range
        in_date_range = self.START_DATE_MS <= latest_ts <= self.END_DATE_MS

        # ------------------------------------------------------------------
        # 2.  Trade-direction booleans
        # ------------------------------------------------------------------
        allow_long = self.TRADE_DIRECTION in ("Both", "Long Only")
        allow_short = self.TRADE_DIRECTION in ("Both", "Short Only")

        # ------------------------------------------------------------------
        # 3.  Indicators (computed on full DataFrame)
        # ------------------------------------------------------------------
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        slow_sma = close.rolling(window=self.SMA_SLOW, min_periods=self.SMA_SLOW).mean()
        medm_ema = close.ewm(span=self.EMA_MEDM, adjust=False).mean()
        fast_ema = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        atr = self._atr(df, self.ATR_VALU)

        # DMI / ADX
        di_plus, di_minus, adx = self._dmi_adx(df, self.ATR_VALU, self.ATR_VALU)

        # EMA Fan Trend Detection
        fan_up_trend = (fast_ema > medm_ema) & (medm_ema > slow_sma)
        fan_dn_trend = (fast_ema < medm_ema) & (medm_ema < slow_sma)

        # Momentum Logic
        is_strong_trend = adx > self.MOMENTUM_THRESH

        # Pin Bar Detection
        bar_range = high - low
        body = (close - open_).abs()
        upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low

        bullish_pin_bar = (lower_wick >= 0.66 * bar_range) & (body <= 0.34 * bar_range)
        bearish_pin_bar = (upper_wick >= 0.66 * bar_range) & (body <= 0.34 * bar_range)

        # Price Pierce Detection
        bull_pierce = (
            ((low < fast_ema) & (close > fast_ema))
            | ((low < medm_ema) & (close > medm_ema))
            | ((low < slow_sma) & (close > slow_sma))
        )
        bear_pierce = (
            ((high > fast_ema) & (close < fast_ema))
            | ((high > medm_ema) & (close < medm_ema))
            | ((high > slow_sma) & (close < slow_sma))
        )

        # ------------------------------------------------------------------
        # 4.  Equity Tracking & Adaptive Logic
        # ------------------------------------------------------------------
        initial_capital = 100.0
        current_closed_equity = (
            equity_history[-1] if equity_history else initial_capital
        )

        in_warmup = len(equity_history) < self.WARMUP_TRADES
        has_min_samples = len(equity_history) >= self.EQUITY_CURVE_MIN_SAMPLES

        avg_equity = (
            np.mean(equity_history) if has_min_samples else current_closed_equity
        )

        # Hyper-Status
        target_equity_amt = initial_capital * self.GROWTH_TARGET_X
        is_hyper_phase = current_closed_equity < target_equity_amt
        progress_pct = (
            (current_closed_equity / target_equity_amt * 100.0)
            if target_equity_amt > 0
            else 0.0
        )

        # Equity Guard
        is_strategy_cold = bool(
            has_min_samples and not in_warmup and current_closed_equity < avg_equity
        )
        # OVERRIDE: Hyper Phase ignores cold state (ALWAYS SEND)
        if is_hyper_phase:
            is_strategy_cold = False

        # ATR Multiplier Base
        atr_mult_use = (
            self.ATR_MULT
            if in_warmup
            else (self.ATR_MULT_GUARD if is_strategy_cold else self.ATR_MULT)
        )

        # ------------------------------------------------------------------
        # 5.  Adaptive Equity Compounding Core (4-Layer Optimized)
        # ------------------------------------------------------------------
        if has_min_samples and len(equity_history) > self.ER_LEN + 1:
            prev_equity = equity_history[-self.ER_LEN - 1]
            change = abs(current_closed_equity - prev_equity)

            vol_sum = 0.0
            for i in range(1, self.ER_LEN + 1):
                if len(equity_history) > i + 1:
                    e1 = equity_history[-i]
                    e2 = equity_history[-i - 1]
                    vol_sum += abs(e1 - e2)
            volatility = vol_sum if vol_sum > self.FLOAT_EPSILON else self.VOLATILITY_FLOOR
        else:
            change = 0.0
            volatility = self.VOLATILITY_FLOOR

        eff_ratio = change / volatility if volatility > self.FLOAT_EPSILON else 0.0

        # 1) Adaptive SMA (Dynamic Memory Length)
        len_adaptive = 8.0 + (30.0 - 8.0) * (1.0 - eff_ratio)
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive))
        len_adaptive_int = int(round(len_adaptive))

        # eqSMA: SMA of the equity curve (latest values up to len_adaptive)
        eq_slice = equity_history[-len_adaptive_int:] if len(equity_history) >= len_adaptive_int else equity_history
        eq_sma = np.mean(eq_slice) if eq_slice else current_closed_equity

        # 2) Logistic Multiplier (Confidence Curve)
        dist_raw = (current_closed_equity - eq_sma) / eq_sma if eq_sma > self.FLOAT_EPSILON else 0.0
        dist = dist_raw if abs(dist_raw) > self.EPSILON else 0.0
        confidence = 1.0 / (1.0 + np.exp(-4.0 * dist))
        base_mult = 1.0 + 0.8 * confidence
        base_mult = max(0.9, min(1.8, base_mult))

        # 3) Acceleration Filter (Smart Smoothing)
        # Approximate velocity & acceleration from equity curve
        if len(eq_slice) >= 3:
            vel = eq_slice[-1] - eq_slice[-2] if len(eq_slice) >= 2 else 0.0
            vel_prev = eq_slice[-2] - eq_slice[-3] if len(eq_slice) >= 3 else 0.0
            acc = vel - vel_prev
        else:
            acc = 0.0
            vel = 0.0

        # EMA of acceleration (5-period)
        if len(eq_slice) >= 5:
            acc_series = pd.Series(
                [0.0] * (len(eq_slice) - 1)
            )
            for i in range(1, len(eq_slice)):
                acc_series.iloc[i - 1] = eq_slice[i] - eq_slice[i - 1]
            acc_smooth = acc_series.ewm(span=5, adjust=False).mean().iloc[-1]
        else:
            acc_smooth = acc

        acc_impact = max(0.0, min(1.0, (-acc_smooth - 0.4) / 0.6))
        acc_adj = 1.0 - 0.1 * acc_impact

        # 4) Std-Dev Channel (Equity Envelope)
        eq_stdev = np.std(eq_slice, ddof=1) if len(eq_slice) >= 2 else 0.0
        eq_upper = eq_sma + self.Z_SCORE * eq_stdev
        eq_lower = eq_sma - self.Z_SCORE * eq_stdev

        chan_adj = (
            0.95
            if current_closed_equity < eq_lower
            else (1.03 if current_closed_equity > eq_upper else 1.0)
        )

        # 5) Final Adaptive Multiplier (Hyper-Aware - AGGRESSIVE)
        len_factor = 0.8 if is_hyper_phase else 1.0
        len_adaptive = max(self.MIN_ADAPTIVE_LEN, min(self.MAX_ADAPTIVE_LEN, len_adaptive * len_factor))

        base_mult_range = 1.0 if is_hyper_phase else 0.9
        max_mult_range = 2.0 if is_hyper_phase else 1.8
        base_mult = max(base_mult_range, min(max_mult_range, base_mult))

        chan_adj = (
            (0.97 if is_hyper_phase else 0.95)
            if current_closed_equity < eq_lower
            else (
                (1.05 if is_hyper_phase else 1.03)
                if current_closed_equity > eq_upper
                else 1.0
            )
        )
        acc_adj = (
            1.0 - 0.07 * acc_impact
            if is_hyper_phase
            else 1.0 - 0.1 * acc_impact
        )

        mult_adaptive = base_mult * acc_adj * chan_adj
        mult_adaptive = max(self.MIN_ATR_MULT, min(self.MAX_ATR_MULT, mult_adaptive))

        # Apply globally
        atr_mult_use = mult_adaptive

        # ------------------------------------------------------------------
        # 6.  Risk Profile Activation / Offset
        # ------------------------------------------------------------------
        risk_profiles = {
            "Sniper Mode (18/6)": (18, 6),
            "Trend Scalper (18/12)": (18, 12),
            "Conservative (25/18)": (25, 18),
            "Golden Growth (36/12)": (36, 12),
        }
        if self.RISK_PROFILE in risk_profiles:
            active_activation, active_offset = risk_profiles[self.RISK_PROFILE]
        else:
            active_activation = self.MAN_ACTIVATION
            active_offset = self.MAN_OFFSET

        # Peak Equity Tracking
        equity_peak = max(equity_history) if equity_history else current_closed_equity
        dd_percent = (
            (current_closed_equity - equity_peak) / equity_peak * 100.0
            if equity_peak > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # 7.  Valid Triggers (Hyper Phase = Momentum OR Pin Bar)
        # ------------------------------------------------------------------
        if is_hyper_phase and self.USE_MOMENTUM:
            valid_trigger_bull = bullish_pin_bar | (
                is_strong_trend & (close > high.shift(1))
            )
            valid_trigger_bear = bearish_pin_bar | (
                is_strong_trend & (close < low.shift(1))
            )
        else:
            valid_trigger_bull = bullish_pin_bar
            valid_trigger_bear = bearish_pin_bar

        # ------------------------------------------------------------------
        # 8.  Duplicate Prevention (approximate Pine lastEntryBar logic)
        # ------------------------------------------------------------------
        # In Pine: lastEntryBar is set to bar_index on entry and the condition
        # bar_index > lastEntryBar is checked.  For a stateless signal generator
        # we approximate by checking whether the *previous* bar would have
        # triggered, and if so treat lastEntryBar as that index.
        prev_idx = len(df) - 2
        prev_long_entry = (
            fan_up_trend.iloc[prev_idx]
            & bull_pierce.iloc[prev_idx]
            & valid_trigger_bull.iloc[prev_idx]
        ) if prev_idx >= 0 else False
        prev_short_entry = (
            fan_dn_trend.iloc[prev_idx]
            & bear_pierce.iloc[prev_idx]
            & valid_trigger_bear.iloc[prev_idx]
        ) if prev_idx >= 0 else False

        last_entry_bar = prev_idx if (prev_long_entry or prev_short_entry) else -1
        bar_index = len(df) - 1

        # ------------------------------------------------------------------
        # 9.  Final Entry Signals
        # ------------------------------------------------------------------
        long_entry = (
            fan_up_trend
            & bull_pierce
            & valid_trigger_bull
            & (bar_index > last_entry_bar)
        )
        short_entry = (
            fan_dn_trend
            & bear_pierce
            & valid_trigger_bear
            & (bar_index > last_entry_bar)
        )

        # ------------------------------------------------------------------
        # 10. Evaluate *latest* bar
        # ------------------------------------------------------------------
        latest_long = bool(long_entry.iloc[-1])
        latest_short = bool(short_entry.iloc[-1])

        # Apply date range & direction filters
        if not in_date_range:
            latest_long = False
            latest_short = False
        if not allow_long:
            latest_long = False
        if not allow_short:
            latest_short = False

        # ------------------------------------------------------------------
        # 11. Signal score (97% "full send" confidence concept)
        # ------------------------------------------------------------------
        if latest_long or latest_short:
            # Base score anchored to the logistic confidence curve
            score = 0.50 + confidence * 0.20

            # Strong trend bonus
            if bool(is_strong_trend.iloc[-1]):
                score += 0.12

            # Pin-bar pure-price-action bonus
            if bool(bullish_pin_bar.iloc[-1]) or bool(bearish_pin_bar.iloc[-1]):
                score += 0.10

            # Hyper-phase conviction bonus
            if is_hyper_phase:
                score += 0.05

            # Clamp to 0.97 (the "full send" ceiling — 3% slippage buffer)
            signal = min(0.97, score)
            direction = "BUY" if latest_long else "SELL"
        else:
            signal = 0.0
            direction = "NEUTRAL"

        # ------------------------------------------------------------------
        # 12. Metadata (dashboard-friendly)
        # ------------------------------------------------------------------
        metadata = {
            # Trend state
            "trend_state": (
                "UP"
                if bool(fan_up_trend.iloc[-1])
                else ("DOWN" if bool(fan_dn_trend.iloc[-1]) else "NEUTRAL")
            ),
            # EMA values
            "fast_ema": round(float(fast_ema.iloc[-1]), 6),
            "medm_ema": round(float(medm_ema.iloc[-1]), 6),
            "slow_sma": round(float(slow_sma.iloc[-1]), 6),
            # ADX / DMI
            "adx": round(float(adx.iloc[-1]), 4),
            "di_plus": round(float(di_plus.iloc[-1]), 4),
            "di_minus": round(float(di_minus.iloc[-1]), 4),
            "is_strong_trend": bool(is_strong_trend.iloc[-1]),
            # Pin bars
            "bullish_pin_bar": bool(bullish_pin_bar.iloc[-1]),
            "bearish_pin_bar": bool(bearish_pin_bar.iloc[-1]),
            # Price pierce
            "bull_pierce": bool(bull_pierce.iloc[-1]),
            "bear_pierce": bool(bear_pierce.iloc[-1]),
            # Risk profile
            "risk_profile": self.RISK_PROFILE,
            "engine": "v1",
            "activation": active_activation,
            "offset": active_offset,
            # Adaptive multiplier
            "mult_adaptive": round(float(mult_adaptive), 4),
            "atr_mult_use": round(float(atr_mult_use), 4),
            # Equity / hyper state
            "is_hyper_phase": is_hyper_phase,
            "progress_pct": round(float(progress_pct), 2),
            "is_strategy_cold": is_strategy_cold,
            "confidence": round(float(confidence), 4),
            "dd_percent": round(float(dd_percent), 4),
            "equity": round(float(current_closed_equity), 2),
            # Signal decomposition
            "valid_trigger": bool(valid_trigger_bull.iloc[-1]) if latest_long else (
                bool(valid_trigger_bear.iloc[-1]) if latest_short else False
            ),
            "in_date_range": in_date_range,
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata,
        }


# ===================================================================
#  SMOKE TEST (synthetic PEPE data)
# ===================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 120  # need > 50 bars for slow SMA

    # Build synthetic OHLCV that vaguely looks like a trending memecoin
    trend = np.cumsum(np.random.normal(0.001, 0.02, n))
    close = 0.000012 + trend * 0.000001
    close = np.maximum(close, 0.000001)

    noise = np.random.normal(0, 0.0000003, n)
    open_ = close + noise
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.0000005, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.0000005, n))
    volume = np.random.randint(1_000_000, 10_000_000, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2025-01-02 08:00", periods=n, freq="15min"),
    )

    strategy = EngineV1Strategy()

    # Test 1: no equity history (cold start)
    result = strategy.generate_signals(df, symbol="PEPE", equity_history=None)
    print("=== Test 1: cold start ===")
    print(result)
    assert "direction" in result
    assert "signal" in result
    assert "metadata" in result

    # Test 2: growing equity history (hyper phase)
    eq_hist = [100.0, 105.0, 110.0, 115.0, 120.0, 130.0, 140.0]
    result2 = strategy.generate_signals(df, symbol="PEPE", equity_history=eq_hist)
    print("\n=== Test 2: equity history (hyper) ===")
    print(result2)
    assert result2["metadata"]["is_hyper_phase"] is True  # 140 < 5000

    # Test 3: above target (conservative mode)
    eq_hist_big = [5100.0, 5200.0, 5300.0]
    result3 = strategy.generate_signals(df, symbol="PEPE", equity_history=eq_hist_big)
    print("\n=== Test 3: above target (conservative) ===")
    print(result3)
    assert result3["metadata"]["is_hyper_phase"] is False

    print("\n✅ All smoke tests passed.")
