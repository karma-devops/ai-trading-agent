"""
Engine v6.1 Strategy
====================
Pine-to-Python translation of Magic Trend v6 PRO.
Runs best on PEPE and FARTCOIN due to their volatility and meme momentum.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class EngineV6_1Strategy(BaseStrategy):
    name = "Engine v6.1"
    description = "Trend-follower with pin-bar/momentum triggers. Best on PEPE and FARTCOIN."

    # Default parameters matching the Pine script
    SLOW_SMA = 50
    MEDM_EMA = 18
    FAST_EMA = 6
    ATR_PERIOD = 14
    ATR_MULT = 1.8
    MOMENTUM_THRESH = 18  # ADX threshold

    def __init__(self):
        super().__init__(self.name)

    def generate_signals(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Generate a trading signal from OHLCV data.

        Returns {
            'token': symbol,
            'signal': 0-1 confidence,
            'direction': 'BUY' | 'SELL' | 'NEUTRAL',
            'metadata': dict
        }
        """
        if df is None or len(df) < self.SLOW_SMA + 5:
            return {"token": symbol, "signal": 0.0, "direction": "NEUTRAL", "metadata": {"error": "insufficient data"}}

        # Indicators
        df = df.copy()
        df["slow_sma"] = df["close"].rolling(self.SLOW_SMA).mean()
        df["medm_ema"] = df["close"].ewm(span=self.MEDM_EMA, adjust=False).mean()
        df["fast_ema"] = df["close"].ewm(span=self.FAST_EMA, adjust=False).mean()
        df["atr"] = self._atr(df, self.ATR_PERIOD)

        # DMI / ADX approximation
        df["adx"] = self._adx(df, 14)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Trend fan
        fan_up = latest["fast_ema"] > latest["medm_ema"] > latest["slow_sma"]
        fan_down = latest["fast_ema"] < latest["medm_ema"] < latest["slow_sma"]

        # Pin bars
        body = abs(latest["close"] - latest["open"])
        range_ = latest["high"] - latest["low"]
        lower_wick = min(latest["open"], latest["close"]) - latest["low"]
        upper_wick = latest["high"] - max(latest["open"], latest["close"])

        bullish_pin = (range_ > 0) and (lower_wick > 0.66 * range_)
        bearish_pin = (range_ > 0) and (upper_wick > 0.66 * range_)

        # Momentum
        strong_trend = latest["adx"] > self.MOMENTUM_THRESH

        # Pierce
        bull_pierce = (
            (latest["low"] < latest["fast_ema"] and latest["open"] > latest["fast_ema"] and latest["close"] > latest["fast_ema"]) or
            (latest["low"] < latest["medm_ema"] and latest["open"] > latest["medm_ema"] and latest["close"] > latest["medm_ema"]) or
            (latest["low"] < latest["slow_sma"] and latest["open"] > latest["slow_sma"] and latest["close"] > latest["slow_sma"])
        )
        bear_pierce = (
            (latest["high"] > latest["fast_ema"] and latest["open"] < latest["fast_ema"] and latest["close"] < latest["fast_ema"]) or
            (latest["high"] > latest["medm_ema"] and latest["open"] < latest["medm_ema"] and latest["close"] < latest["medm_ema"]) or
            (latest["high"] > latest["slow_sma"] and latest["open"] < latest["slow_sma"] and latest["close"] < latest["slow_sma"])
        )

        # Entries
        long_entry = fan_up and bull_pierce and bullish_pin
        short_entry = fan_down and bear_pierce and bearish_pin

        # If momentum mode is enabled (default), allow ADX + close break
        if not long_entry and fan_up and strong_trend and latest["close"] > prev["high"]:
            long_entry = True
        if not short_entry and fan_down and strong_trend and latest["close"] < prev["low"]:
            short_entry = True

        direction = "NEUTRAL"
        signal = 0.0
        if long_entry:
            direction = "BUY"
            signal = min(1.0, latest["adx"] / 50.0 + 0.5)
        elif short_entry:
            direction = "SELL"
            signal = min(1.0, latest["adx"] / 50.0 + 0.5)

        metadata = {
            "fast_ema": round(float(latest["fast_ema"]), 6),
            "medm_ema": round(float(latest["medm_ema"]), 6),
            "slow_sma": round(float(latest["slow_sma"]), 6),
            "atr": round(float(latest["atr"]), 6),
            "adx": round(float(latest["adx"]), 4),
            "fan_up": bool(fan_up),
            "fan_down": bool(fan_down),
            "bullish_pin": bool(bullish_pin),
            "bearish_pin": bool(bearish_pin),
        }

        return {
            "token": symbol,
            "signal": round(float(signal), 4),
            "direction": direction,
            "metadata": metadata
        }

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0.0
        minus_dm[minus_dm < 0] = 0.0

        tr = EngineV6_1Strategy._atr(df, period) * period
        atr = tr.rolling(period).mean()
        atr = atr.replace(0, np.nan)

        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.replace([np.inf, -np.inf], np.nan)
        adx = dx.rolling(period).mean()
        return adx.fillna(0)
