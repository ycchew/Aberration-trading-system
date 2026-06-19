"""
Aberration trading strategy implementation.

Based on Keith Fitschen's Aberration system (1986):
- Bollinger-band-style breakout with volatility filter
- Long-only or long-short mode
- Signal generated at close of day t, applied at open of day t+1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AberrationParams:
    """Parameters for the Aberration trading strategy."""

    period: int = 80
    std_mult_up: float = 2.0
    std_mult_dn: float = 2.0
    long_only: bool = True
    use_vol_filter: bool = True
    atr_period: int = 20
    vol_lookback: int = 252
    min_vol_ratio: float = 0.5
    max_vol_ratio: float = 2.0
    transaction_cost: float = 0.001


class AberrationStrategy:
    """
    Implements the Aberration breakout strategy.

    Bands
    -----
    Middle = SMA(close, period)
    Upper  = Middle + std_mult_up x StdDev(close, period)
    Lower  = Middle - std_mult_dn x StdDev(close, period)

    Signal Rules (long-only by default)
    ------------------------------------
    Long entry  : close > upper AND vol_filter passes
    Long exit   : close < middle
    Short entry : close < lower AND vol_filter passes  (long_only=False only)
    Short exit  : close > middle                       (long_only=False only)

    Execution
    ---------
    Signal on close of day t -> position applied at open of day t+1
    (implemented via .shift(1) on strategy returns)
    """

    def __init__(self, params: Optional[AberrationParams] = None) -> None:
        self.params = params or AberrationParams()

    def calculate_bands(
        self, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return (upper, mid, lower) Bollinger-style bands."""
        p = self.params
        mid = close.rolling(window=p.period, min_periods=p.period).mean()
        std = close.rolling(window=p.period, min_periods=p.period).std(ddof=1)
        upper = mid + p.std_mult_up * std
        lower = mid - p.std_mult_dn * std
        return upper, mid, lower

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """ATR = SMA(TrueRange, atr_period)."""
        p = self.params
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).rename("hl"),
                (high - prev_close).abs().rename("hpc"),
                (low - prev_close).abs().rename("lpc"),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=p.atr_period, min_periods=p.atr_period).mean()

    def volatility_filter(self, atr: pd.Series) -> pd.Series:
        """True where ATR is within [min_ratio, max_ratio] x its 252-day median."""
        p = self.params
        med = atr.rolling(window=p.vol_lookback, min_periods=p.vol_lookback // 2).median()
        ratio = atr / med
        return (ratio >= p.min_vol_ratio) & (ratio <= p.max_vol_ratio)

    def generate_signals(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Bar-by-bar state machine: returns pd.Series of {-1, 0, 1}.
        Value = position held at close of that bar (applied next bar).
        """
        p = self.params
        upper, mid, lower = self.calculate_bands(close)

        if p.use_vol_filter and high is not None and low is not None:
            atr = self.calculate_atr(high, low, close)
            vol_ok = self.volatility_filter(atr)
        else:
            vol_ok = pd.Series(True, index=close.index)

        close_arr = close.values
        upper_arr = upper.values
        mid_arr   = mid.values
        lower_arr = lower.values
        vol_arr   = vol_ok.values
        n = len(close_arr)

        signals = np.zeros(n, dtype=float)
        pos = 0  # 0=flat, 1=long, -1=short

        for i in range(n):
            c   = close_arr[i]
            u   = upper_arr[i]
            m   = mid_arr[i]
            lo  = lower_arr[i]
            vok = bool(vol_arr[i])

            if np.isnan(c) or np.isnan(u) or np.isnan(m) or np.isnan(lo):
                pos = 0
                signals[i] = 0
                continue

            if pos == 0:
                if c > u and vok:
                    pos = 1
                elif not p.long_only and c < lo and vok:
                    pos = -1

            elif pos == 1:
                if c < m:
                    pos = 0
                    if not p.long_only and c < lo and vok:
                        pos = -1

            else:  # pos == -1
                if c > m:
                    pos = 0
                    if c > u and vok:
                        pos = 1

            signals[i] = pos

        return pd.Series(signals, index=close.index, name="signal")

    def compute_returns(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Daily strategy returns after transaction costs.

        return[t] = signal[t-1] x price_return[t]  -  cost[t]
        cost applied on each bar where position changes.
        """
        p = self.params
        signals       = self.generate_signals(close, high=high, low=low)
        daily_returns = close.pct_change()
        strategy_returns = signals.shift(1) * daily_returns

        position_changes = signals.diff().abs()
        position_changes.iloc[0] = abs(signals.iloc[0]) if not pd.isna(signals.iloc[0]) else 0.0
        costs = position_changes * p.transaction_cost
        strategy_returns = strategy_returns - costs.shift(1)
        return strategy_returns.fillna(0.0)
