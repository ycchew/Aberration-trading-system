"""
Aberration trading strategy implementation.

Based on Keith Fitschen's Aberration system (1986):
- Bollinger-band-style breakout with volatility filter
- Long-only or long-short mode
- Signal generated at close of day t, filled at market-on-close (same bar)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AberrationParams:
    """Parameters for the Aberration trading strategy."""

    # Band calculation
    period: int = 80          # SMA / StdDev lookback (N-bar middle band)
    std_mult_up: float = 2.0  # Upper band multiplier
    std_mult_dn: float = 2.0  # Lower band multiplier

    # Mode
    long_only: bool = True    # If False, allows short positions

    # Volatility filter
    use_vol_filter: bool = True
    atr_period: int = 20          # ATR calculation period
    vol_lookback: int = 252       # Rolling window for ATR median
    min_vol_ratio: float = 0.5    # ATR must be >= min_vol_ratio × median
    max_vol_ratio: float = 2.0    # ATR must be <= max_vol_ratio × median

    # Execution
    transaction_cost: float = 0.001  # 0.1% per trade (one-way)


class AberrationStrategy:
    """
    Implements the Aberration breakout strategy.

    Bands
    -----
    Middle = SMA(close, period)
    Upper  = Middle + std_mult_up × StdDev(close, period)
    Lower  = Middle - std_mult_dn × StdDev(close, period)

    Entry/Exit rules (long-only by default)
    ----------------------------------------
    Long entry  : close > upper AND vol_filter passes
    Long exit   : close < middle
    Short entry : close < lower AND vol_filter passes  (long_only=False only)
    Short exit  : close > middle                       (long_only=False only)

    Execution
    ---------
    Signal on close of day t → market-on-close fill at same day t's close.
    Implemented by shifting signals one bar: lagged_signal[t] = signal[t-1],
    so position earns the close[t-1]→close[t] return — i.e. the trade fills
    at the closing price of the signal bar (MOC order).
    """

    def __init__(self, params: Optional[AberrationParams] = None) -> None:
        self.params = params or AberrationParams()

    # ------------------------------------------------------------------
    # Band calculation
    # ------------------------------------------------------------------

    def calculate_bands(
        self, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute upper, middle, and lower Aberration bands.

        Parameters
        ----------
        close : pd.Series
            Daily closing prices.

        Returns
        -------
        upper, mid, lower : tuple of pd.Series
        """
        p = self.params
        mid = close.rolling(window=p.period, min_periods=p.period).mean()
        std = close.rolling(window=p.period, min_periods=p.period).std(ddof=1)
        upper = mid + p.std_mult_up * std
        lower = mid - p.std_mult_dn * std
        return upper, mid, lower

    # ------------------------------------------------------------------
    # Volatility (ATR) filter
    # ------------------------------------------------------------------

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        True Range-based ATR using an SMA smoother.

        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = SMA(TR, atr_period)

        Parameters
        ----------
        high, low, close : pd.Series  (same index)

        Returns
        -------
        atr : pd.Series
        """
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
        atr = tr.rolling(window=p.atr_period, min_periods=p.atr_period).mean()
        return atr

    def volatility_filter(self, atr: pd.Series) -> pd.Series:
        """
        Return boolean Series: True where the 20-day ATR sits between
        min_vol_ratio × median_252 and max_vol_ratio × median_252.

        Parameters
        ----------
        atr : pd.Series  (output of calculate_atr)

        Returns
        -------
        pd.Series[bool]
        """
        p = self.params
        med = atr.rolling(window=p.vol_lookback, min_periods=p.vol_lookback // 2).median()
        ratio = atr / med
        return (ratio >= p.min_vol_ratio) & (ratio <= p.max_vol_ratio)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate position signals using a bar-by-bar state machine.

        Returns
        -------
        signals : pd.Series of {-1, 0, 1} (float)
            Value represents position held at *close* of that day
            (applied at next open via .shift(1) in compute_returns).
        """
        p = self.params

        # --- bands ---
        upper, mid, lower = self.calculate_bands(close)

        # --- volatility filter ---
        if p.use_vol_filter and high is not None and low is not None:
            atr = self.calculate_atr(high, low, close)
            vol_ok = self.volatility_filter(atr)
        else:
            # If no OHLC data or filter disabled, always pass
            vol_ok = pd.Series(True, index=close.index)

        # Convert to numpy for speed in the state-machine loop
        close_arr = close.values
        upper_arr = upper.values
        mid_arr = mid.values
        lower_arr = lower.values
        vol_arr = vol_ok.values
        n = len(close_arr)

        signals = np.zeros(n, dtype=float)
        pos = 0  # current position: 0=flat, 1=long, -1=short

        for i in range(n):
            c = close_arr[i]
            u = upper_arr[i]
            m = mid_arr[i]
            lo = lower_arr[i]
            vok = bool(vol_arr[i])

            # If any band value is NaN → no signal, reset
            if np.isnan(c) or np.isnan(u) or np.isnan(m) or np.isnan(lo):
                pos = 0
                signals[i] = 0
                continue

            if pos == 0:
                # Flat: check entries
                if c > u and vok:
                    pos = 1
                elif not p.long_only and c < lo and vok:
                    pos = -1
                signals[i] = pos

            elif pos == 1:
                # Long: check exit
                if c < m:
                    pos = 0
                    # Immediate short re-entry?
                    if not p.long_only and c < lo and vok:
                        pos = -1
                # else: stay long (no re-entry on breakout needed while in pos)
                signals[i] = pos

            else:  # pos == -1
                # Short: check exit
                if c > m:
                    pos = 0
                    # Immediate long re-entry?
                    if c > u and vok:
                        pos = 1
                signals[i] = pos

        return pd.Series(signals, index=close.index, name="signal")

    # ------------------------------------------------------------------
    # Returns calculation
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute daily strategy returns.

        Market-on-close execution: signal[t] fills at close[t]; the
        resulting position earns close[t]→close[t+1], implemented as
        lagged_signal[t+1] × pct_change(close)[t+1] − cost[t+1].
        Cost is applied on every bar where the position changes.

        Parameters
        ----------
        close : pd.Series
        high  : pd.Series (optional, used by vol filter & ATR)
        low   : pd.Series (optional)

        Returns
        -------
        strategy_returns : pd.Series
            NaN-free daily P&L series (NaN → 0).
        """
        p = self.params

        signals = self.generate_signals(close, high=high, low=low)

        # Daily log returns → arithmetic returns (avoids compounding issues)
        daily_returns = close.pct_change()

        # Signal on day t → applies to return on day t+1
        lagged_signals = signals.shift(1)

        # Raw strategy returns
        strategy_returns = lagged_signals * daily_returns

        # Transaction costs: charged on every position change
        position_changes = signals.diff().abs()
        # First non-zero signal counts as a trade (diff is NaN for first bar)
        position_changes.iloc[0] = abs(signals.iloc[0]) if not pd.isna(signals.iloc[0]) else 0.0
        costs = position_changes * p.transaction_cost

        # Shift costs by 1 to match execution timing
        strategy_returns = strategy_returns - costs.shift(1)

        # Fill NaN (from pct_change or shift) with 0
        strategy_returns = strategy_returns.fillna(0.0)

        return strategy_returns
