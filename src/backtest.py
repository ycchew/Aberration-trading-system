"""
Backtesting engine for the Aberration ETF trading system.

Supports single-ticker and equal-weight multi-ticker portfolio backtests.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .aberration import AberrationParams, AberrationStrategy
from .metrics import PerformanceMetrics


class Backtester:
    """Runs Aberration strategy backtests on individual ETFs or portfolios."""

    def __init__(self, params: Optional[AberrationParams] = None) -> None:
        self.params   = params or AberrationParams()
        self.strategy = AberrationStrategy(self.params)

    def run_single(
        self,
        close:  pd.Series,
        high:   Optional[pd.Series] = None,
        low:    Optional[pd.Series] = None,
        ticker: str = "",
    ) -> Dict:
        """
        Backtest on a single price series.

        Returns dict: ticker, returns, signals, equity_curve, metrics.
        """
        signals          = self.strategy.generate_signals(close, high=high, low=low)
        strategy_returns = self.strategy.compute_returns(close, high=high, low=low)
        equity_curve     = (1.0 + strategy_returns).cumprod()
        equity_curve.name = ticker or "equity"
        metrics = PerformanceMetrics.summary(strategy_returns, signals=signals)
        return {
            "ticker":       ticker,
            "returns":      strategy_returns,
            "signals":      signals,
            "equity_curve": equity_curve,
            "metrics":      metrics,
        }

    def run_portfolio(
        self,
        close_df: pd.DataFrame,
        high_df:  Optional[pd.DataFrame] = None,
        low_df:   Optional[pd.DataFrame] = None,
        weight_method: str = "equal",
    ) -> Dict:
        """
        Run Aberration on each column independently; combine equal-weight.

        Returns dict: tickers, combined_returns, individual_returns,
        individual_signals, equity_curve, metrics, individual_results.
        """
        if weight_method != "equal":
            raise NotImplementedError("Only 'equal' weighting is supported.")

        tickers: list = list(close_df.columns)
        ind_returns: Dict[str, pd.Series] = {}
        ind_signals: Dict[str, pd.Series] = {}
        ind_results: Dict[str, Dict]      = {}

        for tkr in tickers:
            high = high_df[tkr] if high_df is not None and tkr in high_df.columns else None
            low  = low_df[tkr]  if low_df  is not None and tkr in low_df.columns  else None
            res  = self.run_single(close=close_df[tkr], high=high, low=low, ticker=tkr)
            ind_results[tkr]  = res
            ind_returns[tkr]  = res["returns"]
            ind_signals[tkr]  = res["signals"]

        returns_df       = pd.DataFrame(ind_returns)
        combined_returns = returns_df.mean(axis=1) if tickers else pd.Series(dtype=float)
        combined_returns.name = "portfolio"
        equity_curve     = (1.0 + combined_returns).cumprod()
        equity_curve.name = "portfolio_equity"
        metrics = PerformanceMetrics.summary(combined_returns, signals=None)

        return {
            "tickers":            tickers,
            "combined_returns":   combined_returns,
            "individual_returns": returns_df,
            "individual_signals": pd.DataFrame(ind_signals),
            "equity_curve":       equity_curve,
            "metrics":            metrics,
            "individual_results": ind_results,
        }

    def print_summary(self, result: Dict, title: str = "Backtest Results") -> None:
        """Pretty-print metrics from run_single or run_portfolio."""
        metrics = result.get("metrics", {})
        ticker  = result.get("ticker", result.get("tickers", "Portfolio"))
        if isinstance(ticker, list):
            ticker_str = ", ".join(ticker[:5])
            if len(ticker) > 5:
                ticker_str += f" (+{len(ticker)-5} more)"
        else:
            ticker_str = str(ticker)

        sep = "─" * 52
        print(f"\n{sep}\n  {title}\n  Ticker(s) : {ticker_str}\n{sep}")

        fmt_map = {
            "cagr":                    ("CAGR",               "{:.2%}"),
            "annual_vol":              ("Annual Volatility",   "{:.2%}"),
            "sharpe":                  ("Sharpe Ratio",        "{:.3f}"),
            "sortino":                 ("Sortino Ratio",       "{:.3f}"),
            "max_drawdown":            ("Max Drawdown",        "{:.2%}"),
            "calmar":                  ("Calmar Ratio",        "{:.3f}"),
            "win_rate":                ("Win Rate",            "{:.2%}"),
            "trade_count":             ("Trade Count",         "{:d}"),
            "avg_trade_duration_days": ("Avg Trade Duration",  "{:.1f} days"),
        }
        for key, (label, fmt) in fmt_map.items():
            val = metrics.get(key)
            if val is None:
                continue
            try:
                formatted = (
                    "N/A" if isinstance(val, float) and not np.isfinite(val)
                    else fmt.format(int(val)) if key == "trade_count"
                    else fmt.format(val)
                )
            except (ValueError, TypeError):
                formatted = str(val)
            print(f"  {label:<26} {formatted}")

        ret = result.get("returns") or result.get("combined_returns")
        if ret is not None and len(ret) > 0:
            print(f"  {'Period':<26} {ret.index[0]:%Y-%m-%d} -> {ret.index[-1]:%Y-%m-%d}")
        print(sep)
