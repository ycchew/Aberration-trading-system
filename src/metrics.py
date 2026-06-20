"""
Performance metrics for the Aberration ETF strategy.

All methods accept daily arithmetic return series (e.g. 0.01 = +1%).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Static helpers for common trading strategy performance metrics."""

    @staticmethod
    def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
        r = returns.dropna()
        if len(r) == 0:
            return 0.0
        terminal = (1.0 + r).prod()
        if terminal <= 0:
            return -1.0
        return float(terminal ** (periods_per_year / len(r)) - 1.0)

    @staticmethod
    def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        r = returns.dropna()
        return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) >= 2 else 0.0

    @staticmethod
    def sharpe(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        r = returns.dropna()
        if len(r) < 2:
            return 0.0
        daily_rf = risk_free_rate / periods_per_year
        std = r.std(ddof=1)
        if std == 0:
            return 0.0
        return float((r - daily_rf).mean() / std * np.sqrt(periods_per_year))

    @staticmethod
    def sortino(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        r = returns.dropna()
        if len(r) < 2:
            return 0.0
        daily_rf = risk_free_rate / periods_per_year
        excess   = r - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
        return float(excess.mean() * periods_per_year / downside_std) if downside_std > 0 else 0.0

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) == 0:
            return 0.0
        equity = (1.0 + r).cumprod()
        dd = (equity - equity.cummax()) / equity.cummax()
        return float(dd.min())

    @staticmethod
    def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
        mdd = PerformanceMetrics.max_drawdown(returns)
        return 0.0 if mdd == 0 else float(
            PerformanceMetrics.cagr(returns, periods_per_year) / abs(mdd)
        )

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        r = returns.dropna()
        active = r[r != 0.0]
        return float((active > 0).mean()) if len(active) > 0 else 0.0

    @staticmethod
    def trade_count(signals: pd.Series) -> int:
        if signals is None or len(signals) == 0:
            return 0
        changes = signals.diff().fillna(signals.iloc[0] if len(signals) > 0 else 0)
        return int((changes != 0).sum())

    @staticmethod
    def avg_trade_duration(signals: pd.Series, periods_per_year: int = 252) -> float:
        if signals is None or len(signals) == 0:
            return 0.0
        in_trade = (signals != 0).astype(int)
        trade_id = (in_trade.diff().fillna(
            in_trade.iloc[0] if len(in_trade) > 0 else 0
        ) != 0).cumsum()
        durations = [
            len(grp) for _, grp in in_trade.groupby(trade_id) if grp.iloc[0] == 1
        ]
        return float(np.mean(durations)) if durations else 0.0

    @staticmethod
    def summary(
        returns: pd.Series,
        signals: Optional[pd.Series] = None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Union[float, int]]:
        pm = PerformanceMetrics
        metrics: Dict[str, Union[float, int]] = {
            "cagr":        pm.cagr(returns, periods_per_year),
            "annual_vol":  pm.annual_volatility(returns, periods_per_year),
            "sharpe":      pm.sharpe(returns, risk_free_rate, periods_per_year),
            "sortino":     pm.sortino(returns, risk_free_rate, periods_per_year),
            "max_drawdown":pm.max_drawdown(returns),
            "calmar":      pm.calmar(returns, periods_per_year),
            "win_rate":    pm.win_rate(returns),
        }
        if signals is not None:
            metrics["trade_count"]            = pm.trade_count(signals)
            metrics["avg_trade_duration_days"]= pm.avg_trade_duration(signals)
        else:
            metrics["trade_count"]            = 0
            metrics["avg_trade_duration_days"]= 0.0
        return metrics
