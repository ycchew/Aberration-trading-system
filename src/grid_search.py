"""
Grid search over all C(n, k) ETF combinations for the Aberration system.

For each k-ETF combination an equal-weight portfolio is evaluated;
results are filtered by minimum CAGR and ranked by Sharpe ratio.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .aberration import AberrationParams, AberrationStrategy


@dataclass
class GridSearchConfig:
    """Configuration for the ETF portfolio grid search."""
    min_cagr:         float = 0.10
    risk_free:        float = 0.02
    max_results:      int   = 200
    n_etfs_per_combo: int   = 5
    transaction_cost: float = 0.001


class GridSearch:
    """
    Exhaustive search over all C(n, k) ETF combinations.

    Usage
    -----
    gs = GridSearch()
    results = gs.run(data_dict)   # data_dict from DataFetcher.download()
    gs.print_results(results)
    """

    def __init__(
        self,
        aberration_params: Optional[AberrationParams] = None,
        config:            Optional[GridSearchConfig]  = None,
    ) -> None:
        self.params   = aberration_params or AberrationParams()
        self.config   = config or GridSearchConfig()
        self.strategy = AberrationStrategy(self.params)

    def _precompute_returns(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run Aberration on every ETF; return wide DataFrame of daily returns.
        Tickers that generate zero trades are dropped.
        """
        per_ticker: Dict[str, pd.Series] = {}

        for ticker, df in data_dict.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"]
            high  = df["High"] if "High" in df.columns else None
            low   = df["Low"]  if "Low"  in df.columns else None
            try:
                returns = self.strategy.compute_returns(close, high=high, low=low)
                signals = self.strategy.generate_signals(close, high=high, low=low)
            except Exception as exc:
                print(f"  [WARN] {ticker}: {exc}")
                continue

            n_trades = int(
                (signals.diff().fillna(signals.iloc[0] if len(signals) else 0) != 0).sum()
            )
            if n_trades == 0:
                continue
            per_ticker[ticker] = returns

        if not per_ticker:
            return pd.DataFrame()

        ret_df = pd.DataFrame(per_ticker)
        ret_df.dropna(how="all", inplace=True)
        ret_df.fillna(0.0, inplace=True)
        return ret_df.sort_index()

    def run(self, data_dict: Dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate all C(n, k) combinations and return results sorted by Sharpe.

        Columns: etfs, cagr, sharpe, max_drawdown, calmar, annual_vol, n_trades_avg
        """
        cfg = self.config
        k, rf, ppy = cfg.n_etfs_per_combo, cfg.risk_free, 252

        if verbose:
            print("Pre-computing per-ETF strategy returns...")
        ret_df = self._precompute_returns(data_dict)

        if ret_df.empty or ret_df.shape[1] < k:
            print(f"Not enough tradeable ETFs ({ret_df.shape[1]}) for k={k}.")
            return pd.DataFrame()

        tickers  = list(ret_df.columns)
        n        = len(tickers)
        ret_mat  = ret_df.values.astype(np.float64)   # (T, n)
        T        = ret_mat.shape[0]
        daily_rf = rf / ppy

        # Per-ticker trade counts for reporting
        trade_count_arr = np.zeros(n, dtype=float)
        for i, tkr in enumerate(tickers):
            df  = data_dict.get(tkr)
            if df is None:
                continue
            close = df["Close"]
            high  = df["High"] if "High" in df.columns else None
            low   = df["Low"]  if "Low"  in df.columns else None
            try:
                sigs = self.strategy.generate_signals(close, high=high, low=low)
                trade_count_arr[i] = int(
                    (sigs.diff().fillna(sigs.iloc[0] if len(sigs) else 0) != 0).sum()
                )
            except Exception:
                pass

        total = _comb(n, k)
        if verbose:
            print(f"Searching {total:,} combinations of {k} ETFs from {n} candidates...")

        results: List[Dict] = []
        combo_iter = itertools.combinations(range(n), k)

        try:
            from tqdm import tqdm
            if verbose:
                combo_iter = tqdm(combo_iter, total=total, desc="Grid search", unit="combo")
        except ImportError:
            pass

        for combo_num, idx in enumerate(combo_iter, start=1):
            port = ret_mat[:, idx].mean(axis=1)

            terminal = np.prod(1.0 + port)
            if terminal <= 0:
                continue
            cagr = terminal ** (ppy / T) - 1.0
            if cagr < cfg.min_cagr:
                continue

            std      = port.std(ddof=1)
            ann_vol  = std * np.sqrt(ppy)
            sharpe   = (port - daily_rf).mean() / std * np.sqrt(ppy) if std > 0 else 0.0

            equity   = np.cumprod(1.0 + port)
            mdd      = float(((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min())
            calmar   = cagr / abs(mdd) if mdd != 0 else 0.0
            avg_tc   = float(trade_count_arr[list(idx)].mean())

            results.append({
                "etfs":         tuple(tickers[i] for i in idx),
                "cagr":         round(cagr,    6),
                "sharpe":       round(sharpe,  6),
                "max_drawdown": round(mdd,     6),
                "calmar":       round(calmar,  6),
                "annual_vol":   round(ann_vol, 6),
                "n_trades_avg": round(avg_tc,  1),
            })

            if verbose and combo_num % 10_000 == 0 and not isinstance(combo_iter, type(iter([]))):
                print(f"  {combo_num:>10,}/{total:,}  ({100*combo_num/total:.1f}%)  "
                      f"{len(results)} passed")

        if not results:
            print("No combinations passed the min_cagr filter.")
            return pd.DataFrame(
                columns=["etfs","cagr","sharpe","max_drawdown","calmar","annual_vol","n_trades_avg"]
            )

        df_out = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
        if verbose:
            print(f"\nSearch complete: {len(df_out):,} combos passed CAGR >= {cfg.min_cagr:.0%}.")
        return df_out.head(cfg.max_results)

    def get_top_n(self, results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return results_df.head(n).copy()

    def print_results(self, results_df: pd.DataFrame, n: int = 20) -> None:
        if results_df is None or results_df.empty:
            print("No results to display.")
            return
        top = results_df.head(n)
        sep = "─" * 92
        print(f"\n{sep}")
        print(f"  Top {len(top)} ETF Combinations (sorted by Sharpe Ratio)")
        print(sep)
        print(f"  {'Rank':>4}  {'ETFs':<36}  {'CAGR':>7}  {'Sharpe':>7}  "
              f"{'MaxDD':>7}  {'Calmar':>7}  {'AnnVol':>7}  {'Trades':>6}")
        print(sep)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            etfs = ", ".join(row["etfs"])
            if len(etfs) > 36:
                etfs = etfs[:33] + "..."
            print(f"  {rank:>4}  {etfs:<36}  {row['cagr']:>7.2%}  {row['sharpe']:>7.3f}  "
                  f"{row['max_drawdown']:>7.2%}  {row['calmar']:>7.3f}  "
                  f"{row['annual_vol']:>7.2%}  {row['n_trades_avg']:>6.1f}")
        print(sep)


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r
