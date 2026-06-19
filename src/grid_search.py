"""
Grid search over all C(n, k) ETF combinations for the Aberration system.

For each combination of k ETFs an equal-weight portfolio return series is
computed on the IN-SAMPLE portion of the data, filtered by a minimum CAGR,
and the winning combos are also evaluated on the OUT-OF-SAMPLE period.

Methodology
-----------
- Data is split chronologically: first *train_frac* of dates = in-sample,
  remainder = out-of-sample.
- Per-combo returns are computed on the inner join of each ticker's live
  history, so late-starting ETFs don't dilute the combo's return stream.
- CAGR is annualised over T_eff (actual joint trading days), not the full
  date range.
- Combos are selected on in-sample Sharpe; OOS metrics are reported
  alongside but not used for selection.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .aberration import AberrationParams, AberrationStrategy
from .metrics import PerformanceMetrics


@dataclass
class GridSearchConfig:
    """Configuration for the portfolio grid search."""

    min_cagr: float = 0.10          # Minimum in-sample CAGR to keep a combo
    risk_free: float = 0.02         # Annual risk-free rate for Sharpe/Sortino
    max_results: int = 50           # Maximum rows in the returned DataFrame
    n_etfs_per_combo: int = 5       # k (number of ETFs per combination)
    transaction_cost: float = 0.001 # Passed through to AberrationParams if no params given
    train_frac: float = 0.70        # Fraction of history used for in-sample selection


class GridSearch:
    """
    Exhaustive search over all C(n, k) ETF combinations.

    Usage
    -----
    >>> from src import DataFetcher, GridSearch, AberrationParams
    >>> fetcher = DataFetcher()
    >>> data = fetcher.download(ETF_UNIVERSE, start="2010-01-01")
    >>> gs = GridSearch()
    >>> results = gs.run(data, verbose=True)
    >>> gs.print_results(results)

    Parameters
    ----------
    aberration_params : AberrationParams | None
        Strategy configuration; defaults to AberrationParams().
    config : GridSearchConfig | None
        Grid-search configuration; defaults to GridSearchConfig().
    """

    def __init__(
        self,
        aberration_params: Optional[AberrationParams] = None,
        config: Optional[GridSearchConfig] = None,
    ) -> None:
        self.params = aberration_params or AberrationParams()
        self.config = config or GridSearchConfig()
        self.strategy = AberrationStrategy(self.params)

        # Set by run() — date boundaries of the IS / OOS split
        self.train_start_date: Optional[pd.Timestamp] = None
        self.train_end_date: Optional[pd.Timestamp] = None
        self.oos_start_date: Optional[pd.Timestamp] = None
        self.oos_end_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_returns(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Run AberrationStrategy on every ETF in *data_dict* and return a
        wide DataFrame of daily strategy returns plus per-ticker trade counts.

        NaN values are PRESERVED (not zero-filled) so that the per-combo
        inner-join logic in run() can determine each ticker's live start date.

        Parameters
        ----------
        data_dict : dict[str, pd.DataFrame]
            Output of DataFetcher.download(): ticker → OHLCV DataFrame.

        Returns
        -------
        returns_df : pd.DataFrame
            T × n_etfs of daily strategy returns; NaN where ticker has no history.
        trade_counts : dict[str, int]
            Number of position changes per ticker.
        """
        per_ticker_returns: Dict[str, pd.Series] = {}
        trade_counts: Dict[str, int] = {}

        for ticker, df in data_dict.items():
            if df is None or df.empty:
                continue

            close = df["Close"] if "Close" in df.columns else None
            if close is None or close.empty:
                continue

            high = df["High"] if "High" in df.columns else None
            low = df["Low"] if "Low" in df.columns else None

            try:
                returns = self.strategy.compute_returns(close, high=high, low=low)
                signals = self.strategy.generate_signals(close, high=high, low=low)
            except Exception as exc:
                print(f"  [WARN] Error computing returns for {ticker}: {exc}")
                continue

            n_trades = int(
                (signals.diff().fillna(signals.iloc[0] if len(signals) else 0) != 0).sum()
            )
            if n_trades == 0:
                print(f"  [SKIP] {ticker}: no trades generated")
                continue

            per_ticker_returns[ticker] = returns
            trade_counts[ticker] = n_trades

        if not per_ticker_returns:
            return pd.DataFrame(), {}

        # Outer join preserves each ticker's full date range; NaN where absent.
        returns_df = pd.DataFrame(per_ticker_returns)
        returns_df.dropna(how="all", inplace=True)
        # Intentionally NOT calling fillna(0) — NaN = "ticker not yet live"
        returns_df.sort_index(inplace=True)

        return returns_df, trade_counts

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate all C(n, k) ETF combinations and return ranked results.

        Selection is performed on the in-sample portion (first *train_frac*
        of dates). Out-of-sample metrics are reported for every selected
        combo but are not used for ranking.

        Parameters
        ----------
        data_dict : dict[str, pd.DataFrame]
            Output of DataFetcher.download().
        verbose : bool
            Show a tqdm progress bar if True.

        Returns
        -------
        pd.DataFrame with columns:
            etfs, is_cagr, is_sharpe, oos_cagr, oos_sharpe,
            oos_max_drawdown, max_drawdown, calmar, annual_vol, n_trades_avg
        Sorted by *is_sharpe* descending.
        """
        cfg = self.config
        k = cfg.n_etfs_per_combo
        rf = cfg.risk_free
        min_cagr = cfg.min_cagr
        ppy = 252

        # --- Pre-compute per-ETF returns (NaN intact) ---
        if verbose:
            print("Pre-computing per-ETF strategy returns...")
        returns_df, trade_counts = self._precompute_returns(data_dict)

        if returns_df.empty or returns_df.shape[1] < k:
            print(
                f"Not enough ETFs with trades to form combinations of size {k}. "
                f"Got {returns_df.shape[1]} usable ETFs."
            )
            return pd.DataFrame()

        tickers = list(returns_df.columns)
        n = len(tickers)

        # Raw returns matrix — NaN kept to represent "not yet live"
        ret_matrix = returns_df.values.astype(np.float64)  # shape: (T, n)
        T = ret_matrix.shape[0]

        # --- Train / test split ---
        train_end = max(int(T * cfg.train_frac), ppy)  # at least 1 year IS
        train_end = min(train_end, T - 1)               # leave at least 1 OOS day
        has_oos = (T - train_end) >= 60

        all_dates = returns_df.index
        self.train_start_date = all_dates[0]
        self.train_end_date = all_dates[train_end - 1]
        self.oos_start_date = all_dates[train_end] if has_oos else None
        self.oos_end_date = all_dates[-1] if has_oos else None

        # --- Per-ticker first non-NaN row (enables fast per-combo inner join) ---
        starts = np.array([
            int(np.argmax(~np.isnan(ret_matrix[:, i])))
            if np.any(~np.isnan(ret_matrix[:, i])) else T
            for i in range(n)
        ], dtype=np.intp)

        trade_count_arr = np.array(
            [trade_counts.get(t, 0) for t in tickers], dtype=float
        )

        # --- Iterate combinations ---
        total_combos = _comb(n, k)
        if verbose:
            pct_oos = f"{1 - cfg.train_frac:.0%}"
            print(
                f"Searching {total_combos:,} combinations of {k} ETFs "
                f"from {n} candidates  "
                f"(IS={cfg.train_frac:.0%} / OOS={pct_oos}) ..."
            )

        results: List[Dict] = []
        combo_iter = itertools.combinations(range(n), k)

        try:
            from tqdm import tqdm
            use_tqdm = verbose
        except ImportError:
            use_tqdm = False

        if use_tqdm:
            combo_iter = tqdm(combo_iter, total=total_combos, desc="Grid search", unit="combo")

        daily_rf = rf / ppy

        for combo_num, idx_combo in enumerate(combo_iter, start=1):
            col = list(idx_combo)

            # First row where ALL k tickers have non-NaN data
            t0 = int(max(starts[i] for i in idx_combo))

            # Need at least ppy joint days within the in-sample window
            if t0 >= train_end or (train_end - t0) < ppy:
                continue

            # ---- In-sample returns (inner join for this combo) ----
            sub_is = ret_matrix[t0:train_end, :][:, col]  # (IS_len, k)
            valid_is = ~np.isnan(sub_is).any(axis=1)
            port_is = sub_is[valid_is].mean(axis=1)
            T_eff_is = len(port_is)
            if T_eff_is < ppy:
                continue

            # IS CAGR (fast numpy; T_eff_is = actual joint trading days)
            terminal_is = np.prod(1.0 + port_is)
            if terminal_is <= 0:
                continue
            cagr_is = terminal_is ** (ppy / T_eff_is) - 1.0
            if cagr_is < min_cagr:
                continue

            # Remaining IS metrics
            ann_vol = port_is.std(ddof=1) * np.sqrt(ppy)
            excess_is = port_is - daily_rf
            std_is = port_is.std(ddof=1)
            sharpe_is = (excess_is.mean() / std_is * np.sqrt(ppy)) if std_is > 0 else 0.0
            equity_is = np.cumprod(1.0 + port_is)
            roll_max_is = np.maximum.accumulate(equity_is)
            mdd = float(((equity_is - roll_max_is) / roll_max_is).min())
            calmar = cagr_is / abs(mdd) if mdd != 0 else 0.0

            # ---- Out-of-sample returns ----
            oos_cagr = oos_sharpe = oos_mdd = np.nan
            if has_oos:
                # OOS inner join: start at max(global_combo_start, train_end)
                t0_oos = max(t0, train_end)
                if T - t0_oos >= 60:
                    sub_oos = ret_matrix[t0_oos:, :][:, col]  # (OOS_len, k)
                    valid_oos = ~np.isnan(sub_oos).any(axis=1)
                    port_oos = sub_oos[valid_oos].mean(axis=1)
                    T_eff_oos = len(port_oos)
                    if T_eff_oos >= 60:
                        term_oos = np.prod(1.0 + port_oos)
                        oos_cagr = (term_oos ** (ppy / T_eff_oos) - 1.0
                                    if term_oos > 0 else -1.0)
                        std_oos = port_oos.std(ddof=1)
                        oos_sharpe = (
                            ((port_oos - daily_rf).mean() / std_oos * np.sqrt(ppy))
                            if std_oos > 0 else 0.0
                        )
                        eq_oos = np.cumprod(1.0 + port_oos)
                        peak_oos = np.maximum.accumulate(eq_oos)
                        oos_mdd = float(((eq_oos - peak_oos) / peak_oos).min())

            avg_trades = float(trade_count_arr[col].mean())
            combo_tickers = tuple(tickers[i] for i in idx_combo)

            results.append(
                {
                    "etfs": combo_tickers,
                    "is_cagr": round(cagr_is, 6),
                    "is_sharpe": round(sharpe_is, 6),
                    "oos_cagr": round(oos_cagr, 6) if not np.isnan(oos_cagr) else np.nan,
                    "oos_sharpe": round(oos_sharpe, 6) if not np.isnan(oos_sharpe) else np.nan,
                    "oos_max_drawdown": round(oos_mdd, 6) if not np.isnan(oos_mdd) else np.nan,
                    "max_drawdown": round(mdd, 6),
                    "calmar": round(calmar, 6),
                    "annual_vol": round(ann_vol, 6),
                    "n_trades_avg": round(avg_trades, 1),
                }
            )

            # Verbose progress fallback
            if not use_tqdm and verbose and combo_num % 10_000 == 0:
                pct = 100 * combo_num / total_combos
                print(
                    f"  Progress: {combo_num:>10,} / {total_combos:,} "
                    f"({pct:.1f}%) — {len(results)} combos passed filter"
                )

        if not results:
            print("No combinations passed the min_cagr filter.")
            return pd.DataFrame(
                columns=[
                    "etfs", "is_cagr", "is_sharpe", "oos_cagr", "oos_sharpe",
                    "oos_max_drawdown", "max_drawdown", "calmar",
                    "annual_vol", "n_trades_avg",
                ]
            )

        df_results = pd.DataFrame(results)
        df_results.sort_values("is_sharpe", ascending=False, inplace=True)
        df_results.reset_index(drop=True, inplace=True)

        if verbose:
            n_pass = len(df_results)
            print(
                f"\nSearch complete. {n_pass:,} combinations passed "
                f"IS min_cagr={min_cagr:.0%} filter."
            )

        return df_results.head(cfg.max_results)

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_top_n(self, results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return the top *n* rows of a results DataFrame (sorted by IS Sharpe)."""
        return results_df.head(n).copy()

    def print_results(self, results_df: pd.DataFrame, n: int = 20) -> None:
        """Pretty-print the top *n* rows of the results table."""
        if results_df is None or results_df.empty:
            print("No results to display.")
            return

        top = results_df.head(n)
        sep = "─" * 110

        print(f"\n{sep}")
        print(f"  Top {min(n, len(top))} ETF Combinations (sorted by IS Sharpe)")
        print(sep)
        header = (
            f"  {'Rank':>4}  "
            f"{'ETFs':<35}  "
            f"{'IS_CAGR':>8}  "
            f"{'IS_Shrp':>7}  "
            f"{'OOS_CAGR':>8}  "
            f"{'OOS_Shrp':>8}  "
            f"{'OOS_MaxDD':>9}  "
            f"{'MaxDD':>7}  "
            f"{'AnnVol':>7}  "
            f"{'Trades':>6}"
        )
        print(header)
        print(sep)

        def _fmt(v: float, fmt: str) -> str:
            return fmt.format(v) if (v is not None and not np.isnan(v)) else "    N/A"

        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            etfs_str = ", ".join(row["etfs"])
            if len(etfs_str) > 35:
                etfs_str = etfs_str[:32] + "..."
            line = (
                f"  {rank:>4}  "
                f"{etfs_str:<35}  "
                f"{_fmt(row['is_cagr'], '{:>8.2%}')}  "
                f"{_fmt(row['is_sharpe'], '{:>7.3f}')}  "
                f"{_fmt(row.get('oos_cagr', float('nan')), '{:>8.2%}')}  "
                f"{_fmt(row.get('oos_sharpe', float('nan')), '{:>8.3f}')}  "
                f"{_fmt(row.get('oos_max_drawdown', float('nan')), '{:>9.2%}')}  "
                f"{_fmt(row['max_drawdown'], '{:>7.2%}')}  "
                f"{_fmt(row['annual_vol'], '{:>7.2%}')}  "
                f"{row['n_trades_avg']:>6.1f}"
            )
            print(line)

        print(sep)


# ---------------------------------------------------------------------------
# Helper: C(n, k) without scipy dependency
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    """Return C(n, k) using integer arithmetic."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
