#!/usr/bin/env python3
"""
run_grid_search.py — Aberration ETF System: Parameter & Portfolio Grid Search
==============================================================================

Searches all C(n, k) ETF combinations (default k=5) from the standard ~35-ETF
universe using the Aberration breakout strategy (Keith Fitschen, 1986).

For each combination the equal-weight portfolio CAGR, Sharpe, max drawdown,
Calmar, and annualised volatility are evaluated.  Results are ranked by Sharpe
ratio and the best combinations are printed and saved.

Usage
-----
    python run_grid_search.py
    python run_grid_search.py --start 2015-01-01 --min-cagr 0.12 --top-n 15
    python run_grid_search.py --long-short --period 100 --std-mult 2.5
    python run_grid_search.py --no-vol-filter --top-n 30

Arguments
---------
    --start         ISO date for backtest start              [default: 2010-01-01]
    --min-cagr      Minimum CAGR to keep a combination       [default: 0.10]
    --top-n         Number of top results to display         [default: 20]
    --long-short    Enable long+short mode (default: long-only)
    --period        SMA / StdDev lookback period             [default: 80]
    --std-mult      Std-dev multiplier for both bands        [default: 2.0]
    --no-vol-filter Disable the 20-day ATR volatility filter
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so 'src' is importable when the
# script is invoked from a different working directory.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aberration ETF system — exhaustive portfolio grid search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        metavar="YYYY-MM-DD",
        help="Backtest start date (ISO format)",
    )
    parser.add_argument(
        "--min-cagr",
        type=float,
        default=0.10,
        metavar="FLOAT",
        help="Minimum portfolio CAGR required to include a combo in results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="INT",
        help="Number of top combos to print in the summary table",
    )
    parser.add_argument(
        "--long-short",
        action="store_true",
        default=False,
        help="Enable long-and-short mode (default: long-only for ETFs)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=80,
        metavar="INT",
        help="SMA / StdDev lookback period for Aberration bands",
    )
    parser.add_argument(
        "--std-mult",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="Standard-deviation multiplier for upper AND lower bands",
    )
    parser.add_argument(
        "--no-vol-filter",
        action="store_true",
        default=False,
        help="Disable the 20-day ATR volatility filter",
    )
    return parser.parse_args()


def _print_banner(args: argparse.Namespace) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  Aberration ETF Trading System — Grid Search")
    print("  Based on Keith Fitschen's Aberration system (1986)")
    print(sep)
    print(f"  Strategy rules:")
    print(f"    Middle Band  = SMA({args.period}) of closing prices")
    print(f"    Upper Band   = Middle + {args.std_mult}× StdDev({args.period})")
    print(f"    Lower Band   = Middle − {args.std_mult}× StdDev({args.period})")
    print(f"    Long entry   : close > upper  AND  vol filter passes")
    print(f"    Long exit    : close < middle")
    if args.long_short:
        print(f"    Short entry  : close < lower  AND  vol filter passes")
        print(f"    Short exit   : close > middle")
    print(f"")
    print(f"  Run configuration:")
    print(f"    Start date   : {args.start}")
    print(f"    Mode         : {'Long-Short' if args.long_short else 'Long-Only (ETF)'}")
    print(f"    Vol filter   : {'Disabled' if args.no_vol_filter else 'Enabled (20-day ATR)'}")
    print(f"    Min CAGR     : {args.min_cagr:.1%}")
    print(f"    Top-N display: {args.top_n}")
    print(sep)
    print()


def main() -> None:
    args = _parse_args()
    _print_banner(args)

    # ------------------------------------------------------------------
    # Imports (deferred so --help works without optional deps installed)
    # ------------------------------------------------------------------
    try:
        from src import DataFetcher, GridSearch, GridSearchConfig, AberrationParams, Backtester
        from src.data_fetcher import ETF_UNIVERSE
    except ImportError as exc:
        print(f"[ERROR] Could not import from src/: {exc}")
        print("        Make sure you are running from the project root directory.")
        sys.exit(1)

    try:
        from tabulate import tabulate
        _HAS_TABULATE = True
    except ImportError:
        _HAS_TABULATE = False

    try:
        from tqdm import tqdm as _tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download ETF data
    # ------------------------------------------------------------------
    print(f"Downloading data for {len(ETF_UNIVERSE)} ETFs (start={args.start})...")
    fetcher = DataFetcher(cache_dir=os.path.join(_HERE, "data", "cache"))
    data_dict = fetcher.download(
        tickers=ETF_UNIVERSE,
        start=args.start,
        use_cache=True,
    )

    if not data_dict:
        print("[ERROR] No data downloaded. Check your internet connection or cache.")
        sys.exit(1)

    n_etfs = len(data_dict)
    print(f"Ready: {n_etfs} ETFs loaded.\n")

    # ------------------------------------------------------------------
    # 2. Build AberrationParams and GridSearchConfig from CLI args
    # ------------------------------------------------------------------
    ab_params = AberrationParams(
        period=args.period,
        std_mult_up=args.std_mult,
        std_mult_dn=args.std_mult,
        long_only=not args.long_short,
        use_vol_filter=not args.no_vol_filter,
    )

    gs_config = GridSearchConfig(
        min_cagr=args.min_cagr,
        max_results=max(args.top_n * 5, 200),  # Keep a generous buffer for CSV
        n_etfs_per_combo=5,
        transaction_cost=ab_params.transaction_cost,
    )

    # ------------------------------------------------------------------
    # 3. Run grid search
    # ------------------------------------------------------------------
    gs = GridSearch(aberration_params=ab_params, config=gs_config)

    print("Running grid search...")
    results_df = gs.run(data_dict, verbose=True)

    # ------------------------------------------------------------------
    # 4. Handle the case where no combo passes min_cagr
    # ------------------------------------------------------------------
    if results_df is None or results_df.empty:
        print(
            f"\n[WARN] No combination achieved CAGR >= {args.min_cagr:.1%}.\n"
            f"       Showing top 10 by Sharpe with no CAGR floor.\n"
        )
        # Re-run without the CAGR filter
        gs_fallback = GridSearch(
            aberration_params=ab_params,
            config=GridSearchConfig(
                min_cagr=-999.0,
                max_results=10,
                n_etfs_per_combo=5,
            ),
        )
        results_df = gs_fallback.run(data_dict, verbose=False)
        if results_df is None or results_df.empty:
            print("[ERROR] Grid search returned no results at all. Exiting.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Print top-N results table
    # ------------------------------------------------------------------
    top_df = results_df.head(args.top_n).copy()

    print(f"\nTop {min(args.top_n, len(top_df))} ETF Combinations (sorted by Sharpe)\n")

    # Build display rows
    display_rows = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        etfs_str = " | ".join(row["etfs"]) if isinstance(row["etfs"], (list, tuple)) else str(row["etfs"])
        display_rows.append([
            rank,
            etfs_str,
            f"{row['cagr']:.2%}",
            f"{row['sharpe']:.3f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['calmar']:.3f}",
            f"{row['annual_vol']:.2%}",
            f"{row['n_trades_avg']:.1f}",
        ])

    headers = ["Rank", "ETFs", "CAGR", "Sharpe", "MaxDD", "Calmar", "AnnVol", "AvgTrades"]

    if _HAS_TABULATE:
        print(tabulate(display_rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # Fallback: manual alignment
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in display_rows), default=0))
                      for i, h in enumerate(headers)]
        sep_line = "  " + "  ".join("-" * w for w in col_widths)
        hdr_line = "  " + "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        print(sep_line)
        print(hdr_line)
        print(sep_line)
        for row_vals in display_rows:
            print("  " + "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row_vals)))
        print(sep_line)

    # ------------------------------------------------------------------
    # 6. Save full results to CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(results_dir, "grid_search_results.csv")
    save_df = results_df.copy()
    # Flatten tuple column to string for CSV
    if "etfs" in save_df.columns:
        save_df["etfs"] = save_df["etfs"].apply(
            lambda v: " | ".join(v) if isinstance(v, (list, tuple)) else str(v)
        )
    save_df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")

    # ------------------------------------------------------------------
    # 7. Per-ETF metrics for the best combination
    # ------------------------------------------------------------------
    best_row = results_df.iloc[0]
    best_tickers = list(best_row["etfs"]) if isinstance(best_row["etfs"], (list, tuple)) else list(best_row["etfs"])

    print(f"\n{'=' * 68}")
    print(f"  Best Combination: {' | '.join(best_tickers)}")
    print(f"  CAGR={best_row['cagr']:.2%}  Sharpe={best_row['sharpe']:.3f}  MaxDD={best_row['max_drawdown']:.2%}")
    print(f"{'=' * 68}")
    print("  Individual ETF metrics:\n")

    bt_best = Backtester(ab_params)
    etf_metric_rows = []
    for tkr in best_tickers:
        df_tkr = data_dict.get(tkr)
        if df_tkr is None:
            continue
        single = bt_best.run_single(
            close=df_tkr["Close"],
            high=df_tkr.get("High") if hasattr(df_tkr, "get") else (df_tkr["High"] if "High" in df_tkr.columns else None),
            low=df_tkr.get("Low")  if hasattr(df_tkr, "get") else (df_tkr["Low"]  if "Low"  in df_tkr.columns else None),
            ticker=tkr,
        )
        m = single["metrics"]
        etf_metric_rows.append([
            tkr,
            f"{m['cagr']:.2%}",
            f"{m['sharpe']:.3f}",
            f"{m['max_drawdown']:.2%}",
            f"{m['calmar']:.3f}",
            f"{m.get('win_rate', 0):.2%}",
            f"{int(m.get('trade_count', 0))}",
        ])

    etf_headers = ["Ticker", "CAGR", "Sharpe", "MaxDD", "Calmar", "WinRate", "Trades"]
    if _HAS_TABULATE:
        print(tabulate(etf_metric_rows, headers=etf_headers, tablefmt="simple"))
    else:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in etf_metric_rows), default=0))
                      for i, h in enumerate(etf_headers)]
        print("  " + "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(etf_headers)))
        print("  " + "  ".join("-" * w for w in col_widths))
        for row_vals in etf_metric_rows:
            print("  " + "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row_vals)))

    # ------------------------------------------------------------------
    # 8. Detailed portfolio backtest on best 5-ETF combo
    # ------------------------------------------------------------------
    print(f"\n{'=' * 68}")
    print(f"  Detailed Portfolio Backtest — {' | '.join(best_tickers)}")
    print(f"{'=' * 68}")

    close_df = pd.DataFrame({t: data_dict[t]["Close"] for t in best_tickers if t in data_dict})
    high_df  = pd.DataFrame({t: data_dict[t]["High"]  for t in best_tickers
                              if t in data_dict and "High" in data_dict[t].columns})
    low_df   = pd.DataFrame({t: data_dict[t]["Low"]   for t in best_tickers
                              if t in data_dict and "Low"  in data_dict[t].columns})

    # Inner-join alignment
    close_df = close_df.dropna(how="any")
    high_df  = high_df.reindex(close_df.index) if not high_df.empty else None
    low_df   = low_df.reindex(close_df.index)  if not low_df.empty  else None

    portfolio = bt_best.run_portfolio(
        close_df=close_df,
        high_df=high_df if (high_df is not None and not high_df.empty) else None,
        low_df=low_df   if (low_df  is not None and not low_df.empty)  else None,
    )
    bt_best.print_summary(portfolio, title="Best-Combo Portfolio Summary")

    # ------------------------------------------------------------------
    # 9. Save equity curve
    # ------------------------------------------------------------------
    equity_curve = portfolio["equity_curve"]
    equity_path = os.path.join(results_dir, "equity_curve.csv")
    equity_curve.to_frame(name="portfolio_equity").to_csv(equity_path)
    print(f"\nEquity curve saved to: {equity_path}")
    print("\nGrid search complete.")


# ---------------------------------------------------------------------------
# Lazy import — pandas is needed here but shouldn't block --help
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

if __name__ == "__main__":
    main()
