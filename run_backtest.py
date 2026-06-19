#!/usr/bin/env python3
"""
run_backtest.py — Aberration ETF Single Portfolio Backtest
==========================================================
Runs the Aberration strategy on a custom set of ETFs and prints
full performance metrics.  Optionally saves an equity curve plot.

Usage
-----
    python run_backtest.py
    python run_backtest.py --tickers SPY QQQ TLT GLD IEF
    python run_backtest.py --tickers SPY TLT GLD --start 2015-01-01 --plot
    python run_backtest.py --tickers QQQ XLK XLV --long-short --period 60
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aberration backtest for a custom ETF portfolio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tickers", nargs="+", default=["SPY", "QQQ", "TLT", "GLD", "IEF"],
        metavar="TICKER",
        help="Space-separated list of ETF tickers to backtest",
    )
    p.add_argument("--start",      default="2010-01-01", metavar="YYYY-MM-DD",
                   help="Backtest start date")
    p.add_argument("--end",        default=None,         metavar="YYYY-MM-DD",
                   help="Backtest end date (default: today)")
    p.add_argument("--period",     type=int,   default=80,  metavar="INT",
                   help="Aberration band period")
    p.add_argument("--std-mult",   type=float, default=2.0, metavar="FLOAT",
                   help="StdDev multiplier for both bands")
    p.add_argument("--long-short", action="store_true",
                   help="Allow short positions (default: long-only)")
    p.add_argument("--no-vol-filter", action="store_true",
                   help="Disable the ATR volatility filter")
    p.add_argument("--plot",       action="store_true",
                   help="Save equity curve plot to results/")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from src import AberrationParams, Backtester
    from src.data_fetcher import DataFetcher

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)

    tickers = [t.upper() for t in args.tickers]

    print(f"\nAberration Backtest")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Period  : {args.period}   Std mult: {args.std_mult}")
    print(f"  Mode    : {'Long-Short' if args.long_short else 'Long-Only'}")
    print(f"  Start   : {args.start}")
    print()

    fetcher = DataFetcher(cache_dir=os.path.join(_HERE, "data", "cache"))
    data = fetcher.download(tickers=tickers, start=args.start, end=args.end, min_years=2)

    if not data:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    available = list(data.keys())
    missing   = [t for t in tickers if t not in data]
    if missing:
        print(f"WARNING: Could not load: {missing}")

    params = AberrationParams(
        period=args.period,
        std_mult_up=args.std_mult,
        std_mult_dn=args.std_mult,
        long_only=not args.long_short,
        use_vol_filter=not args.no_vol_filter,
    )

    import pandas as pd
    close_df = pd.DataFrame({t: data[t]["Close"] for t in available})
    high_df  = pd.DataFrame({t: data[t]["High"]  for t in available if "High" in data[t].columns})
    low_df   = pd.DataFrame({t: data[t]["Low"]   for t in available if "Low"  in data[t].columns})
    close_df = close_df.dropna(how="any")
    high_df  = high_df.reindex(close_df.index) if not high_df.empty else None
    low_df   = low_df.reindex(close_df.index)  if not low_df.empty  else None

    bt = Backtester(params=params)
    result = bt.run_portfolio(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
    )

    bt.print_summary(result, title="Portfolio Backtest Summary")

    # Individual ETF breakdown
    print("\n  Per-ETF Metrics:")
    print(f"  {'Ticker':<8}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}  {'Trades':>6}")
    print(f"  {'─'*45}")
    for tkr, res in result["individual_results"].items():
        m = res["metrics"]
        print(
            f"  {tkr:<8}  {m['cagr']:>7.2%}  "
            f"{m['sharpe']:>7.3f}  {m['max_drawdown']:>7.2%}  "
            f"{int(m.get('trade_count', 0)):>6}"
        )

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(12, 6))

            # Individual ETF curves (gray, thin)
            for tkr, res in result["individual_results"].items():
                eq = (1 + res["returns"]).cumprod()
                ax.plot(eq.index, eq.values, color="gray", alpha=0.35,
                        linewidth=0.8, label=f"_{tkr}")

            # Portfolio equity curve (blue, bold)
            eq_port = result["equity_curve"]
            ax.plot(eq_port.index, eq_port.values, color="royalblue",
                    linewidth=2.2, label="Portfolio (equal-weight)")

            # Buy-and-hold SPY benchmark (if in universe)
            if "SPY" in data:
                spy_close = data["SPY"]["Close"].reindex(close_df.index).dropna()
                spy_bah   = spy_close / spy_close.iloc[0]
                ax.plot(spy_bah.index, spy_bah.values, color="darkorange",
                        linewidth=1.5, linestyle="--", label="SPY Buy & Hold")

            ax.set_title(
                f"Aberration Strategy — {', '.join(available)}\n"
                f"CAGR: {result['metrics']['cagr']:.2%}  "
                f"Sharpe: {result['metrics']['sharpe']:.3f}  "
                f"MaxDD: {result['metrics']['max_drawdown']:.2%}",
                fontsize=11,
            )
            ax.set_ylabel("Equity (normalised, start=1.0)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()

            plot_path = os.path.join(_HERE, "results", "backtest_equity_curve.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"\n  Equity curve plot saved → {plot_path}")

        except ImportError:
            print("\n  [WARN] matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
