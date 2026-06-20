"""
ETF data downloader with local disk cache.

Downloads OHLCV data from Yahoo Finance via yfinance and stores
individual ticker CSV files under cache_dir for fast re-use.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


ETF_UNIVERSE: List[str] = [
    # US Broad Equity
    "SPY", "QQQ", "IWM", "VTI", "MDY",
    # Sector SPDR
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
    # International
    "EFA", "EEM", "EWJ", "VEU",
    # Fixed Income
    "TLT", "IEF", "AGG", "LQD", "HYG",
    # Commodities
    "GLD", "SLV", "DBC", "DBA", "IAU",
    # REITs
    "VNQ",
    # Diversifiers
    "BIL", "TIP", "USO", "UNG",
]


class DataFetcher:
    """
    Downloads and caches ETF OHLCV data from Yahoo Finance.

    Each ticker is stored as a CSV under cache_dir; subsequent calls
    read from disk and only fetch new bars from the network.
    """

    CACHE_SUFFIX = "_daily.csv"

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}{self.CACHE_SUFFIX}")

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=False)
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        df.to_csv(self._cache_path(ticker))

    def _fetch_from_yahoo(
        self, ticker: str, start: str, end: Optional[str]
    ) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
        except Exception as exc:
            print(f"  [WARN] yfinance error for {ticker}: {exc}")
            return None

        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if "Close" not in raw.columns:
            return None

        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        df = raw[needed].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def download(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: Optional[str] = None,
        min_years: float = 5,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for a list of tickers.

        Tickers with fewer than min_years x 252 trading days are skipped.
        Data is cached to disk; only new bars are fetched on subsequent calls.
        """
        if end is None:
            end = date.today().isoformat()

        min_rows = int(min_years * 252)
        result: Dict[str, pd.DataFrame] = {}
        skipped, from_cache, from_net = [], [], []

        for ticker in tickers:
            df: Optional[pd.DataFrame] = None

            if use_cache and not refresh_cache:
                df = self._load_from_cache(ticker)
                if df is not None:
                    last = df.index.max()
                    if last < pd.Timestamp(end) - pd.Timedelta(days=5):
                        fresh = self._fetch_from_yahoo(
                            ticker,
                            start=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            end=end,
                        )
                        if fresh is not None and not fresh.empty:
                            df = pd.concat([df, fresh[~fresh.index.isin(df.index)]])
                            df.sort_index(inplace=True)
                            self._save_to_cache(ticker, df)
                    from_cache.append(ticker)

            if df is None:
                df = self._fetch_from_yahoo(ticker, start=start, end=end)
                if df is not None:
                    self._save_to_cache(ticker, df)
                    from_net.append(ticker)

            if df is None or df.empty:
                skipped.append(ticker)
                continue

            df = df.loc[start:end]
            if len(df) < min_rows:
                skipped.append(ticker)
                print(f"  [SKIP] {ticker}: {len(df)} rows (need {min_rows})")
                continue

            result[ticker] = df

        print(
            f"\nLoaded {len(result)}/{len(tickers)} tickers "
            f"({len(from_cache)} from cache, {len(from_net)} downloaded)"
        )
        if skipped:
            print(f"Skipped: {skipped}")
        return result

    def download_close_prices(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: Optional[str] = None,
        min_years: float = 5,
        min_data_pct: float = 0.8,
        ffill_limit: int = 5,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download and align close prices for multiple tickers."""
        data = self.download(tickers, start=start, end=end,
                             min_years=min_years, use_cache=use_cache)
        if not data:
            return pd.DataFrame()

        wide = pd.concat(
            {tkr: df["Close"] for tkr, df in data.items()}, axis=1
        ).sort_index()

        threshold = int(np.ceil(min_data_pct * wide.shape[1]))
        wide = wide.loc[wide.notna().sum(axis=1) >= threshold]
        return wide.ffill(limit=ffill_limit)
