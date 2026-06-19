# Aberration ETF Trading System

A full-stack implementation of Keith Fitschen's **Aberration** mechanical trading
system adapted for Exchange-Traded Funds (ETFs), with an exhaustive grid search
for optimal 5-ETF portfolio combinations.

---

## What is Aberration?

The Aberration system was developed by Keith Fitschen in **1986**, commercially
released in 1993, and named one of the *"Top Ten Trading Systems of All Time"*
four times by *Futures Truth* magazine.

It is a **breakout trend-following system** built on Bollinger-band-style channels:

| Component    | Formula |
|-------------|----------|
| Middle Band  | SMA(close, N) — default N = 80 |
| Upper Band   | Middle + 2 × StdDev(close, N) |
| Lower Band   | Middle − 2 × StdDev(close, N) |

### Signal Rules (Long-Only / ETF Mode)

| Signal       | Condition |
|-------------|------------|
| **Long Entry**  | close > Upper Band AND volatility filter passes |
| **Long Exit**   | close < Middle Band |

### Signal Rules (Long-Short Mode)

| Signal        | Condition |
|--------------|-----------|
| **Short Entry** | close < Lower Band AND volatility filter passes |
| **Short Exit**  | close > Middle Band |

### Volatility Filter

Prevents trading in abnormally quiet or violently volatile regimes:

```
vol_ratio = ATR(20) / rolling_median(ATR(20), 252 days)
Filter passes when: 0.5 ≤ vol_ratio ≤ 2.0
```

### Execution Model

Signals are generated at the **close of day t** and applied to the **return of
day t+1** (one-bar execution lag). Transaction cost: **0.1% per trade** (one-way).

---

## Project Structure

```
aberration-trading-system/
├── src/
│   ├── aberration.py      # Core strategy (signal generation, returns)
│   ├── data_fetcher.py    # yfinance download + disk cache
│   ├── backtest.py        # Single-ETF and portfolio backtester
│   ├── metrics.py         # CAGR, Sharpe, Sortino, MaxDD, Calmar…
│   └── grid_search.py     # Exhaustive C(n,5) portfolio grid search
├── tradingview/
│   └── aberration_strategy.pine   # TradingView Pine Script v5
├── results/               # CSV outputs and plots (git-ignored)
├── run_grid_search.py     # Main entry point — full grid search
├── run_backtest.py        # Quick backtest for a custom portfolio
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Full ETF Grid Search (find best 5-ETF combo)

```bash
# Classic Aberration parameters, long-only, minimum 10% CAGR
python run_grid_search.py

# Custom date range and CAGR threshold
python run_grid_search.py --start 2012-01-01 --min-cagr 0.12 --top-n 20

# Long-short mode, custom period
python run_grid_search.py --long-short --period 60 --std-mult 2.5

# Disable volatility filter
python run_grid_search.py --no-vol-filter
```

The grid search evaluates **all C(n, 5)** combinations of ETFs from the ~35-ticker
universe. For each combination, an equal-weight portfolio is formed and CAGR,
Sharpe ratio, maximum drawdown, and Calmar ratio are computed.

Results are saved to `results/grid_search_results.csv`.

### 2. Custom Portfolio Backtest

```bash
# Default portfolio: SPY QQQ TLT GLD IEF
python run_backtest.py

# Custom tickers
python run_backtest.py --tickers QQQ XLK XLV TLT GLD

# With equity curve plot (saved to results/)
python run_backtest.py --tickers SPY TLT GLD --plot

# Long-short mode
python run_backtest.py --tickers SPY TLT GLD EEM VNQ --long-short
```

### 3. TradingView Strategy

Open `tradingview/aberration_strategy.pine` in the
[TradingView Pine Editor](https://www.tradingview.com/pine-editor/) and click
**Add to chart**. The strategy supports all key parameters as configurable inputs.

**Recommended TradingView settings:**
- Timeframe: Daily (1D)
- Long Only: ✅ (for ETFs)
- Band Period: 80
- Std Multiplier: 2.0
- Enable Volatility Filter: ✅

---

## ETF Universe

The grid search covers **~35 ETFs** across 7 asset classes:

| Class | Tickers |
|-------|---------|
| US Broad Equity | SPY, QQQ, IWM, VTI, MDY |
| US Sectors | XLK, XLV, XLF, XLE, XLI, XLY, XLP, XLB, XLU, XLRE |
| International | EFA, EEM, EWJ, VEU |
| Fixed Income | TLT, IEF, AGG, LQD, HYG |
| Commodities | GLD, SLV, DBC, DBA, IAU |
| REITs | VNQ |
| Other | BIL, TIP, USO, UNG |

---

## Performance Metrics Computed

| Metric | Description |
|--------|-------------|
| CAGR | Compound Annual Growth Rate |
| Sharpe | Annualised Sharpe (risk-free = 2%) |
| Sortino | Annualised Sortino ratio |
| Max Drawdown | Peak-to-trough decline |
| Calmar | CAGR / |Max Drawdown| |
| Win Rate | Fraction of active-trading days with positive return |
| Trade Count | Number of position changes |
| Avg Trade Duration | Average holding period in trading days |

---

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 80 | SMA and StdDev lookback (Fitschen original) |
| `std_mult_up` | 2.0 | Upper band standard deviation multiplier |
| `std_mult_dn` | 2.0 | Lower band standard deviation multiplier |
| `long_only` | True | ETF mode — no short positions |
| `use_vol_filter` | True | ATR-based volatility gating |
| `atr_period` | 20 | ATR smoothing period |
| `vol_lookback` | 252 | Rolling window for ATR median baseline |
| `min_vol_ratio` | 0.5 | Minimum ATR/median ratio to allow entry |
| `max_vol_ratio` | 2.0 | Maximum ATR/median ratio to allow entry |
| `transaction_cost` | 0.001 | Cost per trade (0.1%, one-way) |

---

## Grid Search Details

With ~35 ETFs in the universe, there are **C(35, 5) ≈ 324,632** combinations.
Each combination is evaluated as an **equal-weight portfolio**.  The search
runs in a single process using vectorised NumPy operations and typically
completes in **2–5 minutes** on a modern CPU.

Results are ranked by **Sharpe ratio** and filtered by a minimum CAGR
threshold (default 10%).

---

## Disclaimer

This implementation is for **educational and research purposes only**.
Past performance does not guarantee future results. Always conduct your
own due diligence before trading.
