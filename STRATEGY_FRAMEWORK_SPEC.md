# Strategy Framework Specification

*Derived from the Aberration ETF Trading System. Use as the canonical template for all future strategies.*

---

## 1. Project Structure

```
<strategy-name>/
├── src/
│   ├── __init__.py          # Exports: Strategy, Params, DataFetcher, Backtester,
│   │                        #          PerformanceMetrics, GridSearch, GridSearchConfig
│   ├── <strategy>.py        # Signal logic only — no I/O, no side effects
│   ├── data_fetcher.py      # Download + disk cache (one CSV per ticker)
│   ├── backtest.py          # run_single / run_portfolio wrappers
│   ├── metrics.py           # Pure functions: CAGR, Sharpe, Sortino, MDD, Calmar
│   └── grid_search.py       # C(n,k) portfolio search with IS/OOS split
├── tradingview/
│   └── <strategy>.pine      # Pine Script v5 mirror of Python logic
├── data/cache/              # Auto-created; one CSV per ticker
├── run_grid_search.py       # CLI entry point
└── requirements.txt
```

---

## 2. Data Layer

### 2.1 Source & Cache

- **Source:** Yahoo Finance via `yfinance`; columns `Open, High, Low, Close, Volume`.
- **Cache:** `data/cache/<TICKER>_daily.csv` — read first, update incrementally when stale (> 5 days behind today).
- **Index:** `pd.DatetimeIndex`, timezone-naive (`tz_localize(None)`).

### 2.2 Universe Filter

Tickers are dropped if they have fewer than `min_years × 252` trading days in the requested date range. Default `min_years = 5`.

### 2.3 Wide Close Matrix

`download_close_prices()` returns a date-aligned `DataFrame` (rows = dates, columns = tickers).

- Keep dates where ≥ `min_data_pct` (default 0.8) of tickers have data.
- Forward-fill NaN up to `ffill_limit` (default 5) days.

---

## 3. Strategy Module (`src/<strategy>.py`)

### 3.1 Params Dataclass

All tuneable constants live in a single `@dataclass` named `<Strategy>Params`. Provide sensible defaults. No mutable objects as defaults (`field(default_factory=...`).

```python
@dataclass
class <Strategy>Params:
    # --- signal parameters ---
    period: int = 80
    ...
    # --- execution ---
    transaction_cost: float = 0.001   # one-way, fraction of trade value
```

### 3.2 Signal Generation — State Machine Rule

`generate_signals(close, high, low) → pd.Series[{-1, 0, 1}]`

- Run a **bar-by-bar state machine** in a NumPy loop (not pandas vectorised) so that position state persists across bars.
- Output value = position held **at the close of that bar**.
- Treat any NaN band/indicator value as "no signal, stay flat".
- Return value:
  - `+1` = long
  - `-1` = short
  - ` 0` = flat

### 3.3 Return Computation — Execution Model (Critical)

`compute_returns(close, high, low) → pd.Series`

**Market-on-close (MOC) execution:**

```
signal[t] is generated at close[t].
The position is filled at close[t] (MOC order).
The position earns the return from close[t] to close[t+1].
```

Implementation via pandas shift:

```python
daily_returns   = close.pct_change()          # return[t] = (close[t] - close[t-1]) / close[t-1]
lagged_signals  = signals.shift(1)            # position held going into day t
strategy_returns = lagged_signals * daily_returns

# Transaction cost on every position change, shifted to match execution day
position_changes = signals.diff().abs()
position_changes.iloc[0] = abs(signals.iloc[0])
costs = position_changes * params.transaction_cost
strategy_returns = strategy_returns - costs.shift(1)
strategy_returns = strategy_returns.fillna(0.0)
```

**Never** use `signals * daily_returns` without the `.shift(1)` — that is look-ahead bias.

---

## 4. Performance Metrics (`src/metrics.py`)

All inputs are **daily arithmetic returns** (`pd.Series`). NaN values are dropped before each calculation.

| Metric | Formula |
|--------|----------|
| **CAGR** | `(∏(1+r))^(252/n) − 1` where `n = len(r.dropna())` |
| **Annual Vol** | `r.std(ddof=1) × √252` |
| **Sharpe** | `(mean(r) − rf/252) / std(r,ddof=1) × √252` |
| **Sortino** | `mean(excess)×252 / (√mean(min(excess,0)²) × √252)` |
| **Max Drawdown** | `min((equity − cummax(equity)) / cummax(equity))` — negative |
| **Calmar** | `CAGR / |MaxDrawdown|` |
| **Win Rate** | `mean(r[r≠0] > 0)` — active (in-market) bars only |
| **Trade Count** | `count(signals.diff() ≠ 0)` |

Risk-free rate default: **2% p.a.** Periods per year: **252**.

`PerformanceMetrics.summary(returns, signals, periods_per_year=252, risk_free_rate=0.02)` returns a dict with all of the above.

---

## 5. Backtester (`src/backtest.py`)

### 5.1 Single-Ticker

`run_single(close, high, low, ticker) → dict`

Returns:
```python
{
    "ticker":       str,
    "returns":      pd.Series,   # daily strategy returns
    "signals":      pd.Series,   # {-1, 0, 1}
    "equity_curve": pd.Series,   # cumprod(1 + returns), starts at 1.0
    "metrics":      dict,        # PerformanceMetrics.summary(...)
}
```

### 5.2 Portfolio

`run_portfolio(close_df, high_df, low_df, weight_method="equal") → dict`

- Runs `run_single` independently on each ticker.
- Equal-weight: `combined_returns = returns_df.mean(axis=1)`.
- Portfolio metrics computed on `combined_returns`.
- Returns the same structure as above, plus `individual_results`, `individual_returns`, `individual_signals`.

**Only equal-weight is implemented.** Do not add other weighting schemes unless explicitly required.

---

## 6. Grid Search (`src/grid_search.py`)

### 6.1 Configuration

```python
@dataclass
class GridSearchConfig:
    min_cagr:         float = 0.10   # IS CAGR floor for a combo to pass
    risk_free:        float = 0.02
    max_results:      int   = 50     # rows returned (top by IS Sharpe)
    n_etfs_per_combo: int   = 5      # k
    transaction_cost: float = 0.001
    train_frac:       float = 0.70   # fraction of dates used for IS
```

### 6.2 Pre-computation

`_precompute_returns(data_dict) → (returns_df: DataFrame, trade_counts: dict)`

- Runs `AberrationStrategy.compute_returns` and `generate_signals` on every ticker **once**.
- **Do NOT call `fillna(0)`** on the returns DataFrame — NaN means "ticker not yet listed"; zero-filling hides this and inflates early combo history.
- Returns `trade_counts` dict so the main loop doesn't need a second pass.

### 6.3 Main Loop — Correctness Rules

```
For each combo of k tickers (index columns col ⊂ [0, n)):

  t0 = max(first_non_nan_row[i] for i in col)   ← inner join start

  if t0 >= train_end or (train_end - t0) < 252:
      skip                                        ← need ≥ 1 year of joint IS history

  port_is = ret_matrix[t0:train_end, col][valid rows].mean(axis=1)
  T_eff_is = len(port_is)                        ← ACTUAL joint trading days

  CAGR_is = prod(1 + port_is)^(252 / T_eff_is) - 1   ← annualise over T_eff, NOT global T
```

**Never** annualise CAGR over the global date range `T`. Use `T_eff` (actual joint trading days) per combo.

### 6.4 In-Sample / Out-of-Sample Split

```
IS  : dates[0 : train_end]    (first train_frac of all dates, default 70%)
OOS : dates[train_end :]      (remaining 30%)
```

- **Selection criterion:** IS Sharpe (only IS metrics are used to rank).
- **OOS metrics** (`oos_cagr`, `oos_sharpe`, `oos_max_drawdown`) are computed and reported **but never used to filter or rank**. They exist for honest forward-looking validation.
- OOS is skipped for a combo if `T_eff_oos < 60` days.
- Expose `train_start_date`, `train_end_date`, `oos_start_date`, `oos_end_date` as instance attributes on `GridSearch` after `run()`.

### 6.5 Result Schema

| Column | Type | Description |
|--------|------|-------------|
| `etfs` | tuple[str] | Ticker symbols in this combo |
| `is_cagr` | float | In-sample CAGR |
| `is_sharpe` | float | In-sample Sharpe (sort key) |
| `oos_cagr` | float\|NaN | Out-of-sample CAGR |
| `oos_sharpe` | float\|NaN | Out-of-sample Sharpe |
| `oos_max_drawdown` | float\|NaN | OOS maximum drawdown (negative) |
| `max_drawdown` | float | IS maximum drawdown (negative) |
| `calmar` | float | IS Calmar ratio |
| `annual_vol` | float | IS annualised volatility |
| `n_trades_avg` | float | Mean trade count across k tickers |

Sorted descending by `is_sharpe`. Capped at `max_results` rows.

---

## 7. CLI Entry Point (`run_grid_search.py`)

Must expose at minimum:

```
--start        Start date (default 2010-01-01)
--end          End date (default today)
--period       Band/indicator period
--train-frac   IS fraction (default 0.70)
--top-n        Rows to display
--refresh      Force re-download (bypass cache)
```

Print flow:
1. Banner with key parameters and IS/OOS ratio.
2. After `gs.run()`, print the IS and OOS date ranges.
3. Results table via `GridSearch.print_results()` (NaN-safe formatting).
4. Best combo summary with both IS and OOS metrics.
5. Detailed `Backtester.run_portfolio()` on the best combo (full date range, inner join).

---

## 8. TradingView Pine Script (`tradingview/<strategy>.pine`)

### 8.1 Mandatory Settings

```pine
//@version=5
strategy(
    ...
    calc_on_every_tick      = false,
    process_orders_on_close = true   // ← must match Python's MOC model
)
```

`process_orders_on_close = true` is required so fills happen at the signal bar's close, matching the Python `.shift(1)` approach. Setting it to `false` would fill at the next bar's open — a different execution model.

### 8.2 Structure

Mirror Python exactly:
- Band / indicator calculations
- Volatility or regime filter
- Entry / exit conditions using `strategy.position_size` for state
- `strategy.entry()` / `strategy.close()` calls
- `plot()` for bands; `plotshape()` for signals
- Info table at `barstate.islast` showing key parameters and live state

### 8.3 Input Groups

Organise inputs into named `group` strings: `"Core Parameters"`, `"Volatility Filter"`, `"Display"` (or equivalent). Every input must have a `tooltip`.

---

## 9. Anti-Snooping Rules

These rules are non-negotiable for honest results:

1. **No OOS data in selection.** All ranking and filtering uses IS metrics only.
2. **No global-T annualisation.** CAGR exponent uses `T_eff` (joint trading days per combo), not the length of the full date range.
3. **No zero-fill for missing history.** NaN = ticker not yet listed. Zero-filling inflates a combo's IS history and understates risk.
4. **No post-hoc parameter tuning on OOS.** If parameters are adjusted after seeing OOS results, the OOS period must be discarded and a new holdout window defined.
5. **Report OOS honestly.** If OOS CAGR < IS CAGR (almost always true), report both. Do not hide or suppress OOS figures.
6. **No look-ahead in signals.** `signals.shift(1)` in `compute_returns` is mandatory. Any indicator that uses `close[t]` to generate a fill at `open[t]` is look-ahead bias.
7. **Transaction costs always on.** Default 0.1% one-way. Never set to zero for "clean" results.

---

## 10. Testing & Verification

### Minimum Smoke Test

Given a synthetic price series with known properties:

1. Run `generate_signals` → verify no signal fires before the warmup period ends.
2. Run `compute_returns` with a flat line → verify returns = 0 (no spurious P&L).
3. Introduce a step-up trend → verify a long signal fires and earns the correct return.
4. Run `GridSearch.run()` on 6 synthetic tickers (k=2) → verify IS CAGR matches an independent `Backtester.run_portfolio()` on the IS slice with the same inner-join start.

### IS/OOS Consistency Check

For the top combo from `GridSearch`:
- Run `Backtester.run_portfolio()` on the IS date slice (inner-joined from `t0`).
- Compare `Backtester` CAGR to `gs_results.iloc[0]["is_cagr"]`.
- **They must match to 4 decimal places.** Any divergence indicates a bug in annualisation, date slicing, or NaN handling.

---

## 11. Dependencies

```
yfinance>=0.2
pandas>=2.0
numpy>=1.24
tqdm            # optional; progress bar in grid search
```

No ML libraries. No scipy (use integer arithmetic for C(n,k)).
