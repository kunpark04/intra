## Purpose

This repository is a **backtest-first** trading bot project for **MNQ economics** on **1-minute bars**, using an **explosive / high-frequency** mean-reversion strategy: **Z-score band stretch + reversal**, confirmed by a **sensitive EMA regime filter**.

This file is the **implementation contract** for Claude Code (and contributors): folder layout, what to build, required artifacts, and non-negotiable constraints.

The strategy rules themselves are defined in [`STRATEGY.md`](STRATEGY.md). The exact ML-ready log schema is defined in [`LOG_SCHEMA.md`](LOG_SCHEMA.md).

## Non-negotiable constraints

- **Timeframe**: 1-minute.
- **Instrument economics**: compute PnL/sizing using **MNQ** (\($2 per full index point per contract\); tick = 0.25 points = $0.50).
  - Data is NQ price series; treat it as the underlying price path but use MNQ $/point for dollars.
- **Starting equity**: **$2000**.
- **Risk model**: risk **5% of equity** per open position/trade (\($100 at start\)).
- **Minimum R:R**: **1:2** (planned reward distance must be ≥ 2× planned risk distance).
- **Backtest mode first**: use CSV input; no live broker integration yet.
- **Notebooks run in-place**: notebooks must run from repo root and write artifacts to the correct folders without manual path edits.

## Performance requirements (Parquet + Numba)

- **Parquet**: prefer Parquet for large intermediate data (bars with indicators, large logs, and any ML datasets). CSV remains acceptable for human-facing outputs (e.g. `trader_log.csv`).
- **Numba**: performance-critical loops (e.g. bar-by-bar backtest loop, Monte Carlo simulation, and potentially indicator loops) should be written in a Numba-friendly way where practical (NumPy arrays, typed loops, minimal Python objects).
  - Keep a pure-Python fallback if Numba is unavailable, but design for speed.

## Data inputs

Primary CSV file:

- `data/NQ_1min.csv`

Expected columns (from provided dataset):

- `Time` (parseable datetime)
- `Open`, `High`, `Low`
- `Latest` (treat as **close**)
- `Volume`
- `session_break` (boolean-like)

Loader must normalize to canonical names:

- `time`, `open`, `high`, `low`, `close`, `volume`, `session_break`

## Train / test split policy

Chronological split on the full bar series \(no shuffling\):

- Let bars be indexed `t = 0 .. T-1` after sorting by time.
- **Training** partition: `t < floor(0.8 * T)`.
- **Test** partition: remaining bars.

Important workflow rule:

- **Iterations are training-set backtests only** (you will run many).
- **Evaluation** is a **single final hold-out** run on the test partition for the most refined version.

## Folder architecture (source of truth)

```text
intra/
  data/
    NQ_1min.csv
    NQ_1min_gaps.csv
  src/
    __init__.py
    config.py
    data_loader.py
    indicators.py
    strategy.py
    risk.py
    backtest.py
    reporting.py
    io_paths.py
  scripts/
    run_backtest.py
  notebooks/
    01_backtest_and_log.ipynb
    02_plotly_exploration.ipynb
  iterations/
    .gitkeep
    V1/
      metadata.json
      trades.csv
      trader_log.csv
      daily_ledger.csv
      equity_curve.csv
      plotly_price_indicators.html
  evaluation/
    .gitkeep
    metadata.json           # includes source_iteration: Vn
    trades.csv
    trader_log.csv
    daily_ledger.csv
    equity_curve.csv
    plotly_price_indicators.html
    monte_carlo.json        # optional summary of evaluation risk metrics
  requirements.txt
  .gitignore
  README.md
  CLAUDE.md
  STRATEGY.md
  LOG_SCHEMA.md
  lessons.md
```

### Purpose of key folders

- `src/`: reusable core logic (data loading, indicators, strategy, risk, backtest, reporting).
- `scripts/`: command-line entry points (repeatable runs without notebooks).
- `notebooks/`: interactive analysis; must run from repo root and write outputs to `iterations/` and `evaluation/`.
- `iterations/`: generated training backtest artifacts, **one folder per version** (`V1/`, `V2/`, …).
- `evaluation/`: generated artifacts for the **single final** hold-out evaluation (no `Vn` subfolders).

## Required outputs (artifacts)

For each training iteration version `Vn`:

- `iterations/Vn/metadata.json`
- `iterations/Vn/trades.csv` (ML-ready schema)
- `iterations/Vn/trader_log.csv` (simple, trades-only)
- `iterations/Vn/daily_ledger.csv` (includes no-trade days)
- `iterations/Vn/equity_curve.csv` (optional but recommended)
- `iterations/Vn/plotly_price_indicators.html` (Plotly interactive)
- `iterations/Vn/monte_carlo.json` (required; robust Monte Carlo risk metrics)

For the final hold-out evaluation:

- `evaluation/metadata.json` with at least:
  - `source_iteration: "Vn"` indicating which iteration version is evaluated
  - all strategy/config parameters used
  - data range and split indices
- `evaluation/trades.csv`, `evaluation/trader_log.csv`, `evaluation/daily_ledger.csv`, `evaluation/equity_curve.csv`, `evaluation/plotly_price_indicators.html`
- `evaluation/monte_carlo.json` (optional; same schema as iteration Monte Carlo summary)

## Monte Carlo simulation (required per iteration)

Each iteration `Vn` must run a robust Monte Carlo procedure on the **training backtest outcomes** to estimate risk metrics distributionally (not single-point estimates). Save summary results as:

- `iterations/Vn/monte_carlo.json`

Recommended approach (implementation can evolve, but must be deterministic with a seed):

- Use bootstrapping / resampling of trade outcomes (e.g. `r_multiple` or `net_pnl_dollars`) to create many simulated equity paths.
- Preserve or optionally model autocorrelation (block bootstrap) if desired; document the chosen method in `metadata.json`.

Minimum metrics to report (point estimates + percentiles):

- `n_trades`, `n_sims`, `seed`
- distribution of max drawdown (e.g. p50, p90, p95, p99)
- probability of drawdown exceeding thresholds (configurable)
- VaR / CVaR on trade returns or daily returns (define horizon explicitly)
- risk-of-ruin estimate (define ruin threshold; e.g. equity <= 0 or X% drawdown)

## Claude iterative learning (mistakes log)

To help Claude Code avoid repeating command/tooling mistakes, maintain a persistent, human-readable log at:

- `lessons.md`

Policy:

- When a command or workflow fails (bad flags, wrong working directory, missing deps, etc.), record:
  - what was attempted
  - the error message summary
  - the fix
  - the new rule/prevention

This file is intended to be referenced before re-running similar commands.

## Reporting & styling requirements

- **Win/Loss row shading**: in notebooks, trades tables should be styled:
  - winning trades: green row background
  - losing trades: red row background
  - zero PnL: neutral
- Because CSV cannot store colors, additionally export **HTML** if you want preserved coloring:
  - `trader_log.html` and/or `trades_styled.html` (optional; recommended for human review)

## Plotly requirements

Create an interactive Plotly visualization over raw price data + indicators:

- Price: candlestick \(or close line\)
- EMA fast/slow overlays
- Z-score subplot with horizontal lines at `±k`
- Zoom + rangeslider enabled
- Save HTML:
  - training chart → `iterations/Vn/plotly_price_indicators.html`
  - evaluation chart → `evaluation/plotly_price_indicators.html`

## Dependencies (Python)

At minimum:

- `pandas`, `numpy`

For notebooks + charts:

- `plotly`, `ipykernel`, `nbformat`

Optional:

- `pyarrow` (Parquet read/write)
- `numba` (performance acceleration)
- `matplotlib` (quick equity chart; not required if Plotly used)

## Backtest execution model (implementation guidance)

- Prefer deterministic bar-by-bar simulation.
- Entry execution must be consistent:
  - either **next-bar open** fills (recommended for reproducibility) or close-of-signal-bar; pick one and document.
- Stops/targets:
  - must enforce minimum 1:2 planned R:R at order creation.
  - if both SL and TP are touched in the same bar, define and document a deterministic rule (used for `label_hit_tp_first`).
- One position at a time initially.
- Commission/slippage may start as constants (0 by default) but must be represented in logs as separate fields.

## Future scope (do not implement yet)

- Live volume “bubbles”
- DOM / Level 2 features
- Broker execution adapters

