## Purpose

This repository is a **backtest-first** trading bot project for **MNQ economics** on **1-minute bars**, using an **explosive / high-frequency** mean-reversion strategy: **Z-score band stretch + reversal**, confirmed by a **sensitive EMA regime filter**.

This file is the **implementation contract** for Claude Code (and contributors): folder layout, what to build, required artifacts, and non-negotiable constraints.

The strategy rules themselves are defined in [`STRATEGY.md`](STRATEGY.md). The exact ML-ready log schema is defined in [`LOG_SCHEMA.md`](LOG_SCHEMA.md).

## Non-negotiable constraints

- **Timeframe**: 1-minute.
- **Instrument economics**: compute PnL/sizing using **MNQ** (\($2 per full index point per contract\); tick = 0.25 points = $0.50).
  - Data is NQ price series; treat it as the underlying price path but use MNQ $/point for dollars.
- **Starting equity**: **$50,000**.
- **Risk model**: risk **5% of equity** per open position/trade.
- **Minimum R:R**: **1:3** (planned reward distance must be ≥ 3× planned risk distance).
- **Backtest mode first**: use CSV input; no live broker integration yet.
- **Notebooks run in-place**: notebooks must run from repo root and write artifacts to the correct folders without manual path edits.

## Performance requirements (Cython + Numba)

The backtest engine uses a **three-tier dispatch** in order of priority:

1. **Cython AOT** (`src/cython_ext/backtest_core.pyx`) — compiled with MSVC via `setup_cython.py`; fastest for repeated runs.
2. **Numba JIT** (`src/backtest.py`) — compiled on first call; good for development.
3. **NumPy pure Python** — fallback when neither is available.

Build Cython extension:

```bash
python setup_cython.py build_ext --inplace
```

- **Parquet**: prefer Parquet for large intermediate data (bars with indicators, large logs, and any ML datasets). CSV remains acceptable for human-facing outputs (e.g. `trader_log.csv`).

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
    scoring.py
    io_paths.py
    cython_ext/
      __init__.py
      backtest_core.pyx
      backtest_core.c        (generated; gitignored)
      backtest_core.*.pyd    (compiled; gitignored)
  scripts/
    run_backtest.py
    gen_analysis_notebook.py
    exec_analysis.py
    rerun_and_benchmark.py
  notebooks/
    01_backtest_and_log.ipynb
  iterations/
    .gitkeep
    V1/
      metadata.json
      trades.csv
      trader_log.csv
      daily_ledger.csv
      equity_curve.csv
      monte_carlo.json
      analysis.ipynb
    V2/ ... (same structure)
    V3/ ... (same structure)
  evaluation/
    .gitkeep
    metadata.json           # includes source_iteration: Vn
    trades.csv
    trader_log.csv
    daily_ledger.csv
    equity_curve.csv
    monte_carlo.json
    analysis.ipynb
  setup_cython.py
  requirements.txt
  .gitignore
  README.md
  CLAUDE.md
  STRATEGY.md
  LOG_SCHEMA.md
  lessons.md
```

### Purpose of key folders

- `src/`: reusable core logic (data loading, indicators, strategy, risk, backtest, reporting, scoring).
- `src/cython_ext/`: Cython AOT extension module for the bar-by-bar backtest core.
- `scripts/`: command-line entry points (repeatable runs without notebooks).
  - `gen_analysis_notebook.py`: generates 7-cell `analysis.ipynb` for each iteration folder.
  - `exec_analysis.py`: executes `analysis.ipynb` in-place via nbclient with correct cwd.
  - `rerun_and_benchmark.py`: reruns V1/V2 and benchmarks all three engine tiers.
- `notebooks/`: interactive analysis; must run from repo root and write outputs to `iterations/` and `evaluation/`.
- `iterations/`: generated training backtest artifacts, **one folder per version** (`V1/`, `V2/`, …).
- `evaluation/`: generated artifacts for the **single final** hold-out evaluation (no `Vn` subfolders).

## Required outputs (artifacts)

For each training iteration version `Vn`:

- `iterations/Vn/metadata.json`
- `iterations/Vn/trades.csv` (ML-ready schema; includes `entry_score`, `mfe_mae_ratio`)
- `iterations/Vn/trader_log.csv` (simple, trades-only)
- `iterations/Vn/daily_ledger.csv` (includes no-trade days)
- `iterations/Vn/equity_curve.csv`
- `iterations/Vn/monte_carlo.json` (required; IID bootstrap + permutation test on win rate)
- `iterations/Vn/analysis.ipynb` (7-cell notebook; generated by `scripts/gen_analysis_notebook.py`)

For the final hold-out evaluation:

- `evaluation/metadata.json` with at least:
  - `source_iteration: "Vn"` indicating which iteration version is evaluated
  - all strategy/config parameters used
  - data range and split indices
- `evaluation/trades.csv`, `evaluation/trader_log.csv`, `evaluation/daily_ledger.csv`, `evaluation/equity_curve.csv`
- `evaluation/monte_carlo.json` (optional; same schema as iteration Monte Carlo summary)
- `evaluation/analysis.ipynb`

**Note**: Plotly HTML files are no longer produced. All visualisations use matplotlib and are rendered inline in `analysis.ipynb`.

## Analysis notebook structure (`analysis.ipynb`)

7-cell matplotlib notebook generated per iteration by `scripts/gen_analysis_notebook.py`:

| Cell | Content |
|------|---------|
| 0 | Equity curve vs S&P 500 (yfinance `^GSPC`, normalised to same start equity) |
| 1 | Drawdown (%) curve vs S&P 500 |
| 2 | Key metrics table vs S&P 500: total return, annualised return, Sharpe, max drawdown, win rate, trades, exit breakdown |
| 3 | Monte Carlo risk table + permutation test on win rate |
| 4 | Track A: MFE/MAE ratio distribution, wins vs losses overlay |
| 5 | Track A: Entry score distribution + win rate by score quartile bar chart |
| 6 | Track A: Win-rate bucket heatmap (|z-score| quartile × volume quartile) |

Execute notebooks:

```bash
python scripts/gen_analysis_notebook.py V1 V2 V3
python scripts/exec_analysis.py V1 V2 V3
```

## Monte Carlo simulation (required per iteration)

Each iteration `Vn` must run a robust Monte Carlo procedure on the **training backtest outcomes** to estimate risk metrics distributionally. Saved as `iterations/Vn/monte_carlo.json`.

Current implementation (`src/reporting.py`):

- **IID bootstrap** on `net_pnl_dollars` (`MC_BOOTSTRAP = "iid"`), 10,000 sims (`MC_N_SIMS`), seeded (`MC_SEED = 42`).
- Max drawdown distribution: p50, p90, p95, p99, worst.
- VaR / CVaR at 5th percentile trade level.
- Risk-of-ruin probability (ruin = max drawdown ≥ 50% of starting equity).
- **Permutation test on win rate** embedded under `"permutation_test"` key:
  - H0: strategy wins at break-even rate `1 / (1 + avg_rr_planned)`
  - H1: win rate > break-even (one-tailed)
  - Significance flags: `significant_05`, `significant_01`

## Track A: Entry quality analytics

All trades in `trades.csv` include two analytical diagnostics (not ML features):

- **`entry_score`** ∈ [0, 1]: weighted composite of z-score stretch, volume, EMA spread, bar body, session time. See `src/scoring.py`.
- **`mfe_mae_ratio`**: `mfe_points / abs(mae_points)` — cleaner entries have higher ratios.

Score weights are configurable in `src/config.py` via `SCORE_W_*` variables.

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

## Dependencies (Python)

At minimum:

- `pandas`, `numpy`

For notebooks + charts:

- `matplotlib`, `yfinance`, `ipykernel`, `nbformat`, `nbclient`

Optional:

- `pyarrow` (Parquet read/write)
- `numba` (performance acceleration)
- `cython` (AOT compilation; requires MSVC on Windows or GCC on Linux/Mac)

## Backtest execution model (implementation guidance)

- Z-score band default: **`k = 2.5`** (configurable; must be stored in `metadata.json` and logged as `z_band_k` in `trades.csv`).
- `SIGNAL_MODE`: `"ema_crossover"` (V1/V2) or `"zscore_reversal"` (V3+, default).
- Prefer deterministic bar-by-bar simulation.
- Fill model: **next-bar open** (`FILL_MODEL = "next_bar_open"`).
- Stops/targets:
  - must enforce minimum `MIN_RR` planned R:R at order creation.
  - if both SL and TP are touched in the same bar: `SAME_BAR_COLLISION = "tp_first"` (logged as `label_hit_tp_first`).
- One position at a time initially.
- Commission/slippage may start as constants (0 by default) but must be represented in logs as separate fields.

## Future scope (do not implement yet)

- Track B: ML pipeline — train classifier on `trades.csv` features to predict `label_win`
- Parameter sweep for ML training data diversity
- Live volume "bubbles"
- DOM / Level 2 features
- Broker execution adapters
