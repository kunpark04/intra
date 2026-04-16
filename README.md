# MNQ 1-Minute Z-EMA Mean Reversion Bot

A **backtest-first** algorithmic trading system for **MNQ futures** on **1-minute bars**, implementing an explosive mean-reversion strategy: Z-score band stretch + reversal confirmed by a sensitive EMA regime filter.

## Strategy Overview

**Type**: Mean reversion with EMA regime confirmation

**Instrument**: MNQ (Micro Nasdaq Futures) — $2/point per contract, tick = 0.25 pts

**Signal logic**:
- **Long**: Z-score ≤ −k (stretched below band) + reversal tick up + `ema_fast > ema_slow`
- **Short**: Z-score ≥ +k (stretched above band) + reversal tick down + `ema_fast < ema_slow`
- **Entry fill**: next-bar open after signal confirmation

**Indicators**:
| Indicator | Default |
|-----------|---------|
| Z-score window | 20 bars |
| Z-score band k | 1.5–2.5 (configurable) |
| EMA fast | 3 bars |
| EMA slow | 6 bars |

**Risk model**:
- Starting equity: $50,000
- Risk per trade: 5% of equity ($2,500 at start)
- Minimum R:R: configurable per combo (data-driven — sweeps v4–v10 and the
  V2-filtered ML#1 surrogate both prefer R:R near 1:1 under MNQ economics;
  the 1:3 default from early iterations is no longer a floor)
- One position at a time per combo; portfolio runs are sized independently

See [`STRATEGY.md`](STRATEGY.md) for the full specification.

## Folder Structure

```
intra/
  data/
    NQ_1min.csv                         # raw 1-min bars (not committed — too large)
    NQ_1min_gaps.csv                    # session-break reference
    ml/                                 # parameter-sweep + ML artifacts (gitignored)
      originals/                        # original sweep parquets (no MFE/MAE)
      mfe/                              # MFE/MAE-enriched re-runs (adaptive R:R input)
      adaptive_rr_v1/                   # V1 trade-grain model + per-R:R calibrators
      adaptive_rr_v2/                   # V2 booster (production ML#2)
      adaptive_rr_v3/                   # V3 booster + pooled per-R:R isotonic (current)
      ml1_results/                      # ML#1 combo-grain surrogate outputs
      strategy.db                       # sqlite backfill of combos/trades
  src/
    config.py                           # all tunable parameters
    data_loader.py                      # CSV → canonical DataFrame
    indicators/                         # ema, zscore, zscore_variants, atr, pipeline
    strategy.py                         # signal generation
    risk.py                             # position sizing
    backtest.py                         # bar-by-bar simulation (Numba + NumPy fallback)
    reporting.py                        # artifact writers
    scoring.py                          # entry-quality analytic diagnostics
    io_paths.py                         # path constants
    cython_ext/backtest_core.pyx        # AOT-compiled backtest loop (fastest tier)
  scripts/
    run_backtest.py                     # single-version iteration runner
    param_sweep.py                      # Track B training-data generator
    gen_analysis_notebook.py            # writes 7-cell analysis.ipynb per iteration
    exec_analysis.py                    # executes analysis.ipynb in-place
    create_notebooks.py                 # bootstrap notebook
    rerun_and_benchmark.py              # reruns V1/V2 across Cython/Numba/NumPy tiers
    models/                             # ML training + inference (V1/V2/V3 + heldout/monotonic, ml1, inference_v3)
    calibration/                        # isotonic + per-combo calibration
    backtests/                          # filter / Kelly / adaptive-vs-fixed / constrained-band
    evaluation/                         # walk-forward, held-out, final hold-out, portfolio sim
    analysis/                           # SHAP, permutation tests, feature engineering, combo features
    runners/                            # paramiko+screen launchers for sweep-runner-1 jobs
    data_pipeline/                      # init_db, sweep_status, validate_mfe, rr cache
  notebooks/
    01_backtest_and_log.ipynb           # run backtest, write artifacts
  iterations/
    V1/ V2/ V3/                         # training backtest artifacts (one folder per version)
      metadata.json
      trades.csv                        # ML-ready, full feature schema
      trader_log.csv                    # human-readable summary
      daily_ledger.csv                  # includes no-trade days
      equity_curve.csv
      monte_carlo.json                  # distributional risk metrics
      analysis.ipynb                    # 7-cell matplotlib analysis
  evaluation/                           # single final hold-out run (no Vn subfolders)
    metadata.json                       # includes source_iteration: "Vn"
    trades.csv
    trader_log.csv
    daily_ledger.csv
    equity_curve.csv
    monte_carlo.json
    analysis.ipynb
  tasks/                                # plans + findings (ML analysis, Part B, v3 follow-up)
  setup_cython.py                       # builds backtest_core.pyx via MSVC/GCC
  STRATEGY.md                           # strategy specification
  LOG_SCHEMA.md                         # ML-ready log schema
  CLAUDE.md                             # implementation contract for Claude Code
  lessons.md                            # failure log and prevention rules
```

## Data

The primary input is `data/NQ_1min.csv` (NQ 1-minute OHLCV bars, ~233 MB — not committed).

Expected columns: `Time`, `Open`, `High`, `Low`, `Latest` (close), `Volume`, `session_break`

The loader normalizes these to: `time`, `open`, `high`, `low`, `close`, `volume`, `session_break`

**Train/test split**: chronological 80/20 — no shuffling. All iteration runs use training bars only; evaluation is a single final hold-out run.

## Output Artifacts

Each training iteration `Vn` produces:

| File | Description |
|------|-------------|
| `trades.csv` | ML-ready trade log with full feature + label schema (includes `entry_score`, `mfe_mae_ratio`) |
| `trader_log.csv` | Human-readable trades-only summary |
| `daily_ledger.csv` | Calendar coverage including no-trade days |
| `equity_curve.csv` | Equity path per bar |
| `monte_carlo.json` | Bootstrapped risk metrics + permutation test on win rate |
| `analysis.ipynb` | 7-cell matplotlib analysis (equity vs. S&P, drawdown, MFE/MAE, score buckets, heatmap) |
| `metadata.json` | All parameters + data fingerprint |

All visualisations use matplotlib and render inline in `analysis.ipynb` — Plotly HTML files are no longer produced.

See [`LOG_SCHEMA.md`](LOG_SCHEMA.md) for exact column definitions and ML leakage rules.

## Monte Carlo Risk Metrics

Each iteration runs a bootstrap Monte Carlo (seeded, reproducible) on trade outcomes to produce distributional estimates:

- Max drawdown distribution (p50, p90, p95, p99)
- VaR / CVaR on trade or daily returns
- Risk-of-ruin probability (equity ≤ ruin threshold)

## Setup

```bash
pip install pandas numpy matplotlib yfinance ipykernel nbformat nbclient pyarrow numba lightgbm scikit-learn
```

Optionally build the Cython extension for the fastest backtest tier:

```bash
python setup_cython.py build_ext --inplace
```

Run a backtest (from repo root):

```bash
python scripts/run_backtest.py
```

Or use the notebook interactively — it must be run from the repo root and writes artifacts to `iterations/` automatically.

## Performance Design

Three-tier dispatch in priority order (see `src/backtest.py`):

1. **Cython AOT** (`src/cython_ext/backtest_core.pyx`) — MSVC/GCC-compiled; fastest for repeated runs.
2. **Numba JIT** (`src/backtest.py`) — compiled on first call; good for development.
3. **NumPy pure Python** — fallback when neither is available.

**Parquet** is used for all large intermediate data (bars + indicators, ML sweep datasets, MFE/MAE re-runs).

## ML Pipeline (Track B)

The repo ships with two trained surrogate/prediction models:

- **ML#1** (`scripts/models/ml1_surrogate.py`): LightGBM combo-grain surrogate trained on sweep outcomes (one row per combo). Predicts per-combo Sharpe/return from static parameters.
- **ML#2** (`scripts/models/adaptive_rr_model_v{1,2,3}.py`): trade-grain adaptive R:R model. **Current production stack is V3**: booster + pooled per-R:R isotonic calibrator + fixed 5% sizing (per-combo two-stage calibrator was deprecated in Phase 5D — see `tasks/part_b_findings.md`).

Inference helpers: `scripts/models/inference_v3.py` exposes `predict_pwin_v3()` for downstream filter/portfolio backtests.

## References

- [`STRATEGY.md`](STRATEGY.md) — full signal, entry/exit, and risk specification
- [`LOG_SCHEMA.md`](LOG_SCHEMA.md) — exact column schema for all output logs
- [`CLAUDE.md`](CLAUDE.md) — implementation contract, folder layout, non-negotiable constraints
- [`lessons.md`](lessons.md) — persistent log of tooling failures and prevention rules
- [`tasks/part_b_findings.md`](tasks/part_b_findings.md) — ML Phase 3–5 experimental findings and Phase 5D verdict
- [`tasks/ml_full_analysis.md`](tasks/ml_full_analysis.md) — synthesized ML#1 + ML#2 cross-analysis and decision framework
