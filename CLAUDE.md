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
- **Minimum R:R**: configurable per combo (was originally set at 1:3 as a
  default). The 1:3 floor is **not** a non-negotiable — parameter sweeps
  (v4–v10) and the retrained V2-filtered ML#1 surrogate (see
  `tasks/part_b_findings.md`) both prefer R:R near 1:1 under MNQ economics,
  and the calibrated ML#2 model (`adaptive_rr_v3` — current production)
  confirms this. Use whatever R:R the data supports for the chosen combo.
- **ML#2 production stack** — 🛑 **REVOKED 2026-04-21 UTC**. The V3
  LightGBM booster + pooled per-R:R isotonic at `data/ml/adaptive_rr_v3/`
  was shown to depend on a `global_combo_id` memorization channel (same
  leak pattern as V4, revoked 2026-04-21 00:34 UTC). Combo-agnostic
  refit (`data/ml/adaptive_rr_v3_no_gcid/`, OOF AUC 0.8293 — identical
  to shipped V3) produces s6_net Sharpe p50 **0.31** / ruin **53.62%** /
  pos_prob **64%**, a catastrophic collapse from shipped V3's 1.78 /
  6.93% / 98.7%. Both V3 and V4 are out of production. See
  `tasks/v3_no_gcid_audit_verdict.md` and `tasks/ship_decision.md`
  V3 revocation banner (2026-04-21 04:00 UTC). Paper-trading halted.
  The per-combo two-stage calibrator was **also deprecated** (Phase 5D
  post-mortem, Kelly+two-stage failed the B16 gate with 91% DD in
  portfolio sim). Do not reintroduce either stack without first passing
  a combo-agnostic audit of comparable design. See
  `tasks/part_b_findings.md` Phase 5D + the Phase 3 V3 verdict above.
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
    ml/                               # parameter-sweep ML datasets (Parquet)
      originals/                      # original sweep outputs (no MFE/MAE)
        ml_dataset_v{N}.parquet       # one per sweep version (v2–v10)
        ml_dataset_v{N}_manifest.json # per-combo completion manifest
      mfe/                            # MFE/MAE-enriched re-runs (adaptive R:R input)
        ml_dataset_v{N}_mfe.parquet
        ml_dataset_v{N}_mfe_manifest.json
      ml1_results/                    # ML#1 combo-grain surrogate outputs
      adaptive_rr_v1/                 # adaptive R:R LightGBM model + artifacts
      strategy.db                     # sqlite backfill of combos/trades
  src/
    __init__.py
    config.py
    data_loader.py
    indicators/                       # one module per indicator + pipeline wrapper
      __init__.py
      ema.py
      zscore.py
      zscore_variants.py              # generalised z-score (input/anchor/denom/type)
      atr.py
      pipeline.py                     # add_indicators() — attaches all indicators
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
    run_backtest.py                   # single-version iteration runner
    gen_analysis_notebook.py          # writes 7-cell analysis.ipynb per iteration
    exec_analysis.py                  # executes analysis.ipynb in-place
    create_notebooks.py
    rerun_and_benchmark.py            # reruns V1/V2 across Cython/Numba/NumPy tiers
    param_sweep.py                    # parameter sweep → data/ml/originals/ml_dataset_vN.parquet (or mfe/ suffix variant)
    models/                           # ML training + inference (adaptive_rr_model_v{1,2,3}, heldout/monotonic variants, ml1_surrogate, inference_v3)
    calibration/                      # isotonic + per-combo calibration (recal_*, per_combo_*, export_calibrators_v3)
    backtests/                        # filter / kelly / adaptive_vs_fixed / constrained_band backtests
    evaluation/                       # walk_forward_v2, heldout_time_eval_v2, final_holdout_eval_v3, portfolio_sim_v3
    analysis/                         # shap_audit_v2, permutation_test_*, validate_top_combos_ml1, feature_engineering_v2, build_combo_features_ml1
    runners/                          # `run_*` orchestrators that upload+launch remote jobs
    data_pipeline/                    # init_db, sweep_status, validate_mfe_parquets, build_adaptive_rr_cache_v2
  notebooks/
    01_backtest_and_log.ipynb
  iterations/
    V1/ V2/ V3/                       # same artifact set per version (see below)
      metadata.json
      trades.csv
      trader_log.csv
      daily_ledger.csv
      equity_curve.csv
      monte_carlo.json
      analysis.ipynb
  evaluation/
    .gitkeep
    metadata.json           # includes source_iteration: Vn
    trades.csv
    trader_log.csv
    daily_ledger.csv
    equity_curve.csv
    monte_carlo.json
    analysis.ipynb
  tasks/
    plan.md                           # V1 build plan (historical)
    v10_sweep_monitor_plan.md         # latest sweep launch + monitor protocol
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
- `src/indicators/`: one module per indicator (`ema.py`, `zscore.py`, `zscore_variants.py`, `atr.py`) plus `pipeline.py` which exposes `add_indicators()` for DataFrame enrichment.
- `src/cython_ext/`: Cython AOT extension module for the bar-by-bar backtest core.
- `scripts/`: command-line entry points (repeatable runs without notebooks). Canonical top-level entry points:
  - `run_backtest.py`: runs a single iteration version end-to-end.
  - `gen_analysis_notebook.py`: generates 7-cell `analysis.ipynb` for each iteration folder.
  - `exec_analysis.py`: executes `analysis.ipynb` in-place via nbclient with correct cwd.
  - `rerun_and_benchmark.py`: reruns V1/V2 and benchmarks all three engine tiers.
  - `create_notebooks.py`: bootstraps iteration notebooks.
  - `param_sweep.py`: parameter sweep for Track-B ML training data; writes one row per closed trade to `data/ml/originals/ml_dataset_v{N}.parquet` (or `data/ml/mfe/ml_dataset_v{N}_mfe.parquet` for MFE re-runs). Supports `--range-mode` `default | winrate | zscore_variants | v4 | v5 | v6 | v7 | v8 | v9 | v10`.
  - `scripts/models/`: ML training + inference (`adaptive_rr_model_v{1,2,3}.py`, `adaptive_rr_model_heldout_v2.py`, `adaptive_rr_model_monotonic_v2.py`, `ml1_surrogate.py`, `inference_v3.py`). Sibling modules import each other directly via `sys.path.insert(repo / "scripts" / "models")`.
  - `scripts/calibration/`: isotonic + per-combo calibration (`recal_robustness_v3`, `recal_window_compare_v3`, `rolling_recal_v3`, `per_combo_{ece_audit,isotonic,rr_isotonic}_v3`, `export_calibrators_v3`, `recalibrate_adaptive_rr_v1`, `calibration_audit_v1`).
  - `scripts/backtests/`: filter / Kelly / adaptive-vs-fixed / constrained-band backtests (`filter_backtest_*_v{2,3}.py`, `kelly_backtest_v2`, `adaptive_vs_fixed_backtest_v1`, `constrained_band_backtest_v1`).
  - `scripts/evaluation/`: held-out and walk-forward evaluators (`walk_forward_v2`, `heldout_time_eval_v2`, `final_holdout_eval_v3`, `portfolio_sim_v3`).
  - `scripts/analysis/`: diagnostics + feature engineering (`shap_audit_v2`, `permutation_test_{adaptive_rr_v1,ml1}`, `validate_top_combos_ml1`, `feature_engineering_v2`, `build_combo_features_ml1`).
  - `scripts/runners/`: `run_*` orchestrators that upload scripts to sweep-runner-1 via paramiko and launch remote screen jobs.
  - `scripts/data_pipeline/`: utility data scripts (`init_db`, `sweep_status`, `validate_mfe_parquets`, `build_adaptive_rr_cache_v2`).
- `notebooks/`: interactive analysis; must run from repo root and write outputs to `iterations/` and `evaluation/`.
- `iterations/`: generated training backtest artifacts for canonical strategy versions (currently `V1/`, `V2/`, `V3/`). Sweep versions `v4`–`v10` are training-data generators for ML, not strategy iterations — they write Parquet to `data/ml/` rather than producing an `iterations/Vn/` folder.
- `evaluation/`: generated artifacts for the **single final** hold-out evaluation (no `Vn` subfolders).
- `data/ml/`: parameter-sweep output datasets (one Parquet + manifest per sweep version). See [`LOG_SCHEMA.md`](LOG_SCHEMA.md) for the parquet schema.

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

## Reviewing Agent Protocol

After **every logical code change** — new modules, modified algorithms, new sweep
parameters, indicator formulas, backtest engine changes — spawn a reviewing agent
before marking the task complete.

### When to spawn

| Change type | Review required |
|---|---|
| New indicator / formula | Yes |
| Modified signal logic | Yes |
| New sweep parameters / sampling | Yes |
| Cache / indexing logic | Yes |
| Backtest engine changes | Yes |
| Config additions only | No |
| Pure refactor with no logic change | No |

### Reviewing agent prompt structure

The reviewing agent prompt must always include:

1. **Context** — what the code does and why it was changed
2. **Files to read** — exact paths for every changed file
3. **What to check** — explicit checklist:
   - Mathematical / algorithmic correctness
   - Edge cases (NaN, zero, empty, first bar, last bar)
   - Compatibility rules enforced correctly
   - Cache / key correctness (no aliasing)
   - ML metadata accuracy (stored metadata matches what was computed)
   - Off-by-one in time series operations
4. **Severity schema** — CRITICAL / WARN / INFO
5. **Explicit OK confirmation** — agent must state which checks passed, not only what failed

### Fix rule

- **CRITICAL** — fix before proceeding, no exceptions
- **WARN** — fix or document in `lessons.md` with justification
- **INFO** — optional; note inline if relevant

### Pre-sweep gate (mandatory)

Before launching **any parameter sweep of 3000+ combinations**, spawn a dedicated
pre-sweep reviewing agent that audits all sweep-related code for:

1. **Type consistency** — every column written to Parquet must have a single
   consistent dtype across all combo branches (e.g. `int` vs `float` vs `None`
   mixing that PyArrow will reject at merge time).
   - **Mandatory sub-check**: For every nullable numeric column (e.g. `swing_lookback`,
     `stop_fixed_pts`, `atr_multiplier`), verify that **all** `range_mode` branches
     assign the **same Python type** when the value is non-`None`. A common failure
     pattern: one new branch uses `int(rng.integers(...))` while all older branches
     use `float(rng.integers(...))`. When a PyArrow batch contains only `None` for
     that column, pandas infers `float64`; a subsequent batch with `int` values
     produces `int64`, and `pa.concat_tables` raises `ArrowTypeError: incompatible
     types double vs int64`. Rule: nullable stop/lookback columns must always be
     `float(...)` when non-`None`, never bare `int`.
2. **Schema completeness** — all keys in `_COMBO_META_KEYS` are present in every
   combo dict for every `range_mode` branch.
3. **Logical correctness** — parameter sampling ranges are sensible, no branch
   silently produces no trades (e.g. unreachable thresholds).
4. **Append / merge safety** — `_append_parquet` and any schema-promotion logic
   correctly handles optional/nullable columns.

The sweep **must not launch** until the reviewing agent gives an explicit all-clear
on the above four points. Any CRITICAL finding blocks the sweep.

## Reporting & styling requirements

- **Win/Loss row shading**: in notebooks, trades tables should be styled:
  - winning trades: green row background
  - losing trades: red row background
  - zero PnL: neutral
- **Sharpe basis**: Sizing is flat `fixed_dollars_500` ($500 risk per trade)
  project-wide. Sharpe = `mean(pnl_$) / std(pnl_$) × sqrt(trades_per_year)`,
  annualized. Compounding / %-of-equity sizing (`pct5_compound`) was removed
  April 2026 — historical notebooks under `evaluation/v{10,11,12}_topk*/` still
  reference it as frozen phase records, but the generators
  (`scripts/evaluation/_build_v2_notebooks.py`, `_top_perf_common.py`) emit
  only fixed-$500 outputs. `src/reporting.py`'s `log1p`/`RISK_FRAC` helpers
  remain as library primitives but are no longer exercised by eval notebooks.
- **Remote eval-notebook memory cap**: `run_eval_notebooks_remote.py` launches
  under `systemd-run --scope -p MemoryMax=9G` on sweep-runner-1 (9.7G RAM).
  MC-heavy notebooks (§3 unfiltered bootstrap over ~24k trades) peak around
  2 GB per 10k-sim matrix; single-policy MC after the sizing-policy drop
  fits comfortably. Historical 2-policy MC notebooks needed reduced n_sims
  (s3 at 2,000) to avoid OOM. Net-suite s3 still needs `n_sims=2000` pinning
  on raw-Sharpe pools even in single-policy mode — see the shipped V4 pattern
  in commit 2105b45 and apply the same patch to any new s3 notebook before
  remote launch.
- **Net-only eval notebooks (policy from 2026-04-20)**: Any new variant added
  to `scripts/evaluation/_build_v2_notebooks.py` must use `_build_net_variant`
  (net-of-friction 6-section suite only), **not** the legacy `_build_variant`
  (which emits paired gross+net). Gross/zero-friction notebooks are not
  load-bearing for ship decisions — the ship criterion reads s6_net Sharpe
  exclusively, and `_ml2_net_ev_mask` gates ML#2 on net E[R]. Gross numbers
  are at best a diagnostic and at worst leak into write-ups as misleading
  "upside" figures. Historical variants (v10, v11, v12, raw_sharpe_v3/v4,
  full_pool_*, v4_no_gcid) stay on the dual API for reproduction only.

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

## Parameter sweep (Track B training data)

`scripts/param_sweep.py` generates a diverse ML training set by running the
backtest core over thousands of randomly-sampled parameter combos. Each combo
contributes its closed trades (feature + label columns) to a shared Parquet
file at `data/ml/originals/ml_dataset_v{N}.parquet` (or
`data/ml/mfe/ml_dataset_v{N}_mfe.parquet` for MFE re-runs), with a sidecar
`_manifest.json` tracking per-combo status (enables resume).

**Version history** (see `lessons.md` for per-version post-mortems):

| Mode | Purpose |
|---|---|
| `default` / `winrate` / `zscore_variants` | Original 3k sweeps (V1 diversity + winrate-biased + z-score formulation grid) |
| `v4` | Data-driven tightened ranges from V3 correlation analysis |
| `v5` | High-signal ranges + V5 filters (vol regime, session, tod_exit, volume entry) |
| `v6` | Ultra-tight ranges; V5 filters disabled (V5 analysis showed most hurt WR) |
| `v7` | Correlation-driven refinement of V6 |
| `v8` | Diversity pivot — broader ranges for ML coverage |
| `v9` | Ultra-wide diversity; hard-fixes z-score formulation to `close/rolling_mean/rolling_std/parametric` |
| `v10` | V9 numeric ranges + sampled z-score formulation + sampled exit/confirmation flags (max qualitative diversity) |
| `v11` | **Friction-aware ranges** (stop_fixed_pts ∈ [15, 100], no tight stops) + `$5/contract RT` cost baked into sweep engine PnL (`COST_PER_CONTRACT_RT`). Emits `gross_pnl_dollars`, `friction_dollars`, `mfe_points`, `mae_points`, `entry_bar_idx` inline — no separate MFE pass required. |

**Key invariant**: every nullable numeric column (`stop_fixed_pts`, `atr_multiplier`,
`swing_lookback`, …) must be `float(...)` when non-`None`. Mixing `int` and
`float` across branches causes `ArrowTypeError` at Parquet concat time — this
is the dominant past failure mode; see the Pre-sweep gate below.

## Track B: ML pipeline — implemented

- **ML#1 combo-grain surrogate**: LightGBM regressor(s) on combo-level outcomes.
  - Legacy v1–v10: `scripts/models/ml1_surrogate.py` (target = gross Sharpe).
  - v11 (`scripts/models/ml1_surrogate_v11.py`, `data/ml/ml1_results_v11/`): retargeted to net Sharpe after $5/contract RT; leakage-fixed (dropped `gross_sharpe` + `gross_net_sharpe_gap`).
  - **v12 (`scripts/models/ml1_surrogate_v12.py`, `data/ml/ml1_results_v12/`)**: parameter-only feature set + fold-wise parameter-space KNN + quantile heads (p10/p50/p90); target = robust walk-forward Sharpe (`median − 0.5·std` across K=5 ordinal windows). OOF R²=0.93, Spearman=0.96 on v11 sweep (13,814 combos post-gate). Trained but **retired as a production ranker** — see below.
  - **Production ranker (current, 2026-04-20): raw-Sharpe top-50**. The 2026-04-19 Pool B ablation (`project_ml1_v12_ablation_failure`) showed v12's UCB ranking (`p50 + κ·(p90−p10)/2`) is statistically indistinguishable from random at any κ. Production now ranks combos by `audit_full_net_sharpe` (the ML#1 training target, already pre-computed in `combo_features_v12.parquet` under `MIN_TRADES_GATE=500`) via `scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py` → `evaluation/top_strategies_v12_raw_sharpe_top50.json`. s6_net MC on this pipeline: Sharpe p50 2.13, CI [+0.56, +3.64], ruin 0.02%. The ML#1 **surrogate model** is therefore not loaded in the production inference path, but the **feature-engineering pipeline** that produces `combo_features_v12.parquet` remains upstream and required.
  - **Top-K truncation is load-bearing** (do not propose "drop ML#1 for a simpler pipeline"). The 2026-04-20 Step 4 ablation ran `ML#2 V4` against the full post-gate pool (500-combo uniform subsample of 13,814) with no top-K gate: s6_net collapsed to Sharpe 0.00, DD worst 280%, ruin prob 43.45% on 24,542 trades. See `tasks/ship_decision.md` and `evaluation/v12_full_pool_net_v4_2k/s6_mc_combined_ml2_net.ipynb`. Concentration to a small basket (currently 50) is an architectural invariant.
- **ML#2 trade-grain adaptive R:R** (`scripts/models/adaptive_rr_model_v{1,2,3}.py`): LightGBM binary classifier predicting `P(win | features, candidate_rr)`. **Current production stack is V3**: booster + pooled per-R:R isotonic + fixed 5% sizing (`data/ml/adaptive_rr_v3/`). Phase 5D held-out eval retired the per-combo two-stage calibrator (four null-to-negative results across Phases 3, 5A, 5C, 5D).
- **Net-of-friction E[R] filter** (added April 2026, commit 6662292, wired into `scripts/evaluation/_top_perf_common.py`): when building the ML#2-filtered portfolio in eval notebooks, `E[R]` is computed against net-of-cost payoffs (not gross), so the filter bar rises in proportion to contract count. Consumed by the `*_net` notebook suites under `evaluation/v{10,11,12}_topk_net/`.
- **Inference entry point**: `scripts/models/inference_v3.py::predict_pwin_v3()` consumed by filter/Kelly/portfolio backtests.
- Full Phase 3–5 experimental record: `tasks/part_b_findings.md`.

## Future scope (do not implement yet)

- Live volume "bubbles"
- DOM / Level 2 features
- Broker execution adapters
