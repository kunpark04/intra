> **HISTORICAL — superseded.** This document is the original V1 plan from
> 2026-04-13. The actual production stack is V3; see
> `tasks/adaptive_rr_decisions.md` D14 and `tasks/part_b_findings.md`
> Phase 5D for the current state. Kept for provenance; do not follow this
> plan verbatim.

# Adaptive R:R Model — Phase 1 Validation + Phase 2 Training Plan

**Purpose**: Continuation doc for the next Claude session (intended to run on the
DigitalOcean droplet at `195.88.25.157` where the `_mfe.parquet` files already
live). Picks up immediately after the 10-version MFE sweep completed.

**Prereq context** (memory files):
- `memory/project_adaptive_rr_plan.md` — full approach summary
- `memory/project_adaptive_rr.md` — original MFE/MAE idea
- `memory/project_trade_count_bias.md` — watch for low-trade-count combos in diagnostics

---

## Current state (2026-04-13)

All 9 MFE sweeps completed on server. Parquets exist at:

```
data/ml/ml_dataset_v{2..10}_mfe.parquet        # 14M–? rows each, 7.0 GB total
data/ml/ml_dataset_v{2..10}_mfe_manifest.json  # 37,000/37,000 combos, 0 errors
```

Status dashboard (`python scripts/data_pipeline/sweep_status.py`) shows **DONE** for every
version with zero errors. Files are present on both server and local (local
pulled via scp earlier this session).

**Do NOT re-run any sweep.** Data is complete.

Prior LightGBM run at `data/ml/ml1_results/` (2026-04-12) operated at
**combo-grain** (27,171 aggregate rows). The adaptive R:R model is different —
it works at **trade-grain** with synthetic labels expanded across R:R levels.

---

## Phase 1 — Data validation

**Goal**: confirm all 9 `_mfe.parquet` files have the required columns and that
MFE/MAE values are sane before spending hours training.

Validation must be **memory-safe**: v3 is 2.7 GB, v10 is 2.1 GB. Use
`pyarrow.parquet` metadata or row-group iteration, not `pd.read_parquet` on the
whole file.

### Checks

1. **Schema presence** — every file has columns:
   `mfe_points`, `mae_points`, `stop_distance_pts`, `hold_bars`, `combo_id`,
   `r_multiple`, `label_win`, plus the 15 entry features used by Phase 2.
2. **Row counts** — report per-version row count and `combo_id.nunique()`.
   Expect ~3k combos for v2–v8, ~6k for v9, ~10k for v10.
3. **Sign conventions** — `mfe_points >= 0`, `mae_points <= 0`, `hold_bars > 0`
   (per Cython core lines 123–133).
4. **Sanity vs r_multiple** — for each version, sample 1000 trades and verify:
   - Wins (`label_win == 1`) have `mfe_points >= r_multiple * stop_distance_pts`
     within rounding tolerance.
   - Losses have `mae_points <= -stop_distance_pts` (stop hit) OR exited early
     on opposite signal (check exit_reason).
5. **No NaN in MFE/MAE** — these are raw path extremes and should always
   resolve (unless the trade was zero-duration, which shouldn't happen).

### Suggested script

Create `scripts/data_pipeline/validate_mfe_parquets.py`. Use this pattern:

```python
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

VERSIONS = ['v2','v3','v4','v5','v6','v7','v8','v9','v10']
REQUIRED = {'mfe_points','mae_points','stop_distance_pts','hold_bars',
            'combo_id','r_multiple','label_win','exit_reason'}

for v in VERSIONS:
    p = Path(f'data/ml/ml_dataset_{v}_mfe.parquet')
    pf = pq.ParquetFile(p)
    cols = set(pf.schema_arrow.names)
    missing = REQUIRED - cols
    # Read only the required columns to stay under memory
    df = pf.read(columns=list(REQUIRED)).to_pandas()
    # ...checks...
```

If any CRITICAL issue (missing columns, all-NaN MFE, wrong signs) — STOP and
investigate before Phase 2. Spawn a reviewing agent.

---

## Phase 2 — Adaptive R:R LightGBM model

Full spec is in `memory/project_adaptive_rr_plan.md`. Condensed execution plan:

### Dataset assembly

- Load each `_mfe.parquet` **column-subset** (not full file). Only need:
  15 entry features + `mfe_points`, `mae_points`, `stop_distance_pts`,
  `r_multiple`, `label_win`, `combo_id`, `source_version`.
- Concatenate with a `source_version` column (`'v2'`…`'v10'`).
- **Base sample**: randomly subsample to ~5M trades total, stratified by
  `source_version` so each version contributes proportionally (but cap v10 to
  avoid dominance — it has 3× the combos of others).
- **R:R expansion**: for each base trade, duplicate across 17 candidate R:R
  levels: `[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
  4.0, 4.25, 4.5, 4.75, 5.0]`. Result: ~85M rows. Subsample to **10M** for
  training.

### Synthetic labels

For each (trade, candidate_rr) pair:
```
would_win = (mfe_points >= candidate_rr * stop_distance_pts)
```
This is the gold-standard MFE/MAE approach: we don't need to actually re-simulate
each R:R choice — the path data tells us whether any given R:R target would have
been hit before the stop.

Also compute `would_hit_stop_first = (mae_points <= -stop_distance_pts)` and
use path-ordering (hold_bars + sign) to resolve same-bar collisions per
strategy's `SAME_BAR_COLLISION = "tp_first"`.

### Features (20 total)

15 entry features from the parquet:
`zscore_entry, zscore_prev, zscore_delta, volume_zscore, ema_spread,
bar_body_points, bar_range_points, atr_points, parkinson_vol_pct,
parkinson_vs_atr, session_minute, minute_of_day, day_of_week,
ema_fast_slope, ema_slow_slope` (verify names in `param_sweep.py` line 908+).

Plus 5 engineered:
- `candidate_rr` (the expansion key)
- `stop_method` (categorical: atr / fixed / swing)
- `exit_on_opposite_signal` (bool)
- `abs_zscore_entry`
- `rr_x_atr` (`candidate_rr * atr_points`)

### Model

- LightGBM binary classifier, `objective='binary'`, metric=`'binary_logloss'`.
- 5-fold stratified CV on `(combo_id, source_version)` — critical: **do NOT
  split by row**, that leaks trades from the same combo across train/test.
  Split by combo_id to keep a combo's trades together.
- Hyperparams: start with same grid as the 2026-04-12 run
  (`data/ml/ml1_results/run_metadata.json`) but switch objective to binary.
- Early stopping on log-loss, 50 rounds patience.

### Inference — adaptive R:R selection

Given a new trade's features, at inference time:
1. Sweep all 17 candidate R:R values.
2. Predict `P(win | features, candidate_rr)`.
3. Compute expected R-multiple: `E[R] = P(win)*candidate_rr - (1-P(win))*1`.
4. Pick `candidate_rr` maximizing `E[R]`.

### Output structure

```
data/ml/adaptive_rr_v1/
  model.lgb                    # saved LightGBM model
  model_metadata.json          # features, R:R grid, training rows, CV scores
  cv_results.json              # per-fold log-loss, AUC, calibration
  feature_importance.png
  partial_dependence_rr.png    # P(win) as function of candidate_rr
  calibration_plot.png
  inference_example.py         # minimal "given features, pick R:R" example
```

### Validation of the model

Before trusting for deployment:

1. **Calibration** — reliability diagram; P(win) should track actual win
   frequencies within ±5%.
2. **Backtest adaptive vs fixed** — re-run v10_9955 (or whatever the current
   best combo is — check `data/ml/ml1_results/top_combos.csv`) with
   adaptive R:R and compare Sharpe / total return / max DD vs its original
   fixed-R:R result.
3. **No-lookahead check** — confirm all features are computed from data
   available AT entry, not after. Sampled 20 random trades, cross-checked
   against bar data.
4. **Permutation test** — shuffle `candidate_rr` within trade groups, retrain,
   confirm AUC collapses to ~0.5. If not, leakage.

---

## Execution order for the new session

1. `cd ~/intra` on server, `git pull` (sync latest param_sweep + scripts).
2. Verify env has: `lightgbm`, `pandas`, `pyarrow`, `scikit-learn`,
   `matplotlib`. Install if missing (`pip install lightgbm scikit-learn`).
3. Run `python scripts/data_pipeline/sweep_status.py` — confirm all DONE.
4. Write + run `scripts/data_pipeline/validate_mfe_parquets.py` (Phase 1). Report result.
5. STOP if Phase 1 fails; investigate.
6. Write `scripts/models/adaptive_rr_model_v1.py` (Phase 2 training).
   - **Spawn a reviewing agent** after writing, before running (per
     CLAUDE.md "Reviewing Agent Protocol"). This is a new non-trivial
     module touching labels + CV split — review required.
7. Run training. Expect hours; use `screen` or `nohup`.
8. Generate validation artifacts. Check calibration.
9. Run adaptive-vs-fixed comparison on a known good combo.
10. Update `lessons.md` with anything surprising. Update memory with outcomes.

---

## Known gotchas

- **v3 OOM on full read**: 2.7 GB parquet cannot be loaded via
  `pd.read_parquet(path)` in one shot on low-RAM machines. Use
  `pq.ParquetFile(p).iter_batches()` or `pf.read(columns=[...])`.
- **Combo-grain split**: CV split MUST be by `combo_id`, not by row.
- **R:R expansion explodes row count 17×**. Subsample to 10M before training
  or training will OOM even on the droplet.
- **Type consistency** (per past sweep failures): any new columns written
  must be `float` not `int` to survive Parquet concat. Not relevant for pure
  reading, but is relevant if Phase 2 writes intermediate parquets.
- **v10 has mixed z-score formulations** (per v10 sweep design). Keep
  `zscore_formulation` as a categorical feature or results will be confused.

---

## Pointers

- Full plan: `C:\Users\kunpa\.claude\plans\deep-munching-lynx.md` (local only)
- Memory: `memory/project_adaptive_rr_plan.md`
- Prior LightGBM combo-grain run (reference): `data/ml/ml1_results/`
- Param sweep trade dict: `scripts/param_sweep.py` lines 908–960
- Cython MFE/MAE computation: `src/cython_ext/backtest_core.pyx` lines 73–74,
  123–133, 193–194
- Strategy rules: `STRATEGY.md`
- Trade schema: `LOG_SCHEMA.md`
