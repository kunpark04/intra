# V3 Follow-up Plan — after the retrain finishes

**Context**: `scripts/adaptive_rr_model_v3.py` is running on sweep-runner-1
(screen `50718.v3_run`). When it finishes it will produce:

- `data/ml/adaptive_rr_v3/booster_v3.txt` — LightGBM booster
- `data/ml/adaptive_rr_v3/metrics_v3.json` — OOF AUC/LL/Brier/ECE (raw+cal),
  per-fold metrics, feature gain
- `data/ml/adaptive_rr_v3/isotonic_calibrators_v3.json` — per-R:R thresholds
- `data/ml/adaptive_rr_v3/feature_importance_v3.png`

Feature set = V2's 20 + Family A (`prior_wr_10/50`, `prior_r_ma10`,
`has_history_50`) + `global_combo_id` as LightGBM categorical.

This plan covers **post-retrain integration** in phases that are each
self-contained enough to execute in a fresh chat context.

---

## Phase 0 — Documentation Discovery (done)

### Files that currently load the V2 booster

| Script | Load site | Output |
|---|---|---|
| `scripts/filter_backtest.py` | L22 `V2_MODEL = ...adaptive_rr_v2/adaptive_rr_model.txt` | `filter_backtest.json` |
| `scripts/filter_backtest_per_combo.py` | Imports `V2_MODEL` from `filter_backtest.py`; L78 `lgb.Booster(...)` | `filter_backtest_per_combo.json` |
| `scripts/filter_backtest_percentile.py` | Imports `V2_MODEL`; L78 `lgb.Booster(...)` | `filter_backtest_percentile.json` |
| `scripts/filter_backtest_surrogate.py` | Imports `V2_MODEL`; L89 `lgb.Booster(...)` | `filter_backtest_surrogate.json` |
| `scripts/b6_heldout_time_eval.py` | L82 `default=...adaptive_rr_v2/adaptive_rr_model.txt` | `b6_temporal_ood.json` |

### Existing calibrator

- `scripts/recalibrate_adaptive_rr.py` — L47–58 trains **static** per-R:R
  `IsotonicRegression` from OOF predictions. **Not time-rolling.**

### Known baselines (pre-V3)

- B1 (`filter_backtest_per_combo.json`): bimodal threshold distribution,
  WR×threshold ρ=0.341.
- B2 (`filter_backtest_percentile.json`): top-25% wins 34/48 combos,
  median Sharpe lift **+3.87**.
- B4 (`filter_backtest_surrogate.json`): median Sharpe lift **+2.65** across
  50 surrogate combos × 3 thresholds.
- B6 (`b6_temporal_ood.json`): test AUC 0.8014 (Δ −0.004 vs train), but
  **ECE 0.062** — calibration drift is the gating blocker for B16.

### Anti-patterns to avoid

- Do **not** re-run the V2 scripts with V3 artifacts by string-replacing
  paths. V3's feature matrix has Family A + `global_combo_id`; V2's
  `build_feature_matrix` in `filter_backtest.py` does not compute these.
  Always add V3-specific feature construction, don't pretend the models
  are drop-in.
- Do **not** invent LightGBM APIs — V3 uses the documented
  `lgb.Booster(model_file=...)` load + `.predict(df)` pattern. Confirm in
  existing `filter_backtest.py::predict_pwin()` before mirroring.
- `global_combo_id` **unseen at inference** falls back to LightGBM's
  default category. For held-out eval (B16/B6), unseen combos are
  expected; report both "seen" and "unseen" subset metrics.

---

## Phase 1 — V3 artifact validation (≤30 min, no compute)

**Goal**: sanity-check `metrics_v3.json` before any downstream work.

### What to do

1. Read `data/ml/adaptive_rr_v3/metrics_v3.json`.
2. Verify:
   - `oof_auc` ≥ 0.830 (B8 hit 0.8406 on smaller testbed; full-set CV will
     be conservative because `global_combo_id` is unseen in every fold,
     so anywhere near that range is a pass).
   - `oof_ece_cal` ≤ 0.01 (matching V2's 0.004 post-isotonic is the bar).
   - Top-10 `feature_importance_gain` contains at least two Family A
     features among the top 5 (confirms they survived full-set training).
3. Read `feature_importance_v3.png` to visually confirm orange (Family A)
   and red (`global_combo_id`) bars are substantive vs blue (V2).
4. Append `### V3 retrain complete` section to `tasks/ml_cross_analysis.md`
   with the AUC/ECE delta table and one-paragraph verdict.

### Verification

- `grep -c oof_auc data/ml/adaptive_rr_v3/metrics_v3.json` = 1 (file exists
  and well-formed).
- The findings subsection cites specific numbers, not the plan's
  placeholders.

### Anti-patterns

- Don't claim production readiness based solely on OOF AUC — calibration
  ECE matters equally. Report both.
- Don't over-index on `global_combo_id` importance: as documented above,
  it's useless during CV (always unseen), so its gain is driven by
  in-fold-training splits only — still meaningful for production, but
  the CV metric understates what it contributes live.

---

## Phase 2 — V3 inference helper (1–2 hrs, local, small)

**Goal**: factor out a reusable `predict_pwin_v3(df, booster, calibrators)`
that computes Family A features + applies booster + applies per-R:R
isotonic. Needed by Phases 3 and 4.

### What to do

1. Open `scripts/filter_backtest.py` and copy its `predict_pwin()` /
   `build_feature_matrix()` pattern. The V3 version must additionally:
   - Before prediction: call a `compute_family_a(df)` that mirrors
     `scripts/adaptive_rr_model_v3.py::add_family_a` (same sort key, same
     rolling windows, same NaN fills). Import it directly — do not
     duplicate the logic.
   - After prediction: load `isotonic_calibrators_v3.json`, construct a
     dict of R:R → `IsotonicRegression` (fit via
     `iso.X_thresholds_ = np.array(X)` / `iso.y_thresholds_ = np.array(y)`
     pattern; check `sklearn.isotonic` source if unsure), apply per-R:R.
2. Save as `scripts/v3_inference.py` with a single exported function
   `predict_pwin_v3(base_trade_df, rr_levels) -> np.ndarray` returning a
   calibrated P(win) matrix shape `(n_trades, len(rr_levels))`.
3. Unit-test by round-tripping the OOF predictions:
   - Load V3 booster + V3 calibrators.
   - Run `predict_pwin_v3` on a 10k subsample of the training data.
   - Verify ECE on that subsample is within 0.005 of `oof_ece_cal`.

### Verification

- `python -c "from v3_inference import predict_pwin_v3; ..."` returns
  expected shape and values close to OOF.

### Anti-patterns

- Don't hand-roll isotonic reconstruction from thresholds if
  `IsotonicRegression` has a `from_params`-style loader. Check sklearn
  docs; don't invent API.
- Don't skip the round-trip test. Getting inference wrong silently
  corrupts every downstream backtest.

---

## Phase 3 — Re-run B1/B2/B4 filter backtests on V3 (2–3 hrs, remote)

**Goal**: quantify V3's downstream lift on the established filter
benchmarks. Only proceed if Phase 1 passed the AUC/ECE gates.

### What to do

1. For each of `filter_backtest_per_combo.py`, `filter_backtest_percentile.py`,
   `filter_backtest_surrogate.py`:
   - Copy the file to a `_v3.py` variant.
   - Replace the `V2_MODEL` constant with
     `V3_MODEL = REPO / "data/ml/adaptive_rr_v3/booster_v3.txt"`.
   - Swap the `predict_pwin` call for `v3_inference.predict_pwin_v3`.
   - Change output JSON path to
     `data/ml/adaptive_rr_v3/filter_backtest_<name>_v3.json`.
2. Run each on sweep-runner-1 under `systemd-run -p MemoryMax=5G
   -p CPUQuota=400%` (Rule 3). Serialised — one at a time.
3. Produce a delta table in `tasks/part_b_findings.md` under a new
   `### V3 filter backtest lift` subsection:
   - Median Sharpe lift, per-combo win rate of V3 over V2.
   - For B2: does top-25% still win 34/48+ combos? Does median lift
     exceed V2's +3.87?
   - For B4: does median +2.65 improve meaningfully?

### Verification

- The three `*_v3.json` files exist and match the schema of their V2
  counterparts (same keys).
- Commit with a concrete claim ("V3 lifts B2 median Sharpe from +3.87 to
  +X.XX across N combos") or a null result statement.

### Anti-patterns

- Don't compare V3 filter output to V2 filter output with different random
  seeds or different combo sets. Keep seeds fixed and combo sets identical
  — apples to apples.

---

## Phase 4 — Rolling isotonic recalibrator (3–4 hrs, local, medium)

**Goal**: the B6/B16 calibration-drift mitigation. V3 alone doesn't fix
the post-2024-10-22 regime shift; a time-rolling recalibrator does.

### What to do

1. Open `scripts/recalibrate_adaptive_rr.py` L47–58. The existing pattern
   fits one `IsotonicRegression` per R:R on all OOF data. Extend it:
   - Add a `--window` arg (trades, default 5000) and `--refit-every` arg
     (trades, default 500).
   - Instead of one calibrator per R:R, produce a **series** of
     calibrators indexed by anchor-timestamp or trade-index.
   - At inference: look up the most recent calibrator whose training
     window ends before the trade's entry time.
2. Output to
   `data/ml/adaptive_rr_v3/rolling_calibrators/{ts}_rr_{rr}.joblib` +
   a manifest `rolling_manifest.json`.
3. Update `scripts/v3_inference.py` (Phase 2) to optionally use the
   rolling calibrators.

### Verification

- On the B6 held-out bars, rolling-calibrated ECE drops from 0.062
  (static) to < 0.015. This is the gate; if it doesn't drop, the rolling
  approach is wrong and we need to think harder before B16.
- Commit a `b6_rolling_recal.json` with before/after ECE per window.

### Anti-patterns

- Don't train the rolling calibrator on data that includes the bar being
  predicted (off-by-one leakage). Every window must end strictly before
  the prediction timestamp.
- Don't ship a calibrator without a version manifest — live inference
  needs to know which calibrator to use for which trade.

---

## Phase 5 — B16 final held-out evaluation (1 day)

**Goal**: the live-capital gate. Single run. Don't re-run; per
`CLAUDE.md` §Train/test split, the test bars are spent after this.

### What to do

1. Copy `scripts/b6_heldout_time_eval.py` to `scripts/b16_final_eval.py`.
2. Update model load path to V3 (`booster_v3.txt`) and integrate
   `v3_inference.predict_pwin_v3` + rolling calibrator from Phase 4.
3. Run the top stack (ML#1-v2filtered top combos + V3 filter + per-combo
   or percentile threshold from Phase 3 winner) on the 20% test bars.
4. Report: AUC, ECE (static + rolling), Sharpe / DD / WR / total return
   on the filtered + sized stack. Compare to in-sample numbers.

### Verification

- `evaluation/metadata.json` created per `CLAUDE.md` §Required outputs,
  with `source_iteration = "V3"`, full strategy config, data range and
  split indices.
- `evaluation/trades.csv`, `trader_log.csv`, `daily_ledger.csv`,
  `equity_curve.csv`, `monte_carlo.json`, `analysis.ipynb` all exist.
- `analysis.ipynb` executes end-to-end without manual edits.

### Anti-patterns

- Don't iterate. One shot. If the numbers are disappointing, document and
  move to B17 paper-trade rather than tweaking anything against test data.
- Don't skip the Monte Carlo permutation test (`CLAUDE.md` requires it).

---

## Phase 6 — Documentation + commit

After each phase above, update:

- `tasks/part_b_findings.md` — scoreboard row + numerical delta section
- `tasks/ml_cross_analysis.md` — update the confidence tiers + critical
  path with what was validated
- `tasks/ml1_ml2_synthesis_roadmap.md` — mark phase complete

Commits should be one per phase with a concrete numerical claim in the
message (e.g., "V3 retrain: OOF AUC 0.838, ECE 0.006" — use the actual
numbers, not placeholders).

---

## Sequencing note

Phases 1 → 2 → 3 are strictly sequential (each depends on the previous).
Phase 4 can run in parallel with Phase 3 (different compute, different
artifacts). Phase 5 depends on Phase 3 winner + Phase 4 rolling
calibrator. Phase 6 runs continuously as each phase closes.

---

## Phase 4b — Post-Phase-4 robustness tests (pre-B16)

Phase 4 passed the static gate (rolling ECE 0.0070 vs 0.015). Before
committing to B16 final eval, validate the result with three tests:

### 4b.1 — Per-regime ECE decomposition

Split the held-out tail at the 2024-10-22 regime boundary (or the
chronological midpoint if exact date unavailable in the test parquet).
Report raw / static / rolling ECE separately for each slice.

- **Goal**: confirm static degrades post-break while rolling recovers
  — the mechanistic claim behind Phase 4.
- **Output**: `data/ml/adaptive_rr_v3/b6_regime_split_ece.json` +
  section in `part_b_findings.md`.
- **Runtime**: ~2 minutes (reuses Phase 4 pipeline, adds split).
- **Gate**: rolling ECE in post-break slice < 0.02 (allowing slack
  for smaller n). Static ECE in post-break should be visibly worse
  than pre-break, else the drift story is wrong.

### 4b.2 — Bootstrap CI on Phase 4 ECE

1000× bootstrap resamples of the held-out expanded rows; compute
rolling ECE per resample; report 95% CI and probability(ECE < 0.015).

- **Goal**: quantify how much of the 2.1× margin is signal vs noise.
- **Output**: `data/ml/adaptive_rr_v3/b6_rolling_ece_bootstrap.json`.
- **Runtime**: ~15 minutes local (no booster predict; reuse cached
  p_rolling / y arrays — dump these from Phase 4 first).
- **Gate**: P(ECE < 0.015) > 0.95.

### 4b.3 — 4-hour test: **Rolling window × refit_every grid sweep**

Full Phase 4 rerun over the grid:

- `window ∈ {1000, 2500, 5000, 10000, 20000}`
- `refit_every ∈ {100, 500, 2000}`

→ 15 configs × ~90s ≈ 25 min compute, but with SFTP round-trips,
logging, per-run Family A recomputation, and sensitivity-report
generation the whole loop is ~3–4 hrs wall clock.

- **Goal**: map the ECE surface; verify 5000/500 wasn't a lucky
  pick; find the robust plateau for productionization.
- **Implementation**: extend `scripts/b6_rolling_recal_v3.py` to
  accept a `--grid` JSON arg, loop over configs, emit one row per
  config to `data/ml/adaptive_rr_v3/b6_rolling_grid.json`. Remote
  launcher mirrors Phase 4's pattern.
- **Output**: heatmap (window × refit) of rolling ECE + markdown
  table in `part_b_findings.md`.
- **Gate**: ≥ 60% of grid cells satisfy ECE < 0.015, and the
  5000/500 cell is within 0.002 of the grid minimum. If 5000/500
  is a sharp spike, redo Phase 4 with the new robust optimum.

Sequence: 4b.1 and 4b.2 are cheap — run them first. 4b.3 is the
~4hr test; only launch after 4b.1/4b.2 confirm the headline number
is real.
