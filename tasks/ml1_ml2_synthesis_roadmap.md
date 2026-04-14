# ML#1 + ML#2 — Synthesis & Next-Step Roadmap

**Created**: 2026-04-14
**Purpose**: Authoritative reference for the state of the two ML models in this
repository, what has been validated, what is still unknown, and the ranked
sequence of experiments to close out before any live-capital decision. Anchor
for future sessions — read this before proposing new ML work.

---

## Part A — What we know now

### ML #1 (combo-grain)

- **Script**: `scripts/ml_optimizer.py`
- **Output**: `data/ml/lgbm_results/`
- **Purpose**: rank parameter configurations by predicted composite score
  (Sharpe 0.25 + return 0.25 + DD 0.20 + WR 0.15 + trade count 0.15).
- **Data**: ~33K combos from sweeps v2–v10, aggregated from ~80M trades to
  combo-grain.
- **CV**: 5-fold random KFold on combos.
- **Outputs produced**:
  - `top_combos.csv` (20)
  - `high_freq_combos.csv` (5)
  - `low_freq_combos.csv` (5)
  - `surrogate_top_combos.csv` (50 from random sampling of 50k candidates)
- **Known weaknesses**:
  - Trained on fixed-R:R outcomes.
  - No calibration audit.
  - No permutation test.
  - No held-out time test for its picks.

### ML #2 (trade-grain adaptive R:R)

- **Script**: `scripts/adaptive_rr_model.py` → V1 (broken)
  and `scripts/adaptive_rr_model_v2.py` → V2 (fixed)
- **Output**: `data/ml/adaptive_rr/` and `data/ml/adaptive_rr_v2/`
- **Purpose**: predict P(win) for each trade at each of 17 R:R levels
  (1.0 → 5.0, step 0.25); use for R:R selection or filtering.
- **Data**: ~10M expanded rows (~600K base trades × 17 R:R levels) subsampled
  from 138M via stream-Bernoulli.
- **CV**: StratifiedGroupKFold on `global_combo_id`.

#### V1 (broken)

- `is_unbalance=True` inflated high-R:R probabilities.
- ECE 0.097, Sharpe 9.3 on top combo.
- `best_iter=5` → dramatically underfit.

#### V2 (fixed)

- Removed `is_unbalance`; `lr=0.02`; `min_child=20`; 2000 rounds.
- **AUC 0.806, log-loss 0.340, Brier 0.105, ECE 0.004** (post isotonic-per-RR).

#### Held-out variant (`adaptive_rr_heldout/`)

- Excludes `v10_9955` from training set.
- Identical AUC/log-loss/Brier to V2 → **no per-combo overfitting** at the
  model level.

### Joint findings on `v10_9955` (training split)

| Variant                       | Sharpe | Return | DD     | WR     |
|-------------------------------|--------|--------|--------|--------|
| Fixed (ML#1 combo only)       | 16.1   | 2285%  | 16.8%  | 70.2%  |
| V2 adaptive R:R picker        | 14.0   | 1634%  | 13.5%  | 73.6%  |
| Constrained-band [1.5, 3.0]   | 14.0   | 3244%  | 30.5%  | 56.0%  |
| Filter E[R] ≥ 0.0             | 29.9   | 358%   | 2.5%   | 98.1%  |

### Filter test (11 combos)

- **Low-freq (≤3500 trades)**: 2–4× Sharpe, DD halved, return preserved.
  Clean win.
- **High-freq (>5000 trades)**: over-filters. Return destroyed at `thr≥0.1`.

### Core insights

1. **ML#2-as-R:R-picker hurts**. The calibrated model prefers R:R ≈ 1.0,
   which cuts profits before they compound. Also violates `MIN_RR=1:3`
   constraint in `CLAUDE.md`.
2. **ML#2-as-filter wins on low-freq combos**. Discriminates real signal from
   noise; ML#1-chosen params + ML#2 filter = best combo.
3. **ML#1 optimized under fixed-R:R assumption**. Stack-integration errors
   are real (`v10_9955` adaptive Sharpe 16 → 9 until the filter approach).
4. **No `is_unbalance` on MFE-expanded data**. The class imbalance isn't a
   bug — it's the signal (high R:R = genuinely low true P(win)).

---

## Part B — What's still unexplored (ranked by expected value)

### Tier 1 — Cheap, high-value

#### B1. Per-combo optimal filter threshold
**What**: Learn filter threshold as a function of combo's fixed win rate.
Low-freq combos with WR=0.7 may want thr=0.25; high-freq WR=0.5 may want
thr=−0.3. Meta-model or simple per-combo quantile rule.
**Why**: Current single threshold across combos over-prunes high-freq ones.
**Effort**: ~1 hr.

#### B2. Adaptive threshold — E[R] percentile, not absolute
**What**: Skip if E[R] < quantile(E[R], 25%) within the combo, instead of
absolute E[R] < 0.
**Why**: Ensures consistent skip rate regardless of combo's P(win) baseline.
Tests whether the "over-filter" on high-freq combos is a threshold problem.
**Effort**: ~1 hr.

#### B3. Permutation test on ML#1
**What**: Shuffle target labels → retrain ML#1 → confirm AUC drops to random.
**Why**: Never done. Critical sanity check — matches ML#2's existing test.
**Effort**: ~30 min.

#### B4. Filter applied to `surrogate_top_combos.csv` (50 combos)
**What**: Extend the current 11-combo filter test to all 50 surrogate combos.
**Why**: Population-level evidence of where the filter works.
**Effort**: ~10 min.

### Tier 2 — Medium effort, major payoff

#### B5. Re-train ML#1 on adaptive-R:R-replayed outcomes
**What**: Re-compute combo metrics with V2 filter applied, then retrain ML#1.
**Why**: This is the correct feedback loop — ML#1 learns which combos work
well when ML#2 is filtering. Could completely reshuffle the top combos.
**Effort**: ~4 hrs (recompute metrics on 33K combos).

#### B6. ML#2 temporal OOD test
**What**: The 20% held-out test bars have never been used for ML#2. Rebuild
MFE parquet for test bars, apply V2 model, check whether ECE/AUC degrade vs
training split.
**Why**: Answers "does this generalize across time?"
**Effort**: ~2 hrs.

#### B7. Walk-forward validation for ML#2
**What**: Current CV splits by combo. Add time-based CV: train on 2022 bars,
validate on 2023 bars, etc.
**Why**: Tests temporal stationarity of the signal.
**Effort**: ~3 hrs.

#### B8. Feature engineering for ML#2
**What**: Current 20 features are mostly instantaneous at entry. Add:
- Rolling win-rate over prior N trades (auto-correlation)
- Regime features (vol regime, session trend)
- Time-since-last-trade
- Prior trade's P(win) residual
**Why**: Test whether these lift AUC meaningfully.
**Effort**: ~3 hrs.

### Tier 3 — Larger experiments

#### B9. Monotonic constraint on `candidate_rr`
**What**: P(win) should monotonically decrease as R:R increases (path-fact).
Enforce via LightGBM's `monotone_constraints`.
**Why**: Should improve calibration and close the "R:R=1.0 preference" loop.
May let us trust R:R picking again.
**Effort**: ~1 hr.

#### B10. Kelly-fraction sizing from calibrated P(win)
**What**: Currently risk 5% per trade regardless of signal strength. Kelly:
risk ∝ edge. With V2's calibrated P(win), compute Kelly per trade and
compare Sharpe.
**Why**: Orthogonal edge to R:R choice.
**Effort**: ~2 hrs.

#### B11. Regime-split ML#2
**What**: Train separate V2 models for high-vol vs low-vol bars. Test if
regime-conditional models outperform single model.
**Effort**: ~4 hrs.

#### B12. Multi-combo portfolio
**What**: Run top-5 low-freq combos concurrently with shared 5% total risk,
not 5% per combo. Test diversification.
**Why**: Current backtests are single-combo.
**Effort**: ~3 hrs.

#### B13. Stack ensemble
**What**: Combine V2 + isotonic-V1 + V1-ranking via simple averaging.
**Why**: Test if ensemble improves OOD calibration beyond V2 alone.
**Effort**: ~1 hr.

### Tier 4 — Structural / long-horizon

#### B14. SHAP interpretability on ML#1
**What**: Explain why a combo ranks high. Identify whether ranking depends on
spurious features (e.g. trade count as leakage).
**Effort**: ~2 hrs.

#### B15. Fixed-dollar risk test
**What**: Rerun filter comparisons with fixed-dollar risk instead of
percentage-of-equity risk.
**Why**: Compound effects distort `v10_9955`'s filter results (Sharpe 29.9
but return gutted). Fixed-dollar risk decouples Sharpe from trade count and
gives a cleaner signal-quality measure.
**Effort**: ~1 hr.

#### B16. Final held-out time evaluation
**What**: Run V2 + filter on the top recommended stack against the 20%
held-out test bars.
**Why**: Per `CLAUDE.md`, the test bars are the "single final" evaluation
slot and have never been touched. This is the gate for any live deployment.
**Effort**: ~2 hrs.

#### B17. Paper-trade forward phase
**What**: Before real capital, run live against recent data
(post-test-split bars) for 1–2 months.
**Why**: No ML project should skip this.
**Effort**: No estimate — calendar-bound.

---

## Part C — Recommended sequence

### Week 1 — Consolidate V2 findings
1. **B3** permutation on ML#1 (sanity)
2. **B4** filter on `surrogate_top_combos` (population view)
3. **B2** adaptive threshold (fixes high-freq over-pruning)
4. **B1** per-combo threshold (picks up residual)
5. **B9** monotonic constraint (may let us trust R:R picking again)

### Week 2 — The real integration
6. **B5** re-train ML#1 on V2-filtered metrics — this is the proper stack
7. **B10** Kelly sizing — orthogonal edge

### Week 3 — Validate
8. **B6** temporal OOD for ML#2
9. **B7** walk-forward
10. **B16** final held-out time evaluation

### Week 4+
11. **B17** paper trade

---

## Part C.1 — Standing operational rules for every Part B task

All five rules below apply by **default** to any script in Part B (or any
other ML training / backtest / parquet analysis in this repo). Set per user
policy 2026-04-14.

### Rule 2 — Cap LightGBM `num_threads=3`
Every LightGBM call must set `num_threads=3` (or `n_jobs=3`) in `LGB_PARAMS`
and in `model.predict(..., num_threads=3)`. Host is a 4-vCPU Kamatera VM;
leaving one vCPU for the OS / pyarrow IO threads avoids context-switch
thrash while still using 3/4 cores for compute. LightGBM's default
"all cores" spawns 16+ threads per job and caused load avg ~8 with two
concurrent jobs under the old 2-core host. Revised 2026-04-14 after the
server was upgraded to 4 vCPU and concurrent jobs were retired.

### Rule 3 — `systemd-run` resource limits on every long job
```
systemd-run --scope -p MemoryMax=5G -p CPUQuota=150% python scripts/<job>.py
```
Tune `MemoryMax` / `CPUQuota` to the job, but always set both. Host is a
~2-core / 9.7 GB box shared with interactive SSH sessions — a single
unconstrained job can OOM the machine.

### Rule 4 — Subsample to 3M rows before scaling up
Default `max_rows = 3_000_000` for any ML#2-style training. Scale to 10M
only after a measured AUC comparison shows the smaller sample loses more
than 0.01 AUC. V1 hit AUC 0.786 at `best_iter=5` on 10M; the plateau is
well below 10M.

### Rule 5 — Column-subset parquet reads, never whole-file
```python
pq.ParquetFile(p).read(columns=[...])   # good
pq.ParquetFile(p).iter_batches()        # good
pd.read_parquet(p)                      # not allowed on _mfe.parquet
```
`v3_mfe` is 2.7 GB, `v10_mfe` is 2.1 GB. Whole-file reads peak at 3–5×
the needed memory and have been the dominant OOM cause on this box.

### Rule 6 — Cache the expanded feature matrix once
The 10M-row expansion (base trades × 17 R:R levels + derived features) is
identical across V1, V2, heldout, b9, and every future Part B ML#2 variant.
Build it once into `data/ml/adaptive_rr_cache.parquet` (versioned path if
`ALL_FEATURES` or the R:R grid changes) and have new variants load it
instead of rebuilding from the 9 `_mfe.parquet` files.

**Note**: rules 1 (serialize jobs) and 7 (rent bigger hardware) were
discussed but **not** adopted as standing policy. Use only when re-requested.

---

## Part D — Honest limits

- **In-sample only**: Everything is still on the training bars (2011–~2023).
  The 20% test bars have never been touched. Any claimed Sharpe above could
  degrade on test.
- **Execution assumptions**: Backtest uses `next_bar_open` fill, zero
  slippage, zero commission. Live execution will erode edge.
- **Wide CIs on high Sharpe**: Compound equity with 5% risk has high
  variance; confidence intervals on Sharpe values > 15 are wide.
- **Overlapping training data**: ML#1 + ML#2 were trained on overlapping
  trades (ML#2's 10M rows came from the sweeps ML#1 summarized). No true
  information split between the two models.

---

## Pointers

- ML#1 decisions: `tasks/ml_decisions.md`
- ML#2 decisions: `tasks/adaptive_rr_decisions.md`
- ML#2 phase-2 training plan (original): `tasks/adaptive_rr_phase2_plan.md`
- Sweep monitor protocol: `tasks/v10_sweep_monitor_plan.md`
- Reviewing-agent protocol: `CLAUDE.md` § Reviewing Agent Protocol
- Train/test split policy: `CLAUDE.md` § Train / test split policy
  (80/20 chronological, test bars reserved for single final evaluation)
