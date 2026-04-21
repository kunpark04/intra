# Probe 1 Pre-Registration — 15m + 1h Timeframe Lift

**Date**: 2026-04-21 UTC
**Authority**: LLM Council verdict, `tasks/council/council-report-2026-04-21-nq-mnq-fork.html`
**Template**: Mirrors `tasks/phase5_kill_criterion.md` (Option C preregistration pattern).
**Process discipline**: Thresholds committed before the experiment runs, applied mechanically after. Ties go to the stricter (sunset) side.

---

## Status

**SIGNED 2026-04-21 18:11:57 UTC at commit `d0ee506`.**

Probe 1 execution (bar-aggregation infrastructure, sweep-harness
extension, parallel remote sweeps, gross-ceiling readout, and
conditional K-fold audit) may now proceed. The decision rule below is
binding regardless of which branch the observed metrics fall into.

---

## 1. Purpose

Test whether the Z-score mean-reversion strategy family has a tradeable
edge on NQ/MNQ at **bar timeframes other than 1 minute**. The v11 sweep
established that on 1-minute NQ bars the gross Sharpe ceiling across 13,814
combos is 1.108 (single combo, v11_23634), below the user's ship bar of
Sharpe 1.0 net of $5/contract RT friction.

The probe is scoped to answer two binary questions:

1. **Gross-ceiling gate**: does either 15-minute or 1-hour NQ produce a
   universe with ≥10 combos at gross Sharpe ≥ 1.3 and ≥ bar-count-adjusted
   trade floor?
2. **Ship gate** (conditional on gross-ceiling PASS): does a combo-agnostic
   refit of the V3 architecture (with `v3_no_memory` feature set) produce
   cross-fold mean s6_net Sharpe p50 ≥ 1.0 and ruin ≤ 20%?

FAIL on both timeframes at the gross-ceiling gate → **family-level sunset**
of Z-score mean-reversion on NQ/MNQ.

---

## 2. Combo-Level K-Fold Protocol

### Partition

- **Estimator**: `sklearn.model_selection.GroupKFold(n_splits=5)`.
- **Groups**: `combo_id` from the new sweep parquet
  (`data/ml/originals/ml_dataset_v11_15m.parquet` and/or `_1h.parquet`).
- **Semantics**: each fold trains on 4/5 of combos' trades and predicts
  OOS on the held-out 1/5 of combos. A combo's trades never appear in
  both train and OOS within a single fold.

### Feature set (v3_no_memory)

**Strip all per-combo memorization channels**:

- `global_combo_id` (LightGBM categorical)
- `prior_wr_10`, `prior_wr_50` (running win rate on this combo's past trades)
- `prior_r_ma10` (running r-multiple MA on this combo's past trades)
- `has_history_50` (flag for ≥50 past trades on this combo)

**Retained features** (mirrors the superseded `tasks/root_ablation_criterion.md`):

- *Trade-level market state at entry* (15): `zscore_entry`, `zscore_prev`,
  `zscore_delta`, `volume_zscore`, `ema_spread`, `bar_body_points`,
  `bar_range_points`, `atr_points`, `parkinson_vol_pct`,
  `parkinson_vs_atr`, `time_of_day_hhmm`, `day_of_week`,
  `distance_to_ema_fast_points`, `distance_to_ema_slow_points`, `side`.
- *Combo parameters — borderline, retained* (2): `stop_method` (3
  categories), `exit_on_opposite_signal` (boolean). Combined cardinality
  of 6 cannot uniquely identify thousands of combos.
- *Decision variable* (1): `candidate_rr`.
- *Derived interactions* (2): `abs_zscore_entry`, `rr_x_atr`.
- *New in this probe* (3): microstructure sweep parameters
  `entry_timing_offset`, `fill_slippage_ticks`, `cooldown_after_exit_bars`
  (see §4).

### Model

- **Booster**: LightGBM binary classifier on `P(win | features, candidate_rr)`
  — identical hyperparameters to `data/ml/adaptive_rr_v3/` (the revoked
  production stack), but refit per fold on the 4/5 training partition.
- **Calibration**: pooled per-R:R isotonic, fit per fold on the fold's
  OOF predictions.
- **Basket construction**: union of OOF predictions across folds, above
  the net-E[R] gate (`_ml2_net_ev_mask` from `_top_perf_common.py`).

### Evaluation

- **Kernel**: s6_net MC on the OOF basket, `n_sims=2000`, seeded at
  `MC_SEED=42`, `fixed_dollars_500` sizing, `$5/contract RT` friction.
- **Cross-fold metrics reported**: mean ± std of s6_net Sharpe p50, ruin,
  combo-count-in-basket, pos_prob.
- **Artifact layout**: `evaluation/probe1_{15m,1h}_kfold/` matching the
  standard eval-notebook structure.

---

## 3. Sunset Threshold

### Gross-ceiling pre-gate (applied to sweep output, before K-fold)

Let `N_1.3(tf)` = count of combos in the timeframe-`tf` sweep satisfying:

- gross Sharpe ≥ 1.3 (trade-level annualized, zero-friction), AND
- trade count ≥ bar-count-adjusted `MIN_TRADES_GATE` (see table below)

| Timeframe | Raw floor (× 500 / bar minutes) | Statistical floor | Applied floor |
|---|---|---|---|
| 15m | 33 | 50 | **50** |
| 1h | 8 | 50 | **50** |

(Floored at 50 for statistical significance of the per-combo Sharpe
estimate regardless of bar count.)

**Branch A — FAMILY-LEVEL SUNSET:**
`N_1.3(15m) < 10 AND N_1.3(1h) < 10`

Z-score mean-reversion on NQ/MNQ is declared falsified across the
bar-timeframe axis. No K-fold audit runs. Next actions:

1. Update `CLAUDE.md` to deprecate the family from production research.
2. Add lesson to `lessons.md` documenting the family-level falsification
   process.
3. Spawn a fresh LLM Council to decide the next fork (signal-family swap,
   session structure, or project sunset).

**Branch B — PROCEED TO K-FOLD:**
`N_1.3(15m) ≥ 10 OR N_1.3(1h) ≥ 10`

Run §2's K-fold audit on the passing timeframe(s).

### Post-K-fold ship gate (applied to cross-fold metrics)

All three conditions must hold on at least one timeframe:

1. `mean(cross-fold Sharpe p50) ≥ 1.0`
2. `mean(cross-fold ruin probability) ≤ 0.20`
3. `≥ 10 combos survive the basket in at least 4 of 5 folds`

**PASS** → the passing timeframe is the new production configuration.
Proceed to production wiring (update `CLAUDE.md`, re-point eval notebooks,
standard ship-decision banner).

**FAIL** → the timeframe cleared gross-ceiling but did not survive the
combo-agnostic audit. Document in `tasks/probe1_verdict.md` as a
"gross-but-not-robust" outcome. No ship. Fresh council for the next step.

### Tie-breaking

- Gross-ceiling gate: `N_1.3 = 9` on both timeframes → treat as Branch A
  (sunset). `N_1.3 = 10` on either → treat as Branch B (proceed).
- Ship gate: `mean Sharpe p50 = 0.99` or `mean ruin = 0.21` → FAIL.
  "Approximately meets" is not meets.

---

## 4. Microstructure Sweep Parameter Spec

Add to the v11 sweep parameter ranges (`param_sweep.py --range-mode v11`):

| Parameter | Values | Semantics |
|---|---|---|
| `entry_timing_offset` | `{0, 1, 2}` | Bars to wait after signal before order fill |
| `fill_slippage_ticks` | `{0, 1, 2}` | Ticks of slippage on top of existing 2-ticks-per-side model |
| `cooldown_after_exit_bars` | `{0, 3, 10}` | Minimum bars between exit and next entry |

Total new cells: 27. Existing v11 combinatoric ≈ 3000 sampled combos; the
sweep continues to sample the microstructure axes via the same random
mechanism as other `--range-mode v11` parameters. No Cartesian blow-up.

**Rationale**: the Executor's "free-ride axis" addresses the Contrarian's
microstructure concern at zero additional sweep cost. If the 1m family
failure is in fact an execution-friction failure rather than a signal
failure, the microstructure axes surface that within the same sweep rather
than requiring a separate diagnostic.

---

## 5. Walk-Forward Time-Slice (Regime Sanity Probe)

Chronological split of the sweep's OOS trade log into calendar-year bins
covering the NQ data range (2019-01-01 through the latest available bar):

| Slice | Range |
|---|---|
| S1 | 2019-01-01 — 2019-12-31 |
| S2 | 2020-01-01 — 2020-12-31 |
| S3 | 2021-01-01 — 2022-12-31 (2-year span — rate cycle) |
| S4 | 2023-01-01 — 2023-12-31 |
| S5 | 2024-01-01 — (latest) |

(Adjust S5's end per actual data range; adjust S3 span if data coverage
differs. Preserve the 4-5 slice count.)

**Sanity probe** (applied only on Branch B, after §2 K-fold completes):

For each slice, compute filtered-basket Sharpe p50 using the OOF
predictions restricted to trades in that slice. Define:

- `S_2024` = Sharpe p50 on S5 (most recent slice)
- `S_pre2024` = mean Sharpe p50 across S1-S4

**Flag condition**: `S_pre2024 > S_2024 × 1.30`.

If flagged, the edge is regime-dependent — it performed better in older
regimes (2019-2023) than in 2024. This does NOT auto-fail the probe, but
it must be called out in `tasks/probe1_verdict.md` as a flag against
production readiness, and the ship decision is deferred pending a regime-
stability council.

---

## 6. Timeframes

**Included**: 15-minute (`15min`) and 1-hour (`1h`). Run as parallel
remote sweeps per Phase C3 of `tasks/todo.md`.

**Deprecated**: 5-minute. Council's Expansionist argued (and peer review
endorsed) that 5m shares the 1m microstructure noise profile and is
correlated with the 1m failure mode; running 5m adds compute cost without
adding independent information.

**Sweep counts**:
- 15-minute: 3000 combos (matching v11 density).
- 1-hour: 1500 combos (lower combinatorial richness per unit of compute
  justifies smaller N).

---

## 7. Irrevocable Commitments (carried forward from prior pre-regs)

1. **Results read mechanically.** No post-hoc reinterpretation of
   thresholds once the gross-ceiling and K-fold metrics are read.
2. **No selective metric switching.** Gross pre-gate reads trade-level
   annualized zero-friction Sharpe. Ship gate reads s6_net Sharpe p50
   under `fixed_dollars_500` sizing, `$5/contract RT` friction,
   `n_sims=2000`, `MC_SEED=42`. Any deviation requires a written,
   timestamped amendment to this document pre-commit.
3. **AUC parity is never sufficient evidence for a ship.** V3 and V4
   both shipped on AUC parity and were both revoked for `global_combo_id`
   memorization. The post-K-fold ship gate is the authoritative criterion;
   AUC is logged but not decision-weighted.
4. **Combo-agnostic partition is mandatory.** `GroupKFold(groups=combo_id)`
   is the only acceptable partition for §2's K-fold. `KFold` without
   `groups` or any time-based partition alone does not satisfy this
   criterion (the 100%-combo-overlap structural finding in
   `project_combo_overlap_structural.md` means time partitions leak
   through parameter-space neighbors).
5. **Ties go to the stricter side** per §3.
6. **FAIL is terminal for this axis within this probe.** If Branch A
   fires, Z-score mean-reversion at the bar-timeframe axis is closed;
   workarounds (intermediate timeframes like 30min / 2h) are not
   admissible without a new pre-registration cycle and fresh council.
7. **Scope lock stands.** `memory/feedback_nq_mnq_scope_only.md` is
   binding throughout; no cross-instrument or cross-asset probes are
   admitted on any branch.

---

## 8. Compute Budget

| Step | Wall-clock | Notes |
|---|---|---|
| Bar aggregation infra (C1) | 2-3h | Local |
| Sweep harness extension (C2) | 1-2h + smoke | Local; pre-sweep gate required |
| Parallel sweeps (C3) | 12-24h | Remote, `sweep-runner-1`, monitor 10min |
| Gross-ceiling readout (C4) | 30 min | Local |
| K-fold audit (C5) | 3-5h × ≤ 2 timeframes | Remote, conditional |
| Verdict document (C6) | 1h | Local |
| **Total wall-clock** | ~1 day if Branch A; ~2 days if Branch B | |

---

## 9. Signature

By signing below, the signer commits to this document's decision rule
without post-hoc reinterpretation. §3 thresholds are mechanically applied
to the observed metrics. §7's irrevocable commitments stand.

- **Signed**: kunpa (confirmed via explicit "signed" reply 2026-04-21 18:11 UTC)
- **Date/time of signature**: 2026-04-21 18:11:57 UTC
- **Commit hash at time of signature**: `d0ee506` (d0ee5066abfebdb191bdf9fa9adb449078d49bc8)

---

## 10. References

- LLM Council verdict: `tasks/council/council-report-2026-04-21-nq-mnq-fork.html`
- Council transcript: `tasks/council/council-transcript-2026-04-21-nq-mnq-fork.md`
- Superseded ablation (`v3_no_memory` feature set origin): `tasks/root_ablation_criterion.md`
- Prior pre-registration template: `tasks/phase5_kill_criterion.md`
- Scope constraint: `memory/feedback_nq_mnq_scope_only.md`
- Universe-ceiling finding: `memory/project_friction_constant_unvalidated.md`
- Combo-overlap pathology: `memory/project_combo_overlap_structural.md`
- V3 and V4 revocations: `tasks/v3_no_gcid_audit_verdict.md`, `tasks/v4_no_gcid_audit_verdict.md`, `tasks/ship_decision.md`
- Plan document: `tasks/todo.md` (Phase C)
