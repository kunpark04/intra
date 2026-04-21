# Root Ablation Kill-Criterion — V3 No-Memory Refit

**Date**: 2026-04-21 UTC
**Authority**: User directive "Conduct the root ablation. ML models should have
absolutely no memorization." (2026-04-21, following V3-no-gcid feature-importance
audit surfacing the `prior_wr_*` family as an implicit per-combo memorization
channel).
**Process discipline**: Same pre-registration pattern as
`tasks/phase5_kill_criterion.md` (Option C) — thresholds committed before the
experiment runs, applied mechanically after.

---

## Status

**SUPERSEDED** — 2026-04-21 UTC. Never signed; never executed. The chairman
sweep completed the same day (see
`memory/project_friction_constant_unvalidated.md`) established that the
v11 combo universe has a **gross Sharpe ceiling of 1.108** (single combo of
13,814). Because any ML#2 filter operates *within* the gross distribution —
it reweights trades, it does not create them — no V3-no-memory refit can
produce net Sharpe ≥ 1.0 across a usable Pool B. The ablation's Branch 1
(PASS) is now foreclosed on information-theoretic grounds, independent of
how cleanly memorization is stripped.

Follow-up plan (supersedes this document): `tasks/todo.md` →
Phase A / B / C.

---

**Original text below, retained for historical reference only.**

---

## Purpose

Test whether ML#2 has *any* honest trade-level edge by stripping every feature
that could serve as a per-combo memorization channel and retraining V3 on the
same training data.

Scoping principle: **the model must be able to predict on a combo it has never
seen before, with zero trade history.** Any feature unavailable for such a combo
is stripped.

## Features stripped (V3 → V3-no-memory)

### Already stripped in `v3_no_gcid` (baseline for this ablation)

- `global_combo_id` — explicit per-combo lookup table (LightGBM categorical)

### Newly stripped in this ablation (FAMILY_A from `adaptive_rr_model_v3.py:63`)

- `prior_wr_10`  — running win rate, last 10 trades of this combo
- `prior_wr_50`  — running win rate, last 50 trades of this combo (**rank #3 in V3-no-gcid importance**, gain 3.40M)
- `prior_r_ma10` — running r-multiple MA, last 10 trades of this combo (**rank #4**, gain 2.17M)
- `has_history_50` — flag for ≥50 past trades on this combo

Rationale: all four are computed from the combo's own past trade stream. For a
brand-new combo with no history, these are undefined. Their survival after
`global_combo_id` removal is the "leak in a costume" pathway the LLM Council's
peer-review round flagged (`tasks/council-transcript-2026-04-21-phase5.md`).

### Retained (20 features)

**Trade-level market state at entry** (15):
`zscore_entry`, `zscore_prev`, `zscore_delta`, `volume_zscore`, `ema_spread`,
`bar_body_points`, `bar_range_points`, `atr_points`, `parkinson_vol_pct`,
`parkinson_vs_atr`, `time_of_day_hhmm`, `day_of_week`,
`distance_to_ema_fast_points`, `distance_to_ema_slow_points`, `side`

**Combo parameters — borderline, retained** (2):
`stop_method` (3 categories), `exit_on_opposite_signal` (boolean). Combined
cardinality of 6 cannot uniquely identify 588,235 combos, so these are not a
viable memorization channel. They encode design-choice signal that generalizes
to new combos configured with the same parameter set. If the ablation passes
ambiguously, a stricter second ablation (`v3_pure_entry`) can additionally
strip these.

**Decision variable** (1): `candidate_rr`

**Derived interactions** (2): `abs_zscore_entry`, `rr_x_atr`

## The Decision Rule

Let **`S_no_memory`** = filtered portfolio Sharpe p50 on Pool B s6_net MC,
V3-no-memory filter applied, fixed_dollars_500, $5/contract RT, `n_sims=2000`.

Let **`R_no_memory`** = risk_of_ruin_prob on the same MC.

### Reference points (Option C)

| Condition | Sharpe p50 | Ruin prob |
|---|---|---|
| Unfiltered Pool B (no ML#2) | −0.19 | 97.3% |
| V3-no-gcid filtered (with prior_wr_*) | +0.31 | 53.6% |
| V3-no-memory filtered (target of this experiment) | `S_no_memory` | `R_no_memory` |

### Branches (apply mechanically)

**Branch 1 — PASS** (trade-level edge exists beyond memorization):

Condition: **`S_no_memory ≥ +0.15` AND `R_no_memory ≤ 0.70`**

Interpretation: roughly half of V3-no-gcid's Sharpe uplift and meaningful ruin
reduction survive the stripping of the `prior_wr_*` channel. The model has
genuine trade-level signal; memorization was supplementary but not load-bearing.

Next actions:
1. ML#2 architecture is defensible. Option A (combo-level K-fold retrain)
   becomes justified for the V3-no-memory feature set.
2. Task #17's ranker-basket swap becomes the follow-up audit to sanity-check
   generalization across combo selections.
3. Any production ML#2 stack must be on the no-memory feature set; the
   V3-no-gcid stack (with `prior_wr_*`) is retired as a ship candidate.

**Branch 2 — FAIL** (ML#2 was memorization):

Condition: **`S_no_memory < +0.15` OR `R_no_memory > 0.70`**

Interpretation: the `prior_wr_*` channel carried most of V3-no-gcid's signal.
Without it, the model does not have a meaningful trade-level edge. The ML#2
architecture as designed is a memorization artifact.

Next actions:
1. **Retire the ML#2 trade-grain-filter line of work.** No further K-fold
   retrains, calibration variants, or per-combo stacks of V3-family models.
2. **Option B becomes the default**: ship raw top-50 combos by
   `audit_full_net_sharpe`, no ML#2 filter. Portfolio Sharpe ceiling at
   whatever the unfiltered Pool B provides (currently −0.19, but Option B
   may use a different basket selection).
3. Task #17's basket swap is moot — nothing to audit.
4. Document ablation result in `lessons.md` as the definitive post-mortem
   on the ML#2 architecture.

## Additional irrevocable commitments

Per the Option C discipline carried forward:

1. **Results read mechanically.** No post-hoc reinterpretation of thresholds
   once `S_no_memory` and `R_no_memory` are read.
2. **Ties go to the stricter side** (Branch 2). The precedent of two false
   positives (V3 shipped Sharpe 1.78, V4 shipped Sharpe 2.13, both
   memorization) argues for the stricter prior.
3. **No selective metric switching.** The criterion reads s6_net Sharpe p50
   with fixed_dollars_500, $5/contract RT, n_sims=2000, exactly as Option C
   measured V3-no-gcid. Any deviation must be pre-approved in writing.
4. **Stricter follow-up on PASS.** A strong pass (S ≥ +0.25 AND R ≤ 0.55)
   would additionally justify a second ablation stripping `stop_method` +
   `exit_on_opposite_signal` to confirm no residual parameter-memorization.
   This is optional — the Branch 1 verdict is binding regardless.
5. **FAIL is terminal for ML#2 as currently designed.** Do not propose
   workarounds (per-combo calibration, stacked filters, etc.) without first
   demonstrating a non-memorization edge exists in some other way.

## Compute budget

- Code change in `scripts/models/adaptive_rr_model_v3.py` (add `--no-memory`): ~10 min
- New inference shim `scripts/models/inference_v3_no_memory.py`: ~5 min
- Remote runner `scripts/runners/run_v3_no_memory_refit_remote.py`: ~10 min
- Remote training on sweep-runner-1: **~2h** (V3-no-gcid was 7,029s / 117 min per `metrics_v3.json:177`)
- Remote s3-s6 eval on Pool B (production basket): ~10 min
- Verdict synthesis: ~30 min
- **Total**: ~3h wall-clock, ~2h unattended

## Signature block

By signing below, the signer commits to Branch 1 or Branch 2 as the next
action, without post-hoc reinterpretation of what counts as "≥ +0.15" or
"≤ 0.70". Ties go to Branch 2.

**Signed**: _______________
**Date/time of signature**: _______________
**Commit hash at time of signature**: _______________

---

## References

- V3-no-gcid audit: `tasks/v3_no_gcid_audit_verdict.md`
- V3-no-gcid feature importance: `data/ml/adaptive_rr_v3_no_gcid/metrics_v3.json`
- Option C verdict: `tasks/option_c_verdict.md`
- Phase 5 criterion (pattern reference): `tasks/phase5_kill_criterion.md`
- V3 model code: `scripts/models/adaptive_rr_model_v3.py`
- Task #17 plan (now scoped secondary to this ablation): `tasks/task17_ranker_audit_plan.md`
