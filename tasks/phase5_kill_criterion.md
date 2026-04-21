# Phase 5 Kill-Criterion — Pre-Registered Before Reading Option C's Result

**Date**: 2026-04-21 UTC
**Authority**: LLM Council verdict, `tasks/council-report-2026-04-21-phase5.html`
**Purpose**: Pre-commit the decision rule for Option C (unfiltered Pool B
diagnostic) **before** the result is read. This is the single process
discipline the council identified that would have caught both V3 and V4
revocations ex-ante. AUC parity fooled the pipeline twice; the only
structural fix is to write the decision rule before the evidence is
observable.

---

## Status

**SIGNED 2026-04-21 07:18:58 UTC at commit `5135706`.**

Option C notebook extraction may now proceed. The decision rule below
is binding regardless of which branch the observed (S, R) falls into.

---

## The Decision Rule

Let **`S_unfiltered_C`** = p50 Sharpe from the s6_net (or s3 unfiltered,
if that is what the executed notebook exposes) Monte Carlo on Pool B
top-50 under V3-no-ID training, *with no ML#2 filter applied*, computed
on the identical kernel / bootstrap / friction parameters that produced
V3-no-ID filtered Sharpe 0.31 (ruin 53.62%).

Let **`R_unfiltered_C`** = the corresponding risk-of-ruin probability.

**If `S_unfiltered_C ≥ 0.31` AND `R_unfiltered_C ≤ 0.54`:**

1. **Deprecate ML#2 filtering from the production pipeline.** The
   `scripts/models/adaptive_rr_model_v{1,2,3,4}.py`, inference paths
   (`scripts/models/inference_v3.py`), isotonic calibrators, Kelly
   variants, and per-combo calibrators are all declared overhead. Their
   apparent Sharpe uplift was combo-memorization leak value, not edge.
2. **Do not train another trade-grain filter (any new ML#2 attempt)
   without first auditing the ranker layer** (`audit_full_net_sharpe`
   top-50 selection over `combo_features_v12.parquet`) for per-combo
   memorization via a pre-2024 / walk-forward partition refit.
3. **Option A (combo-level K-fold retrain) is also deprecated** under
   this branch — if the filter is overhead, K-fold validation of the
   filter is overhead squared.
4. **Focus shifts upstream**: combo-level ranker audit (Task #17), then
   decide between accepting the raw-Sharpe top-50 baseline or
   investigating a clean learn-to-rank redesign (not a per-trade
   filter).

**If `S_unfiltered_C < 0.31` OR `R_unfiltered_C > 0.54`:**

5. ML#2 is additive (or at least not net-negative) under clean training.
   This does **not** authorize re-shipping V3 or V4 — the combo-ID
   revocations stand.
6. **Option A remains conditional on first completing the Task #17
   ranker audit.** Auto-launch of K-fold retrain Monday PM is rejected
   (against Executor's recommendation) because the 100%-overlap critique
   means K-fold OOS combos are parameter-space neighbors of training
   combos; running K-fold before the ranker is audited risks a third
   AUC-parity false pass.
7. Option B (accept 0.31 ceiling) remains on the table but requires a
   named, committed next `.py` file per Executor's forcing-function
   objection. A "pivot to combo-scouring" without a concrete script is
   declared procrastination.

## Additional irrevocable commitments (independent of C's result)

8. **Reject Option D** (multi-day learning-to-rank rebuild as a trade
   filter). v12 raw-Sharpe already ranks combos well; the problem is not
   ranking, it is the trade-level filter layer. (Note: a combo-level
   learn-to-rank over `combo_features_v12` is a *different* proposal
   and remains in scope under Task #17's downstream decisions — but
   trade-grain learn-to-rank is rejected.)
9. **Reject per-combo meta-models (Expansionist's Option E).** Unanimous
   peer-review finding: training calibrators on each of 50 combos' own
   trade histories, when those 50 combos were selected from the same
   universe V3/V4 trained on, is the `global_combo_id` memorization
   channel formalized. Phase 5D already killed this approach (91% DD,
   B16 fail). Not to be reopened.
10. **Mandatory pre-ship audit for any future ML#2 stack**: no future
    trade-grain filter may be declared production-ready without (a) a
    combo-agnostic refit achieving ship-criterion Sharpe on Pool B, AND
    (b) a ranker-layer audit on a partition excluding the ML#2 training
    window. AUC parity is a precondition, never sufficient evidence.

---

## Signature

By signing below, I commit to the above decision rule and will execute
it mechanically based on `S_unfiltered_C` and `R_unfiltered_C` once
read, without post-hoc reinterpretation of what counts as "≥ 0.31" or
"≤ 0.54". Ties (exact equality) go to the kill branch — the precedent
of two false positives argues for the stricter prior.

**Signed**: kunpa (confirmed via explicit "signed" reply 2026-04-21 07:19 UTC)
**Date/time of signature**: 2026-04-21 07:18:58 UTC
**Commit hash at time of signature**: `5135706` (513570602046e5d7526973a136346c1955394aa0)

---

## Reference

- Council HTML report: `tasks/council-report-2026-04-21-phase5.html`
- Council transcript: `tasks/council-transcript-2026-04-21-phase5.md`
- V3 verdict: `tasks/v3_no_gcid_audit_verdict.md`
- V4 verdict: `tasks/v4_no_gcid_audit_verdict.md`
- Ship decision: `tasks/ship_decision.md` (V3 + V4 REVOKED banners)
- Lesson: `lessons.md` entry `2026-04-21 auc_parity_is_not_a_ship_gate`
