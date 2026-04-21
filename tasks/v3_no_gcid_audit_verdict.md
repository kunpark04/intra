# V3 Combo-Agnostic Ship-Blocker Audit — VERDICT: FAIL

**Date**: 2026-04-21 UTC
**Result**: 🛑 **REVOKE V3 production status — halt paper-trading**
**Authority**: `tasks/plan_v3_audit_and_ranker_null.md` Phase 3 (post-V4-FAIL contingency)
**Precedent**: V4 combo-agnostic audit FAIL (`tasks/v4_no_gcid_audit_verdict.md`, 2026-04-20)

## Summary

The shipped V3 + Pool B configuration — which the 2026-04-18 s6_net MC
produced at **Sharpe p50 1.78, 95% CI (+0.23, +3.33), ruin 6.93%** — was
leaking via the `global_combo_id` LightGBM categorical feature (same
pattern as V4, lines 64, 67, 69 of `scripts/models/adaptive_rr_model_v3.py`).
When V3 is refit without `global_combo_id`
(`data/ml/adaptive_rr_v3_no_gcid/`, commits `d66af88`, `4c8bd98`,
`3e40223` for V4 infra; V3 training run 2026-04-20, OOF AUC 0.8293) and
re-audited on the identical Pool B top-50 with identical MC kernel, every
ship-decision metric collapses.

## Head-to-head (s6 net, identical Pool B, identical MC kernel)

| Metric | Shipped V3 (with `global_combo_id`) | V3 combo-agnostic (`v3_no_gcid`) | Delta |
|---|---|---|---|
| `n_trades` filtered | 2,791 | **3,299** | +18% retention |
| `win_rate` | 30.07% (CI 28.41–31.75) | **26.30%** (CI 24.77–27.80) | **−3.77 pp, CIs disjoint** |
| `sharpe_p50` | +1.7822 | **+0.3054** | **−1.48 Sharpe** |
| `sharpe_ci_95` | (+0.231, +3.327) | **(−1.369, +1.929)** | **crosses zero** |
| `sharpe_pos_prob` | 98.7% | **64.17%** | **−34.5 pp** |
| `dd_worst_pct` | 123.69% | **271.6%** | **2.2× worse** |
| `risk_of_ruin_prob` | 6.93% | **53.62%** | **7.7× worse — ruin-majority regime** |
| `trades_per_year` | 1,910.3 | 2,258.0 | +18% |

Source notebooks:
- Baseline → `evaluation/v12_topk_top50_raw_sharpe_net_v3/s6_mc_combined_ml2_net.ipynb`
- Combo-agnostic → `evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s6_mc_combined_ml2_net.ipynb` (this commit)

V3 combo-agnostic training metrics (`data/ml/adaptive_rr_v3_no_gcid/metrics_v3.json`):
- OOF AUC **0.8293** (vs shipped V3 AUC 0.8293 — near-identical, confirming the model is still learning the same signal)
- 24 features (no `global_combo_id`)
- 9,999,995 rows × 588,235 combos, 5-fold CV, 7029s runtime

## Interpretation

### V3 leak is reweighting, not rejection

Where V4-no-ID admitted **+126% more trades** (1,446 vs 641 — the ID was
acting as a rejector of combos it had seen lose), V3-no-ID admits only
**+18% more trades** (3,299 vs 2,791). Yet V3's Sharpe collapse is almost
as severe. The V3 combo-ID channel was not primarily a rejector — it was
a **per-combo reweighter**, biasing `P(win | features, candidate_rr)`
upward for the 50 top combos that the model had trained on, then a net-E[R]
filter let those biased P(win) values gate more trades per combo. A
modest basket expansion (+18%) is enough to reveal the bias because the
biased P(win) values were pushing the filter bar artificially low on the
original combos too.

### 64% pos_prob / 54% ruin is the cleanest falsifier

`sharpe_pos_prob = 0.6417` means 36% of 10,000 IID bootstrap paths
produce a **negative** Sharpe. The shipped baseline sat at 98.7%.
`risk_of_ruin_prob = 0.5362` means more than half of 10k paths exceed
50% drawdown. No sizing or cost tweak will rescue this regime.

### The OOF AUC invariant did not save us

V3-no-ID OOF AUC is 0.8293 — essentially identical to V3-with-ID. That
means the model still learns the per-trade signal well from the 24
remaining features. But **AUC is a trade-level ranking metric and does
not capture the combo-level calibration bias** that drives the s6_net
ship decision. The AUC invariant from Phase 2 (commit TBD) was a
necessary but not sufficient gate. Future audits must treat AUC parity
as a PRE-CONDITION for deeper tests, not as evidence of ship readiness.

### Both V3 and V4 are now out

As of 2026-04-21 UTC, **both V3 and V4 ML#2 stacks fail the combo-agnostic
ship-blocker audit**. The 2026-04-20 V4 revocation (Pool B + V4) now
extends to V3. CLAUDE.md's line 58 — "ML#2 production stack (as of Phase
5D, 2026-04-15): V3 LightGBM booster" — must be updated to reflect that
V3 is no longer certified for production until a clean redesign passes
the same audit.

## Action items

1. ✅ **Revoke V3 production status**. Add permanent REVOCATION banner to
   `tasks/ship_decision.md` (now listing V3 + V4 as revoked).
2. ✅ **Halt paper-trading** (already halted after V4 revocation).
3. ✅ **Update CLAUDE.md** to reflect V3 + V4 both out of production.
4. ✅ **Update memory index** (`project_council_plan_stage.md`,
   `project_v4_combo_id_leak_confirmed.md`, add new
   `project_v3_combo_id_leak_confirmed.md`).
5. ❌ **Phase 5 fork → deep redesign escalation**. Neither V3 nor V4 can
   be shipped in their current form. The redesign options are:

   **Option A — Combo-level K-fold retrain**. Partition the 588,235
   training combos into 5 folds by combo identity; train V3 on 4/5,
   predict on 1/5. This removes the memorization channel without
   removing the feature. Cost: ~5× a single training run (~10 hours on
   sweep-runner-1). Gate: if the K-fold OOS Sharpe on the held-out
   combos matches the unfolded Sharpe, the combo-ID feature is
   justified. If not, the feature is confirmed as a leak and must be
   permanently dropped.

   **Option B — Feature-only redesign (status quo combo-agnostic)**.
   Accept the current V3-no-ID metrics as the honest ceiling. Sharpe
   p50 0.31 with pos_prob 64% is not shippable, but it is the true
   out-of-sample expectation. No further model work required — we
   would instead return to combo-scouring (finding better raw-Sharpe
   combos in the v11 sweep) or feature engineering on the 24-feature
   set.

   **Option C — Abandon ML#2 entirely**. If the unfiltered Pool B
   stream has an honest Sharpe ≥ the ML#2-filtered stream, ML#2 is not
   adding value post-leak. Audit s3_mc_combined_net (unfiltered) under
   V3-no-ID to check. If unfiltered Sharpe ≥ 0.31, the ML#2 layer is
   pure overhead.

   **Option D — Rank-based redesign**. Replace the binary classifier
   with a learning-to-rank objective over combo-level Sharpe, using
   feature-only inputs. This is a larger architectural change but may
   align better with the actual use case (picking the best 50 combos
   from a universe of 13,814).

   Recommendation is to start with Option A (K-fold retrain) because it
   answers the question directly: *is combo-ID a leak or a legitimate
   memorization signal?* If A passes, we re-ship V3 with K-fold
   training. If A fails, Options B/C/D are the only remaining paths.

## Reproduction

```bash
# From repo root, all on sweep-runner-1 (root@195.88.25.157)
python scripts/models/adaptive_rr_model_v3.py --no-combo-id \
  --output-dir data/ml/adaptive_rr_v3_no_gcid
# -> data/ml/adaptive_rr_v3_no_gcid/booster_v3.txt
#    data/ml/adaptive_rr_v3_no_gcid/isotonic_calibrators_v3.json
#    data/ml/adaptive_rr_v3_no_gcid/metrics_v3.json  (AUC 0.8293)

python tasks/_restart_v3_no_gcid_s3plus.py  # local driver
# -> evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s{3..6}_*.ipynb executed
```

MC kernel parameters (all bootstrap samples):
- `n_sims = 2000` for s3 unfiltered (pin for 9G memory cap; see CLAUDE.md)
- `n_sims = 10000` for s6 filtered (post-filter pool fits comfortably)
- IID bootstrap on `net_pnl_dollars` (cost: $5.00/contract RT)
- `fixed_dollars_500` sizing ($500 risk per trade)
- YEARS_SPAN = 1.461 (test partition 2024-10-22 → 2026-04-08)
