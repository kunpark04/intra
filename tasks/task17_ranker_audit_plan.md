# Task #17 — Pool B Ranker Audit (Plan)

**Date**: 2026-04-21 UTC
**Authority**: Signed Phase 5 kill-criterion clause 6 (`tasks/phase5_kill_criterion.md`)
**Trigger**: Option C Branch 2 triggered — ML#2 filter is additive (+0.50 Sharpe,
−43.7pp ruin) under combo-agnostic training. Option A (K-fold retrain) and
Option B (accept ceiling) are both gated behind this audit.

---

## Operational framing

The "ranker" is not a learned model —
`scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py:96` is literally
`eligible.sort_values("audit_full_net_sharpe").head(50)`. "Ranker leakage"
therefore means **basket selection bias**: Pool B is the 50 combos that
performed best on the train partition V3 was trained on. The 0.31 heldout
Sharpe may partially reflect those 50 combos being robust on heldout simply
because they were the most robust combos in the universe.

**Test**: Rank combos by a different **combo-agnostic** robustness signal.
Evaluate filter uplift on the alternate basket. If filter uplift persists,
V3-no-gcid's filter carries weight beyond basket selection → Option A
authorized. If filter uplift collapses, the 0.31 floor is inflated by
combo selection → Option A's K-fold will inherit the same bias; redesign
as combo-level temporal holdout instead.

## Pre-registered interpretation thresholds (binding, clause 6 discipline)

Let **`U_prod` = +0.50** (Option C measurement: filtered +0.31 − unfiltered −0.19).
Let **`U_alt`** = filter uplift on alternate basket.

| `U_alt` | Verdict |
|---|---|
| ≥ **+0.40** | Filter generalizes beyond production basket → Option A (K-fold retrain) authorized |
| [+0.20, +0.40) | Ambiguous → require second alternate basket (C by `target_robust_sharpe`) before Option A |
| < **+0.20** | Selection-dependent → Option A's K-fold will inherit bias → pivot to combo-level temporal holdout or Option B's forcing-function script |

Ties at exact thresholds resolve against Option A (same stricter-prior
discipline as the kill-criterion).

## Alternate rankers (all combo-agnostic, no learned parameters)

All three live in `data/ml/ml1_results_v12/combo_features_v12.parquet`:

1. **Production**: `audit_full_net_sharpe` — full-train-partition 1-contract net Sharpe
2. **Alternate B**: `target_median_wf_sharpe` — median across N=5 ordinal walk-forward
   windows inside the train partition (temporal-robustness signal,
   `build_combo_features_ml1_v12.py:248`)
3. **Alternate C** (tiebreaker if Phase 3 hits the ambiguous band):
   `target_robust_sharpe = median − 0.5·std` — ML#1 v12's training target,
   penalizes within-train window-to-window instability

Eligibility gate (same for all three): `audit_n_trades >= 500`.

## Phases

### Phase 0 — Basket-overlap baseline (local, ~15 min)

**Script**: `tasks/_ranker_audit_overlap.py`

- Load `combo_features_v12.parquet`; apply `audit_n_trades >= 500` gate
- Compute top-50 baskets for rankers A, B, C
- Report Jaccard overlap matrix (A∩B, A∩C, B∩C)
- Report Spearman rank correlation on full eligible pool
- Report per-basket distribution stats

**Decision gate**:

- If `Jaccard(A,B) ≥ 0.80` **and** `Jaccard(A,C) ≥ 0.80` → baskets
  near-identical, no divergence to probe → land clearance verdict, skip
  Phases 1–2, Option A authorized on structural grounds in Phase 3.
- Otherwise → alternate basket B is meaningfully different → run Phases 1–3.

### Phase 1 — Alternate-basket payload + notebook generation (local, ~20 min)

Only if Phase 0 fails the 80% gate.

- Generalize `extract_top_combos_by_raw_sharpe_v12.py` with `--ranking-col`
  → emit `evaluation/top_strategies_v12_median_wf_top50.json`
- Extend `scripts/evaluation/_build_v2_notebooks.py` to generate
  `evaluation/v12_topk_top50_median_wf_net_v3_no_gcid/` via `_build_net_variant`
- Ship `n_sims=2000` patch into generated s3 at build time (commit 2105b45
  pattern) — avoid the Apr 20 22:42 kernel-death repro

### Phase 2 — Remote s3-s6 execution (sweep-runner-1, ~5–10 min)

Only if Phase 1 ran. V3-no-gcid already trained — no refit.

- Git sync + paramiko upload of new notebooks
- Execute under `systemd-run --scope -p MemoryMax=9G` screen session
- Pull artifacts via `_pull_v3_no_gcid_s6.py` pattern

### Phase 3 — Verdict

**Artifact**: `tasks/task17_ranker_audit_verdict.md`

Fill in the table:

| Basket | Unfiltered Sharpe p50 | Filtered Sharpe p50 | Filter uplift |
|---|---|---|---|
| Production `audit_full_net_sharpe` | −0.19 | +0.31 | +0.50 |
| Alternate `target_median_wf_sharpe` | ? | ? | `U_alt` |

Apply pre-registered thresholds mechanically, land the verdict, update
`tasks/ship_decision.md` + `CLAUDE.md` if Option A is authorized or blocked.

## References

- Option C verdict: `tasks/option_c_verdict.md`
- Signed kill-criterion: `tasks/phase5_kill_criterion.md`
- Council report: `tasks/council-report-2026-04-21-phase5.html`
- Combo-overlap structural finding: `memory/project_combo_overlap_structural.md`
- Production ranker: `scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py`
- Feature builder: `scripts/analysis/build_combo_features_ml1_v12.py`
