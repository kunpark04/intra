# Phase 4.1 Finding — 100% Combo Overlap, Null Test Infeasible

**Date**: 2026-04-21 UTC
**Authority**: LLM Council step 6 (ranker-label overlap null)
**Verdict**: Phase 4.2–4.5 BLOCKED — the held-out partition is empty.

## Summary

Streamed `data/ml/originals/ml_dataset_v11.parquet` (102,220,938 rows) against the
13,814 post-gate combos in `data/ml/ml1_results_v12/combo_features_v12.parquet`
to count per-combo trade-row overlap.

## Result

| Metric | Value |
|---|---|
| Combos labeled | 13,814 |
| Combos with `trades_in_training > 0` | 13,814 (100%) |
| Combos with `trades_in_training == 0` | 0 |
| `overlap_pct` min / median / max | 1.000 / 1.000 / 1.000 |
| `overlap_pct` mean | 1.000 |
| Total trade rows streamed | 102,220,938 |
| Trade rows in-universe (matched to the 13,814 combos) | 100,983,394 |
| Runtime | 5.6 s (18.2M rows/s) |

## Interpretation

This is a **structural** result, not a bug. `combo_features_v12.parquet` was
aggregated from the same v11 sweep trades that V3/V4 train on; every combo's
trade rows are necessarily present in the training partition. There is no
clean held-out partition available under the current data construction.

## Implications

1. **Phase 4.2 (held-out Pool B construction) is infeasible** as designed —
   zero combos survive the `trades_in_training == 0` filter, and per the
   plan's contingency at line 202, threshold-based filtering with
   `overlap_pct < threshold` also fails because all `overlap_pct` values are
   exactly 1.0.
2. **The Phase 3 V3 combo-agnostic audit now bears full weight** as the
   remaining test of whether the ranker+filter interaction leaks via
   per-combo memorization. By stripping `global_combo_id`, V3 combo-agnostic
   removes the memorization channel directly, which is a stronger
   intervention than the held-out-partition null would have provided.
3. **If a full ranker-label overlap null is ever needed**, it must be
   redesigned as **combo-level K-fold**: partition the 13,814 combos into K
   folds, train V3 on 4/5 and evaluate Pool B ranking on the held-out 1/5.
   This is ~5x the training cost and is not on the current plan. Defer until
   Phase 3 verdict is known.

## Artifacts

- Output: `data/ml/ranker_null/combo_overlap_labels.parquet` (13,814 rows, 224 KB)
- Log: `tasks/_overlap_labels_remote.log` (full remote run transcript)
- Commits: `ce2be3a` (script), `f1822fd` (path fix)

## Next step

Block on Phase 3 verdict. If Phase 3 passes combo-agnostic, the
ranker+filter leak concern is substantially attenuated. If Phase 3 fails,
a combo-level K-fold redesign may be worth the cost. Either way, Phase
4.2–4.5 as originally specified is not actionable.
