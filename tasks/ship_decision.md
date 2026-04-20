# Pool B Ship Decision

**Date**: 2026-04-19 / 2026-04-20
**Status**: ⚠️ **PROVISIONAL — ship decision on HOLD pending V4 combo-ID leak audit** (2026-04-20 14:00 CDT)
**Plan**: `C:\Users\kunpa\.claude\plans\sleepy-roaming-kettle.md`

## ⚠️ PROVISIONAL BANNER (2026-04-20)

The Sharpe p50 2.13 figure below was produced by an ML#2 V4 filter that
trains `global_combo_id` as a LightGBM categorical feature
(`scripts/models/adaptive_rr_model_v4.py` lines 69, 72, 74). This is a
per-combo memorization channel: V4 has seen every shipped top-50 combo
during training and can recover its in-sample win rate directly from the
ID. A naive random-K null test would be fooled by this leak.

The LLM Council (transcript 2026-04-20 13:49 CDT, `evaluation/council/
council-transcript-20260420_134920.md`) unanimously recommended Option 1:
refit V4 combo-agnostic (drop `global_combo_id`) and re-audit s6_net on
the shipped top-50 before any further evaluation work. `adaptive_rr_v3`
is contaminated by the same pattern (L64, L67, L69) but is not on the
shipped path; V3's audit is scoped after V4.

**All downstream claims in this document are contingent on the refit
V4's s6_net CI overlapping the shipped 2.13**. If the CI collapses,
paper-trading halts and the pipeline is re-architected.

Tracked in tasks #7–#11. Banner will be removed once the combo-agnostic
V4 ship-blocker audit completes.

## Context

The 2026-04-19 Pool B ship work (raw-Sharpe top-50 eval suite, commits
`2105b45` + `734d3a3`) produced Sharpe p50 2.13 on the `s6_net` MC
(ML#2 V4-filtered, net of $5/contract RT friction). The LLM council
issued a HOLD verdict. Three validation gates were defined:

1. **Rank stability** — is the top-50 a stable set or a lottery ticket?
2. **Bundle attribution** — Pool B top-K vs V4 upgrade, which moved the needle?
3. **ML#1 necessity** — does ML#1 add value over raw full-pool + V4 filter?

All three were executed remotely on sweep-runner-1. This document
consolidates the verdicts.

## Arm comparison

| # | Arm | Sharpe p50 | CI [5, 95] | DD worst | Ruin % | Trades | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Prod (UCB + V4) | 0.88 | [-0.72, +2.45] | — | 1.37 | — | Prior production baseline |
| 2 | Pool B + V4 (shipped) | 2.13 | [+0.56, +3.64] | ~0 | 0.02 | — | Raw-Sharpe top-50, Step 0 |
| 3 | Pool B + V3 | 1.78 | [+0.23, +3.33] | 123.7% | 6.9 | — | Step 3 unbundle — V3 filter arm |
| 4 | Full pool + V4 (500 subsample) | **0.00** | **[0.00, +2.02]** | **280.4%** | **43.45** | 24,542 | Step 4 — catastrophic collapse; ML#1 concentration is load-bearing |
| 5 | (ref) Pool B + no filter | n/a | n/a | n/a | n/a | n/a | not run — no filter means no ML#2 gate |

Source notebooks:
- Row 2 → `evaluation/v12_topk_top50_raw_sharpe_net_v4/s6_mc_combined_ml2_net.ipynb`
- Row 3 → `evaluation/v12_topk_top50_raw_sharpe_net_v3/s6_mc_combined_ml2_net.ipynb`
- Row 4 → `evaluation/v12_full_pool_net_v4_2k/s6_mc_combined_ml2_net.ipynb` (pending)

## Rank stability (Step 2)

Ordinal-shift resplit approximation (calendar-time shift infeasible
without per-trade timestamps in the v12 features). Shifts applied to
the 80/20 train/val cut in `audit_n_trades` space:

| Arm shift | Jaccard vs baseline top-50 |
|---|---|
| −180d equivalent | 0.59 |
| −90d equivalent | 0.71 |
| +90d equivalent | 0.70 |
| +180d equivalent | 0.61 |
| **Mean** | **0.653** |

**Verdict**: 0.653 sits in the 0.40–0.70 "noisy but real signal" band.
Above the 0.40 abort floor, below the 0.70 stability threshold. Ship
is not blocked by rank stability; noise is a caveat to note.

Artifact: `data/ml/ml1_results_v12/rank_stability_pool_b.json`

## Bundle attribution (Step 3)

Pool B + V3 (row 3): Sharpe 1.78 [+0.23, +3.33]
Pool B + V4 (row 2): Sharpe 2.13 [+0.56, +3.64]
UCB + V4 (row 1): Sharpe 0.88 [-0.72, +2.45]

- Pool B + V3 > UCB + V4 by 0.90 Sharpe → **Pool B selection contributes meaningfully on its own**.
- Pool B + V4 > Pool B + V3 by 0.35 Sharpe → **V4 filter adds incremental value**; the CIs overlap, so the V4 effect size is uncertain but directionally positive.
- Pool B + V3 DD worst 123.7% is concerning — the V4 filter appears to do real drawdown-containment work on top of V3's selection.

**Verdict**: genuinely bundled. V3 carries most of the signal, V4 adds
drawdown containment. Both components pull weight.

## ML#1 necessity (Step 4 — RESOLVED)

Full-pool 500-uniform subsample (seed=42) of the 13,814 post-gate combos
(plan §Risk & rollback fallback). Two upstream failures delayed the
run — per-cell timeout at 7200s, then s3 OOM at the 9G cgroup cap on
unfiltered MC. Third attempt (`run_skip_s3.sh`) skipped s3 in both
dirs and completed all 8 remaining notebooks in 2 min 1 sec once
`results_raw.pkl` (254.8 MB) + `_ml2_cache_v4.pkl` (49.6 MB) caches
were warm. Decision relies on s6 only (plan §Step 4 explicit fallback
— "gate the decision on s6 only" if s3 OOMs).

**Outcome: catastrophic collapse, matching the third expected outcome.**

| Metric | Full pool + V4 | Pool B + V4 | Ratio |
|---|---|---|---|
| Sharpe p50 | 0.00 | 2.13 | — |
| Sharpe upper 95% | 2.02 | 3.64 | ≈ Pool B's p50 |
| DD worst | 280.4% | ~0% | 2.8× blown out |
| Ruin prob | 43.45% | 0.02% | 2173× worse |
| Trades | 24,542 | ~200 | 123× more churn |
| Sharpe pos-prob | 5.25% | high | — |

The full-pool portfolio generates 123× more trades per year (16,798
vs Pool B's few hundred) but with a 34.98% win rate and no
concentration gate. ML#2 V4's per-trade filtering alone cannot
compensate. A 43% ruin probability on a $50k account is a
non-starter.

**Verdict**: ML#1 top-K selection is doing **critical concentration
work**. Removing it — even with V4 filtering retained — produces a
portfolio that blows out the account with 43% probability. Keep the
two-stage pipeline.

Artifact: `evaluation/v12_full_pool_net_v4_2k/s6_mc_combined_ml2_net.ipynb`

## Decision logic (resolved)

```
if jaccard_mean < 0.40:
    ABORT — Pool B signal too noisy
    → NOT triggered (0.653 ≥ 0.40)                              [PASS]

if Full pool + V4 Sharpe CI ≥ Pool B + V4 CI (overlap at lower bound):
    SHIP Full pool + V4 (simpler — drop ML#1)
    → NOT triggered (full-pool Sharpe p50=0.00 vs Pool B 2.13;
       upper 95% of full-pool = 2.02 < Pool B p50 = 2.13;
       DD 280% disqualifies regardless)                         [FAIL]

elif Pool B + V3 ≈ Pool B + V4 within CI overlap:
    SHIP Pool B + V3 (simpler — drop V4 upgrade)
    → NOT triggered (V3 DD 123.7% vs V4 ~0% disqualifies
       the V3-only path)                                        [FAIL]

elif Pool B + V4 strictly dominates:
    SHIP Pool B + V4 (current bundle)
    → TRIGGERED                                                 [SHIP]

else:
    HOLD — reopen investigation
```

## Final verdict: SHIP Pool B + V4

All three council validation gates passed or triggered the ship branch:

1. **Rank stability (0.653)** — noisy but real signal; above the 0.40 abort floor. Recorded as caveat.
2. **Bundle attribution** — V3 carries most of the raw signal (+0.90 Sharpe over UCB baseline); V4 adds drawdown containment (+0.35 Sharpe, and crucially drops DD from 123.7% to ~0%). Both components pull weight; the V3-only path is disqualified by drawdown.
3. **ML#1 necessity** — catastrophic collapse without the top-K gate. Full pool + V4 produces a 43.45% ruin probability and 280% peak drawdown. ML#1 is the load-bearing concentration mechanism.

The shipped pipeline (raw-Sharpe top-50 from v12 post-gate pool → ML#2 V4 filter → fixed-$500 sizing) produced the best Sharpe CI this project has ever recorded on OOS data (p50 = 2.13, CI [+0.56, +3.64], ruin 0.02%). All proposed simplifications make it worse.

## Rollback path

- All new artifacts live in new directories:
  `evaluation/v12_full_pool_v4_2k/`, `evaluation/v12_full_pool_net_v4_2k/`,
  `evaluation/v12_topk_top50_raw_sharpe_{v3,net_v3}/`.
- Revert = delete new notebook dirs + revert generator commits on
  `scripts/evaluation/_build_v2_notebooks.py` + delete
  `evaluation/top_strategies_v12_full_pool_2k.json`.
- No production signal change until this doc's verdict is committed.

## Follow-up queue (explicitly out of scope for ship decision)

- ML#1 v13 retrain aimed at OOS Sharpe (not just in-sample fit).
- Calendar-time rank stability once per-trade timestamps are wired
  into the v12 feature pipeline.
- Multi-instrument expansion.
- `CLAUDE.md` update retiring v12 UCB — deferred to post-ship docs commit.
- `tasks/lessons.md` entry: "always ablate vs trivial baselines" — separate commit.

## Next action (one concrete step)

**Commit Pool B + V4 as the production pipeline.** Specifically:

1. Commit the Step 4 notebooks + logs + this doc (this repo commit
   constitutes the ship gate).
2. In a follow-up commit, update `CLAUDE.md` to note that v12 UCB is
   retired in favor of v12 raw-Sharpe top-50 + V4 filter.
3. In a follow-up commit, add a `tasks/lessons.md` entry: "always
   ablate vs the simplest possible baseline" — the Step 4 catastrophic
   result was the single most decisive piece of evidence in this
   gauntlet; without it, the HOLD could not have resolved to SHIP.

## Step 4 execution notes (for incident record)

Three attempts were needed:

| Attempt | Wrapper | Screen | Outcome |
|---|---|---|---|
| 1 | `run_full_pool_v4_2k.sh` (timeout 7200s) | `full_pool_2k` | s1 `CellTimeoutError` at 2h wall on 500-combo composed backtest |
| 2 | same, timeout 14400s | `full_pool_2k` (relaunched) | s1 + s2 completed with warm cache; s3 OOM-killed at 9G cgroup cap on unfiltered MC |
| 3 | `run_skip_s3.sh` (skips s3 in both dirs) | `full_pool_skip` | SUITE COMPLETE in 2 min 1 sec, all 8 target notebooks executed, caches warm |

Logs preserved at:
- `evaluation/_step4_full_pool_v4_2k.log.timeout1`
- `evaluation/_step4_full_pool_v4_2k.log.oom_s3`
- `evaluation/_step4_skip_s3.log`

s3 skeletons (`s3_mc_combined.ipynb` in both dirs) are intentionally
left as unexecuted — per plan §Step 4 fallback, the decision rests
on s6 (ML#2-filtered MC). s3 would be an unfiltered-MC reference only,
and the decision does not depend on it.
