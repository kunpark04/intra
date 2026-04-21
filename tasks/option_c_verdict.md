# Option C Diagnostic — Verdict

**Date**: 2026-04-21 07:19 UTC
**Signed commitment**: `tasks/phase5_kill_criterion.md` (signed 07:18:58 UTC at commit `5135706`)
**Source notebook**: `evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb`
**Extraction script**: `tasks/_extract_s3_unfiltered_mc.py`

## Verdict: Branch 2 — ML#2 is additive

The kill branch (deprecate ML#2) was **not triggered**. ML#2 is carrying
meaningful weight even under combo-agnostic V3 training.

## Numbers

### Unfiltered Pool B (V3-no-ID training, no ML#2 applied)

| Metric | Value |
|---|---|
| `sharpe_p50` | **−0.1935** |
| `sharpe_ci_95` | (−1.72, +1.45) |
| `sharpe_pos_prob` | 41.15% |
| `risk_of_ruin_prob` | **97.3%** |
| `dd_p50_pct` | 159.3% |
| `dd_worst_pct` | 977.5% |
| `win_rate` | 28.39% (CI 27.98–28.82%) |
| `n_trades` | 43,846 |
| `trades_per_year` | 30,010.8 |
| `var_5pct_trade` | −$509.16 |
| `cvar_5pct_trade` | −$510.13 |

### Three-way comparison

| | Unfiltered (s3) | V3-no-ID filtered (s6) | Shipped V3 with combo-ID (baseline) |
|---|---|---|---|
| n_trades | 43,846 | 3,299 | 2,791 |
| sharpe_p50 | **−0.19** | **+0.31** | +1.78 |
| sharpe_ci_95 | (−1.72, +1.45) | (−1.37, +1.93) | (+0.23, +3.33) |
| sharpe_pos_prob | 41% | 64% | 98.7% |
| risk_of_ruin_prob | **97.3%** | **53.6%** | 6.9% |
| dd_worst_pct | 977.5% | 271.6% | 123.7% |

### Filter delta (ML#2 contribution, combo-ID-free)

| Impact | Value |
|---|---|
| Trade rejection rate | 92.5% (43,846 → 3,299) |
| Sharpe improvement | +0.50 (−0.19 → +0.31) |
| Ruin reduction | −43.7 pp (97.3% → 53.6%) |
| pos_prob improvement | +23 pp (41% → 64%) |
| dd_worst reduction | 3.6× smaller (977% → 272%) |

**This is real filter work.** Even after removing the `global_combo_id`
leak, ML#2 is knocking ruin from near-certainty (97%) down to
coin-flip-plus (54%) and flipping the Sharpe sign. The filter is
*not* ornamental.

## Mechanical rule application

Per signed criterion clause 1:
- Kill branch trigger: `S ≥ 0.31 AND R ≤ 0.54`
- Observed: `S = −0.19 (FAIL, < 0.31)` AND `R = 0.973 (FAIL, > 0.54)`
- Both conditions fail → kill branch **NOT triggered**

Per signed criterion clause 5:
- Branch 2 applies: "ML#2 is additive (or at least not net-negative) under clean training."
- This does NOT authorize re-shipping V3 or V4. The combo-ID revocations stand.

## What this means

1. The filter layer has a legitimate job. It was not living purely off
   the combo-ID memorization channel — even without `global_combo_id`,
   the 24 remaining features extract enough signal to cut ruin by 44pp
   and recover 0.5 Sharpe over the raw unfiltered pool.

2. The shipped V3 uplift (1.78 Sharpe) was still *mostly* leakage:
   V3-no-ID filtered hits only 0.31. But the leak-free floor is above
   the leak-free ceiling of "no filter at all." The filter does work.
   It just doesn't do enough work to reach the ship gate on its own.

3. The Expansionist's framing ("combo identity is valuable") is
   partially vindicated at the *layer* level — combo-aware signal does
   carry value. But the peer-review rejection of Expansionist's
   *mechanism* (per-combo meta-models / revoked leak in a costume)
   stands: the value has to come from generalizable features, not from
   per-combo memorization.

4. The Contrarian/First-Principles framing ("ML#2 is ornamental") is
   **falsified** by this result. The council's dominant hypothesis was
   wrong. The filter carries 0.5 Sharpe and 44pp ruin reduction under
   clean training.

5. The pre-registered kill criterion did its job: a single-model answer
   might have anchored on the council's hypothesis and misread these
   numbers. The mechanical rule forced the correct reading.

## Next actions per signed criterion

### Upstream (mandatory, independent of any A/B decision) — Task #17

**Audit the Pool B ranker layer itself** for combo-agnostic soundness.
`audit_full_net_sharpe` top-50 was computed over the 13,814-combo pool
that 100% overlaps V3/V4 training. If the ranker is leaky, any A result
is uninterpretable and the 0.31 floor itself may be inflated.

Concrete audit: recompute top-50 raw-Sharpe ranker on a pre-2024 /
walk-forward slice that excludes V3/V4's training window, and compare
basket overlap + s6_net metrics.

### Downstream (after Task #17) — Option A or Option B

- If ranker audit PASSES → Option A (combo-level K-fold retrain with
  pre-committed kill-criterion for the K-fold OOS Sharpe gate) is
  justified. The goal is to test whether combo-ID carries legitimate
  memorization value on combos unseen by that fold.
- If ranker audit FAILS → both A and "accept 0.31 ceiling" are
  invalidated because the 0.31 baseline itself is inflated. In that
  branch, the ship-readiness conversation has to move to a combo-level
  temporal-holdout design, not a modeling decision.

## Irrevocable commitments (independent of any outcome)

Per signed criterion clauses 8–10:

- Option D (multi-day trade-grain learn-to-rank rebuild): **REJECTED**.
- Per-combo meta-models (Expansionist's Option E): **REJECTED**.
- Mandatory pre-ship audit for any future ML#2: combo-agnostic refit +
  ranker-layer audit must both pass before any trade-grain filter is
  declared production-ready. AUC parity is a precondition, never
  sufficient evidence.

## Artifacts

- Source notebook: `evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb`
- Extraction script: `tasks/_extract_s3_unfiltered_mc.py`
- Signed criterion: `tasks/phase5_kill_criterion.md`
- Council report: `tasks/council-report-2026-04-21-phase5.html`
- Council transcript: `tasks/council-transcript-2026-04-21-phase5.md`
