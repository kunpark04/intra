# ML Cross-Analysis Report — MNQ Adaptive R:R Stack

**Snapshot**: 2026-04-15
**Purpose**: Synthesized cross-model view of ML#1 + ML#2, what the integrated
stack looks like, where it generalises, and what gates remain before live
capital. Pairs with `tasks/part_b_findings.md` (per-task numerical detail)
and `tasks/ml1_ml2_synthesis_roadmap.md` (task sequencing).

---

## 1. The two models

| | **ML#1** (combo-grain ranker) | **ML#2** (trade-grain P(win)) |
|---|---|---|
| Script | `scripts/ml_optimizer.py` | `scripts/adaptive_rr_model_v2.py` |
| Grain | One row per parameter combo (~33k) | One row per trade × R:R level (~10M) |
| Target | Composite score (Sharpe, return, DD, WR, count) | Binary `would_win` from MFE/MAE synthetic label |
| CV | 5-fold KFold on combos | StratifiedGroupKFold on `global_combo_id` |
| Output | Ranked `top_combos.csv` | Calibrated P(win \| features, R:R) |
| Role in stack | Picks *which* parameter combo to trade | Filters *which* trades to take within a combo |

They were trained on overlapping data (ML#2's expanded rows came from the
same sweeps ML#1 aggregated) — no clean information split. That constrains
joint-validation claims.

---

## 2. What each model is good for — and what it isn't

### ML#1 — combo ranker

- **Strength**: surfaces parameter combos with genuine edge above noise. B3
  permutation confirmed the AUC collapses to random on shuffled labels → the
  ranking signal is real.
- **Original weakness**: it was trained on fixed-R:R outcomes, so it ranked
  combos under an assumption the live stack (ML#2 filtering) violates.
- **Fix (B5)**: retrained on V2-filtered outcomes → produces
  `lgbm_results_v2filtered/`. This is now the canonical combo ranker. The
  pre-B5 `top_combos.csv` is obsolete — zero overlap with the post-B5 top-20.

### ML#2 — per-trade P(win)

- **V1 was broken** (`is_unbalance=True` + `best_iter=5` underfit) — inflated
  high-R:R probabilities, ECE 0.097.
- **V2 fixed** it: AUC 0.806, Brier 0.105, ECE 0.004 post isotonic-per-RR.
- **Held-out variant** (drop `v10_9955` from train) matched V2 → no
  per-combo overfitting at the model level.
- **B9** closed the "use V2 to pick R:R" avenue: the calibrated model prefers
  R:R≈1.0, which cuts profits and violates the original 1:3 floor (later
  relaxed, but the picker is still worse than a fixed R:R per combo).
- **B8** lifted the discrimination ceiling: +0.0333 AUC from four
  autocorrelation features (`prior_wr_10/50`, `prior_r_ma10`,
  `has_history_50`). Recency and regime families added nothing.
- **B8-SHAP** validated the lift is genuine time-local signal, not
  `combo_id` leakage (2 dynamic, 2 partial, 0 identity-proxy).

---

## 3. The integrated stack (what actually wins on in-sample data)

1. **ML#1-v2filtered** ranks combos → candidate set =
   `lgbm_results_v2filtered/top_combos.csv`.
2. **ML#2 (V2) as a filter**, not a picker.
   - B4 extended this to the 50-combo surrogate set.
   - B1 shows the right filter threshold is **per-combo** (low-freq/high-WR
     want strict, high-freq/volume-driven want loose).
   - B2 shows a within-combo E[R] percentile threshold transfers more safely
     across regimes than an absolute cutoff.
3. **Sizing**: 5% fixed baseline, with Kelly-cap5 (B10) on combos where
   calibration permits.
4. **`MIN_RR`**: per-combo, data-driven. The original 1:3 floor was relaxed
   after B5/B9 showed the data prefers ~1:1 under MNQ economics.

Example observed impact on `v10_9955` (training split):

| Variant                       | Sharpe | Return | DD    | WR    |
|-------------------------------|--------|--------|-------|-------|
| Fixed (ML#1 combo only)       | 16.1   | 2285%  | 16.8% | 70.2% |
| V2 adaptive R:R picker        | 14.0   | 1634%  | 13.5% | 73.6% |
| Constrained-band [1.5, 3.0]   | 14.0   | 3244%  | 30.5% | 56.0% |
| Filter E[R] ≥ 0.0             | 29.9   | 358%   | 2.5%  | 98.1% |

(Filter variant gains Sharpe but collapses return from over-pruning on
high-freq combos — the motivation for B1 per-combo and B2 percentile
thresholds.)

---

## 4. Temporal generalisation — where it does and doesn't hold

Three tests stress time:

| Test | Result | Interpretation |
|---|---|---|
| **B6** (V2 on 20% held-out test bars) | AUC 0.8057 → 0.8014 (Δ −0.004); ECE 0.062, mean-bias +6.2 pp | Ranking holds; absolute calibration drifts |
| **B7** (9-fold walk-forward, 2020–2024) | AUC 0.811–0.827, ECE median 0.0037 | Signal stable within training; rolling ≈ expanding → local-in-time, not aggregation-built |
| **B8-SHAP** (within-combo signal decomposition) | 2 dynamic, 2 partial Family-A features | Autocorrelation is real dynamics, not identity proxying |

**Synthesis**: the B6 miscalibration is a post-2024-10-22 regime-shift
event, not continuous decay. The mitigation is a **rolling isotonic
recalibrator** on recent realised trades before any *absolute-probability*
use (Kelly sizing, absolute-E[R] filter thresholds). **Rank-based use**
(percentile filters, combo ranking) transfers without recalibration.

---

## 5. Orthogonal findings that shaped the stack

- **Trade-count bias in sweeps**: high-WR combos in sweep outputs tend to
  be small-sample (median ~180 trades). Any ranking that weights WR
  linearly gets gamed by sparse combos. ML#1 features absorb this;
  hand-picked filters should bucket by trade count.
- **Per-combo behaviour dominates**: every task that looked at filter
  thresholds (B1) or Kelly fractions (B10) found combo-specific answers
  beat any single global parameter.
- **`is_unbalance` is poison on MFE-expanded data**. The imbalance *is*
  the signal — high R:R is genuinely low true P(win).
- **Data pipeline**: MFE parquets cost 2–5× their size on whole-file
  reads; column-subset reads + batched iteration are mandatory. The
  expanded 10M-row feature matrix should be cached once and reused across
  variants.

---

## 6. What is NOT yet validated

- **B16 — the single held-out time evaluation** has never run. Until it
  does, every Sharpe/return number is in-sample.
- **Full-9.5M production retrain on Family A** (green-lit today by
  B8-SHAP) hasn't run. B8's +0.033 AUC was on the 1.18M-trade testbed;
  the full retrain could come in lower after regularisation re-tunes.
- **Portfolio behaviour** (B12): all backtests are single-combo with
  independent 5% risk. Correlations, shared risk budgets,
  diversification — unknown.
- **Regime-conditional models** (B11): whether high-vol vs low-vol
  separate boosters beat one global model is an open question.
- **Live-execution assumptions**: `next_bar_open` fill, zero slippage,
  zero commission. Realistic execution will erode edge, by how much is
  untested.
- **Paper-trade forward** (B17): not started.

---

## 7. Confidence tiers (honest self-assessment)

| Confidence | Claim |
|---|---|
| **High**   | ML#1 ranking is a real signal (B3); V2-as-filter beats V2-as-picker (B9+B4+B5+B10); per-combo thresholds > global (B1+B2); Family-A features are genuine time-local signal (B8+B8-SHAP) |
| **Medium** | V2 calibration drift is regime-local not continuous (B6+B7); rolling isotonic would fix absolute-probability use; full-9.5M retrain will retain most of the +0.033 AUC |
| **Low**    | Any specific Sharpe/return number on filtered stacks (wide CIs from 5% compounding + in-sample only); portfolio behaviour; live execution quality |

---

## 8. Critical path to a live decision

1. **Full-9.5M V2 retrain on Family A + `combo_id` categorical** → produce
   new booster + isotonic calibrator.
2. **Re-run downstream filter backtests** (B1-per-combo, B2-percentile,
   B4-surrogate-50) with the new model; confirm lift holds.
3. **Build rolling-isotonic recalibrator** for absolute-probability use
   cases (Kelly, absolute-E[R]).
4. **B16 final held-out evaluation** — single run on 20% test bars using
   the post-retrain stack. This is the live-capital gate.
5. **B17 paper-trade** forward for 1–2 months before any real capital.

Everything else (B11 regime, B12 portfolio, B13 stack, B14 SHAP-on-ML#1,
B15 fixed-dollar) is optional refinement that can happen in parallel with
or after paper-trade.

---

## Pointers

- Per-task numerical detail: `tasks/part_b_findings.md`
- Task sequencing / roadmap: `tasks/ml1_ml2_synthesis_roadmap.md`
- ML#1 decisions: `tasks/ml_decisions.md`
- ML#2 decisions: `tasks/adaptive_rr_decisions.md`
