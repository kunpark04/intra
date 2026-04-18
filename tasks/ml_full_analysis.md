# V3 ML Stack — Full-Scale Analysis

**Date**: 2026-04-16
**Scope**: Everything from ML#1 combo ranker through V3 Phase 4f calibration
**Purpose**: Synthesize all results into a coherent picture before B16

---

## 1. Architecture: Two Models, Two Jobs

| | **ML#1 (combo ranker)** | **ML#2 (trade-level classifier)** |
|---|---|---|
| Grain | 1 row per parameter combo (~33k) | 1 row per trade x R:R (~10M) |
| Target | Composite score (Sharpe 0.25, return 0.25, DD 0.2, WR 0.15, PF 0.15) | Binary `would_win` (MFE >= R:R x stop) |
| Output | Ranked combo list | Calibrated P(win \| features, R:R) per trade |
| CV | 5-fold KFold on combos | StratifiedGroupKFold on `global_combo_id` |
| Role | *Which* parameter set to trade | *Which* trades to take, at *what* R:R |

ML#1 selects the instrument; ML#2 operates it. They're complementary, not competing.

**Key metric split**:
- ML#1: R^2 = 0.771 on composite, 0.885 on Sharpe, 0.960 on win-rate
- ML#2 (V3): AUC = 0.808, Brier = 0.106, ECE = 9.3e-7 (OOF, post-isotonic)

ML#1's permutation test confirms signal is real (shuffled R^2 ~ 0.0 vs actual 0.77).

---

## 2. V2 to V3: What Changed, What Improved

### 2.1 Feature evolution

V2 had 20 features. V3 adds:
- **Family A** (4 features): `prior_wr_10`, `prior_wr_50`, `prior_r_ma10`, `has_history_50`
- **`global_combo_id`** as LightGBM native categorical

Family A captures autocorrelation — whether a combo that's been winning
recently is more likely to win next. B8-SHAP confirmed these are dynamic
signals, not identity proxies (within-combo SHAP std > between-combo std
for `prior_r_ma10` ratio=1.77 and `prior_wr_10` ratio=1.32).

### 2.2 The AUC paradox

| Context | V2 AUC | V3 AUC | Delta |
|---|---|---|---|
| B8 testbed (1.18M trades, 418 combos) | 0.8072 | **0.8406** | **+0.033** |
| Full retrain (10M rows, 588k combos) | 0.8057 | 0.8077 | +0.002 |
| Held-out test (B6 baseline) | 0.8014 | 0.8387* | +0.037 |

*B6 V3 number is from `rolling_recal_v3.json` which uses the full V3 booster on test data.

**Why +0.033 on testbed but +0.002 on full retrain?**

The testbed used 418 combos; the full retrain uses 588k subsampled combo
groups. `global_combo_id` as categorical absorbed much of the information
that Family A provided in the small testbed. With 588k combo embeddings,
the model can learn per-combo base rates directly — Family A's
contribution shrinks to incremental temporal dynamics within combos the
model already knows.

Evidence: in V3 feature importance by gain, `global_combo_id` is #2
(16.3M) right behind `candidate_rr` (20.3M). `prior_wr_50` is #4
(2.8M), `prior_r_ma10` #9 (0.7M), `prior_wr_10` #10 (0.5M). Family A
survived but didn't dominate.

### 2.3 Calibration: V3's real win

| Metric | V2 | V3 |
|---|---|---|
| OOF ECE (post-isotonic) | 0.004 | **9.3e-7** |
| OOF Brier (raw) | 0.1048 | 0.1057 |
| OOF Brier (cal) | 0.1048 | 0.1053 |

V3's calibration is essentially perfect in-sample. The near-zero ECE
means the per-R:R isotonic calibrators fit the OOF predictions with no
residual systematic bias. This is a prerequisite for Kelly sizing
(B10) and absolute E[R] thresholds to work.

---

## 3. Temporal Stability: The Good and the Bad

### 3.1 The good: discrimination holds

| Test | AUC range | Interpretation |
|---|---|---|
| B7 walk-forward (2020-2024) | 0.811 – 0.827 | Stable across 5 years |
| B6 temporal OOD (V2) | 0.8014 vs 0.8057 train | Δ = −0.004, negligible |
| B6 V3 on test | 0.8387 | Stronger than V2 OOD |
| Rolling = Expanding in B7 | Nearly identical | Signal is local in time |

The model's ability to *rank* trades (good vs bad) is robust across
time. AUC barely degrades. This means filter-based decisions (top-25%,
EV threshold) transfer to unseen time periods.

### 3.2 The bad: calibration drifts

| Context | ECE |
|---|---|
| V2 OOF (training) | 0.004 |
| V3 OOF (training) | ~0.000 |
| B6 held-out (V2 raw) | **0.062** |
| B6 held-out (V3 raw) | 0.026 |
| B6 held-out (V3 static isotonic) | 0.034 (worse than raw!) |

Static OOF-trained isotonic calibrators **actively harm** held-out
calibration. Mean prediction bias: V2 predicts 0.175 vs observed 0.113
(55% too high). V3 raw predicts 0.139 vs observed 0.113 (23% too high,
better than V2).

**Root cause**: the training distribution's conditional P(win|features)
shifts over time. The isotonic curve fit on OOF data encodes the
training era's base rate, not the test era's. This is *not* a model
failure — it's a data distribution shift. Rolling recalibration is the
correct fix.

### 3.3 The fix: rolling/expanding causal calibration

The Phase 4 series systematically solved this:

| Calibrator | Pooled ECE | Phase |
|---|---|---|
| V3 raw | 0.0260 | — |
| V3 static OOF isotonic | 0.0336 | — |
| Rolling (5000, 500) | 0.0070 | 4 |
| Rolling (20000, 100) — grid optimum | **0.0035** | 4b.3 |
| Expanding causal (refit=100) | 0.0039 | 4c |
| Expanding causal, per-combo R:R-agnostic | 0.0042 | 4f |

**Production pick**: two-stage calibrator from Phase 4f:
1. Per-combo expanding isotonic (R:R-agnostic, refit every 100 trades)
2. Pooled per-R:R expanding isotonic as fallback

This achieves pooled ECE 0.0042 AND median per-combo ECE 0.014.

---

## 4. The Calibration Paradox: Pooled vs Per-Combo

Phase 4d uncovered the most important finding of the calibration work:

| Metric | Value |
|---|---|
| Pooled ECE (expanding) | 0.0039 |
| Median per-combo ECE | 0.043 |
| Ratio | **11x** |

Pooled ECE 0.0039 is dominated by a handful of high-volume,
well-calibrated combos (v10_163 n=95k ECE=0.009, v10_128 n=76k
ECE=0.011). These large combos pull the volume-weighted average down
while hundreds of smaller combos have ECE 0.04–0.18.

**The worst offenders are high-win-rate combos** (mean_y 0.24–0.37,
3x the global mean 0.11). A global isotonic regresses their predictions
toward the pool mean, systematically under-predicting.

Phase 4f fixed this with per-combo R:R-agnostic expanding isotonic:

| Metric | Before (pooled) | After (per-combo) | Change |
|---|---|---|---|
| Median per-combo ECE | 0.043 | **0.014** | −67% |
| Max per-combo ECE | 0.184 | 0.097 | −47% |
| Pooled ECE | 0.0039 | 0.0042 | +0.0003 (flat) |

**Implication for live trading**: pooled calibration is fine for
portfolio-level risk (Kelly sizing on the book). But per-combo EV
thresholds, go/no-go decisions, and combo-specific Kelly fractions
require per-combo calibration. Without it, the best combos get their
signals suppressed.

---

## 5. The Phase 3 Null: Why V3 Filters Underperformed V2

Phase 3 ran V3 through all three filter backtest variants:

| Benchmark | V2 baseline | V3 | V3 > V2 rate |
|---|---|---|---|
| B1 per-combo threshold (40 combos) | median Sharpe 14.35 | 13.00 (−1.69) | 8/40 (20%) |
| B2 top-25% percentile (46 combos) | median Sharpe 7.33 | 7.15 (−1.42) | 8/46 (17%) |
| B4 surrogate EV filter (35 combos) | lift +2.13 | lift +1.66 | 12/35 (34%) |

**V3's +0.033 AUC testbed gain produced worse downstream Sharpe.**

This is the discrimination-filter decoupling problem. V3 ranks trades
better (AUC up), but the *thresholds* calibrated on V2's probability
scale don't transfer to V3's probability scale. The features that
V3 added (Family A + combo_id) help discrimination but don't change
what V2's `candidate_rr` and volume features already captured for
the E[R] filter's decision boundary.

V2's probability-to-EV mapping was already well-tuned to the combos
in the backtest. V3's shifted probabilities need re-optimized thresholds.
The null result says "V3 isn't plug-compatible with V2's filter
parameters," not "V3 is worse."

---

## 6. Feature Importance: What Actually Drives Predictions

### 6.1 V3 feature importance (gain)

| Rank | Feature | Gain | Category |
|---|---|---|---|
| 1 | `candidate_rr` | 20.3M | R:R label |
| 2 | `global_combo_id` | 16.3M | Combo identity |
| 3 | `atr_points` | 3.0M | Volatility |
| 4 | `prior_wr_50` | 2.8M | Family A |
| 5 | `stop_method` | 1.8M | Strategy param |
| 6 | `rr_x_atr` | 1.4M | Interaction |
| 7 | `bar_range_points` | 1.3M | Price action |
| 8 | `parkinson_vol_pct` | 0.9M | Volatility |
| 9 | `prior_r_ma10` | 0.7M | Family A |
| 10 | `prior_wr_10` | 0.5M | Family A |

**Reading**: `candidate_rr` dominates because it's the mechanical
determinant of the label — higher R:R means higher profit target,
lower base rate. `global_combo_id` encodes per-combo intercepts
(some combos just win more). After those two, it's a volatility story
(`atr_points`, `parkinson_vol_pct`, `bar_range_points`) mixed with
autocorrelation (`prior_wr_50`, `prior_r_ma10`, `prior_wr_10`).

### 6.2 B8-SHAP audit: Family A is real signal

| Feature | Within:Between Std | Verdict |
|---|---|---|
| `prior_r_ma10` | 1.77 | **Dynamic** (changes within combo) |
| `prior_wr_10` | 1.32 | **Dynamic** |
| `has_history_50` | 0.91 | Partial |
| `prior_wr_50` | 0.78 | Partial |

Ratios > 1.0 mean the feature's SHAP values vary *more* within a
single combo's trades than between different combos. This confirms
`prior_r_ma10` and `prior_wr_10` capture genuine temporal dynamics,
not combo identity by another name.

`prior_wr_50` has ratio 0.78 but the *highest* mean |SHAP| (0.494) —
it partially encodes combo identity but is the most predictive Family A
feature overall. This is the expected tradeoff: longer lookback windows
are more stable (more identity-like) but also more informative.

---

## 7. Kelly Sizing: Calibration Enables Position Sizing

B10 tested Kelly-fraction sizing on 19 combos:

| Variant | Median lift | Mean lift | Skip rate |
|---|---|---|---|
| kelly_full | +5.15 | +0.24 | 89% |
| kelly_full_cap5 | +7.85 | **+5.31** | 89% |
| kelly_half_cap5 | +7.89 | +3.83 | 90% |
| kelly_quarter_cap5 | **+8.10** | +3.12 | 91% |

Kelly sizing skips ~90% of trades (negative or tiny Kelly fraction).
The remaining 10% are high-confidence, and the capped variants
(max 5% per trade) dramatically improve Sharpe.

**Critical dependency**: Kelly sizing requires *calibrated*
probabilities. With B6's ECE 0.062 (uncalibrated), Kelly fractions
would be systematically wrong. The Phase 4 calibration stack (ECE
0.0042 pooled, 0.014 median per-combo) makes Kelly viable.

This is the causal chain: **Phase 4f calibration → reliable P(win)
→ Kelly fraction → B10's +5-8 Sharpe lift becomes deployable.**

---

## 8. What's Proven vs What's Speculative

### High confidence (multiple independent tests confirm)

| Claim | Evidence |
|---|---|
| ML#1 ranking is real | B3 permutation test (shuffled R^2 ~ 0) |
| V2-as-filter beats V2-as-picker | B9 (monotonic), B4, B5, B10 all agree |
| Per-combo > global threshold | B1, B2, B10 unanimous |
| Family A is genuine signal | B8 (+0.033 AUC), B8-SHAP (dynamic ratios) |
| Discrimination stable over time | B6 (Δ −0.004), B7 (0.811-0.827 range) |
| Static calibration drifts | B6 (ECE 0.062), Phase 4 (static worse than raw) |
| Rolling/expanding fixes calibration | Phase 4 (0.0070), 4b bootstrap (P<gate=1.0) |
| Per-combo calibration matters | 4d (11x gap), 4f (67% median improvement) |

### Medium confidence (single test, strong result)

| Claim | Evidence | Risk |
|---|---|---|
| Kelly-cap5 yields +5-8 Sharpe lift | B10 (19 combos) | Small sample, in-sample only |
| Expanding = Rolling calibration | Phase 4c (Δ 0.0004) | Tested on one distribution |
| V3 OOF calibration ≈ perfect | metrics_v3.json (ECE 9.3e-7) | Could overfit OOF |
| Filter beats fixed on 95% of 27k combos | B5 | In-sample only |

### Low confidence (extrapolation or assumption)

| Claim | Risk |
|---|---|
| Any specific Sharpe/return number transfers to live | All backtest numbers are in-sample or simulated |
| Kelly skip rate is stable in live | Distribution of P(win) may shift |
| Per-combo calibrator generalizes to new combos | No data for novel combos in production |
| 90% skip rate is acceptable for portfolio management | Untested at portfolio level |

---

## 9. The Production Stack (as validated — updated 2026-04-18)

> **Phase 5D update (2026-04-15):** The per-combo two-stage calibrator and
> Kelly sizing described in the pre-Phase-5D version of this diagram have
> been **retired**. Four null-to-negative results across Phases 3, 5A, 5C
> (in-sample) and 5D (OOS / B16) on the per-combo calibrator — plus a 91%
> drawdown for Kelly+two-stage in the B16 portfolio sim — are sufficient
> evidence to drop both. See `tasks/part_b_findings.md` Phase 5D for the
> full post-mortem.

```
Live trade arrives
    |
    v
ML#1 combo ranker (v12 parameter-only surrogate) → is this combo in the top-K?
    |                                              (no → skip)
    v
V3 booster predicts P(win | features, R:R) for 17 R:R levels
    |
    v
Pooled per-R:R isotonic calibration  (data/ml/adaptive_rr_v3/isotonic_calibrators_v3.json)
    |
    v
E[R] = P(win_cal) * R:R - (1 - P(win_cal)) * 1.0           (legacy gross form)
    |          -or-
E[R_net] = P(win_cal) * (R:R * risk - contracts * cost)
           - (1 - P(win_cal)) * (risk + contracts * cost)   (net-of-friction form,
                                                            evaluation notebooks
                                                            since April 2026)
    |
    v
Per-combo / percentile filter: take trade if E[R] > gate
    |
    v
Fixed 5% of equity sizing                                   (not Kelly)
    |
    v
Execute at next-bar open
```

### Key parameters

| Parameter | Value | Source |
|---|---|---|
| R:R levels | 17 (1.0 to 5.0, step 0.25) | V2 design |
| Calibrator | Pooled per-R:R isotonic (17 calibrators, static) | Phase 3 |
| Sizing | Fixed 5% of current equity per trade | Phase 5D |
| Friction | $5/contract round-trip (in-sim since v11 sweep) | commit 8b4bda8 |
| Fill model | Next-bar open | CLAUDE.md |

### Retired components (do not reintroduce without new mandate)

- **Per-combo two-stage calibrator** (`per_combo_calibrators_v3.json`) — artifact retained for reproducibility only.
- **Kelly sizing variants** (`kelly_cap5`, `kelly_twostage`) — failed B16 portfolio gate at 91% drawdown. A narrow overlay (`kelly_simple` with pooled calibrator) is acceptable *only* for low-frequency high-conviction filtering, not the main stack.

---

## 10. Gaps and Risks Before B16

### 10.1 Known gaps

1. **No portfolio-level test (B12)**: all results are per-combo.
   Multi-combo interaction (correlation, drawdown overlap) is untested.

2. **No regime-split model (B11)**: V3 uses one model for all regimes.
   B7 showed 2022 is different (higher base rate, lower AUC). A
   regime-aware model might help.

3. **V3 filter thresholds not re-optimized**: Phase 3 null shows V2
   thresholds don't transfer. V3 needs its own per-combo threshold
   sweep before B16.

4. **Execution model is optimistic**: zero slippage, zero commission,
   next-bar-open fill. Any of these could erode the ~90%-skip Kelly
   regime where marginal trades have thin edge.

5. **Per-combo calibrator cold-start**: new combos have no history.
   Falls back to pooled, which is 11x worse per-combo. B16 should
   report separately for warm (>300 trades) vs cold combos.

### 10.2 What B16 should test

B16 is the **single final held-out eval** — no re-running. It should:

1. Run the full production stack (V3 + two-stage calibrator + Kelly-cap5)
   on the 20% held-out test bars.
2. Report **per-combo** Sharpe, return, drawdown, win-rate, trade count.
3. Report **pooled** ECE and **median per-combo** ECE on test.
4. Compare vs fixed 5%-per-trade sizing as baseline.
5. **Separate warm vs cold combo results** to quantify cold-start risk.
6. Compute portfolio-level equity curve if running multiple combos.

### 10.3 Decision framework for B16

| Outcome | Interpretation | Action |
|---|---|---|
| Kelly-cap5 Sharpe > fixed Sharpe for >60% of warm combos | Stack works | Proceed to B17 paper-trade |
| Kelly-cap5 Sharpe > fixed for 40-60% | Marginal | Investigate failed combos; consider combo-specific sizing |
| Kelly-cap5 Sharpe > fixed for <40% | Calibration doesn't transfer | Abandon Kelly, use rank-only filtering |
| Median per-combo ECE > 0.03 on test | Calibrator needs more work | Add more sophistication to calibrator |
| Pooled ECE > 0.015 on test | Phase 4 results don't transfer | Fundamental recalibration failure |

---

## 11. Summary Numbers (reference card)

### Model performance

| Metric | V2 (OOF) | V3 (OOF) | V3 (held-out) |
|---|---|---|---|
| AUC | 0.8057 | 0.8077 | 0.8387 |
| Brier | 0.1048 | 0.1057 | 0.0841 |
| ECE (calibrated) | 0.004 | 9.3e-7 | 0.0042* |
| Log-loss | 0.3402 | 0.3399 | 0.2756 |

*Phase 4f two-stage calibrator

### Calibration journey (held-out tail)

| Stage | Pooled ECE | Median per-combo ECE |
|---|---|---|
| Raw V3 | 0.0260 | — |
| Static OOF isotonic | 0.0336 | — |
| Expanding per-R:R | 0.0039 | 0.043 |
| Per-combo R:R-agnostic | 0.0042 | **0.014** |

### Filter backtest (in-sample)

| Method | Median Sharpe lift | Win rate (combos) |
|---|---|---|
| B2 top-25% percentile (V2) | +3.87 | 34/48 |
| B4 surrogate EV filter (V2) | +2.65 | — |
| B5 retrain filter | +0.849 | 26,062/27,326 (95%) |
| B10 Kelly-cap5 | +5.31 (mean) | 14/19 (74%) |
