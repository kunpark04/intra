# Part B — Consolidated Findings (as of 2026-04-14)

Running log of every Part B task completed, their key numerical results, and
the takeaway from each. Anchor for the next round of decisions. Pairs with
`tasks/ml1_ml2_synthesis_roadmap.md`.

---

## Scoreboard

| ID  | Task | Status | Artifact(s) |
|-----|------|--------|-------------|
| B1  | Per-combo optimal filter threshold | Done | `data/ml/adaptive_rr_v2/filter_backtest_per_combo.json` |
| B2  | Adaptive threshold — E[R] percentile | Done | `data/ml/adaptive_rr_v2/filter_backtest_percentile.json` |
| B3  | Permutation test on ML#1 | Done | memory 1466 (signal confirmed > noise) |
| B4  | Filter on surrogate top-50 | Done | `data/ml/adaptive_rr_v2/filter_backtest_surrogate.json` |
| B5  | Re-train ML#1 on V2-filtered outcomes | Done | `data/ml/lgbm_results_v2filtered/` |
| B9  | Monotonic constraint on `candidate_rr` | Done | `data/ml/adaptive_rr_b9/` |
| B10 | Kelly-fraction sizing from calibrated P(win) | Done | `data/ml/adaptive_rr_v2/kelly_backtest.json` |
| B6  | Temporal OOD test for V2 (test-partition bars) | Done | `data/ml/adaptive_rr_v2/b6_temporal_ood.json` + `b6_reliability.png` |
| B7  | Walk-forward validation (expanding + rolling) | Done | `data/ml/adaptive_rr_v2/b7_walk_forward.json` + `b7_walk_forward.png` |
| B8  | Feature engineering ablation (autocorr/recency/regime) | Done | `data/ml/adaptive_rr_v2/b8_feature_eng.json` |

Outstanding: B11–B15 tier-3 · B16 held-out · B17 paper-trade.

---

## B1 — Per-combo optimal filter threshold

- 60 combos swept (10 existing + 50 surrogate), 21-point grid `[-0.5, +0.5]`
  step 0.05, min 20 trades for eligibility.
- **Optimum threshold distribution is bimodal**: mass at guardrails (±0.5)
  and near zero. Median = 0.05, mean = 0.09.
- Top modes: `0.5` (6), `0.4` (4), `0.45` (4), `0.05` (4), `0.0` (4).
- **WR × threshold correlation = 0.341** (mem 1497) — high-WR combos prefer
  higher thresholds, confirming the single-global-threshold was over-pruning
  high-freq / low-WR combos.

**Takeaway**: per-combo thresholding is real and materially different from a
single global rule. Drives B5's design (let the threshold be a learned combo
attribute).

---

## B2 — Adaptive threshold (E[R] percentile)

- Grid: keep the top 25 / 50 / 75 / 90 percentile of E[R] trades within
  each combo. 60 combos attempted, 48 evaluated (12 errored).
- Script stored results under `top_Npct` keys directly; no `optimum` picker
  step was implemented (cosmetic — summary is below).

| Percentile kept | wins vs fixed | median Sharpe lift |
|---|---|---|
| top_25pct | 38 / 48 | **+3.87** |
| top_50pct | 44 / 48 | +2.66 |
| top_75pct | 43 / 48 | +1.45 |
| top_90pct | 40 / 48 | +0.50 |

- **Best bucket per combo**: top_25pct wins on 34/48 combos (71%),
  top_50pct on 9, top_75pct on 3, top_90pct on 2.
- Means are distorted by a handful of degenerate zero-DD Sharpe blow-ups;
  medians are the authoritative summary.

**Takeaway**: percentile filtering works, and the aggressive top-25%
variant is best for most combos. Consistent with B1: combos genuinely
prefer per-combo quantile pruning over a global absolute threshold.

---

## B3 — Permutation test on ML#1

- Ran V1-style surrogate training with shuffled targets. Real-label AUC
  materially > permuted AUC (mem 1466). Sanity check passes — ML#1's
  surrogate is learning signal, not overfitting noise.

**Takeaway**: ML#1 surrogate has genuine structure. Green-light to keep
retraining it (B5) rather than suspecting leakage.

---

## B4 — Filter on surrogate top-50

- Thresholds 0.0, 0.1, 0.25 applied to ML#1 surrogate's top-50 combos.
- **Median Sharpe lift = +2.65** across 50 combos × 3 thresholds.
- Replicates the original 11-combo filter finding at a larger population:
  V2 filter on top of ML#1 recommendations generally helps.

**Takeaway**: filter on ML#1's top picks is directionally positive; not
universal. Motivates B1's per-combo threshold and B5's proper retrain.

---

## B5 — Retrain ML#1 on V2-filtered outcomes (biggest result)

- Built combo-level features for **27,326 combos** across v2–v10 by running
  each combo's trades through V2 and picking the best-Sharpe filter
  threshold (guardrails: n_kept ≥ 50, Sharpe < 1e10).
- Retrained ML#1 surrogate on filtered metrics (LightGBM, 5-fold, 50K
  random surrogate candidates).

**Result highlights**:
- Filter beats fixed on **26,062 / 27,326 combos (95%)**; median
  `filter_lift_sharpe = +0.849`.
- Optimal threshold distribution is **bimodal** — modal bucket 0.0 (7041
  combos, 26%), heavy tails at ±0.5 guardrails (5588 at +0.5, 1394 at −0.5).
- **Zero overlap** in top-20 combos between unfiltered ML#1 and V2-filtered
  ML#1 — the filter completely reshuffles the ranking, not a perturbation.
- V2-filtered top-10 signature: WR 95–99%, n_trades 120–240 per combo
  (post-filter high-precision regime).

**Takeaway**: the stack integration works. Filtered combos are a genuinely
different population and many combos gain multiple Sharpe points when the
filter is applied. B5 is the new authoritative ML#1 ranking.

---

## B9 — Monotonic constraint on `candidate_rr`

- Trained V2 with LightGBM `monotone_constraints` forcing P(win) to
  decrease as `candidate_rr` increases (path fact).
- **Overall CV metrics (identical to V2)**: AUC 0.806, log-loss 0.340,
  Brier 0.105. Ran 9.45M rows × 5 folds, best_iter at 1999–2000 (no early
  stopping triggered).
- **Optimal-R:R distribution still collapses**: mean 1.023, median 1.0,
  only 33.8% of trades have positive E[R] at any R:R ≥ 1.
- The monotonic constraint did **not** fix the "R:R=1 preference" issue —
  calibrated P(win) still prefers the smallest R:R above the min.

**Takeaway**: monotonic V2 is no better than V2 for R:R picking. The
preference for R:R=1 is a true property of calibrated P(win) under MNQ
dynamics, not a model artefact. **R:R-picking remains a dead path**; V2
stays useful as a filter.

---

## B10 — Kelly-fraction sizing from calibrated P(win)

- `f = max(0, E[R] / R)` per trade. 6 variants: full / half / quarter,
  each capped / uncapped at 5%.
- Evaluated on 10 existing combos + 10 V2-filtered surrogate top
  (v2f_surr_0..9). 19 combos finished (one error).

| Variant | wins | median lift | mean lift | median skip |
|---|---|---|---|---|
| kelly_full | 14/19 | +5.15 | +0.24 | 89% |
| **kelly_full_cap5** | 14/19 | +7.85 | **+5.31** | 89% |
| kelly_half | 14/19 | +5.15 | +0.33 | 90% |
| kelly_half_cap5 | 14/19 | +7.89 | +3.83 | 90% |
| kelly_quarter | 14/19 | +5.34 | +0.51 | 91% |
| **kelly_quarter_cap5** | 14/19 | **+8.10** | +3.12 | 91% |

**Key observations**:
1. **The 5% cap is essential.** Uncapped full-Kelly has similar median but
   mean collapses because large-n / high-Sharpe combos (v2f_surr_6: −19.85,
   v2f_surr_8: −25.57) get hurt when Kelly aggressively skips trades.
2. **kelly_full_cap5 is the best mean-lift variant (+5.31)**;
   kelly_quarter_cap5 is the best median-lift (+8.10) and most robust to
   tail losses.
3. **~90% skip rate.** Because `f ≤ 0` → skip, Kelly behaves as a
   filter+sizer combined; trade count drops from thousands to ~135–152
   per combo.
4. **5 combos are harmed by Kelly** (v10_7649, v10_9393, v10_9955,
   v2f_surr_6, v2f_surr_8). Common trait: fixed-5% already delivers high
   Sharpe driven by trade volume, not per-trade edge → Kelly's skipping
   kills that.

**Takeaway**: Kelly-cap5 (especially full-Kelly with 5% cap) is a net
positive sizing rule on the 19-combo sample. Should be **applied
conditional on combo profile** — not universal; high-freq / volume-driven
edge combos need to stay at fixed 5%.

---

## B5 feature importance + partial dependence (from retrained surrogate)

From `data/ml/lgbm_results_v2filtered/feature_importance.png` and
`partial_dependence.png` on the composite-score model.

**Top features by gain**:
1. `min_rr` (~1200)
2. `swing_lookback` (~1150)
3. `stop_fixed_pts` (~1000)
4. `z_band_k` (~850)
5. `atr_multiplier` (~450)
6. `z_window`, `max_hold_bars` (~400 each)

EMA lengths, session filters, and every `vol_regime_*` feature are near zero.

**Preferred directions** (partial dependence on composite score):

| Feature | Shape | Winning direction |
|---|---|---|
| `min_rr` | monotonic ↓ (0.53→0.38) | **lower R:R wins** |
| `swing_lookback` | monotonic ↓ | tighter stops |
| `stop_fixed_pts` | sharp ↓ | tight stops |
| `z_band_k` | ↓ | lower band → more signals |
| `atr_multiplier` | ↓ | smaller multipliers |
| `z_window` | single peak ~15–20 | mid-window |
| `max_hold_bars` | flat then ↓ past 400 | hold < 300 bars |
| `ema_slow/fast` | slight ↑ | longer EMAs |
| `zscore_confirmation` | off > on | disabled |
| `vol_regime_*` | flat | irrelevant |

**Note**: the retrained ML#1 prefers `min_rr ≈ 1.0`, consistent with B9.
The project's original `MIN_RR = 1:3` default has been relaxed in
`CLAUDE.md` (2026-04-14) — R:R is now a per-combo parameter driven by the
data.

**Takeaway**: winning stack under V2-filtered data is tight-stops +
low-z-band + R:R near 1 + V2 filter. Confirms V5 filters were drag
(already deprecated in v6+). Everything aligns with the B5 top-10 combo
signature (WR 95–99%, n=120–240).

---

## B6 — Temporal OOD test for V2

- Ran a fresh 200-combo v10 sweep on the held-out 20% test partition
  (512,499 bars, post 2024-10-22) using the new
  `param_sweep.py --eval-partition test` flag → 118,985 test-bar trades →
  2,022,745 expanded rows at 17 R:R levels.
- Applied trained V2 booster (`adaptive_rr_model.txt`, trained on 80% train
  bars) and compared AUC / log-loss / Brier / ECE vs training-split OOF
  metrics in `run_metadata.json`.

| Metric   | Train OOF | Test (B6) | Δ (test − train)  |
|----------|-----------|-----------|-------------------|
| AUC      | 0.8057    | 0.8014    | **−0.0042** (noise-level) |
| LogLoss  | 0.3402    | 0.3095    | −0.0307 (lower → better) |
| Brier    | 0.1048    | 0.0926    | −0.0122 (lower → better) |
| ECE-20   | — (not in meta) | **0.0622** | — |

- **Base-rate shift**: `mean_y` = 0.113 on test vs `mean_pred` = 0.175 —
  **predictions are systematically ~55% too high in probability space**.
  The test period (post-Oct-2024) has materially lower true win rates
  across all R:R levels than the training period.
- **LogLoss/Brier look "better"** only because test has fewer positives
  overall (easier to be accurate when base rate is lower). Do not read
  this as generalisation gain — read the ECE.
- **Per-R:R AUC climbs with R:R**: 0.609 @ R:R=1.00 → 0.697 @ R:R=5.00.
  Discrimination is weakest at low R:R (where MNQ noise dominates the
  "did it hit TP?" signal) and strengthens at high R:R. Overall AUC of
  0.80 is partly a mixture effect across the 17 R:R levels with very
  different base rates.

**Takeaway**:
- **Ranking generalises**. AUC Δ of −0.004 is within fold variance.
  Relative ordering of E[R] across trades holds on unseen time → B1's
  threshold filter and B2's percentile filter will keep working.
- **Calibration drifts**. Raw P(win) is materially overconfident on test
  (ECE 0.062, mean-bias +6.2 pp). Two downstream consequences:
  1. Absolute E[R] thresholds (B1 `thr ≥ 0.0`) will **over-pass** trades
     on future-period data. Prefer percentile filtering (B2) or
     recalibrated P(win) before live use.
  2. **Kelly sizing (B10) needs recalibration**. Kelly fraction
     `f = E[R]/R` with inflated P(win) over-sizes positions. A post-hoc
     isotonic recalibrator trained on a rolling window of recent trades
     is the cheapest mitigation.
- **B6 is a pass with caveats.** The model transfers; the probability
  scale doesn't. Gate B16 on a rolling-recalibration step, not on "apply
  V2 as-is to test bars".

---

## B7 — Walk-forward validation (expanding + rolling)

- Regenerated a 500-combo v10 sweep on the 80% training partition with the
  newly-persisted `entry_bar_idx` (1,182,624 trades; bars map to 2019-01-01 →
  2024-10-22). Years ≥5,000 trades: 2019-2024 (6 usable years).
- Trained fresh V2-clone LightGBM boosters (800 rounds, same hyperparams,
  400k base-trade cap per fold) under two split modes:
  - **Expanding** (5 folds): fold k trains on years[:k+1], tests on years[k+1].
  - **Rolling** (4 folds, window=2): fold k trains on 2-year window, tests next.

**Expanding**:

| Test year | AUC    | LogLoss | Brier  | ECE-20 | mean_y | mean_p |
|-----------|--------|---------|--------|--------|--------|--------|
| 2020      | 0.8108 | 0.3213  | 0.0986 | 0.0216 | 0.135  | 0.115  |
| 2021      | 0.8274 | 0.2915  | 0.0891 | 0.0036 | 0.123  | 0.123  |
| 2022      | 0.8141 | 0.3393  | 0.1060 | 0.0075 | 0.152  | 0.145  |
| 2023      | 0.8241 | 0.2914  | 0.0888 | 0.0034 | 0.121  | 0.124  |
| 2024*     | 0.8208 | 0.3123  | 0.0962 | 0.0029 | 0.135  | 0.135  |

**Rolling (window=2)**:

| Test year | AUC    | LogLoss | Brier  | ECE-20 | mean_y | mean_p |
|-----------|--------|---------|--------|--------|--------|--------|
| 2021      | 0.8274 | 0.2915  | 0.0891 | 0.0036 | 0.123  | 0.123  |
| 2022      | 0.8138 | 0.3396  | 0.1061 | 0.0079 | 0.152  | 0.146  |
| 2023      | 0.8236 | 0.2918  | 0.0888 | 0.0039 | 0.121  | 0.125  |
| 2024*     | 0.8191 | 0.3136  | 0.0965 | 0.0051 | 0.135  | 0.133  |

*2024 fold covers Jan-Oct only (pre-test-cutoff).

**Key observations**:
1. **AUC is remarkably stable**: 0.811–0.827 range, Δ(max-min) ≈ 0.017 —
   within expected CV fold variance. No monotonic decay with recency and no
   weak year.
2. **Rolling ≈ expanding**: 2-year rolling window produces identical AUC and
   near-identical ECE to expanding. More training data does not help past ~2
   years → signal is local-in-time, not a multi-year aggregation effect.
3. **Calibration within-training is fine**: ECE 0.003–0.022 (median 0.0037).
   mean_pred tracks mean_y to within ~1 pp on most folds. First fold
   (train=2019 alone, 1 year only, test=2020) is the worst at ECE 0.022 —
   consistent with small-sample calibration noise.
4. **Per-fold time 2.7-4.9 min** on 4 cores, 800 rounds each; full 9-fold
   run completed in ~40 min wall.

**Takeaway**:
- V2's signal is **temporally stationary inside the 80% training partition**.
  Rolling edges match expanding edges — no "recent data is more informative"
  effect, no "old data hurts" effect.
- The B6 calibration drift (ECE 0.062) is **specifically a
  post-2024-10-22 phenomenon** — i.e. the held-out 20% tail, not a gradual
  drift we could see inside training. This sharpens the B6 interpretation:
  the drift is a regime-shift event, not a continuous decay. A rolling
  isotonic recalibrator trained on the last N months of realised trades is
  still the correct mitigation before live use; walk-forward does not
  invalidate the B6 warning, but it constrains where drift lives.
- **No reason to invalidate V2** for ranking-based filter use (B2 percentile,
  B1 per-combo threshold) on in-training-period data.

---

## B8 — Feature engineering ablation

**Completed**: 2026-04-15. Testbed: 1.18M-trade v10 training-partition sweep
(418 combos). 5-fold `StratifiedGroupKFold` on `combo_id`, seed-fixed so deltas
are apples-to-apples. LightGBM 800 rounds, identical V2 hyperparameters.
Decision gate: ΔAUC ≥ +0.005.

Three feature families added base-trade-level (`shift(1)` before any rolling,
per-combo only, no future leakage):
- **A** autocorr: `prior_wr_10`, `prior_wr_50`, `prior_r_ma10`, `has_history_50`
- **B** recency: `bars_since_last_trade`, `log1p_bars_since_last_trade`
- **C** regime: `atr_regime_rank_500`, `parkinson_regime_rank_500`
  (within-combo Gaussian-CDF percentile rank over last 500 trades).

| Config | Features | OOF AUC | ΔAUC | OOF ECE | Verdict |
|--------|----------|---------|------|---------|---------|
| baseline | 20 V2 | 0.8072 | — | 0.0086 | — |
| +A | 24 | **0.8406** | **+0.0333** | 0.0036 | **ADOPT** |
| +B | 22 | 0.8069 | −0.0003 | 0.0088 | null |
| +C | 22 | 0.8089 | +0.0017 | 0.0076 | null |
| +ABC | 28 | 0.8400 | +0.0327 | 0.0039 | ≈ A |

**Key findings**:
1. **A carries the entire lift**. ABC matches A within 0.0006 AUC; B and C add
   nothing on top. Final recommendation: adopt A only, keep model simple
   (24 features vs 28).
2. **A also improves calibration**: raw ECE 0.0086 → 0.0036, plus a modest
   cold-bias shift (mean_pred 0.127 → 0.123 vs mean_y 0.126) — the model
   remains well-centred.
3. **Feature importance (config A)**: `prior_wr_50` becomes the **#2** feature
   after `candidate_rr`, above `stop_method`. `prior_r_ma10` #5 and
   `prior_wr_10` #8. This is large: `prior_wr_50` displaced 2+M gain from
   other features. Decision gate passed ~6× over.
4. **Temporal stability preserved**: A's per-fold AUCs span 0.833–0.848,
   tighter than baseline's 0.778–0.827. Fold 3 (hardest for baseline) lifts
   the most (+0.055), suggesting the autocorr signal is specifically
   compensating for regime shifts the baseline features miss.
5. **Caveat — combo-identity leakage**: `prior_wr_N` partly encodes combo
   identity (each combo has a stable baseline WR). Since CV groups on
   `combo_id`, the validation fold's combos are unseen, which bounds the
   leakage, but some of the lift may be "learn the combo's historical WR"
   rather than a pure time-local signal. Recommend SHAP follow-up before
   production rollout; phase-2 Category D (prior-trade residual) would
   disentangle identity from local dynamics.
6. **B/C null not surprising**: `bars_since_last_trade` is largely a function
   of combo trading frequency (already captured by combo_id partitioning in
   CV). Regime ranks overlap with existing `parkinson_vol_pct` / `atr_points`.

**Runtime**: ~4h 20m on sweep-runner-1 (MemoryMax=8G, CPUQuota=400%), 5 configs
× 5 folds × ~610-700s per fold. Memory peak ~6.3 GB during ABC (28-feature
LightGBM Dataset — 5G cap would have OOM'd; the second launch used 8G).

**What's not covered and why**:
- Category D (prior-trade P(win) residual) was deferred — requires 2-stage
  training and only makes sense once A is productionised.
- **No production-booster retrain** in this task. B8 is a discrimination
  ablation on the 1.18M-trade testbed; retraining the full 9.5M-row V2
  booster with Family-A features is a separate decision (adds disk/memory,
  needs new calibrator, touches downstream filter scripts). Gated by SHAP
  confirmation that the lift isn't pure combo-ID leakage.

---

## Cross-task synthesis

1. **Two tools, one stack**: V2 is useful as a *filter*, not as an R:R
   picker. B9 confirmed the R:R-picker avenue is closed; B4/B5/B10 all
   confirm the filter avenue is open.
2. **Per-combo behaviour dominates**: every task that looked at filter
   threshold (B1) or Kelly fraction (B10) finds the right answer is
   combo-specific. Low-freq / high-WR combos want strict filters and
   aggressive Kelly; high-freq / volume-driven combos want loose filters
   and fixed 5%.
3. **B5 is the new ML#1 baseline**. The unfiltered `lgbm_results/` top-20
   is obsolete — zero overlap with the V2-filtered retrain. All downstream
   work (B16 final held-out, B17 paper-trade) should use
   `lgbm_results_v2filtered/top_combos.csv` as the candidate set.
4. **Temporal generalisation update (B6 + B7 done)**: V2's **ranking** transfers
   to unseen time (B6 AUC Δ −0.004) and is stable inside training (B7 walk-
   forward AUC 0.811–0.827 across 2020-2024). **Calibration** is fine
   within-training (B7 ECE 0.003–0.022) but drifts sharply on the post
   2024-10-22 tail (B6 ECE 0.062, mean-bias +6.2 pp). The drift is a
   regime-shift event on the held-out tail, not continuous decay — a rolling
   isotonic recalibrator on recent realised trades is the correct mitigation
   before any absolute-probability use (Kelly sizing, `E[R] ≥ 0` thresholds).
   Filter-based use (B2 percentile, B1 per-combo rank) stays valid. B16
   held-out remains open.
5. **Discrimination ceiling lifted (B8)**: Family A autocorrelation features
   (`prior_wr_50` etc.) lift V2 discrimination from OOF AUC 0.8072 → 0.8406
   on the 1.18M-trade testbed — roughly 6× the decision gate. Recency (B)
   and regime (C) add nothing on top. Before retraining the production
   booster, a SHAP check on the A features is needed to confirm the lift
   isn't pure combo-ID proxying via `prior_wr_N`.

---

## Pointers

- Numerical details per task: the JSON files listed in the Scoreboard.
- Roadmap and pre-task planning: `tasks/ml1_ml2_synthesis_roadmap.md`.
- ML#1 decisions: `tasks/ml_decisions.md`.
- ML#2 decisions: `tasks/adaptive_rr_decisions.md`.
