# Adaptive R:R Model Decisions

Key design choices for `scripts/adaptive_rr_model_v1.py`.

---

## D1: Unit of Analysis — Trade-Level, Not Combo-Level

**Decision**: Each row is one trade at one candidate R:R level, predicting P(win).

**Reasoning**: Unlike ml1_surrogate.py which answers "which combo settings are best?",
this model answers "for THIS specific trade setup, what R:R maximizes expected value?"
Trade-level granularity is necessary because different market conditions within the
same combo warrant different R:R selections.

---

## D2: Synthetic Labels via MFE

**Decision**: `would_win = (mfe_points >= candidate_rr * stop_distance_pts)`

**Reasoning**: MFE (max favorable excursion) tells us the best price the trade ever
reached. If MFE exceeded the hypothetical take-profit level, the trade would have hit
that TP in reality. This is a gold-standard label — it uses actual path data, not just
entry/exit outcome.

**Limitation**: This assumes the stop loss stays the same regardless of R:R. In
reality, changing the TP doesn't change the SL, so the stop_distance_pts is correct.
However, hold_bars might differ — a 5:1 TP takes longer to hit than 2:1, and the trade
might hit SL first. We mitigate this by using `mfe_points` which captures the actual
peak regardless of when it occurred during the trade's life.

---

## D3: 17 R:R Levels (1.0 to 5.0, step 0.25)

**Decision**: 17 discrete candidate R:R values from 1.0 to 5.0.

**Reasoning**: Finer than 0.5 steps for smooth EV curves. 1.0 lower bound because
R:R < 1 is never acceptable (you risk more than you gain). 5.0 upper bound because
MFE analysis shows very few trades exceed 5× risk distance.

---

## D4: Row Expansion and Subsampling

**Decision**: Expand each base trade to 17 rows, subsample to 10M total if needed.

**Reasoning**: With ~80M base trades × 17 = 1.36B rows, this exceeds memory. 10M rows
(~588K base trades × 17) provides sufficient statistical power while fitting in RAM.
Subsampling is random with seed=42 for reproducibility.

---

## D5: LightGBM Binary Classification

**Decision**: Binary classification with log-loss, not regression.

**Reasoning**: The target (would_win) is binary. Log-loss produces well-calibrated
probability estimates, which we need for the E[R] calculation. A regression model
would need post-hoc calibration.

---

## D6: Expected Value Optimization

**Decision**: `E[R] = P(win) * R - (1 - P(win))` where R is the candidate R:R.

**Reasoning**: This is the Kelly-style expected return per unit of risk. P(win)*R is
the expected gain when you win, and (1-P(win))*1 is the expected loss (always 1 unit
of risk). The R:R that maximizes E[R] is the optimal risk-adjusted choice.

---

## D7: Stratified CV

**Decision**: 5-fold stratified by `would_win` label.

**Reasoning**: At high R:R levels (4:1, 5:1), win rates are very low (~10-15%).
Stratification ensures each fold has representative class balance. This matters
because `is_unbalance=True` in LightGBM adjusts for class weights.

---

## D8: Feature Set — 20 Features

**Decision**: 15 entry features + stop_method + exit_on_opposite + candidate_rr +
2 derived (abs_zscore, rr_x_atr).

**Reasoning**: Entry features capture market state. candidate_rr is the key variable
we're optimizing. The interaction rr_x_atr captures whether a higher R:R is achievable
given current volatility. abs_zscore_entry captures signal strength regardless of direction.

Excluded: combo-level params like z_band_k, z_window (these determine which trades
exist, not which R:R is best for a given trade). The model should generalize across
combo settings.

---

## D9: Calibration Monitoring

**Decision**: Plot reliability diagram + Brier score.

**Reasoning**: The E[R] calculation depends on P(win) being accurate, not just
discriminative (AUC). A model with AUC=0.8 but poor calibration would suggest
wrong R:R levels. The calibration curve reveals if P(win)=0.3 actually wins 30%
of the time.

---

## D10: V2-as-R:R-picker is a dead path; V2-as-filter is the live path (2026-04-14)

**Decision**: Do not use V2 to pick per-trade R:R. Use V2 to filter trades
at the combo's own `min_rr`.

**Reasoning**: The calibrated V2 model consistently prefers the smallest
available R:R across its 17-step grid. B9 tested this with a monotonic
constraint on `candidate_rr` (V2 retrained under `monotone_constraints`):
overall CV metrics are identical to V2 (AUC 0.806, log-loss 0.340,
Brier 0.105), and the optimal-R:R distribution still collapses
(mean 1.023, median 1.0, only 33.8% of trades have positive E[R] at any
R:R ≥ 1). The R:R=1 preference is a true property of calibrated P(win)
under MNQ 1-minute economics, not a model artefact.

**What works instead**: using P(win) at the combo's `min_rr` to compute
E[R] and filter trades below a per-combo threshold. B1 finds the
threshold is bimodal (heavy mass at ±0.5 guardrails and near zero).
B2 finds top-25% percentile filtering lifts median Sharpe by +3.87.
B5 retrain confirms 95% of 27,326 combos benefit from this filter.

---

## D11: Kelly-fraction sizing — cap at 5%, apply conditionally (2026-04-14)

**Decision**: Default to `f = max(0, E[R] / R)` capped at 5%. Prefer
full-Kelly-cap5 (best mean lift) or quarter-Kelly-cap5 (most robust) over
uncapped Kelly. Do not apply universally — gate on combo profile.

**Reasoning**: B10 evaluated 6 Kelly variants on 19 combos. Capped variants
win: `kelly_full_cap5` mean Sharpe lift +5.31, `kelly_quarter_cap5` median
lift +8.10. All three cap5 variants improve 14/19 combos with ~90% skip
rate — Kelly acts as filter+sizer together because `f ≤ 0` → skip. Five
combos are *harmed* by Kelly: ones whose edge comes from trade volume +
high fixed-5% compounding rather than per-trade edge (e.g. v2f_surr_6,
v2f_surr_8 where Sharpe dropped ~20–25 points). Conclusion: Kelly-cap5 is
a net-positive sizing rule on average but needs combo-level gating before
live use.

---

## D12: MIN_RR constraint relaxed (2026-04-14)

**Decision**: `MIN_RR ≥ 3` is no longer a project floor. Combos run at the
R:R their data supports, which is typically near 1:1 under V2 filter.

**Reasoning**: Same as `tasks/ml_decisions.md` §D16. `CLAUDE.md` updated
to reflect.

---

## D13: V2 OOD calibration drift — use rank/percentile filters, recalibrate before Kelly (2026-04-14)

**Decision**: Under current V2 as trained on the 80% training partition:

- **Filter use (ranking)** — safe to deploy unchanged. Use B2 percentile
  or B1 per-combo thresholds rather than a global absolute `E[R] ≥ 0`.
- **Kelly sizing (B10)** — must be gated on a rolling isotonic
  recalibrator trained on a recent window of realised trades before live
  use. Raw V2 output is materially overconfident on unseen time.

**Reasoning**: B6 tested V2 on a 200-combo v10 sweep run on the held-out
20% test bars (post-2024-10-22, 118,985 base trades, 2M expanded rows).
Ranking (AUC 0.8057 train → 0.8014 test, Δ −0.004) transferred cleanly —
fold variance alone is ~0.005. Calibration did not: ECE 0.062, mean
predicted 0.175 vs mean observed 0.113. The test period has lower true
win rates across all R:R levels than the training period; raw P(win) is
about +6 pp inflated.

Consequences:
- B1's absolute-E[R] threshold (`thr = 0.0`) would over-pass trades on
  future bars. B2's percentile filter is drift-robust.
- B10 Kelly `f = E[R] / R` with inflated P(win) systematically oversizes.
- A post-hoc recalibrator rebuilt on a rolling window of recent trades
  (e.g. last 6 months) is the cheapest fix. Do not retrain V2 from
  scratch for this alone.

See `tasks/part_b_findings.md` §B6 and
`data/ml/adaptive_rr_v2/heldout_time_eval_v2.json`.
