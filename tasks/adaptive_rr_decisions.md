# Adaptive R:R Model Decisions

Key design choices for `scripts/adaptive_rr_model.py`.

---

## D1: Unit of Analysis — Trade-Level, Not Combo-Level

**Decision**: Each row is one trade at one candidate R:R level, predicting P(win).

**Reasoning**: Unlike ml_optimizer.py which answers "which combo settings are best?",
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
