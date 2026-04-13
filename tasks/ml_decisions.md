# ML Pipeline Decisions

Every key decision in `scripts/ml_optimizer.py` with reasoning.

---

## D1: Model Choice — LightGBM Regression

**Decision**: Use LightGBM gradient-boosted trees as the surrogate model.

**Reasoning**: The input data is tabular (parameter settings → performance metrics).
Gradient-boosted trees consistently outperform neural networks and linear models on
structured/tabular data. LightGBM specifically is faster than XGBoost, handles
categorical features natively (no one-hot encoding needed), and handles NaN/missing
values without imputation. With ~33K rows, a neural network would be overkill and
prone to overfitting.

**Alternatives considered**:
- XGBoost: comparable quality but slower; LightGBM's native categorical support is cleaner
- Random Forest: decent but GBMs consistently outperform on structured data
- Neural networks: overkill for 33K tabular rows, harder to interpret
- Linear regression: cannot capture parameter interactions (e.g., "ATR stop + high min_rr" behaves differently than either alone)

---

## D2: Unit of Analysis — Combo Level, Not Trade Level

**Decision**: Aggregate 80M trade rows into ~33K combo-level rows. The model predicts
combo performance from parameter settings.

**Reasoning**: The goal is "which parameter settings produce the best outcomes?" not
"will this individual trade win?" A combo-level model directly answers the question.
Trade-level classification would require a separate filtering step and doesn't optimize
for portfolio-level metrics like Sharpe or max drawdown.

---

## D3: Using All Sweep Versions (v2–v10)

**Decision**: Combine all 9 sweep versions into one training set (~33K combos, 80M trades).

**Reasoning**: More data = better generalization. Each version explored different
parameter regions (v2 was broad, v6 was ultra-tight, v8 was diversity-focused, etc.).
Combining them gives the model visibility into the full parameter landscape. The model
can learn that certain regions are consistently bad regardless of how they were sampled.

**Handling schema differences**: v2 has 31 columns, v3–v4 have 39, v5–v10 have 45.
Missing columns in older versions are filled with default/disabled values since those
combos actually used the default z-score formulation and had no V5 filters active.

---

## D4: Sharpe Ratio — Un-Annualized Trade-Level

**Decision**: Compute Sharpe as `mean(net_pnl_dollars) / std(net_pnl_dollars)` without
annualization.

**Reasoning**: The parquet data lacks entry timestamps (only `time_of_day_hhmm` and
`day_of_week`), so we cannot group trades into daily buckets for a proper daily Sharpe.
However, since ALL combos run on the same historical price data over the same time
period, the annualization factor would be a monotonic transformation that doesn't change
relative rankings. For the purpose of ranking combos against each other, un-annualized
Sharpe is mathematically equivalent.

---

## D5: Composite Score — Rank-Percentile Weighted Sum

**Decision**: Normalize each metric using rank percentile [0, 1], then combine with
a weighted sum.

**Weights**:
- Sharpe ratio: 0.25 (risk-adjusted return — best single predictor of out-of-sample performance)
- Total return: 0.25 (absolute performance — the primary goal)
- Max drawdown: 0.20 (inverted — lower is better; critical for live trading psychology)
- Win rate: 0.15 (consistency and execution confidence)
- Trade count: 0.15 (more trades = more statistical confidence + more opportunities)

**Why rank percentile over z-score**: Trading metrics have extreme outliers (some combos
return +6000%, others -5600%). Z-score normalization would be dominated by these extremes.
Rank percentile is robust — it maps everything to [0, 1] uniformly regardless of outlier
magnitude.

**Why these specific weights**: Sharpe and return are weighted equally (0.25 each) because
the user wants both high returns AND risk-adjusted quality. Drawdown gets 0.20 because
a 50% drawdown is psychologically devastating in live trading. Win rate gets 0.15 because
the user explicitly asked for higher win rates. Trade count gets 0.15 because the user
explicitly stated "higher trades taken is good" and more trades also means more statistical
reliability.

---

## D6: Cross-Validation — 5-Fold KFold

**Decision**: Standard 5-fold KFold cross-validation, no stratification.

**Reasoning**: The 33K combos are independent random parameter samples that all ran on
the same historical data. There is no temporal ordering between combos (combo 500 doesn't
"come before" combo 5000 in any meaningful sense). Therefore standard random k-fold is
appropriate — there's no information leakage between folds.

Note: the risk of overfitting is NOT between combos (which is what CV protects against).
The real risk is overfitting to the historical price period. This is addressed by the
held-out 20% test bars (a separate backtest step, not part of this ML pipeline).

---

## D7: Multi-Target — Separate Models Per Metric

**Decision**: Train 5 independent LightGBM models: one for composite_score, and one each
for sharpe_ratio, total_return_pct, max_drawdown_pct, and win_rate.

**Reasoning**: Each metric captures a different aspect of strategy quality. Separate models
allow inspecting which parameters drive each metric independently. For example, you might
find that `min_rr` is the top driver of Sharpe but irrelevant for win rate. The composite
model is the primary one for ranking; the individual models are for interpretability.

**Alternative considered**: Multi-output regression (one model, multiple heads). LightGBM
doesn't support this natively, and wrapping it adds complexity for marginal benefit at
this data size.

---

## D8: Minimum Trade Filter — 30 Trades

**Decision**: Exclude combos with fewer than 30 trades from the training data.

**Reasoning**: Performance metrics computed from <30 trades are statistically unreliable.
A combo with 5 trades and 100% win rate tells us almost nothing — it could be luck.
30 is a standard statistical threshold for the central limit theorem to apply (means
and standard deviations become approximately normal). This also prevents the model from
learning spurious patterns from low-sample combos.

---

## D9: Missing Categorical Defaults

**Decision**: Fill missing z-score formulation columns (`z_input`, `z_anchor`, `z_denom`,
`z_type`) with their actual default values: `close`, `rolling_mean`, `rolling_std`,
`parametric`.

**Reasoning**: Combos from v2 and early v10 (IDs 0–2999) used the default z-score
formulation — they weren't "missing" this information, they just didn't record it.
Leaving them as NaN would create an artificial "unknown" category that doesn't represent
reality. Filling with the actual defaults is more truthful.

Similarly, V5 filter fields (`session_filter_mode`, `tod_exit_hour`, `vol_regime_*`,
`volume_entry_threshold`) are filled with their "disabled" values (0, 0.0, 1.0) for
v2–v4, since those versions had no filtering active.

---

## D10: Parameter Extraction — Three Methods

**Decision**: Use three complementary methods to extract optimal parameters:

1. **Top-N from OOF predictions**: Rank all ~33K combos by their out-of-fold predicted
   composite score. Report the top 20. This is the most directly actionable output.

2. **Partial Dependence Plots (PDP)**: Show how each parameter affects the predicted
   composite score in isolation. This gives interpretable insight into parameter sensitivity.

3. **Surrogate search**: Generate 50,000 random parameter combos, score them with the
   trained model, and report the top candidates. This explores parameter regions that the
   original sweep may have missed.

**Reasoning**: Top-N gives immediate answers. PDP gives understanding. Surrogate search
gives discovery of potentially better combos outside the training data. Together they
provide a complete picture: what works, why it works, and what else might work.

---

## D11: LightGBM Hyperparameters

**Decision**: Use conservative, regularized defaults.

```python
num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
bagging_fraction=0.8, min_child_samples=20,
reg_alpha=0.1, reg_lambda=1.0, n_estimators=500, early_stopping=50
```

**Reasoning**: With ~33K rows, the risk is overfitting rather than underfitting.
- `num_leaves=31`: default; 33K rows doesn't justify more complex trees
- `learning_rate=0.05`: slower learning = more robust ensemble
- `feature_fraction=0.8` + `bagging_fraction=0.8`: random subsampling per tree for diversity
- `min_child_samples=20`: prevents leaves from fitting tiny groups
- `reg_alpha=0.1` + `reg_lambda=1.0`: L1 + L2 regularization penalizes complexity
- `early_stopping=50`: stops training when validation error plateaus, prevents overfitting
- `n_estimators=500`: upper bound; early stopping will typically cut this shorter

No hyperparameter tuning (Optuna/GridSearch) is done in this first iteration to keep
things simple and avoid meta-overfitting. If R² is low, tuning can be added later.

---

## D12: Output Structure

**Decision**: All outputs go to `data/ml/lgbm_results/`.

**Reasoning**: Keeps ML artifacts separate from the raw sweep data. The directory contains
both human-readable files (CSV, PNG, JSON) and machine-readable files (parquet, model .txt).
`run_metadata.json` captures all settings for reproducibility.

---

## D13: Global Combo ID

**Decision**: Create a unique combo ID as `f"v{version}_{combo_id}"` (e.g., `v10_4523`).

**Reasoning**: `combo_id` starts at 0 in every sweep version. Without a global ID,
combo 0 from v2 would collide with combo 0 from v10. The version prefix makes IDs
unique and traceable back to their source dataset.
