"""Audit: does frequency-bucket filtering throw out high-composite combos?

Loads ML#1 combo_features + composite booster, predicts composite for every
combo, then compares:
  • Overall top-20 combos (any n_trades)
  • Top-5 per frequency bucket under several threshold schemes
  • Distribution of composite vs n_trades

Tells us whether the bucket boundaries are sacrificing quality.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ML1_DIR = Path("/root/intra/data/ml/lgbm_results_v2filtered")
FEATURES = ML1_DIR / "combo_features.parquet"
MODEL = ML1_DIR / "models" / "composite_score.txt"


def main() -> None:
    """Audit frequency-bucket filtering vs unbucketed ranking."""
    df = pd.read_parquet(FEATURES)
    booster = lgb.Booster(model_file=str(MODEL))
    feat_cols = booster.feature_name()
    X = df[feat_cols].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")
    df["pred"] = booster.predict(X)
    df["realised"] = df["composite_score"]

    print(f"N combos: {len(df):,}")
    print(f"composite_score range: {df['realised'].min():.3f} - {df['realised'].max():.3f}")
    print(f"predicted_composite range: {df['pred'].min():.3f} - {df['pred'].max():.3f}")
    print()

    # Correlation between n_trades and predicted/realised composite
    print("Correlation:")
    print(f"  pred vs n_trades:     {df['pred'].corr(df['n_trades']):.3f}")
    print(f"  realised vs n_trades: {df['realised'].corr(df['n_trades']):.3f}")
    print(f"  pred vs realised:     {df['pred'].corr(df['realised']):.3f}")
    print()

    # Overall top-20 by predicted
    print("=" * 70)
    print("OVERALL TOP-20 by predicted_composite (no freq filter):")
    print("=" * 70)
    top20 = df.nlargest(20, "pred")[["global_combo_id", "n_trades",
                                      "pred", "realised", "sharpe_ratio",
                                      "win_rate", "total_return_pct"]]
    print(top20.to_string(index=False))
    print()
    print(f"  n_trades distribution of overall top-20:")
    print(f"    min={top20['n_trades'].min()}  max={top20['n_trades'].max()}  "
          f"median={top20['n_trades'].median():.0f}")
    print()

    # Where would percentile thresholds put us?
    p50 = int(df["n_trades"].quantile(0.50))
    p95 = int(df["n_trades"].quantile(0.95))
    high_min = max(p95, 10 * p50)
    low_max = p50
    print(f"Percentile thresholds: low_max=p50={low_max}  "
          f"high_min=max(p95,10*p50)={high_min}")
    print()

    # Top-5 in each bucket
    hi = df[df["n_trades"] >= high_min].nlargest(5, "pred")
    lo = df[df["n_trades"] <= low_max].nlargest(5, "pred")
    mid = df[(df["n_trades"] > low_max) & (df["n_trades"] < high_min)].nlargest(5, "pred")
    print(f"  HIGH-FREQ pool (n_trades >= {high_min}): {(df['n_trades'] >= high_min).sum():,} combos")
    print(f"  LOW-FREQ  pool (n_trades <= {low_max}): {(df['n_trades'] <= low_max).sum():,} combos")
    print(f"  MIDDLE    pool ({low_max} < n_trades < {high_min}): "
          f"{((df['n_trades'] > low_max) & (df['n_trades'] < high_min)).sum():,} combos (THROWN OUT)")
    print()

    print("Top-5 HIGH:")
    print(hi[["global_combo_id", "n_trades", "pred", "realised", "sharpe_ratio"]].to_string(index=False))
    print()
    print("Top-5 LOW:")
    print(lo[["global_combo_id", "n_trades", "pred", "realised", "sharpe_ratio"]].to_string(index=False))
    print()
    print("Top-5 MIDDLE (excluded by current bucketing):")
    print(mid[["global_combo_id", "n_trades", "pred", "realised", "sharpe_ratio"]].to_string(index=False))
    print()

    # What's the predicted composite of the BEST combo in each band?
    print("Best predicted composite per frequency band:")
    bands = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800),
             (800, 1600), (1600, 3200), (3200, 10_000_000)]
    for lo_b, hi_b in bands:
        sub = df[(df["n_trades"] >= lo_b) & (df["n_trades"] < hi_b)]
        if len(sub) == 0:
            continue
        best = sub.nlargest(1, "pred").iloc[0]
        print(f"  n_trades [{lo_b:>5},{hi_b:>9}): n={len(sub):>6,}  "
              f"best_pred={best['pred']:.3f}  realised={best['realised']:.3f}  "
              f"sharpe={best['sharpe_ratio']:.2f}  "
              f"id={best['global_combo_id']}")


if __name__ == "__main__":
    main()
