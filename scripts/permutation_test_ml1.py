"""B3: Permutation test for ML#1 surrogate models.

Shuffle target values before training to confirm CV R² collapses to ~0
(i.e. the real model is learning signal, not structure from feature leakage).
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("mlo", REPO / "scripts/ml1_surrogate.py")
mlo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlo)

OUT = REPO / "data/ml/ml1_results/permutation_test.json"


def main():
    cached = mlo.OUTPUT_DIR / "combo_features.parquet"
    if not cached.exists():
        print(f"Missing {cached}; run ml1_surrogate first.")
        sys.exit(1)
    df = pd.read_parquet(cached)
    for col in mlo.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    weights = {"sharpe_ratio": 0.4, "total_return_pct": 0.3,
               "max_drawdown_pct": 0.2, "win_rate": 0.1}
    df["composite_score"] = mlo.compute_composite_score(df, weights)

    X, feature_cols = mlo.prepare_features(df)
    print(f"Data: {len(df):,} combos, {len(feature_cols)} features")

    rng = np.random.default_rng(42)
    df_shuf = df.copy()
    for t in mlo.TARGETS:
        df_shuf[t] = rng.permutation(df_shuf[t].values)

    print("\n=== REAL targets ===")
    real = mlo.train_surrogate_models(df, feature_cols, mlo.TARGETS, 5, 42)
    print("\n=== SHUFFLED targets ===")
    shuf = mlo.train_surrogate_models(df_shuf, feature_cols, mlo.TARGETS, 5, 42)

    summary = {}
    for t in mlo.TARGETS:
        r_mean = float(np.mean(real[t]["cv_r2"]))
        s_mean = float(np.mean(shuf[t]["cv_r2"]))
        summary[t] = {"real_cv_r2": r_mean, "shuffled_cv_r2": s_mean,
                      "delta": r_mean - s_mean}
        print(f"{t}: real R²={r_mean:.4f}  shuffled R²={s_mean:.4f}  "
              f"Δ={r_mean - s_mean:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
