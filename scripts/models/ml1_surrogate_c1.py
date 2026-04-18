"""ML#1 retrain with C1 target: fixed_sharpe * log1p(n_trades).

Replaces the rank-weighted 5-metric composite (which was distorted by
5% compounding in its dollar-sharpe and total-return-pct components)
with a sizing-invariant single-formula target validated by the
composite-score bakeoff (see data/ml/composite_bakeoff_v1.json):

    composite_score := fixed_sharpe * log1p(n_trades)

where fixed_sharpe = mean(r_multiple) / std(r_multiple).

Inputs
------
    data/ml/ml1_results/combo_features.parquet   (reuse existing cache)
    data/ml/full_combo_r_stats.parquet           (from build_full_combo_r_stats.py)

Outputs → data/ml/ml1_results_c1/
    combo_features_c1.parquet                    (cache + fixed_sharpe + new target)
    models/composite_score.txt                    (LightGBM booster)
    cv_results.json                               (5-fold OOF R²/RMSE)
    top_combos.csv                                (top-N by OOF predicted C1)
    feature_importance.png
    run_metadata.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM not installed"); sys.exit(1)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
COMBO_CACHE = REPO / "data" / "ml" / "ml1_results" / "combo_features.parquet"
R_STATS = REPO / "data" / "ml" / "full_combo_r_stats.parquet"
OUTPUT_DIR = REPO / "data" / "ml" / "ml1_results_c1"

PARAM_COLS = [
    "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop",
    "max_hold_bars", "zscore_confirmation",
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]
CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]
BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 500,
    "num_threads": 3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-top", type=int, default=30)
    p.add_argument("--min-trades", type=int, default=30,
                   help="Floor on n_trades_full for C1 eligibility")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_combo_features() -> pd.DataFrame:
    print(f"[c1] loading combo cache: {COMBO_CACHE.relative_to(REPO)}")
    df = pd.read_parquet(COMBO_CACHE)
    print(f"     {len(df):,} combos, {len(df.columns)} cols")
    return df


def load_r_stats() -> pd.DataFrame:
    print(f"[c1] loading r-stats: {R_STATS.relative_to(REPO)}")
    r = pd.read_parquet(R_STATS)
    print(f"     {len(r):,} combos")
    return r


def merge_and_compute_c1(df: pd.DataFrame, r: pd.DataFrame,
                         min_trades: int) -> pd.DataFrame:
    merged = df.merge(r, on="global_combo_id", how="inner")
    print(f"[c1] merged: {len(merged):,} combos "
          f"(dropped {len(df) - len(merged):,} without r-stats)")
    # Consistency check between cached n_trades and full n_trades_full
    mism = (merged["n_trades"] != merged["n_trades_full"]).sum()
    if mism:
        print(f"     WARN: {mism:,} combos have n_trades != n_trades_full; "
              f"using n_trades_full.")
    # Apply eligibility floor
    before = len(merged)
    merged = merged[merged["n_trades_full"] >= min_trades].copy()
    print(f"[c1] after min_trades >= {min_trades}: {len(merged):,} "
          f"(dropped {before - len(merged):,})")
    # C1 target
    merged["fixed_sharpe"] = merged["fixed_sharpe_full"]
    merged["composite_score"] = (
        merged["fixed_sharpe"] * np.log1p(merged["n_trades_full"])
    )
    c1 = merged["composite_score"]
    print(f"[c1] target range: [{c1.min():.4f}, {c1.max():.4f}]  "
          f"mean={c1.mean():.4f}  std={c1.std():.4f}")
    return merged


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0).astype(int)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    feat_cols = [c for c in PARAM_COLS if c in df.columns]
    return df[feat_cols].copy(), feat_cols


def train(df: pd.DataFrame, feat_cols: list[str], n_folds: int,
          seed: int) -> dict:
    X = df[feat_cols]
    y = df["composite_score"].values
    cat_feats = [c for c in feat_cols if c in CATEGORICAL_COLS]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_r2, fold_rmse, fold_train_r2 = [], [], []
    importance_acc = np.zeros(len(feat_cols))

    for fi, (tr, va) in enumerate(kf.split(X)):
        Xt, Xv = X.iloc[tr], X.iloc[va]
        yt, yv = y[tr], y[va]
        dtrain = lgb.Dataset(Xt, label=yt, categorical_feature=cat_feats,
                             free_raw_data=False)
        dval = lgb.Dataset(Xv, label=yv, categorical_feature=cat_feats,
                           reference=dtrain, free_raw_data=False)
        m = lgb.train(LGB_PARAMS, dtrain,
                      num_boost_round=LGB_PARAMS["n_estimators"],
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pv = m.predict(Xv); pt = m.predict(Xt)
        oof[va] = pv
        fold_r2.append(r2_score(yv, pv))
        fold_rmse.append(np.sqrt(mean_squared_error(yv, pv)))
        fold_train_r2.append(r2_score(yt, pt))
        importance_acc += m.feature_importance(importance_type="gain")
        print(f"  fold {fi+1}/{n_folds}: val_R²={fold_r2[-1]:.4f}  "
              f"train_R²={fold_train_r2[-1]:.4f}  "
              f"rmse={fold_rmse[-1]:.4f}  trees={m.num_trees()}")

    overall_r2 = r2_score(y, oof)
    overall_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    gap = float(np.mean(fold_train_r2) - np.mean(fold_r2))
    print(f"  overall OOF R²={overall_r2:.4f}  rmse={overall_rmse:.4f}  "
          f"gap={gap:.4f}")

    # Refit final model on full data
    dfull = lgb.Dataset(X, label=y, categorical_feature=cat_feats,
                        free_raw_data=False)
    best_trees = int(np.median([len(l) for l in [fold_r2]]) or 500)  # fallback
    # Re-train with same num_boost_round as the last fold's best
    final = lgb.train(LGB_PARAMS, dfull,
                      num_boost_round=m.num_trees())

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": importance_acc / n_folds,
    }).sort_values("importance", ascending=False)

    return {
        "model": final,
        "oof": oof,
        "overall_r2": overall_r2,
        "overall_rmse": overall_rmse,
        "overfit_gap": gap,
        "fold_r2": fold_r2,
        "fold_rmse": fold_rmse,
        "fold_train_r2": fold_train_r2,
        "importance": imp_df,
    }


def save_outputs(df: pd.DataFrame, feat_cols: list[str], res: dict,
                 n_top: int, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Save combo features w/ new target
    df_save = df.copy()
    for c in CATEGORICAL_COLS:
        if c in df_save.columns:
            df_save[c] = df_save[c].astype(str)
    df_save.to_parquet(out_dir / "combo_features_c1.parquet", index=False)

    # Top combos ranked by OOF predicted C1
    df_ranked = df_save.copy()
    df_ranked["predicted_composite"] = res["oof"]
    top = df_ranked.sort_values("predicted_composite", ascending=False).head(n_top)
    summary_cols = [
        "global_combo_id", "predicted_composite", "composite_score",
        "fixed_sharpe", "n_trades_full", "mean_r_full", "std_r_full",
        "win_rate", "avg_r_multiple",
    ]
    summary_cols = [c for c in summary_cols if c in top.columns]
    top[summary_cols].to_csv(out_dir / "top_combos.csv", index=False)

    # CV JSON
    cv = {
        "overall_r2": res["overall_r2"],
        "overall_rmse": res["overall_rmse"],
        "overfit_gap": res["overfit_gap"],
        "fold_r2": res["fold_r2"],
        "fold_rmse": res["fold_rmse"],
        "fold_train_r2": res["fold_train_r2"],
        "top_10_features": res["importance"].head(10).to_dict("records"),
    }
    (out_dir / "cv_results.json").write_text(json.dumps(cv, indent=2, default=str))

    # Importance plot
    imp = res["importance"].head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(imp)), imp["importance"].values, color="#4CAF50")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title("ML#1-C1 Feature Importance (target = fixed_sharpe × log1p n)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

    # Save model
    res["model"].save_model(str(models_dir / "composite_score.txt"))

    # Metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_formula": "fixed_sharpe * log1p(n_trades_full)",
        "min_trades_floor": args.min_trades,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_combos": len(df),
        "lgb_params": LGB_PARAMS,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[c1] outputs written to {out_dir.relative_to(REPO)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    print("=" * 70)
    print("ML#1 C1 retrain — target = fixed_sharpe * log1p(n_trades)")
    print("=" * 70)
    t0 = time.time()
    df = load_combo_features()
    r = load_r_stats()
    merged = merge_and_compute_c1(df, r, args.min_trades)
    X, feat_cols = prepare_features(merged)
    print(f"[c1] feature matrix: {X.shape}")
    res = train(merged, feat_cols, args.n_folds, args.seed)
    save_outputs(merged, feat_cols, res, args.n_top, out_dir, args)
    print(f"\nDONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
