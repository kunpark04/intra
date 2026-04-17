"""ML#1 v11 surrogate trainer — friction-aware target + quantile uncertainty.

Trains four LightGBM boosters on `combo_features_v11.parquet`:
  - net_sharpe_point : L2 (MSE)              primary regressor
  - net_sharpe_p10   : pinball alpha=0.1     10th percentile
  - net_sharpe_p50   : pinball alpha=0.5     median
  - net_sharpe_p90   : pinball alpha=0.9     90th percentile

CV strategy is random 5-fold on combos. ML#1 is combo-grain (one row per
combo, all trade-stream stats aggregated); combo rows are independent, not
a time series, so random folds are valid. Temporal robustness is verified
downstream on the held-out test bars [80:100%], not here.

UCB at inference: score = p50 + kappa * (p90 - p10) / 2, default kappa=0.
kappa=0 recovers pure exploit (p50-only ranking); raise kappa to reward
combos with wide predictive intervals (exploration).

Run remotely on sweep-runner-1 after `build_combo_features_ml1_v11.py`
produces the feature parquet.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed", file=sys.stderr)
    sys.exit(1)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "data" / "ml" / "ml1_results_v11"
FEATURES_PARQUET = OUTPUT_DIR / "combo_features_v11.parquet"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COL = "target_net_sharpe"

# Feature columns — everything on the parquet that is NOT a target or ID.
EXCLUDE_COLS = {
    "global_combo_id", "combo_id",
    "target_net_sharpe",
    # Scale-dependent leaks: n_trades and trades_per_year are fine (they're
    # predictive signal, not labels). gross_sharpe and gross_net_sharpe_gap
    # are kept deliberately — they tell the model about friction sensitivity.
}

CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]
BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
    "tight_stop_flag",
]

LGB_BASE = {
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
    "num_threads": 4,
}
N_ESTIMATORS = 500
EARLY_STOP = 50
N_FOLDS = 5
SEED = 42

BOOSTERS = [
    {"name": "net_sharpe_point", "objective": "regression", "metric": "rmse"},
    {"name": "net_sharpe_p10", "objective": "quantile", "alpha": 0.1,
     "metric": "quantile"},
    {"name": "net_sharpe_p50", "objective": "quantile", "alpha": 0.5,
     "metric": "quantile"},
    {"name": "net_sharpe_p90", "objective": "quantile", "alpha": 0.9,
     "metric": "quantile"},
]


def _prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cast booleans / categoricals for LightGBM consumption."""
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].copy()
    for c in BOOLEAN_COLS:
        if c in X.columns:
            X[c] = X[c].astype(float).fillna(0).astype(int)
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, feat_cols


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss at quantile alpha. Strictly positive for any miscalibration."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def train_booster(cfg: dict, X: pd.DataFrame, y: np.ndarray,
                  cat_features: list[str]) -> dict:
    """5-fold CV + final retrain on all data for one booster config."""
    params = {**LGB_BASE, "objective": cfg["objective"], "metric": cfg["metric"]}
    if "alpha" in cfg:
        params["alpha"] = cfg["alpha"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y))
    fold_losses: list[float] = []
    fold_best_iters: list[int] = []
    importance_acc = np.zeros(len(X.columns))

    t0 = time.time()
    for fold_idx, (tr, va) in enumerate(kf.split(X)):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr],
                          categorical_feature=cat_features, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va], label=y[va],
                          categorical_feature=cat_features,
                          reference=dtr, free_raw_data=False)
        mdl = lgb.train(
            params, dtr,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        pred = mdl.predict(X.iloc[va])
        oof_preds[va] = pred
        fold_best_iters.append(int(mdl.num_trees()))
        importance_acc += mdl.feature_importance(importance_type="gain")

        if cfg["objective"] == "regression":
            loss = float(np.sqrt(mean_squared_error(y[va], pred)))
            label = "RMSE"
        else:
            loss = _pinball_loss(y[va], pred, cfg["alpha"])
            label = f"pinball@{cfg['alpha']}"
        fold_losses.append(loss)
        print(f"    fold {fold_idx+1}/{N_FOLDS}: {label}={loss:.4f}  "
              f"trees={mdl.num_trees()}", flush=True)

    summary: dict = {
        "name": cfg["name"],
        "objective": cfg["objective"],
        "alpha": cfg.get("alpha"),
        "fold_losses": fold_losses,
        "mean_loss": float(np.mean(fold_losses)),
        "fold_best_iters": fold_best_iters,
        "mean_best_iters": int(np.mean(fold_best_iters)),
    }
    if cfg["objective"] == "regression":
        summary["oof_r2"] = float(r2_score(y, oof_preds))
        summary["oof_rmse"] = float(np.sqrt(mean_squared_error(y, oof_preds)))
    else:
        summary["oof_pinball"] = _pinball_loss(y, oof_preds, cfg["alpha"])

    avg_importance = importance_acc / N_FOLDS
    summary["feature_importance"] = sorted(
        [{"feature": c, "gain": float(g)}
         for c, g in zip(X.columns, avg_importance)],
        key=lambda d: d["gain"], reverse=True,
    )

    # Final retrain on all data using mean best_iters from CV.
    dfull = lgb.Dataset(X, label=y, categorical_feature=cat_features,
                        free_raw_data=False)
    final = lgb.train(params, dfull,
                      num_boost_round=summary["mean_best_iters"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{cfg['name']}.txt"
    final.save_model(str(model_path))
    summary["model_path"] = str(model_path.relative_to(REPO)).replace("\\", "/")

    print(f"  [{cfg['name']}] {time.time()-t0:.1f}s  "
          f"mean_loss={summary['mean_loss']:.4f}  "
          f"mean_trees={summary['mean_best_iters']}", flush=True)
    return summary


def plot_feature_importance(results: list[dict], out_path: Path) -> None:
    """Grid of top-15 features for each of the 4 boosters."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, r in zip(axes.flat, results):
        fi = r["feature_importance"][:15]
        names = [d["feature"] for d in fi][::-1]
        gains = [d["gain"] for d in fi][::-1]
        ax.barh(names, gains, color="#1f77b4")
        ax.set_title(f"{r['name']} (top-15 by gain)")
        ax.set_xlabel("gain")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not FEATURES_PARQUET.exists():
        print(f"ERROR: {FEATURES_PARQUET} missing — "
              f"run build_combo_features_ml1_v11.py first", file=sys.stderr)
        sys.exit(1)

    print(f"[v11-train] loading {FEATURES_PARQUET.relative_to(REPO)}", flush=True)
    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"[v11-train] {len(df):,} combos, {len(df.columns)} cols", flush=True)

    X, feat_cols = _prepare_frame(df)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    cat_features = [c for c in feat_cols if c in CATEGORICAL_COLS]
    print(f"[v11-train] {len(feat_cols)} features, "
          f"{len(cat_features)} categorical: {cat_features}", flush=True)
    print(f"[v11-train] target p5={np.quantile(y,0.05):.3f} "
          f"p50={np.median(y):.3f} p95={np.quantile(y,0.95):.3f}", flush=True)

    results: list[dict] = []
    for cfg in BOOSTERS:
        print(f"\n[v11-train] {cfg['name']} ({cfg['objective']}"
              + (f", alpha={cfg.get('alpha')}" if 'alpha' in cfg else "")
              + ")", flush=True)
        results.append(train_booster(cfg, X, y, cat_features))

    cv_path = OUTPUT_DIR / "cv_results.json"
    cv_path.write_text(json.dumps({
        "n_combos": int(len(df)),
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "categorical_features": cat_features,
        "target": TARGET_COL,
        "lgb_params": LGB_BASE,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "boosters": results,
    }, indent=2, default=str))
    print(f"[v11-train] wrote {cv_path.relative_to(REPO)}", flush=True)

    fig_path = OUTPUT_DIR / "feature_importance.png"
    plot_feature_importance(results, fig_path)
    print(f"[v11-train] wrote {fig_path.relative_to(REPO)}", flush=True)

    for r in results:
        if "oof_r2" in r:
            print(f"  {r['name']}: OOF R²={r['oof_r2']:.4f}  "
                  f"RMSE={r['oof_rmse']:.4f}")
        else:
            print(f"  {r['name']}: OOF pinball@{r['alpha']}={r['oof_pinball']:.4f}")


if __name__ == "__main__":
    main()
