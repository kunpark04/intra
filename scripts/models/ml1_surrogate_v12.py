"""ML#1 v12 surrogate trainer — parameter-only forecasting with fold-wise KNN.

Differences from v11:
  1. Target: target_robust_sharpe (median - 0.5*std across ordinal walk-forward
     windows within the train partition). Best proxy for OOS Sharpe on unseen
     bars without leaking the test partition.
  2. Features: parameter columns + engineered (friction_pct_of_risk) + KNN
     features computed fold-wise in parameter space. No trade-derived
     summaries (audit_* columns in the parquet are dropped).
  3. Parameter-space KNN features (nn10_mean_target, nn10_std_target) act
     as a GP-flavored smoother: per-combo neighborhood expected Sharpe +
     uncertainty. Computed inside each CV fold using only fold-train targets
     to prevent target leakage across folds.
  4. Reports BOTH OOF R² and Spearman rank correlation. Spearman is the
     operative metric for top-K selection.

Four boosters: point regression + quantile p10/p50/p90 (same shape as v11).
UCB ranking happens in extract_top_combos_v12.py.

Run locally:
    .venv/Scripts/python scripts/models/ml1_surrogate_v12.py
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "data" / "ml" / "ml1_results_v12"
FEATURES_PARQUET = OUTPUT_DIR / "combo_features_v12.parquet"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COL = "target_robust_sharpe"

# The `audit_` prefix is our convention for "computed from trade stream —
# do NOT feed to the model." Matches the naming in build_combo_features_ml1_v12.
EXCLUDE_PREFIX = "audit_"
EXCLUDE_EXACT = {
    "global_combo_id", "combo_id",
    "target_robust_sharpe", "target_median_wf_sharpe", "target_std_wf_sharpe",
    "target_n_valid_windows",
}

CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]
BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
]

# KNN over parameter space
KNN_K = 10

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
    {"name": "robust_sharpe_point", "objective": "regression", "metric": "rmse"},
    {"name": "robust_sharpe_p10", "objective": "quantile", "alpha": 0.1,
     "metric": "quantile"},
    {"name": "robust_sharpe_p50", "objective": "quantile", "alpha": 0.5,
     "metric": "quantile"},
    {"name": "robust_sharpe_p90", "objective": "quantile", "alpha": 0.9,
     "metric": "quantile"},
]


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in EXCLUDE_EXACT and not c.startswith(EXCLUDE_PREFIX)]


def _encode_for_knn(df_feats: pd.DataFrame) -> np.ndarray:
    """Encode combo features to a numeric matrix for KNN distance.

    Numeric cols -> standardized (fit on input subset's own stats for fold-wise
    honesty; this is a static transform given any input). Categorical cols ->
    integer codes with categories aligned on the full pool, then cast to
    one-hot. Boolean cols -> 0/1 floats.
    """
    parts: list[np.ndarray] = []
    for c in df_feats.columns:
        s = df_feats[c]
        if c in CATEGORICAL_COLS:
            codes = s.astype("category").cat.codes.to_numpy()
            # Integer codes are fine for Euclidean distance as long as the same
            # mapping is applied to every call (we pass through the same frame).
            parts.append(codes.astype(np.float64)[:, None])
        elif c in BOOLEAN_COLS:
            parts.append(s.astype(np.float64).to_numpy()[:, None])
        else:
            v = s.astype(np.float64).to_numpy()
            v = np.nan_to_num(v, nan=np.nanmean(v))
            parts.append(v[:, None])
    X = np.concatenate(parts, axis=1)
    # Standardize across rows — equal weight per feature regardless of scale.
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def _knn_features(X_encoded: np.ndarray, targets: np.ndarray,
                  train_idx: np.ndarray, query_idx: np.ndarray,
                  k: int = KNN_K) -> tuple[np.ndarray, np.ndarray]:
    """Compute (nn_mean, nn_std) for query_idx using only train_idx as the
    neighbor pool and their targets.

    When query is in train_idx (self), exclude the nearest match (which is
    itself) by querying k+1 and dropping the nearest.
    """
    train_X = X_encoded[train_idx]
    train_y = targets[train_idx]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(train_X)

    query_X = X_encoded[query_idx]
    dists, idxs = nn.kneighbors(query_X, n_neighbors=k + 1)

    out_mean = np.empty(len(query_idx), dtype=np.float64)
    out_std = np.empty(len(query_idx), dtype=np.float64)
    query_set = set(int(i) for i in query_idx)
    for i, qi in enumerate(query_idx):
        nbr_pos = idxs[i]
        # If the query point is itself in train_idx, its row will appear as
        # distance 0. Remove it and take the next k neighbors.
        if int(qi) in set(int(train_idx[p]) for p in nbr_pos):
            mask = np.array([int(train_idx[p]) != int(qi) for p in nbr_pos])
            nbr_pos = nbr_pos[mask][:k]
        else:
            nbr_pos = nbr_pos[:k]
        nbr_targets = train_y[nbr_pos]
        out_mean[i] = float(np.mean(nbr_targets))
        out_std[i] = float(np.std(nbr_targets, ddof=1)) if len(nbr_targets) >= 2 else 0.0
    return out_mean, out_std


def _prepare_lgb(df: pd.DataFrame, feat_cols: list[str]
                 ) -> tuple[pd.DataFrame, list[str]]:
    """Cast booleans / categoricals for LightGBM consumption. Returns X and
    the final list of categorical col names that actually appear in X."""
    X = df[feat_cols].copy()
    for c in BOOLEAN_COLS:
        if c in X.columns:
            X[c] = X[c].astype(float).fillna(0).astype(int)
    cat_in_X = []
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
            cat_in_X.append(c)
    return X, cat_in_X


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _train_booster(cfg: dict, X: pd.DataFrame, y: np.ndarray, cat_features: list[str],
                   X_encoded_for_knn: np.ndarray
                   ) -> dict:
    params = dict(LGB_BASE)
    params["objective"] = cfg["objective"]
    params["metric"] = cfg["metric"]
    if "alpha" in cfg:
        params["alpha"] = cfg["alpha"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    oof = np.full(len(X), np.nan, dtype=np.float64)
    best_trees = []

    for fi, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        # Compute KNN features fold-wise: pool = tr_idx, query = all; but
        # we only use predictions on va_idx for OOF scoring. For the
        # fold-train rows we still need KNN features as model inputs, so
        # compute them too (self-exclusion handled inside _knn_features).
        nn_mean, nn_std = _knn_features(
            X_encoded_for_knn, y, train_idx=tr_idx,
            query_idx=np.arange(len(X)),
        )
        X_fold = X.copy()
        X_fold["nn10_mean_target"] = nn_mean
        X_fold["nn10_std_target"] = nn_std

        Xtr, Xva = X_fold.iloc[tr_idx], X_fold.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_features,
                             free_raw_data=False)
        dvalid = lgb.Dataset(Xva, label=yva, categorical_feature=cat_features,
                             reference=dtrain, free_raw_data=False)
        booster = lgb.train(
            params, dtrain, num_boost_round=N_ESTIMATORS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        pred = booster.predict(Xva, num_iteration=booster.best_iteration)
        oof[va_idx] = pred

        if cfg["objective"] == "regression":
            loss = float(np.sqrt(np.mean((yva - pred) ** 2)))
            label = "RMSE"
        else:
            loss = _pinball_loss(yva, pred, cfg["alpha"])
            label = f"pinball@{cfg['alpha']}"
        fold_metrics.append(loss)
        best_trees.append(booster.best_iteration)
        print(f"    fold {fi}/{N_FOLDS}: {label}={loss:.4f}  "
              f"trees={booster.best_iteration}", flush=True)

    summary = {
        "name": cfg["name"],
        "objective": cfg["objective"],
        "alpha": cfg.get("alpha"),
        "fold_losses": fold_metrics,
        "mean_loss": float(np.mean(fold_metrics)),
        "mean_best_trees": int(round(np.mean(best_trees))),
    }

    # OOF metrics against the training-fold target. R² and Spearman.
    y_oof_mask = ~np.isnan(oof)
    y_true = y[y_oof_mask]
    y_pred = oof[y_oof_mask]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rho, _ = spearmanr(y_true, y_pred)
    summary["oof_r2"] = r2
    summary["oof_spearman"] = float(rho) if rho is not None else float("nan")
    summary["oof_rmse"] = float(np.sqrt(ss_res / len(y_true)))
    if cfg["objective"] == "quantile":
        summary["oof_pinball"] = _pinball_loss(y_true, y_pred, cfg["alpha"])

    # Full-data refit using mean best_iteration — final artifact.
    # Use full-data KNN (self-excluded).
    nn_mean_full, nn_std_full = _knn_features(
        X_encoded_for_knn, y,
        train_idx=np.arange(len(X)), query_idx=np.arange(len(X)),
    )
    Xfull = X.copy()
    Xfull["nn10_mean_target"] = nn_mean_full
    Xfull["nn10_std_target"] = nn_std_full
    dfull = lgb.Dataset(Xfull, label=y, categorical_feature=cat_features,
                        free_raw_data=False)
    final = lgb.train(params, dfull, num_boost_round=summary["mean_best_trees"])
    summary["_booster"] = final
    summary["_feature_names"] = list(Xfull.columns)
    summary["_oof_pred"] = oof
    return summary


def _plot_feature_importance(models: list[dict], path: Path) -> None:
    n_m = len(models)
    fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 6), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        booster = m["_booster"]
        imp = pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=True).tail(20)
        ax.barh(imp["feature"], imp["gain"])
        ax.set_title(f"{m['name']} (gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)

    print(f"[v12-train] loading {args.features.relative_to(REPO)}", flush=True)
    df = pd.read_parquet(args.features)
    feat_cols = _select_feature_cols(df)
    print(f"[v12-train] {len(df)} combos, {len(df.columns)} cols total, "
          f"{len(feat_cols)} model-input features", flush=True)
    print(f"[v12-train] excluded prefix '{EXCLUDE_PREFIX}': "
          f"{[c for c in df.columns if c.startswith(EXCLUDE_PREFIX)]}",
          flush=True)

    X, cat_features = _prepare_lgb(df, feat_cols)
    print(f"[v12-train] {X.shape[1]} features, {len(cat_features)} categorical: "
          f"{cat_features}", flush=True)

    # Encode once for KNN (same mapping used for every fold).
    X_encoded = _encode_for_knn(X)
    print(f"[v12-train] KNN encoding shape: {X_encoded.shape}", flush=True)

    y = df[TARGET_COL].astype(np.float64).to_numpy()
    print(f"[v12-train] target p5={np.quantile(y,0.05):.3f} "
          f"p50={np.median(y):.3f} p95={np.quantile(y,0.95):.3f}", flush=True)

    results = []
    t0 = time.time()
    for cfg in BOOSTERS:
        print(f"\n[v12-train] {cfg['name']} ({cfg['objective']}"
              + (f", alpha={cfg.get('alpha')}" if 'alpha' in cfg else "")
              + ")", flush=True)
        summary = _train_booster(cfg, X, y, cat_features, X_encoded)
        results.append(summary)
        print(f"  [{cfg['name']}] OOF R²={summary['oof_r2']:.4f}  "
              f"Spearman={summary['oof_spearman']:.4f}  "
              f"RMSE={summary['oof_rmse']:.4f}", flush=True)

        out_path = args.output_dir / "models" / f"{cfg['name']}.txt"
        summary["_booster"].save_model(str(out_path))
        print(f"  saved -> {out_path.relative_to(REPO)}", flush=True)

    dt = time.time() - t0

    cv_payload = {
        "target_col": TARGET_COL,
        "feature_names": feat_cols + ["nn10_mean_target", "nn10_std_target"],
        "categorical_features": cat_features,
        "n_combos": int(len(df)),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "knn_k": KNN_K,
        "lgb_base": LGB_BASE,
        "train_seconds": dt,
        "boosters": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ],
    }
    (args.output_dir / "cv_results.json").write_text(
        json.dumps(cv_payload, indent=2, default=float))
    print(f"\n[v12-train] wrote {(args.output_dir / 'cv_results.json').relative_to(REPO)}",
          flush=True)

    _plot_feature_importance(results, args.output_dir / "feature_importance.png")
    print(f"[v12-train] wrote feature_importance.png", flush=True)

    print("\n=== OOF summary ===")
    for r in results:
        extras = ""
        if "oof_pinball" in r:
            extras = f"  pinball@{r['alpha']}={r['oof_pinball']:.4f}"
        print(f"  {r['name']:25}  R²={r['oof_r2']:+.4f}  "
              f"Spearman={r['oof_spearman']:+.4f}{extras}")


if __name__ == "__main__":
    main()
