"""Extract top-K combos under ML#1 v12 UCB ranking.

Loads the four v12 boosters, computes parameter-space KNN features on the
full eligible pool (final model was retrained on all combos), scores each,
ranks by UCB = p50 + kappa * (p90 - p10) / 2, and writes
`evaluation/top_strategies_v12.json` in the schema consumed by
`composed_strategy_runner`.

Default kappa=0 => pure exploit on p50.

Sanity checks printed:
  - Spearman(predicted_p50, target_robust_sharpe) on the eligible pool —
    the honest ranking-quality metric on the training distribution.
  - v10_9264 rank — regression check that the friction-toxic outlier still
    lands in the bottom decile.

Usage:
    .venv/Scripts/python scripts/analysis/extract_top_combos_v12.py
    .venv/Scripts/python scripts/analysis/extract_top_combos_v12.py --kappa 0.5 --top-k 20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
ML1_DIR = REPO / "data" / "ml" / "ml1_results_v12"
FEATURES_PARQUET = ML1_DIR / "combo_features_v12.parquet"
MODELS_DIR = ML1_DIR / "models"
OUTPUT_JSON = REPO / "evaluation" / "top_strategies_v12.json"

TARGET_COL = "target_robust_sharpe"
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
KNN_K = 10

# Schema-compatible with legacy top_strategies.json — composed_strategy_runner
# reads entry["parameters"] + stop_fixed_pts_resolved.
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
    "stop_fixed_pts_resolved",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--output", type=Path, default=OUTPUT_JSON)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-trades", type=int, default=500)
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="UCB exploration weight (0 = exploit = p50 only)")
    return ap.parse_args()


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in EXCLUDE_EXACT and not c.startswith(EXCLUDE_PREFIX)]


def _prepare_for_lgb(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = df[feat_cols].copy()
    for c in BOOLEAN_COLS:
        if c in X.columns:
            X[c] = X[c].astype(float).fillna(0).astype(int)
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _encode_for_knn(df_feats: pd.DataFrame) -> np.ndarray:
    parts: list[np.ndarray] = []
    for c in df_feats.columns:
        s = df_feats[c]
        if c in CATEGORICAL_COLS:
            codes = s.astype("category").cat.codes.to_numpy()
            parts.append(codes.astype(np.float64)[:, None])
        elif c in BOOLEAN_COLS:
            parts.append(s.astype(np.float64).to_numpy()[:, None])
        else:
            v = s.astype(np.float64).to_numpy()
            v = np.nan_to_num(v, nan=np.nanmean(v))
            parts.append(v[:, None])
    X = np.concatenate(parts, axis=1)
    return StandardScaler().fit_transform(X)


def _knn_features_full(X_encoded: np.ndarray, targets: np.ndarray,
                       k: int = KNN_K) -> tuple[np.ndarray, np.ndarray]:
    """Full-pool KNN — matches what the final model saw during refit."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_encoded)
    dists, idxs = nn.kneighbors(X_encoded, n_neighbors=k + 1)
    # Drop column 0 (self).
    nbr_idx = idxs[:, 1:]
    nbr_targets = targets[nbr_idx]
    return nbr_targets.mean(axis=1), nbr_targets.std(axis=1, ddof=1)


def _entry_payload(row: pd.Series) -> dict:
    params = {c: row[c] for c in PARAM_COLS if c in row.index}
    # LightGBM-friendly types end up as numpy scalars / pandas types; coerce to
    # plain python so json.dumps round-trips cleanly.
    clean = {}
    for k, v in params.items():
        if pd.isna(v):
            clean[k] = None
        elif hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v
    return {
        "global_combo_id": row["global_combo_id"],
        "combo_id": int(row["combo_id"]),
        "parameters": clean,
        "stop_fixed_pts_resolved": float(row["stop_fixed_pts_resolved"]),
        "predicted_p10": float(row["pred_p10"]),
        "predicted_p50": float(row["pred_p50"]),
        "predicted_p90": float(row["pred_p90"]),
        "predicted_point": float(row["pred_point"]),
        "ucb_score": float(row["ucb_score"]),
        "realised": {
            "target_robust_sharpe": float(row[TARGET_COL]),
            "target_median_wf_sharpe": float(row["target_median_wf_sharpe"]),
            "target_std_wf_sharpe": float(row["target_std_wf_sharpe"]),
            "audit_n_trades": int(row["audit_n_trades"]),
            "audit_full_net_sharpe": float(row["audit_full_net_sharpe"]),
            "audit_full_gross_sharpe": float(row["audit_full_gross_sharpe"]),
        },
    }


def main() -> None:
    args = _parse_args()
    print(f"[v12-extract] loading {args.features.relative_to(REPO)}", flush=True)
    df = pd.read_parquet(args.features)
    print(f"[v12-extract] {len(df)} combos", flush=True)

    feat_cols = _select_feature_cols(df)
    X = _prepare_for_lgb(df, feat_cols)
    X_encoded = _encode_for_knn(X)
    y_known = df[TARGET_COL].astype(np.float64).to_numpy()

    # Full-pool KNN features — matches trainer's final refit.
    nn_mean, nn_std = _knn_features_full(X_encoded, y_known)
    X["nn10_mean_target"] = nn_mean
    X["nn10_std_target"] = nn_std

    preds = {}
    for name in ["robust_sharpe_point", "robust_sharpe_p10",
                 "robust_sharpe_p50", "robust_sharpe_p90"]:
        booster = lgb.Booster(model_file=str(args.models_dir / f"{name}.txt"))
        fn = booster.feature_name()
        missing = [c for c in fn if c not in X.columns]
        if missing:
            raise SystemExit(f"[v12-extract] model expects {missing} — "
                             f"feature set mismatch")
        preds[name] = booster.predict(X[fn])
        print(f"[v12-extract] scored {name}: p50={np.median(preds[name]):.3f}",
              flush=True)

    df["pred_point"] = preds["robust_sharpe_point"]
    df["pred_p10"] = preds["robust_sharpe_p10"]
    df["pred_p50"] = preds["robust_sharpe_p50"]
    df["pred_p90"] = preds["robust_sharpe_p90"]
    df["ucb_score"] = df["pred_p50"] + args.kappa * (
        df["pred_p90"] - df["pred_p10"]) / 2.0

    # Eligibility gate.
    eligible = df[df["audit_n_trades"] >= args.min_trades].copy()
    print(f"[v12-extract] n_trades >= {args.min_trades}: "
          f"{len(eligible)} eligible, {len(df)-len(eligible)} rejected",
          flush=True)

    # Ranking sanity: Spearman of predicted_p50 vs realised target_robust_sharpe
    # on the eligible pool. This is an in-sample metric (final model saw all),
    # useful for schema-sanity not generalization.
    rho, _ = spearmanr(eligible[TARGET_COL], eligible["pred_p50"])
    print(f"[v12-extract] in-sample Spearman(pred_p50, target)={rho:.4f}",
          flush=True)

    # v10_9264 regression check
    regression_checks = {}
    if "v10_9264" in eligible["global_combo_id"].values:
        sorted_all = eligible.sort_values("ucb_score", ascending=False).reset_index(drop=True)
        idx_9264 = int(sorted_all.index[
            sorted_all["global_combo_id"] == "v10_9264"
        ][0])
        regression_checks["v10_9264_rank"] = idx_9264
        regression_checks["v10_9264_percentile"] = idx_9264 / len(sorted_all)
        print(f"[v12-extract] v10_9264 rank: {idx_9264}/{len(sorted_all)} "
              f"({regression_checks['v10_9264_percentile']:.1%})", flush=True)

    top = eligible.sort_values("ucb_score", ascending=False).head(args.top_k)
    print(f"[v12-extract] top-{args.top_k}: {list(top['global_combo_id'])}",
          flush=True)

    payload = {
        "source_features": str(args.features.relative_to(REPO)),
        "source_models_dir": str(args.models_dir.relative_to(REPO)),
        "ranking_metric": f"UCB(p50, p90, p10, kappa={args.kappa})",
        "kappa": args.kappa,
        "min_trades": args.min_trades,
        "top_k": args.top_k,
        "pool_sizes": {
            "total": int(len(df)),
            "eligible": int(len(eligible)),
        },
        "in_sample_rank_metrics": {
            "spearman_pred_p50_vs_target": float(rho),
        },
        "regression_checks": regression_checks,
        "top": [_entry_payload(r) for _, r in top.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[v12-extract] wrote {args.output.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
