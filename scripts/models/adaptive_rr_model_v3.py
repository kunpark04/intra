"""
adaptive_rr_model_v3.py — V2 + Family A autocorrelation features, full-9.5M retrain.

Extends V2 with the four autocorrelation features validated by B8 + B8-SHAP:
  prior_wr_10, prior_wr_50, prior_r_ma10, has_history_50

Also adds `global_combo_id` as an explicit categorical feature so LightGBM can
absorb static combo-quality signal there (defending against prior_wr_50's
partial identity component, per the B8-SHAP audit).

Uses V2's stream-Bernoulli parquet loader across v2-v10 MFE parquets. Since
only the v10_train_wf parquet has entry_bar_idx, V3 synthesises per-combo
ordering via groupby(global_combo_id).cumcount() — this is valid because
param_sweep.py writes trade rows in bar order within each combo.

Outputs to data/ml/adaptive_rr_v3/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))

# Reuse V2's stream loader + stratified subsample.
from adaptive_rr_model_v2 import (  # noqa: E402
    load_mfe_parquets, filter_combos, _stratified_subsample,
)

# ── Constants ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data/ml")
OUTPUT_DIR = DATA_DIR / "adaptive_rr_v3"

ENTRY_FEATURES = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side",
]
COMBO_FEATURES = ["stop_method", "exit_on_opposite_signal"]
RR_FEATURE = "candidate_rr"
DERIVED_FEATURES = ["abs_zscore_entry", "rr_x_atr"]
FAMILY_A = ["prior_wr_10", "prior_wr_50", "prior_r_ma10", "has_history_50"]
ID_FEATURE = "global_combo_id"  # LightGBM categorical

ALL_FEATURES = (ENTRY_FEATURES + COMBO_FEATURES + [RR_FEATURE]
                + DERIVED_FEATURES + FAMILY_A + [ID_FEATURE])

CATEGORICAL_COLS = ["stop_method", "side", ID_FEATURE]

NUMERIC_FEATURE_COLS = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
]

RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2).tolist()

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "num_threads": 4,
    "max_cat_to_onehot": 4,  # high-cardinality combo_id → split-based, not one-hot
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the V3 adaptive-R:R trainer."""
    p = argparse.ArgumentParser(description="V3 adaptive R:R — V2 + Family A")
    p.add_argument("--versions", type=int, nargs="+", default=list(range(2, 11)))
    p.add_argument("--max-rows", type=int, default=10_000_000)
    p.add_argument("--min-trades-per-combo", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--target-base-trades", type=int, default=1_200_000,
                   help="Base trades to load pre-expansion (default 1.2M)")
    return p.parse_args()


# Monkey-patch V2's PARQUET_COLUMNS to include label_win + r_multiple
# so Family A computation has the data it needs.
import adaptive_rr_model_v2 as _v2  # noqa: E402
_v2.PARQUET_COLUMNS = list(_v2.PARQUET_COLUMNS) + ["label_win", "r_multiple"]


# ── Family A feature engineering ──────────────────────────────────────────

def add_family_a(df: pd.DataFrame) -> pd.DataFrame:
    """Per-combo rolling prior-N win-rate and realised-R features.

    Row order within a combo is assumed chronological (param_sweep.py writes
    trades in bar order); we don't need entry_bar_idx because cumcount on
    that preserved order *is* the chronological position.
    """
    # Stable sort keeps intra-combo row order while grouping combos together.
    df = df.sort_values("global_combo_id", kind="stable").reset_index(drop=True)

    g = df.groupby("global_combo_id", sort=False, observed=True)
    win_prev = g["label_win"].shift(1)
    r_prev = g["r_multiple"].shift(1)

    gp_win = win_prev.groupby(df["global_combo_id"], sort=False, observed=True)
    gp_r = r_prev.groupby(df["global_combo_id"], sort=False, observed=True)

    # fillna BEFORE cast — pandas 2.x emits "invalid value encountered in cast"
    # when .astype(np.float32) sees NaN from short rolling windows (min_periods).
    df["prior_wr_10"] = gp_win.transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    ).fillna(0.5).astype(np.float32)
    df["prior_wr_50"] = gp_win.transform(
        lambda s: s.rolling(50, min_periods=10).mean()
    ).fillna(0.5).astype(np.float32)
    df["prior_r_ma10"] = gp_r.transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    ).fillna(0.0).astype(np.float32)

    combo_rank = g.cumcount()
    df["has_history_50"] = (combo_rank >= 25).astype(np.int8)

    print(f"  Family A computed: prior_wr_10 mean={df['prior_wr_10'].mean():.3f}, "
          f"prior_wr_50 mean={df['prior_wr_50'].mean():.3f}, "
          f"has_history_50 rate={df['has_history_50'].mean():.3f}")
    return df


# ── R:R expansion (V3: adds Family A + global_combo_id through to expanded df) ──

def expand_rr_levels(df: pd.DataFrame, rr_levels: list[float],
                     max_rows: int) -> pd.DataFrame:
    """Expand a base trade frame into long format along `RR_LEVELS`.

    Args:
        df: Base trade DataFrame.

    Returns:
        Long-format frame with synthetic `would_win` label and R:R feature.
    """
    print(f"\n[expand] {len(df):,} base trades × {len(rr_levels)} R:R...")

    rng = np.random.default_rng(42)
    n_rr = len(rr_levels)
    rr_arr = np.array(rr_levels, dtype=np.float32)

    sample_n = max_rows // n_rr
    if len(df) > sample_n:
        print(f"  Subsampling stratified by sweep_version: "
              f"{len(df):,} -> {sample_n:,} base trades")
        df = _stratified_subsample(df, sample_n, rng)
    else:
        df = df.reset_index(drop=True)

    n_base = len(df)
    out: dict[str, np.ndarray] = {}

    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(dtype=np.float32), n_rr)

    for col in ["stop_method", "side"]:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(), n_rr)

    if "exit_on_opposite_signal" in df.columns:
        out["exit_on_opposite_signal"] = np.repeat(
            df["exit_on_opposite_signal"].to_numpy(dtype=np.int8), n_rr)

    # Family A numerics
    for col in FAMILY_A:
        if col in df.columns:
            dtype = np.int8 if col == "has_history_50" else np.float32
            out[col] = np.repeat(df[col].to_numpy(dtype=dtype), n_rr)

    # global_combo_id stays as-is for both CV grouping and LightGBM categorical
    out[ID_FEATURE] = np.repeat(df[ID_FEATURE].to_numpy(), n_rr)
    out["sweep_version"] = np.repeat(
        df["sweep_version"].to_numpy(dtype=np.int8), n_rr)

    out[RR_FEATURE] = np.tile(rr_arr, n_base)

    mfe = df["mfe_points"].to_numpy(dtype=np.float32)
    stop = df["stop_distance_pts"].to_numpy(dtype=np.float32)
    would_win_2d = mfe[:, None] >= rr_arr[None, :] * stop[:, None]
    out["would_win"] = would_win_2d.ravel().astype(np.int8)

    if "zscore_entry" in out:
        out["abs_zscore_entry"] = np.abs(out["zscore_entry"])
    if "atr_points" in out:
        out["rr_x_atr"] = out[RR_FEATURE] * out["atr_points"]

    del would_win_2d, mfe, stop

    expanded = pd.DataFrame(out, copy=False)
    for col in CATEGORICAL_COLS:
        if col in expanded.columns:
            expanded[col] = expanded[col].astype("category")

    print(f"  Expanded: {len(expanded):,} rows, "
          f"{expanded.memory_usage(deep=True).sum() / 1e9:.2f} GB in-memory")
    return expanded


# ── Training ──────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, n_folds: int) -> dict:
    """Fit the V3 LightGBM booster on a prepared train/val split.

    Args:
        X_tr: Training feature matrix.
        y_tr: Training labels.
        X_va: Validation feature matrix.
        y_va: Validation labels.
        params: LightGBM parameter dict.

    Returns:
        Trained `lightgbm.Booster`.
    """
    print(f"\n[train] Training V3 on {len(df):,} rows, {n_folds}-fold CV...")
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    X = df[feature_cols]
    y = df["would_win"].to_numpy(dtype=np.int8)
    groups = df[ID_FEATURE].to_numpy()
    rr_arr = df[RR_FEATURE].to_numpy(dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    fold_metrics = []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        t0 = time.time()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train,
                                 categorical_feature=CATEGORICAL_COLS,
                                 free_raw_data=True)
        val_data = lgb.Dataset(X_val, label=y_val,
                               categorical_feature=CATEGORICAL_COLS,
                               free_raw_data=True)

        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False),
                     lgb.log_evaluation(period=0)]

        model = lgb.train(LGB_PARAMS, train_data,
                          num_boost_round=LGB_PARAMS["n_estimators"],
                          valid_sets=[val_data], callbacks=callbacks)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        ll = log_loss(y_val, val_preds)
        auc = roc_auc_score(y_val, val_preds)
        brier = brier_score_loss(y_val, val_preds)
        fold_metrics.append({
            "fold": fold_i, "log_loss": float(ll), "auc": float(auc),
            "brier": float(brier), "best_iter": int(model.best_iteration or 0),
            "time_s": time.time() - t0,
        })
        print(f"  fold {fold_i}: AUC={auc:.4f} LL={ll:.4f} Brier={brier:.4f} "
              f"best_iter={model.best_iteration} t={time.time()-t0:.0f}s")
        models.append(model)

    return {
        "oof_preds": oof_preds, "y": y, "rr": rr_arr,
        "fold_metrics": fold_metrics, "models": models,
        "feature_cols": feature_cols,
    }


# ── Isotonic-per-RR calibration ───────────────────────────────────────────

def fit_isotonic_per_rr(y: np.ndarray, p: np.ndarray, rr: np.ndarray) -> dict:
    """Fit one `IsotonicRegression` calibrator per R:R level.

    Args:
        p: Raw booster probabilities (long-format vector).
        y: Outcomes aligned to `p`.
        rr: Per-row R:R level.

    Returns:
        Dict mapping R:R → fitted `IsotonicRegression`.
    """
    calibrators = {}
    p_cal = np.empty_like(p)
    for rr_val in RR_LEVELS:
        mask = np.isclose(rr, rr_val)
        if mask.sum() < 100:
            p_cal[mask] = p[mask]
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p[mask], y[mask])
        p_cal[mask] = iso.predict(p[mask])
        calibrators[f"{rr_val:.2f}"] = {
            "X": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }
    return {"calibrators": calibrators, "p_cal": p_cal}


def ece_20bin(y: np.ndarray, p: np.ndarray) -> float:
    """Convenience ECE wrapper with a fixed 20-bin equal-width grid."""
    bins = np.linspace(0, 1, 21)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(y)
    for b in range(20):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Train the V3 stack: LightGBM booster + per-R:R pooled isotonic calibrator."""
    args = parse_args()
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[v3] Loading v{args.versions}, target {args.target_base_trades:,} base trades...")
    df = load_mfe_parquets(args.versions, args.target_base_trades)
    df = filter_combos(df, args.min_trades_per_combo)

    # Need label_win + r_multiple for Family A
    if "label_win" not in df.columns or "r_multiple" not in df.columns:
        sys.exit("[v3] FATAL: MFE parquets missing label_win/r_multiple.")

    print(f"\n[v3] Computing Family A features...")
    df = add_family_a(df)

    expanded = expand_rr_levels(df, RR_LEVELS, args.max_rows)
    del df

    results = train_model(expanded, args.n_folds)

    # Raw OOF metrics
    y = results["y"]; p = results["oof_preds"]; rr = results["rr"]
    oof_auc = float(roc_auc_score(y, p))
    oof_ll = float(log_loss(y, p))
    oof_brier = float(brier_score_loss(y, p))
    oof_ece_raw = ece_20bin(y, p)
    print(f"\n[v3] OOF raw: AUC={oof_auc:.4f} LL={oof_ll:.4f} "
          f"Brier={oof_brier:.4f} ECE={oof_ece_raw:.4f}")

    # Isotonic-per-RR calibration
    cal = fit_isotonic_per_rr(y, p, rr)
    p_cal = cal["p_cal"]
    oof_ece_cal = ece_20bin(y, p_cal)
    oof_brier_cal = float(brier_score_loss(y, p_cal))
    print(f"[v3] OOF calibrated: Brier={oof_brier_cal:.4f} ECE={oof_ece_cal:.4f}")

    # Refit on all data for production booster
    print(f"\n[v3] Training final booster on full data...")
    feat = results["feature_cols"]
    full = lgb.Dataset(expanded[feat], label=y,
                       categorical_feature=CATEGORICAL_COLS,
                       free_raw_data=True)
    best_iter = int(np.mean([m["best_iter"] for m in results["fold_metrics"]]))
    print(f"  Using best_iter={best_iter} (mean across folds)")
    prod_params = dict(LGB_PARAMS); prod_params["n_estimators"] = best_iter
    booster = lgb.train(prod_params, full, num_boost_round=best_iter)
    booster.save_model(str(OUTPUT_DIR / "booster_v3.txt"))
    print(f"  Saved {OUTPUT_DIR/'booster_v3.txt'}")

    # Feature importance
    imp_gain = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat, imp_gain), key=lambda x: -x[1])

    # Save metrics
    metrics = {
        "oof_auc": oof_auc, "oof_log_loss": oof_ll,
        "oof_brier_raw": oof_brier, "oof_brier_cal": oof_brier_cal,
        "oof_ece_raw": oof_ece_raw, "oof_ece_cal": oof_ece_cal,
        "fold_metrics": results["fold_metrics"],
        "feature_cols": feat,
        "feature_importance_gain": [{"feature": f, "gain": float(g)}
                                    for f, g in feat_imp],
        "n_rows_expanded": len(expanded),
        "n_combos": int(pd.Series(results["y"]).size // 17),
        "best_iter_mean": best_iter,
        "runtime_seconds": time.time() - t0,
        "versions": args.versions,
    }
    (OUTPUT_DIR / "metrics_v3.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Saved {OUTPUT_DIR/'metrics_v3.json'}")

    (OUTPUT_DIR / "isotonic_calibrators_v3.json").write_text(
        json.dumps(cal["calibrators"], indent=2))
    print(f"  Saved {OUTPUT_DIR/'isotonic_calibrators_v3.json'}")

    # Top-15 feature importance plot
    top = feat_imp[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [x[0] for x in top][::-1]
    gains = [x[1] for x in top][::-1]
    colors = ["tab:orange" if n in FAMILY_A else
              ("tab:red" if n == ID_FEATURE else "tab:blue") for n in names]
    ax.barh(names, gains, color=colors)
    ax.set_xlabel("LightGBM gain")
    ax.set_title(f"V3 top-15 features (orange=Family A, red={ID_FEATURE})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance_v3.png", dpi=140)
    plt.close(fig)

    print(f"\n[v3] Done in {(time.time()-t0)/60:.1f} min. "
          f"AUC={oof_auc:.4f} vs V2 baseline 0.806 (Δ={oof_auc-0.806:+.4f})")
    print(f"[v3] ECE cal={oof_ece_cal:.4f} vs V2 0.004")
    print(f"\n[v3] Top 10 features by gain:")
    for f, g in feat_imp[:10]:
        print(f"  {f:30s}  {g:>12.0f}")


if __name__ == "__main__":
    main()
