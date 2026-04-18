"""
adaptive_rr_model_v4.py — V3 architecture retrained on v11 friction-aware sweep.

Motivation (Phase 6.7): V3 was trained on v2-v10 sweep economics and rejects
100% of v11 top-combo trades as out-of-distribution. The top-50 κ=0 variant
surfaced a +0.54 Sharpe portfolio but with 81% drawdown — the filter is the
tail-cutter, and without it the portfolio bleeds on the worst combos.

V4 is byte-identical to V3 (same features, LGB params, Family A, per-R:R
pooled isotonic) except for the input data source:

  V3: data/ml/mfe/ml_dataset_v{2..10}_mfe.parquet  (multi-version MFE re-runs)
  V4: data/ml/originals/ml_dataset_v11.parquet     (v11 inline MFE sweep)

v11 emits mfe_points / mae_points / stop_distance_pts / entry_bar_idx inline
per the v11 sweep contract (COST_PER_CONTRACT_RT baked in, no separate MFE
pass needed), so the loader drops multi-version budget math entirely.

Outputs to data/ml/adaptive_rr_v4/ — parallel to adaptive_rr_v3/; nothing in
V3 is modified.

Usage:
    python scripts/models/adaptive_rr_model_v4.py
    python scripts/models/adaptive_rr_model_v4.py --max-rows 5000000
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
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Constants (mirror V3) ─────────────────────────────────────────────────

DATA_DIR = Path("data/ml")
V11_PARQUET = DATA_DIR / "originals" / "ml_dataset_v11.parquet"
OUTPUT_DIR = DATA_DIR / "adaptive_rr_v4"
TRAIN_MATRIX_CACHE = OUTPUT_DIR / "train_matrix.parquet"

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
ID_FEATURE = "global_combo_id"

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

# Columns pulled from the v11 parquet — only what's needed downstream.
PARQUET_COLUMNS = [
    "combo_id",
    "mfe_points", "mae_points", "stop_distance_pts",
    "entry_bar_idx",            # chronological ordering for Family A
    "label_win", "r_multiple",  # Family A inputs
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
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
    "num_threads": 3,
    "max_cat_to_onehot": 4,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the V4 adaptive-R:R trainer."""
    p = argparse.ArgumentParser(description="V4 adaptive R:R — V3 retrained on v11")
    p.add_argument("--max-rows", type=int, default=10_000_000)
    p.add_argument("--min-trades-per-combo", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--target-base-trades", type=int, default=1_200_000,
                   help="Base trades to load pre-expansion (default 1.2M)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Ignore cached train_matrix.parquet and rebuild from v11.")
    return p.parse_args()


# ── v11 stream loader ─────────────────────────────────────────────────────

def load_v11_parquet(target_base_trades: int) -> pd.DataFrame:
    """Bernoulli-stream the v11 parquet into a ~target_base_trades sample.

    v11 is ~102M rows at ~6.6 GB compressed; we cannot materialise it. This
    iterates row-group batches via pyarrow and keeps each row with probability
    p = target / total_rows. Peak RAM ≈ one row group (~10 MB) plus the
    growing sample.
    """
    if not V11_PARQUET.exists():
        sys.exit(f"[v4] FATAL: {V11_PARQUET} not found.")

    pf = pq.ParquetFile(V11_PARQUET)
    total_rows = pf.metadata.num_rows
    available = {f.name for f in pf.schema_arrow}
    missing = [c for c in PARQUET_COLUMNS if c not in available]
    if missing:
        sys.exit(f"[v4] FATAL: v11 parquet missing columns: {missing}")
    cols = [c for c in PARQUET_COLUMNS if c in available]

    p_keep = min(1.0, target_base_trades / total_rows)
    print(f"[load] v11 parquet: {total_rows:,} rows, p_keep={p_keep:.5f}, "
          f"target={target_base_trades:,}")

    rng = np.random.default_rng(42)
    f32_cols_set = set(NUMERIC_FEATURE_COLS + [
        "mfe_points", "mae_points", "stop_distance_pts", "r_multiple"])

    kept_batches: list[pd.DataFrame] = []
    rows_scanned = 0
    for batch in pf.iter_batches(batch_size=65536, columns=cols):
        df_b = batch.to_pandas()
        rows_scanned += len(df_b)
        mask = rng.random(len(df_b)) < p_keep
        if mask.any():
            df_b = df_b.loc[mask]
            dtype_map = {c: np.float32 for c in f32_cols_set if c in df_b.columns}
            if dtype_map:
                df_b = df_b.astype(dtype_map)
            if "exit_on_opposite_signal" in df_b.columns:
                df_b["exit_on_opposite_signal"] = df_b[
                    "exit_on_opposite_signal"].astype(np.int8)
            if "label_win" in df_b.columns:
                df_b["label_win"] = df_b["label_win"].astype(np.int8)
            if "entry_bar_idx" in df_b.columns:
                df_b["entry_bar_idx"] = df_b["entry_bar_idx"].astype(np.int64)
            kept_batches.append(df_b)
        del batch, df_b

    if not kept_batches:
        sys.exit("[v4] FATAL: empty sample (p_keep too small?)")

    df = pd.concat(kept_batches, ignore_index=True, copy=False)
    del kept_batches

    # v11 has a single sweep version; emit a placeholder column so downstream
    # code that expects it (if any) doesn't choke, and build global_combo_id.
    df["sweep_version"] = np.int8(11)
    df["global_combo_id"] = "v11_" + df["combo_id"].astype(str)

    print(f"  Scanned {rows_scanned:,} rows, kept {len(df):,} trades "
          f"({df['global_combo_id'].nunique()} combos)")
    return df


def filter_combos(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Drop combos with fewer than min_trades samples (same as V3)."""
    combo_counts = df.groupby("global_combo_id").size()
    valid = combo_counts[combo_counts >= min_trades].index
    before = df["global_combo_id"].nunique()
    df = df[df["global_combo_id"].isin(valid)].copy()
    after = df["global_combo_id"].nunique()
    print(f"  Filtered combos: {before} -> {after} "
          f"(removed {before - after} with <{min_trades} trades)")
    print(f"  Filtered trades: {len(df):,}")
    return df


# ── Family A (chronological) ──────────────────────────────────────────────

def add_family_a(df: pd.DataFrame) -> pd.DataFrame:
    """V3 Family A formula, but sorted by entry_bar_idx (not cumcount).

    v11 emits entry_bar_idx inline, so per-combo chronological ordering is
    explicit rather than inferred from write order. Prevents subtle drift if
    a future sweep shuffles emission order.
    """
    df = df.sort_values(
        ["global_combo_id", "entry_bar_idx"], kind="stable"
    ).reset_index(drop=True)

    g = df.groupby("global_combo_id", sort=False, observed=True)
    win_prev = g["label_win"].shift(1)
    r_prev = g["r_multiple"].shift(1)

    gp_win = win_prev.groupby(df["global_combo_id"], sort=False, observed=True)
    gp_r = r_prev.groupby(df["global_combo_id"], sort=False, observed=True)

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

    print(f"  Family A: prior_wr_10 mean={df['prior_wr_10'].mean():.3f}, "
          f"prior_wr_50 mean={df['prior_wr_50'].mean():.3f}, "
          f"has_history_50 rate={df['has_history_50'].mean():.3f}")
    return df


# ── R:R expansion ─────────────────────────────────────────────────────────

def expand_rr_levels(df: pd.DataFrame, rr_levels: list[float],
                     max_rows: int) -> pd.DataFrame:
    """Expand base trades × RR levels into the long-format training frame.

    Single-version variant of V3's expand: no sweep_version stratification
    (v11 is the only source). If we're over the row budget, straight random
    subsample of base trades.
    """
    print(f"\n[expand] {len(df):,} base trades × {len(rr_levels)} R:R...")

    rng = np.random.default_rng(42)
    n_rr = len(rr_levels)
    rr_arr = np.array(rr_levels, dtype=np.float32)

    sample_n = max_rows // n_rr
    if len(df) > sample_n:
        print(f"  Subsampling {len(df):,} -> {sample_n:,} base trades")
        idx = rng.choice(len(df), size=sample_n, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
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

    for col in FAMILY_A:
        if col in df.columns:
            dtype = np.int8 if col == "has_history_50" else np.float32
            out[col] = np.repeat(df[col].to_numpy(dtype=dtype), n_rr)

    out[ID_FEATURE] = np.repeat(df[ID_FEATURE].to_numpy(), n_rr)
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
    wr_by_rr = expanded.groupby(RR_FEATURE, observed=True)["would_win"].mean()
    print("  Win rate by R:R:")
    for rr, wr in wr_by_rr.items():
        print(f"    R:R {rr:.2f}: {wr:.1%}")
    return expanded


# ── Training (byte-identical to V3) ───────────────────────────────────────

def train_model(df: pd.DataFrame, n_folds: int) -> dict:
    """StratifiedGroupKFold CV, mirrors V3."""
    print(f"\n[train] Training V4 on {len(df):,} rows, {n_folds}-fold CV...")
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


# ── Isotonic-per-RR calibration (byte-identical to V3) ────────────────────

def fit_isotonic_per_rr(y: np.ndarray, p: np.ndarray, rr: np.ndarray) -> dict:
    """Fit one IsotonicRegression per R:R level."""
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
    """20-bin equal-width ECE."""
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
    """Train V4: LightGBM booster + per-R:R pooled isotonic on v11 sweep."""
    args = parse_args()
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Adaptive R:R V4 — V3 retrained on v11 friction-aware sweep")
    print("=" * 70)

    if TRAIN_MATRIX_CACHE.exists() and not args.rebuild_cache:
        print(f"\n[cache] Loading expanded training matrix from {TRAIN_MATRIX_CACHE}")
        expanded = pd.read_parquet(TRAIN_MATRIX_CACHE)
        for col in CATEGORICAL_COLS:
            if col in expanded.columns and not pd.api.types.is_categorical_dtype(expanded[col]):
                expanded[col] = expanded[col].astype("category")
        print(f"  Loaded {len(expanded):,} rows, {expanded.memory_usage(deep=True).sum()/1e9:.2f} GB in-memory")
    else:
        df = load_v11_parquet(args.target_base_trades)
        df = filter_combos(df, args.min_trades_per_combo)

        if "label_win" not in df.columns or "r_multiple" not in df.columns:
            sys.exit("[v4] FATAL: v11 parquet missing label_win/r_multiple.")

        print("\n[v4] Computing Family A features...")
        df = add_family_a(df)

        expanded = expand_rr_levels(df, RR_LEVELS, args.max_rows)
        del df

        print(f"\n[cache] Saving expanded matrix to {TRAIN_MATRIX_CACHE}")
        # Drop categorical dtype for safe parquet round-trip; reapplied on read.
        to_write = expanded.copy()
        for col in CATEGORICAL_COLS:
            if col in to_write.columns and pd.api.types.is_categorical_dtype(to_write[col]):
                to_write[col] = to_write[col].astype(str)
        to_write.to_parquet(TRAIN_MATRIX_CACHE, index=False, compression="snappy")
        del to_write
        sz = TRAIN_MATRIX_CACHE.stat().st_size / 1e9
        print(f"  Wrote {sz:.2f} GB to {TRAIN_MATRIX_CACHE}")

    results = train_model(expanded, args.n_folds)

    y = results["y"]; p = results["oof_preds"]; rr = results["rr"]
    oof_auc = float(roc_auc_score(y, p))
    oof_ll = float(log_loss(y, p))
    oof_brier = float(brier_score_loss(y, p))
    oof_ece_raw = ece_20bin(y, p)
    print(f"\n[v4] OOF raw: AUC={oof_auc:.4f} LL={oof_ll:.4f} "
          f"Brier={oof_brier:.4f} ECE={oof_ece_raw:.4f}")

    cal = fit_isotonic_per_rr(y, p, rr)
    p_cal = cal["p_cal"]
    oof_ece_cal = ece_20bin(y, p_cal)
    oof_brier_cal = float(brier_score_loss(y, p_cal))
    print(f"[v4] OOF calibrated: Brier={oof_brier_cal:.4f} ECE={oof_ece_cal:.4f}")

    print("\n[v4] Training final booster on full data...")
    feat = results["feature_cols"]
    full = lgb.Dataset(expanded[feat], label=y,
                       categorical_feature=CATEGORICAL_COLS,
                       free_raw_data=True)
    best_iter = int(np.mean([m["best_iter"] for m in results["fold_metrics"]]))
    print(f"  Using best_iter={best_iter} (mean across folds)")
    prod_params = dict(LGB_PARAMS); prod_params["n_estimators"] = best_iter
    booster = lgb.train(prod_params, full, num_boost_round=best_iter)
    booster.save_model(str(OUTPUT_DIR / "booster_v4.txt"))
    print(f"  Saved {OUTPUT_DIR/'booster_v4.txt'}")

    imp_gain = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat, imp_gain), key=lambda x: -x[1])

    metrics = {
        "oof_auc": oof_auc, "oof_log_loss": oof_ll,
        "oof_brier_raw": oof_brier, "oof_brier_cal": oof_brier_cal,
        "oof_ece_raw": oof_ece_raw, "oof_ece_cal": oof_ece_cal,
        "fold_metrics": results["fold_metrics"],
        "feature_cols": feat,
        "feature_importance_gain": [{"feature": f, "gain": float(g)}
                                    for f, g in feat_imp],
        "n_rows_expanded": len(expanded),
        "n_isotonic_keys": len(cal["calibrators"]),
        "best_iter_mean": best_iter,
        "runtime_seconds": time.time() - t0,
        "source_parquet": str(V11_PARQUET),
        "target_base_trades": args.target_base_trades,
        "max_rows": args.max_rows,
        "n_folds": args.n_folds,
        "rr_levels": RR_LEVELS,
        "lgb_params": LGB_PARAMS,
    }
    (OUTPUT_DIR / "metrics_v4.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Saved {OUTPUT_DIR/'metrics_v4.json'}")

    (OUTPUT_DIR / "isotonic_calibrators_v4.json").write_text(
        json.dumps(cal["calibrators"], indent=2))
    print(f"  Saved {OUTPUT_DIR/'isotonic_calibrators_v4.json'} "
          f"({len(cal['calibrators'])} keys)")

    top = feat_imp[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [x[0] for x in top][::-1]
    gains = [x[1] for x in top][::-1]
    colors = ["tab:orange" if n in FAMILY_A else
              ("tab:red" if n == ID_FEATURE else "tab:blue") for n in names]
    ax.barh(names, gains, color=colors)
    ax.set_xlabel("LightGBM gain")
    ax.set_title(f"V4 top-15 features (orange=Family A, red={ID_FEATURE})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance_v4.png", dpi=140)
    plt.close(fig)

    print(f"\n[v4] Done in {(time.time()-t0)/60:.1f} min. "
          f"AUC={oof_auc:.4f} (V3 baseline ~0.81)")
    print(f"[v4] ECE cal={oof_ece_cal:.4f}")
    print(f"\n[v4] Top 10 features by gain:")
    for f, g in feat_imp[:10]:
        print(f"  {f:30s}  {g:>12.0f}")


if __name__ == "__main__":
    main()
