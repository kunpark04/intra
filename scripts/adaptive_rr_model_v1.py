"""
adaptive_rr_model_v1.py — LightGBM model for adaptive R:R selection.

Instead of a fixed min_rr, this model learns P(win | features, R:R) from
MFE/MAE path data. At inference time, it sweeps candidate R:R values and
picks the one that maximizes expected value: E[R] = P(win)*R - (1-P(win)).

Key design choices documented in tasks/adaptive_rr_decisions.md.

Usage:
    python scripts/adaptive_rr_model_v1.py
    python scripts/adaptive_rr_model_v1.py --versions 8 9 10
    python scripts/adaptive_rr_model_v1.py --max-rows 5000000
"""

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
    print("ERROR: LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        log_loss, roc_auc_score, brier_score_loss,
        precision_recall_curve, average_precision_score,
    )
    from sklearn.calibration import calibration_curve
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/ml")
OUTPUT_DIR = DATA_DIR / "adaptive_rr"

# 15 entry features from the sweep trade dicts
ENTRY_FEATURES = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points",
    "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side",
]

# Combo-level context features (categorical or boolean)
COMBO_FEATURES = [
    "stop_method",
    "exit_on_opposite_signal",
]

# The candidate R:R level is an additional feature
RR_FEATURE = "candidate_rr"

# Derived interaction features
DERIVED_FEATURES = [
    "abs_zscore_entry",   # |zscore_entry|
    "rr_x_atr",           # candidate_rr * atr_points (R:R scaled by volatility)
]

ALL_FEATURES = ENTRY_FEATURES + COMBO_FEATURES + [RR_FEATURE] + DERIVED_FEATURES

CATEGORICAL_COLS = ["stop_method", "side"]

# 17 candidate R:R levels from 1.0 to 5.0 in 0.25 steps
RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2).tolist()

# LightGBM hyperparameters — binary classification
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,        # more complex than combo-level model (many more rows)
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,  # higher floor for huge dataset
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 1000,
    "is_unbalance": True,     # handle win/loss imbalance at high R:R
}

# Sweep version → range_mode mapping
VERSION_RANGE_MODE = {
    2: "default",
    3: "zscore_variants",
    4: "v4", 5: "v5", 6: "v6", 7: "v7",
    8: "v8", 9: "v9", 10: "v10",
}

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive R:R model — P(win | features, candidate_rr)")
    p.add_argument("--versions", type=int, nargs="+",
                   default=list(range(2, 11)),
                   help="Sweep versions to include (default: 2-10)")
    p.add_argument("--max-rows", type=int, default=10_000_000,
                   help="Max rows after R:R expansion (default: 10M)")
    p.add_argument("--min-trades-per-combo", type=int, default=30,
                   help="Skip combos with fewer trades (default: 30)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="CV folds (default: 5)")
    p.add_argument("--skip-expansion", action="store_true",
                   help="Reuse cached expanded dataset")
    return p.parse_args()


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_mfe_parquets(versions: list[int]) -> pd.DataFrame:
    """Load _mfe.parquet files and concatenate."""
    frames = []
    for v in versions:
        path = DATA_DIR / "mfe" / f"ml_dataset_v{v}_mfe.parquet"
        if not path.exists():
            print(f"  WARN: {path} not found, skipping v{v}")
            continue
        print(f"  Loading v{v}...", end="", flush=True)
        df = pd.read_parquet(path)
        df["sweep_version"] = v
        # Add global combo ID
        df["global_combo_id"] = df["combo_id"].apply(lambda cid: f"v{v}_{cid}")
        print(f" {len(df):,} trades, {df['combo_id'].nunique()} combos")
        frames.append(df)

    if not frames:
        print("ERROR: No _mfe.parquet files found. Run sweeps first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} trades, "
          f"{combined['global_combo_id'].nunique()} combos")
    return combined


def filter_combos(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Remove combos with too few trades for reliable labels."""
    combo_counts = df.groupby("global_combo_id").size()
    valid_combos = combo_counts[combo_counts >= min_trades].index
    before = df["global_combo_id"].nunique()
    df = df[df["global_combo_id"].isin(valid_combos)].copy()
    after = df["global_combo_id"].nunique()
    print(f"  Filtered combos: {before} -> {after} "
          f"(removed {before - after} with <{min_trades} trades)")
    print(f"  Filtered trades: {len(df):,}")
    return df


# ── Synthetic Label Expansion ────────────────────────────────────────────────

def expand_rr_levels(df: pd.DataFrame, rr_levels: list[float],
                     max_rows: int) -> pd.DataFrame:
    """
    For each trade, create 17 rows (one per R:R level) with synthetic label:
      would_win = (mfe_points >= candidate_rr * stop_distance_pts)

    This answers: "If we had set the take-profit at candidate_rr × risk,
    would the price have reached it before the trade ended?"
    """
    print(f"\n[expand] Expanding {len(df):,} trades × {len(rr_levels)} R:R levels...")

    # Pre-compute: only keep needed columns to save memory
    keep_cols = (ENTRY_FEATURES + COMBO_FEATURES +
                 ["mfe_points", "mae_points", "stop_distance_pts",
                  "hold_bars", "global_combo_id"])
    df_slim = df[keep_cols].copy()

    # Subsample base trades if expansion would exceed max_rows
    n_base = len(df_slim)
    n_expanded = n_base * len(rr_levels)
    if n_expanded > max_rows:
        sample_n = max_rows // len(rr_levels)
        print(f"  Subsampling: {n_base:,} -> {sample_n:,} base trades "
              f"(expanded would be {n_expanded:,} > {max_rows:,})")
        df_slim = df_slim.sample(n=sample_n, random_state=42).reset_index(drop=True)
        n_base = len(df_slim)

    # Vectorized expansion: repeat each trade row len(rr_levels) times
    rr_arr = np.array(rr_levels, dtype=np.float64)
    n_rr = len(rr_arr)

    # Tile the dataframe
    expanded = pd.DataFrame(
        np.repeat(df_slim.values, n_rr, axis=0),
        columns=df_slim.columns,
    )
    # Assign candidate_rr
    expanded[RR_FEATURE] = np.tile(rr_arr, n_base)

    # Ensure numeric types survived the repeat
    for col in ["mfe_points", "stop_distance_pts", RR_FEATURE]:
        expanded[col] = pd.to_numeric(expanded[col], errors="coerce")

    # Synthetic label: would the trade have hit this R:R target?
    target_pts = expanded[RR_FEATURE] * expanded["stop_distance_pts"]
    expanded["would_win"] = (expanded["mfe_points"] >= target_pts).astype(np.int8)

    # Derived features
    expanded["abs_zscore_entry"] = pd.to_numeric(
        expanded["zscore_entry"], errors="coerce").abs()
    expanded["rr_x_atr"] = (
        expanded[RR_FEATURE] *
        pd.to_numeric(expanded["atr_points"], errors="coerce")
    )

    # Ensure all numeric columns are proper types
    numeric_cols = [c for c in ENTRY_FEATURES if c not in CATEGORICAL_COLS + ["side"]]
    for col in numeric_cols:
        if col in expanded.columns:
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce")

    print(f"  Expanded dataset: {len(expanded):,} rows")
    print(f"  Win rate by R:R level:")
    wr_by_rr = expanded.groupby(RR_FEATURE)["would_win"].mean()
    for rr, wr in wr_by_rr.items():
        print(f"    R:R {rr:.2f}: {wr:.1%}")

    return expanded


# ── Model Training ───────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, n_folds: int) -> dict:
    """Train LightGBM binary classifier with stratified CV."""
    print(f"\n[train] Training on {len(df):,} rows, {n_folds}-fold CV...")

    # Prepare features
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["would_win"].values.astype(np.int32)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Ensure boolean columns are int
    for col in ["exit_on_opposite_signal"]:
        if col in X.columns:
            X[col] = X[col].astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    fold_metrics = []
    models = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train,
                                 categorical_feature=CATEGORICAL_COLS,
                                 free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val,
                               categorical_feature=CATEGORICAL_COLS,
                               free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            LGB_PARAMS,
            train_data,
            num_boost_round=LGB_PARAMS["n_estimators"],
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        ll = log_loss(y_val, val_preds)
        auc = roc_auc_score(y_val, val_preds)
        brier = brier_score_loss(y_val, val_preds)

        fold_metrics.append({
            "fold": fold_i,
            "log_loss": ll,
            "auc": auc,
            "brier": brier,
            "best_iter": model.best_iteration,
            "time_s": time.time() - t0,
        })
        models.append(model)
        print(f"  Fold {fold_i}: log_loss={ll:.4f}  AUC={auc:.4f}  "
              f"brier={brier:.4f}  iters={model.best_iteration}  "
              f"({time.time()-t0:.0f}s)")

    # Overall OOF metrics
    overall_ll = log_loss(y, oof_preds)
    overall_auc = roc_auc_score(y, oof_preds)
    overall_brier = brier_score_loss(y, oof_preds)
    print(f"\n  OOF: log_loss={overall_ll:.4f}  AUC={overall_auc:.4f}  "
          f"brier={overall_brier:.4f}")

    return {
        "models": models,
        "oof_preds": oof_preds,
        "y_true": y,
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "overall": {
            "log_loss": overall_ll,
            "auc": overall_auc,
            "brier": overall_brier,
        },
        "X": X,
    }


# ── Optimal R:R Selection ───────────────────────────────────────────────────

def compute_optimal_rr(df: pd.DataFrame, oof_preds: np.ndarray,
                       rr_levels: list[float]) -> pd.DataFrame:
    """
    For each trade, find the R:R that maximizes expected value:
      E[R] = P(win) * R - (1 - P(win)) * 1
    where R is the candidate R:R and P(win) is the model's prediction.
    """
    print("\n[optimal] Computing optimal R:R per trade...")

    n_rr = len(rr_levels)
    n_trades = len(df) // n_rr

    # Reshape predictions: (n_trades, n_rr)
    preds_2d = oof_preds.reshape(n_trades, n_rr)
    rr_arr = np.array(rr_levels)

    # E[R] = P(win) * R - (1 - P(win))
    ev_2d = preds_2d * rr_arr[None, :] - (1 - preds_2d)

    # Best R:R per trade
    best_idx = np.argmax(ev_2d, axis=1)
    best_rr = rr_arr[best_idx]
    best_ev = ev_2d[np.arange(n_trades), best_idx]
    best_pwin = preds_2d[np.arange(n_trades), best_idx]

    results = pd.DataFrame({
        "optimal_rr": best_rr,
        "best_ev": best_ev,
        "best_pwin": best_pwin,
    })

    print(f"  Optimal R:R distribution:")
    vc = results["optimal_rr"].value_counts().sort_index()
    for rr, cnt in vc.items():
        print(f"    R:R {rr:.2f}: {cnt:,} trades ({cnt/n_trades:.1%})")

    print(f"\n  Mean optimal R:R: {results['optimal_rr'].mean():.2f}")
    print(f"  Mean best E[R]:   {results['best_ev'].mean():.4f}")
    print(f"  Trades with E[R] > 0: {(results['best_ev'] > 0).sum():,} "
          f"({(results['best_ev'] > 0).mean():.1%})")

    return results


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                     output_dir: Path) -> None:
    """Reliability diagram — is P(win) well-calibrated?"""
    fig, ax = plt.subplots(figsize=(8, 6))
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=20)
    ax.plot(mean_pred, fraction_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted P(win)")
    ax.set_ylabel("Observed fraction of wins")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved calibration_curve.png")


def plot_pwin_by_rr(df: pd.DataFrame, oof_preds: np.ndarray,
                    rr_levels: list[float], output_dir: Path) -> None:
    """P(win) vs R:R level — should decrease monotonically."""
    n_rr = len(rr_levels)
    n_trades = len(df) // n_rr
    preds_2d = oof_preds.reshape(n_trades, n_rr)

    mean_pwin = preds_2d.mean(axis=0)
    actual_wr = df.groupby(RR_FEATURE)["would_win"].mean().values

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(rr_levels)
    ax.plot(x, mean_pwin, "o-", label="Predicted P(win)", color="blue")
    ax.plot(x, actual_wr, "s--", label="Actual win rate", color="green")
    ax.set_xlabel("Candidate R:R")
    ax.set_ylabel("Win probability")
    ax.set_title("Win Probability vs R:R Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pwin_vs_rr.png", dpi=150)
    plt.close(fig)
    print("  Saved pwin_vs_rr.png")


def plot_ev_by_rr(df: pd.DataFrame, oof_preds: np.ndarray,
                  rr_levels: list[float], output_dir: Path) -> None:
    """Expected value curve — where is the sweet spot?"""
    n_rr = len(rr_levels)
    n_trades = len(df) // n_rr
    preds_2d = oof_preds.reshape(n_trades, n_rr)
    rr_arr = np.array(rr_levels)
    ev_2d = preds_2d * rr_arr[None, :] - (1 - preds_2d)
    mean_ev = ev_2d.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(rr_arr, mean_ev, width=0.2, alpha=0.7, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Candidate R:R")
    ax.set_ylabel("Mean E[R]")
    ax.set_title("Expected Value by R:R Level (higher = better)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "ev_by_rr.png", dpi=150)
    plt.close(fig)
    print("  Saved ev_by_rr.png")


def plot_feature_importance(models: list, feature_cols: list[str],
                            output_dir: Path) -> None:
    """Average feature importance across folds."""
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.feature_importance(importance_type="gain")
    importances /= len(models)

    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_cols))

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[sorted_idx[:top_n]], align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Gain")
    ax.set_title("Feature Importance (Top 20)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved feature_importance.png")


def plot_optimal_rr_distribution(results: pd.DataFrame,
                                 output_dir: Path) -> None:
    """Distribution of optimal R:R selections."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of optimal R:R
    axes[0].hist(results["optimal_rr"], bins=len(RR_LEVELS), edgecolor="black",
                 alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Optimal R:R")
    axes[0].set_ylabel("Trade count")
    axes[0].set_title("Distribution of Optimal R:R")

    # Right: E[R] distribution
    axes[1].hist(results["best_ev"], bins=50, edgecolor="black",
                 alpha=0.7, color="forestgreen")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Best E[R]")
    axes[1].set_ylabel("Trade count")
    axes[1].set_title("Distribution of Best Expected Value")

    fig.tight_layout()
    fig.savefig(output_dir / "optimal_rr_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved optimal_rr_distribution.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Adaptive R:R Model — P(win | features, candidate_rr)")
    print("=" * 70)

    expanded_path = OUTPUT_DIR / "expanded_dataset.parquet"

    if args.skip_expansion and expanded_path.exists():
        print("\n[load] Loading cached expanded dataset...")
        expanded = pd.read_parquet(expanded_path)
        print(f"  Loaded: {len(expanded):,} rows")
    else:
        # Step 1: Load MFE parquets
        print(f"\n[load] Loading _mfe.parquet files for versions: {args.versions}")
        df = load_mfe_parquets(args.versions)

        # Step 2: Filter combos with too few trades
        df = filter_combos(df, args.min_trades_per_combo)

        # Step 3: Expand with synthetic labels at each R:R level
        expanded = expand_rr_levels(df, RR_LEVELS, args.max_rows)

        # Cache the expanded dataset
        print(f"\n[cache] Saving expanded dataset to {expanded_path}...")
        expanded.to_parquet(expanded_path, compression="snappy", index=False)
        print(f"  Saved: {expanded_path} ({expanded_path.stat().st_size / 1e9:.2f} GB)")

        del df

    # Step 4: Train model
    result = train_model(expanded, args.n_folds)

    # Step 5: Compute optimal R:R per trade
    optimal_results = compute_optimal_rr(
        expanded, result["oof_preds"], RR_LEVELS)

    # Step 6: Generate plots
    print("\n[plots] Generating visualizations...")
    plot_calibration(result["y_true"], result["oof_preds"], OUTPUT_DIR)
    plot_pwin_by_rr(expanded, result["oof_preds"], RR_LEVELS, OUTPUT_DIR)
    plot_ev_by_rr(expanded, result["oof_preds"], RR_LEVELS, OUTPUT_DIR)
    plot_feature_importance(result["models"], result["feature_cols"], OUTPUT_DIR)
    plot_optimal_rr_distribution(optimal_results, OUTPUT_DIR)

    # Step 7: Save model and metadata
    print("\n[save] Saving model and metadata...")

    # Save the best model (by AUC)
    best_fold = max(range(len(result["fold_metrics"])),
                    key=lambda i: result["fold_metrics"][i]["auc"])
    best_model = result["models"][best_fold]
    model_path = OUTPUT_DIR / "adaptive_rr_model.txt"
    best_model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Save OOF predictions + optimal R:R
    oof_df = pd.DataFrame({
        "oof_pwin": result["oof_preds"],
        "y_true": result["y_true"],
    })
    oof_path = OUTPUT_DIR / "oof_predictions.parquet"
    oof_df.to_parquet(oof_path, compression="snappy", index=False)

    # Save optimal R:R results
    opt_path = OUTPUT_DIR / "optimal_rr_results.parquet"
    optimal_results.to_parquet(opt_path, compression="snappy", index=False)

    # Save run metadata
    metadata = {
        "versions_used": args.versions,
        "max_rows": args.max_rows,
        "min_trades_per_combo": args.min_trades_per_combo,
        "n_folds": args.n_folds,
        "rr_levels": RR_LEVELS,
        "n_features": len(result["feature_cols"]),
        "features": result["feature_cols"],
        "total_rows": len(expanded),
        "overall_metrics": result["overall"],
        "fold_metrics": result["fold_metrics"],
        "optimal_rr_summary": {
            "mean": float(optimal_results["optimal_rr"].mean()),
            "median": float(optimal_results["optimal_rr"].median()),
            "std": float(optimal_results["optimal_rr"].std()),
            "pct_positive_ev": float((optimal_results["best_ev"] > 0).mean()),
        },
        "lgb_params": LGB_PARAMS,
        "best_fold": best_fold,
        "runtime_seconds": time.time() - t_start,
    }
    meta_path = OUTPUT_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"  Metadata saved: {meta_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  OOF Log-loss: {result['overall']['log_loss']:.4f}")
    print(f"  OOF AUC:      {result['overall']['auc']:.4f}")
    print(f"  OOF Brier:    {result['overall']['brier']:.4f}")
    print(f"  Mean optimal R:R: {optimal_results['optimal_rr'].mean():.2f}")
    print(f"  Trades with E[R] > 0: {(optimal_results['best_ev'] > 0).mean():.1%}")
    print(f"  Runtime: {(time.time()-t_start)/60:.1f} min")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
