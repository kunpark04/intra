"""
ml1_surrogate.py — LightGBM surrogate optimizer for MNQ strategy parameters.

Reads parameter sweep datasets (v2–v10), aggregates 80M+ trade rows into ~33K
combo-level performance metrics, trains LightGBM surrogate models to predict
performance from parameter settings, and extracts optimal parameter regions.

Key decisions are documented in tasks/ml_decisions.md.

Usage:
    python scripts/models/ml1_surrogate.py
    python scripts/models/ml1_surrogate.py --min-trades 50 --n-top 30
    python scripts/models/ml1_surrogate.py --skip-aggregation   # reuse cached combo_features.parquet
    python scripts/models/ml1_surrogate.py --versions 8 9 10    # only use specific versions

See tasks/ml_decisions.md for reasoning behind every design choice.
"""

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Dependency checks ────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────

STARTING_EQUITY = 50_000.0

# Sweep versions available and their data paths
DATA_DIR = Path("data/ml")
OUTPUT_DIR = DATA_DIR / "ml1_results"

# All parameter columns that LightGBM will use as features.
# These are the strategy hyperparameters (what we're optimizing).
PARAM_COLS = [
    # Core z-score / indicator params
    "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow",
    # Stop-loss configuration
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    # Risk / exit params
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop",
    "max_hold_bars", "zscore_confirmation",
    # Z-score variant formulation (see D9 in ml_decisions.md)
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    # V5+ filter params
    "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]

# Columns that are categorical (LightGBM handles these natively)
CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]

# Boolean columns — treat as integer 0/1
BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
]

# Default values for missing fields in older sweep versions.
# These represent what those combos ACTUALLY used (not imputation).
# See D9 in ml_decisions.md.
ZSCORE_DEFAULTS = {
    "z_input": "close",
    "z_anchor": "rolling_mean",
    "z_denom": "rolling_std",
    "z_type": "parametric",
    "z_window_2": 0,
    "z_window_2_weight": 0.0,
}

# V5 filter defaults (disabled = no filtering active)
V5_FILTER_DEFAULTS = {
    "volume_entry_threshold": 0.0,
    "vol_regime_lookback": 0,
    "vol_regime_min_pct": 0.0,
    "vol_regime_max_pct": 1.0,
    "session_filter_mode": 0,
    "tod_exit_hour": 0,
}

# Composite score weights (see D5 in ml_decisions.md)
DEFAULT_WEIGHTS = {
    "sharpe_ratio": 0.25,
    "total_return_pct": 0.25,
    "max_drawdown_pct": 0.20,   # inverted: lower drawdown = higher score
    "win_rate": 0.15,
    "n_trades": 0.15,
}

# LightGBM hyperparameters (see D11 in ml_decisions.md)
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

# Targets to model (see D7 in ml_decisions.md)
TARGETS = [
    "composite_score",
    "sharpe_ratio",
    "total_return_pct",
    "max_drawdown_pct",
    "win_rate",
]

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ML#1 combo-grain surrogate trainer."""
    p = argparse.ArgumentParser(description="LightGBM surrogate optimizer for strategy parameters")
    p.add_argument("--versions", nargs="+", type=int, default=list(range(2, 11)),
                   help="Sweep versions to include (default: 2–10)")
    p.add_argument("--min-trades", type=int, default=30,
                   help="Exclude combos with fewer trades (default: 30)")
    p.add_argument("--n-top", type=int, default=20,
                   help="Number of top combos to report (default: 20)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Cross-validation folds (default: 5)")
    p.add_argument("--skip-aggregation", action="store_true",
                   help="Reuse cached combo_features.parquet instead of re-aggregating")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory (default: data/ml/ml1_results)")
    p.add_argument("--surrogate-candidates", type=int, default=50_000,
                   help="Number of random candidates for surrogate search (default: 50000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    # Allow overriding composite weights via CLI
    p.add_argument("--w-sharpe", type=float, default=DEFAULT_WEIGHTS["sharpe_ratio"])
    p.add_argument("--w-return", type=float, default=DEFAULT_WEIGHTS["total_return_pct"])
    p.add_argument("--w-drawdown", type=float, default=DEFAULT_WEIGHTS["max_drawdown_pct"])
    p.add_argument("--w-winrate", type=float, default=DEFAULT_WEIGHTS["win_rate"])
    p.add_argument("--w-trades", type=float, default=DEFAULT_WEIGHTS["n_trades"])
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Combo-Level Aggregation
# Transforms 80M trade rows into ~33K combo rows with performance metrics.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_sharpe(pnls: np.ndarray) -> float:
    """
    Un-annualized trade-level Sharpe ratio: mean(pnl) / std(pnl).

    See D4 in ml_decisions.md: no timestamps available for daily grouping,
    and annualization cancels out for ranking purposes since all combos
    run on the same historical data.
    """
    if len(pnls) < 2:
        return 0.0
    std = np.std(pnls, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(pnls) / std)


def _compute_max_drawdown(pnls: np.ndarray, starting_equity: float) -> float:
    """
    Max peak-to-trough drawdown as a percentage.

    Builds a cumulative equity curve from trade PnLs, then finds the
    largest percentage drop from any peak to subsequent trough.
    """
    equity = starting_equity + np.cumsum(pnls)
    # Prepend starting equity so the first trade's drawdown is relative to start
    equity_full = np.concatenate([[starting_equity], equity])
    running_peak = np.maximum.accumulate(equity_full)
    # Avoid division by zero if peak is ever 0 (shouldn't happen with $50k start)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (running_peak - equity_full) / running_peak
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)
    return float(drawdowns.max() * 100)


def _compute_max_consecutive_losses(wins: np.ndarray) -> int:
    """
    Longest streak of consecutive losing trades.

    Uses a simple run-length approach: count consecutive 0s in the
    win/loss array.
    """
    if len(wins) == 0:
        return 0
    max_streak = 0
    current_streak = 0
    for w in wins:
        if w == 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def _compute_profit_factor(pnls: np.ndarray) -> float:
    """
    Sum of winning PnLs divided by absolute sum of losing PnLs.

    Returns inf if no losses (perfect combo), 0.0 if no wins.
    Capped at 100.0 to prevent extreme values from distorting analysis.
    """
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return 100.0  # cap instead of inf
    return min(float(wins / losses), 100.0)


def _aggregate_single_combo(pnls: np.ndarray, wins: np.ndarray,
                             r_multiples: np.ndarray) -> dict:
    """
    Compute all performance metrics for one combo from its trade arrays.

    Parameters
    ----------
    pnls : array of net_pnl_dollars for each trade
    wins : array of label_win (0 or 1) for each trade
    r_multiples : array of r_multiple for each trade

    Returns
    -------
    Dict of metric_name -> value
    """
    n = len(pnls)
    total_pnl = float(pnls.sum())

    return {
        "n_trades": n,
        "win_rate": float(wins.mean()) if n > 0 else 0.0,
        "total_return_pct": total_pnl / STARTING_EQUITY * 100,
        "total_return_dollars": total_pnl,
        "final_equity": STARTING_EQUITY + total_pnl,
        "sharpe_ratio": _compute_sharpe(pnls),
        "max_drawdown_pct": _compute_max_drawdown(pnls, STARTING_EQUITY),
        "profit_factor": _compute_profit_factor(pnls),
        "avg_r_multiple": float(np.mean(r_multiples)) if n > 0 else 0.0,
        "median_r_multiple": float(np.median(r_multiples)) if n > 0 else 0.0,
        "avg_trade_pnl": float(np.mean(pnls)) if n > 0 else 0.0,
        "std_trade_pnl": float(np.std(pnls, ddof=1)) if n > 1 else 0.0,
        "max_consecutive_losses": _compute_max_consecutive_losses(wins),
        "calmar_ratio": (
            (total_pnl / STARTING_EQUITY * 100) /
            max(_compute_max_drawdown(pnls, STARTING_EQUITY), 0.01)
        ),
    }


def build_combo_features(versions: list[int], min_trades: int) -> pd.DataFrame:
    """
    Load trade data from all sweep versions, aggregate to combo-level metrics,
    and merge with parameter settings from manifests.

    Memory strategy (see plan): read only 3 columns from each parquet
    (combo_id, net_pnl_dollars, label_win, r_multiple), accumulate per-combo
    arrays in a dict, then aggregate. Peak memory ~2-3 GB.

    Parameters
    ----------
    versions : list of sweep version numbers (e.g., [2, 3, ..., 10])
    min_trades : minimum trades per combo to include

    Returns
    -------
    DataFrame with one row per combo: parameter columns + performance metrics
    """
    print("[ml] Loading trade data from all sweep versions...", flush=True)

    # ── Phase 1: Accumulate per-combo trade data ──
    # Key = global combo ID (e.g., "v10_4523")
    # Value = dict of numpy-appendable lists
    combo_pnls: dict[str, list] = defaultdict(list)
    combo_wins: dict[str, list] = defaultdict(list)
    combo_rmul: dict[str, list] = defaultdict(list)

    cols_needed = ["combo_id", "net_pnl_dollars", "label_win", "r_multiple"]

    for v in versions:
        parquet_path = DATA_DIR / "originals" / f"ml_dataset_v{v}.parquet"
        if not parquet_path.exists():
            print(f"  [WARN] v{v} parquet not found, skipping.", flush=True)
            continue

        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(parquet_path))
        n_groups = pf.metadata.num_row_groups

        print(f"  v{v}: {pf.metadata.num_rows:>12,} rows, {n_groups} row groups", flush=True)

        # Read row group by row group to control memory
        for rg_idx in range(n_groups):
            chunk = pf.read_row_group(rg_idx, columns=cols_needed).to_pandas()
            for cid, grp in chunk.groupby("combo_id"):
                gid = f"v{v}_{int(cid)}"  # global unique ID (see D13)
                combo_pnls[gid].extend(grp["net_pnl_dollars"].tolist())
                combo_wins[gid].extend(grp["label_win"].tolist())
                combo_rmul[gid].extend(grp["r_multiple"].tolist())
            del chunk
        gc.collect()

    print(f"  Total unique combos with trades: {len(combo_pnls):,}", flush=True)

    # ── Phase 2: Aggregate each combo into performance metrics ──
    print("[ml] Computing combo-level performance metrics...", flush=True)
    rows = []
    for gid in sorted(combo_pnls.keys()):
        pnls = np.array(combo_pnls[gid], dtype=np.float64)
        wins = np.array(combo_wins[gid], dtype=np.int8)
        rmul = np.array(combo_rmul[gid], dtype=np.float64)

        if len(pnls) < min_trades:
            continue

        metrics = _aggregate_single_combo(pnls, wins, rmul)
        metrics["global_combo_id"] = gid
        rows.append(metrics)

    # Free the large accumulation dicts
    del combo_pnls, combo_wins, combo_rmul
    gc.collect()

    metrics_df = pd.DataFrame(rows)
    print(f"  Combos after min_trades={min_trades} filter: {len(metrics_df):,}", flush=True)

    # ── Phase 3: Load parameter settings from manifests ──
    print("[ml] Loading parameter settings from manifests...", flush=True)
    param_rows = []
    for v in versions:
        manifest_path = DATA_DIR / "originals" / f"ml_dataset_v{v}_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            if entry.get("status") != "completed":
                continue
            if entry.get("n_trades", 0) == 0:
                continue

            gid = f"v{v}_{entry['combo_id']}"
            row = {"global_combo_id": gid, "sweep_version": v}

            # Extract all parameter columns, applying defaults for missing fields
            for col in PARAM_COLS:
                if col in entry:
                    row[col] = entry[col]
                elif col in ZSCORE_DEFAULTS:
                    row[col] = ZSCORE_DEFAULTS[col]
                elif col in V5_FILTER_DEFAULTS:
                    row[col] = V5_FILTER_DEFAULTS[col]
                else:
                    row[col] = np.nan  # truly missing

            param_rows.append(row)

    params_df = pd.DataFrame(param_rows)
    print(f"  Parameter entries loaded: {len(params_df):,}", flush=True)

    # ── Phase 4: Merge metrics + parameters on global_combo_id ──
    df = metrics_df.merge(params_df, on="global_combo_id", how="inner")
    print(f"  Final merged dataset: {len(df):,} combos", flush=True)

    # Convert booleans to int (0/1) for LightGBM compatibility
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0).astype(int)

    # Convert categoricals to pandas category dtype (LightGBM native support)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Composite Score
# Rank-percentile normalization + weighted sum across objectives.
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute a single composite score combining multiple objectives.

    Each metric is normalized to [0, 1] using rank percentile (robust to
    outliers — see D5 in ml_decisions.md). Max drawdown is inverted so
    lower drawdown = higher score. The weighted sum produces the final score.

    Parameters
    ----------
    df : DataFrame with metric columns
    weights : dict mapping metric name -> weight (should sum to ~1.0)

    Returns
    -------
    Series of composite scores, one per combo
    """
    score = pd.Series(0.0, index=df.index)

    for metric, weight in weights.items():
        if metric not in df.columns:
            print(f"  [WARN] metric '{metric}' not in DataFrame, skipping.", flush=True)
            continue

        # Rank percentile: maps values to [0, 1] uniformly
        ranked = df[metric].rank(pct=True, na_option="bottom")

        # Invert drawdown: lower drawdown = higher rank
        if metric == "max_drawdown_pct":
            ranked = 1.0 - ranked

        score += weight * ranked

    return score


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LightGBM Training
# 5-fold CV with separate models per target metric.
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare the feature matrix (X) for LightGBM.

    - Selects only parameter columns (not metrics — those are targets)
    - Maintains categorical dtypes for native LightGBM handling
    - Returns (X_dataframe, list_of_feature_column_names)
    """
    feature_cols = [c for c in PARAM_COLS if c in df.columns]
    X = df[feature_cols].copy()
    return X, feature_cols


def train_surrogate_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train one LightGBM regressor per target using k-fold cross-validation.

    For each target:
    1. Split data into k folds
    2. Train on k-1 folds, validate on the held-out fold
    3. Record out-of-fold (OOF) predictions for every row
    4. Report per-fold and overall R² and RMSE

    See D6 and D7 in ml_decisions.md for CV and multi-target reasoning.

    Parameters
    ----------
    df : full DataFrame (features + targets)
    feature_cols : list of parameter column names
    targets : list of target metric names to model
    n_folds : number of CV folds
    seed : random seed for reproducibility

    Returns
    -------
    Dict mapping target_name -> {
        'model': LGBMRegressor (trained on full data),
        'cv_r2': list of per-fold R² scores,
        'cv_rmse': list of per-fold RMSE scores,
        'oof_predictions': array of out-of-fold predictions,
        'feature_importance': DataFrame with feature importance,
        'train_r2': list of per-fold training R² (for overfitting detection),
    }
    """
    X = df[feature_cols]
    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Identify categorical feature names for LightGBM
    cat_features = [c for c in feature_cols if c in CATEGORICAL_COLS]

    for target in targets:
        print(f"\n[ml] Training model for: {target}", flush=True)
        y = df[target].values

        oof_preds = np.zeros(len(y))
        fold_r2 = []
        fold_rmse = []
        fold_train_r2 = []
        importance_acc = np.zeros(len(feature_cols))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create LightGBM datasets with categorical feature support
            dtrain = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=cat_features,
                free_raw_data=False,
            )
            dval = lgb.Dataset(
                X_val, label=y_val,
                categorical_feature=cat_features,
                reference=dtrain,
                free_raw_data=False,
            )

            # Train with early stopping on validation set
            model = lgb.train(
                LGB_PARAMS,
                dtrain,
                num_boost_round=LGB_PARAMS["n_estimators"],
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            # Predictions
            val_pred = model.predict(X_val)
            train_pred = model.predict(X_train)
            oof_preds[val_idx] = val_pred

            # Metrics for this fold
            r2 = r2_score(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_r2 = r2_score(y_train, train_pred)

            fold_r2.append(r2)
            fold_rmse.append(rmse)
            fold_train_r2.append(train_r2)

            # Accumulate feature importance (gain-based)
            importance_acc += model.feature_importance(importance_type="gain")

            print(f"  Fold {fold_idx+1}/{n_folds}: "
                  f"val R²={r2:.4f}  train R²={train_r2:.4f}  "
                  f"RMSE={rmse:.4f}  trees={model.num_trees()}", flush=True)

        # Average importance across folds
        avg_importance = importance_acc / n_folds
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": avg_importance,
        }).sort_values("importance", ascending=False)

        # Overall OOF metrics
        overall_r2 = r2_score(y, oof_preds)
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))

        # Overfitting gap: how much better the model does on train vs val
        avg_train_r2 = np.mean(fold_train_r2)
        avg_val_r2 = np.mean(fold_r2)
        overfit_gap = avg_train_r2 - avg_val_r2

        print(f"  Overall OOF R²={overall_r2:.4f}  RMSE={overall_rmse:.4f}", flush=True)
        print(f"  Overfitting gap: train R²={avg_train_r2:.4f} - val R²={avg_val_r2:.4f} "
              f"= {overfit_gap:.4f}", flush=True)
        if overfit_gap > 0.2:
            print(f"  [WARN] Large overfitting gap ({overfit_gap:.3f} > 0.2) — "
                  f"consider stronger regularization.", flush=True)

        # Retrain final model on ALL data for deployment / surrogate search
        dtrain_full = lgb.Dataset(
            X, label=y,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        final_model = lgb.train(
            LGB_PARAMS,
            dtrain_full,
            num_boost_round=model.num_trees(),  # use best iteration count from CV
        )

        results[target] = {
            "model": final_model,
            "cv_r2": fold_r2,
            "cv_rmse": fold_rmse,
            "train_r2": fold_train_r2,
            "overall_r2": overall_r2,
            "overall_rmse": overall_rmse,
            "overfit_gap": overfit_gap,
            "oof_predictions": oof_preds,
            "feature_importance": importance_df,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Parameter Extraction
# Top-N combos, feature importance, PDP, and surrogate search.
# ══════════════════════════════════════════════════════════════════════════════

def extract_top_combos(
    df: pd.DataFrame,
    oof_preds: np.ndarray,
    n_top: int = 20,
) -> pd.DataFrame:
    """
    Rank combos by out-of-fold predicted composite score and return the top N.

    Includes both predicted and actual metrics so you can verify the model's
    predictions are trustworthy (predicted vs actual should be correlated).
    """
    df_ranked = df.copy()
    df_ranked["predicted_composite"] = oof_preds
    df_ranked = df_ranked.sort_values("predicted_composite", ascending=False)
    return df_ranked.head(n_top)


def plot_feature_importance(
    results: dict,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """
    Bar chart of feature importance for the composite_score model.
    Shows which parameters have the most influence on overall strategy quality.
    """
    if "composite_score" not in results:
        return

    imp = results["composite_score"]["feature_importance"].head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(imp)), imp["importance"].values, color="#2196F3")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["feature"].values)
    ax.invert_yaxis()  # highest importance at top
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance — Composite Score Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}", flush=True)


def plot_partial_dependence_charts(
    results: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
    top_n: int = 12,
) -> None:
    """
    Partial Dependence Plots for the top-N most important features.

    PDP shows the marginal effect of each parameter on the predicted
    composite score — i.e., "how does changing this parameter affect
    strategy quality, holding everything else constant?"

    Only plots numeric features (categorical PDP requires special handling).
    """
    if "composite_score" not in results:
        return

    model = results["composite_score"]["model"]
    imp = results["composite_score"]["feature_importance"]

    # Filter to numeric features only (PDP doesn't work well with categoricals)
    numeric_features = [f for f in imp["feature"].values
                        if f not in CATEGORICAL_COLS and f in feature_cols]
    features_to_plot = numeric_features[:top_n]

    if not features_to_plot:
        print("  [WARN] No numeric features for PDP.", flush=True)
        return

    # Prepare X for manual PDP computation.
    # sklearn's partial_dependence requires a fitted sklearn estimator, but we
    # have a raw LightGBM Booster. We compute PDP manually instead: for each
    # grid value of the feature, replace that column for ALL rows, predict,
    # and take the mean prediction. This is exactly what PDP does.
    X = df[feature_cols].copy()

    n_plots = len(features_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features_to_plot):
        col_vals = X[feat].dropna()
        grid = np.linspace(col_vals.min(), col_vals.max(), 50)
        pdp_values = []

        for val in grid:
            X_temp = X.copy()
            X_temp[feat] = val
            preds = model.predict(X_temp)
            pdp_values.append(preds.mean())

        ax = axes[i]
        ax.plot(grid, pdp_values, color="#2196F3", linewidth=2)
        ax.set_xlabel(feat)
        ax.set_ylabel("Partial Dependence")
        ax.set_title(feat, fontsize=10)

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Partial Dependence — Composite Score", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}", flush=True)


def search_surrogate(
    model,
    feature_cols: list[str],
    df: pd.DataFrame,
    n_candidates: int = 50_000,
    seed: int = 42,
    n_top: int = 50,
) -> pd.DataFrame:
    """
    Generate random parameter combos and score them with the trained surrogate.

    This explores parameter regions the original sweep may have missed.
    The surrogate evaluation is instant (~microseconds per prediction),
    so 50K candidates is trivial. See D10 in ml_decisions.md.

    Sampling strategy: for each parameter, sample uniformly within the
    observed min/max range in the training data. For categoricals, sample
    uniformly from observed categories.
    """
    rng = np.random.default_rng(seed)
    candidates = {}

    for col in feature_cols:
        if col in CATEGORICAL_COLS:
            # Sample from observed categories
            categories = df[col].cat.categories.tolist()
            candidates[col] = pd.Categorical(
                rng.choice(categories, size=n_candidates),
                categories=categories,
            )
        elif col in BOOLEAN_COLS:
            candidates[col] = rng.integers(0, 2, size=n_candidates)
        else:
            # Numeric: uniform between observed min and max
            col_min = df[col].min()
            col_max = df[col].max()
            if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                candidates[col] = np.full(n_candidates, col_min if not pd.isna(col_min) else 0)
            else:
                candidates[col] = rng.uniform(col_min, col_max, size=n_candidates)

    cand_df = pd.DataFrame(candidates)

    # Score all candidates with the surrogate model
    cand_df["predicted_composite"] = model.predict(cand_df[feature_cols])
    cand_df = cand_df.sort_values("predicted_composite", ascending=False)

    return cand_df.head(n_top)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Output & Reporting
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    df: pd.DataFrame,
    results: dict,
    top_combos: pd.DataFrame,
    surrogate_top: pd.DataFrame,
    weights: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Save all pipeline outputs to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combo features dataset (the aggregated ~33K row dataset)
    combo_path = output_dir / "combo_features.parquet"
    # Convert categoricals to strings for parquet compatibility
    df_save = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df_save.columns:
            df_save[col] = df_save[col].astype(str)
    df_save.to_parquet(combo_path, index=False)
    print(f"  Saved: {combo_path}", flush=True)

    # 2. Top combos CSV (the primary actionable output)
    top_path = output_dir / "top_combos.csv"
    top_save = top_combos.copy()
    for col in CATEGORICAL_COLS:
        if col in top_save.columns:
            top_save[col] = top_save[col].astype(str)
    top_save.to_csv(top_path, index=False)
    print(f"  Saved: {top_path}", flush=True)

    # 3. Surrogate search results
    surr_path = output_dir / "surrogate_top_combos.csv"
    surr_save = surrogate_top.copy()
    for col in CATEGORICAL_COLS:
        if col in surr_save.columns:
            surr_save[col] = surr_save[col].astype(str)
    surr_save.to_csv(surr_path, index=False)
    print(f"  Saved: {surr_path}", flush=True)

    # 4. CV results JSON
    cv_results = {}
    for target, res in results.items():
        cv_results[target] = {
            "overall_r2": res["overall_r2"],
            "overall_rmse": res["overall_rmse"],
            "overfit_gap": res["overfit_gap"],
            "fold_r2": res["cv_r2"],
            "fold_rmse": res["cv_rmse"],
            "fold_train_r2": res["train_r2"],
            "top_10_features": res["feature_importance"].head(10)[
                ["feature", "importance"]
            ].to_dict("records"),
        }
    cv_path = output_dir / "cv_results.json"
    cv_path.write_text(json.dumps(cv_results, indent=2, default=str))
    print(f"  Saved: {cv_path}", flush=True)

    # 5. Run metadata (for reproducibility)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions_used": args.versions,
        "min_trades": args.min_trades,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "composite_weights": weights,
        "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
        "total_combos": len(df),
        "surrogate_candidates": args.surrogate_candidates,
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"  Saved: {meta_path}", flush=True)

    # 6. Save LightGBM model files
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for target, res in results.items():
        model_path = models_dir / f"{target}.txt"
        res["model"].save_model(str(model_path))
    print(f"  Saved: {len(results)} model files to {models_dir}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Train the ML#1 surrogate model from per-combo aggregate features."""
    args = parse_args()
    t_start = time.time()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    # Build composite weight dict from CLI args
    weights = {
        "sharpe_ratio": args.w_sharpe,
        "total_return_pct": args.w_return,
        "max_drawdown_pct": args.w_drawdown,
        "win_rate": args.w_winrate,
        "n_trades": args.w_trades,
    }

    print("=" * 70)
    print("ML Parameter Optimizer — LightGBM Surrogate")
    print("=" * 70)
    print(f"  Versions: {args.versions}")
    print(f"  Min trades: {args.min_trades}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Weights: {weights}")
    print(f"  Seed: {args.seed}")
    print()

    # ── Step 1: Build or load combo features ──
    cached_path = OUTPUT_DIR / "combo_features.parquet"
    if args.skip_aggregation and cached_path.exists():
        print("[ml] Loading cached combo_features.parquet...", flush=True)
        df = pd.read_parquet(cached_path)
        # Restore categorical dtypes
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        print(f"  Loaded {len(df):,} combos from cache.", flush=True)
    else:
        df = build_combo_features(args.versions, args.min_trades)

    # ── Step 2: Compute composite score ──
    print("\n[ml] Computing composite scores...", flush=True)
    df["composite_score"] = compute_composite_score(df, weights)
    print(f"  Composite score range: [{df['composite_score'].min():.4f}, "
          f"{df['composite_score'].max():.4f}]", flush=True)

    # ── Step 3: Prepare features ──
    X, feature_cols = prepare_features(df)
    print(f"\n[ml] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features", flush=True)

    # ── Step 4: Train surrogate models ──
    results = train_surrogate_models(df, feature_cols, TARGETS, args.n_folds, args.seed)

    # ── Step 5: Extract top combos from OOF predictions ──
    print(f"\n[ml] Extracting top {args.n_top} combos...", flush=True)
    oof_composite = results["composite_score"]["oof_predictions"]
    top_combos = extract_top_combos(df, oof_composite, args.n_top)

    # Print top combos summary
    print(f"\n{'='*70}")
    print(f"TOP {args.n_top} COMBOS BY PREDICTED COMPOSITE SCORE")
    print(f"{'='*70}")
    summary_cols = [
        "global_combo_id", "predicted_composite", "composite_score",
        "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
        "win_rate", "n_trades",
    ]
    print(top_combos[summary_cols].to_string(index=False))

    # ── Step 6: Surrogate search for novel optimal combos ──
    print(f"\n[ml] Surrogate search: scoring {args.surrogate_candidates:,} random candidates...",
          flush=True)
    composite_model = results["composite_score"]["model"]
    surrogate_top = search_surrogate(
        composite_model, feature_cols, df,
        n_candidates=args.surrogate_candidates,
        seed=args.seed,
    )
    print(f"  Top surrogate candidate score: {surrogate_top['predicted_composite'].iloc[0]:.4f}")

    # ── Step 7: Generate plots ──
    print("\n[ml] Generating plots...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(results, OUTPUT_DIR / "feature_importance.png")
    plot_partial_dependence_charts(
        results, df, feature_cols, OUTPUT_DIR / "partial_dependence.png"
    )

    # ── Step 8: Save all outputs ──
    print("\n[ml] Saving results...", flush=True)
    save_results(df, results, top_combos, surrogate_top, weights, args, OUTPUT_DIR)

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Combo dataset:    {len(df):,} combos")
    print(f"  Top combos:       {OUTPUT_DIR / 'top_combos.csv'}")
    print(f"  Surrogate top:    {OUTPUT_DIR / 'surrogate_top_combos.csv'}")
    print()
    for target, res in results.items():
        print(f"  {target:25s}  R²={res['overall_r2']:.4f}  "
              f"gap={res['overfit_gap']:.4f}")


if __name__ == "__main__":
    main()
