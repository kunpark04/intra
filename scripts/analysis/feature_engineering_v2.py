"""
B8 — Feature engineering ablation for ML#2 (adaptive R:R V2).

Tests whether three families of temporal features lift CV AUC above V2's
0.806 baseline on the B7 500-combo training-partition parquet. Runs the
same 5-fold StratifiedGroupKFold scheme as V2 (groups=combo_id) so deltas
are directly comparable across configurations.

Feature families:
  A — autocorr:  prior_wr_10, prior_wr_50, prior_r_ma10, has_history_50
  B — recency:   bars_since_last_trade, log1p_bars_since_last_trade
  C — regime:    atr_regime_rank_500, parkinson_regime_rank_500

Five configs are evaluated on identical folds:
  baseline · +A · +B · +C · +ABC

Decision gate: adopt a family only if OOF AUC lifts ≥ +0.005 vs baseline.

All rolling features use `groupby(combo_id).shift(1).rolling(N)` — no
trade sees its own outcome or any post-entry information.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---- V2-compatible feature lists (matches adaptive_rr_model_v2.py) -----

V2_ENTRY_FEATURES = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points",
    "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side",
]
V2_COMBO_FEATURES = ["stop_method", "exit_on_opposite_signal"]
RR_FEATURE = "candidate_rr"
V2_DERIVED_FEATURES = ["abs_zscore_entry", "rr_x_atr"]

CATEGORICAL_COLS = ["stop_method", "side"]

# Numeric features that get np.repeat during R:R expansion. Includes V2
# base numerics PLUS any new B8 features we've computed at base-trade level.
V2_NUMERIC = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
]

# New feature families.
FAMILY_A = ["prior_wr_10", "prior_wr_50", "prior_r_ma10", "has_history_50"]
FAMILY_B = ["bars_since_last_trade", "log1p_bars_since_last_trade"]
FAMILY_C = ["atr_regime_rank_500", "parkinson_regime_rank_500"]

RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2).tolist()

PARQUET_COLUMNS = [
    "combo_id",
    "mfe_points", "mae_points", "stop_distance_pts",
    "entry_bar_idx",
    "label_win", "r_multiple",
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
]

# V2 hyperparams, 800 rounds (matches B7 walk-forward).
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
    "n_estimators": 800,
    "num_threads": 4,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the B8 V2 feature-engineering ablation.

    Returns:
        `argparse.Namespace` with feature-family flags and output paths.
    """
    p = argparse.ArgumentParser(description="B8 feature-engineering ablation")
    p.add_argument("--parquet", type=Path,
                   default=REPO_ROOT / "data/ml/mfe/ml_dataset_v10_train_wf_mfe.parquet")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "data/ml/adaptive_rr_v2")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--configs", nargs="+",
                   default=["baseline", "A", "B", "C", "ABC"],
                   help="Configurations to run")
    p.add_argument("--max-base-trades", type=int, default=0,
                   help="Subsample base trades (0 = use all ~1.18M)")
    return p.parse_args()


# ---------------------------- Feature engineering ----------------------------

def _assert_sorted(df: pd.DataFrame) -> None:
    """Every combo's entry_bar_idx must be non-decreasing for rolling to
    be chronologically valid. Cheap check; fail fast if upstream changed."""
    monotone = df.groupby("combo_id", sort=False, observed=True)["entry_bar_idx"] \
                 .apply(lambda s: s.is_monotonic_increasing).all()
    if not monotone:
        raise RuntimeError("df must be sorted by (combo_id, entry_bar_idx) "
                           "before feature engineering")


def add_family_a(df: pd.DataFrame) -> pd.DataFrame:
    """Autocorrelation — per-combo rolling prior WR and prior realised R.

    All computed on values *shifted by one trade* within each combo, so the
    current trade's outcome never enters its own feature.
    """
    _assert_sorted(df)
    g = df.groupby("combo_id", sort=False, observed=True)
    win_prev = g["label_win"].shift(1)
    r_prev = g["r_multiple"].shift(1)

    # Rolling means of the shifted series, still grouped by combo.
    gp_win = win_prev.groupby(df["combo_id"], sort=False, observed=True)
    gp_r = r_prev.groupby(df["combo_id"], sort=False, observed=True)

    df["prior_wr_10"] = gp_win.transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    ).astype(np.float32)
    df["prior_wr_50"] = gp_win.transform(
        lambda s: s.rolling(50, min_periods=10).mean()
    ).astype(np.float32)
    df["prior_r_ma10"] = gp_r.transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    ).astype(np.float32)

    # Within-combo position; flag when < 25 prior trades exist.
    combo_rank = g.cumcount()
    df["has_history_50"] = (combo_rank >= 25).astype(np.int8)

    # NaN fill for early trades (shift(1) + min_periods>=3 leaves the first
    # few rows per combo NaN). Use neutral constants so LGBM doesn't see
    # NaN; has_history_50 acts as the "no real history" flag.
    df["prior_wr_10"] = df["prior_wr_10"].fillna(0.5).astype(np.float32)
    df["prior_wr_50"] = df["prior_wr_50"].fillna(0.5).astype(np.float32)
    df["prior_r_ma10"] = df["prior_r_ma10"].fillna(0.0).astype(np.float32)
    return df


def add_family_b(df: pd.DataFrame) -> pd.DataFrame:
    """Recency — bar gap to previous trade within the same combo."""
    _assert_sorted(df)
    g = df.groupby("combo_id", sort=False, observed=True)
    diff = g["entry_bar_idx"].diff()
    # First trade in combo → no predecessor; fill with large sentinel so
    # LGBM learns "first trade" as a distinct regime.
    gap = diff.fillna(-1).astype(np.int64)
    df["bars_since_last_trade"] = gap.astype(np.float32)
    # log1p with sign protection for the -1 sentinel.
    df["log1p_bars_since_last_trade"] = np.where(
        gap < 0, -1.0, np.log1p(gap.clip(lower=0).to_numpy())
    ).astype(np.float32)
    return df


def add_family_c(df: pd.DataFrame) -> pd.DataFrame:
    """Regime — within-combo rolling percentile of atr and parkinson vol."""
    _assert_sorted(df)

    def rolling_rank_pct(s: pd.Series, window: int = 500) -> pd.Series:
        """Rolling-window percentile rank of a series.

        Args:
            s: 1-D numeric series.
            window: Lookback length in bars.

        Returns:
            Series of rank percentiles in `[0, 1]`; first `window-1` values NaN.
        """
        # Current value's percentile rank among previous `window` values
        # (shift 1 to exclude self). Implemented via rolling().apply.
        shifted = s.shift(1)
        def _rank(x: np.ndarray) -> float:
            """Compute the percentile rank of the last element within a window array.

            Args:
                a: 1-D array of length `window`.

            Returns:
                Fraction of `a` strictly less than `a[-1]`.
            """
            if len(x) == 0:
                return 0.5
            v = x[-1]  # this is the shifted-current value inside the window
            return float((x < v).sum() / max(len(x), 1))
        # We want: for each row i, the rank of s[i] among s[i-window:i].
        # Easier: compare s to its rolling(window).rank via expanding scheme.
        # Use rolling quantile approximation: rank = (rolling rank / count).
        roll = shifted.rolling(window, min_periods=30)
        # Rank within window via .apply is O(N*W) — use numeric proxy:
        # (current_value - rolling_mean) / rolling_std gives a z-score;
        # convert to percentile via normal CDF.
        from scipy.stats import norm
        mu = roll.mean()
        sd = roll.std(ddof=0)
        z = (s - mu) / sd.replace(0.0, np.nan)
        pct = pd.Series(norm.cdf(z.to_numpy()), index=s.index, dtype=np.float32)
        return pct.fillna(0.5).astype(np.float32)

    g = df.groupby("combo_id", sort=False, observed=True)
    df["atr_regime_rank_500"] = g["atr_points"].transform(rolling_rank_pct)
    df["parkinson_regime_rank_500"] = g["parkinson_vol_pct"].transform(rolling_rank_pct)
    return df


# ---------------------------- Label expansion ----------------------------

def expand(df: pd.DataFrame, extra_numeric: list[str]) -> pd.DataFrame:
    """Expand each base trade into 17 R:R rows.

    Identical to adaptive_rr_model_v2.expand_rr_levels, but accepts a list
    of extra numeric columns to np.repeat alongside the V2 base numerics.
    """
    n_rr = len(RR_LEVELS)
    rr_arr = np.array(RR_LEVELS, dtype=np.float32)
    n_base = len(df)
    out: dict[str, np.ndarray] = {}

    numeric = list(V2_NUMERIC) + [c for c in extra_numeric if c in df.columns]
    for col in numeric:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(dtype=np.float32), n_rr)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(), n_rr)

    if "exit_on_opposite_signal" in df.columns:
        out["exit_on_opposite_signal"] = np.repeat(
            df["exit_on_opposite_signal"].to_numpy(dtype=np.int8), n_rr)

    out[RR_FEATURE] = np.tile(rr_arr, n_base)
    mfe = df["mfe_points"].to_numpy(dtype=np.float32)
    stop = df["stop_distance_pts"].to_numpy(dtype=np.float32)
    out["would_win"] = (
        (mfe[:, None] >= rr_arr[None, :] * stop[:, None]).ravel().astype(np.int8)
    )

    if "zscore_entry" in out:
        out["abs_zscore_entry"] = np.abs(out["zscore_entry"])
    if "atr_points" in out:
        out["rr_x_atr"] = out[RR_FEATURE] * out["atr_points"]

    # Group label for StratifiedGroupKFold.
    out["__combo_id__"] = np.repeat(df["combo_id"].to_numpy(), n_rr)

    ex = pd.DataFrame(out, copy=False)
    for col in CATEGORICAL_COLS:
        if col in ex.columns:
            ex[col] = ex[col].astype("category")
    return ex


def ece_20bin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Equal-width 20-bin Expected Calibration Error.

    Args:
        y: Binary outcomes.
        p: Predicted probabilities.

    Returns:
        ECE as a float.
    """
    bins = np.linspace(0.0, 1.0, 21)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, 19)
    ece = 0.0
    n = len(y_pred)
    for b in range(20):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(y_pred[m].mean() - y_true[m].mean())
    return float(ece)


# ---------------------------- Config runner ----------------------------

def build_extra(config: str) -> list[str]:
    """Build candidate extra features (Family A, B, C) on top of V2 base.

    Args:
        df: Base per-trade feature frame.

    Returns:
        Copy of `df` with the extra feature columns added.
    """
    extra = []
    if "A" in config:
        extra += FAMILY_A
    if "B" in config:
        extra += FAMILY_B
    if "C" in config:
        extra += FAMILY_C
    return extra


def run_config(df_base: pd.DataFrame, config: str, n_folds: int, seed: int = 42) -> dict:
    """Run 5-fold CV for one configuration on pre-engineered base df."""
    extra = build_extra(config)
    ex = expand(df_base, extra)

    # Feature columns for this config.
    base_feats = V2_ENTRY_FEATURES + V2_COMBO_FEATURES + [RR_FEATURE] + V2_DERIVED_FEATURES
    feat_cols = base_feats + [c for c in extra if c in ex.columns]

    X = ex[feat_cols]
    y = ex["would_win"].to_numpy(dtype=np.int8)
    groups = ex["__combo_id__"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []
    feat_gain = np.zeros(len(feat_cols), dtype=np.float64)

    for fi, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        t0 = time.time()
        lgb_train = lgb.Dataset(X.iloc[tr], label=y[tr],
                                categorical_feature=CATEGORICAL_COLS,
                                free_raw_data=True)
        model = lgb.train(LGB_PARAMS, lgb_train,
                          num_boost_round=LGB_PARAMS["n_estimators"])
        p = model.predict(X.iloc[va], num_threads=4).astype(np.float32)
        oof_pred[va] = p
        fm = {
            "fold": fi,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "auc": float(roc_auc_score(y[va], p)),
            "log_loss": float(log_loss(y[va], p)),
            "brier": float(brier_score_loss(y[va], p)),
            "ece_20bin": ece_20bin(y[va], p),
            "time_s": time.time() - t0,
        }
        fold_metrics.append(fm)
        gains = model.feature_importance(importance_type="gain")
        feat_gain += gains
        print(f"  [{config}] fold {fi}: AUC={fm['auc']:.4f} ECE={fm['ece_20bin']:.4f} "
              f"({fm['time_s']:.0f}s)")

    oof = {
        "auc": float(roc_auc_score(y, oof_pred)),
        "log_loss": float(log_loss(y, oof_pred)),
        "brier": float(brier_score_loss(y, oof_pred)),
        "ece_20bin": ece_20bin(y, oof_pred),
        "mean_y": float(y.mean()),
        "mean_pred": float(oof_pred.mean()),
    }
    top_gain = sorted(zip(feat_cols, (feat_gain / n_folds).tolist()),
                      key=lambda kv: -kv[1])[:15]
    result = {
        "config": config,
        "features_added": extra,
        "n_features": len(feat_cols),
        "oof": oof,
        "folds": fold_metrics,
        "top_feature_gain": top_gain,
    }
    del ex, X, y, groups, oof_pred, lgb_train, model
    gc.collect()
    return result


# ---------------------------- Main ----------------------------

def main() -> None:
    """B8: V2 feature-engineering ablation.

    Trains the adaptive R:R model under each feature-family subset, reports
    AUC/ECE deltas vs V2 baseline, and writes JSON + CSV.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if not args.parquet.exists():
        sys.exit(f"[b8] FATAL: parquet missing: {args.parquet}")

    pf = pq.ParquetFile(args.parquet)
    have = {f.name for f in pf.schema_arrow}
    cols = [c for c in PARQUET_COLUMNS if c in have]
    print(f"[b8] Loading {args.parquet.name}, {pf.metadata.num_rows:,} rows")
    df = pf.read(columns=cols).to_pandas()
    df = df.dropna(subset=["mfe_points", "stop_distance_pts"]).reset_index(drop=True)

    if args.max_base_trades and len(df) > args.max_base_trades:
        rng = np.random.default_rng(42)
        df = df.iloc[rng.choice(len(df), size=args.max_base_trades,
                                 replace=False)].reset_index(drop=True)
        print(f"[b8] Subsampled to {len(df):,} base trades")

    # Sort per-combo temporally — REQUIRED for autocorr/recency features.
    df = df.sort_values(["combo_id", "entry_bar_idx"]).reset_index(drop=True)
    print(f"[b8] Sorted by (combo_id, entry_bar_idx). "
          f"{df['combo_id'].nunique():,} unique combos.")

    # Compute feature families once, up-front. run_config picks which to expand.
    print("[b8] Computing family A (autocorr)...")
    df = add_family_a(df)
    print("[b8] Computing family B (recency)...")
    df = add_family_b(df)
    print("[b8] Computing family C (regime ranks)...")
    df = add_family_c(df)
    print(f"[b8] Base df ready — {len(df):,} trades, "
          f"{df.shape[1]} cols, mem={df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    results = {
        "script": "scripts/analysis/feature_engineering_v2.py",
        "parquet": str(args.parquet),
        "n_base_trades": int(len(df)),
        "n_combos": int(df["combo_id"].nunique()),
        "n_folds": args.n_folds,
        "configs": {},
    }

    for cfg in args.configs:
        print(f"\n[b8] === config={cfg} ===")
        results["configs"][cfg] = run_config(df, cfg, args.n_folds)
        r = results["configs"][cfg]["oof"]
        print(f"  [{cfg}] OOF AUC={r['auc']:.4f} LogLoss={r['log_loss']:.4f} "
              f"Brier={r['brier']:.4f} ECE={r['ece_20bin']:.4f}")

    results["runtime_seconds"] = time.time() - t0

    # Delta table.
    if "baseline" in results["configs"]:
        base_auc = results["configs"]["baseline"]["oof"]["auc"]
        base_ece = results["configs"]["baseline"]["oof"]["ece_20bin"]
        deltas = {}
        for cfg, r in results["configs"].items():
            deltas[cfg] = {
                "auc_delta": r["oof"]["auc"] - base_auc,
                "ece_delta": r["oof"]["ece_20bin"] - base_ece,
                "passes_gate": (r["oof"]["auc"] - base_auc) >= 0.005,
            }
        results["deltas_vs_baseline"] = deltas

    out_path = args.output_dir / "feature_engineering_v2.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[b8] Saved {out_path}")

    # Summary.
    print("\n[b8] Summary (OOF AUC / ECE / Δauc vs baseline):")
    for cfg, r in results["configs"].items():
        d = results.get("deltas_vs_baseline", {}).get(cfg, {})
        delta = d.get("auc_delta", 0.0)
        gate = "ADOPT" if d.get("passes_gate") else "null"
        print(f"  {cfg:10s}  AUC={r['oof']['auc']:.4f}  "
              f"ECE={r['oof']['ece_20bin']:.4f}  Δ={delta:+.4f}  {gate}")


if __name__ == "__main__":
    main()
