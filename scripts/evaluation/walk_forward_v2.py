"""
B7 — Walk-forward validation for the V2 adaptive R:R signal.

Takes a sweep parquet generated on the 80% training partition (must carry
`entry_bar_idx` so each trade can be tagged with its calendar date). Trains
fresh V2-clone LightGBM models on sliding calendar windows and evaluates
on the next window. Tests temporal stationarity of the ML#2 signal.

Two split modes:

- **expanding**: training window grows each fold (fold 0 trains on window 0,
  fold 1 trains on windows 0–1, …). Standard "does the signal work?" test.
- **rolling**:   fixed-width training window slides forward (fold k trains
  on window k-W..k-1 and tests on window k). Answers "is the edge
  strongest recently?".

Both modes use the same V2 hyperparameters and feature set as
`scripts/models/adaptive_rr_model_v2.py`. No retraining of the live V2 model —
all folds train a fresh clone for analysis only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.data_loader import load_bars, split_train_test  # noqa: E402

# Feature set — must match V2 exactly.
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
COMBO_FEATURES = ["stop_method", "exit_on_opposite_signal"]
RR_FEATURE = "candidate_rr"
DERIVED_FEATURES = ["abs_zscore_entry", "rr_x_atr"]
ALL_FEATURES = ENTRY_FEATURES + COMBO_FEATURES + [RR_FEATURE] + DERIVED_FEATURES
CATEGORICAL_COLS = ["stop_method", "side"]
NUMERIC_FEATURE_COLS = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
]
RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2).tolist()

PARQUET_COLUMNS = [
    "combo_id",
    "mfe_points", "mae_points", "stop_distance_pts",
    "entry_bar_idx",
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
]

# Same hyperparams as V2, but reduce rounds for per-fold speed. 800 rounds
# at lr=0.02 reaches plateau on 1–2M row training sets (V2 hit plateau by
# ~1500 on 9.5M rows — smaller folds converge faster).
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
    p = argparse.ArgumentParser(description="B7 walk-forward for V2")
    p.add_argument("--parquet", type=Path,
                   default=REPO_ROOT / "data/ml/mfe/ml_dataset_v10_train_wf_mfe.parquet",
                   help="Training-partition sweep parquet with entry_bar_idx")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "data/ml/adaptive_rr_v2",
                   help="Where to write walk_forward_v2.json + plot")
    p.add_argument("--modes", nargs="+", choices=["expanding", "rolling"],
                   default=["expanding", "rolling"],
                   help="Walk-forward modes to run")
    p.add_argument("--rolling-window-years", type=int, default=2,
                   help="Rolling window size in years (default 2)")
    p.add_argument("--max-base-trades", type=int, default=400_000,
                   help="Cap base trades per fold train set before expansion")
    return p.parse_args()


def load_and_date(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        sys.exit(f"[b7] FATAL: parquet missing: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)
    have = {f.name for f in pf.schema_arrow}
    if "entry_bar_idx" not in have:
        sys.exit("[b7] FATAL: parquet missing entry_bar_idx — "
                 "regenerate with B7.1 patched param_sweep.py.")
    cols = [c for c in PARQUET_COLUMNS if c in have]
    df = pf.read(columns=cols).to_pandas()
    df = df.dropna(subset=["mfe_points", "stop_distance_pts"]).reset_index(drop=True)
    print(f"[b7] Loaded {len(df):,} trades from {parquet_path.name}")

    # Map entry_bar_idx → calendar date via the train partition of NQ_1min.
    print("[b7] Loading NQ_1min and mapping bar indices to dates...")
    bars = load_bars(REPO_ROOT / "data/NQ_1min.csv")
    train_bars, _ = split_train_test(bars, 0.8)
    times = train_bars["time"].to_numpy()
    idx = df["entry_bar_idx"].to_numpy()
    if idx.max() >= len(times):
        sys.exit(f"[b7] FATAL: entry_bar_idx {idx.max()} >= train bar count "
                 f"{len(times)} — parquet was generated on the wrong partition?")
    df["entry_time"] = times[idx]
    df["entry_year"] = pd.DatetimeIndex(df["entry_time"]).year
    print(f"[b7] Trade date range: {df['entry_time'].min()} → {df['entry_time'].max()}")
    print(f"[b7] Year counts:\n{df['entry_year'].value_counts().sort_index()}")
    return df


def expand(df: pd.DataFrame) -> pd.DataFrame:
    """R:R expansion — identical to adaptive_rr_model_v2.expand_rr_levels."""
    n_rr = len(RR_LEVELS)
    rr_arr = np.array(RR_LEVELS, dtype=np.float32)
    n_base = len(df)
    out: dict[str, np.ndarray] = {}
    for col in NUMERIC_FEATURE_COLS:
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
    ex = pd.DataFrame(out, copy=False)
    for col in CATEGORICAL_COLS:
        if col in ex.columns:
            ex[col] = ex[col].astype("category")
    return ex


def ece_20bin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    cap_base: int) -> dict:
    """Train V2-clone on train_df; evaluate on test_df."""
    if len(train_df) > cap_base:
        rng = np.random.default_rng(42)
        train_df = train_df.iloc[
            rng.choice(len(train_df), size=cap_base, replace=False)
        ].reset_index(drop=True)

    train_ex = expand(train_df)
    test_ex = expand(test_df)

    feat_cols = [c for c in ALL_FEATURES if c in train_ex.columns]
    X_tr = train_ex[feat_cols]
    y_tr = train_ex["would_win"].to_numpy(dtype=np.int8)
    X_te = test_ex[feat_cols]
    y_te = test_ex["would_win"].to_numpy(dtype=np.int8)

    lgb_train = lgb.Dataset(X_tr, label=y_tr,
                            categorical_feature=CATEGORICAL_COLS,
                            free_raw_data=True)
    model = lgb.train(LGB_PARAMS, lgb_train,
                      num_boost_round=LGB_PARAMS["n_estimators"])
    preds = model.predict(X_te, num_threads=4).astype(np.float32)

    return {
        "n_train_base": int(len(train_df)),
        "n_train_rows": int(len(X_tr)),
        "n_test_base": int(len(test_df)),
        "n_test_rows": int(len(X_te)),
        "auc": float(roc_auc_score(y_te, preds)),
        "log_loss": float(log_loss(y_te, preds)),
        "brier": float(brier_score_loss(y_te, preds)),
        "ece_20bin": ece_20bin(y_te, preds),
        "mean_y": float(y_te.mean()),
        "mean_pred": float(preds.mean()),
    }


def make_folds_expanding(years: list[int]) -> list[tuple[list[int], int]]:
    """Expanding: fold k trains on years[:k+1], tests on years[k+1]."""
    return [(years[:k + 1], years[k + 1]) for k in range(len(years) - 1)]


def make_folds_rolling(years: list[int], window: int) -> list[tuple[list[int], int]]:
    """Rolling: fold k trains on last `window` years, tests on years[k+window]."""
    out = []
    for k in range(len(years) - window):
        out.append((years[k:k + window], years[k + window]))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_and_date(args.parquet)

    years_sorted = sorted(df["entry_year"].unique())
    # Drop years with too few trades to be a meaningful test window.
    counts = df["entry_year"].value_counts().to_dict()
    usable = [y for y in years_sorted if counts.get(y, 0) >= 5_000]
    print(f"[b7] Years used (≥5000 trades each): {usable}")

    results: dict = {
        "script": "scripts/evaluation/walk_forward_v2.py",
        "parquet": str(args.parquet),
        "usable_years": usable,
        "year_counts": {int(y): int(counts[y]) for y in usable},
        "modes": {},
    }

    for mode in args.modes:
        if mode == "expanding":
            folds = make_folds_expanding(usable)
        else:
            folds = make_folds_rolling(usable, args.rolling_window_years)
        if not folds:
            print(f"[b7] mode={mode}: not enough years for any fold, skipping")
            continue

        fold_results = []
        print(f"\n[b7] === mode={mode}, {len(folds)} folds ===")
        for i, (train_years, test_year) in enumerate(folds):
            print(f"\n[b7] {mode} fold {i}: train={train_years} test={test_year}")
            train_df = df[df["entry_year"].isin(train_years)].reset_index(drop=True)
            test_df = df[df["entry_year"] == test_year].reset_index(drop=True)
            t_fold = time.time()
            m = train_and_eval(train_df, test_df, args.max_base_trades)
            m["fold"] = i
            m["train_years"] = list(train_years)
            m["test_year"] = int(test_year)
            m["time_s"] = time.time() - t_fold
            print(f"  AUC={m['auc']:.4f}  LogLoss={m['log_loss']:.4f}  "
                  f"Brier={m['brier']:.4f}  ECE={m['ece_20bin']:.4f}  "
                  f"mean_y={m['mean_y']:.3f}  mean_p={m['mean_pred']:.3f}  "
                  f"({m['time_s']:.0f}s)")
            fold_results.append(m)
        results["modes"][mode] = fold_results

    results["runtime_seconds"] = time.time() - t0

    # Plot AUC and ECE per fold per mode.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"expanding": "tab:blue", "rolling": "tab:orange"}
    for mode, folds in results["modes"].items():
        xs = [f["test_year"] for f in folds]
        axes[0].plot(xs, [f["auc"] for f in folds], "o-",
                     label=mode, color=colors.get(mode))
        axes[1].plot(xs, [f["ece_20bin"] for f in folds], "o-",
                     label=mode, color=colors.get(mode))
    axes[0].set_title("Walk-forward AUC (B7 folds — 800-round clones; trend only)")
    axes[0].set_xlabel("Test year")
    axes[0].set_ylabel("AUC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Walk-forward ECE (20 bin, raw predictions)")
    axes[1].set_xlabel("Test year")
    axes[1].set_ylabel("ECE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png_path = args.output_dir / "walk_forward_v2.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\n[b7] Saved {png_path}")

    out_path = args.output_dir / "walk_forward_v2.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[b7] Saved {out_path}")

    print("\n[b7] Summary (test-year AUC / ECE):")
    for mode, folds in results["modes"].items():
        row = "  ".join(
            f"{f['test_year']}:AUC={f['auc']:.3f}/ECE={f['ece_20bin']:.3f}"
            for f in folds
        )
        print(f"  {mode:9s}  {row}")


if __name__ == "__main__":
    main()
