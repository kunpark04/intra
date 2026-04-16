"""
B6 — Temporal out-of-distribution evaluation of the V2 adaptive R:R model.

Loads the V2 LightGBM booster trained on the 80% training partition, applies
it to trade rows generated on the held-out 20% test partition (produced by
`param_sweep.py --eval-partition test`), and reports AUC / log-loss / Brier /
ECE alongside the training-split OOF metrics from `run_metadata.json`.

Purpose:
- Detect temporal drift in calibration or discrimination.
- Gate follow-on work (B16 final held-out) on a pass.

The V2 cross-validation was StratifiedGroupKFold on `global_combo_id` — so
combos didn't leak across folds, but bars did. This test is the first one
that holds out *time* from V2's training data.
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

# Feature set must match V2 exactly.
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
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the B6 temporal-OOD V2 evaluator.

    Returns:
        `argparse.Namespace` with holdout fraction and output paths.
    """
    p = argparse.ArgumentParser(description="B6 temporal-OOD eval for V2")
    p.add_argument("--test-parquet", type=Path,
                   default=REPO_ROOT / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet",
                   help="Test-partition MFE parquet produced by param_sweep.py "
                        "--eval-partition test")
    p.add_argument("--model", type=Path,
                   default=REPO_ROOT / "data/ml/adaptive_rr_v2/adaptive_rr_model.txt",
                   help="V2 LightGBM model file")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "data/ml/adaptive_rr_v2",
                   help="Where to write heldout_time_eval_v2.json + reliability PNG")
    p.add_argument("--max-base-trades", type=int, default=500_000,
                   help="Cap base trades before R:R expansion (default: 500k "
                        "→ 8.5M expanded rows)")
    return p.parse_args()


def load_test_trades(path: Path, max_base: int) -> pd.DataFrame:
    """Load test-partition MFE parquet, keep only the columns V2 uses."""
    if not path.exists():
        sys.exit(f"[b6] Test parquet missing: {path}. "
                 f"Run param_sweep.py --eval-partition test first.")
    pf = pq.ParquetFile(path)
    have = {f.name for f in pf.schema_arrow}
    cols = [c for c in PARQUET_COLUMNS if c in have]
    missing = set(PARQUET_COLUMNS) - have
    if missing:
        print(f"[b6] WARN: test parquet missing columns: {sorted(missing)}")
    required = {"mfe_points", "stop_distance_pts"}
    if required - have:
        sys.exit(f"[b6] FATAL: label columns missing from test parquet: "
                 f"{sorted(required - have)}")
    df = pf.read(columns=cols).to_pandas()
    n_raw = len(df)
    # Drop rows without labels (should not happen in mfe parquets, defensive).
    df = df.dropna(subset=["mfe_points", "stop_distance_pts"]).reset_index(drop=True)
    print(f"[b6] Loaded {n_raw:,} test trades ({len(df):,} labelable)")
    if len(df) > max_base:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=max_base, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"[b6] Subsampled to {len(df):,} base trades")
    return df


def expand(df: pd.DataFrame) -> pd.DataFrame:
    """R:R expansion — mirrors adaptive_rr_model_v2.expand_rr_levels."""
    n_rr = len(RR_LEVELS)
    rr_arr = np.array(RR_LEVELS, dtype=np.float32)
    n_base = len(df)
    print(f"[b6] Expanding {n_base:,} × {n_rr} R:R levels = {n_base*n_rr:,} rows")

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


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                n_bins: int = 20) -> float:
    """ECE = Σ (|bin|/N) * |mean(p_pred) − mean(y_true)| per bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_pred)
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(y_pred[m].mean() - y_true[m].mean())
    return float(ece)


def main() -> None:
    """B6: temporal OOD evaluation of the V2 adaptive R:R model.

    Splits the test partition by time into a holdout tail and reports
    per-R:R ECE and return delta vs the training-time distribution.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_test_trades(args.test_parquet, args.max_base_trades)
    exp = expand(df)

    print(f"[b6] Loading model: {args.model}")
    booster = lgb.Booster(model_file=str(args.model))

    feature_cols = [c for c in ALL_FEATURES if c in exp.columns]
    X = exp[feature_cols]
    y = exp["would_win"].to_numpy(dtype=np.int8)

    print(f"[b6] Predicting {len(X):,} rows...")
    preds = booster.predict(X, num_threads=4).astype(np.float32)

    overall = {
        "n_rows": int(len(X)),
        "n_base_trades": int(len(df)),
        "auc": float(roc_auc_score(y, preds)),
        "log_loss": float(log_loss(y, preds)),
        "brier": float(brier_score_loss(y, preds)),
        "ece_20bin": expected_calibration_error(y, preds, 20),
        "mean_y": float(y.mean()),
        "mean_pred": float(preds.mean()),
    }
    print("[b6] OVERALL (test partition):")
    for k, v in overall.items():
        print(f"    {k}: {v}")

    # Per-R:R metrics (compare calibration across the 17 levels).
    rr_vals = exp[RR_FEATURE].to_numpy()
    per_rr = []
    for rr in RR_LEVELS:
        m = rr_vals == rr
        if m.sum() < 100:
            continue
        try:
            auc_rr = float(roc_auc_score(y[m], preds[m]))
        except ValueError:
            auc_rr = None  # single-class slice
        per_rr.append({
            "rr": float(rr),
            "n": int(m.sum()),
            "auc": auc_rr,
            "brier": float(brier_score_loss(y[m], preds[m])),
            "log_loss": float(log_loss(y[m], preds[m], labels=[0, 1])),
            "mean_y": float(y[m].mean()),
            "mean_pred": float(preds[m].mean()),
            "ece_20bin": expected_calibration_error(y[m], preds[m], 20),
        })

    # Reference: training-split metrics from V2 run_metadata.json.
    meta = json.loads((REPO_ROOT / "data/ml/adaptive_rr_v2/run_metadata.json").read_text())
    train_overall = meta.get("overall_metrics", {})

    # Reliability diagram.
    import matplotlib.pyplot as plt
    frac_pos, mean_pred = calibration_curve(y, preds, n_bins=20)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="Test partition (B6)")
    ax.set_xlabel("Mean predicted P(win)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"B6 Reliability — test bars  "
                 f"(AUC={overall['auc']:.3f}, Brier={overall['brier']:.3f}, "
                 f"ECE={overall['ece_20bin']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    png_path = args.output_dir / "heldout_time_reliability_v2.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[b6] Saved {png_path}")

    # Degradation deltas (test − train, where lower log_loss/brier is better).
    degradation = {
        "auc_delta":      overall["auc"] - float(train_overall.get("auc", np.nan)),
        "log_loss_delta": overall["log_loss"] - float(train_overall.get("log_loss", np.nan)),
        "brier_delta":    overall["brier"] - float(train_overall.get("brier", np.nan)),
    }

    result = {
        "script": "scripts/evaluation/heldout_time_eval_v2.py",
        "test_parquet": str(args.test_parquet),
        "model": str(args.model),
        "test_metrics": overall,
        "train_overall_ref": train_overall,
        "degradation": degradation,
        "per_rr": per_rr,
        "runtime_seconds": time.time() - t0,
    }
    out_path = args.output_dir / "heldout_time_eval_v2.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[b6] Saved {out_path}")

    print("\n[b6] Train-vs-test summary:")
    print(f"  AUC     train={train_overall.get('auc'):.4f}  "
          f"test={overall['auc']:.4f}  Δ={degradation['auc_delta']:+.4f}")
    print(f"  LogLoss train={train_overall.get('log_loss'):.4f}  "
          f"test={overall['log_loss']:.4f}  Δ={degradation['log_loss_delta']:+.4f}")
    print(f"  Brier   train={train_overall.get('brier'):.4f}  "
          f"test={overall['brier']:.4f}  Δ={degradation['brier_delta']:+.4f}")


if __name__ == "__main__":
    main()
