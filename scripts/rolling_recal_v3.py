"""Phase 4 — V3 + rolling isotonic recalibrator on the B6 held-out tail.

Compares three calibration strategies on the held-out 20% test partition:

  raw      — V3 booster output, no calibration
  static   — V3 booster + V3's OOF-trained per-R:R IsotonicRegression
             (from `isotonic_calibrators_v3.json`)
  rolling  — V3 booster + a causal rolling per-R:R IsotonicRegression re-fit
             every `--refit-every` trades on the most recent `--window`
             labelled trades. At step i the fit uses ONLY trades [i-window, i).

Gate (from tasks/v3_followup_plan.md §Phase 4): rolling ECE < 0.015 on the
held-out bars (static baseline 0.062). If that holds, rolling recalibration
addresses the post-2024-10-22 regime drift; if not, document the null.

Output: data/ml/adaptive_rr_v3/b6_rolling_recal.json
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from adaptive_rr_model_v3 import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_COLS,
    FAMILY_A,
    ID_FEATURE,
    NUMERIC_FEATURE_COLS,
    RR_FEATURE,
    RR_LEVELS,
    add_family_a,
)
from inference_v3 import _apply_calibrator, _load_calibrators  # noqa: E402

V3_DIR = REPO / "data/ml/adaptive_rr_v3"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
V3_CALIBRATORS = V3_DIR / "isotonic_calibrators_v3.json"
OUT_PATH = V3_DIR / "b6_rolling_recal.json"

PARQUET_COLUMNS = [
    "combo_id", "mfe_points", "stop_distance_pts",
    "label_win", "r_multiple",
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
]


def ece_20(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def load_test(path: Path, max_base: int, sweep_version: int = 10) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    have = {f.name for f in pf.schema_arrow}
    cols = [c for c in PARQUET_COLUMNS if c in have]
    df = pf.read(columns=cols).to_pandas()
    df = df.dropna(subset=["mfe_points", "stop_distance_pts",
                           "label_win", "r_multiple"]).reset_index(drop=True)
    if len(df) > max_base:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(df), size=max_base, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    df[ID_FEATURE] = f"v{sweep_version}_" + df["combo_id"].astype(str)
    return df


def expand_rr(df: pd.DataFrame) -> pd.DataFrame:
    rr_arr = np.array(RR_LEVELS, dtype=np.float32)
    n_rr = len(rr_arr)
    n_base = len(df)
    out: dict[str, np.ndarray] = {}
    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(dtype=np.float32), n_rr)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            out[col] = np.repeat(df[col].to_numpy(), n_rr)
    for col in FAMILY_A:
        if col in df.columns:
            dtype = np.int8 if col == "has_history_50" else np.float32
            out[col] = np.repeat(df[col].to_numpy(dtype=dtype), n_rr)
    if "exit_on_opposite_signal" in df.columns:
        out["exit_on_opposite_signal"] = np.repeat(
            df["exit_on_opposite_signal"].to_numpy(dtype=np.int8), n_rr)
    out[RR_FEATURE] = np.tile(rr_arr, n_base)
    mfe = df["mfe_points"].to_numpy(dtype=np.float32)
    stop = df["stop_distance_pts"].to_numpy(dtype=np.float32)
    out["would_win"] = (
        (mfe[:, None] >= rr_arr[None, :] * stop[:, None]).ravel().astype(np.int8)
    )
    out["_base_idx"] = np.repeat(np.arange(n_base, dtype=np.int64), n_rr)
    if "zscore_entry" in out:
        out["abs_zscore_entry"] = np.abs(out["zscore_entry"])
    if "atr_points" in out:
        out["rr_x_atr"] = out[RR_FEATURE] * out["atr_points"]
    ex = pd.DataFrame(out, copy=False)
    for col in CATEGORICAL_COLS:
        if col in ex.columns:
            ex[col] = ex[col].astype("category")
    return ex


def rolling_recal_1d(p_raw: np.ndarray, y: np.ndarray,
                     window: int, refit_every: int,
                     min_fit: int = 200) -> np.ndarray:
    """Causal rolling isotonic: at anchor i, fit on p_raw[i-window:i] / y[...]
    and predict for [i, i+refit_every). For i < min_fit, leave as raw."""
    n = len(p_raw)
    p_cal = p_raw.copy()
    for anchor in range(min_fit, n, refit_every):
        lo = max(0, anchor - window)
        bp = p_raw[lo:anchor]
        by = y[lo:anchor]
        if len(np.unique(by)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(bp, by)
        hi = min(n, anchor + refit_every)
        p_cal[anchor:hi] = iso.predict(p_raw[anchor:hi])
    return p_cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", type=Path,
                    default=REPO / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet")
    ap.add_argument("--max-base", type=int, default=80_000)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--refit-every", type=int, default=500)
    ap.add_argument("--sweep-version", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    df = load_test(args.test_parquet, args.max_base, args.sweep_version)
    print(f"[phase4] {len(df):,} base trades loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    df = add_family_a(df)
    print(f"[phase4] Family A computed in {time.time()-t1:.1f}s")

    t2 = time.time()
    exp = expand_rr(df)
    print(f"[phase4] expanded to {len(exp):,} rows in {time.time()-t2:.1f}s")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t3 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[phase4] raw predictions in {time.time()-t3:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    # Static calibration: reuse V3 OOF-trained per-R:R isotonics.
    cals = _load_calibrators(V3_CALIBRATORS)
    p_static = p_raw.copy()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        knots = cals.get(f"{float(rr):.2f}")
        if knots is not None:
            p_static[m] = _apply_calibrator(p_raw[m], knots)

    # Rolling calibration: per-R:R chronological by _base_idx.
    p_rolling = p_raw.copy()
    per_rr_rows = []
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        if m.sum() == 0:
            continue
        idx_m = np.where(m)[0]
        order = idx_m[np.argsort(base_idx[idx_m], kind="stable")]
        p_cal_sorted = rolling_recal_1d(p_raw[order], y[order],
                                        args.window, args.refit_every)
        p_rolling[order] = p_cal_sorted
        if m.sum() >= 100:
            row = {
                "rr": float(rr),
                "n": int(m.sum()),
                "ece_raw":     ece_20(y[m], p_raw[m]),
                "ece_static":  ece_20(y[m], p_static[m]),
                "ece_rolling": ece_20(y[m], p_rolling[m]),
                "mean_y":      float(y[m].mean()),
                "mean_p_raw":  float(p_raw[m].mean()),
                "mean_p_cal":  float(p_rolling[m].mean()),
            }
            try:
                row["auc"] = float(roc_auc_score(y[m], p_raw[m]))
            except ValueError:
                row["auc"] = None
            per_rr_rows.append(row)

    overall = {
        "n_base_trades": int(len(df)),
        "n_expanded":    int(len(exp)),
        "auc":           float(roc_auc_score(y, p_raw)),
        "mean_y":        float(y.mean()),
        "mean_p_raw":    float(p_raw.mean()),
        "mean_p_static": float(p_static.mean()),
        "mean_p_rolling": float(p_rolling.mean()),
        "brier_raw":     float(brier_score_loss(y, p_raw)),
        "brier_static":  float(brier_score_loss(y, p_static)),
        "brier_rolling": float(brier_score_loss(y, p_rolling)),
        "log_loss_raw":     float(log_loss(y, np.clip(p_raw, 1e-6, 1 - 1e-6))),
        "log_loss_static":  float(log_loss(y, np.clip(p_static, 1e-6, 1 - 1e-6))),
        "log_loss_rolling": float(log_loss(y, np.clip(p_rolling, 1e-6, 1 - 1e-6))),
        "ece_raw":      ece_20(y, p_raw),
        "ece_static":   ece_20(y, p_static),
        "ece_rolling":  ece_20(y, p_rolling),
    }

    result = {
        "script": "scripts/rolling_recal_v3.py",
        "test_parquet": str(args.test_parquet),
        "model": str(V3_BOOSTER),
        "config": {
            "window": args.window,
            "refit_every": args.refit_every,
            "sweep_version": args.sweep_version,
            "max_base": args.max_base,
        },
        "gate_pass": overall["ece_rolling"] < 0.015,
        "overall": overall,
        "per_rr": per_rr_rows,
        "runtime_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[phase4] Saved {OUT_PATH}")
    print(f"[phase4] ECE: raw={overall['ece_raw']:.4f} "
          f"static={overall['ece_static']:.4f} "
          f"rolling={overall['ece_rolling']:.4f}")
    print(f"[phase4] Gate (rolling < 0.015): "
          f"{'PASS' if result['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
