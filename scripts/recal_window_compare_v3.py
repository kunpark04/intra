"""Phase 4c — lock in production config + test expanding causal window.

Compares four calibration strategies on the V3 held-out tail:

  rolling (20000, 100)   — new production candidate from Phase 4b grid
  rolling (5000, 500)    — old default, for reference
  expanding refit=100    — causal expanding (all past trades, refit every 100)
  expanding refit=500    — cheaper expanding variant

Goal: confirm (20000, 100) is the right production pick, and determine
whether a simpler expanding-window causal isotonic matches it (if yes,
productionization doesn't need a sliding window at all).

Output: data/ml/adaptive_rr_v3/recal_window_compare_v3.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

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

V3_DIR = REPO / "data/ml/adaptive_rr_v3"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
OUT_PATH = V3_DIR / "recal_window_compare_v3.json"
TEST_PARQUET = REPO / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet"

MAX_BASE = 80_000
SWEEP_VERSION = 10
GATE = 0.015
MIN_FIT = 200

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
    if n == 0:
        return 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def load_test() -> pd.DataFrame:
    pf = pq.ParquetFile(TEST_PARQUET)
    have = {f.name for f in pf.schema_arrow}
    cols = [c for c in PARQUET_COLUMNS if c in have]
    df = pf.read(columns=cols).to_pandas()
    df = df.dropna(subset=["mfe_points", "stop_distance_pts",
                           "label_win", "r_multiple"]).reset_index(drop=True)
    if len(df) > MAX_BASE:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(df), size=MAX_BASE, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    df[ID_FEATURE] = f"v{SWEEP_VERSION}_" + df["combo_id"].astype(str)
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
                     window: int, refit_every: int) -> np.ndarray:
    """Causal rolling (bounded window) isotonic."""
    n = len(p_raw)
    p_cal = p_raw.copy()
    for anchor in range(MIN_FIT, n, refit_every):
        lo = max(0, anchor - window)
        bp = p_raw[lo:anchor]
        by = y[lo:anchor]
        if len(np.unique(by)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(bp, by)
        hi = min(n, anchor + refit_every)
        p_cal[anchor:hi] = iso.predict(p_raw[anchor:hi])
    return p_cal


def expanding_recal_1d(p_raw: np.ndarray, y: np.ndarray,
                       refit_every: int) -> np.ndarray:
    """Causal expanding isotonic: fit on all past [0, anchor)."""
    n = len(p_raw)
    p_cal = p_raw.copy()
    for anchor in range(MIN_FIT, n, refit_every):
        bp = p_raw[:anchor]
        by = y[:anchor]
        if len(np.unique(by)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(bp, by)
        hi = min(n, anchor + refit_every)
        p_cal[anchor:hi] = iso.predict(p_raw[anchor:hi])
    return p_cal


def apply_per_rr(p_raw: np.ndarray, y: np.ndarray,
                 rr_col: np.ndarray, base_idx: np.ndarray,
                 fn) -> np.ndarray:
    """Apply calibration fn per-R:R in chronological (_base_idx) order."""
    p_out = p_raw.copy()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        if m.sum() == 0:
            continue
        idx_m = np.where(m)[0]
        order = idx_m[np.argsort(base_idx[idx_m], kind="stable")]
        p_cal_sorted = fn(p_raw[order], y[order])
        p_out[order] = p_cal_sorted
    return p_out


def main() -> None:
    t0 = time.time()
    df = load_test()
    print(f"[4c] loaded {len(df):,} base trades in {time.time()-t0:.1f}s")

    df = add_family_a(df)
    exp = expand_rr(df)
    print(f"[4c] expanded to {len(exp):,} rows")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t1 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[4c] raw predictions in {time.time()-t1:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    variants = [
        ("rolling_20000_100", lambda p, y_: rolling_recal_1d(p, y_, 20000, 100)),
        ("rolling_5000_500",  lambda p, y_: rolling_recal_1d(p, y_, 5000, 500)),
        ("expanding_100",     lambda p, y_: expanding_recal_1d(p, y_, 100)),
        ("expanding_500",     lambda p, y_: expanding_recal_1d(p, y_, 500)),
    ]

    results = {"ece_raw": ece_20(y, p_raw)}
    for name, fn in variants:
        t = time.time()
        p_cal = apply_per_rr(p_raw, y, rr_col, base_idx, fn)
        e = ece_20(y, p_cal)
        dt = time.time() - t
        results[f"ece_{name}"] = e
        results[f"seconds_{name}"] = dt
        results[f"pass_{name}"] = e < GATE
        print(f"[4c] {name:22s} ECE={e:.4f} "
              f"({'PASS' if e < GATE else 'FAIL'}) [{dt:.1f}s]")

    # Simplicity check: does expanding_100 match rolling_20000_100?
    rolling_new = results["ece_rolling_20000_100"]
    expanding_best = min(results["ece_expanding_100"],
                         results["ece_expanding_500"])
    results["expanding_matches_rolling"] = abs(
        expanding_best - rolling_new) < 0.001
    results["production_pick"] = (
        "expanding_100" if (
            results["ece_expanding_100"] <= rolling_new + 0.0005
        ) else "rolling_20000_100"
    )

    summary = {
        "script": "scripts/recal_window_compare_v3.py",
        "n_base": int(len(df)),
        "n_expanded": int(len(exp)),
        "gate": GATE,
        "results": results,
        "runtime_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[4c] saved {OUT_PATH}")
    print(f"[4c] production pick: {results['production_pick']}")


if __name__ == "__main__":
    main()
