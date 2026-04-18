"""Phase 4d — per-combo ECE audit of the expanding_100 calibrator.

Breaks pooled ECE 0.0039 down by global_combo_id to see whether residual
error is uniform or concentrated in a few combos.

Output: data/ml/adaptive_rr_v3/per_combo_ece_audit_v3.json
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

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "models"))

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
OUT_PATH = V3_DIR / "per_combo_ece_audit_v3.json"
TEST_PARQUET = REPO / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet"

MAX_BASE = 80_000
SWEEP_VERSION = 10
REFIT_EVERY = 100
MIN_FIT = 200
MIN_COMBO_ROWS = 500   # for per-combo ECE stability

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
    """Equal-width Expected Calibration Error over `n_bins` buckets.

    Args:
        y: Binary outcomes.
        p: Predicted probabilities aligned to `y`.
        n_bins: Number of equal-width `[0, 1]` buckets (default 20).

    Returns:
        Count-weighted mean absolute gap between predicted and observed
        per bin; `0.0` for empty input.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = len(p)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def load_test() -> pd.DataFrame:
    """Load the test-partition parquet as a pandas frame.

    Drops rows missing `mfe_points`, `stop_distance_pts`, `label_win`, or
    `r_multiple`, subsamples to `MAX_BASE` rows if larger, and attaches
    the `combo_id_v{SWEEP_VERSION}` feature used by the booster.

    Returns:
        DataFrame ready for `expand_rr`.
    """
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
    """Expand each base trade into one row per `RR_LEVELS` candidate.

    Repeats numeric/categorical/Family-A features along the R:R axis,
    computes the `would_win` synthetic label `(mfe_points ≥ rr × stop)`,
    preserves a `_base_idx` back-pointer, and adds the interaction
    features `abs_zscore_entry` and `rr_x_atr`.

    Args:
        df: Base trade frame loaded by `load_test`.

    Returns:
        Long-format DataFrame (`len(df) × len(RR_LEVELS)` rows).
    """
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


def expanding_recal_1d(p_raw: np.ndarray, y: np.ndarray,
                       refit_every: int) -> np.ndarray:
    """Expanding-window isotonic calibration over a chronologically sorted slice.

    Fits `IsotonicRegression` on the first `anchor` rows and applies it
    to the next `refit_every` rows; walks forward until the end. Skips
    refits where fewer than 2 unique labels are present. Rows before
    `MIN_FIT` keep their raw score.

    Args:
        p_raw: Raw booster predictions in chronological order.
        y: Binary outcomes aligned to `p_raw`.
        refit_every: Step size between refits.

    Returns:
        Calibrated probabilities, same shape as `p_raw`.
    """
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


def main() -> None:
    """Audit per-combo ECE after pooled per-R:R expanding recalibration.

    Loads the test partition, expands R:R, predicts with the V3 booster,
    applies expanding isotonic per R:R, then reports pooled ECE plus
    per-combo ECE (raw vs calibrated) and top/bottom-10 concentration.
    Writes JSON to `OUT_PATH`.
    """
    t0 = time.time()
    df = load_test()
    print(f"[4d] loaded {len(df):,} base trades")

    df = add_family_a(df)
    exp = expand_rr(df)
    combo_col_base = df[ID_FEATURE].to_numpy()
    combo_col = np.repeat(combo_col_base, len(RR_LEVELS))
    print(f"[4d] expanded to {len(exp):,} rows, "
          f"{len(np.unique(combo_col_base))} unique combos")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t1 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=3).astype(np.float64)
    print(f"[4d] raw predictions in {time.time()-t1:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    # Apply expanding_100 per-R:R in chronological order.
    p_cal = p_raw.copy()
    t2 = time.time()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        if m.sum() == 0:
            continue
        idx_m = np.where(m)[0]
        order = idx_m[np.argsort(base_idx[idx_m], kind="stable")]
        p_cal[order] = expanding_recal_1d(p_raw[order], y[order], REFIT_EVERY)
    print(f"[4d] expanding_100 calibration in {time.time()-t2:.1f}s")

    pooled_ece = ece_20(y, p_cal)
    print(f"[4d] pooled ECE: {pooled_ece:.4f}")

    # Per-combo ECE.
    combos, inv = np.unique(combo_col, return_inverse=True)
    per_combo_rows = []
    for i, c in enumerate(combos):
        m = inv == i
        n = int(m.sum())
        if n < MIN_COMBO_ROWS:
            continue
        per_combo_rows.append({
            "combo_id": str(c),
            "n": n,
            "ece_raw": ece_20(y[m], p_raw[m]),
            "ece_cal": ece_20(y[m], p_cal[m]),
            "mean_y":  float(y[m].mean()),
            "mean_p_cal": float(p_cal[m].mean()),
        })

    per_combo_rows.sort(key=lambda r: r["ece_cal"], reverse=True)
    n_combos = len(per_combo_rows)
    top10 = per_combo_rows[:10]
    bottom10 = per_combo_rows[-10:]

    # Concentration analysis: how much total ECE contribution from top K combos?
    # Contribution to pooled ECE is weighted by n; approximate with
    # n_c * ece_c / sum(n_c) to get each combo's weighted share.
    total_n = sum(r["n"] for r in per_combo_rows)
    weighted = sorted(
        ((r["n"] * r["ece_cal"] / total_n, r) for r in per_combo_rows),
        key=lambda t: t[0], reverse=True,
    )
    total_weighted = sum(w for w, _ in weighted)
    cum = 0.0
    concentration = {}
    for k in (5, 10, 20, 50):
        share = sum(w for w, _ in weighted[:k]) / total_weighted
        concentration[f"top_{k}_share"] = share

    ece_array = np.array([r["ece_cal"] for r in per_combo_rows])
    summary = {
        "script": "scripts/calibration/per_combo_ece_audit_v3.py",
        "calibrator": "expanding_100",
        "pooled_ece": pooled_ece,
        "n_combos_audited": n_combos,
        "min_combo_rows": MIN_COMBO_ROWS,
        "per_combo_ece_stats": {
            "mean":   float(ece_array.mean()),
            "median": float(np.median(ece_array)),
            "p25":    float(np.percentile(ece_array, 25)),
            "p75":    float(np.percentile(ece_array, 75)),
            "p90":    float(np.percentile(ece_array, 90)),
            "p99":    float(np.percentile(ece_array, 99)),
            "max":    float(ece_array.max()),
        },
        "concentration": concentration,
        "top10_worst": top10,
        "bottom10_best": bottom10,
        "runtime_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[4d] saved {OUT_PATH}")
    print(f"[4d] top-10 worst combos contribute "
          f"{concentration['top_10_share']*100:.1f}% of weighted ECE")


if __name__ == "__main__":
    main()
