"""Phase 4e — per-(combo, R:R) expanding isotonic with pooled fallback.

Phase 4d showed pooled calibration (ECE 0.0039) hides median per-combo
ECE 0.043 — the 10 worst combos were actively harmed by calibration.
This test keys the expanding isotonic on (combo_id, R:R): fit on each
combo's own past trades at that R:R. Fall back to pooled-per-R:R
expanding isotonic when the combo has fewer than MIN_FIT past trades.

Output: data/ml/adaptive_rr_v3/per_combo_rr_isotonic_v3.json
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
OUT_PATH = V3_DIR / "per_combo_rr_isotonic_v3.json"
TEST_PARQUET = REPO / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet"

MAX_BASE = 80_000
SWEEP_VERSION = 10
REFIT_EVERY = 100
MIN_FIT = 200
MIN_COMBO_ROWS_AUDIT = 500

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


def expanding_recal_1d(p_raw: np.ndarray, y: np.ndarray,
                       refit_every: int) -> np.ndarray:
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
    t0 = time.time()
    df = load_test()
    print(f"[4e] loaded {len(df):,} base trades")

    df = add_family_a(df)
    exp = expand_rr(df)
    combo_col_base = df[ID_FEATURE].to_numpy()
    combo_col = np.repeat(combo_col_base, len(RR_LEVELS))
    n_combos_total = len(np.unique(combo_col_base))
    print(f"[4e] expanded to {len(exp):,} rows, {n_combos_total} combos")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t1 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[4e] raw predictions in {time.time()-t1:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    # Stage 1: pooled expanding per R:R — the fallback.
    t2 = time.time()
    p_pooled = p_raw.copy()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        if m.sum() == 0:
            continue
        idx_m = np.where(m)[0]
        order = idx_m[np.argsort(base_idx[idx_m], kind="stable")]
        p_pooled[order] = expanding_recal_1d(p_raw[order], y[order],
                                             REFIT_EVERY)
    ece_pooled = ece_20(y, p_pooled)
    print(f"[4e] pooled-per-RR (fallback) in {time.time()-t2:.1f}s, "
          f"ECE={ece_pooled:.4f}")

    # Stage 2: per-(combo, R:R) expanding — override when combo stream
    # has >= MIN_FIT + REFIT_EVERY trades at that R:R.
    t3 = time.time()
    p_cal = p_pooled.copy()
    combos_overridden = 0
    combos_fallback = 0
    rows_overridden = 0
    rows_fallback = 0
    for rr in RR_LEVELS:
        m_rr = rr_col == np.float32(rr)
        if m_rr.sum() == 0:
            continue
        idx_rr = np.where(m_rr)[0]
        # Within this RR, group by combo.
        combos_at_rr = combo_col[idx_rr]
        uniq = np.unique(combos_at_rr)
        for combo in uniq:
            mc = combos_at_rr == combo
            idx_combo = idx_rr[mc]
            # Chronological order within (combo, R:R).
            order = idx_combo[np.argsort(base_idx[idx_combo], kind="stable")]
            if len(order) < MIN_FIT + REFIT_EVERY:
                combos_fallback += 1
                rows_fallback += len(order)
                continue  # keep pooled fallback
            # Per-combo expanding isotonic; leaves first MIN_FIT as raw.
            # Replace those with pooled values to avoid raw leakage.
            p_stream = expanding_recal_1d(p_raw[order], y[order], REFIT_EVERY)
            # First MIN_FIT positions: use pooled, not raw.
            p_stream[:MIN_FIT] = p_pooled[order[:MIN_FIT]]
            p_cal[order] = p_stream
            combos_overridden += 1
            rows_overridden += len(order)
    print(f"[4e] per-(combo,rr) override in {time.time()-t3:.1f}s; "
          f"{combos_overridden} streams overridden, "
          f"{combos_fallback} streams kept pooled; "
          f"rows override={rows_overridden:,} fallback={rows_fallback:,}")

    ece_per_combo = ece_20(y, p_cal)
    print(f"[4e] per-(combo,rr) expanding ECE={ece_per_combo:.4f} "
          f"(pooled was {ece_pooled:.4f})")

    # Audit: per-combo ECE distribution under the new calibrator.
    combos, inv = np.unique(combo_col, return_inverse=True)
    per_combo_rows = []
    for i, c in enumerate(combos):
        m = inv == i
        n = int(m.sum())
        if n < MIN_COMBO_ROWS_AUDIT:
            continue
        per_combo_rows.append({
            "combo_id": str(c),
            "n": n,
            "ece_raw":    ece_20(y[m], p_raw[m]),
            "ece_pooled": ece_20(y[m], p_pooled[m]),
            "ece_per_combo": ece_20(y[m], p_cal[m]),
            "mean_y":     float(y[m].mean()),
        })
    per_combo_rows.sort(key=lambda r: r["ece_per_combo"], reverse=True)
    top10 = per_combo_rows[:10]

    # Target comparison: the 10 worst under pooled (from 4d).
    pooled_sorted = sorted(per_combo_rows,
                           key=lambda r: r["ece_pooled"], reverse=True)
    top10_under_pooled = pooled_sorted[:10]

    ece_arr = np.array([r["ece_per_combo"] for r in per_combo_rows])
    summary = {
        "script": "scripts/calibration/per_combo_rr_isotonic_v3.py",
        "n_base": int(len(df)),
        "n_expanded": int(len(exp)),
        "pooled_ece_reference": ece_pooled,
        "per_combo_pooled_ece": ece_per_combo,
        "delta_vs_pooled": ece_per_combo - ece_pooled,
        "n_combos_overridden": combos_overridden,
        "n_combos_fallback": combos_fallback,
        "rows_overridden": int(rows_overridden),
        "rows_fallback": int(rows_fallback),
        "per_combo_ece_stats_new": {
            "mean":   float(ece_arr.mean()),
            "median": float(np.median(ece_arr)),
            "p25":    float(np.percentile(ece_arr, 25)),
            "p75":    float(np.percentile(ece_arr, 75)),
            "p90":    float(np.percentile(ece_arr, 90)),
            "max":    float(ece_arr.max()),
        },
        "top10_worst_under_new": top10,
        "pooled_top10_tracked": [
            {
                "combo_id":       r["combo_id"],
                "n":              r["n"],
                "ece_raw":        r["ece_raw"],
                "ece_pooled":     r["ece_pooled"],
                "ece_per_combo":  r["ece_per_combo"],
                "mean_y":         r["mean_y"],
                "improvement":    r["ece_pooled"] - r["ece_per_combo"],
            } for r in top10_under_pooled
        ],
        "runtime_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[4e] saved {OUT_PATH}")
    print(f"[4e] new median per-combo ECE: "
          f"{summary['per_combo_ece_stats_new']['median']:.4f} "
          f"(was 0.043 under pooled)")


if __name__ == "__main__":
    main()
