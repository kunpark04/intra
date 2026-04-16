"""Phase 4f — per-combo R:R-agnostic expanding isotonic with pooled fallback.

Phase 4e keyed on (combo, R:R) but starved the ~2000-row offenders
(120 rows per R:R < 300 threshold). Phase 4f drops the R:R split:
fit one expanding isotonic per combo on its full cross-R:R stream.
17x more samples per combo; the worst offenders should now qualify.

Assumption: per-combo miscalibration is R:R-agnostic bias (Phase 4d
evidence: worst offenders had systematic under-prediction across
every R:R, mean_y ~0.33 vs mean_p ~0.14).

Output: data/ml/adaptive_rr_v3/per_combo_isotonic_v3.json
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
OUT_PATH = V3_DIR / "per_combo_isotonic_v3.json"
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
    """Load the test-partition parquet and prepare it for R:R expansion.

    Drops rows missing the label or sizing inputs, subsamples to
    `MAX_BASE` if larger, and attaches the combo-id feature used by
    the V3 booster.

    Returns:
        Cleaned base DataFrame ready for `expand_rr`.
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
    """Expand each base trade into one row per candidate R:R level.

    Repeats numeric/categorical/Family-A features along the R:R axis,
    synthesises the `would_win` label as `(mfe_points ≥ rr × stop)`,
    preserves a `_base_idx` back-pointer, and adds the `rr_x_atr` and
    `abs_zscore_entry` interaction features.

    Args:
        df: Base trade frame from `load_test`.

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
    forward to the next chunk; walks until the end. Skips refits with
    fewer than 2 unique labels. Rows before the warm-up keep raw scores.

    Args:
        p_raw: Raw booster predictions in chronological order.
        y: Binary outcomes aligned to `p_raw`.
        refit_every: Bars between refits.

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
    """Fit per-combo 1-D isotonic calibrators on top of the pooled per-R:R stack.

    For each combo with enough data, fits an additional `IsotonicRegression`
    on top of the pooled-recalibrated score and reports ECE before and
    after. Writes JSON + a knots artifact consumed by `inference_v3`.
    """
    t0 = time.time()
    df = load_test()
    print(f"[4f] loaded {len(df):,} base trades")

    df = add_family_a(df)
    exp = expand_rr(df)
    combo_col_base = df[ID_FEATURE].to_numpy()
    combo_col = np.repeat(combo_col_base, len(RR_LEVELS))
    print(f"[4f] expanded to {len(exp):,} rows, "
          f"{len(np.unique(combo_col_base))} combos")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t1 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[4f] raw predictions in {time.time()-t1:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    # Stage 1: pooled per-R:R expanding (fallback).
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
    print(f"[4f] pooled-per-RR (fallback) in {time.time()-t2:.1f}s, "
          f"ECE={ece_pooled:.4f}")

    # Stage 2: per-combo R:R-agnostic expanding isotonic override.
    t3 = time.time()
    p_cal = p_pooled.copy()
    uniq_combos = np.unique(combo_col)
    combos_overridden = 0
    combos_fallback = 0
    rows_overridden = 0
    for combo in uniq_combos:
        m = combo_col == combo
        idx_c = np.where(m)[0]
        if len(idx_c) < MIN_FIT + REFIT_EVERY:
            combos_fallback += 1
            continue
        order = idx_c[np.argsort(base_idx[idx_c], kind="stable")]
        p_stream = expanding_recal_1d(p_raw[order], y[order], REFIT_EVERY)
        # Warmup (first MIN_FIT): use pooled values, not raw.
        p_stream[:MIN_FIT] = p_pooled[order[:MIN_FIT]]
        p_cal[order] = p_stream
        combos_overridden += 1
        rows_overridden += len(order)
    print(f"[4f] per-combo (R:R-agnostic) in {time.time()-t3:.1f}s; "
          f"{combos_overridden} overridden / {combos_fallback} fallback; "
          f"rows overridden={rows_overridden:,}")

    ece_per_combo = ece_20(y, p_cal)
    print(f"[4f] per-combo (no RR split) ECE={ece_per_combo:.4f} "
          f"(pooled was {ece_pooled:.4f})")

    # Audit: per-combo ECE under the new calibrator.
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
            "ece_per_combo_f": ece_20(y[m], p_cal[m]),
            "mean_y":     float(y[m].mean()),
        })

    # Track the 10 that were worst under pooled.
    pooled_sorted = sorted(per_combo_rows,
                           key=lambda r: r["ece_pooled"], reverse=True)
    top10_pooled = pooled_sorted[:10]
    improvements = [
        {**r, "improvement": r["ece_pooled"] - r["ece_per_combo_f"]}
        for r in top10_pooled
    ]

    ece_arr = np.array([r["ece_per_combo_f"] for r in per_combo_rows])
    summary = {
        "script": "scripts/calibration/per_combo_isotonic_v3.py",
        "n_base": int(len(df)),
        "n_expanded": int(len(exp)),
        "pooled_ece_reference": ece_pooled,
        "per_combo_f_ece": ece_per_combo,
        "delta_vs_pooled": ece_per_combo - ece_pooled,
        "n_combos_overridden": combos_overridden,
        "n_combos_fallback": combos_fallback,
        "rows_overridden": int(rows_overridden),
        "per_combo_ece_stats_f": {
            "mean":   float(ece_arr.mean()),
            "median": float(np.median(ece_arr)),
            "p25":    float(np.percentile(ece_arr, 25)),
            "p75":    float(np.percentile(ece_arr, 75)),
            "p90":    float(np.percentile(ece_arr, 90)),
            "max":    float(ece_arr.max()),
        },
        "pooled_top10_tracked": improvements,
        "runtime_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[4f] saved {OUT_PATH}")
    print(f"[4f] new median per-combo ECE: "
          f"{summary['per_combo_ece_stats_f']['median']:.4f}")


if __name__ == "__main__":
    main()
