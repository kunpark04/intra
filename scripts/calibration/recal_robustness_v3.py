"""Phase 4b — robustness tests for V3 rolling recalibrator.

Runs three diagnostics in one pass (shares the expensive booster predict):

  4b.1  regime-split ECE: chronological halves of the held-out tail
  4b.2  bootstrap CI on rolling ECE at default (5000, 500) config
  4b.3  grid sweep: window x refit_every, report rolling ECE per cell

Output: data/ml/adaptive_rr_v3/recal_robustness_v3.json
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
from inference_v3 import _apply_calibrator, _load_calibrators  # noqa: E402

V3_DIR = REPO / "data/ml/adaptive_rr_v3"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
V3_CALIBRATORS = V3_DIR / "isotonic_calibrators_v3.json"
OUT_PATH = V3_DIR / "recal_robustness_v3.json"
TEST_PARQUET = REPO / "data/ml/mfe/ml_dataset_v10_test_mfe.parquet"

MAX_BASE = 80_000
SWEEP_VERSION = 10
DEFAULT_WINDOW = 5000
DEFAULT_REFIT = 500
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 777
GATE = 0.015

GRID_WINDOWS = [1000, 2500, 5000, 10000, 20000]
GRID_REFITS = [100, 500, 2000]

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


def rolling_recal_1d(p_raw: np.ndarray, y: np.ndarray,
                     window: int, refit_every: int,
                     min_fit: int = 200) -> np.ndarray:
    """Rolling-window isotonic calibration over a chronologically sorted slice.

    Slides a fixed-width training window and refits `IsotonicRegression`
    every `refit_every` rows. Complementary to `expanding_recal_1d`:
    better for detecting regime drift, worse for small samples.

    Args:
        p_raw: Raw booster predictions in chronological order.
        y: Binary outcomes aligned to `p_raw`.
        window: Training window size.
        refit_every: Bars between refits.

    Returns:
        Calibrated probabilities, same shape as `p_raw`.
    """
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


def apply_rolling_per_rr(p_raw: np.ndarray, y: np.ndarray,
                         rr_col: np.ndarray, base_idx: np.ndarray,
                         window: int, refit_every: int) -> np.ndarray:
    """Apply `rolling_recal_1d` independently per R:R slice.

    Args:
        p_raw: Raw predictions (full long-format vector).
        y: Outcomes aligned to `p_raw`.
        rr_col: Per-row R:R level (used to split).
        base_idx: Per-row base-trade index (used to sort chronologically).
        window: Training window size.
        refit_every: Bars between refits.

    Returns:
        Calibrated probabilities, same shape as `p_raw`.
    """
    p_out = p_raw.copy()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        if m.sum() == 0:
            continue
        idx_m = np.where(m)[0]
        order = idx_m[np.argsort(base_idx[idx_m], kind="stable")]
        p_cal_sorted = rolling_recal_1d(p_raw[order], y[order],
                                        window, refit_every)
        p_out[order] = p_cal_sorted
    return p_out


def regime_split_metrics(y: np.ndarray, p_raw: np.ndarray,
                         p_static: np.ndarray, p_rolling: np.ndarray,
                         base_idx: np.ndarray) -> dict:
    """Per-regime ECE on a supplied train/test split.

    Args:
        y: Outcomes.
        p: Predicted probabilities aligned to `y`.
        mask_train: Bool mask for the training half.
        mask_test: Bool mask for the held-out half.

    Returns:
        Dict with `ece_train`, `ece_test`, and sample counts.
    """
    n_base = int(base_idx.max()) + 1
    mid = n_base // 2
    m_early = base_idx < mid
    m_late = base_idx >= mid
    return {
        "split_base_idx_mid": mid,
        "n_base_total": n_base,
        "early": {
            "n": int(m_early.sum()),
            "ece_raw":     ece_20(y[m_early], p_raw[m_early]),
            "ece_static":  ece_20(y[m_early], p_static[m_early]),
            "ece_rolling": ece_20(y[m_early], p_rolling[m_early]),
            "mean_y":      float(y[m_early].mean()),
            "mean_p_raw":  float(p_raw[m_early].mean()),
        },
        "late": {
            "n": int(m_late.sum()),
            "ece_raw":     ece_20(y[m_late], p_raw[m_late]),
            "ece_static":  ece_20(y[m_late], p_static[m_late]),
            "ece_rolling": ece_20(y[m_late], p_rolling[m_late]),
            "mean_y":      float(y[m_late].mean()),
            "mean_p_raw":  float(p_raw[m_late].mean()),
        },
    }


def bootstrap_ece(y: np.ndarray, p_rolling: np.ndarray,
                  n_boot: int, seed: int, gate: float) -> dict:
    """Bootstrap confidence interval for ECE via row resampling.

    Args:
        y: Outcomes.
        p: Predicted probabilities aligned to `y`.
        n_boot: Number of bootstrap draws (default defined in body).
        seed: RNG seed.

    Returns:
        Dict with `mean`, `p2_5`, `p97_5` ECE values.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    eces = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        eces[i] = ece_20(y[idx], p_rolling[idx])
    eces.sort()
    return {
        "n_boot": n_boot,
        "mean":   float(eces.mean()),
        "median": float(np.median(eces)),
        "ci_95_lo": float(eces[int(0.025 * n_boot)]),
        "ci_95_hi": float(eces[int(0.975 * n_boot) - 1]),
        "p_below_gate": float((eces < gate).mean()),
        "gate": gate,
    }


def main() -> None:
    """Robustness audit of per-R:R recalibration via regime splits + bootstraps.

    Computes per-regime ECE, bootstrap CIs, and compares rolling vs
    expanding recalibration across multiple `refit_every` cadences.
    Writes an aggregate JSON report.
    """
    t0 = time.time()
    df = load_test()
    print(f"[4b] loaded {len(df):,} base trades in {time.time()-t0:.1f}s")

    t1 = time.time()
    df = add_family_a(df)
    print(f"[4b] Family A in {time.time()-t1:.1f}s")

    t2 = time.time()
    exp = expand_rr(df)
    print(f"[4b] expanded to {len(exp):,} rows in {time.time()-t2:.1f}s")

    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t3 = time.time()
    p_raw = booster.predict(exp[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[4b] raw predictions in {time.time()-t3:.1f}s")

    y = exp["would_win"].to_numpy(np.int8)
    rr_col = exp[RR_FEATURE].to_numpy(np.float32)
    base_idx = exp["_base_idx"].to_numpy(np.int64)

    cals = _load_calibrators(V3_CALIBRATORS)
    p_static = p_raw.copy()
    for rr in RR_LEVELS:
        m = rr_col == np.float32(rr)
        knots = cals.get(f"{float(rr):.2f}")
        if knots is not None:
            p_static[m] = _apply_calibrator(p_raw[m], knots)

    # Default rolling (5000, 500) — used for 4b.1 + 4b.2.
    t4 = time.time()
    p_rolling_default = apply_rolling_per_rr(
        p_raw, y, rr_col, base_idx, DEFAULT_WINDOW, DEFAULT_REFIT)
    print(f"[4b] default rolling ({DEFAULT_WINDOW},{DEFAULT_REFIT}) "
          f"in {time.time()-t4:.1f}s")

    headline = {
        "ece_raw":     ece_20(y, p_raw),
        "ece_static":  ece_20(y, p_static),
        "ece_rolling": ece_20(y, p_rolling_default),
    }
    print(f"[4b] headline ECE: raw={headline['ece_raw']:.4f} "
          f"static={headline['ece_static']:.4f} "
          f"rolling={headline['ece_rolling']:.4f}")

    # 4b.1 — regime split.
    t5 = time.time()
    regime = regime_split_metrics(y, p_raw, p_static, p_rolling_default,
                                  base_idx)
    print(f"[4b.1] regime split in {time.time()-t5:.1f}s: "
          f"early rolling ECE={regime['early']['ece_rolling']:.4f}, "
          f"late rolling ECE={regime['late']['ece_rolling']:.4f}")

    # 4b.2 — bootstrap CI on default rolling ECE.
    t6 = time.time()
    boot = bootstrap_ece(y, p_rolling_default, N_BOOTSTRAP,
                         BOOTSTRAP_SEED, GATE)
    print(f"[4b.2] bootstrap {N_BOOTSTRAP}x in {time.time()-t6:.1f}s: "
          f"CI95=[{boot['ci_95_lo']:.4f},{boot['ci_95_hi']:.4f}] "
          f"P(<{GATE})={boot['p_below_gate']:.3f}")

    # 4b.3 — grid sweep.
    grid_rows = []
    t7 = time.time()
    for w in GRID_WINDOWS:
        for r in GRID_REFITS:
            tc = time.time()
            p_cfg = apply_rolling_per_rr(p_raw, y, rr_col, base_idx, w, r)
            ece = ece_20(y, p_cfg)
            grid_rows.append({
                "window": w,
                "refit_every": r,
                "ece_rolling": ece,
                "pass_gate": ece < GATE,
                "seconds": time.time() - tc,
            })
            print(f"[4b.3] w={w:>5} r={r:>4} -> ECE {ece:.4f} "
                  f"({'PASS' if ece < GATE else 'FAIL'}) "
                  f"[{time.time()-tc:.1f}s]")
    print(f"[4b.3] grid done in {time.time()-t7:.1f}s")

    pass_rate = sum(1 for g in grid_rows if g["pass_gate"]) / len(grid_rows)
    grid_min = min(g["ece_rolling"] for g in grid_rows)
    default_cell = next(g for g in grid_rows
                        if g["window"] == DEFAULT_WINDOW
                        and g["refit_every"] == DEFAULT_REFIT)
    default_delta_from_min = default_cell["ece_rolling"] - grid_min
    robust_default = default_delta_from_min < 0.002

    summary = {
        "script": "scripts/calibration/recal_robustness_v3.py",
        "test_parquet": str(TEST_PARQUET),
        "model": str(V3_BOOSTER),
        "n_base": int(len(df)),
        "n_expanded": int(len(exp)),
        "headline_ece": headline,
        "regime_4b1": regime,
        "bootstrap_4b2": boot,
        "grid_4b3": {
            "cells": grid_rows,
            "pass_rate": pass_rate,
            "grid_min_ece": grid_min,
            "default_cell_ece": default_cell["ece_rolling"],
            "default_delta_from_min": default_delta_from_min,
            "robust_default": robust_default,
            "pass_rate_target": 0.60,
            "pass_rate_ok": pass_rate >= 0.60,
        },
        "gates": {
            "regime_late_rolling_under_0_02":
                regime["late"]["ece_rolling"] < 0.02,
            "static_worse_post_break":
                regime["late"]["ece_static"] > regime["early"]["ece_static"],
            "bootstrap_p_below_gate_over_0_95":
                boot["p_below_gate"] > 0.95,
            "grid_pass_rate_over_0_60": pass_rate >= 0.60,
            "default_within_0_002_of_min": robust_default,
        },
        "runtime_seconds": time.time() - t0,
    }
    summary["all_gates_pass"] = all(summary["gates"].values())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[4b] saved {OUT_PATH}")
    print(f"[4b] all gates pass: {summary['all_gates_pass']}")


if __name__ == "__main__":
    main()
