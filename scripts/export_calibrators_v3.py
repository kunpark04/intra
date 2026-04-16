"""Export two-stage per-combo calibrators for V3 production stack.

Fits two sets of isotonic knots on training data:
  Stage 1: Per-combo R:R-agnostic isotonic (for combos with >= MIN_COMBO rows)
  Stage 2: Pooled per-R:R isotonic (fallback for small combos + warmup)

Uses the production V3 booster to predict on training MFE parquets, then fits
IsotonicRegression on (p_raw, would_win) for each combo (across all R:R)
and per-R:R pooled.

Output: data/ml/adaptive_rr_v3/per_combo_calibrators_v3.json
  {
    "per_combo": { "v10_123": {"X": [...], "y": [...]}, ... },
    "pooled_per_rr": { "1.00": {"X": [...], "y": [...]}, ... },
    "meta": { "n_combos": ..., "n_fallback": ..., ... }
  }

Run on sweep-runner-1 (needs ~6GB RAM for MFE loading + expansion).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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
    expand_rr_levels,
)
from adaptive_rr_model_v2 import (  # noqa: E402
    load_mfe_parquets,
    filter_combos,
)

from sklearn.isotonic import IsotonicRegression

V3_DIR = REPO / "data/ml/adaptive_rr_v3"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
OUT_PATH = V3_DIR / "per_combo_calibrators_v3.json"

# Minimum rows per combo to fit a per-combo calibrator (matching Phase 4f).
MIN_COMBO_ROWS = 300
# Training data loading parameters (matching V3 training defaults).
TARGET_BASE_TRADES = 1_200_000
MAX_ROWS = 10_000_000
MIN_TRADES_PER_COMBO = 30
VERSIONS = list(range(2, 11))


def main() -> None:
    t0 = time.time()

    # ── Load training data (same pipeline as V3 training) ────────────────
    print("[export] Loading training MFE parquets...")
    df = load_mfe_parquets(VERSIONS, TARGET_BASE_TRADES)
    df = filter_combos(df, MIN_TRADES_PER_COMBO)

    if "label_win" not in df.columns or "r_multiple" not in df.columns:
        sys.exit("[export] FATAL: MFE parquets missing label_win/r_multiple.")

    print(f"[export] Computing Family A features on {len(df):,} base trades...")
    df = add_family_a(df)

    print("[export] Expanding R:R levels...")
    expanded = expand_rr_levels(df, RR_LEVELS, MAX_ROWS)

    # Save combo_id before it becomes categorical in the feature frame.
    combo_col = expanded[ID_FEATURE].astype(str).to_numpy()
    rr_col = expanded[RR_FEATURE].to_numpy(dtype=np.float32)
    y = expanded["would_win"].to_numpy(dtype=np.int8)

    # ── Raw predictions ──────────────────────────────────────────────────
    print("[export] Loading V3 booster and predicting...")
    booster = lgb.Booster(model_file=str(V3_BOOSTER))
    t1 = time.time()
    p_raw = booster.predict(expanded[ALL_FEATURES], num_threads=4).astype(np.float64)
    print(f"[export] Predicted {len(p_raw):,} rows in {time.time()-t1:.1f}s")

    del expanded  # free memory

    # ── Stage 1: Per-combo R:R-agnostic isotonic ─────────────────────────
    print("[export] Fitting per-combo isotonic calibrators...")
    t2 = time.time()
    uniq_combos, inv = np.unique(combo_col, return_inverse=True)
    per_combo_knots: dict[str, dict] = {}
    n_fallback = 0

    for i, combo in enumerate(uniq_combos):
        m = inv == i
        n = int(m.sum())
        if n < MIN_COMBO_ROWS:
            n_fallback += 1
            continue
        yi, pi = y[m], p_raw[m]
        if len(np.unique(yi)) < 2:
            n_fallback += 1
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pi, yi)
        per_combo_knots[str(combo)] = {
            "X": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }

    n_combos_fitted = len(per_combo_knots)
    print(f"[export] Fitted {n_combos_fitted} per-combo calibrators, "
          f"{n_fallback} fallback ({time.time()-t2:.1f}s)")

    # ── Stage 2: Pooled per-R:R isotonic (fallback) ──────────────────────
    print("[export] Fitting pooled per-R:R isotonic calibrators...")
    t3 = time.time()
    pooled_knots: dict[str, dict] = {}

    for rr_val in RR_LEVELS:
        m = np.isclose(rr_col, rr_val)
        n = int(m.sum())
        if n < 100:
            continue
        yi, pi = y[m], p_raw[m]
        if len(np.unique(yi)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pi, yi)
        pooled_knots[f"{rr_val:.2f}"] = {
            "X": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }

    print(f"[export] Fitted {len(pooled_knots)} pooled per-R:R calibrators "
          f"({time.time()-t3:.1f}s)")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "per_combo": per_combo_knots,
        "pooled_per_rr": pooled_knots,
        "meta": {
            "n_combos_fitted": n_combos_fitted,
            "n_combos_fallback": n_fallback,
            "n_pooled_rr": len(pooled_knots),
            "min_combo_rows": MIN_COMBO_ROWS,
            "n_rows_total": int(len(p_raw)),
            "rr_levels": RR_LEVELS,
            "runtime_seconds": time.time() - t0,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[export] Saved {OUT_PATH} "
          f"({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"[export] Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
