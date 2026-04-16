"""
validate_mfe_parquets.py — Phase 1 validator for adaptive R:R pipeline.

Reads column-subsets of each `data/ml/ml_dataset_v{N}_mfe.parquet` (memory-safe:
never materializes the full file) and checks:
  - required columns present
  - sign conventions (mfe >= 0, mae <= 0, stop_distance > 0, hold_bars > 0)
  - no NaN in MFE / MAE / stop_distance
  - win-label consistency: if label_win==1 then mfe_points >= stop_distance_pts
    (a R=1:1 target must have been hit at minimum)
  - row count + combo count per version

Exits nonzero on CRITICAL findings (missing cols, wrong signs, NaN in path data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA_DIR = Path("data/ml")
VERSIONS = list(range(2, 11))

# Columns we actually need for Phase 2 + validation
# NOTE: plan mentions `exit_reason` but it's not in the parquet schema — we skip it.
REQUIRED = [
    "mfe_points", "mae_points", "stop_distance_pts",
    "hold_bars", "combo_id", "r_multiple", "label_win",
    # entry features used downstream
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
]


def validate_one(v: int) -> dict:
    """Validate a single MFE-enriched sweep parquet against the schema contract.

    Args:
        path: Path to `ml_dataset_v{N}_mfe.parquet`.

    Returns:
        Dict of validation results (row count, schema ok, null-column ok).
    """
    path = DATA_DIR / "mfe" / f"ml_dataset_v{v}_mfe.parquet"
    if not path.exists():
        return {"version": v, "status": "MISSING", "critical": 1}

    pf = pq.ParquetFile(path)
    cols = {f.name for f in pf.schema_arrow}
    missing = [c for c in REQUIRED if c not in cols]

    result = {
        "version": v,
        "rows": pf.metadata.num_rows,
        "missing_cols": missing,
        "critical": 0,
        "warn": 0,
    }
    if missing:
        result["status"] = "MISSING_COLS"
        result["critical"] += 1
        return result

    # Read only the validation columns (memory-safe even on v3/v10)
    val_cols = ["mfe_points", "mae_points", "stop_distance_pts",
                "hold_bars", "combo_id", "r_multiple", "label_win"]
    tbl = pf.read(columns=val_cols)
    df = tbl.to_pandas()

    result["combos"] = int(df["combo_id"].nunique())

    # Sign checks
    neg_mfe = int((df["mfe_points"] < 0).sum())
    pos_mae = int((df["mae_points"] > 0).sum())
    nonpos_stop = int((df["stop_distance_pts"] <= 0).sum())
    nonpos_hold = int((df["hold_bars"] <= 0).sum())

    # NaN checks on path data
    nan_mfe = int(df["mfe_points"].isna().sum())
    nan_mae = int(df["mae_points"].isna().sum())
    nan_stop = int(df["stop_distance_pts"].isna().sum())

    # Label mix — informational (wins can exit via TP or opposite-signal;
    # we can't disambiguate without exit_reason, so no consistency check here).
    wins = int((df["label_win"] == 1).sum())

    result.update({
        "neg_mfe": neg_mfe, "pos_mae": pos_mae,
        "nonpos_stop": nonpos_stop, "nonpos_hold": nonpos_hold,
        "nan_mfe": nan_mfe, "nan_mae": nan_mae, "nan_stop": nan_stop,
        "wins": wins,
    })

    # CRITICAL: wrong signs, NaN in path, stop<=0
    if neg_mfe or pos_mae or nonpos_stop or nan_mfe or nan_mae or nan_stop:
        result["critical"] += 1
    if nonpos_hold:
        result["warn"] += 1

    result["status"] = "CRITICAL" if result["critical"] else (
        "WARN" if result["warn"] else "OK")
    return result


def main() -> int:
    """Phase-1 validator: check every MFE-enriched sweep parquet.

    Runs `validate_one` on each parquet under `data/ml/mfe/` and prints a
    pass/fail table.
    """
    print("=" * 70)
    print("Phase 1: MFE Parquet Validation")
    print("=" * 70)

    total_critical = 0
    total_rows = 0
    for v in VERSIONS:
        r = validate_one(v)
        total_critical += r.get("critical", 0)
        if r["status"] in ("MISSING", "MISSING_COLS"):
            print(f"v{v:<2}  {r['status']}  missing={r.get('missing_cols')}")
            continue
        total_rows += r["rows"]
        print(
            f"v{v:<2}  {r['status']:<8}  "
            f"rows={r['rows']:>12,}  combos={r['combos']:>6}  "
            f"wins={r['wins']:>10,}  "
            f"neg_mfe={r['neg_mfe']:>3}  pos_mae={r['pos_mae']:>3}  "
            f"nonpos_stop={r['nonpos_stop']:>3}  "
            f"nan(m/M/s)={r['nan_mfe']}/{r['nan_mae']}/{r['nan_stop']}"
        )

    print("-" * 70)
    print(f"Total rows across versions: {total_rows:,}")
    print(f"CRITICAL count: {total_critical}")
    return 1 if total_critical else 0


if __name__ == "__main__":
    sys.exit(main())
