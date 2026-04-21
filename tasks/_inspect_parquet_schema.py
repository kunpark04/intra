"""Inspect the 15m and 1h parquet schemas — column list + a row sample.

Goal: confirm `gross_pnl_dollars`, friction column, and any time/date column
we can use to infer `YEARS_SPAN_TRAIN` directly rather than hardcoding 5.8056.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for tf in ("15m", "1h"):
    p = REPO / "data" / "ml" / "originals" / f"ml_dataset_v11_{tf}.parquet"
    df = pd.read_parquet(p)
    print(f"\n=== {tf} ===  shape={df.shape}  unique_combos={df['combo_id'].nunique()}")
    print("columns:")
    for c in df.columns:
        dt = df[c].dtype
        print(f"  {c}: {dt}")
    print("sample row:")
    r = df.iloc[0]
    for c in ("gross_pnl_dollars", "friction_dollars", "net_pnl_dollars",
              "entry_bar_idx", "entry_timing_offset", "fill_slippage_ticks",
              "cooldown_after_exit_bars", "combo_id"):
        if c in df.columns:
            print(f"  {c}={r[c]!r}")
