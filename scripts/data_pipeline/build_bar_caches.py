"""build_bar_caches.py - One-time cache of NQ_15min and NQ_1h parquets from NQ_1min.csv.

Reusable across sweeps and eval notebooks, so the CSV is loaded and synthetic
bars dropped exactly once. Also runs the three validations the Probe 1
preregistration calls out (round-trip, volume conservation, hand-check).

Usage:
    python scripts/data_pipeline/build_bar_caches.py

Outputs:
    data/NQ_15min.parquet
    data/NQ_1h.parquet

Re-running is safe and idempotent; existing parquets are overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data_loader import load_bars
from src.indicators.bar_resample import resample_bars


CSV_PATH = REPO / "data" / "NQ_1min.csv"
OUT_15 = REPO / "data" / "NQ_15min.parquet"
OUT_1H = REPO / "data" / "NQ_1h.parquet"


def _validate(df_1min: pd.DataFrame, df_15: pd.DataFrame, df_1h: pd.DataFrame) -> None:
    """Run the three validations from probe1_preregistration §C1."""
    # 1) Round-trip identity at 1min (NaN-aware; any NaN carries through)
    roundtrip = resample_bars(df_1min, "1min")
    assert len(roundtrip) == len(df_1min), (
        f"round-trip len mismatch: {len(roundtrip)} vs {len(df_1min)}"
    )
    for col in ("open", "high", "low", "close", "volume"):
        assert np.array_equal(
            roundtrip[col].to_numpy(),
            df_1min[col].to_numpy(),
            equal_nan=True,
        ), f"round-trip {col} diverged"
    print(f"[ok] round-trip 1min identity ({len(df_1min):,} rows)")

    # 2) Volume conservation across the kept windows. Because partial windows
    # are dropped, volume from those fragments is excluded; what we assert is
    # the equivalent-window volume sum.
    #
    # Derive per-1min which target windows it belongs to, then sum only rows
    # whose target window was kept (full-count) in the resampled output.
    for freq, df_out in (("15min", df_15), ("1h", df_1h)):
        starts = df_out["time"].to_numpy()
        kept_mask = df_1min["time"].dt.floor(freq).isin(starts)
        kept_vol_1min = df_1min.loc[kept_mask, "volume"].sum()
        kept_vol_out = df_out["volume"].sum()
        assert kept_vol_1min == kept_vol_out, (
            f"{freq} volume mismatch: 1min kept-sum {kept_vol_1min} "
            f"vs resampled sum {kept_vol_out}"
        )
        print(
            f"[ok] {freq} volume conservation "
            f"(kept {kept_mask.sum():,}/{len(df_1min):,} 1-min bars, "
            f"sum={int(kept_vol_out):,})"
        )

    # 3) Hand-check: pick 4 consecutive 15m windows, verify OHLCV
    sample = df_15.iloc[1000:1004].copy()
    for _, row in sample.iterrows():
        start = row["time"]
        end = start + pd.Timedelta(minutes=15)
        window = df_1min[(df_1min["time"] >= start) & (df_1min["time"] < end)]
        assert len(window) == 15, f"window at {start} has {len(window)} bars, expected 15"
        assert row["open"] == window["open"].iloc[0], f"open mismatch at {start}"
        assert row["high"] == window["high"].max(), f"high mismatch at {start}"
        assert row["low"] == window["low"].min(), f"low mismatch at {start}"
        assert row["close"] == window["close"].iloc[-1], f"close mismatch at {start}"
        assert row["volume"] == window["volume"].sum(), f"volume mismatch at {start}"
    print(f"[ok] hand-check 4 consecutive 15m windows (starting {sample['time'].iloc[0]})")


def main() -> None:
    print(f"[load] {CSV_PATH}")
    df_1min = load_bars(CSV_PATH)
    n_raw = len(df_1min)
    bad_time = df_1min["time"].isna()
    if bad_time.any():
        print(f"[load] dropping {int(bad_time.sum()):,} rows with NaT time")
        df_1min = df_1min.loc[~bad_time].reset_index(drop=True)
    # Drop any remaining rows with NaN OHLCV (partial records) — resample would
    # otherwise let them corrupt whichever window they fall into.
    nan_ohlcv = df_1min[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    if nan_ohlcv.any():
        print(f"[load] dropping {int(nan_ohlcv.sum()):,} rows with NaN OHLCV")
        df_1min = df_1min.loc[~nan_ohlcv].reset_index(drop=True)
    print(f"[load] {len(df_1min):,} 1-min bars (from {n_raw:,} raw), "
          f"range {df_1min['time'].iloc[0]} -> {df_1min['time'].iloc[-1]}")

    df_15 = resample_bars(df_1min, "15min")
    df_1h = resample_bars(df_1min, "1h")
    print(f"[resample] 15min -> {len(df_15):,} bars")
    print(f"[resample] 1h    -> {len(df_1h):,} bars")

    _validate(df_1min, df_15, df_1h)

    df_15.to_parquet(OUT_15, index=False)
    df_1h.to_parquet(OUT_1H, index=False)
    print(f"[write] {OUT_15.relative_to(REPO)}")
    print(f"[write] {OUT_1H.relative_to(REPO)}")


if __name__ == "__main__":
    main()
