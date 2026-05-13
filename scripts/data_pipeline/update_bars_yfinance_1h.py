"""update_bars_yfinance_1h.py — extend `data/NQ_1h.parquet` with fresh yfinance bars.

Pulls NQ=F 1h bars from yfinance starting just after the last bar in the
existing parquet, converts UTC-aware -> naive CT (per LOCAL_TZ vendor marker
at `update_bars_yfinance.py:37`), aligns to the parquet schema, de-dupes,
recomputes `session_break`, and atomically replaces the parquet.

This is the 1h sibling to `update_bars_yfinance.py` (which targets 1m bars
and is bound by yfinance's 7-day 1m historical cap). yfinance allows 730
days of history at the **1h** interval, so this script bridges gaps of
arbitrary size up to that ceiling — it does NOT have the 7-day-cadence
constraint of the 1m pipeline.

Cadence: run whenever a fresh paper-trade backfill is desired. Append-only;
historical bars are not modified.

Usage (from repo root):
    python scripts/data_pipeline/update_bars_yfinance_1h.py
    python scripts/data_pipeline/update_bars_yfinance_1h.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = REPO / "data" / "NQ_1h.parquet"
TICKER = "NQ=F"
LOCAL_TZ = "America/Chicago"  # vendor marker, matches update_bars_yfinance.py:37

sys.path.insert(0, str(REPO))
from src.tz_contract import assert_naive_ct


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + report counts but do not modify the parquet.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load existing parquet + validate source-TZ contract
    df_old = pd.read_parquet(PARQUET_PATH)
    assert_naive_ct(df_old, time_col="time")
    last_ts_ct = pd.Timestamp(df_old["time"].max())
    print(f"[update_bars_1h] existing: {len(df_old):,} bars  "
          f"first={df_old['time'].min()}  last={last_ts_ct} (CT-naive)")

    # 2. Compute UTC range to query yfinance.
    # Widen start by 24h before last_ts_utc — yfinance's 1h historical endpoint
    # silently drops bars near a mid-day `start=` boundary (observed: 7 bars
    # missed at 17:00-23:00 CT on the seam day when start was the next-bar
    # timestamp). De-dupe in step 6 safely discards the overlap, so widening
    # is free; the alternative is a 5-7h gap right at the splice point.
    last_ts_utc = last_ts_ct.tz_localize(LOCAL_TZ).tz_convert("UTC").tz_localize(None)
    start_utc = last_ts_utc - pd.Timedelta(hours=24)
    now_utc = pd.Timestamp.now("UTC").tz_localize(None)
    if start_utc >= now_utc:
        print(f"[update_bars_1h] no new bars to pull "
              f"(start_utc={start_utc} >= now_utc={now_utc})")
        return

    gap_days = (now_utc - start_utc).total_seconds() / 86400
    print(f"[update_bars_1h] yfinance NQ=F 1h: {start_utc} -> {now_utc} UTC  "
          f"(gap~{gap_days:.1f}d)")

    # 3. Pull yfinance
    raw = yf.download(TICKER, interval="1h", start=start_utc, end=now_utc,
                      auto_adjust=False, progress=False, prepost=False)
    if raw is None or raw.empty:
        print("[update_bars_1h] yfinance returned no bars")
        return

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # 4. UTC-aware -> naive CT
    raw = raw.reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else raw.columns[0]
    times_utc = pd.to_datetime(raw[time_col])
    if times_utc.dt.tz is None:
        times_utc = times_utc.dt.tz_localize("UTC")
    times_ct = times_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # 5. Build schema-aligned new-bars frame
    new = pd.DataFrame({
        "time": times_ct,
        "open": raw["Open"].astype(float),
        "high": raw["High"].astype(float),
        "low": raw["Low"].astype(float),
        "close": raw["Close"].astype(float),
        "volume": raw["Volume"].astype(float),
        "session_break": False,
    })
    # Honor any extra columns the parquet might carry (filled with NA)
    for c in df_old.columns:
        if c not in new.columns:
            new[c] = pd.NA

    # 6. De-dupe against existing rows
    new = new[new["time"] > last_ts_ct].copy().reset_index(drop=True)
    if len(new) == 0:
        print("[update_bars_1h] all yfinance bars are duplicates of existing parquet")
        return
    print(f"[update_bars_1h] yfinance returned {len(new):,} fresh bars  "
          f"first={new['time'].iloc[0]}  last={new['time'].iloc[-1]} (CT-naive)")

    # 7. Concat + sort + recompute session_break (time-gap heuristic)
    combined = pd.concat([df_old, new[df_old.columns]], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    deltas = combined["time"].diff()
    combined["session_break"] = (deltas > pd.Timedelta(hours=1.5)).fillna(False)

    # 8. Validate post-write
    assert_naive_ct(combined, time_col="time")
    assert combined["time"].is_monotonic_increasing, "time column not sorted"
    assert not combined["time"].duplicated().any(), "duplicate timestamps after concat"

    if args.dry_run:
        print(f"[update_bars_1h] DRY-RUN — would write {len(combined):,} rows "
              f"(={len(df_old):,} existing + {len(new):,} new)")
        return

    # 9. Atomic write
    tmp_path = PARQUET_PATH.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(PARQUET_PATH)
    print(f"[update_bars_1h] wrote {len(combined):,} rows ({len(new):,} new)  "
          f"-> {PARQUET_PATH}")
    print(f"[update_bars_1h] new last bar: {combined['time'].iloc[-1]} (CT-naive)")


if __name__ == "__main__":
    main()
