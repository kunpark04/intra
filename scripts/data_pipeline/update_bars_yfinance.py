"""Append the latest NQ 1-minute bars from yfinance to `data/NQ_1min.csv`.

Reads the existing CSV, finds the most recent real (non-footer) timestamp,
fetches `NQ=F` 1-minute bars from yfinance for the gap up to "now"
(yfinance limits 1-minute history to ~7 days), normalises into the
canonical Barchart-export schema, de-dupes against existing rows, and
appends. Strips any trailing Barchart provenance footer line before append.

Run cadence ≤ weekly to avoid coverage holes (yfinance's 7-day 1m limit).

The frozen training cutoff (`src.config.TRAIN_END_FROZEN`) is unaffected;
every appended bar lands in the test partition consumed by the evaluation
notebooks.

Usage:
    python scripts/data_pipeline/update_bars_yfinance.py
    python scripts/data_pipeline/update_bars_yfinance.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance",
          file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO / "data" / "NQ_1min.csv"
TICKER = "NQ=F"
LOCAL_TZ = "America/Chicago"  # CSV timestamps are CT (Barchart export)
SESSION_GAP_THRESHOLD = pd.Timedelta(minutes=5)

CSV_COLS = [
    "Symbol", "Time", "Open", "High", "Low", "Latest",
    "Change", "%Change", "Volume", "Open Int", "synthetic", "session_break",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI args (`--dry-run` to preview without writing)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + report counts but do not modify the CSV.")
    return ap.parse_args()


def _read_existing_last_ts(csv_path: Path) -> pd.Timestamp | None:
    """Return the most recent real (non-footer, parseable) timestamp."""
    df = pd.read_csv(csv_path, usecols=["Time"])
    times = pd.to_datetime(df["Time"], errors="coerce").dropna()
    return None if times.empty else times.max()


def _fetch_yf(start: pd.Timestamp) -> pd.DataFrame:
    """Pull 1-minute `NQ=F` bars from yfinance from `start+1m` to now.

    Returns a DataFrame with columns `time, Open, High, Low, Close,
    Volume`, where `time` is naive CT. Empty if the gap is fully outside
    yfinance's 7-day 1m window.
    """
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    yf_window_start_utc = now_utc - pd.Timedelta(days=7)

    last_ct = pd.Timestamp(start).tz_localize(LOCAL_TZ)
    last_utc = last_ct.tz_convert("UTC").tz_localize(None)
    eff_start_utc = max(last_utc + pd.Timedelta(minutes=1), yf_window_start_utc)
    if eff_start_utc >= now_utc:
        return pd.DataFrame()

    df = yf.download(
        tickers=TICKER,
        interval="1m",
        start=eff_start_utc,
        end=now_utc,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(LOCAL_TZ).tz_localize(None)
    df = df.reset_index().rename(columns={df.index.name or "Datetime": "time"})
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    return df


def _to_csv_rows(yf_df: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance OHLCV → Barchart-schema rows ready to append.

    `Symbol`, `Change`, `%Change`, `Open Int` are left blank (load_bars
    drops them). `session_break` is recomputed from inter-bar gaps
    (>5 minutes = break).
    """
    out = pd.DataFrame()
    out["Symbol"] = ""
    out["Time"] = yf_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["Open"] = yf_df["Open"].astype(float).round(2)
    out["High"] = yf_df["High"].astype(float).round(2)
    out["Low"] = yf_df["Low"].astype(float).round(2)
    out["Latest"] = yf_df["Close"].astype(float).round(2)
    out["Change"] = ""
    out["%Change"] = ""
    out["Volume"] = yf_df["Volume"].astype(float).fillna(0).round(0)
    out["Open Int"] = ""
    out["synthetic"] = False
    out["session_break"] = False

    times = pd.to_datetime(out["Time"])
    gaps = times.diff() > SESSION_GAP_THRESHOLD
    out.loc[gaps, "session_break"] = True
    return out[CSV_COLS]


def _strip_footer(csv_path: Path) -> None:
    """Remove any trailing Barchart provenance line (`Downloaded from...`)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    keep = [ln for ln in lines if not ln.startswith("Downloaded from")]
    if len(keep) != len(lines):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(keep)


def main() -> None:
    """Fetch new yfinance bars and append them to `data/NQ_1min.csv`."""
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        sys.exit(1)

    last_ts = _read_existing_last_ts(CSV_PATH)
    if last_ts is None:
        print("ERROR: existing CSV has no parseable timestamps", file=sys.stderr)
        sys.exit(1)
    print(f"[update] existing last bar: {last_ts}")

    yf_df = _fetch_yf(last_ts)
    if yf_df.empty:
        print("[update] no new bars from yfinance "
              "(already up to date or beyond 7-day 1m window)")
        return

    print(f"[update] fetched {len(yf_df):,} candidate bars from yfinance "
          f"({yf_df['time'].iloc[0]} -> {yf_df['time'].iloc[-1]})")

    new_rows = _to_csv_rows(yf_df)
    new_times = pd.to_datetime(new_rows["Time"])
    new_rows = new_rows[new_times > last_ts].reset_index(drop=True)
    if new_rows.empty:
        print("[update] all fetched bars already present; nothing to append")
        return

    print(f"[update] {len(new_rows):,} new bars after de-dupe")
    if args.dry_run:
        print("[update] dry-run: not modifying CSV")
        print(new_rows.head().to_string(index=False))
        print("...")
        print(new_rows.tail().to_string(index=False))
        return

    _strip_footer(CSV_PATH)
    new_rows.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"[update] appended {len(new_rows):,} bars; new last bar: "
          f"{new_rows['Time'].iloc[-1]}")


if __name__ == "__main__":
    main()
