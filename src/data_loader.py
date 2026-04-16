"""CSV ingestion + chronological train/test split for NQ 1-minute bars.

Normalises the provided dataset's column casing (`Time` → `time`, `Latest` →
`close`, etc.) and drops synthetic gap-fill bars so downstream indicators see
only real price action. The train/test split is time-ordered — never random.
"""
import pandas as pd
from pathlib import Path

RENAME_MAP = {
    "Time": "time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Latest": "close",
    "Volume": "volume",
    "session_break": "session_break",
}
KEEP_COLS = list(RENAME_MAP.values())

def load_bars(csv_path) -> pd.DataFrame:
    """Load NQ 1-minute bars from CSV into the canonical schema.

    Renames raw columns to lowercase canonical names, drops any rows flagged
    `synthetic` (gap-fill artifacts with no real price action), coerces
    `session_break` to bool, and sorts by timestamp.

    Args:
        csv_path: Path to the NQ 1-minute CSV. Must contain a `Time` column
            parseable as datetime plus OHLCV + `session_break`.

    Returns:
        DataFrame with columns `time, open, high, low, close, volume,
        session_break`, sorted ascending by `time` and with a fresh RangeIndex.
    """
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.rename(columns=RENAME_MAP)
    # Drop synthetic bars (gap-fill artifacts — no real price action)
    if "synthetic" in df.columns:
        df = df[df["synthetic"] != True].copy()
    df = df[KEEP_COLS].copy()
    df["session_break"] = df["session_break"].astype(bool)
    df = df.sort_values("time").reset_index(drop=True)
    return df

def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8,
                     train_end=None):
    """Chronologically split a bar series into training and test partitions.

    Two modes — never shuffles:
      • `train_end` provided (preferred for production eval): training is
        rows where `time <= train_end`; test is the remainder. Locks the
        boundary even as new bars are appended over time.
      • `train_end` is None: rows `[0, floor(N*train_ratio))` go to training;
        the remainder goes to test.

    Args:
        df: Bar DataFrame, already time-sorted (e.g. from `load_bars`).
        train_ratio: Fraction of rows to assign to training when
            `train_end` is None. Default 0.8.
        train_end: Optional timestamp (string or `pd.Timestamp`) that
            freezes the training partition's right edge. When set,
            overrides `train_ratio`.

    Returns:
        Tuple `(train, test)` — both DataFrames with fresh RangeIndex.
    """
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        train = df[df["time"] <= cutoff].reset_index(drop=True)
        test = df[df["time"] > cutoff].reset_index(drop=True)
        return train, test

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test
