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

def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    """Chronologically split a bar series into training and test partitions.

    The split preserves bar order — rows `[0, floor(N*train_ratio))` go to
    training; the remainder goes to test. Never shuffles.

    Args:
        df: Bar DataFrame, already time-sorted (e.g. from `load_bars`).
        train_ratio: Fraction of rows to assign to training. Default 0.8.

    Returns:
        Tuple `(train, test)` — both DataFrames with fresh RangeIndex.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test
