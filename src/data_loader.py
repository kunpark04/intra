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
    """Load NQ 1-min CSV, normalize columns, drop synthetic bars, sort by time."""
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
    """Chronological split — no shuffling."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test
