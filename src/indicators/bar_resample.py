"""bar_resample.py - Aggregate 1-minute OHLCV bars to coarser timeframes.

Left-anchored windows (``time`` = window start) preserve the backtest engine's
``next_bar_open`` fill semantics at any bar width: a signal on the window ending
at T fills at the following window's open, just as 1-min bars do today.

Partial windows (fewer constituent 1-minute bars than the frequency expects)
are dropped. This removes both series-boundary fragments and any window that
straddles a trading halt — a 15m bar built from 8 of 15 expected minutes is
not a 15-minute observation, its ``open`` would be 3 minutes inside the
nominal window instead of at the window edge.
"""
from __future__ import annotations

import pandas as pd


_FREQ_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "4h": 240,
}

_REQUIRED_COLS = ("time", "open", "high", "low", "close", "volume", "session_break")


def resample_bars(df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate 1-minute bars to a coarser timeframe with left-anchored labels.

    Aggregation rules:
      - ``open``   = first in window
      - ``high``   = max
      - ``low``    = min
      - ``close``  = last
      - ``volume`` = sum
      - ``session_break`` = any (carry the flag forward if any constituent bar has it)

    Windows are left-closed and left-labeled: a 15m window at 09:00 covers
    ``[09:00, 09:15)``. The returned ``time`` column is the window start — this
    matches the backtest engine's ``next_bar_open`` fill model, so the same
    bar-T-fills-at-T+1 convention holds for 15m and 1h without any engine edit.

    Partial windows (group size != expected 1-min count) are removed.

    Args:
        df_1min: DataFrame with columns ``time, open, high, low, close, volume,
            session_break``. ``time`` must be datetime and pre-sorted ascending
            (``load_bars`` guarantees this).
        freq: Target frequency. One of ``1min, 5min, 15min, 30min, 1h, 4h``.
            Gated to this list so callers can't accidentally request an exotic
            pandas alias with different semantics.

    Returns:
        DataFrame with the same seven columns, fresh RangeIndex, sorted by
        ``time``. Passing ``freq='1min'`` returns a validated copy.
    """
    if freq not in _FREQ_MINUTES:
        raise ValueError(
            f"Unsupported freq '{freq}'. Supported: {sorted(_FREQ_MINUTES)}"
        )

    missing = [c for c in _REQUIRED_COLS if c not in df_1min.columns]
    if missing:
        raise ValueError(f"df_1min missing required columns: {missing}")

    if freq == "1min":
        return df_1min.loc[:, list(_REQUIRED_COLS)].reset_index(drop=True).copy()

    expected = _FREQ_MINUTES[freq]

    g = df_1min.set_index("time").resample(freq, label="left", closed="left")
    out = g.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "session_break": "any",
    })
    counts = g.size()
    out = out[counts == expected].copy()

    out = out.reset_index()
    out["session_break"] = out["session_break"].astype(bool)
    out = out[list(_REQUIRED_COLS)].reset_index(drop=True)
    return out
