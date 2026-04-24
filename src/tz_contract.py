"""Source-TZ contract for NQ bar timestamps.

Timestamps in `data/NQ_*.parquet` are naive Central Time (Barchart
vendor — see `scripts/data_pipeline/update_bars_yfinance.py:37`).
Call `assert_naive_ct(df)` at every parquet load in TZ-sensitive
code and `localize_ct_to_et(series)` for ET-minute math.

Context: `lessons.md` 2026-04-23 `tz_bug_in_session_decomposition`.
"""
from __future__ import annotations

from typing import Union

import pandas as pd


SOURCE_TZ = "America/Chicago"
ET_TZ = "America/New_York"

_MIN_PLAUSIBLE_YEAR = 2005  # NQ went electronic in 1999; 2005+ covers the modern liquid regime


def assert_naive_ct(df: pd.DataFrame, time_col: str = "time") -> None:
    """Assert df[time_col] is tz-naive with a plausible year.

    Mechanical guard: does not detect semantic TZ shifts (e.g. someone
    converted CT→UTC then stripped tz). Vendor contract at
    update_bars_yfinance.py:37 is the sole source of truth upstream.
    Raises AssertionError with an actionable message on violation.
    """
    if time_col not in df.columns:
        raise AssertionError(
            f"assert_naive_ct: column {time_col!r} not found (columns: {list(df.columns)})"
        )
    col = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(col):
        col = pd.to_datetime(col)
    tz = getattr(col.dt, "tz", None)
    if tz is not None:
        raise AssertionError(
            f"assert_naive_ct: column {time_col!r} is tz-aware (tz={tz!r}); "
            f"vendor NQ bars are naive Central Time. Do NOT call "
            f"tz_localize('UTC') on this data. Use src.tz_contract.localize_ct_to_et() "
            f"for ET math. See memory/feedback_tz_source_ct.md."
        )
    if len(col) == 0:
        raise AssertionError("assert_naive_ct: empty time column")
    first = col.iloc[0]
    first_year = first.year
    now_year = pd.Timestamp.now("UTC").year
    if first_year < _MIN_PLAUSIBLE_YEAR:
        raise AssertionError(
            f"assert_naive_ct: first bar {first} is before {_MIN_PLAUSIBLE_YEAR}; "
            f"vendor NQ bars should post-date electronic-trading adoption."
        )
    if first_year > now_year:
        raise AssertionError(
            f"assert_naive_ct: first bar {first} is in the future ({first_year} > {now_year}); "
            f"file is corrupt or wall-clock skew."
        )


def localize_ct_to_et(ts: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    """Canonical CT-naive → ET-aware conversion for NQ bar timestamps.

    Uses `ambiguous="infer"` / `nonexistent="shift_forward"` to pass
    DST transitions without raising. Input must be tz-naive (raises
    AssertionError otherwise — run `assert_naive_ct` first).
    """
    series = pd.to_datetime(ts if isinstance(ts, pd.Series) else pd.Series(ts))
    tz = getattr(series.dt, "tz", None)
    if tz is not None:
        raise AssertionError(
            f"localize_ct_to_et: input is tz-aware (tz={tz!r}); "
            f"this helper expects naive CT. Run assert_naive_ct() first."
        )
    return (
        series.dt.tz_localize(SOURCE_TZ, ambiguous="infer", nonexistent="shift_forward")
              .dt.tz_convert(ET_TZ)
    )


def _self_test() -> int:
    """Validate the harness against the three NQ parquets shipped in data/.

    Return 0 on success, 1 on failure. Run as `python src/tz_contract.py`.
    """
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    parquets = [data_dir / f"NQ_{tf}.parquet" for tf in ("1min", "15min", "1h")]

    failures: list[str] = []

    for p in parquets:
        if not p.exists():
            print(f"[tz_contract] SKIP  {p.name:18s} (not present)")
            continue
        try:
            df = pd.read_parquet(p, columns=["time"])
            assert_naive_ct(df, "time")
            et = localize_ct_to_et(df["time"].iloc[:5])
            assert et.dt.tz is not None, "localize_ct_to_et returned naive output"
            print(
                f"[tz_contract] OK    {p.name:18s} "
                f"(n={len(df):>10,}, first={df['time'].iloc[0]}, last={df['time'].iloc[-1]})"
            )
        except Exception as exc:  # noqa: BLE001 — diagnostics
            failures.append(f"{p.name}: {exc}")
            print(f"[tz_contract] FAIL  {p.name:18s} — {exc}")

    if failures:
        print(f"\n[tz_contract] {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[tz_contract] all NQ parquets pass source-TZ contract.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
