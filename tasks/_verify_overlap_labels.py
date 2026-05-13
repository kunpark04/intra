"""Local verification of combo_overlap_labels.parquet."""
from __future__ import annotations
import sys
import pandas as pd

PATH = r"C:\Users\kunpa\Downloads\Projects\intra\data\ml\ranker_null\combo_overlap_labels.parquet"


def main() -> int:
    df = pd.read_parquet(PATH)
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")
    print()

    expected_cols = {"combo_id", "combo_id_int", "trades_in_training", "trades_in_training_bool", "overlap_pct"}
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    print(f"expected cols present: {missing == set()}")
    if missing:
        print(f"  MISSING: {missing}")
    if extra:
        print(f"  EXTRA: {extra}")
    print()

    # assertions
    n = len(df)
    n_true = int(df["trades_in_training_bool"].sum())
    n_false = int((~df["trades_in_training_bool"]).sum())
    overlap_min = float(df["overlap_pct"].min())
    overlap_max = float(df["overlap_pct"].max())
    overlap_median = float(df["overlap_pct"].median())
    overlap_mean = float(df["overlap_pct"].mean())
    total_trades = int(df["trades_in_training"].sum())

    print(f"row count: {n}  (expected 13814)  OK={n == 13814}")
    print(f"trades_in_training_bool TRUE: {n_true}  (expected 13814)  OK={n_true == 13814}")
    print(f"trades_in_training_bool FALSE: {n_false}  (expected 0)  OK={n_false == 0}")
    print(f"overlap_pct min: {overlap_min}")
    print(f"overlap_pct median: {overlap_median}")
    print(f"overlap_pct max: {overlap_max}")
    print(f"overlap_pct mean: {overlap_mean}")
    print(f"trades_in_training sum: {total_trades:,}")

    # head
    print()
    print("head:")
    print(df.head(5).to_string())
    print()
    print("describe (trades_in_training):")
    print(df["trades_in_training"].describe().to_string())

    # final assertion block
    ok = (
        n == 13814
        and n_true == 13814
        and n_false == 0
        and overlap_min >= 0.999
        and overlap_max <= 1.0001
        and missing == set()
    )
    print()
    print(f"ALL PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
