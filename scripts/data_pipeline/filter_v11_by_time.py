"""filter_v11_by_time.py — temporal partition for V4 refit null-test.

Reads the full v11 sweep parquet and writes a new parquet containing only
trades whose `exit_time` is strictly before the cutoff (default 2024-01-01).
Streams via pyarrow row-group batches; peak RAM ≈ one batch.

Used once by Phase 1 of the random-K null test plan to produce
`data/ml/originals/ml_dataset_v11_pre2024.parquet`, which feeds a clean V4
refit that has not seen any 2024+ trade (preventing the V4 in-partition
leak flagged in LLM council peer review 2026-04-20).

Usage:
    python scripts/data_pipeline/filter_v11_by_time.py \
        --input  data/ml/originals/ml_dataset_v11.parquet \
        --output data/ml/originals/ml_dataset_v11_pre2024.parquet \
        --cutoff 2024-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal filter for v11 parquet")
    p.add_argument("--input", required=True,
                   help="Path to source parquet (e.g. data/ml/originals/ml_dataset_v11.parquet)")
    p.add_argument("--output", required=True,
                   help="Path to filtered output parquet")
    p.add_argument("--cutoff", default="2024-01-01",
                   help="ISO date; rows with exit_time < cutoff are kept (default 2024-01-01)")
    p.add_argument("--time-col", default="exit_time",
                   help="Column to filter on (default exit_time)")
    p.add_argument("--batch-size", type=int, default=65536,
                   help="pyarrow iter_batches size (default 65536)")
    p.add_argument("--max-null-frac", type=float, default=0.01,
                   help="Abort if null time fraction exceeds this (default 0.01)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        sys.exit(f"[filter] FATAL: source parquet not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cutoff = pd.Timestamp(args.cutoff)
    print(f"[filter] src     = {src}")
    print(f"[filter] dst     = {dst}")
    print(f"[filter] cutoff  = {cutoff.isoformat()} (column: {args.time_col})")

    pf = pq.ParquetFile(src)
    total_rows = pf.metadata.num_rows
    schema = pf.schema_arrow
    if args.time_col not in schema.names:
        sys.exit(f"[filter] FATAL: {args.time_col} not in parquet schema. "
                 f"Available: {schema.names[:10]}...")

    print(f"[filter] total source rows: {total_rows:,}")

    writer: pq.ParquetWriter | None = None
    rows_kept = 0
    rows_dropped_future = 0
    rows_null_time = 0
    rows_scanned = 0

    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            rows_scanned += len(df)

            t = pd.to_datetime(df[args.time_col], errors="coerce")
            null_mask = t.isna()
            rows_null_time += int(null_mask.sum())

            keep_mask = (~null_mask) & (t < cutoff)
            rows_kept += int(keep_mask.sum())
            rows_dropped_future += int((~null_mask & (t >= cutoff)).sum())

            if keep_mask.any():
                kept = df.loc[keep_mask]
                table = pa.Table.from_pandas(kept, preserve_index=False,
                                             schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(dst, schema, compression="snappy")
                writer.write_table(table)

            if rows_scanned % (args.batch_size * 50) == 0:
                print(f"[filter] scanned={rows_scanned:,} kept={rows_kept:,} "
                      f"dropped_future={rows_dropped_future:,} null={rows_null_time:,}")

    finally:
        if writer is not None:
            writer.close()

    null_frac = rows_null_time / max(1, rows_scanned)
    print("-" * 60)
    print(f"[filter] rows scanned      : {rows_scanned:,}")
    print(f"[filter] rows kept (<cutoff): {rows_kept:,}")
    print(f"[filter] rows dropped (>=cutoff): {rows_dropped_future:,}")
    print(f"[filter] rows with null time    : {rows_null_time:,} "
          f"({null_frac:.4%})")
    print(f"[filter] kept fraction     : {rows_kept / max(1, rows_scanned):.4%}")

    if null_frac > args.max_null_frac:
        sys.exit(f"[filter] FATAL: null time fraction {null_frac:.4%} exceeds "
                 f"--max-null-frac {args.max_null_frac:.4%}. "
                 f"Investigate before trusting the partition.")

    if rows_kept == 0:
        sys.exit("[filter] FATAL: zero rows kept — cutoff or time column wrong?")

    out_sz = dst.stat().st_size / 1e9
    print(f"[filter] wrote {dst} ({out_sz:.2f} GB)")


if __name__ == "__main__":
    main()
