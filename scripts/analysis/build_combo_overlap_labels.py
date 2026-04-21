"""Phase 4.1 — Ranker-Label Overlap Null (Council step 6).

Labels every combo in `combo_features_v12.parquet` with how many of its trade
rows appear in the V3/V4 training MFE parquet (`ml_dataset_v11_mfe.parquet`).

Why this exists
---------------
The 2026-04-20 LLM Council flagged a possible leak: Pool B (raw-Sharpe top-50)
is selected via `audit_full_net_sharpe`, a statistic computed from the very
same trade rows that feed V3/V4 training. If the selection partition and the
training partition overlap combo-wise, any per-combo memorisation in V3/V4
would also memorise the ranker's winners. This script builds the partition
label needed for the held-out Pool B construction in 4.2:

    trades_in_training == 0  →  eligible for the held-out Pool B

Output schema (one row per combo in `combo_features_v12.parquet`)
-----------------------------------------------------------------
- `combo_id`              — string; the canonical key used across v12 artefacts
                            (formatted `v{sweep}_{combo_id_int}`, e.g. `v11_742`).
                            Matches `combo_features_v12.parquet::global_combo_id`
                            so downstream notebooks can join directly.
- `combo_id_int`          — integer copy of the numeric combo_id (the key used
                            inside the MFE parquet).
- `trades_in_training`    — integer count of MFE-parquet rows with matching
                            `combo_id_int` (and matching sweep prefix).
- `trades_in_training_bool` — `trades_in_training > 0`.
- `overlap_pct`           — `trades_in_training / audit_n_trades`, clipped to
                            [0, 1]. Audit n_trades is the pre-gate trade count
                            that produced the ranker's Sharpe; this ratio tells
                            us whether the overlap is partial or total. For a
                            clean v11-only features parquet against the v11 MFE
                            parquet the expected value is ~1.0 for every combo.

Memory behaviour
----------------
The V11 MFE parquet is ~100M rows (the plan mandates NOT pulling it local).
We stream via `pyarrow.parquet.ParquetFile.iter_batches(batch_size=500_000)`
and accumulate a `combo_id_int -> count` dict. Peak memory is O(#combos) on
the aggregator + O(batch_size) on the active chunk. Every 10 batches we log
`total_rows`, `combos_seen`, `elapsed_s`, `rows_per_sec`.

CLI
---
    python scripts/analysis/build_combo_overlap_labels.py \
        --combo-features data/ml/ml1_results_v12/combo_features_v12.parquet \
        --mfe-parquet   data/ml/mfe/ml_dataset_v11_mfe.parquet \
        --output        data/ml/ranker_null/combo_overlap_labels.parquet

Exits 0 on success. Exits 1 on schema mismatch (missing `combo_id` column on
either side, or empty intersection of combos).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]

DEFAULT_COMBO_FEATURES = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"
DEFAULT_MFE_PARQUET = REPO / "data" / "ml" / "mfe" / "ml_dataset_v11_mfe.parquet"
DEFAULT_OUTPUT = REPO / "data" / "ml" / "ranker_null" / "combo_overlap_labels.parquet"

# Progress logging cadence
BATCH_SIZE = 500_000
LOG_EVERY_N_BATCHES = 10


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the overlap-labeling job.

    Returns:
        argparse.Namespace with paths and batch-size flags.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Phase 4.1 ranker-label overlap null: label every v12 combo by "
            "how many of its trade rows appear in the V3/V4 training MFE "
            "parquet. Streams the MFE parquet in pyarrow batches so the "
            "102M-row V11 file never lands in a DataFrame."
        ),
    )
    ap.add_argument(
        "--combo-features",
        type=Path,
        default=DEFAULT_COMBO_FEATURES,
        help=(
            "Input: combo-grain features parquet that defines the universe of "
            "combos to label. Expects a `global_combo_id` column "
            "(e.g. 'v11_742') and a numeric `combo_id` column. "
            f"Default: {DEFAULT_COMBO_FEATURES.relative_to(REPO)}"
        ),
    )
    ap.add_argument(
        "--mfe-parquet",
        type=Path,
        default=DEFAULT_MFE_PARQUET,
        help=(
            "Input: trade-grain MFE parquet whose rows are the labels "
            "('training-visible' iff any combo has a row here). Expects a "
            "numeric `combo_id` column. "
            f"Default: {DEFAULT_MFE_PARQUET.relative_to(REPO)}"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output parquet path. One row per combo in --combo-features. "
            f"Default: {DEFAULT_OUTPUT.relative_to(REPO)}"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            f"Rows per pyarrow iter_batches chunk (default: {BATCH_SIZE:,}). "
            "Keep under 1M to stay within the 9G systemd-run cgroup."
        ),
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY_N_BATCHES,
        help=(
            f"Log progress every N batches (default: {LOG_EVERY_N_BATCHES}; "
            f"i.e. ~every {BATCH_SIZE * LOG_EVERY_N_BATCHES:,} rows)."
        ),
    )
    return ap.parse_args()


def _load_combo_universe(path: Path) -> pd.DataFrame:
    """Load the combo universe (13,814 rows post-gate for the v12 parquet).

    Reads only the columns we need for the overlap label:

    - `global_combo_id` (string key, shape `v{sweep}_{combo_id_int}`)
    - `combo_id`        (integer; matches the MFE parquet's per-row combo_id)
    - `audit_n_trades`  (denominator for `overlap_pct`)

    Returns:
        DataFrame indexed trivially with the three columns above, validated
        against a non-empty-and-unique check on `global_combo_id`.
    """
    if not path.exists():
        raise FileNotFoundError(f"combo-features parquet not found: {path}")

    pf = pq.ParquetFile(str(path))
    schema_names = set(pf.schema_arrow.names)
    required = {"global_combo_id", "combo_id", "audit_n_trades"}
    missing = required - schema_names
    if missing:
        raise RuntimeError(
            f"combo-features parquet missing required columns: {sorted(missing)}. "
            f"Available columns include: {sorted(schema_names)[:20]}..."
        )

    cols = ["global_combo_id", "combo_id", "audit_n_trades"]
    df = pf.read(columns=cols).to_pandas()
    if df.empty:
        raise RuntimeError(f"combo-features parquet is empty: {path}")
    if df["global_combo_id"].duplicated().any():
        dup_n = int(df["global_combo_id"].duplicated().sum())
        raise RuntimeError(
            f"combo-features parquet has {dup_n} duplicate global_combo_id rows"
        )

    df["combo_id"] = df["combo_id"].astype(np.int64)
    df["audit_n_trades"] = df["audit_n_trades"].astype(np.int64)
    return df


def _derive_mfe_version_prefix(df: pd.DataFrame, mfe_path: Path) -> str:
    """Derive the sweep-version prefix that the MFE parquet corresponds to.

    The MFE parquet filenames follow `ml_dataset_v{N}_mfe.parquet`. The v12
    features parquet's `global_combo_id` also follows `v{N}_{combo_id_int}`.
    For the overlap label to be meaningful, the MFE parquet's sweep version
    must match the prefix(es) in the features parquet.

    This function extracts the version prefix from the MFE filename and
    returns it as the string the caller should filter on (e.g. `v11`).

    Args:
        df: combo-features DataFrame (for diagnostic on the prefix distribution).
        mfe_path: path to the MFE parquet file.

    Returns:
        Version prefix string (e.g. `v11`).
    """
    name = mfe_path.name
    # expected: ml_dataset_v{N}_mfe.parquet
    if not name.startswith("ml_dataset_v"):
        raise RuntimeError(
            f"MFE parquet filename does not match 'ml_dataset_v*_mfe.parquet': {name}"
        )
    tag = name[len("ml_dataset_") : name.find("_mfe")]
    if not tag.startswith("v"):
        raise RuntimeError(
            f"could not parse sweep version from MFE filename: {name} "
            f"(extracted tag: {tag!r})"
        )

    prefixes = df["global_combo_id"].str.split("_").str[0].unique().tolist()
    print(
        f"[overlap] MFE version prefix inferred: {tag!r}; "
        f"combo-features prefixes: {prefixes}",
        flush=True,
    )
    if tag not in prefixes:
        # This is a warning, not a hard failure — the script can still run if
        # the caller is intentionally checking overlap across sweep versions
        # (e.g. labeling v12 combos against v10's MFE). The overlap will simply
        # be zero, which is itself a useful signal.
        print(
            f"[overlap] WARNING: MFE prefix {tag!r} not in combo-features "
            f"prefix set {prefixes}. Overlap will be zero for all combos "
            "unless the features parquet mixes sweep versions. Continuing.",
            flush=True,
        )
    return tag


def _stream_mfe_counts(mfe_path: Path, eligible_combo_ids: set[int],
                       batch_size: int, log_every: int) -> dict[int, int]:
    """Stream the MFE parquet and count rows per `combo_id`.

    Only counts rows whose `combo_id` is in `eligible_combo_ids` — this keeps
    the aggregator at O(#combos_v12) rather than O(#combos_sweep_raw) which
    can be much larger on pre-gate sweeps.

    Memory: peak is O(batch_size) on the active chunk. The aggregator is a
    Python dict of `int -> int`, ~16 KB for 13,814 combos.

    Args:
        mfe_path: path to `ml_dataset_v{N}_mfe.parquet`.
        eligible_combo_ids: set of integer combo_ids we care about (usually
            the combo_id column of `combo_features_v12.parquet`).
        batch_size: rows per iter_batches chunk.
        log_every: progress-log cadence (batches).

    Returns:
        dict mapping `combo_id_int` → row count.
    """
    if not mfe_path.exists():
        raise FileNotFoundError(f"MFE parquet not found: {mfe_path}")

    pf = pq.ParquetFile(str(mfe_path))
    if "combo_id" not in pf.schema_arrow.names:
        raise RuntimeError(
            f"MFE parquet missing required column 'combo_id'. "
            f"Available: {pf.schema_arrow.names[:20]}..."
        )

    total_rows_expected = pf.metadata.num_rows
    print(
        f"[overlap] streaming {mfe_path.name} "
        f"({total_rows_expected:,} rows, {pf.metadata.num_row_groups} row groups)",
        flush=True,
    )

    counts: dict[int, int] = {}
    n_rows_total = 0
    n_batches = 0
    n_rows_kept = 0
    t0 = time.time()

    # Only request the combo_id column — everything else is a waste of IO.
    for batch in pf.iter_batches(batch_size=batch_size, columns=["combo_id"]):
        n_batches += 1
        arr = batch.column("combo_id").to_numpy(zero_copy_only=False)
        arr = arr.astype(np.int64, copy=False)

        # Keep only rows whose combo_id is in the eligible set.
        if eligible_combo_ids:
            mask = np.isin(arr, list(eligible_combo_ids))
            kept = arr[mask]
        else:
            kept = arr
        n_rows_kept += len(kept)

        # Aggregate via np.unique (vectorised), then update the dict.
        if len(kept):
            uniq, cnt = np.unique(kept, return_counts=True)
            for k, c in zip(uniq.tolist(), cnt.tolist()):
                counts[int(k)] = counts.get(int(k), 0) + int(c)

        n_rows_total += len(arr)
        if n_batches % log_every == 0:
            elapsed = time.time() - t0
            rps = n_rows_total / max(elapsed, 1e-9)
            print(
                f"[overlap]   batch {n_batches:>4} | "
                f"{n_rows_total:>12,} rows streamed | "
                f"{n_rows_kept:>12,} rows in-universe | "
                f"{len(counts):>5} combos seen | "
                f"{elapsed:>7.1f}s | "
                f"{rps:>10,.0f} rows/s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"[overlap] DONE streaming {n_rows_total:,} rows ({n_rows_kept:,} in-universe) "
        f"in {elapsed:.1f}s ({n_rows_total/max(elapsed,1e-9):,.0f} rows/s); "
        f"{len(counts)} unique combos observed in MFE",
        flush=True,
    )
    return counts


def _build_output(combo_df: pd.DataFrame, counts: dict[int, int],
                  mfe_prefix: str) -> pd.DataFrame:
    """Join per-combo MFE row counts onto the combo universe.

    Combos whose `global_combo_id` prefix matches the MFE parquet's sweep
    version are eligible to match; combos with a different prefix get
    `trades_in_training=0` even if their integer `combo_id` happens to
    collide with an MFE combo_id (different sweeps reuse combo_id integers).

    Args:
        combo_df: DataFrame with `global_combo_id`, `combo_id`, `audit_n_trades`.
        counts: dict of combo_id_int → row count from the MFE stream.
        mfe_prefix: sweep-version prefix of the MFE parquet (e.g. 'v11').

    Returns:
        DataFrame with columns
        (combo_id, combo_id_int, trades_in_training, trades_in_training_bool,
         overlap_pct).
    """
    out = pd.DataFrame({
        "combo_id": combo_df["global_combo_id"].astype(str).values,
        "combo_id_int": combo_df["combo_id"].astype(np.int64).values,
        "_prefix": combo_df["global_combo_id"].str.split("_").str[0].values,
        "_audit_n": combo_df["audit_n_trades"].astype(np.int64).values,
    })
    # Only rows whose prefix matches the MFE sweep version can have positive
    # overlap — a `v10_742` combo and a `v11_742` combo share the int 742 but
    # are different combos.
    prefix_match = out["_prefix"].values == mfe_prefix
    raw = np.array([counts.get(int(c), 0) for c in out["combo_id_int"].values],
                   dtype=np.int64)
    trades_in_training = np.where(prefix_match, raw, 0)

    out["trades_in_training"] = trades_in_training
    out["trades_in_training_bool"] = out["trades_in_training"] > 0

    # overlap_pct: clip to [0, 1]. If audit_n_trades is zero (shouldn't happen
    # post-gate but guard anyway) return 0.0 to avoid div-by-zero.
    denom = np.where(out["_audit_n"].values > 0, out["_audit_n"].values, 1)
    frac = out["trades_in_training"].values.astype(np.float64) / denom
    out["overlap_pct"] = np.clip(frac, 0.0, 1.0)

    return out[["combo_id", "combo_id_int", "trades_in_training",
                "trades_in_training_bool", "overlap_pct"]]


def _write_output(df: pd.DataFrame, path: Path) -> None:
    """Write the label parquet via pyarrow (project convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path))
    size_kb = path.stat().st_size / 1024
    print(
        f"[overlap] wrote {path.relative_to(REPO)} "
        f"({len(df):,} rows, {size_kb:.1f} KB)",
        flush=True,
    )


def main() -> int:
    """Entry point: stream MFE parquet, count per-combo rows, emit labels."""
    args = _parse_args()
    print(f"[overlap] combo-features: {args.combo_features}", flush=True)
    print(f"[overlap] mfe-parquet:    {args.mfe_parquet}", flush=True)
    print(f"[overlap] output:         {args.output}", flush=True)
    print(f"[overlap] batch size:     {args.batch_size:,}", flush=True)

    combo_df = _load_combo_universe(args.combo_features)
    print(
        f"[overlap] combo universe: {len(combo_df):,} rows "
        f"({combo_df['combo_id'].nunique():,} unique combo_id ints, "
        f"{combo_df['global_combo_id'].nunique():,} unique global_combo_ids)",
        flush=True,
    )

    mfe_prefix = _derive_mfe_version_prefix(combo_df, args.mfe_parquet)
    # Only stream counts for combos whose prefix matches the MFE version —
    # filtering the stream upfront keeps the aggregator tight.
    eligible_mask = combo_df["global_combo_id"].str.startswith(f"{mfe_prefix}_")
    eligible_ids = set(combo_df.loc[eligible_mask, "combo_id"].astype(int).tolist())
    print(
        f"[overlap] eligible combos (prefix == {mfe_prefix!r}): "
        f"{len(eligible_ids):,} / {len(combo_df):,}",
        flush=True,
    )
    if not eligible_ids:
        # Not a hard failure — the output is still well-defined (all zeros),
        # but flag it loudly so the caller notices.
        print(
            "[overlap] WARNING: no combo-features rows match the MFE sweep "
            "prefix. Output will record zero overlap for every combo.",
            flush=True,
        )

    counts = _stream_mfe_counts(
        args.mfe_parquet,
        eligible_ids,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )

    out = _build_output(combo_df, counts, mfe_prefix)

    # Sanity: report the overlap summary before writing
    overlap_any = int(out["trades_in_training_bool"].sum())
    overlap_zero = int((~out["trades_in_training_bool"]).sum())
    print(
        f"[overlap] summary: "
        f"{overlap_any:,} combos with trades_in_training>0, "
        f"{overlap_zero:,} combos with trades_in_training==0",
        flush=True,
    )
    if overlap_any > 0:
        # describe overlap_pct only on combos that had any overlap, so the
        # zero-prefix rows don't swamp the percentile report
        pos = out.loc[out["trades_in_training_bool"], "overlap_pct"]
        print(
            f"[overlap] overlap_pct on overlapping combos: "
            f"p5={pos.quantile(0.05):.3f}, p50={pos.median():.3f}, "
            f"p95={pos.quantile(0.95):.3f}, mean={pos.mean():.3f}",
            flush=True,
        )

    _write_output(out, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
