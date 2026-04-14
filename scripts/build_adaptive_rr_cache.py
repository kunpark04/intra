"""Build the shared expanded-features cache used by every ML#2 variant.

Reads the 9 `_mfe.parquet` files, stratified-subsamples to the configured
base size, expands by 17 R:R levels with synthetic labels, and writes to
`data/ml/adaptive_rr_cache_<hash>.parquet`. The hash is keyed on
ALL_FEATURES + RR_LEVELS (imported from `adaptive_rr_model_v2`) so any
schema/grid drift lands on a new path, guaranteeing no stale reads.

Runs expansion only — no training. Use this once per host/dataset to
populate the cache; `adaptive_rr_model_v2.py`, `adaptive_rr_model_b9.py`,
and future variants will load from it automatically on start.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import adaptive_rr_model_v2 as v2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--versions", type=int, nargs="+", default=list(range(2, 11)))
    p.add_argument("--max-rows", type=int, default=10_000_000)
    p.add_argument("--min-trades-per-combo", type=int, default=30)
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if cache already exists")
    args = p.parse_args()

    cache_path = v2.shared_cache_path()
    print(f"Cache path: {cache_path}")
    if cache_path.exists() and not args.force:
        size_gb = cache_path.stat().st_size / 1e9
        print(f"Cache already exists ({size_gb:.2f} GB). Pass --force to rebuild.")
        return

    t0 = time.time()
    target_base = args.max_rows // len(v2.RR_LEVELS)
    target_load = int(target_base * 1.2)
    print(f"[load] versions={args.versions} target_base={target_load:,}")
    df = v2.load_mfe_parquets(args.versions, target_load)
    df = v2.filter_combos(df, args.min_trades_per_combo)
    expanded = v2.expand_rr_levels(df, v2.RR_LEVELS, args.max_rows)
    del df

    print(f"\n[cache] Writing {len(expanded):,} rows to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(cache_path, compression="snappy", index=False)
    print(f"  Saved: {cache_path.stat().st_size / 1e9:.2f} GB")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
