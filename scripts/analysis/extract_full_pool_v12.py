"""Extract the full post-gate pool of v12 combos (no top-K, no ranking).

Pool B validation gauntlet Step 4 — emits every combo that passes the
`audit_n_trades >= 500` eligibility gate from
`data/ml/ml1_results_v12/combo_features_v12.parquet`, in the same
composed_strategy_runner-compatible JSON schema as
`top_strategies_v12_raw_sharpe_top50.json`. The goal is to measure
whether ML#1 pre-selection adds value by running ML#2 V4 on the entire
eligible universe.

Usage:
    .venv/Scripts/python scripts/analysis/extract_full_pool_v12.py
    .venv/Scripts/python scripts/analysis/extract_full_pool_v12.py --output path.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

# Reuse PARAM_COLS + _entry_payload from the raw-Sharpe extractor so the
# emitted JSON schema stays byte-for-byte compatible with the top-50 file
# (minus the top-level ranking metadata).
from extract_top_combos_by_raw_sharpe_v12 import (  # noqa: E402
    PARAM_COLS,  # noqa: F401  (imported for downstream visibility / docs)
    _entry_payload,
)

FEATURES_PARQUET = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"
OUTPUT_JSON = REPO / "evaluation" / "top_strategies_v12_full_pool.json"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--output", type=Path, default=OUTPUT_JSON)
    ap.add_argument("--min-trades", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[full-pool-extract] loading {args.features.relative_to(REPO)}",
          flush=True)
    df = pd.read_parquet(args.features)
    print(f"[full-pool-extract] {len(df)} combos total", flush=True)

    eligible = df[df["audit_n_trades"] >= args.min_trades].copy()
    print(f"[full-pool-extract] n_trades >= {args.min_trades}: "
          f"{len(eligible)} eligible, "
          f"{len(df) - len(eligible)} rejected", flush=True)

    # No sort, no slice — emit every eligible combo.
    eligible = eligible.reset_index(drop=True)

    payload = {
        "source_features": str(args.features.relative_to(REPO)),
        "ranking_metric": "none",
        "min_trades": args.min_trades,
        "pool_sizes": {
            "total": int(len(df)),
            "eligible": int(len(eligible)),
        },
        "provenance": (
            "Pool B validation gauntlet Step 4 — the full post-gate v12 "
            "combo universe (audit_n_trades >= 500), with no ranking and "
            "no top-K cutoff. Paired with the V4 ML#2 filter to test "
            "whether ML#1 pre-selection adds value over running ML#2 on "
            "everything that clears the trade-count gate."
        ),
        "top": [_entry_payload(r) for _, r in eligible.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {len(eligible)} entries to "
          f"{args.output.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
