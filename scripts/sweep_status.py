#!/usr/bin/env python3
"""Quick status dashboard for the MFE sweep pipeline."""

import json
import re
from pathlib import Path

VERSIONS = [
    {"name": "v2",  "mode": "default",          "combos": 3000,  "seed": 0, "suffix": "mfe"},
    {"name": "v3",  "mode": "zscore_variants",   "combos": 3000,  "seed": 0, "suffix": "mfe"},
    {"name": "v4",  "mode": "v4",                "combos": 3000,  "seed": 4, "suffix": "mfe"},
    {"name": "v5",  "mode": "v5",                "combos": 3000,  "seed": 5, "suffix": "mfe"},
    {"name": "v6",  "mode": "v6",                "combos": 3000,  "seed": 6, "suffix": "mfe"},
    {"name": "v7",  "mode": "v7",                "combos": 3000,  "seed": 7, "suffix": "mfe"},
    {"name": "v8",  "mode": "v8",                "combos": 3000,  "seed": 8, "suffix": "mfe"},
    {"name": "v9",  "mode": "v9",                "combos": 6000,  "seed": 9, "suffix": "mfe"},
    {"name": "v10", "mode": "v10",               "combos": 10000, "seed": 0, "suffix": "mfe"},
]

ML_DIR = Path("data/ml")
LOG_DIR = Path(".")  # sweep logs live in repo root


def _parse_rate_from_log(name: str, suffix: str, is_running: bool) -> float | None:
    """Parse the most recent progress line from the sweep log to get combos/min.

    Only parses for versions that are currently running to avoid stale/wrong data.
    Picks the most recently modified log among candidates.
    """
    if not is_running:
        return None

    # Candidate logs: version-specific and generic fallback
    candidates = [
        LOG_DIR / f"sweep_run_{name}_{suffix}.log",
        LOG_DIR / "sweep_run.log",
    ]
    # Pick the most recently modified log that exists
    existing = [(p, p.stat().st_mtime) for p in candidates if p.exists()]
    if not existing:
        return None
    log_path = max(existing, key=lambda x: x[1])[0]

    # Read last 50 lines to find the most recent progress line
    try:
        lines = log_path.read_text().splitlines()[-50:]
    except Exception:
        return None

    # Pattern: [sweep] 1400/3000 (47%) | 42.4m | last combo #1399
    pattern = re.compile(r"\[sweep\]\s+(\d+)/\d+\s+\(\d+%\)\s+\|\s+([\d.]+)m")
    last_match = None
    for line in lines:
        m = pattern.search(line)
        if m:
            last_match = m

    if last_match:
        done = int(last_match.group(1))
        elapsed_m = float(last_match.group(2))
        if elapsed_m > 0:
            return done / elapsed_m
    return None


def check_version(v: dict) -> dict:
    name = v["name"]
    total = v["combos"]
    suffix = v["suffix"]

    manifest_path = ML_DIR / f"ml_dataset_{name}_{suffix}_manifest.json"
    parquet_path = ML_DIR / f"ml_dataset_{name}_{suffix}.parquet"
    chunks_dir = ML_DIR / f"ml_dataset_{name}_{suffix}_chunks"

    result = {"name": name, "total": total, "completed": 0, "errors": 0,
              "error_types": [], "chunks": 0, "merged": False,
              "status": "not started", "rate": None, "eta_min": None}

    # Check manifest
    if manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path))
            completed = sum(1 for e in manifest if e.get("status") == "completed")
            errors = sum(1 for e in manifest if str(e.get("status", "")).startswith("error"))
            result["completed"] = completed
            result["errors"] = errors

            # Collect unique error types
            err_msgs = {}
            for e in manifest:
                s = str(e.get("status", ""))
                if s.startswith("error"):
                    short = s[:80]
                    err_msgs[short] = err_msgs.get(short, 0) + 1
            result["error_types"] = sorted(err_msgs.items(), key=lambda x: -x[1])[:3]

        except (json.JSONDecodeError, Exception):
            pass

    # Check chunks
    if chunks_dir.exists():
        result["chunks"] = len(list(chunks_dir.glob("chunk_*.parquet")))

    # Check merged parquet
    if parquet_path.exists() and parquet_path.stat().st_size > 100:
        result["merged"] = True

    # Determine status
    processed = result["completed"] + result["errors"]
    if result["completed"] == 0 and result["errors"] == 0 and result["chunks"] == 0:
        result["status"] = "not started"
    elif result["merged"]:
        result["status"] = "DONE"
    elif processed >= total:
        result["status"] = "DONE (unmerged)" if not result["merged"] else "DONE"
    elif processed > 0:
        result["status"] = "running"

    # Parse rate from log (only for running versions)
    rate = _parse_rate_from_log(name, suffix, is_running=(result["status"] == "running"))
    if rate and rate > 0:
        result["rate"] = round(rate, 1)
        remaining = total - processed
        if remaining > 0:
            result["eta_min"] = round(remaining / rate)

    return result


def main():
    print()
    print("=" * 72)
    print("  MFE SWEEP PIPELINE STATUS")
    print("=" * 72)
    print()
    print(f"  {'Ver':<5} {'Progress':<14} {'%':<7} {'Rate':<12} {'ETA':<10} {'Errors':<8} {'Status'}")
    print(f"  {'---':<5} {'--------':<14} {'-':<7} {'----':<12} {'---':<10} {'------':<8} {'------'}")

    total_completed = 0
    total_target = 0
    total_errors = 0
    current_rate = None

    for v in VERSIONS:
        r = check_version(v)
        total_completed += r["completed"]
        total_target += r["total"]
        total_errors += r["errors"]

        pct = r["completed"] / r["total"] * 100 if r["total"] else 0
        progress_str = f"{r['completed']}/{r['total']}"
        pct_str = f"{pct:.0f}%"
        err_str = str(r["errors"]) if r["errors"] else "-"

        rate_str = f"{r['rate']}/min" if r["rate"] else "-"
        if r["eta_min"] is not None and r["status"] == "running":
            h, m = divmod(r["eta_min"], 60)
            eta_str = f"{h}h {m}m" if h else f"{m}m"
        else:
            eta_str = "-"

        if r["status"].startswith("DONE"):
            status_str = "DONE"
        elif r["status"] == "running":
            status_str = "running"
            current_rate = r["rate"]
        elif r["status"] == "not started":
            status_str = "waiting"
        else:
            status_str = r["status"]

        print(f"  {r['name']:<5} {progress_str:<14} {pct_str:<7} {rate_str:<12} {eta_str:<10} {err_str:<8} {status_str}")

        for msg, count in r["error_types"]:
            print(f"         {count}x {msg}")

    print()
    overall_pct = total_completed / total_target * 100 if total_target else 0
    print(f"  Overall: {total_completed:,}/{total_target:,} ({overall_pct:.1f}%)  |  Errors: {total_errors}")
    if current_rate:
        remaining = total_target - total_completed - total_errors
        total_eta = round(remaining / current_rate)
        h, m = divmod(total_eta, 60)
        print(f"  Est. remaining: {h}h {m}m (at {current_rate}/min)")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
