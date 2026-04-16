"""
init_db.py — Create and populate the SQLite strategy database.

Consolidates parameter combos, sweep metrics, validation results, and ML
run metadata from scattered CSVs/Parquets/JSONs into a single queryable
database at data/ml/strategy.db.

Usage:
    python scripts/init_db.py              # create + backfill (idempotent)
    python scripts/init_db.py --force      # drop all tables first, then recreate
    python scripts/init_db.py --db-path custom.db

Example queries after creation:
    SELECT cp.global_combo_id, vr.win_rate, vr.sharpe_r
    FROM validation_results vr
    JOIN combo_params cp ON cp.global_combo_id = vr.global_combo_id
    WHERE vr.partition = 'test' AND vr.win_rate > 0.60
    ORDER BY vr.sharpe_r DESC;
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path("data/ml/strategy.db")
COMBO_FEATURES_PATH = Path("data/ml/ml1_results/combo_features.parquet")
TOP_COMBOS_PATH = Path("data/ml/ml1_results/top_combos.csv")
VALIDATION_DIR = Path("data/ml/ml1_results/validation")
RUN_METADATA_PATH = Path("data/ml/ml1_results/run_metadata.json")
CV_RESULTS_PATH = Path("data/ml/ml1_results/cv_results.json")
ML_DATA_DIR = Path("data/ml")

# ── Parameter columns in combo_features.parquet ────────────────────────────
# These are the strategy parameter columns (features for LightGBM).
# They become columns in the combo_params table.

PARAM_COLS = [
    "z_band_k", "z_window", "volume_zscore_window", "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop", "max_hold_bars",
    "zscore_confirmation", "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight", "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]

# Sweep-level performance metric columns in combo_features.parquet.
METRIC_COLS = [
    "n_trades", "win_rate", "total_return_pct", "total_return_dollars",
    "final_equity", "sharpe_ratio", "max_drawdown_pct", "profit_factor",
    "avg_r_multiple", "median_r_multiple", "avg_trade_pnl", "std_trade_pnl",
    "max_consecutive_losses", "calmar_ratio", "composite_score",
]


# ── Schema DDL ──────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Strategy parameter combos: one row per unique parameter configuration.
CREATE TABLE IF NOT EXISTS combo_params (
    global_combo_id         TEXT PRIMARY KEY,   -- e.g. "v10_9955"
    sweep_version           INTEGER NOT NULL,
    combo_id                INTEGER NOT NULL,   -- per-version 0-9999

    -- Core z-score / indicator params
    z_band_k                REAL,
    z_window                INTEGER,
    volume_zscore_window    INTEGER,
    ema_fast                INTEGER,
    ema_slow                INTEGER,

    -- Stop-loss configuration
    stop_method             TEXT,       -- 'fixed', 'atr', 'swing'
    stop_fixed_pts          REAL,
    atr_multiplier          REAL,
    swing_lookback          REAL,
    stop_fixed_pts_resolved REAL,      -- actual stop distance used in sweep

    -- Risk / exit params
    min_rr                      REAL,
    exit_on_opposite_signal     INTEGER,  -- 0/1
    use_breakeven_stop          INTEGER,  -- 0/1
    max_hold_bars               REAL,
    zscore_confirmation         INTEGER,  -- 0/1

    -- Z-score variant formulation
    z_input                 TEXT,
    z_anchor                TEXT,
    z_denom                 TEXT,
    z_type                  TEXT,
    z_window_2              INTEGER,
    z_window_2_weight       REAL,

    -- V5+ filter params
    volume_entry_threshold  REAL,
    vol_regime_lookback     INTEGER,
    vol_regime_min_pct      REAL,
    vol_regime_max_pct      REAL,
    session_filter_mode     TEXT,
    tod_exit_hour           INTEGER,

    UNIQUE(sweep_version, combo_id)
);

-- Sweep-level performance metrics (1:1 with combo_params).
CREATE TABLE IF NOT EXISTS sweep_metrics (
    global_combo_id         TEXT PRIMARY KEY
                            REFERENCES combo_params(global_combo_id),
    n_trades                INTEGER,
    win_rate                REAL,
    total_return_pct        REAL,
    total_return_dollars    REAL,
    final_equity            REAL,
    sharpe_ratio            REAL,
    max_drawdown_pct        REAL,
    profit_factor           REAL,
    avg_r_multiple          REAL,
    median_r_multiple       REAL,
    avg_trade_pnl           REAL,
    std_trade_pnl           REAL,
    max_consecutive_losses  INTEGER,
    calmar_ratio            REAL,
    composite_score         REAL,
    predicted_composite     REAL       -- NULL for combos not in top-N
);

-- Validation results: one row per combo × partition × tag × run.
-- Supports multiple reruns (different run_timestamp) and tagged groups
-- (low_freq, high_freq, etc.).
CREATE TABLE IF NOT EXISTS validation_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    global_combo_id     TEXT NOT NULL
                        REFERENCES combo_params(global_combo_id),
    partition           TEXT NOT NULL CHECK (partition IN ('train', 'test')),
    tag                 TEXT NOT NULL DEFAULT '',

    n_trades            INTEGER,
    n_signals           INTEGER,
    win_rate            REAL,
    total_return_pct    REAL,
    max_drawdown_pct    REAL,
    max_drawdown_R      REAL,
    sharpe_r            REAL,
    profit_factor       REAL,
    avg_r_multiple      REAL,
    runtime_seconds     REAL,
    run_timestamp       TEXT NOT NULL
);

-- ML optimizer run metadata (one row per ml1_surrogate.py execution).
CREATE TABLE IF NOT EXISTS ml_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp           TEXT NOT NULL,
    versions_used           TEXT NOT NULL,       -- JSON array
    min_trades              INTEGER,
    n_folds                 INTEGER,
    seed                    INTEGER,
    total_combos            INTEGER,
    surrogate_candidates    INTEGER,

    -- Composite score weights
    w_sharpe                REAL,
    w_return                REAL,
    w_drawdown              REAL,
    w_winrate               REAL,
    w_trades                REAL,

    -- LGB hyperparameters (JSON blob — queried rarely)
    lgb_params              TEXT,

    -- Full CV results per target (JSON blob)
    cv_results              TEXT,

    -- Top-level summary for composite_score model
    overall_r2              REAL,
    overall_rmse            REAL,
    overfit_gap             REAL
);

-- ── Indexes ────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_combo_version
    ON combo_params(sweep_version);

CREATE INDEX IF NOT EXISTS idx_sweep_winrate
    ON sweep_metrics(win_rate);
CREATE INDEX IF NOT EXISTS idx_sweep_drawdown
    ON sweep_metrics(max_drawdown_pct);
CREATE INDEX IF NOT EXISTS idx_sweep_sharpe
    ON sweep_metrics(sharpe_ratio);
CREATE INDEX IF NOT EXISTS idx_sweep_composite
    ON sweep_metrics(composite_score);
CREATE INDEX IF NOT EXISTS idx_sweep_ntrades
    ON sweep_metrics(n_trades);

CREATE INDEX IF NOT EXISTS idx_val_combo
    ON validation_results(global_combo_id);
CREATE INDEX IF NOT EXISTS idx_val_partition_tag
    ON validation_results(partition, tag);
CREATE INDEX IF NOT EXISTS idx_val_winrate
    ON validation_results(win_rate);
CREATE INDEX IF NOT EXISTS idx_val_drawdown
    ON validation_results(max_drawdown_pct);
"""


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create and populate strategy.db")
    p.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                   help="Path for the SQLite database file")
    p.add_argument("--force", action="store_true",
                   help="Drop all tables before recreating")
    return p.parse_args()


# ── Schema creation ─────────────────────────────────────────────────────────

def create_schema(conn: sqlite3.Connection, force: bool = False) -> None:
    """Create all tables and indexes. If force=True, drop existing tables first."""
    if force:
        print("[init_db] --force: dropping existing tables...")
        for table in ["validation_results", "sweep_metrics", "combo_params", "ml_runs"]:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.executescript(SCHEMA_SQL)
    print("[init_db] Schema created (4 tables + indexes)")


# ── Backfill: combo_params ──────────────────────────────────────────────────

def _load_resolved_stops() -> dict:
    """
    Load stop_fixed_pts_resolved from all manifests (v2-v10).
    Returns {global_combo_id: resolved_stop_value}.
    """
    resolved = {}
    for v in range(2, 11):
        manifest_path = ML_DATA_DIR / "originals" / f"ml_dataset_v{v}_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            gid = f"v{v}_{entry['combo_id']}"
            val = entry.get("stop_fixed_pts_resolved")
            if val is not None:
                resolved[gid] = float(val)
    return resolved


def backfill_combos(conn: sqlite3.Connection) -> int:
    """
    Insert combo parameter rows from combo_features.parquet + manifests.
    Returns the number of rows inserted.
    """
    if not COMBO_FEATURES_PATH.exists():
        print(f"  [SKIP] {COMBO_FEATURES_PATH} not found")
        return 0

    df = pd.read_parquet(COMBO_FEATURES_PATH)
    resolved_stops = _load_resolved_stops()

    # Parse combo_id from global_combo_id (e.g. "v10_9955" -> 9955)
    df["combo_id"] = df["global_combo_id"].str.split("_").str[-1].astype(int)

    # Build rows for INSERT
    cols = ["global_combo_id", "sweep_version", "combo_id"] + PARAM_COLS + ["stop_fixed_pts_resolved"]
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)

    rows = []
    for _, r in df.iterrows():
        gid = r["global_combo_id"]
        row_vals = [gid, int(r["sweep_version"]), int(r["combo_id"])]

        for col in PARAM_COLS:
            val = r.get(col)
            # Convert NaN/NaT to None for SQLite NULL
            if pd.isna(val):
                row_vals.append(None)
            elif col in ("exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation"):
                row_vals.append(int(bool(val)))
            else:
                row_vals.append(val)

        # stop_fixed_pts_resolved from manifest
        row_vals.append(resolved_stops.get(gid))
        rows.append(tuple(row_vals))

    conn.executemany(
        f"INSERT OR REPLACE INTO combo_params ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    print(f"  combo_params: {len(rows):,} rows")
    return len(rows)


# ── Backfill: sweep_metrics ─────────────────────────────────────────────────

def backfill_sweep_metrics(conn: sqlite3.Connection) -> int:
    """
    Insert sweep performance metrics from combo_features.parquet.
    Merges predicted_composite from top_combos.csv where available.
    """
    if not COMBO_FEATURES_PATH.exists():
        print(f"  [SKIP] {COMBO_FEATURES_PATH} not found")
        return 0

    df = pd.read_parquet(COMBO_FEATURES_PATH)

    # Load predicted_composite from top_combos.csv (only ~20 combos have it)
    predicted = {}
    if TOP_COMBOS_PATH.exists():
        top_df = pd.read_csv(TOP_COMBOS_PATH)
        if "predicted_composite" in top_df.columns:
            for _, r in top_df.iterrows():
                predicted[r["global_combo_id"]] = float(r["predicted_composite"])

    cols = ["global_combo_id"] + METRIC_COLS + ["predicted_composite"]
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)

    rows = []
    for _, r in df.iterrows():
        gid = r["global_combo_id"]
        row_vals = [gid]

        for col in METRIC_COLS:
            val = r.get(col)
            if pd.isna(val):
                row_vals.append(None)
            elif col in ("n_trades", "max_consecutive_losses"):
                row_vals.append(int(val))
            else:
                row_vals.append(float(val))

        # predicted_composite: only for combos that appear in top_combos.csv
        row_vals.append(predicted.get(gid))
        rows.append(tuple(row_vals))

    conn.executemany(
        f"INSERT OR REPLACE INTO sweep_metrics ({col_names}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    print(f"  sweep_metrics: {len(rows):,} rows")
    return len(rows)


# ── Backfill: validation_results ────────────────────────────────────────────

# Mapping from filename pattern to tag
VALIDATION_FILES = {
    "validation_results.csv": "",
    "validation_low_freq.csv": "low_freq",
    "validation_high_freq.csv": "high_freq",
}

# The old validation_results.csv uses different column names than the
# rewritten script. Normalize to the new schema.
COLUMN_RENAMES = {
    "sharpe_ratio": "sharpe_r",
    "final_equity": None,  # drop — not in the new schema
}

# Columns expected in validation_results table (excluding id and run_timestamp)
VAL_COLS = [
    "global_combo_id", "partition", "tag",
    "n_trades", "n_signals", "win_rate", "total_return_pct",
    "max_drawdown_pct", "max_drawdown_R", "sharpe_r",
    "profit_factor", "avg_r_multiple", "runtime_seconds",
]


def backfill_validation(conn: sqlite3.Connection) -> int:
    """
    Import validation CSVs. Handles schema differences between old and new
    validation scripts (sharpe_ratio vs sharpe_r, etc.).
    """
    if not VALIDATION_DIR.exists():
        print(f"  [SKIP] {VALIDATION_DIR} not found")
        return 0

    total = 0
    for filename, tag in VALIDATION_FILES.items():
        csv_path = VALIDATION_DIR / filename
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Normalize column names from old schema to new
        if "sharpe_ratio" in df.columns and "sharpe_r" not in df.columns:
            df = df.rename(columns={"sharpe_ratio": "sharpe_r"})
        if "final_equity" in df.columns:
            df = df.drop(columns=["final_equity"], errors="ignore")

        # Add missing columns with NULL
        if "max_drawdown_R" not in df.columns:
            df["max_drawdown_R"] = None
        if "n_signals" not in df.columns:
            df["n_signals"] = None

        # Set tag and run_timestamp
        df["tag"] = tag
        file_mtime = datetime.fromtimestamp(
            os.path.getmtime(csv_path), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S")
        df["run_timestamp"] = file_mtime

        all_cols = VAL_COLS + ["run_timestamp"]
        placeholders = ", ".join(["?"] * len(all_cols))
        col_names = ", ".join(all_cols)

        rows = []
        for _, r in df.iterrows():
            row_vals = []
            for col in all_cols:
                val = r.get(col)
                if pd.isna(val):
                    row_vals.append(None)
                else:
                    row_vals.append(val)
            rows.append(tuple(row_vals))

        # Clear existing rows for this tag before reinserting (idempotent)
        conn.execute(
            "DELETE FROM validation_results WHERE tag = ? AND run_timestamp = ?",
            (tag, file_mtime),
        )
        conn.executemany(
            f"INSERT INTO validation_results ({col_names}) VALUES ({placeholders})",
            rows,
        )
        conn.commit()
        total += len(rows)
        print(f"  validation_results: {len(rows)} rows from {filename} (tag='{tag}')")

    if total == 0:
        print("  validation_results: no CSV files found")
    return total


# ── Backfill: ml_runs ──────────────────────────────────────────────────────

def backfill_ml_runs(conn: sqlite3.Connection) -> int:
    """
    Insert ML optimizer run metadata from run_metadata.json + cv_results.json.
    """
    if not RUN_METADATA_PATH.exists():
        print(f"  [SKIP] {RUN_METADATA_PATH} not found")
        return 0

    meta = json.loads(RUN_METADATA_PATH.read_text())
    cv = {}
    if CV_RESULTS_PATH.exists():
        cv = json.loads(CV_RESULTS_PATH.read_text())

    # Extract composite weights
    weights = meta.get("composite_weights", {})

    # Extract top-level CV metrics for the composite_score model
    comp_cv = cv.get("composite_score", {})

    # Extract LGB hyperparameters from meta
    lgb_params = meta.get("lgb_params", {})

    # Deduplicate: if a run with this timestamp already exists, skip
    run_ts = meta.get("timestamp", "")
    existing = conn.execute(
        "SELECT COUNT(*) FROM ml_runs WHERE run_timestamp = ?", (run_ts,)
    ).fetchone()[0]
    if existing > 0:
        print("  ml_runs: already exists (skipped)")
        return 0

    conn.execute(
        """INSERT INTO ml_runs (
            run_timestamp, versions_used, min_trades, n_folds, seed,
            total_combos, surrogate_candidates,
            w_sharpe, w_return, w_drawdown, w_winrate, w_trades,
            lgb_params, cv_results,
            overall_r2, overall_rmse, overfit_gap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            meta.get("timestamp", ""),
            json.dumps(meta.get("versions_used", [])),
            meta.get("min_trades"),
            meta.get("n_folds"),
            meta.get("seed"),
            meta.get("total_combos"),
            meta.get("surrogate_candidates"),
            weights.get("sharpe_ratio"),
            weights.get("total_return_pct"),
            weights.get("max_drawdown_pct"),
            weights.get("win_rate"),
            weights.get("n_trades"),
            json.dumps(lgb_params),
            json.dumps(cv),
            comp_cv.get("overall_r2"),
            comp_cv.get("overall_rmse"),
            comp_cv.get("overfit_gap"),
        ),
    )
    conn.commit()
    print("  ml_runs: 1 row")
    return 1


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Initializing strategy database: {db_path}")
    print("=" * 60)

    conn = sqlite3.connect(str(db_path))

    # Step 1: Create schema
    create_schema(conn, force=args.force)

    # Step 2-5: Backfill data
    print("\n[init_db] Backfilling data...")
    backfill_combos(conn)
    backfill_sweep_metrics(conn)
    backfill_validation(conn)
    backfill_ml_runs(conn)

    # Summary: row counts per table
    print("\n[init_db] Final row counts:")
    for table in ["combo_params", "sweep_metrics", "validation_results", "ml_runs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,}")

    db_size_kb = db_path.stat().st_size / 1024
    print(f"\n  Database size: {db_size_kb:.0f} KB")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
