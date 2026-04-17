"""Build combo_features_v11.parquet for ML#1 v11 retrain.

Target: annualized net Sharpe per combo, computed after a $5/contract
round-trip friction charge on a fixed 1-contract PnL stream (scale-
invariant). Friction-blind gross Sharpe is what made v10_9264 selectable;
net Sharpe at 1 contract exposes the friction cost directly and prevents
tight-stop / high-frequency combos from masquerading as elite.

Source: data/ml/mfe/ml_dataset_v10_mfe.parquet (v10 only — max diversity,
10k combos, ~37M trade rows). Output: data/ml/ml1_results_v11/
combo_features_v11.parquet (one row per combo, ~41 features + target).

Run locally:
    .venv/Scripts/python scripts/analysis/build_combo_features_ml1_v11.py
"""
from __future__ import annotations
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "ml"
V10_PARQUET = DATA_DIR / "mfe" / "ml_dataset_v10_mfe.parquet"
V10_MANIFEST = DATA_DIR / "originals" / "ml_dataset_v10_manifest.json"
OUTPUT_DIR = DATA_DIR / "ml1_results_v11"
OUTPUT_PATH = OUTPUT_DIR / "combo_features_v11.parquet"

# MNQ economics and cost model (matches _top_perf_common.DEFAULT_COST_PER_CONTRACT_RT)
DOLLARS_PER_POINT = 2.0
COST_PER_CONTRACT_RT = 5.0

# Chronological 80/20 split on NQ_1min.csv: train partition spans 5.8056 years.
# Computed once from bar timestamps; hardcoded here to keep the feature builder
# independent of the raw CSV path. If the source data changes, recompute.
YEARS_SPAN_TRAIN = 5.8056

# n_trades gate — flat floor per the approved plan. Combos below this are
# dropped before training (noisy targets, unstable Sharpe).
MIN_TRADES = 500

# Parameter columns copied directly from the parquet (one value per combo,
# denormalised on every trade row). `stop_fixed_pts_resolved` comes from the
# manifest since the parquet stores the raw (nullable) param.
PARAM_COLS = [
    "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop",
    "max_hold_bars", "zscore_confirmation",
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]

CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]

BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
]

# Columns read from each parquet row group.
READ_COLS = [
    "combo_id",
    # Params (per-trade denormalised; first row of each combo is canonical)
    *PARAM_COLS,
    # Trade outcomes for target + summary features
    "label_win", "net_pnl_dollars", "r_multiple",
    "mfe_points", "mae_points", "stop_distance_pts",
    "hold_bars", "zscore_entry",
]


def load_manifest_resolved_stops() -> dict[int, float]:
    """combo_id -> stop_fixed_pts_resolved from v10 manifest.

    For stop_method='fixed', this equals the raw stop_fixed_pts param.
    For stop_method='atr'/'swing', it's the training-period median resolved
    stop — the actual stop size the backtest engine used.
    """
    data = json.loads(V10_MANIFEST.read_text())
    out: dict[int, float] = {}
    for e in data:
        if e.get("status") != "completed":
            continue
        resolved = e.get("stop_fixed_pts_resolved")
        if resolved is None:
            continue
        out[int(e["combo_id"])] = float(resolved)
    return out


def accumulate_from_parquet() -> dict[int, dict]:
    """Stream v10 parquet row groups, accumulate per-combo trade arrays.

    Memory strategy: peak usage ~ one row group decoded + per-combo lists of
    small ndarrays. At 337 row groups / 37M rows, this stays well under 4GB.
    """
    pf = pq.ParquetFile(str(V10_PARQUET))
    cols = [c for c in READ_COLS if c in pf.schema_arrow.names]
    missing = set(READ_COLS) - set(cols)
    if missing:
        print(f"  [WARN] missing cols in parquet: {sorted(missing)}", flush=True)

    acc_pnl: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_win: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_rmul: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_stop: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_mfe: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_mae: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_hold: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_zentry: dict[int, list[np.ndarray]] = defaultdict(list)
    # One parameter row per combo — captured from the first row seen.
    combo_params: dict[int, dict] = {}

    t0 = time.time()
    n_rows_total = 0
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=cols).to_pandas()
        cid_arr = chunk["combo_id"].astype(int).to_numpy()
        pnl = chunk["net_pnl_dollars"].to_numpy(dtype=np.float64)
        win = chunk["label_win"].to_numpy(dtype=np.int8)
        rmul = chunk["r_multiple"].to_numpy(dtype=np.float64)
        stop = chunk["stop_distance_pts"].to_numpy(dtype=np.float64)
        mfe = chunk["mfe_points"].to_numpy(dtype=np.float64)
        mae = chunk["mae_points"].to_numpy(dtype=np.float64)
        hold = chunk["hold_bars"].to_numpy(dtype=np.float64)
        zentry = chunk["zscore_entry"].to_numpy(dtype=np.float64)

        order = np.argsort(cid_arr, kind="stable")
        cid_sorted = cid_arr[order]
        uniq, starts = np.unique(cid_sorted, return_index=True)
        ends = np.append(starts[1:], len(cid_sorted))
        for cid, st, en in zip(uniq, starts, ends):
            cid = int(cid)
            idx = order[st:en]
            acc_pnl[cid].append(pnl[idx])
            acc_win[cid].append(win[idx])
            acc_rmul[cid].append(rmul[idx])
            acc_stop[cid].append(stop[idx])
            acc_mfe[cid].append(mfe[idx])
            acc_mae[cid].append(mae[idx])
            acc_hold[cid].append(hold[idx])
            acc_zentry[cid].append(zentry[idx])
            if cid not in combo_params:
                first = int(idx[0])
                combo_params[cid] = {c: chunk[c].iat[first] for c in PARAM_COLS
                                     if c in chunk.columns}

        n_rows_total += len(chunk)
        if (rg + 1) % 20 == 0 or rg == pf.metadata.num_row_groups - 1:
            elapsed = time.time() - t0
            rate = n_rows_total / max(elapsed, 1e-9)
            print(f"  rg {rg+1}/{pf.metadata.num_row_groups} | "
                  f"{n_rows_total:,} rows | {rate:,.0f} rows/s | "
                  f"combos={len(combo_params)}", flush=True)
        del chunk

    combos: dict[int, dict] = {}
    for cid in list(acc_pnl.keys()):
        combos[cid] = {
            "params": combo_params[cid],
            "pnl": np.concatenate(acc_pnl.pop(cid)),
            "win": np.concatenate(acc_win.pop(cid)),
            "rmul": np.concatenate(acc_rmul.pop(cid)),
            "stop": np.concatenate(acc_stop.pop(cid)),
            "mfe": np.concatenate(acc_mfe.pop(cid)),
            "mae": np.concatenate(acc_mae.pop(cid)),
            "hold": np.concatenate(acc_hold.pop(cid)),
            "zentry": np.concatenate(acc_zentry.pop(cid)),
        }
    gc.collect()
    print(f"  total: {n_rows_total:,} rows, {len(combos)} combos, "
          f"{time.time()-t0:.1f}s", flush=True)
    return combos


def compute_combo_row(cid: int, bundle: dict, resolved_stop: float | None) -> dict | None:
    """Compute target + features for one combo. Returns None if gated out."""
    n = len(bundle["pnl"])
    if n < MIN_TRADES:
        return None

    rmul = bundle["rmul"]
    stop = bundle["stop"]
    # 1-contract gross PnL per trade = r_multiple * stop_pts * $/pt.
    # This is the scale-invariant baseline (matches the gross PnL at any
    # fixed contract count; we pick 1 so the friction charge below has a
    # concrete dollar anchor).
    gross_pnl = rmul * stop * DOLLARS_PER_POINT
    net_pnl = gross_pnl - COST_PER_CONTRACT_RT  # 1 contract RT

    std_net = float(np.std(net_pnl, ddof=1)) if n >= 2 else 0.0
    std_gross = float(np.std(gross_pnl, ddof=1)) if n >= 2 else 0.0
    trades_per_year = n / YEARS_SPAN_TRAIN
    if std_net == 0.0 or not np.isfinite(std_net):
        target_net_sharpe = 0.0
    else:
        target_net_sharpe = float(
            np.mean(net_pnl) / std_net * np.sqrt(trades_per_year)
        )
    if std_gross == 0.0 or not np.isfinite(std_gross):
        gross_sharpe = 0.0
    else:
        gross_sharpe = float(
            np.mean(gross_pnl) / std_gross * np.sqrt(trades_per_year)
        )

    # Trade-stream summary features
    win = bundle["win"]
    mfe = bundle["mfe"]
    mae = bundle["mae"]
    hold = bundle["hold"]
    zentry = bundle["zentry"]

    mfe_mae = np.where(np.abs(mae) > 0, mfe / np.abs(mae), 0.0)

    row: dict = {
        "global_combo_id": f"v10_{cid}",
        "combo_id": cid,
        # Target
        "target_net_sharpe": target_net_sharpe,
        # Trade-stream summaries
        "n_trades": int(n),
        "trades_per_year": float(trades_per_year),
        "gross_win_rate": float(np.mean(win)),
        "median_stop_pts": float(np.median(stop)),
        "median_hold_bars": float(np.median(hold)),
        "p90_hold_bars": float(np.quantile(hold, 0.9)),
        "median_mfe_pts": float(np.median(mfe)),
        "median_mae_pts": float(np.median(mae)),
        "median_mfe_mae_ratio": float(np.median(mfe_mae)),
        "gross_sharpe": gross_sharpe,
        "gross_net_sharpe_gap": float(gross_sharpe - target_net_sharpe),
        "zscore_entry_median_abs": float(np.median(np.abs(zentry))),
        # Engineered friction-awareness features
        "stop_fixed_pts_resolved": float(resolved_stop) if resolved_stop is not None
                                   else float(np.median(stop)),
        "cost_as_pct_of_edge": float(
            COST_PER_CONTRACT_RT / max(float(np.mean(gross_pnl)), 1e-6)
        ) if float(np.mean(gross_pnl)) > 0 else float("inf"),
    }

    # tight_stop_flag uses resolved stop (handles atr/swing combos correctly)
    row["tight_stop_flag"] = int(row["stop_fixed_pts_resolved"] < 3.0)

    # Parameter features — copy through, boolean-normalised, categorical as str
    for col in PARAM_COLS:
        val = bundle["params"].get(col)
        if col in BOOLEAN_COLS:
            row[col] = int(bool(val)) if val is not None and not pd.isna(val) else 0
        elif col in CATEGORICAL_COLS:
            row[col] = str(val) if val is not None and not pd.isna(val) else "none"
        else:
            row[col] = float(val) if val is not None and not pd.isna(val) else np.nan
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v11] Source: {V10_PARQUET}", flush=True)
    print(f"[v11] Manifest: {V10_MANIFEST}", flush=True)
    print(f"[v11] Output: {OUTPUT_PATH}", flush=True)
    print(f"[v11] Gate: n_trades >= {MIN_TRADES}", flush=True)
    print(f"[v11] Target: annualized net Sharpe @ "
          f"${COST_PER_CONTRACT_RT}/contract RT (1 contract)", flush=True)
    print(f"[v11] Train span: {YEARS_SPAN_TRAIN} years", flush=True)
    print()

    resolved_stops = load_manifest_resolved_stops()
    print(f"[v11] Loaded {len(resolved_stops)} resolved stops from manifest",
          flush=True)

    print("[v11] Streaming parquet...", flush=True)
    combos = accumulate_from_parquet()

    print("[v11] Computing per-combo features + target...", flush=True)
    rows: list[dict] = []
    gated_out = 0
    for cid, bundle in combos.items():
        row = compute_combo_row(cid, bundle, resolved_stops.get(cid))
        if row is None:
            gated_out += 1
            continue
        rows.append(row)
    print(f"[v11] Kept {len(rows)} / {len(combos)} combos "
          f"({gated_out} dropped by n_trades gate)", flush=True)

    df = pd.DataFrame(rows)

    # Sanity checks
    n_nan_target = int(df["target_net_sharpe"].isna().sum())
    assert n_nan_target == 0, f"NaN in target_net_sharpe: {n_nan_target}"
    print(f"[v11] target_net_sharpe: "
          f"p5={df['target_net_sharpe'].quantile(0.05):.3f}, "
          f"p50={df['target_net_sharpe'].median():.3f}, "
          f"p95={df['target_net_sharpe'].quantile(0.95):.3f}", flush=True)
    print(f"[v11] gross_sharpe p50={df['gross_sharpe'].median():.3f} | "
          f"gap p50={df['gross_net_sharpe_gap'].median():.3f}", flush=True)
    print(f"[v11] tight_stop_flag=1 count: "
          f"{int(df['tight_stop_flag'].sum())}", flush=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[v11] Wrote {OUTPUT_PATH} "
          f"({OUTPUT_PATH.stat().st_size/1e6:.1f} MB, "
          f"{len(df)} rows, {len(df.columns)} cols)", flush=True)


if __name__ == "__main__":
    main()
