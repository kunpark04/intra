"""Build combo_features_v12.parquet — parameter-only forecasting target.

Key differences from v11:

1. **Walk-forward target.** Each combo's trade stream is split into K ordinal
   windows (default K=5). Per-window 1-contract net Sharpe is computed,
   annualized to the per-window time fraction. The model target is the robust
   aggregate: `target_robust_sharpe = median - 0.5 * std` across windows.
   This is our best proxy for OOS Sharpe on unseen bars without leaking the
   true test partition.

2. **Audit vs model-input features explicitly separated.** Trade-derived
   summary features (n_trades, win_rate, mfe/mae stats, cost_as_pct_of_edge,
   etc.) are stored in the parquet with the `audit_` prefix. The trainer
   excludes every `audit_*` column from the feature set. This keeps the
   diagnostic panel alive without letting trade-derived signal leak into a
   supposedly parameter-only forecast.

3. **Feature set is parameter-only + param-derived engineered features.**
   No trade summaries reach the model.

4. **Parameter-space KNN features are NOT computed here.** They depend on
   targets and so must be fold-aware in the trainer to avoid target leakage.

Note on ordinal windows: the v10 parquet doesn't carry per-trade timestamps,
only per-trade order within a combo. Trades are appended in entry order by
the sweep runner, so ordinal-trade splits correspond to monotone time slices
within each combo — just not equal-calendar-time slices (trade rate varies
with market regime). For forecasting cross-window stability this is a
reasonable proxy; a follow-up v12.1 could use real entry_bar once the v11
sweep carries it.

Run locally:
    .venv/Scripts/python scripts/analysis/build_combo_features_ml1_v12.py
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "ml"

DEFAULT_SOURCE  = DATA_DIR / "mfe" / "ml_dataset_v10_mfe.parquet"
DEFAULT_MANIFEST = DATA_DIR / "originals" / "ml_dataset_v10_manifest.json"
DEFAULT_OUTPUT  = DATA_DIR / "ml1_results_v12" / "combo_features_v12.parquet"
DEFAULT_VERSION_TAG = "v10"  # source of the combos → global_combo_id prefix

# MNQ economics and cost model
DOLLARS_PER_POINT = 2.0
COST_PER_CONTRACT_RT = 5.0

# Chronological 80/20 split on NQ_1min.csv: train partition spans 5.8056 years.
YEARS_SPAN_TRAIN = 5.8056

# Walk-forward window count.
N_WINDOWS = 5
# Minimum trades per window to count the window's Sharpe as valid. Fewer than
# this and the window's Sharpe is noise, so we skip that window.
MIN_TRADES_PER_WINDOW = 30
# Minimum valid windows per combo. Below this and the combo itself gets
# dropped — can't compute a stable median/std.
MIN_VALID_WINDOWS = 4

# Overall n_trades floor (pre-window). A combo with N_WINDOWS * MIN_TRADES_PER_WINDOW
# = 150 trades is the theoretical minimum; we set this higher to filter out
# degenerate low-frequency combos where even a full-period Sharpe is noise.
MIN_TRADES = 500

# Parameter columns — copied from the source parquet. These go to the model.
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
    *PARAM_COLS,
    "label_win", "net_pnl_dollars", "r_multiple",
    "mfe_points", "mae_points", "stop_distance_pts",
    "hold_bars", "zscore_entry",
]


def load_manifest_resolved_stops(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text())
    out: dict[int, float] = {}
    for e in data:
        if e.get("status") != "completed":
            continue
        resolved = e.get("stop_fixed_pts_resolved")
        if resolved is None:
            continue
        out[int(e["combo_id"])] = float(resolved)
    return out


def accumulate_from_parquet(src: Path) -> dict[int, dict]:
    """Stream parquet row groups, accumulate per-combo trade arrays in trade order.

    Within a combo, np.concatenate of the per-row-group slices preserves the
    trade entry order (the sweep runner writes trades in entry-bar order).
    """
    pf = pq.ParquetFile(str(src))
    cols = [c for c in READ_COLS if c in pf.schema_arrow.names]

    acc_pnl: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_win: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_rmul: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_stop: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_mfe: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_mae: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_hold: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_zentry: dict[int, list[np.ndarray]] = defaultdict(list)
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


def _window_sharpe(gross_pnl: np.ndarray, n_window: int,
                   window_fraction_of_year: float) -> float | None:
    """Annualized 1-contract net Sharpe for a single ordinal window.

    Returns None if the window has <2 trades or zero std — the caller filters
    those out of the median/std aggregates.
    """
    if n_window < 2:
        return None
    net = gross_pnl - COST_PER_CONTRACT_RT
    std = float(np.std(net, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return None
    trades_per_year_window = n_window / window_fraction_of_year
    return float(np.mean(net) / std * np.sqrt(trades_per_year_window))


def compute_combo_row(cid: int, bundle: dict, resolved_stop: float | None,
                      version_tag: str) -> dict | None:
    n = len(bundle["pnl"])
    if n < MIN_TRADES:
        return None

    rmul = bundle["rmul"]
    stop = bundle["stop"]
    gross_pnl = rmul * stop * DOLLARS_PER_POINT  # 1-contract gross per trade
    net_pnl = gross_pnl - COST_PER_CONTRACT_RT

    # Walk-forward windows: equal-size ordinal slices.
    window_edges = np.linspace(0, n, N_WINDOWS + 1, dtype=int)
    window_fraction = YEARS_SPAN_TRAIN / N_WINDOWS
    window_sharpes: list[float] = []
    for wi in range(N_WINDOWS):
        lo, hi = int(window_edges[wi]), int(window_edges[wi + 1])
        w_n = hi - lo
        if w_n < MIN_TRADES_PER_WINDOW:
            continue
        s = _window_sharpe(gross_pnl[lo:hi], w_n, window_fraction)
        if s is None:
            continue
        window_sharpes.append(s)
    if len(window_sharpes) < MIN_VALID_WINDOWS:
        return None

    ws = np.asarray(window_sharpes, dtype=np.float64)
    target_median_wf = float(np.median(ws))
    target_std_wf = float(np.std(ws, ddof=1)) if len(ws) >= 2 else 0.0
    target_robust = target_median_wf - 0.5 * target_std_wf

    # Full-period Sharpe (audit only — NOT the training target)
    std_net_full = float(np.std(net_pnl, ddof=1)) if n >= 2 else 0.0
    std_gross_full = float(np.std(gross_pnl, ddof=1)) if n >= 2 else 0.0
    trades_per_year = n / YEARS_SPAN_TRAIN
    full_net_sharpe = (float(np.mean(net_pnl) / std_net_full * np.sqrt(trades_per_year))
                       if std_net_full > 0 else 0.0)
    full_gross_sharpe = (float(np.mean(gross_pnl) / std_gross_full * np.sqrt(trades_per_year))
                        if std_gross_full > 0 else 0.0)

    # Audit-only trade-stream summaries (prefix `audit_` so the trainer's
    # EXCLUDE_PREFIX rule drops them cleanly)
    win = bundle["win"]
    mfe = bundle["mfe"]
    mae = bundle["mae"]
    hold = bundle["hold"]
    zentry = bundle["zentry"]
    mfe_mae = np.where(np.abs(mae) > 0, mfe / np.abs(mae), 0.0)

    resolved_stop_final = float(resolved_stop) if resolved_stop is not None \
                          else float(np.median(stop))

    row: dict = {
        "global_combo_id": f"{version_tag}_{cid}",
        "combo_id": cid,
        # === TARGETS (all three stored; model trains on target_robust_sharpe)
        "target_robust_sharpe": target_robust,
        "target_median_wf_sharpe": target_median_wf,
        "target_std_wf_sharpe": target_std_wf,
        "target_n_valid_windows": int(len(ws)),
        # === ENGINEERED FEATURES (parameter-derived only — model SEES these)
        "stop_fixed_pts_resolved": resolved_stop_final,
        # friction as fraction of $2500 risk at 5% equity:
        #   contracts = 2500 / (stop_pts * 2); friction = contracts * 5
        #   friction_pct_of_risk = friction / 2500 = 5 / (stop_pts * 2) = 2.5 / stop_pts
        "friction_pct_of_risk": 2.5 / max(resolved_stop_final, 1e-6),
        # === AUDIT ONLY (model does NOT see these — EXCLUDE_PREFIX='audit_')
        "audit_n_trades": int(n),
        "audit_trades_per_year": float(trades_per_year),
        "audit_full_net_sharpe": full_net_sharpe,
        "audit_full_gross_sharpe": full_gross_sharpe,
        "audit_gross_net_gap": full_gross_sharpe - full_net_sharpe,
        "audit_gross_win_rate": float(np.mean(win)),
        "audit_median_stop_pts": float(np.median(stop)),
        "audit_median_hold_bars": float(np.median(hold)),
        "audit_p90_hold_bars": float(np.quantile(hold, 0.9)),
        "audit_median_mfe_pts": float(np.median(mfe)),
        "audit_median_mae_pts": float(np.median(mae)),
        "audit_median_mfe_mae_ratio": float(np.median(mfe_mae)),
        "audit_zscore_entry_median_abs": float(np.median(np.abs(zentry))),
        "audit_cost_as_pct_of_edge": (
            float(COST_PER_CONTRACT_RT / max(float(np.mean(gross_pnl)), 1e-6))
            if float(np.mean(gross_pnl)) > 0 else float("inf")
        ),
    }

    # Parameter features — typed for LightGBM
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="Input MFE parquet (default: v10 MFE)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="Input manifest JSON (default: v10 originals manifest)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG,
                    help="Prefix for global_combo_id (e.g. v10, v11)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[v12] Source: {args.source}", flush=True)
    print(f"[v12] Output: {args.output}", flush=True)
    print(f"[v12] Target: robust_wf_sharpe = median - 0.5*std, K={N_WINDOWS} windows",
          flush=True)
    print(f"[v12] Gates: n_trades >= {MIN_TRADES}, "
          f"min {MIN_VALID_WINDOWS} valid windows, "
          f"min {MIN_TRADES_PER_WINDOW} trades/window", flush=True)
    print()

    resolved_stops = load_manifest_resolved_stops(args.manifest)
    print(f"[v12] Loaded {len(resolved_stops)} resolved stops from manifest",
          flush=True)

    print("[v12] Streaming parquet...", flush=True)
    combos = accumulate_from_parquet(args.source)

    print("[v12] Computing per-combo features + walk-forward target...", flush=True)
    rows: list[dict] = []
    gated_n = gated_windows = 0
    for cid, bundle in combos.items():
        if len(bundle["pnl"]) < MIN_TRADES:
            gated_n += 1
            continue
        row = compute_combo_row(cid, bundle, resolved_stops.get(cid),
                                args.version_tag)
        if row is None:
            gated_windows += 1
            continue
        rows.append(row)
    print(f"[v12] Kept {len(rows)} / {len(combos)} combos "
          f"(gated: {gated_n} by n_trades, {gated_windows} by valid windows)",
          flush=True)

    df = pd.DataFrame(rows)
    n_nan_target = int(df["target_robust_sharpe"].isna().sum())
    assert n_nan_target == 0, f"NaN in target_robust_sharpe: {n_nan_target}"

    print(f"[v12] target_robust_sharpe: "
          f"p5={df['target_robust_sharpe'].quantile(0.05):.3f}, "
          f"p50={df['target_robust_sharpe'].median():.3f}, "
          f"p95={df['target_robust_sharpe'].quantile(0.95):.3f}",
          flush=True)
    print(f"[v12] median vs full-period correlation: "
          f"{df[['target_median_wf_sharpe','audit_full_net_sharpe']].corr().iloc[0,1]:.3f}",
          flush=True)

    df.to_parquet(args.output, index=False)
    print(f"[v12] Wrote {args.output} "
          f"({args.output.stat().st_size/1e6:.1f} MB, "
          f"{len(df)} rows, {len(df.columns)} cols)", flush=True)


if __name__ == "__main__":
    main()
