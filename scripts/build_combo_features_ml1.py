"""B5: Build filtered combo_features.parquet for ML#1 retraining.

For each combo across v2-v10 mfe parquets:
  1. Load its trades + V2 ENTRY_FEATURES columns.
  2. Predict P(win | features, combo.min_rr) using V2 model.
  3. Sweep absolute E[R] thresholds, pick the one maximising Sharpe with
     guardrail (n_trades >= MIN_N, sharpe < MAX_SHARPE). Fallback: thr=0.
  4. Aggregate filtered trades into combo-level metrics.

Output: data/ml/lgbm_results_v2filtered/combo_features.parquet
Next: run ml1_surrogate.py --skip-aggregation --output-dir data/ml/lgbm_results_v2filtered
"""
from __future__ import annotations
import gc
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import adaptive_rr_model_v2 as v2
import ml1_surrogate as mlo

SIDE_MAP = {"long": 0, "short": 1}
STOP_METHOD_MAP = {"fixed": 0, "atr": 1, "swing": 2}

DATA_DIR = REPO / "data/ml"
OUTPUT_DIR = DATA_DIR / "lgbm_results_v2filtered"
V2_MODEL_PATH = DATA_DIR / "adaptive_rr_v2" / "adaptive_rr_model_v1.txt"

THRESHOLDS = np.round(np.arange(-0.5, 0.51, 0.1), 2).tolist()  # 11 points
MIN_N_KEEP = 50           # guardrail on filtered trade count
MAX_SHARPE = 1e10         # guardrail on degenerate zero-dd Sharpe
MIN_TRADES_COMBO = 30     # same floor ml1_surrogate uses

ENTRY_FEATS = v2.ENTRY_FEATURES               # 15 direct
COMBO_FEATS = v2.COMBO_FEATURES               # 2 (stop_method, exit_on_opposite_signal)
READ_COLS_BASE = ["combo_id", "label_win", "net_pnl_dollars", "r_multiple"]
READ_COLS = READ_COLS_BASE + [c for c in ENTRY_FEATS] + [c for c in COMBO_FEATS if c in ENTRY_FEATS + COMBO_FEATS]


def load_manifest_params(version: int) -> dict[int, dict]:
    path = DATA_DIR / "originals" / f"ml_dataset_v{version}_manifest.json"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for entry in json.loads(path.read_text()):
        if entry.get("status") != "completed":
            continue
        out[int(entry["combo_id"])] = entry
    return out


def build_feature_matrix(grp: pd.DataFrame, combo: dict) -> pd.DataFrame:
    """Build the 20-column V2 feature matrix for a combo's trades.

    Replicates scripts/adaptive_vs_fixed_backtest_v1.build_features dtype
    conventions (int8 for side/stop_method/exit_on_opp, float32 elsewhere).
    """
    out = pd.DataFrame(index=grp.index)
    float_feats = [c for c in ENTRY_FEATS
                   if c not in ("side", "time_of_day_hhmm", "day_of_week")]
    for c in float_feats:
        out[c] = grp[c].astype(np.float32) if c in grp.columns else np.float32(np.nan)
    out["time_of_day_hhmm"] = grp["time_of_day_hhmm"].astype(int).astype(np.float32)
    out["day_of_week"] = grp["day_of_week"].astype(np.float32)
    out["side"] = grp["side"].map(SIDE_MAP).astype(np.int8)
    out["stop_method"] = np.int8(STOP_METHOD_MAP[str(combo["stop_method"])])
    out["exit_on_opposite_signal"] = np.int8(int(bool(combo["exit_on_opposite_signal"])))
    out["candidate_rr"] = np.float32(float(combo["min_rr"]))
    out["abs_zscore_entry"] = out["zscore_entry"].abs()
    out["rr_x_atr"] = (out["candidate_rr"] * out["atr_points"]).astype(np.float32)
    return out[v2.ALL_FEATURES]


def aggregate(pnls: np.ndarray, wins: np.ndarray, rmul: np.ndarray) -> dict:
    return mlo._aggregate_single_combo(pnls, wins, rmul)


def pick_threshold(ev: np.ndarray, pnls: np.ndarray, wins: np.ndarray,
                   rmul: np.ndarray) -> tuple[float, dict, float]:
    """Sweep thresholds, pick best Sharpe with guardrails. Returns (thr, metrics, skip_rate)."""
    best = None
    for thr in THRESHOLDS:
        keep = ev >= thr
        n_kept = int(keep.sum())
        if n_kept < MIN_N_KEEP:
            continue
        m = aggregate(pnls[keep], wins[keep], rmul[keep])
        s = m.get("sharpe_ratio", 0.0)
        if not np.isfinite(s) or s >= MAX_SHARPE:
            continue
        if best is None or s > best[1]["sharpe_ratio"]:
            best = (thr, m, float(1 - keep.mean()))
    if best is not None:
        return best
    # Fallback: thr=0 unconditionally
    keep = ev >= 0.0
    if keep.sum() == 0:
        keep = np.ones_like(ev, dtype=bool)
    m = aggregate(pnls[keep], wins[keep], rmul[keep])
    return (0.0, m, float(1 - keep.mean()))


def process_version(v: int, model, rows: list[dict]) -> None:
    """Row-group streaming: per-RG build features + predict, accumulate
    (pnl, win, rmul, ev) per combo. Keeps peak memory low so 9 versions
    fit in a 7 GB cgroup without OOM between versions."""
    path = DATA_DIR / "mfe" / f"ml_dataset_v{v}_mfe.parquet"
    if not path.exists():
        print(f"  [WARN] {path} missing", flush=True)
        return
    params = load_manifest_params(v)
    print(f"  v{v}: streaming {path.name} ({len(params)} combos in manifest)",
          flush=True)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    cols = [c for c in READ_COLS if c in pf.schema_arrow.names]

    cid_to_rr = {cid: float(params[cid]["min_rr"]) for cid in params}
    cid_to_sm = {cid: STOP_METHOD_MAP[str(params[cid]["stop_method"])]
                 for cid in params}
    cid_to_eo = {cid: int(bool(params[cid]["exit_on_opposite_signal"]))
                 for cid in params}

    # Per-combo arrays accumulated across row groups
    acc_pnl: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_win: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_rmul: dict[int, list[np.ndarray]] = defaultdict(list)
    acc_ev: dict[int, list[np.ndarray]] = defaultdict(list)

    t0 = time.time()
    n_rows_total = 0
    float_feats = [c for c in ENTRY_FEATS
                   if c not in ("side", "time_of_day_hhmm", "day_of_week")]

    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=cols).to_pandas()
        cid_int = chunk["combo_id"].astype(int).to_numpy()
        # Filter rows whose combo isn't in manifest
        mask = np.array([cid in cid_to_rr for cid in cid_int])
        if not mask.any():
            del chunk; continue
        chunk = chunk[mask].reset_index(drop=True)
        cid_int = cid_int[mask]

        feat = pd.DataFrame(index=chunk.index)
        for c in float_feats:
            feat[c] = chunk[c].astype(np.float32)
        feat["time_of_day_hhmm"] = chunk["time_of_day_hhmm"].astype(int).astype(np.float32)
        feat["day_of_week"] = chunk["day_of_week"].astype(np.float32)
        feat["side"] = chunk["side"].map(SIDE_MAP).astype(np.int8)
        feat["stop_method"] = pd.Series(cid_int).map(cid_to_sm).astype(np.int8).values
        feat["exit_on_opposite_signal"] = pd.Series(cid_int).map(cid_to_eo).astype(np.int8).values
        rr_arr = pd.Series(cid_int).map(cid_to_rr).astype(np.float32).values
        feat["candidate_rr"] = rr_arr
        feat["abs_zscore_entry"] = feat["zscore_entry"].abs()
        feat["rr_x_atr"] = (rr_arr * feat["atr_points"].values).astype(np.float32)
        X = feat[v2.ALL_FEATURES].values

        pwin = model.predict(X, num_threads=4)
        ev = pwin * rr_arr - (1.0 - pwin)
        pnls = chunk["net_pnl_dollars"].to_numpy(dtype=np.float64)
        wins = chunk["label_win"].to_numpy(dtype=np.int8)
        rmul = chunk["r_multiple"].to_numpy(dtype=np.float64)

        # Scatter per combo via argsort
        order = np.argsort(cid_int, kind="stable")
        cid_sorted = cid_int[order]
        uniq, starts = np.unique(cid_sorted, return_index=True)
        ends = np.append(starts[1:], len(cid_sorted))
        for cid, st, en in zip(uniq, starts, ends):
            idx = order[st:en]
            cid = int(cid)
            acc_pnl[cid].append(pnls[idx])
            acc_win[cid].append(wins[idx])
            acc_rmul[cid].append(rmul[idx])
            acc_ev[cid].append(ev[idx])

        n_rows_total += len(chunk)
        if (rg + 1) % 5 == 0 or rg == pf.metadata.num_row_groups - 1:
            elapsed = time.time() - t0
            print(f"    v{v}: rg {rg+1}/{pf.metadata.num_row_groups}, "
                  f"{n_rows_total:,} rows, {n_rows_total/max(elapsed,1e-9):.0f} rows/s",
                  flush=True)
        del chunk, feat, X, pwin, ev, pnls, wins, rmul

    print(f"  v{v}: streamed {n_rows_total:,} rows in {time.time()-t0:.1f}s, "
          f"{len(acc_pnl)} combos", flush=True)
    t0 = time.time()
    n_done = 0
    for cid in list(acc_pnl.keys()):
        pnls = np.concatenate(acc_pnl.pop(cid))
        wins = np.concatenate(acc_win.pop(cid))
        rmul = np.concatenate(acc_rmul.pop(cid))
        ev = np.concatenate(acc_ev.pop(cid))
        if len(pnls) < MIN_TRADES_COMBO:
            continue
        try:
            fixed = aggregate(pnls, wins, rmul)
            thr, filt, skip = pick_threshold(ev, pnls, wins, rmul)
            row = dict(filt)
            row["global_combo_id"] = f"v{v}_{cid}"
            row["optimal_threshold"] = thr
            row["filtered_n_trades"] = filt["n_trades"]
            row["skip_rate"] = skip
            fixed_sharpe = fixed.get("sharpe_ratio", 0.0)
            row["filter_lift_sharpe"] = (filt["sharpe_ratio"] - fixed_sharpe) if np.isfinite(fixed_sharpe) else np.nan
            row["fixed_sharpe_ratio"] = fixed_sharpe
            row["fixed_n_trades"] = fixed["n_trades"]
            rows.append(row)
            n_done += 1
            if n_done % 500 == 0:
                print(f"    v{v}: {n_done} combos aggregated", flush=True)
        except Exception as e:
            print(f"    [ERR] v{v}_{cid}: {e}", flush=True)
    print(f"  v{v}: processed {n_done} combos in {time.time()-t0:.1f}s", flush=True)
    gc.collect()


def main() -> None:
    import lightgbm as lgb
    print(f"[B5] Loading V2 model: {V2_MODEL_PATH}", flush=True)
    model = lgb.Booster(model_file=str(V2_MODEL_PATH))

    rows: list[dict] = []
    for v in range(2, 11):
        process_version(v, model, rows)

    print(f"\n[B5] Aggregated {len(rows)} combos total", flush=True)
    metrics_df = pd.DataFrame(rows)

    # Merge with parameter settings (reuse ml1_surrogate's manifest loader logic)
    print("[B5] Loading manifest params for merge...", flush=True)
    param_rows = []
    for v in range(2, 11):
        mp = DATA_DIR / "originals" / f"ml_dataset_v{v}_manifest.json"
        if not mp.exists():
            continue
        for entry in json.loads(mp.read_text()):
            if entry.get("status") != "completed" or entry.get("n_trades", 0) == 0:
                continue
            gid = f"v{v}_{entry['combo_id']}"
            row = {"global_combo_id": gid, "sweep_version": v}
            for col in mlo.PARAM_COLS:
                if col in entry:
                    row[col] = entry[col]
                elif col in mlo.ZSCORE_DEFAULTS:
                    row[col] = mlo.ZSCORE_DEFAULTS[col]
                elif col in mlo.V5_FILTER_DEFAULTS:
                    row[col] = mlo.V5_FILTER_DEFAULTS[col]
                else:
                    row[col] = np.nan
            param_rows.append(row)
    params_df = pd.DataFrame(param_rows)

    df = metrics_df.merge(params_df, on="global_combo_id", how="inner")
    print(f"[B5] Final merged: {len(df):,} combos", flush=True)

    for col in mlo.BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float).fillna(0).astype(int)
    for col in mlo.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)  # parquet-friendly; ml1_surrogate re-casts

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "combo_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[B5] Saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # Quick summary
    print("\n[B5] Threshold distribution:")
    print(df["optimal_threshold"].value_counts().sort_index().to_string())
    print(f"\n[B5] Median filter_lift_sharpe: {df['filter_lift_sharpe'].median():.3f}")
    print(f"[B5] Combos where filter beats fixed: "
          f"{(df['filter_lift_sharpe'] > 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
