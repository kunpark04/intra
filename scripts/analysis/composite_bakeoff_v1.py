"""Walk-forward bakeoff for candidate composite score formulas.

For every combo in the sweep parquets, splits its per-trade rows into
early (first 80%) vs late (last 20%) — a per-combo chronological slice
(rows are emitted in trade-close order). Aggregates sizing-invariant
primitives on each slice, then asks:

    For each candidate composite formula computed on EARLY primitives,
    how well does its cross-combo rank predict the combo's LATE-period
    realised metrics?

Correlation via Spearman rank. The composite whose early rank best
predicts late performance across multiple truth criteria is the most
defensible generalisation of "composite_score".

Output: `data/ml/composite_bakeoff_v1.json` with a full rho table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
ORIG_DIR = REPO / "data" / "ml" / "originals"
OUT_PATH = REPO / "data" / "ml" / "composite_bakeoff_v1.json"
SWEEP_VERSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RISK_DOLLARS = 500.0
EARLY_FRACTION = 0.80
MIN_EARLY = 80
MIN_LATE = 20


def aggregate_slice(r_mult: np.ndarray, label_win: np.ndarray) -> dict:
    """Sizing-invariant aggregates for one slice of a combo's trades."""
    n = len(r_mult)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "sharpe": np.nan,
                "pf": np.nan, "total_pnl": 0.0, "max_dd": 0.0}
    pnl = RISK_DOLLARS * r_mult
    avg_r = float(r_mult.mean())
    std_r = float(r_mult.std(ddof=1)) if n > 1 else 0.0
    sharpe = avg_r / std_r if std_r > 0 else 0.0
    wins_pnl = float(pnl[r_mult > 0].sum())
    loss_pnl = float(-pnl[r_mult < 0].sum())
    pf = wins_pnl / loss_pnl if loss_pnl > 0 else np.inf
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max())
    return {
        "n": int(n),
        "wr": float(label_win.mean()),
        "avg_r": avg_r,
        "sharpe": sharpe,
        "pf": pf,
        "total_pnl": float(pnl.sum()),
        "max_dd": max_dd,
    }


def split_combo(r_mult: np.ndarray, label_win: np.ndarray) -> dict:
    n = len(r_mult)
    split = int(EARLY_FRACTION * n)
    e = aggregate_slice(r_mult[:split], label_win[:split])
    l = aggregate_slice(r_mult[split:], label_win[split:])
    return {
        "early_n": e["n"], "early_wr": e["wr"], "early_avg_r": e["avg_r"],
        "early_sharpe": e["sharpe"], "early_pf": e["pf"],
        "early_total_pnl": e["total_pnl"], "early_max_dd": e["max_dd"],
        "late_n": l["n"], "late_wr": l["wr"], "late_avg_r": l["avg_r"],
        "late_sharpe": l["sharpe"], "late_pf": l["pf"],
        "late_total_pnl": l["total_pnl"], "late_max_dd": l["max_dd"],
    }


def process_version(version: int) -> pd.DataFrame:
    path = ORIG_DIR / f"ml_dataset_v{version}.parquet"
    print(f"[v{version}] reading {path.name}")
    tbl = pq.read_table(path, columns=["combo_id", "r_multiple", "label_win"])
    df = tbl.to_pandas()
    del tbl
    n_trades = len(df)
    r = df["r_multiple"].to_numpy(np.float64)
    w = df["label_win"].to_numpy(np.int8)
    cids = df["combo_id"].to_numpy()
    del df

    # Find contiguous groups. Combos are written contiguously per the sweep
    # runner; verify via np.diff and fall back to pandas groupby if not.
    change = np.flatnonzero(np.diff(cids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_trades]])
    if len(np.unique(cids[starts])) != len(starts):
        print(f"[v{version}]   non-contiguous combos detected; using groupby")
        tmp = pd.DataFrame({"combo_id": cids, "r_multiple": r, "label_win": w})
        rows = []
        for cid, g in tmp.groupby("combo_id", sort=False):
            rec = split_combo(g["r_multiple"].to_numpy(np.float64),
                              g["label_win"].to_numpy(np.int8))
            rec["global_combo_id"] = f"v{version}_{cid}"
            rows.append(rec)
    else:
        rows = []
        for s, e in zip(starts, ends):
            rec = split_combo(r[s:e], w[s:e])
            rec["global_combo_id"] = f"v{version}_{cids[s]}"
            rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"[v{version}]   {n_trades:,} trades, {len(out):,} combos")
    return out


def build_composites(elig: pd.DataFrame) -> list[str]:
    """Compute candidate composite scores on EARLY primitives; return col names."""
    e_n = elig["early_n"].to_numpy(np.float64)
    e_r = elig["early_avg_r"].to_numpy(np.float64)
    e_s = elig["early_sharpe"].to_numpy(np.float64)
    e_w = elig["early_wr"].to_numpy(np.float64)
    e_pf = elig["early_pf"].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
    e_pnl = elig["early_total_pnl"].to_numpy(np.float64)
    e_dd = elig["early_max_dd"].to_numpy(np.float64)
    e_pf_clipped = np.clip(e_pf, 0, 10)

    elig["C1_fixed_sharpe_x_logn"] = e_s * np.log1p(e_n)
    elig["C2_avg_r_x_sqrtn"] = e_r * np.sqrt(e_n)
    elig["C3_avg_r_x_n"] = e_r * e_n
    elig["C4_pf_clip_x_logn"] = e_pf_clipped * np.log1p(e_n)
    elig["C5_fixed_sharpe"] = e_s
    elig["C6_avg_r"] = e_r
    elig["C7_wr_edge_x_sqrtn"] = (e_w - 0.5) * np.sqrt(e_n)
    elig["C8_pnl_minus_dd"] = e_pnl - e_dd
    elig["C9_sortino_proxy"] = e_r / np.where(np.sqrt(e_dd) > 0, np.sqrt(e_dd), np.nan)
    return [c for c in elig.columns if c.startswith("C")]


def main() -> None:
    frames = [process_version(v) for v in SWEEP_VERSIONS]
    agg = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combos aggregated: {len(agg):,}")

    elig = agg[(agg["early_n"] >= MIN_EARLY) & (agg["late_n"] >= MIN_LATE)].copy()
    print(f"Eligible (early>={MIN_EARLY}, late>={MIN_LATE}): {len(elig):,}")

    composites = build_composites(elig)
    truths = ["late_total_pnl", "late_sharpe", "late_avg_r", "late_pf", "late_wr"]

    for c in composites + truths:
        elig[c] = elig[c].replace([np.inf, -np.inf], np.nan)

    rho_tbl = pd.DataFrame(index=composites, columns=truths, dtype=float)
    for c in composites:
        for t in truths:
            mask = elig[c].notna() & elig[t].notna()
            if mask.sum() < 10:
                continue
            rho, _ = spearmanr(elig.loc[mask, c], elig.loc[mask, t])
            rho_tbl.loc[c, t] = rho

    rho_tbl["mean_signed"] = rho_tbl[truths].mean(axis=1)
    rho_tbl["mean_abs"] = rho_tbl[truths].abs().mean(axis=1)
    rho_tbl = rho_tbl.sort_values("mean_signed", ascending=False)

    print("\n=== Spearman rho: early composite vs late realised ===")
    print(rho_tbl.round(3).to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "within-combo 80/20 rank-based split (no wall-clock); "
                  "rho = cross-combo Spearman(early_composite, late_realised)",
        "risk_dollars_per_trade": RISK_DOLLARS,
        "eligibility": {"min_early_n": MIN_EARLY, "min_late_n": MIN_LATE,
                        "eligible_combos": int(len(elig)),
                        "total_combos": int(len(agg))},
        "truths": truths,
        "rho_table": rho_tbl.round(4).to_dict(orient="index"),
    }
    import json
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
