"""Probe 3 gate §4.2 — Parameter neighborhood on combo 865 1h test partition.

Builds a 3-axis x 3-level grid (27 combos) varying:
  - z_band_k        ∈ { ×0.95, ×1.00, ×1.05 } around 2.3043838978381697
  - stop_fixed_pts  ∈ { ×0.95, ×1.00, ×1.05 } around 17.017749351216196
  - min_rr          ∈ { ×0.95, ×1.00, ×1.05 } around 1.8476511046163293

All other 865 parameters frozen at the Probe 2 manifest values
(data/ml/probe2/combo865_1h_test_manifest.json). Writes the 27-combo list
to data/ml/probe3/param_nbhd/combos.json, then invokes param_sweep.py via
subprocess with --explicit-combos-json to produce the parquet.

Gate §4.2 composite: ≥ 14 of 27 combos pass the Probe 2 §4 three-gate set:
  net_sharpe ≥ 1.3  AND  n_trades ≥ 50  AND  net_$/yr ≥ 5000  (per combo)

References:
- tasks/probe3_preregistration.md §2.2, §3.2.2, §4.2
- tasks/probe3_multiplicity_memo.md §3 (binomial calibration)
- Probe 2 manifest (authoritative combo 865 engine-grade parameters).

Runs on sweep-runner-1 after remote `git pull`. Expected wall-clock ~5 min.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Base combo 865 (engine-grade values from Probe 2 1h manifest) ────────────
BASE_COMBO_865: dict = {
    "combo_id":                 865,
    "z_band_k":                 2.3043838978381697,
    "z_window":                 41,
    "volume_zscore_window":     47,
    "ema_fast":                 6,
    "ema_slow":                 48,
    "stop_method":              "fixed",
    "stop_fixed_pts":           17.017749351216196,
    "atr_multiplier":           None,
    "swing_lookback":           None,
    "min_rr":                   1.8476511046163293,
    "exit_on_opposite_signal":  False,
    "use_breakeven_stop":       False,
    "max_hold_bars":            120,
    "zscore_confirmation":      False,
    "z_input":                  "returns",
    "z_anchor":                 "rolling_mean",
    "z_denom":                  "parkinson",
    "z_type":                   "parametric",
    "z_window_2":               47,
    "z_window_2_weight":        0.3308286558845456,
    "volume_entry_threshold":   0.0,
    "vol_regime_lookback":      0,
    "vol_regime_min_pct":       0.0,
    "vol_regime_max_pct":       1.0,
    "session_filter_mode":      0,
    "tod_exit_hour":            0,
    "entry_timing_offset":      1,
    "fill_slippage_ticks":      1,
    "cooldown_after_exit_bars": 3,
}

# ── Gate thresholds (Probe 2 §4 three-gate set; see preregistration §4.2) ────
YEARS_SPAN_TEST  = 1.4799
SHARPE_GATE      = 1.3
NTRADES_GATE     = 50
DOLLARS_GATE     = 5_000.0
COMPOSITE_PASS_N = 14    # of 27
GRID_N           = 27

OUT_DIR     = REPO_ROOT / "data" / "ml" / "probe3" / "param_nbhd"
COMBOS_JSON = OUT_DIR / "combos.json"
PARQUET_OUT = OUT_DIR / "combos.parquet"
READOUT_OUT = REPO_ROOT / "data" / "ml" / "probe3" / "param_nbhd.json"


def _build_grid() -> list[dict]:
    """3 axes × 3 levels = 27 combos. combo_id = 10_000 + i (disjoint from sampler)."""
    levels = [0.95, 1.00, 1.05]
    combos: list[dict] = []
    i = 0
    for zk in levels:
        for sp in levels:
            for rr in levels:
                c = dict(BASE_COMBO_865)
                c["combo_id"] = 10_000 + i
                c["z_band_k"]       = BASE_COMBO_865["z_band_k"]       * zk
                c["stop_fixed_pts"] = BASE_COMBO_865["stop_fixed_pts"] * sp
                c["min_rr"]         = BASE_COMBO_865["min_rr"]         * rr
                c["_grid_z_band_k_mult"]       = zk
                c["_grid_stop_fixed_pts_mult"] = sp
                c["_grid_min_rr_mult"]         = rr
                combos.append(c)
                i += 1
    assert len(combos) == GRID_N, f"expected {GRID_N} combos, got {len(combos)}"
    return combos


def _write_combos_json(combos: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Strip diagnostic keys (_grid_*) before passing to engine — engine only
    # tolerates _COMBO_META_KEYS. Preserve them in a sidecar for post-hoc audit.
    engine_combos: list[dict] = []
    grid_sidecar: list[dict] = []
    for c in combos:
        ec = {k: v for k, v in c.items() if not k.startswith("_grid_")}
        engine_combos.append(ec)
        grid_sidecar.append({
            "combo_id": c["combo_id"],
            "z_band_k_mult":       c["_grid_z_band_k_mult"],
            "stop_fixed_pts_mult": c["_grid_stop_fixed_pts_mult"],
            "min_rr_mult":         c["_grid_min_rr_mult"],
        })
    COMBOS_JSON.write_text(json.dumps(engine_combos, indent=2, default=str))
    (OUT_DIR / "grid_sidecar.json").write_text(
        json.dumps(grid_sidecar, indent=2, default=str)
    )
    print(f"[param_nbhd] wrote {len(engine_combos)} engine combos -> "
          f"{COMBOS_JSON}")


def _run_sweep() -> None:
    """Invoke param_sweep.py with --explicit-combos-json on 1h test partition."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "param_sweep.py"),
        "--explicit-combos-json", str(COMBOS_JSON),
        "--timeframe",            "1h",
        "--eval-partition",       "test",
        "--output",               str(PARQUET_OUT),
        "--workers",              "1",
    ]
    print(f"[param_nbhd] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"param_sweep.py returned {proc.returncode}")


def _combo_gate(pnl: np.ndarray) -> dict:
    """Apply Probe 2 §4 three-gate set to one combo's trade PnL series."""
    n = int(pnl.size)
    if n < 2:
        return {
            "n_trades":             n,
            "net_sharpe":           0.0,
            "net_dollars_per_year": 0.0,
            "sharpe_pass":          False,
            "n_trades_pass":        n >= NTRADES_GATE,
            "dollars_pass":         False,
            "combo_pass":           False,
        }
    mean_pnl = float(pnl.mean())
    std_pnl  = float(pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        net_sharpe = 0.0
    else:
        net_sharpe = (mean_pnl / std_pnl) * math.sqrt(n / YEARS_SPAN_TEST)
    net_dollars_per_year = mean_pnl * n / YEARS_SPAN_TEST
    sharpe_pass  = net_sharpe          >= SHARPE_GATE
    n_trades_pass = n                  >= NTRADES_GATE
    dollars_pass = net_dollars_per_year >= DOLLARS_GATE
    return {
        "n_trades":             n,
        "net_sharpe":           float(net_sharpe),
        "net_dollars_per_year": float(net_dollars_per_year),
        "sharpe_pass":          bool(sharpe_pass),
        "n_trades_pass":        bool(n_trades_pass),
        "dollars_pass":         bool(dollars_pass),
        "combo_pass":           bool(sharpe_pass and n_trades_pass and dollars_pass),
    }


def _apply_gates(combos: list[dict]) -> dict:
    print(f"[param_nbhd] reading {PARQUET_OUT}")
    df = pd.read_parquet(PARQUET_OUT)
    per_combo: list[dict] = []
    for c in combos:
        cid = c["combo_id"]
        sub = df[df["combo_id"] == cid]
        pnl = sub["net_pnl_dollars"].to_numpy() if len(sub) else np.array([])
        g = _combo_gate(pnl)
        per_combo.append({
            "combo_id":            cid,
            "z_band_k_mult":       c["_grid_z_band_k_mult"],
            "stop_fixed_pts_mult": c["_grid_stop_fixed_pts_mult"],
            "min_rr_mult":         c["_grid_min_rr_mult"],
            "z_band_k":            c["z_band_k"],
            "stop_fixed_pts":      c["stop_fixed_pts"],
            "min_rr":              c["min_rr"],
            **g,
        })
    n_pass = sum(1 for r in per_combo if r["combo_pass"])
    gate_pass = n_pass >= COMPOSITE_PASS_N
    return {
        "gate_id":      "§4.2_param_nbhd",
        "grid_n":       GRID_N,
        "n_pass":       int(n_pass),
        "threshold":    COMPOSITE_PASS_N,
        "thresholds":   {
            "net_sharpe":           SHARPE_GATE,
            "n_trades":             NTRADES_GATE,
            "net_dollars_per_year": DOLLARS_GATE,
            "composite_pass_n":     COMPOSITE_PASS_N,
        },
        "years_span_test": YEARS_SPAN_TEST,
        "per_combo":    per_combo,
        "gate_pass":    bool(gate_pass),
    }


def main() -> None:
    combos = _build_grid()
    _write_combos_json(combos)
    _run_sweep()
    readout = _apply_gates(combos)
    READOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    READOUT_OUT.write_text(json.dumps(readout, indent=2, default=str))
    print(f"[param_nbhd] wrote {READOUT_OUT}")
    print(f"[param_nbhd] n_pass = {readout['n_pass']} / {GRID_N}  "
          f"(threshold = {COMPOSITE_PASS_N})")
    print(f"[param_nbhd] GATE §4.2 -> "
          f"{'PASS' if readout['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
