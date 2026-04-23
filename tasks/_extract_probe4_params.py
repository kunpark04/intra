"""One-shot introspection: pull frozen parameter dicts for combo_ids 1298 and 664
from the v11 1h sweep parquet so the Probe 4 prereg §2.1/§2.2 can be populated.

Verifies (a) every targeted combo has trades in the parquet, (b) every
parameter column is constant across trades within a combo (so a single
dict represents the combo), (c) prints the dicts in Probe-3 §2.1 table format.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

PARQUET = Path("data/ml/originals/ml_dataset_v11_1h.parquet")
TARGET_COMBOS = [1298, 664]

PROBE3_PARAM_ORDER = [
    "z_band_k", "z_window", "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight", "volume_zscore_window",
    "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "max_hold_bars",
    "exit_on_opposite_signal", "use_breakeven_stop",
    "zscore_confirmation", "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
    "entry_timing_offset", "fill_slippage_ticks", "cooldown_after_exit_bars",
]

def main() -> None:
    print(f"Reading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    print(f"  rows={len(df):,}  cols={len(df.columns)}")
    cols_missing = [c for c in PROBE3_PARAM_ORDER if c not in df.columns]
    if cols_missing:
        print(f"  WARN: missing param cols in parquet: {cols_missing}")

    cols_present = [c for c in PROBE3_PARAM_ORDER if c in df.columns]

    out = {}
    for cid in TARGET_COMBOS:
        sub = df[df["combo_id"] == cid]
        n = len(sub)
        print(f"\n=== combo_id={cid}: {n} trades ===")
        if n == 0:
            print("  ABSENT in parquet")
            continue
        dict_for_combo = {}
        for col in cols_present:
            uniq = sub[col].dropna().unique()
            if len(uniq) == 0:
                # all NaN - record explicitly
                dict_for_combo[col] = None
            elif len(uniq) == 1:
                v = uniq[0]
                if hasattr(v, "item"):
                    v = v.item()
                dict_for_combo[col] = v
            else:
                # not constant - parameter varies within combo, abort
                dict_for_combo[col] = f"VARIES({uniq.tolist()[:5]})"
        out[cid] = dict_for_combo
        for col in cols_present:
            print(f"  {col} = {dict_for_combo[col]!r}")

    out_path = Path("tasks/_probe4_param_dicts.json")
    out_path.write_text(json.dumps({str(k): v for k, v in out.items()}, indent=2, default=str))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
