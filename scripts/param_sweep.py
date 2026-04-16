"""
param_sweep.py — Random parameter sweep for ML dataset generation.

Runs N random-sampled backtests across the strategy parameter space and pools
all resulting trades into a Parquet file. Per-combo stats are saved to a
manifest JSON file.

Usage:
    python scripts/param_sweep.py                              # 3000 combos, default ranges
    python scripts/param_sweep.py --combinations 500
    python scripts/param_sweep.py --hours 7
    python scripts/param_sweep.py --start-combo 300 --combinations 600
    python scripts/param_sweep.py --seed 1 --range-mode winrate --output data/ml/originals/ml_dataset_v2.parquet

Range modes:
    default   — original sweep ranges (seed=0 → data/ml/originals/ml_dataset.parquet)
    winrate   — win-rate-biased ranges + breakeven/time-exit/z-confirmation modes
                (seed=1 → data/ml/originals/ml_dataset_v2.parquet by convention)

Stop mechanisms (any one triggers clean flush + exit):
    Ctrl+C              — SIGINT caught; flushes current batch and exits
    --hours N           — stops at next combo boundary after N hours elapsed
    stop_sweep.txt      — drop this file in repo root; stops at next combo check
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as _BASE_CFG
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.indicators.ema import compute_ema
from src.indicators.zscore import compute_zscore, compute_volume_zscore
from src.indicators.atr import compute_atr
from src.indicators.zscore_variants import compute_zscore_v2, compute_vwap_session
from src.strategy import generate_signals

# Import the low-level core engines directly — we bypass run_backtest's
# equity-curve loop (O(n_bars) in Python = ~7 min on 2M bars per combo).
from src.backtest import (
    CYTHON_AVAILABLE, NUMBA_AVAILABLE,
    backtest_core_cy,        # Cython AOT (None if unavailable)
    _backtest_core,          # Numba JIT
    _backtest_core_numpy,    # pure NumPy fallback
)

# ── Output paths (defaults; overridden by CLI --output) ───────────────────────
_DEFAULT_PARQUET  = Path("data/ml/ml_dataset.parquet")
_DEFAULT_MANIFEST = Path("data/ml/sweep_manifest.json")
STOP_FILE         = Path("stop_sweep.txt")
BATCH_SIZE        = 100   # flush to disk every N completed combos
LOG_INTERVAL      = 200   # print progress every N completed combos

# Module-level references updated in main() after CLI parsing
PARQUET_PATH  = _DEFAULT_PARQUET
MANIFEST_PATH = _DEFAULT_MANIFEST

# ── Cache globals (populated in main(), read by worker processes via fork COW) ─
# On Linux, fork gives children copy-on-write access to the parent's memory.
# These ~1.3 GB caches are read-only in workers, so no physical duplication.
_cache_close_np: np.ndarray | None = None
_cache_high_np: np.ndarray | None = None
_cache_low_np: np.ndarray | None = None
_cache_open_np: np.ndarray | None = None
_cache_volume_np: np.ndarray | None = None
_cache_session_brk_np: np.ndarray | None = None
_cache_time_np: np.ndarray | None = None
_cache_hour_np: np.ndarray | None = None
_cache_atr_arr: np.ndarray | None = None
_cache_vwap_session_arr: np.ndarray | None = None
_cache_zscore: dict[int, np.ndarray] = {}
_cache_vol_zscore: dict[int, np.ndarray] = {}
_cache_ema_period: dict[int, np.ndarray] = {}
_cache_atr_pct: dict[int, np.ndarray] = {}
_cache_ddof: int = 0

# ── Fixed sweep constants (not swept) ────────────────────────────────────────
_FIXED = dict(
    SIGNAL_MODE           = "zscore_reversal",
    STARTING_EQUITY       = 50_000.0,
    RISK_PCT              = 0.05,
    FILL_MODEL            = "next_bar_open",
    SAME_BAR_COLLISION    = "tp_first",
    TICK_SIZE             = 0.25,
    MNQ_DOLLARS_PER_POINT = 2.0,
    ZSCORE_DDOF           = 0,
    SWING_BUFFER_TICKS    = 1,
)

# ── Column groups ─────────────────────────────────────────────────────────────
_COMBO_META_KEYS = [
    "combo_id", "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow", "stop_method", "stop_fixed_pts",
    "atr_multiplier", "swing_lookback", "min_rr", "exit_on_opposite_signal",
    "use_breakeven_stop", "max_hold_bars", "zscore_confirmation",
    # Z-score variant dimensions (V4+)
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    # V5+ entry filters and exit
    "volume_entry_threshold", "vol_regime_lookback",
    "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]
_TRADE_FEATURE_KEYS = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points",
    "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side",
]
_LABEL_KEYS = ["label_win", "net_pnl_dollars", "r_multiple"]
_PATH_METRIC_KEYS = ["mfe_points", "mae_points", "stop_distance_pts", "hold_bars"]

# ── Parkinson volatility constants ────────────────────────────────────────────
# Single-bar Parkinson estimator — no lookback, price-normalized.
# _PARKINSON_SCALE: converts (H-L) in points to Parkinson volatility in points
# _LOG_PARKINSON_SCALE: denominator for pct form: ln(H/L) / sqrt(4*ln2)
import math as _math
_PARKINSON_SCALE     = _math.sqrt(1.0 / (4.0 * _math.log(2)))   # ≈ 0.6006
_LOG_PARKINSON_SCALE = _math.sqrt(4.0 * _math.log(2))            # ≈ 1.6651


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_combos(n: int, rng_seed: int = 0, range_mode: str = "default") -> list[dict]:
    """Sample n random parameter combos (reproducible via rng_seed).

    range_mode="default"        — original sweep ranges
    range_mode="winrate"        — win-rate-biased ranges + new exit/entry modes
    range_mode="zscore_variants"— full grid of z-score formulation variants
    """
    rng = np.random.default_rng(seed=rng_seed)
    combos = []
    for i in range(n):
        if range_mode == "zscore_variants":
            ema_fast    = int(rng.integers(5, 16))
            ema_slow    = int(rng.integers(ema_fast + 5, 36))
            stop_method = str(rng.choice(["fixed", "atr", "swing"]))
            z_window_1  = int(rng.integers(10, 31))
            z_type      = str(rng.choice(["parametric", "quantile_rank"]))
            z_input     = str(rng.choice(["close", "returns", "typical_price"]))

            # Resolve anchor: returns must use rolling_mean (price-space mismatch)
            if z_input == "returns":
                z_anchor = "rolling_mean"
            else:
                z_anchor = str(rng.choice(["rolling_mean", "vwap_session", "ema_fast", "ema_slow"]))

            # Resolve denom: quantile_rank ignores denom → record "n/a" in metadata
            # returns cannot use atr (unit mismatch → zero trades)
            if z_type == "quantile_rank":
                z_denom = "n/a"
            elif z_input == "returns":
                z_denom = str(rng.choice(["rolling_std", "parkinson"]))
            else:
                z_denom = str(rng.choice(["rolling_std", "atr", "parkinson"]))

            # Second window: sample from [z_window_1+5, 51) to prevent overlap
            use_window_2 = bool(rng.integers(0, 2))
            w2_low = z_window_1 + 5
            if use_window_2 and w2_low < 51:
                z_window_2       = int(rng.integers(w2_low, 51))
                z_window_2_weight = float(rng.uniform(0.2, 0.5))
            else:
                z_window_2        = 0
                z_window_2_weight = 0.0

            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(1.5, 3.5)),
                "z_window":               z_window_1,
                "volume_zscore_window":   int(rng.integers(10, 31)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 3.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 11))  if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(2.0, 5.0)),
                "exit_on_opposite_signal": bool(rng.integers(0, 2)),
                "use_breakeven_stop":     bool(rng.integers(0, 2)),
                "max_hold_bars":          int(rng.choice([0, 0, 15, 30, 60, 120])),
                "zscore_confirmation":    bool(rng.integers(0, 2)),
                "z_input":               z_input,
                "z_anchor":              z_anchor,
                "z_denom":               z_denom,
                "z_type":                z_type,
                "z_window_2":            z_window_2,
                "z_window_2_weight":     z_window_2_weight,
                # V5 filters — off for non-v5 modes
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "winrate":
            # Win-rate-biased ranges (derived from correlation analysis of sweep v1)
            ema_fast    = int(rng.integers(5, 9))                           # 5–8 (top-50 median=6)
            ema_slow    = int(rng.integers(ema_fast + 5, min(ema_fast + 20, 26)))  # cap at 25
            stop_method = str(rng.choice(["swing", "swing", "atr", "fixed"]))     # swing weighted 2x
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(2.5, 3.5)),    # was (1.5, 3.5)
                "z_window":               int(rng.integers(20, 31)),        # was (10, 31)
                "volume_zscore_window":   int(rng.integers(10, 31)),        # unchanged
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 3.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 11))  if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(2.0, 3.0)),     # was (2.0, 5.0)
                "exit_on_opposite_signal": bool(rng.choice([True, True, True, False])),  # 75% True
                # New mode parameters
                "use_breakeven_stop":     bool(rng.choice([True, False])),
                "max_hold_bars":          int(rng.choice([0, 0, 15, 30, 60, 120])),  # 33% off
                "zscore_confirmation":    bool(rng.choice([True, False])),
                # Z-score variant keys — baseline values for winrate mode
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — off for non-v5 modes
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v4":
            # V4: data-driven tightened ranges from V3 correlation analysis.
            # Key changes vs V3:
            #   - min_rr: (1.0, 2.5)  [was (2.0, 5.0); r=−0.540, single strongest predictor]
            #   - z_window: (15, 28)   [was (10, 31); r=+0.255, low end underperforms]
            #   - z_band_k: (2.0, 3.2) [was (1.5, 3.5); low end underperforms]
            #   - ema_fast: (5, 9)     [was (5, 16); top combos cluster at 5-8]
            #   - ema_slow: cap 25     [was 36; r=−0.204]
            #   - stop_method: swing 50%, atr 25%, fixed 25%
            #   - boolean flags (use_breakeven_stop, zscore_confirmation, max_hold_bars) now tested
            #   - z-score variants dropped entirely (800-combo V3 test showed zero improvement)
            ema_fast    = int(rng.integers(5, 9))                                    # 5–8
            ema_slow    = int(rng.integers(ema_fast + 5, min(ema_fast + 20, 26)))   # cap at 25
            stop_method = str(rng.choice(["swing", "swing", "atr", "fixed"]))       # swing 50%
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(2.0, 3.2)),
                "z_window":               int(rng.integers(15, 29)),                 # [15, 28]
                "volume_zscore_window":   int(rng.integers(10, 31)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 3.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 11))  if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(1.0, 2.5)),
                "exit_on_opposite_signal": bool(rng.choice([True, True, True, False])),  # 75% True (V3: +1.3% wr)
                "use_breakeven_stop":     bool(rng.integers(0, 2)),
                "max_hold_bars":          int(rng.choice([0, 0, 15, 30, 60, 120])),
                "zscore_confirmation":    bool(rng.integers(0, 2)),
                # Z-score variant keys — fixed to best known values (variants showed no improvement in V3)
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — off for non-v5 modes
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v5":
            # V5: further tightened to high-signal zones from V4 analysis.
            # All parameters still vary — ranges just squeezed to elite performer territory.
            # Key changes vs V4:
            #   - min_rr: (1.0, 1.5)     [was (1.0, 2.5); elite p90=1.76; ultra-tight ≤1.1 → 58.3% WR]
            #   - z_window: (17, 26)      [was (15, 28); z=15 → 40.3% WR, drop low end]
            #   - z_band_k: (2.0, 3.0)   [was (2.0, 3.2); trim top slightly]
            #   - ema_fast: (5, 8)        [unchanged]
            #   - ema_slow: (10, 28)      [was capped at 25; cap was hurting WR by 5.5pp]
            #   - swing_lookback: (3, 5)  [was (3, 10); sharp performance cliff at 5]
            #   - atr_multiplier: (1.0, 1.5) [was (1.0, 3.0); degrades above 1.5]
            #   - stop_method: swing 60%, atr 20%, fixed 20%
            #   - use_breakeven_stop: still varies but weighted 75% False (8.7pp WR penalty when True)
            #   - zscore_confirmation: still varies but weighted 75% False (slight negative)
            #   - exit_on_opposite_signal: 70% True (mild preference)
            #   - max_hold_bars: weighted toward 0 and 15 (elite preference)
            #   - volume_zscore_window: (15, 25) [neutral predictor; tighten for speed]
            ema_fast    = int(rng.integers(5, 9))                                    # 5–8
            ema_slow    = int(rng.integers(max(ema_fast + 5, 10), 29))              # [ema_fast+5, 28]; no cap at 25
            stop_method = str(rng.choice(["swing", "swing", "swing", "atr", "fixed"]))  # swing 60%
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(2.0, 3.0)),
                "z_window":               int(rng.integers(17, 27)),                 # [17, 26]
                "volume_zscore_window":   int(rng.integers(15, 26)),                 # [15, 25]
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 1.5)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 6))   if stop_method == "swing" else None,  # [3, 5]
                "min_rr":                 float(rng.uniform(1.0, 1.5)),
                "exit_on_opposite_signal": bool(rng.choice([True, True, True, False])),         # 75% True
                "use_breakeven_stop":     bool(rng.choice([False, False, False, True])),        # 75% False
                "max_hold_bars":          int(rng.choice([0, 0, 0, 15, 15, 30, 60, 120])),     # weighted 0/15
                "zscore_confirmation":    bool(rng.choice([False, False, False, True])),        # 75% False
                # Z-score variant keys — fixed to best known values
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 entry filters — weighted toward "off" states to preserve HF trade volume
                "volume_entry_threshold": float(rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0])),  # 43% off
                "vol_regime_lookback":    int(rng.choice([0, 0, 0, 50, 100, 200])),                   # 50% off
                "vol_regime_min_pct":     float(rng.choice([0.0, 0.0, 0.1, 0.2, 0.3])),              # 40% no gate
                "vol_regime_max_pct":     float(rng.choice([1.0, 1.0, 0.9, 0.8, 0.7])),              # 40% no gate
                "session_filter_mode":    int(rng.choice([0, 0, 0, 1, 2, 3])),                        # 50% all hours
                "tod_exit_hour":          int(rng.choice([0, 0, 0, 14, 16, 20, 22])),                 # 43% off
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v6":
            # V6: locked on V5-validated winners, flags fixed to proven-best settings.
            # Key changes vs V5:
            #   - min_rr: (1.0, 1.3)    [was 1.0-1.5; elite cluster ≤1.3 dominates]
            #   - z_window: (18, 24)     [was 17-26; trim noisy extremes]
            #   - z_band_k: (2.1, 2.5)  [was 2.0-3.0; top-20 mean=2.34]
            #   - ema_slow: (15, 25)     [was up to 28; top-20 mean=15.3, ≥25 underperforms]
            #   - swing_lookback: (3, 4) [was 3-5; sharp cliff at 5]
            #   - atr_mult: (1.0, 1.2)  [was 1.0-1.5; degrades above 1.2]
            #   - stop_method: swing 70%, atr 20%, fixed 10%
            #   - use_breakeven_stop: FIXED False  (8.7pp WR penalty when True)
            #   - zscore_confirmation: FIXED False  (slight negative in V5)
            #   - exit_on_opposite_signal: FIXED True  (best in V5)
            #   - vol_regime, tod_exit, session_filter, volume_threshold: ALL DISABLED
            #     (vol_regime hurt -4.1pp; session filters halve trade count; tod marginal)
            ema_fast    = int(rng.integers(5, 9))                                   # 5–8 unchanged
            ema_slow    = int(rng.integers(max(ema_fast + 5, 15), 26))             # [max(ef+5,15), 25]
            stop_method = str(rng.choice(["swing", "swing", "swing", "swing", "swing",
                                          "swing", "swing", "atr", "atr", "fixed"]))  # swing 70%
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(2.1, 2.5)),
                "z_window":               int(rng.integers(18, 25)),                # [18, 24]
                "volume_zscore_window":   int(rng.integers(15, 26)),                # unchanged
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 1.2)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 5))   if stop_method == "swing" else None,  # [3, 4]
                "min_rr":                 float(rng.uniform(1.0, 1.3)),
                "exit_on_opposite_signal": True,                                    # FIXED True
                "use_breakeven_stop":     False,                                    # FIXED False
                "max_hold_bars":          int(rng.choice([0, 0, 0, 0, 0, 15, 15, 30])),  # strongly 0
                "zscore_confirmation":    False,                                    # FIXED False
                # Z-score variant keys — fixed to best known values
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — ALL DISABLED for V6 (vol_regime hurts; others filter too aggressively)
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v7":
            # V7: exploit V6's strong correlation signals (3000 combos, mean WR=56.2%).
            # Key changes vs V6:
            #   - z_window: (21, 27)     [was 18-24; r=+0.478 strongest predictor, top combos hit max=24]
            #   - ema_slow: (15, 20)     [was 15-25; r=-0.531 strongest neg predictor, top mean=16.85 vs bot=21.97]
            #   - ema_fast: 5-6 only     [was 5-8; r=-0.304, top mean=6.02, bottom mean=6.98]
            #   - min_rr: (1.0, 1.20)   [was 1.0-1.3; r=-0.312, top mean=1.10 vs bot=1.23]
            #   - stop_method: swing 85%, atr 15%, fixed 0%  [0% fixed in top 10%]
            #   - atr_mult: (1.0, 1.15) [was 1.0-1.2; top atr mean=1.071, max=1.194]
            #   - z_band_k: (2.15, 2.50) [was 2.1-2.5; minor tightening around top mean=2.309]
            #   - All fixed flags inherited from V6 (exit_opp=True, breakeven=False, zscore_confirm=False)
            ema_fast    = int(rng.integers(5, 7))                                    # 5–6 only
            ema_slow    = int(rng.integers(max(ema_fast + 5, 15), 21))              # [max(ef+5,15), 20]
            stop_method = str(rng.choice(["swing"] * 17 + ["atr"] * 3))             # swing 85%, atr 15%
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(2.15, 2.50)),
                "z_window":               int(rng.integers(21, 27)),                 # [21, 26]
                "volume_zscore_window":   int(rng.integers(15, 26)),                 # unchanged
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         None,                                      # fixed method dropped
                "atr_multiplier":         float(rng.uniform(1.0, 1.15)) if stop_method == "atr" else None,
                "swing_lookback":         float(rng.integers(3, 5))  if stop_method == "swing" else None,  # [3, 4]
                "min_rr":                 float(rng.uniform(1.0, 1.20)),
                "exit_on_opposite_signal": True,                                     # FIXED True
                "use_breakeven_stop":     False,                                     # FIXED False
                "max_hold_bars":          int(rng.choice([0, 0, 0, 0, 0, 15, 15, 30])),  # strongly 0
                "zscore_confirmation":    False,                                     # FIXED False
                # Z-score variant keys — fixed to best known values
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — ALL DISABLED (confirmed hurts or marginal in V5/V6)
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v8":
            # V8: DIVERSITY sweep — broad coverage of parameter space for ML training data.
            # Goal shifts from win-rate maximisation to representative signal diversity.
            # Win-rate optimisation (V3–V7) produced thin-sample, clustered combos that
            # risk overfitting the training window. V8 broadens all numeric ranges back
            # toward the full feasible space so the ML classifier trains on the complete
            # distribution of trade setups (bad, mediocre, and good) rather than only
            # cherry-picked low-frequency winners.
            #
            # What is KEPT fixed from prior learning (structural settings that are robust):
            #   - exit_on_opposite_signal=True  (confirmed best across V5/V6)
            #   - use_breakeven_stop=False       (8.7pp WR penalty when True)
            #   - zscore_confirmation=False      (consistently slight negative)
            #   - All V5 entry filters DISABLED  (vol_regime, session, tod all hurt)
            #
            # What is BROADENED (to cover full parameter distribution):
            #   - z_band_k: (1.5, 4.5)          [was 2.15-2.50; need low/high extremes]
            #   - z_window: (10, 31)             [was 21-26; need full range represented]
            #   - ema_fast: (5, 16)              [was 5-6 only]
            #   - ema_slow: (ema_fast+5, 41)     [was capped at 20; need long-slow coverage]
            #   - min_rr: (1.0, 5.0)             [was 1.0-1.20; need high R:R represented]
            #   - stop_method: equal 33/33/33    [was swing-heavy; need all types]
            #   - swing_lookback: (3, 12)        [was (3,4) only]
            #   - atr_multiplier: (1.0, 4.0)     [was 1.0-1.15]
            #   - stop_fixed_pts: (5, 50)        [was disabled; re-enabled for diversity]
            #   - max_hold_bars: equal across all levels including longer holds
            #   - volume_zscore_window: (10, 31) [full range]
            #
            # Trade count note: broader ranges will push median trades higher vs V7's
            # thin-sample (~150 trades) winners. This is intentional — we want combos
            # with 200-2000 trades for statistically robust ML labels.
            ema_fast    = int(rng.integers(5, 16))
            ema_slow    = int(rng.integers(ema_fast + 5, 41))
            stop_method = str(rng.choice(["fixed", "atr", "swing"]))          # equal 33%
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(1.5, 4.5)),
                "z_window":               int(rng.integers(10, 31)),           # [10, 30]
                "volume_zscore_window":   int(rng.integers(10, 31)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(5, 50))   if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 4.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 12))  if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(1.0, 5.0)),
                "exit_on_opposite_signal": True,                               # FIXED True (robust)
                "use_breakeven_stop":     False,                               # FIXED False (8.7pp penalty)
                "max_hold_bars":          int(rng.choice([0, 15, 30, 60, 120, 240])),  # equal spread
                "zscore_confirmation":    False,                               # FIXED False (robust)
                # Z-score variant keys — fixed to best known values
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — ALL DISABLED (confirmed hurts or marginal)
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v9":
            # V9: ULTRA-WIDE diversity sweep — push every parameter to its full feasible range.
            # Goal: maximum coverage of the trade-setup space for ML training.
            # V8 was already broad; V9 extends further into regions never tested:
            #   - z_window: (5, 50)          [was 10-30; short/long windows untested]
            #   - volume_zscore_window: (5, 50) [was 10-30]
            #   - ema_fast: (3, 20)          [was 5-15; periods 3-4 never tried]
            #   - ema_slow: (fast+5, 60)     [was up to 40; very slow EMAs untested]
            #   - stop_fixed_pts: (2, 100)   [was 5-50; tight < 5 and wide > 50 untested]
            #   - atr_multiplier: (0.5, 6.0) [was 1.0-4.0; sub-1 multiplier untested]
            #   - swing_lookback: (2, 20)    [was 3-11; 2-bar and >11 never tested]
            #   - max_hold_bars: adds 480, 720 [8h and 12h holds never tested]
            #   - z_window_2 / weight: 50% chance of active dual-window blend
            #     (only ~5% of all past combos ever used this)
            #
            # Fixed (from confirmed evidence — do not re-open):
            #   - exit_on_opposite_signal=True, use_breakeven_stop=False,
            #     zscore_confirmation=False, all V5 filters off,
            #     z-score type = close/rolling_mean/rolling_std/parametric
            ema_fast    = int(rng.integers(3, 20))
            ema_slow    = int(rng.integers(ema_fast + 5, min(ema_fast + 45, 61)))
            stop_method = str(rng.choice(["fixed", "atr", "swing"]))              # equal 33%

            # Dual-window z-score blend: 50% chance of active second window
            z_window_1 = int(rng.integers(5, 51))
            use_w2     = bool(rng.integers(0, 2))
            w2_low     = z_window_1 + 5
            if use_w2 and w2_low < 51:
                z_window_2       = int(rng.integers(w2_low, 51))
                z_window_2_weight = float(rng.uniform(0.1, 0.5))
            else:
                z_window_2        = 0
                z_window_2_weight = 0.0

            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(1.5, 4.5)),
                "z_window":               z_window_1,
                "volume_zscore_window":   int(rng.integers(5, 51)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(2, 100))   if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(0.5, 6.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(2, 20))   if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(1.0, 5.0)),
                "exit_on_opposite_signal": True,                                   # FIXED
                "use_breakeven_stop":     False,                                   # FIXED
                "max_hold_bars":          int(rng.choice([0, 15, 30, 60, 120, 240, 480, 720])),
                "zscore_confirmation":    False,                                   # FIXED
                # Z-score type — fixed to best known baseline
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         z_window_2,
                "z_window_2_weight":  z_window_2_weight,
                # V5 filters — ALL DISABLED
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        elif range_mode == "v10":
            # V10: ultra-diversity sweep for ML training — extends V9's numeric ranges
            # with variation on the z-score formulation axis and previously-fixed flags.
            # Goal: maximum qualitative regime coverage for downstream ML model.
            #
            # Kept from V9 (already at feasible numeric limits):
            #   - z_band_k, z_window, volume_zscore_window, ema_fast/slow
            #   - stop_fixed_pts (2–100), atr_multiplier (0.5–6.0), swing_lookback (2–20)
            #   - min_rr (1.0–5.0), max_hold_bars ∈ {0, 15, 30, 60, 120, 240, 480, 720}
            #   - Dual-window z-score blend (50% chance active)
            #
            # New diversity axes (were FIXED in V9):
            #   - z_input ∈ {close, returns, typical_price}
            #   - z_anchor ∈ {rolling_mean, vwap_session, ema_fast, ema_slow}
            #       compat: z_input="returns" forces z_anchor="rolling_mean"
            #   - z_denom ∈ {rolling_std, atr, parkinson}
            #       compat: z_input="returns" drops atr; z_type="quantile_rank" → "n/a"
            #   - z_type: 75% parametric, 25% quantile_rank
            #   - exit_on_opposite_signal / use_breakeven_stop / zscore_confirmation: sample bool
            #   - volume_entry_threshold: 50% 0.0, 50% uniform[0.5, 2.5]
            #   - session_filter_mode ∈ {0, 1, 2}
            #
            # Kept off (V5 analysis confirmed they hurt WR):
            #   - vol_regime_* filters, tod_exit_hour
            ema_fast    = int(rng.integers(3, 20))
            ema_slow    = int(rng.integers(ema_fast + 5, min(ema_fast + 45, 61)))
            stop_method = str(rng.choice(["fixed", "atr", "swing"]))

            # Dual-window z-score blend
            z_window_1 = int(rng.integers(5, 51))
            use_w2     = bool(rng.integers(0, 2))
            w2_low     = z_window_1 + 5
            if use_w2 and w2_low < 51:
                z_window_2        = int(rng.integers(w2_low, 51))
                z_window_2_weight = float(rng.uniform(0.1, 0.5))
            else:
                z_window_2        = 0
                z_window_2_weight = 0.0

            # Z-score formulation — sampled with compatibility rules mirrored from
            # zscore_variants branch so Parquet metadata matches what was computed.
            z_type  = str(rng.choice(["parametric", "parametric", "parametric", "quantile_rank"]))
            z_input = str(rng.choice(["close", "returns", "typical_price"]))
            if z_input == "returns":
                z_anchor = "rolling_mean"
            else:
                z_anchor = str(rng.choice(["rolling_mean", "vwap_session", "ema_fast", "ema_slow"]))
            if z_type == "quantile_rank":
                z_denom = "n/a"
            elif z_input == "returns":
                z_denom = str(rng.choice(["rolling_std", "parkinson"]))
            else:
                z_denom = str(rng.choice(["rolling_std", "atr", "parkinson"]))

            # Volume entry threshold: 50% off, 50% active
            if bool(rng.integers(0, 2)):
                volume_entry_threshold = float(rng.uniform(0.5, 2.5))
            else:
                volume_entry_threshold = 0.0

            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(1.5, 4.5)),
                "z_window":               z_window_1,
                "volume_zscore_window":   int(rng.integers(5, 51)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(2, 100))   if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(0.5, 6.0)) if stop_method == "atr"   else None,
                "swing_lookback":         float(rng.integers(2, 20))   if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(1.0, 5.0)),
                "exit_on_opposite_signal": bool(rng.integers(0, 2)),
                "use_breakeven_stop":     bool(rng.integers(0, 2)),
                "max_hold_bars":          int(rng.choice([0, 15, 30, 60, 120, 240, 480, 720])),
                "zscore_confirmation":    bool(rng.integers(0, 2)),
                # Z-score formulation — sampled (see compat rules above)
                "z_input":            z_input,
                "z_anchor":           z_anchor,
                "z_denom":            z_denom,
                "z_type":             z_type,
                "z_window_2":         z_window_2,
                "z_window_2_weight":  z_window_2_weight,
                # V5 entry filters — volume + session sampled; vol_regime/tod kept OFF
                "volume_entry_threshold": volume_entry_threshold,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    int(rng.choice([0, 1, 2])),
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        else:
            # Default ranges (original sweep v1)
            ema_fast    = int(rng.integers(5, 16))
            ema_slow    = int(rng.integers(ema_fast + 5, 36))
            stop_method = str(rng.choice(["fixed", "atr", "swing"]))
            combo = {
                "combo_id":                i,
                "z_band_k":               float(rng.uniform(1.5, 3.5)),
                "z_window":               int(rng.integers(10, 31)),
                "volume_zscore_window":   int(rng.integers(10, 31)),
                "ema_fast":               ema_fast,
                "ema_slow":               ema_slow,
                "stop_method":            stop_method,
                "stop_fixed_pts":         float(rng.uniform(10, 30)) if stop_method == "fixed" else None,
                "atr_multiplier":         float(rng.uniform(1.0, 3.0)) if stop_method == "atr"  else None,
                "swing_lookback":         float(rng.integers(3, 11))  if stop_method == "swing" else None,
                "min_rr":                 float(rng.uniform(2.0, 5.0)),
                "exit_on_opposite_signal": bool(rng.integers(0, 2)),
                # New mode params default to off for backward compat
                "use_breakeven_stop":     False,
                "max_hold_bars":          0,
                "zscore_confirmation":    False,
                # Z-score variant keys — baseline values for default mode
                "z_input":            "close",
                "z_anchor":           "rolling_mean",
                "z_denom":            "rolling_std",
                "z_type":             "parametric",
                "z_window_2":         0,
                "z_window_2_weight":  0.0,
                # V5 filters — off for non-v5 modes
                "volume_entry_threshold": 0.0,
                "vol_regime_lookback":    0,
                "vol_regime_min_pct":     0.0,
                "vol_regime_max_pct":     1.0,
                "session_filter_mode":    0,
                "tod_exit_hour":          0,
                "stop_fixed_pts_resolved": None,
            }
        combos.append(combo)
    return combos


# ── Config factory ────────────────────────────────────────────────────────────

def _make_cfg(combo: dict) -> types.SimpleNamespace:
    """Build a cfg-like SimpleNamespace from base config + combo overrides."""
    ns = types.SimpleNamespace()
    for k in dir(_BASE_CFG):
        if not k.startswith("_"):
            setattr(ns, k, getattr(_BASE_CFG, k))
    for k, v in _FIXED.items():
        setattr(ns, k, v)
    ns.Z_BAND_K                = combo["z_band_k"]
    ns.Z_WINDOW                = combo["z_window"]
    ns.VOLUME_ZSCORE_WINDOW    = combo["volume_zscore_window"]
    ns.EMA_FAST                = combo["ema_fast"]
    ns.EMA_SLOW                = combo["ema_slow"]
    ns.MIN_RR                  = combo["min_rr"]
    ns.EXIT_ON_OPPOSITE_SIGNAL = combo["exit_on_opposite_signal"]
    # New exit/entry mode params
    ns.USE_BREAKEVEN_STOP      = bool(combo.get("use_breakeven_stop", False))
    ns.MAX_HOLD_BARS           = int(combo.get("max_hold_bars", 0))
    ns.ZSCORE_CONFIRMATION     = bool(combo.get("zscore_confirmation", False))
    # V5+ entry filters
    ns.VOLUME_ENTRY_THRESHOLD  = float(combo.get("volume_entry_threshold", 0.0))
    ns.VOL_REGIME_LOOKBACK     = int(combo.get("vol_regime_lookback", 0))
    ns.VOL_REGIME_MIN_PCT      = float(combo.get("vol_regime_min_pct", 0.0))
    ns.VOL_REGIME_MAX_PCT      = float(combo.get("vol_regime_max_pct", 1.0))
    ns.SESSION_FILTER_MODE     = int(combo.get("session_filter_mode", 0))
    ns.TOD_EXIT_HOUR           = int(combo.get("tod_exit_hour", 0))
    # All stop methods normalised to "fixed" for the Cython core
    ns.STOP_METHOD    = "fixed"
    ns.STOP_FIXED_PTS = float(combo["stop_fixed_pts_resolved"])
    return ns


def _make_ind_cfg(ema_fast: int, ema_slow: int, z_window: int,
                  vol_window: int) -> types.SimpleNamespace:
    """Minimal cfg for add_indicators() — only the indicator params."""
    ns = types.SimpleNamespace()
    for k in dir(_BASE_CFG):
        if not k.startswith("_"):
            setattr(ns, k, getattr(_BASE_CFG, k))
    for k, v in _FIXED.items():
        setattr(ns, k, v)
    ns.EMA_FAST            = ema_fast
    ns.EMA_SLOW            = ema_slow
    ns.Z_WINDOW            = z_window
    ns.VOLUME_ZSCORE_WINDOW = vol_window
    return ns


# ── Stop resolution ────────────────────────────────────────────────────────────

def _resolve_stop_pts(combo: dict, df_ind: pd.DataFrame) -> float:
    """Convert ATR/swing stop methods to a fixed-distance approximation."""
    method = combo["stop_method"]
    if method == "fixed":
        return float(combo["stop_fixed_pts"])
    elif method == "atr":
        median_atr = float(np.nanmedian(df_ind["atr"].to_numpy()))
        return median_atr * float(combo["atr_multiplier"])
    else:  # swing
        lb = int(combo["swing_lookback"])
        roll_high = df_ind["high"].rolling(lb).max()
        roll_low  = df_ind["low"].rolling(lb).min()
        swing_dist = (roll_high - roll_low).dropna()
        median_swing = float(np.nanmedian(swing_dist.to_numpy()))
        return median_swing * 0.5 + 0.25


# ── Lightweight backtest (no equity curve) ────────────────────────────────────

_EXIT_REASON_MAP = {1: "stop", 2: "take_profit", 3: "opposite_signal", 4: "end_of_data", 5: "time_exit"}


def _run_backtest_light(df: pd.DataFrame, cfg) -> dict:
    """Run the backtest core engine and post-process trades only.

    Identical to src.backtest.run_backtest but skips the equity-curve loop,
    which iterates over all n_bars in Python (7 min on 2M bars per combo).
    Returns: {"trades": [...], "n_trades": int, "final_equity": float}
    """
    open_arr   = df["open"].to_numpy(dtype=np.float64)
    high_arr   = df["high"].to_numpy(dtype=np.float64)
    low_arr    = df["low"].to_numpy(dtype=np.float64)
    close_arr  = df["close"].to_numpy(dtype=np.float64)
    signal_arr = df["signal"].to_numpy(dtype=np.int8)
    n_bars     = len(df)

    sl_pts             = float(cfg.STOP_FIXED_PTS)
    tp_pts             = sl_pts * float(cfg.MIN_RR)
    same_bar_tp_first  = (cfg.SAME_BAR_COLLISION == "tp_first")
    exit_on_opposite   = bool(cfg.EXIT_ON_OPPOSITE_SIGNAL)
    use_breakeven_stop = bool(getattr(cfg, "USE_BREAKEVEN_STOP", False))
    max_hold_bars      = int(getattr(cfg, "MAX_HOLD_BARS", 0))
    tod_exit_hour      = int(getattr(cfg, "TOD_EXIT_HOUR", 0))

    # Hour array for TOD exit (zero array if column absent)
    if "bar_hour" in df.columns:
        hour_arr = df["bar_hour"].to_numpy(dtype=np.int64)
    else:
        hour_arr = np.zeros(n_bars, dtype=np.int64)

    # Dispatch: Cython → Numba → NumPy
    if CYTHON_AVAILABLE:
        core_fn = backtest_core_cy
    elif NUMBA_AVAILABLE:
        core_fn = _backtest_core
    else:
        core_fn = _backtest_core_numpy

    (
        raw_side, raw_signal_bar, raw_entry_bar, raw_exit_bar,
        raw_entry_price, raw_exit_price, raw_sl, raw_tp,
        raw_exit_reason, raw_mae, raw_mfe, raw_hold_bars, raw_label_tp_first,
    ) = core_fn(
        open_arr, high_arr, low_arr, close_arr,
        signal_arr, n_bars, sl_pts, tp_pts,
        same_bar_tp_first, exit_on_opposite,
        use_breakeven_stop, max_hold_bars,
        hour_arr, tod_exit_hour,
    )

    n_trades_raw = len(raw_side)

    # Pre-index columns for fast .iat access
    col_pos = {col: df.columns.get_loc(col) for col in df.columns}

    def _get(row_idx: int, col: str, default=float("nan")):
        """Safe attribute/index lookup used inside vectorised sampling paths."""
        if col in col_pos:
            return df.iat[row_idx, col_pos[col]]
        return default

    trades = []
    # Use fixed starting equity for contract sizing across ALL trades in this combo.
    # This keeps net_pnl_dollars on a consistent scale across combos (no compounding
    # to astronomical values over 50k-trade combos). r_multiple and label_win are
    # unaffected by this choice.
    fixed_equity = float(cfg.STARTING_EQUITY)

    for i in range(n_trades_raw):
        side           = int(raw_side[i])
        entry_price    = float(raw_entry_price[i])
        exit_price     = float(raw_exit_price[i])
        sl_price       = float(raw_sl[i])
        tp_price       = float(raw_tp[i])
        entry_bar_idx  = int(raw_entry_bar[i])
        signal_bar_idx = int(raw_signal_bar[i])

        stop_distance_pts   = abs(entry_price - sl_price)
        target_distance_pts = abs(tp_price - entry_price)
        rr_planned = (target_distance_pts / stop_distance_pts
                      if stop_distance_pts > 1e-12 else 0.0)

        contracts = (int(fixed_equity * cfg.RISK_PCT
                         // (stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT))
                     if stop_distance_pts > 0 else 0)

        gross_pnl    = (exit_price - entry_price) * side * contracts * cfg.MNQ_DOLLARS_PER_POINT
        net_pnl      = gross_pnl
        risk_dollars = stop_distance_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT
        r_multiple   = net_pnl / risk_dollars if risk_dollars > 0 else 0.0

        entry_time     = df.iat[entry_bar_idx, col_pos["time"]]
        sig_ema_fast   = float(_get(signal_bar_idx, "ema_fast"))
        sig_ema_slow   = float(_get(signal_bar_idx, "ema_slow"))
        sig_ema_spread = float(_get(signal_bar_idx, "ema_spread"))
        sig_zscore     = float(_get(signal_bar_idx, "zscore"))
        sig_vol_z      = float(_get(signal_bar_idx, "volume_zscore"))
        sig_atr        = float(_get(signal_bar_idx, "atr"))
        sig_high       = float(_get(signal_bar_idx, "high"))
        sig_low        = float(_get(signal_bar_idx, "low"))
        sig_open       = float(_get(signal_bar_idx, "open"))
        sig_close      = float(_get(signal_bar_idx, "close"))

        prev_idx    = max(0, signal_bar_idx - 1)
        prev_zscore = float(_get(prev_idx, "zscore"))
        zscore_delta = (sig_zscore - prev_zscore
                        if not (math.isnan(sig_zscore) or math.isnan(prev_zscore))
                        else float("nan"))

        # Parkinson single-bar volatility (no lookback, price-normalized)
        # parkinson_vol_pct: volatility as fraction of price (dimensionless)
        # parkinson_vs_atr:  ratio of bar's Parkinson vol to ATR-based baseline
        if sig_low > 0 and not math.isnan(sig_high) and not math.isnan(sig_low):
            parkinson_vol_pct = math.log(sig_high / sig_low) / _LOG_PARKINSON_SCALE
        else:
            parkinson_vol_pct = float("nan")

        if sig_atr > 0 and not math.isnan(sig_high) and not math.isnan(sig_low):
            parkinson_vs_atr = (sig_high - sig_low) * _PARKINSON_SCALE / sig_atr
        else:
            parkinson_vs_atr = float("nan")

        trades.append({
            # ML features
            "zscore_entry":                sig_zscore,
            "zscore_prev":                 prev_zscore,
            "zscore_delta":                zscore_delta,
            "volume_zscore":               sig_vol_z,
            "ema_spread":                  sig_ema_spread,
            "bar_body_points":             abs(sig_close - sig_open),
            "bar_range_points":            sig_high - sig_low,
            "atr_points":                  sig_atr,
            "parkinson_vol_pct":           parkinson_vol_pct,
            "parkinson_vs_atr":            parkinson_vs_atr,
            "time_of_day_hhmm":            entry_time.strftime("%H%M"),
            "day_of_week":                 entry_time.dayofweek,
            "distance_to_ema_fast_points": sig_close - sig_ema_fast,
            "distance_to_ema_slow_points": sig_close - sig_ema_slow,
            "side":                        "long" if side == 1 else "short",
            # Labels
            "label_win":       int(net_pnl > 0),
            "net_pnl_dollars": net_pnl,
            "r_multiple":      r_multiple,
            # Path metrics (for adaptive R:R model)
            "mfe_points":        float(raw_mfe[i]),
            "mae_points":        float(raw_mae[i]),
            "stop_distance_pts": stop_distance_pts,
            "hold_bars":         int(raw_hold_bars[i]),
            # Partition-relative entry bar index (B7 walk-forward needs
            # calendar date → map via partition df.iloc[entry_bar_idx].time).
            "entry_bar_idx":     entry_bar_idx,
        })

    total_pnl = sum(t["net_pnl_dollars"] for t in trades)
    return {
        "trades":       trades,
        "n_trades":     n_trades_raw,
        "final_equity": fixed_equity + total_pnl,
    }


# ── Row extractor ─────────────────────────────────────────────────────────────

def _extract_trade_row(trade: dict, combo: dict) -> dict:
    """Merge combo metadata + ML features + labels into a single flat row."""
    row: dict = {k: combo[k] for k in _COMBO_META_KEYS}
    row.update(trade)   # trade already contains only the ML-safe keys
    return row


# ── Module-level indicator builders (read _cache_* globals via fork COW) ─────

def _get_zscore_variant(c: dict, ema_f: np.ndarray, ema_s: np.ndarray) -> np.ndarray:
    """Return the z-score array for a variant combo.

    Reads module-level _cache_* globals. In multiprocessing mode these are
    shared via fork copy-on-write — no memory duplication.
    """
    z_input  = c.get("z_input",  "close")
    z_anchor = c.get("z_anchor", "rolling_mean")
    z_denom  = c.get("z_denom",  "rolling_std")
    z_type   = c.get("z_type",   "parametric")
    w1       = int(c["z_window"])
    w2       = int(c.get("z_window_2", 0))
    wt       = float(c.get("z_window_2_weight", 0.0))

    # Use legacy fast path for the baseline case (default/winrate combos)
    if z_input == "close" and z_anchor == "rolling_mean" \
            and z_denom == "rolling_std" and z_type == "parametric" and w2 == 0:
        return _cache_zscore[w1]

    return compute_zscore_v2(
        _cache_close_np, _cache_high_np, _cache_low_np,
        _cache_volume_np, _cache_session_brk_np,
        ema_f, ema_s, _cache_atr_arr,
        w1, z_input, z_anchor, z_denom, z_type, _cache_ddof,
        _cache_vwap_session_arr, w2, wt,
    ).astype(np.float32)


def _build_indicator_df(c: dict) -> pd.DataFrame:
    """Assemble indicator DataFrame for one combo from pre-computed cache arrays."""
    ema_f = _cache_ema_period[c["ema_fast"]]
    ema_s = _cache_ema_period[c["ema_slow"]]
    zscore_arr = _get_zscore_variant(c, ema_f, ema_s)
    vol_z_arr  = _cache_vol_zscore[c["volume_zscore_window"]]
    fast_above = ema_f > ema_s
    prev_fab   = np.concatenate([[False], fast_above[:-1]])
    lb = int(c.get("vol_regime_lookback", 0))
    atr_pct_arr = _cache_atr_pct.get(lb, np.full(len(_cache_atr_arr), np.nan))
    return pd.DataFrame({
        "time":            _cache_time_np,
        "open":            _cache_open_np,
        "high":            _cache_high_np,
        "low":             _cache_low_np,
        "close":           _cache_close_np,
        "volume":          _cache_volume_np,
        "session_break":   _cache_session_brk_np,
        "ema_fast":        ema_f,
        "ema_slow":        ema_s,
        "ema_spread":      ema_f - ema_s,
        "ema_cross_up":    fast_above & ~prev_fab,
        "ema_cross_down":  ~fast_above & prev_fab,
        "zscore":          zscore_arr,
        "atr":             _cache_atr_arr,
        "volume_zscore":   vol_z_arr,
        "atr_pct_rank":    atr_pct_arr,
        "bar_hour":        _cache_hour_np,
    })


def _process_one_combo(combo: dict) -> dict:
    """Process a single combo end-to-end. Returns serializable result dict.

    Used both in single-process mode (called directly) and multi-process mode
    (called via Pool.imap_unordered — must be picklable top-level function).
    """
    t0 = time.time()
    combo_id = combo["combo_id"]
    try:
        df_ind = _build_indicator_df(combo)
        combo["stop_fixed_pts_resolved"] = _resolve_stop_pts(combo, df_ind)
        cfg_c  = _make_cfg(combo)
        df_sig = generate_signals(df_ind, cfg_c)
        del df_ind

        results = _run_backtest_light(df_sig, cfg_c)
        del df_sig

        trades_list  = results["trades"]
        n_trades     = results["n_trades"]
        final_equity = results["final_equity"]
        del results

        trade_rows = [_extract_trade_row(t, combo) for t in trades_list]

        n_wins = sum(1 for t in trades_list if t["label_win"])
        wr = n_wins / n_trades if n_trades else 0.0

        _start_eq = float(cfg_c.STARTING_EQUITY)
        _equity = _start_eq
        _peak   = _equity
        _max_dd = 0.0
        for _t in trades_list:
            _equity += _t["net_pnl_dollars"]
            if _equity > _peak:
                _peak = _equity
            if _peak > 0:
                _dd = (_peak - _equity) / _peak
                if _dd > _max_dd:
                    _max_dd = _dd
        total_return_pct = (_equity - _start_eq) / _start_eq * 100.0

        manifest_entry = {
            **{k: combo[k] for k in _COMBO_META_KEYS},
            "stop_fixed_pts_resolved": round(combo["stop_fixed_pts_resolved"], 4),
            "n_trades":          n_trades,
            "win_rate":          round(wr, 4),
            "final_equity":      round(final_equity, 2),
            "total_return_pct":  round(total_return_pct, 2),
            "max_drawdown_pct":  round(_max_dd * 100.0, 2),
            "runtime_seconds":   round(time.time() - t0, 3),
            "status":            "completed",
        }
        return {"combo_id": combo_id, "manifest_entry": manifest_entry,
                "trade_rows": trade_rows, "error": None}

    except Exception as exc:
        import traceback
        traceback.print_exc()
        manifest_entry = {
            **{k: combo[k] for k in _COMBO_META_KEYS},
            "stop_fixed_pts_resolved": None,
            "n_trades": 0, "win_rate": None, "final_equity": None,
            "runtime_seconds": round(time.time() - t0, 3),
            "status": f"error: {exc}",
        }
        return {"combo_id": combo_id, "manifest_entry": manifest_entry,
                "trade_rows": [], "error": str(exc)}


def _worker_init():
    """Ignore SIGINT in pool workers — main process handles shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _get_chunks_dir() -> Path:
    """Return the chunks subdirectory for the current PARQUET_PATH."""
    return PARQUET_PATH.parent / (PARQUET_PATH.stem + "_chunks")


_chunk_counter = 0   # incremented per flush; survives resume via chunk file count


def _append_parquet(rows: list[dict]) -> None:
    """Write a batch of trade rows as a numbered chunk file.

    Old approach read + concatenated the full parquet every flush — OOM on
    large sweeps (v3: 14M+ rows).  Now we write independent chunk files and
    merge once at the end via _merge_chunks().
    """
    global _chunk_counter
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks_dir = _get_chunks_dir()
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Build table in 50k-row sub-chunks to avoid OOM on large batches.
    _CHUNK = 50_000
    sub_chunks = [rows[i:i + _CHUNK] for i in range(0, len(rows), _CHUNK)]
    new_table = pa.concat_tables(
        [pa.Table.from_pandas(pd.DataFrame(chunk), preserve_index=False)
         for chunk in sub_chunks],
        promote_options="default",
    )

    chunk_path = chunks_dir / f"chunk_{_chunk_counter:05d}.parquet"
    pq.write_table(new_table, chunk_path, compression="snappy")
    _chunk_counter += 1


def _merge_chunks() -> None:
    """Merge all chunk files into the final PARQUET_PATH.

    Reads chunk files one at a time via pyarrow.dataset to avoid loading
    everything into memory simultaneously.
    """
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    chunks_dir = _get_chunks_dir()
    if not chunks_dir.exists():
        return

    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        return

    print(f"[sweep] Merging {len(chunk_files)} chunk files...", flush=True)
    dataset = ds.dataset(chunk_files, format="parquet")
    # Stream rows via scanner — avoids loading all rows into memory at once
    scanner = dataset.scanner()
    with pq.ParquetWriter(PARQUET_PATH, scanner.projected_schema,
                          compression="snappy") as writer:
        for batch in scanner.to_batches():
            writer.write_batch(batch)
    print(f"[sweep] Merged to {PARQUET_PATH} "
          f"({PARQUET_PATH.stat().st_size / 1e6:.0f} MB)", flush=True)

    # Clean up chunk files
    for f in chunk_files:
        f.unlink()
    chunks_dir.rmdir()


def _save_manifest(manifest: list[dict]) -> None:
    """Persist the per-combo completion manifest alongside the parquet dataset."""
    MANIFEST_PATH.write_text(json.dumps(manifest, default=str))


def _load_manifest() -> list[dict]:
    """Load the per-combo manifest if it exists, else return an empty skeleton."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return []


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the parameter sweep runner."""
    p = argparse.ArgumentParser(description="Parameter sweep for ML dataset generation.")
    p.add_argument("--combinations", type=int, default=3000,
                   help="Total combos in the sample space (default: 3000)")
    p.add_argument("--start-combo", type=int, default=0,
                   help="First combo index to run (default: 0); enables resuming")
    p.add_argument("--end-combo", type=int, default=None,
                   help="Last combo index (exclusive). Default: --combinations value.")
    p.add_argument("--hours", type=float, default=0,
                   help="Max runtime in hours (0 = unlimited, default: 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for combo sampling (default: 0). Use 1 for second sweep.")
    p.add_argument("--range-mode", choices=["default", "winrate", "zscore_variants", "v4", "v5", "v6", "v7", "v8", "v9", "v10"], default="default",
                   help="Parameter range mode: 'default' | 'winrate' | 'zscore_variants' | 'v4'–'v10'")
    p.add_argument("--output", type=str, default=None,
                   help="Output parquet path (default: data/ml/ml_dataset.parquet)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes (default: 1). "
                        "Requires Linux (fork COW). 0 = auto (cpu_count - 1).")
    p.add_argument("--eval-partition", choices=["train", "test"], default="train",
                   help="Which chronological partition to sweep on. 'train' (default) "
                        "= first 80% of bars (standard). 'test' = last 20% (B6 "
                        "temporal-OOD eval — do not use for iteration backtests).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the parameter sweep: sample N combos, backtest each, append to parquet."""
    global PARQUET_PATH, MANIFEST_PATH, _chunk_counter

    args = _parse_args()
    t_start = time.time()

    # Workers: platform guard + auto-detect
    import platform as _plat
    if args.workers > 1 and _plat.system() == "Windows":
        print("[sweep] ERROR: --workers > 1 requires Linux (fork start method). "
              "Windows uses 'spawn' which copies 1.3 GB caches per worker. "
              "Exiting.", flush=True)
        sys.exit(1)
    if args.workers == 0:
        import os as _os
        args.workers = max(1, _os.cpu_count() - 1)
        print(f"[sweep] Auto-detected {args.workers} workers", flush=True)

    # Set output paths from CLI (or derive manifest from parquet path)
    if args.output:
        PARQUET_PATH  = Path(args.output)
        MANIFEST_PATH = PARQUET_PATH.with_suffix("").with_name(
            PARQUET_PATH.stem + "_manifest"
        ).with_suffix(".json")
    else:
        PARQUET_PATH  = _DEFAULT_PARQUET
        MANIFEST_PATH = _DEFAULT_MANIFEST

    # Resume: set chunk counter past existing chunks so we don't overwrite
    chunks_dir = _get_chunks_dir()
    if chunks_dir.exists():
        existing_chunks = list(chunks_dir.glob("chunk_*.parquet"))
        _chunk_counter = len(existing_chunks)
        if _chunk_counter:
            print(f"[sweep] Found {_chunk_counter} existing chunk files "
                  f"(resuming)", flush=True)

    engine = "Cython" if CYTHON_AVAILABLE else ("Numba" if NUMBA_AVAILABLE else "NumPy")
    print(f"[sweep] engine={engine}  combinations={args.combinations}  "
          f"start_combo={args.start_combo}  hours={'unlimited' if args.hours == 0 else args.hours}  "
          f"seed={args.seed}  range_mode={args.range_mode}",
          flush=True)

    # 1. Load chosen partition once
    print("[sweep] Loading data...", flush=True)
    df_raw           = load_bars("data/NQ_1min.csv")
    train_part, test_part = split_train_test(df_raw, 0.8)
    if args.eval_partition == "test":
        train_df = test_part  # reuse the same local name downstream; semantics: "the bars we backtest on"
        print(f"[sweep] eval_partition=test  Test bars: {len(train_df):,} "
              f"(B6 temporal-OOD eval)", flush=True)
    else:
        train_df = train_part
        print(f"[sweep] eval_partition=train  Train bars: {len(train_df):,}", flush=True)

    # 2. Sample all combos upfront (deterministic via seed)
    all_combos = _sample_combos(args.combinations, rng_seed=args.seed,
                                range_mode=args.range_mode)
    end_combo = args.end_combo if args.end_combo is not None else args.combinations
    run_combos = [c for c in all_combos
                  if args.start_combo <= c["combo_id"] < end_combo]
    print(f"[sweep] Combos to run: {len(run_combos)}  "
          f"(#{args.start_combo} to #{args.combinations - 1})", flush=True)

    # 3. Pre-compute all unique indicators up front.
    #    With nearly-all-unique parameter combos, computing indicators once per
    #    combo from scratch (via add_indicators) costs ~5s each = hours for 3000 combos.
    #    Instead: compute ATR once, zscore/vol_zscore per unique window, EMA per unique pair.
    #    Cost: O(unique_values) instead of O(n_combos).
    print("[sweep] Pre-computing indicator caches...", flush=True)
    t_cache = time.time()
    use_numba = True
    ddof      = 0

    close_np       = train_df["close"].to_numpy(dtype=np.float64)
    high_np        = train_df["high"].to_numpy(dtype=np.float64)
    low_np         = train_df["low"].to_numpy(dtype=np.float64)
    open_np        = train_df["open"].to_numpy(dtype=np.float64)   # pre-extract for del train_df
    volume_np      = train_df["volume"].to_numpy(dtype=np.float64)
    session_brk_np = train_df["session_break"].to_numpy()
    time_np        = train_df["time"].to_numpy()                   # pre-extract for del train_df

    # ATR: fixed window (ATR_WINDOW=14 always) — compute once; kept float64 (stop distance precision)
    atr_arr   = compute_atr(high_np, low_np, close_np, _BASE_CFG.ATR_WINDOW, use_numba)

    # Session VWAP: computed once (resets on session_break), used by vwap_session anchor
    vwap_session_arr = compute_vwap_session(high_np, low_np, close_np, volume_np, session_brk_np)

    # zscore: unique z_window values — float32 (signal use only; halves per-array memory)
    unique_z_windows  = sorted(set(c["z_window"] for c in all_combos))
    zscore_cache      = {w: compute_zscore(close_np, w, ddof, use_numba).astype(np.float32)
                         for w in unique_z_windows}

    # volume_zscore: unique vol_window values — float32
    unique_vol_windows = sorted(set(c["volume_zscore_window"] for c in all_combos))
    vol_zscore_cache   = {w: compute_volume_zscore(volume_np, w, ddof, use_numba).astype(np.float32)
                          for w in unique_vol_windows}

    # EMA: cache by individual period, not by (fast, slow) pair.
    # With 286 unique pairs but only ~36 unique periods, the pair cache stored
    # 572 redundant float64 arrays (~9 GB). Per-period cache stores ~36 arrays (~600 MB).
    unique_ema_periods = sorted(set(
        p for c in all_combos for p in [c["ema_fast"], c["ema_slow"]]
    ))
    ema_period_cache: dict[int, np.ndarray] = {
        p: compute_ema(close_np, p, use_numba).astype(np.float32)  # float32; signal use only
        for p in unique_ema_periods
    }

    # ATR percentile rank: unique non-zero vol_regime_lookback values — computed once
    # (ATR uses fixed 14-bar window, independent of any combo param)
    unique_vol_regime_lookbacks = sorted(set(
        int(c.get("vol_regime_lookback", 0)) for c in all_combos
        if int(c.get("vol_regime_lookback", 0)) > 0
    ))
    atr_pct_cache: dict[int, np.ndarray] = {}
    for lb in unique_vol_regime_lookbacks:
        atr_pct_cache[lb] = pd.Series(atr_arr).rolling(lb).rank(pct=True).to_numpy()

    # Bar hour array: constant for all combos
    hour_np = pd.DatetimeIndex(time_np).hour.to_numpy(dtype=np.int64)

    print(f"[sweep] Cache built in {time.time()-t_cache:.1f}s — "
          f"z_windows={len(unique_z_windows)}  vol_windows={len(unique_vol_windows)}  "
          f"ema_periods={len(unique_ema_periods)}", flush=True)

    # Free the raw DataFrame — all columns are now in numpy arrays above
    del train_df

    # Publish caches to module-level globals so _build_indicator_df /
    # _get_zscore_variant / _process_one_combo can access them — both in
    # single-process mode and via fork COW in multi-process mode.
    global _cache_close_np, _cache_high_np, _cache_low_np, _cache_open_np
    global _cache_volume_np, _cache_session_brk_np, _cache_time_np, _cache_hour_np
    global _cache_atr_arr, _cache_vwap_session_arr
    global _cache_zscore, _cache_vol_zscore, _cache_ema_period, _cache_atr_pct
    global _cache_ddof

    _cache_close_np        = close_np
    _cache_high_np         = high_np
    _cache_low_np          = low_np
    _cache_open_np         = open_np
    _cache_volume_np       = volume_np
    _cache_session_brk_np  = session_brk_np
    _cache_time_np         = time_np
    _cache_hour_np         = hour_np
    _cache_atr_arr         = atr_arr
    _cache_vwap_session_arr = vwap_session_arr
    _cache_zscore          = zscore_cache
    _cache_vol_zscore      = vol_zscore_cache
    _cache_ema_period      = ema_period_cache
    _cache_atr_pct         = atr_pct_cache
    _cache_ddof            = ddof

    # 4. SIGINT handler
    stop_flag = [False]
    def _on_sigint(sig, frame):
        """SIGINT handler — flush pending rows to parquet before exiting cleanly."""
        stop_flag[0] = True
        print("\n[sweep] Ctrl+C — will flush and exit after current combo.", flush=True)
    signal.signal(signal.SIGINT, _on_sigint)

    # 5. Load existing manifest (supports resuming mid-sweep)
    manifest       = _load_manifest()
    done_ids       = {e["combo_id"] for e in manifest if e.get("status") == "completed"}
    if done_ids:
        print(f"[sweep] Skipping {len(done_ids)} already-completed combo IDs.", flush=True)
    pending_trades: list[dict] = []
    n_done         = 0
    todo_combos    = [c for c in run_combos if c["combo_id"] not in done_ids]
    n_total        = len(todo_combos)

    # 6. Main loop — single-process or multi-process
    if args.workers <= 1:
        # ── Single-process path (original behavior) ──
        for c in todo_combos:
            # Stop checks
            if stop_flag[0]:
                break
            if args.hours > 0 and (time.time() - t_start) / 3600 >= args.hours:
                stop_flag[0] = True
                break
            if STOP_FILE.exists():
                print("[sweep] stop_sweep.txt detected — stopping.", flush=True)
                stop_flag[0] = True
                break

            result = _process_one_combo(c)
            manifest.append(result["manifest_entry"])
            pending_trades.extend(result["trade_rows"])
            n_done += 1

            if n_done % 50 == 0:
                gc.collect()
            if n_done % BATCH_SIZE == 0:
                if pending_trades:
                    _append_parquet(pending_trades)
                    pending_trades.clear()
                _save_manifest(manifest)
            if n_done % LOG_INTERVAL == 0:
                elapsed_m = (time.time() - t_start) / 60
                pct = n_done / n_total * 100
                print(f"[sweep] {n_done}/{n_total} ({pct:.0f}%) | "
                      f"{elapsed_m:.1f}m | last combo #{result['combo_id']}",
                      flush=True)

    else:
        # ── Multi-process path (Linux only, fork COW) ──
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)

        print(f"[sweep] Starting pool with {args.workers} workers for "
              f"{n_total} combos", flush=True)

        pool = mp.Pool(processes=args.workers, initializer=_worker_init)
        try:
            # chunksize=4: each combo ~0.5-2s, so 4 combos per IPC round-trip
            # balances overhead vs load-balancing for variable-cost combos.
            result_iter = pool.imap_unordered(
                _process_one_combo, todo_combos, chunksize=4,
            )
            for result in result_iter:
                manifest.append(result["manifest_entry"])
                pending_trades.extend(result["trade_rows"])
                n_done += 1

                if result["error"]:
                    print(f"[sweep] combo {result['combo_id']} error: "
                          f"{result['error']}", flush=True)

                if n_done % BATCH_SIZE == 0:
                    if pending_trades:
                        _append_parquet(pending_trades)
                        pending_trades.clear()
                    _save_manifest(manifest)

                if n_done % LOG_INTERVAL == 0:
                    elapsed_m = (time.time() - t_start) / 60
                    pct = n_done / n_total * 100
                    print(f"[sweep] {n_done}/{n_total} ({pct:.0f}%) | "
                          f"{elapsed_m:.1f}m", flush=True)

                # Stop checks (checked in main process between results)
                if stop_flag[0]:
                    break
                if args.hours > 0 and (time.time() - t_start) / 3600 >= args.hours:
                    stop_flag[0] = True
                    break
                if STOP_FILE.exists():
                    print("[sweep] stop_sweep.txt detected — stopping.",
                          flush=True)
                    stop_flag[0] = True
                    break

                # Periodic GC in the main process
                if n_done % 50 == 0:
                    gc.collect()

        except KeyboardInterrupt:
            print("\n[sweep] Ctrl+C — terminating workers...", flush=True)
            stop_flag[0] = True
        finally:
            pool.terminate()
            pool.join()

    # 7. Final flush
    if pending_trades:
        _append_parquet(pending_trades)
    _save_manifest(manifest)

    elapsed_m  = (time.time() - t_start) / 60
    total_rows = sum(e.get("n_trades", 0) for e in manifest
                     if e.get("status") == "completed")

    # 8. Merge chunks into final parquet (only on full completion, not stop)
    if not stop_flag[0]:
        _merge_chunks()
        print(f"[sweep] Done — {n_done} combos in {elapsed_m:.1f}m | "
              f"~{total_rows:,} trade rows | {PARQUET_PATH}", flush=True)
    else:
        print(f"[sweep] Stopped — {n_done} combos in {elapsed_m:.1f}m | "
              f"chunks saved in {_get_chunks_dir()}", flush=True)


if __name__ == "__main__":
    import traceback as _tb
    # Mirror all output to a log file named after the output parquet (prevents
    # successive sweeps from clobbering each other's logs).
    # e.g. --output data/ml_dataset_v7.parquet → sweep_run_v7.log
    _log_name = "sweep_run.log"
    for _j in range(1, len(sys.argv) - 1):
        if sys.argv[_j] == "--output":
            _stem = Path(sys.argv[_j + 1]).stem       # e.g. "ml_dataset_v7"
            _suffix = _stem.replace("ml_dataset", "")  # "_v7" or ""
            _log_name = f"sweep_run{_suffix}.log"
            break
    _log = open(_log_name, "w", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr

    class _Tee:
        """Minimal tee file-like that mirrors writes to stdout and a log file."""

        def __init__(self, *streams):
            """Store the destination streams."""
            self._streams = streams

        def write(self, data):
            """Forward `data` to all underlying streams and flush each."""
            for s in self._streams:
                s.write(data)
                s.flush()

        def flush(self):
            """Flush all underlying streams."""
            for s in self._streams:
                s.flush()

        def fileno(self):
            """Delegate `fileno` to the primary (first) stream."""
            return self._streams[0].fileno()

    sys.stdout = _Tee(_orig_stdout, _log)
    sys.stderr = _Tee(_orig_stderr, _log)

    _exit_code = 0
    try:
        main()
    except Exception as _e:
        print(f"[sweep] FATAL: {_e}", flush=True)
        _tb.print_exc()
        _exit_code = 1
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log.close()
    sys.exit(_exit_code)
