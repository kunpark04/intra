"""Central configuration constants for the backtest, strategy, and risk model.

All tunable parameters live here so the bar-by-bar engine, signal generator,
Monte Carlo simulator, and reporting layer read from a single source of truth.
Values are loaded at import time; mutations at runtime are not supported.
Grouped by concern (indicators, signal logic, execution, risk, exits, scoring).
"""

# ── Indicators ────────────────────────────────────────────────────────────────
EMA_FAST = 10                # fast EMA span (bars)  — V3: 10/20
EMA_SLOW = 20                # slow EMA span (bars)

Z_WINDOW = 20                # rolling Z-score window
Z_BAND_K = 2.5               # Z-score band threshold (optional filter)
ZSCORE_DDOF = 0              # ddof for rolling std; fixed for V1
VOLUME_ZSCORE_WINDOW = 20    # rolling window for volume Z-score

ATR_WINDOW = 14              # ATR window (used for reporting/context; not SL sizing)

# ── Signal logic ──────────────────────────────────────────────────────────────
# Primary signal: EMA crossover (8/21)
#   fast crosses above slow  →  long signal
#   fast crosses below slow  →  short signal
#
# Optional Z-score filter (USE_ZSCORE_FILTER=True):
#   long only when zscore <= -Z_BAND_K  (buying stretched-low)
#   short only when zscore >= +Z_BAND_K (selling stretched-high)
# Signal mode
#   "ema_crossover"   — V1/V2: EMA crossover is primary signal; optional z-score gate
#   "zscore_reversal" — V3+:  z-score crossing back through band is primary;
#                              EMA direction (fast vs slow) is confirming filter
SIGNAL_MODE = "zscore_reversal"
USE_ZSCORE_FILTER = True     # used only when SIGNAL_MODE="ema_crossover"

# ── Execution ─────────────────────────────────────────────────────────────────
FILL_MODEL = "next_bar_open"
SAME_BAR_COLLISION = "tp_first"   # if SL+TP both hit same bar: assume TP first

# ── Risk / sizing ─────────────────────────────────────────────────────────────
STARTING_EQUITY = 50000.0
RISK_PCT = 0.05                   # risk 5% of equity per trade
# When set to a positive float, overrides RISK_PCT and sizes every trade to
# risk this many dollars regardless of running equity (decouples sizing from
# compounding — useful for signal-quality comparisons across combos).
FIXED_RISK_DOLLARS = None
MNQ_DOLLARS_PER_POINT = 2.0       # MNQ: $2 per full index point per contract
TICK_SIZE = 0.25                  # minimum price increment (points)
MIN_RR = 3.0                      # minimum reward-to-risk ratio (V2: 1:3)

# ── Stop-loss ─────────────────────────────────────────────────────────────────
STOP_METHOD = "fixed"             # "fixed" | "atr" | "swing"
STOP_FIXED_PTS = 20.0             # fixed SL distance in points (V2: tighter)
ATR_MULTIPLIER = 1.5              # used when STOP_METHOD="atr"
SWING_LOOKBACK = 5                # bars for swing high/low (STOP_METHOD="swing")
SWING_BUFFER_TICKS = 1            # ticks beyond swing extreme

# ── Exit logic ────────────────────────────────────────────────────────────────
# Exit triggered by whichever comes first:
#   1. SL hit (low <= sl for long, high >= sl for short)
#   2. TP hit (high >= tp for long, low <= tp for short)
#   3. Opposite EMA crossover signal
#   4. Time exit (if MAX_HOLD_BARS > 0)
EXIT_ON_OPPOSITE_SIGNAL = True    # close position on opposite crossover
USE_BREAKEVEN_STOP      = False   # once +1R profit, move SL to entry price
MAX_HOLD_BARS           = 0       # exit at next open after N bars (0 = disabled)

# ── Entry confirmation ────────────────────────────────────────────────────────
ZSCORE_CONFIRMATION = False       # require |z| already declining before entry

# ── New entry filters (V5+) ───────────────────────────────────────────────────
VOLUME_ENTRY_THRESHOLD = 0.0     # min volume_zscore to enter (0.0 = disabled)
VOL_REGIME_LOOKBACK    = 0       # rolling window for ATR pct rank (0 = disabled)
VOL_REGIME_MIN_PCT     = 0.0     # min ATR percentile to allow entry (0.0 = no gate)
VOL_REGIME_MAX_PCT     = 1.0     # max ATR percentile to allow entry (1.0 = no gate)
SESSION_FILTER_MODE    = 0       # 0=all, 1=daytime(7-20h), 2=core_us(9-16h), 3=overnight(20-7h)

# ── Time-of-day exit (V5+) ────────────────────────────────────────────────────
TOD_EXIT_HOUR          = 0       # force close open position at this hour (0 = disabled)

# ── Train / test ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
# Frozen training cutoff: bars at or before this timestamp are training; bars
# after it are test. Locks the partition boundary even as new bars are appended
# to data/NQ_1min.csv via scripts/data_pipeline/update_bars_yfinance.py, so the
# V3 production stack keeps being evaluated on truly held-out future data.
# Value matches the 80% chronological boundary at the time V3 was trained.
TRAIN_END_FROZEN = "2024-10-22 05:07:00"

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_SIMS = 10_000
MC_SEED = 42
MC_BOOTSTRAP = "iid"             # "iid" | "block"
MC_RUIN_THRESHOLD = 0.5          # ruin = equity drops >= 50% from start

# ── Performance ───────────────────────────────────────────────────────────────
USE_NUMBA = True                  # set False if numba unavailable

# ── Track A: Entry quality scoring ────────────────────────────────────────────
# Weights for each component of the entry_score (sum need not equal 1;
# the scorer normalises by the sum of non-NaN weights automatically).
SCORE_W_ZSCORE  = 0.25   # abs(zscore_entry) / Z_BAND_K — price stretch
SCORE_W_VOLUME  = 0.20   # volume_zscore — participation confirmation
SCORE_W_EMA     = 0.20   # abs(ema_spread) / SCORE_EMA_NORM — trend strength
SCORE_W_BODY    = 0.20   # bar_body / bar_range — directional conviction
SCORE_W_SESSION = 0.15   # time-of-day quality (RTH > extended > overnight)
SCORE_EMA_NORM  = 5.0    # EMA spread normalisation constant (index points)
