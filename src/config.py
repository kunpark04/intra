# ── Indicators ────────────────────────────────────────────────────────────────
EMA_FAST = 8                 # fast EMA span (bars)
EMA_SLOW = 21                # slow EMA span (bars)

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
USE_ZSCORE_FILTER = False    # baseline: trade every crossover

# ── Execution ─────────────────────────────────────────────────────────────────
FILL_MODEL = "next_bar_open"
SAME_BAR_COLLISION = "tp_first"   # if SL+TP both hit same bar: assume TP first

# ── Risk / sizing ─────────────────────────────────────────────────────────────
STARTING_EQUITY = 2000.0
RISK_PCT = 0.05                   # risk 5% of equity per trade
MNQ_DOLLARS_PER_POINT = 2.0       # MNQ: $2 per full index point per contract
TICK_SIZE = 0.25                  # minimum price increment (points)
MIN_RR = 2.0                      # minimum reward-to-risk ratio

# ── Stop-loss ─────────────────────────────────────────────────────────────────
STOP_METHOD = "fixed"             # "fixed" | "atr" | "swing"
STOP_FIXED_PTS = 30.0             # fixed SL distance in points (V1 baseline)
ATR_MULTIPLIER = 1.5              # used when STOP_METHOD="atr"
SWING_LOOKBACK = 5                # bars for swing high/low (STOP_METHOD="swing")
SWING_BUFFER_TICKS = 1            # ticks beyond swing extreme

# ── Exit logic ────────────────────────────────────────────────────────────────
# Exit triggered by whichever comes first:
#   1. SL hit (low <= sl for long, high >= sl for short)
#   2. TP hit (high >= tp for long, low <= tp for short)
#   3. Opposite EMA crossover signal
EXIT_ON_OPPOSITE_SIGNAL = True    # close position on opposite crossover

# ── Train / test ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_SIMS = 10_000
MC_SEED = 42
MC_BOOTSTRAP = "iid"             # "iid" | "block"
MC_RUIN_THRESHOLD = 0.5          # ruin = equity drops >= 50% from start

# ── Performance ───────────────────────────────────────────────────────────────
USE_NUMBA = True                  # set False if numba unavailable
