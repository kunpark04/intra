# --- Indicators ---
Z_WINDOW = 20
Z_BAND_K = 2.5
EMA_FAST = 3
EMA_SLOW = 6
ZSCORE_DDOF = 0          # ddof=0 for rolling std; fixed for V1
VOLUME_ZSCORE_WINDOW = 20

# --- Execution ---
FILL_MODEL = "next_bar_open"
SAME_BAR_COLLISION = "tp_first"  # if both SL+TP hit same bar: assume TP hit first

# --- Risk ---
STARTING_EQUITY = 2000.0
RISK_PCT = 0.05
MNQ_DOLLARS_PER_POINT = 2.0
TICK_SIZE = 0.25
MIN_RR = 2.0

# --- Stop placement ---
STOP_METHOD = "swing"        # "swing" or "atr"
SWING_LOOKBACK = 5           # bars to look back for swing high/low
SWING_BUFFER_TICKS = 1       # ticks beyond swing: long stop = swing_low - 1*TICK_SIZE
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.5

# --- Train/test ---
TRAIN_RATIO = 0.8

# --- Monte Carlo ---
MC_N_SIMS = 10_000
MC_SEED = 42
MC_BOOTSTRAP = "iid"         # "iid" or "block"
MC_RUIN_THRESHOLD = 0.5      # ruin = equity drops to <= (1 - threshold) * starting_equity

# --- Performance ---
USE_NUMBA = True             # set False if numba not available
