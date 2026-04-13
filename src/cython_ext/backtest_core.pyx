# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
backtest_core.pyx — AOT Cython bar-by-bar backtest state machine.

Identical logic to _backtest_core (Numba) and _backtest_core_numpy in
backtest.py. Returns the same 13-tuple of NumPy arrays.

Build:
    python setup_cython.py build_ext --inplace
"""
import numpy as np
cimport numpy as cnp

cnp.import_array()


def backtest_core_cy(
    const double[::1] open_arr,
    const double[::1] high_arr,
    const double[::1] low_arr,
    const double[::1] close_arr,
    const cnp.int8_t[::1] signal_arr,
    int n_bars,
    double sl_pts,
    double tp_pts,
    bint same_bar_tp_first,
    bint exit_on_opposite,
    bint use_breakeven_stop,
    int max_hold_bars,
    const cnp.int64_t[::1] hour_arr,
    int tod_exit_hour,
):
    """AOT-compiled bar-by-bar state machine.

    Parameters
    ----------
    open_arr, high_arr, low_arr, close_arr : price arrays (float64)
    signal_arr  : int8 array — 1=long, -1=short, 0=flat
    n_bars      : number of bars
    sl_pts      : stop-loss distance in points (positive scalar)
    tp_pts      : take-profit distance in points (positive scalar)
    same_bar_tp_first : if SL+TP both touched same bar, assume TP first
    exit_on_opposite  : exit on opposite signal
    use_breakeven_stop: once +1R profit, move SL to entry price
    max_hold_bars     : exit at next open after this many bars (0 = disabled)
    hour_arr          : bar hour-of-day (0–23); used for TOD exit
    tod_exit_hour     : force close at this hour (0 = disabled)

    Exit reason codes
    -----------------
    1 = stop_loss
    2 = take_profit
    3 = opposite_signal
    4 = end_of_data
    5 = time_exit

    Returns
    -------
    13-tuple of sliced NumPy arrays (n_trades rows each).
    """
    cdef int max_trades = n_bars // 2 + 1

    # ── Preallocate output arrays ─────────────────────────────────────────────
    cdef cnp.ndarray[cnp.int8_t,   ndim=1] out_side          = np.empty(max_trades, dtype=np.int8)
    cdef cnp.ndarray[cnp.int64_t,  ndim=1] out_signal_bar    = np.empty(max_trades, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t,  ndim=1] out_entry_bar     = np.empty(max_trades, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t,  ndim=1] out_exit_bar      = np.empty(max_trades, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_entry_price   = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_exit_price    = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_sl            = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_tp            = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.int8_t,   ndim=1] out_exit_reason   = np.empty(max_trades, dtype=np.int8)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_mae           = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] out_mfe           = np.empty(max_trades, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t,  ndim=1] out_hold_bars     = np.empty(max_trades, dtype=np.int64)
    cdef cnp.ndarray[cnp.int8_t,   ndim=1] out_label_tp_first= np.empty(max_trades, dtype=np.int8)

    # ── State variables ───────────────────────────────────────────────────────
    cdef bint   in_trade           = False
    cdef int    side               = 0
    cdef int    entry_bar_i        = -1
    cdef int    signal_bar_i       = -1
    cdef double entry_price        = 0.0
    cdef double sl_price           = 0.0
    cdef double effective_sl       = 0.0   # may equal entry_price after breakeven
    cdef double tp_price           = 0.0
    cdef double mae                = 0.0
    cdef double mfe                = 0.0
    cdef bint   breakeven_activated= False
    cdef int    bars_held          = 0
    cdef int    n_trades           = 0

    # ── Per-bar temporaries ───────────────────────────────────────────────────
    cdef int    t, sig
    cdef double bar_adv, bar_fav
    cdef bint   sl_hit, tp_hit
    cdef int    exit_reason
    cdef double exit_price_val
    cdef int    label_tp_first

    for t in range(n_bars):
        # ── Entry logic ───────────────────────────────────────────────────────
        if not in_trade:
            sig = signal_arr[t]
            if sig != 0 and t + 1 < n_bars:
                in_trade           = True
                side               = sig
                signal_bar_i       = t
                entry_bar_i        = t + 1
                entry_price        = open_arr[t + 1]
                sl_price           = entry_price - sl_pts * side
                effective_sl       = sl_price
                tp_price           = entry_price + tp_pts * side
                mae                = 0.0
                mfe                = 0.0
                breakeven_activated= False
                bars_held          = 0

        # ── In-trade bar processing ───────────────────────────────────────────
        if in_trade and t >= entry_bar_i:
            bars_held += 1

            if side == 1:
                bar_adv = low_arr[t]  - entry_price   # negative = adverse
                bar_fav = high_arr[t] - entry_price   # positive = favorable
            else:
                bar_adv = entry_price - high_arr[t]
                bar_fav = entry_price - low_arr[t]

            if bar_adv < mae:
                mae = bar_adv
            if bar_fav > mfe:
                mfe = bar_fav

            # Breakeven: once +1R profit, move effective SL to entry
            if use_breakeven_stop and not breakeven_activated:
                if bar_fav >= sl_pts:
                    effective_sl       = entry_price
                    breakeven_activated= True

            # SL / TP hit detection (SL uses effective_sl post-breakeven)
            if side == 1:
                sl_hit = low_arr[t]  <= effective_sl
                tp_hit = high_arr[t] >= tp_price
            else:
                sl_hit = high_arr[t] >= effective_sl
                tp_hit = low_arr[t]  <= tp_price

            exit_reason    = 0
            exit_price_val = 0.0
            label_tp_first = 0

            if sl_hit and tp_hit:
                if same_bar_tp_first:
                    exit_reason    = 2
                    exit_price_val = tp_price
                    label_tp_first = 1
                else:
                    exit_reason    = 1
                    exit_price_val = effective_sl
            elif tp_hit:
                exit_reason    = 2
                exit_price_val = tp_price
                label_tp_first = 1
            elif sl_hit:
                exit_reason    = 1
                exit_price_val = effective_sl
            elif exit_on_opposite and signal_arr[t] == -side and t + 1 < n_bars:
                exit_reason    = 3
                exit_price_val = open_arr[t + 1]
            elif max_hold_bars > 0 and bars_held >= max_hold_bars and t + 1 < n_bars:
                exit_reason    = 5
                exit_price_val = open_arr[t + 1]
            elif tod_exit_hour > 0 and hour_arr[t] == tod_exit_hour and t + 1 < n_bars:
                exit_reason    = 5
                exit_price_val = open_arr[t + 1]

            # End of data: close at last bar's close
            if exit_reason == 0 and t == n_bars - 1:
                exit_reason    = 4
                exit_price_val = close_arr[t]

            if exit_reason != 0:
                out_side[n_trades]           = side
                out_signal_bar[n_trades]     = signal_bar_i
                out_entry_bar[n_trades]      = entry_bar_i
                out_exit_bar[n_trades]       = t
                out_entry_price[n_trades]    = entry_price
                out_exit_price[n_trades]     = exit_price_val
                out_sl[n_trades]             = sl_price   # original (pre-breakeven) for sizing
                out_tp[n_trades]             = tp_price
                out_exit_reason[n_trades]    = exit_reason
                out_mae[n_trades]            = mae
                out_mfe[n_trades]            = mfe
                out_hold_bars[n_trades]      = t - entry_bar_i + 1
                out_label_tp_first[n_trades] = label_tp_first
                n_trades += 1
                in_trade = False
                side     = 0

    return (
        out_side[:n_trades],
        out_signal_bar[:n_trades],
        out_entry_bar[:n_trades],
        out_exit_bar[:n_trades],
        out_entry_price[:n_trades],
        out_exit_price[:n_trades],
        out_sl[:n_trades],
        out_tp[:n_trades],
        out_exit_reason[:n_trades],
        out_mae[:n_trades],
        out_mfe[:n_trades],
        out_hold_bars[:n_trades],
        out_label_tp_first[:n_trades],
    )
