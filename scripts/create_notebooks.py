"""
Script to create notebooks/01_backtest_and_log.ipynb and
notebooks/02_plotly_exploration.ipynb using nbformat v4.
Run from repo root: python scripts/create_notebooks.py
"""
import nbformat

# ---------------------------------------------------------------------------
# Notebook 1: 01_backtest_and_log.ipynb
# ---------------------------------------------------------------------------

cell1 = nbformat.v4.new_code_cell("""\
import sys
sys.path.insert(0, ".")

import pandas as pd
import src.config as cfg
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.strategy import generate_signals
from src.backtest import run_backtest
from src.reporting import save_iteration, run_monte_carlo
import json

VERSION = "V1"
DATA_PATH = "data/NQ_1min.csv"
""")

cell2 = nbformat.v4.new_code_cell("""\
df = load_bars(DATA_PATH)
train, test = split_train_test(df, cfg.TRAIN_RATIO)
print(f"Total: {len(df):,}  Train: {len(train):,}  Test: {len(test):,}")
""")

cell3 = nbformat.v4.new_code_cell("""\
train = add_indicators(train, cfg)
print("Indicator columns:", [c for c in train.columns if c not in ['time','open','high','low','close','volume','session_break']])
train.head(3)
""")

cell4 = nbformat.v4.new_code_cell("""\
train = generate_signals(train, cfg)
n_long = (train.signal == 1).sum()
n_short = (train.signal == -1).sum()
print(f"Signals — long: {n_long:,}  short: {n_short:,}")
""")

cell5 = nbformat.v4.new_code_cell("""\
results = run_backtest(train, cfg, version=VERSION)
trades = results["trades"]
n = results["n_trades"]
wins = sum(1 for t in trades if t["label_win"])
print(f"Trades: {n:,}  Wins: {wins:,}  Win rate: {wins/n*100:.1f}%  Final equity: ${results['final_equity']:,.2f}")
""")

cell6 = nbformat.v4.new_code_cell("""\
out_dir = save_iteration(VERSION, results, train, cfg)
print(f"Artifacts saved to: {out_dir}")
""")

cell7 = nbformat.v4.new_code_cell("""\
mc_path = out_dir / "monte_carlo.json"
with open(mc_path) as f:
    mc = json.load(f)
print(f"Monte Carlo ({mc['n_sims']:,} sims, seed={mc['seed']})")
print(f"  Max Drawdown p50: ${mc['max_drawdown']['p50']:.2f}  p90: ${mc['max_drawdown']['p90']:.2f}  p99: ${mc['max_drawdown']['p99']:.2f}")
print(f"  VaR (5th pct): ${mc['var_trade_pnl']:.2f}  CVaR: ${mc['cvar_trade_pnl']:.2f}")
print(f"  Risk of ruin: {mc['risk_of_ruin_prob']*100:.1f}%")
""")

cell8 = nbformat.v4.new_code_cell("""\
tlog = pd.read_csv(out_dir / "trader_log.csv")

def color_rows(row):
    if row["net_pnl_dollars"] > 0:
        return ["background-color: #d4edda"] * len(row)
    elif row["net_pnl_dollars"] < 0:
        return ["background-color: #f8d7da"] * len(row)
    else:
        return [""] * len(row)

styled = tlog.head(50).style.apply(color_rows, axis=1)
styled
""")

cell9 = nbformat.v4.new_code_cell("""\
html_path = out_dir / "trader_log.html"
tlog.head(200).style.apply(color_rows, axis=1).to_html(html_path)
print(f"Styled HTML saved: {html_path}")
""")

nb1 = nbformat.v4.new_notebook()
nb1.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb1.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}
nb1.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

with open("notebooks/01_backtest_and_log.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb1, f)

print("Written: notebooks/01_backtest_and_log.ipynb")

# ---------------------------------------------------------------------------
# Notebook 2: 02_plotly_exploration.ipynb
# ---------------------------------------------------------------------------

nb2_cell1 = nbformat.v4.new_code_cell("""\
import sys
sys.path.insert(0, ".")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.config as cfg
from src.io_paths import iteration_dir

VERSION = "V1"
out_dir = iteration_dir(VERSION)
""")

nb2_cell2 = nbformat.v4.new_code_cell("""\
# Load trades and equity curve from iteration artifacts
trades_df = pd.read_csv(out_dir / "trades.csv", parse_dates=["entry_time","exit_time"])

# Load a sample of raw bars with indicators for the chart
# We'll use the first 5000 bars of training data for a manageable chart
import src.config as cfg
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.strategy import generate_signals

df_raw = load_bars("data/NQ_1min.csv")
train, _ = split_train_test(df_raw, cfg.TRAIN_RATIO)
train = add_indicators(train, cfg)
train = generate_signals(train, cfg)

# Chart the first 5000 bars
CHART_BARS = 5000
chart_df = train.head(CHART_BARS).reset_index(drop=True)

# Filter trades that fall within the chart window
chart_start = chart_df["time"].min()
chart_end = chart_df["time"].max()
chart_trades = trades_df[(trades_df["entry_time"] >= chart_start) & (trades_df["entry_time"] <= chart_end)]
print(f"Chart window: {chart_start} to {chart_end}  |  Trades in window: {len(chart_trades)}")
""")

nb2_cell3 = nbformat.v4.new_code_cell("""\
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    subplot_titles=["Price + EMA 8/21", "Z-score"],
    vertical_spacing=0.05,
)

# --- Row 1: Candlestick ---
fig.add_trace(go.Candlestick(
    x=chart_df["time"],
    open=chart_df["open"],
    high=chart_df["high"],
    low=chart_df["low"],
    close=chart_df["close"],
    name="Price",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# EMA fast
fig.add_trace(go.Scatter(
    x=chart_df["time"], y=chart_df["ema_fast"],
    name=f"EMA {cfg.EMA_FAST}", line=dict(color="#1976D2", width=1),
), row=1, col=1)

# EMA slow
fig.add_trace(go.Scatter(
    x=chart_df["time"], y=chart_df["ema_slow"],
    name=f"EMA {cfg.EMA_SLOW}", line=dict(color="#FB8C00", width=1),
), row=1, col=1)

# Trade entry/exit markers
for _, trade in chart_trades.iterrows():
    color = "#26a69a" if trade["side"] == "long" else "#ef5350"
    symbol = "triangle-up" if trade["side"] == "long" else "triangle-down"
    fig.add_trace(go.Scatter(
        x=[trade["entry_time"]], y=[trade["entry_fill_price"]],
        mode="markers",
        marker=dict(symbol=symbol, size=8, color=color),
        name=f"{trade['side']} entry",
        showlegend=False,
    ), row=1, col=1)

# --- Row 2: Z-score ---
fig.add_trace(go.Scatter(
    x=chart_df["time"], y=chart_df["zscore"],
    name="Z-score", line=dict(color="#7B1FA2", width=1),
), row=2, col=1)

# Z-score band lines
for level, label in [(cfg.Z_BAND_K, f"+{cfg.Z_BAND_K}"), (-cfg.Z_BAND_K, f"-{cfg.Z_BAND_K}")]:
    fig.add_hline(y=level, line=dict(color="gray", dash="dash", width=1),
                  row=2, col=1, annotation_text=label, annotation_position="right")
fig.add_hline(y=0, line=dict(color="gray", width=0.5), row=2, col=1)

fig.update_layout(
    title=f"MNQ 1-min | EMA {cfg.EMA_FAST}/{cfg.EMA_SLOW} Crossover | {VERSION}",
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=True,
    height=700,
    template="plotly_dark",
)
fig.show()
""")

nb2_cell4 = nbformat.v4.new_code_cell("""\
html_path = out_dir / "plotly_price_indicators.html"
fig.write_html(str(html_path))
print(f"Chart saved: {html_path}")
""")

nb2 = nbformat.v4.new_notebook()
nb2.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb2.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}
nb2.cells = [nb2_cell1, nb2_cell2, nb2_cell3, nb2_cell4]

with open("notebooks/02_plotly_exploration.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb2, f)

print("Written: notebooks/02_plotly_exploration.ipynb")
print("Done.")
