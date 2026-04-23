# Scope D — SES_2 Sub-Window Split (Characterization Footnote)

**Date**: 2026-04-23 UTC
**Stage**: Characterization (not a gate, not a preregistered confirmation)
**Council authority**: `tasks/council-report-2026-04-23-b1-scope.html` + `tasks/council-transcript-2026-04-23-b1-scope.md`
**Chairman "one thing to do first"**: quoted verbatim below
**Predecessors**: Probe 2 (`a49f370`), Probe 3 (`b68fe62`), Probe 4 (`c419391` + `ec60bf0`)

---

## Purpose

Answer a single binary question: **within combos {865, 1298, 664}'s SES_2 trade concentration on the 1h test partition, is the edge located in the tradable GLOBEX overnight window (18:00–09:30 ET) or in the partially-untradable post-RTH + settlement halt window (16:00–18:00 ET)?**

The answer gates whether a paper-trade preregistration on any of these combos can proceed. Per the council chairman:

> Run Scope D as a 10-minute patch to `tasks/_probe4_readout.py` — split SES_2 into 16:00–18:00 (post-RTH/settlement halt) and 18:00–09:30 (GLOBEX overnight) on combos {865, 1298, 664} against the 1h test partition, and read out per-trade net $ and n_trades per cell. The answer to "is this overnight edge or settlement-halt artifact?" decides whether paper-trade prereg goes forward at all.

This is explicitly **not** a B1 preregistration. It is a mechanism footnote that costs zero backtest engine compute (post-hoc re-bucketing of existing per-trade parquets) and produces no gate-bound verdict.

---

## Motivation

Amendment 1 of Probe 4's preregistration redefined SES_2 as the complement-of-RTH (`ET min ∈ [0, 570) ∪ [960, 1440)`), which **bundles two mechanically different windows**:

| Sub-bucket | ET clock | ET-minute range | Mechanical character |
|---|---|---|---|
| **SES_2a** — pure GLOBEX overnight | 18:00 → 09:30 next day | `[1080, 1440) ∪ [0, 570)` | Asia + Europe continuous electronic trading. Thin but fillable. Economic story for Z-score mean-reversion is plausible (overnight flows push price in thin book; reversion occurs as liquidity returns). |
| **SES_2b** — post-RTH + settlement halt | 16:00 → 18:00 | `[960, 1080)` | 1 hour of post-RTH continuation plus the CME settlement halt from 17:00–18:00 ET, during which **no trading happens at all**. A backtest that books "fills" here is capturing phantom trades. |

The LLM Council Outsider flagged this bundling as a category error. If a material fraction of Probe 4's reported SES_2 Sharpe (4.05 on combo-1298, 1.44 on combo-664) is sourced from SES_2b, then:

1. The "overnight session-structure inefficiency" narrative is wrong — what's being measured is a post-close continuation / settlement-mechanics artifact.
2. A portion of the measured edge may not be executable (phantom fills during the 17:00–18:00 halt).
3. The thesis that would motivate a paper-trade preregistration — "trade overnight GLOBEX on these combos" — has a different economic basis than the data actually supports.

Scope D is the minimum-cost test that can distinguish these cases.

---

## Method

Input artifacts (all local, already produced):

| File | Source probe | Combo | Trades |
|---|---|---|---|
| `data/ml/probe2/combo865_1h_test.parquet` | Probe 2 | 865 | 220 |
| `data/ml/probe4/combo1298_SES_0_trades.parquet` | Probe 4 | 1298 | 123 |
| `data/ml/probe4/combo664_SES_0_trades.parquet` | Probe 4 | 664 | 780 |
| `data/NQ_1h.parquet` | engine cache | — | 1h bar timeline |

Procedure:

1. Load `data/NQ_1h.parquet`, apply `split_train_test(df, 0.8)` to isolate the 1h test partition (2024-10-22 → 2026-04-08).
2. For each bar, compute ET minute-of-day via `pd.to_datetime(time).dt.tz_convert("America/New_York")`. This handles EDT/EST transitions correctly across the multi-year window.
3. For each combo's per-trade parquet, read `entry_bar_idx` and `net_pnl_dollars`. Map each trade to its ET minute-of-day by indexing into the test-partition bar timeline.
4. Apply three mutually exclusive masks per trade:
   - `SES_1 RTH`: `et_min >= 570 AND et_min < 960`
   - `SES_2a pure overnight`: `et_min >= 1080 OR et_min < 570`
   - `SES_2b post-RTH + halt`: `et_min >= 960 AND et_min < 1080`
5. Compute per-bucket metrics (n_trades, mean_pnl, std_pnl, Sharpe annualized over `YEARS_SPAN_TEST = 1.4799`, net $/yr). Sharpe formula mirrors `_probe4_readout.py` exactly (same `YEARS_SPAN_TEST` inherited conservative bias; preserved for cross-probe comparability per `feedback_years_span_cross_tf.md`).
6. Emit `data/ml/scope_d/readout.json` with the 9-cell table (3 combos × 3 buckets) plus a short "SES_2a dominates?" interpretation field computed mechanically.

**No gates. No pass/fail. No multiplicity correction.** This is a characterization readout only. Stage = characterization; notional denominator = 3 combos × 2 sub-windows = 6 cells (plus 3 SES_1 RTH cells for reference). Nothing downstream auto-fires from the numbers.

---

## Outcome interpretations (for human reading, not mechanical branching)

After the readout lands, a human (or a follow-up council on paper-trade prereg scope) will read the 9-cell table and judge one of four rough regimes:

1. **SES_2a dominates** — pure GLOBEX overnight carries ≥ ~80% of per-combo net $ and ≥ ~75% of SES_2 trade count for all three combos, and SES_2b per-trade mean is near zero or negative. Interpretation: the "overnight inefficiency" narrative holds, paper-trade preregistration should bind execution to 18:00–09:30 ET only.
2. **SES_2b dominates** — post-RTH + halt carries ≥ ~50% of per-combo net $. Interpretation: what's really being measured is a post-close continuation or settlement-mechanics artifact; the overnight thesis is wrong and paper-trade preregistration must be rescoped (or abandoned).
3. **Mixed (both contribute materially)** — both sub-windows carry ≥ ~25% of per-combo net $. Interpretation: not purely overnight, not purely halt; the "session-structure" label papers over two distinct sources of edge that need separate investigation. Paper-trade preregistration cannot bind cleanly; council reconvene.
4. **Combos diverge** — one combo shows SES_2a dominance while others show SES_2b or mixed. Interpretation: the cross-combo generalization weaker than Probe 4's aggregate suggested; Reviewer 3's "effective n = 1" concern is reinforced.

None of these regimes auto-trigger an action. Human reading of the table + a paper-trade scope council (if applicable) is the next step.

---

## Caveats and disclosures

- **Partition reuse.** The 1h test partition has now been consumed by Probes 2, 3, 4, and this footnote. Scope D does not pretend to be partition-disciplined — it is explicitly a re-read of already-measured data under a finer mask. Any interpretation must price this (numbers at this depth of reuse are progressively less trustworthy as independent confirmation; they remain useful as mechanistic characterization).
- **No holding-time axis.** The Probe 4 per-trade parquets for 1298 and 664 saved only `(net_pnl_dollars, entry_bar_idx)` — no `hold_bars`. The Probe 2 combo865 parquet has `hold_bars`, but without cross-combo symmetry we cannot report holding-time effects. The Outsider's holding-time/opportunity-cost concern from the council is noted but unaddressed here; if the SES_2a/SES_2b result merits a paper-trade prereg, that prereg must include holding-time as a first-class metric.
- **Phantom fills in SES_2b.** The backtest engine does not model the 17:00–18:00 CME settlement halt. If SES_2b trades cluster in 17:00–18:00 ET specifically, those are by construction phantom — the reported PnL assumes a fill that a live broker could not have given. Scope D's sub-partitioning exposes this but does not re-partition SES_2b into "16:00–17:00 tradable" vs "17:00–18:00 halt" — if SES_2b is material, that finer split is an immediate follow-up.
- **Insufficient-n cells allowed.** SES_2b may contain very few trades (especially for combo-1298, which has only 123 SES_0 trades total). Per-bucket metrics with `n_trades < 30` should be read as directional-only, not statistically resolved.
- **Readout is local.** Pure pandas, no engine compute, matches the `_probe4_readout.py` convention ("Runs anywhere (no engine compute). Wall-clock < 1 s.") Per `feedback_remote_execution_only.md`, engine/sweep compute is remote-only; readout-grade pandas is local-OK under the same convention already in use.

---

## Outputs

- `data/ml/scope_d/readout.json` — 9-cell session decomposition table + cross-combo summary
- `tasks/_scope_d_readout.py` — the script that produced it
- `tasks/scope_d_brief.md` — this document

## Authority

- `tasks/council-report-2026-04-23-b1-scope.html` — council verdict, 5 advisors + 5 peer reviews + chairman
- `tasks/council-transcript-2026-04-23-b1-scope.md` — full transcript, anonymization revealed
- `tasks/probe4_verdict.md` §Amendment 1 + §Interpretation — surfaced the bundling issue disclosed here
- `memory/feedback_council_methodology.md` — Rule 1 / Rule 2 framing applied to the council that authorized this footnote
