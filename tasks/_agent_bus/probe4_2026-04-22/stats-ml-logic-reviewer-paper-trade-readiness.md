---
from: stats-ml-logic-reviewer
run_id: probe4_2026-04-22
timestamp: 2026-04-23T08:15:00Z
scope_reviewed: Paper-trade readiness decision for combos {865, 1298, 664} 1h NQ post Probe 4 + Scope D
verdict: NOT_READY (CRITICAL timezone bug invalidates session decomposition underlying current paper-trade scope)
critical_flags: 3
warn_flags: 4
info_flags: 3
cross_references:
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md (Pass 1)
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-pass2.md (Pass 2)
  - tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer-pass4.md
  - tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer-scripts.md
  - tasks/probe3_verdict.md
  - tasks/probe4_verdict.md
  - tasks/scope_d_brief.md
  - data/ml/probe2/combo865_1h_test.parquet (n=220, full schema)
  - data/ml/probe4/combo1298_SES_0_trades.parquet (n=123, net_pnl + idx only)
  - data/ml/probe4/combo664_SES_0_trades.parquet (n=780, net_pnl + idx only)
  - data/ml/scope_d/readout.json
  - data/ml/probe4/readout.json
  - data/ml/probe3/1h_ritual.json
  - tasks/_probe3_1h_ritual.py (timezone bug at line 186)
  - tasks/_probe4_readout.py (timezone bug at line 129)
  - tasks/_scope_d_readout.py (timezone bug at line 133)
---

# Paper-Trade Readiness Review — Combos {865, 1298, 664} 1h NQ

## Verdict

**NOT_READY.** A previously undetected timezone bug runs through **Probe 3 §4.4, Probe 4 §5 SESSION_CONFOUND routing, and Scope D's entire readout** — all three artifacts use `tz_localize("UTC").tz_convert("America/New_York")` on `data/NQ_1h.parquet`, but the raw timestamps in that file are in **US Central Time (CT)**, not UTC. This is triple-confirmed by (i) the CME Globex maintenance halt appearing at raw hour 16 (= 16:00-17:00 CT = 17:00-18:00 ET, the correct halt window, which would fall at 22:00-23:00 UTC if raw were UTC), (ii) `session_break=True` appearing exclusively at raw hour 17 (= CME reopening at 17:00 CT), and (iii) the hourly volume peak at raw hours 8-14 (= RTH 08:30-15:00 CT = 09:30-16:00 ET). Under the correct timezone, the "SES_2a overnight" bucket that Scope D reports as dominating is actually **~80% RTH trades mislabeled as overnight**, and the real per-session picture is inverted: combos 865 and 1298 show their edge in RTH, not overnight. Combo 664 is the only one with a genuine (weak) overnight lean, and its overnight Sharpe drops from the reported 1.65 to 1.17 with a 95% bootstrap CI of [-0.43, +2.80] that spans zero. Three of the five open concerns I was asked to evaluate (partition reuse, effective-n, session-decomposition validity) either materially worsen or completely flip direction under the corrected analysis. The Probe 2 pass, Probe 4 absolute gates, and Probe 4 Welch-t 3.85 are all mechanically untouched by this bug (they are timezone-independent total-combo statistics), but the **paper-trade scope you are about to preregister — "trade only SES_2a overnight" — is anchored on the buggy session decomposition**, so a preregistration at this scope would commit you to forward-walking a window that the corrected data does not support. Do not launch paper-trade preregistration until the timezone bug is fixed, all session-dependent verdicts are re-run, and a fresh council adjudicates whether the corrected picture supports the same PAPER_TRADE branch or a different one.

---

## Independent Re-Derivations

I independently verified every headline number you supplied and found:

### Headline numbers (all reproduce exactly)

| Claim | Reported | Re-derived | Match? |
|---|---:|---:|---|
| Combo 865 net Sharpe 1h test | 2.89 | 2.8949 | ✓ |
| Combo 865 total $/yr | $124,896 | $124,896 | ✓ |
| Combo 865 n_trades | 220 | 220 | ✓ |
| Combo 1298 net Sharpe | 3.551 | 3.5510 | ✓ |
| Combo 1298 $/yr | $153,259 | $153,259 | ✓ |
| Combo 1298 n_trades | 123 | 123 | ✓ |
| Combo 664 net Sharpe | 1.340 | 1.3397 | ✓ |
| Combo 664 $/yr | $83,426 | $83,426 | ✓ |
| Combo 664 n_trades | 780 | 780 | ✓ |
| Welch-t 1298 vs 664 (full 24h) | 3.851 | 3.8506 | ✓ |
| Welch-t p (one-tailed) | — | 0.000091 | — |

All core Probe 4 gate statistics reproduce to 4 decimal places.

### Parameter-space distance between {865, 1298, 664}

Direct inspection of per-combo config JSON (data/ml/probe4/combo{1298,664}_SES_0_combos.json and data/ml/probe2/combo865_1h_test_manifest.json):

| Pair | Differ / total parameters |
|---|---:|
| 865 vs 1298 | 14 / 21 |
| 865 vs 664 | 13 / 21 |
| 1298 vs 664 | 15 / 21 |

**The Probe 4 prereg's claim that 1298 is "Δ=1 microstructure neighbor" of 865 is FALSE.** 865 and 1298 differ on:
- z_band_k (2.30 vs 2.27 — close)
- z_window (41 vs 14 — 3× different)
- volume_zscore_window (47 vs 12)
- ema_fast / ema_slow (6/48 vs 4/15)
- stop_fixed_pts (17.0 vs 15.2)
- min_rr (1.85 vs 2.79)
- exit_on_opposite_signal (False vs True)
- max_hold_bars (120 vs 480 — 4× different)
- zscore_confirmation (False vs True)
- z_window_2 / z_window_2_weight (47/0.33 vs 0/0)
- entry_timing_offset (1 vs 0)
- fill_slippage_ticks (1 vs 2)

Combo 664 differs further — including `z_type` switching from `parametric` to `quantile_rank`, a completely different z-score formulation.

**This is GOOD news for Probe 4's original "independent confirmations" framing** — the combos are genuinely distinct parameter-space points, not microstructure perturbations of each other. But it contradicts the prereg text you supplied.

### Trade-bar overlap between combos (per-bar entry indices)

| Pair | Overlap (N) | % of smaller | PnL correlation on overlap |
|---|---:|---:|---:|
| 865 ∩ 1298 | 6 | 4.9% | -0.33 |
| 865 ∩ 664 | **147** | **66.8%** | **+0.76** |
| 1298 ∩ 664 | 5 | 4.1% | -0.61 |
| 865 ∩ 1298 ∩ 664 | 4 | — | — |

**Combos 865 and 664 share 147 trade bars out of combo-865's 220 trades and have 0.76 PnL correlation on those shared bars.** This IS effective-sample-size collapse: 865 and 664 are substantially the same system observed from two slightly different parameter vantage points. Combo 1298 is distinct. So the "three combos = three independent confirmations" framing should be replaced with "**two effectively independent strategies**: {865, 664} as one family, {1298} as another."

### CRITICAL: Timezone bug in session decomposition

The repo-wide assumption (encoded in `tasks/_probe3_1h_ritual.py:186`, `tasks/_probe4_readout.py:129`, `tasks/_scope_d_readout.py:133`) is:

```python
ts_utc = ts.dt.tz_localize("UTC")      # assumes raw is UTC
ts_et  = ts_utc.dt.tz_convert("America/New_York")
```

Four independent lines of evidence say the raw data is **CT**, not UTC:

1. **Missing hour 16**: `data/NQ_1h.parquet` has 0 bars at raw hour 16, 1821 at raw hour 17. The CME Globex maintenance halt is 16:00-17:00 **CT** (= 17:00-18:00 ET). Under the UTC hypothesis, the hole would be at raw UTC 22:00/23:00, not raw hour 16. Under the ET hypothesis, the hole would be at raw hour 17 (= 17:00 ET = the halt), not at raw hour 16.
2. **session_break=True is ALL on raw hour 17** (1821 of 1821). That's CME reopening at 17:00 CT = 18:00 ET. If raw were UTC, reopening would be at 22:00/23:00 UTC, not hour 17. If raw were ET, reopening would be at 18:00 ET, not 17:00 ET.
3. **Volume peak at raw hours 8-14** (medians 15k-86k vs 3-8k elsewhere). RTH in CT is 08:30-15:00 CT; in ET is 09:30-16:00 ET; in UTC is 13:30-20:00 UTC (winter) or 14:30-21:00 UTC (summer). Only CT's RTH maps to the observed peak.
4. **First bar 2019-01-01 17:01**: 17:01 CT = 18:01 ET = 1 minute after the NYD holiday Globex reopen at 17:00 CT / 18:00 ET. No other timezone makes this timestamp a sensible first-bar.

Under the **corrected** CT→ET mapping (`tz_localize("America/Chicago").tz_convert("America/New_York")`), the session decomposition inverts almost completely. Numbers reported / corrected:

| Combo | Bucket | Reported n | Reported Sharpe | Corrected n | Corrected Sharpe | 95% boot CI |
|---|---|---:|---:|---:|---:|---|
| 865 | SES_1 RTH | 65 | 0.64 | **126** | **2.64** | [+1.02, +4.41] |
| 865 | SES_2a OV | 143 | 3.32 | **82** | **1.47** | [-0.16, +3.19] |
| 865 | SES_2b 16-18 | 12 | -0.46 | 12 | +0.02 | (too few) |
| 1298 | SES_1 RTH | 23 | -0.17 | **77** | **4.41** | [+2.79, +6.47] |
| 1298 | SES_2a OV | 98 | 4.21 | **40** | **0.53** | [-1.23, +2.06] |
| 1298 | SES_2b | 2 | 0 | 6 | — | (too few) |
| 664 | SES_1 RTH | 169 | 0.14 | **243** | **0.54** | [-1.04, +2.14] |
| 664 | SES_2a OV | 562 | 1.65 | **505** | **1.17** | [-0.43, +2.80] |
| 664 | SES_2b | 49 | -0.51 | 32 | +0.45 | — |

### Welch-t 1298 vs 664, re-derived by session

| Session | n_1298 | n_664 | Welch-t | p (1-sided) |
|---|---:|---:|---:|---:|
| Full 24h | 123 | 780 | 3.85 | 9.1e-5 |
| RTH (corrected) | 77 | 243 | **4.89** | 6.5e-7 |
| Overnight (corrected) | 40 | 505 | 0.40 | 0.34 |

**The 3.85 Welch-t that passed Probe 4's §4.3 gate is dominated by the RTH difference, not overnight.** The overnight-only Welch-t is a null result (p=0.34).

### Friction robustness (concern #6)

Using the per-trade parquets + reverse-engineered contracts (865: 73 ct @ $6/RT; 1298: 82 ct @ $7/RT; 664: 32 ct @ $6/RT — derived from the `fill_slippage_ticks` axis that bumps COST_PER_CONTRACT_RT in `scripts/param_sweep.py:895-901`):

**Under the (incorrect) Scope D SES_2a bucket** — Sharpe at 1.0x / 1.5x / 2.0x / 2.5x / 3.0x friction:

| Combo | 1.0x | 1.5x | 2.0x | 2.5x | 3.0x |
|---|---:|---:|---:|---:|---:|
| 865 SES_2a | 3.32 | 2.71 | 2.09 | 1.48 | 0.86 |
| 1298 SES_2a | 4.21 | 3.71 | 3.21 | 2.72 | 2.22 |
| 664 SES_2a | 1.65 | 0.96 | 0.27 | **-0.42** | **-1.11** |

Combo 664 goes negative at **1.5× friction** (= ~$9/contract RT, a realistic overnight spread estimate). Combo 865 drops below the 1.3 implied floor at ~2.5-3× friction. Combo 1298 is friction-robust across the range. But because the corrected timezone says this "SES_2a" bucket is actually mostly RTH, the friction-sensitivity numbers for 864's "overnight" bucket are misleading — the true overnight bucket (n=82 for 865, n=40 for 1298, n=505 for 664) needs separate sensitivity analysis that the current artifacts do not support.

### Hold-time (concern #7)

For combo 865 (the only combo with `hold_bars` in the parquet): median 1 bar, 95th percentile 2 bars, max 5 bars. Total bars held = 249 of 8380 test bars = 3.0% of partition time. **Not a load-bearing concern for 865.** Cannot be verified for 1298/664 because their Probe 4 parquets only store net_pnl + entry_bar_idx.

### Paper-trade forward CI projection

Using the formula SE(Sharpe) ≈ sqrt((1+S²/2)/n_obs) and the **corrected** (RTH-focused) per-combo statistics, the forward 95% CI half-width on realized Sharpe after T years of paper trading would be:

| Combo / window | Backtest Sharpe | trades/yr | 0.5yr HW | 1.0yr HW | 1.5yr HW |
|---|---:|---:|---:|---:|---:|
| 865 RTH | 2.64 | 85 | ±0.64 | ±0.45 | ±0.37 |
| 1298 RTH | 4.41 | 52 | ±1.26 | ±0.89 | ±0.73 |
| 664 Overnight | 1.17 | 341 | ±0.20 | ±0.14 | ±0.11 |

**For 1298 specifically**, the backtest Sharpe 4.41 has a 95% CI half-width of ±0.89 after 1 year of paper trading. A realized Sharpe of 2.0 would be 2.7 backtest-SDs below the point estimate but inside a 99.5% CI — i.e., 1 year of forward data would be insufficient to tell a "true Sharpe 4" edge apart from a "true Sharpe 2" or even "true Sharpe 1" outcome. This is Concern #8 confirmed: **the paper-trade phase needs multiple years of forward data to discriminate the point estimates this backtest is producing from more pedestrian true Sharpes.**

---

## Methodological Audit

### Choice of test / model

The mechanical test batteries (Probe 2 gates, Probe 3 four-gate suite, Probe 4 absolute + Welch-t gates) are individually well-specified for the questions they pose, with the corrections and fixes from the Pass-1 / Pass-2 reviews already incorporated. The Welch-t is the right statistic for comparing two per-trade means with unequal variance and unequal n. Probe 2's three-gate set and Probe 4's absolute gates are defensible engine-level filters.

**What is wrong is the plumbing underneath all of them**: the `_et_min` computation that partitions trades into sessions silently misinterprets the raw timezone. The statistical tests then operate on mislabeled buckets and produce results that look coherent but do not mean what the code says they mean.

### Assumptions

| Assumption | Holds? | Notes |
|---|---|---|
| Per-trade independence (Welch-t) | Probably | Hold time ~1 bar, cooldown 3 bars, so weakly autocorrelated at best. Not invalidating. |
| Chronological train/test split | Holds | `split_train_test(df, 0.8)` is a straight date cut. No leakage from that axis. |
| Session partition represents deployment | **DOES NOT HOLD** | The "18:00-09:30 ET" window that the code implements is actually a mislabeled mixture of RTH + overnight (with RTH dominant for 865/1298). Deploying a bot that trades only the "SES_2a" window as computed would trade a fundamentally different wall-clock window than the label suggests. |
| Friction is session-invariant | **DOES NOT HOLD** | $5/contract RT is a single constant; overnight bid-ask widens 2-4× on NQ. Combo 664 goes negative at 1.5× friction — and 664's edge is what the data "actually" puts overnight. |
| Partition reuse is independent across probes | **DOES NOT HOLD** | Probes 2, 3, 4, Scope D all read the same 1h test partition. Each probe pre-registers against statistics already shown by the prior probe to be favorable for combo-865. A paper-trade prereg anchored on any of these statistics is the 5th read. The B1 council Contrarian is right to flag this. |

### Alternatives not considered

1. **Correct timezone handling.** The most basic fix: verify the raw data's timezone with a single-minute volume-peak and halt-hour sanity check before partitioning. This was not done in any of Probe 3, Probe 4, or Scope D scripts.
2. **Purged / embargoed cross-validation** on the 1h parquet, holding out combo-specific entry bars. This would let you empirically estimate the partition-reuse discount instead of arguing about it in prose.
3. **Permutation null on entry times**: shuffle entry times within-session (or within-day) and re-compute each combo's per-session Sharpe. A real session-dependent edge should not survive within-session time shuffling; a label-confusion artifact would.
4. **Bootstrap-sharpe confidence intervals** on every cell of Probe 3 §4.4's 16-cell ritual — not just the 8/16 count. An "exactly-at-threshold" result with 5 of the 8 pass cells showing CI lower bound < 1.3 is qualitatively different from 8/16 with all CIs safely above.

---

## Data Audit

- **Descriptive reality (corrected TZ)**: Across the 1.46-year test partition (8380 bars): RTH (09:30-16:00 ET) = 1810 bars; Overnight (18:00-09:30 ET) = 5838 bars; Transition (16:00-18:00 ET) = ~730 bars. Overnight dominates bar count 3.2:1, but for combo 865 and combo 1298, **RTH carries most of the trade edge**. For 664 (the 780-trade combo), trades are roughly 2:1 overnight vs RTH (505 : 243), and the overnight lean IS real (Sharpe 1.17 vs 0.54 RTH), just much weaker than reported.
- **Leakage / contamination risks**:
  - **Partition reuse**: 5th read on the same 1h test partition at paper-trade prereg. Each marginal probe was pre-registered, but the partition was pre-selected by Probe 1's training-partition sunset + Probe 2's carve-out, both of which optimized over the same historical path. This is a genuine multiplicity load.
  - **Effective-n collapse between 865 and 664**: 66.8% trade-bar overlap + 0.76 PnL correlation. {865, 664} should be treated as one parametric realization with TWO independent survivors = 1298 + {865 ∪ 664}.
  - **Combo-865/1298's RTH signal comes from a small-n bucket**: n=126 and n=77 respectively. The Probe 2 verdict framed combo-865 as a 220-trade confirmation; the corrected breakdown says ~60% of those 220 are RTH, which is where the edge is, but the RTH-only n is 126.
- **Representativeness**: The test partition is 2024-10-22 → 2026-04-08. This spans one regime transition (late-2024 bull-leg + 2025 consolidation + early-2026 recovery). Whether this is representative of a 2026-forward regime is the core unanswerable question. Forward-walking would begin with a different implied-vol regime than the mid-2025 window that dominates the per-combo statistics.

---

## Result Interpretation

### Statistical significance

- **Probe 4 full-24h Welch-t 3.85 (p=9e-5)** holds. The corrected RTH-session Welch-t is actually 4.89 (stronger) and the overnight Welch-t is 0.40 (null). Since Probe 4's gate tested full-24h, the gate PASS is valid.
- **Absolute gates** (Sharpe ≥ 1.3, n ≥ 50, $/yr ≥ $5k) for both combos on full 24h: all pass with margin for 1298, pass by ~0.04 Sharpe margin for 664 (flagged WARN below — only 32% chance 664's true Sharpe is above 1.3 under bootstrap).

### Practical significance (in domain units)

- **Combo 865 RTH (corrected)**: $1,008/trade net mean, 126 trades in 1.46 years, = ~85 trades/yr * $1,008 = ~$85,700/yr realized backtest. Expected forward range (1 yr): $85k ± $50k on point Sharpe 2.64 ± 0.45 CI.
- **Combo 1298 RTH (corrected)**: $2,817/trade mean, 77 trades in 1.46 years, = ~52 trades/yr * $2,817 = ~$146k/yr backtest. Forward range (1 yr): $146k ± $100k on point Sharpe 4.41 ± 0.89 CI.
- **Combo 664 Overnight (corrected)**: $173/trade mean, 341 trades/yr, = ~$59k/yr backtest. Forward range (1 yr): $59k ± $25k on point Sharpe 1.17 ± 0.14 CI, but with 32% probability of true Sharpe < 1.3 and 2.3% probability of true Sharpe < 0.

### Multiple testing / power

- **Multiplicity load is heavy but has been accounted for correctly at the Probe 1, 2, 3, and 4 stages individually.** The cross-probe partition reuse adds a multiplicity layer that none of the individual prereg docs bind. Concretely: every probe implicitly selected combo-865 as the survivor from a 1500-combo pool (Probe 1 → Probe 2), and Probe 4's 1298 and 664 were the Probe 3 council's A/B picks from an even larger partition-ranking. A fair posterior that accounts for partition-level selection would discount each absolute Sharpe by ~sqrt(log(1500)/n) ≈ 0.25 units for 220-trade combos.
- **Power for paper-trade** (concern #8): At 5% fixed-risk on $50k ($2500/trade at ~43-82 contracts), 1 year of forward data delivers a ±0.45 to ±0.89 Sharpe half-width — insufficient to falsify a "half the backtest Sharpe" null at 95% confidence.

### What the result does and does not license you to conclude

**Licensed conclusions (survive timezone correction)**:
- Combo 865 has a real full-24h edge on 220 1h-test trades: net Sharpe 2.89, 95% CI that stays above 1.3 under trade-level bootstrap.
- Combo 1298 has a strong full-24h edge on 123 trades: net Sharpe 3.55.
- Combo 1298's per-trade edge is significantly larger than combo-664's per-trade edge (Welch-t 3.85 on full 24h, equivalent t=4.89 on RTH-only corrected buckets).
- Combos 865 and 664 overlap 66.8% on trade bars and correlate 0.76 — they are ONE effective observation, not two.

**NOT licensed (invalidated by timezone bug)**:
- "Edge concentrates in GLOBEX overnight" — FALSE for 865 and 1298 under corrected TZ; TRUE but weak for 664.
- "SES_2a dominates unanimously across all three combos" — the statement is true under the buggy code's labeling but WRONG under wall-clock ET interpretation.
- "Overnight per-trade ratio ~3.45× RTH" — this figure came from the buggy partition.
- "Combo 664's 1.65 overnight Sharpe" — actual overnight Sharpe 1.17 with CI spanning zero.
- Probe 3 §4.4's "exactly-at-threshold 8/16" pass. The 8 pass cells disproportionately include the "SES_2_overnight_only" cells, which the corrected TZ says are actually RTH-dominant trades. Under correct labeling, those cells should be re-sorted to SES_1, and the SES_1 cells (thought to be RTH-dominant) become actually overnight — which will change the cell-pass count.

---

## Impact on Project Scope

### If the result stands (it doesn't, as stated)

The current paper-trade scope — "trade only SES_2a (18:00-09:30 ET)" — would execute on a different wall-clock window than the one the backtest evaluated for 865 and 1298. You'd be live-trading a window that corresponds to what the backtest called "SES_1 RTH" performance (Sharpe 0.64 for 865, Sharpe -0.17 for 1298 in the reported partition). **That is the opposite of what the paper-trade plan intends.** This is a catastrophic mis-specification risk for forward trading.

### If the result is fragile (it is)

The following downstream decisions are premature:
- Any paper-trade preregistration anchored on "SES_2a overnight."
- Any sizing rule calibrated to the "overnight per-trade σ" reported by Scope D.
- Any broker-adapter selection that assumes overnight-only execution is sufficient.
- The Probe 3 "PAPER_TRADE branch, posterior [0.65, 0.85]" verdict itself, because §4.4 (one of the 4 gates) used the buggy session mapping. If the §4.4 pass-count drops below 8 under corrected labeling, the F-count becomes 1, and the branch becomes COUNCIL_RECONVENE — not PAPER_TRADE — per Probe 3's own preregistered routing.

### Consistency with prior findings

- Probe 2 combo-865 PASS: **UNAFFECTED** (total-combo statistic).
- Probe 3 §4.1 regime halves (H1/H2): UNAFFECTED (total-combo, time-halved).
- Probe 3 §4.2 parameter neighborhood: UNAFFECTED (27/27 is over parameter perturbations, not session masks).
- Probe 3 §4.3 15m negative control: UNAFFECTED (0/16 is a 15m test on total trades; the intra-day session filter inside it would be affected, but since the result was 0/16 the affected cells pre-passed anyway).
- Probe 3 §4.4 1h session/exit ritual: **AFFECTED**. The 8/16 exact pass is potentially no longer 8/16 under corrected labeling.
- Probe 4 absolute + Welch-t gates: UNAFFECTED.
- Probe 4 §5 SESSION_CONFOUND verdict: **AFFECTED**. The row-2 condition `(SES_2 Sharpe − SES_1 Sharpe) > 1.0` used the buggy labeling. Under correct labeling, for combo-1298: RTH Sharpe 4.41 vs Overnight Sharpe 0.53 — differential +3.88 FAVORS RTH. The row-2 clause would STILL FIRE (the sign magnitude is the same), but the interpretation reverses: "edge concentrates in RTH, not in SES_2 overnight." The branch label would need a different name.
- Scope D's unanimous "SES_2a dominates": **INVALIDATED**.

---

## Recommended Alternatives (ranked by information value)

### 1. FIX THE TIMEZONE BUG (highest info value, lowest effort)

**What**: Change three files:
- `tasks/_probe3_1h_ritual.py:186`: `ts.dt.tz_localize("UTC")` → `ts.dt.tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")`
- `tasks/_probe4_readout.py:129`: same
- `tasks/_scope_d_readout.py:133`: same
- Any other `_probe*` or `_scope*` script that does a `tz_localize("UTC")` on `data/NQ_1h.parquet`.

**Why better**: Answers the paper-trade-session-definition question definitively. Cannot proceed without it.

**Verification**: Re-run Probe 3 §4.4 ritual; Scope D's readout; Probe 4 §5 decomposition. Check that (a) §4.4 cells' SES_1 and SES_2 swap relative pass-fail patterns, (b) cross-combo "overnight dominance" becomes "RTH dominance" for 865 and 1298, (c) session_break=True bars fall inside SES_2a (18:00 ET reopen) in the corrected bar map.

**Decision rule for paper-trade**: If Probe 3 §4.4 still passes ≥8/16 after correction, PAPER_TRADE is recoverable at a DIFFERENT session-scope (RTH-focused, not overnight). If it falls to ≤7/16, F-count = 1 and Probe 3 branches to COUNCIL_RECONVENE per its own preregistered logic.

### 2. Combo-agnostic K-fold on v11 1h (if §1 result implies paper-trade is still gated)

**What**: K-fold the v11 1h ML sweep dataset on COMBO_ID (not trade-level); refit a regression of combo-level Sharpe on parameters on K-1 combos; predict on held-out combo; compare out-of-fold Sharpe distribution to the top-3 selection's in-sample Sharpe.

**Why better**: Answers the question the council's Reviewer 4 raised — "is Probe 1 falsification + Probes 2-4 carve-out the same phenomenon (thin-overnight-noise harvesting)?" If the top-50 combos' out-of-fold Sharpe distribution is indistinguishable from random selection, it's the same phenomenon. If it's a right-tailed distribution around 0.5-1.0 with a long tail catching 865/1298, it's a real (but selection-heavy) edge.

**Decision rule**: If out-of-fold median Sharpe of "high-scoring combos" ≤ in-sample median of "random combos," paper-trade is not defensible.

### 3. Friction sensitivity curve for each combo's ACTUAL (corrected) session (medium info value)

**What**: Redo the friction-robustness table (1.0×, 1.5×, 2.0×, 2.5×, 3.0×) on the CORRECTED RTH bucket for 865 and 1298, and the CORRECTED overnight bucket for 664. Add a bid-ask-spread estimate from actual NQ RTH vs overnight tape (or a conservative proxy) to establish which friction level is realistic.

**Why better**: Paper-trade sizing and cost budget both depend on this. 664's 1.5× → negative edge under the *buggy* overnight bucket is a pre-existing alarm; the corrected buckets may be worse or better.

**Decision rule**: If any combo goes negative at 1.5× friction on its corrected edge-carrying session, exclude it from paper-trade scope.

### 4. Redo Probe 4's §5 SESSION_CONFOUND row under corrected labeling (medium info value)

**What**: Under corrected TZ, does the row-2 condition still fire? Yes — for 1298, RTH (4.41) − overnight (0.53) = +3.88 > 1.0. But the CONFOUND interpretation flips direction: the edge is SESSION-RTH-concentrated, not SESSION-OVERNIGHT-concentrated. Recompute what this means for the paper-trade scope. RTH is liquid, has tighter spreads, is tradable for retail — this is a *less* adverse finding than overnight dominance.

**Why better**: Changes the risk profile from "fragile overnight edge that may collapse on wider spreads" to "RTH edge that is in the more liquid session." Material for go/no-go.

### 5. Bootstrap CIs on every Probe 3 §4.4 cell, not just the count (low effort, high info)

**What**: For each of the 16 cells, compute a 95% bootstrap Sharpe CI (not just pass/fail). Tally how many cells have CI_lower ≥ 1.3 vs how many barely clear the threshold.

**Why better**: An exactly-at-threshold 8/16 where all 8 have CI_lower ≥ 1.3 is a different epistemic state than 8/16 where 3 have CI_lower < 1.0. The former supports PAPER_TRADE; the latter is within noise of PAPER_HOLD.

**Decision rule**: If ≤5 of the 8 pass-cells have CI_lower ≥ 1.3, treat the gate as EFFECTIVELY FAILED even at an 8/16 mechanical pass.

### 6. Regenerate combo 1298 and 664 full parquets (with friction_dollars, gross_pnl, mfe, hold_bars) — high info value over long run

**What**: Re-run the engine for combos 1298 and 664 on the 1h test partition with the full Probe-2 schema. This is a pure recomputation (no model retrain), takes <1 minute per combo.

**Why better**: Currently, friction robustness, hold-time, and per-trade diagnostics are only computable for 865. The full schema is required for any serious paper-trade sizing analysis on 1298/664. Right now, the only published artifacts for those combos are 2-column parquets — this blocks most of the sizing-floor and friction-floor analyses the paper-trade prereg should cite.

---

## Severity Flags

### CRITICAL

**CRITICAL-1: Timezone bug in all 1h session decompositions.** `tz_localize("UTC")` on `data/NQ_1h.parquet` is wrong — the raw data is CT. Fix required in `tasks/_probe3_1h_ritual.py:186`, `tasks/_probe4_readout.py:129`, `tasks/_scope_d_readout.py:133`, and verified-absent in any other `_probe*`/`_scope*` script on 1h. **Paper-trade preregistration anchored on the current SES_2a-dominates narrative would execute on the wrong wall-clock window.** Effort to fix: ~10 lines across 3 files + re-run of Probe 3 §4.4, Probe 4 §5 routing, and Scope D readout.

**CRITICAL-2: Probe 3 §4.4's "exactly-8-of-16" pass is built on buggy session labels.** Because the result was exactly at threshold, even small changes in cell assignments under the correct mapping may flip the pass. If it becomes 7/16, Probe 3's own preregistered routing (F-count = 1) triggers a COUNCIL_RECONVENE, not PAPER_TRADE. This directly affects the posterior-range citation [0.65, 0.85] that the council has been using to authorize paper-trade prep.

**CRITICAL-3: Effective-n collapse between 865 and 664** (66.8% trade-bar overlap + 0.76 PnL correlation). Probes 2 + 4 have been presented as 3 independent combo confirmations but are really 2 (= {865 ∪ 664} + {1298}). Every post-Probe-2 multiplicity correction that assumed independent survivors is too permissive.

### WARN

**WARN-1: Partition reuse at paper-trade prereg = 5th read.** Probes 2, 3, 4, Scope D each read the 1h test partition; paper-trade prereg anchored on these statistics is read #5. This is the Contrarian's "denom=1 dressed up" concern. No formal fix; the correct response is either (a) reserve a fresh temporal holdout before paper-trade commits, or (b) accept that the posterior range should be discounted further and set a higher paper-trade bar.

**WARN-2: Combo 664 on its CORRECTED overnight bucket has 95% CI spanning 0.** Bootstrap CI [-0.43, +2.80]. 32% probability true Sharpe < 1.3, 2.3% probability true Sharpe < 0. Including 664 in the paper-trade basket buys little discrimination power and adds an edge-possibly-nonexistent combo.

**WARN-3: Combo 664's narrow-miss on Probe 4 absolute gate (+0.04 Sharpe margin over 1.3) looks even thinner with corrected-TZ friction sensitivity.** Under 1.5× overnight spreads, 664's CORRECTED overnight Sharpe was 0.96 in the uncorrected analysis; I don't have the corrected-overnight-only friction curve yet because I'd need to re-bucket 664's gross_pnl which is not in its 2-column parquet.

**WARN-4: Probe 4 prereg's claim "1298 is Δ=1 microstructure neighbor of 865" is FALSE.** Actual distance: 14/21 parameters differ. This is a WARN rather than CRITICAL because the direction-of-error is favorable (combos are MORE independent than claimed) but it means the verdict's own §7.5 basin-counting language is inaccurate as written.

### INFO

**INFO-1: The Welch-t α=0.025 gate calibration from Pass-2 §5.2 remains correct.** t_crit at Satterthwaite df is 1.98, so a cutoff of 2.0 is conservatively inside the nominal α. No issue.

**INFO-2: Forward paper-trade CI widths are wide at 1 year.** Discrimination between "true Sharpe 2" and "true Sharpe 4" would require 2-3 years of forward data at these per-combo trade rates. The paper-trade prereg's stopping rules should not be set assuming 1 year will be decisive.

**INFO-3: The EX_2 TOD exit at `tod_exit_hour=19 UTC` in Probe 3's §4.4 is ALSO mis-specified if engine's tod_exit_hour interprets raw CT hours.** Depends on exactly which source the engine reads for `entry_time.hour` — worth a 10-minute audit but not load-bearing for the §4.4 pass-count correction.

---

## Exact Conditions for Paper-Trade Prereg (if you want to proceed despite NOT_READY)

**Proceed to paper-trade preregistration only when all of the following are satisfied:**

1. **Timezone bug fixed** in `tasks/_probe3_1h_ritual.py`, `tasks/_probe4_readout.py`, `tasks/_scope_d_readout.py`. Verification: run a sanity script that confirms (a) `session_break=True` bars fall in corrected SES_2a window (GLOBEX reopen), (b) CME halt hour is absent from RTH/SES_1 window, (c) reported RTH volume is 5-10× overnight volume (consistent with real market microstructure).

2. **Probe 3 §4.4 pass count re-verified under corrected TZ.** If the count drops to ≤7/16, Probe 3 F-count becomes 1 and branch becomes COUNCIL_RECONVENE, not PAPER_TRADE — which would terminate this paper-trade plan entirely. If it remains ≥8/16, the PAPER_TRADE branch holds but the scope must be rewritten to reflect RTH-dominant (not overnight-dominant) edge.

3. **Scope D readout regenerated under corrected TZ.** The current scope_d readout.json is invalid and should be replaced. New readout should report per-combo Sharpe bootstrap CIs (not just point estimates) on SES_1 (RTH) and SES_2a (overnight) — as you'll see, the story inverts.

4. **Probe 4 §5 row-2 interpretation updated.** Under corrected TZ, row 2 still fires, but the *meaning* of SESSION_CONFOUND changes. Decide whether "edge concentrates in RTH" is a positive feature (RTH is liquid, paper-trade is cheaper) or negative (requires narrower discipline on time-of-day entries).

5. **Full per-trade schema regenerated for combos 1298 and 664** (re-run engine with Probe 2-style output columns: friction_dollars, gross_pnl_dollars, mfe_points, mae_points, hold_bars, stop_distance_pts). Without this, no paper-trade sizing analysis is defensible for those two combos.

6. **Combo-664 friction sensitivity on CORRECTED overnight bucket re-computed.** If 664 goes net-negative at 1.5× friction (=~$9/contract RT) on its corrected overnight bucket, it must be dropped from the paper-trade basket.

7. **Effective-n acknowledged**: paper-trade prereg must state explicitly that 865 and 664 share 66.8% trade-bar overlap (66.8% of 865's trades AND 18.8% of 664's trades are on the same bars) and 0.76 PnL correlation, and therefore the "three combo confirmations" framing is reduced to "two effectively independent strategies." Any sizing or stopping rule that assumes independence must be re-derived.

8. **Pre-commit a fresh holdout window** that no probe has touched, ideally a 3-6 month fresh forward backtest on combo-865 parameters only (1298 and 664 cannot be held out because they were selected on this partition and have no surplus data). This gives the paper-trade phase a first decision point that is NOT on the reused partition.

9. **Fresh LLM Council** to adjudicate whether corrected picture + effective-n = 2 still supports PAPER_TRADE branch, or whether the branch should shift to COUNCIL_RECONVENE per the spirit of the updated evidence.

---

## Pre-Preregistration Checklist (numbers that must land in the prereg)

The paper-trade preregistration, to be audit-ready after the CRITICAL fixes land, must state:

1. **Per-combo in-sample corrected-TZ Sharpe with 95% bootstrap CI** for each proposed trading session (e.g., "combo 865, RTH 09:30-16:00 ET: Sharpe 2.64, 95% CI [+1.02, +4.41], n=126").
2. **Forward-walking stopping rule** anchored on a per-combo cumulative-P&L loss that maps to a specific quantile of the bootstrap Sharpe distribution (e.g., "stop if realized 90-day Sharpe falls below CI_lower minus 0.5, OR cumulative DD exceeds 15% of starting capital").
3. **Sizing rule** that accounts for per-trade σ asymmetry: combo 1298 per-trade std $4,734 vs combo 664 per-trade std $2,712. A uniform 5%-risk policy does NOT mean uniform $-risk; sizing must cap $-risk per combo to avoid 1298-dominance in forward variance. Explicit $-risk cap per combo.
4. **Per-combo trading session declared in wall-clock ET terms** (not "SES_2a" label) to avoid the labeling-vs-reality gap. E.g., "combo 865 trades only 09:30-15:30 ET, no entries after 15:30, position closed by 16:00 ET."
5. **Friction assumption**: state the $/contract RT assumed and cite the sensitivity curve showing Sharpe at 1.0x, 1.5x, 2.0x friction on the corrected session bucket. Commit to a falsification trigger if realized friction exceeds 1.5× the backtest assumption.
6. **Correlation caveat**: state the 66.8% trade-bar overlap between 865 and 664 explicitly and declare whether they are traded as (a) one combined strategy (union of entry signals, no double-sizing on overlap bars) or (b) two independent strategies with shared-bar size caps.
7. **Expected calendar duration to decision**: projected ±half-width on realized Sharpe at 6, 12, 18 months. Commit to a minimum paper-trade window that delivers enough data to falsify at the 95% level against a pre-specified null (e.g., "null: true Sharpe < 1.3; minimum paper-trade duration = max(6 months, whatever yields SE < 0.5 Sharpe half-width)").
8. **Multiplicity discount**: acknowledge 5th-partition-read and either (a) raise the paper-trade bar by sqrt(log(k)/n) adjustment, or (b) state explicitly that the preregistered bar is inherited from the prior probes' posterior and no further multiplicity correction is applied.
9. **Power analysis for rejection**: for each combo's corrected point Sharpe, what is the probability of correctly detecting "true Sharpe < 1.3" within 6 / 12 / 18 months if that's the truth? (Hint: for combo 664 overnight at Sharpe 1.17 point, 1-year paper with ±0.14 CI half-width will likely NOT fail at 1.3 — 664 may not fail even under a null-true scenario.)

---

## Bottom Line

The paper-trade readiness question as posed assumed the session decomposition was mechanically correct and the open question was "is the evidence strong enough?" The actual situation is that **the session decomposition itself is incorrect across Probes 3, 4, and Scope D** — and under corrected decomposition, the story flips so materially that the paper-trade scope ("trade only SES_2a overnight") is the opposite of what the data supports. The Probe 2 combo-865 PASS and Probe 4 absolute/Welch-t gates survive the bug (they're timezone-independent). The "SESSION_CONFOUND" narrative, the "3.45× overnight/RTH ratio," the "SES_2a dominates unanimously" Scope D conclusion, and the posterior-interval citation [0.65, 0.85] all require re-computation. Paper-trade preregistration at the current scope would commit you to forward-walking a window the corrected backtest does not support. **Fix the bug first. Re-run the three affected readouts. Reconvene the council on the corrected picture. Only then consider paper-trade prereg.** All effort: approximately 2-4 hours of coding + 1 council. No new engine compute required (both are pure post-hoc re-bucketing of existing per-trade parquets, except for CRITICAL-item-5 which needs an engine re-run on 1298 and 664 for full schema).
