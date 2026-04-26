# Probe 5 — Combo-865 Forward Paper-Trade Preregistration (SKELETON)

**Status:** 🚧 DRAFT (not signed)
**Date opened:** 2026-04-24
**Required by:** LLM Council verdict 2026-04-24 (`tasks/council-report-2026-04-24-independent-data.html`) — chairman's One Thing To Do First.
**Predecessors:**
- `tasks/probe1_preregistration.md` (signed `d0ee506`) — §3 tie-breaking rule named combo-865.
- `tasks/probe2_preregistration.md` (signed `a49f370`) — §4 absolute gate PASS on combo-865 1h test partition (TZ-agnostic, stands).
- `tasks/probe3_preregistration.md` (signed `f8447af` / fixes `8636167`) — 4-gate robustness, 3/4 PASS under TZ correction (§4.3 fail → COUNCIL_RECONVENE).
- `tasks/probe4_preregistration.md` (signed `432fb3d`) — second-combo carve-out, row 4 → COUNCIL_RECONVENE.
- `tasks/tz_cascade_closure.md` (signed 2026-04-24) — TZ contract harness + scope of session-decomposition retractions.

**Graceful fallback clause:** If §4 gates cannot be pinned without breaching the multiplicity-adjusted prior (i.e. the pass threshold consistent with 1/13,814 selection bias exceeds combo-865's historical Sharpe), this preregistration becomes unwriteable and the verdict collapses to **honest full-sunset including combo-865** per chairman verdict §"Graceful fallback to user's original lean".

---

## §1 Purpose and prior framing

This is a **forward paper-bar accrual test** on a frozen combo-865 rule, with binding §-gates signed before bar 1 of accrual.

**Hypothesis under test (H1):** combo-865's per-trade edge persists out-of-sample on forward NQ 1h bars at the parameter freeze defined in §2.

**Null hypothesis (H0):** combo-865's Probe 2 PASS was a 1/13,814 selection-bias artifact; forward edge is statistically indistinguishable from a randomly-selected combo from the v11 sweep.

**This test is confirmation under selection bias, not independent confirmation.** Specifically:

1. **Probe 1 §3 selection bias:** combo-865 was named by `argmax` over 1500 combos on 1h training fold. The 1/1500 selection fraction is the *direct* multiplicity prior for this test.
2. **v11 sweep multiplicity:** the broader sweep generated 13,814 combos under $5/contract RT friction; only 1 (v11_23634) clears gross Sharpe ≥ 1.0. The 1/13,814 prior is the *upstream* selection-bias horizon.
3. **100%-combo-overlap leakage (`project_combo_overlap_structural`):** combo-865 has trades in V3/V4 ML#2 training partitions; its parameters may have been implicitly weighted by ML#1 feature engineering (`combo_features_v12.parquet`) before Probe 1 §3 selection. Forward paper-accrual resolves this leakage *if* the freeze is exact and ingest is genuinely live (not Barchart batch).

**Per-bar partition contamination accounting:**
- 1h test partition (2024-10-22 → 2026-04-08, 1.48 yr, ~2,400 bars) was read 4× pre-2026-04-24 (Probes 2, 3, 4, Scope D).
- This forward test reads bars *not yet generated* at the partition boundary 2026-04-08. By construction, no overlap with prior reads.
- The 5th read concern named by the previous council (2026-04-24 post-TZ-cascade) does **not** apply here: forward bars are a fresh sample.

**Mechanism story (parallel research, not blocking):** A microstructural articulation of why combo-865's specific (z_band_k=2.30, stop_pts=17, min_rr=1.85) configuration captures a durable inefficiency on 1h NQ has *not* been produced. First Principles Thinker (council 2026-04-24) named this as the load-bearing missing piece. Documented as `tasks/combo865_microstructure_hypothesis.md` (TBD), runs concurrent with this paper-trade window. Resolution of the mechanism story does not gate this preregistration but informs §5 verdict interpretation.

---

## §1.A Analyst pre-sign assessment (audit record only — not a §-gate)

**Pre-sign verdict (2026-04-26, AI analyst):** This prereg is **NOT READY** for forward paper-trade execution as of this date, and combo-865 is **NOT READY for live deployment** at any sizing. Recorded here per user directive 2026-04-26 to capture the AI counterfactual against the user's sizing-override decision, so the audit trail preserves both perspectives. **The user has overridden this assessment for the backfill execution scope only** (running `scripts/paper_trade/backfill_combo865.py` at $2k/$50 sizing) and has NOT authorized forward paper-trade or live deployment.

**Six load-bearing blockers (in order of severity):**

1. **Probe 3 PAPER_TRADE is retracted (CLAUDE.md status banner 2026-04-23).** F-count went 0→1 under the TZ-corrected reading, collapsing the branch to COUNCIL_RECONVENE per Probe 3 §5.2. There is no current outstanding authorization to deploy combo-865 anywhere — paper or live.

2. **This prereg is unsigned.** §2 (full parameter dict + hash), §3.1 (broker), §3.2 (live ingest vendor), §A.1 (window), §A.2 / §A.3 (PASS / KILL thresholds) are all `TODO` as of 2026-04-26. Per §6.2: *"Accrual ingest may not begin before the signing commit is in master."* The 2026-04-26 backfill is NOT accrual ingest — it is historical replay against a frozen parquet, explicitly framed as a continuity bridge in `metadata.json:51` (`"NOT_SIGNED — these numbers do NOT count toward Probe 5 §-gates"`).

3. **The new $2k sizing makes §A.3.1 KILL gate already-near-tripped on historical data — a structural-unsignability finding.** $600 absolute trigger vs $519 realized historical DD (87% of trigger); MC p95 DD $556 (92%); MC p99 DD $708 (already past). Even under H1 (historical edge persists exactly), 5–8% of empirical-distribution trade orderings trip the kill gate before completing the §A.1 minimum window. Either §A.3.1 or §3.3 must be re-tuned before signing; the backfill numbers are sufficient evidence that the current pairing as drafted is structurally unsignable. Critically: §6.3 forbids retroactive gate softening, so the re-tune must happen *before* the signing commit.

4. **Friction overhead is 44.7% of starting equity per year at $2k sizing.** $6/contract RT × 1 contract × 149 trades/yr ≈ $894/yr against $2,000 starting. The strategy must out-earn this drag *before* edge appears, and *before* selection-bias discounting. At the project-wide $50k/14-contract reference the same friction-tax-percent (17.6%) was 1.8% of starting equity per year — nearly invisible. The small-account configuration amplifies friction in absolute-equity terms by ~25× without changing the underlying per-trade economics. This is not a concern at the $50k/$500 reference; it becomes load-bearing at $2k/$50.

5. **No broker integration, no live ingest pipeline, no mechanism story.** §3.1 broker = TBD; §3.2 vendor = TBD; First Principles Thinker mechanism story (council 2026-04-24) = open. The 2026-04-26 backfill reads from `data/NQ_1h.parquet` (Barchart batch), not a live feed; there is no code path that places real orders. Going-forward live deployment requires all three to be resolved.

6. **Selection bias is unaddressed by re-sizing.** The whole prereg framework (per §1) exists because combo-865 was named via `argmax` over 1500 combos (Probe 1 §3 direct selection) embedded in a 13,814-combo v11 sweep (`memory/project_friction_constant_unvalidated.md`), and has 100% trade overlap with V3/V4 ML#2 training partitions (`memory/project_combo_overlap_structural.md`). Sizing changes per-trade $; it does nothing to the multiplicity priors. Forward-bar evidence under signed gates is the only mechanism the project has accepted as resolving these.

**MC ruin probability of 0.04% does not certify live readiness.** It is an IID bootstrap on the *historical* per-trade pnl distribution — answers the question *"if the forward edge equals the in-sample edge, what's ruin probability under random trade ordering?"* — NOT *"will the forward edge equal the in-sample edge?"* This prereg framework exists because the latter question is unresolved. Conflating the two would amount to using the in-sample edge as evidence of itself.

**Concrete recommendation, ordered:**

a. Re-tune §A.3 KILL thresholds in lockstep with §3.3 forward-window sizing (this commit's edits flag the issue and propose -$3,500/yr at small-account sizing for §A.3.3, but do not pin a final number for §A.3.1 — that is the user's call before signing).
b. Resolve §2 parameter freeze TODOs (full v11 row + dict SHA-256 hash).
c. Resolve §3.1 broker / §3.2 vendor TODOs.
d. Sign per §6.2 with engine commit hash + combo param hash + frozen `src/tz_contract.py` revision.
e. Run forward window with all four §A.2 PASS gates clearing AND no §A.3 KILL gate firing across the full §A.1 window.
f. Convene LLM Council per §6.4 PASS branch on live-trading scope (broker, sizing, kill-switch authority for live).

**Audit traceability:** AI analyst exchange with user 2026-04-26 (chat session). User question: *"Is the $2k/$50 ready for live deployment?"* AI analyst response: enumerated the 6 blockers above and answered *"No — not by a wide margin, and the new sizing arguably makes the gap larger rather than smaller."* User decision recorded: *"My decision to paper trade backfill overrides, but mention why you chose not to."* Interpretation: backfill execution at $2k/$50 authorized; forward paper-trade and live deployment NOT authorized; AI counterfactual recorded as this §1.A.

---

## §2 Combo-865 parameter freeze

All parameters lifted from the v11 sweep canonical source.

**Sweep source:** `data/ml/originals/ml_dataset_v11.parquet` WHERE `combo_id = 865`. Sanity-check sources: `data/ml/probe3/param_nbhd.json` row `combo_id=10013` (1.0/1.0/1.0 perturbation cell) + `data/ml/probe2/combo865_combos.json`.

**Surfaced parameters (from `data/ml/probe3/param_nbhd.json:10013`):**
- `z_band_k = 2.3043838978381697`
- `stop_fixed_pts = 17.017749351216196`
- `min_rr = 1.8476511046163293`

**TODO (sign-off blocker):** Lift the full v11 combo-865 row including all of:
- `signal_mode` (likely `zscore_reversal` per V3 default)
- `z_score_formulation` (input / anchor / denom / type — v11 uses `close / rolling_mean / rolling_std / parametric` per CLAUDE.md v11 spec)
- `ema_fast`, `ema_slow` (regime filter)
- `swing_lookback` OR `atr_multiplier` (whichever is non-null per v11 schema)
- `session_filter_mode` (must verify ET/CT semantics per `tasks/tz_cascade_closure.md` §3.C — engine default is CT-by-design)
- `entry_timing_offset`, `fill_slippage_ticks`, `cooldown_after_exit_bars` (v11 microstructure axes)
- `volume_confirmation_*` flags
- Any exit-rule flags (tp_first / sl_first / maxhold)

The full parameter dict must be embedded verbatim as a JSON block here, with a hash of the dict committed alongside the prereg signing commit. **No live tuning. No session sub-rule additions. No EX swaps mid-window.** Any deviation requires a signed amendment to this prereg.

---

## §3 Block B — Operations

### 3.1 Broker / paper-trade platform

**TODO[USER CHOICE]:** select one. Must support NQ futures simulated execution + programmatic API access.

| Candidate | Pros | Cons |
|---|---|---|
| **IBKR Paper (Trader Workstation simulated)** | Programmatic API (ib_insync), real-time NQ market data co-located, fills via IBKR-realistic queue | Requires IBKR account; market-data subscription cost |
| **Tradovate Demo** | Free, futures-focused, REST + WebSocket API, NQ-native | API less mature than IBKR; fill realism documented less rigorously |
| **NinjaTrader 8 Simulator** | Free sim, good UI for monitoring, large user base | Programmatic execution requires NinjaScript C# integration; less Pythonic |

**Recommendation:** IBKR Paper as default — best programmatic API alignment with project's existing Python toolchain.

### 3.2 Bar source (live ingest)

**Requirement:** real-time NQ 1h bar stream consumed in time-aligned manner. **Cannot use** `data/NQ_1h.parquet` (Barchart batch source — latency-incompatible).

**TODO[USER CHOICE]:** select one.

| Candidate | Pros | Cons |
|---|---|---|
| **IBKR market data** | If using IBKR Paper, single integration | Subscription cost (~$10-25/mo for futures bundle) |
| **Polygon.io** | Polygon REST + WebSocket; well-documented | $79/mo Stocks Starter does not include futures; futures tier ~$200+/mo |
| **Databento** | High-quality CME direct feed | Paid; overkill for 1h bars |

**Source-TZ contract:** ALL bars must be passed through `src.tz_contract.assert_naive_ct` at load time, then `src.tz_contract.localize_ct_to_et` for any ET-conditional logic. Live vendor feeds typically deliver UTC-aware timestamps; the contract must handle vendor-tz → CT-naive normalization explicitly. Vendor-TZ assumption documented per `tasks/tz_cascade_closure.md` §5.A.

**Vendor consistency check:** the live vendor's reported NQ 1h bars must align (bar boundaries, OHLCV semantics) with `data/NQ_1h.parquet` historical bars. A 30-bar overlap probe at accrual start (re-fetch the last 30 historical bars from the live vendor and diff vs Barchart cache) is required as a sanity check; >2% bar-by-bar OHLC divergence aborts the run.

### 3.3 Sizing policy

**Backfill scope (user directive 2026-04-26 — applied):** Fixed **$50 risk per trade** on **$2,000 starting paper equity** — small-account OOS-scoped override of CLAUDE.md project-wide $500 / $50,000 defaults. Implemented at `scripts/paper_trade/backfill_combo865.py:115-122` and propagated to `evaluation/probe5_combo865_backfill/` artifacts (regenerated 2026-04-26 via `jupyter nbconvert --execute`).

Contract math at this sizing: `int(50 // ($2/pt × 17.0177pt)) = int(50 // 34.04) = 1 contract/trade`, realized risk per trade **$34.04** (the $15.96 residual is intentionally unallocated — rounding up to 2 contracts would put 3.4% of a $2k account at risk per trade, a different risk regime). Friction at `1 × $6 RT = $6/trade ≈ 17.6% of risk` (same friction-tax-percent as the $50k/$500 reference, because the integer floor cuts both numerator and denominator). Annualized friction at combo-865's ~149 trades/yr ≈ **$894/yr = 44.7% of starting equity** — the strategy must out-earn this drag before edge appears in the equity curve.

**Forward-window sizing (TBD — sign-off blocker, coupled to §A.3 retuning):** the §A.3 KILL gates below were originally drafted at the $50k/$500 reference where the 30%-DD trigger ($15,000 absolute) was structurally unreachable by the historical trade distribution. At $2k/$50, the same 30% trigger = **$600 absolute**, against historical realized peak DD = **$519 (87% of trigger)** and MC p99 DD = **$708 (already past trigger)**. The forward-window sizing block must be re-pinned in lockstep with §A.3 thresholds before the prereg can be signed; do not assume backfill-scope sizing carries forward to live ingest without explicit decision. See §1.A analyst pre-sign assessment for the structural-unsignability framing of this issue.

**Reference reproduction (the prior $50k/$500 sizing, pre-2026-04-26):** Risk-per-trade = stop distance (in MNQ $/point × points) × contracts → contracts = $500 / ($2/pt × 17.0177pt) ≈ 14.7 contracts → round to 14 (floor). Recorded here for traceability against `tasks/probe2_verdict.md` (signed `a49f370`) and `tasks/probe3_verdict.md` numerics.

### 3.4 Execution model

- **Fill:** next-bar open per `FILL_MODEL = "next_bar_open"`.
- **Slippage assumption:** $5/contract round-trip (consistent with v11 sweep `COST_PER_CONTRACT_RT`). For 14 contracts → $70/trade RT friction. Forward results compared net-of-this-friction.
- **Order types:** market on next-bar open for entry; stop-loss + take-profit limit orders bracketed at fill.
- **Same-bar collision rule:** `SAME_BAR_COLLISION = "tp_first"` per CLAUDE.md.

### 3.5 Position management

- **One position at a time** per CLAUDE.md.
- No pyramiding, no scaling in.
- Cooldown after exit per `cooldown_after_exit_bars` from §2 freeze.

---

## §4 Block A — Methodology gates (TODOs with suggestions)

**These are sign-off-blocking decisions.** Each must be pinned with explicit numbers before commit.

### A.1 Accrual window

**TODO[USER]:** specify min and max bars OR min and max wall-clock weeks.

**SUGGESTION:** **24 weeks min / 32 weeks max.** Rationale:
- Combo-865 historical frequency: 220 trades / 1.48 yr ≈ **149 trades/yr** equilibrium.
- 24 weeks ≈ 0.46 yr ≈ **~69 expected trades**.
- 32 weeks ≈ 0.62 yr ≈ **~92 expected trades**.
- 12 weeks (council's initial suggestion) yields ~35 trades, which is power-underspec'd for any honest Sharpe-1.3-vs-Sharpe-0 detection (rough rule: ~80 trades to detect per-trade Sharpe 0.236 — combo-865's historical equivalent — at 80% power).
- Trade-off: 32 weeks delays go/no-go decision by ~5 months relative to 12 weeks but produces a much firmer signal.

**Alternative:** 16 weeks min / 24 weeks max, paired with softer gates (Sharpe ≥ 1.5, n ≥ 30) and explicit "directional" framing. **Higher retraction risk.** Not recommended unless the 32-week window is judged operationally unacceptable.

### A.2 PASS gates (conjunction — ALL must hold)

**TODO[USER]:** specify each threshold.

**SUGGESTION:**
1. **Net Sharpe ≥ 1.3** — matches Probe 2/3/4 floor, consistent with v11 friction model; this is the project-wide "tradable" threshold.
2. **n_trades ≥ 50** — matches Probe 3 §4.4 threshold; below this, Sharpe SE balloons.
3. **Welch t-stat (per-trade $ vs zero) ≥ 2.0** — matches Probe 4 §4.3 Welch-t gate; corrects for non-normality of per-trade pnl.
4. **Net dollars/year (annualized over actual accrual span) ≥ $5,000** — matches Probe 3 §4.4 dollar floor; rules out edge that exists in Sharpe but not in dollars.

ALL four must pass simultaneously for §5 PASS verdict.

### A.3 KILL gates (disjunction — ANY one triggers unilateral abandonment)

**TODO[USER]:** specify each threshold.

**SUGGESTION:**
1. **Maximum drawdown ≥ 30% of starting paper equity** = ≥ $15,000 absolute drawdown on $50k starting. Triggers immediate kill regardless of trade count or Sharpe at that point. **⚠️ Sizing-coupled retuning required for $2k backfill scope (per §3.3 amendment 2026-04-26):** at $2k starting, the 30% trigger = **$600 absolute**, against historical realized peak DD = $519 (87% of trigger), MC p95 DD = $556 (92%), and MC p99 DD = $708 (already past). Under H1 (historical edge persists exactly), 5–8% of empirical-distribution trade orderings trip the kill gate before completing the §A.1 minimum window. Either the threshold percent or the starting equity must be re-pinned before this gate can be set on the forward window. **Inadmissible to soften the gate retroactively post-sign per §6.3** — re-tuning happens before signing or not at all.
2. **n_trades < 5 by week 12 of accrual** — zero-activity kill. If combo-865 is firing at < 40% of its expected ~3 trades/week rate halfway through the minimum window, the regime has changed and the test is uninformative.
3. **Net dollars/year (annualized over actual accrual span) ≤ -$50,000** — catastrophic-underperformance kill. Ranges that look like Probe 3 §4.3 SES_2 disasters (-$140k/yr Sharpe -3) trigger early termination instead of riding to time budget. **⚠️ Sizing-coupled retuning:** at backfill-scope $2k/$50 (1 contract/trade), per-trade $ scales to ~1/14× of the $50k/$500 reference, so the original -$50k/yr threshold is unreachable in absolute dollars; suggested forward retune ≈ **-$3,500/yr** at small-account sizing (preserving the same percent-of-equity bite). Re-pin in lockstep with §A.3.1 before signing.
4. **Time budget exceeded** = max of (32 weeks, A.1 max). Default abandonment if no PASS verdict has fired.

ANY ONE triggers kill. Kill is unilateral and requires no further authorization (per §6.3).

### A.4 Multiplicity-adjusted prior — framing in §1

**TODO[USER]:** approve the §1 framing language (above) or amend.

**SUGGESTION:** the language in §1 is the council's recommended framing. Specifically:
- Frame as **confirmation under selection bias**, not independent confirmation.
- Acknowledge 1/1500 (Probe 1 §3 direct selection) and 1/13,814 (v11 sweep horizon).
- Acknowledge 100%-combo-overlap leakage from `project_combo_overlap_structural`.
- State explicitly that PASS does not equal "edge proven" but "edge survives selection-bias-aware forward sample."
- State explicitly that FAIL or KILL does equal "edge does not persist out-of-sample" — i.e. the test has more falsification power than confirmation power. This asymmetry is intentional.

### A.5 Power calc statement (additive — recommended for §1)

**TODO[USER]:** include a power-calc paragraph in §1 with the chosen window's expected statistical power.

**SUGGESTION:** add the following paragraph to §1 once A.1 is pinned:

> *Assuming the historical Sharpe 2.89 holds (per-trade Sharpe ≈ 0.236), the chosen accrual window of [N] weeks yields ~[K] expected trades. At n=[K], the standard error of the Sharpe estimator is approximately √(1+0.236²·N/2)/√n ≈ [value]. The Sharpe-1.3 PASS threshold sits ~[Z]σ below the expected mean under H1, giving ~[power]% power to clear the gate. Under H0 (true Sharpe 0), observed Sharpe ≥ 1.3 occurs with probability ~[α]. Multiplicity-adjusted false-positive rate (Bonferroni over 1500-combo pool) is ~[α × 1500].*

Numbers to be computed from A.1 final values.

---

## §5 Branching / verdict logic

| Condition | Branch | Action |
|---|---|---|
| ALL §4 PASS gates clear AND no §4 KILL gate fired | **PASS** | Output: `data/ml/probe5/readout.json` with `branch="PASS"`. Triggers council-convene on next steps (live trading scope; sizing policy beyond paper; broker/account decision). |
| ANY §4 KILL gate fires (regardless of accrual progress) | **KILL** | Output: `data/ml/probe5/readout.json` with `branch="KILL"`. Combo-865 forward-edge null hypothesis NOT rejected; sunset including 865. |
| Time budget reached without §4 PASS gates clearing AND no KILL fired | **NARROW_MISS** | Output: `data/ml/probe5/readout.json` with `branch="NARROW_MISS"`. Council-convene on inconclusive evidence: optionally extend accrual via signed amendment, or accept as null result. |

**No room for a PAPER_TRADE_EXTEND or AMEND_GATES branch.** Any extension requires a signed amendment to this prereg. Any gate softening is structurally retroactive and inadmissible.

---

## §6 Block C — Procedure

### 6.1 Code freeze

At sign-off, the following must be pinned:
- **Engine commit:** `git rev-parse HEAD` at sign-off → recorded in `frozen_commit` field of §8.
- **`src/tz_contract.py` revision:** must be at or downstream of commit `15f450d` (TZ contract harness committed 2026-04-24).
- **Combo-865 parameter dict:** SHA-256 hash of the JSON dict from §2 → recorded in `combo_param_hash` field.
- **Bar-source vendor + endpoint:** explicit vendor name + API endpoint URL → recorded in `bar_source` field.
- **Broker + paper account ID:** explicit broker name + paper account identifier → recorded in `broker` field.

No code changes to engine, indicators, scoring, risk, or backtest paths between sign-off and §5 verdict fire. Telemetry/logging-only changes permitted; must be committed to a separate branch and not merged into the engine path.

### 6.2 Signing protocol

Mirror the pattern of `a49f370` (Probe 2) and `d0ee506` (Probe 1):
1. User reviews this document (post-Block-A fill).
2. User commits this file with sign-off message: `"Sign Probe 5 combo-865 paper-trade preregistration"`.
3. The signing commit hash becomes `signed_commit` in §8.
4. Accrual ingest may not begin before the signing commit is in `master`.

### 6.3 Authority and amendment

- **Kill-switch authority:** §4 KILL gates fire automatically and unilaterally upon condition match, with no human-in-the-loop override. Authority to silence a KILL trigger requires a signed amendment to this prereg, committed before the trigger fires (impossible by construction; this is the point).
- **Time-budget extension:** requires signed amendment with explicit new max date and rationale. Cannot be retroactive (i.e. cannot be signed after the original time budget expires).
- **Gate amendment:** any change to §4 PASS or KILL thresholds is structurally inadmissible mid-window. New thresholds = new prereg + new accrual window.

### 6.4 Result publication

On §5 verdict fire:
1. Write `data/ml/probe5/readout.json` with full per-trade trade list, gate-by-gate evaluation, branch, and timestamps.
2. Append `tasks/probe5_verdict.md` with a §-by-§ summary.
3. Commit both with message `"Probe 5 [BRANCH] verdict — combo-865 forward paper-trade"`.
4. If branch=PASS, fire LLM Council on live-trading scope (broker, sizing, kill-switch authority for live).
5. If branch=KILL or NARROW_MISS, fire LLM Council on next-fork (full sunset vs alternative scope).

---

## §7 References

- `tasks/probe1_preregistration.md`, `tasks/probe1_verdict.md`
- `tasks/probe2_preregistration.md`, `tasks/probe2_verdict.md`
- `tasks/probe3_preregistration.md`, `tasks/probe3_verdict.md` (with Amendment 2 retraction)
- `tasks/probe4_preregistration.md`, `tasks/probe4_verdict.md` (with retraction)
- `tasks/scope_d_brief.md` (with Amendment retraction)
- `tasks/tz_cascade_closure.md` (signed 2026-04-24)
- `tasks/council-report-2026-04-24-independent-data.html` (chairman verdict)
- `tasks/council-transcript-2026-04-24-independent-data.md` (full deliberation)
- `CLAUDE.md` §Non-negotiable constraints, §Performance requirements, §Reporting & styling requirements
- `data/ml/originals/ml_dataset_v11.parquet` (combo-865 canonical parameter source)
- `src/tz_contract.py` (source-TZ validation harness)

---

## §8 Sign-off

| Field | Value |
|---|---|
| `frozen_commit` | TBD (set at sign-off) |
| `combo_param_hash` | TBD (SHA-256 of §2 JSON dict, set at sign-off) |
| `bar_source` | TBD (vendor + endpoint, per §6.1) |
| `broker` | TBD (broker + paper account ID, per §6.1) |
| `accrual_window_min_weeks` | TBD (§A.1) |
| `accrual_window_max_weeks` | TBD (§A.1) |
| `pass_sharpe_floor` | TBD (§A.2) |
| `pass_n_trades_floor` | TBD (§A.2) |
| `pass_t_stat_floor` | TBD (§A.2) |
| `pass_dollars_per_year_floor` | TBD (§A.2) |
| `kill_max_dd_pct` | TBD (§A.3) |
| `kill_n_trades_floor_at_week12` | TBD (§A.3) |
| `kill_dollars_per_year_floor` | TBD (§A.3) |
| `signed_commit` | TBD (set at signing per §6.2) |
| `signed_date` | TBD |
| `signed_by` | TBD (user) |

---

*Document opened 2026-04-24 as DRAFT. Fields marked TBD must be resolved before signing. Signing commit ≠ this commit; this commit is the unsigned skeleton.*
