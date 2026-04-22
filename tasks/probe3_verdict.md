# Probe 3 Verdict — Combo-865 4-Gate Robustness PASS on 1h Holdout

**Date**: 2026-04-22 UTC
**Probe**: Combo-865 session/exit structure + robustness gates on 1h test partition (2024-10-22 → 2026-04-08)
**Preregistration**: `tasks/probe3_preregistration.md` (signed commit `f8447af` / review fixes `8636167`)
**Readout script**: `tasks/_probe3_readout.py`
**Machine record**: `data/ml/probe3/readout.json`
**Remote launcher**: `tasks/_run_probe3_remote.py`

---

## Bottom line

**F = 0. All four gates PASS. Branch = PAPER_TRADE per preregistration §5.**

Posterior P(genuine edge | F=0) = **0.91** (prior 0.167; BF ≈ 54× in favor of H1
per `tasks/probe3_multiplicity_memo.md` §6.4).

Mechanically-applied gates:

| Gate | Type | Threshold | Observed | Margin | Status |
|---|---|---|---:|---:|:---:|
| §4.1 regime halves | two-half split @ 2025-07-15 | both halves net Sharpe ≥ 1.3, n ≥ 25, $/yr ≥ $5k | H1: 3.235/102/$136k · H2: 2.580/118/$114k | +1.28 / +1.94 on Sharpe | **PASS** |
| §4.2 param ±5% neighborhood | 27-combo 3-axis grid | ≥ 14 / 27 cells clear all three sub-gates | **27 / 27** cells pass | +13 cells over threshold | **PASS** |
| §4.3 15m negative control | 16 session×exit cells on 15m | ≤ 2 / 16 cells pass | **0 / 16** cells pass | +2 below fail-ceiling | **PASS** |
| §4.4 1h session/exit ritual | 16 session×exit cells on 1h | ≥ 8 / 16 cells pass | **8 / 16** cells pass | exactly at threshold | **PASS** |

All four PASS flags sourced directly from `data/ml/probe3/readout.json`. Verdict
applied by `tasks/_probe3_readout.py` §5 branch routing: F=0 → PAPER_TRADE.

---

## §4.1 — Regime halves (temporal stability)

Test-partition cut into two halves at **2025-07-15 00:00 UTC** (median bar of test
window). Both halves clear all three per-half sub-gates independently.

| Metric | H1 (pre-2025-07-15) | H2 (post-2025-07-15) | Threshold |
|---|---:|---:|---:|
| `n_bars` | 4,079 | 4,301 | — |
| `years_span` | 0.720 | 0.760 | — |
| `n_trades` | **102** | **118** | ≥ 25 per half |
| `net_sharpe` | **3.235** | **2.580** | ≥ 1.3 per half |
| `net_dollars_per_year` | **+$136,198** | **+$114,178** | ≥ $5,000 per half |
| `mean_pnl` / trade | +$961.86 | +$734.95 | — |
| `std_pnl` / trade | $3,537.97 | $3,550.67 | — |

H1 is stronger than H2 (~25% higher mean per trade), but both clear comfortably.
No "edge concentrated in one half" signature — the chronological halving does
not reveal regime-dependent collapse.

**Aggregate test vs halves cross-check**: 220 full-test trades (Probe 2) = 102 + 118 ✓.
Full-test net Sharpe 2.89 sits between the two half-Sharpes (3.235 H1, 2.580 H2)
as expected under standard variance math, no anomaly.

---

## §4.2 — Parameter ±5% neighborhood (parameter stability)

27-combo grid varying three orthogonal parameters by {×0.95, ×1.00, ×1.05}:
- `z_band_k` (entry band)
- `stop_fixed_pts` (stop distance)
- `min_rr` (minimum R:R gate)

All 23 other parameters frozen at the Probe 2 1h manifest.

Observed: **27 / 27 cells pass** (threshold was ≥ 14 / 27 per memo §3 binomial
calibration). The lowest cell (`combo 10025`: z_band_k=×1.05, stop=×1.05,
min_rr=×1.00) still clears net Sharpe **1.988** on 193 trades — above the 1.3
gate with margin. The highest cell (`combo 10013`: the exact frozen baseline)
hits Sharpe **2.895** on 220 trades (matches Probe 2 1h Sharpe 2.89 to 2 d.p. ✓).

Observed envelope:
- `net_sharpe` ∈ [1.988, 2.895] · mean ≈ 2.485
- `n_trades` ∈ {193, 220, 237} — determined entirely by `z_band_k` level (tighter
  band → more trades; frozen band → 220; wider band → 193)
- `net_dollars_per_year` ∈ [$79,932, $125,065]

The `n_trades` structure deserves a note: `z_band_k` is the **only** axis here
that changes entry count (because `stop_fixed_pts` and `min_rr` affect exits, not
entries, and the base combo already filters to a fixed entry rule). So within
each of the three `z_band_k` slabs, `n_trades` is constant across the 9 cells.

**Interpretation**: Combo 865's edge does not sit on a parameter knife-edge at
1h. All neighbors within ±5% of all three most-sensitive parameters deliver a
tradeable Sharpe. This is the classic binomial overkill profile predicted by
memo §3 (at H1 = Sharpe 1.8 baseline, ≥ 14 / 27 was ~97% power).

---

## §4.3 — 15m negative control (TF boundary clarity)

Full 4 × 4 session × exit grid on the **15m** test partition. Threshold: "NO MORE
THAN 2 / 16 cells clear the three sub-gates" — framed as a negative control to
confirm that session/exit manipulation cannot rescue 15m.

Observed: **0 / 16 cells pass**. Cleanest possible outcome.

| Session \ Exit | EX_0 native | EX_1 maxhold=60h | EX_2 TOD_1500ET | EX_3 breakeven_1R |
|---|---:|---:|---:|---:|
| SES_0 all | −0.552 | −0.552 | −0.567 | −0.284 |
| SES_1 RTH | −0.777 | −0.777 | −0.842 | **−1.114** |
| SES_2 overnight | −0.314 | −0.314 | −0.314 | +0.059 |
| SES_3 RTH−lunch | +0.464 | +0.464 | +0.314 | +0.101 |

(values = net Sharpe; all fail sharpe_pass regardless of any other sub-gate)

Two structural observations worth naming explicitly:

1. **EX_0 ≡ EX_1 on 15m is a fixture**, not a finding. `EX_1` caps `max_hold_bars`
   at 240 (= 60h × 4 bars/h on 15m); `EX_0` uses the combo's native 120 bars (= 30h).
   In practice EX_1's cap of 240 is *less* binding than EX_0's 120 — so identity
   arises because neither cap fires in this window. Read the EX_1 column as
   "same trades as EX_0 with a longer never-hit ceiling."
2. **EX_3 amplifies sign**, both ways. On 15m SES_1 RTH, EX_3 breakeven-after-1R
   drives Sharpe from −0.777 to **−1.114** — breakeven exit bleeds when the
   underlying edge is negative. This is the mirror of its +45–48% upside on 1h
   (§4.4 below). It's not a universal upgrade.

**Mechanical interpretation**: 15m's failure is fundamental at the friction level
(Probe 2 analysis: $438/trade mean friction vs $355/trade mean gross edge). No
session filter or exit rule inside the preregistered grid can rescue
sub-friction per-trade edge. Combo 865 @ 15m is a training-partition artifact,
confirmed independently here for the second time.

---

## §4.4 — 1h session/exit ritual (the main gate for deployment-config discovery)

Full 4 × 4 session × exit grid on the **1h** test partition. Threshold:
"AT LEAST 8 / 16 cells clear the three sub-gates." This is the discovery gate —
passing here means "the edge survives session/exit manipulation AND at least
one session-structure config meets the floor."

Observed: **8 / 16 cells pass — exactly at threshold**.

| Session \ Exit | EX_0 native | EX_1 maxhold=60h | EX_2 TOD_1500ET | EX_3 breakeven_1R |
|---|---:|---:|---:|---:|
| SES_0 all (220 tr) | **2.895** ✓ | **2.895** ✓ | **2.913** ✓ | **4.207** ✓ |
| SES_1 RTH (65 tr) | 0.639 | 0.639 | 0.660 | 0.781 |
| SES_2 overnight (143 tr) | **3.322** ✓ | **3.322** ✓ | **3.322** ✓ | **4.945** ✓ |
| SES_3 RTH−lunch (36 tr) | 0.035 | 0.035 | −0.129 | 0.365 |

(values = net Sharpe; ✓ = cell_pass with all three sub-gates cleared)

The 8 / 8 split has clean structure: **every cell in SES_0 and SES_2 passes, every
cell in SES_1 and SES_3 fails**. The session axis cleanly partitions the grid;
the exit axis does not.

Why the RTH failures happen:
- **SES_3 RTH−lunch** fails on **n_trades** (36 < 50 gate) in every exit variant.
- **SES_1 RTH** fails on **Sharpe** (~0.64–0.78 across exits) — the RTH slice
  delivers 65 trades but the per-trade edge drops below the 1.3 floor. Per-trade
  gross edge is ~$350–$400 lower than the overnight slice.

Why the passes happen:
- **SES_2 overnight-only** (143 trades, net Sharpe 3.322 native / 4.945 EX_3) is
  stronger than SES_0 all-sessions (220 trades, 2.895 native / 4.207 EX_3) in
  Sharpe, modestly weaker in absolute $/yr. The per-trade edge is larger
  overnight ($1,592 mean native) than in RTH ($232 mean native).
- **EX_3 breakeven-after-1R is the strongest exit** in every session, with a
  +45% lift over native on SES_0 and +49% on SES_2. On PASS cells, EX_3 gains
  are structural (it converts early-runner partials into locked gains when price
  retraces), not statistical noise.

**Concentration-of-edge reading**: The 8 / 16 margin looks tight on paper but
isn't a fragility signature. The 8 failures are 4 RTH-Sharpe-fails and 4
RTH−lunch-n_trade-fails — every RTH subset fails for **a single consistent
reason** (undersized sample or undersized per-trade edge), not for scattershot
reasons. Combo 865's 1h edge is a **GLOBEX-overnight phenomenon**, not an
all-sessions phenomenon that happens to pass on average.

This is a deployment-config signal: if paper-trading through a broker window
that excludes GLOBEX (many US retail accounts restrict overnight futures hours),
the edge will collapse. Broker/venue selection becomes non-trivial.

---

## §5 Branch routing (mechanical)

Per `tasks/probe3_readout.py` §5 table (memo §6 authority):

| F | Branch | Posterior | Next action |
|---|---|---:|---|
| 0 | **PAPER_TRADE** | **0.91** | Fresh LLM Council on paper-trade scope, then preregister paper-trade plan, then sign |
| 1 | COUNCIL_RECONVENE | 0.038–0.10 | Council diagnoses which gate failed and why |
| ≥ 2 | SUNSET_OPTION_Z | ≤ 0.01 | Stand down; carve-out retired |

**F = 0 fires. Branch = PAPER_TRADE.**

The Bayes factor ≈ 54× is strong but not "absolute." What raises confidence
substantially above the prior 0.167:
- H1 predicted §4.1, §4.2, §4.4 to clear with high probability (regime stability
  and parameter neighborhood were the strongest H1-aligned priors per memo §6.4).
- H0 (training artifact) predicted §4.3 should sometimes fire by chance, §4.4
  would flip between 4 / 16 and 6 / 16, and §4.1 H2-half could slip below 1.3.
  Nothing in the observed data supports the H0 trajectory.
- The §4.4 "exactly-at-threshold 8 / 16" point could have been a warning sign if
  the failures were scattered; their tight concentration in the two RTH session
  modes makes them structurally lawful and strengthens the H1 reading.

---

## Relationship to Probe 1 Branch A and Probe 2

**Probe 1 (family-level sunset on bar-timeframe axis) remains binding.**
Probe 3 tested a **single combo's robustness at its validated timeframe (1h)** —
not the family, not a new timeframe, not a new instrument. §7.6 of Probe 1
(terminal on bar-timeframe axis) is unaffected.

**Probe 2 (combo-865 1h PASS) is confirmed and deepened.**
- Probe 2 verified: combo 865 at 1h passes a single Sharpe/trades/$/yr gate on
  the held-out window. It did **not** distinguish "genuine edge" from "lucky
  window."
- Probe 3 verified: the edge holds in **both halves of that window**, holds
  under **±5% perturbations of the three most-sensitive parameters**, and holds
  at **a specific session-structure signature** (overnight-concentrated).
- Probe 3 adds: a **deployment-config finding** — EX_3 breakeven-after-1R
  amplifies edge by ~45% when edge exists; SES_2 overnight-only is the
  cleanest slice; RTH-only is not a viable venue for combo 865.

---

## Irrevocable commitments honored (per preregistration §6)

1. **No mid-flight gate edits.** All four gate thresholds (1.3 / 50 / $5k;
   14/27; 2/16; 8/16) applied as written at signing commit `8636167`. No
   relaxation, no tightening.
2. **No early-stop inspection.** The full 4-gate suite ran to completion in
   ~60s on sweep-runner-1 before any readout was read. Readout was written
   atomically by `_probe3_readout.py` after all four gate JSONs landed.
3. **Results read mechanically.** F-count computed by summing `gate_pass == false`
   boolean flags. Branch routing looked up from §5 table. No subjective
   re-interpretation.
4. **No post-hoc methodology shift.** The EX_3 upside finding is reported as
   an observation, not promoted from "audit" to "primary metric."
5. **Scope lock stands.** Probe 3 tested combo 865 at 1h only. No spillover
   attempts at other combos, other timeframes, other instruments.
6. **Council precondition for next phase honored.** Paper-trade phase (E1) will
   not be signed without a fresh LLM Council per §5.6 and per user directive.

---

## §4.4 tightness disclosure

The §4.4 pass is **exactly at threshold** (8 / 16; threshold was ≥ 8). Per
disclosure discipline: this is not tighter than the preregistered gate
expected — memo §3 binomial calibration put the H1 expected pass rate around
10 / 16 at Sharpe 1.8 baseline, giving the gate ~90% power. Observing 8 / 16
sits at the lower end of the H1 predictive interval.

Interpretation: the clean SES_0 + SES_2 vs SES_1 + SES_3 partition explains
this tightness structurally (8 RTH-session-cells all fail for session-structure
reasons rather than "edge weakness" reasons), but a reader should note that
§4.4 did not pass with wide margin. Paper-trade scope decisions should respect
this — deployment configurations that exclude overnight hours would likely have
pushed §4.4 to FAIL under the same preregistration.

---

## Known limitations (transparency for Phase E1 council)

1. **EX_2 TOD_1500ET is a UTC-hour approximation.** Preregistration §2.4
   specifies "tod_exit_hour=15" with intent "15:00 ET". Engine's tod_exit_hour
   is UTC hour-of-day. Applied `tod_exit_hour=19 UTC` = 15:00 EDT exactly /
   14:00 EST (1h early). Test window is ~53% EDT / 47% EST, so the EX_2 cells
   approximate 15:00 ET correctly for half the window and close trades 1h
   early for the other half. Both effects are small (EX_2 passes where EX_0
   passes, with near-identical Sharpe and $/yr); flagging the deviation for
   auditability. No post-hoc edit — this was noted in the gate JSON
   `ex2_approximation_note` fields.
2. **Test-window sample size is small.** 1h test partition has 220 total
   trades. §4.1 half-splits yield 102 + 118; §4.4 sub-session cells yield
   36–220. At these sample sizes, per-cell Sharpe confidence intervals are
   wide — expect post-deployment Sharpe to regress from 2.89 toward something
   like [1.0, 3.5] under typical sample-size decay.
3. **§4.2 n_trades clustering.** 27 / 27 cells passing is overkill vs
   threshold of 14. But the 27 cells cluster on 3 distinct `n_trades` values
   (193, 220, 237) driven entirely by `z_band_k` level. Statistical
   independence across the 27 is limited. This was preregistered and
   binomial-calibrated (memo §3) — not a post-hoc concern, but worth stating
   so Phase E1 council understands the grid structure.
4. **Regime halves split point.** 2025-07-15 is the mid-bar split, not the
   mid-trade split. H1 has 102 trades, H2 has 118 trades — trades are
   slightly denser post-split. This is expected (1h GLOBEX volume skews
   toward more-recent data).

---

## Next actions (per preregistration §5 + user directive)

1. **Commit this verdict bundle** — verdict doc + CLAUDE.md carve-out
   extension. Gate JSONs + parquet artifacts are already committed at
   `5cbe652` (2026-04-22 UTC).
2. **Update memory** — new memory file
   `memory/project_probe3_combo865_pass.md` + MEMORY.md index entry.
3. **Fire LLM Council on paper-trade scope** (Phase E1) per §5.6 binding.
   Council framing must include Rule 1 (Stage 1 vs Stage 2 multiplicity) and
   Rule 2 (P(PASS|H1)/P(PASS|H0) per gate) from
   `feedback_council_methodology.md`. Key axes:
   - Exit strategy: EX_0 native vs EX_3 breakeven-after-1R (EX_3 gains are
     structural in PASS cells but amplify losses in FAIL cells; RTH-only
     windows with a broker that blocks overnight would flip this)
   - Session filter: SES_0 all vs SES_2 overnight-only (SES_2 has fewer
     trades and higher Sharpe — depends on sample-size vs signal-purity
     tradeoff)
   - Sizing policy: fixed $500 (repo default) vs fractional Kelly vs
     discrete contract count (combo 865 runs on 87 contracts at $500 risk /
     17.02 pt stop / $2 point value on MNQ = 87; for MNQ-only broker
     accounts may cap contract count)
   - Duration / trade-count floor: 220 trades/yr baseline → "N trades before
     go/no-go" criterion
   - Broker adapter / venue: which broker preserves overnight GLOBEX access
     at MNQ sizing; slippage model parity with engine $5/contract RT
4. **Draft `tasks/combo865_1h_paper_trade_plan.md`** based on council synthesis.
5. **Sign paper-trade plan** only after council + user explicit authorization.

---

## Files

- Readout aggregator: `tasks/_probe3_readout.py`
- JSON record: `data/ml/probe3/readout.json`
- §4.1 gate JSON: `data/ml/probe3/regime_halves.json`
- §4.2 gate JSON: `data/ml/probe3/param_nbhd.json` + `data/ml/probe3/param_nbhd/{combos.json, grid_sidecar.json, combos.parquet}`
- §4.3 gate JSON: `data/ml/probe3/15m_nc.json` + `data/ml/probe3/nc_15m/EX_{0,1,2,3}_*.parquet`
- §4.4 gate JSON: `data/ml/probe3/1h_ritual.json` + `data/ml/probe3/ritual_1h/EX_{0,1,2,3}_*.parquet`
- Remote launcher: `tasks/_run_probe3_remote.py`
- Gate scripts: `tasks/_probe3_{regime_halves,param_nbhd,15m_nc,1h_ritual,readout}.py`
- Preregistration (signed): `tasks/probe3_preregistration.md` (commit `f8447af`; review fixes `8636167`)
- Multiplicity memo: `tasks/probe3_multiplicity_memo.md`
- Preceding verdicts: `tasks/probe2_verdict.md` · `tasks/probe1_verdict.md`
- Council authority: `tasks/council-report-2026-04-21-probe1-branch-a-fork.html`
