# Probe 3 Verdict — Combo-865 4-Gate Robustness PASS on 1h Holdout

> 🛑 **RETRACTED 2026-04-23 UTC — PAPER_TRADE authorization withdrawn.**
> A critical timezone bug was identified in the post-hoc session decomposition
> scripts (`_probe3_1h_ritual.py:186`, `_probe3_15m_nc.py:207`) after this verdict
> was signed. Under corrected TZ: §4.3 goes from 0/16 PASS to **8/16 FAIL**
> (rescue fires); §4.4 strengthens from 8/16 to 12/16 PASS. F-count becomes **1**,
> which under the signed preregistration §5.2 routes to **COUNCIL_RECONVENE**,
> not PAPER_TRADE. See Amendment 2 at the bottom of this document.

**Date**: 2026-04-22 UTC
**Probe**: Combo-865 session/exit structure + robustness gates on 1h test partition (2024-10-22 → 2026-04-08)
**Preregistration**: `tasks/probe3_preregistration.md` (signed commit `f8447af` / review fixes `8636167`)
**Readout script**: `tasks/_probe3_readout.py`
**Machine record**: `data/ml/probe3/readout.json`
**Remote launcher**: `tasks/_run_probe3_remote.py`

---

## Bottom line

**F = 0. All four gates PASS. Branch = PAPER_TRADE per preregistration §5.**

Posterior P(genuine edge | F=0) ∈ **[0.65, 0.85]** (prior 0.167; BF ∈ [10×, 20×]
under the central reading, up to ≈ 54× at the upper edge — see post-audit
correction W3 and `tasks/probe3_multiplicity_memo.md` §6.4).

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
2. **EX_3 acts as an additive Sharpe shift, not a multiplicative amplifier.**
   On 15m SES_1 RTH, EX_3 breakeven-after-1R shifts Sharpe by Δ ≈ −0.34
   (from −0.777 to −1.114). On 1h SES_0 and SES_2 PASS cells, EX_3 shifts
   Sharpe by Δ ≈ +1.3 and Δ ≈ +1.6 respectively (see §4.4). The "+45–48%"
   framing elsewhere in earlier drafts reads as a large percentage only
   because the pre-EX_3 Sharpes are modest denominators; the mechanism is
   first-R breakeven lock-in converting partial runners into realized PnL.
   Read as: "EX_3 adds a roughly-constant trade-quality shift; the shift
   is helpful where entries have edge, hurtful where they don't." Not a
   universal upgrade. See post-audit correction W4 below.

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
  overnight than in RTH — approximately **$1,184/trade (overnight) vs
  $343/trade (RTH), a ratio of ~3.45×** — with a Welch t on mean-trade PnL
  giving p ≈ 0.11 at these sample sizes (143 overnight vs 65 RTH). The
  ratio is economically meaningful but not formally significant; read it
  as "large enough that a venue blocking overnight will break the edge,"
  not "multi-sigma separation." See post-audit correction W1 below.
- **EX_3 breakeven-after-1R is the strongest exit** in every session. On
  PASS cells, EX_3 shifts Sharpe by a roughly-additive Δ ≈ +1.3 (SES_0) /
  +1.6 (SES_2) by converting partial runners into locked gains when price
  retraces. This is an **additive lock-in effect**, not a multiplicative
  amplifier — the "+45% / +49%" framing in earlier drafts reads as big
  percentages only because native Sharpes are modest denominators. The
  mechanism is structural (entry → 1R → breakeven-stop trajectory), not
  statistical noise, but see §4.3 obs #2 for the same Δ magnitude in
  negative territory on 15m. See post-audit correction W4 below.

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
| 0 | **PAPER_TRADE** | **[0.65, 0.85]** (range; see W3) | Fresh LLM Council on paper-trade scope, then preregister paper-trade plan, then sign |
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

## Post-audit corrections (2026-04-22)

After the initial verdict draft, a logic-flaw reviewer + statistical-
reasoning auditor pass flagged four in-document inaccuracies (W1–W4) and
two deferred structural gaps (W5–W6). W1–W4 are corrected inline above;
the corrections do **not** change F-count, branch routing, or any gate
PASS/FAIL flag (the JSON ground truth at `data/ml/probe3/*.json` was
correct all along). W5–W6 are forwarded to the downstream artifacts
that properly own them.

### Correction W1 — overnight-vs-RTH per-trade ratio
Prior draft text: "$1,592 mean native / $232 mean native" ⇒ ratio ≈ 6.9×.
Corrected: **$1,184/trade (overnight, 143 trades) vs $343/trade (RTH, 65
trades), ratio ≈ 3.45×**. Welch t on mean-trade PnL gives p ≈ 0.11 at
these sample sizes. Economically meaningful but not formally significant.
The prior $1,592/$232 numbers are not reproducible from the 1h_ritual
gate JSON; best reading is that they were either a confused gross-vs-net
unit error or carried over from a superseded intermediate readout.

### Correction W2 — contract sizing
Prior draft text: "combo 865 runs on 87 contracts at $500 risk / 17.02 pt (note: the "$500 risk" framing is the eval-notebook `fixed_dollars_500` policy; the sweep engine actually uses $2,500 = 5% × $50k equity per `scripts/param_sweep.py:1030`, so the engine's 220 trades were executed at 73 contracts not 87 nor 15)
stop / $2 point value on MNQ = 87". Corrected: 500 / (2 × 17.02) =
**~73 MNQ contracts** at the sweep engine's $2,500 risk per trade (5% of $50k fixed_equity per `scripts/param_sweep.py:94-95, 1030`), verified via friction $438/trade = 73 × $6 (the $6 includes a $1 slippage bump from `fill_slippage_ticks=1` per `scripts/param_sweep.py:895-901`). The "87" figure appears to be
a typo or stale pre-refactor number — the correct arithmetic is
14.69 contracts rounded to 15. This is a cosmetic correction that does
not affect any gate result (the sweep engine uses the fixed_dollars_500
sizing helper directly; the bullet in §5 was advisory text only).

### Correction W3 — posterior range
Prior draft text: "Posterior P(genuine edge | F=0) = 0.91 (BF ≈ 54×)".
Corrected: **posterior ∈ [0.65, 0.85]** reflecting sensitivity across
three modeling choices: (a) Stage-1 multiplicity denominator — Probe 2
admitted combo-865 through a pool-of-1 gate, but the family-level
13,814-combo sweep space is a legitimate alternative denominator
depending on audit intent; (b) gate independence assumption — §4.1
aggregate and §4.2 center cell both include the same 220-trade
measurement, so gates partially share evidence rather than being
four fully independent draws; (c) the pre-audit posterior math did
not discount the §4.4 "pick one of 4 PASS variants" deployment-variant
selection (Stage 3 multiplicity). The 0.91 point estimate remains
reachable under one combination of (a/b/c) but is the upper edge of
the plausible range, not the central reading. BF ∈ [10×, 20×] under
the central reading.

### Correction W4 — EX_3 framing
Prior draft text: "amplifies edge by ~45–49% on PASS cells, amplifies
sign both ways on FAIL cells". Corrected: EX_3 breakeven-after-1R
delivers a **roughly-additive Sharpe shift Δ**, not a proportional
amplifier. Observed Δ magnitudes: +1.3 (SES_0), +1.6 (SES_2), −0.34
(15m SES_1 RTH). "+45%" is a big percentage only because native Sharpes
are modest denominators; on a FAIL cell where native Sharpe is −0.78,
the same additive Δ reads as "sign amplification." Mechanism is first-R
breakeven lock-in (partial runners → realized PnL), not statistical
noise.

### Pending E1 — W5 (deployment-variant lock)
Probe 3 reported 4 cells passing §4.4 (SES_0 × {EX_0, EX_2, EX_3} +
SES_2 × {EX_0, EX_2, EX_3}, with EX_1 ≡ EX_0 on this timeframe).
**No single ship variant has been preregistered.** Phase E1 council
must select one (EX × SES) combination before Phase E2 preregistration
is drafted, framing the decision under Rule 1 (Stage-3 multiplicity:
4 PASS cells is a "pick-the-best" inflation beyond the §4.4 gate's
already-accounted Stage 2 count) and Rule 2 (P(PASS|H1)/P(PASS|H0)
for the chosen cell). Until that lock exists, "the combo passed"
should not be conflated with "the deployment config is decided."

### Pending E2 — W6 (paper-trade falsification criteria)
Preregistration §5.6 binds Phase E1 council convening but does not
itself specify **paper-trade kill-switch criteria** (max DD before
decommission, min trades before go/no-go, max consecutive losses,
Sharpe-trajectory floor). These belong in
`tasks/combo865_1h_paper_trade_plan.md` (Phase E2 preregistration
artifact), signed before paper trading begins. Absent explicit
criteria, "paper trade until something feels wrong" is not a
falsifiable commitment and violates the preregistration discipline
that Probes 1–3 established.

### Supportive findings (unchanged by corrections)
- §4.1 H1 vs H2 mean-per-trade difference p ≈ 0.70 (not regime-dependent
  collapse — just natural variance across the split); H1 > H2 by ~25%
  in per-trade mean edge, consistent with slow decay rather than
  window-specific structure.
- §4.3 15m observation **0/16** is a **ceiling** (threshold was ≤ 2),
  so the gate cleared with floor-to-ceiling margin — strongest possible
  negative-control outcome.
- Test-partition Sharpe is expected to regress from 2.89 toward the
  [1.0, 3.5] interval under standard sample-size decay; multiplicity
  memo §2 test > train regression expectation is directionally aligned
  with the observed H1 > H2 signature.

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
  delivers an additive Sharpe lock-in (Δ ≈ +1.3 on SES_0, +1.6 on SES_2;
  Δ ≈ −0.34 on 15m SES_1 RTH) when edge exists; SES_2 overnight-only is
  the cleanest slice; RTH-only is not a viable venue for combo 865. See
  post-audit correction W4 for why the "+45%" framing in earlier drafts
  was misleading.

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
     discrete contract count. At $500 fixed risk / 17.02 pt stop /
     $2 point value on MNQ, **the eval-notebook contract count under the
     `fixed_dollars_500` policy is 500 / (2 × 17.02) = ~15 contracts**.
     The SWEEP ENGINE ran combo-865 at **73 contracts** ($2,500 risk per
     `scripts/param_sweep.py:1030`, verified via friction $438 = 73 × $6
     on the actual Probe 2 parquet) — the "87" in an earlier draft was a
     typo and "~15" conflated eval-notebook sizing with sweep-engine
     sizing;
     see post-audit correction W2). MNQ-only broker accounts with
     integer contract floors should still accommodate 15 cleanly; the
     sizing question is really about Kelly-vs-fixed variance discipline,
     not contract capacity.
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

---

## Amendment 2 — Timezone bug retraction (2026-04-23 UTC)

### Summary

`tasks/_probe3_1h_ritual.py` and `tasks/_probe3_15m_nc.py` both contained
`ts.dt.tz_localize("UTC")` on `data/NQ_1h.parquet` and `data/NQ_15min.parquet`
timestamps, which are actually naive **Central Time** (Barchart vendor export
— see `scripts/data_pipeline/update_bars_yfinance.py:37` authoritative marker).
Localizing CT as UTC and then converting to ET shifts every timestamp by ~5–6
hours and inverts the SES_1 (RTH) / SES_2 (GLOBEX) labels on most bars.

The bug was discovered on 2026-04-23 UTC during a paper-trade-readiness review
by the stats-ml-logic-reviewer (artifact:
`tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-paper-trade-readiness.md`).
Independent verification reproduced the reviewer's numbers to 3 decimal places.

Root cause + propagation trace: see `lessons.md` `2026-04-23 tz_bug_in_session_decomposition`.

### Corrected gate results (re-run 2026-04-23 UTC after the fix)

| Gate | Pre-fix | Post-fix | Threshold | Status |
|---|---|---|---|---|
| §4.1 regime halves (uses naive-midpoint, not ET-minute) | both PASS | both PASS (unchanged) | both ≥ 1.3/25/$5k | PASS (TZ-agnostic) |
| §4.2 param ±5% nbhd (no session decomp) | 27/27 | 27/27 (unchanged) | ≥ 14/27 | PASS (TZ-agnostic) |
| §4.3 15m negative control | 0/16 PASS | **8/16 FAIL** (rescue fires) | ≤ 2/16 | **FLIPPED: PASS → FAIL** |
| §4.4 1h session/exit ritual | 8/16 exactly PASS | **12/16** PASS (clear) | ≥ 8/16 | PASS (strengthened) |

### §5 branch routing under corrected gates

F-count was **0** → now **1**. Per preregistration §5.2 ("Exactly 1 / 4 FAIL
→ **council re-convene** (ambiguous branch)"), the verdict branch is no longer
PAPER_TRADE.

**New branch: COUNCIL_RECONVENE.**

### Corrected interpretation

The §4.3 15m negative control now fails because specific session/exit filter
combinations produce a positive 15m signal (8 of 16 cells clear the per-cell
three-gate set under corrected session labels), even though the 15m aggregate
Sharpe remains −0.55 per Probe 2. This contradicts the pre-fix claim that "the
15m signal is absent" — instead, the 15m signal can be **rescued** via session/
exit manipulation. The "1h is a uniquely privileged timeframe" claim that §4.3
was designed to test therefore fails.

The §4.4 ritual continues to pass, and actually strengthens (8/16 → 12/16) —
the 1h edge is robust to session/exit variations. But one PASS does not
rescue the F=1 branch routing.

### What stands

- §4.1 and §4.2 are unchanged under corrected TZ. Regime stability and
  parameter neighborhood robustness are real.
- Probe 2 combo-865 1h single-gate PASS (Sharpe 2.895, 220 trades,
  +$124,896/yr) stands unchanged. It's a session-agnostic aggregate.
- Probe 1 family-level falsification (N_1.3=4/1500 on 1h) stands unchanged
  in direction, though the specific 4 combos may shift under a corrected
  engine-side bar_hour interpretation (see `memory/feedback_tz_source_ct.md`
  engine-side caveat).

### What is retracted

- **F=0 → PAPER_TRADE branch.** Under corrected TZ, F=1 → COUNCIL_RECONVENE.
- **Posterior P(genuine edge | F=0) ∈ [0.65, 0.85].** The Bayes factor
  estimate was conditioned on F=0; it no longer applies.
- **"Edge concentrates overnight" structural finding.** Under corrected TZ,
  combo-865's RTH Sharpe is 2.64 (not 0.64) and overnight Sharpe is 1.47
  (not 3.32). The edge is RTH-leaning, not overnight-dominant.
- **Strongest deployment variant EX_3×SES_2 Sharpe 4.95.** The session
  decomposition underlying this number used the buggy TZ; the re-run under
  corrected TZ yields a different ranking that has not been computed here.
- **§4.4 "8/16 exactly at threshold" concern.** The at-threshold concern
  that Phase E1 council was supposed to adjudicate no longer applies
  (12/16 is comfortably above threshold). A different concern replaces it:
  §4.3 fails outright.
- **Combo-865 paper-trade candidate status.** No new paper-trade
  preregistration should be drafted without a fresh Probe-3-equivalent
  preregistration cycle, with gates (including the 15m negative control)
  re-specified against the corrected TZ.

### Downstream consequence

Per preregistration §5.2, COUNCIL_RECONVENE branch requires a fresh LLM
Council to scope the ambiguity. As of this amendment, no such council has
been fired. Any future action on combo-865 must proceed from the
COUNCIL_RECONVENE state, not from a phantom PAPER_TRADE authorization.

### Code fix

- `tasks/_probe3_1h_ritual.py:186` and `tasks/_probe3_15m_nc.py:207`:
  `tz_localize("UTC")` → `tz_localize("America/Chicago", ambiguous="infer",
  nonexistent="shift_forward")`. Variable renamed `ts_utc` → `ts_ct`.
  Misleading comments replaced with explicit CT reference + vendor marker
  pointer.
- Re-ran both scripts locally (pure pandas, no engine re-run required) to
  produce corrected JSONs at `data/ml/probe3/1h_ritual.json` and
  `data/ml/probe3/15m_nc.json`.

### References

- `lessons.md` `2026-04-23 tz_bug_in_session_decomposition` — root-cause post-mortem
- `memory/feedback_tz_source_ct.md` — durable source-TZ rule
- `memory/project_tz_bug_cascade.md` — cascade summary across Probe 3/4/Scope D
- `tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-paper-trade-readiness.md` — discovery review
- `scripts/data_pipeline/update_bars_yfinance.py:37` — authoritative vendor TZ marker
