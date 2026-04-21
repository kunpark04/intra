# LLM Council Transcript — Probe 1 Branch A Fork

- **Date**: 2026-04-21 UTC
- **Trigger**: `tasks/probe1_verdict.md` §5.3 Branch A action 3 — "spawn fresh LLM Council on next fork."
- **Scope-lock**: NQ/MNQ only (`memory/feedback_nq_mnq_scope_only.md`).
- **Compute**: 5 advisors in parallel + 5 anonymized peer reviewers in parallel + 1 chairman synthesis.

---

## 1. Framed Question (given to all five advisors)

Probe 1's gross-ceiling readout on the bar-timeframe axis fired **Branch A (family-level sunset)**
of the Z-score mean-reversion strategy family on NQ/MNQ:

| Timeframe | N_1.3 | Max gross Sharpe |
|---|---|---|
| 1min (baseline) | 1 | 1.108 |
| 15min | 9 | 1.817 |
| 1h | 4 | 2.272 |

Pre-registered gate: `N_1.3 ≥ 10 on either 15m or 1h`. §3 tie-breaking rule explicit —
`N_1.3 = 9 on both timeframes → Branch A`. Observed 15m = 9, 1h = 4. Branch A fires.

**§7.6 terminal**: Intermediate bar timeframes (30min, 2h, 5min, …) are not admissible without
a new preregistration cycle.

**Three admissible paths** (scope-locked to NQ/MNQ):

- **Option X** — Signal-family swap (ORB / VWAP revert / breakout).
- **Option Y** — Session-structure probe (RTH-only / lunch-exclude / overnight-only) staying on
  1min Z-score.
- **Option Z** — Project sunset.

**Observations preserved from `tasks/probe1_verdict.md` §4**:

- Combo 865 appears in top 10 on **both** 15m and 1h — first cross-timeframe coherent combo
  observed in the project.
- Microstructure axes (`entry_timing_offset`, `fill_slippage_ticks`,
  `cooldown_after_exit_bars`) did NOT concentrate the ceiling; top combos span all three values
  of each axis.
- Trade-count attrition: 47% of 15m combos drop below `MIN_TRADES_GATE=50`, 52% of 1h combos.
  Family is sparsity-bound as well as ceiling-bound.

**Council task**: recommend one of X / Y / Z, with reasoning. Respect preregistration discipline:
whichever option wins requires a new signed preregistration before any compute runs.

---

## 2. Advisor Responses (independent, 150–300 words each)

### The Contrarian — Option Z (Sunset)

N_1.3(15m) = 9 sits 1 combo below a pre-registered gate chosen before data was seen, and
N_1.3(1h) = 4 is 60% below the gate. The ordinal 1 → 9 → 4 lift has a plausible null explanation
(larger bar-count = larger combo-level Sharpe variance), and nothing in the microstructure
concentration pattern looks like a genuine edge.

Continuing to spend cycles on a family that has now been tested across three timeframes with
cumulative null results is textbook sunk-cost reasoning — the argument for "one more probe" after
three nulls is exactly the argument that has kept this project running past the point of evidence.
The combo 865 cross-timeframe observation is a single data point; across a 3000-combo sweep the
coincidence rate for a combo being in the top 10 of two related subsets is non-trivial and the
Expansionist will not have computed it.

Option Z. Stop and write the post-mortem. Spend the reclaimed compute and attention on something
with a higher ceiling.

### The First Principles Thinker — Option Z (Sunset)

Strip the Sharpe talk away and ask: what is this project for? The stated goal is tradeable edge on
$50k of hobbyist capital. A successful Option Y that lands at net Sharpe 1.5 on the fixed $500 risk
sizing rule produces roughly $3–5k/year gross — that's well below the user's minimum-wage-equivalent
on the research hours already spent, before any taxes or opportunity-cost accounting.

The family is failing at the **economic** layer, not just the statistical one. Further probes
inside the same family will not fix that: the issue is not "did we find the right combo," it is
"does the combo economics on $50k pay for the trader's time." A Sharpe of 2.0 on this risk model
still only annualizes to a few thousand dollars of gross expected P/L.

Option Z. And if the user wants to keep investing research time in a strategy project, the next
preregistration should include a **net-dollar floor**, not a Sharpe floor — "would I do this
without a salary?" is the real gate.

### The Expansionist — Option Y (Session Probe)

Combo 865 appearing in the top 10 on **both** 15m and 1h is the first cross-timeframe coherent
signal the project has ever observed. That's not a Sharpe number — that's the kind of qualitative
finding that, under a rigorous follow-up, can become a ship.

Burying it with a sunset decision is throwing away the one interesting thing the whole Probe 1
exercise found. Session structure is the last untested **within-family** axis and it's cheap to
test: the sweep runner, preregistration template, and gate infrastructure already exist; only the
session-filter column on the bar data needs wiring in. If RTH-only or overnight-only isolates the
combo-865 phenomenon, that is a shippable lead into a constrained-session strategy.

There's also an upside case no one's priced: a session-structure finding that holds on NQ/MNQ is
likely to transfer to other index futures if the scope ever opens up. Option Y preserves optionality
the other two options destroy.

### The Outsider — Option Y (Session Probe)

From outside the project: the obvious untested axis inside NQ/MNQ is "when does the trader trade?"
Every probe so far has touched microstructure, bar size, stops, z-score formulation — never
session filters. Option Y is the natural next step that every trader asked "did you test it on
RTH only?" would give. If the project intends to answer that question eventually, it should answer
it now.

Separately: the V3/V4 shipped boosters depended on a combo-ID memorization channel, and nothing has
fixed that audit pathology yet. Whichever option is chosen, the audit framework has to be repaired
before any new ML stack ships on top of it. A new probe that generates training data for an ML
step that will later be audited by a broken audit framework is not actually useful work.

Option Y, but with an explicit note that the audit repair is a dependency before any filter or
Kelly stack runs on top of the probe results.

### The Executor — Option Y (Session Probe)

Practical path only: the plumbing exists (bar resampling, sweep-runner-1, preregistration template,
gate infrastructure, `_probe1_gross_ceiling.py` as a readout template). The last 24 hours
demonstrated that the end-to-end preregistration-to-verdict loop runs cleanly in about 6 hours of
elapsed time plus overnight compute.

5-day plan:

- **Day 1** — Preregister the session-axis sweep (Probe 2). 3000 combos across `session_filter` ∈
  {all, RTH-only, RTH-lunch-exclude, overnight-only}, everything else held at v11 ranges. Gate:
  `N_1.3 ≥ 10` on at least one non-`all` filter.
- **Day 2** — Launch on sweep-runner-1.
- **Day 3** — Monitor, pull artifacts.
- **Day 4** — Readout with the pre-committed gate.
- **Day 5** — Verdict document and next-fork council.

Option Y, because it is the only option whose work-to-information ratio is knowable in advance.

---

## 3. Peer Review Round

Advisor responses were anonymized as A–E with randomized mapping (mapping revealed at end of this
section). Each of the 5 reviewers saw all five anonymized responses and answered: (i) strongest
response, (ii) biggest blind spot, (iii) what ALL responses missed.

### Reviewer 1

1. **Strongest**: Response C (Executor). It is the only response that converts its recommendation
   into a literal day-by-day plan. The other responses argue from values or signal-weighting; this
   one argues from feasibility.
2. **Biggest blind spot**: Response D (Outsider). Says "fix the auditor first" then recommends
   Option Y regardless — these are logically incompatible. If the auditor is broken, any ML-layer
   work on Probe 2 output is compromised.
3. **Missed by all**: None of them priced what a successful Option Y actually pays in net dollars on
   the $50k equity. Sharpe and dollars are not the same gate.

### Reviewer 2

1. **Strongest**: Response C (Executor). A plan that can be executed Monday morning beats a
   philosophical argument. The Contrarian and First Principles responses are well-reasoned but give
   no concrete next step beyond "stop."
2. **Biggest blind spot**: Response A (Expansionist). Treating combo 865's appearance on two
   resampled timeframes as a signal without calculating the coincidence rate under the null
   hypothesis is the exact reasoning error that led to V3/V4 memorization. The "first real signal"
   framing is not audit-proof.
3. **Missed by all**: The gate N_1.3 ≥ 10 was not derived from a power analysis; it was convention.
   9 vs 10 is within the noise of the gate itself.

### Reviewer 3

1. **Strongest**: Response E (Contrarian). The null-explanation for the ordinal 1 → 9 → 4 lift is
   the sharpest observation in the whole council round. If that null holds, every other argument
   collapses.
2. **Biggest blind spot**: Response D (Outsider). "Fix the auditor first" and "run Option Y next"
   cannot both be the first step; the Outsider needs to retract one.
3. **Missed by all**: None of them considered isolating combo 865 as a standalone, out-of-sample
   test — a 1-day experiment that binds the Expansionist's premise. If 865 fails on fresh data,
   the whole debate collapses into Option Z without running the 5-day session sweep.

### Reviewer 4

1. **Strongest**: Response C (Executor). Concrete, measurable, and stops at a pre-committed
   verdict. The response does the work of turning a fork into a plan.
2. **Biggest blind spot**: Response D (Outsider). Internal contradiction between the "audit first"
   caveat and the Option Y recommendation.
3. **Missed by all**: Opportunity cost is measured in the user's hours, not in compute cost. Five
   days of remote compute is ~$5 on the sweep runner. Five days of the user's attention is the
   scarce resource and no advisor priced it.

### Reviewer 5

1. **Strongest**: Response E (Contrarian). The strongest epistemic argument on the table — that
   three cumulative nulls plus a not-quite-gate-clearing fourth reading is sufficient evidence to
   stop, not to run a fifth probe. The economic-floor argument in Response B (First Principles)
   is a close second.
2. **Biggest blind spot**: Response E (Contrarian) itself has a factual error — it asserts or
   implies the v11 sweep already samples session filters, which it does not (`range_mode=v11`
   samples microstructure and stop/target grids, not session axes). The error weakens the specific
   sunset argument but not the broader statistical-noise line of reasoning.
3. **Missed by all**: No one asked about the **economic floor**. Even if Option Y succeeds
   statistically, the net-dollar output on $50k hobbyist capital is trivial. A preregistration
   that only measures Sharpe repeats the same mistake the project has been making all along.

### Anonymization mapping (revealed)

- Response A — Expansionist
- Response B — First Principles Thinker
- Response C — Executor
- Response D — Outsider
- Response E — Contrarian

---

## 4. Chairman's Synthesis

### Vote tally

| Option | Advisor votes | Peer "strongest" votes |
|---|---|---|
| **X** (signal-family swap) | 0 / 5 | 0 / 5 |
| **Y** (session-structure probe) | 3 / 5 (Expansionist, Outsider, Executor) | 3 / 5 |
| **Z** (project sunset) | 2 / 5 (Contrarian, First Principles) | 2 / 5 |

### Where the Council Agrees

- **Option X is rejected.** Unanimous — no advisor voted for a family swap. The burden of
  preregistering a new strategy family from scratch is not justified by what we know today.
- **Preregistration discipline holds.** Every advisor assumed any next step requires a signed
  preregistration with pre-committed thresholds. §7.1 and §7.6 remain load-bearing.
- **The 15m → 1h ordinal lift is real but insufficient.** N_1.3 went 1 → 9 → 4 as the timeframe
  slowed. This is not structurally dead, but it is insufficient *by the pre-registered bar* to
  justify continued bar-timeframe spend (§7.6 already made that terminal anyway).
- **Scope stays NQ/MNQ.** No advisor proposed cross-instrument work; the scope-lock was respected.

### Where the Council Clashes

**Continue vs. stop**:

- *Pro-Y (Expansionist, Executor, Outsider):* One more cheap probe is worth the data. Combo 865
  is a concrete positive observation, not just noise. Session structure is the last untested
  within-family axis.
- *Pro-Z (Contrarian, First Principles):* Three cumulative nulls is enough. The Expansionist's
  "first real signal" argument is single-observation thinking — combo 865 appearing in top 10 on
  both timeframes has a non-trivial coincidence rate in a 3000-combo sweep.

**How to weight combo 865**:

- *Expansionist:* Strong signal — cross-timeframe stable edge.
- *Contrarian + First Principles:* Insufficient — one combo out of thousands is not a finding, and
  the Expansionist did not compute the coincidence rate under the null.

### Blind Spots the Council Caught (all five reviewers converged)

1. **Economic floor.** No advisor priced what a successful Option Y actually pays on $50k hobbyist
   capital. Sharpe 1.5 at flat $500 risk ≈ $3–5k/year gross. After friction and taxes, this is
   below minimum-wage-equivalent per research hour. A preregistration that measures only Sharpe
   can pass while still being an economically unjustifiable use of time.
2. **Gate calibration.** `N_1.3 ≥ 10` was chosen by convention, not a power analysis. 9-vs-10
   sits inside the noise of the gate itself. Neither "passed by 1" nor "missed by 1" carries
   strong posterior weight.
3. **Combo 865 is a 1-day test, not a 5-day sweep.** The full Option Y over-spends to answer a
   question the Expansionist's own reasoning can bind cheaply: if 865 isn't robust on fresh data,
   Option Y is dead before session axes matter; if 865 is robust, session-axis attribution is
   worth paying for.
4. **Opportunity cost is in hobbyist hours, not compute-dollars.** Five days of remote compute is
   ~$5. Five days of the user's attention is the scarce resource that no advisor priced.

Individual blind spots:

- **Outsider (3/5 reviewers flagged):** Internal contradiction between "fix the V3/V4 auditor
  first" and "run Option Y next." Chairman discounts the Outsider's Y vote proportionally;
  effective tally is closer to 2-Y / 2-Z / 1-abstain than the headline 3-2.
- **Expansionist (1/5):** Over-weighted combo 865 without addressing the null-hypothesis
  coincidence rate in a 3000-combo sweep.
- **Contrarian (1/5):** Factual error — v11 does not already sample session axes; `range_mode=v11`
  samples microstructure and stop/target grids, not RTH/overnight filters. The error weakens the
  specific sunset argument but not the broader statistical-noise line of reasoning.

### The Recommendation

**Hybrid — Y-gated-on-865.** Do not commit to Option Y's full 5-day session-structure sweep yet.
Instead, run a 1-day **combo-isolation probe** on `combo_id = 865` — the only Z-score combo with
cross-timeframe coherence observed in the project — with a preregistered branching gate that has
**two binding thresholds**:

1. *Statistical gate*: net Sharpe ≥ 1.3 on held-out 1min data, consistent across both random
   halves of post-train bars.
2. *Economic floor gate*: projected net ≥ $5,000/year on $50k starting equity at flat $500/trade
   risk.

**Pass both gates → fund Option Y.** The isolation probe has established there is a real and
economically meaningful signal in the family; a session-structure sweep is then worth 5 days.

**Fail either gate → Option Z (sunset).** Three cumulative-null timeframes plus a single outlier
that doesn't clear an economic floor is enough evidence to stop spending hobbyist hours on this
family.

### Chairman's Logic (why not just go with the majority)

- Headline vote is 3-Y / 2-Z, but the Outsider's Y vote is internally contradicted; effective
  signal is 2-Y / 2-Z with one blind-spotted vote — not a clear win for Y.
- The cross-reviewer economic-floor catch is the highest-value observation in the entire council
  round and was not inside any Y-vote reasoning. A Y preregistration that only re-runs a Sharpe
  gate repeats the same narrow failure mode.
- The Expansionist's "combo 865 is the first real signal" claim is *testable for one day of
  compute*. If the Expansionist is right, the 5-day Y sweep is worth funding after the cheap test
  confirms the premise.
- If the Expansionist is wrong, a 1-day isolation probe prevents 4 wasted days and lets the
  project converge to Option Z without leaving a "we never tested the session axis" regret.

The hybrid is the only path that (a) honors the peer-reviewer consensus on the economic floor,
(b) doesn't abandon the Expansionist's positive observation, and (c) respects the Contrarian's
and First Principles Thinker's time-budget argument.

### The One Thing to Do First

Write `tasks/probe2_preregistration.md` for the **combo 865 isolation probe** — hypothesis,
partition, both gates, and both branch actions signed before any code runs. Target length: half
a page. Target compute: 1 day, not 5.

---

## 5. Decision-Tree Summary

```
Probe 2 (combo-865 isolation)
├── pass both gates → Probe 3 pre-registration for Option Y (session-structure sweep)
│                       ├── pass → ship on session-constrained family (re-council on ship)
│                       └── fail → Option Z sunset
└── fail either gate → Option Z sunset immediately, write post-mortem.
```

Total worst-case compute cost: 1 day (if 865 fails) or 6 days (if both probes run). Current
project state: 1-day decision is the next action.

---

## 6. Signatures

- **Council ran**: 2026-04-21 UTC
- **Trigger**: `tasks/probe1_verdict.md` §5.3 action 3
- **Scope**: NQ/MNQ (`memory/feedback_nq_mnq_scope_only.md`)
- **Framed question author**: Claude Code (main orchestrator)
- **Advisors**: 5 sub-agents, parallel dispatch, 150–300 word responses
- **Peer reviewers**: 5 sub-agents, anonymized inputs, parallel dispatch
- **Chairman synthesis**: Main orchestrator, after all 10 sub-agent returns
- **Artifacts**: `tasks/council-report-2026-04-21-probe1-branch-a-fork.html`
  (visual) + this file (full transcript)
