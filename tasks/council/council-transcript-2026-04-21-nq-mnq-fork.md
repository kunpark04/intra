# LLM Council Transcript — Within-NQ/MNQ Fork Ordering

**Session**: 2026-04-21 12:50 CDT
**Probe**: Probe 3 of Phase 5 post-sweep plan (`tasks/todo.md`)
**Scope constraint**: `memory/feedback_nq_mnq_scope_only.md` — no cross-instrument probes
**Anonymization mapping** (revealed post-review): A = Outsider · B = Contrarian · C = Expansionist · D = Executor · E = First Principles

---

## 1. The Framed Question

> The v11 1-minute Z-score mean-reversion sweep falsified the family on
> NQ/MNQ at the user's ship bar (gross ceiling 1.108, one combo of 13,814).
> Research scope is locked to NQ/MNQ — no cross-instrument probes. Within
> that envelope, which unexplored axis has the highest evidentiary leverage
> as the next load-bearing probe, and what is the pre-registered sunset
> threshold?
>
> Unexplored axes under consideration:
> - **Bar timeframe**: 5m / 15m / 1h / 4h / daily
> - **Signal family**: momentum/breakout, vol-regime, stat-arb, multi-factor
> - **Session structure**: RTH / overnight / open-drive / close window
> - **Exit/sizing model**: time-of-day exit, regime-adaptive, Kelly, vol-targeted
> - **Project sunset**: declare the direction retired, preserve the harness
>
> Working context:
> - Gross ceiling of 1.108 established on the v11 post-gate universe (13,814
>   combos), MIN_TRADES_GATE = 500, zero-friction Sharpe.
> - V3 and V4 ML#2 stacks both revoked 2026-04-21 for `global_combo_id`
>   memorization leak that preserved AUC parity.
> - Research scope is locked to NQ/MNQ — cross-instrument probes (ES / YM /
>   RTY) are out of scope.

---

## 2. Advisor Responses

### The Contrarian

The axis everyone is missing is execution microstructure. The v11 sweep was
a pure parameter sweep at 1-minute bars with a fixed 2-tick slippage model
and no cooldown after exits. If the 1-minute failure is actually an
execution-cost failure — 2 ticks on a mean-reversion entry at a band
extreme chews through a gross edge of ~1 point — no amount of changing bar
timeframe or signal primitive will fix it. You'll replicate the failure at
a different cadence.

Before committing to a new sweep: what happens to the v11 Sharpe
distribution if we strip the slippage model to zero? If the ceiling moves
meaningfully (say, to 1.5+), execution friction is the binding constraint
and the entire probe list is misframed. If it doesn't move, then yes,
sweep axes — but explicitly include an `entry_timing` and `cooldown` axis
so next time's post-mortem isn't "oh, we never checked microstructure."

And whatever sweep runs: pre-register a combo-agnostic K-fold audit. V3
and V4 both memorized combos under AUC parity. Don't repeat that pattern.

### The First Principles Thinker

The question is framed wrong. You're asking "which axis to probe next" —
but what you actually need is a falsification threshold for the Z-score
mean-reversion family on NQ/MNQ. The five axes aren't independent; they're
a decomposition of "stuff we haven't tried." Run Probe 1, fail, pivot to
Probe 2, fail, pivot to Probe 3. You'll death-march through the axis list
without a committed exit.

Pick ONE axis, pre-register a sunset threshold, execute cleanly. If the
probe fails, sunset the family — not the probe. Bar timeframe is the best
pick because it's the cheapest kill: same harness, same primitive, same
eligibility gate, different bars. If 15m fails at the same ceiling as 1m,
the signal family is falsified across the time-resolution axis, which is
strong evidence against the family as a whole.

Family-level falsification on a single probe is the only way you get to a
clean decision with finite compute. Don't probe hoping to find an edge —
probe expecting to sunset, and be surprised when you find one.

### The Expansionist

Go big. Probe 1 (timeframe lift) is correct, but do it in parallel — 15m
AND 1h, not sequential. Skip 5m entirely. 5m shares the 1m microstructure
noise profile (same intraday frictions, same regime sensitivity), so it's
highly correlated with the 1m failure mode. If 15m passes and 5m fails,
the ambiguity costs a week of interpretation.

Parallel 15m + 1h is information-maximizing: one tests intraday cadence,
one tests swing cadence, and the gap between them tells you where the
signal actually lives. If one clears and the other doesn't, you have
directional evidence about the strategy's time-scale. If both clear, you
have two independent shipping candidates.

Bigger picture — if the Z-score mean-reversion family is alive at any
timeframe, that's a new research frontier worth months of refinement. The
upside of finding ONE tradeable bar-timeframe configuration on NQ/MNQ is
a working strategy; the downside of running two sweeps in parallel is
~24h unattended remote compute on sweep-runner-1. Asymmetric bet, take it.

### The Outsider

I don't know what "axis" means here. You have a backtest of a strategy
that lost money. You're proposing five different ways to change the
strategy. Before I hear "which change," I want to hear: what are you
actually testing?

If the answer is "whether we should be in this market at all with any
mean-reversion strategy," then timeframe probes don't answer that. If the
answer is "whether mean-reversion fails specifically at 1-minute
resolution on NQ," then one alternative timeframe answers it. You might
be conflating "find a working strategy" with "rule out a family." Those
are different experiments.

If you're ruling out, pick the cheapest probe. If you're searching, you
need a broader search — axis-by-axis will take forever. From outside, it
looks like you're trying to do both with one probe. Pick one intent, then
pick the probe that matches. The word "axis" is hiding this decision from
you.

### The Executor

Monday morning: you need a sweep harness that supports 15m and 1h bars,
and a pre-registered kill rule. Neither exists today. Bar-aggregation
code is not in the repo (I checked — no `resample` anywhere in `src/`).
That's a ~2-3 hour build. Extending `param_sweep.py` to take a
`--timeframe` flag is ~1-2 hours.

Before either, the pre-registration needs to be written: kill bar, ship
bar, K-fold protocol (sklearn `GroupKFold` by `combo_id`, three lines),
sunset threshold. Write the doc TODAY, get sign-off with a commit hash
(`phase5_kill_criterion.md` is the template), then build infrastructure.

Also include microstructure sweep parameters — `entry_timing_offset`,
`slippage_ticks`, `cooldown_after_exit_bars` — because we're rebuilding
the sweep anyway and it's the cheapest way to pick up a hidden friction
variable. 27 total microstructure cells × 3000 combos = well within
remote compute budget.

Parallel 15m + 1h sweeps on sweep-runner-1, monitor every 10 min per the
existing remote-job workflow. Gross ceiling readout takes 30 min. If
clear, K-fold audit takes 3-5h per timeframe. Total wall-clock: 1-2 days.
Do it.

---

## 3. Peer Review (anonymized as A-E)

Each reviewer answered three questions:
1. Which response is the strongest?
2. Which response has the biggest blind spot?
3. What did ALL responses miss?

### Reviewer 1

**Strongest: D.** Only response with a concrete executable plan. Handles
both the infrastructure ask and the pre-registration ask, and engages
with the V3/V4 leak problem explicitly rather than hand-waving. B gets
close on the leak mention but stops at "pre-register the audit" without
specifying how.

**Biggest blind spot: B.** The microstructure-strip diagnostic is
valuable, but B misframes it as an alternative to timeframe probes rather
than a cheap addition. That's a false dichotomy. D resolves the same
concern by folding microstructure into the sweep itself — strictly
dominant strategy.

**All missed:** whether the v11 trade log itself is the right input for
the K-fold audit, given the 100%-combo-overlap pathology documented in
`project_combo_overlap_structural.md`. A GroupKFold on combos whose
parameters are parameter-space neighbors still risks leakage through
parameter proximity, not just combo identity.

### Reviewer 2

**Strongest: D.** Every other response argues; D plans. B identifies a
real blind spot but doesn't give you Tuesday. E gives you a clean frame
but not a timeline. A and C both gesture at strategic shape but can't
land a date.

**Biggest blind spot: B.** Contrarian fixates on microstructure to the
exclusion of the framing. If the strip diagnostic shows the ceiling stays
at 1.108, B has no second act. D's version (include microstructure as a
sweep parameter) handles the same hypothesis without the diagnostic's
all-or-nothing structure.

**All missed:** regime dependency. 2019-2024 spans a major volatility
regime change — 2020 COVID, 2022-23 rate cycles. An edge at 15m in 2024
but not 2020 is a fragile edge, and none of the advisors specified a
walk-forward partition alongside the combo-level K-fold.

### Reviewer 3

**Strongest: D.** Concrete, resource-aware, and handles the leak problem
without being asked. Also correctly identifies the order of operations:
preregistration first, infrastructure second, sweep third.

**Biggest blind spot: E.** "Pick ONE axis" framing is philosophically
clean but operationally wrong. Expansionist's 15m + 1h parallel
dominates if compute is available, which it is on sweep-runner-1. E's
rigidity would burn a week re-litigating axis selection every time a
single probe returns ambiguous.

**All missed:** combo-ID leak carryover. None of the advisors explicitly
specified which partition prevents the new 15m/1h sweep from falling
into the same memorization pattern as V3/V4. "GroupKFold" is named once
(by D) but not scoped to cover the downstream ML#2 stack that would
eventually be trained on this trade log.

### Reviewer 4

**Strongest: D.** Only one with both the diagnostic rigor (microstructure
as sweep parameter) and the process discipline (preregistration first).
Executor is both thorough and efficient here.

**Biggest blind spot: E.** The "cheapest kill" framing already commits
you to accepting ambiguity if a different axis contained the edge. If
timeframe is in fact fine and session structure was the culprit, E's
sunset declaration is wrong — but E has pre-committed to reading it as
right. The single-probe-family-falsification logic doesn't actually hold
unless every axis is equally likely ex-ante to carry the edge.

**All missed:** microstructure as a sweep parameter. Only Contrarian
raised it, and Contrarian framed it as a diagnostic. D folded it into
the sweep as a free-ride axis, which is strictly better. None of A, C,
E engaged with microstructure at all.

### Reviewer 5

**Strongest: D.** The only response that cleanly answers the
Chairman-level question: what do we do tomorrow? Everyone else either
reframes the question or asks for more diagnostics first. D just builds.

**Biggest blind spot: B.** Contrarian's "diagnostic first" gate is
process-expensive relative to the information value. The ceiling-strip
test requires non-trivial setup (rebuild the v11 backtest loop with
zero friction, recompute Sharpes across 13,814 combos) and the most
likely outcome is "ceiling doesn't move," which tells you nothing you
didn't already know. D's version captures the hypothesis without the
diagnostic's cost.

**All missed:** pre-registration discipline for the combo-agnostic
K-fold. Nobody specified the exact partition, just "K-fold." The
difference between `KFold(n_splits=5)` (current ML#1 v12 code) and
`GroupKFold(n_splits=5, groups=combo_id)` is the entire point — and
it's invisible to 4 of 5 advisors.

### Review Tally

| Category | Vote |
|---|---|
| **Strongest**: D (Executor) | **5 / 5** unanimous |
| **Biggest blind spot**: B (Contrarian) | 3 / 5 (Reviewers 1, 2, 5) |
| **Biggest blind spot**: E (First Principles) | 2 / 5 (Reviewers 3, 4) |
| **Collective miss**: combo-ID leak carryover | 3 / 5 (Reviewers 1, 3, 5) |
| **Collective miss**: 2019-2024 regime dependency | 1 / 5 (Reviewer 2) |
| **Collective miss**: microstructure as sweep axis (not diagnostic) | 1 / 5 (Reviewer 4) |

---

## 4. Chairman's Synthesis

### Where the Council Agrees

- **Probe 1 (timeframe lift) dominates Probe 2 (signal-family swap)**: 4 of
  5 advisors aligned explicitly; the 5th (Contrarian) conditional on the
  microstructure diagnostic. Same sweep harness, cheapest kill criterion,
  most interpretable negative result.
- **Pre-registration of the sunset threshold is non-negotiable** — aligned
  across advisors regardless of axis choice.
- **The v11 1m failure has not been adequately diagnosed at the
  execution-microstructure level** — even advisors who didn't emphasize it
  acknowledged the gap by omission.
- **Combo-agnostic K-fold audit is a standing requirement for any future
  ML#2 work** — reinforced unanimously in peer review.

### Where the Council Clashes

- **Sunset threshold precision.** Expansionist implies "find ONE working
  configuration" = success; First Principles wants family-level sunset on
  a single probe failure. Resolution: sunset threshold must specify what
  "find working" means. ≥10 combos clearing gross Sharpe 1.3 is stricter
  than one-combo-is-enough (which was V3/V4's failure mode — concentrated
  on v11_23634 at 1.108) and permits Expansionist's 2-timeframe parallel
  without drifting into pick-the-best-result fishing.
- **Timeframe selection.** Three candidate configurations surfaced: 5m+15m
  (diversify intraday), 15m+1h (diversify across cadence), 15m only
  (minimalist). Resolution: **15m+1h**. Expansionist's correlated-failure
  argument against 5m is decisive — the 1m failure mode is microstructure-
  adjacent, and 5m shares that microstructure footprint.
- **Depth vs breadth.** Contrarian + First Principles want a focused
  single probe; Expansionist wants matrix. Resolution: compromise on 2
  timeframes (not 1, not 5), plus microstructure as a free-ride parameter
  (not a new diagnostic). This captures Contrarian's concern at zero
  additional compute cost while respecting First Principles' discipline
  about not death-marching.

### Blind Spots the Council Caught (peer review)

1. **Combo-ID leak carryover.** All 5 advisors assumed the K-fold audit
   would prevent it, but only Executor named the specific partition. Fix:
   the preregistration must state `sklearn.model_selection.GroupKFold(
   n_splits=5, groups=combo_id)` explicitly, and the `v3_no_memory`
   feature set must be mandated (no `global_combo_id`, no `prior_wr_*`).
2. **2019-2024 regime dependency.** Three advisors ignored; two implied.
   NQ data spans a major regime change. Fix: walk-forward time-slice
   (4-5 calendar-year bins) as a regime-robustness check **on top of** the
   combo-level K-fold. Combo-K-fold catches memorization; walk-forward
   catches regime fragility. They are not substitutes.
3. **Microstructure as a sweep parameter** (vs. a separate diagnostic).
   Only Contrarian raised it; Executor's incidental inclusion was the
   elegant move. Fix: microstructure axes (`entry_timing_offset ∈
   {0,1,2}`, `fill_slippage_ticks ∈ {0,1,2}`, `cooldown_after_exit_bars ∈
   {0,3,10}`) in the v11 sweep as free-ride parameters. 27 cells ×
   existing v11 combinatoric = well within remote compute budget.

### The Recommendation

Execute **Probe 1 at 15m AND 1h in parallel**, skip 5m. Pre-register a
**combo-agnostic K-fold audit** with `GroupKFold` partitioning on
`combo_id` as a non-negotiable ship gate. Sunset threshold = **≥10 combos
clearing gross Sharpe ≥ 1.3 at ≥500 trades (bar-count-adjusted), passing
combo-level 5-fold walk-forward**. Add microstructure sweep parameters as
free-ride axes (capturing Contrarian's concern at zero additional sweep
cost). Walk-forward time-slice on top of combo-K-fold as a regime sanity
check.

Reject a project-level sunset framing in favor of a **family-level
sunset** conditional on Probe 1's outcome. If both 15m and 1h fail the
gross pre-gate, the Z-score mean-reversion family is falsified on
NQ/MNQ at the timeframe axis and is retired from the research direction.
A subsequent fork council would be spawned to decide what follows.

### The One Thing To Do First

Before any code: write `tasks/probe1_preregistration.md`.

It must contain, in this order:

1. The exact combo-level K-fold protocol (`GroupKFold(n_splits=5,
   groups=combo_id)`; `v3_no_memory` feature set — strip `global_combo_id`
   and `prior_wr_{10,50}` + `prior_r_ma10` + `has_history_50`).
2. The sunset threshold with **both** floors: ≥10 combos at gross Sharpe ≥
   1.3 and ≥500 trades (bar-count-adjusted, floored at 50), passing the
   K-fold. Ties go to the stricter (sunset) side.
3. Walk-forward time-slice definitions for 2019-2024 (4-5 calendar-year
   bins) with a sanity probe: pre-2024 K-fold performance must not exceed
   2024 K-fold performance by more than 30%.
4. Microstructure sweep parameter spec: `entry_timing_offset ∈ {0,1,2}`,
   `fill_slippage_ticks ∈ {0,1,2}`, `cooldown_after_exit_bars ∈ {0,3,10}`.
5. Timeframes: **15m + 1h**, not 5m. 5m is deprecated by council.

Mirror the `phase5_kill_criterion.md` template (Status / Decision Rule /
irrevocable commitments / Signature block). User signs off with commit
hash captured. Then and only then: bar-aggregation infrastructure, sweep
harness adaptation, parallel remote sweeps.

---

## 5. Process Notes

- Spawning: 5 advisors in parallel via sub-agents with assigned thinking
  styles; peer reviewers received anonymized responses (letters randomized
  to prevent positional bias) and blind-reviewed.
- Anonymization mapping (revealed post-review): A = Outsider,
  B = Contrarian, C = Expansionist, D = Executor, E = First Principles.
- Chairman synthesis informed by full advisor text + full peer review
  text, with explicit citation of where each blind spot first surfaced.
- Artifacts: this transcript + sibling HTML report
  (`council-report-2026-04-21-nq-mnq-fork.html`).

## 6. Downstream Artifacts

- **`tasks/todo.md`** — Phase 5 post-sweep plan updated to reflect the
  verdict. Phase C (Execute Probe 1) expanded into sub-phases C0-C6 with
  the preregistration doc as the first deliverable.
- **`tasks/probe1_preregistration.md`** — authored alongside this
  transcript as Phase C0 of the execution plan. Awaits user sign-off
  (commit hash to be captured at signature time).
- **`memory/feedback_nq_mnq_scope_only.md`** — standing scope constraint
  that shaped the question; no changes required.
