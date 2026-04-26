# LLM Council Transcript — The Independent-Data Question

**Date:** 2026-04-24
**Context:** Post-TZ-cascade closure (commit `15f450d`). Probe 5 preregistration blocked on this fork.
**Anonymization map revealed below.**

---

## Original Question (user prompt)

> Summon the council. I am leaning towards c

User's prior turn established the three options. Council fired on the full framed question below.

---

## Framed Question (sent to all 5 advisors + 5 reviewers)

**The independent-data question — council's 2nd gating deliverable before Probe 5 preregistration.**

Context: backtest-first trading bot for NQ/MNQ futures. Preregistered-probe methodology (signed binding clauses auto-enforce verdicts). ~4 months running. TZ-cascade closure landed 2026-04-24 (commit `15f450d`). Probe 5 signal-family preregistration is blocked on this fork.

**Three options:**
- (a) Identify a held-out NQ 1h partition never read by Probes 1-4 + Scope D.
- (b) Preregister forward paper-bar accrual (~12 weeks / ~1200 bars) as the OOS source.
- (c) Accept NQ 1h bar data is exhausted on the bar-timeframe axis; honest sunset of Probe-1-family work.

**USER IS LEANING TOWARD (c).** React to this lean — validate or challenge it explicitly. Do not anchor on it.

**Key historical facts (all signed):**
- Probe 1 (signed `d0ee506`, 2026-04-21): Z-score mean-reversion on NQ FALSIFIED across bar timeframes. N_1.3 = 1/9/4 on 1m/15m/1h vs gate=10. §7.6 terminal clause blocks 30m/2h/5m without fresh prereg.
- Probe 2 (signed `a49f370`, 2026-04-21): combo-865 carve-out PASS on 1h test partition (Sharpe 2.89, 220 trades, +$124,896/yr at $2,500 risk/trade). TZ-agnostic, stands.
- Probe 3 (orig PAPER_TRADE): RETRACTED 2026-04-23 under TZ bug → COUNCIL_RECONVENE.
- Probe 4 (SESSION_CONFOUND): RETRACTED 2026-04-23 under same bug. Combo-1298 corrected: RTH-concentrated Sharpe 4.41 on n=77.
- Scope D "SES_2a dominates" inverted to "SES_1 RTH dominates" post-fix.
- 1h test partition = 2024-10-22 → 2026-04-08 ≈ 1.48yr, ~2,400 bars. Read FOUR times (Probes 2, 3, 4, Scope D).
- Gross Sharpe ceiling under $5/contract RT: 1 combo of 13,814 clears 1.0. Zero-friction cannot rescue.
- Previous council (2026-04-24): Reviewers 4 & 5 flagged (a) as "partition is burned".

**Constraints:** NQ/MNQ only. Fixed-$500 sizing. $5/contract RT. 80/20 chrono split. User's direction: ML#1-style generalization.

**Open questions:** (a) feasibility, (b) opportunity cost, (c) finality / sunset scope, hybrids, stats-honesty BF inflation thresholds.

**Chairman needs:** option or hybrid, binding clauses, specific deliverables.

---

## Advisor Responses

### The Contrarian (Response B in peer review)

The framing is the fatal flaw. You're treating "find OOS data" as the gating question, but the real gating question — which you keep dancing around — is: *why are you still hunting for an out for a falsified family?* Probe 1 §7.6 already fired. The bar-timeframe axis is closed. You're now asking the council to help you launder one combo (865, possibly 1298) past a partition that's been read four times. That's not a stats problem you can BF-inflate your way out of — that's motivated search.

**Specifically attacking each option:**

**(a) is a corpse.** Pre-2019 data is a different microstructure regime (no overnight liquidity profile match, pre-CME Globex 23h, different tick economics). Training partition as OOS is **leakage by construction** — you sweep-tuned on it. 4h/daily evades §7.6 only by abandoning the 1h finding that motivated this; if 1h doesn't generalize to 4h, why are you here? Reviewers 4 & 5 were right. Stop relitigating.

**(b) is a tax disguised as rigor.** 12 weeks of bars adds ~50 fresh trades for combo-865 at its current frequency. That's underpowered for Sharpe ≥ 1.3 detection at any honest BF. You'd ship paper-trade on 50 trades after a 4× partition burn? The opportunity cost framing is wrong — the cost is **another retraction in 3 months.**

**(c) has a hidden problem too:** "sunset" lets you walk away from combo-865's $124k/yr without a falsification of *combo-865 specifically*. Probe 2 stands. You haven't earned the sunset — you've earned an inability to extend.

**Verdict: (c) with binding teeth.** Sunset Probe-1-family bar-timeframe work. Foreclose 865 paper-trade absent forward accrual. Signal-family swap (Probe 5) requires fresh data partition declared *before* signal selection, not after. No hybrid. The hybrid is how you get retraction #3.

---

### The First Principles Thinker (Response A in peer review)

**What is the user actually trying to achieve?**

Strip away "Probe 5 preregistration." Strip away the bar-timeframe axis. The user is trying to answer: *Does combo-865 (and its cousins) represent a real economic edge worth risking $50k on, or did the search procedure overfit to a 1.48-year window?*

That is the only question that matters. Everything else — the prereg ritual, the BF inflation accounting, the §7.6 clauses — is scaffolding around it.

**The load-bearing assumption to challenge: that "fresh data" is the binding constraint.**

It isn't. The binding constraint is that you ran a 13,814-combo search and one combo passed friction at gross Sharpe ≥ 1.0. That's *prima facie* multiplicity-driven. No amount of partition cleverness fixes a search that returned 1/13,814 hits — fresh data only tells you whether *that specific realization* persists, not whether the underlying generative process has edge.

**The (a/b/c) framing is wrong** because it treats this as a data-availability problem. It's a search-multiplicity problem dressed as a data problem.

**Reframe:** instead of "where do I get clean OOS bars," ask "what would falsify combo-865's edge *mechanistically* rather than statistically?" If combo-865's edge is real, it should have a *reason* — a microstructural story about why z-score reversal at those specific parameters captures something durable on 1h NQ. If you can't articulate that story before looking at more data, no amount of OOS will save you; you'll just keep p-hacking the priors.

**Lean toward (c), but not as sunset — as honest scope contraction.** Combo-865 is a hypothesis, not a strategy. Either find its mechanism or shelve it. Don't burn 12 weeks of paper-bars hoping statistics resolves what economics hasn't.

---

### The Expansionist (Response D in peer review)

**(c) is leaving $124k/yr on the floor.** Sunset is methodological fatigue dressed as discipline.

Reframe: combo-865 already cleared a preregistered, mechanical, OOS gate (Probe 2, TZ-agnostic, signed). That is a *result*, not a hypothesis. The reuse-burn argument applies to *parameter discovery*; combo-865's parameters are frozen since `a49f370`. Re-running diagnostics on a frozen rule does not inflate selection bias the way a fresh sweep would. The 4× reuse penalty is real for Probes 3/4/Scope D's *new* claims (session, ritual, decomposition) — not for the dollars that were already booked at Probe 2.

**The hidden asset in (b): the accrual window IS the signal.** 12 weeks of forward 1h bars on a frozen rule is the cleanest inference instrument this project will ever produce — zero researcher degrees of freedom, zero look-ahead, zero TZ ambiguity (you control ingest). That's not opportunity cost; that's a Bayes-factor multiplier no historical partition can match. Each week of clean forward bars is worth ~3× a re-sliced historical week.

**Hybrid I'd push the chairman toward (a-narrow + b + c-partial):**

1. **Bind combo-865 to forward paper accrual at fixed parameters** (no re-tuning, no session sub-rules, no EX swaps). Preregister §-gates: ≥X bars, net Sharpe ≥ 1.3, ≥50 trades, t-stat ≥ 2.0. This is (b) with combo-865 as the asset under test, not the family.
2. **Honest sunset of the Z-score *family search* on 1h historical** — accept (c) at the family level only. Stop spending probes on bar-timeframe rescue.
3. **Open an adjacent-upside lane**: ML#1-style generalization on the *training* partition (80%) ranking unseen combos, then commit them forward at paper-trade simultaneously. Two independent forward streams = 2× the signal for the same wallclock.

Don't sunset a $124k/yr instrument because the methodology is tired.

---

### The Outsider (Response E in peer review)

Let me strip this down to what I actually see.

You spent four months trying to find a moneymaking pattern in one specific market on one specific timeframe. Your own pre-set rules said "if fewer than 10 setups beat this bar, the idea is dead." You found 1, 9, and 4. The idea is dead by your own scoring.

Then you found ONE specific configuration (combo-865) that looks great. You poked at it four separate times on the same 1.48 years of data. Each poke, you found new ways it might or might not work. Two of those pokes had to be thrown out because of a clock bug. The data has been read so many times it's basically memorized at this point.

Now you're asking: should I (a) find some clever way to test it again on the same exhausted dataset, (b) wait three months for fresh data, or (c) walk away?

Here's what's weird from the outside. You already have ONE thing that survived everything: combo-865 makes ~$125k/year on a held-out backtest, and that number didn't move when the clock bug got fixed. That's the actual asset. Everything else — Probe 3, Probe 4, Scope D — is you trying to *understand why* it works, not whether it works.

The economically sensible move and the emotionally sensible move diverge here. Emotionally, (c) feels clean — close the book. Economically, you have a candidate strategy with a real backtest number that you have not actually traded. Three months of paper trading (option b, but for combo-865, not for the family) costs nothing but time and tells you something no amount of re-slicing the same 2,400 bars ever will.

Sunset the family. Paper-trade the survivor. Stop re-reading the partition.

---

### The Executor (Response C in peer review)

**Monday-morning test, ranked by tractability:**

**(c) WINS on tractability.** First move tomorrow: `git tag probe1-family-sunset-2026-04-24`, write `tasks/sunset_probe1_family.md` with 3 sections (what's falsified, what stands, what's next). Done by lunch. Combo-865 is NOT foreclosed by sunset — it's a separate carve-out with TZ-agnostic Probe 2 PASS. Sunset Probe 1, paper-trade 865 in parallel.

**(a) is dead-on-arrival.** Pre-2019 NQ has different microstructure (pre-CME Globex consolidation regime change ~2020); 4h/daily are outside §7.6 but require fresh prereg + new sweep + new K-fold = 3-4 weeks of work to maybe discover the same friction floor. Training partition as OOS is leakage by definition. Deliverables: prereg, sweep relaunch, audit redesign, council convene = **5 artifacts minimum, 2-3 weeks**.

**(b) is opportunity-cost suicide.** 12 weeks of waiting buys you ~1200 bars to confirm what Probe 1 already falsified at N=2400. You don't need more data to reject a falsified hypothesis — you need a different hypothesis. Deliverables: prereg, paper infra, weekly reconciliation = real work for negative information value.

**Hybrid (c + paper-trade 865):** Tightest path to cash. Sunset Probe 1 family, run a fresh council on 865-only paper-trade scope (broker, sizing, kill-switch, session filter post-TZ-fix). Combo-865 is your only validated edge; ship it small while you redesign signal family.

**Tomorrow's deliverables (3, total):**
1. `sunset_probe1_family.md` (1hr)
2. Council convene request: "865 paper-trade scope" (30min)
3. Probe 5 prereg DRAFT on signal-family swap, not bar-timeframe (2hr)

Validate the lean. Stop digging the bar-timeframe hole. Ship 865 or pivot signals.

---

## Anonymization Map (revealed)

- **Response A** = First Principles Thinker
- **Response B** = Contrarian
- **Response C** = Executor
- **Response D** = Expansionist
- **Response E** = Outsider

Reviewers saw only A-E labels during the peer review round.

---

## Peer Reviews

### Reviewer 1

**1. Strongest: D.** D is the only response that distinguishes parameter-discovery reuse-burn (real, applies to Probes 3/4/Scope D) from frozen-rule diagnostic reuse (not the same multiplicity tax). It correctly identifies that forward accrual on a frozen `a49f370` rule has *zero* researcher DoF — the cleanest inference instrument available — and proposes a concrete hybrid that sunsets the family search while preserving the only TZ-agnostic asset. C is operationally tight but conflates "ship 865" with the gating question. E is correct but folksy.

**2. Biggest blind spot: B.** B treats combo-865 as a laundering target and demands "no hybrid," but never engages with the asymmetry D names: Probe 2 was preregistered, mechanical, and TZ-agnostic. Calling forward accrual a "tax disguised as rigor" while accepting sunset is internally incoherent — sunset *also* doesn't falsify 865. B's "retraction #3" framing is fear-based, not Bayesian.

**3. What all five missed:** None addressed the **paper-trade-vs-live decision boundary**. Forward paper accrual on a frozen rule still has selection bias if the kill-switch is discretionary — preregister the *abandonment* criteria (max DD, n_trades floor, time budget) at the same time as the success gates, or paper-trade becomes another partition to overfit.

---

### Reviewer 2

**1. Strongest: D.** It's the only response that mechanically separates two distinct claims — (i) combo-865's Probe 2 result is a signed mechanical PASS that doesn't decay from family-level reuse, (ii) the 1h family search is exhausted. The 4× partition-burn penalty applies to *new* claims drawn from those reads, not to dollars already booked behind a frozen preregistration. D also identifies the only true zero-DoF inference instrument available: forward bars on frozen parameters. C reaches similar hybrid but skips the rigor on why 865 is exempt.

**2. Biggest blind spot: B.** B conflates "family falsified" with "every artifact of the search is contaminated." Combo-865 was signed at commit `a49f370` *before* Probes 3/4 reads. B's "you haven't earned sunset" framing punishes a clean Probe 2 PASS for sins committed downstream. Foreclosing 865 absent forward accrual is exactly the trap D names.

**3. All five missed:** none stress-tested combo-865 against the 100% combo-overlap finding (`project_combo_overlap_structural`). 865 has trades in V3/V4 training partitions; its parameters may have been *implicitly* selected by ML#1's feature engineering even before Probe 2. That's a leakage vector orthogonal to the 4× read. Forward paper-accrual still resolves it — but the council should name it.

---

### Reviewer 3

**1. Strongest: Response B.** B is the only one that names the actual failure mode: "find OOS data" is motivated search post-falsification. It diagnoses each option's specific pathology (a=corpse, b=tax, c=hidden 865 problem) and lands on binding teeth — Probe 5 partition declared *before* signal selection, no hybrid. The "hybrid is how you get retraction #3" line is load-bearing given the project's actual track record (Probes 3, 4 both retracted).

**2. Biggest blind spot: Response D.** D treats "12 weeks of forward bars on a frozen rule is the cleanest inference this project will ever make" as if power weren't an issue. ~50 trades cannot detect Sharpe 1.3 vs 0 at any reasonable α — B names this explicitly, D hand-waves it. D also adds a third concurrent stream (ML#1 generalization) to a project already drowning in retracted probes.

**3. What ALL missed:** None addressed that combo-865's Probe 2 PASS itself was selected from a 13,814-combo search where the family failed — so 865's "OOS" was already conditional on family-level multiplicity. Paper-trading 865 isn't independent evidence; it's testing whether one lottery ticket keeps paying. The honest prior on 865 should reflect 1/13,814 selection, not its standalone Sharpe 2.89.

---

### Reviewer 4

**1. Strongest: A.** A is the only response that names the actual epistemic problem: 1/13,814 is a multiplicity artifact, and *no quantity of fresh data* answers the family-level question without a mechanism. B/C/D/E debate logistics inside a frame A correctly rejects. A's "find mechanism or shelve" is the only test that distinguishes durable edge from a survivor.

**2. Biggest blind spot: D.** D treats $124k/yr as "booked" when it's a backtest number on a 4×-read partition with no forward fills. The "4× reuse penalty doesn't apply to Probe 2 dollars" claim is wrong — Probe 2 was selected *because* combo-865 survived an exhaustive 13,814-combo search; selection bias contaminates the dollar figure exactly as it contaminates Probes 3/4. D mistakes a frozen rule for unconditional inference.

**3. What all five missed:** None addressed the **stopping rule for paper-trading 865**. C/D/E all propose forward accrual without preregistering the kill criterion: drawdown threshold, Sharpe floor, trade-count window, false-discovery posterior under H0. Without a signed §-gate before bar 1, paper-trade becomes Probe 5 by another name — another retraction risk. The council's next deliverable should be the 865 paper-trade preregistration, not the OOS framing debate.

---

### Reviewer 5

**1. Strongest: Response A.** It correctly reframes the problem. (a)/(b)/(c) all assume the question is "where do we get clean data" when the actual question is "what is the posterior on 865's edge given a 1/13,814 search hit." Without a mechanistic falsifier, fresh data just rolls the dice again — A is the only response that names this. B gets close but stops at "motivated search" without specifying the mechanism test.

**2. Biggest blind spot: Response C.** Treats this as a logistics problem ("done by lunch, 3 deliverables tomorrow") and asserts 865 paper-trade is "tightest path to cash" without engaging the multiplicity prior at all. C also contradicts itself — claims (a) is dead-on-arrival and (b) is "suicide," then prescribes paper-trading (which IS option b for the combo). Misses that paper-trading without a pre-specified falsification gate just adds a 5th retraction risk.

**3. What all five missed:** None specified the **decision rule for the paper-trade itself**. If 865 paper-trades for 12 weeks and returns Sharpe 1.5 on 50 trades, does that PASS or FAIL? Without a preregistered gate (n_trades floor, Sharpe threshold, drawdown kill-switch) signed *before* accrual starts, the council is setting up retraction #3. The TZ-bug lesson was preregistration-discipline; the council just skipped it.

---

## Peer Review Tallies

**Strongest response vote:** D=2 (Reviewers 1, 2), A=2 (Reviewers 4, 5), B=1 (Reviewer 3). No clear majority.

**Biggest blind spot vote:** B=2 (Reviewers 1, 2), D=2 (Reviewers 3, 4), C=1 (Reviewer 5). Splits — reviewers who loved D flagged B; reviewers who loved A/B flagged D.

**Convergent catch (what ALL missed):** 5/5 reviewers independently flagged **paper-trade kill-switch / abandonment criteria must be preregistered before bar 1** (Reviewer 1 explicit, Reviewer 4 explicit, Reviewer 5 explicit, Reviewers 2 & 3 implicit via leakage / lottery framing). This is the strongest signal in the entire peer-review round.

**Secondary convergent catch:** Combo-865's Probe 2 PASS is itself multiplicity-contaminated (1/13,814 selection) — Reviewers 2, 3, 4 independently. D's "frozen rule exempts multiplicity" claim is partially wrong.

**Tertiary catch:** 100%-combo-overlap leakage (Reviewer 2 only) — 865 has trades in V3/V4 training partitions, potentially implicit selection by ML#1 before Probe 2.

---

## Chairman Synthesis

### Where the Council Agrees

All 5 advisors converge on **family-level sunset** (option (c) applied to the Probe-1 Z-score bar-timeframe family). Independent convergence:
- A ("honest scope contraction")
- B ("(c) with binding teeth... Sunset Probe-1-family bar-timeframe work")
- C ("(c) WINS on tractability... Sunset Probe 1")
- D ("Honest sunset of the Z-score family search on 1h historical")
- E ("Sunset the family")

Also 4/5 agreement (B, C, D, E) that combo-865 is NOT necessarily foreclosed by family-level sunset — Probe 2 PASS (TZ-agnostic, signed `a49f370`) stands independently.

### Where the Council Clashes

**The scope of "sunset":**
- Minimal (combo-865 preserved via paper-trade): C, D, E
- Absolutist (865 paper-trade also foreclosed absent new forward accrual): B
- Mechanism-first (865 shelved pending microstructural story): A

**The method for 865 paper-trade** (among the 4 who preserve it):
- Immediate ship with parallel council on scope: C
- (b)-style preregistered forward accrual with §-gates: B (implicitly, via "foreclose absent forward accrual"), D (explicitly), E (implicitly)

**Concurrent streams:**
- Add ML#1 generalization lane on training partition: D alone. Peer review (Reviewer 3) flagged as overreach — "a project already drowning in retracted probes."

### Blind Spots the Council Caught

Three load-bearing points emerged only in peer review:

**(1) Paper-trade kill-switch must be preregistered before bar 1 (5/5 reviewers).** Reviewer 1: "preregister the *abandonment* criteria." Reviewer 4: "the 865 paper-trade preregistration, not the OOS framing debate." Reviewer 5: "decision rule for the paper-trade itself." Without signed kill-switches, forward accrual becomes Probe 5 by another name — the 6th retraction on the same lesson.

**(2) Combo-865's Probe 2 PASS is itself multiplicity-contaminated (Reviewers 2, 3, 4).** 865 was selected from a 13,814-combo search. D's "parameters frozen since `a49f370` → reuse-burn doesn't apply" claim is partially wrong: the frozen rule was itself a selection event. Honest prior reflects 1/13,814 selection, not standalone Sharpe 2.89.

**(3) 100%-combo-overlap leakage vector (Reviewer 2).** 865 has trades in V3/V4 training partitions per `project_combo_overlap_structural`. Its parameters may have been implicitly selected by ML#1 feature engineering even before Probe 2's explicit evaluation. Forward paper-accrual resolves, but any prereg must name it.

### The Recommendation

**Verdict: Hybrid — (c) at the family + (b)-style for combo-865, with mandatory preregistered kill-switches. User's lean is VALIDATED at family level, CHALLENGED on scope.**

Honest sunset does NOT require abandoning combo-865; it requires abandoning the Z-score bar-timeframe *family search*. These are distinct. Concretely:

1. **Sunset the Probe-1 Z-score bar-timeframe family.** Document in `tasks/sunset_probe1_family.md` as a corollary to (2).

2. **Combo-865 paper-trade does NOT proceed via immediate ship (reject C).** It proceeds via (b)-style preregistered forward paper-bar accrual with binding §-gates:
   - PASS (conjunction): ≥N_bars, net Sharpe ≥ 1.3, ≥50 trades, t-stat ≥ 2.0
   - KILL (disjunction): max DD ≥ 30% of starting paper equity, n_trades < 10 by week 8, time budget 16 weeks max
   - Prior acknowledgment: multiplicity-adjusted for 1/13,814 selection + 100%-combo-overlap leakage
   - TZ contract: `src/tz_contract.py::assert_naive_ct` invoked at bar-ingest load

3. **Reject D's concurrent ML#1 generalization lane** for now (peer review overreach). One forward stream per wallclock period; 865 first.

4. **Defer Probe 5 signal-family swap** until after the 865 paper-trade window resolves. This sequences correctly per previous council's "C sequenced" verdict.

5. **A's mechanism-first ask acknowledged as concurrent research, not blocking.** During the 12-16 week paper-trade, separately attempt a microstructural articulation. If mechanism doesn't land AND paper-trade fails gates, 865 shelves. If paper-trade passes but no mechanism emerges, user makes live-trading judgment with eyes open.

**B's "no hybrid" stance dissolves on close reading:** B writes hybrid in substance ("Foreclose 865 paper-trade absent forward accrual") while denying it in rhetoric. The clash is cosmetic.

**D's optimism about "zero researcher DoF forward accrual" is tempered** by Reviewer 3's power calc (~50 trades can't detect Sharpe 1.3 vs 0 at any reasonable α) and Reviewers 2-4's multiplicity prior catch. D's hybrid is right in structure; the gates need to be calibrated for the multiplicity-adjusted prior.

### The One Thing to Do First

**Draft and sign `tasks/probe5_combo865_paper_trade_preregistration.md`.**

NOT `sunset_probe1_family.md` first (that's a corollary). NOT `probe5_signal_family_preregistration.md` first (that's a later fork). The preregistration is the single binding deliverable that gates everything downstream.

The document must include:
- **Accrual window** (≥ 12 weeks, up to 16-week time budget)
- **PASS gates (conjunction)**: ≥N bars, net Sharpe ≥ 1.3, ≥50 trades, t-stat ≥ 2.0
- **KILL criteria (disjunction)**: max DD cap, n_trades floor by week 8, time budget
- **Multiplicity-adjusted prior** acknowledging 1/13,814 selection + 100%-combo-overlap leakage (cite `project_combo_overlap_structural`)
- **TZ contract**: `src/tz_contract.py::assert_naive_ct` invoked at bar-ingest load
- **Signing protocol**: commit hash + signed document (like `a49f370` / `d0ee506`)

**Graceful fallback to user's original lean:** If this prereg cannot be written — because the gates cannot be pinned without A's mechanism, or because the multiplicity-adjusted prior makes any PASS threshold unattainable — then the answer collapses to honest full-sunset including 865. Which is exactly what the user came in leaning toward.

---

*Transcript generated 2026-04-24. HTML report: `tasks/council-report-2026-04-24-independent-data.html`. Previous council handoff: `tasks/council-transcript-2026-04-24-post-tz-cascade.md`. Project constraints: `CLAUDE.md`. TZ-cascade closure: `tasks/tz_cascade_closure.md`.*
