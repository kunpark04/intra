# LLM Council Transcript — Post-TZ-Cascade Fork

**Date:** 2026-04-24
**Project:** NQ/MNQ Z-score mean-reversion
**Trigger:** Multiple signed probe verdicts routed to `COUNCIL_RECONVENE` after 2026-04-23/24 TZ retraction cascade; no auto-follow-up rule exists
**Artifacts:** [`council-report-2026-04-24-post-tz-cascade.html`](council-report-2026-04-24-post-tz-cascade.html)

---

## The Original Question

> Launch council — what is the next preregistered action after the 2026-04-23/24 TZ retraction cascade?

---

## The Framed Question (sent to all 5 advisors)

**Decision:** Choose a mechanical action from:
- **(A)** Draft paper-trade preregistration on combo-865 1h despite Probe 3 F=1
- **(B)** Draft fresh preregistration for a combo-1298 RTH-carve-out probe
- **(C)** Declare Branch A sunset per Probe 1 §7.6 and pivot to signal-family swap within NQ/MNQ
- **(D)** Some bounded alternative within NQ/MNQ-only scope

**Hard constraints:** NQ/MNQ-only scope; Probe 1 §7.6 terminal (no 30m/2h/5m without fresh preregistration); remote-execution-only on sweep-runner-1; preregistration mandatory for every new probe.

**Current factual state (post-pipeline-review `40bc05d`, 2026-04-24):**

*What stands (TZ-agnostic):*
- Probe 1 Branch A falsification on bar-timeframe axis: N_1.3 = 1/9/4 combos at 1m/15m/1h vs gate of 10. Signed `d0ee506`.
- Probe 2 combo-865 1h PASS: net Sharpe 2.89 on 220 test trades, +$124,896/yr at 73 contracts. Session-agnostic, stands. Signed `a49f370`.
- Probe 4 absolute gates (TZ-agnostic): combo-1298 full-session Sharpe 3.55 on n=250; combo-664 Sharpe 1.34 on n=910; Welch-t 3.851.
- Gross Sharpe ceiling on NQ v11 pool: 1.108 (one combo, v11_23634) across 13,814 combos under $5/contract RT. Zero-friction cannot rescue.

*What retracted under TZ fix (2026-04-23):*
- Probe 3 PAPER_TRADE (signed `b68fe62`) → RETRACTED. Corrected: §4.3 15m NC rescue fires (0/16 PASS → 8/16 FAIL); §4.4 1h ritual strengthens (8/16 → 12/16). F=0→1. §5.2 binding fires: COUNCIL_RECONVENE.
- Probe 4 SESSION_CONFOUND (signed `c419391`) → RETRACTED. Combo-1298 RTH Sharpe 4.41 on n=77; SES_2 Sharpe 0.265 on n=46 (fails n≥50). Row 4 fires: COUNCIL_RECONVENE. B1 session-structure sweep authorization withdrawn.
- Scope D: combos diverge. 865 RTH-leaning (2.64 vs 1.47); 1298 strongly RTH-concentrated (4.41 vs 0.53); 664 weakly overnight-leaning.

*Multiplicity exposure:*
- Stage 1: 1500-combo pool at 1h; combo-865 chosen post-hoc.
- Stage 2: 1h test partition read **4 times** (Probes 2, 3, 4, Scope D).
- Combo-1298 RTH Sharpe 4.41 sits on n=77 (1500-combo selection × 4-reuse discount).

*ML-stack status (context only):*
- ML#2 V3 and V4 both revoked 2026-04-21 for combo-ID memorization leak.

**Signed rules the council cannot override:**
- Probe 3 §5.2: F=1 → COUNCIL_RECONVENE (firing).
- Probe 4 §5 row 4: COUNCIL_RECONVENE (firing).
- Probe 1 §7.6 terminal: no intermediate-timeframe workarounds without fresh preregistration.

**Three questions each advisor must address:**
1. **Probe 3 interpretability** — does the post-TZ-fix state support any action-level conclusion beyond mechanical §5.2 re-convene? §4.4 strengthened (8→12/16); §4.3 rescue fires on negative control. Principled override criterion?
2. **Combo-1298 RTH fresh carve-out** — scientifically sound given 1500-combo selection + 4× partition reuse + n=77 variance? Is a 5th re-read defensible?
3. **Admissible-probe space enumeration** — rank NQ/MNQ axes (signal family, session, exit, sizing) by (i) preregistration-able? (ii) prior P(H1)? (iii) cost?

---

## Anonymization Map (revealed)

| Letter | Advisor |
|---|---|
| Response A | The Executor |
| Response B | The Contrarian |
| Response C | The Outsider |
| Response D | The First Principles Thinker |
| Response E | The Expansionist |

Peer reviewers saw only the letters.

---

## Advisor Round (5 parallel sub-agents)

### The Contrarian — Verdict: C (with pre-probe methodological audit)

**The fatal flaw nobody is naming: this project has a compounding multiplicity problem that every proposed action makes worse.**

**On Q1 — Probe 3 "§4.4 strengthening":** This framing is a trap. You cannot cherry-pick the gate that got better and ignore the gate that got worse. §4.3 was a *negative control* — its entire purpose was to fail. It didn't fail; it rescued. That is not "informational," it is a direct contradiction of the Probe 1 15m falsification that underwrites Branch A. You cannot simultaneously hold "15m is falsified" and "15m can be recovered via session/exit combos" without admitting the gating framework is unstable under TZ perturbation. §4.4's 12/16 is also suspicious — a mechanical gate that moves from exactly-at-threshold to comfortably-clear after a timestamp rewrite is not robust, it's *sensitive to labeling noise*. A signed §5.2 binding exists precisely to stop post-hoc rationalization. Overriding it requires a pre-declared amendment protocol. None exists. Do not override.

**On Q2 — Combo-1298 RTH:** n=77. Let me say that again: **n=77**. Selected from a 1500-combo pool, on a test partition that has now been read four times, after a TZ retraction that inverted the prior session conclusion. Stage 1 multiplicity discount ~log(1500), Stage 2 discount ~4× selective-reporting penalty, Stage 3 small-sample variance on 77 trades. Welch-t 3.851 sounds impressive until you apply a Bonferroni correction for the number of (combo, session) cells implicitly searched — which is at least 1500×3. A hypothetical 4-gate probe with typical P(PASS|H0) ≈ 0.05 per gate compounds to maybe 0.01 jointly under H0, but P(PASS|H1) collapses because the effect you'd be testing is *already* inside the multiplicity-inflated estimate. BF will look strong and mean nothing. **A 5th read of the 1h partition is indefensible.** The partition is burned. If you want 1298, you need fresh data — which means waiting.

**On Q3:** Signal-family swap is the only axis where prior P(H1) hasn't been destroyed by the 1.108 gross ceiling + family falsification. Session/exit/sizing all operate downstream of a signal family that is dead. Stop trying to resurrect mean-reversion on NQ.

**The uncomfortable fourth option nobody is saying out loud: pause execution entirely and do a methodological audit before any new probe.** The TZ bug cascaded across four probes because no one validated source timestamps — what other assumptions are unvalidated?

**Verdict: C**, but only after a pre-probe methodological audit. B is statistical suicide. A violates signed bindings. Do not draft paper-trade on a retracted probe.

---

### The First Principles Thinker — Verdict: C

You're asking "which probe next?" when the real question is: **"Have we accumulated enough evidence to conclude that the NQ bar-data regime does not contain a mechanical edge discoverable by post-hoc parameter selection, and if so, what's the exit plan?"**

Strip away the ceremony and look at what you actually have:

1. **Probe 1 falsified the family** at the gate level (N_1.3 = 14 combined vs gate of 10 per-timeframe).
2. **Probe 2 survived on one combo at one TF** — a carve-out, acknowledged as not generalizing.
3. **Probe 3 had to be retracted** because your own scripts mislabeled timezones.
4. **Probe 4 had to be retracted** for the same reason, and its "structural" finding (overnight dominance) inverted.
5. **Both ML#2 stacks (V3, V4) were revoked** for combo-ID memorization.

That's five mechanical falsifications/retractions in <2 weeks. The pattern isn't "we keep almost finding something" — it's **"every result that looked affirming dissolved under audit."** The combos that survive absolute-gate checks (865, 1298, 664) sit on post-hoc selection from 1500-combo pools with multiple partition re-reads.

Now look at what you're being asked to choose:

- **(A)** ships on a probe whose signed rule says RECONVENE. Violates your own prereg discipline.
- **(B)** does a 5th read of the same 20% test partition on a combo that scored Sharpe 4.41 on **n=77**. The variance on n=77 is large; 1500×4 multiplicity after correction collapses any nominal p-value. This is drawing from the same well that just poisoned you twice.
- **(C)** honors the signed Probe 1 §7.6 binding and moves to a genuinely new hypothesis space.

The first-principles question: **what's the prior P(H1) that a discoverable 1h edge exists on NQ given the accumulated evidence?** It was ~0.167 going into Probe 3. After two retractions showing your measurement apparatus was broken, plus gross-Sharpe ceiling of 1.108 across 13,814 combos under documented friction, that prior should be *lower*, not higher. Yet B and A treat it as effectively higher by proposing more extraction from the same partition.

The information-value calculus:
- **B**: expected info ≈ noise; failure mode already demonstrated.
- **C**: burns a new data partition or new instrument-equivalent axis, forces a genuinely independent test.
- **A**: not a probe, a violation.
- **D (carve-out to D)**: only defensible if it's a signal-family swap *with held-out data never read before* — i.e., de facto C.

You're not running out of combos. You're running out of **independent data**. Every additional read of the 20% test partition compounds the selection discount until posteriors are uninterpretable. The correct action is to stop extracting and pivot.

**Verdict: C.**

---

### The Expansionist — Verdict: D (session-restricted Probe 1 redux + combo-1298 embedded)

Everyone's going to crowd around the multiplicity penalty and miss what the corrected TZ just handed us: the dataset has a **genuine mechanical signal** hiding in plain sight. Two independent combos (865 and 1298) both lean RTH-concentrated after the fix. That's not noise — that's a **session-structural feature of NQ liquidity** that the family-level Probe 1 sweep averaged away because it ran session-agnostic. Probe 1's N_1.3=4 on 1h was computed over the *whole session*. Nobody has run N_1.3 restricted to RTH-only 1h bars. That number could be dramatically different, and it's the single preregistrable experiment that could unfalsify the family at the axis §7.6 already permits (session, not timeframe).

**The undervalued asset:** combo-1298 at Sharpe 4.41 on n=77 RTH is being framed as a multiplicity hazard. Flip it. n=77 at that effect size implies **~$1,900 per trade net** in a liquidity window that broker/microstructure risk is *lowest*. If the true RTH Sharpe is even half that (2.2), annualized on a ~150-trade/yr RTH cadence scaled across a 5-combo basket, you're looking at **$400–600k/yr on $50k starting equity** — a 10× outcome the base-case is not pricing.

**The adjacent opportunity nobody's naming:** the TZ bug itself is a gift. It means every prior Probe-1-style gross-ceiling read was run on mislabeled session bars. The "1 of 13,814 gross Sharpe ≥ 1.0" ceiling was computed against session-confounded data. Re-running the v11 gross ceiling on **TZ-corrected RTH-only 1h bars** is cheap (~4 CPU-hours remote), fully preregistrable, and could reveal the family isn't binding — it's *session-binding*. If N_1.3(RTH-1h) ≥ 10, the entire Branch A sunset premise collapses into a signal-structure victory.

**Upside decision today:** don't pick between A/B/C. Pick a preregistration that does **both** in one remote job — a session-restricted family-wide gross-ceiling re-read (Probe 1 redux, RTH-only, 1h) *and* an out-of-sample validation window for combo-1298 RTH. If the family re-read passes, 1298 is no longer a 5th re-read — it's a confirmed member of a rediscovered family. If it fails, we still have clean evidence on 1298 and a defensible pivot to signal-family swap with priors updated.

**Mechanical verdict: D** — preregister a **session-restricted Probe 1 redux (RTH-only, 1h, TZ-corrected)** with combo-1298 as an embedded confirmatory cell on a disjoint time window. Single job, single signature, maximum optionality preserved.

---

### The Outsider — Verdict: C

Let me say what I'm seeing without the jargon.

You ran thousands of parameter combinations on a trading strategy. Out of 13,814 attempts, exactly ONE cleared a gross performance bar before costs. After costs, ONE specific combination (865) looked good on a specific test window. You then designed four probes to validate it. Probe 1 said the strategy family is broken. Probe 2 said "but this one works." Probe 3 said "yes paper-trade it." Then you found a timezone bug. Now Probe 3 says "reconvene" and Probe 4's story inverted — the edge isn't where you thought it was, it's somewhere else. And the new "somewhere else" (combo-1298 RTH, Sharpe 4.41 on 77 trades) looks amazing.

Here's the dumb question: **why does the amazing thing keep moving?**

Every time you look at this test partition, a different story emerges. Overnight dominates — no wait, RTH dominates. Combo-865 is the carve-out — no wait, combo-1298 is stronger. Each re-read on the same 2024-10-22 → 2026-04-08 window is another multiple comparison. You're at read #4. The headline number (Sharpe 4.41) rests on 77 trades. That's nothing. Seventy-seven coin flips.

On Question 1: "Can the user override the mechanical rule because §4.4 got stronger?" No. You wrote the rule before you saw the data specifically so you couldn't do this. The §4.3 control was designed to kill a spurious 15m signal. It now *fails*. That means your negative control is telling you the 15m signal is rescuable — which means your probes can't distinguish real edge from session/exit tweaking. That's a *worse* outcome, not a better one, even if §4.4 looks prettier. Overriding because the pretty number got prettier is textbook motivated reasoning.

On Question 2: A 5th read of the same partition isn't science, it's data-snooping with preregistration theater. Combo-1298 was pulled from 1500 candidates and looked at 4 times already. There is no other data. That's the actual answer to "what other data is there?" — nothing, which is the problem.

On Question 3: The 1.108 gross ceiling is the load-bearing fact everyone's dancing around. Before costs, across 13,814 tries, your best is barely above 1. That's not a strategy that needs better exits or sessions. That's a signal family with no juice.

**Verdict: C.** Sunset this signal family. Pivot to a genuinely different signal family within NQ/MNQ with fresh preregistration. Stop re-reading the same test window hoping a different combo waves.

---

### The Executor — Verdict: C

Monday-morning reality check. Forget interpretability gymnastics. Here's what each path actually costs in keystrokes and wall-clock.

**(A) Paper-trade prereg on combo-865 1h despite F=1**
First file: `tasks/paper_trade_preregistration.md` (new). Then `scripts/runners/run_paper_trade_865.py` → broker adapter you don't have yet. Readable result: never, because signed §5.2 says COUNCIL_RECONVENE and you'd be laundering the override. Even if you write it Monday, you can't ship it Friday without burning the ritual. Operational cost: high (broker adapter = new infra). **Dead on arrival.**

**(B) Fresh prereg for combo-1298 RTH carve-out**
First file: `tasks/probe5_preregistration.md`. Data already on disk (`data/NQ_1h.parquet`, CT-naive, you now know the localize rule). Scripts are 90% copy-paste from `tasks/_probe3_1h_ritual.py` with the TZ fix already merged. Remote launch: `python scripts/runners/run_probe5_remote.py`. Readable result: **2–4 days** (prereg draft Mon, council Tue, launch Wed, artifacts Fri). Problem: n=77 is the whole edge, and you're re-reading the 1h test partition for the **5th** time. Even with a clean prereg, posterior is dominated by multiplicity, not compute.

**(C) Branch A sunset + signal-family swap prereg**
First file: `tasks/probe5_signal_family_preregistration.md` specifying the candidate family (momentum breakout? VWAP reversion? opening-range?). Then `scripts/param_sweep.py --range-mode v12_momentum` (new branch in sweep code — ~1 day of careful edits per the pre-sweep gate). Remote sweep: 1500 combos × 1h ≈ 6–12h on sweep-runner-1. Readable result: **5–7 days**. Higher, but it's a **fresh partition of hypothesis space** — no multiplicity debt, no reused holdout.

**(D) Bounded alt — exit mechanics sweep on Probe 2's already-passed combo-865 1h**
First file: prereg for `EX_*` mechanics sweep (breakeven, trailing, time-stop) on the combo-865 1h params, pre-declaring the holdout slice. Scripts exist. **3–4 days.** But this is just Probe 3 redux under a new name and burns the partition a 5th time for a probably-marginal gain.

**Fastest signal-to-decision:** (B). **Least operational friction:** (B) — zero new infra, TZ fix already in tree, scripts exist.
**Best learning-per-day accounting for multiplicity:** (C). You stop strip-mining the 1h holdout.

You can't do A. D is (B)-but-worse. Pick between (B) and (C) on how much holdout debt you're willing to take.

**Verdict: C.** Stop re-reading a depleted partition. Monday: open `tasks/probe5_signal_family_preregistration.md`.

---

## Peer Review Round (5 parallel sub-agents, blind to advisor identity)

### Reviewer 1

**1. Strongest: D (First Principles).** D is the only response that frames the decision correctly — not "which probe next" but "what is the posterior P(H1) and does any remaining action have positive expected information value?" It integrates the five retractions/revocations into a coherent pattern (measurement apparatus failures, not near-misses), applies the prior-update correctly (P(H1) should *decrease* after TZ retractions, not increase), and shows why A/B/D-narrow reduce to violations or noise-extraction. C's ranking falls out mechanically from the calculus rather than being asserted.

**2. Biggest blind spot: E (Expansionist).** E treats n=77 Sharpe 4.41 as a "genuine mechanical signal" and proposes *yet another* read of the 1h test partition (the 5th, now RTH-filtered) as the salvation. This ignores that RTH-restriction is a *search dimension already implicit in the 1500×3-session multiplicity E itself would need to correct for*. The "session-binding, not family-binding" pitch is exactly the post-hoc rationalization §5.2 was signed to block. The $400–600k/yr arithmetic compounds a multiplicity-inflated point estimate — the error bar on n=77 swallows the headline. E also misreads §7.6: session is permitted *with fresh prereg*, not as an escape hatch for the falsified family reading.

**3. What ALL missed:** None named the **infrastructure debt** as a blocker. The TZ bug cascaded across four probes because there is no source-of-truth timestamp contract or validation harness. Whichever path is chosen (even C), the next probe will be built on the same fragile measurement stack. Before any prereg, a methodological audit of the pipeline (timestamp handling, gate computation, partition-read accounting) should be a gating deliverable. B gestured at this; none made it mechanical.

---

### Reviewer 2

**1. Response D (First Principles)** is strongest. It names the meta-pattern correctly (five retractions in two weeks, every affirming result dissolves under audit) and updates the prior in the right direction. A/B/C do individual-verdict reasoning; D does portfolio reasoning across the retraction sequence. E is clever but conflates "TZ fix revealed session structure" with "TZ fix creates a new partition" — it does not.

**2. Response E (Expansionist)** has the biggest blind spot: it treats the TZ-corrected data as a fresh partition. It isn't. The 1h test partition has been read 4 times already; "RTH-filtered 1h" is a subset of the same bars, not independent data. Its ~$400-600k/yr scaling from n=77 is exactly the inflation §5.2 was designed to prevent. E also misframes Probe 1 §7.6 as permissive — RTH-restricted 1h is plausibly a new bar-timeframe axis variant requiring fresh prereg scaffolding, not a quick sweep.

**3. All five miss the write-up cost of the TZ cascade itself.** Before any Probe 5 (C or D), the preregistration framework needs a methodological audit: every post-hoc script that touched `tz_localize("UTC")` must be re-audited (not just `_probe3_*` and `_probe4_*`), `memory/feedback_tz_source_ct.md` enforcement needs a CI/pre-commit check, and the Scope D divergence needs a retraction document. B gestures at this ("pre-probe methodological audit") but none quantify it. Shipping Probe 5 before closing the TZ cascade risks a 6th retraction on the same root cause. The correct sequence is: close cascade → audit → then C (or D with E's caveats addressed).

---

### Reviewer 3

**1. Response B (Contrarian) is strongest.** It names the actual governance failure — §5.2 COUNCIL_RECONVENE was designed to halt exactly the post-hoc rationalization happening now (§4.4 "strengthening" framing, §4.3 rescue re-reading). B is alone in demanding a methodological audit *before* any new preregistration, which addresses the root cause D correctly identifies (every affirming result dissolves). A/C/E converge on conclusions without auditing why the prior four probes produced retractable outputs.

**2. Response E (Expansionist) has the biggest blind spot.** It reframes the TZ bug as a "gift" enabling a session-restricted Probe 1 redux — but this is the 5th read of the same 20% test partition and effectively weaponizes the retraction into a new affirming probe. E also treats Sharpe 4.41 on n=77 as economically load-bearing ("$400-600k/yr") when B and C correctly flag 77 trades post-4×-reuse post-Bonferroni as within noise. E's "maximum optionality" framing ignores that optionality on a burned partition is illusory.

**3. All five missed: the decision isn't actually between A/B/C/D — it's whether the project's preregistration machinery is still trustworthy.** Four signed preregistrations (Probes 1-4) produced findings that required retraction under a single TZ bug. No response asks: should Probe 5 wait until the TZ fix is propagated through *all* historical artifacts (combo_features_v12.parquet, Probe 1 N_1.3 counts, Scope D) and re-verified? Launching Probe 5 on a codebase with unaudited TZ-dependent artifacts risks retraction #6. B gestures at this; none operationalize it.

---

### Reviewer 4

**1. Strongest: D (First Principles).** D correctly reframes the decision from "which probe next" to "what is the posterior probability any NQ bar-mechanical edge exists." It names the actual pattern — 5 retractions in 2 weeks, every affirming result dissolving under audit — and correctly flags that we are **running out of independent data** (the 1h partition has been read 4×, now 5× under any continuation). This is the load-bearing constraint; everything else is tactics. A, B, C all arrive at C but D earns it.

**2. Biggest blind spot: E (Expansionist).** E misses that a "session-restricted Probe 1 redux" is a **5th read of the same 1h partition** with a post-hoc axis (RTH) suggested by the very TZ-correction that retracted Probes 3/4. This is textbook post-hoc selection on exhausted data — the exact failure mode B and D flag. The "$400-600k/yr" extrapolation from n=77 is the Outsider's "77 coin flips" dressed up as optionality. D=option is C-redux with a session-flavor coat.

**3. What all five missed:** None quantified **remaining independent NQ data** as a hard budget. The 1h partition is burned; 15m is burned (Probe 2 FAIL + Probe 3 §4.3); 1m is burned (Probe 1). Any Probe 5 — signal-family OR session-redux — still draws from the same NQ price series across the same dates. A signal-family swap on NQ 1h is **not** an independent test; it's a new hypothesis on contaminated data. The council should preregister an **out-of-sample data acquisition step** (forward paper-bar accrual, or a held-in-reserve partition never touched) before *any* Probe 5, or acknowledge the project is data-exhausted and sunset is the honest call.

---

### Reviewer 5

**1. Strongest: D (First Principles).** D reframes correctly. The question isn't "which probe next" — it's "does accumulated evidence warrant concluding no discoverable edge exists?" Five retractions in <2 weeks, 1.108 gross ceiling across 13,814 combos, Probe 1 §7.6 terminal, and 1h test partition read 4× (5× if we run again). D is the only response that treats prior P(H1) as a quantity that should *decrease* under this evidence stream, not increase with each reframe. B makes the same point but wraps it in methodology; D names the actual epistemic fact.

**2. Biggest blind spot: E (Expansionist).** E treats the TZ bug as a "gift" that *unconfounds* past reads. It does the opposite — it tells us the prior reads were wrong about session structure, not that session structure is now the answer. Combo-1298's n=77 is still n=77. Embedding a known-hot combo in a "redux" on a "disjoint window" is the definition of double-dipping: the window only looks disjoint because 1298 was discovered using the full partition. E also ignores that Probe 1's N_1.3=4 on 1h was already on data that *contained* RTH hours; RTH-only can at best move the count, not create 6 new passing combos from nothing.

**3. What all five missed.** None quantified partition exhaustion as a *stopping rule*. The 1h test partition has been read 4 times under distinct preregistrations; a 5th read (any option B/D, arguably C if it touches 1h) has no statistical validity regardless of prereg quality. The council should declare the 1h test partition **burned** before choosing any path — and C's "fresh signal family" only escapes this if it uses a genuinely held-out partition, which none of the responses verified exists.

---

### Peer Review Tally

| | Strongest | Blind spot |
|---|---|---|
| Reviewer 1 | D (First Principles) | E (Expansionist) |
| Reviewer 2 | D (First Principles) | E (Expansionist) |
| Reviewer 3 | B (Contrarian) | E (Expansionist) |
| Reviewer 4 | D (First Principles) | E (Expansionist) |
| Reviewer 5 | D (First Principles) | E (Expansionist) |

**Strongest advisor:** First Principles (4/5) · Contrarian (1/5)
**Biggest blind spot:** Expansionist (5/5 unanimous)
**What all advisors missed** (converging themes): (i) infrastructure debt / methodological audit before any new probe (Reviewers 1, 2, 3); (ii) data-partition exhaustion applies to C too — signal-family swap on NQ 1h is not an independent test (Reviewers 4, 5).

---

## Chairman Synthesis

### Where the Council Agrees

Four of five advisors (Contrarian, First Principles, Outsider, Executor) converge on **Verdict C** — Branch A sunset with signal-family swap. The convergence is not tactical; it's rooted in three independently-reached conclusions:

**(1) §5.2 COUNCIL_RECONVENE is binding, not advisory.** Contrarian names this most sharply ("§5.2 binding exists precisely to stop post-hoc rationalization"); Outsider translates it to plain English ("you wrote the rule before you saw the data specifically so you couldn't override it"); Executor treats A as dead-on-arrival on this basis alone. Option A requires override of a signed prereg binding. No amendment protocol exists.

**(2) The 1h test partition is burned.** Contrarian (Bonferroni over 1500×3 cells + 4× reuse), First Principles ("running out of independent data"), and Outsider ("data-snooping with preregistration theater") all independently flag that a 5th read of the 1h partition — whether for 1298 RTH (B) or Probe 3 redux (D) — is statistically indefensible regardless of prereg quality.

**(3) The accumulated-evidence pattern lowers prior P(H1), not raises it.** First Principles names this cleanly: 5 retractions in 2 weeks + gross ceiling 1.108/13,814 + measurement apparatus proven broken ⇒ posterior should compress toward sunset, not expand toward a new carve-out. Contrarian's "every result that looked affirming dissolved under audit" captures the same pattern from a governance angle.

### Where the Council Clashes

**Expansionist vs. the other four** is the only real clash, and it doesn't survive peer review. Expansionist reframes the TZ bug as a "gift" unconfounding prior reads and proposes session-restricted Probe 1 redux (D) embedding combo-1298 as a confirmatory cell. All five reviewers flagged this as the weakest response, for convergent reasons:

- Reviewer 1: treats RTH-restriction as fresh partition (it isn't — subset of same bars).
- Reviewer 2: $400-600k/yr scaling from n=77 is "exactly the inflation §5.2 was designed to prevent."
- Reviewer 3: weaponizes the retraction into a new affirming probe.
- Reviewer 4: "textbook post-hoc selection on exhausted data."
- Reviewer 5: TZ fix tells us prior reads were *wrong* about session structure — it doesn't tell us session structure is *now* the answer. n=77 is still n=77.

**Secondary clash:** Executor's tactical ranking (B fastest, C best) vs. the methodological framing. Reviewers 1-3 all surface that even C as proposed is not immediately executable — it inherits the broken measurement stack, and shipping Probe 5 before closing the TZ cascade risks a 6th retraction on the same root cause.

### Blind Spots the Council Caught

Three load-bearing points emerged only in peer review, not in the advisor responses themselves:

**(1) Infrastructure debt as gating deliverable (Reviewer 1).** No timestamp contract, no validation harness. The TZ bug cascaded across 4 probes because the measurement apparatus has no assertions. Any Probe 5 — C or otherwise — inherits this fragility.

**(2) TZ cascade is not closed (Reviewer 2).** Every post-hoc script that touched `tz_localize("UTC")` needs re-audit, not just `_probe3_*` and `_probe4_*`. `memory/feedback_tz_source_ct.md` needs enforcement as a pre-commit or CI check. Scope D divergence needs its own retraction document. Historical artifacts (`combo_features_v12.parquet`, Probe 1 N_1.3 counts) need re-verification under corrected TZ before any downstream probe treats them as ground truth.

**(3) Partition exhaustion applies to C too (Reviewers 4, 5).** This is the sharpest blind spot. A signal-family swap on NQ 1h using the same 20% test partition is *not* an independent test — it's a new hypothesis on contaminated data. None of the five advisors quantified remaining independent NQ data as a hard budget. 1m, 15m, and 1h are all burned under Probe 1 + Probe 2. C only escapes this if it uses a genuinely held-out partition, and no one verified one exists.

### The Recommendation

**Verdict: C, but sequenced — not executable this week.** The 4-1 majority is correct on direction; the peer review is correct that C-as-proposed skips two gating deliverables.

Concretely, the council's mechanical action is:

1. **Close the TZ cascade before any new prereg.** Audit every script that calls `tz_localize`; add a validation harness that asserts source parquets are naive CT at load time; retract Scope D's "SES_2a dominates" reading as a signed artifact; re-verify historical N_1.3 counts under corrected TZ.

2. **Declare the 1h test partition burned in writing**, and resolve the independent-data question before Probe 5. Either (a) identify a genuinely held-out partition never touched by Probes 1-4 + Scope D, or (b) preregister forward paper-bar accrual as the out-of-sample source, or (c) acknowledge NQ bar-data is exhausted and sunset is the honest call.

3. **Only then** open `tasks/probe5_signal_family_preregistration.md`.

Skipping (1) risks a 6th retraction on the same root cause. Skipping (2) means C collapses into "new hypothesis on contaminated data" — which the peer review correctly flags is not actually an independent test.

B is blocked by §5.2 + partition exhaustion. A is blocked by §5.2. D is B-but-worse. C is the only survivable direction, but the TZ cascade and partition-exhaustion questions are upstream of it.

### The One Thing to Do First

Open `tasks/tz_cascade_closure.md` and enumerate every historical script and artifact that touched `tz_localize("UTC")` on timestamps sourced from `data/NQ_*.parquet`. Until that enumeration is complete and the Scope D retraction + historical N_1.3 re-verification are signed off, no Probe 5 preregistration should be drafted.

---

*End of transcript.*
