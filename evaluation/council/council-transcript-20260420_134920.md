# LLM Council Transcript — 2026-04-20 13:49 CDT

**Topic:** V4 leak-isolation strategy for the random-K null test (4 options fork)
**Context anchor:** Prior council verdict `council-report-20260420_043015.html` prescribed a random-K null test; Phase 1 today hit two structural blockers revealing a deeper leak.

---

## Framed Question

A trading-strategy ML pipeline ships at Sharpe 2.13 (Pool B + V4 filter on MNQ futures, 641 trades, ruin prob 0.02%). The v12 UCB ranker behind it was **proven statistically indistinguishable from random** in an ablation (`project_ml1_v12_ablation_failure`). A full-pool variant (no top-K truncation) collapses to ruin prob 43% — so concentration to 50 combos IS load-bearing, but we cannot yet prove those *specific* 50 are better than 50 random. An LLM council yesterday prescribed a "random-K null test": 500+ random 50-combo baskets vs shipped top-50, all passed through a V4 refit on a pre-2024 clean partition.

Plan Phase 1 today hit two problems:

1. **Structural:** v11 parquet (the training data) has NO datetime columns at all. Only `entry_bar_idx`, `time_of_day_hhmm`, `day_of_week`. A calendar-based temporal filter is impossible without joining to bars CSV and choosing an arbitrary bar-index cutoff.
2. **Deeper leak discovered:** V4's feature list includes `global_combo_id` as a **trained LightGBM categorical** (`ID_FEATURE = "global_combo_id"`, categorical_feature in training). V4 has literally memorized per-combo win rates for every combo in the pool. A temporal holdout does NOT fix this — V4 still sees the same combo IDs before and after any time cutoff.

### Four options

1. **Combo-agnostic V4** (recommended): drop `global_combo_id` from features, retrain on full v11. Clean leak isolation; no combo memorization. Runtime ~same.
2. **Entry-bar temporal holdout**: map `entry_bar_idx → timestamp` via NQ_1min.csv, filter by arbitrary bar-index cutoff. Matches plan intent but does NOT fix combo-ID memorization.
3. **Both** (combo-agnostic + temporal). 2× runtime, strictly most conservative.
4. **Accept shipped V4**: skip refit, note caveat in Phase 4. Null test remains leak-confounded, conclusion bounded.

**State:** local Phase 1 code committed + pushed; remote screen killed; no artifacts produced. Compute budget for remote retrain is ~hours on sweep-runner-1 (9G RAM cap).

**The decision:** which option to pursue — or is the framing itself wrong?

---

## Advisor Responses

### The Contrarian

The framing is wrong, and all four options are bandaids on a deeper rot.

You just discovered that V4 — the filter inside your shipped Sharpe 2.13 system — is a combo-ID memorizer. That is not a leak to isolate for the null test. That is a leak in the *production pipeline you already declared ship-ready yesterday*. The null test is suddenly the least of your problems. You are debating whether to refit V4 cleanly for an ablation while the shipped V4 is still looking up win rates by ID.

Ask the uncomfortable question: if V4 memorizes `global_combo_id`, what does the Sharpe 2.13 actually measure? It measures how well LightGBM recalls per-combo base rates on combos *it was trained on*. Your top-50 basket is drawn from that same pool. The filter isn't filtering — it's retrieving. The 43% ruin on full-pool isn't evidence that concentration is load-bearing; it's evidence that V4 can't rank combos it has over-memorized once you stop pre-filtering by raw Sharpe.

Option 1 isn't a null-test prep step. It's a **ship-blocker audit**. Retrain combo-agnostic V4, re-run s6_net on the shipped top-50, and see if Sharpe 2.13 survives. If it does, then run the null test. If it collapses, you didn't have a shippable system yesterday and the council's "SHIP Pool B + V4" decision was based on a memorized artifact.

Option 4 is indefensible — you'd be documenting a caveat on a system whose core filter is doing lookup, not learning.

Do Option 1. But reframe it: not as null-test prep, as ship-decision re-litigation. The null test waits.

### The First Principles Thinker

You're asking "which null test" when the real question is "what would change our mind?"

Stop. The ablation already told you UCB ranking ≡ random. The full-pool collapse told you top-K concentration is load-bearing. Those are two established facts. The null test you're now designing asks: "are the shipped 50 combos better than 50 random combos from the same post-gate pool?" If the answer is NO, you already have the hypothesis ready: concentration is the mechanism, combo identity is noise. If the answer is YES, you need to explain WHY given the ranker is random — and the only remaining mechanism is `audit_full_net_sharpe` itself having OOS signal, which is just raw in-sample Sharpe persistence, which is a well-known weak-but-nonzero effect.

So the framing is wrong in a specific way: you're treating this as a validation gate for shipping. It isn't. You already shipped. This is a **mechanism attribution experiment**, and mechanism attribution demands you isolate one variable at a time.

Option 1. Combo-agnostic V4. Here's why the others are wrong:

- **Option 2 alone** is worthless — you've already identified `global_combo_id` as a memorization channel. Running a "clean" temporal test through a memorizing filter tells you nothing.
- **Option 3** is 2× compute to answer a question you don't have. What does "combo-agnostic + temporal" disambiguate that combo-agnostic alone doesn't? Nothing, unless you suspect non-stationarity in the feature→win mapping, which is a different experiment.
- **Option 4** concedes the leak and publishes a bounded conclusion. Fine if compute were scarce. It isn't — hours on one VM.

The parquet having no datetime columns is the universe telling you temporal holdout was never the real experiment. Drop `global_combo_id`, retrain, run the random-K test. One variable. One answer.

### The Expansionist

The framing is too small. You've discovered `global_combo_id` is a trained categorical — that's not a cleanup task, that's a **product pivot hiding in plain sight**.

Go with **Option 1 (combo-agnostic V4) — but reframe what you're building.** A V4 that only sees *parameter-space* features (no combo identity) is no longer a combo memorizer. It's a **combo generator's scoring function**. That changes the ceiling dramatically:

1. **Unshackle from the 13,814-combo pool.** Once V4 scores parameters rather than IDs, you can sample *new* combos outside the v11 grid — denser around the current top-50, or into unexplored regions v11 undersampled (e.g., mid-band z + tight stops). The sweep is no longer your universe; it's your training set.

2. **The null test becomes almost free bycatch.** Run the random-K test on the combo-agnostic V4 *in parallel* with the real prize: a **V4-guided active search** that proposes 500 synthetic combos, scores them, backtests the top ones. If even one survives the same s6_net gate at Sharpe > 2.13, you've proven the pipeline generalizes — which is the actual scientific claim the council was circling.

3. **Top-K concentration stops being a black-box constraint.** Right now "50 is load-bearing" is a fact you can't explain. A combo-agnostic scorer lets you ask *why 50* — is it parameter-region coverage? diversification across z-formulations? You can answer this and likely push to Sharpe > 2.5 by selecting 50 **diverse** high-scorers instead of 50 top-ranked ones.

Skip Option 2 entirely. Temporal holdout on bar-index is theater compared to removing the ID feature. Don't do Option 3 — 2× runtime for marginal extra rigor when Option 1 unlocks the generator.

**Ship Option 1. Then immediately scope the active-search extension.** That's the 10× outcome hiding here.

### The Outsider

The framing buries the lede. You're not deciding between four refit options — you're deciding whether a null test can even answer your actual question.

Your actual question: "Are these specific 50 combos better than 50 random combos from the same pool?"

A model that trained on `global_combo_id` as a categorical feature cannot answer that question. Ever. It has memorized which IDs win. Running random-K through it measures "how well did V4 memorize each basket," not "are these combos inherently better." Option 2 is dead on arrival — time-splitting a lookup table doesn't un-leak it. Option 4 ships a conclusion you can't defend.

So it's Option 1 or Option 3. And here's the outsider question nobody's asking: **why does a production filter have combo ID as a feature at all?** That's not a filter — that's a hash table with gradient boosting bolted on. If the ID feature genuinely helps OOS, your pipeline isn't learning generalizable trade quality, it's learning "combo 4471 wins, combo 9823 loses." That should terrify you regardless of the null test.

Pick Option 1. Reasons:
- It's the only option that makes the null test *mean* something.
- If combo-agnostic V4 still produces Sharpe ~2 on the shipped 50, you've proven the ranker picked combos with learnable non-ID structure.
- If it collapses, you've learned something far more important than the null test was going to tell you: your shipped edge is partly ID-memorization, and the 43% full-pool ruin might be closer to the truth than the 0.02% top-50 number.
- Option 3 is 2x compute to answer a question (temporal drift) you haven't motivated.

Do Option 1. Treat the ID-feature discovery as the headline finding, not a footnote.

### The Executor

Option 1. Ship it today. Here's what you do Monday morning — or rather, right now:

1. `cp scripts/backtests/<v4_train_script>.py scripts/backtests/train_v4_combo_agnostic.py`
2. Delete two lines: the `global_combo_id` entry in the feature list, and the `categorical_feature=[...]` arg in the LightGBM `train()` call.
3. `git add && git commit && git push`, SSH into sweep-runner-1, `git pull`, launch under `systemd-run --scope -p MemoryMax=9G` in a screen. Poll at 600s.
4. When it finishes, run the 500-basket null test against this refit V4.

Why not the others:

**Option 2 is actively broken.** You said it yourself — temporal holdout does NOT fix combo-ID memorization. You'd burn hours producing an artifact that doesn't answer the question. Skip.

**Option 3 is Option 2's waste plus Option 1's work.** The temporal holdout adds no information once combo-ID is dropped, because v11 parquet has no real calendar anyway — you'd be inventing a bar-index cutoff and calling it temporal. That's theater, not evidence. 2× runtime for 1× signal.

**Option 4 ships a known-confounded null test.** The whole point of running this is to defend the 50-combo concentration against the ablation finding. A leak-confounded null with a caveat in Phase 4 is worse than no null — it looks like due diligence while providing none.

The framing is right. One fix (drop the ID feature), one retrain, one clean null test. Don't let "most conservative" (Option 3) masquerade as "most rigorous" when half of it is provably useless. Go.

---

## Anonymization Map

| Letter | Advisor |
|---|---|
| A | The Executor |
| B | The Contrarian |
| C | The First Principles Thinker |
| D | The Outsider |
| E | The Expansionist |

---

## Peer Reviews

### Reviewer 1

1. **Response B is strongest.** It refuses the framing and names the real stakes: if V4 memorizes `global_combo_id`, the shipped Sharpe 2.13 is retrieval, not filtering. That makes null-test prep secondary to a ship-decision re-litigation. B is the only one that protocol-corrects the "shipped yesterday" posture rather than proceeding to experiment design on top of a possibly-broken production claim.

2. **Response E has the biggest blind spot.** Pivoting to a "combo generator's scoring function" and active search assumes the shipped edge is real and ID-independent — the exact thing B and D flag as unverified. E skips the prerequisite audit (does combo-agnostic V4 preserve Sharpe 2.13?) and jumps to a 10× vision built on unverified ground. Classic premature scope expansion.

3. **All five missed rollback/disclosure mechanics.** Per CLAUDE.md, ML#2 V3 (not V4) is the documented production stack; the Pool B ship decision (d25fbe3) used V4 + raw-Sharpe top-50. None of the responses address: (a) is the 2026-04-15 SHIP commit reversible pending the combo-agnostic refit, (b) should `tasks/ship_decision.md` get a provisional-status flag now, (c) what does V3's feature list look like — does V3 also carry `global_combo_id`? The leak may contaminate more than V4.

### Reviewer 2

**1. Strongest: B.** B is the only one that names the actual stakes. A, C, D, E all treat Option 1 as null-test prep with a side-benefit. B correctly inverts: if V4 memorizes `global_combo_id`, the Sharpe 2.13 ship decision from yesterday is already compromised, because the filter that produced it was retrieving per-combo base rates, not filtering trade quality. The null test is downstream of "does the shipped system survive ID removal." Every other response tacitly accepts the shipped number.

**2. Biggest blind spot: E.** E leaps to an active-search product pivot before establishing whether combo-agnostic V4 works at all. If the refit collapses (B's scenario), there is no scoring function to guide synthetic combo generation — E is scoping a 10× outcome on top of an unverified foundation. Classic over-extension.

**3. What all five missed.** None questioned whether `audit_full_net_sharpe` — the current production ranker — is itself leaky. It was computed on the same sweep used to train V4. If the top-50 was selected by in-sample Sharpe and then filtered by an ID-memorizing V4, **both** stages see combo identity. The random-K null needs a combo-agnostic ranker too, or it only tests half the pipeline.

### Reviewer 3

**1. Strongest: B.** B is the only response that treats the ID-leak discovery as what it actually is — a ship-decision invalidator. Every other response accepts yesterday's ship decision and asks "what null test design?" B correctly inverts the stack: if the production filter memorizes `global_combo_id`, then Sharpe 2.13 may be measuring LightGBM's recall of per-combo base rates on combos it saw in training. The null test is downstream of answering that. B's sequencing (retrain combo-agnostic V4 → re-run s6_net on top-50 → *then* null test) is the only path where every subsequent result is interpretable. D reaches the same insight but still frames it as null-test prep; B names it as ship-litigation.

**2. Biggest blind spot: E.** E leaps to active-search product scope before confirming the current shipped system isn't hollow. "10× outcome hiding here" assumes V4-without-ID still has signal — the exact hypothesis B/D say must be tested first. Building a combo generator on top of a filter that might be a hash table is castle-on-sand.

**3. All five missed:** none asked whether `audit_full_net_sharpe` — the raw-Sharpe ranker now in production — is itself computed from the same training window V4 saw. If yes, top-50 selection *and* V4 filtering share an in-sample snooping channel independent of the ID feature. The ID fix is necessary but insufficient.

### Reviewer 4

**1. Strongest: B.** It's the only response that names the actual risk correctly. The others treat V4's ID leak as a nuisance blocking the null test; B recognizes it's a nuisance in the *shipped* system. If V4 memorized `global_combo_id`, then the Sharpe 2.13 on the top-50 was scored *through that same memorizing filter on combos it trained on* — that's not an edge, it's retrieval. B's sequencing (refit → re-run s6_net on top-50 → *then* null test) is the only order that protects against shipping a confounded result. C and D reach the same refit but miss the ship-decision implication.

**2. Biggest blind spot: E.** Active-search/combo-generation is a fantasy built on an unverified premise. If combo-agnostic V4 collapses (B's scenario), there's no scoring function to generate from — you'd be sampling from noise. E skips the verification step entirely and jumps to 10× scope.

**3. All five missed:** the top-50 basket was *selected* by `audit_full_net_sharpe` computed on the same v11 sweep V4 trained on. Even combo-agnostic V4 re-scoring that same basket inherits selection bias from the ranker. The honest test needs an unseen combo pool or a held-out slice of v11 — otherwise refit-V4 Sharpe is still optimistic.

### Reviewer 5

**1. Strongest: Response B.** It's the only one that refuses to let "run the null test" distract from the actual discovery. If V4 memorizes `global_combo_id`, then yesterday's ship decision (Sharpe 2.13, s6_net, CI [+0.56, +3.64]) was made through a retrieval layer, not a filter. B correctly re-sequences: re-litigate ship first, null test second. D reaches the same insight but softens it ("more important than the null itself"). B names it a ship-blocker.

**2. Biggest blind spot: Response E.** Pivoting to active search / synthetic combo generation before confirming the shipped system's Sharpe survives the ID-feature removal is building the second floor before inspecting the foundation. If combo-agnostic V4 collapses s6_net, E's "parameter scoring function" has no demonstrated edge to extrapolate from. It mistakes a contamination finding for a capability.

**3. What all five missed:** None asked whether the *top-50 selection itself* (raw-Sharpe ranking of `audit_full_net_sharpe`) is also ID-contaminated upstream — `combo_features_v12.parquet` was built under MIN_TRADES_GATE=500 on the same v11 sweep V4 trained on. The null test and the V4 refit both assume the top-50 *list* is independent of the leak. It isn't. The audit target and V4's training labels share the same in-sample trades.

---

## Chairman Synthesis

### Where the Council Agrees

- **All five advisors picked Option 1** (combo-agnostic V4 refit). Options 2, 3, and 4 were unanimously rejected — temporal holdout doesn't address the combo-ID leak, a combined refit doubles compute for no added signal, and shipping with a caveat on a retrieval-based filter is indefensible.
- **All five peer reviewers independently named B as strongest** and **E as biggest blind spot.** The rank unanimity across anonymized reviewers is itself signal — B's ship-litigation framing is the clearest read of the situation.
- Four of five advisors (all except A) explicitly flagged the ID-feature discovery as the real finding, more important than the null test itself.

### Where the Council Clashes

The clash is about what Option 1 *is for*:

- **B (Contrarian):** ship-decision re-litigation. Null test waits until we confirm Sharpe 2.13 survives ID removal.
- **C (First Principles):** mechanism attribution experiment. We already shipped; this is learning, not a gate.
- **D (Outsider):** headline discovery about pipeline quality. Treat the ID feature existing in a production filter as the news.
- **E (Expansionist):** foundation for active combo search. Parameter-space V4 → synthetic combo generator.
- **A (Executor):** execute the steps; null test follows.

The underlying disagreement: **is the shipped Sharpe 2.13 currently trusted?** B and D say no (needs verification). C and E implicitly say yes (C "you already shipped," E "build on top"). The chairman sides with B and D — the reviewers unanimously confirmed this ordering.

### Blind Spots the Council Caught

Four of five peer reviewers independently identified the same universal miss:

**`audit_full_net_sharpe` — the raw-Sharpe top-50 selector — is computed on the same v11 sweep V4 was trained on.** The top-50 basket selection and V4's training labels share in-sample trades. Even after a clean combo-agnostic V4 refit, the null test's "shipped-50" anchor is drawn from a leak-contaminated list. Fixing V4 alone is necessary but insufficient. An honest null test needs either (a) an unseen combo pool or (b) a held-out slice of v11 used neither for top-50 selection nor V4 training.

Additional blind spots from single reviewers:

- V3 (still documented as production in CLAUDE.md Phase 5D) may *also* carry `global_combo_id`. The scope of the leak audit should begin with a grep across all production models, not just V4.
- No rollback/provisional-status mechanics proposed. If the audit finds the shipped edge doesn't survive, `tasks/ship_decision.md` needs a formal reversion path, not just a new section.

### The Recommendation

**Do Option 1, adopting B's re-sequencing and the reviewers' caught blind spots.** Concretely:

1. **Audit the leak scope first.** Grep V3, V4, and any ancillary model for `global_combo_id` as a trained/categorical feature before retraining. If V3 carries it too, the rebuild is broader.
2. **Mark `tasks/ship_decision.md` as PROVISIONAL** pending audit outcome. Single-line flag at top.
3. **Refit V4 combo-agnostic:** drop `global_combo_id` from `ALL_FEATURES`, `CATEGORICAL_COLS`, and the LightGBM `categorical_feature` arg. No other changes.
4. **Re-run s6_net on the shipped top-50 list using refit V4.** This is the ship-blocker audit.
5. **Only if step 4 returns Sharpe whose 95% CI overlaps the shipped 2.13** do you proceed to the random-K null test against refit V4.
6. **Plan a second-round null** that breaks the `audit_full_net_sharpe` leak by using a combo-agnostic selector (or a held-out combo pool) — otherwise even a passing null test under refit V4 only proves half the story.

### The One Thing to Do First

**Grep the production model training scripts for `global_combo_id` as a feature, before retraining anything.** This scopes the audit — if V3 also has it (CLAUDE.md documents V3 as production, not V4), the rebuild is broader and the ship-decision re-litigation covers both. If only V4 has it, the fix is contained. Either way, ten minutes of grep prevents committing to a retrain scope that's wrong.
