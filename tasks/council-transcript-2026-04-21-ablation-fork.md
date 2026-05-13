# LLM Council Transcript — Root Ablation Fork

**Date**: 2026-04-21 UTC
**Repository**: `intra`
**Skill**: `userSettings:llm-council`
**Trigger**: Three-way fork surfaced after Sharpe-distribution audit revealed universe ceiling below user's tradeability bar.

---

## The Original Question

The user had drafted a root-ablation kill-criterion (`tasks/root_ablation_criterion.md`) to retrain V3 with all per-combo memorization channels stripped — `global_combo_id` plus the `prior_wr_*` running win-rate family. Criterion was ready for sign-off. ~2h compute on the remote GPU (sweep-runner-1).

Then a pre-signature scan of the full v12 combo universe (`tasks/_sharpe_distribution.py`) returned:

- **0 combos with net Sharpe ≥ 1.0** across 13,814 eligible combos
- Max `audit_full_net_sharpe` = 0.736 (v11_7872)
- Max gross Sharpe = 1.108 (v11_23634) → drops to **net 0.601** after $5/contract RT friction
- Mean net Sharpe = −9.835; median = −6.162

User's binding constraint (stated in same conversation): *"A Sharpe of less than one is not tradeable as it poses too much risk."*

User invoked `/llm-council` on the resulting three-way fork:

- **(a)** Cancel the ablation — ceiling below tradeability bar, ablation outcome pre-determined.
- **(b)** Run the ablation anyway — ~2h compute, completes the Phase 5 record.
- **(c)** Pivot the project — change instrument, timeframe, strategy family, or friction model.

## Framed Question (what each advisor saw)

> **Situation**: MNQ 1-min mean-reversion strategy. V3 + V4 ML#2 filters revoked today after `global_combo_id` memorization leak — shipped Sharpes 1.78/2.13 collapsed to 0.31/−0.42 on clean refit. Root ablation drafted + unsigned, ~2h compute.
>
> **New finding**: In the full 13,814-combo v12 universe, max net Sharpe = 0.736. Mean = −9.8, median = −6.2. Friction alone ($5/contract RT) strips ~0.5 Sharpe.
>
> **User's hard gate**: "Sharpe < 1 is not tradeable."
>
> **Three-way fork**: (a) cancel the ablation, (b) run it for completeness, (c) pivot the project.
>
> **What's at stake**: ~6 months of sunk work on a filter layer that may not be the binding constraint. Pivoting earlier means more runway for a tradeable variant.
>
> **Question**: Which path? If pivot, what's the cheapest probe to identify where the tradeable surface lives?

---

## Step 2 — Advisor Responses (de-anonymized)

### The Contrarian

The ablation is dead weight. Run it and you get a formally-signed "we confirmed the leak was real" stamp on a pool that can't trade regardless. That's not closure, that's ceremony. Cancel (a).

But don't pivot yet either. You're about to make the same mistake in reverse: declaring the surface dead based on a filtered, already-degenerate universe. The 13,814 combos in v12 were **selected by a training-data generator tuned for ML signal diversity**, not by a search for tradeable Sharpe. A 0.736 ceiling in a diversity-biased pool tells you almost nothing about whether a Sharpe-seeking search over the same economics would clear 1.0.

Questions you're avoiding:

1. **What does the v12 ceiling look like at N=50,000 random combos drawn from wider ranges?** You've explored a narrow slice. The "ceiling" is a sampling artifact until proven otherwise.
2. **Is $5 RT friction real for you, or a conservative placeholder?** MNQ at a real retail broker is ~$1.20 RT. If you're gating on inflated friction, you may have killed the strategy on paper that trades fine in reality. Verify the number before pivoting the whole project.
3. **Is 1-minute the problem or the z-score family the problem?** You don't know. A 5-min z-score sweep is cheap.
4. **"Sharpe < 1 is not tradeable" — whose rule?** Sharpe 0.7 with 50% DD is untradeable. Sharpe 0.7 with 8% DD and monthly payout is a lifestyle business. The binding constraint is ruin probability and DD, not the Sharpe integer.

Cheapest probe: re-verify friction ($30), then 5k-combo wide sweep at 5-min with real friction (~4h compute). If **that** ceiling is also <1.0, pivot. Until then, you're pivoting on one data point.

### The First Principles Thinker

You're asking the wrong question entirely.

The question isn't "V3 or pivot?" It's: **what problem were you actually trying to solve, and does the evidence say it's solvable at all?**

Strip it down. The core claim of this entire project is: *there exists a parameterization of Z-score mean reversion on MNQ 1-min that, net of friction, earns Sharpe ≥ 1.0 — and ML can find it.* That's the load-bearing assumption. Everything else (ML#1 ranker, ML#2 filter, calibration, sizing) is machinery built on top of it.

You just falsified the assumption. Not with a model's output — with a **census** of 13,814 combos. Max = 0.736. No filter fixes that, because a filter is a *selection operator on an existing distribution*. You cannot select your way to a ceiling above the max. This is arithmetic, not statistics.

So (a) and (b) are the same answer dressed differently. The ablation's output is **already determined**: even a perfect leak-free V3 cannot produce a portfolio whose unfiltered constituent Sharpes exceed 0.736. Running it to "complete the record" is paying 2h compute to confirm what a census already proved. Don't.

Pick (c). But the cheapest probe is not "change instrument." It's this: **does a Sharpe ≥ 1.0 exist at all for 1-min mean-reversion on liquid US index futures net of realistic friction?** Three days, no ML, no sweep. Pull ES, NQ, YM, RTY 1-min. Run a dozen hand-tuned Z-score/EMA combos at current friction. If the best gross Sharpe across *all* of them is still sub-1.0 net, the strategy family is the binding constraint and you've been filtering noise for six months.

Falsify the family before rebuilding the stack.

### The Expansionist

The raw-Sharpe ceiling of 0.736 isn't a failure signal — it's a **lens problem**. You've been looking through MNQ-1min-meanreversion glasses at a universe that's vastly bigger than that.

**Pivot to (c), but reframe what "pivot" means.** You haven't built a trading strategy. You've built a **combinatorial alpha-discovery factory**: 13,814-combo sweep engine, parameter-KNN ranker, Monte Carlo, held-out evaluator, remote trainer, Council workflow. That stack is instrument-agnostic and strategy-family-agnostic. The hidden asset isn't the MNQ edge — it's the infrastructure's ability to chew through parameter spaces on ANY 1-min OHLCV series with friction.

**The cheapest probe**: Swap the data file. `data/NQ_1min.csv` → `ES_1min.csv`, `CL_1min.csv`, `GC_1min.csv`, `BTC_1min.csv`. Keep the entire sweep engine frozen. Rerun v11 ranges against 3-5 instruments in parallel on sweep-runner-1. Cost: ~6 hours. If ANY instrument shows pre-ML Sharpe >1.0 in the raw sweep, you've found the universe where your machinery has edge — and you can THEN apply the ML#2 filter work as a multiplier.

**Bigger upside nobody's pricing in**: Your v12 parameter-only KNN ranker is a **meta-model**. It doesn't care what the underlying instrument is. Train it cross-instrument and you get a parameter-space prior that transfers. That's a genuinely novel asset — most quant shops tune per-instrument from scratch.

**And this**: mean-reversion on 1-min MNQ failing at $5 RT friction is the **expected** outcome for retail friction on a liquid e-mini. Move to crypto perps (2-3bps friction, 24/7, higher vol-of-vol) and the same engine may light up. That's not a pivot — that's the engine finally meeting its natural substrate.

Don't cancel. Don't re-ablate a dead horse. **Port the factory.**

### The Outsider

Stop. I'm going to say what someone deep in this probably can't hear anymore.

You just told me the best result out of **13,814 attempts** is 0.736. You told me the average is -9.8, which I had to look up because I wasn't sure a Sharpe ratio could even go that negative in a coherent way. You told me you caught yourself cheating — twice — and only because you went looking.

A normal person hearing this would ask: how do you know the thing you're trying to do is possible at all?

Not "possible with better features." Not "possible with a smarter filter." Possible, period. You searched 13,814 configurations of a specific idea (mean-reversion on 1-minute bars of one instrument) and the entire surface is underwater. That's not a tuning problem. That's the search telling you the signal isn't there, or isn't there in the form you're looking for it.

The ablation is a 2-hour experiment to confirm something the data already screamed at you. You're not running it for information — you're running it because stopping feels like admitting six months went sideways. Those are different motivations and you should be honest about which one is driving.

My vote: **(c), but not the version you wrote.** Don't pivot to "different contract, different timeframe." That's just re-rolling the same dice. Pivot to figuring out *why* your entire search space is a loss-generator. Is it transaction costs eating everything? Is 1-minute mean-reversion on index futures just a solved/crowded edge? Does anyone — anywhere — publish evidence this class of strategy works after friction?

If you can't answer that, a new contract won't save you. Neither will a new model.

### The Executor

Cancel the ablation. Run option (c), specifically the friction re-score, today. Here's why: you already know the ceiling is 0.736 Sharpe at $5/contract RT. Two hours of GPU compute to confirm a revoked model still can't pass a bar it can't structurally reach is tuition you don't need to pay twice.

The friction re-score is the only probe that costs essentially nothing and tests a specific, plausible hypothesis: retail MNQ quotes are $0.50-$1.25/contract RT, not $5. You have every trade's gross PnL already in the v11 Parquet. You're multiplying one column by a scalar. That's it.

**Monday morning, three commands:**

1. `python -c "import pandas as pd; df = pd.read_parquet('data/ml/originals/ml_dataset_v11.parquet'); print(df.groupby('combo_id').apply(lambda g: (g['gross_pnl_dollars'] - 1.50*g.get('contracts',1)).sum() / ((g['gross_pnl_dollars'] - 1.50*g.get('contracts',1)).std() or 1) * (252*390)**0.5 / len(g)**0.5).describe())"` — quick-and-dirty net-Sharpe distribution at $1.50/contract to see if the ceiling moves above 1.0.
2. If step 1 clears 1.0: `python scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py --cost-per-contract 1.50 --output evaluation/top_strategies_v12_lowcost.json` — re-rank with corrected friction.
3. `python scripts/evaluation/final_holdout_eval_v3.py --combos evaluation/top_strategies_v12_lowcost.json --no-ml2` — run the shipped pipeline minus the revoked ML#2 on the held-out partition.

If step 1 ceiling stays <1.0, kill the project on this strategy family by Monday noon. Don't pivot instruments; pivot the strategy. You're not running the ablation either way.

---

## Step 3 — Peer Review (anonymized during review)

**Anonymization mapping used in this round** (revealed here for reference only — reviewers did not see this):

| Letter | Advisor |
|---|---|
| A | The Expansionist |
| B | The Outsider |
| C | The First Principles Thinker |
| D | The Executor |
| E | The Contrarian |

### Reviewer 1

1. **Strongest: D.** Only response that converts diagnosis into hour-scale experiment with kill-switch. Names exact file/flag (`extract_top_combos_by_raw_sharpe_v12.py --cost-per-contract 1.50`), Monday-noon decision. A/C/E argue philosophy; D ships. E raises the same friction point but dilutes it across four questions instead of forcing the test.
2. **Biggest blind spot: A.** "Port the factory" assumes the factory is sound. But the factory just produced two leak-contaminated models the author only caught by accident. Porting machinery with an unaudited leak surface to ES/CL/BTC replicates the `global_combo_id` failure mode across more instruments. A should have said: fix the audit protocol *before* scaling the search universe.
3. **All missed:** whether 0.736 is even the true ceiling. The v12 pool was gated at `MIN_TRADES_GATE=500` — a selection filter that truncates the right tail (high-Sharpe combos often have fewer trades). The "census" framing in B and C is wrong: 13,814 is a *filtered* sample, not a population. Before pivoting or ablating, re-run the ceiling calculation without the trade-count gate.

### Reviewer 2

1. **Strongest: D.** Only response with concrete, same-day, near-zero-cost test that falsifies the binding question. Three commands, one column multiply against existing v11 Parquet. Pre-commits to Monday noon kill decision. D converts the debate into a 2-hour experiment cheaper than the ablation it's replacing.
2. **Biggest blind spot: A.** "Port the factory" assumes the factory works. The factory just produced a 13,814-combo census with max 0.736 and caught itself leaking twice. Porting a leak-prone, selection-biased pipeline to 5 new instruments multiplies the leakage surface and the multiple-testing problem — not the alpha.
3. **All missed:** None questioned whether the **$5/contract RT assumption itself** (baked into v11 and all downstream net-Sharpe claims) was ever validated against actual retail MNQ costs (~$1.20 RT). D gestures at it but treats it as a pivot option, not a pre-existing bug. If friction was mis-specified by 4x, every "ceiling" number — including the 0.736 driving the kill decision — is wrong. Validating the cost constant is logically prior.

### Reviewer 3

1. **Strongest: D.** D treats the user's actual constraint (Sharpe < 1 is a hard gate) as a *falsifiable one-day test* rather than a philosophical pivot. It identifies the load-bearing assumption everyone else glossed: $5/contract RT is ~4x retail MNQ. That single parameter could move the ceiling from 0.736 to >1.0 without any new data, model, or instrument. This is the elegant fix `CLAUDE.md` demands.
2. **Biggest blind spot: A.** Wants to "port the factory" to ES/CL/GC/BTC but ignores that the factory itself encodes a revoked leak pattern, a diversity-biased combo pool, and potentially wrong friction. Porting a contaminated pipeline to four new instruments multiplies the contamination, not the evidence. Also: BTC 1-min mean-reversion has radically different microstructure — the sweep ranges won't transfer.
3. **All missed:** whether 0.736 is even a *clean* ceiling. Given V3/V4 both had combo-ID leaks, the v11 sweep's `audit_full_net_sharpe` may itself be contaminated by the same partition-memorization pattern the refits exposed. Before cancelling, pivoting, or re-friction-scoring, verify 0.736 survives a walk-forward recomputation.

### Reviewer 4

1. **Strongest: D.** Only response that converts diagnosis into same-day falsification with concrete dollar assumption ($1.50 RT vs baked-in $5), known artifact (v11 parquet has `gross_pnl_dollars` and `friction_dollars` — CLAUDE.md confirms), and Monday-noon kill gate. A/C/E gesture at the right moves; D executes one.
2. **Biggest blind spot: A.** Proposing NQ→ES/CL/GC/BTC with the "frozen sweep engine" ignores that the engine hardcodes MNQ economics ($2/pt, $5 RT) and the z-score/EMA parameterization was tuned on NQ 1-min distributions. Porting isn't a data swap — it's re-validating the factory. "Crypto perps" reframes a failed equity-index strategy as a different asset class entirely. Not a probe — it's abandoning the project while calling it a pivot.
3. **All missed:** The V3/V4 leak itself constrains the ablation's upside. The 100%-combo-overlap finding (`project_combo_overlap_structural`) means any clean refit inherits the same 13,814-combo ceiling — the ablation can *at best* reproduce 0.736, not exceed it. That makes (a) the correct answer on pure information grounds, independent of friction or instrument. No respondent cited this structural cap.

### Reviewer 5

1. **Strongest: D.** Cheapest falsification path with concrete commands. Friction assumption ($5/contract RT vs retail ~$1.20) is the single highest-leverage unchecked variable — if wrong, it rewrites the entire 0.736 ceiling. D is the only response that names a specific column operation, a deadline, and a kill criterion. Everything else is downstream of this check.
2. **Biggest blind spot: A.** Proposes porting the factory to new instruments without first validating the factory itself produces tradeable signal *anywhere*. The same combo-ID leak pathology and friction-assumption error would port straight to ES/CL/BTC. "Instrument-agnostic" is a feature only if the methodology is sound — which the last two revocations suggest it isn't. A is solving for novelty when the binding constraint is epistemic hygiene.
3. **All missed:** None asked whether the **0.736 itself is trustworthy**. Given two confirmed memorization leaks in the ML#2 stack, the ML#1 feature pipeline and sweep-engine friction accounting deserve the same audit before any decision. The census max is a measurement — and measurements from a system that just failed two integrity checks need re-verification before being treated as ground truth for a project-killing pivot.

### Peer-review convergence

- **Strongest (unanimous 5/5)**: Response D — The Executor.
- **Biggest blind spot (unanimous 5/5)**: Response A — The Expansionist.
- **Collective miss (all 5, distributed across five distinct reasons)**: none of the five advisors questioned whether 0.736 itself is a trustworthy ceiling.

---

## Step 4 — Chairman Synthesis

### Where the Council Agrees

All five advisors reject option (b) — running the 2h ablation is tuition on a question the 13,814-combo census already answered. A selection operator cannot exceed the max of its input distribution; 0.736 is a ceiling no filter moves.

Four of five (Contrarian, Executor, and both implicitly via peer reviews) converge on the same highest-leverage unchecked variable: **the $5/contract RT friction assumption is ~4x retail MNQ reality (~$1.20)**. Every "ceiling" number in the project — including the 0.736 driving the kill decision — is downstream of a constant that was never validated.

Peer review is unanimous: the Executor's response is strongest. It is the only one that converts the three-way fork into a same-day falsification with a concrete kill-gate.

### Where the Council Clashes

**First Principles vs Expansionist** on what the 0.736 ceiling means:

- First Principles reads it as a falsification of the strategy family. The cheapest next probe is hand-tuned Z-score across ES/NQ/YM/RTY to see if the *family* works anywhere before rebuilding the stack.
- Expansionist reads it as a lens problem — the factory is the asset, port it to crypto/commodities where microstructure and friction differ.
- Peer reviewers 1-5 side decisively against the Expansionist: porting an unaudited factory that just caught itself leaking twice multiplies the contamination surface across more instruments.

**Outsider vs everyone**: asks whether this class of strategy works after friction for anyone, anywhere. Nobody else treated that as a prerequisite question. Uncomfortable but correct — it's logically prior to any pivot.

### Blind Spots the Council Caught

Three load-bearing issues surfaced only in peer review, not in any primary response:

1. **The ceiling itself may be contaminated.** V3 and V4 both had `global_combo_id` memorization leaks. The v11 sweep's `audit_full_net_sharpe` was computed by the same pipeline and has never been audited for the same pathology. 0.736 may not be a clean number. (Reviewers 3, 5)
2. **The 13,814 pool is a filtered sample, not a population.** `MIN_TRADES_GATE=500` truncates the right tail of the Sharpe distribution — high-Sharpe-low-trade-count combos were removed before ranking. (Reviewer 1)
3. **100% combo-overlap structural finding means any clean refit inherits the same ceiling.** The ablation, even if run, cannot produce a combo that wasn't in the 13,814 pool. It can at best reproduce 0.736. This is the strongest argument for (a) on pure information grounds. (Reviewer 4)

These three findings together mean: the binding number (0.736) is derived from a filtered pool by a possibly-contaminated pipeline using a probably-wrong friction constant. Three unverified assumptions stacked.

### The Recommendation

**Cancel the ablation. Execute the Executor's three-step friction re-score today.** Do not pivot instruments. Do not run the ablation "for completeness."

The council is right that the ablation is dead weight. Running it costs 2h compute and produces one more number constrained by the same ceiling. The reviewers are right that (a) is the defensible answer on pure information grounds.

But (a) alone leaves three unverified assumptions in place. The Executor's path is cheaper than (a) and tests the single most load-bearing one: is $5/contract RT real? Peer review consensus plus the cost asymmetry (one column multiply vs. 2h GPU) makes this the dominant move.

The chairman adds one modification the council didn't make explicit: before multiplying the column, **verify $5 is actually what the sweep engine used and where `COST_PER_CONTRACT_RT` comes from in `scripts/param_sweep.py`**. If the constant is wrong, it's wrong everywhere — fix it at source, don't patch at the output layer.

Kill-gate: if the net-Sharpe distribution at $1.20 RT still tops out below 1.0, the strategy family is falsified on this instrument. At that point option (c) is the only live branch — but it's a pivot off a clean ceiling, not a contaminated one. First Principles' hand-tuned cross-instrument Z-score probe becomes the next move, not the Expansionist's "port the whole factory."

What the council rejects: Expansionist's crypto-perps pivot. Porting a pipeline that just confessed two memorization leaks and uses an unvalidated friction constant to four new instruments is how a clean kill becomes a four-instrument mess.

### The One Thing to Do First

Before running any re-score: open `scripts/param_sweep.py` and whatever code shipped the v11 parquet, find where `COST_PER_CONTRACT_RT = 5` is set, confirm it was actually applied to `friction_dollars` and `audit_full_net_sharpe`, and find any documented source (a commit, a comment, a `CLAUDE.md` line) justifying $5 over retail ~$1.20. That single file read resolves whether the 0.736 ceiling is even the number this decision should hinge on. Then run the Executor's step (1) — net-Sharpe distribution at $1.50 RT from the existing parquet. Monday noon kill decision stands.

---

## Artifacts

- HTML report: `tasks/council-report-2026-04-21-ablation-fork.html`
- This transcript: `tasks/council-transcript-2026-04-21-ablation-fork.md`
- Root ablation criterion (unsigned, superseded by this verdict): `tasks/root_ablation_criterion.md`
- Sharpe distribution script that surfaced the ceiling: `tasks/_sharpe_distribution.py`
- Previous Phase 5 council (ML#2 redesign fork): `tasks/council-report-2026-04-21-phase5.html`, `tasks/council-transcript-2026-04-21-phase5.md`

## References

- v11 friction constant lives in: `scripts/param_sweep.py` (`COST_PER_CONTRACT_RT`) — see `CLAUDE.md` "v11" row of parameter-sweep version history
- Feature builder that produces `audit_full_net_sharpe`: `scripts/analysis/build_combo_features_ml1_v12.py`
- Top-combo extractor (production ranker): `scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py`
- 100%-combo-overlap finding: `memory/project_combo_overlap_structural.md`
