# Probe 2 Verdict — Combo-865 Isolation PASS on 1h Holdout

**Date**: 2026-04-21 UTC
**Probe**: Combo-865 cross-TF isolation on test-partition bars (2024-10-22 → 2026-04-08)
**Preregistration**: `tasks/probe2_preregistration.md` (signed commit `a49f370`)
**Readout script**: `tasks/_probe2_readout.py`
**Machine record**: `data/ml/probe2/readout.json`

---

## Bottom line

**PASS — Probe 3 authorized** per §4 of the preregistration.

The 1h timeframe clears all three gates simultaneously with substantial
margin. The 15m timeframe fails dramatically. Per §4, "all three conditions
must hold on at least one timeframe" — no cross-timeframe mixing is
required or performed here, because 1h satisfies all three gates on its
own.

---

## Observed metrics

| Metric | 15m | 1h | Gate | 15m | 1h |
|---|---:|---:|:---:|:---:|:---:|
| `n_trades` | 780 | 220 | ≥ 50 | PASS | PASS |
| `net_sharpe` | **−0.5519** | **2.8949** | ≥ 1.3 | FAIL | PASS |
| `net_dollars_per_year` | **−$43,968** | **+$124,896** | ≥ $5,000 | FAIL | PASS |
| **All three** | — | — | — | **FAIL** | **PASS** |

Audit-only (not gate-bound):

| Audit metric | 15m | 1h |
|---|---:|---:|
| `gross_sharpe` | 2.346 | 4.404 |
| `total_friction` | $341,640 | $96,360 |
| `friction_per_trade` | $438 | $438 |
| `win_rate` | 0.4013 | 0.5318 |
| `mean_net_pnl` / trade | −$83.42 | +$840.15 |
| `std_net_pnl` / trade | $3,470.21 | $3,538.51 |
| `total_net_pnl` | −$65,068 | +$184,834 |

---

## Why 15m failed and 1h passed — mechanical interpretation

Friction in the v11 sweep engine is **per-contract** (stop-distance keyed,
not per-trade). Combo 865's `stop_fixed_pts = 17.018` fixes the contract
count identically at both timeframes, so mean friction is ~$438/trade at
**both** 15m and 1h.

- **15m**: 780 trades × mean gross edge ~$355/trade → **sub-friction**;
  friction carves off the entire gross Sharpe and then some.
- **1h**: 220 trades × mean gross edge ~$1,278/trade → **3.6× friction**;
  edge comfortably covers costs and leaves a substantial net.

This is the friction-intensity regime: 15m takes 3.5× more swings for a
much smaller per-trade move, and the fixed per-contract friction dominates
the smaller per-trade signal. 1h trades less often but captures the larger
move. No post-hoc interpretation — this is what the data says mechanically
and what the preregistered gates were designed to detect (§3: "net-of-
friction... on the project's stated capital base ($50k)").

---

## Relationship to Probe 1 Branch A

Probe 1 (`tasks/probe1_verdict.md`, 2026-04-21 earlier UTC) falsified the
**Z-score family at large** on the bar-timeframe axis: N_1.3 = 1 / 9 / 4
combos clearing gross Sharpe ≥ 1.3 on 1min / 15m / 1h respectively,
against the preregistered gate of 10. §7.6 of Probe 1 is **terminal on the
bar-timeframe axis** for the family as a whole — 30min, 2h, etc. are not
admissible without a new preregistration cycle.

Probe 2 tests a **single combo's persistence at its own two timeframes**
on the held-out 20% — not the family as a whole. Probe 1's family-level
falsification remains intact. What Probe 2 shows is:

- **Combo 865 at 1h specifically is not a 3000-sweep coincidence.** The
  training-partition gross Sharpe of 2.098 on 747 trades at 1h translates
  to test-partition **net Sharpe of 2.89 on 220 trades** — actually
  stronger on held-out bars, which is unusual and worth scrutinizing
  before Probe 3 commits capital-level resources. Possible explanations:
  (a) genuine regime-persistent edge for this specific parameter dict at
  1h on NQ; (b) the 2024-10-22 → 2026-04-08 test window happened to be
  favorable to this combo's bias; (c) smaller-sample variance amplifying a
  real-but-moderate edge. Probe 3 should pressure-test (b) and (c) via
  session/sub-window structure.

- **Combo 865 at 15m is a training-partition artifact.** Its training
  Sharpe of 1.386 on 2,786 trades collapses to net Sharpe −0.55 on 780
  test trades. The 3000-sweep-coincidence hypothesis is the best
  explanation here — 2,786 trades at $5/contract × ~87 contracts per
  trade = absolute friction headwind the training edge never had to
  actually pay (v10 sweeps used zero friction).

- **Signal family on NQ remains falsified at the bar-timeframe axis**
  per Probe 1. What's not falsified is **one specific parameter
  realization at one specific timeframe** — which is the narrow thing
  Probe 2 was designed to test.

---

## §4 Branch decision (mechanical)

**PASS → Probe 3 authorized.** Both timeframes were tested; 1h passed all
three gates; 15m failed all but the trade-count gate. Per §4 branches:

> PASS — Probe 3 authorized:
> - At least one of {15m, 1h} satisfies gates (1) AND (2) AND (3)
>   simultaneously.
> - Next action: draft `tasks/probe3_preregistration.md` for the full
>   Option Y session-structure sweep... on the test-passing timeframe.
>   Spawn a fresh LLM Council BEFORE signing Probe 3.

The test-passing timeframe is **1h**. Probe 3 scope is therefore pinned
to 1h only — not 15m, not 1min — per §4's "test-passing timeframe" clause.

---

## Irrevocable commitments honored

Per §5 of the preregistration:

1. **Results read mechanically** — gates applied as written; no threshold
   relaxation attempted despite the 15m collapse.
2. **Combo 865 parameters frozen** — no substitution, no "865-ish"
   neighbors.
3. **No post-hoc timeframe switching** — N/A (1h passed on its merits; no
   need to escape to a different TF).
4. **No post-hoc gate relaxation** — N/A (1h passed comfortably).
5. **Scope lock stands** — NQ/MNQ only.
6. **Probe 3 still requires its own council** — to be convened before
   Probe 3 signing.

---

## Next actions

1. **Commit this verdict bundle** (verdict doc + readout + parquets +
   CLAUDE.md banner update).
2. **Convene LLM Council** on Probe 3 scope:
   - Option Y axes: session_filter × parameter-holdover rule × sample
     size (the 1h 2026-04-08 test window gave 220 trades — a session sweep
     halves or quarters that; preregister a trade-count floor).
   - Key axis decisions for council: which session filter set? Should
     combo 865's non-session params stay frozen, or sweep a neighborhood?
     How many combos × filters before "we're just re-running Probe 1 on a
     different axis"?
3. **Draft `tasks/probe3_preregistration.md`** based on council synthesis.
4. **Sign Probe 3** only after council + user explicit authorization.
5. **Update memory** with Probe 2 verdict (new `memory/project_probe2_combo865_1h_pass.md`).

---

## Files

- Readout: `tasks/_probe2_readout.py`
- JSON record: `data/ml/probe2/readout.json`
- Trade parquets: `data/ml/probe2/combo865_{15m,1h}_test.parquet`
- Manifests: `data/ml/probe2/combo865_{15m,1h}_test_manifest.json`
- Remote launcher: `tasks/_run_probe2_remote.py`
- Preregistration (signed): `tasks/probe2_preregistration.md`
- Council authority: `tasks/council-report-2026-04-21-probe1-branch-a-fork.html`
- Branch A verdict (precedent): `tasks/probe1_verdict.md`
