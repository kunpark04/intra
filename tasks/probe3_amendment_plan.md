# Probe 3 Verdict Amendment Plan — Post-Audit Corrections

**Date authored**: 2026-04-22 UTC
**Scope**: Landing the code-logic-reviewer + stats-ml-logic-reviewer findings
(W1–W6) into the three artifacts that describe Probe 3 (verdict, out-of-repo
memory, CLAUDE.md carve-out), then committing the in-repo subset.

**Authority**: User authorized "Option A" (verdict amendment with W1–W4 fixes
plus Pending E1/E2 callouts for W5/W6) before the session compacted.

**Anti-mission-creep**: This plan covers **narrative amendment only**. It does
**not** re-run gates, re-compute posteriors from first principles, or touch
`data/ml/probe3/**`. The JSON ground truth is correct; the errors live
entirely in interpretive prose that drifted from the JSON.

---

## Phase 0 — Documentation Discovery (already complete)

Audit findings already synthesized pre-compaction. Reproduced here so the
Do-phase has a self-contained brief and does not need to re-invoke the
reviewers.

### Audit sources
- `C:\Users\kunpa\.claude\agents\code-logic-reviewer.md` — code-review
  specialist, CRITICAL/WARN/INFO severity schema. Reviewed
  `tasks/probe3_verdict.md` + gate scripts + JSON ground-truth.
- `C:\Users\kunpa\.claude\agents\stats-ml-logic-reviewer.md` —
  read-only statistical auditor with adversarial posture. Reviewed
  posterior arithmetic + gate-independence claims + sample-size
  disclosures.

### Ground-truth files consulted
- `data/ml/probe3/readout.json` — branch routing truth
- `data/ml/probe3/1h_ritual.json` — §4.4 cell-level numbers
- `data/ml/probe3/15m_nc.json` — §4.3 cell-level numbers
- `data/ml/probe3/regime_halves.json` — §4.1 half-split numbers
- `data/ml/probe3/param_nbhd.json` — §4.2 neighborhood numbers
- `tasks/probe3_multiplicity_memo.md` §6.4 — posterior derivation

### Findings catalog

| ID | Severity | Finding | Fix layer |
|---|---|---|---|
| W1 | WARN | Overnight-vs-RTH per-trade ratio written as `$1,592 / $232 ≈ 6.9×`. Re-derivation from 1h_ritual cells gives **$1,184 / $343 ≈ 3.45×** with p ≈ 0.11 under a Welch t on mean-trade PnL. The 6.9× figure cannot be reproduced from the JSON. | Verdict narrative + memory + CLAUDE.md |
| W2 | WARN | Sizing policy bullet reads "combo 865 runs on 87 contracts at $500 risk / 17.02 pt stop / $2 point value on MNQ = 87". Arithmetic: 500 / (2 × 17.02) = **14.69 ≈ 15 contracts**, not 87. The "87" appears to be a typo or stale pre-refactor number. | Verdict §5-next + memory |
| W3 | WARN | Posterior `0.91` is a single-point claim; BF ≈ 54× is tight to memo §6.4's specific multiplicity accounting. Sensitivity across plausible Stage-1 denominators + gate-independence assumptions puts the posterior in **[0.65, 0.85]**. Reporting a point estimate understates the model-uncertainty layer. | Verdict §5 + CLAUDE.md + memory frontmatter |
| W4 | WARN | EX_3 framed as "amplifies edge by ~45% when edge exists" and "amplifies sign both ways" — this is multiplicative language, but the underlying mechanism is **additive lock-in of first-R breakeven bar**: it shifts Sharpe by a roughly constant additive Δ that reads as big percentages on small-denominator Sharpes. On 15m SES_1 RTH the shift is Δ ≈ −0.34 Sharpe (from −0.78 to −1.11), similar magnitude to 1h PASS-cell additions. | Verdict §4.3 obs #2 + §4.4 "Why passes happen" + Relationship bullet + CLAUDE.md + memory |
| W5 | WARN (deferred) | No pre-commitment to **which** deployment variant (EX_0 vs EX_3, SES_0 vs SES_2) is the single ship candidate. Probe 3 reports 4 variants passing §4.4 and implicitly lets Phase E1 pick — this is a Stage-3 multiplicity inflation (pick-the-best within the PASS set). | Deferred to Phase E1 council scope — callout only in verdict post-audit section |
| W6 | WARN (deferred) | No preregistered falsification criteria for paper-trade phase (kill-switch trade count, min Sharpe floor, max DD before decommission). Post-audit section can flag the gap; resolution is a Phase E2 preregistration artifact. | Deferred to Phase E2 preregistration — callout only in verdict post-audit section |

### Allowed-APIs note
- The `Edit` tool operates on exact-string match with unique `old_string`. For
  each target below, the `old_string` has been verified unique by line-range
  read.
- The `Write` tool fully replaces file contents. Used only for new files
  (this plan) — never for partial edits.
- `git add <explicit-paths>` — do NOT use `git add -A` or `git add .` because
  the working tree has ~30 untracked scratch `.py` files from prior phases
  (`tasks/_check_v3_proc.py`, etc.) that must not be swept into the
  amendment commit.

### Anti-patterns to avoid
- Do NOT re-run gate scripts; the JSON is authoritative.
- Do NOT recompute the posterior from first principles in this plan; the
  range 0.65–0.85 is the audit output, not a derivation to redo.
- Do NOT touch `data/ml/probe3/**` artifacts.
- Do NOT amend the preregistration (`tasks/probe3_preregistration.md`) —
  it's signed and frozen at `f8447af`/`8636167`.
- Do NOT amend out-of-repo memory files via `git commit` (they're not
  tracked).

---

## Phase 1 — Amend `tasks/probe3_verdict.md` (in-repo)

**Goal**: Land W1–W4 in the verdict's interpretive prose + add a new
"Post-audit corrections (2026-04-22)" section + Pending E1/E2 callouts for
W5/W6.

**Task tracker**: `#48`.

### Target edits

Edit 1 — §4.3 observation #2 (reframe W4 from "amplifies sign" to "additive
lock-in"):

- **Target lines**: 115–118 (`tasks/probe3_verdict.md`)
- **Old string** (unique):
  `2. **EX_3 amplifies sign**, both ways. On 15m SES_1 RTH, EX_3 breakeven-after-1R\n   drives Sharpe from −0.777 to **−1.114** — breakeven exit bleeds when the\n   underlying edge is negative. This is the mirror of its +45–48% upside on 1h\n   (§4.4 below). It's not a universal upgrade.`
- **New content**:
  `2. **EX_3 acts as an additive Sharpe shift, not a multiplicative amplifier.**\n   On 15m SES_1 RTH, EX_3 breakeven-after-1R shifts Sharpe by Δ ≈ −0.34\n   (from −0.777 to −1.114). On 1h SES_0 and SES_2 PASS cells, EX_3 shifts\n   Sharpe by Δ ≈ +1.3 and Δ ≈ +1.6 respectively. The "+45–48%" framing\n   elsewhere in earlier drafts reads as a large percentage only because the\n   pre-EX_3 Sharpes are modest denominators; the mechanism is first-R\n   breakeven lock-in converting partial runners into realized PnL. Read as:\n   "EX_3 adds a roughly-constant trade-quality shift; the shift is helpful\n   where entries have edge, hurtful where they don't." Not a universal\n   upgrade.`
- **Acceptance check**: After edit, `Grep "amplifies sign"` on the file
  returns zero matches.

Edit 2 — §4.4 "Why passes happen" (W1 arithmetic fix + W4 reframe):

- **Target lines**: 156–164
- **Old string** (unique):
  `Why the passes happen:\n- **SES_2 overnight-only** (143 trades, net Sharpe 3.322 native / 4.945 EX_3) is\n  stronger than SES_0 all-sessions (220 trades, 2.895 native / 4.207 EX_3) in\n  Sharpe, modestly weaker in absolute $/yr. The per-trade edge is larger\n  overnight ($1,592 mean native) than in RTH ($232 mean native).\n- **EX_3 breakeven-after-1R is the strongest exit** in every session, with a\n  +45% lift over native on SES_0 and +49% on SES_2. On PASS cells, EX_3 gains\n  are structural (it converts early-runner partials into locked gains when price\n  retraces), not statistical noise.`
- **New content**:
  `Why the passes happen:\n- **SES_2 overnight-only** (143 trades, net Sharpe 3.322 native / 4.945 EX_3) is\n  stronger than SES_0 all-sessions (220 trades, 2.895 native / 4.207 EX_3) in\n  Sharpe, modestly weaker in absolute $/yr. The per-trade edge is larger\n  overnight than in RTH — **approximately $1,184/trade (overnight) vs\n  $343/trade (RTH), a ratio of ~3.45×** — with a Welch t on mean-trade PnL\n  giving p ≈ 0.11 at these sample sizes (143 overnight vs 65 RTH). The ratio\n  is economically meaningful but not formally significant; read it as\n  "large enough that a venue blocking overnight will break the edge,"\n  not "multi-sigma separation."\n- **EX_3 breakeven-after-1R is the strongest exit** in every session. On\n  PASS cells, EX_3 shifts Sharpe by a roughly-additive Δ ≈ +1.3 (SES_0) /\n  +1.6 (SES_2) by converting partial runners into locked gains when price\n  retraces. This is an **additive lock-in effect**, not a multiplicative\n  amplifier — the "+45% / +49%" framing in earlier drafts reads as big\n  percentages only because native Sharpes are modest denominators. The\n  mechanism is structural (entry→1R→breakeven-stop trajectory), not\n  statistical noise, but see §4.3 obs #2 for the same Δ magnitude in\n  negative territory.`
- **Acceptance check**: `Grep "\\$1,592"` returns zero matches; `Grep "3\\.45"`
  returns ≥ 1 match.

Edit 3 — Insert new "Post-audit corrections (2026-04-22)" section:

- **Target location**: Between `## §5 Branch routing (mechanical)` block
  (ending at line 202) and `## Relationship to Probe 1 Branch A and Probe 2`
  (starting at line 204).
- **New section content**:

  ```markdown
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

  ### Correction W2 — contract sizing
  Prior draft text: "combo 865 runs on 87 contracts at $500 risk / 17.02 pt
  stop / $2 point value on MNQ = 87". Corrected: 500 / (2 × 17.02) =
  **~15 MNQ contracts** at fixed $500 risk. The "87" figure appears to be
  a typo or stale pre-refactor number.

  ### Correction W3 — posterior range
  Prior draft text: "Posterior P(genuine edge | F=0) = 0.91 (BF ≈ 54×)".
  Corrected: **posterior ∈ [0.65, 0.85]** reflecting sensitivity across
  three modeling choices: (a) Stage-1 multiplicity denominator (Probe 2
  pool of 1 vs family-level sweep space); (b) gate independence assumption
  (§4.1 aggregate and §4.2 center cell both include the same 220-trade
  measurement, so gates share evidence); (c) pre-audit posterior math
  did not discount the §4.4 "pick one of 4 PASS variants" deployment-
  variant selection. The 0.91 point estimate remains reachable under one
  combination of (a/b/c) but is the upper edge of the plausible range,
  not the central reading. BF in [10×, 20×] under the central reading.

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
  Probe 3 reported 4 cells passing §4.4 (SES_0 × {EX_0, EX_2, EX_3} + SES_2
  × {EX_0, EX_2, EX_3}, with EX_1 ≡ EX_0 on this timeframe). **No single
  ship variant has been preregistered.** Phase E1 council must select one
  (EX × SES) combination before Phase E2 preregistration is drafted,
  framing the decision under Rule 1 (Stage-3 multiplicity: 4 PASS cells
  is a "pick-the-best" inflation beyond the §4.4 gate's already-
  accounted Stage 2 count) and Rule 2 (P(PASS|H1)/P(PASS|H0) for the
  chosen cell).

  ### Pending E2 — W6 (paper-trade falsification criteria)
  Preregistration §5.6 binds Phase E1 council convening but does not
  itself specify **paper-trade kill-switch criteria** (max DD before
  decommission, min trades before go/no-go, max consecutive losses,
  Sharpe-trajectory floor). These belong in `tasks/combo865_1h_paper_trade_plan.md`
  (Phase E2 preregistration artifact), signed before paper trading begins.
  Absent explicit criteria, "paper trade until something feels wrong" is
  not a falsifiable commitment and violates the preregistration discipline
  that Probes 1–3 established.

  ### Supportive findings (unchanged by corrections)
  - §4.1 H1 vs H2 mean-per-trade difference p ≈ 0.70 (not regime-dependent
    collapse — just natural variance across the split).
  - §4.3 15m observation 0/16 is a **ceiling** (threshold was ≤ 2), so the
    gate cleared with floor-to-ceiling margin.
  - Test-partition Sharpe is expected to regress from 2.89 toward the
    [1.0, 3.5] interval under standard sample-size decay; memo §2 test>train
    regression expectation is directionally aligned.
  ```

- **Acceptance check**: `Grep "Post-audit corrections \(2026-04-22\)"`
  returns exactly one match.

Edit 4 — Relationship section EX_3 bullet (L218–220):

- **Old string** (unique):
  `- Probe 3 adds: a **deployment-config finding** — EX_3 breakeven-after-1R\n  amplifies edge by ~45% when edge exists; SES_2 overnight-only is the\n  cleanest slice; RTH-only is not a viable venue for combo 865.`
- **New content**:
  `- Probe 3 adds: a **deployment-config finding** — EX_3 breakeven-after-1R\n  delivers an additive Sharpe lock-in (Δ ≈ +1.3 on SES_0, +1.6 on SES_2;\n  Δ ≈ −0.34 on 15m SES_1 RTH) when edge exists; SES_2 overnight-only is\n  the cleanest slice; RTH-only is not a viable venue for combo 865. See\n  post-audit corrections above for why "+45%" framing is misleading.`

Edit 5 — §5 next-actions sizing bullet (L307–310):

- **Old string** (unique):
  `   - Sizing policy: fixed $500 (repo default) vs fractional Kelly vs\n     discrete contract count (combo 865 runs on 87 contracts at $500 risk /\n     17.02 pt stop / $2 point value on MNQ = 87; for MNQ-only broker\n     accounts may cap contract count)`
- **New content**:
  `   - Sizing policy: fixed $500 (repo default) vs fractional Kelly vs\n     discrete contract count. At $500 fixed risk / 17.02 pt stop /\n     $2 point value on MNQ, **the contract count is 500 / (2 × 17.02) =\n     ~15 contracts** (the "87" figure in an earlier draft was a typo; see\n     post-audit correction W2). MNQ-only broker accounts with integer\n     contract floors should still accommodate 15 cleanly; the sizing\n     question is really about Kelly-vs-fixed variance discipline, not\n     contract capacity.`

Edit 6 — §5 branch-table posterior cell (L185):

- **Old string** (unique):
  `| 0 | **PAPER_TRADE** | **0.91** | Fresh LLM Council on paper-trade scope, then preregister paper-trade plan, then sign |`
- **New content**:
  `| 0 | **PAPER_TRADE** | **0.65–0.85** (range; see W3) | Fresh LLM Council on paper-trade scope, then preregister paper-trade plan, then sign |`

Edit 7 — Bottom-line posterior line (L16–17):

- **Old string** (unique):
  `Posterior P(genuine edge | F=0) = **0.91** (prior 0.167; BF ≈ 54× in favor of H1\nper \`tasks/probe3_multiplicity_memo.md\` §6.4).`
- **New content**:
  `Posterior P(genuine edge | F=0) ∈ **[0.65, 0.85]** (prior 0.167; BF ∈ [10×, 20×]\nunder the central reading, up to ≈ 54× at the upper edge — see post-audit\ncorrection W3 and \`tasks/probe3_multiplicity_memo.md\` §6.4).`

### Verification checklist (Phase 1)
- [ ] `Grep "amplifies sign"` on `tasks/probe3_verdict.md` → zero matches
- [ ] `Grep "\\$1,592"` on verdict → zero matches
- [ ] `Grep "87 contracts"` on verdict → zero matches
- [ ] `Grep "Post-audit corrections \\(2026-04-22\\)"` on verdict → exactly one match
- [ ] `Grep "0\\.65–0\\.85"` on verdict → ≥ one match
- [ ] `Grep "Δ ≈"` on verdict → ≥ three matches (additive-shift language)
- [ ] Visual: bottom-line block + §5 table + post-audit section all agree on
  posterior range [0.65, 0.85]
- [ ] Visual: no orphan references to 0.91 as a bare point estimate outside
  the upper-edge context

### Anti-pattern guards
- Do NOT delete the §5 branch routing table — only edit the F=0 row.
- Do NOT modify gate numbers (Sharpe, n_trades, $/yr) anywhere — those
  come from JSON and are correct.
- Do NOT change the "Bottom line: F=0, all four gates PASS, branch =
  PAPER_TRADE" claim. Corrections affect confidence estimates + framing,
  not the branch decision.

---

## Phase 2 — Amend `memory/project_probe3_combo865_pass.md` (out-of-repo)

**Goal**: Propagate W1–W4 to the out-of-repo memory pointer so future
sessions loading MEMORY.md see the corrected numbers.

**Task tracker**: `#49`.

**Critical**: This file lives at
`C:\Users\kunpa\.claude\projects\C--Users-kunpa-Downloads-Projects-intra\memory\project_probe3_combo865_pass.md`
— it is **not** in the git working tree. Phase 4's `git add` must NOT
include this path.

### Target edits
1. **Frontmatter description** (L3): replace `posterior 0.91` with
   `posterior range 0.65–0.85 after audit`.
2. **Body posterior line** (L18): `P(genuine edge | F=0) = 0.91 from prior
   0.167 (Bayes factor ≈ 54×)` → `P(genuine edge | F=0) ∈ [0.65, 0.85]
   from prior 0.167 (BF ∈ [10×, 20×] central, up to ~54× at upper edge;
   see verdict W3 correction)`.
3. **Structural finding block** (L44): `Per-trade gross edge is ~4–7× larger
   overnight than in RTH` → `Per-trade net edge is **~3.45×** larger
   overnight ($1,184/trade vs $343/trade, p ≈ 0.11)`.
4. **DO NOT bullet** (L54–57): `DO NOT assume EX_3 breakeven-after-1R is a
   universal upgrade. It gains +45% Sharpe on §4.4 PASS cells but amplifies
   losses on FAIL cells...` → reframe as "EX_3 is an additive Sharpe
   shift (Δ ≈ +1.3 / +1.6 / −0.34 across cells), not a proportional
   amplifier. Helpful where entries have edge, hurtful where they don't."
5. **Paper-trade scope bullet** (L69–70): `combo 865 runs on 87 MNQ
   contracts @ $500 risk / 17.02 pt stop / $2 point value` → `combo 865
   runs on **~15 MNQ contracts** @ $500 risk / 17.02 pt stop / $2 point
   value (500/(2×17.02) = 14.69)`.

### Verification checklist (Phase 2)
- [ ] `Grep "0\\.91"` on memory file → matches are only inside "upper edge"
  context, not as bare point estimate
- [ ] `Grep "87 MNQ contracts"` on memory file → zero matches
- [ ] `Grep "4–7×"` on memory file → zero matches
- [ ] `Grep "3\\.45"` on memory file → ≥ one match

### Anti-pattern guards
- Do NOT `git add` this file in Phase 4.
- Do NOT regenerate MEMORY.md index — description-field change does not
  require index regeneration since the index one-liner doesn't quote
  the numeric posterior.

---

## Phase 3 — Amend `CLAUDE.md` Probe 3 carve-out bullet (in-repo)

**Goal**: Propagate W1, W3, W4 to the canonical project contract so that
every future session reading CLAUDE.md sees the corrected summary.

**Task tracker**: `#50`.

### Target edits
- **Target region**: lines 69–93 (the Probe 3 carve-out bullet).
- **L79**: `F-count = **0**. Branch = **PAPER_TRADE** per §5 (posterior
  0.91, prior 0.167; BF ≈ 54×)` → `F-count = **0**. Branch =
  **PAPER_TRADE** per §5 (posterior ∈ **[0.65, 0.85]** range after
  post-audit correction W3; prior 0.167; BF in [10×, 20×] central, up to
  ~54× at upper edge)`.
- **L83–85**: `EX_3 breakeven-after-1R is the strongest exit variant
  (+45% Sharpe lift over native on SES_0, +49% on SES_2) but also
  amplifies losses where edge is negative — not a universal upgrade` →
  `EX_3 breakeven-after-1R is the strongest exit variant, delivering an
  **additive Sharpe shift Δ** (≈ +1.3 on SES_0, +1.6 on SES_2, −0.34 on
  15m SES_1 RTH) — first-R breakeven lock-in, not a proportional
  amplifier; helpful where entries have edge, hurtful where they don't`.
- Additional qualifier after the overnight concentration finding: add
  "per-trade ratio ~3.45×, p ≈ 0.11 on Welch t — economically
  meaningful, not formally significant".

### Verification checklist (Phase 3)
- [ ] `Grep "posterior 0\\.91"` on `CLAUDE.md` → zero matches
- [ ] `Grep "\\+45%"` on `CLAUDE.md` in Probe 3 bullet region → zero matches
- [ ] `Grep "0\\.65"` on `CLAUDE.md` → ≥ one match in Probe 3 region
- [ ] Diff shows ONLY Probe 3 bullet region changed; no other strategy
  contract blocks altered

### Anti-pattern guards
- Do NOT modify the Probe 1 or Probe 2 bullets (they're correct).
- Do NOT modify any of the other ML#2 / signal-family / friction blocks —
  scope is strictly the Probe 3 carve-out.
- Do NOT change the instrument scope, R:R language, or remote-execution
  policy — those are unrelated.

---

## Phase 4 — Commit + push amendment bundle

**Goal**: Stage exactly the two in-repo files (verdict + CLAUDE.md),
commit with a descriptive message referencing W1–W4 + E1/E2 callouts,
push to `origin/master`.

**Task tracker**: `#51`.

### Steps
1. `git status` → confirm working tree has `tasks/probe3_verdict.md` and
   `CLAUDE.md` modified; confirm untracked scratch files are still
   untracked (do not stage them).
2. `git diff tasks/probe3_verdict.md CLAUDE.md` → eyeball the diff one
   final time.
3. `git add tasks/probe3_verdict.md CLAUDE.md` (explicit paths, NOT
   `-A` or `.`).
4. `git commit` with HEREDOC message:

   ```
   Probe 3 verdict amendment: audit corrections (W1-W4) + pending E1/E2 callouts

   Post-audit pass (code-logic-reviewer + stats-ml-logic-reviewer):
   - W1: overnight/RTH per-trade ratio corrected from ~6.9× to ~3.45×
     ($1,184/$343, p ≈ 0.11). The $1,592/$232 numbers in the prior draft
     were not reproducible from 1h_ritual.json.
   - W2: contract sizing corrected from "87 contracts" (typo) to ~15 MNQ
     contracts at $500 fixed risk / 17.02 pt stop / $2 point value.
   - W3: posterior reported as range [0.65, 0.85] rather than point
     estimate 0.91. Sensitivity sources: Stage-1 denominator choice,
     gate-independence assumption, deployment-variant-selection
     (Stage 3) inflation. BF in [10×, 20×] central reading.
   - W4: EX_3 breakeven-after-1R reframed as additive Sharpe shift Δ
     (≈ +1.3 SES_0, +1.6 SES_2, −0.34 15m SES_1 RTH) rather than
     multiplicative "+45%/+49%" amplifier. Mechanism: first-R
     breakeven lock-in converting partial runners into realized PnL.
   - W5 (deferred to Phase E1 council): no single deployment variant
     preregistered among the 4 PASS cells. Stage-3 multiplicity is
     unaddressed.
   - W6 (deferred to Phase E2 preregistration): no paper-trade kill-
     switch criteria. "Paper trade until something feels wrong" is
     not a falsifiable commitment.

   F-count, branch routing (F=0 → PAPER_TRADE), and gate PASS/FAIL flags
   are unchanged. JSON ground truth at data/ml/probe3/*.json was correct;
   corrections affect interpretive prose only.
   ```

5. `git push origin master`.
6. `git log -1` → confirm commit landed with correct message.

### Verification checklist (Phase 4)
- [ ] `git status` post-commit: working tree clean except for the same
  untracked scratch files that were untracked pre-commit
- [ ] `git log -1 --stat` shows exactly 2 files changed: `tasks/probe3_verdict.md`
  and `CLAUDE.md`
- [ ] Push succeeds; `git log origin/master -1` shows the amendment
  commit as HEAD
- [ ] No memory file (`memory/project_probe3_combo865_pass.md`) appears
  in the staged diff

### Anti-pattern guards
- Do NOT `git add -A`. Scratch files (`_check_v3_proc.py`, `_poll_v3_eval.py`,
  `_remote_cache_sizes.py`, `_restart_v3_from_s3.py`, `_restart_v3_isolated.py`,
  `_run_v3_eval_remote.py`, `_wait_v3_eval.py`, `_wait_v3_subproc.py`,
  `_wait_v3_v2.py`) are unrelated to this amendment.
- Do NOT force-push. This is a forward commit on master.
- Do NOT amend the preceding verdict commit (`18a22ee`). Keep this a
  separate commit so the audit trail is visible.
- Do NOT `--no-verify`. Any pre-commit hook failure should be diagnosed,
  not bypassed.

---

## Phase 5 (deferred) — Phase E1 LLM Council on paper-trade scope

**Not authorized for this session.** Resolves W5 (deployment-variant
lock).

Sketch only:
- Frame the council under `feedback_council_methodology.md` Rule 1
  (Stage-3 multiplicity: picking 1 of 4 §4.4 PASS variants inflates
  the effective gate) + Rule 2 (P(PASS|H1)/P(PASS|H0) for the chosen
  cell).
- Axes: EX_0 vs EX_3; SES_0 vs SES_2; fixed-$500 vs Kelly vs discrete-15
  sizing; broker adapter selection; min trades before go/no-go.
- Output: single preregistered paper-trade variant + scope lock.

---

## Phase 6 (deferred) — Phase E2 paper-trade preregistration

**Not authorized for this session.** Resolves W6 (paper-trade
falsification criteria).

Sketch only:
- Artifact: `tasks/combo865_1h_paper_trade_plan.md` (unsigned draft →
  pre-sign review → signing commit, matching the Probe 3 discipline).
- Required sections: kill-switch trade count, min Sharpe floor,
  max DD before decommission, max consecutive losses, go/no-go
  decision horizon.
- Signing condition: Phase E1 council output lock + user explicit
  authorization.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Edit-string mismatch (file drifted since audit read) | Phase 0 verified line numbers match expected via full Read. If Edit fails with "old_string not unique / not found", re-Read the surrounding block before retrying. |
| Unintended staging of scratch files | Phase 4 explicitly uses `git add <paths>`, not `-A`. Verification checklist catches it. |
| Memory file accidentally committed | Memory file lives outside the git tree; physically cannot be staged. No mitigation needed, but verification checklist confirms anyway. |
| Posterior range [0.65, 0.85] is too lossy / too tight | Audit output is directional; wider/narrower sensitivity analysis belongs in Phase E1 council if raised. Not a blocker for the narrative correction. |
| W4 reframing disagrees with Probe 2/3 framing elsewhere | Scope strictly Probe 3 artifacts; Probe 2 verdict predates EX_3 language and does not need amendment. |

---

## Done definition

- All W1–W4 corrections land in verdict + memory + CLAUDE.md.
- Post-audit corrections section appears in verdict with W1–W4 fixes
  + W5/W6 deferred callouts.
- Commit on `origin/master` stages exactly 2 files and references
  W1–W6 in message body.
- No JSON ground-truth file, no preregistration file, no unrelated
  scratch file appears in the diff.
- Task tracker: #48, #49, #50, #51 all marked `completed`.

---

## References

- `tasks/probe3_verdict.md` — target of Phase 1
- `tasks/probe3_preregistration.md` (signed `f8447af` / `8636167`) — frozen, do not touch
- `tasks/probe3_multiplicity_memo.md` §6.4 — posterior derivation authority
- `data/ml/probe3/readout.json` — branch routing ground truth
- `data/ml/probe3/{1h_ritual,15m_nc,regime_halves,param_nbhd}.json` — gate ground truth
- `CLAUDE.md` lines 69–93 — Probe 3 carve-out contract
- `memory/project_probe3_combo865_pass.md` — out-of-repo pointer memory
- `C:\Users\kunpa\.claude\agents\code-logic-reviewer.md` — audit agent 1
- `C:\Users\kunpa\.claude\agents\stats-ml-logic-reviewer.md` — audit agent 2
- `feedback_council_methodology.md` — Rule 1 / Rule 2 framing for Phase E1
