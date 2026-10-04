# Context 1.4 in-season confirmation preregistration (issue 164)

**This is a pre-score protocol amendment.** The original, unmerged PR #165 draft
(commit `a118ef21364232090ab084d6ea58fdb81f0c18e7`) used the eventual final
2026 Massey target. Before merge, and before any Context 1.4 2026 predictive
score was opened, that endpoint was corrected to in-season posterior-predictive
game validation through Week 6. No Context 1.4 2026 score influenced this
amendment. The final merged commit containing this document, the
[registered configuration](../config/context_v1_4_validation.json), and its
[SHA-256 registration](../data/processed/context_v1_4_validation/preregistration.json)
is the authoritative preregistration. The draft commit is retained as amendment
history, not as an alternative decision rule. **Commit this correction before
running or inspecting Context 1.4's 2026 confirmation.**

## Question and existing evidence

The question is whether the frozen Context 1.4 candidate behaves acceptably on
candidate-score-unseen 2026 games, or whether it shows a sufficiently clear
out-of-sample regression to stop promotion over Context 1.3. This is a
confirmation and veto gate, not a fresh search for a candidate. Issue #158 / PR
#159 selected the fixed positive-only moderation from a constrained family:
`alpha = 0.75` improved final posterior rank NLL over Context 1.3 in every
2022–2025 development season, including 2022, and at every shared 2023–2025
checkpoint. It preserved essentially all of Context 1.3's preseason advantage.
Issue #160 / PR #161 froze the builder before 2026 candidate results were
inspected. The semantic SHA-256 is
`db84045d2f7d80b1648360da52ef1e71f76093696cad4e4e7bbdbd57a086573f`.
There is no tunable parameter in this validation.

History 1.1 remains a first-class production lineage. PR #159 already measured
the pooled 2023–2025 Context 1.4 minus History 1.1 posterior NLL: preseason
−0.0331; September +0.0076; October 1 +0.0185; October 2 +0.0208; November 1
+0.0337; November 2 +0.0372; December +0.0420. Context 1.4 improves Context
1.3 but does not remove the historical later-season History crossover. Whether
2026 repeats that crossover is a future observation. History is a descriptive
comparison arm here; beating it is **not** required for Context promotion.

## Two 2026 evidence strata

**Candidate-score-unseen confirmation, Weeks 1–5.** Score all eligible
completed 2026 regular-season FBS/FCS games from the beginning of the season
through Week 5. These outcomes existed during later research, so this is not a
pristine prospective sample. They did not choose the `0.75` candidate, the
candidate was frozen independently, and its 2026 scores have not been viewed.

**Strictly prospective confirmation, Week 6.** After all Week 6 games are
complete, forecast every eligible Week 6 game from the official **September 27
Context 1.3 evidence state**. Construct frozen Context 1.4 from that same state.
The Week 6 games occur after the candidate freeze and protocol correction. Do
not use a new snapshot generated after those games to forecast them. Score and
make the promotion decision before generating the normal Week 6 production
snapshot if operations permit. Week 6 receives its own prominent table with
the same statistics as the full sample; it does not independently approve or
veto based on a noisy weekly estimate. A structural correctness failure in
Week 6 does stop promotion.

## Frozen official forecast origins

Use only these registered official Context 1.3 publication slots and their
pinned snapshot-local `metadata.json`, `included_games.csv`, and
`posterior_pmfs.csv`. The configuration also pins the matching History files.
These retained included-game rows establish each evidence boundary; do not
reconstruct earlier states from a mutable current-season response or use live
or temporary snapshots to make a denser grid.

| Publication slot | Logical UTC evidence cutoff | Included games |
| --- | --- | ---: |
| `2026-preseason-context-1.3` | 2026-08-22 23:59:59 | 0 |
| `2026-09-08` | 2026-09-08 11:43:00.275833 | 172 |
| `2026-09-13` | 2026-09-13 12:02:55.255941 | 291 |
| `2026-09-20` | 2026-09-20 11:00:14.294077 | 410 |
| `2026-09-27` | 2026-09-27 12:27:35.698895 | 530 |

The preseason and September 8/13 Context 1.3 slots are official retrospective
backfills, generated after their logical evidence cutoffs. Their Weeks 1–5
scores are cutoff-safe reconstructions, not literally real-time forecasts.
September 20 and 27 were available before the games they forecast. The
September 27 state is the registered Week 6 forecast even if another official
state becomes available after Week 5; this keeps Week 6 genuinely prospective
under the corrected protocol.

For each completed regular-season Week 1–6 game with a valid final score,
actual UTC kickoff, and two FBS/FCS participants, select the **latest**
registered logical cutoff **strictly before kickoff**. Score the game once,
under that origin only. A game already present in that origin's included-game
evidence is leakage and aborts. Collapse identical FBS/FCS response rows by
canonical game ID before assignment; conflicting duplicate rows, a missing origin, unknown
kickoff, invalid result, incomplete source coverage, or an unexpected Week 6
origin aborts or is explicitly audited under the registered failure policy.
Use the actual kickoff for rescheduled games. After Week 6, verify the
canonical schedule is terminal or every exception is explicitly accounted for.
The operational sufficiency floors are 500 scored games overall and 40 Week 6
games; source completeness and pairing checks still apply even above them.
These floors cannot be used to omit inconvenient completed games.

## Frozen candidate and paired inference

The locked population is the exact 138 FBS team IDs in the retained 2026
Context 1.3 and History 1.1 preseason predictions; Sacramento State (`16`)
and North Dakota State (`2449`) keep their Context 1.3 cold-start History
transition fallbacks. The 128 FCS fallback IDs and uniform full-subdivision
rank support are pinned from the registered schedule. If a scored future game
has an FCS team absent from an origin's inference network, give that team the
same locked uniform 128-rank fallback for forecast scoring in every arm;
future outcomes never enter that origin's inference. For each origin, build
Context 1.4 by calling PR #161's `construct_candidate_prior` on the exact
Context 1.3 prior and fit/transfer state. Verify source-PMF parity, bitwise
identity for nonpositive Context-only subtotal and cold starts, positive-only
`0.75` moderation, centered-mixture offsets, residual scale, uncertainty,
mixture and fallback semantics. The 2026 Context fit has a retained legacy
attestation rather than a verified training-corpus digest; report that limit.

Infer Context 1.4 and 1.3 with identical included-game evidence, FCS
fallbacks, Historical Likelihood V1, team population, and deterministic
configuration: 500 iterations, `1e-9` tolerance, `0.35` damping. Require
convergence, matching game and team identities, and unchanged model/source
bytes. History 1.1 uses the same forecast origins as a descriptive arm. Abort
on any candidate, baseline, evidence, inference, provenance, or source mismatch.
Outcome scores may enter only the scoring step, after the prior and posterior
forecast states are constructed and hashed. Do not refit or recalibrate the
likelihood.

## Development-only predictive bridge

Before opening a 2026 Context 1.4 predictive score, run the frozen candidate
on the existing **2022–2025** leakage-safe future-game validation panel. Use
the standard midseason cutoff (index 3) in each season, exact historical game
evidence and Historical Likelihood V1, and the existing
[`posterior.predictive`](../src/gippyrank/posterior/predictive.py) Student-t
mixture scoring from
[`validate_predictive_game_v1.py`](../scripts/validate_predictive_game_v1.py)
and [its metric definition](predictive_validation.md). Compare Context 1.4
against Context 1.3 on identical future-game rows, reporting at least margin
NLL, margin MAE, and win Brier by season and pooled. This reuses development
data, is **not** new holdout evidence, and must not change the candidate. If it
reveals a major systematic contradiction between the selected rank-NLL gains
and game prediction, report it explicitly and suspend the decision for
explicit review. Do not search other alphas or select a convenient metric.

The bridge was run before any 2026 candidate score access. Its exact
[machine artifact](../data/processed/context_v1_4_validation/development_predictive_bridge.json)
records the season cutoffs and source hashes. The paired game counts are 618,
642, 716, and 724 (2,700 pooled). Lower is better for each metric:

| Season | C1.3 NLL | C1.4 NLL | C1.3 MAE | C1.4 MAE | C1.3 Brier | C1.4 Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 4.257291 | 4.256934 | 13.529924 | 13.525241 | 0.204875 | 0.204823 |
| 2023 | 4.240886 | 4.241738 | 13.070849 | 13.078135 | 0.187507 | 0.187686 |
| 2024 | 4.175703 | 4.175136 | 12.524298 | 12.520534 | 0.194465 | 0.194324 |
| 2025 | 4.171629 | 4.171513 | 12.462811 | 12.460822 | 0.176963 | 0.176897 |
| **Pooled** | **4.208784** | **4.208724** | **12.867945** | **12.867074** | **0.190500** | **0.190476** |

The pooled Context 1.4 minus 1.3 differences are −0.000061 margin NLL,
−0.000871 margin MAE, and −0.000024 win Brier. The 2023 game metrics regress
slightly, while the other three seasons improve slightly. The large
development rank-NLL gain does not translate into a large game-prediction gain
at this fixed midseason panel, but the bridge shows no major systematic
opposition. This observation does not retune the candidate or replace the
registered 2026 confirmation rule.

## Registered 2026 score and decision

The primary observation is each game's exact marginalized posterior-predictive
home-margin NLL under the Student-t rank-mixture, as defined in the existing
game validator. The paired difference is **Context 1.4 − Context 1.3**:
negative favors 1.4; positive favors 1.3. Give each unique game equal weight.
For the full Weeks 1–6 sample and Week 6 separately, report eligible game
count, each Context mean NLL, paired mean and median difference, deterministic
paired interval, and results by official forecast window. Use a two-sided 95%
paired percentile bootstrap of game IDs, 10,000 size-N samples with replacement,
lexically sorted IDs, NumPy `Generator(PCG64(1642026))`, and linear quantiles.
The interval describes variation among scored games in this season; common
teams and shared inference mean it is not an independent multi-season estimate.

After all guardrails and sample sufficiency pass:

- **Retain Context 1.3** only if the full-sample paired mean is above zero **and** the registered interval lies wholly above zero.
- **Promote Context 1.4** for every other valid full-sample result. The strong historical development evidence is part of this asymmetric decision; this partial season need not independently prove superiority at conventional significance.
- **Inconclusive or abort** only for substantive validation failures, including insufficient eligible data, a provenance/evidence/inference failure, or a major unresolved predictive-bridge contradiction.

Week 6 alone is not a significance gate and an opposite-sign weekly estimate
cannot override the full evidence stack. Inspect it for catastrophic predictive
degradation, unexpected calibration failure, evidence mismatch, and candidate
construction bugs. A genuine structural failure stops promotion. The registered
secondary metrics on the **same common game rows** are margin MAE, win Brier,
and central 50%, 80%, and 95% margin-interval coverage, by origin and Week 6,
for Context 1.4, Context 1.3, and History 1.1 where appropriate. They explain
behavior; do not select whichever metric favors one model after viewing results.

## Artifacts, amendment, and handoff

The follow-up validator must load this registered configuration, produce sorted,
deterministic per-game and per-origin files, summary, evidence/exclusion audit,
source and code hashes, failure report when applicable, and a human-readable
result. Provenance must name this final preregistration's commit and digest,
the frozen PR #161 candidate, each origin's snapshot/evidence hash, final game
source, team population, and Historical Likelihood. Repeated runs on identical
inputs must be byte-identical. A changed endpoint, origin list, population,
weight, interval, decision rule, or candidate after 2026 score access is a
protocol deviation and a new research cycle.

If the confirmation passes, promote the frozen candidate as the active Context
model. Regenerate the complete 2026 Context season under 1.4—preseason and
Weeks 1, 2, 3, 4, 5, and 6—at each original official evidence boundary. Those
regenerated snapshots become the canonical production 2026 Context lineage;
continue producing Context 1.4 afterward. Retain Context 1.3 artifacts and
validation provenance internally. Do not add user-facing “retrospective
reconstruction” labels to the regenerated lineage. History 1.1 remains active
independently.

## Later postseason corroboration

The eventual final 2026 Massey target may later corroborate the in-season
decision. It is **not** a prerequisite for completing issue #164 or promoting
Context 1.4 under this gate, and it cannot silently replace the registered
in-season decision.
