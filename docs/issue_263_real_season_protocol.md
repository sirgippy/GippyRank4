# Issue 263: frozen real-season Quality of Results protocol

This design authorizes research execution only. It runs no candidate inference,
selects no weight, and changes no production rankings. The objective is to examine
whether season accomplishments receive defensible credit without systematically
rewarding easier scheduling, while giving wins and losses greater significance
than Quality of Play. Modest difficult-schedule preferences are permissible;
there is no arbitrary schedule bonus or numerical equivalence target. All five
existing weights receive the same analyses. Prediction accuracy, polls, committee
rankings, selections, and agreement with any other ranking are not objectives.

The executable specification is
`data/research/issue_263_real_season_design/protocol.json`. The case inventory is
`cases.json`; `reference_descriptors.csv` and `reference_games.csv` contain only
actual outcomes and established endpoint/Context references used to select it.
This report and those artifacts together constitute the protocol. Candidate
rankings from issue 242 are deliberately not inputs to case selection.

## Evidence audit and selected population

The audit freezes repository revision
`95401c6b1699ee9750884fab2163772420951a6e` (the merged issue 261 study), the
publication configuration, exact source paths, file hashes, game-row hashes,
requested/effective cutoffs, likelihood, Context priors, inference configuration,
FBS populations, and FCS network variables. `snapshot_audit` covers every official
weekly publication available at this revision, including unselected checkpoints.

| Official 2026 checkpoint | Effective cutoff (UTC) | Eligible games | Network variables | Execute |
| --- | --- | ---: | ---: | --- |
| Week 2 | 2026-09-08 11:43:00.275833 | 171 | 253 | yes: early evidence |
| Week 3 | 2026-09-13 12:02:55.255941 | 291 | 266 | no: intermediate |
| Week 4 | 2026-09-20 11:00:14.294077 | 410 | 266 | no: intermediate |
| Week 5 | 2026-09-27 12:27:35.698895 | 530 | 266 | yes: middle checkpoint |
| Week 6 | 2026-10-04 14:51:53.251364 | 644 | 266 | no: intermediate |
| Week 7 | 2026-10-11 12:47:27.781981 | 748 | 266 | no: intermediate |
| Week 8 | 2026-10-18 13:04:45.309059 | 864 | 266 | no: intermediate |
| Week 9 | 2026-10-25 12:59:22.246522 | 975 | 266 | yes: latest available |

All are partial regular-season information, with 138 FBS teams, Historical
Likelihood V1, Context 1.4, retained Quality of Play and Quality of Record results,
and the same frozen prior artifact. Requested cutoffs equal effective cutoffs.
Week 2's 115 observed FCS variables and later weeks' 128 variables use retained
uniform full-subdivision fallback PMFs (128-rank support); keep these uncertain
variables and all eligible FCS–FCS games in the inference graph. Lower-division
exclusions and their counts remain as documented in the source metadata. Never
use only the 138 FBS rows as an inference network. There is no postseason or final
2026 evidence in this study. Week 9 is a later observed checkpoint, not late-season
completion. Three separated checkpoints bound the compute and expose differences
in evidence depth; the remaining five are audited, not an adaptive reserve pool.
Preseason has no observed Performance evidence and temporary publications are
excluded. New official weeks do not enter this frozen population automatically.

### 2025 pre-bowl and final feasibility

2025 is **unavailable for execution from retained inputs**. The normalized game
corpus and raw CFBD game responses in this checkout contain 2026 only; there are
no 2025 snapshot metadata/game/PMF bundles under `data/processed/snapshots`.
The repository does retain 136-team 2025 Context prior rows in
`data/processed/preseason/context/predictions.csv`, historical candidate priors,
DB-repair historical prior/model files, and `posterior_backtest/2025_rolling.json`.
The latter holds scores, IDs, and runtimes rather than reproducible game networks
and endpoint PMFs. The issue 200 target manifest identifies a December 7, 2025
pre-bowl ranking measurement; it is not a Context/game publication bundle.
Hashes of these inspected artifacts are in `audit_only_input_sha256`. Historical priors alone
cannot recover the missing game rows, cutoff availability, FCS population,
likelihood/source identity, or matched endpoint outputs. The current Context 1.4
annual production directory contains only 2026. Do not apply its prior to 2025 or
reinterpret repaired historical forecasts as a published 2025 checkpoint.

Pre-bowl and final should eventually be separate experiments: pre-bowl excludes
bowl/playoff results, whereas final includes postseason evidence under potentially
different roster/coaching conditions. The retained December 7 measurement does
not establish a safe game-evidence cutoff; there is no justified exact final
cutoff to freeze here. Both desired snapshots are explicitly unselected in the
specification. The bounded alternative is the three available 2026 checkpoints;
it supplies no completed-season substitute. Completed-season conclusions remain
unavailable. A separate prerequisite must acquire and freeze 2025 games, original
availability/cutoff evidence, season-specific established priors, FCS coverage,
and endpoint bundles, then amend the protocol before looking at candidate results.
No historical reconstruction, new acquisition, refit, or repair is required for
the follow-up 2026 execution ticket.

### Campbell correction, historical lineage, and availability limits

Every audited weekly bundle passes the existing Context/Performance/Quality of
Record validators and frozen loader. Context and both endpoints have identical
included rows. Every bundle excludes phantom completed 0–0 game `401866625` and
contains real Campbell–Western Carolina game `401917058`, Campbell 28–19, IDs
2097/2717. Week 2 uses the explicit issue 238 lineage (171 games versus 172 before
repair). Its correction metadata preserves source requested/effective cutoffs and
prior version; later official bundles use already corrected evidence. The issue
238 audit is itself hashed. Superseded artifacts still exist and must not be
selected by filename globbing. The September 6 temporary snapshot correctly omits
the later replacement result, as documented by issue 238; this design does not
rebuild or select it. Issue 237 corrected acquisition/normalization; issue 238
repaired retained publications. Current normalized data cannot stand in for those
cutoff-local publication rows.

These are the **established repaired publication lineages**, not proof that every
model artifact was originally deployable on each date. Context 1.4/DB repairs and
the issue 238 rebuild are retrospective. Comparisons are controlled within these
retained lineages, without replacing their priors with a new fit; they do not
claim a strictly prospective backtest. In particular, early publications use
`legacy_first_response` provenance: Week 2's FCS response was retrieved
0.284999 seconds after its recorded cutoff. Other retrieval gaps are preserved
per source in `retrieval_seconds_after_cutoff`. Do not change the historical
cutoffs, assert that kickoff proves completion availability, or silently ingest
future results to resolve this discrepancy. The as-published frozen evidence is
suitable for a fixed-input normative comparison; exact information availability
at every historical instant is not established. Do not pool across chronology as
if these were independent prospective trials. Record evidence corrections and
changes between cutoffs explicitly.

## Frozen candidate inference and gates

Use `infer_quality_of_results` at λ = **0, 0.25, 0.50, 0.75, 1** with the byte-pinned
V1 likelihood and each snapshot's retained Context prior and complete opponent
network. The game factor is `P(sign(m)|ranks)^(1−λ) p(m|ranks)^λ`. Preserve site,
subdivision, rematch grouping, full PMFs, existing prior stripping, and uniform
Performance distributions for zero-game FBS teams. No scalar strength/Elo layer,
likelihood refit, new λ, empirical schedule correction, or endpoint averaging is
permitted. Load snapshots directly from specification paths, never rediscover
current publication slots in the execution runner.

All selected sources retain deterministic damped loopy sum-product BP with
max_iterations=500, tolerance=1e−9, damping=0.35. Use cold starts, canonical team
and game ordering, and the same eligible rows for every candidate. Keep inference
rank coordinates separate from display ranks obtained by sorting expected rank
(ascending, canonical string ID for ties). Rank coordinates need not form a joint
permutation of the FBS population. FCS variables remain uncertain but are excluded
from FBS display ranking.

Before inference verify every execution-input and frozen artifact hash, row population, cutoff, version, game
ID, prior/support, Campbell correction and endpoint identity. Reuse
`_input_bundle`/`_baseline_rankings` with the pinned `PublishedInput` paths, not
`run_experiment` from issue 242: that runner discovers weeks and makes additional
adaptive selections/sweeps outside this design. The frozen loader must supply
retained FCS fallbacks. Hash the post-fallback prior PMFs as well as the prior file.

Require convergence and finite nonnegative PMFs summing to 1 within 1e−9. Rerun
λ=0/1 and compare all FBS prior-stripped PMFs to retained Quality of Record/Play:
maximum absolute cell difference ≤1e−8, expected-rank difference ≤138×1e−8.
Require identical ordinal order except for means tied within that propagated
numerical tolerance; report all such ties. A failure stops that snapshot without
relaxing settings. Retain partial failures and timings. No candidate gets a
special retry. At Week 9 reverse both input lists and rerun λ=0, .5, 1; require
max-cell agreement ≤1e−8. This checks deterministic order invariance without
adding a chronological model. No production artifact is overwritten.

## Whole-population evaluation

Infer every network variable; report all 138 FBS PMFs and summaries, retaining
zero-game teams as unrated/uniform. Rank contrasts use teams with ≥1 eligible
game; longitudinal comparisons use the intersection of rated populations and
report entrants separately. Never silently discard failed candidates or teams.
The reference descriptor table freezes the planned population and counts for
each selected cutoff.

For every snapshot and all ten unordered candidate pairs, orient differences as
higher λ minus lower λ; positive expected/display-rank differences mean a worse
rank. Report signed mean, median, mean absolute change, p05/p95 and maximum
absolute change with ID (NumPy linear quantiles), as well as complete team rows.
Compute ordinal pair inversions divided by n(n−1)/2, top 10/25 overlap, symmetric
difference, named entries/exits, and movement of the union of the two top sets.
Report boundary bands ±3 positions around 10, 12 and 25 in either endpoint
reference order; 12 is a consequential inspection boundary, not an assertion of
committee selection equivalence. For these bands report adjacent gaps, crossing
IDs and changes in PMF mass at latent ranks ≤10, ≤12 and ≤25. PMF mass is not
probability of occupying a display-ranking slot. Report adjacent λ differences
and slopes divided by .25 for each team; do not infer an optimum from smoothness.

For each team/candidate preserve the full PMF, expected/median/mode rank, rank SD,
entropy in bits, and equal-tail 50/80/95% integer quantiles and inclusive widths
(high−low+1). Quantiles use the first rank with CDF ≥ probability. For all candidate
pairs compute total variation `0.5 sum(abs(p−q))` and rank Wasserstein-1
`sum(abs(CDF_p−CDF_q))`. Show whether ordinal changes accompany substantial mean
or distribution changes; proximity alone can create large ordinal moves. Put
mean shifts alongside each candidate's SD/intervals and TV/W1; do not convert
interval overlap into a significance test. Single-team marginals do not identify
joint cross-team superiority probabilities. There are no Monte Carlo confidence
intervals for this deterministic observed-season census and no independence
assumption across connected teams, repeated snapshots or candidates. Numerical
convergence tolerance is not a measure of model uncertainty.

For each candidate track Week2→5 and Week5→9 mean/ordinal rank and SD/width
changes, per-team new-game counts, and top-set transitions. Break out changes in
opponent evidence versus own games descriptively; do not attribute all movement
to own games. Do not compare the three checkpoints as independent replications.

### Accomplishments and opponent-aware descriptions

Use every team's actual eligible W–L, game count, margins (mean, median, complete
list), close results (|margin|≤7), dominant wins (margin≥28), and FBS/FCS mix.
Schedule quality must use the integrated V1 opponent distributions at actual
venues rather than raw opponent records or mean opponent ranks. The frozen
reference is Western Michigan's complete preseason Context PMF, selected nearest
expected rank 65 (canonical ID tie break). Replace each focal team with that
same PMF in each matchup and integrate against the opponent's retained Context
posterior. The mean standardized win probability is the schedule descriptor;
smaller means harder. Also retain the complete game-probability profile and
expected win total, since a mean can conceal mixed schedules. No adjustment is
added to the rankings.

Compute a separate retrospective expectation for the actual focal and opponent
Context PMFs at that cutoff, wins minus sum of integrated win probabilities, and
per-game expected margins through the same V1 surface. These use all evidence
at the cutoff, including that game: they describe opponent-aware accomplishment,
not an out-of-sample forecast or independent validation. This inexpensive fixed
reference allows reproducible case selection without running new inference.
Cavity or game-removal diagnostics below address self-influence for a bounded
subset. FCS effects and generic fallback priors limit all these descriptions.

Before consulting candidate outputs, fix QoP ordinal bands 1–10 (elite), 11–40
(strong), 41–95 (middle), 96–138 (weak). Within each band/snapshot sort by schedule
reference mean, then string ID; partition using `numpy.array_split(...,3)` into
hard/middle/easy terciles. Also tabulate win fractions [0,.3), [.3,.6], (.6,.8),
[.8,1], raw W–L/game-count groups, and near-expected-record teams satisfying
`abs(wins−context_expected_wins)≤1`. Report sample sizes and empty cells; do not
change cutpoints after seeing candidates.

For each panel and all pairwise candidate changes report signed mean, median,
mean absolute change, SD/width changes, full distributions and top/boundary
crossings. Show descriptive Spearman associations of candidate contrasts with
win fraction, schedule descriptor, average margin and wins minus expectation,
with denominators and undefined constant-column cases. No maximizing correlations,
regression-derived schedule bonus, predictive validation score, or p-value ranking
of candidates is authorized. Contrast excellent records/easy schedules with
middling records/hard schedules and exact-W–L pairs within each reference band;
report all such pairs with harder/easier identities fixed by the schedule measure.
For near-expected-record panels compare hard/easy groups without requiring their
records or ranks to coincide. Opponent-quality wins/losses use actual game rows,
opponent Context expected rank/SD, QoP ordinal, and integrated expectation, with
FCS labels. These summaries examine systematic kinds of credit; they cannot
identify a causal schedule effect from nonrandom real schedules.

Population panels are primary. Frozen cases illustrate their mechanisms and
limitations. Candidate-dependent outliers discovered during execution belong in
a separate exploratory appendix and cannot replace the inventory or count as
independent confirmation.

## Predetermined case inventory and questions

`cases.json` lists each selected ID/name, season, snapshot, selection categories,
W–L, game count, margin, both endpoint references, and schedule/expectation context.
Selection occurs at Week 9, then follows those same IDs at Weeks 2/5/9; it does
not select new cases at an intermediate λ or extrapolate early results into the
selection. Endpoint references are permitted established information, but this
selection emphasizes endpoint disagreement and is not an unbiased population
sample. Use all population rows to judge prevalence.

The concrete Week 9 selection is below. The machine inventory contains the same
IDs at all three checkpoints, with cutoff-specific record/schedule context and
the specific question for each category.

| Team | ID | Week 9 W–L | Selection categories |
| --- | --- | --- | --- |
| Alabama | 333 | 6–2 | elite |
| Kentucky | 96 | 7–1 | strong |
| UTSA | 2636 | 7–1 | middle |
| Central Michigan | 2117 | 3–5 | weak |
| Liberty | 2335 | 6–1 | excellent record/easy schedule |
| Mississippi State | 344 | 4–3 | middling record/difficult schedule |
| Michigan | 130 | 4–3 | competitive elite loss |
| Memphis | 235 | 6–2 | narrow weaker-opponent win |
| Oregon | 2483 | 4–3 | dominant overmatched win |
| Stanford | 24 | 4–4 | endpoint disagreement; weak schedule pair |
| Massachusetts | 113 | 4–4 | weak schedule pair |
| Auburn | 2 | 4–3 | middle schedule pair |
| UConn | 41 | 4–3 | middle schedule pair |
| LSU | 99 | 6–2 | boundary 10 |
| Georgia | 61 | 6–1 | boundary 12 |
| Missouri | 142 | 5–2 | boundary 25 |

The middle pair's standardized schedule win chances are approximately .308
(Auburn) versus .697 (UConn); the weak pair's are .438 (Stanford) versus .769
(Massachusetts). These large descriptive gaps support informative case inspection,
without identifying a causal schedule effect. The four-state probes are Alabama
at Tennessee (`401856723`) and Central Michigan versus Miami (OH) (`401866476`).
The five distinct removal games are pinned in `game_diagnostics.removal_game_ids`.

The executable `select_cases` rules, applied before any new candidate inference,
are as follows. All ties use canonical string ID; game ties then use string game
ID. Deduplicate teams and retain all categories rather than create quotas.

| Rule | Selection | Question |
| --- | --- | --- |
| elite/strong/middle/weak | QoP display positions nearest 5/20/65/120 | How do winning, margins and uncertainty interact across the ranking population? |
| excellent record/easy slate | ≥5 games and ≥.8 win fraction, highest standardized schedule win chance | Does an excellent record get disproportionate easy-schedule credit? |
| middling record/hard slate | ≥5 games and .3–.6 win fraction, lowest standardized chance | Does opponent evidence explain the apparently ordinary record? |
| competitive elite loss | loss by ≤7 to QoP top10, largest absolute eligible margin | What information does a competitive loss retain? |
| narrow weaker-opponent win | win by ≤7 with retrospective Context win chance ≥.8, largest eligible margin | How much credit does crossing the outcome boundary carry? |
| dominant overmatched win | win by ≥28 with Context win chance ≥.8, largest eligible margin | Does a blowout dominate the evaluation? |
| endpoint disagreement | maximum absolute QoR−QoP expected rank | Which actual observations explain the endpoint disagreement? |
| boundaries | QoR display positions nearest 10/12/25 | Do candidate changes matter near consequential boundaries or only reorder close teams? |
| middle/weak schedule pairs | within QoP positions 41–95/96–138, identical W–L, maximize standardized schedule gap | Does the middle ambiguity or weak uncertainty reversal appear under comparable records? |

For paired rules tie-break by the ascending ID pair after absolute schedule gap.
Missing qualifying categories must be reported explicitly, with no substitute
chosen using candidate results. These rules yield a compact overlapping inventory;
the concrete rows and diagnostic game IDs are machine-frozen, so execution does
not repeat a subjective selection step. Every selected case asks its category's
question at each available cutoff and includes all five candidates, eligible
opponents and posterior uncertainty. The inventory is a 2026 team-season
inventory; there are no defensible frozen 2025 cases from the available evidence.

## Game-level diagnostics and limits

The specification freezes two FBS games, one for each elite/weak anchor, chosen
closest to retrospective Context p(win)=.5 among that team's eligible FBS games,
then game ID. For each, replace only that observation with margins **−3, +3,
+10, +35** from the focal perspective. Fix opponent points to
`max(actual_opponent_points,3)` and focal points to opponent points plus margin;
orient back to original home/away. Keep IDs, opponents, venues, dates, priors,
cutoffs and every other observation fixed. These are controlled evidence probes,
not purported football predictions or additive adjustments to actual records.
For every λ rerun the entire network and strip the same priors. Report focal,
opponent and all other FBS effects, full PMF distances and convergence.

Define outcome-boundary benefit `E(rank at −3)−E(rank at +3)` and within-win
margin benefit `E(rank at +10)−E(rank at +35)`, with positive benefit better.
Report both and their difference for every λ; avoid ratios when the denominator
is small. At λ=0, +3/+10/+35 must agree within the endpoint tolerance. These elite
and weak examples fill gaps left by issue 252 without repeating its simulations.
Retain the actual evidence baseline separately from all four artificial states.

Also remove, one at a time, the two frozen counterfactual game IDs and the
three category-selected close-loss/narrow-win/blowout game IDs, deduplicated.
`removal_game_ids` is the exact bounded inventory. Recompute the **entire**
connected network at all five weights, using unchanged priors and remaining
actual observations. Recalculate evidence counts (uniform Performance if a team
has no remaining games). Compare removed versus actual baseline expected rank,
SD, intervals, TV/W1, top-set movement and opponent effects. If a selected game is
a rematch, remove just that observation, retain the other meeting, and let the
existing engine rebuild grouped factors; do not remove the whole pair silently.
For each case show its eligible games even if not all are perturbation targets.

Each removal is a total response to changing that game with the rest fixed.
Responses overlap through cycles and opponent updates; they are neither
independent contributions nor a sum explaining the baseline. No Shapley values,
general attribution framework, exhaustive leave-one-out campaign or game-cherry
picking is needed. If helpful, the existing converged BP pair-cavity state may be
shown as a descriptive local check, clearly labeled as removing a grouped pair
and approximate in a cyclic network. It is not a replacement for the frozen
full-network perturbations or an exact causal attribution.

## Carry-forward questions from issue 261

The middle synthetic λ=.25 unconditional advantage was about .90 ranks for the
easy slate, with an interval allowing effects around two ranks. Inspect the full
middle-band record/schedule panels, all exact-record pairs, the frozen middle
pair, and changes over the three cutoffs at **every** λ. Do not subtract .90,
demand equivalence, or call an observed group gap a causal easy-schedule bias.
Real schedules, priors, selection, injuries and opponent networks differ.

Approximately-expected-record synthetic comparisons favored difficult schedules
at .25 across four profiles. The near-expected-record panels and matched W–L
pairs test the coherence of actual records under opponent-aware expectations;
identical records or ranks are not required across real schedules. Both difficult
preferences and easy advantages must be explained, without rewarding a candidate
simply for producing the smallest difference.

Weak synthetic teams reversed the uncertainty pattern. In the weak band and
frozen weak pair report SD, entropy, 50/80/95% widths and hard/easy schedule
profiles per λ and cutoff. Weak real teams exist here, so the investigation is
feasible, but a small selected group and bounded rank support limit conclusions.
Weak-case uncertainty is not sampling error of the synthetic mean. The elite/weak
four-state probes assess whether crossing a win/loss boundary has greater effect
than stretching an existing victory; the real-network answer can differ from
issue 252 and is not a universal point-to-victory exchange rate.

## Compute, reproducibility and bounded execution ticket

No smoke-test inference is necessary: retained source diagnostics already record
converged real networks. Week2/5/9 Context solves took approximately
0.388/2.018/13.540 seconds (existing hardware, 253/266/266 variables). These are
**endpoint reference timings, not measured new candidate runtimes**. Five solves
per checkpoint give a baseline estimate of `5×sum(t)=79.73 seconds`. Network
cycles, intermediate weights, binary factors and host variance can increase cost.
Budget four times retained timing per solve, at least one minute per solve,
plus IO/analysis; allow two wall-clock hours and 4 GiB RSS overall, with a
600-second single-solve limit. Record actual elapsed and peak RSS in execution.

The plan uses 15 baseline solves, 40 four-state counterfactual solves, five
removal solves per unique frozen game, and three Week9 input-order reversals.
The exact total is in `execution.maximum_network_solves` (at most 83 with five
unique removal games). At the largest reference timing, 83 solves are about
19 minutes before the safety multiplier; a fourfold bound is about 75 minutes,
within the two-hour cap; applying the one-minute planning floor yields about
83 minutes plus IO/analysis. The baseline's 79.73 seconds is not a promise for the
whole diagnostic study. Zero simulations or acceptance searches are required.

Use a serial worker to bound memory; share binary/margin factor surfaces within
a snapshot, discard between snapshots. Two dense 138² float64 surfaces for 975
games occupy about 283 MiB before BP/grouped factors, messages and Python overhead;
FCS surfaces are smaller. Keep PMFs in deterministic gzip (`mtime=0`). Checkpoint
by input hashes, λ, complete inference configuration and diagnostic perturbation,
never by label alone; only resume verified matching converged results. Preserve
failed attempts and stop affected snapshots at resource limits. Do not relax
convergence, shrink populations, drop FCS games, alter cutoffs or silently change
the design. Report incomplete coverage and request a separate amendment if needed.

The next execution ticket should implement a narrowly scoped runner under
`src/gippyrank/research` and a script wrapper, with this sequence:

1. Run `uv run python scripts/design_issue_263_real_seasons.py` to verify frozen
   inputs/artifacts. Load only specification-selected paths and inventory IDs.
   Record the execution commit, pinned model hashes, dependency lock and runtime.
2. Execute 15 baseline solves with endpoint/numerical/population gates; retain full
   PMFs and `rankings.csv`, `convergence.csv`, `endpoint_parity.json`.
3. Produce all ten pairwise contrasts per snapshot, accomplishments/schedule/
   uncertainty panels and longitudinal transitions before interpreting cases.
   Save `population_contrasts.csv` and `accomplishment_panels.csv` plus full rows.
4. Report frozen cases using `cases.json` and per-game actual evidence; save
   `case_results.csv`. Run the fixed removal/four-state/order diagnostics with
   entire-network reruns; save `game_diagnostics.csv` and uncertainty details.
5. Save a manifest with every input/output hash, exact perturbations, timings,
   convergence and missing/failed cells. Write a report distinguishing population
   findings, frozen examples and explicitly exploratory observations. Compare all
   weights fairly; do not select/activate a final production methodology.

The design can be regenerated intentionally with
`uv run python scripts/design_issue_263_real_seasons.py --write` (use a separate
output directory when checking determinism). Default invocation only verifies
hashes and never updates the design to newer publications. Test the freeze gates,
case-selection independence, input-order determinism and score orientation in
the execution ticket. Run Ruff and the sandbox-compatible Python suite; CI checks
API tests. No browser validation is required because no site files change.

`publication_config.json` preserves the design-time configuration. The original
mutable configuration, normalized corpus and historical availability-audit files
are identified by `audit_only_input_sha256`; they are not execution dependencies.
New publications or updated current acquisition files therefore do not block the
frozen study or expand its population. All retained snapshot bundles, priors,
models, implementation sources and the copied configuration remain hash-gated.

The full 2025 comparison, prospective availability proof, likelihood/prior
misspecification, roster/coach transition effects and causal schedule bias remain
outside what these retained snapshots can establish. There is no unresolved
choice blocking the bounded 2026 study; completed-season evidence requires the
separate provenance prerequisite described above. These limits must accompany
any execution conclusion.
