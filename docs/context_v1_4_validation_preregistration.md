# Context 1.4 holdout validation preregistration (issue 164)

**Registered before Context 1.4's 2026 outcome scores are opened.** The executable
choices are in [`config/context_v1_4_validation.json`](../config/context_v1_4_validation.json).
The frozen candidate is PR #161's `construct_candidate_prior`, semantic SHA-256
`db84045d2f7d80b1648360da52ef1e71f76093696cad4e4e7bbdbd57a086573f`.
The configuration pins its audit mirror and builder bytes, the retained 2026
Context 1.3 fit/predictions, and the retained History 1.1 predictions. The
[registration provenance](../data/processed/context_v1_4_validation/preregistration.json)
pins the configuration SHA-256 for later Git ancestry checks. Validation
must load this configuration and call the frozen builder without an alpha option;
it must not refit, repair, or change any of the three models.

## Holdout status and locked population

Issue #154 diagnosed 2022–2025 Context location errors; issue #158 selected from
the same development seasons; issue #160 and PR #161 froze and checked the
candidate against development and outcome-free fixtures. PR #161 also constructed
a **2026 cold-start prior for a provenance test**, without a 2026 target or score.
The separate #140 Context/History study and weekly operations viewed 2026
**baseline** game/posterior snapshots through September 27. No final 2026 Massey
rank-distribution target was available in that study. These already viewed
2026 games and baseline displays are not pristine new season evidence. The
as-yet-unavailable final 2026 target and games after this preregistration are
untouched validation evidence; no evidence in the cited candidate-selection
artifacts shows 2026 outcomes choosing the candidate. The report must retain
this exposure distinction and may call the test *candidate-score-unseen 2026
validation*, not an entirely unobserved season.

The population is the exact **138 FBS team IDs** in the committed 2026 Context
1.3 and History 1.1 preseason predictions, recorded individually and hashed in
the configuration. Later conference or subdivision labels cannot change it.
Sacramento State (`16`) and North Dakota State (`2449`) remain in the population
with their frozen History 1.1 transition fallbacks; fitted Context rows are
never substituted for them. Legitimately absent transfer or feature values use
Context 1.3's recorded missingness semantics and the same retrospective 2026
transfer reconstruction in both Context arms. No new transfer repair, imputation,
or historical refit is permitted. The Context 1.3 fit has a retained legacy
attestation, not a verified training-corpus digest; that limit must be reported.

For scoring, remove a team **at all checkpoints and from all three arms** only
if its final canonical target PMF is missing or invalid. A valid row must
uniquely identify that 2026 FBS team, contain unique supported ranks within
1–138 (unlisted ranks have zero target mass), finite nonnegative probabilities, and sum to one within `1e-8`.
A missing global target source aborts. Log every excluded team and checkpoint. All other teams have equal weight: one team, one paired NLL
observation per checkpoint. The same target-complete set drives the headline and
History comparisons, so each checkpoint has a three-model common population.
At least **132 of 138** targets are required for a promotion or retention
decision; lower coverage yields *inconclusive*. A missing/mismatched baseline
prior, unmatched model team, unverifiable Context source row, or unconstructible
candidate invalidates the run instead of shrinking the population.

## Seven fixed evidence boundaries

The seven UTC cutoffs are the established 2025 rolling-backtest cutoffs shifted
by exactly 364 days, preserving their weekday positions:

| Checkpoint | UTC cutoff |
| --- | --- |
| Preseason | 2026-08-22 23:59:59 |
| Early season | 2026-09-18 23:59:59 |
| Early midseason | 2026-10-07 23:59:59 |
| Midseason | 2026-10-23 23:59:59 |
| Late season 1 | 2026-11-08 23:59:59 |
| Late season 2 | 2026-11-22 23:59:59 |
| **Final primary endpoint** | **2026-12-12 23:59:59** |

Preseason has zero game evidence. Each later boundary uses completed scores
with a valid actual kickoff at least **48 hours before the cutoff**, the #140
conservative availability rule. Use the canonical finalized kickoff for a
rescheduled game. Exclude and audit unscored/unknown-kickoff games. A known
suspended or delayed completion with no time proving cutoff safety is excluded
at every checkpoint and audited. If canonical schedule coverage itself is
incomplete, stop the run. Each arm receives the same eligible game rows, frozen
Historical Likelihood V1, FCS fallback population, and posterior settings
(`500` iterations, `1e-9` tolerance, `0.35` damping). The FCS fallback is
uniform over ranks 1–128: those 128 FCS IDs are locked from the hashed 2026
CFBD full-season schedules present at registration. At each checkpoint, include
FCS teams appearing in eligible or then-scheduled games using this same
128-team denominator; an ID outside the locked roster aborts. The eventual
Massey target corpus cannot supply the FCS fallback population. A team without a newly
completed game stays in the shared inference network; its posterior may still
move through opponents. Do not pick alternate dates after seeing scores.

The target is the final **2026 Massey constituent-rank PMF** produced by the
existing historical-modeling target semantics, on ranks 1–138. It is reused at
every checkpoint as in #140. Target files may be loaded for scoring only after
all three prior/posterior arrays and cutoff evidence sets have been constructed
and hashed. The target never enters features, priors, game selection, or
posterior inference.

## Scores, uncertainty, and decision

For each team/model/checkpoint, NLL is
`-sum_r target[r] * ln(max(posterior[r], 1e-15))`. The primary statistic at the
**final** cutoff is the equal-team-weight mean of paired
`Context 1.4 NLL − Context 1.3 NLL`: **negative favors the candidate**.
Report N, each model's mean NLL, paired mean and median, and a two-sided 95%
percentile interval for the paired mean. Resample team IDs with replacement
10,000 times, sample size N, using NumPy `Generator(PCG64(1642026))`; sort IDs
lexically before sampling and use linear interpolation at the 2.5th and 97.5th
percentiles. No team/checkpoint is an independent resampling unit beyond its
one final paired team difference. This interval describes variation across
teams in one season; team outcomes share games, so it is not a multi-season
generalization interval.

The practical threshold is **0.010 NLL nats per team**. This was set from
2022–2025 development evidence: the selected candidate gained 0.0308 pooled
final NLL, the weakest development season gained 0.0138, and Context 1.3's
2023–2025 final gap to History was 0.0728. Thus 0.010 is roughly one third of
the pooled gain and one seventh of that baseline gap, without using a 2026
candidate score. The registered classifications are:

- **Promote:** paired mean ≤ −0.010 and bootstrap upper bound < 0.
- **Retain Context 1.3:** paired mean ≥ +0.010 and bootstrap lower bound > 0.
- **Inconclusive:** every other valid result, including N < 132.

An identity, guardrail, target-isolation, or evidence failure **aborts without a
registered decision**. Secondary scores cannot rescue a missed primary rule.
This is a three-way decision on magnitude and uncertainty, not a generic
significance label.

Secondary output uses the same paired population: mean posterior NLL for all
three models and all three pairwise differences at all seven checkpoints;
Context 1.3/History and Context 1.4/History signed gaps at every checkpoint;
and preseason/final expected-rank mean absolute error, median absolute error,
and mean signed error (`forecast − target`). Report team-level final candidate
minus Context 1.3 NLL differences: improved (<−`1e-10`), worsened (>`1e-10`),
otherwise unchanged; mean, min/max, 10th/25th/50th/75th/90th percentiles using
linear interpolation; and ten largest gains and harms, breaking ties by team
ID. These are explanations, never decision selectors.

Only four descriptive strata are allowed: sign of the Context-only net subtotal
(negative/zero/positive, with cold starts separate); positive-subtotal size
using **0.529952779961** and **0.795976339044** (the positive-value median
and 75th percentile from the hashed 2022–2025 fitted contribution artifact);
fitted versus cold-start; and existing transfer coverage states
`complete`/`incomplete`/`history_fallback`. A fitted row is `complete` only when
both offensive-usage and defensive-impact transfer inputs are present; a
cold-start is `history_fallback`. Subgroup counts and scores are descriptive.
No 2026 cutpoint, subgroup selection, or alpha search is allowed.

## Failure gates and later handoff

Before scoring, verify all configured hashes, the candidate semantic identity,
Context 1.3 and History 1.1 model provenance, exact roster, source prior parity,
and identical game evidence. For `context_only_subtotal <= 0` and cold starts,
the candidate prior PMF must equal Context 1.3 bit for bit. For positive fitted
rows, verify the exact 0.75 net moderation, unchanged centered mixture offsets
(with `1e-12` numeric tolerance) and residual scale, and unchanged uncertainty
and transfer semantics. Reject an altered candidate. A missing target excludes
only that team's seven paired rows; record team, checkpoint, reason, and all
three affected models. Missing History input, missing/invalid Context input,
missing canonical game coverage, target-source ambiguity, or candidate
provenance mismatch aborts with a machine-readable failure report. Never fill
in a model or target score to reach 132.

The follow-up implementation must create the configured script, read the
registered JSON through
`gippyrank.context_v1_4_validation_protocol.load_registered_protocol`,
and produce the team/checkpoint CSV, summary JSON, exclusions CSV, evidence
audit, provenance JSON, and results document. An aborted run writes the
configured `failure.json` with the failed gate and input hashes, without a
performance decision. Deterministic
machine files must sort team IDs/checkpoints, use stable numeric formatting,
and be byte-identical on repeat runs. The provenance JSON must include issue
164, the Git commit containing this final preregistration, this config's
SHA-256, PR #161 candidate semantic hash and builder hash, both baseline
identities, script/code and dependency hashes, all baseline/transfer/game/
likelihood/target input hashes, and an evidence hash per checkpoint. The final
report opens with the registered decision and primary table, then the secondary
checkpoint/History/rank/team distribution results, descriptive strata, exact
population/exclusions, provenance, and a conclusion applying this rule.

Required follow-up tests use historical or synthetic fixtures until the
protocol is merged: candidate hash and no tunable alpha; exact population and
exclusion accounting; paired row identity and NLL orientation; 48-hour cutoff
and postponed-game safety; target isolation; bootstrap determinism; negative
and cold-start identity; centered-mixture preservation; model immutability;
and byte-identical artifacts. Validation must stop if any guardrail fails.
Any later change to population, cutoffs, primary score, weights, uncertainty,
0.010 threshold, or decision rule after scores are opened is a documented
protocol deviation and a new research cycle, never an unrecorded edit to this
registration. This preregistration creates no Context 1.4 2026 performance
artifact and does not promote the candidate.
