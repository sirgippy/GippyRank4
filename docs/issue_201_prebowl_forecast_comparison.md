# Issue 201: pre-bowl History and Context forecast comparison

## Frozen protocol

This protocol records the comparison before the issue 201 forecast scores are
computed. The result is exploratory development, not untouched confirmation:
2022–2025 have informed earlier project work. No result promotes a model or
changes production defaults.

The four fixed arms cross the prior family with the matched issue 200 policy:

| Arm | Prior target | Likelihood policy |
| --- | --- | --- |
| History existing | Corrected final ranks | Existing regular-season and postseason games |
| Context existing | Corrected final ranks | Existing regular-season and postseason games |
| History pre-bowl | Pre-bowl FBS ranks | Regular-season FBS evidence; unchanged FCS postseason policy |
| Context pre-bowl | Pre-bowl FBS ranks | Regular-season FBS evidence; unchanged FCS postseason policy |

The expected ordering, recorded before scoring, is Context pre-bowl, History
existing, History pre-bowl, Context existing, from lower to higher mean game
margin CRPS. It is a hypothesis, not a success rule. The intervention changes
the FBS response target and FBS postseason likelihood policy together; it does
not isolate either contribution. Existing prior-season feature inputs,
architectures, optimizer recipes, cold starts, and common fitting seasons
remain those in the matched issue 200 bundle (2003–2021). The FCS target and
postseason policy are unchanged.

## Pregame game replay

The weekly schedule is every Tuesday at **00:00:00 UTC**, beginning with the
Tuesday strictly before the season's first scheduled regular-season kickoff
and ending with the last Tuesday before its last scheduled regular-season
game. Each scheduled game is assigned to the latest such cutoff strictly
before its kickoff. This rule uses only game IDs, kickoff timestamps, season
type, team IDs, subdivisions, and site metadata. The runner writes and hashes
the complete schedule assignment plan before it reads completion or score
fields; scoring refuses a different schedule plan. Each eligible game appears
at one cutoff only.

Primary forecasts cover completed regular-season FBS–FBS games. They include
conference championships and Army–Navy when the source marks them regular
season. FBS bowls and CFP games are excluded from forecast scoring in every
arm. Completed regular-season FBS–FCS games are scored and reported separately.
Army–Navy is identified separately because it may follow the pre-bowl target
snapshot. The fixed final cached schedule supplies kickoffs; historical
as-of-schedule snapshots are not available, so kickoff certainty records the
source timestamp precision and explicitly marks schedule availability as
retrospective.

At each cutoff, evidence consists of completed games with valid FBS/FCS teams
and a kickoff at least **48 hours before** the cutoff. The kickoff age is the
conservative availability proxy used by the retained benchmark because exact
completion timestamps are unavailable. Existing-policy replay retains its
eligible postseason evidence. Pre-bowl replay removes postseason games that
involve an FBS team, while preserving FCS-only postseason evidence. Every
scored game must start strictly after its assigned cutoff and must be absent
from that arm's evidence IDs. The runner persists included evidence IDs and
canonical row hashes, excluded near-cutoff IDs, kickoff certainty, and the
schedule assignment for every scored game.

For each cutoff, issue 200's replay API combines the selected bundle prior,
policy likelihood, and the allowed evidence using the bundle's fixed belief
propagation settings. FCS teams use the bundle's full-season uniform fallback
population. The same posterior-product approximation is used for all arms.
For each forecast, the existing predictive API integrates the full pair of
posterior rank PMFs into its Student-t margin mixture; no expected-rank plug-in
or display histogram is scored.

### Metrics and aggregation

The primary score is continuous-margin CRPS, lower is better. For mixture
components with locations `mu_i`, weights `w_i`, shared scale `s`, degrees of
freedom `nu`, observed margin `y`, and `T ~ Student-t(nu)`, the implementation
uses

```text
CRPS = sum_i w_i E|mu_i + s T - y| - integral F(z) (1 - F(z)) dz.
```

The first term is evaluated from the Student-t CDF and first absolute-tail
moment. The second is deterministic Gauss–Legendre integration over
`[min(mu) - As, max(mu) + As]`, starting at `A=40`. The analytic Student–t
tail-integral bound must use no more than half the `2e-6` margin-point error
budget; if the bound is too large, `A` doubles until it fits. The runner checks
successive quadrature orders 128, 256, and, if needed, 512 on that fixed
interval. The reported numerical error diagnostic is the last successive
quadrature difference plus the analytic tail-integral bound beyond the
interval. A forecast that does not meet the full tolerance fails the run. The
chosen `A`, order, error diagnostic, and tolerance are recorded per forecast.

Supporting game measures are stable marginalized continuous-margin NLL,
central 50/80/95% interval coverage and width, absolute error of the predictive
median, and home-win Brier score. The win outcome gives ties half credit. The
report includes per-game rows, per-season means, an equal-game pooled mean,
an equal-season sensitivity summary, paired arm differences, and
pre-Thanksgiving, rivalry-window, conference-championship, and Army–Navy
results under the fixed date mapping in the runner. Four seasons and repeated
team matchups are not independent confirmation; uncertainty is described by
season-level stability and paired differences, not an independent-game
significance claim.

Preseason-only predictions use the same priors and scheduled game population
with no current-season evidence. They are a separate diagnostic and are never
added to rolling headline game counts.

## Secondary ranking evaluation

The four rolling arms are scored on common FBS team supports against the
corrected full pre-bowl target PMFs. Ranking NLL is secondary; discrete rank
CRPS, expected-rank error, and central-80% coverage/width are reported beside
it. Only cutoffs strictly before the target observation date enter the paired
ranking comparison. Each included posterior's evidence is audited to confirm
that every included game kickoff precedes the target date. The weekly date
mapping includes the event-aligned pre-Thanksgiving cutoff; common cutoffs and
team supports are listed explicitly. Cutoffs and games after the target date
are excluded rather than moved or relabeled.

FCS results and targets keep the #204-corrected final-ranking contract and
existing postseason policy. The report separately records Army–Navy games
that occur after a pre-bowl FBS target, as accepted by the experiment contract.

## Frozen-model target sensitivity

The original #192 History 1.1 and repaired Context 1.4 panel, its four stored
PMF arms, original seven checkpoint dates, inference evidence, and final
scoring targets are read-only. The runner verifies the panel/source hashes and
computes a frozen-PMF identity digest before and after scoring. It re-scores
the original final targets and corrected pre-bowl targets on one common
team-season/checkpoint cohort and reproduces original final-target History and
Context posterior scores against the retained `team_scores.csv` values before
interpreting target differences. Native cold starts are kept. Missing IDs or
rank supports are excluded with an explicit reason; no PMF is padded, clipped,
or renormalized onto a different support.

For each retained checkpoint, evidence IDs and kickoff timestamps are audited
against the pre-bowl target date. Any checkpoint whose evidence includes a
game after that date is excluded from the paired frozen forecast comparison;
the cutoff is never shifted. The report gives per-season/checkpoint NLL and
paired Context-minus-History differences under each target, plus rank CRPS and
calibration summaries. It keeps 2022 visible and maps the original checkpoints
to pre-Thanksgiving, rivalry, and championship phases, including missing phase
coverage. This frozen sensitivity measures evaluation-target sensitivity; it
does not show that postseason exclusion improves forecasts or explain a
mechanism.

## Decision rules

Report the four-arm ordering and the interaction

```text
(C_pre_bowl - C_existing) - (H_pre_bowl - H_existing)
```

in game CRPS; negative values mean the policy improved Context more. Also
report Context-minus-History within each policy. State any disagreement among
CRPS, margin NLL/calibration, and secondary ranking NLL. Conclude improve, no
useful improvement, or inconclusive, then recommend proceed, defer, or revise
for issues #193–195. No extra arms, feature searches, residual probes, or
automatic follow-on work are included.

## Fixed event windows and result classification

Game phases use the kickoff's `America/New_York` calendar date. Army–Navy is a
separate overriding label. The rivalry window runs from Thanksgiving Thursday
through Sunday. The conference-championship window runs from one day before
through the last FBS conference championship date in the #198 schedule
manifest. Other regular-season games before Thanksgiving are pre-Thanksgiving;
remaining dates are reported as other regular season.

Checkpoint phases use the cutoff's Eastern calendar date: pre-Thanksgiving is
21 through 8 days before Thanksgiving; the rivalry window is 7 days before
through 3 days after Thanksgiving; the conference-championship period is 7
days before the last FBS conference championship date through the first FBS
postseason date. Original checkpoints outside those windows are labeled other;
phase coverage is reported even when a target-safety rule excludes a checkpoint.

Classify the primary result as **improve** only if Context pre-bowl has lower
CRPS than Context existing under both the equal-game and equal-season
summaries, wins in at least three of four seasons, and the policy interaction
is negative. Classify it as **no useful improvement** when Context pre-bowl
fails to lower CRPS in both aggregate summaries and wins in at most one season.
Otherwise classify it as **inconclusive**. If margin NLL or calibration has a
consistent opposite direction in at least three seasons, use inconclusive even
when the CRPS rule would say improve. These are descriptive rules for the four
previously studied seasons, not significance tests.


Calibration opposition for the result rule is the per-season mean absolute deviation
of 50/80/95% empirical interval coverage from its nominal level. If the
Context pre-bowl arm has worse margin NLL or worse average coverage deviation
than Context existing in at least three seasons, an otherwise improving CRPS
classification is downgraded to inconclusive. The recommendation mapping is
fixed: improve → proceed to separately scoped follow-up issues; no useful
improvement → defer them pending a new rationale; inconclusive → revise the
comparison plan before follow-up work. This mapping does not dispatch or
create follow-on work.
