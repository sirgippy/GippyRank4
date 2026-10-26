# Issue 252: Quality of Results schedule-equivalence protocol

## Purpose and freeze

This research-only experiment evaluates the five frozen issue 242 candidates
(`λ = 0, 0.25, 0.50, 0.75, 1`) for schedule equivalence and outcome sensitivity.
It does not change or publish a ranking. The evaluation compares season
accomplishments; predictive accuracy is not a selection criterion.

The protocol, candidate schedules, seeds, acceptance rules, and summaries are
fixed before the candidate rank results are computed. No schedule, seed,
counterfactual, or interpretation threshold may be changed in response to a
candidate result.

## Frozen inputs and synthetic population

Use the 2026 Context 1.4 preseason PMFs for the complete FBS population and the
byte-pinned Historical Likelihood V1 artifact. The population size, rank
support, prior artifact hash, selected PMF hashes, likelihood hash, and source
versions are written to the run manifest. The median expected-rank FBS prior
is the common prior for all focal teams; its PMF is retained, and each focal
team is assigned that same PMF. This is one representative focal-strength
stratum. Opponent beliefs are the corresponding independent preseason FBS
PMFs, including their uncertainty; no opponent rank is treated as known during
inference.

For each candidate opponent, compute the focal win probability by integrating
the Student-t CDF over both discrete rank PMFs, using the frozen V1 coefficient
surface and the fixed venue mix below. Opponents are chosen independently of
synthetic outcomes. Candidate schedules are selected without replacement by
matching each target probability, in ascending order, to the nearest unused
opponent probability; ties use canonical team ID.

| Schedule role | Target probabilities by game |
| --- | --- |
| Strong-opponent schedule (sensitivity only) | 0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.31, 0.34 |
| Approximately even schedule | 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68 |
| Weaker-opponent schedule | 0.66, 0.69, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88, 0.91, 0.94 |

Every schedule uses the same ten ordered venue slots: home, away, home, away,
neutral, home, away, home, away, neutral (4 home, 4 away, 2 neutral). The
computed schedule-average probabilities and the opponent-specific probabilities
are reported. If the chosen slate misses its target average by more than 0.03,
the run fails before inference; no replacement search is allowed.

## Outcome generation and inference

Each season draws one latent focal rank from the common focal PMF. Each opponent
rank is drawn once from that team's own PMF. For each game, draw a continuous
margin from the exact V1 Student-t surface at those ranks and the assigned
site. Synthetic margins are internal model observations, not literal football
scores. Game signs define wins and losses. Shared rank quantiles and Student-t
quantiles across the approximately even and weaker schedules provide matched
random numbers. The same-schedule control uses two copies of the same
approximately even opponent-PMF slate, shared latent ranks and independent
margin draws.

Each trial is inferred with all four focal teams and their opponent nodes in a
disconnected forest, the complete 2026 FBS rank support, identical focal prior
treatment, the issue 242 `infer_quality_of_results` implementation, and its
default deterministic BP settings. All five λ values receive identical game
inputs. The focal output is the frozen Performance prior-stripped PMF; opponent
PMFs remain uncertain in the inference network. A run fails if any candidate
does not converge or the focal support differs.

## Simulation panels

### Bounded pilot

Before candidate results, a disjoint pilot used seeds 2510000–2510199 to check
record variability and the exact-record acceptance rate. All 200 seeds were
used for generation; the first three also timed all five inference weights,
with their candidate rank outputs discarded and not inspected. Nine draws
(4.5%) matched 5–5 and 8–2; the mean records were 5.17–4.83
and 7.74–2.26, with win-count standard deviations 2.47 and 1.96. A discarded
three-trial run through all five λ values took 6.1 seconds, about 2.0 seconds
per four-component inference trial. No candidate rank outputs from this timing
run were inspected. Based on the pilot, the frozen main budget is 500
unconditional pairs and 200 accepted record-conditioned pairs; the estimated
main inference budget is about 24 minutes, plus generation and sensitivity
counterfactuals. The pilot seeds are excluded from the main seed ranges.

The unconditional panel contains 500 season pairs, seeds 2520000–2520499, and
retains every generated record. It reports the approximately-even versus
weaker-schedule pair and the same-schedule control pair.

The record-conditioned panel accepts the first 200 seeds from 2530000 onward
that produce exactly 5–5 on the approximately-even schedule and 8–2 on the
weaker-opponent schedule. The deterministic search ceiling is 20,000 seeds.
All rejected seeds are generation-only draws and are recorded by total count;
only accepted seasons require inference. This estimates the comparison
conditional on the motivating records without mixing it with the unconditional
panel. No score is edited to obtain a target record.

The representative comparison is selected from the accepted conditional
panel: choose the season pair whose combined median squared standardized
margin residual is closest to the conditional-panel median. This rule uses
only the generating model and outcomes, not inferred ranks or λ results.

## Schedule-equivalence measures

For each trial and λ, define rank difference as
`expected_rank(weaker-schedule team) − expected_rank(even-schedule team)`;
negative values mean the weaker-schedule team receives the better expected
rank. Report its mean, median, 5th/25th/75th/95th percentiles, and a 95% Monte
Carlo interval for the mean. Also report the probability that the weaker-
schedule team outranks the even-schedule team by expected rank, with a 95%
Wilson interval.

Report paired differences in rank SD, entropy, and central 80% interval width,
plus PMF total variation and 1-Wasserstein distance. The same measures are
reported separately for the same-schedule control, unconditional trials, and
record-conditioned trials. A result is not called equivalent from a small
mean difference alone; the interval and full difference distribution are
considered together. No equivalence threshold is imposed after seeing results.

## Outcome-sensitivity panel

Use one fixed generated season for each of the three controlled schedule roles
(strong, approximately even, weaker). Select the season by the same
model-residual median rule from seeds 2540000–2540999. For each schedule, vary
one selected game while preserving all opponents, venue, other game results,
priors, and inference settings:

- Win/loss boundary: −3 versus +3 points against the selected opponent.
- Within-win margin: +10 versus +35 points, including the weaker-opponent
  case.
- Within-loss margin: −10 versus −35 points.
- Large weak-opponent win: +14 versus +35 points.

The boundary comparison is repeated for the approximately-even,
weaker-opponent, and strong-opponent games. The same unchanged game is
counterfactually replaced within each comparison. For all five λ values report
changes in focal expected rank, posterior PMF total variation, rank SD, and
entropy. Scores are explicit synthetic margins and do not represent real
games. These contrasts describe model response; they do not set a universal
win-to-points exchange rate.

## Monte Carlo reporting and artifacts

Report the sample count and Monte Carlo interval for every aggregate mean and
the Wilson interval for every outrank probability. Publish all per-trial
expected-rank and uncertainty summaries, all accepted/unconditional seeds,
synthetic game margins, fixed schedule definitions, selected team priors, the
byte-identical likelihood artifact, and a manifest with source hashes and
inference configuration. Deterministic gzip uses `mtime=0`. A rerun from the
same source artifacts and seed ranges must reproduce substantive outputs.

The conclusion will separate observed candidate behavior from the normative
decision about how Quality of Results should value wins and margins. The
recommendation can retain a candidate for further study or conclude that the
present evidence supports no candidate; it cannot trigger a model correction
in this issue.
