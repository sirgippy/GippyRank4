# Issue 258: Multi-strength Quality of Results design

## Decision and scope

This is a research-only feasibility and design artifact. It selects four deterministic focal priors, estimates feasible FBS opponent and schedule win-probability ranges, fixes schedule pairs, and runs a generation-only record-acceptance pilot. It does not run posterior inference, compare any λ candidate rankings, select a margin weight, or change ranking behavior.

The four main profiles are selected by nearest 2026 Context 1.4 preseason expected rank to 5, 20, 65, and 120, with canonical team ID breaking ties. The complete prior PMFs are retained. Colorado is included as a reference to the issue 252 focal prior; it is not an additional stratum in the main follow-up budget.

## Frozen profiles and feasibility

| Profile | Selected team | ID | Target rank | Expected rank | Rank SD | 80% interval |
|---|---|---|---|---|---|---|
| elite | Oregon | 2483 | 5.000 | 6.456 | 7.912 | 1–15 |
| strong | Oklahoma | 201 | 20.000 | 20.165 | 18.330 | 4–45 |
| middle | Western Michigan | 2711 | 65.000 | 64.951 | 33.267 | 21–112 |
| weak | Sam Houston | 2534 | 120.000 | 120.052 | 19.579 | 93–136 |
| issue_252_reference | Colorado | 38 | 63.143 | 63.143 | 33.443 | 19–111 |

The game-level range is the minimum and maximum integrated focal win probability over every other FBS opponent and the fixed home, away, or neutral venue. Schedule-average limits use a one-to-one assignment of ten distinct FBS opponents to the ten venue slots. The 50% and 80% feasibility flags below refer to those achievable average limits. A feasible average does not guarantee a schedule with a balanced game-probability distribution; both the selected slates and their full ten-game distributions are therefore retained.

| Profile | Game p range | Home | Away | Neutral | Feasible schedule means | 50% | 80% |
|---|---|---|---|---|---|---|---|
| elite | 0.431–0.995 | 0.592–0.995 | 0.431–0.987 | 0.512–0.992 | 0.597–0.978 | False | True |
| strong | 0.251–0.988 | 0.385–0.988 | 0.251–0.973 | 0.315–0.982 | 0.398–0.951 | True | True |
| middle | 0.075–0.938 | 0.139–0.938 | 0.075–0.884 | 0.103–0.915 | 0.148–0.813 | True | True |
| weak | 0.012–0.712 | 0.027–0.712 | 0.012–0.571 | 0.018–0.644 | 0.026–0.451 | False | False |
| issue_252_reference | 0.079–0.941 | 0.146–0.941 | 0.079–0.888 | 0.109–0.918 | 0.155–0.819 | True | True |

## Frozen schedule pairs

Every schedule has ten distinct FBS opponents and the issue 252 venue sequence (four home, four away, two neutral). Win probabilities integrate the focal and opponent rank PMFs through Historical Likelihood V1. The schedule builder matches a common ±0.08 target-probability profile to the closest unique opponents by venue; target-profile centers are calibrated using only these frozen model inputs and the requested schedule mean. Full game-level probabilities and opponent prior PMFs are in `schedule_games.csv` and `opponent_prior_pmfs.csv.gz`.

| Profile | Arm | Requested mean | Achieved mean | SD | Game p range | Median |
|---|---|---|---|---|---|---|
| elite | lower_win_probability | 0.600 | 0.599 | 0.089 | 0.464–0.731 | 0.592 |
| elite | higher_win_probability | 0.800 | 0.801 | 0.051 | 0.724–0.881 | 0.802 |
| strong | lower_win_probability | 0.500 | 0.499 | 0.049 | 0.426–0.580 | 0.495 |
| strong | higher_win_probability | 0.800 | 0.800 | 0.052 | 0.720–0.878 | 0.803 |
| middle | lower_win_probability | 0.500 | 0.501 | 0.050 | 0.428–0.581 | 0.500 |
| middle | higher_win_probability | 0.800 | 0.800 | 0.043 | 0.731–0.884 | 0.803 |
| weak | lower_win_probability | 0.200 | 0.200 | 0.052 | 0.123–0.281 | 0.198 |
| weak | higher_win_probability | 0.400 | 0.388 | 0.041 | 0.322–0.462 | 0.387 |
| issue_252_reference | lower_win_probability | 0.500 | 0.500 | 0.052 | 0.421–0.583 | 0.501 |
| issue_252_reference | higher_win_probability | 0.800 | 0.801 | 0.046 | 0.730–0.888 | 0.803 |

### Game-level schedule pairs

### elite: Oregon (ID 2483)

| Game | Venue | Lower schedule opponent | p(win) | Higher schedule opponent | p(win) |
|---|---|---|---|---|---|
| 1 | home | Texas Tech | 0.670 | Michigan | 0.724 |
| 2 | away | Georgia | 0.464 | Iowa State | 0.738 |
| 3 | home | Indiana | 0.666 | Texas A&M | 0.754 |
| 4 | away | Ole Miss | 0.503 | Washington State | 0.777 |
| 5 | neutral | Ohio State | 0.512 | South Carolina | 0.794 |
| 6 | home | LSU | 0.731 | Penn State | 0.809 |
| 7 | away | Miami | 0.546 | Mississippi State | 0.826 |
| 8 | home | Notre Dame | 0.716 | TCU | 0.845 |
| 9 | away | Michigan | 0.585 | Kentucky | 0.862 |
| 10 | neutral | Texas | 0.600 | North Texas | 0.881 |

### strong: Oklahoma (ID 201)

| Game | Venue | Lower schedule opponent | p(win) | Higher schedule opponent | p(win) |
|---|---|---|---|---|---|
| 1 | home | Georgia | 0.426 | Arizona State | 0.720 |
| 2 | away | Texas A&M | 0.440 | Tulane | 0.734 |
| 3 | home | Ole Miss | 0.462 | California | 0.754 |
| 4 | away | USC | 0.471 | North Carolina | 0.774 |
| 5 | neutral | Utah | 0.488 | Western Michigan | 0.792 |
| 6 | home | Miami | 0.503 | North Texas | 0.814 |
| 7 | away | Northwestern | 0.521 | Central Michigan | 0.828 |
| 8 | home | Michigan | 0.540 | East Carolina | 0.846 |
| 9 | away | Arizona | 0.560 | UAB | 0.862 |
| 10 | neutral | Washington | 0.580 | Wyoming | 0.878 |

### middle: Western Michigan (ID 2711)

| Game | Venue | Lower schedule opponent | p(win) | Higher schedule opponent | p(win) |
|---|---|---|---|---|---|
| 1 | home | Washington State | 0.428 | UAB | 0.731 |
| 2 | away | Kentucky | 0.438 | Middle Tennessee | 0.753 |
| 3 | home | Georgia Tech | 0.459 | Nevada | 0.772 |
| 4 | away | UTSA | 0.472 | UL Monroe | 0.770 |
| 5 | neutral | Purdue | 0.491 | Ball State | 0.809 |
| 6 | home | North Dakota State | 0.508 | New Mexico State | 0.831 |
| 7 | away | Oregon State | 0.528 | Sam Houston | 0.797 |
| 8 | home | Arkansas | 0.544 | UTEP | 0.838 |
| 9 | away | Georgia Southern | 0.562 | Massachusetts | 0.884 |
| 10 | neutral | Stanford | 0.581 | Charlotte | 0.813 |

### weak: Sam Houston (ID 2534)

| Game | Venue | Lower schedule opponent | p(win) | Higher schedule opponent | p(win) |
|---|---|---|---|---|---|
| 1 | home | Washington State | 0.123 | Missouri State | 0.322 |
| 2 | away | East Carolina | 0.133 | Buffalo | 0.346 |
| 3 | home | Rutgers | 0.153 | UAB | 0.353 |
| 4 | away | Utah State | 0.172 | Middle Tennessee | 0.371 |
| 5 | neutral | Marshall | 0.189 | Kent State | 0.389 |
| 6 | home | Western Kentucky | 0.208 | Sacramento State | 0.412 |
| 7 | away | South Alabama | 0.226 | Charlotte | 0.393 |
| 8 | home | UNLV | 0.246 | San José State | 0.447 |
| 9 | away | Eastern Michigan | 0.266 | Ball State | 0.384 |
| 10 | neutral | Air Force | 0.281 | UL Monroe | 0.462 |

### issue_252_reference: Colorado (ID 38)

| Game | Venue | Lower schedule opponent | p(win) | Higher schedule opponent | p(win) |
|---|---|---|---|---|---|
| 1 | home | SMU | 0.421 | Florida International | 0.730 |
| 2 | away | Ohio | 0.438 | New Mexico State | 0.747 |
| 3 | home | Vanderbilt | 0.456 | Rice | 0.769 |
| 4 | away | East Carolina | 0.466 | UL Monroe | 0.777 |
| 5 | neutral | Arkansas | 0.493 | UTEP | 0.803 |
| 6 | home | Mississippi State | 0.509 | Kent State | 0.815 |
| 7 | away | Hawai'i | 0.525 | Sam Houston | 0.803 |
| 8 | home | West Virginia | 0.544 | Ball State | 0.855 |
| 9 | away | New Mexico | 0.563 | Massachusetts | 0.888 |
| 10 | neutral | Texas State | 0.583 | Charlotte | 0.820 |

The original 50% versus 80% average contrast is achievable for the strong profile, and the mean limits also bracket both targets for the middle profile. The balanced target-profile builder reaches approximately 50% and 80% for the strong profile. For the middle profile it reaches the 80% mean only with a target-profile center near the upper end of the FBS opponent range, so its final achieved mean is used and documented rather than substituting expected ranks or forcing a high-variance schedule. Elite teams cannot average 50% across ten FBS games; their proposed comparison is approximately 60% versus 80%. Weak teams cannot reach either 50% or 80%; their proposed comparison is approximately 20% versus 40%. The narrower alternatives preserve a meaningful separation while keeping each slate's game probabilities clustered, rather than assembling near-certain wins and near-certain losses to hit an average.

## Record-conditioned panels and generation-only pilot

Record targets follow each pair's achieved schedule means: elite 6–4 versus 8–2; strong 5–5 versus 8–2; middle 5–5 versus 8–2; weak 2–8 versus 4–6. The pilot draws one focal rank per seed, samples opponent ranks from their full independent priors, and samples win/loss using the exact conditional V1 Student-t probability at the sampled rank pair and venue. The follow-up draws continuous synthetic margins from that same Student-t distribution; margin signs determine wins and losses. Shared focal ranks, opponent rank quantiles, and margin quantiles match the two schedule arms. Two copies of the lower-win-probability schedule use independent margin draws for the same-schedule control. No margins are altered to obtain a record.

| Profile | Accepted / pilot | Rate | Wilson 95% | Mean wins L/H | Planned search |
|---|---|---|---|---|---|
| elite | 774 | 0.077 | 7.232%–8.280% | 6.00/8.02 | 3083 |
| strong | 514 | 0.051 | 4.724%–5.590% | 4.98/7.98 | 4726 |
| middle | 431 | 0.043 | 3.929%–4.726% | 5.02/8.00 | 5684 |
| weak | 544 | 0.054 | 5.012%–5.902% | 1.98/3.85 | 4453 |
| issue_252_reference | 433 | 0.043 | 3.948%–4.747% | 5.01/7.99 | 5657 |

The separate same-schedule record-conditioned control panel accepts seeds only when both duplicated control teams attain the lower schedule's target record. It is not selected from the primary schedule-pair seeds.

| Profile | Accepted / pilot | Rate | Wilson 95% | Mean wins A/B | Planned search |
|---|---|---|---|---|---|
| elite | 464 | 0.046 | 4.245%–5.070% | 6.02/6.02 | 5261 |
| strong | 292 | 0.029 | 2.608%–3.269% | 4.97/4.97 | 8571 |
| middle | 296 | 0.030 | 2.645%–3.311% | 5.02/5.02 | 8448 |
| weak | 533 | 0.053 | 4.907%–5.788% | 1.99/2.00 | 4549 |
| issue_252_reference | 276 | 0.028 | 2.457%–3.100% | 5.01/5.02 | 9099 |

The next ticket should use 500 unconditional paired seasons, 200 accepted primary record-conditioned pairs, and 200 accepted lower-schedule same-schedule control pairs per strength stratum. Each candidate λ receives identical game evidence within a trial. The primary conditional and control conditional panels use separate, disjoint seed ranges. The unconditional control is observed at every unconditional seed; controls copied onto primary-conditioned seeds are labeled as primary-seed controls and are not used as own-record-conditioned controls. The generation pilot also checks that simulated mean win counts track the schedule probabilities before posterior inference.

## Follow-up analysis protocol

- Reuse issue 252's disconnected inference forest, issue 242 Quality of Results implementation, complete 138-rank support, default deterministic BP settings, focal prior stripping, and five frozen candidates (λ = 0, 0.25, 0.50, 0.75, 1). Keep each opponent prior uncertain in inference; do not plug in expected ranks or treat opponents as known.
- Generate identical trial evidence for all five candidates. The two primary teams in a pair share the same underlying focal PMF, focal latent-rank draw, starting-prior treatment, schedule game count, and venue distribution. Opponents are independently sampled from their own Context 1.4 PMFs; no opponent repeats within one slate.
- Report results separately by strength stratum. The primary schedule effect is the paired difference in expected rank (higher-win-probability arm minus lower-win-probability arm), with a 95% normal Monte Carlo interval for the mean, the difference distribution, outrank probability with a 95% Wilson interval, rank SD, entropy, central 80% width, PMF total variation, and Wasserstein distance. Compare schedule effects across strata; do not compare absolute expected ranks across strata as if those were schedule effects.
- Use same-schedule controls to estimate outcome-driven variation at the lower schedule distribution. Report them beside, not subtracted from, the primary schedule contrast. Keep unconditional controls unconditional. Build a separate control-conditioned panel accepted on the two controls' own records; never label controls selected only by a primary pair's seed as record-conditioned controls.
- Separate schedule-driven differences (paired schedule contrast), realized-outcome variation (within-panel difference distributions and same-schedule controls), posterior uncertainty (paired rank SD, entropy, interval width), and between-stratum context (within-stratum effects and their contrasts). Do not select or revise schedules after seeing candidate posterior ranks.

## Sample and compute budget

The unconditional and primary conditioned panels require 700 four-component inference trials per stratum; the separately conditioned control panel adds 200 two-component trials per stratum. Scaling issue 252's measured 6.089 seconds for three five-weight, four-component trials gives an estimated **108 minutes** total for the four strata, including the separate control panel. This is a planning estimate, not a new timing run. Generation and seed search are expected to be small relative to posterior inference. The pilot estimates a per-panel seed search that targets a 95% chance of finding 200 accepted draws using the lower Wilson acceptance bound, capped at 20,000 attempts. If a panel exceeds that cap, pause before inference and report the conditioning limitation; do not change records in response to candidate results.

## Reproducibility and frozen decisions

The manifest pins the source prior CSV, Historical Likelihood V1 bytes, issue 252 schedule protocol, this design code, selected focal PMFs, opponent PMFs, schedule definitions, pilot seed ranges, NumPy/SciPy versions, and SHA-256 hashes for every output. Gzip output uses a fixed timestamp. All profile selection, schedule construction, pilot criteria, outcome targets, and follow-up budgets are fixed without issue 258 candidate posterior inference. If a comparison cannot be run within the stated FBS population and budgets, the follow-up should report that limitation rather than introduce FCS opponents, duplicate opponents, schedule bonuses, likelihood changes, or invented profiles.

Base repository revision: `fdf6fc095a864b65907fc3c2056bde559f018546`.
