# Issue 253: in-season posterior movement and accumulated surprise

## Finding

The retained panel contains 534 FBS team-seasons, 7,476 family/checkpoint observations, and 84 native cold-start team-checkpoint rows. Each row retains full prior and posterior rank PMFs.
At least three evidence-available prior games contribute to 6,088 surprise comparisons. The full movement summaries keep all team-checkpoints, including those below that threshold.

Movement uses positive values for expected-rank improvement. Rank-location, standard deviation, and interval-width changes are divided by the season’s rank range (`N - 1`) for cross-season comparison. KL is posterior-to-prior in nats; adjacent-checkpoint total variation ranges from 0 to 1.

## Typical movement across checkpoints

Medians and 10th–90th percentiles pool the four exploratory seasons for compact display; checkpoint-specific, season-specific, and cold-start distributions are in `aggregate_by_season_checkpoint.csv`.

| Family | Checkpoint | Median normalized expected-rank improvement | P10–P90 | Median KL | Median adjacent TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| History | 1 | 0.000 | 0.000 to 0.000 | 0.000 | — |
| Context | 1 | 0.000 | 0.000 to 0.000 | 0.000 | — |
| History | 4 | -0.005 | -0.203 to 0.196 | 0.418 | 0.141 |
| Context | 4 | -0.009 | -0.190 to 0.179 | 0.380 | 0.134 |
| History | 7 | -0.010 | -0.228 to 0.209 | 0.640 | 0.095 |
| Context | 7 | -0.011 | -0.199 to 0.200 | 0.588 | 0.096 |

| Family | Checkpoint | Prior SD / range | Posterior SD / range | SD change / range | Prior 80% width / range | Posterior 80% width / range | Width change / range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| History | 1 | 0.234 | 0.234 | 0.000 | 0.631 | 0.631 | 0.000 |
| Context | 1 | 0.220 | 0.220 | 0.000 | 0.584 | 0.584 | 0.000 |
| History | 4 | 0.234 | 0.122 | -0.109 | 0.631 | 0.313 | -0.304 |
| Context | 4 | 0.220 | 0.119 | -0.097 | 0.584 | 0.302 | -0.269 |
| History | 7 | 0.234 | 0.096 | -0.134 | 0.631 | 0.245 | -0.376 |
| Context | 7 | 0.220 | 0.093 | -0.124 | 0.584 | 0.235 | -0.338 |

A negative normalized posterior-SD change indicates narrower uncertainty than preseason; it is descriptive and does not by itself indicate overconfidence. Expected-rank movement can also result from information elsewhere in the joint inference graph, so it is not attributed solely to a team’s own games.

Checkpoint-7 median location movement changed sign across the four years: 2022–2023 medians moved toward worse expected ranks, while 2024–2025 medians improved. Median posterior SD was lower than preseason in all four years for both families. These are distribution summaries, not independent seasonal replications.

## Checkpoint-7 season stability

| Season | Family | Median rank improvement from preseason | Median normalized SD change | Median posterior normalized SD | Median KL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2022 | History | -1.479 ranks (-0.011) | -0.133 | 0.098 | 0.608 |
| 2022 | Context | -2.231 ranks (-0.017) | -0.126 | 0.097 | 0.576 |
| 2023 | History | -3.744 ranks (-0.028) | -0.135 | 0.098 | 0.634 |
| 2023 | Context | -2.259 ranks (-0.017) | -0.121 | 0.093 | 0.601 |
| 2024 | History | 2.780 ranks (0.021) | -0.132 | 0.095 | 0.652 |
| 2024 | Context | 3.108 ranks (0.023) | -0.121 | 0.092 | 0.582 |
| 2025 | History | 1.010 ranks (0.007) | -0.137 | 0.092 | 0.648 |
| 2025 | Context | 0.289 ranks (0.002) | -0.127 | 0.088 | 0.588 |

## Relationship with accumulated surprise

The #247 groups retain their fixed ±0.5 cutoffs and require at least three games available in the exact #192 evidence set. Pooled group differences and percentile intervals are descriptive. The regressions use continuous surprise, preseason strength and uncertainty, game count, and season/checkpoint fixed effects. The main continuous surprise slopes are evaluated at mean preseason strength. Positive location slopes mean positive prior game surprises are associated with rank improvement in the next checkpoint; negative SD-change slopes mean greater surprises are associated with narrowing rank uncertainty.

| Family | Outcome | Term | Estimate | 95% interval | Team-checkpoint rows |
| --- | --- | --- | ---: | ---: | ---: |
| History | directional_location_movement | accumulated_surprise | 0.1562 | 0.1059 to 0.2033 | 3,044 |
| History | directional_location_movement | accumulated_surprise_x_preseason_strength | -0.0549 | -0.1638 to 0.0411 | 3,044 |
| History | absolute_location_movement | absolute_accumulated_surprise | 0.0839 | 0.0235 to 0.1365 | 3,044 |
| History | rank_sd_change | accumulated_surprise | 0.0017 | -0.0029 to 0.0069 | 3,044 |
| History | rank_sd_change | accumulated_surprise_x_preseason_strength | -0.0895 | -0.1446 to -0.0314 | 3,044 |
| History | central80_width_change | accumulated_surprise | 0.0053 | -0.0083 to 0.0228 | 3,044 |
| History | central80_width_change | accumulated_surprise_x_preseason_strength | -0.2362 | -0.4051 to -0.0764 | 3,044 |
| Context | directional_location_movement | accumulated_surprise | 0.1463 | 0.1039 to 0.1874 | 3,044 |
| Context | directional_location_movement | accumulated_surprise_x_preseason_strength | -0.0552 | -0.1445 to 0.0273 | 3,044 |
| Context | absolute_location_movement | absolute_accumulated_surprise | 0.0794 | 0.0311 to 0.1271 | 3,044 |
| Context | rank_sd_change | accumulated_surprise | 0.0014 | -0.0032 to 0.0067 | 3,044 |
| Context | rank_sd_change | accumulated_surprise_x_preseason_strength | -0.0831 | -0.1307 to -0.0351 | 3,044 |
| Context | central80_width_change | accumulated_surprise | 0.0046 | -0.0094 to 0.0196 | 3,044 |
| Context | central80_width_change | accumulated_surprise_x_preseason_strength | -0.2215 | -0.3769 to -0.0895 | 3,044 |

Group mean contrasts (sustained group minus neutral) use two-way pigeonhole bootstrap intervals resampling team IDs and season-checkpoint snapshots.

| Family | Outcome | Contrast | Difference | 95% interval | Eligible rows A / B |
| --- | --- | --- | ---: | ---: | ---: |
| History | signed_location_change_since_previous_normalized | sustained_negative_minus_approximately_neutral | -0.1116 | -0.1648 to -0.0522 | 112 / 2,793 |
| History | signed_location_change_since_previous_normalized | sustained_positive_minus_approximately_neutral | 0.1103 | 0.0538 to 0.1530 | 139 / 2,793 |
| History | rank_sd_change_since_previous_normalized | sustained_negative_minus_approximately_neutral | -0.0343 | -0.0555 to -0.0120 | 112 / 2,793 |
| History | rank_sd_change_since_previous_normalized | sustained_positive_minus_approximately_neutral | -0.0290 | -0.0459 to -0.0097 | 139 / 2,793 |
| Context | signed_location_change_since_previous_normalized | sustained_negative_minus_approximately_neutral | -0.1004 | -0.1446 to -0.0470 | 98 / 2,817 |
| Context | signed_location_change_since_previous_normalized | sustained_positive_minus_approximately_neutral | 0.1010 | 0.0530 to 0.1429 | 129 / 2,817 |
| Context | rank_sd_change_since_previous_normalized | sustained_negative_minus_approximately_neutral | -0.0273 | -0.0461 to -0.0068 | 98 / 2,817 |
| Context | rank_sd_change_since_previous_normalized | sustained_positive_minus_approximately_neutral | -0.0232 | -0.0372 to -0.0072 | 129 / 2,817 |

The continuous directional slopes are positive in both families, and absolute movement also increases with absolute surprise. This matches the expected direction of posterior response in the retained data. The adjusted average surprise slopes for adjacent-checkpoint SD and central-80%-width changes include zero; their surprise-by-preseason-strength terms are negative in both families, so the uncertainty association varies with preseason strength in this descriptive model.

## History and Context differences

Paired differences use the same team-season-checkpoint in both families. The table reports checkpoint-7 History-minus-Context movement differences; continuous surprise-slope differences are Context minus History.

| Movement outcome at checkpoint 7 | History minus Context | 95% interval | Matched team-seasons |
| --- | ---: | ---: | ---: |
| signed_location_change_from_preseason_normalized | 0.0003 | -0.0075 to 0.0083 | 534 |
| rank_sd_change_from_preseason_normalized | -0.0127 | -0.0155 to -0.0101 | 534 |
| kl_posterior_from_prior | 0.0510 | 0.0254 to 0.0761 | 534 |
| total_variation_since_previous | 0.0011 | -0.0001 to 0.0022 | 534 |

| Context minus History slope | Difference | 95% interval | Matched team-checkpoints |
| --- | ---: | ---: | ---: |
| directional_location_movement | -0.0100 | -0.0192 to -0.0010 | 3,044 |
| absolute_location_movement | -0.0045 | -0.0153 to 0.0076 | 3,044 |
| rank_sd_change | -0.0003 | -0.0023 to 0.0016 | 3,044 |
| central80_width_change | -0.0007 | -0.0072 to 0.0061 | 3,044 |

All pooled intervals use 1,000 two-way pigeonhole resamples. These are approximate: posterior inference couples teams through shared evidence, only four exploratory seasons are retained, and repeated checkpoints are not independent. A relationship here is observational, not a causal effect of particular games.

## Lineage and temporal reconciliation

| Check | Result |
| --- | --- |
| #201 frozen #192 panel and evidence hashes | match |
| #247 recorded #201/#200 input hashes | match |
| Existing-policy History/Context labels | match |
| #192 and #201 existing likelihood hashes | 89eff21a304939a62beed6fd58dd642e0e4dbba698d7bdaa9aaca9f09f2de153 vs c56520a9c8fd1c62985941d9de413edecdfa89f1ab7a3baf714e5d111154e8e1 |
| Likelihood configurations identical | False |
| FBS focal-game counts matching exact #192 evidence | 7,476/7,476 |
| Joined #247 games at/after the checkpoint cutoff | 0 |

The rank posterior and game-margin surprise forecasts have distinct likelihood configurations. They are not treated as one fitted model: only #192 PMFs define posterior movement, and only #247 game surprises define the joined descriptive predictor.

The #192 and #200 existing-policy preseason PMFs also differ. Per-family exact hash matches and L1/expected-rank differences are recorded in `compatibility_audit.json`. The #247 forecast PMFs and preseason summaries were not substituted into #192 movement or strength calculations. The #192 evidence set is the join authority, rather than #201’s different rolling forecast cutoff; the joined focal-game counts exactly reproduce #192 `games_played`.

#247 forecast populations are retained separately from the #192 FBS rank supports: fbs_fbs 11,992 oriented rows, fbs_fcs 970 oriented rows.

| Family | #192/#200 exact prior PMF hashes | Median L1 | Maximum L1 | Median absolute expected-rank difference | Maximum absolute expected-rank difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| History | 0/534 | 0.00246 | 0.0556 | 0.048 | 2.020 |
| Context | 0/534 | 0.00218 | 0.3253 | 0.034 | 12.026 |

## Selected trajectories

Two examples were selected using only checkpoint-7 accumulated surprise, as the minimum and maximum mean across the History and Context #247 signals among team-seasons with at least three eligible games in both families. Selection did not use posterior movement or eventual ranking outcomes. Full seven-checkpoint PMFs for both families are in `selected_trajectory_pmfs.jsonl.gz`.

| Selection | Team-season | Mean family surprise at checkpoint 7 | History / Context |
| --- | --- | ---: | ---: |
| negative-surprise | Massachusetts / 2025 | -0.723 | -0.701 / -0.745 |
| positive-surprise | Indiana / 2025 | 0.738 | 0.756 / 0.721 |

## Recommendation under #244

Continue Phase 1 with a bounded conditional predictive-calibration investigation. Test whether posterior-derived future-game distributions are calibrated within preseason-strength and uncertainty strata and across the retained season checkpoints, preserving the leakage-safe forecast cutoffs. The current analysis characterizes how beliefs move; it does not establish whether the resulting beliefs improve or calibrate future predictions. The #247 study found no detectable persistent directional next-game error, so this next step should focus on conditional distribution calibration rather than changing priors or likelihood weights. Keep the historical seasons exploratory and do not start model interventions without predictive evidence.

## Reproducibility and interpretation

Reproduce with `uv run python scripts/run_issue_253_posterior_movement.py`. The input/output manifest records hashes, row counts, configuration, and environment. The PMF JSONL retains all serialized prior/posterior distributions used in the calculations. This analysis changes no production model and uses no 2026 outcomes.
