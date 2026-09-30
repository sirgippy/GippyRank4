# Partial DB transfer-impact coverage study (issue #142)

## Design

Ground truth is 281 complete incoming-DB team-seasons in the 2021–2025 frozen retrospective Context panel; 174 have at least two players and can retain a nonempty observed subset after masking. 3830 team-level masks were evaluated. The full player sum is checked against the committed Context feature before inclusion. Exclusions: {'incomplete_impact': 257, 'outside_context_panel': 5}. Only DB impacts are in scope; offensive usage is a separate feature.

For each complete roster, masks keep the roster fixed and hide one or more player impacts. Small combinations are exhaustive; strata with more than 80 masks retain the first and last plus a deterministic seeded sample. Policies: current all-or-nothing zero with availability 0; observed sum; observed sum multiplied by total/observed count; observed sum plus missing count times the other seasons' complete-player mean. The last policy never replaces known contributions. The primary table weights masks equally; a sensitivity analysis first averages within each historical (team-season, exact n/k coverage) cell and then gives those cells equal weight. These are feature-recovery diagnostics under controlled masking, not real missingness bias or predictive accuracy.

## Feature recovery

The machine-readable `policy_summary.csv` includes signed bias, MAE, RMSE, correlation with the full feature, frequency of errors above one impact unit, and fraction of masks beating the all-or-nothing zero. It breaks results out by exact coverage, coverage band, and missing-contribution magnitude and direction.

| Coverage | Masks | Policy | MAE | Bias | P(|error| > 1) |
| --- | ---: | --- | ---: | ---: | ---: |
| high_80_plus | 264 | all_or_nothing | 1.861 | -1.287 | 49.6% |
| high_80_plus | 264 | observed | 0.842 | -0.203 | 31.8% |
| high_80_plus | 264 | coverage_scaled | 0.885 | -0.000 | 39.0% |
| high_80_plus | 264 | missing_mean | 0.899 | +0.205 | 39.0% |
| moderate_50_to_80 | 2052 | all_or_nothing | 2.351 | -1.770 | 56.2% |
| moderate_50_to_80 | 2052 | observed | 1.389 | -0.711 | 52.5% |
| moderate_50_to_80 | 2052 | coverage_scaled | 1.432 | -0.005 | 56.9% |
| moderate_50_to_80 | 2052 | missing_mean | 1.375 | +0.136 | 57.4% |
| low_under_50 | 1514 | all_or_nothing | 2.342 | -1.780 | 55.6% |
| low_under_50 | 1514 | observed | 1.985 | -1.221 | 65.0% |
| low_under_50 | 1514 | coverage_scaled | 3.010 | -0.058 | 76.8% |
| low_under_50 | 1514 | missing_mean | 2.095 | +0.413 | 73.2% |

### Equal historical team-season/coverage weight

Each exact n/k mask set for one historical team-season contributes one average error before broad coverage aggregation. This limits the influence of rosters with many mask combinations. A team with several distinct coverage levels can contribute once to each level.

| Coverage | Team-season/coverage cells | Policy | MAE | P(|error| > 1) | Beats zero |
| --- | ---: | --- | ---: | ---: | ---: |
| high_80_plus | 33 | all_or_nothing | 2.384 | 66.7% | 0.0% |
| high_80_plus | 33 | observed | 0.773 | 26.1% | 77.7% |
| high_80_plus | 33 | coverage_scaled | 0.770 | 32.5% | 74.7% |
| high_80_plus | 33 | missing_mean | 0.793 | 34.2% | 75.5% |
| moderate_50_to_80 | 227 | all_or_nothing | 1.982 | 63.0% | 0.0% |
| moderate_50_to_80 | 227 | observed | 1.016 | 38.1% | 73.7% |
| moderate_50_to_80 | 227 | coverage_scaled | 1.026 | 45.1% | 64.7% |
| moderate_50_to_80 | 227 | missing_mean | 0.992 | 43.3% | 71.6% |
| low_under_50 | 137 | all_or_nothing | 2.352 | 67.2% | 0.0% |
| low_under_50 | 137 | observed | 1.866 | 61.3% | 64.5% |
| low_under_50 | 137 | coverage_scaled | 2.601 | 73.8% | 46.5% |
| low_under_50 | 137 | missing_mean | 1.776 | 67.3% | 58.5% |

Exact high-coverage strata:

| Coverage | Masks | Baseline MAE | Observed MAE | Scaled MAE | Missing-mean MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4/5 | 50 | 2.350 | 0.802 | 0.841 | 0.800 |
| 7/8 | 16 | 4.581 | 0.907 | 0.744 | 0.806 |
| 9/10 | 0 | unavailable | unavailable | unavailable | unavailable |
| 11/12 | 12 | 0.046 | 0.818 | 0.893 | 0.927 |

There is no complete ten-player DB team-season in the 2021–2025 historical panel. The 7/8 result comes from two complete eight-player rosters; interpret it as directional, not a precise population estimate.

## Consequential missing contributions

| Hidden contribution | Masks | Baseline MAE | Observed MAE | Scaled MAE | Missing-mean MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| small_le_1, positive | 807 | 1.266 | 0.495 | 1.264 | 0.663 |
| small_le_1, negative | 878 | 1.020 | 0.496 | 1.314 | 1.472 |
| large_gt_1, positive | 1587 | 3.990 | 2.722 | 2.429 | 1.718 |
| large_gt_1, negative | 558 | 1.099 | 1.653 | 3.049 | 3.005 |

For one hidden player in the bottom or top decile of complete-player impact (cutoffs -0.747 and 1.727):

| Hidden player impact | Masks | Baseline MAE | Observed MAE |
| --- | ---: | ---: | ---: |
| top_decile | 57 | 4.046 | 2.100 |
| bottom_decile | 79 | 1.272 | 0.749 |

## Missing-part uncertainty check

A secondary uncertainty proxy assigns the missing contribution a leave-season-out player mean and standard deviation times the square root of missing count. The known sum stays fixed. This assumes independent missing players, so empirical interval coverage is checked below rather than presumed. The interval is not a fitted Context posterior and is not recommended for production without calibration.

| Coverage | Masks | Nominal 90% interval coverage | Mean half-width |
| --- | ---: | ---: | ---: |
| high_80_plus | 264 | 89.4% | 1.669 |
| moderate_50_to_80 | 2052 | 86.2% | 2.382 |
| low_under_50 | 1514 | 80.1% | 3.322 |

## Empirical coverage and limitations

The separate issue #141 audit identifies 58 partially observed 2026 DB teams among 132 teams with incoming DB players; 51 partial teams have a roster size represented by at least one complete historical maskable team. The committed `empirical_2026_coverage.csv` records each actual n/k. This is a coverage comparison only: masking complete teams uniformly cannot establish how unresolved identities, origin coverage, or position conflicts select high-impact players. The retrospective source snapshots also do not establish August 15 availability.

Exact historical support for each 2026 n/k cell with at least one observed player is shown below. Cells backed by only one or two historical team-seasons are flagged even if they contain many masks or are reused for several current teams.

| Coverage | 2026 teams | Historical team-seasons | Historical masks | Support |
| --- | ---: | ---: | ---: | --- |
| 1/2 | 5 | 80 | 160 | multi_team |
| 2/3 | 7 | 35 | 105 | multi_team |
| 2/4 | 3 | 27 | 162 | multi_team |
| 3/4 | 3 | 27 | 108 | multi_team |
| 2/5 | 3 | 10 | 100 | multi_team |
| 3/5 | 3 | 10 | 100 | multi_team |
| 4/5 | 3 | 10 | 50 | multi_team |
| 4/6 | 3 | 13 | 195 | multi_team |
| 5/6 | 8 | 13 | 78 | multi_team |
| 6/7 | 6 | 6 | 42 | multi_team |
| 4/8 | 1 | 2 | 140 | sparse_two_teams |
| 6/8 | 1 | 2 | 56 | sparse_two_teams |
| 7/8 | 2 | 2 | 16 | sparse_two_teams |
| 6/9 | 1 | 0 | 0 | unmatched |
| 7/10 | 1 | 0 | 0 | unmatched |
| 9/10 | 1 | 0 | 0 | unmatched |
| 10/11 | 3 | 0 | 0 | unmatched |
| 10/12 | 2 | 1 | 66 | sparse_one_team |
| 12/13 | 1 | 0 | 0 | unmatched |

The exact n/k empirical replay matches 50 of 57 partially covered 2026 teams with observed values. Equal weighting by those teams gives all_or_nothing MAE 2.329, observed MAE 0.976, coverage_scaled MAE 0.980, missing_mean MAE 0.947. Each historical cell MAE is averaged across its distinct team-seasons before current teams are weighted. This transports historical masking errors to today's coverage frequencies; it cannot correct selection bias in which players are missing.

Excluding single-team historical cells leaves 48 current teams; their equal-team replay gives all_or_nothing MAE 2.425, observed MAE 0.973, coverage_scaled MAE 0.968, missing_mean MAE 0.935. The excluded cells are diagnostic only and should not set a coverage threshold.

One complete 2026 ten-player DB roster permits a separate 9/10 one-missing sensitivity (10 masks): all_or_nothing MAE 8.060, observed MAE 0.972, coverage_scaled MAE 1.038, missing_mean MAE 0.891. This is a current-season feature reconstruction, not a historical outcome test.

## Downstream Context response

The frozen 2026 Context 1.3 model and its committed preseason PMFs were used for one-missing masks on complete 2026 DB teams. Because the historical rank-distribution input panel is absent from this checkout, the exact posterior replay is unavailable: the model's fitted DB impact and binary availability location coefficients translate each saved rank PMF by interpolating its CDF on the rank-logit axis. The DB scale coefficients are zero in this fit; a new partial-aware indicator has no fitted coefficient. These are approximate local sensitivity diagnostics, not out-of-sample ranking scores or newly published predictions. Ordinal displacement compares the changed team with all other 2026 teams held at their published expected ranks.

| Policy | Cases | Mean absolute expected-rank shift | Mean PMF total variation | Mean absolute ordinal shift |
| --- | ---: | ---: | ---: | ---: |
| all_or_nothing | 248 | 2.368 | 0.0348 | 2.786 |
| observed | 248 | 1.101 | 0.0167 | 1.367 |
| coverage_scaled | 248 | 1.365 | 0.0210 | 1.718 |
| missing_mean | 248 | 1.323 | 0.0199 | 1.677 |

## Recommendation

Preserve the observed incoming-DB impact sum whenever at least one relevant player's impact resolves, including at low coverage; zero remains a valid observed contribution. This is a representation recommendation, not a claim that Context should trust a 1/n partial value as strongly as a complete sum. In the low-coverage mask-weighted band, observed-sum MAE falls from 2.342 to 1.985, but P(|error| > 1) rises from 55.6% to 65.0% and observed beats zero on only 54.8% of masks. Under equal historical team-season/coverage weight, the same tail frequency falls from 67.2% to 61.3%; the tail comparison depends on weighting. The missing-part 90% proxy covers only about 80% there. The model should account for coverage and learn or calibrate reduced confidence before a production change; the study does not establish that simply swapping numeric values into frozen Context improves low-coverage posteriors. Do not scale by count or fill the entire team feature. Keep the unobserved portion unknown pending a separately validated uncertainty model. For no observed players, keep impact unavailable and numeric neutral zero. For no incoming DB transfers, retain the current natural zero with complete coverage. The results do not justify a hard fractional cutoff: average recovery and tail risk move differently, and sparse exact n/k cells cannot locate a stable break point.

Persist `incoming_db_count`, `observed_db_impact_count`, and `observed_db_impact_sum` with source/season provenance. Derive missing count and fraction from the first two counts. These fields distinguish complete, partial, and absent coverage; zero incoming is a complete natural zero. Keep player-level contributor/status audit links for traceability. The existing binary availability flag cannot express partial coverage, and changing the numeric feature while retaining frozen coefficients changes the model's input semantics. Treat adoption as a new Context model version with a trained and validated coverage-aware contract, not a silent Context 1.3 clarification. No production behavior or published rankings changed in this issue.

## Reproduce

`UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/study_partial_db_transfer_impact.py` writes this report and the machine-readable masks, errors, coverage, downstream diagnostics, and input hashes. `summary.json` records the seed and leave-season-out player means. The downstream response is an interpolation approximation, not a causal estimate of future-season performance.
