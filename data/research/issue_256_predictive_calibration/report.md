# Issue 256: conditional in-season predictive calibration

## Finding

**Recommendation under #244: C — Inconclusive evidence.**

This study evaluates the retained #201 game-margin predictive distributions for the existing-policy History and Context families. It does not directly validate the #253 rank-PMF uncertainty trajectories, which use a different retained configuration. The analysis is historical and exploratory.

The primary population contains one rolling forecast per FBS–FBS regular-season game, family, and #201 cutoff. Preseason forecasts are shown separately as a reference. FBS–FCS results are separate. The retained #201 interval indicators passed row-level recalculation parity.

## Primary FBS–FBS rolling calibration

Coverage error is observed minus nominal coverage in percentage points. Intervals are 95% two-way team/game pigeonhole bootstrap percentile intervals.

| Family | Games | 80% coverage [95% interval] | Error (pp) | Upper / lower misses | Mean / median 80% width | CRPS | Margin NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| History | 2998 | 80.2% [77.9%, 82.2%] | +0.19 | 9.8% / 10.0% | 41.44 / 40.52 | 8.997 | 4.193 |
| Context | 2998 | 80.4% [78.3%, 82.3%] | +0.39 | 9.7% / 9.9% | 41.14 / 40.31 | 8.908 | 4.184 |

### Equal-season summary

Each year contributes equally to these four-season estimates.

| Family | 80% coverage [95% interval] | Error (pp) | Mean width | CRPS | Margin NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| History | 80.2% [78.1%, 82.2%] | +0.18 | 41.44 | 8.999 | 4.193 |
| Context | 80.4% [78.4%, 82.4%] | +0.38 | 41.14 | 8.909 | 4.184 |

### By season

| Season | Family | Games | 80% coverage [95% interval] | Error (pp) | Mean width |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2022 | History | 734 | 79.3% [75.1%, 83.0%] | -0.71 | 41.51 |
| 2022 | Context | 734 | 80.0% [75.8%, 83.6%] | -0.03 | 41.23 |
| 2023 | History | 750 | 80.4% [76.7%, 84.1%] | +0.40 | 41.47 |
| 2023 | Context | 750 | 80.7% [76.5%, 84.5%] | +0.67 | 41.15 |
| 2024 | History | 752 | 78.9% [74.7%, 82.9%] | -1.14 | 41.35 |
| 2024 | Context | 752 | 78.7% [74.6%, 82.6%] | -1.28 | 41.03 |
| 2025 | History | 762 | 82.2% [78.2%, 85.8%] | +2.15 | 41.44 |
| 2025 | Context | 762 | 82.2% [78.2%, 86.0%] | +2.15 | 41.14 |

### Season progression and preseason strength

The following are team-oriented conditional estimates. FBS–FBS games contribute both team perspectives; `unique games` exposes that shared-game dependence. Family-specific strength quartiles use distinct FBS team-seasons pooled across 2022–2025 and linear sample quantiles, before examining coverage; they are descriptive only.

Strength reference counts and cut points (q25 / q50 / q75): History: 534 team-seasons, 0.343554 / 0.498786 / 0.642604; Context: 534 team-seasons, 0.321379 / 0.493047 / 0.676477.

| Analysis | Group | Family | Team-game rows | Unique games | 80% coverage [95% interval] | Error (pp) | Mean width |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| season_progression | early_0_2 | History | 1234 | 661 | 81.5% [77.1%, 85.2%] | +1.52 | 46.21 |
| season_progression | early_0_2 | Context | 1234 | 661 | 82.6% [78.3%, 86.9%] | +2.58 | 45.29 |
| season_progression | middle_3_6 | History | 2072 | 1116 | 81.9% [78.5%, 85.2%] | +1.95 | 41.59 |
| season_progression | middle_3_6 | Context | 2072 | 1116 | 82.3% [78.9%, 85.4%] | +2.29 | 41.36 |
| season_progression | late_7_plus | History | 2690 | 1381 | 78.2% [75.0%, 81.3%] | -1.78 | 39.14 |
| season_progression | late_7_plus | Context | 2690 | 1381 | 77.9% [74.8%, 81.0%] | -2.08 | 39.06 |
| preseason_strength | q1_low | History | 1488 | 1186 | 80.2% [76.7%, 83.6%] | +0.17 | 41.34 |
| preseason_strength | q1_low | Context | 1491 | 1168 | 79.5% [75.4%, 83.2%] | -0.46 | 41.03 |
| preseason_strength | q2_lower_middle | History | 1491 | 1300 | 79.7% [75.8%, 83.3%] | -0.32 | 41.73 |
| preseason_strength | q2_lower_middle | Context | 1481 | 1268 | 80.1% [76.2%, 83.7%] | +0.08 | 41.45 |
| preseason_strength | q3_upper_middle | History | 1488 | 1289 | 80.8% [77.2%, 84.3%] | +0.78 | 41.73 |
| preseason_strength | q3_upper_middle | Context | 1495 | 1261 | 80.8% [77.1%, 84.6%] | +0.80 | 41.49 |
| preseason_strength | q4_high | History | 1529 | 1233 | 80.1% [76.6%, 83.4%] | +0.12 | 40.98 |
| preseason_strength | q4_high | Context | 1529 | 1199 | 81.1% [77.5%, 84.3%] | +1.10 | 40.59 |

## Accumulated surprise

Only issue 247 observations with at least three evidence-eligible prior games enter this section. Surprise groups reuse the frozen ±0.5 thresholds. Counts are team-game perspectives; unique game counts show the shared outcomes.

| Prior surprise | Family | Team-game rows | Unique games | 80% coverage [95% interval] | Error (pp) | Upper / lower misses | Mean width | CRPS | NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sustained_negative | History | 187 | 181 | 77.5% [66.5%, 87.3%] | -2.46 | 10.2% / 12.3% | 40.60 | 8.611 | 4.144 |
| sustained_negative | Context | 160 | 157 | 78.8% [67.9%, 89.1%] | -1.25 | 12.5% / 8.8% | 40.51 | 8.445 | 4.128 |
| approximately_neutral | History | 4328 | 2377 | 79.9% [77.6%, 82.2%] | -0.08 | 9.8% / 10.2% | 40.18 | 8.844 | 4.175 |
| approximately_neutral | Context | 4375 | 2384 | 79.8% [77.5%, 82.1%] | -0.16 | 9.9% / 10.3% | 40.04 | 8.841 | 4.175 |
| sustained_positive | History | 247 | 240 | 80.2% [70.9%, 88.2%] | +0.16 | 12.6% / 7.3% | 40.36 | 8.918 | 4.189 |
| sustained_positive | Context | 227 | 221 | 80.2% [70.5%, 88.8%] | +0.18 | 11.5% / 8.4% | 40.19 | 8.806 | 4.174 |

### Prespecified continuous miss models

Linear-probability coefficients are percentage-point changes in 80% interval miss probability. Preseason terms are per 0.1 change in normalized strength or uncertainty; surprise terms are per 0.5 surprise units. Interaction terms are adjusted for season and progression as specified in the frozen protocol. They are descriptive associations, not causal effects.

| Analysis | Family | Term | Estimate (pp) [95% interval] | Team-game rows |
| --- | --- | --- | ---: | ---: |
| preseason_expectations | History | strength_per_0_1_early | +0.58 [-1.27, +2.38] | 5996 |
| preseason_expectations | History | uncertainty_per_0_1_early | -8.11 [-18.28, +1.44] | 5996 |
| preseason_expectations | History | strength_per_0_1_x_middle_3_6 | -0.90 [-3.40, +1.58] | 5996 |
| preseason_expectations | History | uncertainty_per_0_1_x_middle_3_6 | +5.91 [-7.03, +17.88] | 5996 |
| preseason_expectations | History | strength_per_0_1_x_late_7_plus | -0.88 [-3.17, +1.58] | 5996 |
| preseason_expectations | History | uncertainty_per_0_1_x_late_7_plus | +12.11 [-0.14, +24.14] | 5996 |
| accumulated_surprise_continuous | History | signed_surprise_per_0_5 | -0.11 [-3.24, +2.94] | 4762 |
| accumulated_surprise_continuous | History | absolute_surprise_per_0_5 | +0.65 [-4.82, +6.05] | 4762 |
| accumulated_surprise_continuous | History | negative_surprise_magnitude_x_strength_per_0_5_by_0_1 | -0.91 [-3.74, +2.17] | 4762 |
| preseason_expectations | Context | strength_per_0_1_early | -0.10 [-1.93, +1.63] | 5996 |
| preseason_expectations | Context | uncertainty_per_0_1_early | -6.89 [-16.29, +2.26] | 5996 |
| preseason_expectations | Context | strength_per_0_1_x_middle_3_6 | -0.39 [-2.54, +1.90] | 5996 |
| preseason_expectations | Context | uncertainty_per_0_1_x_middle_3_6 | +7.21 [-4.81, +18.73] | 5996 |
| preseason_expectations | Context | strength_per_0_1_x_late_7_plus | -0.16 [-2.32, +2.14] | 5996 |
| preseason_expectations | Context | uncertainty_per_0_1_x_late_7_plus | +9.70 [-1.34, +20.38] | 5996 |
| accumulated_surprise_continuous | Context | signed_surprise_per_0_5 | +0.08 [-3.23, +3.57] | 4762 |
| accumulated_surprise_continuous | Context | absolute_surprise_per_0_5 | +1.92 [-4.14, +7.54] | 4762 |
| accumulated_surprise_continuous | Context | negative_surprise_magnitude_x_strength_per_0_5_by_0_1 | -0.60 [-3.44, +2.30] | 4762 |

## Paired History–Context comparison

Differences are History minus Context on matched FBS–FBS games and compatible #201 cutoffs. Negative CRPS or NLL differences favor History on that score; coverage differences are descriptive. Intervals use the same two-way dependence-aware bootstrap.

| Metric | Matched games | Mean difference [95% interval] |
| --- | ---: | ---: |
| coverage_80_difference_pp (percentage_points) | 2998 | -0.200 [-1.028, +0.661] |
| width_difference (margin_points) | 2998 | +0.306 [+0.278, +0.337] |
| crps_difference (margin_points) | 2998 | +0.089 [+0.036, +0.144] |
| margin_nll_difference (log_density) | 2998 | +0.009 [+0.004, +0.015] |

## Separate reference results

| Population / stage | Family | Forecasts | 80% coverage | Mean width | CRPS | Margin NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FBS–FCS rolling | History | 485 | 84.1% | 47.32 | 9.613 | 4.253 |
| FBS–FCS rolling | Context | 485 | 84.3% | 47.01 | 9.493 | 4.241 |
| FBS–FBS preseason reference | History | 2998 | 83.3% | 49.10 | 9.900 | 4.295 |
| FBS–FBS preseason reference | Context | 2998 | 83.8% | 47.90 | 9.634 | 4.268 |
| FBS–FCS preseason reference | History | 485 | 84.5% | 51.07 | 10.029 | 4.302 |
| FBS–FCS preseason reference | Context | 485 | 84.9% | 50.67 | 9.859 | 4.287 |

## Decision and limitations

The frozen decision screen returned **C**. Both families' pooled FBS–FBS coverage intervals fall within the ±5 percentage-point global band, and no prespecified subgroup shows a ≥10-point error with the required consistency. However, the sustained-negative and sustained-positive surprise groups have fewer than 250 unique games per family, and their intervals are too wide to rule out a meaningful localized deviation. The continuous miss models also remain imprecise. The four exploratory seasons are not independent replications.

This supports an inconclusive Phase 1 result rather than a mechanism investigation: do not initiate Phase 2 from these data alone, and do not close #244 as calibrated. If the sustained-surprise question remains important, the specific next evidence would be prospective out-of-time calibration with enough repeated-surprise team-games to narrow those intervals; it does not justify prior, likelihood, or inference changes now.

CRPS and NLL provide scoring context; wider intervals can improve coverage without improving sharpness.

No calibration result alone identifies a cause. This study does not attribute deviations to preseason priors, game likelihood, posterior inference, or latent-rank assumptions. Do not change production behavior from these historical results.

## Reproducibility

Run `uv run python -m gippyrank.research.issue_256_predictive_calibration` from the repository root. The runner verifies issue 201/247 input and evidence hashes, interval parity, exact cutoff joins, prior-game IDs, prior surprise means, and target-game exclusion. It uses the retained #201 scores and does not rebuild predictive mixtures.

Counts: {"coverage_parity_checks": 41796, "forecast_rows": 13932, "forecast_rows_by_stage_population_family": {"preseason/fbs_fbs/Context": 2998, "preseason/fbs_fbs/History": 2998, "preseason/fbs_fcs/Context": 485, "preseason/fbs_fcs/History": 485, "rolling/fbs_fbs/Context": 2998, "rolling/fbs_fbs/History": 2998, "rolling/fbs_fcs/Context": 485, "rolling/fbs_fcs/History": 485}, "issue201_aggregate_parity": {"cross_season_aggregate_checks": 64, "phase_summary_checks": 672, "season_summary_checks": 256}, "pairing": {"matched_fbs_fbs_forecasts": 2998, "paired_team_orientations": 5996, "unmatched_game_forecasts": 0}, "prior_history_checks": 12962, "rolling_forecasts_joined": 6966, "surprise_eligible_team_rows_by_family": {"Context": 4762, "History": 4762}, "target_game_excluded_from_prior_histories": true, "team_oriented_rows": 12962, "team_rows_by_population_family": {"fbs_fbs/Context": 5996, "fbs_fbs/History": 5996, "fbs_fcs/Context": 485, "fbs_fcs/History": 485}, "unique_team_forecasts": 12962}

See `manifest.json` for input/output hashes, pairing audits, bootstrap configuration, and row counts.
