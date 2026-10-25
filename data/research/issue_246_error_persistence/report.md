# Issue 246: sequential in-season forecast-error persistence

## Result

This leakage-safe descriptive analysis asks whether a team's signed pregame margin error in one regular-season game predicts the error in that team's next game, after the earlier result enters the next forecast's evidence.

The four previously studied seasons (2022–2025) are exploratory historical data, not independent confirmation. No models are refit and no production behavior is changed.

The retained #201 primary FBS–FBS score summaries passed parity for **16 arm-season rows**; the largest absolute difference from its stored mean CRPS or margin NLL was 1.78e-15. The #201 inputs and retained #200 rank-PMF bundle are recorded by hash in manifest.json.

**Finding:** No consistent sequential error-persistence signal was detected. Across the four pooled arms, lag-1 correlations range from 0.004 to 0.008, and directional excess over the sign-marginal independence baseline ranges from -0.003 to -0.002; their 95% intervals include zero. Season, progress, and preseason-rank-prior contrasts also include their no-effect values. These results fail to detect a consistent effect; they do not demonstrate that meaningful effects are absent.

**Recommendation:** Defer model changes whose rationale is sequential error persistence based on this exploratory retrospective analysis. This does not resolve the broader calibration epic or establish that the prior-likelihood balance is correct; assess other calibration questions using their own evidence.

## Sample and eligibility

The panel contains **25,924 oriented FBS team-games** across **3,483 matchups** and **2,136 model-arm/team-seasons**. It includes both FBS perspectives for FBS–FBS games and the FBS perspective for FBS–FCS games, preserving each FBS team's regular-season sequence.

The retained #200 preseason-rank PMFs classify **2,136 team-seasons**, once per model arm. There are **23,788 adjacent candidate transitions**; **21,900** pass the evidence check and **1,888** are excluded because the previous game is absent from the next forecast's evidence IDs. Excluded pairs and reasons remain in lagged_error_pairs.csv.

| Season | Prior family | Policy | Candidate pairs | Eligible pairs | Excluded |
| --- | --- | --- | ---: | ---: | ---: |
| 2022 | History | existing | 1457 | 1329 | 128 |
| 2023 | History | existing | 1485 | 1362 | 123 |
| 2024 | History | existing | 1491 | 1385 | 106 |
| 2025 | History | existing | 1514 | 1399 | 115 |
| 2022 | Context | existing | 1457 | 1329 | 128 |
| 2023 | Context | existing | 1485 | 1362 | 123 |
| 2024 | Context | existing | 1491 | 1385 | 106 |
| 2025 | Context | existing | 1514 | 1399 | 115 |
| 2022 | History | pre_bowl | 1457 | 1329 | 128 |
| 2023 | History | pre_bowl | 1485 | 1362 | 123 |
| 2024 | History | pre_bowl | 1491 | 1385 | 106 |
| 2025 | History | pre_bowl | 1514 | 1399 | 115 |
| 2022 | Context | pre_bowl | 1457 | 1329 | 128 |
| 2023 | Context | pre_bowl | 1485 | 1362 | 123 |
| 2024 | Context | pre_bowl | 1491 | 1385 | 106 |
| 2025 | Context | pre_bowl | 1514 | 1399 | 115 |

## Lag-1 results by season and prior family

Signed error is observed team margin minus expected team margin; positive means the team performed better than expected. Lag-1 correlation is Pearson correlation between consecutive eligible errors. For nonzero error pairs, the table shows the observed same-direction share and the independence baseline computed from the observed positive-sign shares for the previous and next errors. The directional excess is observed persistence minus that baseline; zero is the no-association reference.

Intervals are 95% percentile intervals from 1,000 deterministic multiway pigeonhole bootstrap repetitions, resampling focal team IDs and game IDs jointly across both lag positions. The directional-excess interval recomputes the sign marginals and independence baseline on every resample.

| Season | Prior family | Policy | Pairs | Teams | Lag-1 r [95% interval] | Same direction / baseline | Excess over independence [95% interval] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | History | existing | 1329 | 131 | -0.010 [-0.153, 0.135] | 0.490 / 0.500 | -0.010 [-0.080, 0.061] |
| 2022 | Context | existing | 1329 | 131 | -0.012 [-0.167, 0.133] | 0.497 / 0.500 | -0.003 [-0.077, 0.065] |
| 2022 | History | pre_bowl | 1329 | 131 | -0.010 [-0.147, 0.119] | 0.490 / 0.500 | -0.010 [-0.080, 0.057] |
| 2022 | Context | pre_bowl | 1329 | 131 | -0.012 [-0.159, 0.124] | 0.494 / 0.500 | -0.006 [-0.074, 0.059] |
| 2023 | History | existing | 1362 | 133 | 0.003 [-0.138, 0.141] | 0.491 / 0.500 | -0.009 [-0.080, 0.055] |
| 2023 | Context | existing | 1362 | 133 | 0.003 [-0.139, 0.143] | 0.501 / 0.500 | 0.001 [-0.070, 0.061] |
| 2023 | History | pre_bowl | 1362 | 133 | 0.002 [-0.141, 0.146] | 0.493 / 0.500 | -0.007 [-0.079, 0.061] |
| 2023 | Context | pre_bowl | 1362 | 133 | 0.002 [-0.141, 0.133] | 0.500 / 0.500 | -0.000 [-0.068, 0.066] |
| 2024 | History | existing | 1385 | 134 | 0.020 [-0.116, 0.164] | 0.510 / 0.500 | 0.010 [-0.065, 0.079] |
| 2024 | Context | existing | 1385 | 134 | 0.014 [-0.137, 0.162] | 0.500 / 0.500 | 0.000 [-0.070, 0.070] |
| 2024 | History | pre_bowl | 1385 | 134 | 0.020 [-0.118, 0.161] | 0.510 / 0.500 | 0.010 [-0.058, 0.080] |
| 2024 | Context | pre_bowl | 1385 | 134 | 0.013 [-0.131, 0.156] | 0.499 / 0.500 | -0.001 [-0.071, 0.062] |
| 2025 | History | existing | 1399 | 136 | 0.016 [-0.119, 0.152] | 0.498 / 0.500 | -0.002 [-0.069, 0.065] |
| 2025 | Context | existing | 1399 | 136 | 0.015 [-0.126, 0.142] | 0.494 / 0.500 | -0.006 [-0.074, 0.057] |
| 2025 | History | pre_bowl | 1399 | 136 | 0.015 [-0.116, 0.153] | 0.497 / 0.500 | -0.003 [-0.066, 0.064] |
| 2025 | Context | pre_bowl | 1399 | 136 | 0.014 [-0.131, 0.148] | 0.500 / 0.500 | -0.001 [-0.068, 0.065] |

## Pooled results by model arm

The previous/next positive-sign shares show why 0.5 is not assumed as the null. With marginal positive shares p1 and p2, the independence baseline is p1*p2 + (1-p1)*(1-p2).

| Prior family | Policy | Pairs | Positive previous / next | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| History | existing | 5475 | 0.503 / 0.503 | 0.497 / 0.500 | -0.003 [-0.037, 0.033] | 0.008 [-0.064, 0.078] |
| Context | existing | 5475 | 0.503 / 0.504 | 0.498 / 0.500 | -0.002 [-0.036, 0.033] | 0.005 [-0.066, 0.079] |
| History | pre_bowl | 5475 | 0.503 / 0.503 | 0.498 / 0.500 | -0.002 [-0.038, 0.032] | 0.007 [-0.066, 0.079] |
| Context | pre_bowl | 5475 | 0.504 / 0.504 | 0.498 / 0.500 | -0.002 [-0.040, 0.034] | 0.004 [-0.069, 0.078] |

## Strong preseason rank priors

A team-season is classified independently within each arm from its retained #200 preseason rank PMF. Strong prior means P(rank <= 10) >= 0.50. The group label is fixed for every game and transition in that team-season; it does not depend on opponents or game-level win probabilities.

| Prior family | Policy | Stratum | Pairs | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| History | existing | strong_preseason_rank_prior | 151 | 0.570 / 0.520 | 0.050 [-0.154, 0.233] | 0.059 [-0.331, 0.430] |
| History | existing | other | 5324 | 0.495 / 0.500 | -0.005 [-0.040, 0.029] | 0.005 [-0.065, 0.076] |
| Context | existing | strong_preseason_rank_prior | 277 | 0.570 / 0.512 | 0.058 [-0.097, 0.189] | 0.088 [-0.200, 0.361] |
| Context | existing | other | 5198 | 0.494 / 0.500 | -0.006 [-0.040, 0.030] | -0.000 [-0.070, 0.074] |
| History | pre_bowl | strong_preseason_rank_prior | 151 | 0.570 / 0.517 | 0.052 [-0.131, 0.216] | 0.059 [-0.313, 0.416] |
| History | pre_bowl | other | 5324 | 0.495 / 0.500 | -0.005 [-0.043, 0.030] | 0.005 [-0.069, 0.078] |
| Context | pre_bowl | strong_preseason_rank_prior | 266 | 0.564 / 0.511 | 0.053 [-0.102, 0.182] | 0.091 [-0.234, 0.349] |
| Context | pre_bowl | other | 5209 | 0.495 / 0.500 | -0.005 [-0.039, 0.029] | -0.001 [-0.072, 0.078] |

Paired strong-prior-minus-other differences:

| Prior family | Policy | Metric | Difference [95% interval] |
| --- | --- | --- | ---: |
| History | existing | lag1_correlation | 0.054 [-0.368, 0.423] |
| History | existing | directional_excess_over_independence | 0.055 [-0.147, 0.225] |
| Context | existing | lag1_correlation | 0.088 [-0.236, 0.362] |
| Context | existing | directional_excess_over_independence | 0.064 [-0.094, 0.200] |
| History | pre_bowl | lag1_correlation | 0.054 [-0.382, 0.449] |
| History | pre_bowl | directional_excess_over_independence | 0.057 [-0.167, 0.256] |
| Context | pre_bowl | lag1_correlation | 0.092 [-0.239, 0.364] |
| Context | pre_bowl | directional_excess_over_independence | 0.058 [-0.102, 0.201] |

## Change over the season

Transitions are assigned by the destination team's game number: early games 2–4, middle games 5–8, and late games 9+.

| Prior family | Policy | Progress | Pairs | Same direction / independence baseline | Excess over independence [95% interval] | Lag-1 r [95% interval] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| History | existing | Early (games 2–4) | 1466 | 0.489 / 0.500 | -0.011 [-0.075, 0.051] | 0.019 [-0.118, 0.150] |
| History | existing | Middle (games 5–8) | 2041 | 0.504 / 0.500 | 0.004 [-0.052, 0.063] | 0.003 [-0.109, 0.126] |
| History | existing | Late (game 9+) | 1968 | 0.497 / 0.500 | -0.003 [-0.063, 0.051] | 0.003 [-0.124, 0.119] |
| Context | existing | Early (games 2–4) | 1466 | 0.508 / 0.500 | 0.008 [-0.054, 0.072] | 0.005 [-0.133, 0.140] |
| Context | existing | Middle (games 5–8) | 2041 | 0.494 / 0.500 | -0.006 [-0.065, 0.047] | 0.004 [-0.121, 0.115] |
| Context | existing | Late (game 9+) | 1968 | 0.494 / 0.500 | -0.006 [-0.069, 0.052] | 0.005 [-0.112, 0.119] |
| History | pre_bowl | Early (games 2–4) | 1466 | 0.490 / 0.500 | -0.010 [-0.078, 0.053] | 0.018 [-0.117, 0.154] |
| History | pre_bowl | Middle (games 5–8) | 2041 | 0.502 / 0.500 | 0.002 [-0.052, 0.058] | 0.002 [-0.110, 0.126] |
| History | pre_bowl | Late (game 9+) | 1968 | 0.498 / 0.500 | -0.002 [-0.063, 0.055] | 0.002 [-0.111, 0.120] |
| Context | pre_bowl | Early (games 2–4) | 1466 | 0.504 / 0.500 | 0.004 [-0.067, 0.069] | 0.004 [-0.130, 0.142] |
| Context | pre_bowl | Middle (games 5–8) | 2041 | 0.494 / 0.500 | -0.006 [-0.064, 0.049] | 0.004 [-0.123, 0.121] |
| Context | pre_bowl | Late (game 9+) | 1968 | 0.498 / 0.500 | -0.002 [-0.059, 0.058] | 0.004 [-0.117, 0.126] |

Paired late-minus-early differences:

| Prior family | Policy | Metric | Difference [95% interval] |
| --- | --- | --- | ---: |
| History | existing | lag1_correlation | -0.016 [-0.195, 0.160] |
| History | existing | directional_excess_over_independence | 0.008 [-0.080, 0.093] |
| Context | existing | lag1_correlation | -0.000 [-0.176, 0.174] |
| Context | existing | directional_excess_over_independence | -0.014 [-0.100, 0.073] |
| History | pre_bowl | lag1_correlation | -0.015 [-0.193, 0.163] |
| History | pre_bowl | directional_excess_over_independence | 0.008 [-0.085, 0.098] |
| Context | pre_bowl | lag1_correlation | 0.000 [-0.172, 0.173] |
| Context | pre_bowl | directional_excess_over_independence | -0.006 [-0.091, 0.075] |

## Interpretation and limits

Intervals that include zero mean the analysis did not detect a clear effect at this resolution; they do not demonstrate that meaningful effects are absent. These intervals are not equivalence tests. Likewise, lack of lag-1 persistence does not establish that the prior-likelihood balance is correct.

The retained #201 forecasts account for opponent strength and game site. Error persistence alone does not identify a cause or justify changing a likelihood, prior, or production behavior. The broader calibration epic remains a separate question.

This differs from issue #34, which examined historical performance/rank residuals. Issue 246 uses leakage-safe pregame margin forecasts and requires the earlier game to appear in the next forecast's evidence. The four arms are shown separately; repeated teams and shared matchups are handled in the bootstrap, while the four seasons remain exploratory historical data.

## Reproducibility

Reproduce from the repository root with: uv run python scripts/run_issue_246_error_persistence.py

The runner verifies the retained #201 forecast artifacts and the #200 pipeline bundle and PMF hashes. It does not refit models or rebuild the forecasting pipeline. preseason_rank_classification.csv, team_game_errors.csv, and lagged_error_pairs.csv retain the team-season labels, game errors, and transition eligibility decisions; all outputs and inputs are hashed in manifest.json.

## Source parity

| Arm-season rows checked | Maximum absolute difference | Status |
| ---: | ---: | --- |
| 16 | 1.78e-15 | passed |
