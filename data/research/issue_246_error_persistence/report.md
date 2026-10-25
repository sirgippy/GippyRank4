# Issue 246: sequential in-season forecast-error persistence

## Result

This is a leakage-safe descriptive analysis of retained issue 201 forecasts. It asks whether a team's signed pregame margin error in one regular-season game predicts the error in that same team's next game, after the earlier result is present in the next forecast's evidence.

The sample contains **four previously studied seasons (2022–2025)**, so the estimates are exploratory and are not independent confirmation. The analysis makes no model or production changes.

The retained issue 201 primary FBS–FBS score summaries passed parity for **16 arm-season rows**; the largest absolute difference from its stored mean CRPS or margin NLL was `1.78e-15`. The source score, summary, evidence, and manifest hashes are recorded in `manifest.json`.

**Finding:** No consistent sequential directional-error signal was detected. Across the four pooled arms, lag-1 correlations range from 0.004 to 0.008, and same-direction rates range from 0.497 to 0.498. Their intervals cover zero and 0.5 respectively; the season, progress, and preseason-favorite contrasts also include their no-effect values. Weak effects remain possible because these intervals are not equivalence tests.

**Recommendation:** Defer model changes and further retrospective responsiveness probes. If this question remains decision-relevant, run a pre-registered prospective replication on a genuinely later holdout, keeping the same evidence-membership rule and team/game dependence handling. Do not tune the model from this historical panel.

## Sample and eligibility

The team-game panel has **25,924 oriented FBS team-games** across **3,483 scheduled matchups** and **2,136 model-arm/team-seasons**. It includes each FBS–FBS game from both FBS teams' perspectives, and each FBS–FCS game once from the FBS team's perspective. FBS–FCS rows are needed to keep the FBS team's full regular-season sequence; FCS-only team histories are not inferred from this corpus.

There are **23,788 adjacent candidate transitions**; **21,900** pass the evidence check and **1,888** are excluded because the previous game is absent from the next forecast's included evidence IDs. The excluded pairs remain in `lagged_error_pairs.csv` with reasons. The next game's own game ID is checked to be absent from its evidence, and its cutoff must precede kickoff.

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

Signed error is `observed team margin − expected team margin`. Positive means the team scored better than the model expected from its perspective; negative means worse. The retained #201 predictive mean is used. Lag-1 correlation is Pearson correlation between consecutive eligible signed errors. Directional persistence is the share of nonzero pairs with the same sign; 0.5 is the no-directional-persistence reference.

Intervals are 95% percentile intervals from 1,000 deterministic multiway pigeonhole bootstrap repetitions, resampling focal team IDs and game IDs jointly across both lag positions. This preserves repeated-team and shared-matchup dependence, including the two orientations of an FBS–FBS matchup. The intervals describe this historical panel and do not create four independent seasons.

| Season | Prior family | Policy | Eligible pairs | Teams | Lag-1 r [95% interval] | Same-direction rate [95% interval] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2022 | History | existing | 1329 | 131 | -0.010 [-0.153, 0.135] | 0.490 [0.420, 0.562] |
| 2022 | Context | existing | 1329 | 131 | -0.012 [-0.167, 0.133] | 0.497 [0.424, 0.566] |
| 2022 | History | pre_bowl | 1329 | 131 | -0.010 [-0.147, 0.119] | 0.490 [0.420, 0.558] |
| 2022 | Context | pre_bowl | 1329 | 131 | -0.012 [-0.159, 0.124] | 0.494 [0.427, 0.559] |
| 2023 | History | existing | 1362 | 133 | 0.003 [-0.138, 0.141] | 0.491 [0.420, 0.556] |
| 2023 | Context | existing | 1362 | 133 | 0.003 [-0.139, 0.143] | 0.501 [0.431, 0.563] |
| 2023 | History | pre_bowl | 1362 | 133 | 0.002 [-0.141, 0.146] | 0.493 [0.420, 0.562] |
| 2023 | Context | pre_bowl | 1362 | 133 | 0.002 [-0.141, 0.133] | 0.500 [0.433, 0.568] |
| 2024 | History | existing | 1385 | 134 | 0.020 [-0.116, 0.164] | 0.510 [0.435, 0.580] |
| 2024 | Context | existing | 1385 | 134 | 0.014 [-0.137, 0.162] | 0.500 [0.430, 0.571] |
| 2024 | History | pre_bowl | 1385 | 134 | 0.020 [-0.118, 0.161] | 0.510 [0.442, 0.582] |
| 2024 | Context | pre_bowl | 1385 | 134 | 0.013 [-0.131, 0.156] | 0.499 [0.429, 0.564] |
| 2025 | History | existing | 1399 | 136 | 0.016 [-0.119, 0.152] | 0.498 [0.431, 0.566] |
| 2025 | Context | existing | 1399 | 136 | 0.015 [-0.126, 0.142] | 0.494 [0.427, 0.556] |
| 2025 | History | pre_bowl | 1399 | 136 | 0.015 [-0.116, 0.153] | 0.497 [0.435, 0.564] |
| 2025 | Context | pre_bowl | 1399 | 136 | 0.014 [-0.131, 0.148] | 0.500 [0.433, 0.569] |

## Pooled results by model arm

| Prior family | Policy | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |
| --- | --- | ---: | ---: | ---: |
| History | existing | 5475 | 0.008 [-0.064, 0.078] | 0.497 [0.463, 0.534] |
| Context | existing | 5475 | 0.005 [-0.066, 0.079] | 0.498 [0.464, 0.534] |
| History | pre_bowl | 5475 | 0.007 [-0.066, 0.079] | 0.498 [0.462, 0.532] |
| Context | pre_bowl | 5475 | 0.004 [-0.069, 0.078] | 0.498 [0.460, 0.535] |

## Strong preseason favorites

A destination game is a strong preseason favorite case when the focal team's issue 201 preseason-only win probability is at least 0.80. This fixed threshold uses no in-season outcome. Each arm's rows are classified independently from its own preseason forecast.

| Prior family | Policy | Stratum | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |
| --- | --- | --- | ---: | ---: | ---: |
| History | existing | strong_preseason_favorite | 594 | 0.071 [-0.131, 0.267] | 0.524 [0.411, 0.627] |
| History | existing | other | 4881 | -0.000 [-0.079, 0.070] | 0.494 [0.458, 0.530] |
| Context | existing | strong_preseason_favorite | 670 | 0.044 [-0.167, 0.240] | 0.524 [0.421, 0.623] |
| Context | existing | other | 4805 | -0.001 [-0.073, 0.074] | 0.494 [0.460, 0.532] |
| History | pre_bowl | strong_preseason_favorite | 600 | 0.072 [-0.157, 0.276] | 0.527 [0.423, 0.631] |
| History | pre_bowl | other | 4875 | -0.001 [-0.086, 0.079] | 0.494 [0.461, 0.529] |
| Context | pre_bowl | strong_preseason_favorite | 668 | 0.039 [-0.166, 0.245] | 0.519 [0.403, 0.621] |
| Context | pre_bowl | other | 4807 | -0.000 [-0.073, 0.077] | 0.495 [0.458, 0.532] |

The next table reports the paired stratum difference (strong favorite minus other). The difference intervals use the same team and matchup resamples for both strata.

| Prior family | Policy | Metric | Difference [95% interval] |
| --- | --- | --- | ---: |
| History | existing | lag1_correlation | 0.071 [-0.170, 0.281] |
| History | existing | directional_persistence | 0.029 [-0.078, 0.138] |
| Context | existing | lag1_correlation | 0.045 [-0.183, 0.252] |
| Context | existing | directional_persistence | 0.029 [-0.077, 0.134] |
| History | pre_bowl | lag1_correlation | 0.073 [-0.138, 0.302] |
| History | pre_bowl | directional_persistence | 0.033 [-0.078, 0.143] |
| Context | pre_bowl | lag1_correlation | 0.039 [-0.175, 0.268] |
| Context | pre_bowl | directional_persistence | 0.024 [-0.077, 0.134] |

## Change over the season

Destination team-game ordinals 2–4 are early, 5–8 are middle, and 9 or later are late. Each transition is assigned according to the next game's ordinal.

| Prior family | Policy | Season progress | Eligible pairs | Lag-1 r [95% interval] | Same-direction rate [95% interval] |
| --- | --- | --- | ---: | ---: | ---: |
| History | existing | Early (games 2–4) | 1466 | 0.019 [-0.118, 0.150] | 0.489 [0.424, 0.553] |
| History | existing | Middle (games 5–8) | 2041 | 0.003 [-0.109, 0.126] | 0.504 [0.449, 0.565] |
| History | existing | Late (game 9+) | 1968 | 0.003 [-0.124, 0.119] | 0.497 [0.438, 0.550] |
| Context | existing | Early (games 2–4) | 1466 | 0.005 [-0.133, 0.140] | 0.508 [0.447, 0.573] |
| Context | existing | Middle (games 5–8) | 2041 | 0.004 [-0.121, 0.115] | 0.494 [0.435, 0.547] |
| Context | existing | Late (game 9+) | 1968 | 0.005 [-0.112, 0.119] | 0.494 [0.431, 0.552] |
| History | pre_bowl | Early (games 2–4) | 1466 | 0.018 [-0.117, 0.154] | 0.490 [0.422, 0.554] |
| History | pre_bowl | Middle (games 5–8) | 2041 | 0.002 [-0.110, 0.126] | 0.502 [0.448, 0.559] |
| History | pre_bowl | Late (game 9+) | 1968 | 0.002 [-0.111, 0.120] | 0.498 [0.436, 0.555] |
| Context | pre_bowl | Early (games 2–4) | 1466 | 0.004 [-0.130, 0.142] | 0.504 [0.434, 0.571] |
| Context | pre_bowl | Middle (games 5–8) | 2041 | 0.004 [-0.123, 0.121] | 0.494 [0.435, 0.549] |
| Context | pre_bowl | Late (game 9+) | 1968 | 0.004 [-0.117, 0.126] | 0.498 [0.441, 0.558] |

Paired late-minus-early differences (late minus early):

| Prior family | Policy | Metric | Difference [95% interval] |
| --- | --- | --- | ---: |
| History | existing | lag1_correlation | -0.016 [-0.195, 0.160] |
| History | existing | directional_persistence | 0.008 [-0.082, 0.094] |
| Context | existing | lag1_correlation | -0.000 [-0.176, 0.174] |
| Context | existing | directional_persistence | -0.014 [-0.101, 0.073] |
| History | pre_bowl | lag1_correlation | -0.015 [-0.193, 0.163] |
| History | pre_bowl | directional_persistence | 0.008 [-0.085, 0.099] |
| Context | pre_bowl | lag1_correlation | 0.000 [-0.172, 0.173] |
| Context | pre_bowl | directional_persistence | -0.006 [-0.091, 0.077] |

Read sparse late-season rows with their interval widths.

## Interpretation and limits

The existing #201 forecasts already account for opponent strength and game site. A positive sequential association therefore describes residual errors that remain after the earlier game was available to the later posterior. It does not identify whether a specific information source, likelihood feature, or changing team state caused the association.

This differs from issue #34: that study examined persistence in historical performance/rank residuals and recency-style probes. Issue 246 measures errors in leakage-safe pregame game-margin forecasts, then requires the preceding game to appear in the next forecast's evidence. No rankings or in-sample fitted-game residuals are substituted here.

The four arms are shown separately: History and Context, each under the existing and pre-bowl policies. Policies are not pooled as independent replications. Repeated team-game pairs and shared matchup orientations are handled in the bootstrap, but the underlying four seasons remain previously studied exploratory data. This report supports a bounded decision about posterior responsiveness; it does not justify a model change or parameter tuning.

## Reproducibility

Run from the repository root with:

```bash
uv run python scripts/run_issue_246_error_persistence.py
```

The runner reads only the retained issue 201 score, summary, and replay-evidence artifacts; it verifies their stored hashes and reproduces issue 201's primary per-season CRPS and margin NLL means before calculating sequential errors. `team_game_errors.csv` and `lagged_error_pairs.csv` retain the auditable observations and eligibility decisions. The complete input and output hashes are in `manifest.json`.

## Source parity

| Arm-season rows checked | Maximum absolute difference | Status |
| ---: | ---: | --- |
| 16 | 1.78e-15 | passed |
