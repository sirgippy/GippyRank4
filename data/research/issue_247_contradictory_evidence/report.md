# Issue 247: accumulated contradictory evidence and subsequent forecast errors

## Finding

This analysis asks whether the mean of a team's evidence-available prior-game surprises predicts the signed error in its next forecast, after those results were incorporated. The 80% interval half-width is the uncertainty scale; signed error divided by that half-width is the standardized surprise. The primary outcome remains signed margin error in points.

The eligible panel contains **9,682 team-game observations** across **534 team-seasons** in the existing-policy History and Context arms. At least 3 eligible prior games are required. All 12,962 oriented target team-games, including those below the threshold, are retained in `team_game_observations.csv`.

The retained #201 score summary passed parity for **16 arm-season checks**; the maximum absolute CRPS or margin-NLL difference was 1.78e-15. Input artifacts are hash-verified by the reused #246 readers. The four seasons remain exploratory historical data, not independent confirmation.

History's adjusted pooled margin-error slope was -1.513 [-4.546, 1.096] points per surprise unit; its 95% interval includes zero.
Context's adjusted pooled margin-error slope was -1.879 [-4.905, 0.836] points per surprise unit; its 95% interval includes zero.
Neither prior family shows a detectable adjusted pooled association between accumulated surprise and next-game margin error at this resolution. The point estimates are negative, but their intervals also allow a positive association.
The continuous preseason-strength interaction intervals include zero in both families, so this panel does not show a clear stronger or weaker surprise/error relationship for stronger preseason expectations.
The sustained-negative versus neutral, sustained-positive versus neutral, and sustained-positive versus sustained-negative contrasts all have intervals including zero in both families.

## Pooled adjusted association

OLS adjusts for continuous preseason rank strength and uncertainty, their interaction with accumulated surprise, eligible prior-game count, target team-game number, and season indicators. The surprise slope is the expected change in the next signed margin error (points) per one-unit increase in mean prior surprise, evaluated at mean preseason strength. The secondary outcome divides the next error by its own 80% interval half-width. Intervals are 95% multiway team/target-game bootstrap percentile intervals.

| Prior family | Observations | Surprise slope, points [95% interval] | Surprise slope, interval-width units [95% interval] | Surprise × preseason strength, points [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| History | 4841 | -1.513 [-4.546, 1.096] | -0.071 [-0.221, 0.058] | 0.874 [-1.686, 3.333] |
| Context | 4841 | -1.879 [-4.905, 0.836] | -0.089 [-0.242, 0.046] | 0.588 [-2.018, 3.068] |

## Preseason expectation and underperformance

Preseason strength is the continuous reverse expected-rank percentile from the retained #200 distribution (higher means stronger expectation); uncertainty is rank SD divided by population minus one. The interaction term above tests whether the surprise/error relationship changes with preseason strength.

The model-implied next-error shift compares mean prior surprise -1 with 0, holding the included covariates fixed at their reference values. Strength is set to the observed 25th and 75th percentiles. Negative shifts mean worse-than-forecast future performance after a one-unit sustained negative history.

| Prior family | Strength percentile | Shift in next error, points [95% interval] | Shift in standardized next error [95% interval] |
| --- | ---: | ---: | ---: |
| History | 0.344 | 2.204 [-1.125, 6.264] | 0.106 [-0.064, 0.304] |
| History | 0.646 | 0.882 [-2.157, 4.310] | 0.040 [-0.110, 0.214] |
| History | P75 minus P25 | -1.322 [-5.045, 2.552] | — |
| Context | 0.323 | 2.359 [-1.145, 5.994] | 0.113 [-0.064, 0.291] |
| Context | 0.680 | 1.422 [-1.929, 4.949] | 0.067 [-0.104, 0.245] |
| Context | P75 minus P25 | -0.937 [-4.891, 3.218] | — |

## Frozen surprise groups

Sustained negative means mean prior standardized surprise ≤ -0.5; sustained positive means ≥ +0.5; values between are approximately neutral. Group boundaries were fixed before comparing them with future errors.

| Prior family | Prior history | Observations | Mean next error, points [95% interval] | Mean standardized next error [95% interval] |
| --- | --- | ---: | ---: | ---: |
| History | Sustained negative | 193 | -0.575 [-4.166, 3.489] | -0.030 [-0.206, 0.168] |
| History | Approximately neutral | 4400 | -0.020 [-0.685, 0.619] | -0.001 [-0.034, 0.031] |
| History | Sustained positive | 248 | 0.678 [-3.324, 4.254] | 0.037 [-0.159, 0.215] |
| Context | Sustained negative | 167 | 1.202 [-2.977, 5.143] | 0.056 [-0.150, 0.252] |
| Context | Approximately neutral | 4446 | -0.038 [-0.717, 0.633] | -0.002 [-0.035, 0.032] |
| Context | Sustained positive | 228 | -0.302 [-4.158, 3.722] | -0.012 [-0.204, 0.193] |

| Prior family | Contrast | Difference in next error, points [95% interval] | Difference in standardized error [95% interval] |
| --- | --- | ---: | ---: |
| History | Negative minus neutral | -0.554 [-4.226, 3.597] | -0.029 [-0.211, 0.176] |
| History | Positive minus neutral | 0.699 [-3.368, 4.359] | 0.038 [-0.162, 0.223] |
| History | Positive minus negative | 1.253 [-4.413, 6.062] | 0.067 [-0.223, 0.303] |
| Context | Negative minus neutral | 1.240 [-2.919, 5.366] | 0.058 [-0.152, 0.263] |
| Context | Positive minus neutral | -0.264 [-4.320, 3.889] | -0.010 [-0.209, 0.198] |
| Context | Positive minus negative | -1.504 [-6.919, 4.658] | -0.068 [-0.336, 0.244] |

## Season stability

Season-specific surprise slopes use the same adjustment for prior-game count and season progress, without season indicators. These checks describe stability across the four exploratory years.

| Season | Prior family | Eligible observations | Surprise slope, points [95% interval] | Surprise slope, interval-width units [95% interval] |
| ---: | --- | ---: | ---: | ---: |
| 2022 | History | 1183 | -3.344 [-8.807, 1.007] | -0.166 [-0.437, 0.050] |
| 2022 | Context | 1183 | -3.860 [-9.223, 0.980] | -0.192 [-0.462, 0.055] |
| 2023 | History | 1211 | -2.978 [-9.625, 2.612] | -0.141 [-0.472, 0.134] |
| 2023 | Context | 1211 | -3.733 [-10.559, 2.150] | -0.176 [-0.512, 0.118] |
| 2024 | History | 1216 | 0.802 [-3.970, 5.471] | 0.039 [-0.197, 0.274] |
| 2024 | Context | 1216 | 0.579 [-4.880, 5.164] | 0.028 [-0.239, 0.249] |
| 2025 | History | 1231 | -1.813 [-8.126, 3.155] | -0.080 [-0.396, 0.166] |
| 2025 | Context | 1231 | -1.706 [-7.655, 3.439] | -0.076 [-0.378, 0.180] |

| Season | Prior family | All target games | At least three eligible prior games |
| ---: | --- | ---: | ---: |
| 2022 | History | 1588 | 1183 |
| 2022 | Context | 1588 | 1183 |
| 2023 | History | 1618 | 1211 |
| 2023 | Context | 1618 | 1211 |
| 2024 | History | 1625 | 1216 |
| 2024 | Context | 1625 | 1216 |
| 2025 | History | 1650 | 1231 |
| 2025 | Context | 1650 | 1231 |

## Selected examples

Examples are the most negative and positive accumulated-surprise observations within each prior family, requiring at least 5 eligible prior games. Selection used accumulated surprise only and did not use the target game's prediction error.

| Family | Team / season | History | Prior surprise | Preseason strength | Future game | Predicted margin | Actual margin | Signed error |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| History | Hawai'i / 2022 | Sustained negative (5 games) | -1.136 | 0.401 | San Diego State (401415235) | -17.7 | -2.0 | 15.7 |
| History | Indiana / 2025 | Sustained positive (5 games) | 0.926 | 0.726 | Oregon (401752870) | -5.4 | 10.0 | 15.4 |
| Context | Hawai'i / 2022 | Sustained negative (5 games) | -0.980 | 0.256 | San Diego State (401415235) | -18.5 | -2.0 | 16.5 |
| Context | UConn / 2024 | Sustained positive (5 games) | 0.884 | 0.207 | Temple (401629033) | 23.4 | 9.0 | -14.4 |

## Interpretation and recommendation

Because signed error is observed margin minus predicted margin, positive future errors mean the team outperformed its forecast and negative errors mean it performed worse than forecast. A positive surprise slope is directional persistence; a negative slope is a reversal. The preseason interaction assesses whether that relationship changes continuously with preseason expectations. None of these associations identifies prior overconfidence, likelihood misspecification, posterior inference error, or a latent-rank limitation.

These analyses are observational and exploratory; bootstrap intervals describe dependence in the retained panel, not independent confirmation. A null interval is failure to detect at this resolution, not proof of no predictive bias. Group contrasts are descriptive complements to the adjusted continuous model.

**Recommendation under #244:** Retain existing model behavior; these findings provide no justification for adjusting priors, likelihoods, or inference. Across 2022–2025, accumulated contradictory evidence did not show a detectable systematic association with subsequent directional forecast errors. This weakens the predictive-underreaction hypothesis, but does not establish that posterior responsiveness or probabilistic calibration is satisfactory. Continue Phase 1 with a bounded investigation of posterior movement and uncertainty as game evidence accumulates, including whether the resulting predictive distributions are appropriately calibrated. Defer prior rescaling, likelihood modifications, parameter searches, and other model interventions until diagnostics identify a deficiency worth investigating. The frozen analysis may be revisited after the 2026 season, with 2027 offering an opportunity for untouched confirmation. Neither prospective check is a prerequisite for completing the remaining Phase 1 diagnostics.

## Reproducibility and limits

Reproduce with `uv run python scripts/run_issue_247_contradictory_evidence.py` (1,000 bootstrap repetitions). The output manifest records the protocol, retained #200/#201 inputs, configuration, row counts, and SHA-256 hashes. No 2026 outcome, model refit, parameter search, or production change is used.

Source parity covered 16 #201 arm-season rows with maximum absolute metric difference 1.78e-15.
