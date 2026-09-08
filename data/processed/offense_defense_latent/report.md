# Offense/defense latent-state investigation

Issue #40 is research-only. Historical Likelihood V1, Context/History priors, Posterior V1, Performance V1, weekly publication, and published artifacts were not modified.

## Frozen design

- Training: 2008–2017; development: 2018–2021; evaluation: 2022–2025. 2026 outcomes are excluded.
- Context is primary. Existing Context priors are consumed; missing development priors use the repository's frozen Context fit trained through 2017.
- V1 expected margin is averaged over the historical rank-pair observations. A training-only pairing/site total environment gives expected scores `(T + margin) / 2` and `(T - margin) / 2`, with FBS-first orientation for cross-subdivision games.
- Offensive residual is points scored minus expected points scored. Defensive residual is expected opponent points minus opponent points allowed. Both are reported for both teams in every game.
- OD score means are `environment + O_team - D_opponent`; offense and defense each obey a population sum-to-zero constraint. The candidate score distribution is independent Normal with a training-frozen scale.

## Stage 0

The residual panel contains 52,140 focal-team games and 4,512 team-seasons with at least three games.

| Period | Scale | Relation | Control | Lag | Bin | Pairs | Correlation |
|---|---|---|---|---:|---|---:|---:|
| development | demeaned | defense->defense | observed | 1 | all | 9789 | -0.091572 |
| development | demeaned | defense->offense | observed | 1 | all | 9789 | 0.036316 |
| development | demeaned | offense->defense | observed | 1 | all | 9789 | 0.028841 |
| development | demeaned | offense->offense | observed | 1 | all | 9789 | -0.098559 |
| development | raw | defense->defense | observed | 1 | all | 9789 | 0.083783 |
| development | raw | defense->offense | observed | 1 | all | 9789 | -0.076922 |
| development | raw | offense->defense | observed | 1 | all | 9789 | -0.086625 |
| development | raw | offense->offense | observed | 1 | all | 9789 | 0.089025 |
| evaluation | demeaned | defense->defense | observed | 1 | all | 11594 | -0.072575 |
| evaluation | demeaned | defense->offense | observed | 1 | all | 11594 | 0.004369 |
| evaluation | demeaned | offense->defense | observed | 1 | all | 11594 | 0.011999 |
| evaluation | demeaned | offense->offense | observed | 1 | all | 11594 | -0.065861 |
| evaluation | raw | defense->defense | observed | 1 | all | 11594 | 0.075226 |
| evaluation | raw | defense->offense | observed | 1 | all | 11594 | -0.091593 |
| evaluation | raw | offense->defense | observed | 1 | all | 11594 | -0.088460 |
| evaluation | raw | offense->offense | observed | 1 | all | 11594 | 0.090535 |
| training | demeaned | defense->defense | observed | 1 | all | 26205 | -0.080934 |
| training | demeaned | defense->offense | observed | 1 | all | 26205 | 0.027946 |
| training | demeaned | offense->defense | observed | 1 | all | 26205 | 0.025312 |
| training | demeaned | offense->offense | observed | 1 | all | 26205 | -0.080551 |
| training | raw | defense->defense | observed | 1 | all | 26205 | 0.098247 |
| training | raw | defense->offense | observed | 1 | all | 26205 | -0.098509 |
| training | raw | offense->defense | observed | 1 | all | 26205 | -0.103226 |
| training | raw | offense->offense | observed | 1 | all | 26205 | 0.102073 |
| development | demeaned | defense->defense | shuffle_components | 1 | all | 88097 | 0.006757 |
| development | demeaned | defense->defense | shuffle_order | 1 | all | 97890 | -0.099789 |
| development | demeaned | defense->defense | unrelated_team | 1 | all | 90686 | -0.003237 |
| development | demeaned | defense->offense | shuffle_components | 1 | all | 88097 | -0.000830 |
| development | demeaned | defense->offense | shuffle_order | 1 | all | 97890 | 0.016654 |
| development | demeaned | defense->offense | unrelated_team | 1 | all | 90686 | -0.000842 |
| development | demeaned | offense->defense | shuffle_components | 1 | all | 88097 | -0.004106 |
| development | demeaned | offense->defense | shuffle_order | 1 | all | 97890 | 0.010778 |
| development | demeaned | offense->defense | unrelated_team | 1 | all | 90686 | -0.002409 |
| development | demeaned | offense->offense | shuffle_components | 1 | all | 88097 | -0.004388 |
| development | demeaned | offense->offense | shuffle_order | 1 | all | 97890 | -0.103255 |
| development | demeaned | offense->offense | unrelated_team | 1 | all | 90686 | -0.005321 |
| development | raw | defense->defense | shuffle_components | 1 | all | 88097 | 0.010795 |
| development | raw | defense->defense | shuffle_order | 1 | all | 97890 | 0.077523 |
| development | raw | defense->defense | unrelated_team | 1 | all | 90686 | 0.001956 |
| development | raw | defense->offense | shuffle_components | 1 | all | 88097 | -0.006347 |
| development | raw | defense->offense | shuffle_order | 1 | all | 97890 | -0.097452 |
| development | raw | defense->offense | unrelated_team | 1 | all | 90686 | -0.006096 |
| development | raw | offense->defense | shuffle_components | 1 | all | 88097 | -0.006812 |
| development | raw | offense->defense | shuffle_order | 1 | all | 97890 | -0.092355 |
| development | raw | offense->defense | unrelated_team | 1 | all | 90686 | -0.008753 |
| development | raw | offense->offense | shuffle_components | 1 | all | 88097 | 0.000553 |
| development | raw | offense->offense | shuffle_order | 1 | all | 97890 | 0.081467 |
| development | raw | offense->offense | unrelated_team | 1 | all | 90686 | 0.000208 |
| evaluation | demeaned | defense->defense | shuffle_components | 1 | all | 108017 | 0.003933 |
| evaluation | demeaned | defense->defense | shuffle_order | 1 | all | 115940 | -0.088754 |
| evaluation | demeaned | defense->defense | unrelated_team | 1 | all | 108956 | 0.002178 |
| evaluation | demeaned | defense->offense | shuffle_components | 1 | all | 108017 | 0.001778 |
| evaluation | demeaned | defense->offense | shuffle_order | 1 | all | 115940 | 0.008527 |
| evaluation | demeaned | defense->offense | unrelated_team | 1 | all | 108956 | -0.000788 |
| evaluation | demeaned | offense->defense | shuffle_components | 1 | all | 108017 | -0.000610 |
| evaluation | demeaned | offense->defense | shuffle_order | 1 | all | 115940 | 0.017352 |
| evaluation | demeaned | offense->defense | unrelated_team | 1 | all | 108956 | 0.003139 |
| evaluation | demeaned | offense->offense | shuffle_components | 1 | all | 108017 | -0.005820 |
| evaluation | demeaned | offense->offense | shuffle_order | 1 | all | 115940 | -0.090536 |
| evaluation | demeaned | offense->offense | unrelated_team | 1 | all | 108956 | -0.001795 |
| evaluation | raw | defense->defense | shuffle_components | 1 | all | 108017 | 0.002300 |
| evaluation | raw | defense->defense | shuffle_order | 1 | all | 115940 | 0.055282 |
| evaluation | raw | defense->defense | unrelated_team | 1 | all | 108956 | 0.003830 |
| evaluation | raw | defense->offense | shuffle_components | 1 | all | 108017 | 0.002906 |
| evaluation | raw | defense->offense | shuffle_order | 1 | all | 115940 | -0.083870 |
| evaluation | raw | defense->offense | unrelated_team | 1 | all | 108956 | -0.002999 |
| evaluation | raw | offense->defense | shuffle_components | 1 | all | 108017 | 0.000114 |
| evaluation | raw | offense->defense | shuffle_order | 1 | all | 115940 | -0.085370 |
| evaluation | raw | offense->defense | unrelated_team | 1 | all | 108956 | -0.001362 |
| evaluation | raw | offense->offense | shuffle_components | 1 | all | 108017 | -0.007374 |
| evaluation | raw | offense->offense | shuffle_order | 1 | all | 115940 | 0.067328 |
| evaluation | raw | offense->offense | unrelated_team | 1 | all | 108956 | 0.000948 |
| training | demeaned | defense->defense | shuffle_components | 1 | all | 239565 | 0.001607 |
| training | demeaned | defense->defense | shuffle_order | 1 | all | 262050 | -0.095034 |
| training | demeaned | defense->defense | unrelated_team | 1 | all | 243177 | -0.000594 |
| training | demeaned | defense->offense | shuffle_components | 1 | all | 239565 | -0.001676 |
| training | demeaned | defense->offense | shuffle_order | 1 | all | 262050 | 0.019068 |
| training | demeaned | defense->offense | unrelated_team | 1 | all | 243177 | -0.003334 |
| training | demeaned | offense->defense | shuffle_components | 1 | all | 239565 | 0.000350 |
| training | demeaned | offense->defense | shuffle_order | 1 | all | 262050 | 0.018701 |
| training | demeaned | offense->defense | unrelated_team | 1 | all | 243177 | -0.004218 |
| training | demeaned | offense->offense | shuffle_components | 1 | all | 239565 | -0.002138 |
| training | demeaned | offense->offense | shuffle_order | 1 | all | 262050 | -0.091242 |
| training | demeaned | offense->offense | unrelated_team | 1 | all | 243177 | 0.001239 |
| training | raw | defense->defense | shuffle_components | 1 | all | 239565 | 0.005931 |
| training | raw | defense->defense | shuffle_order | 1 | all | 262050 | 0.088243 |
| training | raw | defense->defense | unrelated_team | 1 | all | 243177 | 0.008031 |
| training | raw | defense->offense | shuffle_components | 1 | all | 239565 | -0.007676 |
| training | raw | defense->offense | shuffle_order | 1 | all | 262050 | -0.106407 |
| training | raw | defense->offense | unrelated_team | 1 | all | 243177 | -0.012135 |
| training | raw | offense->defense | shuffle_components | 1 | all | 239565 | -0.007324 |
| training | raw | offense->defense | shuffle_order | 1 | all | 262050 | -0.109085 |
| training | raw | offense->defense | unrelated_team | 1 | all | 243177 | -0.013443 |
| training | raw | offense->offense | shuffle_components | 1 | all | 239565 | 0.005968 |
| training | raw | offense->offense | shuffle_order | 1 | all | 262050 | 0.090424 |
| training | raw | offense->offense | unrelated_team | 1 | all | 243177 | 0.010210 |

Development/evaluation demeaned lag-1 same-component mean correlation: `-0.08544320128223684`; cross-component mean: `0.020471405116477635`; difference: `-0.10591460639871447`.
Early-vs-late correlations, lag-2 results, elapsed-time bins, season rows, shuffle-order nulls, and unrelated-team nulls are retained in the CSV artifacts.

## Stage 1

The predeclared family is OD0 (independent component prior) and OD1 (rho=0.25 offense/defense prior correlation). The development gate requires margin NLL improvement >= 0.010, margin MAE improvement >= 0.10 points, win Brier worsening <= 0.001, improvement in at least 3 of 4 seasons, and no season NLL worsening > 0.015.
Selected candidate: **none**. Evaluation triggered: **False**.

| Candidate | NLL improvement | MAE improvement | Brier delta | Improved seasons | Worst season NLL delta | Pass |
|---|---:|---:|---:|---:|---:|---|
| OD0 | -0.022727 | -0.148831 | +0.003500 | 1 | +0.051186 | False |
| OD1 | -0.020211 | -0.087707 | +0.002887 | 1 | +0.046603 | False |

## Recommendation: C

Stage 0 does not show convincing distinct component persistence and no OD candidate clears the development gate.

Production integrity hashes before and after are identical. The research runner writes only `data/processed/offense_defense_latent/` and does not change production artifacts.

Runtime seconds: 341.021.
