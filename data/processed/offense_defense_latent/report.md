# Offense/defense latent-state investigation

Issue #40 is research-only. Historical Likelihood V1, Context/History priors, Posterior V1, Performance V1, weekly publication, and published artifacts were not modified.

## Frozen design

- Training: 2008–2017; development: 2018–2021; evaluation: 2022–2025. 2026 outcomes are excluded.
- Context is primary. Existing Context priors are consumed; missing development priors use the repository's frozen Context fit trained through 2017.
- Stage 0 expected margins use only outcome-free preseason Context PMFs (or uniform cold-start PMFs); historical `rank_pairs` are deliberately excluded because they are end-of-season observations. Stage 1 uses the existing pregame Posterior V1 construction.
- A training-only pairing/site total environment gives expected scores `(T + margin) / 2` and `(T - margin) / 2`, with FBS-first orientation for cross-subdivision games.
- Offensive residual is points scored minus expected points scored. Defensive residual is expected opponent points minus opponent points allowed. Both are reported for both teams in every game.
- OD score means are `environment + O_team - D_opponent`; offense and defense each obey a population sum-to-zero constraint. The candidate score distribution is independent Normal with a training-frozen scale.
- The OD margin is `environment margin + (O + D)_home - (O + D)_away`, so a margin gate mostly tests the scalar sum `O + D`; component separation is additionally diagnosed through score and total predictions.
- Component diagnostics report offense/defense-to-scalar-quality alignment and offense-minus-defense variance. Pace/scoring-environment variation can confound score residuals, so this is not interpreted as a pure possession-level offense/defense decomposition.

## Stage 0

The residual panel contains 52,140 focal-team games and 4,512 team-seasons with at least three games.

| Period | Scale | Relation | Control | Lag | Bin | Pairs | Correlation |
|---|---|---|---|---:|---|---:|---:|
| development | demeaned | defense->defense | observed | 1 | all | 9789 | -0.079323 |
| development | demeaned | defense->offense | observed | 1 | all | 9789 | 0.033495 |
| development | demeaned | offense->defense | observed | 1 | all | 9789 | 0.021120 |
| development | demeaned | offense->offense | observed | 1 | all | 9789 | -0.079342 |
| development | raw | defense->defense | observed | 1 | all | 9789 | 0.185967 |
| development | raw | defense->offense | observed | 1 | all | 9789 | 0.062413 |
| development | raw | offense->defense | observed | 1 | all | 9789 | 0.046211 |
| development | raw | offense->offense | observed | 1 | all | 9789 | 0.178024 |
| evaluation | demeaned | defense->defense | observed | 1 | all | 11594 | -0.065331 |
| evaluation | demeaned | defense->offense | observed | 1 | all | 11594 | -0.001809 |
| evaluation | demeaned | offense->defense | observed | 1 | all | 11594 | -0.003012 |
| evaluation | demeaned | offense->offense | observed | 1 | all | 11594 | -0.067286 |
| evaluation | raw | defense->defense | observed | 1 | all | 11594 | 0.153617 |
| evaluation | raw | defense->offense | observed | 1 | all | 11594 | 0.058993 |
| evaluation | raw | offense->defense | observed | 1 | all | 11594 | 0.051665 |
| evaluation | raw | offense->offense | observed | 1 | all | 11594 | 0.211684 |
| training | demeaned | defense->defense | observed | 1 | all | 26205 | -0.074681 |
| training | demeaned | defense->offense | observed | 1 | all | 26205 | 0.019003 |
| training | demeaned | offense->defense | observed | 1 | all | 26205 | 0.017043 |
| training | demeaned | offense->offense | observed | 1 | all | 26205 | -0.064781 |
| training | raw | defense->defense | observed | 1 | all | 26205 | 0.175001 |
| training | raw | defense->offense | observed | 1 | all | 26205 | 0.057035 |
| training | raw | offense->defense | observed | 1 | all | 26205 | 0.050935 |
| training | raw | offense->offense | observed | 1 | all | 26205 | 0.227759 |
| development | demeaned | defense->defense | shuffle_components | 1 | all | 88097 | 0.004895 |
| development | demeaned | defense->defense | shuffle_order | 1 | all | 97890 | -0.102828 |
| development | demeaned | defense->defense | unrelated_team | 1 | all | 90686 | -0.002489 |
| development | demeaned | defense->offense | shuffle_components | 1 | all | 88097 | -0.000307 |
| development | demeaned | defense->offense | shuffle_order | 1 | all | 97890 | -0.005060 |
| development | demeaned | defense->offense | unrelated_team | 1 | all | 90686 | -0.000135 |
| development | demeaned | offense->defense | shuffle_components | 1 | all | 88097 | -0.005747 |
| development | demeaned | offense->defense | shuffle_order | 1 | all | 97890 | -0.009258 |
| development | demeaned | offense->defense | unrelated_team | 1 | all | 90686 | -0.002971 |
| development | demeaned | offense->offense | shuffle_components | 1 | all | 88097 | -0.004744 |
| development | demeaned | offense->offense | shuffle_order | 1 | all | 97890 | -0.102660 |
| development | demeaned | offense->offense | unrelated_team | 1 | all | 90686 | -0.005022 |
| development | raw | defense->defense | shuffle_components | 1 | all | 88097 | 0.006248 |
| development | raw | defense->defense | shuffle_order | 1 | all | 97890 | 0.165324 |
| development | raw | defense->defense | unrelated_team | 1 | all | 90686 | 0.001053 |
| development | raw | defense->offense | shuffle_components | 1 | all | 88097 | -0.003433 |
| development | raw | defense->offense | shuffle_order | 1 | all | 97890 | 0.026737 |
| development | raw | defense->offense | unrelated_team | 1 | all | 90686 | -0.006207 |
| development | raw | offense->defense | shuffle_components | 1 | all | 88097 | -0.008684 |
| development | raw | offense->defense | shuffle_order | 1 | all | 97890 | 0.031069 |
| development | raw | offense->defense | unrelated_team | 1 | all | 90686 | -0.004851 |
| development | raw | offense->offense | shuffle_components | 1 | all | 88097 | 0.000190 |
| development | raw | offense->offense | shuffle_order | 1 | all | 97890 | 0.156357 |
| development | raw | offense->offense | unrelated_team | 1 | all | 90686 | 0.000761 |
| evaluation | demeaned | defense->defense | shuffle_components | 1 | all | 108017 | 0.001798 |
| evaluation | demeaned | defense->defense | shuffle_order | 1 | all | 115940 | -0.091241 |
| evaluation | demeaned | defense->defense | unrelated_team | 1 | all | 108956 | 0.005344 |
| evaluation | demeaned | defense->offense | shuffle_components | 1 | all | 108017 | 0.003633 |
| evaluation | demeaned | defense->offense | shuffle_order | 1 | all | 115940 | -0.009148 |
| evaluation | demeaned | defense->offense | unrelated_team | 1 | all | 108956 | 0.002230 |
| evaluation | demeaned | offense->defense | shuffle_components | 1 | all | 108017 | -0.003066 |
| evaluation | demeaned | offense->defense | shuffle_order | 1 | all | 115940 | -0.003268 |
| evaluation | demeaned | offense->defense | unrelated_team | 1 | all | 108956 | 0.003940 |
| evaluation | demeaned | offense->offense | shuffle_components | 1 | all | 108017 | -0.003892 |
| evaluation | demeaned | offense->offense | shuffle_order | 1 | all | 115940 | -0.089617 |
| evaluation | demeaned | offense->offense | unrelated_team | 1 | all | 108956 | -0.000999 |
| evaluation | raw | defense->defense | shuffle_components | 1 | all | 108017 | 0.003368 |
| evaluation | raw | defense->defense | shuffle_order | 1 | all | 115940 | 0.127294 |
| evaluation | raw | defense->defense | unrelated_team | 1 | all | 108956 | 0.004026 |
| evaluation | raw | defense->offense | shuffle_components | 1 | all | 108017 | -0.001223 |
| evaluation | raw | defense->offense | shuffle_order | 1 | all | 115940 | 0.048798 |
| evaluation | raw | defense->offense | unrelated_team | 1 | all | 108956 | -0.001519 |
| evaluation | raw | offense->defense | shuffle_components | 1 | all | 108017 | -0.002138 |
| evaluation | raw | offense->defense | shuffle_order | 1 | all | 115940 | 0.048976 |
| evaluation | raw | offense->defense | unrelated_team | 1 | all | 108956 | -0.004694 |
| evaluation | raw | offense->offense | shuffle_components | 1 | all | 108017 | -0.012628 |
| evaluation | raw | offense->offense | shuffle_order | 1 | all | 115940 | 0.186063 |
| evaluation | raw | offense->offense | unrelated_team | 1 | all | 108956 | -0.003498 |
| training | demeaned | defense->defense | shuffle_components | 1 | all | 239565 | -0.001038 |
| training | demeaned | defense->defense | shuffle_order | 1 | all | 262050 | -0.094427 |
| training | demeaned | defense->defense | unrelated_team | 1 | all | 243177 | -0.002640 |
| training | demeaned | defense->offense | shuffle_components | 1 | all | 239565 | -0.002146 |
| training | demeaned | defense->offense | shuffle_order | 1 | all | 262050 | -0.003394 |
| training | demeaned | defense->offense | unrelated_team | 1 | all | 243177 | -0.002352 |
| training | demeaned | offense->defense | shuffle_components | 1 | all | 239565 | -0.001487 |
| training | demeaned | offense->defense | shuffle_order | 1 | all | 262050 | -0.002276 |
| training | demeaned | offense->defense | unrelated_team | 1 | all | 243177 | -0.006973 |
| training | demeaned | offense->offense | shuffle_components | 1 | all | 239565 | -0.001968 |
| training | demeaned | offense->offense | shuffle_order | 1 | all | 262050 | -0.093979 |
| training | demeaned | offense->offense | unrelated_team | 1 | all | 243177 | 0.000096 |
| training | raw | defense->defense | shuffle_components | 1 | all | 239565 | 0.001129 |
| training | raw | defense->defense | shuffle_order | 1 | all | 262050 | 0.158534 |
| training | raw | defense->defense | unrelated_team | 1 | all | 243177 | 0.002977 |
| training | raw | defense->offense | shuffle_components | 1 | all | 239565 | -0.006959 |
| training | raw | defense->offense | shuffle_order | 1 | all | 262050 | 0.036729 |
| training | raw | defense->offense | unrelated_team | 1 | all | 243177 | -0.007253 |
| training | raw | offense->defense | shuffle_components | 1 | all | 239565 | -0.008847 |
| training | raw | offense->defense | shuffle_order | 1 | all | 262050 | 0.034101 |
| training | raw | offense->defense | unrelated_team | 1 | all | 243177 | -0.014941 |
| training | raw | offense->offense | shuffle_components | 1 | all | 239565 | 0.002714 |
| training | raw | offense->offense | shuffle_order | 1 | all | 262050 | 0.201622 |
| training | raw | offense->offense | unrelated_team | 1 | all | 243177 | 0.006871 |

Development-only demeaned lag-1 same-component mean correlation: `-0.07933249529171021`; cross-component mean: `0.02730747741134803`; raw difference: `-0.10663997270305825`.
The Stage 0 decision uses the per-relation observed-minus-shuffle effects, summarized as same-component `0.02341149572054633`, cross-component `0.03446680396999115`, and same-minus-cross excess `-0.011055308249444819`. Unrelated-team effects are reported as a descriptive secondary null because team-season demeaning can make their comparison negative.
The evaluation-period signal is descriptive only. Early-vs-late correlations, lag-2 results, elapsed-time bins, season rows, shuffle-order nulls, and unrelated-team nulls are retained in the CSV artifacts.

Per-relation observed and null-relative effects:

| Relation | Observed | Shuffle null | Unrelated null | Observed - shuffle | Observed - unrelated |
|---|---:|---:|---:|---:|---:|
| offense->offense | -0.079342 | -0.102660 | -0.005022 | +0.023318 | -0.074320 |
| defense->defense | -0.079323 | -0.102828 | -0.002489 | +0.023505 | -0.076835 |
| offense->defense | +0.021120 | -0.009258 | -0.002971 | +0.030378 | +0.024091 |
| defense->offense | +0.033495 | -0.005060 | -0.000135 | +0.038556 | +0.033631 |

## Stage 1

The predeclared family is OD0 (independent component prior) and OD1 (rho=0.25 offense/defense prior correlation), compared with a matched scalar scoreboard control using the same environment, Normal likelihood, prior regularization, and cutoff construction. The development gate requires margin NLL improvement >= 0.010, margin MAE improvement >= 0.10 points, win Brier worsening <= 0.001, improvement in at least 3 of 4 seasons, no season NLL worsening > 0.015, and no worse NLL or MAE than the matched scalar control.
Stage 0 gate passed: **False**. Stage 1 ran: **False**. Selected candidate: **none**. Evaluation triggered: **False**.

| Candidate | NLL improvement | MAE improvement | vs scalar NLL | vs scalar MAE | Brier delta | Improved seasons | Worst season NLL delta | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Recommendation: C

Stage 0 does not show convincing distinct component persistence; Stage 1 was not run.

Production integrity hashes before and after are identical. The research runner writes only `data/processed/offense_defense_latent/` and does not change production artifacts.

Runtime seconds: 56.254.
