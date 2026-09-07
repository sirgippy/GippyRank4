# YPP conditional-likelihood investigation

## Question and conclusion

This research asks whether yards per play adds latent team-quality information beyond score margin, opponent quality, and site in Historical Likelihood V1. The predeclared recommendation is **B**.

The production V1 margin factor is frozen. YPP is evaluated only through the conditional decomposition `p(margin | ranks, site) × p(YPP | margin, ranks, site)`. The independent product is retained as a double-counting diagnostic and is not a promotion candidate.

## Data audit

The research uses the frozen historical modeling corpus for 2003–2025; 2003 has no usable team-game YPP, so the defensible common-data fit starts in 2004. Training is 2004-2017 (2003 has no usable YPP); development is 2018-2021; the final test is 2022-2025 untouched until model selection and fit were complete and was not used for model selection.

CFBD raw `/games/teams` responses contain no direct `yardsPerPlay`, `plays`, `offensivePlays`, or `totalPlays` category in the cached corpus. The derivation remains `plays = rushingAttempts + pass attempts parsed from completionAttempts`, then `totalYards / plays`. On 42398 raw/processed common team-game rows, derivation mismatches were 0.

The raw audit found 4280 exact duplicate team-game rows from overlapping classification/week queries and 0 conflicting duplicates. The schedule audit found 2379 exact overlaps and 0 conflicts.

Usable YPP ranges from -0.868 to 13.943; negative values=1, zero values=48, values below 1=69, values above 12=26. Coverage changes materially over time: True.

Representative raw payloads contain the expected `rushingAttempts`, `completionAttempts`, `totalYards`, and `sacks` fields. No direct YPP/plays value was available to compare. In the stored 2024 Tennessee–Arkansas example, CFBD gives 36 rush attempts + 29 pass attempts = 65 plays for Tennessee and 44 + 30 = 74 for Arkansas; the official team notes report total offensive plays of 65 and 74 while listing sacks separately. This supports the current NCAA-style attempt semantics for this audit, without changing the corpus derivation. Because CFBD did not expose a source-wide official-play field in these payloads, this one official cross-check cannot establish a season-wide sacks mismatch rate; that remains a follow-up data-quality check if a direct play field becomes available.

The current corpus builder has an operational sidecar hazard: a broad `game_stats/*.json` glob will see `.provenance.json` objects if a refreshed corpus is present. The audit excluded sidecars and did not rewrite or silently correct the stored corpus. This does not create a YPP-value mismatch in the common rows above, but it should be fixed in a separate acquisition-maintenance change.

CFBD API schema reference: https://apinext.collegefootballdata.com/api/games. Official attempt/play cross-check: https://utsports.com/documents/download/2024/11/4/G9_UT_Notes_MSU.pdf.

### Coverage by era

| Season | Pairing | Games | Both YPP | Team-row coverage |
|---:|:---|---:|---:|---:|
| 2004 | all | 1235 | 699 | 56.6% |
| 2005 | all | 1246 | 708 | 56.8% |
| 2006 | all | 1317 | 787 | 59.8% |
| 2007 | all | 1376 | 787 | 57.4% |
| 2008 | all | 1384 | 801 | 58.0% |
| 2009 | all | 1394 | 808 | 58.0% |
| 2010 | all | 1391 | 808 | 58.1% |
| 2011 | all | 1419 | 805 | 56.7% |
| 2012 | all | 1338 | 835 | 62.5% |
| 2013 | all | 1444 | 855 | 59.2% |
| 2014 | all | 1507 | 867 | 57.5% |
| 2015 | all | 1481 | 870 | 58.7% |
| 2016 | all | 1493 | 872 | 58.4% |
| 2017 | all | 1503 | 874 | 58.2% |
| 2018 | all | 1495 | 883 | 59.1% |
| 2019 | all | 1554 | 888 | 57.1% |
| 2020 | all | 829 | 568 | 68.5% |
| 2021 | all | 1522 | 887 | 58.3% |
| 2022 | all | 1541 | 1524 | 98.9% |
| 2023 | all | 1545 | 1529 | 99.0% |
| 2024 | all | 1611 | 1605 | 99.6% |
| 2025 | all | 1625 | 1619 | 99.6% |

## Conditional signal

Y1 uses only the fixed margin/site/pairing basis; Y2 adds the predeclared rank-percentile contrast basis. Student-t degrees of freedom were selected on 2018–2021 equal-game marginalized YPP NLL only. Selected values: `{"naive": {"development_marginalized_nll": 1.9718990352633612, "final_fit_seasons": "2004-2021", "include_margin": false, "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0}, "y1": {"development_marginalized_nll": 1.6854220894622056, "final_fit_seasons": "2004-2021", "include_margin": true, "rank_signal": false, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0}, "y2": {"development_marginalized_nll": 1.6824966624833098, "final_fit_seasons": "2004-2021", "include_margin": true, "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0}}`.

The residual is observed YPP differential minus the Y1 conditional mean. Positive residual means the V1- oriented side produced more YPP than its margin/site/pairing relationship predicted. `conditional_signal.csv` reports margin and residual bins; `temporal_stability.csv` reports season/pairing effects.

| Period | Season | Pairing | Site | N | Quality slope | Quality Spearman | Next-game slope |
|:---|---:|:---|:---|---:|---:|---:|---:|
| development_2018_2021 | 2018 | fbs-fbs | home_site | 710 | 0.026 | 0.103 | 0.182 |
| development_2018_2021 | 2018 | fbs-fbs | neutral | 61 | 0.027 | 0.200 | -6.209 |
| development_2018_2021 | 2018 | fbs-fcs | home_site | 111 | 0.036 | 0.107 | 0.919 |
| development_2018_2021 | 2018 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| development_2018_2021 | 2019 | fbs-fbs | home_site | 715 | 0.052 | 0.168 | 3.164 |
| development_2018_2021 | 2019 | fbs-fbs | neutral | 59 | 0.002 | 0.007 | -4.407 |
| development_2018_2021 | 2019 | fbs-fcs | home_site | 114 | 0.037 | 0.129 | 2.586 |
| development_2018_2021 | 2020 | fbs-fbs | home_site | 499 | 0.038 | 0.137 | 0.694 |
| development_2018_2021 | 2020 | fbs-fbs | neutral | 35 | 0.023 | 0.108 | 1.024 |
| development_2018_2021 | 2020 | fbs-fcs | home_site | 34 | 0.037 | 0.052 | 4.918 |
| development_2018_2021 | 2021 | fbs-fbs | home_site | 712 | 0.052 | 0.163 | 3.028 |
| development_2018_2021 | 2021 | fbs-fbs | neutral | 58 | 0.017 | 0.052 | -0.557 |
| development_2018_2021 | 2021 | fbs-fcs | home_site | 116 | 0.012 | 0.026 | 2.230 |
| development_2018_2021 | 2021 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2022 | fbs-fbs | home_site | 714 | 0.031 | 0.105 | 1.873 |
| final_2022_2025 | 2022 | fbs-fbs | neutral | 61 | 0.014 | 0.091 | 5.172 |
| final_2022_2025 | 2022 | fbs-fcs | home_site | 119 | 0.045 | 0.146 | 3.655 |
| final_2022_2025 | 2022 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2022 | fcs-fcs | home_site | 611 | -0.008 | -0.677 | -0.292 |
| final_2022_2025 | 2022 | fcs-fcs | neutral | 18 | -0.007 | -0.540 | 0.224 |
| final_2022_2025 | 2023 | fbs-fbs | home_site | 730 | 0.049 | 0.162 | 2.452 |
| final_2022_2025 | 2023 | fbs-fbs | neutral | 62 | 0.033 | 0.045 | 1.062 |
| final_2022_2025 | 2023 | fbs-fcs | home_site | 117 | 0.032 | 0.139 | -0.044 |
| final_2022_2025 | 2023 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2023 | fcs-fcs | home_site | 601 | -0.008 | -0.643 | -0.205 |
| final_2022_2025 | 2023 | fcs-fcs | neutral | 18 | -0.010 | -0.579 | 0.140 |
| final_2022_2025 | 2024 | fbs-fbs | home_site | 722 | 0.045 | 0.145 | 1.553 |
| final_2022_2025 | 2024 | fbs-fbs | neutral | 76 | 0.043 | 0.246 | 0.563 |
| final_2022_2025 | 2024 | fbs-fcs | home_site | 117 | 0.047 | 0.141 | 0.306 |
| final_2022_2025 | 2024 | fbs-fcs | neutral | 4 | -0.472 | -0.200 | -31.204 |
| final_2022_2025 | 2024 | fcs-fcs | home_site | 669 | -0.008 | -0.611 | -0.275 |
| final_2022_2025 | 2024 | fcs-fcs | neutral | 17 | -0.011 | -0.770 | -0.453 |
| final_2022_2025 | 2025 | fbs-fbs | home_site | 744 | 0.048 | 0.174 | 1.369 |
| final_2022_2025 | 2025 | fbs-fbs | neutral | 64 | 0.020 | 0.090 | 4.611 |
| final_2022_2025 | 2025 | fbs-fcs | home_site | 126 | 0.041 | 0.153 | 0.220 |
| final_2022_2025 | 2025 | fcs-fcs | home_site | 671 | -0.008 | -0.628 | -0.373 |
| final_2022_2025 | 2025 | fcs-fcs | neutral | 14 | -0.007 | -0.811 | -0.017 |
| training_2004_2017 | 2004 | fbs-fbs | home_site | 614 | 0.017 | 0.047 | -0.727 |
| training_2004_2017 | 2004 | fbs-fbs | neutral | 29 | 0.009 | 0.175 | 6.538 |
| training_2004_2017 | 2004 | fbs-fcs | home_site | 34 | 0.024 | 0.178 | 3.389 |
| training_2004_2017 | 2004 | fbs-fcs | neutral | 9 | 0.153 | 0.650 | 6.527 |
| training_2004_2017 | 2005 | fbs-fbs | home_site | 626 | 0.015 | 0.013 | 2.993 |
| training_2004_2017 | 2005 | fbs-fbs | neutral | 28 | 0.010 | -0.014 | 5.353 |
| training_2004_2017 | 2005 | fbs-fcs | home_site | 48 | 0.034 | 0.110 | 4.856 |
| training_2004_2017 | 2005 | fbs-fcs | neutral | 6 | -0.151 | -0.771 | -17.570 |
| training_2004_2017 | 2006 | fbs-fbs | home_site | 678 | 0.027 | 0.099 | 2.557 |
| training_2004_2017 | 2006 | fbs-fbs | neutral | 32 | 0.017 | 0.039 | 4.102 |
| training_2004_2017 | 2006 | fbs-fcs | home_site | 70 | -0.005 | -0.068 | 1.823 |
| training_2004_2017 | 2006 | fbs-fcs | neutral | 7 | -0.057 | -0.179 | 5.496 |
| training_2004_2017 | 2007 | fbs-fbs | home_site | 677 | 0.008 | 0.020 | 1.966 |
| training_2004_2017 | 2007 | fbs-fbs | neutral | 32 | 0.001 | -0.057 | -8.058 |
| training_2004_2017 | 2007 | fbs-fcs | home_site | 61 | -0.088 | -0.251 | -2.317 |
| training_2004_2017 | 2007 | fbs-fcs | neutral | 11 | -0.067 | -0.282 | -0.697 |
| training_2004_2017 | 2008 | fbs-fbs | home_site | 670 | 0.012 | 0.050 | 1.991 |
| training_2004_2017 | 2008 | fbs-fbs | neutral | 47 | 0.009 | 0.058 | -2.329 |
| training_2004_2017 | 2008 | fbs-fcs | home_site | 77 | 0.071 | 0.272 | 6.746 |
| training_2004_2017 | 2008 | fbs-fcs | neutral | 7 | 0.015 | 0.000 | 6.417 |
| training_2004_2017 | 2009 | fbs-fbs | home_site | 667 | 0.032 | 0.094 | 0.648 |
| training_2004_2017 | 2009 | fbs-fbs | neutral | 47 | -0.005 | -0.077 | 1.157 |
| training_2004_2017 | 2009 | fbs-fcs | home_site | 84 | 0.020 | 0.076 | 2.664 |
| training_2004_2017 | 2009 | fbs-fcs | neutral | 10 | -0.203 | -0.309 | -5.509 |
| training_2004_2017 | 2010 | fbs-fbs | home_site | 662 | 0.035 | 0.106 | 2.154 |
| training_2004_2017 | 2010 | fbs-fbs | neutral | 56 | 0.025 | 0.060 | 2.349 |
| training_2004_2017 | 2010 | fbs-fcs | home_site | 83 | 0.024 | 0.025 | 2.269 |
| training_2004_2017 | 2010 | fbs-fcs | neutral | 7 | 0.154 | 0.571 | 31.837 |
| training_2004_2017 | 2011 | fbs-fbs | home_site | 654 | 0.041 | 0.139 | 2.630 |
| training_2004_2017 | 2011 | fbs-fbs | neutral | 54 | -0.019 | -0.168 | 4.426 |
| training_2004_2017 | 2011 | fbs-fcs | home_site | 88 | 0.059 | 0.240 | 2.684 |
| training_2004_2017 | 2011 | fbs-fcs | neutral | 9 | -0.172 | -0.283 | -4.010 |
| training_2004_2017 | 2012 | fbs-fbs | home_site | 679 | 0.026 | 0.072 | 1.164 |
| training_2004_2017 | 2012 | fbs-fbs | neutral | 53 | 0.015 | 0.129 | -0.990 |
| training_2004_2017 | 2012 | fbs-fcs | home_site | 97 | 0.049 | 0.167 | 3.085 |
| training_2004_2017 | 2012 | fbs-fcs | neutral | 6 | -0.295 | -0.371 | -38.608 |
| training_2004_2017 | 2013 | fbs-fbs | home_site | 685 | 0.039 | 0.123 | 2.576 |
| training_2004_2017 | 2013 | fbs-fbs | neutral | 52 | 0.003 | 0.078 | 12.702 |
| training_2004_2017 | 2013 | fbs-fcs | home_site | 101 | 0.048 | 0.165 | 0.101 |
| training_2004_2017 | 2013 | fbs-fcs | neutral | 10 | 0.017 | -0.091 | 4.867 |
| training_2004_2017 | 2013 | fcs-fcs | home_site | 1 | n/a | n/a | n/a |
| training_2004_2017 | 2013 | fcs-fcs | neutral | 6 | -0.088 | -0.486 | -6.076 |
| training_2004_2017 | 2014 | fbs-fbs | home_site | 699 | 0.024 | 0.085 | 1.484 |
| training_2004_2017 | 2014 | fbs-fbs | neutral | 61 | -0.027 | -0.144 | 2.665 |
| training_2004_2017 | 2014 | fbs-fcs | home_site | 100 | 0.070 | 0.251 | 3.801 |
| training_2004_2017 | 2014 | fbs-fcs | neutral | 7 | 0.061 | -0.071 | 8.599 |
| training_2004_2017 | 2015 | fbs-fbs | home_site | 704 | 0.034 | 0.108 | 1.177 |
| training_2004_2017 | 2015 | fbs-fbs | neutral | 61 | 0.027 | 0.113 | -2.925 |
| training_2004_2017 | 2015 | fbs-fcs | home_site | 98 | 0.045 | 0.154 | 0.492 |
| training_2004_2017 | 2015 | fbs-fcs | neutral | 7 | -0.070 | -0.286 | -2.132 |
| training_2004_2017 | 2016 | fbs-fbs | home_site | 693 | 0.017 | 0.056 | 0.926 |
| training_2004_2017 | 2016 | fbs-fbs | neutral | 66 | -0.002 | -0.009 | 0.508 |
| training_2004_2017 | 2016 | fbs-fcs | home_site | 111 | 0.036 | 0.104 | 1.394 |
| training_2004_2017 | 2016 | fbs-fcs | neutral | 2 | n/a | n/a | n/a |
| training_2004_2017 | 2017 | fbs-fbs | home_site | 719 | 0.036 | 0.109 | 2.058 |
| training_2004_2017 | 2017 | fbs-fbs | neutral | 57 | 0.015 | 0.013 | 2.099 |
| training_2004_2017 | 2017 | fbs-fcs | home_site | 92 | 0.059 | 0.207 | 7.894 |
| training_2004_2017 | 2017 | fbs-fcs | neutral | 6 | 0.004 | -0.143 | -6.712 |

The descriptive relationship is the decision-relevant evidence: inspect the final-period slopes and correlations rather than treating the extra-variable YPP NLL as comparable with margin-only NLL. A positive quality slope means higher conditional YPP residual was associated with better eventual oriented rank percentile.

## Candidate selection and evaluation

The table below averages the final-cutoff row for each of the four final-test seasons under each prior family. `candidate_metrics.csv` retains every matched cutoff row.

| Candidate | View | NLL | CRPS | Expected-rank MAE | 80% coverage | 80% width | Top-10 Brier | Entropy |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| v1 | full | 3.876 | 0.022 | 8.171 | 0.782 | 29.0 | 0.009 | 3.7 |
| y1 | full | 3.876 | 0.022 | 8.171 | 0.782 | 29.0 | 0.009 | 3.7 |
| y2 | full | 3.955 | 0.026 | 8.862 | 0.753 | 28.3 | 0.009 | 3.7 |
| naive_independent_diagnostic | full | 4.213 | 0.030 | 9.462 | 0.657 | 23.1 | 0.010 | 3.5 |
| v1 | ypp_observed | 3.876 | 0.022 | 8.172 | 0.782 | 29.0 | 0.009 | 3.7 |
| y1 | ypp_observed | 3.876 | 0.022 | 8.172 | 0.782 | 29.0 | 0.009 | 3.7 |
| y2 | ypp_observed | 3.955 | 0.026 | 8.863 | 0.753 | 28.3 | 0.009 | 3.7 |

The primary rank comparison is final-rank quality, with identical priors, cutoffs, rank supports, outcome targets, and comparison keys within each population view. Full production-style keeps every eligible game and gives missing-YPP games exactly V1 margin evidence. The YPP-observed view restricts both V1 and YPP candidates to the same games where YPP could contribute; for runtime control, it is reported at each season's final cutoff, while the full view includes all seven standard cutoffs.

### Final test deltas by season

| Prior | Season | View | Δ NLL Y2−V1 | Δ CRPS | Δ 80% coverage | Δ width |
|:---|---:|:---|---:|---:|---:|---:|
| context | 2022 | full | 0.101 | 0.005 | -0.040 | -0.7 |
| context | 2023 | full | 0.096 | 0.004 | -0.031 | -0.8 |
| context | 2024 | full | 0.066 | 0.003 | -0.018 | -0.4 |
| context | 2025 | full | 0.120 | 0.005 | -0.041 | -0.8 |
| history | 2022 | full | 0.075 | 0.004 | -0.034 | -0.8 |
| history | 2023 | full | 0.070 | 0.004 | -0.027 | -0.7 |
| history | 2024 | full | 0.036 | 0.002 | -0.007 | -0.5 |
| history | 2025 | full | 0.071 | 0.004 | -0.034 | -1.1 |

Y1 is a semantic null: its factor is rank-invariant, and the posterior rows match V1 up to the deterministic alias used by the research evaluator. The naïve independent diagnostic is shown in `candidate_metrics.csv` only at final full cutoffs; any sharper posterior without commensurate rank scores is double-counting warning evidence.

### Future-game check

Future games are scored with the same frozen V1 margin density from each cutoff posterior. The YPP factor is not used to score future margins; it only changes the state estimate.

| Prior | Candidate | Future games | Margin MAE | Win Brier | Margin NLL |
|:---|:---|---:|---:|---:|---:|
| context | v1 | 17284 | 13.109 | 0.202 | 4.235 |
| context | y1 | 17284 | 13.109 | 0.202 | 4.235 |
| context | y2 | 17284 | 13.169 | 0.204 | 4.237 |
| history | v1 | 17284 | 13.180 | 0.204 | 4.239 |
| history | y1 | 17284 | 13.180 | 0.204 | 4.239 |
| history | y2 | 17284 | 13.236 | 0.205 | 4.242 |

## Margin/YPP disagreement games

The examples below are real corpus games. `v1_margin_predictive_nll` is the pre-game V1 predictive evidence; quality shifts are oriented percentile shifts from a context-prior local update; negative YPP-augmented effect means the YPP factor moves the V1-oriented side toward a better latent rank relative to V1 alone.

| Type | Season | Game | Teams | Score | Margin | YPP diff | V1 NLL | V1 shift | YPP effect |
|:---|---:|---:|:---|:---|---:|---:|---:|---:|---:|
| large_win_with_nonpositive_ypp | 2025 | 401752818 | Rutgers–Miami (OH) | 45–17 | 28 | -2.710 | 4.138 | -0.173 | 0.067 |
| large_win_with_nonpositive_ypp | 2024 | 401729780 | Montana State–Idaho | 52–19 | 33 | -1.089 | 4.534 | -0.030 | -0.051 |
| large_win_with_nonpositive_ypp | 2025 | 401756942 | Utah–Cincinnati | 45–14 | 31 | -1.063 | 4.378 | -0.061 | 0.045 |
| close_loss_with_strong_positive_ypp | 2022 | 401424403 | Monmouth–Fordham | 49–52 | -3 | 3.742 | 3.981 | -0.027 | 0.080 |
| close_loss_with_strong_positive_ypp | 2023 | 401540237 | UC Davis–Eastern Washington | 24–27 | -3 | 3.613 | 3.916 | 0.080 | -0.029 |
| close_loss_with_strong_positive_ypp | 2024 | 401636379 | Florida A&M–Mississippi Valley State | 21–24 | -3 | 3.566 | 5.958 | 0.055 | 0.046 |
| close_win_with_strong_negative_ypp | 2025 | 401761648 | Louisiana–Texas State | 42–39 | 3 | -4.402 | 3.710 | -0.020 | 0.019 |
| close_win_with_strong_negative_ypp | 2025 | 401767343 | Sacred Heart–Delaware State | 35–31 | 4 | -3.490 | 3.910 | -0.081 | 0.032 |
| close_win_with_strong_negative_ypp | 2022 | 401420815 | North Dakota–Northern Iowa | 29–27 | 2 | -3.125 | 3.958 | 0.012 | 0.017 |

## Temporal and subdivision stability

The fixed pre-test model is evaluated by season and pairing; no adaptive era weighting, NIL break, or post-test refit is introduced. The result should be read separately for FBS/FBS, FBS/FCS, and FCS/FCS. Sparse or unstable cross-subdivision effects are not promoted by assumption.

Before reading 2022–2025, the promotion gate was fixed at: aggregate final-rank NLL improvement of at least 0.02 nats/team; CRPS degradation no greater than 0.002; 80% coverage drop no greater than 0.03; improvement in at least 3 of 4 held-out seasons for both prior families; no single-season NLL degradation above 0.10 or CRPS degradation above 0.02; future margin MAE degradation no greater than 0.50 points and NLL degradation no greater than 0.02; and positive common-subset quality slope in at least 3 seasons.

The promotion assessment is `{"aggregate_delta_crps": 0.003750880800920692, "aggregate_delta_interval_80_coverage": -0.029148922515876197, "aggregate_delta_nll": 0.07935666793250878, "checks": {"aggregate_crps": false, "aggregate_nll": false, "calibration": true, "common_subset_signal": true, "future_direction": true, "majority_seasons": false, "no_catastrophic_season": false}, "common_subset_positive_quality_slope_seasons": 4, "mean_future_delta_margin_mae": -0.004976520170987708, "mean_future_delta_margin_nll": -6.232186047483435e-05, "recommendation": "B", "seasons_with_nll_improvement": {"context": 0, "history": 0}, "worst_single_season_delta_crps": 0.0050577554717176135, "worst_single_season_delta_nll": 0.11998392059485141}`.

## Production feasibility (not productionized here)

The lightweight weekly updater currently fetches only two `/games` schedule responses for the season and deliberately does not fetch `/games/teams`. The historical builder has 764 cached team-stat response files across the corpus; the cached 2026 shape contains 27 regular-season `/games/teams` response artifacts ({"fbs": 14, "fcs": 13}). That is the current full-season request estimate if each classification/week response is fetched, with fewer requests for an in-season snapshot. Raw/provenance sidecars, a game-id/team-id join, and exact missing-YPP fallback semantics would be required. Current-season YPP acquisition and weekly snapshot runtime were not changed or claimed production-safe in this PR; a later PR should benchmark request latency, rate limits, raw artifact size, and local factor-construction cost before promotion.

## Frozen production boundary and reproducibility

No Historical Likelihood V1, Posterior V1, H 1.1, C 1.2, Performance V1, weekly publication, or website file is modified. The research script reads frozen artifacts and writes only this investigation directory. The input SHA-256 values and output inventory are in `summary.json`.

The branch is research-only. Recommendation A would mean a future Likelihood V2 productionization PR is justified; B means the residual signal exists but formulation/coverage needs more research; C means YPP does not earn the added game-likelihood complexity.
