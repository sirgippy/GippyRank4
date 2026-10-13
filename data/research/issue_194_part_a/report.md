# Issue 194 Part A: historical diagnostics

Part A was selected and run on the existing-policy issue 200 History/Context pair. Part B remains pending. This is an exploratory four-season diagnostic; it does not alter production priors, inference, published snapshots, or website defaults.

## Checkpoint coverage

The analysis uses the 24 safe original checkpoints in #201's frozen-target sensitivity: six per season. The final original checkpoint in each season was excluded for target-safety. These are not a matched December replication.

| Season | Safe checkpoints | Unsafe final checkpoint | Safe phase coverage |
| --- | ---: | --- | --- |
| 2022 | 6 | 2022-12-10T23:59:59+00:00 | other_checkpoint: 4, pre_thanksgiving: 1, rivalry_window: 1 |
| 2023 | 6 | 2023-12-09T23:59:59+00:00 | other_checkpoint: 4, pre_thanksgiving: 1, rivalry_window: 1 |
| 2024 | 6 | 2024-12-14T23:59:59+00:00 | other_checkpoint: 4, pre_thanksgiving: 1, rivalry_window: 1 |
| 2025 | 6 | 2025-12-13T23:59:59+00:00 | other_checkpoint: 4, pre_thanksgiving: 1, rivalry_window: 1 |

Across the 24 safe checkpoints, coverage is 16 `other_checkpoint`, four `pre_thanksgiving`, and four `rivalry_window`. There is no safe `conference_championship_period` checkpoint; the postseason/Army–Navy windows have no matching safe original checkpoint either. The final cutoff was not moved to fill that phase.

## Crossed-message ranking scores

The table averages all teams and six safe checkpoints within each season. NLL and rank CRPS are lower-is-better; target mass is compared with its nominal 0.80 level; interval width and HHI describe spread/concentration. The crossed arms are frozen-message hybrids.

| Season | Arm | Rank NLL | Rank CRPS | 80% target mass | 80% width | HHI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | CC | 4.108214 | 0.039543 | 0.804526 | 43.654 | 0.027030 |
| 2022 | CH | 4.109353 | 0.039671 | 0.803786 | 43.678 | 0.026822 |
| 2022 | HC | 4.132205 | 0.042333 | 0.800475 | 45.107 | 0.026261 |
| 2022 | HH | 4.133853 | 0.042490 | 0.800605 | 45.167 | 0.026049 |
| 2023 | CC | 4.226238 | 0.045074 | 0.781662 | 45.134 | 0.025648 |
| 2023 | CH | 4.212127 | 0.044345 | 0.787650 | 45.211 | 0.025340 |
| 2023 | HC | 4.209562 | 0.044486 | 0.795837 | 47.224 | 0.024214 |
| 2023 | HH | 4.200630 | 0.044064 | 0.797898 | 47.286 | 0.023951 |
| 2024 | CC | 4.202887 | 0.045168 | 0.786535 | 43.761 | 0.027656 |
| 2024 | CH | 4.183655 | 0.044093 | 0.792738 | 44.007 | 0.027139 |
| 2024 | HC | 4.190941 | 0.045716 | 0.792766 | 46.116 | 0.025513 |
| 2024 | HH | 4.177024 | 0.044950 | 0.796331 | 46.335 | 0.025032 |
| 2025 | CC | 4.186951 | 0.042331 | 0.790736 | 43.485 | 0.026521 |
| 2025 | CH | 4.176429 | 0.041619 | 0.796297 | 43.735 | 0.026180 |
| 2025 | HC | 4.168693 | 0.041956 | 0.806937 | 45.420 | 0.025718 |
| 2025 | HH | 4.160272 | 0.041385 | 0.810809 | 45.653 | 0.025379 |

### Crossed-message decomposition

The direct-prior contrast is `CH − HH`, transmitted is `HC − HH`, and interaction is `CC − CH − HC + HH`; all three sum to `CC − HH`. Values below average checkpoint-level team means equally across the six safe checkpoints. These components are descriptive and are not causal shares.

| Season | Metric | Direct prior | Transmitted | Interaction | Native Context − History |
| --- | --- | ---: | ---: | ---: | ---: |
| 2022 | rank NLL | -0.024501 | -0.001648 | +0.000509 | -0.025639 |
| 2022 | rank CRPS | -0.002819 | -0.000157 | +0.000029 | -0.002947 |
| 2022 | 80% target mass | +0.003181 | -0.000130 | +0.000870 | +0.003921 |
| 2022 | 80% interval width | -1.488550 | -0.059796 | +0.035623 | -1.512723 |
| 2022 | HHI concentration | +0.000772 | +0.000212 | -0.000004 | +0.000981 |
| 2023 | rank NLL | +0.011497 | +0.008932 | +0.005180 | +0.025608 |
| 2023 | rank CRPS | +0.000281 | +0.000422 | +0.000307 | +0.001010 |
| 2023 | 80% target mass | -0.010248 | -0.002061 | -0.003926 | -0.016235 |
| 2023 | 80% interval width | -2.075188 | -0.061404 | -0.015038 | -2.151629 |
| 2023 | HHI concentration | +0.001389 | +0.000263 | +0.000046 | +0.001697 |
| 2024 | rank NLL | +0.006631 | +0.013917 | +0.005314 | +0.025863 |
| 2024 | rank CRPS | -0.000857 | +0.000767 | +0.000309 | +0.000218 |
| 2024 | 80% target mass | -0.003593 | -0.003565 | -0.002638 | -0.009797 |
| 2024 | 80% interval width | -2.327114 | -0.218905 | -0.027363 | -2.573383 |
| 2024 | HHI concentration | +0.002107 | +0.000482 | +0.000035 | +0.002624 |
| 2025 | rank NLL | +0.016157 | +0.008421 | +0.002101 | +0.026679 |
| 2025 | rank CRPS | +0.000235 | +0.000572 | +0.000140 | +0.000947 |
| 2025 | 80% target mass | -0.014512 | -0.003872 | -0.001689 | -0.020073 |
| 2025 | 80% interval width | -1.917892 | -0.232843 | -0.017157 | -2.167892 |
| 2025 | HHI concentration | +0.000801 | +0.000338 | +0.000003 | +0.001142 |

`checkpoint_phase_summary.csv` places the ranking measurements alongside #201's retained existing-policy future-game CRPS where a named calendar phase matches. Future-game CRPS measures a different outcome and is contextual, not paired with rank scores. Full season/phase values and game counts are in that file.

### Retained future-game CRPS context

| Season | #201 game phase | Games | History CRPS | Context CRPS | Safe ranking checkpoints in mapped phase |
| --- | --- | ---: | ---: | ---: | ---: |
| 2022 | all_phases | 734 | 9.108537 | 8.996967 | 6 |
| 2022 | army_navy | 1 | 4.474279 | 4.001470 | 0 |
| 2022 | conference_championship_period | 11 | 7.766309 | 7.630610 | 0 |
| 2022 | pre_thanksgiving | 661 | 9.095204 | 8.971137 | 1 |
| 2022 | rivalry_window | 61 | 9.571022 | 9.605147 | 1 |
| 2023 | all_phases | 750 | 8.897336 | 8.844700 | 6 |
| 2023 | army_navy | 1 | 3.640605 | 3.612268 | 0 |
| 2023 | conference_championship_period | 10 | 6.660612 | 6.722846 | 0 |
| 2023 | pre_thanksgiving | 676 | 8.970413 | 8.913151 | 1 |
| 2023 | rivalry_window | 63 | 8.551692 | 8.530070 | 1 |
| 2024 | all_phases | 752 | 9.213743 | 9.098935 | 6 |
| 2024 | army_navy | 1 | 15.647809 | 16.210076 | 0 |
| 2024 | conference_championship_period | 9 | 14.760551 | 15.249923 | 0 |
| 2024 | pre_thanksgiving | 677 | 9.174906 | 9.042046 | 1 |
| 2024 | rivalry_window | 65 | 8.751246 | 8.730371 | 1 |
| 2025 | all_phases | 762 | 8.775371 | 8.695879 | 6 |
| 2025 | army_navy | 1 | 3.608374 | 3.600118 | 0 |
| 2025 | conference_championship_period | 9 | 7.636237 | 7.469590 | 0 |
| 2025 | pre_thanksgiving | 687 | 8.791036 | 8.705611 | 1 |
| 2025 | rivalry_window | 65 | 8.847021 | 8.841212 | 1 |

## Preseason error co-movement

Historical co-movement is classified as **descriptively unsupported** under the frozen directional rule. A season supports the rule only when Context's opponent-minus-matched-control residual co-movement is positive and greater than History's in that same season; a season failing either condition counts as non-supporting. The table gives opponent and matched-control means with their differences; positive centered residual-product differences mean greater same-direction co-movement among scheduled opponents.

| Season | Model | Matched pairs | Pair measure | Opponent | Control | Opponent − control |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 2022 | History_existing | 729 | full-PMF NLL | +4.515518 | +4.538781 | -0.023264 |
| 2022 | History_existing | 729 | signed residual | +0.002235 | -0.001444 | +0.003679 |
| 2022 | History_existing | 729 | centered residual product | +0.000710 | -0.004317 | +0.005027 |
| 2022 | History_existing | 729 | expected-rank residual | -0.031260 | +0.076728 | -0.107988 |
| 2022 | Context_existing | 729 | full-PMF NLL | +4.449680 | +4.465089 | -0.015408 |
| 2022 | Context_existing | 729 | signed residual | +0.004241 | -0.001276 | +0.005517 |
| 2022 | Context_existing | 729 | centered residual product | -0.000422 | -0.002661 | +0.002239 |
| 2022 | Context_existing | 729 | expected-rank residual | -0.056355 | +0.043314 | -0.099669 |
| 2023 | History_existing | 747 | full-PMF NLL | +4.492340 | +4.481801 | +0.010539 |
| 2023 | History_existing | 747 | signed residual | -0.006445 | +0.001000 | -0.007445 |
| 2023 | History_existing | 747 | centered residual product | -0.002247 | -0.000156 | -0.002091 |
| 2023 | History_existing | 747 | expected-rank residual | +0.375752 | -0.379243 | +0.754995 |
| 2023 | Context_existing | 747 | full-PMF NLL | +4.472710 | +4.454891 | +0.017818 |
| 2023 | Context_existing | 747 | signed residual | +0.019612 | +0.026362 | -0.006750 |
| 2023 | Context_existing | 747 | centered residual product | +0.002245 | +0.000318 | +0.001927 |
| 2023 | Context_existing | 747 | expected-rank residual | -1.336862 | -1.974134 | +0.637272 |
| 2024 | History_existing | 749 | full-PMF NLL | +4.575078 | +4.582631 | -0.007553 |
| 2024 | History_existing | 749 | signed residual | -0.004682 | -0.002337 | -0.002345 |
| 2024 | History_existing | 749 | centered residual product | -0.004653 | -0.001685 | -0.002969 |
| 2024 | History_existing | 749 | expected-rank residual | +0.359402 | +0.105170 | +0.254233 |
| 2024 | Context_existing | 749 | full-PMF NLL | +4.522110 | +4.522922 | -0.000813 |
| 2024 | Context_existing | 749 | signed residual | +0.011048 | +0.013077 | -0.002030 |
| 2024 | Context_existing | 749 | centered residual product | -0.004674 | -0.000895 | -0.003779 |
| 2024 | Context_existing | 749 | expected-rank residual | -0.646761 | -0.971630 | +0.324869 |
| 2025 | History_existing | 755 | full-PMF NLL | +4.559202 | +4.560157 | -0.000955 |
| 2025 | History_existing | 755 | signed residual | +0.000714 | +0.000933 | -0.000220 |
| 2025 | History_existing | 755 | centered residual product | -0.003277 | +0.003805 | -0.007082 |
| 2025 | History_existing | 755 | expected-rank residual | +0.207194 | -0.023833 | +0.231028 |
| 2025 | Context_existing | 755 | full-PMF NLL | +4.534310 | +4.530519 | +0.003791 |
| 2025 | Context_existing | 755 | signed residual | -0.001208 | -0.003547 | +0.002339 |
| 2025 | Context_existing | 755 | centered residual product | -0.003515 | +0.002414 | -0.005928 |
| 2025 | Context_existing | 755 | expected-rank residual | +0.631326 | +0.587097 | +0.044229 |
| equal_season_mean | History_existing | equal-season mean | full-PMF NLL | +4.535534 | +4.540842 | -0.005308 |
| equal_season_mean | History_existing | equal-season mean | signed residual | -0.002044 | -0.000462 | -0.001582 |
| equal_season_mean | History_existing | equal-season mean | centered residual product | -0.002367 | -0.000588 | -0.001779 |
| equal_season_mean | History_existing | equal-season mean | expected-rank residual | +0.227772 | -0.055295 | +0.283067 |
| equal_season_mean | Context_existing | equal-season mean | full-PMF NLL | +4.494702 | +4.493355 | +0.001347 |
| equal_season_mean | Context_existing | equal-season mean | signed residual | +0.008423 | +0.008654 | -0.000231 |
| equal_season_mean | Context_existing | equal-season mean | centered residual product | -0.001591 | -0.000206 | -0.001385 |
| equal_season_mean | Context_existing | equal-season mean | expected-rank residual | -0.352163 | -0.578838 | +0.226675 |

Four seasons are only four season clusters. The pair universe overlaps at teams, so no p-values or independent-pair uncertainty claims are reported. The matched null accounts for pair-level schedule-degree level and difference plus preseason rank-level/disagreement features; it does not enforce exact degree balance and is not randomized. Similar levels, schedule structure, shared ranking-measurement error, and common fitted coefficients remain alternative explanations.

## Retained context and limits

#201's existing-policy game forecasts remain the primary operational comparison. Context future-game CRPS was lower than History in every season under the four-season rolling summary, while the previous rank-score difference cannot by itself diagnose shared errors. This historical analysis does not establish that Context causes correlated errors or harms game forecasts.

The ticket's original December final-target pilot values remain a labeled reference only: History NLL 3.83089, Context NLL 3.88793, Context prior with History messages 3.85330, and History prior with Context messages 3.85739. They do not use the corrected pre-bowl target and do not identify a mechanism.

**Part B is pending.** Synthetic dependence degradation and the small-graph BP approximation question remain unresolved until Part B is explicitly selected and run. No full-season correlated prior, new fit, production change, or follow-on experiment was introduced.

Inputs are recorded by logical label, byte count, and SHA-256 in `manifest.json`; no machine-specific input paths are stored.
