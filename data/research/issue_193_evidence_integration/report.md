# Issue 193: evidence-dependent Context integration

## Decision

**Retain unmodified Context / inconclusive**

The primary measure is prospective regular-season FBS–FBS margin CRPS. A lower CRPS is better. 2022–2025 have informed previous research, so these results are exploratory development rather than untouched confirmation.

## Future-game CRPS

| Model path | Pooled equal-game CRPS | Equal-season mean CRPS | 2023 | 2024 | 2025 | Games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `History_existing` | 8.961382 | 8.962150 | 8.897336 | 9.213743 | 8.775371 | 2264 |
| `Context_existing` | 8.879057 | 8.879838 | 8.844700 | 9.098935 | 8.695879 | 2264 |
| `History_calibrated` | 8.968404 | 8.969113 | 8.899150 | 9.213473 | 8.794715 | 2264 |
| `History_calibrated_plus_Context` | 8.912565 | 8.913208 | 8.834065 | 9.153681 | 8.751877 | 2264 |

Path 4 is the only model that uses the Context prior signal. Path 3 measures generic evidence-dependent History calibration. The full per-game score file also reports margin NLL, interval coverage/width, median-based margin MAE, and win Brier.

## Phase-level prospective diagnostics

| Model path | Stage | Population | Season | Phase | Games | Margin CRPS | Margin NLL | Coverage 50/80/95 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `Context_existing` | preseason | fbs_fbs | 2023 | army_navy | 1 | 5.153121 | 3.935535 | 1.000/1.000/1.000 |
| `Context_existing` | preseason | fbs_fbs | 2023 | conference_championship_period | 10 | 8.245280 | 4.108346 | 0.500/1.000/1.000 |
| `Context_existing` | preseason | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.273279 | 4.235877 | 0.576/0.860/0.971 |
| `Context_existing` | preseason | fbs_fbs | 2023 | rivalry_window | 63 | 10.242660 | 4.333145 | 0.556/0.810/0.952 |
| `Context_existing` | preseason | fbs_fbs | 2024 | army_navy | 1 | 12.787580 | 4.466873 | 0.000/1.000/1.000 |
| `Context_existing` | preseason | fbs_fbs | 2024 | conference_championship_period | 9 | 16.088390 | 4.809579 | 0.444/0.556/0.778 |
| `Context_existing` | preseason | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.814827 | 4.279392 | 0.526/0.823/0.970 |
| `Context_existing` | preseason | fbs_fbs | 2024 | rivalry_window | 65 | 10.780787 | 4.408851 | 0.585/0.815/0.923 |
| `Context_existing` | preseason | fbs_fbs | 2025 | army_navy | 1 | 4.829633 | 3.922298 | 1.000/1.000/1.000 |
| `Context_existing` | preseason | fbs_fbs | 2025 | conference_championship_period | 9 | 8.507193 | 4.162600 | 0.444/0.889/1.000 |
| `Context_existing` | preseason | fbs_fbs | 2025 | pre_thanksgiving | 813 | 9.634334 | 4.267699 | 0.571/0.839/0.966 |
| `Context_existing` | preseason | fbs_fbs | 2025 | rivalry_window | 65 | 10.774292 | 4.372242 | 0.538/0.815/0.923 |
| `Context_existing` | rolling | fbs_fbs | 2023 | army_navy | 1 | 3.612268 | 3.640023 | 1.000/1.000/1.000 |
| `Context_existing` | rolling | fbs_fbs | 2023 | conference_championship_period | 10 | 6.722846 | 3.923852 | 0.600/0.900/1.000 |
| `Context_existing` | rolling | fbs_fbs | 2023 | pre_thanksgiving | 794 | 8.965051 | 4.191027 | 0.519/0.811/0.969 |
| `Context_existing` | rolling | fbs_fbs | 2023 | rivalry_window | 63 | 8.530070 | 4.144036 | 0.556/0.794/0.937 |
| `Context_existing` | rolling | fbs_fbs | 2024 | army_navy | 1 | 16.210076 | 4.932204 | 0.000/0.000/1.000 |
| `Context_existing` | rolling | fbs_fbs | 2024 | conference_championship_period | 9 | 15.249923 | 4.932414 | 0.444/0.444/0.667 |
| `Context_existing` | rolling | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.110221 | 4.198964 | 0.495/0.801/0.965 |
| `Context_existing` | rolling | fbs_fbs | 2024 | rivalry_window | 65 | 8.730371 | 4.178553 | 0.538/0.785/0.923 |
| `Context_existing` | rolling | fbs_fbs | 2025 | army_navy | 1 | 3.600118 | 3.624899 | 1.000/1.000/1.000 |
| `Context_existing` | rolling | fbs_fbs | 2025 | conference_championship_period | 9 | 7.469590 | 4.002667 | 0.556/0.778/1.000 |
| `Context_existing` | rolling | fbs_fbs | 2025 | pre_thanksgiving | 813 | 8.770089 | 4.171320 | 0.539/0.834/0.958 |
| `Context_existing` | rolling | fbs_fbs | 2025 | rivalry_window | 65 | 8.841212 | 4.165977 | 0.462/0.785/0.938 |
| `History_calibrated` | preseason | fbs_fbs | 2023 | army_navy | 1 | 5.891374 | 4.010427 | 1.000/1.000/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2023 | conference_championship_period | 10 | 8.438130 | 4.136126 | 0.500/1.000/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.478270 | 4.260395 | 0.583/0.856/0.969 |
| `History_calibrated` | preseason | fbs_fbs | 2023 | rivalry_window | 63 | 10.594485 | 4.361687 | 0.556/0.794/0.889 |
| `History_calibrated` | preseason | fbs_fbs | 2024 | army_navy | 1 | 10.428938 | 4.286420 | 0.000/1.000/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2024 | conference_championship_period | 9 | 13.697463 | 4.572063 | 0.444/0.556/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2024 | pre_thanksgiving | 798 | 10.188835 | 4.317076 | 0.524/0.831/0.970 |
| `History_calibrated` | preseason | fbs_fbs | 2024 | rivalry_window | 65 | 10.951671 | 4.424564 | 0.585/0.815/0.938 |
| `History_calibrated` | preseason | fbs_fbs | 2025 | army_navy | 1 | 4.766811 | 3.939653 | 1.000/1.000/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2025 | conference_championship_period | 9 | 9.914612 | 4.295556 | 0.444/0.778/1.000 |
| `History_calibrated` | preseason | fbs_fbs | 2025 | pre_thanksgiving | 813 | 9.886522 | 4.294158 | 0.567/0.835/0.970 |
| `History_calibrated` | preseason | fbs_fbs | 2025 | rivalry_window | 65 | 11.147164 | 4.407855 | 0.538/0.800/0.923 |
| `History_calibrated` | rolling | fbs_fbs | 2023 | army_navy | 1 | 3.705229 | 3.656413 | 1.000/1.000/1.000 |
| `History_calibrated` | rolling | fbs_fbs | 2023 | conference_championship_period | 10 | 6.703709 | 3.924144 | 0.600/0.900/1.000 |
| `History_calibrated` | rolling | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.030173 | 4.199534 | 0.524/0.810/0.965 |
| `History_calibrated` | rolling | fbs_fbs | 2023 | rivalry_window | 63 | 8.533847 | 4.144906 | 0.571/0.794/0.937 |
| `History_calibrated` | rolling | fbs_fbs | 2024 | army_navy | 1 | 15.505964 | 4.837941 | 0.000/0.000/1.000 |
| `History_calibrated` | rolling | fbs_fbs | 2024 | conference_championship_period | 9 | 14.715539 | 4.852651 | 0.444/0.444/0.667 |
| `History_calibrated` | rolling | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.262598 | 4.214372 | 0.492/0.811/0.962 |
| `History_calibrated` | rolling | fbs_fbs | 2024 | rivalry_window | 65 | 8.753110 | 4.180397 | 0.554/0.769/0.923 |
| `History_calibrated` | rolling | fbs_fbs | 2025 | army_navy | 1 | 3.649115 | 3.640313 | 1.000/1.000/1.000 |
| `History_calibrated` | rolling | fbs_fbs | 2025 | conference_championship_period | 9 | 7.619128 | 4.019829 | 0.556/0.778/1.000 |
| `History_calibrated` | rolling | fbs_fbs | 2025 | pre_thanksgiving | 813 | 8.886726 | 4.184454 | 0.542/0.836/0.958 |
| `History_calibrated` | rolling | fbs_fbs | 2025 | rivalry_window | 65 | 8.854288 | 4.167574 | 0.477/0.800/0.938 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2023 | army_navy | 1 | 5.405315 | 3.967091 | 1.000/1.000/1.000 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2023 | conference_championship_period | 10 | 8.343597 | 4.124585 | 0.500/1.000/1.000 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.293952 | 4.241952 | 0.582/0.864/0.972 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2023 | rivalry_window | 63 | 10.305897 | 4.338062 | 0.587/0.825/0.937 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2024 | army_navy | 1 | 11.627667 | 4.373921 | 0.000/1.000/1.000 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2024 | conference_championship_period | 9 | 14.895019 | 4.684962 | 0.444/0.556/0.889 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.931966 | 4.293132 | 0.530/0.835/0.972 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2024 | rivalry_window | 65 | 10.777845 | 4.408604 | 0.569/0.815/0.938 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2025 | army_navy | 1 | 4.805093 | 3.935874 | 1.000/1.000/1.000 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2025 | conference_championship_period | 9 | 8.976536 | 4.209974 | 0.444/0.889/1.000 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2025 | pre_thanksgiving | 813 | 9.720817 | 4.278699 | 0.579/0.843/0.972 |
| `History_calibrated_plus_Context` | preseason | fbs_fbs | 2025 | rivalry_window | 65 | 10.913237 | 4.386596 | 0.523/0.815/0.923 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2023 | army_navy | 1 | 3.706545 | 3.657454 | 1.000/1.000/1.000 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2023 | conference_championship_period | 10 | 6.719868 | 3.926218 | 0.600/0.900/1.000 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2023 | pre_thanksgiving | 794 | 8.959849 | 4.192016 | 0.521/0.815/0.967 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2023 | rivalry_window | 63 | 8.526680 | 4.144141 | 0.571/0.794/0.937 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2024 | army_navy | 1 | 15.615532 | 4.849598 | 0.000/0.000/1.000 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2024 | conference_championship_period | 9 | 14.740394 | 4.855399 | 0.444/0.444/0.667 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.184672 | 4.207054 | 0.500/0.813/0.965 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2024 | rivalry_window | 65 | 8.752426 | 4.180482 | 0.569/0.769/0.923 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2025 | army_navy | 1 | 3.651562 | 3.641411 | 1.000/1.000/1.000 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2025 | conference_championship_period | 9 | 7.642243 | 4.022443 | 0.556/0.778/1.000 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2025 | pre_thanksgiving | 813 | 8.836175 | 4.179525 | 0.542/0.841/0.961 |
| `History_calibrated_plus_Context` | rolling | fbs_fbs | 2025 | rivalry_window | 65 | 8.861731 | 4.168406 | 0.477/0.800/0.938 |
| `History_existing` | preseason | fbs_fbs | 2023 | army_navy | 1 | 5.904902 | 4.005876 | 1.000/1.000/1.000 |
| `History_existing` | preseason | fbs_fbs | 2023 | conference_championship_period | 10 | 8.366847 | 4.125129 | 0.500/1.000/1.000 |
| `History_existing` | preseason | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.444276 | 4.255319 | 0.581/0.849/0.970 |
| `History_existing` | preseason | fbs_fbs | 2023 | rivalry_window | 63 | 10.581653 | 4.359515 | 0.540/0.794/0.889 |
| `History_existing` | preseason | fbs_fbs | 2024 | army_navy | 1 | 10.389763 | 4.281569 | 0.000/1.000/1.000 |
| `History_existing` | preseason | fbs_fbs | 2024 | conference_championship_period | 9 | 13.781585 | 4.578337 | 0.444/0.556/1.000 |
| `History_existing` | preseason | fbs_fbs | 2024 | pre_thanksgiving | 798 | 10.192792 | 4.316003 | 0.518/0.825/0.967 |
| `History_existing` | preseason | fbs_fbs | 2024 | rivalry_window | 65 | 11.009428 | 4.430661 | 0.569/0.800/0.938 |
| `History_existing` | preseason | fbs_fbs | 2025 | army_navy | 1 | 4.716514 | 3.927263 | 1.000/1.000/1.000 |
| `History_existing` | preseason | fbs_fbs | 2025 | conference_championship_period | 9 | 10.086769 | 4.311639 | 0.444/0.778/1.000 |
| `History_existing` | preseason | fbs_fbs | 2025 | pre_thanksgiving | 813 | 9.850065 | 4.289069 | 0.565/0.834/0.967 |
| `History_existing` | preseason | fbs_fbs | 2025 | rivalry_window | 65 | 11.107095 | 4.404620 | 0.538/0.785/0.923 |
| `History_existing` | rolling | fbs_fbs | 2023 | army_navy | 1 | 3.640605 | 3.643324 | 1.000/1.000/1.000 |
| `History_existing` | rolling | fbs_fbs | 2023 | conference_championship_period | 10 | 6.660612 | 3.917728 | 0.600/0.900/1.000 |
| `History_existing` | rolling | fbs_fbs | 2023 | pre_thanksgiving | 794 | 9.025677 | 4.198326 | 0.519/0.810/0.967 |
| `History_existing` | rolling | fbs_fbs | 2023 | rivalry_window | 63 | 8.551692 | 4.146461 | 0.556/0.794/0.937 |
| `History_existing` | rolling | fbs_fbs | 2024 | army_navy | 1 | 15.647809 | 4.864299 | 0.000/0.000/1.000 |
| `History_existing` | rolling | fbs_fbs | 2024 | conference_championship_period | 9 | 14.760551 | 4.868131 | 0.444/0.444/0.667 |
| `History_existing` | rolling | fbs_fbs | 2024 | pre_thanksgiving | 798 | 9.262759 | 4.214094 | 0.490/0.807/0.960 |
| `History_existing` | rolling | fbs_fbs | 2024 | rivalry_window | 65 | 8.751246 | 4.179714 | 0.538/0.769/0.923 |
| `History_existing` | rolling | fbs_fbs | 2025 | army_navy | 1 | 3.608374 | 3.627704 | 1.000/1.000/1.000 |
| `History_existing` | rolling | fbs_fbs | 2025 | conference_championship_period | 9 | 7.636237 | 4.020446 | 0.556/0.778/1.000 |
| `History_existing` | rolling | fbs_fbs | 2025 | pre_thanksgiving | 813 | 8.864703 | 4.181361 | 0.534/0.827/0.954 |
| `History_existing` | rolling | fbs_fbs | 2025 | rivalry_window | 65 | 8.847021 | 4.166453 | 0.462/0.785/0.938 |

This table keeps preseason forecasts and FBS–FCS rolling forecasts separate from the primary rolling FBS–FBS comparison. Phase labels come from the frozen #201 schedule plan.

## Early-season guardrails

| Candidate vs Context phase | CRPS delta | Calibration-error delta | Passed |
| --- | ---: | ---: | --- |
| preseason | 0.067545 | 0.005742 | True |
| pre_thanksgiving | 0.037656 | 0.004575 | True |

Guardrails compare pooled preseason and pre-Thanksgiving CRPS and mean absolute coverage deviation against fixed tolerances. Per-season checks are in `decision.json`.

## Secondary ranking results

Ranking scores use the corrected pre-bowl target PMFs. Checkpoint rows are weighted so each team-season totals one across its valid checkpoints.

| Model path | Season | Aggregation | NLL | Rank CRPS | Expected-rank error | 80% target mass | 80% width |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `History_existing` | 2023–2025 | pooled_team_season_weighted_all_evaluation_seasons | 4.178687 | 0.043757 | 13.248087 | 0.801765 | 46.475124 |
| `History_existing` | 2023–2025 | equal_season_team_season_weighted | 4.178752 | 0.043762 | 13.248614 | 0.801716 | 46.476661 |
| `History_existing` | 2023 | pooled_team_season_weighted | 4.186595 | 0.043177 | 13.056499 | 0.798769 | 46.636090 |
| `History_existing` | 2024 | pooled_team_season_weighted | 4.179909 | 0.045681 | 13.642992 | 0.796165 | 46.547108 |
| `History_existing` | 2025 | pooled_team_season_weighted | 4.169751 | 0.042428 | 13.046352 | 0.810213 | 46.246783 |
| `Context_existing` | 2023–2025 | pooled_team_season_weighted_all_evaluation_seasons | 4.205907 | 0.044547 | 13.246530 | 0.784216 | 44.164092 |
| `Context_existing` | 2023–2025 | equal_season_team_season_weighted | 4.205977 | 0.044553 | 13.247337 | 0.784184 | 44.165890 |
| `Context_existing` | 2023 | pooled_team_season_weighted | 4.214981 | 0.044465 | 13.192981 | 0.780607 | 44.586967 |
| `Context_existing` | 2024 | pooled_team_season_weighted | 4.206654 | 0.045890 | 13.491426 | 0.783182 | 43.896455 |
| `Context_existing` | 2025 | pooled_team_season_weighted | 4.196297 | 0.043304 | 13.057603 | 0.788764 | 44.014246 |
| `History_calibrated` | 2023–2025 | pooled_team_season_weighted_all_evaluation_seasons | 4.172339 | 0.043561 | 13.310238 | 0.822593 | 49.213131 |
| `History_calibrated` | 2023–2025 | equal_season_team_season_weighted | 4.172419 | 0.043568 | 13.310943 | 0.822527 | 49.211548 |
| `History_calibrated` | 2023 | pooled_team_season_weighted | 4.183083 | 0.043231 | 13.147370 | 0.816402 | 48.866667 |
| `History_calibrated` | 2024 | pooled_team_season_weighted | 4.172424 | 0.045399 | 13.698438 | 0.818535 | 49.409981 |
| `History_calibrated` | 2025 | pooled_team_season_weighted | 4.161749 | 0.042073 | 13.087020 | 0.832645 | 49.357996 |
| `History_calibrated_plus_Context` | 2023–2025 | pooled_team_season_weighted_all_evaluation_seasons | 4.165107 | 0.042567 | 13.054663 | 0.825321 | 48.630542 |
| `History_calibrated_plus_Context` | 2023–2025 | equal_season_team_season_weighted | 4.165177 | 0.042573 | 13.055190 | 0.825248 | 48.627857 |
| `History_calibrated_plus_Context` | 2023 | pooled_team_season_weighted | 4.176092 | 0.042439 | 12.899239 | 0.817187 | 48.163409 |
| `History_calibrated_plus_Context` | 2024 | pooled_team_season_weighted | 4.162807 | 0.044102 | 13.395399 | 0.822616 | 48.783582 |
| `History_calibrated_plus_Context` | 2025 | pooled_team_season_weighted | 4.156630 | 0.041179 | 12.870933 | 0.835940 | 48.936581 |

Preseason, each checkpoint, and equal-checkpoint means over checkpoints with future games are retained in `ranking_summary.csv`. Per-team scores and full PMFs are retained in `ranking_scores.csv` and `ranking_predictions.jsonl.gz`.

### Per-checkpoint ranking scores

| Model path | Season | Cutoff UTC | Phase | Team-seasons | NLL | Rank CRPS | Expected-rank error | 80% target mass | 80% width |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Context_existing` | 2023 | 2023-08-22T00:00:00+00:00 | preseason | 133 | 4.475653 | 0.068233 | 18.672731 | 0.868839 | 70.503759 |
| `Context_existing` | 2023 | 2023-08-29T00:00:00+00:00 | other_checkpoint | 133 | 4.485101 | 0.069666 | 18.830146 | 0.858062 | 69.593985 |
| `Context_existing` | 2023 | 2023-09-05T00:00:00+00:00 | other_checkpoint | 133 | 4.460564 | 0.067633 | 18.297825 | 0.816124 | 61.969925 |
| `Context_existing` | 2023 | 2023-09-12T00:00:00+00:00 | other_checkpoint | 133 | 4.398631 | 0.061360 | 16.874243 | 0.799415 | 54.887218 |
| `Context_existing` | 2023 | 2023-09-19T00:00:00+00:00 | other_checkpoint | 133 | 4.338290 | 0.053610 | 14.948295 | 0.788471 | 49.526316 |
| `Context_existing` | 2023 | 2023-09-26T00:00:00+00:00 | other_checkpoint | 133 | 4.349473 | 0.054168 | 15.006445 | 0.742523 | 45.075188 |
| `Context_existing` | 2023 | 2023-10-03T00:00:00+00:00 | other_checkpoint | 133 | 4.309168 | 0.051689 | 14.739720 | 0.722640 | 42.067669 |
| `Context_existing` | 2023 | 2023-10-10T00:00:00+00:00 | other_checkpoint | 133 | 4.257554 | 0.046255 | 13.453558 | 0.735756 | 40.285714 |
| `Context_existing` | 2023 | 2023-10-17T00:00:00+00:00 | other_checkpoint | 133 | 4.152206 | 0.037524 | 11.690470 | 0.761738 | 38.233083 |
| `Context_existing` | 2023 | 2023-10-24T00:00:00+00:00 | other_checkpoint | 133 | 4.098513 | 0.034176 | 11.037867 | 0.761571 | 36.541353 |
| `Context_existing` | 2023 | 2023-10-31T00:00:00+00:00 | other_checkpoint | 133 | 4.048034 | 0.029992 | 10.147921 | 0.772515 | 35.097744 |
| `Context_existing` | 2023 | 2023-11-07T00:00:00+00:00 | pre_thanksgiving | 133 | 4.028117 | 0.027374 | 9.439548 | 0.764578 | 33.360902 |
| `Context_existing` | 2023 | 2023-11-14T00:00:00+00:00 | pre_thanksgiving | 133 | 3.983726 | 0.024540 | 8.928311 | 0.773266 | 31.857143 |
| `Context_existing` | 2023 | 2023-11-21T00:00:00+00:00 | rivalry_window | 133 | 3.925479 | 0.020635 | 7.875646 | 0.779365 | 30.639098 |
| `Context_existing` | 2023 | 2023-11-28T00:00:00+00:00 | conference_championship_period | 133 | 3.914199 | 0.020126 | 7.951980 | 0.764244 | 29.165414 |
| `Context_existing` | 2024 | 2024-08-20T00:00:00+00:00 | preseason | 134 | 4.524185 | 0.076606 | 20.164771 | 0.824602 | 69.208955 |
| `Context_existing` | 2024 | 2024-08-27T00:00:00+00:00 | other_checkpoint | 134 | 4.530013 | 0.076930 | 20.245286 | 0.818705 | 69.029851 |
| `Context_existing` | 2024 | 2024-09-03T00:00:00+00:00 | other_checkpoint | 134 | 4.551981 | 0.080135 | 20.835266 | 0.768125 | 62.708955 |
| `Context_existing` | 2024 | 2024-09-10T00:00:00+00:00 | other_checkpoint | 134 | 4.429479 | 0.065170 | 17.525056 | 0.785445 | 56.686567 |
| `Context_existing` | 2024 | 2024-09-17T00:00:00+00:00 | other_checkpoint | 134 | 4.384999 | 0.059070 | 16.256227 | 0.769818 | 50.835821 |
| `Context_existing` | 2024 | 2024-09-24T00:00:00+00:00 | other_checkpoint | 134 | 4.324415 | 0.052488 | 14.748240 | 0.783209 | 46.544776 |
| `Context_existing` | 2024 | 2024-10-01T00:00:00+00:00 | other_checkpoint | 134 | 4.234088 | 0.045425 | 13.281851 | 0.792422 | 43.104478 |
| `Context_existing` | 2024 | 2024-10-08T00:00:00+00:00 | other_checkpoint | 134 | 4.192366 | 0.042839 | 12.732056 | 0.773964 | 40.455224 |
| `Context_existing` | 2024 | 2024-10-15T00:00:00+00:00 | other_checkpoint | 134 | 4.162972 | 0.040578 | 12.342318 | 0.759591 | 38.179104 |
| `Context_existing` | 2024 | 2024-10-22T00:00:00+00:00 | other_checkpoint | 134 | 4.102911 | 0.035175 | 11.060642 | 0.783854 | 36.261194 |
| `Context_existing` | 2024 | 2024-10-29T00:00:00+00:00 | other_checkpoint | 134 | 4.109020 | 0.035020 | 11.197280 | 0.756643 | 34.171642 |
| `Context_existing` | 2024 | 2024-11-05T00:00:00+00:00 | other_checkpoint | 134 | 4.051571 | 0.031101 | 10.575063 | 0.764566 | 33.111940 |
| `Context_existing` | 2024 | 2024-11-12T00:00:00+00:00 | pre_thanksgiving | 134 | 4.006962 | 0.028325 | 9.870157 | 0.773996 | 32.052239 |
| `Context_existing` | 2024 | 2024-11-19T00:00:00+00:00 | pre_thanksgiving | 134 | 3.957555 | 0.025262 | 9.150943 | 0.783025 | 31.074627 |
| `Context_existing` | 2024 | 2024-11-26T00:00:00+00:00 | rivalry_window | 134 | 3.903062 | 0.021646 | 8.283172 | 0.791317 | 30.000000 |
| `Context_existing` | 2024 | 2024-12-03T00:00:00+00:00 | conference_championship_period | 134 | 3.840885 | 0.018461 | 7.594491 | 0.801634 | 28.917910 |
| `Context_existing` | 2025 | 2025-08-19T00:00:00+00:00 | preseason | 136 | 4.534482 | 0.071697 | 19.349392 | 0.837785 | 72.566176 |
| `Context_existing` | 2025 | 2025-08-26T00:00:00+00:00 | other_checkpoint | 136 | 4.531027 | 0.071755 | 19.601973 | 0.836081 | 71.816176 |
| `Context_existing` | 2025 | 2025-09-02T00:00:00+00:00 | other_checkpoint | 136 | 4.444391 | 0.063912 | 18.320733 | 0.828348 | 64.617647 |
| `Context_existing` | 2025 | 2025-09-09T00:00:00+00:00 | other_checkpoint | 136 | 4.398434 | 0.059587 | 16.919613 | 0.806476 | 55.073529 |
| `Context_existing` | 2025 | 2025-09-16T00:00:00+00:00 | other_checkpoint | 136 | 4.356100 | 0.055508 | 15.736461 | 0.776424 | 49.507353 |
| `Context_existing` | 2025 | 2025-09-23T00:00:00+00:00 | other_checkpoint | 136 | 4.334195 | 0.053537 | 14.622599 | 0.759104 | 45.191176 |
| `Context_existing` | 2025 | 2025-09-30T00:00:00+00:00 | other_checkpoint | 136 | 4.258622 | 0.046113 | 13.404371 | 0.756868 | 42.323529 |
| `Context_existing` | 2025 | 2025-10-07T00:00:00+00:00 | other_checkpoint | 136 | 4.215528 | 0.042394 | 12.681952 | 0.761979 | 39.661765 |
| `Context_existing` | 2025 | 2025-10-14T00:00:00+00:00 | other_checkpoint | 136 | 4.167952 | 0.039159 | 11.790625 | 0.763314 | 37.808824 |
| `Context_existing` | 2025 | 2025-10-21T00:00:00+00:00 | other_checkpoint | 136 | 4.081216 | 0.033402 | 10.902165 | 0.786281 | 35.948529 |
| `Context_existing` | 2025 | 2025-10-28T00:00:00+00:00 | other_checkpoint | 136 | 4.051695 | 0.031824 | 10.360238 | 0.778661 | 34.588235 |
| `Context_existing` | 2025 | 2025-11-04T00:00:00+00:00 | other_checkpoint | 136 | 4.016153 | 0.029112 | 9.989762 | 0.784201 | 33.286765 |
| `Context_existing` | 2025 | 2025-11-11T00:00:00+00:00 | pre_thanksgiving | 136 | 3.971414 | 0.026651 | 9.455900 | 0.790497 | 32.330882 |
| `Context_existing` | 2025 | 2025-11-18T00:00:00+00:00 | pre_thanksgiving | 136 | 3.948312 | 0.024756 | 9.180754 | 0.788515 | 31.235294 |
| `Context_existing` | 2025 | 2025-11-25T00:00:00+00:00 | rivalry_window | 136 | 3.941734 | 0.023200 | 8.654202 | 0.780957 | 29.727941 |
| `Context_existing` | 2025 | 2025-12-02T00:00:00+00:00 | conference_championship_period | 136 | 3.889505 | 0.020256 | 7.950913 | 0.784727 | 28.544118 |
| `History_calibrated` | 2023 | 2023-08-22T00:00:00+00:00 | preseason | 133 | 4.504891 | 0.071378 | 20.002314 | 0.879198 | 79.466165 |
| `History_calibrated` | 2023 | 2023-08-29T00:00:00+00:00 | other_checkpoint | 133 | 4.509254 | 0.072145 | 20.017606 | 0.872849 | 78.300752 |
| `History_calibrated` | 2023 | 2023-09-05T00:00:00+00:00 | other_checkpoint | 133 | 4.447502 | 0.066007 | 18.379744 | 0.849373 | 68.834586 |
| `History_calibrated` | 2023 | 2023-09-12T00:00:00+00:00 | other_checkpoint | 133 | 4.384982 | 0.060481 | 16.865546 | 0.826483 | 60.556391 |
| `History_calibrated` | 2023 | 2023-09-19T00:00:00+00:00 | other_checkpoint | 133 | 4.297973 | 0.051504 | 14.800384 | 0.824979 | 54.157895 |
| `History_calibrated` | 2023 | 2023-09-26T00:00:00+00:00 | other_checkpoint | 133 | 4.295631 | 0.051460 | 14.746527 | 0.782623 | 49.210526 |
| `History_calibrated` | 2023 | 2023-10-03T00:00:00+00:00 | other_checkpoint | 133 | 4.256681 | 0.049036 | 14.364044 | 0.770844 | 45.744361 |
| `History_calibrated` | 2023 | 2023-10-10T00:00:00+00:00 | other_checkpoint | 133 | 4.207395 | 0.044095 | 13.284140 | 0.776859 | 43.548872 |
| `History_calibrated` | 2023 | 2023-10-17T00:00:00+00:00 | other_checkpoint | 133 | 4.118833 | 0.036420 | 11.652167 | 0.795322 | 41.263158 |
| `History_calibrated` | 2023 | 2023-10-24T00:00:00+00:00 | other_checkpoint | 133 | 4.062370 | 0.032588 | 10.731523 | 0.806600 | 39.390977 |
| `History_calibrated` | 2023 | 2023-10-31T00:00:00+00:00 | other_checkpoint | 133 | 4.005678 | 0.028095 | 9.828626 | 0.811696 | 37.864662 |
| `History_calibrated` | 2023 | 2023-11-07T00:00:00+00:00 | pre_thanksgiving | 133 | 3.971642 | 0.024972 | 8.771877 | 0.808271 | 35.872180 |
| `History_calibrated` | 2023 | 2023-11-14T00:00:00+00:00 | pre_thanksgiving | 133 | 3.935267 | 0.022835 | 8.729581 | 0.809774 | 34.300752 |
| `History_calibrated` | 2023 | 2023-11-21T00:00:00+00:00 | rivalry_window | 133 | 3.879213 | 0.018940 | 7.526146 | 0.817878 | 32.969925 |
| `History_calibrated` | 2023 | 2023-11-28T00:00:00+00:00 | conference_championship_period | 133 | 3.868940 | 0.018515 | 7.510330 | 0.813283 | 31.518797 |
| `History_calibrated` | 2024 | 2024-08-20T00:00:00+00:00 | preseason | 134 | 4.577847 | 0.084071 | 22.528302 | 0.823387 | 79.641791 |
| `History_calibrated` | 2024 | 2024-08-27T00:00:00+00:00 | other_checkpoint | 134 | 4.577703 | 0.083723 | 22.370502 | 0.823294 | 79.313433 |
| `History_calibrated` | 2024 | 2024-09-03T00:00:00+00:00 | other_checkpoint | 134 | 4.576249 | 0.085211 | 22.276962 | 0.791509 | 71.313433 |
| `History_calibrated` | 2024 | 2024-09-10T00:00:00+00:00 | other_checkpoint | 134 | 4.419093 | 0.065430 | 18.223248 | 0.829745 | 64.552239 |
| `History_calibrated` | 2024 | 2024-09-17T00:00:00+00:00 | other_checkpoint | 134 | 4.352665 | 0.057907 | 16.472240 | 0.811779 | 57.201493 |
| `History_calibrated` | 2024 | 2024-09-24T00:00:00+00:00 | other_checkpoint | 134 | 4.284937 | 0.051451 | 14.951502 | 0.817173 | 51.992537 |
| `History_calibrated` | 2024 | 2024-10-01T00:00:00+00:00 | other_checkpoint | 134 | 4.201091 | 0.044442 | 13.366750 | 0.815054 | 48.044776 |
| `History_calibrated` | 2024 | 2024-10-08T00:00:00+00:00 | other_checkpoint | 134 | 4.154057 | 0.041248 | 12.784671 | 0.806267 | 45.141791 |
| `History_calibrated` | 2024 | 2024-10-15T00:00:00+00:00 | other_checkpoint | 134 | 4.114903 | 0.038173 | 11.991012 | 0.802306 | 42.656716 |
| `History_calibrated` | 2024 | 2024-10-22T00:00:00+00:00 | other_checkpoint | 134 | 4.053456 | 0.033290 | 10.901069 | 0.819016 | 40.335821 |
| `History_calibrated` | 2024 | 2024-10-29T00:00:00+00:00 | other_checkpoint | 134 | 4.034097 | 0.031953 | 10.780279 | 0.809158 | 38.097015 |
| `History_calibrated` | 2024 | 2024-11-05T00:00:00+00:00 | other_checkpoint | 134 | 3.978565 | 0.028087 | 10.136554 | 0.815665 | 36.865672 |
| `History_calibrated` | 2024 | 2024-11-12T00:00:00+00:00 | pre_thanksgiving | 134 | 3.943638 | 0.025806 | 9.499435 | 0.819385 | 35.611940 |
| `History_calibrated` | 2024 | 2024-11-19T00:00:00+00:00 | pre_thanksgiving | 134 | 3.890156 | 0.022270 | 8.601446 | 0.826110 | 34.559701 |
| `History_calibrated` | 2024 | 2024-11-26T00:00:00+00:00 | rivalry_window | 134 | 3.831158 | 0.018324 | 7.563741 | 0.836245 | 33.276119 |
| `History_calibrated` | 2024 | 2024-12-03T00:00:00+00:00 | conference_championship_period | 134 | 3.769158 | 0.014992 | 6.727298 | 0.850465 | 31.955224 |
| `History_calibrated` | 2025 | 2025-08-19T00:00:00+00:00 | preseason | 136 | 4.567805 | 0.077807 | 21.532097 | 0.848366 | 81.977941 |
| `History_calibrated` | 2025 | 2025-08-26T00:00:00+00:00 | other_checkpoint | 136 | 4.562782 | 0.077629 | 21.726006 | 0.847738 | 81.264706 |
| `History_calibrated` | 2025 | 2025-09-02T00:00:00+00:00 | other_checkpoint | 136 | 4.469771 | 0.067610 | 19.616651 | 0.847725 | 73.213235 |
| `History_calibrated` | 2025 | 2025-09-09T00:00:00+00:00 | other_checkpoint | 136 | 4.396308 | 0.060079 | 17.420838 | 0.844418 | 62.477941 |
| `History_calibrated` | 2025 | 2025-09-16T00:00:00+00:00 | other_checkpoint | 136 | 4.326989 | 0.053466 | 15.741986 | 0.838484 | 55.992647 |
| `History_calibrated` | 2025 | 2025-09-23T00:00:00+00:00 | other_checkpoint | 136 | 4.287966 | 0.050678 | 14.683121 | 0.816061 | 50.779412 |
| `History_calibrated` | 2025 | 2025-09-30T00:00:00+00:00 | other_checkpoint | 136 | 4.205278 | 0.042488 | 12.935535 | 0.815255 | 47.330882 |
| `History_calibrated` | 2025 | 2025-10-07T00:00:00+00:00 | other_checkpoint | 136 | 4.160619 | 0.038986 | 11.966459 | 0.817495 | 44.441176 |
| `History_calibrated` | 2025 | 2025-10-14T00:00:00+00:00 | other_checkpoint | 136 | 4.109888 | 0.035661 | 11.244822 | 0.814982 | 42.073529 |
| `History_calibrated` | 2025 | 2025-10-21T00:00:00+00:00 | other_checkpoint | 136 | 4.034334 | 0.030642 | 10.347351 | 0.832558 | 40.058824 |
| `History_calibrated` | 2025 | 2025-10-28T00:00:00+00:00 | other_checkpoint | 136 | 4.001728 | 0.029119 | 9.918091 | 0.824219 | 38.294118 |
| `History_calibrated` | 2025 | 2025-11-04T00:00:00+00:00 | other_checkpoint | 136 | 3.962937 | 0.026219 | 9.417524 | 0.834789 | 36.963235 |
| `History_calibrated` | 2025 | 2025-11-11T00:00:00+00:00 | pre_thanksgiving | 136 | 3.914933 | 0.023507 | 8.884401 | 0.839193 | 35.852941 |
| `History_calibrated` | 2025 | 2025-11-18T00:00:00+00:00 | pre_thanksgiving | 136 | 3.890788 | 0.021812 | 8.545951 | 0.836678 | 34.470588 |
| `History_calibrated` | 2025 | 2025-11-25T00:00:00+00:00 | rivalry_window | 136 | 3.874314 | 0.020231 | 8.066388 | 0.831100 | 32.926471 |
| `History_calibrated` | 2025 | 2025-12-02T00:00:00+00:00 | conference_championship_period | 136 | 3.821549 | 0.017239 | 7.345106 | 0.833251 | 31.610294 |
| `History_calibrated_plus_Context` | 2023 | 2023-08-22T00:00:00+00:00 | preseason | 133 | 4.471552 | 0.067251 | 18.846337 | 0.883542 | 75.909774 |
| `History_calibrated_plus_Context` | 2023 | 2023-08-29T00:00:00+00:00 | other_checkpoint | 133 | 4.478338 | 0.068434 | 18.973868 | 0.875773 | 74.902256 |
| `History_calibrated_plus_Context` | 2023 | 2023-09-05T00:00:00+00:00 | other_checkpoint | 133 | 4.432645 | 0.064607 | 17.753326 | 0.845948 | 67.060150 |
| `History_calibrated_plus_Context` | 2023 | 2023-09-12T00:00:00+00:00 | other_checkpoint | 133 | 4.375038 | 0.059583 | 16.544117 | 0.827402 | 59.466165 |
| `History_calibrated_plus_Context` | 2023 | 2023-09-19T00:00:00+00:00 | other_checkpoint | 133 | 4.294711 | 0.051023 | 14.727112 | 0.816207 | 53.593985 |
| `History_calibrated_plus_Context` | 2023 | 2023-09-26T00:00:00+00:00 | other_checkpoint | 133 | 4.292615 | 0.051111 | 14.656893 | 0.783292 | 48.909774 |
| `History_calibrated_plus_Context` | 2023 | 2023-10-03T00:00:00+00:00 | other_checkpoint | 133 | 4.254244 | 0.048764 | 14.323786 | 0.772431 | 45.518797 |
| `History_calibrated_plus_Context` | 2023 | 2023-10-10T00:00:00+00:00 | other_checkpoint | 133 | 4.205210 | 0.043845 | 13.161731 | 0.777861 | 43.473684 |
| `History_calibrated_plus_Context` | 2023 | 2023-10-17T00:00:00+00:00 | other_checkpoint | 133 | 4.114884 | 0.036084 | 11.493242 | 0.797577 | 41.240602 |
| `History_calibrated_plus_Context` | 2023 | 2023-10-24T00:00:00+00:00 | other_checkpoint | 133 | 4.060051 | 0.032426 | 10.723798 | 0.810944 | 39.428571 |
| `History_calibrated_plus_Context` | 2023 | 2023-10-31T00:00:00+00:00 | other_checkpoint | 133 | 4.005979 | 0.028105 | 9.804529 | 0.814119 | 37.924812 |
| `History_calibrated_plus_Context` | 2023 | 2023-11-07T00:00:00+00:00 | pre_thanksgiving | 133 | 3.972495 | 0.025019 | 8.757009 | 0.809190 | 35.984962 |
| `History_calibrated_plus_Context` | 2023 | 2023-11-14T00:00:00+00:00 | pre_thanksgiving | 133 | 3.935368 | 0.022825 | 8.688497 | 0.811445 | 34.451128 |
| `History_calibrated_plus_Context` | 2023 | 2023-11-21T00:00:00+00:00 | rivalry_window | 133 | 3.879647 | 0.018966 | 7.513909 | 0.819967 | 33.037594 |
| `History_calibrated_plus_Context` | 2023 | 2023-11-28T00:00:00+00:00 | conference_championship_period | 133 | 3.868606 | 0.018536 | 7.520434 | 0.812114 | 31.548872 |
| `History_calibrated_plus_Context` | 2024 | 2024-08-20T00:00:00+00:00 | preseason | 134 | 4.531419 | 0.077629 | 21.000441 | 0.842090 | 76.328358 |
| `History_calibrated_plus_Context` | 2024 | 2024-08-27T00:00:00+00:00 | other_checkpoint | 134 | 4.532613 | 0.077563 | 20.991255 | 0.839971 | 76.097015 |
| `History_calibrated_plus_Context` | 2024 | 2024-09-03T00:00:00+00:00 | other_checkpoint | 134 | 4.543647 | 0.080793 | 21.320935 | 0.799710 | 69.507463 |
| `History_calibrated_plus_Context` | 2024 | 2024-09-10T00:00:00+00:00 | other_checkpoint | 134 | 4.405936 | 0.063792 | 17.780439 | 0.831696 | 63.305970 |
| `History_calibrated_plus_Context` | 2024 | 2024-09-17T00:00:00+00:00 | other_checkpoint | 134 | 4.347201 | 0.057309 | 16.334647 | 0.812440 | 56.567164 |
| `History_calibrated_plus_Context` | 2024 | 2024-09-24T00:00:00+00:00 | other_checkpoint | 134 | 4.281270 | 0.050938 | 14.831610 | 0.818740 | 51.634328 |
| `History_calibrated_plus_Context` | 2024 | 2024-10-01T00:00:00+00:00 | other_checkpoint | 134 | 4.198062 | 0.044060 | 13.299001 | 0.816897 | 47.902985 |
| `History_calibrated_plus_Context` | 2024 | 2024-10-08T00:00:00+00:00 | other_checkpoint | 134 | 4.151882 | 0.040999 | 12.685380 | 0.812567 | 45.119403 |
| `History_calibrated_plus_Context` | 2024 | 2024-10-15T00:00:00+00:00 | other_checkpoint | 134 | 4.112984 | 0.038021 | 11.923213 | 0.803964 | 42.619403 |
| `History_calibrated_plus_Context` | 2024 | 2024-10-22T00:00:00+00:00 | other_checkpoint | 134 | 4.051886 | 0.033123 | 10.851157 | 0.820582 | 40.417910 |
| `History_calibrated_plus_Context` | 2024 | 2024-10-29T00:00:00+00:00 | other_checkpoint | 134 | 4.033561 | 0.031882 | 10.754698 | 0.809987 | 38.164179 |
| `History_calibrated_plus_Context` | 2024 | 2024-11-05T00:00:00+00:00 | other_checkpoint | 134 | 3.978057 | 0.028041 | 10.131459 | 0.816955 | 36.955224 |
| `History_calibrated_plus_Context` | 2024 | 2024-11-12T00:00:00+00:00 | pre_thanksgiving | 134 | 3.942654 | 0.025744 | 9.490026 | 0.821780 | 35.723881 |
| `History_calibrated_plus_Context` | 2024 | 2024-11-19T00:00:00+00:00 | pre_thanksgiving | 134 | 3.890844 | 0.022306 | 8.613275 | 0.827769 | 34.656716 |
| `History_calibrated_plus_Context` | 2024 | 2024-11-26T00:00:00+00:00 | rivalry_window | 134 | 3.832543 | 0.018397 | 7.578643 | 0.837074 | 33.425373 |
| `History_calibrated_plus_Context` | 2024 | 2024-12-03T00:00:00+00:00 | conference_championship_period | 134 | 3.770359 | 0.015042 | 6.740199 | 0.849636 | 32.111940 |
| `History_calibrated_plus_Context` | 2025 | 2025-08-19T00:00:00+00:00 | preseason | 136 | 4.539171 | 0.072993 | 20.365519 | 0.857065 | 79.088235 |
| `History_calibrated_plus_Context` | 2025 | 2025-08-26T00:00:00+00:00 | other_checkpoint | 136 | 4.534131 | 0.072794 | 20.516948 | 0.856796 | 78.433824 |
| `History_calibrated_plus_Context` | 2025 | 2025-09-02T00:00:00+00:00 | other_checkpoint | 136 | 4.450622 | 0.064707 | 19.062942 | 0.857320 | 71.544118 |
| `History_calibrated_plus_Context` | 2025 | 2025-09-09T00:00:00+00:00 | other_checkpoint | 136 | 4.388178 | 0.058877 | 17.133947 | 0.847646 | 61.713235 |
| `History_calibrated_plus_Context` | 2025 | 2025-09-16T00:00:00+00:00 | other_checkpoint | 136 | 4.324284 | 0.053038 | 15.556138 | 0.836870 | 55.617647 |
| `History_calibrated_plus_Context` | 2025 | 2025-09-23T00:00:00+00:00 | other_checkpoint | 136 | 4.287214 | 0.050503 | 14.553985 | 0.816779 | 50.764706 |
| `History_calibrated_plus_Context` | 2025 | 2025-09-30T00:00:00+00:00 | other_checkpoint | 136 | 4.205556 | 0.042449 | 12.908637 | 0.819111 | 47.411765 |
| `History_calibrated_plus_Context` | 2025 | 2025-10-07T00:00:00+00:00 | other_checkpoint | 136 | 4.161409 | 0.039006 | 11.986224 | 0.819468 | 44.485294 |
| `History_calibrated_plus_Context` | 2025 | 2025-10-14T00:00:00+00:00 | other_checkpoint | 136 | 4.110626 | 0.035703 | 11.265335 | 0.815879 | 42.272059 |
| `History_calibrated_plus_Context` | 2025 | 2025-10-21T00:00:00+00:00 | other_checkpoint | 136 | 4.034936 | 0.030618 | 10.355534 | 0.832110 | 40.257353 |
| `History_calibrated_plus_Context` | 2025 | 2025-10-28T00:00:00+00:00 | other_checkpoint | 136 | 4.001996 | 0.029082 | 9.911524 | 0.828166 | 38.433824 |
| `History_calibrated_plus_Context` | 2025 | 2025-11-04T00:00:00+00:00 | other_checkpoint | 136 | 3.963734 | 0.026235 | 9.434378 | 0.837750 | 37.191176 |
| `History_calibrated_plus_Context` | 2025 | 2025-11-11T00:00:00+00:00 | pre_thanksgiving | 136 | 3.916557 | 0.023562 | 8.904654 | 0.839373 | 36.022059 |
| `History_calibrated_plus_Context` | 2025 | 2025-11-18T00:00:00+00:00 | pre_thanksgiving | 136 | 3.891872 | 0.021839 | 8.561982 | 0.841342 | 34.801471 |
| `History_calibrated_plus_Context` | 2025 | 2025-11-25T00:00:00+00:00 | rivalry_window | 136 | 3.874438 | 0.020230 | 8.072780 | 0.834239 | 33.132353 |
| `History_calibrated_plus_Context` | 2025 | 2025-12-02T00:00:00+00:00 | conference_championship_period | 136 | 3.821365 | 0.017228 | 7.344399 | 0.835134 | 31.816176 |
| `History_existing` | 2023 | 2023-08-22T00:00:00+00:00 | preseason | 133 | 4.495066 | 0.070347 | 19.707184 | 0.870092 | 76.872180 |
| `History_existing` | 2023 | 2023-08-29T00:00:00+00:00 | other_checkpoint | 133 | 4.502550 | 0.071331 | 19.734191 | 0.864327 | 75.661654 |
| `History_existing` | 2023 | 2023-09-05T00:00:00+00:00 | other_checkpoint | 133 | 4.448331 | 0.066161 | 18.196947 | 0.832331 | 66.067669 |
| `History_existing` | 2023 | 2023-09-12T00:00:00+00:00 | other_checkpoint | 133 | 4.385534 | 0.060368 | 16.635892 | 0.812114 | 57.819549 |
| `History_existing` | 2023 | 2023-09-19T00:00:00+00:00 | other_checkpoint | 133 | 4.297597 | 0.051314 | 14.617418 | 0.811028 | 51.563910 |
| `History_existing` | 2023 | 2023-09-26T00:00:00+00:00 | other_checkpoint | 133 | 4.307920 | 0.051878 | 14.665589 | 0.762740 | 46.744361 |
| `History_existing` | 2023 | 2023-10-03T00:00:00+00:00 | other_checkpoint | 133 | 4.267405 | 0.049404 | 14.373681 | 0.750125 | 43.428571 |
| `History_existing` | 2023 | 2023-10-10T00:00:00+00:00 | other_checkpoint | 133 | 4.218751 | 0.044352 | 13.236993 | 0.754219 | 41.383459 |
| `History_existing` | 2023 | 2023-10-17T00:00:00+00:00 | other_checkpoint | 133 | 4.125397 | 0.036484 | 11.596311 | 0.773517 | 39.142857 |
| `History_existing` | 2023 | 2023-10-24T00:00:00+00:00 | other_checkpoint | 133 | 4.067825 | 0.032629 | 10.727522 | 0.788889 | 37.308271 |
| `History_existing` | 2023 | 2023-10-31T00:00:00+00:00 | other_checkpoint | 133 | 4.008707 | 0.028050 | 9.798874 | 0.792147 | 35.969925 |
| `History_existing` | 2023 | 2023-11-07T00:00:00+00:00 | pre_thanksgiving | 133 | 3.976302 | 0.024959 | 8.784027 | 0.786132 | 33.992481 |
| `History_existing` | 2023 | 2023-11-14T00:00:00+00:00 | pre_thanksgiving | 133 | 3.940276 | 0.022862 | 8.723407 | 0.793734 | 32.578947 |
| `History_existing` | 2023 | 2023-11-21T00:00:00+00:00 | rivalry_window | 133 | 3.882229 | 0.018920 | 7.528123 | 0.798997 | 31.240602 |
| `History_existing` | 2023 | 2023-11-28T00:00:00+00:00 | conference_championship_period | 133 | 3.875039 | 0.018593 | 7.521324 | 0.791145 | 29.766917 |
| `History_existing` | 2024 | 2024-08-20T00:00:00+00:00 | preseason | 134 | 4.576718 | 0.084094 | 22.266327 | 0.815740 | 76.261194 |
| `History_existing` | 2024 | 2024-08-27T00:00:00+00:00 | other_checkpoint | 134 | 4.577013 | 0.083793 | 22.144635 | 0.815371 | 75.940299 |
| `History_existing` | 2024 | 2024-09-03T00:00:00+00:00 | other_checkpoint | 134 | 4.590898 | 0.086488 | 22.340094 | 0.768844 | 67.858209 |
| `History_existing` | 2024 | 2024-09-10T00:00:00+00:00 | other_checkpoint | 134 | 4.421896 | 0.065633 | 18.057724 | 0.803856 | 60.970149 |
| `History_existing` | 2024 | 2024-09-17T00:00:00+00:00 | other_checkpoint | 134 | 4.361949 | 0.058281 | 16.353016 | 0.786735 | 53.813433 |
| `History_existing` | 2024 | 2024-09-24T00:00:00+00:00 | other_checkpoint | 134 | 4.296955 | 0.051867 | 14.955987 | 0.791929 | 48.776119 |
| `History_existing` | 2024 | 2024-10-01T00:00:00+00:00 | other_checkpoint | 134 | 4.210172 | 0.044835 | 13.379423 | 0.790732 | 44.925373 |
| `History_existing` | 2024 | 2024-10-08T00:00:00+00:00 | other_checkpoint | 134 | 4.163574 | 0.041660 | 12.824215 | 0.778075 | 42.231343 |
| `History_existing` | 2024 | 2024-10-15T00:00:00+00:00 | other_checkpoint | 134 | 4.125750 | 0.038572 | 12.030697 | 0.774205 | 39.880597 |
| `History_existing` | 2024 | 2024-10-22T00:00:00+00:00 | other_checkpoint | 134 | 4.063112 | 0.033565 | 10.938213 | 0.792666 | 37.694030 |
| `History_existing` | 2024 | 2024-10-29T00:00:00+00:00 | other_checkpoint | 134 | 4.048526 | 0.032356 | 10.745977 | 0.784835 | 35.626866 |
| `History_existing` | 2024 | 2024-11-05T00:00:00+00:00 | other_checkpoint | 134 | 3.989269 | 0.028308 | 10.094928 | 0.790144 | 34.358209 |
| `History_existing` | 2024 | 2024-11-12T00:00:00+00:00 | pre_thanksgiving | 134 | 3.953984 | 0.025995 | 9.473811 | 0.795338 | 33.253731 |
| `History_existing` | 2024 | 2024-11-19T00:00:00+00:00 | pre_thanksgiving | 134 | 3.897326 | 0.022372 | 8.558207 | 0.804459 | 32.268657 |
| `History_existing` | 2024 | 2024-11-26T00:00:00+00:00 | rivalry_window | 134 | 3.834647 | 0.018290 | 7.499122 | 0.814962 | 31.029851 |
| `History_existing` | 2024 | 2024-12-03T00:00:00+00:00 | conference_championship_period | 134 | 3.766757 | 0.014792 | 6.625491 | 0.830749 | 29.865672 |
| `History_existing` | 2025 | 2025-08-19T00:00:00+00:00 | preseason | 136 | 4.560267 | 0.076786 | 21.036370 | 0.841103 | 78.389706 |
| `History_existing` | 2025 | 2025-08-26T00:00:00+00:00 | other_checkpoint | 136 | 4.555390 | 0.076701 | 21.244653 | 0.840026 | 77.610294 |
| `History_existing` | 2025 | 2025-09-02T00:00:00+00:00 | other_checkpoint | 136 | 4.462856 | 0.067076 | 19.245912 | 0.829783 | 69.345588 |
| `History_existing` | 2025 | 2025-09-09T00:00:00+00:00 | other_checkpoint | 136 | 4.401481 | 0.060532 | 17.341364 | 0.831413 | 58.558824 |
| `History_existing` | 2025 | 2025-09-16T00:00:00+00:00 | other_checkpoint | 136 | 4.337505 | 0.054385 | 15.766585 | 0.811577 | 52.183824 |
| `History_existing` | 2025 | 2025-09-23T00:00:00+00:00 | other_checkpoint | 136 | 4.305775 | 0.051789 | 14.641392 | 0.789956 | 47.227941 |
| `History_existing` | 2025 | 2025-09-30T00:00:00+00:00 | other_checkpoint | 136 | 4.219313 | 0.043319 | 13.014332 | 0.785744 | 44.007353 |
| `History_existing` | 2025 | 2025-10-07T00:00:00+00:00 | other_checkpoint | 136 | 4.176533 | 0.039810 | 12.079637 | 0.789418 | 41.147059 |
| `History_existing` | 2025 | 2025-10-14T00:00:00+00:00 | other_checkpoint | 136 | 4.125335 | 0.036462 | 11.382199 | 0.791029 | 39.139706 |
| `History_existing` | 2025 | 2025-10-21T00:00:00+00:00 | other_checkpoint | 136 | 4.044512 | 0.031340 | 10.452209 | 0.802332 | 37.176471 |
| `History_existing` | 2025 | 2025-10-28T00:00:00+00:00 | other_checkpoint | 136 | 4.012748 | 0.029754 | 9.991997 | 0.798211 | 35.602941 |
| `History_existing` | 2025 | 2025-11-04T00:00:00+00:00 | other_checkpoint | 136 | 3.972406 | 0.026685 | 9.508719 | 0.806977 | 34.330882 |
| `History_existing` | 2025 | 2025-11-11T00:00:00+00:00 | pre_thanksgiving | 136 | 3.921453 | 0.023874 | 8.951946 | 0.812555 | 33.308824 |
| `History_existing` | 2025 | 2025-11-18T00:00:00+00:00 | pre_thanksgiving | 136 | 3.899373 | 0.022168 | 8.586601 | 0.816313 | 32.080882 |
| `History_existing` | 2025 | 2025-11-25T00:00:00+00:00 | rivalry_window | 136 | 3.888032 | 0.020606 | 8.102802 | 0.807231 | 30.522059 |
| `History_existing` | 2025 | 2025-12-02T00:00:00+00:00 | conference_championship_period | 136 | 3.833029 | 0.017555 | 7.394917 | 0.809747 | 29.316176 |

## Fit and parity audit

Baseline game parity checked 10516 rows and baseline ranking parity checked 12630 rows. The maximum permitted score difference was 1e-08.

| Evaluation season | Training seasons | History lambda | Context lambda |
| ---: | --- | ---: | ---: |
| 2023 | 2022 | 0.1 | 0.1 |
| 2024 | 2022,2023 | 0.1 | 0.1 |
| 2025 | 2022,2023,2024 | 0.1 | 0.1 |

The optimizer traces, training-row hashes, standardized evidence parameters, and freely fitted Context-weight coefficients are in `fit_records.json`. `candidate_weights.csv` records the fitted weight against each held-out team's evidence values.

## Fallback and transfer-coverage strata

| Model | Season | Prior status | Repaired transfer coverage | Team-checkpoint rows | NLL | Rank CRPS |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `Context_existing` | 2022 | fitted_prior | available | 1950 | 4.121040 | 0.041023 |
| `Context_existing` | 2022 | native_cold_start_fallback | missing | 15 | 4.539843 | 0.044476 |
| `Context_existing` | 2023 | fitted_prior | available | 1965 | 4.215078 | 0.044545 |
| `Context_existing` | 2023 | native_cold_start_fallback | missing | 30 | 4.208583 | 0.039261 |
| `Context_existing` | 2024 | fitted_prior | available | 2128 | 4.220295 | 0.046156 |
| `Context_existing` | 2024 | native_cold_start_fallback | missing | 16 | 2.392395 | 0.010460 |
| `Context_existing` | 2025 | fitted_prior | available | 2144 | 4.193464 | 0.043495 |
| `Context_existing` | 2025 | native_cold_start_fallback | missing | 32 | 4.386147 | 0.030500 |
| `History_calibrated` | 2023 | fitted_prior | available | 1965 | 4.182860 | 0.043297 |
| `History_calibrated` | 2023 | native_cold_start_fallback | missing | 30 | 4.197705 | 0.038931 |
| `History_calibrated` | 2024 | fitted_prior | available | 2128 | 4.185009 | 0.045629 |
| `History_calibrated` | 2024 | native_cold_start_fallback | missing | 16 | 2.498533 | 0.014779 |
| `History_calibrated` | 2025 | fitted_prior | available | 2144 | 4.158967 | 0.042282 |
| `History_calibrated` | 2025 | native_cold_start_fallback | missing | 32 | 4.348179 | 0.028090 |
| `History_calibrated_plus_Context` | 2023 | fitted_prior | available | 1965 | 4.175766 | 0.042492 |
| `History_calibrated_plus_Context` | 2023 | native_cold_start_fallback | missing | 30 | 4.197491 | 0.038937 |
| `History_calibrated_plus_Context` | 2024 | fitted_prior | available | 2128 | 4.175314 | 0.044324 |
| `History_calibrated_plus_Context` | 2024 | native_cold_start_fallback | missing | 16 | 2.499478 | 0.014653 |
| `History_calibrated_plus_Context` | 2025 | fitted_prior | available | 2144 | 4.153792 | 0.041378 |
| `History_calibrated_plus_Context` | 2025 | native_cold_start_fallback | missing | 32 | 4.346791 | 0.027818 |
| `History_existing` | 2022 | fitted_prior | available | 1950 | 4.148570 | 0.044059 |
| `History_existing` | 2022 | native_cold_start_fallback | missing | 15 | 4.542552 | 0.044509 |
| `History_existing` | 2023 | fitted_prior | available | 1965 | 4.186356 | 0.043245 |
| `History_existing` | 2023 | native_cold_start_fallback | missing | 30 | 4.202244 | 0.038744 |
| `History_existing` | 2024 | fitted_prior | available | 2128 | 4.193239 | 0.045944 |
| `History_existing` | 2024 | native_cold_start_fallback | missing | 16 | 2.407013 | 0.010761 |
| `History_existing` | 2025 | fitted_prior | available | 2144 | 4.166715 | 0.042623 |
| `History_existing` | 2025 | native_cold_start_fallback | missing | 32 | 4.373105 | 0.029355 |

Team-level gain concentration and fitted weight distributions are in `gain_concentration.csv` and `weight_summary.csv`; paired game deltas are in `paired_game_differences.csv`.

## Interpretation

Across 2264 held-out rolling FBS–FBS games, path 4 minus unmodified Context has margin-NLL delta 0.004541 and interval-calibration-error delta 0.004711. The pooled team-season-weighted rank-NLL delta is -0.040800.

| Season | Margin-NLL delta | Interval-calibration-error delta | Rank-NLL delta |
| ---: | ---: | ---: | ---: |
| 2023 | 0.000352 | 0.002222 | -0.038888 |
| 2024 | 0.005994 | -0.006649 | -0.043846 |
| 2025 | 0.007228 | 0.005249 | -0.039667 |

The final comparison follows the preregistered practical-gain threshold and early guardrails. Season/checkpoint rows are dependent; no p-values or independent-row claims are made. The result does not change production priors, likelihoods, inference settings, or site defaults. Stop after this candidate; a separate issue is required for independent confirmation or release.
