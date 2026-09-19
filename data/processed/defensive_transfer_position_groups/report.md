# Defensive-transfer production by position group (issue 111)

## Conclusion

Final recommendation: **advance D5 + DB defensive position-group feature**

P0 D5 aggregate NLL is 4.5081. The individual position-group decisions are DL / EDGE: fail, LB: fail, and DB: pass. P4 has frozen ΔNLL 0.029433 versus D5.

## Frozen player construction

The study uses the PR #105 impact values unchanged. It does not refit the player metric, change component weights, add a position taxonomy, or split any group further.

| Group | Frozen source positions | Frozen impact components | Normalization |
|---|---|---|---|
| DL / EDGE | DL, EDGE, DE, DT, NT | tackles; tackles for loss; sacks; quarterback hurries | prior season × group |
| LB | OLB, ILB, MLB, LB | tackles; tackles for loss; sacks; passes defended | prior season × group |
| DB | CB, DB, S, FS, SS, NB | tackles; passes defended; interceptions | prior season × group |

Impact is the equal-weight mean of standardized `log1p` components. The incoming team features are sums of those frozen player-level impacts.

## Candidate definitions

| Candidate | Definition | Added features beyond D5 |
|---|---|---|
| P0 | D5: C-minus-RP plus returning production and incoming prior usage. | none |
| P1 | D5 plus imported prior defensive impact from DL / EDGE and its availability indicator. | `transfer_in_prior_defensive_impact_dl_edge_sum`, `transfer_in_prior_defensive_impact_dl_edge_available` |
| P2 | D5 plus imported prior defensive impact from LB and its availability indicator. | `transfer_in_prior_defensive_impact_lb_sum`, `transfer_in_prior_defensive_impact_lb_available` |
| P3 | D5 plus imported prior defensive impact from DB and its availability indicator. | `transfer_in_prior_defensive_impact_db_sum`, `transfer_in_prior_defensive_impact_db_available` |
| P4 | D5 plus all three position-specific defensive impact values and availability indicators. | `transfer_in_prior_defensive_impact_dl_edge_sum`, `transfer_in_prior_defensive_impact_lb_sum`, `transfer_in_prior_defensive_impact_db_sum`, `transfer_in_prior_defensive_impact_dl_edge_available`, `transfer_in_prior_defensive_impact_lb_available`, `transfer_in_prior_defensive_impact_db_available` |

## Missing-data treatment

For each group, no incoming transfers are observed zeros with availability 1. Any incoming transfer whose frozen impact cannot be resolved is an unresolved group: its numeric value is neutralized to zero and availability is 0. Team-seasons remain in the model population. Rows before the defensive audit coverage window use the same neutral value with availability 0.

## Protocol and parity

- D5 is the P0 control. Fits use data through 2021 for frozen 2022–2025 scoring; rolling fits train through 2021, 2022, 2023, and 2024 for targets 2022, 2023, 2024, and 2025.
- The production FBS population, Context preprocessing, model family, penalty, optimizer retry, H fallback, August 15 cutoff, response, and scoring implementation are unchanged.
- No target-season outcomes enter feature construction or fitting.
- D5 parity: passed; maximum absolute metric delta 0.000000000 against the stored decomposition artifact.
- Aggregate reconstruction: 226 / 226 compared team-seasons passed at tolerance 1e-06; unresolved groups are not silently treated as reconstructed.

## Aggregate-versus-position reconstruction

The position sum is compared with the existing aggregate only when all three position groups are observed.

| Season | Compared | Passed | Maximum absolute delta |
|---:|---:|---:|---:|
| 2021 | 54 | 54 | 0.000000000 |
| 2022 | 53 | 53 | 0.000000000 |
| 2023 | 53 | 53 | 0.000000000 |
| 2024 | 38 | 38 | 0.000000000 |
| 2025 | 28 | 28 | 0.000000000 |

## Coverage by season and position group

The existing audit supplies an experience-mass coverage proxy. No impact-mass proxy is present in that artifact, so the report leaves that field unavailable rather than inventing one.
Resolved impact counts include the audit's verified zero-recorded-box-score impacts; those are observed zeros, not unresolved players.

| Season | Group | Incoming | Resolved impacts | Unresolved impacts | Experience-mass proxy | Complete teams | No incoming | Unresolved teams | Observed fraction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | DL / EDGE | 110 | 76 | 34 | 0.976 | 41 | 59 | 30 | 0.769 |
| 2021 | LB | 86 | 50 | 36 | 0.856 | 28 | 74 | 28 | 0.785 |
| 2021 | DB | 166 | 112 | 54 | 0.954 | 44 | 39 | 47 | 0.638 |
| 2022 | DL / EDGE | 168 | 121 | 47 | 0.880 | 53 | 41 | 37 | 0.718 |
| 2022 | LB | 118 | 80 | 38 | 0.892 | 43 | 52 | 36 | 0.725 |
| 2022 | DB | 198 | 146 | 52 | 0.967 | 56 | 34 | 41 | 0.687 |
| 2023 | DL / EDGE | 213 | 147 | 66 | 0.914 | 42 | 45 | 46 | 0.654 |
| 2023 | LB | 136 | 108 | 28 | 0.884 | 56 | 51 | 26 | 0.805 |
| 2023 | DB | 284 | 218 | 66 | 0.959 | 61 | 26 | 46 | 0.654 |
| 2024 | DL / EDGE | 369 | 264 | 105 | 0.918 | 50 | 19 | 65 | 0.515 |
| 2024 | LB | 177 | 145 | 32 | 0.968 | 59 | 48 | 27 | 0.799 |
| 2024 | DB | 438 | 364 | 74 | 0.973 | 69 | 12 | 53 | 0.604 |
| 2025 | DL / EDGE | 559 | 407 | 152 | 0.912 | 47 | 9 | 80 | 0.412 |
| 2025 | LB | 219 | 168 | 51 | 0.930 | 58 | 40 | 38 | 0.721 |
| 2025 | DB | 602 | 481 | 121 | 0.980 | 53 | 10 | 73 | 0.463 |

### Observed training coverage by rolling fit

| Target | Train through | Group | Observed team-seasons | Total training team-seasons | Observed fraction |
|---:|---:|---|---:|---:|---:|
| 2022 | 2021 | DL / EDGE | 98 | 2216 | 0.044 |
| 2022 | 2021 | LB | 99 | 2216 | 0.045 |
| 2022 | 2021 | DB | 80 | 2216 | 0.036 |
| 2023 | 2022 | DL / EDGE | 191 | 2346 | 0.081 |
| 2023 | 2022 | LB | 193 | 2346 | 0.082 |
| 2023 | 2022 | DB | 169 | 2346 | 0.072 |
| 2024 | 2023 | DL / EDGE | 276 | 2477 | 0.111 |
| 2024 | 2023 | LB | 298 | 2477 | 0.120 |
| 2024 | 2023 | DB | 254 | 2477 | 0.103 |
| 2025 | 2024 | DL / EDGE | 345 | 2610 | 0.132 |
| 2025 | 2024 | LB | 404 | 2610 | 0.155 |
| 2025 | 2024 | DB | 335 | 2610 | 0.128 |

## Frozen 2022–2025 metrics

| Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 4.5081 | 0.000000 | 0.1080 | 19.49 | 19.68 | 0.838 | 71.96 |
| P1 | 4.5099 | 0.001782 | 0.1080 | 19.44 | 19.63 | 0.836 | 71.99 |
| P2 | 4.5280 | 0.019880 | 0.1094 | 19.56 | 20.06 | 0.830 | 70.53 |
| P3 | 4.5061 | -0.001935 | 0.1078 | 19.41 | 19.65 | 0.837 | 71.62 |
| P4 | 4.5375 | 0.029433 | 0.1103 | 19.57 | 20.34 | 0.823 | 69.51 |

The D5 control P0 has aggregate NLL 4.5081. Negative ΔNLL favors the position-group candidate.

### Frozen year-by-year metrics

| Season | Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | P0 | 4.4564 | 0.000000 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 |
| 2023 | P0 | 4.4684 | 0.000000 | 0.1040 | 19.16 | 19.40 | 0.871 | 72.35 |
| 2024 | P0 | 4.5492 | 0.000000 | 0.1145 | 20.41 | 20.70 | 0.815 | 70.48 |
| 2025 | P0 | 4.5561 | 0.000000 | 0.1078 | 19.84 | 20.04 | 0.832 | 73.29 |
| 2022 | P1 | 4.4607 | 0.004309 | 0.1056 | 18.45 | 18.45 | 0.833 | 71.63 |
| 2023 | P1 | 4.4635 | -0.004905 | 0.1038 | 19.16 | 19.43 | 0.871 | 72.06 |
| 2024 | P1 | 4.5554 | 0.006181 | 0.1148 | 20.33 | 20.64 | 0.816 | 70.60 |
| 2025 | P1 | 4.5577 | 0.001551 | 0.1079 | 19.79 | 19.96 | 0.826 | 73.66 |
| 2022 | P2 | 4.4727 | 0.016285 | 0.1063 | 18.50 | 18.49 | 0.834 | 70.83 |
| 2023 | P2 | 4.4827 | 0.014324 | 0.1052 | 19.01 | 19.71 | 0.856 | 70.54 |
| 2024 | P2 | 4.5697 | 0.020466 | 0.1161 | 20.45 | 21.22 | 0.809 | 68.60 |
| 2025 | P2 | 4.5843 | 0.028200 | 0.1099 | 20.23 | 20.77 | 0.821 | 72.13 |
| 2022 | P3 | 4.4562 | -0.000200 | 0.1055 | 18.47 | 18.52 | 0.832 | 71.58 |
| 2023 | P3 | 4.4676 | -0.000859 | 0.1042 | 19.22 | 19.55 | 0.873 | 71.99 |
| 2024 | P3 | 4.5468 | -0.002440 | 0.1142 | 20.26 | 20.67 | 0.814 | 70.01 |
| 2025 | P3 | 4.5520 | -0.004160 | 0.1072 | 19.66 | 19.81 | 0.830 | 72.88 |
| 2022 | P4 | 4.4828 | 0.026394 | 0.1071 | 18.42 | 18.66 | 0.829 | 70.09 |
| 2023 | P4 | 4.4898 | 0.021416 | 0.1066 | 19.24 | 20.17 | 0.846 | 69.54 |
| 2024 | P4 | 4.5899 | 0.040705 | 0.1177 | 20.51 | 21.81 | 0.800 | 67.40 |
| 2025 | P4 | 4.5852 | 0.029096 | 0.1097 | 20.06 | 20.67 | 0.817 | 70.99 |

## Rolling-origin metrics

### Year-by-year rolling metrics

| Target | Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | P0 | 4.4564 | 0.000000 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 |
| 2022 | P1 | 4.4607 | 0.004309 | 0.1056 | 18.45 | 18.45 | 0.833 | 71.63 |
| 2022 | P2 | 4.4727 | 0.016285 | 0.1063 | 18.50 | 18.49 | 0.834 | 70.83 |
| 2022 | P3 | 4.4562 | -0.000200 | 0.1055 | 18.47 | 18.52 | 0.832 | 71.58 |
| 2022 | P4 | 4.4828 | 0.026394 | 0.1071 | 18.42 | 18.66 | 0.829 | 70.09 |
| 2023 | P0 | 4.4680 | 0.000000 | 0.1041 | 19.16 | 19.36 | 0.870 | 72.01 |
| 2023 | P1 | 4.4659 | -0.002169 | 0.1040 | 19.17 | 19.40 | 0.871 | 71.95 |
| 2023 | P2 | 4.4678 | -0.000247 | 0.1039 | 19.06 | 19.36 | 0.870 | 71.83 |
| 2023 | P3 | 4.4678 | -0.000248 | 0.1042 | 19.22 | 19.45 | 0.870 | 71.80 |
| 2023 | P4 | 4.4672 | -0.000801 | 0.1041 | 19.14 | 19.48 | 0.869 | 71.74 |
| 2024 | P0 | 4.5533 | 0.000000 | 0.1151 | 20.59 | 21.04 | 0.813 | 70.34 |
| 2024 | P1 | 4.5572 | 0.003949 | 0.1151 | 20.52 | 20.95 | 0.804 | 70.38 |
| 2024 | P2 | 4.5538 | 0.000513 | 0.1149 | 20.49 | 20.93 | 0.813 | 70.13 |
| 2024 | P3 | 4.5512 | -0.002065 | 0.1148 | 20.49 | 20.90 | 0.813 | 70.02 |
| 2024 | P4 | 4.5579 | 0.004620 | 0.1150 | 20.46 | 20.94 | 0.814 | 69.99 |
| 2025 | P0 | 4.5524 | 0.000000 | 0.1078 | 19.93 | 20.07 | 0.828 | 73.54 |
| 2025 | P1 | 4.5502 | -0.002184 | 0.1076 | 19.82 | 20.05 | 0.827 | 73.40 |
| 2025 | P2 | 4.5566 | 0.004190 | 0.1081 | 20.00 | 20.23 | 0.827 | 73.47 |
| 2025 | P3 | 4.5504 | -0.001936 | 0.1075 | 19.86 | 20.08 | 0.829 | 73.49 |
| 2025 | P4 | 4.5508 | -0.001566 | 0.1074 | 19.82 | 20.07 | 0.826 | 73.10 |

### Weighted rolling aggregate

| Candidate | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 4.5080 | 0.000000 | 0.1082 | 19.55 | 19.76 | 0.836 | 71.90 |
| P1 | 4.5090 | 0.000952 | 0.1081 | 19.50 | 19.72 | 0.833 | 71.85 |
| P2 | 4.5132 | 0.005129 | 0.1083 | 19.52 | 19.76 | 0.836 | 71.58 |
| P3 | 4.5069 | -0.001122 | 0.1080 | 19.52 | 19.75 | 0.836 | 71.73 |
| P4 | 4.5151 | 0.007036 | 0.1084 | 19.47 | 19.79 | 0.834 | 71.24 |

### Rolling DB availability-only control

P3-M is D5 plus the DB availability indicator, with the numeric DB impact removed. Both P3 and P3-M are refit at each rolling origin. Each row below is P3 minus P3-M, so a negative value means the resolved DB impact adds predictive information beyond resolvability. The aggregate pools the paired team-seasons across the four target seasons.

| Comparison | Scope | Mean ΔNLL | Median ΔNLL | Fraction improved | Team-seasons |
|---|---|---:|---:|---:|---:|
| P3_vs_P3-M | aggregate | -0.001399 | -0.000170 | 0.537 | 534 |
| P3_vs_P3-M | 2022 | 0.000093 | -0.000110 | 0.534 | 131 |
| P3_vs_P3-M | 2023 | -0.001241 | -0.000021 | 0.511 | 133 |
| P3_vs_P3-M | 2024 | -0.001469 | -0.000024 | 0.507 | 134 |
| P3_vs_P3-M | 2025 | -0.002921 | -0.000719 | 0.596 | 136 |

## Paired team-season NLL diagnostics

| Protocol | Comparison | Scope | Mean ΔNLL | Median ΔNLL | Fraction improved | Team-seasons |
|---|---|---|---:|---:|---:|---:|
| frozen_through_2021 | P1_vs_P0 | aggregate | 0.001782 | 0.000233 | 0.485 | 534 |
| frozen_through_2021 | P1_vs_P0 | 2022 | 0.004309 | 0.000000 | 0.496 | 131 |
| frozen_through_2021 | P1_vs_P0 | 2023 | -0.004905 | 0.001074 | 0.474 | 133 |
| frozen_through_2021 | P1_vs_P0 | 2024 | 0.006181 | 0.000142 | 0.493 | 134 |
| frozen_through_2021 | P1_vs_P0 | 2025 | 0.001551 | 0.000823 | 0.478 | 136 |
| frozen_through_2021 | P2_vs_P0 | aggregate | 0.019880 | 0.003325 | 0.442 | 534 |
| frozen_through_2021 | P2_vs_P0 | 2022 | 0.016285 | 0.001552 | 0.466 | 131 |
| frozen_through_2021 | P2_vs_P0 | 2023 | 0.014324 | 0.003945 | 0.429 | 133 |
| frozen_through_2021 | P2_vs_P0 | 2024 | 0.020466 | 0.003429 | 0.433 | 134 |
| frozen_through_2021 | P2_vs_P0 | 2025 | 0.028200 | 0.009200 | 0.441 | 136 |
| frozen_through_2021 | P3_vs_P0 | aggregate | -0.001935 | -0.000574 | 0.528 | 534 |
| frozen_through_2021 | P3_vs_P0 | 2022 | -0.000200 | -0.000542 | 0.557 | 131 |
| frozen_through_2021 | P3_vs_P0 | 2023 | -0.000859 | 0.000850 | 0.436 | 133 |
| frozen_through_2021 | P3_vs_P0 | 2024 | -0.002440 | -0.001042 | 0.560 | 134 |
| frozen_through_2021 | P3_vs_P0 | 2025 | -0.004160 | -0.001267 | 0.559 | 136 |
| frozen_through_2021 | P4_vs_P0 | aggregate | 0.029433 | 0.004054 | 0.461 | 534 |
| frozen_through_2021 | P4_vs_P0 | 2022 | 0.026394 | 0.000839 | 0.489 | 131 |
| frozen_through_2021 | P4_vs_P0 | 2023 | 0.021416 | 0.004954 | 0.466 | 133 |
| frozen_through_2021 | P4_vs_P0 | 2024 | 0.040705 | 0.012281 | 0.396 | 134 |
| frozen_through_2021 | P4_vs_P0 | 2025 | 0.029096 | 0.000000 | 0.493 | 136 |
| frozen_through_2021 | P3_vs_P3-M | aggregate | -0.001874 | -0.000069 | 0.517 | 534 |
| frozen_through_2021 | P3_vs_P3-P | aggregate | -0.023200 | -0.000589 | 0.532 | 534 |
| rolling_origin | P3_vs_P3-M | aggregate | -0.001399 | -0.000170 | 0.537 | 534 |

### Frozen control metrics

| Control | NLL | ΔNLL vs P0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3-M | 4.5080 | -0.000061 | 0.1080 | 19.49 | 19.68 | 0.837 | 71.99 |
| P3-P | 4.5293 | 0.021265 | 0.1106 | 20.03 | 20.37 | 0.831 | 71.53 |

## Coefficient paths and stability

The coefficient artifact reports standardized location coefficients for returning production, incoming usage, each candidate's position-specific value, and its availability indicator. Rolling rows include sign, magnitude, and change from the previous fit.

| Candidate | Group | Value signs | Availability dominates | First | Last | Last/first magnitude |
|---|---|---|---:|---:|---:|---:|
| P1 | DL / EDGE | negative, positive, negative, negative | 4 | 0.0010 | 0.0140 | 13.7618 |
| P2 | LB | negative, negative, negative, negative | 0 | 0.0257 | 0.0173 | 0.6729 |
| P3 | DB | negative, negative, negative, negative | 2 | 0.0069 | 0.0231 | 3.3364 |
| P4 | DL / EDGE | negative, positive, negative, negative | 3 | 0.0038 | 0.0112 | 2.9719 |
| P4 | LB | negative, negative, negative, negative | 1 | 0.0283 | 0.0146 | 0.5165 |
| P4 | DB | negative, negative, negative, negative | 2 | 0.0084 | 0.0220 | 2.6086 |

## Controls and position-group decisions

A group is eligible only if it improves frozen D5 NLL, beats its frozen availability-only control, beats its within-season permutation, improves in every rolling target season, beats its rolling availability-only control in the weighted aggregate, keeps one non-zero coefficient sign, and does not collapse to less than half its first rolling magnitude. The rolling control table reports the target-season pattern as a diagnostic. Availability-coefficient dominance is reported as a stability diagnostic, not used as an automatic promotion veto; the direct availability-only control is the predictive test for a missingness artifact.

| Group candidate | Frozen ΔNLL | Frozen availability control | Rolling availability control (aggregate) | Permutation control | Rolling persistence | Stable value path | Availability dominates (diagnostic) | Passed |
|---|---:|:---:|:---:|:---:|:---:|:---:|---:|:---:|
| P1 (DL / EDGE) | 0.001782 | no | no | no | no | no | 4 / 4 | no |
| P2 (LB) | 0.019880 | no | no | no | no | yes | 0 / 4 | no |
| P3 (DB) | -0.001935 | yes | yes | yes | yes | yes | 2 / 4 | yes |

### Answers

- DL / EDGE: does not pass the full screen.
- LB: does not pass the full screen.
- DB: passes the full screen.
- Combining position groups: P4 does not improve D5 on frozen aggregate NLL.
- Aggregate defensive null: The aggregate null may have obscured a stable position-specific effect, but only the named surviving group is carried forward.

## Artifacts

- `candidate_definitions.json`, `summary.json` — frozen candidates, parity, controls, screens, and recommendation.
- `control_metrics.csv` — frozen availability-only and permutation-control metrics when a primary candidate improves D5.
- `position_coverage.csv`, `training_coverage.csv` — group coverage and rolling training counts.
- `aggregate_reconstruction.csv` — position sum versus existing aggregate sanity check.
- `candidate_summary.csv`, `candidate_annual_metrics.csv` — frozen aggregate and annual metrics.
- `rolling_metrics.csv`, `rolling_aggregate.csv` — rolling annual and weighted aggregate metrics.
- `rolling_control_metrics.csv`, `rolling_control_aggregate.csv` — rolling availability-only control metrics.
- `coefficients.csv` — standardized coefficient paths and source coverage.
- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — paired diagnostics.
- `plots/` — frozen deltas, rolling NLL, coefficients, and paired losses.
