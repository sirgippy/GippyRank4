# Defensive-transfer predictive-value experiment (issue 106)

## Conclusion

Defensive experience improves D5 while impact does not; advance the recorded defensive box-score game-rate proxy to the next stability study, subject to its controls.

The primary D5 control (E2) has aggregate NLL 4.5081. Experience E4 has ΔNLL versus E2 of -0.0009; impact E5 has ΔNLL 0.0032; and E6 has ΔNLL 0.0005 versus E2.
E4's season-level ΔNLL versus E2 is 2022: 0.0043; 2023: 0.0021; 2024: -0.0016; 2025: -0.0080. The experience availability-only and permutation controls are also reported. It does beat the experience availability-only control by mean paired ΔNLL -0.0017 and the within-season permutation by -0.0016; this supports a cautious stability follow-up rather than a production promotion.

The experience measure is prior recorded defensive box-score games divided by source-team games. It is a conservative defensive-experience proxy, not defensive snap share. The impact measure is the frozen position-normalized box-score composite from PR #105, not a direct estimate of player quality.

## Frozen protocol

- Fit through 2021; score unchanged on 2022–2025.
- The production FBS target population, Context preprocessing, H fallback, optimizer retry, penalty, and rank-distribution scoring are unchanged.
- E0, E1, and E2 parity are required before interpreting defensive variants; the study fails if any stored control comparison exceeds tolerance.
- The offensive transfer attachment reads the frozen portal/usage payloads supplied by `--transfer-root`; missing payloads are an error rather than an all-missing feature matrix.
- Defensive values are zero for observed no-incoming rows and for unresolved rows after neutral imputation; an explicit availability indicator separates those cases.
- Partial and no-usable defensive audit rows are never dropped and never treated as observed zeros.
- The defensive audit covers 664 team-seasons: 226 observed and 438 unresolved under this policy.

## Exact primary variants

| Variant | Definition | Defensive inputs |
|---|---|---|
| E0_production_C0 | Existing production Context C0. | none |
| E1_no_returning_production | C-minus-RP control. | none |
| E2_D5 | C-minus-RP plus total returning production and incoming prior offensive usage. | none |
| E3_total_RP_plus_defensive_experience | C-minus-RP plus total returning production and defensive experience; no incoming offensive usage. | transfer_in_prior_defensive_experience_sum |
| E4_D5_plus_defensive_experience | D5 plus defensive experience and its availability indicator. | transfer_in_prior_defensive_experience_sum |
| E5_D5_plus_defensive_impact | D5 plus defensive impact and its availability indicator. | transfer_in_prior_defensive_impact_sum |
| E6_D5_plus_experience_plus_impact | D5 plus defensive experience, defensive impact, and both availability indicators. | transfer_in_prior_defensive_experience_sum, transfer_in_prior_defensive_impact_sum |

## C0 / no-RP / D5 parity and aggregate metrics

| Control | Frozen reference | Maximum absolute metric delta | Tolerance | Status |
|---|---|---:|---:|:---:|
| E0 | E0_production_C0 | 0.00000535 | 0.00100000 | passed |
| E1 | D1_no_returning_production | 0.00000000 | 0.00000100 | passed |
| E2 | D5_total_rp_plus_incoming | 0.00000000 | 0.00000100 | passed |

| Variant | NLL | Δ vs E0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0_production_C0 | 4.5423 | 0.0000 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| E1_no_returning_production | 4.5292 | -0.0132 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| E2_D5 | 4.5081 | -0.0342 | 0.1080 | 19.49 | 19.68 | 0.838 | 71.96 |
| E3_total_RP_plus_defensive_experience | 4.5339 | -0.0085 | 0.1108 | 19.96 | 20.44 | 0.827 | 71.48 |
| E4_D5_plus_defensive_experience | 4.5072 | -0.0351 | 0.1078 | 19.40 | 19.64 | 0.837 | 71.54 |
| E5_D5_plus_defensive_impact | 4.5113 | -0.0310 | 0.1083 | 19.56 | 19.71 | 0.836 | 72.29 |
| E6_D5_plus_experience_plus_impact | 4.5086 | -0.0337 | 0.1079 | 19.50 | 19.66 | 0.839 | 72.08 |
| RP_only_mechanism_control | 4.5354 | -0.0070 | 0.1111 | 20.03 | 20.43 | 0.827 | 72.34 |
| E4-M_experience_availability_only | 4.5089 | -0.0334 | 0.1080 | 19.49 | 19.66 | 0.838 | 72.07 |
| E5-M_impact_availability_only | 4.5089 | -0.0334 | 0.1080 | 19.49 | 19.66 | 0.838 | 72.07 |
| E6-M_availability_only | 4.5089 | -0.0334 | 0.1080 | 19.49 | 19.66 | 0.838 | 72.07 |
| E4P_experience_within_season_permutation | 4.5089 | -0.0335 | 0.1080 | 19.49 | 19.65 | 0.838 | 72.08 |
| E5P_impact_within_season_permutation | 4.5101 | -0.0322 | 0.1081 | 19.53 | 19.68 | 0.835 | 72.02 |
| E6P_defense_within_season_permutation | 4.5103 | -0.0320 | 0.1082 | 19.53 | 19.68 | 0.835 | 72.01 |

## Year-by-year primary metrics

| Season | Variant | NLL | Δ vs E0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | E0_production_C0 | 4.4669 | 0.0000 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 |
| 2023 | E0_production_C0 | 4.4722 | 0.0000 | 0.1044 | 19.28 | 19.55 | 0.867 | 72.12 |
| 2024 | E0_production_C0 | 4.5840 | 0.0000 | 0.1183 | 21.02 | 21.71 | 0.805 | 70.87 |
| 2025 | E0_production_C0 | 4.6425 | 0.0000 | 0.1177 | 21.61 | 22.20 | 0.803 | 72.58 |
| 2022 | E1_no_returning_production | 4.4872 | 0.0203 | 0.1090 | 19.28 | 19.68 | 0.833 | 73.75 |
| 2023 | E1_no_returning_production | 4.5047 | 0.0325 | 0.1078 | 20.10 | 20.41 | 0.861 | 74.68 |
| 2024 | E1_no_returning_production | 4.5539 | -0.0301 | 0.1160 | 20.76 | 20.93 | 0.815 | 73.33 |
| 2025 | E1_no_returning_production | 4.5691 | -0.0734 | 0.1115 | 20.73 | 20.88 | 0.827 | 76.58 |
| 2022 | E2_D5 | 4.4564 | -0.0105 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 |
| 2023 | E2_D5 | 4.4684 | -0.0038 | 0.1040 | 19.16 | 19.40 | 0.871 | 72.35 |
| 2024 | E2_D5 | 4.5492 | -0.0348 | 0.1145 | 20.41 | 20.70 | 0.815 | 70.48 |
| 2025 | E2_D5 | 4.5561 | -0.0864 | 0.1078 | 19.84 | 20.04 | 0.832 | 73.29 |
| 2022 | E3_total_RP_plus_defensive_experience | 4.4746 | 0.0077 | 0.1077 | 18.85 | 19.16 | 0.826 | 71.82 |
| 2023 | E3_total_RP_plus_defensive_experience | 4.4779 | 0.0057 | 0.1051 | 19.52 | 19.97 | 0.869 | 71.75 |
| 2024 | E3_total_RP_plus_defensive_experience | 4.5786 | -0.0055 | 0.1171 | 20.71 | 21.21 | 0.804 | 70.58 |
| 2025 | E3_total_RP_plus_defensive_experience | 4.6017 | -0.0408 | 0.1133 | 20.70 | 21.39 | 0.809 | 71.78 |
| 2022 | E4_D5_plus_defensive_experience | 4.4607 | -0.0062 | 0.1057 | 18.49 | 18.52 | 0.833 | 71.60 |
| 2023 | E4_D5_plus_defensive_experience | 4.4705 | -0.0017 | 0.1045 | 19.38 | 19.82 | 0.870 | 71.85 |
| 2024 | E4_D5_plus_defensive_experience | 4.5477 | -0.0364 | 0.1138 | 20.23 | 20.49 | 0.819 | 70.32 |
| 2025 | E4_D5_plus_defensive_experience | 4.5481 | -0.0944 | 0.1070 | 19.49 | 19.71 | 0.826 | 72.37 |
| 2022 | E5_D5_plus_defensive_impact | 4.4530 | -0.0139 | 0.1049 | 18.34 | 18.28 | 0.838 | 71.98 |
| 2023 | E5_D5_plus_defensive_impact | 4.4706 | -0.0016 | 0.1043 | 19.35 | 19.58 | 0.869 | 72.43 |
| 2024 | E5_D5_plus_defensive_impact | 4.5531 | -0.0310 | 0.1150 | 20.50 | 20.72 | 0.816 | 70.77 |
| 2025 | E5_D5_plus_defensive_impact | 4.5662 | -0.0763 | 0.1089 | 20.03 | 20.23 | 0.823 | 73.96 |
| 2022 | E6_D5_plus_experience_plus_impact | 4.4574 | -0.0095 | 0.1053 | 18.43 | 18.35 | 0.833 | 72.02 |
| 2023 | E6_D5_plus_experience_plus_impact | 4.4732 | 0.0010 | 0.1045 | 19.39 | 19.70 | 0.872 | 72.39 |
| 2024 | E6_D5_plus_experience_plus_impact | 4.5507 | -0.0333 | 0.1146 | 20.51 | 20.73 | 0.822 | 70.73 |
| 2025 | E6_D5_plus_experience_plus_impact | 4.5511 | -0.0914 | 0.1071 | 19.66 | 19.82 | 0.831 | 73.16 |

## Paired NLL diagnostics

Negative ΔNLL means the candidate has lower loss.

| Comparison | Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |
|---|---|---:|---:|---:|---:|
| E4_vs_E2 | aggregate | 534 | -0.0009 | -0.0002 | 0.506 |
| E4_vs_E2 | 2022 | 131 | 0.0043 | -0.0007 | 0.534 |
| E4_vs_E2 | 2023 | 133 | 0.0021 | 0.0011 | 0.444 |
| E4_vs_E2 | 2024 | 134 | -0.0016 | -0.0006 | 0.530 |
| E4_vs_E2 | 2025 | 136 | -0.0080 | -0.0005 | 0.515 |
| E5_vs_E2 | aggregate | 534 | 0.0032 | 0.0000 | 0.491 |
| E5_vs_E2 | 2022 | 131 | -0.0034 | -0.0012 | 0.542 |
| E5_vs_E2 | 2023 | 133 | 0.0022 | 0.0013 | 0.421 |
| E5_vs_E2 | 2024 | 134 | 0.0038 | -0.0017 | 0.530 |
| E5_vs_E2 | 2025 | 136 | 0.0101 | 0.0006 | 0.471 |
| E6_vs_E4 | aggregate | 534 | 0.0014 | 0.0002 | 0.466 |
| E6_vs_E4 | 2022 | 131 | -0.0033 | 0.0000 | 0.489 |
| E6_vs_E4 | 2023 | 133 | 0.0027 | -0.0003 | 0.519 |
| E6_vs_E4 | 2024 | 134 | 0.0030 | 0.0006 | 0.455 |
| E6_vs_E4 | 2025 | 136 | 0.0029 | 0.0006 | 0.404 |
| E6_vs_E5 | aggregate | 534 | -0.0027 | 0.0000 | 0.489 |
| E6_vs_E5 | 2022 | 131 | 0.0044 | -0.0001 | 0.519 |
| E6_vs_E5 | 2023 | 133 | 0.0026 | 0.0000 | 0.489 |
| E6_vs_E5 | 2024 | 134 | -0.0024 | 0.0005 | 0.470 |
| E6_vs_E5 | 2025 | 136 | -0.0152 | 0.0001 | 0.478 |
| E3_vs_RP_only | aggregate | 534 | -0.0015 | -0.0007 | 0.534 |
| E3_vs_RP_only | 2022 | 131 | 0.0052 | -0.0008 | 0.542 |
| E3_vs_RP_only | 2023 | 133 | 0.0100 | 0.0005 | 0.444 |
| E3_vs_RP_only | 2024 | 134 | -0.0018 | -0.0009 | 0.582 |
| E3_vs_RP_only | 2025 | 136 | -0.0188 | -0.0013 | 0.566 |
| E4_vs_E4M | aggregate | 534 | -0.0017 | 0.0000 | 0.487 |
| E4_vs_E4M | 2022 | 131 | 0.0061 | 0.0004 | 0.458 |
| E4_vs_E4M | 2023 | 133 | 0.0015 | 0.0000 | 0.496 |
| E4_vs_E4M | 2024 | 134 | -0.0032 | 0.0001 | 0.485 |
| E4_vs_E4M | 2025 | 136 | -0.0108 | -0.0001 | 0.507 |
| E5_vs_E5M | aggregate | 534 | 0.0024 | -0.0000 | 0.504 |
| E5_vs_E5M | 2022 | 131 | -0.0016 | -0.0001 | 0.565 |
| E5_vs_E5M | 2023 | 133 | 0.0016 | -0.0000 | 0.549 |
| E5_vs_E5M | 2024 | 134 | 0.0022 | 0.0000 | 0.493 |
| E5_vs_E5M | 2025 | 136 | 0.0073 | 0.0001 | 0.412 |
| E6_vs_E6M | aggregate | 534 | -0.0003 | 0.0001 | 0.485 |
| E6_vs_E6M | 2022 | 131 | 0.0028 | -0.0000 | 0.504 |
| E6_vs_E6M | 2023 | 133 | 0.0042 | 0.0000 | 0.489 |
| E6_vs_E6M | 2024 | 134 | -0.0001 | 0.0008 | 0.448 |
| E6_vs_E6M | 2025 | 136 | -0.0079 | -0.0001 | 0.500 |
| E4M_vs_E2 | aggregate | 534 | 0.0008 | 0.0000 | 0.494 |
| E4M_vs_E2 | 2022 | 131 | -0.0018 | -0.0012 | 0.542 |
| E4M_vs_E2 | 2023 | 133 | 0.0006 | 0.0013 | 0.436 |
| E4M_vs_E2 | 2024 | 134 | 0.0016 | -0.0018 | 0.537 |
| E4M_vs_E2 | 2025 | 136 | 0.0028 | 0.0009 | 0.463 |
| E5M_vs_E2 | aggregate | 534 | 0.0008 | 0.0000 | 0.494 |
| E5M_vs_E2 | 2022 | 131 | -0.0018 | -0.0012 | 0.542 |
| E5M_vs_E2 | 2023 | 133 | 0.0006 | 0.0013 | 0.436 |
| E5M_vs_E2 | 2024 | 134 | 0.0016 | -0.0018 | 0.537 |
| E5M_vs_E2 | 2025 | 136 | 0.0028 | 0.0009 | 0.463 |
| E6M_vs_E2 | aggregate | 534 | 0.0008 | 0.0000 | 0.496 |
| E6M_vs_E2 | 2022 | 131 | -0.0018 | -0.0012 | 0.550 |
| E6M_vs_E2 | 2023 | 133 | 0.0006 | 0.0014 | 0.436 |
| E6M_vs_E2 | 2024 | 134 | 0.0016 | -0.0018 | 0.537 |
| E6M_vs_E2 | 2025 | 136 | 0.0028 | 0.0009 | 0.463 |
| E4_vs_E4P | aggregate | 534 | -0.0016 | 0.0001 | 0.485 |
| E4_vs_E4P | 2022 | 131 | 0.0067 | 0.0005 | 0.458 |
| E4_vs_E4P | 2023 | 133 | 0.0012 | 0.0000 | 0.489 |
| E4_vs_E4P | 2024 | 134 | -0.0030 | 0.0002 | 0.478 |
| E4_vs_E4P | 2025 | 136 | -0.0111 | -0.0001 | 0.515 |
| E5_vs_E5P | aggregate | 534 | 0.0012 | 0.0000 | 0.493 |
| E5_vs_E5P | 2022 | 131 | -0.0050 | -0.0001 | 0.511 |
| E5_vs_E5P | 2023 | 133 | 0.0028 | -0.0001 | 0.526 |
| E5_vs_E5P | 2024 | 134 | 0.0008 | 0.0001 | 0.470 |
| E5_vs_E5P | 2025 | 136 | 0.0060 | 0.0000 | 0.463 |
| E6_vs_E6P | aggregate | 534 | -0.0017 | 0.0000 | 0.489 |
| E6_vs_E6P | 2022 | 131 | -0.0002 | -0.0002 | 0.511 |
| E6_vs_E6P | 2023 | 133 | 0.0054 | 0.0002 | 0.481 |
| E6_vs_E6P | 2024 | 134 | -0.0016 | 0.0005 | 0.448 |
| E6_vs_E6P | 2025 | 136 | -0.0104 | -0.0002 | 0.515 |

## Coefficient diagnostics

Numeric coefficients are standardized training-fit location coefficients. Availability indicators are explicit observed/unresolved controls. The defensive inputs enter location only, matching the frozen Context experiment; coefficients are not causal effects.

| Variant | Feature | Included | Training source available / unavailable | Location β | Location missingness |
|---|---|:---:|---:|---:|---:|
| E0_production_C0 | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.1533 | 0.0035 |
| E0_production_C0 | transfer_in_prior_usage_sum | False | None / None | n/a | n/a |
| E0_production_C0 | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E0_production_C0 | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E0_production_C0 | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E0_production_C0 | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E1_no_returning_production | returning_pct_ppa | False | None / None | n/a | n/a |
| E1_no_returning_production | transfer_in_prior_usage_sum | False | None / None | n/a | n/a |
| E1_no_returning_production | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E1_no_returning_production | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E1_no_returning_production | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E1_no_returning_production | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E2_D5 | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2042 | 0.0063 |
| E2_D5 | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0374 | -0.0713 |
| E2_D5 | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2010 | 0.0055 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_usage_sum | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_experience_sum | True | 52.0 / 2164.0 | -0.0242 | 0.0000 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_experience_available | True | 52.0 / 2164.0 | 0.0333 | 0.0000 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E4_D5_plus_defensive_experience | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2031 | 0.0055 |
| E4_D5_plus_defensive_experience | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0330 | -0.0187 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_experience_sum | True | 52.0 / 2164.0 | -0.0164 | 0.0000 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_experience_available | True | 52.0 / 2164.0 | 0.0284 | 0.0000 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2047 | 0.0063 |
| E5_D5_plus_defensive_impact | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0356 | -0.0226 |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_impact_sum | True | 52.0 / 2164.0 | 0.0045 | 0.0000 |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_impact_available | True | 52.0 / 2164.0 | 0.0166 | 0.0000 |
| E6_D5_plus_experience_plus_impact | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2043 | 0.0060 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0305 | -0.0120 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_experience_sum | True | 52.0 / 2164.0 | -0.0309 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_impact_sum | True | 52.0 / 2164.0 | 0.0194 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_experience_available | True | 52.0 / 2164.0 | 0.0165 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_impact_available | True | 52.0 / 2164.0 | 0.0165 | 0.0000 |
| E4-M_experience_availability_only | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2042 | 0.0061 |
| E4-M_experience_availability_only | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0357 | -0.0237 |
| E4-M_experience_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E4-M_experience_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E4-M_experience_availability_only | transfer_in_prior_defensive_experience_available | True | 52.0 / 2164.0 | 0.0176 | 0.0000 |
| E4-M_experience_availability_only | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2042 | 0.0061 |
| E5-M_impact_availability_only | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0357 | -0.0237 |
| E5-M_impact_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_impact_available | True | 52.0 / 2164.0 | 0.0176 | 0.0000 |
| E6-M_availability_only | returning_pct_ppa | True | 1021.0 / 1195.0 | -0.2042 | 0.0061 |
| E6-M_availability_only | transfer_in_prior_usage_sum | True | 94.0 / 2122.0 | -0.0356 | -0.0234 |
| E6-M_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E6-M_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E6-M_availability_only | transfer_in_prior_defensive_experience_available | True | 52.0 / 2164.0 | 0.0088 | 0.0000 |
| E6-M_availability_only | transfer_in_prior_defensive_impact_available | True | 52.0 / 2164.0 | 0.0088 | 0.0000 |

## Missingness and permutation controls

E4 versus its availability-only control: mean paired ΔNLL -0.0017. E5 versus its availability-only control: 0.0024. E6 versus its availability-only control: -0.0003.
Within-season permutations preserve season, availability pattern, and observed defensive-value distributions while destroying team assignment. Real versus permuted mean paired ΔNLL is -0.0016 for experience, 0.0012 for impact, and -0.0017 for E6.

## Experience / impact collinearity

The E6 training panel has 52 rows with both defensive values observed and pairwise correlation 0.5980. E6 coefficient instability, if present, is therefore treated as evidence of redundancy rather than repaired through outcome-tuned regularization.

## Defensive feature semantics and limitations

- Experience is recorded defensive box-score game rate, not defensive snap share or games played.
- Impact is the frozen DL/EDGE, LB, and DB position-normalized log1p box-score composite from PR #105.
- The defensive audit remains a retrospective research oracle with fail-closed identity/source coverage.
- This ticket does not run rolling-origin stability, position decomposition, feature redesign, or production snapshot engineering.

## Artifacts

- `variant_definitions.json`, `summary.json` — frozen configuration, parity, recommendation, and provenance.
- `candidate_annual_metrics.csv`, `candidate_summary.csv` — primary, auxiliary, missingness-control, and permutation metrics.
- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — aggregate/year and team-season paired diagnostics.
- `standardized_coefficients.csv`, `collinearity.json`, `plots/` — structural diagnostics.
