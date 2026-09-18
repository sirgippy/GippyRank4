# Defensive-transfer predictive-value experiment (issue 106)

## Conclusion

Defensive experience improves D5 while impact does not; advance the recorded defensive box-score game-rate proxy to the next stability study, subject to its controls.

The primary D5 control (E2) has aggregate NLL 4.5354. Experience E4 has ΔNLL versus E2 of -0.0015; impact E5 has ΔNLL 0.0072; and E6 has ΔNLL -0.0001 versus E2.
E4 is worse than E2 in 2022 and 2023 but better in 2024 and 2025, so the small aggregate gain is not yet directionally stable. It does beat the experience availability-only control by mean paired ΔNLL -0.0049 and the within-season permutation by -0.0048; this supports a cautious stability follow-up rather than a production promotion.

The experience measure is prior recorded defensive box-score games divided by source-team games. It is a conservative defensive-experience proxy, not defensive snap share. The impact measure is the frozen position-normalized box-score composite from PR #105, not a direct estimate of player quality.

## Frozen protocol

- Fit through 2021; score unchanged on 2022–2025.
- The production FBS target population, Context preprocessing, H fallback, optimizer retry, penalty, and rank-distribution scoring are unchanged.
- E0 parity is required before interpreting defensive variants.
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

E0 parity passed; maximum absolute stored-metric delta 0.000005.

| Variant | NLL | Δ vs E0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0_production_C0 | 4.5423 | 0.0000 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| E1_no_returning_production | 4.5292 | -0.0132 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| E2_D5 | 4.5354 | -0.0070 | 0.1111 | 20.03 | 20.43 | 0.827 | 72.34 |
| E3_total_RP_plus_defensive_experience | 4.5339 | -0.0085 | 0.1108 | 19.96 | 20.44 | 0.827 | 71.48 |
| E4_D5_plus_defensive_experience | 4.5339 | -0.0085 | 0.1108 | 19.96 | 20.44 | 0.827 | 71.48 |
| E5_D5_plus_defensive_impact | 4.5426 | 0.0002 | 0.1119 | 20.17 | 20.58 | 0.827 | 72.57 |
| E6_D5_plus_experience_plus_impact | 4.5353 | -0.0071 | 0.1110 | 20.03 | 20.43 | 0.831 | 72.13 |
| RP_only_mechanism_control | 4.5354 | -0.0070 | 0.1111 | 20.03 | 20.43 | 0.827 | 72.34 |
| E4-M_experience_availability_only | 4.5388 | -0.0035 | 0.1114 | 20.08 | 20.48 | 0.828 | 72.37 |
| E5-M_impact_availability_only | 4.5388 | -0.0035 | 0.1114 | 20.08 | 20.48 | 0.828 | 72.37 |
| E6-M_availability_only | 4.5388 | -0.0035 | 0.1114 | 20.08 | 20.48 | 0.828 | 72.37 |
| E4P_experience_within_season_permutation | 4.5387 | -0.0036 | 0.1114 | 20.09 | 20.50 | 0.828 | 72.37 |
| E5P_impact_within_season_permutation | 4.5388 | -0.0036 | 0.1114 | 20.08 | 20.49 | 0.828 | 72.38 |
| E6P_defense_within_season_permutation | 4.5385 | -0.0038 | 0.1114 | 20.08 | 20.49 | 0.828 | 72.37 |

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
| 2022 | E2_D5 | 4.4694 | 0.0025 | 0.1074 | 18.76 | 18.90 | 0.825 | 72.18 |
| 2023 | E2_D5 | 4.4679 | -0.0043 | 0.1035 | 19.09 | 19.37 | 0.873 | 72.57 |
| 2024 | E2_D5 | 4.5804 | -0.0036 | 0.1180 | 20.95 | 21.54 | 0.803 | 71.21 |
| 2025 | E2_D5 | 4.6205 | -0.0220 | 0.1152 | 21.28 | 21.85 | 0.807 | 73.39 |
| 2022 | E3_total_RP_plus_defensive_experience | 4.4746 | 0.0077 | 0.1077 | 18.85 | 19.16 | 0.826 | 71.82 |
| 2023 | E3_total_RP_plus_defensive_experience | 4.4779 | 0.0057 | 0.1051 | 19.52 | 19.97 | 0.869 | 71.75 |
| 2024 | E3_total_RP_plus_defensive_experience | 4.5786 | -0.0055 | 0.1171 | 20.71 | 21.21 | 0.804 | 70.58 |
| 2025 | E3_total_RP_plus_defensive_experience | 4.6017 | -0.0408 | 0.1133 | 20.70 | 21.39 | 0.809 | 71.78 |
| 2022 | E4_D5_plus_defensive_experience | 4.4746 | 0.0077 | 0.1077 | 18.85 | 19.16 | 0.826 | 71.82 |
| 2023 | E4_D5_plus_defensive_experience | 4.4779 | 0.0057 | 0.1051 | 19.52 | 19.97 | 0.868 | 71.75 |
| 2024 | E4_D5_plus_defensive_experience | 4.5786 | -0.0055 | 0.1171 | 20.71 | 21.20 | 0.804 | 70.58 |
| 2025 | E4_D5_plus_defensive_experience | 4.6017 | -0.0408 | 0.1133 | 20.70 | 21.38 | 0.809 | 71.77 |
| 2022 | E5_D5_plus_defensive_impact | 4.4673 | 0.0004 | 0.1070 | 18.73 | 18.89 | 0.834 | 72.23 |
| 2023 | E5_D5_plus_defensive_impact | 4.4752 | 0.0030 | 0.1044 | 19.32 | 19.56 | 0.871 | 72.66 |
| 2024 | E5_D5_plus_defensive_impact | 4.5868 | 0.0028 | 0.1188 | 21.03 | 21.71 | 0.801 | 71.40 |
| 2025 | E5_D5_plus_defensive_impact | 4.6373 | -0.0052 | 0.1170 | 21.54 | 22.10 | 0.801 | 73.96 |
| 2022 | E6_D5_plus_experience_plus_impact | 4.4707 | 0.0038 | 0.1073 | 18.74 | 18.91 | 0.831 | 72.32 |
| 2023 | E6_D5_plus_experience_plus_impact | 4.4822 | 0.0100 | 0.1052 | 19.47 | 19.84 | 0.871 | 72.46 |
| 2024 | E6_D5_plus_experience_plus_impact | 4.5817 | -0.0023 | 0.1181 | 20.99 | 21.49 | 0.807 | 71.04 |
| 2025 | E6_D5_plus_experience_plus_impact | 4.6036 | -0.0389 | 0.1132 | 20.85 | 21.42 | 0.817 | 72.68 |

## Paired NLL diagnostics

Negative ΔNLL means the candidate has lower loss.

| Comparison | Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |
|---|---|---:|---:|---:|---:|
| E4_vs_E2 | aggregate | 534 | -0.0015 | -0.0007 | 0.536 |
| E4_vs_E2 | 2022 | 131 | 0.0052 | -0.0008 | 0.542 |
| E4_vs_E2 | 2023 | 133 | 0.0100 | 0.0005 | 0.451 |
| E4_vs_E2 | 2024 | 134 | -0.0018 | -0.0009 | 0.582 |
| E4_vs_E2 | 2025 | 136 | -0.0188 | -0.0013 | 0.566 |
| E5_vs_E2 | aggregate | 534 | 0.0072 | -0.0003 | 0.530 |
| E5_vs_E2 | 2022 | 131 | -0.0021 | -0.0008 | 0.550 |
| E5_vs_E2 | 2023 | 133 | 0.0073 | 0.0002 | 0.481 |
| E5_vs_E2 | 2024 | 134 | 0.0064 | -0.0006 | 0.552 |
| E5_vs_E2 | 2025 | 136 | 0.0168 | -0.0004 | 0.537 |
| E6_vs_E4 | aggregate | 534 | 0.0014 | 0.0000 | 0.493 |
| E6_vs_E4 | 2022 | 131 | -0.0039 | -0.0003 | 0.565 |
| E6_vs_E4 | 2023 | 133 | 0.0043 | 0.0000 | 0.496 |
| E6_vs_E4 | 2024 | 134 | 0.0032 | 0.0000 | 0.493 |
| E6_vs_E4 | 2025 | 136 | 0.0019 | 0.0001 | 0.419 |
| E6_vs_E5 | aggregate | 534 | -0.0073 | -0.0002 | 0.532 |
| E6_vs_E5 | 2022 | 131 | 0.0034 | -0.0003 | 0.542 |
| E6_vs_E5 | 2023 | 133 | 0.0070 | 0.0001 | 0.451 |
| E6_vs_E5 | 2024 | 134 | -0.0051 | -0.0003 | 0.545 |
| E6_vs_E5 | 2025 | 136 | -0.0337 | -0.0003 | 0.588 |
| E3_vs_RP_only | aggregate | 534 | -0.0015 | -0.0007 | 0.534 |
| E3_vs_RP_only | 2022 | 131 | 0.0052 | -0.0008 | 0.542 |
| E3_vs_RP_only | 2023 | 133 | 0.0100 | 0.0005 | 0.444 |
| E3_vs_RP_only | 2024 | 134 | -0.0018 | -0.0009 | 0.582 |
| E3_vs_RP_only | 2025 | 136 | -0.0188 | -0.0013 | 0.566 |
| E4_vs_E4M | aggregate | 534 | -0.0049 | -0.0001 | 0.541 |
| E4_vs_E4M | 2022 | 131 | 0.0064 | 0.0000 | 0.489 |
| E4_vs_E4M | 2023 | 133 | 0.0048 | -0.0000 | 0.504 |
| E4_vs_E4M | 2024 | 134 | -0.0054 | -0.0003 | 0.590 |
| E4_vs_E4M | 2025 | 136 | -0.0248 | -0.0004 | 0.581 |
| E5_vs_E5M | aggregate | 534 | 0.0038 | 0.0000 | 0.489 |
| E5_vs_E5M | 2022 | 131 | -0.0009 | -0.0001 | 0.565 |
| E5_vs_E5M | 2023 | 133 | 0.0021 | -0.0000 | 0.519 |
| E5_vs_E5M | 2024 | 134 | 0.0028 | 0.0001 | 0.440 |
| E5_vs_E5M | 2025 | 136 | 0.0108 | 0.0001 | 0.434 |
| E6_vs_E6M | aggregate | 534 | -0.0035 | -0.0001 | 0.519 |
| E6_vs_E6M | 2022 | 131 | 0.0025 | -0.0002 | 0.519 |
| E6_vs_E6M | 2023 | 133 | 0.0091 | 0.0001 | 0.444 |
| E6_vs_E6M | 2024 | 134 | -0.0022 | -0.0002 | 0.530 |
| E6_vs_E6M | 2025 | 136 | -0.0229 | -0.0003 | 0.581 |
| E4M_vs_E2 | aggregate | 534 | 0.0034 | -0.0003 | 0.528 |
| E4M_vs_E2 | 2022 | 131 | -0.0012 | -0.0007 | 0.542 |
| E4M_vs_E2 | 2023 | 133 | 0.0052 | 0.0001 | 0.481 |
| E4M_vs_E2 | 2024 | 134 | 0.0036 | -0.0006 | 0.552 |
| E4M_vs_E2 | 2025 | 136 | 0.0060 | -0.0005 | 0.537 |
| E5M_vs_E2 | aggregate | 534 | 0.0034 | -0.0003 | 0.528 |
| E5M_vs_E2 | 2022 | 131 | -0.0012 | -0.0007 | 0.542 |
| E5M_vs_E2 | 2023 | 133 | 0.0052 | 0.0001 | 0.481 |
| E5M_vs_E2 | 2024 | 134 | 0.0036 | -0.0006 | 0.552 |
| E5M_vs_E2 | 2025 | 136 | 0.0060 | -0.0005 | 0.537 |
| E6M_vs_E2 | aggregate | 534 | 0.0034 | -0.0003 | 0.528 |
| E6M_vs_E2 | 2022 | 131 | -0.0012 | -0.0007 | 0.542 |
| E6M_vs_E2 | 2023 | 133 | 0.0052 | 0.0001 | 0.481 |
| E6M_vs_E2 | 2024 | 134 | 0.0036 | -0.0006 | 0.552 |
| E6M_vs_E2 | 2025 | 136 | 0.0060 | -0.0005 | 0.537 |
| E4_vs_E4P | aggregate | 534 | -0.0048 | -0.0002 | 0.537 |
| E4_vs_E4P | 2022 | 131 | 0.0059 | 0.0000 | 0.489 |
| E4_vs_E4P | 2023 | 133 | 0.0050 | 0.0000 | 0.489 |
| E4_vs_E4P | 2024 | 134 | -0.0055 | -0.0003 | 0.590 |
| E4_vs_E4P | 2025 | 136 | -0.0242 | -0.0004 | 0.581 |
| E5_vs_E5P | aggregate | 534 | 0.0038 | 0.0000 | 0.487 |
| E5_vs_E5P | 2022 | 131 | -0.0007 | -0.0001 | 0.573 |
| E5_vs_E5P | 2023 | 133 | 0.0020 | -0.0000 | 0.511 |
| E5_vs_E5P | 2024 | 134 | 0.0029 | 0.0001 | 0.448 |
| E5_vs_E5P | 2025 | 136 | 0.0107 | 0.0001 | 0.419 |
| E6_vs_E6P | aggregate | 534 | -0.0032 | -0.0001 | 0.522 |
| E6_vs_E6P | 2022 | 131 | 0.0026 | -0.0002 | 0.534 |
| E6_vs_E6P | 2023 | 133 | 0.0090 | 0.0001 | 0.451 |
| E6_vs_E6P | 2024 | 134 | -0.0021 | -0.0000 | 0.530 |
| E6_vs_E6P | 2025 | 136 | -0.0219 | -0.0002 | 0.574 |

## Coefficient diagnostics

Numeric coefficients are standardized training-fit location coefficients. Availability indicators are explicit observed/unresolved controls. The defensive inputs enter location only, matching the frozen Context experiment; coefficients are not causal effects.

| Variant | Feature | Included | Training observed / missing | Location β | Location missingness |
|---|---|:---:|---:|---:|---:|
| E0_production_C0 | returning_pct_ppa | True | 1021 / 1195 | -0.1533 | 0.0035 |
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
| E2_D5 | returning_pct_ppa | True | 1021 / 1195 | -0.2016 | 0.0066 |
| E2_D5 | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0367 |
| E2_D5 | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E2_D5 | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | returning_pct_ppa | True | 1021 / 1195 | -0.2010 | 0.0055 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_usage_sum | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_experience_sum | True | 2216 / 0 | -0.0242 | 0.0000 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_experience_available | True | 2216 / 0 | 0.0333 | 0.0000 |
| E3_total_RP_plus_defensive_experience | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E4_D5_plus_defensive_experience | returning_pct_ppa | True | 1021 / 1195 | -0.2010 | 0.0054 |
| E4_D5_plus_defensive_experience | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0383 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_experience_sum | True | 2216 / 0 | -0.0242 | 0.0000 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_experience_available | True | 2216 / 0 | 0.0333 | 0.0000 |
| E4_D5_plus_defensive_experience | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | returning_pct_ppa | True | 1021 / 1195 | -0.2030 | 0.0062 |
| E5_D5_plus_defensive_impact | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0384 |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_impact_sum | True | 2216 / 0 | 0.0048 | 0.0000 |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E5_D5_plus_defensive_impact | transfer_in_prior_defensive_impact_available | True | 2216 / 0 | 0.0168 | 0.0000 |
| E6_D5_plus_experience_plus_impact | returning_pct_ppa | True | 1021 / 1195 | -0.2029 | 0.0061 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0380 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_experience_sum | True | 2216 / 0 | -0.0421 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_impact_sum | True | 2216 / 0 | 0.0249 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_experience_available | True | 2216 / 0 | 0.0190 | 0.0000 |
| E6_D5_plus_experience_plus_impact | transfer_in_prior_defensive_impact_available | True | 2216 / 0 | 0.0190 | 0.0000 |
| E4-M_experience_availability_only | returning_pct_ppa | True | 1021 / 1195 | -0.2024 | 0.0061 |
| E4-M_experience_availability_only | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0384 |
| E4-M_experience_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E4-M_experience_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E4-M_experience_availability_only | transfer_in_prior_defensive_experience_available | True | 2216 / 0 | 0.0180 | 0.0000 |
| E4-M_experience_availability_only | transfer_in_prior_defensive_impact_available | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | returning_pct_ppa | True | 1021 / 1195 | -0.2024 | 0.0061 |
| E5-M_impact_availability_only | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0384 |
| E5-M_impact_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_experience_available | False | None / None | n/a | n/a |
| E5-M_impact_availability_only | transfer_in_prior_defensive_impact_available | True | 2216 / 0 | 0.0180 | 0.0000 |
| E6-M_availability_only | returning_pct_ppa | True | 1021 / 1195 | -0.2025 | 0.0061 |
| E6-M_availability_only | transfer_in_prior_usage_sum | True | 0 / 2216 | 0.0000 | -0.0384 |
| E6-M_availability_only | transfer_in_prior_defensive_experience_sum | False | None / None | n/a | n/a |
| E6-M_availability_only | transfer_in_prior_defensive_impact_sum | False | None / None | n/a | n/a |
| E6-M_availability_only | transfer_in_prior_defensive_experience_available | True | 2216 / 0 | 0.0090 | 0.0000 |
| E6-M_availability_only | transfer_in_prior_defensive_impact_available | True | 2216 / 0 | 0.0090 | 0.0000 |

## Missingness and permutation controls

E4 versus its availability-only control: mean paired ΔNLL -0.0049. E5 versus its availability-only control: 0.0038. E6 versus its availability-only control: -0.0035.
Within-season permutations preserve season, availability pattern, and observed defensive-value distributions while destroying team assignment. Real versus permuted mean paired ΔNLL is -0.0048 for experience, 0.0038 for impact, and -0.0032 for E6.

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
