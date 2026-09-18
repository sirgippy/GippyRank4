# Transfer-production signal decomposition (issue 96)

## Conclusion

D0 reproduced the stored production C 1.2 panel within the configured 0.001 tolerance. The primary candidate D5 (total RP + incoming prior production) has held-out ΔNLL -0.0342 versus D0; D8 (existing C10) has ΔNLL -0.0263.
Incoming production conditional on total RP (D5 − D2) has mean paired ΔNLL -0.0273; outgoing production beyond total RP + incoming (D7 − D5) has mean paired ΔNLL 0.0019 and improves 0.496 of team-seasons. D5 versus D8 has mean paired ΔNLL -0.0080.
Incoming production also improves the no-RP control (D3 − D1 ΔNLL -0.0029), while outgoing production alone does not (D4 − D1 ΔNLL 0.0018). Outgoing production has a small conditional comparison against total RP (D6 − D2 ΔNLL -0.0052), but adding it after both RP and incoming production is slightly worse. The recommended representation for the subsequent coefficient-stability study is therefore D5: total RP plus incoming prior transfer production, subject to the oracle caveat.
These are retrospective-oracle mechanism results, not a production-safe feature recommendation.

## Exact predeclared variants

| Variant | Definition | Added continuity features |
|---|---|---|
| D0_C0_full | Production Context C 1.2; the parity control. | none |
| D1_no_returning_production | C-minus-RP; remove all returning-production features. | none |
| D2_total_rp_only | C-minus-RP plus total returning production only. | returning_pct_ppa |
| D3_incoming_transfer_production_only | C-minus-RP plus incoming prior transfer production only. | transfer_in_prior_usage_sum |
| D4_outgoing_transfer_production_only | C-minus-RP plus outgoing prior transfer production only. | transfer_out_prior_usage_sum |
| D5_total_rp_plus_incoming | C-minus-RP plus total RP and incoming prior transfer production. | returning_pct_ppa, transfer_in_prior_usage_sum |
| D6_total_rp_plus_outgoing | C-minus-RP plus total RP and outgoing prior transfer production. | returning_pct_ppa, transfer_out_prior_usage_sum |
| D7_total_rp_plus_incoming_plus_outgoing | C-minus-RP plus total RP, incoming, and outgoing prior production. | returning_pct_ppa, transfer_in_prior_usage_sum, transfer_out_prior_usage_sum |
| D8_existing_c10 | Existing C10: full RP family plus incoming and net prior production. | returning_pct_ppa, returning_pct_passing_ppa, returning_pct_receiving_ppa, returning_pct_rushing_ppa, transfer_in_prior_usage_sum, transfer_net_prior_usage |

## C0 parity check

D0 is the existing C 1.2 control. The check passed with maximum absolute metric difference 0.000005.

## Aggregate 2022–2025 comparison

| Variant | NLL | Δ vs D0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0_C0_full | 4.5423 | 0.0000 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| D1_no_returning_production | 4.5292 | -0.0132 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| D2_total_rp_only | 4.5354 | -0.0070 | 0.1111 | 20.03 | 20.43 | 0.827 | 72.34 |
| D3_incoming_transfer_production_only | 4.5263 | -0.0160 | 0.1110 | 20.12 | 20.44 | 0.833 | 74.01 |
| D4_outgoing_transfer_production_only | 4.5309 | -0.0114 | 0.1113 | 20.26 | 20.51 | 0.833 | 74.64 |
| D5_total_rp_plus_incoming | 4.5081 | -0.0342 | 0.1080 | 19.49 | 19.68 | 0.838 | 71.96 |
| D6_total_rp_plus_outgoing | 4.5302 | -0.0121 | 0.1104 | 19.99 | 20.24 | 0.823 | 73.08 |
| D7_total_rp_plus_incoming_plus_outgoing | 4.5099 | -0.0324 | 0.1082 | 19.59 | 19.68 | 0.834 | 72.47 |
| D8_existing_c10 | 4.5160 | -0.0263 | 0.1089 | 19.66 | 19.81 | 0.833 | 71.74 |

## Year-by-year metrics

| Season | Variant | NLL | Δ vs D0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | D0_C0_full | 4.4669 | 0.0000 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 |
| 2023 | D0_C0_full | 4.4722 | 0.0000 | 0.1044 | 19.28 | 19.55 | 0.867 | 72.12 |
| 2024 | D0_C0_full | 4.5840 | 0.0000 | 0.1183 | 21.02 | 21.71 | 0.805 | 70.87 |
| 2025 | D0_C0_full | 4.6425 | 0.0000 | 0.1177 | 21.61 | 22.20 | 0.803 | 72.58 |
| 2022 | D1_no_returning_production | 4.4872 | 0.0203 | 0.1090 | 19.28 | 19.68 | 0.833 | 73.75 |
| 2023 | D1_no_returning_production | 4.5047 | 0.0325 | 0.1078 | 20.10 | 20.41 | 0.861 | 74.68 |
| 2024 | D1_no_returning_production | 4.5539 | -0.0301 | 0.1160 | 20.76 | 20.93 | 0.815 | 73.33 |
| 2025 | D1_no_returning_production | 4.5691 | -0.0734 | 0.1115 | 20.73 | 20.88 | 0.827 | 76.58 |
| 2022 | D2_total_rp_only | 4.4694 | 0.0025 | 0.1074 | 18.76 | 18.90 | 0.825 | 72.18 |
| 2023 | D2_total_rp_only | 4.4679 | -0.0043 | 0.1035 | 19.09 | 19.38 | 0.874 | 72.58 |
| 2024 | D2_total_rp_only | 4.5804 | -0.0036 | 0.1180 | 20.95 | 21.54 | 0.803 | 71.21 |
| 2025 | D2_total_rp_only | 4.6205 | -0.0220 | 0.1152 | 21.28 | 21.85 | 0.807 | 73.39 |
| 2022 | D3_incoming_transfer_production_only | 4.4891 | 0.0222 | 0.1090 | 19.11 | 19.54 | 0.834 | 73.31 |
| 2023 | D3_incoming_transfer_production_only | 4.5125 | 0.0403 | 0.1093 | 20.16 | 20.44 | 0.855 | 74.23 |
| 2024 | D3_incoming_transfer_production_only | 4.5506 | -0.0335 | 0.1156 | 20.64 | 20.85 | 0.807 | 72.52 |
| 2025 | D3_incoming_transfer_production_only | 4.5517 | -0.0908 | 0.1099 | 20.54 | 20.88 | 0.835 | 75.96 |
| 2022 | D4_outgoing_transfer_production_only | 4.4895 | 0.0227 | 0.1094 | 19.30 | 19.69 | 0.835 | 73.87 |
| 2023 | D4_outgoing_transfer_production_only | 4.5067 | 0.0344 | 0.1080 | 20.08 | 20.35 | 0.857 | 74.58 |
| 2024 | D4_outgoing_transfer_production_only | 4.5551 | -0.0289 | 0.1162 | 20.83 | 21.04 | 0.814 | 73.40 |
| 2025 | D4_outgoing_transfer_production_only | 4.5707 | -0.0718 | 0.1118 | 20.80 | 20.93 | 0.826 | 76.68 |
| 2022 | D5_total_rp_plus_incoming | 4.4564 | -0.0105 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 |
| 2023 | D5_total_rp_plus_incoming | 4.4684 | -0.0038 | 0.1040 | 19.16 | 19.40 | 0.871 | 72.35 |
| 2024 | D5_total_rp_plus_incoming | 4.5492 | -0.0348 | 0.1145 | 20.41 | 20.70 | 0.815 | 70.48 |
| 2025 | D5_total_rp_plus_incoming | 4.5561 | -0.0864 | 0.1078 | 19.84 | 20.04 | 0.832 | 73.29 |
| 2022 | D6_total_rp_plus_outgoing | 4.4701 | 0.0032 | 0.1079 | 18.91 | 19.04 | 0.821 | 72.46 |
| 2023 | D6_total_rp_plus_outgoing | 4.4771 | 0.0049 | 0.1038 | 18.92 | 19.20 | 0.872 | 72.88 |
| 2024 | D6_total_rp_plus_outgoing | 4.5666 | -0.0175 | 0.1166 | 21.00 | 21.50 | 0.798 | 71.91 |
| 2025 | D6_total_rp_plus_outgoing | 4.6043 | -0.0382 | 0.1134 | 21.07 | 21.17 | 0.803 | 75.01 |
| 2022 | D7_total_rp_plus_incoming_plus_outgoing | 4.4586 | -0.0083 | 0.1061 | 18.66 | 18.66 | 0.829 | 71.99 |
| 2023 | D7_total_rp_plus_incoming_plus_outgoing | 4.4743 | 0.0021 | 0.1042 | 19.17 | 19.39 | 0.871 | 72.48 |
| 2024 | D7_total_rp_plus_incoming_plus_outgoing | 4.5472 | -0.0368 | 0.1143 | 20.48 | 20.76 | 0.810 | 71.04 |
| 2025 | D7_total_rp_plus_incoming_plus_outgoing | 4.5575 | -0.0850 | 0.1079 | 20.00 | 19.90 | 0.828 | 74.33 |
| 2022 | D8_existing_c10 | 4.4565 | -0.0103 | 0.1057 | 18.55 | 18.52 | 0.829 | 71.74 |
| 2023 | D8_existing_c10 | 4.4752 | 0.0030 | 0.1050 | 19.35 | 19.61 | 0.870 | 71.89 |
| 2024 | D8_existing_c10 | 4.5545 | -0.0295 | 0.1149 | 20.52 | 20.83 | 0.816 | 70.41 |
| 2025 | D8_existing_c10 | 4.5754 | -0.0671 | 0.1100 | 20.17 | 20.24 | 0.817 | 72.91 |

## Required pairwise comparisons

Negative ΔNLL means the first variant has lower loss than the second.

| Comparison | Candidate | Reference | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage |
|---|---|---|---:|---:|---:|---:|---:|
| incoming_unconditional | D3_incoming_transfer_production_only | D1_no_returning_production | -0.0029 | -0.0001 | -0.11 | -0.04 | -0.001 |
| outgoing_unconditional | D4_outgoing_transfer_production_only | D1_no_returning_production | 0.0018 | 0.0002 | 0.03 | 0.03 | -0.001 |
| incoming_conditional_on_rp | D5_total_rp_plus_incoming | D2_total_rp_only | -0.0273 | -0.0031 | -0.54 | -0.76 | 0.011 |
| outgoing_conditional_on_rp | D6_total_rp_plus_outgoing | D2_total_rp_only | -0.0052 | -0.0006 | -0.05 | -0.19 | -0.004 |
| outgoing_beyond_rp_plus_incoming | D7_total_rp_plus_incoming_plus_outgoing | D5_total_rp_plus_incoming | 0.0019 | 0.0002 | 0.10 | 0.01 | -0.003 |
| simplified_vs_c10 | D5_total_rp_plus_incoming | D8_existing_c10 | -0.0080 | -0.0010 | -0.17 | -0.13 | 0.004 |

## Paired team-season NLL diagnostics

| Comparison | Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |
|---|---|---:|---:|---:|---:|
| D5_vs_D2 | aggregate | 534 | -0.0273 | -0.0012 | 0.530 |
| D5_vs_D2 | 2022 | 131 | -0.0130 | -0.0005 | 0.542 |
| D5_vs_D2 | 2023 | 133 | 0.0005 | 0.0000 | 0.496 |
| D5_vs_D2 | 2024 | 134 | -0.0312 | -0.0011 | 0.522 |
| D5_vs_D2 | 2025 | 136 | -0.0644 | -0.0068 | 0.559 |
| D7_vs_D5 | aggregate | 534 | 0.0019 | 0.0000 | 0.496 |
| D7_vs_D5 | 2022 | 131 | 0.0022 | 0.0002 | 0.489 |
| D7_vs_D5 | 2023 | 133 | 0.0059 | 0.0000 | 0.496 |
| D7_vs_D5 | 2024 | 134 | -0.0020 | -0.0012 | 0.515 |
| D7_vs_D5 | 2025 | 136 | 0.0014 | 0.0001 | 0.485 |
| D5_vs_D8 | aggregate | 534 | -0.0080 | -0.0008 | 0.506 |
| D5_vs_D8 | 2022 | 131 | -0.0002 | 0.0009 | 0.481 |
| D5_vs_D8 | 2023 | 133 | -0.0068 | 0.0000 | 0.496 |
| D5_vs_D8 | 2024 | 134 | -0.0053 | -0.0015 | 0.522 |
| D5_vs_D8 | 2025 | 136 | -0.0193 | -0.0019 | 0.522 |

## Standardized roster-continuity coefficients

Numeric inputs are standardized using training-only means and scales. Missingness-indicator coefficients are shown separately; coefficient magnitudes are structural diagnostics, not causal effects.
The frozen C 1.2 fit admits roster-continuity features into the location equation only, so their log-scale coefficients are fixed at zero by protocol.

| Variant | Feature | Observed / missing training rows | Location β | Location missingness | Log-scale γ | Log-scale missingness |
|---|---|---:|---:|---:|---:|---:|
| D2_total_rp_only | returning_pct_ppa | 1021 / 1195 | -0.2016 | 0.0066 | 0.0000 | 0.0000 |
| D3_incoming_transfer_production_only | transfer_in_prior_usage_sum | 94 / 2122 | -0.0240 | 0.0285 | 0.0000 | 0.0000 |
| D4_outgoing_transfer_production_only | transfer_out_prior_usage_sum | 105 / 2111 | -0.0021 | 0.0643 | 0.0000 | 0.0000 |
| D5_total_rp_plus_incoming | returning_pct_ppa | 1021 / 1195 | -0.2042 | 0.0063 | 0.0000 | 0.0000 |
| D5_total_rp_plus_incoming | transfer_in_prior_usage_sum | 94 / 2122 | -0.0374 | -0.0713 | 0.0000 | 0.0000 |
| D6_total_rp_plus_outgoing | returning_pct_ppa | 1021 / 1195 | -0.2059 | 0.0081 | 0.0000 | 0.0000 |
| D6_total_rp_plus_outgoing | transfer_out_prior_usage_sum | 105 / 2111 | -0.0317 | -0.0592 | 0.0000 | 0.0000 |
| D7_total_rp_plus_incoming_plus_outgoing | returning_pct_ppa | 1021 / 1195 | -0.2067 | 0.0075 | 0.0000 | 0.0000 |
| D7_total_rp_plus_incoming_plus_outgoing | transfer_in_prior_usage_sum | 94 / 2122 | -0.0305 | -0.0587 | 0.0000 | 0.0000 |
| D7_total_rp_plus_incoming_plus_outgoing | transfer_out_prior_usage_sum | 105 / 2111 | -0.0209 | -0.0449 | 0.0000 | 0.0000 |
| D8_existing_c10 | returning_pct_ppa | 1021 / 1195 | -0.1565 | 0.0034 | 0.0000 | 0.0000 |
| D8_existing_c10 | returning_pct_passing_ppa | 1021 / 1195 | -0.0253 | 0.0034 | 0.0000 | 0.0000 |
| D8_existing_c10 | returning_pct_receiving_ppa | 1021 / 1195 | -0.0753 | 0.0034 | 0.0000 | 0.0000 |
| D8_existing_c10 | returning_pct_rushing_ppa | 1021 / 1195 | -0.0005 | 0.0034 | 0.0000 | 0.0000 |
| D8_existing_c10 | transfer_in_prior_usage_sum | 94 / 2122 | -0.0391 | -0.1698 | 0.0000 | 0.0000 |
| D8_existing_c10 | transfer_net_prior_usage | 77 / 2139 | 0.0051 | 0.0751 | 0.0000 | 0.0000 |

## Transfer coverage and limitations

CFBD portal and prior-usage inputs retain the issue-91 retrospective-oracle caveats: endpoint responses are not archived as August 15 snapshots, final destinations may be resolved later, and the portal-to-usage join is name-based. Covered seasons with no matching transfer are zero; uncovered seasons remain missing and use the model's training-only imputation and missingness indicators.

| Season | Portal payload | Records | Dated/on cutoff | Destination rate | Rating rate |
|---:|:---:|---:|---:|---:|---:|
| 2004 | False | 0 | 0 | n/a | n/a |
| 2005 | False | 0 | 0 | n/a | n/a |
| 2006 | False | 0 | 0 | n/a | n/a |
| 2007 | False | 0 | 0 | n/a | n/a |
| 2008 | False | 0 | 0 | n/a | n/a |
| 2009 | False | 0 | 0 | n/a | n/a |
| 2010 | False | 0 | 0 | n/a | n/a |
| 2011 | False | 0 | 0 | n/a | n/a |
| 2012 | False | 0 | 0 | n/a | n/a |
| 2013 | False | 0 | 0 | n/a | n/a |
| 2014 | False | 0 | 0 | n/a | n/a |
| 2015 | False | 0 | 0 | n/a | n/a |
| 2016 | False | 0 | 0 | n/a | n/a |
| 2017 | False | 0 | 0 | n/a | n/a |
| 2018 | False | 0 | 0 | n/a | n/a |
| 2019 | False | 0 | 0 | n/a | n/a |
| 2020 | False | 0 | 0 | n/a | n/a |
| 2021 | True | 1770 | 1770 | 0.595 | 0.182 |
| 2022 | True | 2273 | 2268 | 0.601 | 0.459 |
| 2023 | True | 2502 | 2502 | 0.642 | 0.356 |
| 2024 | True | 3378 | 3378 | 0.786 | 0.548 |
| 2025 | True | 4499 | 4497 | 0.838 | 0.558 |

## Artifacts

- `variant_definitions.json`, `summary.json` — frozen configuration, parity check, comparisons, and provenance.
- `candidate_annual_metrics.csv`, `candidate_summary.csv` — aggregate and year-by-year metrics for D0–D8.
- `standardized_coefficients.csv` — D2–D8 roster-continuity coefficient diagnostics.
- `paired_nll_diagnostics.csv`, `comparison_summary.csv` — required paired losses and metric comparisons.
- `coverage_by_season.csv`, `provenance_audit.json`, `plots/` — source audit and visual summaries.
