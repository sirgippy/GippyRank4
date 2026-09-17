# Transfer-aware roster continuity oracle (issue 91)

## Conclusion

No development candidate was selected: the rows available to train through 2017 contain zero observed values for every added transfer feature. The predeclared candidates are exploratory retrospective-oracle comparisons. The clean C10 candidate **C10_transfer_production** has held-out ΔNLL **-0.0263** versus production C0; the strongest descriptive result is **C9_ad_hoc_rp_usage_hybrid**, with ΔNLL **-0.0309**. C10_transfer_production is used only for mechanism diagnostics. This is not a production-safe Context change.
C9 is retained as an explicitly ad hoc hybrid index: returning-production percentage plus incoming prior-usage sum. Those quantities have incompatible denominators, so C9 is not interpreted as reconstructed effective returning production; C10 keeps returning production and incoming prior usage as separate features.
The C0 control reproduction check passed: maximum absolute metric difference from the stored production evaluation was 0.000005.

## Provenance and leakage audit

CFBD `/player/portal` records supply season, origin, destination, position, transfer date, rating, and stars. CFBD `/player/usage` supplies prior-season overall usage. These fields are retained as a retrospective research oracle because endpoint responses are not archived as August 15 snapshots, final destinations may be resolved later, and the portal-to-usage join is name-based. Scholarship-player counts are unavailable because eligibility does not establish scholarship status.

All cutoff features include only records with a transfer date on or before the cutoff. Missing portal seasons remain missing; a covered season with no matching transfer is zero. No target-season outcomes are used as transfer features.

## Evaluation protocol

- C0 is the existing C 1.2 location equation; C1 removes returning production. Transfer candidates were predeclared in the script.
- Fits use rows through 2021 and score unchanged on 2022--2025. The 2018--2021 development comparison is reported for transparency only; it cannot select a transfer candidate because its training rows end before portal coverage begins. The production FBS population and stored H cold-start fallback are retained.
- Training-only imputation and missingness indicators remain in `DirectRankModel`; transfer missingness is never silently converted to zero.
- The negative-control permutation shuffles transfer values within season with seed 7 and never changes the target keys.

## Held-out candidate comparison

| Candidate | NLL | Δ vs C0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0_full | 4.5423 | 0.0000 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| C10_transfer_production | 4.5160 | -0.0263 | 0.1089 | 19.66 | 19.81 | 0.833 | 71.74 |
| C1_minus_rp | 4.5292 | -0.0132 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| C2_transfer_volume | 4.6552 | 0.1129 | 0.1228 | 21.80 | 22.54 | 0.781 | 69.80 |
| C3_minus_rp_transfer_volume | 4.6532 | 0.1109 | 0.1248 | 22.48 | 23.27 | 0.786 | 71.77 |
| C4_rp_x_transfer_volume | 4.5322 | -0.0101 | 0.1109 | 20.03 | 20.43 | 0.827 | 72.36 |
| C5_transfer_talent | 4.7107 | 0.1684 | 0.1273 | 22.54 | 23.85 | 0.758 | 66.79 |
| C6_minus_rp_transfer_talent | 4.7747 | 0.2323 | 0.1331 | 23.54 | 24.82 | 0.742 | 68.14 |
| C7_rp_x_incoming_value | 4.8799 | 0.3376 | 0.1380 | 23.78 | 24.96 | 0.734 | 65.79 |
| C8_rp_x_net_value | 4.5411 | -0.0012 | 0.1118 | 20.15 | 20.59 | 0.825 | 72.01 |
| C9_ad_hoc_rp_usage_hybrid | 4.5115 | -0.0309 | 0.1087 | 19.74 | 19.92 | 0.838 | 73.60 |

## Year-by-year primary metrics

| Season | Candidate | NLL | Δ vs C0 | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | C2_transfer_volume | 4.5337 | 0.0668 | 0.1113 | 19.37 | 19.39 | 0.809 | 68.46 |
| 2023 | C2_transfer_volume | 4.5797 | 0.1075 | 0.1173 | 20.81 | 21.90 | 0.813 | 68.45 |
| 2024 | C2_transfer_volume | 4.7548 | 0.1707 | 0.1320 | 23.04 | 23.72 | 0.747 | 70.96 |
| 2025 | C2_transfer_volume | 4.7480 | 0.1056 | 0.1303 | 23.87 | 25.04 | 0.756 | 71.25 |
| 2022 | C3_minus_rp_transfer_volume | 4.5412 | 0.0743 | 0.1133 | 19.65 | 19.76 | 0.809 | 70.49 |
| 2023 | C3_minus_rp_transfer_volume | 4.6072 | 0.1349 | 0.1212 | 22.08 | 22.98 | 0.821 | 71.03 |
| 2024 | C3_minus_rp_transfer_volume | 4.7536 | 0.1696 | 0.1342 | 23.94 | 24.76 | 0.744 | 72.43 |
| 2025 | C3_minus_rp_transfer_volume | 4.7073 | 0.0648 | 0.1301 | 24.17 | 25.46 | 0.772 | 73.08 |
| 2022 | C4_rp_x_transfer_volume | 4.4666 | -0.0003 | 0.1070 | 18.67 | 18.85 | 0.826 | 72.15 |
| 2023 | C4_rp_x_transfer_volume | 4.4701 | -0.0021 | 0.1041 | 19.26 | 19.56 | 0.868 | 72.29 |
| 2024 | C4_rp_x_transfer_volume | 4.5737 | -0.0103 | 0.1171 | 20.90 | 21.50 | 0.808 | 71.30 |
| 2025 | C4_rp_x_transfer_volume | 4.6153 | -0.0272 | 0.1151 | 21.25 | 21.76 | 0.808 | 73.67 |
| 2022 | C5_transfer_talent | 4.5456 | 0.0787 | 0.1121 | 19.44 | 20.37 | 0.804 | 69.77 |
| 2023 | C5_transfer_talent | 4.6812 | 0.2090 | 0.1250 | 22.18 | 23.66 | 0.779 | 64.29 |
| 2024 | C5_transfer_talent | 4.7779 | 0.1939 | 0.1331 | 23.44 | 24.64 | 0.734 | 65.18 |
| 2025 | C5_transfer_talent | 4.8323 | 0.1898 | 0.1385 | 24.97 | 26.59 | 0.715 | 67.96 |
| 2022 | C6_minus_rp_transfer_talent | 4.5558 | 0.0890 | 0.1126 | 19.82 | 20.79 | 0.806 | 71.44 |
| 2023 | C6_minus_rp_transfer_talent | 4.7607 | 0.2885 | 0.1326 | 23.38 | 24.48 | 0.742 | 65.85 |
| 2024 | C6_minus_rp_transfer_talent | 4.8166 | 0.2325 | 0.1357 | 23.86 | 25.07 | 0.736 | 67.57 |
| 2025 | C6_minus_rp_transfer_talent | 4.9577 | 0.3152 | 0.1509 | 26.98 | 28.80 | 0.687 | 67.75 |
| 2022 | C7_rp_x_incoming_value | 4.5177 | 0.0508 | 0.1130 | 19.56 | 20.05 | 0.813 | 71.21 |
| 2023 | C7_rp_x_incoming_value | 4.5827 | 0.1105 | 0.1117 | 20.09 | 20.85 | 0.839 | 70.01 |
| 2024 | C7_rp_x_incoming_value | 4.8722 | 0.2882 | 0.1446 | 24.56 | 26.23 | 0.713 | 64.57 |
| 2025 | C7_rp_x_incoming_value | 5.5271 | 0.8846 | 0.1814 | 30.67 | 32.46 | 0.577 | 57.64 |
| 2022 | C8_rp_x_net_value | 4.4666 | -0.0002 | 0.1071 | 18.68 | 18.92 | 0.827 | 72.15 |
| 2023 | C8_rp_x_net_value | 4.4718 | -0.0004 | 0.1044 | 19.27 | 19.55 | 0.867 | 72.11 |
| 2024 | C8_rp_x_net_value | 4.5835 | -0.0005 | 0.1181 | 21.01 | 21.69 | 0.805 | 71.03 |
| 2025 | C8_rp_x_net_value | 4.6387 | -0.0038 | 0.1174 | 21.58 | 22.12 | 0.803 | 72.76 |
| 2022 | C9_ad_hoc_rp_usage_hybrid | 4.4688 | 0.0019 | 0.1064 | 18.67 | 18.75 | 0.836 | 73.05 |
| 2023 | C9_ad_hoc_rp_usage_hybrid | 4.4895 | 0.0173 | 0.1067 | 19.72 | 20.00 | 0.864 | 73.82 |
| 2024 | C9_ad_hoc_rp_usage_hybrid | 4.5411 | -0.0429 | 0.1141 | 20.44 | 20.49 | 0.815 | 72.04 |
| 2025 | C9_ad_hoc_rp_usage_hybrid | 4.5448 | -0.0977 | 0.1078 | 20.10 | 20.41 | 0.837 | 75.46 |
| 2022 | C10_transfer_production | 4.4565 | -0.0103 | 0.1057 | 18.55 | 18.52 | 0.829 | 71.74 |
| 2023 | C10_transfer_production | 4.4752 | 0.0030 | 0.1050 | 19.35 | 19.61 | 0.870 | 71.89 |
| 2024 | C10_transfer_production | 4.5545 | -0.0295 | 0.1149 | 20.52 | 20.83 | 0.816 | 70.41 |
| 2025 | C10_transfer_production | 4.5754 | -0.0671 | 0.1100 | 20.17 | 20.24 | 0.817 | 72.91 |
| 2022 | C0_full | 4.4669 | 0.0000 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 |
| 2023 | C0_full | 4.4722 | 0.0000 | 0.1044 | 19.28 | 19.55 | 0.867 | 72.12 |
| 2024 | C0_full | 4.5840 | 0.0000 | 0.1183 | 21.02 | 21.71 | 0.805 | 70.87 |
| 2025 | C0_full | 4.6425 | 0.0000 | 0.1177 | 21.61 | 22.20 | 0.803 | 72.58 |
| 2022 | C1_minus_rp | 4.4872 | 0.0203 | 0.1090 | 19.28 | 19.68 | 0.833 | 73.75 |
| 2023 | C1_minus_rp | 4.5047 | 0.0325 | 0.1078 | 20.10 | 20.41 | 0.861 | 74.68 |
| 2024 | C1_minus_rp | 4.5539 | -0.0301 | 0.1160 | 20.76 | 20.93 | 0.815 | 73.33 |
| 2025 | C1_minus_rp | 4.5691 | -0.0734 | 0.1115 | 20.73 | 20.88 | 0.827 | 76.58 |

## Transfer coverage

| Season | Portal payload | Records | Dated/on cutoff | Destination rate | Rating rate | Usage joins are audited in team-level artifacts |
|---:|:---:|---:|---:|---:|---:|---|
| 2004 | False | 0 | 0 | n/a | n/a | yes |
| 2005 | False | 0 | 0 | n/a | n/a | yes |
| 2006 | False | 0 | 0 | n/a | n/a | yes |
| 2007 | False | 0 | 0 | n/a | n/a | yes |
| 2008 | False | 0 | 0 | n/a | n/a | yes |
| 2009 | False | 0 | 0 | n/a | n/a | yes |
| 2010 | False | 0 | 0 | n/a | n/a | yes |
| 2011 | False | 0 | 0 | n/a | n/a | yes |
| 2012 | False | 0 | 0 | n/a | n/a | yes |
| 2013 | False | 0 | 0 | n/a | n/a | yes |
| 2014 | False | 0 | 0 | n/a | n/a | yes |
| 2015 | False | 0 | 0 | n/a | n/a | yes |
| 2016 | False | 0 | 0 | n/a | n/a | yes |
| 2017 | False | 0 | 0 | n/a | n/a | yes |
| 2018 | False | 0 | 0 | n/a | n/a | yes |
| 2019 | False | 0 | 0 | n/a | n/a | yes |
| 2020 | False | 0 | 0 | n/a | n/a | yes |
| 2021 | True | 1770 | 1770 | 0.595 | 0.182 | yes |
| 2022 | True | 2273 | 2268 | 0.601 | 0.459 | yes |
| 2023 | True | 2502 | 2502 | 0.642 | 0.356 | yes |
| 2024 | True | 3378 | 3378 | 0.786 | 0.548 | yes |
| 2025 | True | 4499 | 4497 | 0.838 | 0.558 | yes |

## Diagnostics and limitations

The bucket, replacement-quality, negative-control, cutoff, QB, Team Talent overlap, and returning-production component tables are the auditable diagnostics for the mechanism question. They use the predeclared C10_transfer_production candidate as a descriptive reference, not a selected model. A positive RP contribution means C-minus-RP has higher NLL than C0, so RP helped; a negative value means RP hurt.

Because historical portal coverage begins late and the endpoint is retrospective, these results cannot establish that a feature was knowable on August 15 in the historical years. A positive result identifies a promising oracle representation; productionization requires an archived cutoff-safe source and a stable player/team identity join.

## Artifacts

- `provenance_audit.json`, `coverage_by_season.csv` — source classification and season coverage.
- `candidate_annual_metrics.csv`, `candidate_summary.csv`, `candidate_per_team_losses.csv` — paired primary scores and losses.
- `transfer_bucket_diagnostics.csv`, `replacement_quality_diagnostics.csv` — RP mechanism diagnostics.
- `negative_control_metrics.csv`, `cutoff_sensitivity.csv`, `qb_sensitivity.csv`, `talent_overlap_sensitivity.csv`, `rp_component_sensitivity.csv` — predeclared robustness checks.
- `summary.json`, `plots/` — configuration, hashes, comparison record, and visual summaries.
