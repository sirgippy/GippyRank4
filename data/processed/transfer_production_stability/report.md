# Transfer-production coefficient stability (issue 97)

## Conclusion

R2 beats R0 by rolling-origin NLL in **3 / 4** target seasons, with weighted mean ΔNLL (R2 − R0) of **-0.0198**. The R2 roster-continuity coefficient paths contain **3** adjacent non-zero sign flips; the largest absolute standardized location coefficient is **0.177**.
These results are a stability diagnostic, not a feature-selection result. The selected representation was fixed before the rolling scores were calculated, and no regularization, interaction, breakpoint, or year-specific term was tuned against 2022–2025.
The coefficient and performance evidence should be read together: the rolling study does not by itself establish a production-safe relationship; it does not replace the original frozen-through-2021 held-out evaluation.

## Selected representation

R2 is the exact C10 representation carried forward from the issue-91 decomposition follow-up:

- existing Context C 1.2 C0 features: coaching, recruiting, Team Talent, and all four returning-production features;
- `transfer_in_prior_usage_sum`;
- `transfer_net_prior_usage` (incoming prior usage minus outgoing prior usage).

R0 is C0 with the existing Context feature family. R1 removes all returning-production features. R2 adds only the two fixed C10 transfer-production inputs to R0.

## Rolling-origin protocol

Each fit uses all FBS team-seasons strictly before its target season and scores the unchanged target-season population. The C 1.2 penalty is 0.25; all model features enter the location equation, H features alone enter scale, and the existing deterministic optimizer retry is retained. Raw transfer fields are filtered to the August 15 season-relative cutoff. Training-only median imputation and missingness indicators remain in `DirectRankModel`. Stored H PMFs remain the cold-start fallback.

| Target | Train through | R0 | R1 | R2 |
|---:|---:|---:|---:|---:|
| 2022 | 2021 | 131 | 131 | 131 |
| 2023 | 2022 | 133 | 133 | 133 |
| 2024 | 2023 | 134 | 134 | 134 |
| 2025 | 2024 | 136 | 136 | 136 |

## Training-data coverage

Counts below are computed from raw transfer values before preprocessing. A complete observed row has both selected R2 transfer-production inputs present.

| Target | Training team-seasons | Transfer-covered seasons | Rows with incoming usage | Rows with net usage | Rows with both | Fraction with both |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 2216 | 1 | 94 | 77 | 77 | 0.035 |
| 2023 | 2346 | 2 | 205 | 180 | 180 | 0.077 |
| 2024 | 2477 | 3 | 322 | 292 | 292 | 0.118 |
| 2025 | 2610 | 4 | 451 | 415 | 415 | 0.159 |

## Rolling-origin predictive metrics

Δ columns are candidate minus the named reference; negative values favor the candidate.

| Target | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | ΔNLL vs R0 | ΔNLL vs R1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | R0 | 4.4669 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 | 0.0000 | -0.0203 |
| 2022 | R1 | 4.4872 | 0.1090 | 19.28 | 19.68 | 0.833 | 73.75 | 0.0203 | 0.0000 |
| 2022 | R2 | 4.4565 | 0.1057 | 18.55 | 18.52 | 0.829 | 71.74 | -0.0103 | -0.0307 |
| 2023 | R0 | 4.4686 | 0.1041 | 19.21 | 19.59 | 0.871 | 71.92 | 0.0000 | -0.0346 |
| 2023 | R1 | 4.5031 | 0.1076 | 20.03 | 20.34 | 0.862 | 74.39 | 0.0346 | 0.0000 |
| 2023 | R2 | 4.4703 | 0.1052 | 19.35 | 19.64 | 0.868 | 71.46 | 0.0018 | -0.0328 |
| 2024 | R0 | 4.5793 | 0.1179 | 20.98 | 21.64 | 0.794 | 70.50 | 0.0000 | 0.0234 |
| 2024 | R1 | 4.5560 | 0.1163 | 20.83 | 21.01 | 0.813 | 73.01 | -0.0234 | 0.0000 |
| 2024 | R2 | 4.5580 | 0.1155 | 20.62 | 21.12 | 0.818 | 69.69 | -0.0213 | 0.0020 |
| 2025 | R0 | 4.6087 | 0.1145 | 21.15 | 21.65 | 0.813 | 73.46 | 0.0000 | 0.0435 |
| 2025 | R1 | 4.5652 | 0.1114 | 20.76 | 20.94 | 0.827 | 76.54 | -0.0435 | 0.0000 |
| 2025 | R2 | 4.5604 | 0.1088 | 19.98 | 20.24 | 0.828 | 72.90 | -0.0484 | -0.0049 |

## R2 standardized roster-continuity coefficients

The reported location coefficients are for numeric inputs after training-only standardization. Missingness-indicator coefficients are shown separately. R2 has no separate outgoing-production feature; its net coefficient implies an outgoing direction through the negative of the net term, conditional on the incoming term.

| Target | Feature | Coefficient | Sign | Change from prior fit | Missingness indicator |
|---:|---|---:|---|---:|---:|
| 2022 | `returning_pct_ppa` | -0.1565 | negative | n/a | 0.0034 |
| 2022 | `returning_pct_passing_ppa` | -0.0253 | negative | n/a | 0.0034 |
| 2022 | `returning_pct_receiving_ppa` | -0.0753 | negative | n/a | 0.0034 |
| 2022 | `returning_pct_rushing_ppa` | -0.0005 | negative | n/a | 0.0034 |
| 2022 | `transfer_in_prior_usage_sum` | -0.0391 | negative | n/a | -0.1698 |
| 2022 | `transfer_net_prior_usage` | 0.0051 | positive | n/a | 0.0751 |
| 2023 | `returning_pct_ppa` | -0.1594 | negative | -0.0030 | -0.0004 |
| 2023 | `returning_pct_passing_ppa` | -0.0289 | negative | -0.0036 | -0.0004 |
| 2023 | `returning_pct_receiving_ppa` | -0.0765 | negative | -0.0012 | -0.0004 |
| 2023 | `returning_pct_rushing_ppa` | 0.0002 | positive | 0.0007 | -0.0004 |
| 2023 | `transfer_in_prior_usage_sum` | -0.0527 | negative | -0.0136 | -0.1118 |
| 2023 | `transfer_net_prior_usage` | -0.0139 | negative | -0.0191 | 0.0352 |
| 2024 | `returning_pct_ppa` | -0.1772 | negative | -0.0178 | -0.0024 |
| 2024 | `returning_pct_passing_ppa` | -0.0280 | negative | 0.0009 | -0.0024 |
| 2024 | `returning_pct_receiving_ppa` | -0.0686 | negative | 0.0079 | -0.0024 |
| 2024 | `returning_pct_rushing_ppa` | -0.0013 | negative | -0.0015 | -0.0024 |
| 2024 | `transfer_in_prior_usage_sum` | -0.0474 | negative | 0.0053 | 0.0367 |
| 2024 | `transfer_net_prior_usage` | -0.0201 | negative | -0.0062 | -0.0051 |
| 2025 | `returning_pct_ppa` | -0.1763 | negative | 0.0009 | -0.0061 |
| 2025 | `returning_pct_passing_ppa` | -0.0238 | negative | 0.0043 | -0.0061 |
| 2025 | `returning_pct_receiving_ppa` | -0.0643 | negative | 0.0043 | -0.0061 |
| 2025 | `returning_pct_rushing_ppa` | -0.0011 | negative | 0.0002 | -0.0061 |
| 2025 | `transfer_in_prior_usage_sum` | -0.0792 | negative | -0.0318 | 0.1932 |
| 2025 | `transfer_net_prior_usage` | -0.0074 | negative | 0.0128 | -0.1393 |

## Pairwise comparisons

| Target | Comparison | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2022 | R2 − R0 | -0.0103 | -0.0013 | -0.13 | -0.34 | 0.004 | -0.36 |
| 2022 | R2 − R1 | -0.0307 | -0.0033 | -0.73 | -1.16 | -0.003 | -2.01 |
| 2022 | R1 − R0 | 0.0203 | 0.0019 | 0.60 | 0.82 | 0.007 | 1.65 |
| 2023 | R2 − R0 | 0.0018 | 0.0011 | 0.14 | 0.05 | -0.003 | -0.46 |
| 2023 | R2 − R1 | -0.0328 | -0.0025 | -0.69 | -0.70 | 0.007 | -2.93 |
| 2023 | R1 − R0 | 0.0346 | 0.0036 | 0.83 | 0.75 | -0.009 | 2.47 |
| 2024 | R2 − R0 | -0.0213 | -0.0025 | -0.36 | -0.52 | 0.024 | -0.81 |
| 2024 | R2 − R1 | 0.0020 | -0.0008 | -0.21 | 0.10 | 0.005 | -3.32 |
| 2024 | R1 − R0 | -0.0234 | -0.0016 | -0.15 | -0.63 | 0.019 | 2.51 |
| 2025 | R2 − R0 | -0.0484 | -0.0057 | -1.17 | -1.41 | 0.015 | -0.56 |
| 2025 | R2 − R1 | -0.0049 | -0.0026 | -0.78 | -0.71 | 0.001 | -3.65 |
| 2025 | R1 − R0 | -0.0435 | -0.0031 | -0.39 | -0.71 | 0.014 | 3.09 |

## Frozen versus rolling behavior

The frozen rows refit each candidate through 2021 and score 2022–2025 without retraining between target seasons. The rolling rows refit before each target. The frozen evaluation answers whether the original 2021-trained hypothesis generalized to untouched future seasons; the rolling evaluation answers whether the relationship remains useful as later portal-era seasons enter training.

| Protocol | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen | R0 | 4.5423 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| frozen | R1 | 4.5292 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| frozen | R2 | 4.5160 | 0.1089 | 19.66 | 19.81 | 0.833 | 71.74 |
| rolling | R0 | 4.5316 | 0.1109 | 20.02 | 20.45 | 0.826 | 72.00 |
| rolling | R1 | 4.5283 | 0.1111 | 20.23 | 20.50 | 0.833 | 74.44 |
| rolling | R2 | 4.5119 | 0.1088 | 19.63 | 19.89 | 0.836 | 71.45 |

## Optional coefficient uncertainty diagnostic

No resampling refit was added. The existing optimizer is already the expensive part of this study, and a season bootstrap would be a descriptive sensitivity check rather than a formal uncertainty model. Coefficient path changes and raw coverage counts are reported instead.

## Provenance and limitations

The portal and prior-usage inputs are a retrospective research oracle: the endpoint responses are not archived as August 15 snapshots, final destinations can be resolved later, and the portal-to-usage join is name-based. The study therefore tests stability of the selected oracle relationship, not production data readiness. No target-season outcomes enter transfer features or fitting.

## Artifacts

- `rolling_metrics.csv`, `rolling_comparisons.csv`, `frozen_metrics.csv` — required scores and pairwise deltas.
- `training_coverage.csv` — training-panel size and raw transfer-feature coverage.
- `coefficients.csv` — standardized roster-continuity coefficients and fit-to-fit changes.
- `summary.json`, `report.md`, `plots/` — protocol, hashes, interpretation, and plots.
