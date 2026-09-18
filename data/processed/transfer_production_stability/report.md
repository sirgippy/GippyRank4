# Transfer-production coefficient stability (issue 97)

## Conclusion

R2 beats R0 by rolling-origin NLL in **4 / 4** target seasons, with weighted mean ΔNLL (R2 − R0) of **-0.0236**. The R2 roster-continuity coefficient paths contain **0** adjacent non-zero sign flips; the largest absolute standardized location coefficient is **0.226**.
These results are a stability diagnostic, not a feature-selection result. The selected representation was fixed before the rolling scores were calculated, and no regularization, interaction, breakpoint, or year-specific term was tuned against 2022–2025.
The coefficient and performance evidence should be read together: the rolling study supports continued investigation; it does not replace the original frozen-through-2021 held-out evaluation.

## Selected representation

R2 is the exact D5 representation selected by the issue-96 decomposition study:

- C-minus-RP baseline features: coaching, recruiting, and Team Talent;
- total returning production: `returning_pct_ppa`;
- incoming prior transfer production: `transfer_in_prior_usage_sum`.

R0 is C0 with the existing Context feature family. R1 removes all returning-production features. R2 is D5 and adds only total RP plus incoming prior transfer production to the C-minus-RP baseline.

## Frozen D5 parity check

The frozen-through-2021 R2 aggregate reproduces the checked-in issue-96 D5 aggregate within the configured tolerance of 1.0e-06; maximum absolute metric difference is 0.00000000.

| Metric | Issue-96 D5 | Frozen R2 | Absolute difference |
|---|---:|---:|---:|
| nll | 4.50808063 | 4.50808063 | 0.00000000 |
| crps | 0.10798618 | 0.10798618 | 0.00000000 |
| expected_rank_mae | 19.48957465 | 19.48957465 | 0.00000000 |
| median_rank_mae | 19.67509363 | 19.67509363 | 0.00000000 |
| interval_80_coverage | 0.83770277 | 0.83770277 | 0.00000000 |
| interval_80_average_width | 71.95880150 | 71.95880150 | 0.00000000 |

## Rolling-origin protocol

Each fit uses all FBS team-seasons strictly before its target season and scores the unchanged target-season population. The C 1.2 penalty is 0.25; all model features enter the location equation, H features alone enter scale, and the existing deterministic optimizer retry is retained. Raw transfer fields are filtered to the August 15 season-relative cutoff. Training-only median imputation and missingness indicators remain in `DirectRankModel`. Stored H PMFs remain the cold-start fallback.

| Target | Train through | R0 | R1 | R2 |
|---:|---:|---:|---:|---:|
| 2022 | 2021 | 131 | 131 | 131 |
| 2023 | 2022 | 133 | 133 | 133 |
| 2024 | 2023 | 134 | 134 | 134 |
| 2025 | 2024 | 136 | 136 | 136 |

## Training-data coverage

Counts below are computed from raw transfer values before preprocessing. The selected transfer-production input is observed when `transfer_in_prior_usage_sum` is present.

| Target | Training team-seasons | Transfer-covered seasons | Rows with incoming usage | Fraction observed |
|---:|---:|---:|---:|---:|
| 2022 | 2216 | 1 | 94 | 0.042 |
| 2023 | 2346 | 2 | 205 | 0.087 |
| 2024 | 2477 | 3 | 322 | 0.130 |
| 2025 | 2610 | 4 | 451 | 0.173 |

## Rolling-origin predictive metrics

Δ columns are candidate minus the named reference; negative values favor the candidate.

| Target | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | ΔNLL vs R0 | ΔNLL vs R1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | R0 | 4.4669 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 | 0.0000 | -0.0203 |
| 2022 | R1 | 4.4872 | 0.1090 | 19.28 | 19.68 | 0.833 | 73.75 | 0.0203 | 0.0000 |
| 2022 | R2 | 4.4564 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 | -0.0105 | -0.0308 |
| 2023 | R0 | 4.4686 | 0.1041 | 19.21 | 19.59 | 0.871 | 71.92 | 0.0000 | -0.0346 |
| 2023 | R1 | 4.5031 | 0.1076 | 20.03 | 20.34 | 0.862 | 74.39 | 0.0346 | 0.0000 |
| 2023 | R2 | 4.4680 | 0.1041 | 19.16 | 19.36 | 0.870 | 72.01 | -0.0005 | -0.0351 |
| 2024 | R0 | 4.5793 | 0.1179 | 20.98 | 21.64 | 0.794 | 70.50 | 0.0000 | 0.0234 |
| 2024 | R1 | 4.5560 | 0.1163 | 20.83 | 21.01 | 0.813 | 73.01 | -0.0234 | 0.0000 |
| 2024 | R2 | 4.5533 | 0.1151 | 20.59 | 21.04 | 0.813 | 70.34 | -0.0261 | -0.0027 |
| 2025 | R0 | 4.6087 | 0.1145 | 21.15 | 21.65 | 0.813 | 73.46 | 0.0000 | 0.0435 |
| 2025 | R1 | 4.5652 | 0.1114 | 20.76 | 20.94 | 0.827 | 76.54 | -0.0435 | 0.0000 |
| 2025 | R2 | 4.5524 | 0.1078 | 19.93 | 20.07 | 0.828 | 73.54 | -0.0564 | -0.0129 |

## R2 standardized roster-continuity coefficients

The reported location coefficients are for the two selected D5 continuity inputs after training-only standardization. Missingness-indicator coefficients are shown separately. No outgoing or net transfer-production feature is included.

| Target | Feature | Coefficient | Sign | Change from prior fit | Missingness indicator |
|---:|---|---:|---|---:|---:|
| 2022 | `returning_pct_ppa` | -0.2042 | negative | n/a | 0.0063 |
| 2022 | `transfer_in_prior_usage_sum` | -0.0374 | negative | n/a | -0.0713 |
| 2023 | `returning_pct_ppa` | -0.2111 | negative | -0.0069 | -0.0059 |
| 2023 | `transfer_in_prior_usage_sum` | -0.0614 | negative | -0.0241 | -0.0846 |
| 2024 | `returning_pct_ppa` | -0.2257 | negative | -0.0146 | -0.0095 |
| 2024 | `transfer_in_prior_usage_sum` | -0.0586 | negative | 0.0028 | 0.0183 |
| 2025 | `returning_pct_ppa` | -0.2220 | negative | 0.0037 | -0.0217 |
| 2025 | `transfer_in_prior_usage_sum` | -0.0824 | negative | -0.0238 | 0.0588 |

## Pairwise comparisons

| Target | Comparison | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2022 | R2 − R0 | -0.0105 | -0.0015 | -0.16 | -0.33 | 0.007 | -0.40 |
| 2022 | R2 − R1 | -0.0308 | -0.0034 | -0.76 | -1.15 | 0.001 | -2.05 |
| 2022 | R1 − R0 | 0.0203 | 0.0019 | 0.60 | 0.82 | 0.007 | 1.65 |
| 2023 | R2 − R0 | -0.0005 | 0.0000 | -0.05 | -0.23 | -0.001 | 0.09 |
| 2023 | R2 − R1 | -0.0351 | -0.0035 | -0.88 | -0.98 | 0.008 | -2.38 |
| 2023 | R1 − R0 | 0.0346 | 0.0036 | 0.83 | 0.75 | -0.009 | 2.47 |
| 2024 | R2 − R0 | -0.0261 | -0.0029 | -0.39 | -0.60 | 0.019 | -0.16 |
| 2024 | R2 − R1 | -0.0027 | -0.0013 | -0.24 | 0.02 | 0.000 | -2.68 |
| 2024 | R1 − R0 | -0.0234 | -0.0016 | -0.15 | -0.63 | 0.019 | 2.51 |
| 2025 | R2 − R0 | -0.0564 | -0.0067 | -1.23 | -1.57 | 0.016 | 0.08 |
| 2025 | R2 − R1 | -0.0129 | -0.0036 | -0.84 | -0.87 | 0.001 | -3.01 |
| 2025 | R1 − R0 | -0.0435 | -0.0031 | -0.39 | -0.71 | 0.014 | 3.09 |

## Frozen versus rolling behavior

The frozen rows refit each candidate through 2021 and score 2022–2025 without retraining between target seasons. The rolling rows refit before each target. The frozen evaluation answers whether the original 2021-trained hypothesis generalized to untouched future seasons; the rolling evaluation answers whether the relationship remains useful as later portal-era seasons enter training.

| Protocol | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen | R0 | 4.5423 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| frozen | R1 | 4.5292 | 0.1111 | 20.23 | 20.48 | 0.834 | 74.60 |
| frozen | R2 | 4.5081 | 0.1080 | 19.49 | 19.68 | 0.838 | 71.96 |
| rolling | R0 | 4.5316 | 0.1109 | 20.02 | 20.45 | 0.826 | 72.00 |
| rolling | R1 | 4.5283 | 0.1111 | 20.23 | 20.50 | 0.833 | 74.44 |
| rolling | R2 | 4.5080 | 0.1082 | 19.55 | 19.76 | 0.836 | 71.90 |

## Optional coefficient uncertainty diagnostic

No resampling refit was added. The existing optimizer is already the expensive part of this study, and a season bootstrap would be a descriptive sensitivity check rather than a formal uncertainty model. Coefficient path changes and raw coverage counts are reported instead.

## Provenance and limitations

The portal and prior-usage inputs are a retrospective research oracle: the endpoint responses are not archived as August 15 snapshots, final destinations can be resolved later, and the portal-to-usage join is name-based. The study therefore tests stability of the selected oracle relationship, not production data readiness. No target-season outcomes enter transfer features or fitting.

## Artifacts

- `rolling_metrics.csv`, `rolling_comparisons.csv`, `frozen_metrics.csv` — required scores and pairwise deltas.
- `training_coverage.csv` — training-panel size and raw transfer-feature coverage.
- `coefficients.csv` — standardized roster-continuity coefficients and fit-to-fit changes.
- `summary.json`, `report.md`, `plots/` — protocol, hashes, interpretation, and plots.
