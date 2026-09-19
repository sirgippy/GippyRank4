# Defensive-transfer experience rolling-origin stability (issue 109)

## Conclusion

Final recommendation: retain D5 and stop defensive feature research

R2 beats D5 (R1) by rolling-origin NLL in 1 / 4 target seasons. The weighted rolling ΔNLL (R2 − R1) is 0.002410.
The defensive-value coefficient signs are negative, positive, positive, negative with 2 adjacent non-zero sign flips. The availability indicator dominates the value coefficient in 4 / 4 fits.

The promotion screen is intentionally stricter than a favorable aggregate: because the frozen incremental gain is small, mixed season-level evidence remains ambiguous and defaults to D5.

## Exact predeclared candidates

| Candidate | Definition | Features added beyond C-minus-RP |
|---|---|---|
| R0 | Production Context C 1.2 (C0). | `returning_pct_ppa`, `returning_pct_passing_ppa`, `returning_pct_receiving_ppa`, `returning_pct_rushing_ppa` |
| R1 | C-minus-RP baseline plus total RP and incoming prior offensive usage. | `returning_pct_ppa`, `transfer_in_prior_usage_sum` |
| R2 | D5 plus imported prior defensive experience and its availability indicator. | `returning_pct_ppa`, `transfer_in_prior_usage_sum`, `transfer_in_prior_defensive_experience_sum`, `transfer_in_prior_defensive_experience_available` |

R0 is production Context C 1.2. R1 is the frozen D5 representation: C-minus-RP baseline features, `returning_pct_ppa`, and `transfer_in_prior_usage_sum`. R2 adds only `transfer_in_prior_defensive_experience_sum` and `transfer_in_prior_defensive_experience_available`.

## Frozen parity controls

Parity uses checked-in study artifacts as the canonical references; rounded issue text is not used as a reference. Every required predictive metric must be within its declared tolerance before rolling results are interpreted.

| Control | Reference | Maximum absolute metric delta | Tolerance | Status |
|---|---|---:|---:|:---:|
| R0 | `production C0` | 0.00000535 | 0.00100000 | passed |
| R1 | `D5_total_rp_plus_incoming` | 0.00000000 | 0.00000100 | passed |
| R2 | `E4_D5_plus_defensive_experience` | 0.00000000 | 0.00000100 | passed |

| Metric | R0 frozen | R1 frozen | R2 frozen |
|---|---:|---:|---:|
| nll | 4.54232675 | 4.50808063 | 4.50722825 |
| crps | 0.11194307 | 0.10798618 | 0.10776299 |
| expected_rank_mae | 20.16301080 | 19.48957465 | 19.40245690 |
| median_rank_mae | 20.59456929 | 19.67509363 | 19.64325843 |
| interval_80_coverage | 0.82497705 | 0.83770277 | 0.83692604 |
| interval_80_average_width | 71.91947566 | 71.95880150 | 71.53558052 |

## Rolling-origin protocol

Each target season is fit using only team-seasons strictly before that target: 2022←2021, 2023←2022, 2024←2023, and 2025←2024. All fits use the existing Context C 1.2 family, penalty 0.25, training-only preprocessing and missingness indicators, the deterministic optimizer retry, the production FBS population, the August 15 season-relative portal cutoff, and stored H PMFs for fallback rows. No target-season outcomes enter fitting or feature construction.

| Target | Train through | R0 N | R1 N | R2 N |
|---:|---:|---:|---:|---:|
| 2022 | 2021 | 131 | 131 | 131 |
| 2023 | 2022 | 133 | 133 | 133 |
| 2024 | 2023 | 134 | 134 | 134 |
| 2025 | 2024 | 136 | 136 | 136 |

## Training-data coverage

Counts are calculated from raw feature availability before DirectRankModel imputation. An observed no-incoming defensive-transfer row has natural experience zero but is available; unresolved rows have neutralized zero plus availability zero.

| Target | Training team-seasons | Transfer-covered seasons | Observed incoming usage | Usage fraction | Defensive source available | Defensive fraction | Defensive unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 2216 | 1 | 94 | 0.042 | 52 | 0.023 | 2164 |
| 2023 | 2346 | 2 | 205 | 0.087 | 104 | 0.044 | 2242 |
| 2024 | 2477 | 3 | 322 | 0.130 | 155 | 0.063 | 2322 |
| 2025 | 2610 | 4 | 451 | 0.173 | 193 | 0.074 | 2417 |

## Rolling predictive metrics

All deltas are candidate minus reference; negative NLL and CRPS favor the candidate.

| Target | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | ΔNLL vs R0 | ΔNLL vs R1 | ΔCRPS vs R0 | ΔCRPS vs R1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | R0 | 4.4669 | 0.1071 | 18.68 | 18.85 | 0.826 | 72.10 | 0.000000 | 0.010500 | 0.000000 | 0.001466 |
| 2022 | R1 | 4.4564 | 0.1056 | 18.52 | 18.53 | 0.833 | 71.69 | -0.010500 | 0.000000 | -0.001466 | 0.000000 |
| 2022 | R2 | 4.4607 | 0.1057 | 18.49 | 18.52 | 0.833 | 71.60 | -0.006159 | 0.004342 | -0.001421 | 0.000046 |
| 2023 | R0 | 4.4686 | 0.1041 | 19.21 | 19.59 | 0.871 | 71.92 | 0.000000 | 0.000514 | 0.000000 | -0.000022 |
| 2023 | R1 | 4.4680 | 0.1041 | 19.16 | 19.36 | 0.870 | 72.01 | -0.000514 | 0.000000 | 0.000022 | 0.000000 |
| 2023 | R2 | 4.4712 | 0.1047 | 19.42 | 19.73 | 0.869 | 72.08 | 0.002634 | 0.003148 | 0.000618 | 0.000596 |
| 2024 | R0 | 4.5793 | 0.1179 | 20.98 | 21.64 | 0.794 | 70.50 | 0.000000 | 0.026082 | 0.000000 | 0.002887 |
| 2024 | R1 | 4.5533 | 0.1151 | 20.59 | 21.04 | 0.813 | 70.34 | -0.026082 | 0.000000 | -0.002887 | 0.000000 |
| 2024 | R2 | 4.5557 | 0.1154 | 20.62 | 21.05 | 0.809 | 70.54 | -0.023625 | 0.002456 | -0.002556 | 0.000331 |
| 2025 | R0 | 4.6087 | 0.1145 | 21.15 | 21.65 | 0.813 | 73.46 | 0.000000 | 0.056361 | 0.000000 | 0.006723 |
| 2025 | R1 | 4.5524 | 0.1078 | 19.93 | 20.07 | 0.828 | 73.54 | -0.056361 | 0.000000 | -0.006723 | 0.000000 |
| 2025 | R2 | 4.5522 | 0.1078 | 19.91 | 20.14 | 0.828 | 73.65 | -0.056580 | -0.000219 | -0.006738 | -0.000015 |

## Explicit pairwise rolling deltas

These rows make the three requested comparisons explicit: R1 − R0, R2 − R1, and R2 − R0.

| Target | Comparison | ΔNLL | ΔCRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ 80% width |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2022 | R1 − R0 | -0.010500 | -0.001466 | -0.16 | -0.33 | 0.007 | -0.40 |
| 2022 | R2 − R1 | 0.004342 | 0.000046 | -0.03 | -0.01 | 0.000 | -0.10 |
| 2022 | R2 − R0 | -0.006159 | -0.001421 | -0.19 | -0.34 | 0.008 | -0.50 |
| 2023 | R1 − R0 | -0.000514 | 0.000022 | -0.05 | -0.23 | -0.001 | 0.09 |
| 2023 | R2 − R1 | 0.003148 | 0.000596 | 0.26 | 0.37 | -0.001 | 0.07 |
| 2023 | R2 − R0 | 0.002634 | 0.000618 | 0.21 | 0.14 | -0.002 | 0.16 |
| 2024 | R1 − R0 | -0.026082 | -0.002887 | -0.39 | -0.60 | 0.019 | -0.16 |
| 2024 | R2 − R1 | 0.002456 | 0.000331 | 0.03 | 0.01 | -0.004 | 0.20 |
| 2024 | R2 − R0 | -0.023625 | -0.002556 | -0.36 | -0.59 | 0.015 | 0.04 |
| 2025 | R1 − R0 | -0.056361 | -0.006723 | -1.23 | -1.57 | 0.016 | 0.08 |
| 2025 | R2 − R1 | -0.000219 | -0.000015 | -0.01 | 0.07 | 0.000 | 0.11 |
| 2025 | R2 − R0 | -0.056580 | -0.006738 | -1.24 | -1.51 | 0.016 | 0.19 |

## Weighted rolling aggregate

The aggregate is team-season weighted across the four target seasons.

| Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 4.5316 | 0.1109 | 20.02 | 20.45 | 0.826 | 72.00 |
| R1 | 4.5080 | 0.1082 | 19.55 | 19.76 | 0.836 | 71.90 |
| R2 | 4.5105 | 0.1084 | 19.62 | 19.87 | 0.835 | 71.97 |

## R2 versus R1 paired NLL diagnostics

Negative paired ΔNLL means R2 improves the same team-season under the same target population.

| Scope | N | Mean ΔNLL | Median ΔNLL | Fraction improved |
|---|---:|---:|---:|---:|
| aggregate | 534 | 0.002410 | -0.000002 | 0.500 |
| 2022 | 131 | 0.004342 | -0.000650 | 0.534 |
| 2023 | 133 | 0.003148 | 0.003443 | 0.429 |
| 2024 | 134 | 0.002456 | -0.001982 | 0.530 |
| 2025 | 136 | -0.000219 | -0.000008 | 0.507 |

## Standardized R2 coefficient paths

Coefficients are for the location equation after training-only standardization. The availability coefficient is reported separately from the defensive value coefficient; source counts are included in the machine-readable CSV.

| Target | Feature | Coefficient | Sign | Change from prior fit | Location missingness coefficient | Source available | Source unresolved |
|---:|---|---:|---|---:|---:|---:|---:|
| 2022 | `returning_pct_ppa` | -0.203089 | negative | n/a | 0.005466 | 1021 | 1195 |
| 2022 | `transfer_in_prior_usage_sum` | -0.033021 | negative | n/a | -0.018652 | 94 | 2122 |
| 2022 | `transfer_in_prior_defensive_experience_sum` | -0.016434 | negative | n/a | 0.000000 | 52 | 2164 |
| 2022 | `transfer_in_prior_defensive_experience_available` | 0.028367 | positive | n/a | 0.000000 | 52 | 2164 |
| 2023 | `returning_pct_ppa` | -0.211536 | negative | -0.008447 | -0.005081 | 1151 | 1195 |
| 2023 | `transfer_in_prior_usage_sum` | -0.063143 | negative | -0.030122 | -0.034223 | 205 | 2141 |
| 2023 | `transfer_in_prior_defensive_experience_sum` | 0.015252 | positive | 0.031686 | 0.000000 | 104 | 2242 |
| 2023 | `transfer_in_prior_defensive_experience_available` | 0.018233 | positive | -0.010135 | 0.000000 | 104 | 2242 |
| 2024 | `returning_pct_ppa` | -0.225452 | negative | -0.013916 | -0.009284 | 1282 | 1195 |
| 2024 | `transfer_in_prior_usage_sum` | -0.057531 | negative | 0.005612 | 0.053418 | 322 | 2155 |
| 2024 | `transfer_in_prior_defensive_experience_sum` | 0.010338 | positive | -0.004914 | 0.000000 | 155 | 2322 |
| 2024 | `transfer_in_prior_defensive_experience_available` | 0.016180 | positive | -0.002053 | 0.000000 | 155 | 2322 |
| 2025 | `returning_pct_ppa` | -0.221820 | negative | 0.003632 | -0.021351 | 1415 | 1195 |
| 2025 | `transfer_in_prior_usage_sum` | -0.079771 | negative | -0.022241 | 0.081306 | 451 | 2159 |
| 2025 | `transfer_in_prior_defensive_experience_sum` | -0.004625 | negative | -0.014962 | 0.000000 | 193 | 2417 |
| 2025 | `transfer_in_prior_defensive_experience_available` | 0.019741 | positive | 0.003561 | 0.000000 | 193 | 2417 |

## Availability-indicator diagnostic

The explicit availability path is not treated as equivalent to defensive experience. A larger absolute availability coefficient is flagged as dominance because it can indicate that resolvability, rather than the numeric experience value, explains the gain.

| Target | Defensive value coefficient | Availability coefficient | Availability dominates |
|---:|---:|---:|:---:|
| 2022 | -0.016434 | 0.028367 | yes |
| 2023 | 0.015252 | 0.018233 | yes |
| 2024 | 0.010338 | 0.016180 | yes |
| 2025 | -0.004625 | 0.019741 | yes |

## Frozen versus rolling comparison

The frozen rows fit through 2021 once and score 2022–2025. Rolling rows refit before each target. The frozen study preserves the original PR #107 result; the rolling study tests whether it persists as portal-era seasons enter training.

| Protocol | Candidate | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---|---:|---:|---:|---:|---:|---:|
| frozen | R0 | 4.5423 | 0.1119 | 20.16 | 20.59 | 0.825 | 71.92 |
| frozen | R1 | 4.5081 | 0.1080 | 19.49 | 19.68 | 0.838 | 71.96 |
| frozen | R2 | 4.5072 | 0.1078 | 19.40 | 19.64 | 0.837 | 71.54 |
| rolling | R0 | 4.5316 | 0.1109 | 20.02 | 20.45 | 0.826 | 72.00 |
| rolling | R1 | 4.5080 | 0.1082 | 19.55 | 19.76 | 0.836 | 71.90 |
| rolling | R2 | 4.5105 | 0.1084 | 19.62 | 19.87 | 0.835 | 71.97 |

## Decision answers

- Does defensive experience remain useful as later portal-era seasons enter training? No on the weighted rolling NLL, but the season-level evidence is decisive only under the promotion screen above.
- Is the coefficient directionally stable? No; the defensive-value path is negative, positive, positive, negative.
- Is the effect broad or concentrated in one season? The paired team-season improvement fraction is 0.500; see the year-level paired table for concentration.
- Is the gain actually driven by availability? Availability dominates in 4 of 4 fits; this is treated as a concern by the promotion screen.
- Should defensive experience be retained in the candidate architecture? The final recommendation is `retain D5 and stop defensive feature research`.

## Provenance and limitations

The portal and prior-usage inputs are the same retrospective research oracle used by the frozen transfer studies. Raw payloads are read unchanged from `--transfer-root`; they are not rewritten or copied into the generated artifacts. The defensive audit is likewise read as a frozen input. This is a stability decision, not production snapshot engineering or a redesign of the defensive feature.

## Artifacts

- `rolling_metrics.csv`, `frozen_metrics.csv`, `rolling_aggregate.csv` — per-target and weighted metrics.
- `rolling_comparisons.csv` — explicit R1−R0, R2−R1, and R2−R0 deltas.
- `training_coverage.csv` — raw training-panel coverage by rolling fit.
- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — paired R2-vs-R1 diagnostics.
- `coefficients.csv`, `availability_diagnostics.csv` — coefficient paths and availability comparison.
- `summary.json`, `report.md`, `plots/` — parity, protocol, provenance, interpretation, and plots.
