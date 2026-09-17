# Returning production as an uncertainty signal (issue 90)

## Conclusion

Removing RP from location (C1) has ΔNLL -0.0132 versus C0 and ΔCRPS -0.0008. This is the strong result in the experiment.
Moving RP into scale (C2) adds only ΔNLL -0.0009 versus C1, with C2 winning in 2023, 2024, 2025 and losing in 2022. Its paired team-season improvement fraction is 0.433, and the season-bootstrap 95% ΔNLL range is -0.0038 to +0.0039.
The representative C2 scale diagnostic says **lower RP increases predicted uncertainty**: scale is 1.2832 at low RP, 1.2051 at median RP, and 1.1448 at high RP, but individual RP scale coefficients have mixed signs. Overall, evidence that RP adds meaningful uncertainty-signal value beyond removing it from location is weak and mixed.

## Frozen protocol

- Every variant is fit once using rows through 2021 and scored unchanged on 2022–2025.
- The target panel is the exact stored production FBS panel, including H fallback for rank-history cold starts.
- Preprocessing, penalty (0.25), optimizer, retry behavior, quadrature, target weighting, and scoring are unchanged.
- C0 is the current production Context location formulation: all Context features enter location and scale retains H only.

## Exact variants

| Variant | Location equation | Scale equation |
|---|---|---|
| C0_current_location | H + coach_tenure_seasons + recruiting_class_rank + recruiting_class_points + recruiting_points_2y_mean + recruiting_points_3y_mean + recruiting_points_4y_mean + recruiting_points_trend + talent_composite + returning_pct_ppa + returning_pct_passing_ppa + returning_pct_receiving_ppa + returning_pct_rushing_ppa | H + none |
| C1_no_rp | H + coach_tenure_seasons + recruiting_class_rank + recruiting_class_points + recruiting_points_2y_mean + recruiting_points_3y_mean + recruiting_points_4y_mean + recruiting_points_trend + talent_composite | H + none |
| C2_rp_scale_only | H + coach_tenure_seasons + recruiting_class_rank + recruiting_class_points + recruiting_points_2y_mean + recruiting_points_3y_mean + recruiting_points_4y_mean + recruiting_points_trend + talent_composite | H + returning_pct_ppa + returning_pct_passing_ppa + returning_pct_receiving_ppa + returning_pct_rushing_ppa |
| C3_rp_both | H + coach_tenure_seasons + recruiting_class_rank + recruiting_class_points + recruiting_points_2y_mean + recruiting_points_3y_mean + recruiting_points_4y_mean + recruiting_points_trend + talent_composite + returning_pct_ppa + returning_pct_passing_ppa + returning_pct_receiving_ppa + returning_pct_rushing_ppa | H + returning_pct_ppa + returning_pct_passing_ppa + returning_pct_receiving_ppa + returning_pct_rushing_ppa |

C1 is the no-RP comparison. C2 is the primary hypothesis: non-RP Context remains in location while RP is removed from location and added to scale. C3 is the secondary both-equations comparator.

## C0 reproduction check

The recomputed C0 maximum absolute metric difference from the stored production evaluation is **5.35e-06**, against tolerance 1e-05; check status: **PASS**.

## Aggregate held-out results

Lower is better for NLL, CRPS, rank MAE, and interval width. Coverage is closer to the nominal 0.80 target.

| Variant | N | NLL | ΔNLL | CRPS | ΔCRPS | Exp. rank MAE | Median rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_current_location | 534 | 4.5423 | +0.0000 | 0.1119 | +0.0000 | 20.16 | 20.59 | 0.825 | 71.9 |
| C1_no_rp | 534 | 4.5292 | -0.0132 | 0.1111 | -0.0008 | 20.23 | 20.48 | 0.834 | 74.6 |
| C2_rp_scale_only | 534 | 4.5282 | -0.0141 | 0.1111 | -0.0009 | 20.22 | 20.49 | 0.840 | 75.3 |
| C3_rp_both | 534 | 4.5397 | -0.0026 | 0.1117 | -0.0002 | 20.12 | 20.56 | 0.818 | 70.8 |

## Year-by-year results

### 2022

| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Exp. MAE | Median MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_current_location | 4.4669 | +0.0000 | 0.1071 | +0.0000 | 18.68 | 18.85 | 0.826 | 72.1 |
| C1_no_rp | 4.4872 | +0.0203 | 0.1090 | +0.0019 | 19.28 | 19.68 | 0.833 | 73.7 |
| C2_rp_scale_only | 4.4935 | +0.0266 | 0.1093 | +0.0022 | 19.27 | 19.66 | 0.834 | 73.7 |
| C3_rp_both | 4.4681 | +0.0013 | 0.1070 | -0.0000 | 18.64 | 18.89 | 0.815 | 70.7 |

### 2023

| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Exp. MAE | Median MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_current_location | 4.4722 | +0.0000 | 0.1044 | +0.0000 | 19.28 | 19.55 | 0.867 | 72.1 |
| C1_no_rp | 4.5047 | +0.0325 | 0.1078 | +0.0034 | 20.10 | 20.41 | 0.861 | 74.7 |
| C2_rp_scale_only | 4.5006 | +0.0284 | 0.1077 | +0.0032 | 20.06 | 20.39 | 0.860 | 75.0 |
| C3_rp_both | 4.4660 | -0.0062 | 0.1042 | -0.0002 | 19.26 | 19.53 | 0.862 | 70.9 |

### 2024

| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Exp. MAE | Median MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_current_location | 4.5840 | +0.0000 | 0.1183 | +0.0000 | 21.02 | 21.71 | 0.805 | 70.9 |
| C1_no_rp | 4.5539 | -0.0301 | 0.1160 | -0.0023 | 20.76 | 20.93 | 0.815 | 73.3 |
| C2_rp_scale_only | 4.5512 | -0.0328 | 0.1158 | -0.0025 | 20.78 | 20.94 | 0.824 | 73.9 |
| C3_rp_both | 4.5864 | +0.0024 | 0.1183 | -0.0000 | 21.01 | 21.64 | 0.793 | 69.6 |

### 2025

| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Exp. MAE | Median MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_current_location | 4.6425 | +0.0000 | 0.1177 | +0.0000 | 21.61 | 22.20 | 0.803 | 72.6 |
| C1_no_rp | 4.5691 | -0.0734 | 0.1115 | -0.0062 | 20.73 | 20.88 | 0.827 | 76.6 |
| C2_rp_scale_only | 4.5661 | -0.0764 | 0.1115 | -0.0063 | 20.72 | 20.95 | 0.842 | 78.7 |
| C3_rp_both | 4.6348 | -0.0077 | 0.1172 | -0.0005 | 21.51 | 22.12 | 0.801 | 72.1 |

## Incremental C1→C2 comparison

These deltas isolate the value of adding RP to scale after RP has already been removed from location. Negative values favor C2. The paired team-season and season-bootstrap summaries are descriptive, not inferential confidence intervals.

| Metric | Δ C2 − C1 |
|---|---:|
| NLL | -0.0009 |
| CRPS | -0.00004 |
| Expected-rank MAE | -0.01 |
| Median-rank MAE | +0.01 |
| 80% coverage | +0.006 |
| 80% interval width | +0.7 |

| Season | ΔNLL | ΔCRPS | Δ80% coverage | Δ80% width |
|---:|---:|---:|---:|---:|
| 2022 | +0.0063 | +0.0003 | +0.001 | -0.1 |
| 2023 | -0.0041 | -0.0002 | -0.001 | +0.3 |
| 2024 | -0.0027 | -0.0002 | +0.009 | +0.6 |
| 2025 | -0.0030 | -0.0001 | +0.015 | +2.1 |

Across 534 paired team-seasons, mean ΔNLL was -0.0009, and C2 was better on 43.3% of team-seasons. The exhaustive ordered season bootstrap over 256 resamples had mean ΔNLL -0.0009, central 95% range -0.0038 to +0.0039, and C2 favored in 73.8% of resamples.

## Scale diagnostics

The scale coefficients are standardized feature coefficients. Positive values mean higher RP raises the modeled scale, conditional on the other features; negative values mean higher RP lowers it.

| Variant | RP feature | In scale? | Standardized coefficient | Missingness coefficient |
|---|---|---:|---:|---:|
| C0_current_location | returning_pct_ppa | no | — | — |
| C0_current_location | returning_pct_passing_ppa | no | — | — |
| C0_current_location | returning_pct_receiving_ppa | no | — | — |
| C0_current_location | returning_pct_rushing_ppa | no | — | — |
| C1_no_rp | returning_pct_ppa | no | — | — |
| C1_no_rp | returning_pct_passing_ppa | no | — | — |
| C1_no_rp | returning_pct_receiving_ppa | no | — | — |
| C1_no_rp | returning_pct_rushing_ppa | no | — | — |
| C2_rp_scale_only | returning_pct_ppa | yes | -0.0181 | +0.0074 |
| C2_rp_scale_only | returning_pct_passing_ppa | yes | +0.0223 | +0.0074 |
| C2_rp_scale_only | returning_pct_receiving_ppa | yes | -0.0152 | +0.0074 |
| C2_rp_scale_only | returning_pct_rushing_ppa | yes | -0.0253 | +0.0074 |
| C3_rp_both | returning_pct_ppa | yes | -0.0025 | +0.0152 |
| C3_rp_both | returning_pct_passing_ppa | yes | +0.0269 | +0.0152 |
| C3_rp_both | returning_pct_receiving_ppa | yes | -0.0091 | +0.0152 |
| C3_rp_both | returning_pct_rushing_ppa | yes | -0.0397 | +0.0152 |

Representative low/median/high RP predictions are in `scale_diagnostics.csv`; the full fitted coefficient table is in `scale_coefficients.csv`.

## Interpretation

- C0 versus C1: C1 changes RP semantics by removing the four RP features from the location equation; its held-out delta is -0.0132 NLL and -0.0008 CRPS.
- C2 versus C1: scale-only RP adds a small aggregate NLL improvement of -0.0009; it loses in 2022 and wins modestly in 2023–2025. The paired and bootstrap summaries above show why this should be treated as weak evidence.
- C2 versus C0: scale-only RP loses in 2022–2023 but improves on C0 in 2024–2025, so it avoids the recent degradation in this panel without being uniformly better across every season.
- C2 calibration: 80% coverage is 0.840 with width 75.3; compare the C0, C1, and C2 rows rather than interpreting coverage alone.
- Directional check: lower RP increases predicted uncertainty, but the individual RP scale coefficients have mixed signs and C2's intervals are wider with coverage farther above nominal. The evidence supports continued investigation of roster continuity as an uncertainty signal, not a production conclusion.

## Limitations

- Returning production is correlated with rank history, recruiting, talent, and coaching; the experiment is a conditional semantic ablation, not a causal transfer analysis.
- The data do not model incoming transfers or reconstruct a transfer-adjusted roster, so low RP may proxy for several roster and measurement processes.
- The 2022–2025 panel contains four season clusters and has appeared in earlier research; uncertainty summaries are descriptive and do not establish independent confirmation.
- The scale diagnostic uses representative feature values and should not be mistaken for a population-average effect; the fitted individual RP coefficients also have mixed signs.

## Reproduction

```text
uv run python scripts/investigate_returning_production_uncertainty.py --source-root /path/to/cached-input-checkout
```

Artifacts: `annual_metrics.csv`, `aggregate_metrics.csv`, `per_team_losses.csv`, `c2_vs_c1_annual.csv`, `c2_vs_c1_per_team.csv`, `c2_vs_c1_summary.json`, `scale_coefficients.csv`, `scale_diagnostics.csv`, `summary.json`, and deterministic plots under `plots/`.
