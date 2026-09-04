# GippyRank4 Preseason Prior V1.1

## Recommendation

The frozen rank-history-only production specification is **long_run_baseline**. Selection used only 2018–2021 development holdouts after fitting on 2004–2017; 2022–2025 was evaluated only after this choice. Cold starts retain V1 handling.

## Development experiments

| Candidate | N | NLL | CRPS | 80% coverage | Notes |
|---|---:|---:|---:|---:|---|
| V1_A2_reference | 513 | 4.5380 | 0.1187 | 0.832 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| full_t1_t2_t3_quadrature | 507 | 4.5448 | 0.1192 | 0.835 | strict three-lag eligible population |
| historical_volatility | 513 | 4.5215 | 0.1161 | 0.839 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| lag_depth_4 | 513 | 4.5302 | 0.1177 | 0.834 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| lag_depth_5 | 513 | 4.5282 | 0.1174 | 0.835 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| long_run_baseline | 513 | 4.5197 | 0.1160 | 0.836 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| student_t_df_5 | 513 | 4.5373 | 0.1186 | 0.807 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |
| trajectory_and_reversion | 513 | 4.5204 | 0.1162 | 0.837 | same complete lag-1 population; missing deeper summaries are training-only imputed with indicators |

## Untouched 2022–2025 FBS production comparison

| Model | N | NLL | CRPS | Expected-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|
| V1_1 | 534 | 4.5442 | 0.1137 | 21.26 | 0.839 | 78.0 |
| V1_A2 | 534 | 4.5677 | 0.1172 | 21.71 | 0.834 | 77.9 |

### Consistency across untouched seasons

ΔNLL = V1.1 − V1; negative values favor V1.1.

| Season | V1 NLL | V1.1 NLL | ΔNLL |
|---|---:|---:|---:|
| 2022 | 4.5547 | 4.5276 | -0.0272 |
| 2023 | 4.5226 | 4.4902 | -0.0324 |
| 2024 | 4.6152 | 4.5882 | -0.0271 |
| 2025 | 4.5774 | 4.5696 | -0.0079 |

V1.1 wins 4 / 4 untouched test seasons by NLL: strong consistency across the available seasons, not a claim of overwhelming inferential proof.

The descriptive season-bootstrap candidate-minus-V1 ΔNLL is -0.0236; its central 95% bootstrap range is -0.0311 to -0.0126. V1.1 wins 100.0% of ordered resamples.

Because the untouched test contains only four seasons, this season-cluster bootstrap is a descriptive robustness check rather than a precise confidence interval.

The JSON artifact contains tier calibration/sharpness, per-season scores, full transition and history diagnostics, quadrature approximation notes, and the final frozen model metadata.

## Historical transition behavior

| Prior-rank tier | N | Mean next-rank percentile | Next-rank percentile SD | Prior constituent disagreement |
|---|---:|---:|---:|---:|
| top_10pct | 200 | 0.144 | 0.136 | 0.482 |
| 11_25pct | 330 | 0.300 | 0.196 | 0.381 |
| 26_50pct | 580 | 0.424 | 0.211 | 0.356 |
| 51_75pct | 565 | 0.571 | 0.228 | 0.366 |
| bottom_25pct | 541 | 0.766 | 0.184 | 0.512 |

## Scope and safety

All candidate inputs are functions solely of final-rank distributions from seasons before the target. No offseason, current-season, external ranking, recruiting, betting, or game data is used. Multi-lag uncertainty uses order-statistic product quadrature (two points per lag, eight joint points); exact Cartesian support is used automatically when a tiny input has at most two observations per lag.
