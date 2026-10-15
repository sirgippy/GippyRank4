# Issue 212: generic likelihood recalibration

**Conclusion:** Retain unmodified Context / no useful gain established

This is a fixed, exploratory replay of the verified #195 generic calibration fits. It does not establish an untouched prospective confirmation or authorize production promotion.

## Primary FBS–FBS future-game CRPS

| Evaluation | Unmodified Context | Recalibrated Context | Candidate minus baseline | Games |
|---|---:|---:|---:|---:|
| Pooled equal game | 8.8791 | 8.8946 | +0.0155 | 2264 |
| Equal season | 8.8798 | 8.8954 | +0.0155 | 2264 |
| 2023 | 8.8447 | 8.8560 | +0.0113 | 750 |
| 2024 | 9.0989 | 9.1213 | +0.0223 | 752 |
| 2025 | 8.6959 | 8.7088 | +0.0130 | 762 |
| Preseason pooled equal game | 9.6831 | 9.6775 | -0.0056 | 2264 |
| Pre-Thanksgiving pooled | 8.8860 | 8.9076 | +0.0216 | 2040 |

The fixed gain rule requires at least 0.05 lower CRPS in both pooled equal-game and equal-season summaries, with gains in at least two of three seasons. The separate preseason and pre-Thanksgiving guardrails apply only to a candidate that clears that primary rule.

## Decision and parity

- Primary practical-gain rule passed: `False`.
- Early guardrails passed: `True`.
- #201 baseline parity passed: `True` across 2264 per-game forecasts; maximum metric difference `0` against tolerance `1e-08`.
- Forecast-cohort audit: `{"preseason/fbs_fbs": {"common_cohort": true, "forecast_rows": 4528, "game_count_per_arm": 2264}, "preseason/fbs_fcs": {"common_cohort": true, "forecast_rows": 730, "game_count_per_arm": 365}, "rolling/fbs_fbs": {"common_cohort": true, "forecast_rows": 4528, "game_count_per_arm": 2264}, "rolling/fbs_fcs": {"common_cohort": true, "forecast_rows": 730, "game_count_per_arm": 365}}`.

## Season-specific calibration parameters

| Season | Mean multiplier | Extra non-neutral home term | Log scale correction |
|---:|---:|---:|---:|
| 2023 | 0.970189 | -0.673180 | -0.061714 |
| 2024 | 0.973320 | -0.515524 | -0.063116 |
| 2025 | 0.990025 | -0.365732 | -0.064851 |

## Secondary rank results

Corrected pre-bowl rank scores use only checkpoints strictly before the target observation date. Each team-season contributes total weight one across its valid checkpoints.

| Evaluation | Rank CRPS (baseline → recalibrated) | Δ NLL | Δ expected-rank error | Δ 80% target mass | Δ 80% interval width |
|---|---:|---:|---:|---:|---:|
| Pooled team-season | 0.04455 → 0.04501 | +0.01051 | +0.0622 | -0.0101 | -0.878 |
| 2023 | 0.04447 → 0.04494 | +0.00995 | +0.0701 | -0.0096 | -0.789 |
| 2024 | 0.04589 → 0.04624 | +0.00985 | +0.0220 | -0.0086 | -0.775 |
| 2025 | 0.04330 → 0.04387 | +0.01170 | +0.0939 | -0.0119 | -1.065 |

The recalibrated arm worsens rank NLL, rank CRPS, expected-rank error, and target mass in the pooled summary and each season; its 80% intervals are narrower.

## Pooled game calibration tradeoffs

Changes are recalibrated minus baseline. The 80% coverage target is 0.80.

| Evaluation | Δ margin NLL | Δ win Brier | Δ median absolute error | Δ 80% coverage | Δ 80% interval width |
|---|---:|---:|---:|---:|---:|
| Rolling all phases | +0.0028 | +0.0004 | +0.0152 | -0.0230 | -2.278 |
| Preseason | -0.0032 | +0.0001 | +0.0153 | -0.0155 | -2.076 |
| Pre-Thanksgiving | +0.0030 | +0.0006 | +0.0239 | -0.0225 | -2.277 |

`forecast_summaries.csv` contains level metrics plus season-, phase-, and population-specific values. FBS–FCS forecasts remain a separate population and use the unchanged conditional likelihood; their FBS rank posterior can change indirectly through FBS–FBS evidence.

## Limitations and next confirmation

The 2022–2025 seasons informed prior development and are not fresh confirmation. A promising result needs a separately registered future-season replay with model settings and predictions frozen before outcomes are available. Repeated teams, games, and checkpoints are not independent observations.

The #201 artifact did not retain per-cutoff posterior PMFs. This run regenerates the baseline and candidate PMFs from the hash-verified #200 bundle and stores both losslessly in `posterior_pmfs.jsonl.gz`; baseline metric parity against #201 is checked per game.
