# Season Forecast Validation

Season Simulation V1 was evaluated with frozen preseason Context PMFs for 2022–2025 at three actual-date phases. Preseason uses no game outcomes; early and late use only completed regular-season games at or before that season's cutoff. The final target is the completed regular-season win total.

The production configuration was used unchanged: 2,000 outer draws, exact conditional Poisson-binomial win totals, seed 49,049, and Historical Likelihood V1. No retuning was performed. Context is the primary reported family; History was not substituted for it.

## Forecast metrics

Expected final-win MAE and CRPS are in wins. Coverage is the fraction of realized final totals inside the central predictive interval.

| Phase | Team forecasts | Expected-win MAE | CRPS | 50% coverage | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| preseason | 534 | 1.771 | 1.238 | 68.4% | 91.2% | 98.1% |
| early | 534 | 1.416 | 0.985 | 66.3% | 87.8% | 98.1% |
| late | 534 | 0.677 | 0.434 | 79.6% | 95.1% | 98.9% |
| all | 1602 | 1.288 | 0.886 | 71.4% | 91.4% | 98.4% |

## Threshold calibration

The table reports the mean predicted probability, observed frequency, and Brier score for selected final-win thresholds.

| Phase | Threshold | Predicted | Observed | Brier |
| --- | ---: | ---: | ---: | ---: |
| preseason | 6+ wins | 60.6% | 61.8% | 0.179 |
| preseason | 8+ wins | 37.7% | 36.9% | 0.180 |
| preseason | 10+ wins | 17.0% | 15.0% | 0.099 |
| early | 6+ wins | 61.5% | 61.8% | 0.136 |
| early | 8+ wins | 37.7% | 36.9% | 0.147 |
| early | 10+ wins | 16.3% | 15.0% | 0.078 |
| late | 6+ wins | 61.7% | 61.8% | 0.056 |
| late | 8+ wins | 36.2% | 36.9% | 0.054 |
| late | 10+ wins | 14.8% | 15.0% | 0.041 |
| all | 6+ wins | 61.3% | 61.8% | 0.124 |
| all | 8+ wins | 37.2% | 36.9% | 0.127 |
| all | 10+ wins | 16.0% | 15.0% | 0.072 |

## Current single-game equivalence

The latest current Context snapshot `2026-weekly-2026-09-08T11-43-00.275833Z-context` contains 789 supported future games. Its outer latent-rank integration has maximum absolute error of 0.016304 in home-win probability and 0.850769 points in expected home margin versus the exact single-game V1 mixture.

## Published payload impact

The manifest measures the durable diagnostics and the browser-facing lazy artifact separately. The browser copy keeps team summaries and distributions, while stripping game marginals, cross-game dependence, validation, Monte Carlo, and event-decomposition diagnostics.

| Measure | Value |
| --- | ---: |
| Published snapshots | 13 |
| Durable season-simulation bytes | 16.8 MiB |
| Browser season-simulation bytes | 5.1 MiB |
| Mean browser lazy increment per snapshot | 404.0 KiB |
| Mean team-summary bytes per snapshot | 6.6 KiB |
| Mean game-marginals diagnostic bytes per snapshot | 354.5 KiB |
| Mean event-decomposition diagnostic bytes per snapshot | 548.0 KiB |
| Existing lazy team-season payload total | 21.6 MiB |

Validation is a historical diagnostic over already-frozen seasons, not independent confirmation: those seasons have been used by prior GippyRank research. The check is still leakage-safe with respect to each selected cutoff and does not alter production artifacts.
