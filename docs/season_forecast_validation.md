# Season Forecast Validation

Season Simulation V1 was evaluated with frozen preseason Context PMFs for 2022–2025 at three actual-date phases. Preseason uses no game outcomes; early and late use only completed regular-season games at or before that season's cutoff. The final target is the completed regular-season win total.

The production configuration was used unchanged: 2,000 outer draws, exact conditional Poisson-binomial win totals, seed 49,049, and Historical Likelihood V1. No retuning was performed. Context is the primary reported family; History was not substituted for it.

The validation schedule is the tracked frozen artifact `data/validation/season_forecast_schedule.csv` (SHA-256 `a90ca3de478997170a9f69fed8c746033b383b39ab913b20b9409c94b44049d6`). Its provenance records the eight raw CFBD response hashes and the deterministic first-seen game-ID merge. Validation does not read the ignored raw-data cache.

## Schedule accounting

Every regular-season row involving a known FBS team is accounted for. Completed scored rows contribute to the fixed record even when the game is outside Historical Likelihood support. Future and unresolved rows enter the candidate forecast scope; unsupported candidates make the affected team-cutoff unavailable and are excluded from scoring.

| Season | Phase | Schedule rows | FBS-involving | Fixed record | Future candidates | Supported | Unresolved | Unsupported | Scored teams | Unavailable teams |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | preseason | 1552 | 854 | 0 | 854 | 854 | 0 | 0 | 131 | 0 |
| 2022 | early | 1552 | 854 | 250 | 604 | 604 | 0 | 0 | 131 | 0 |
| 2022 | late | 1552 | 854 | 651 | 203 | 203 | 0 | 0 | 131 | 0 |
| 2023 | preseason | 1536 | 868 | 0 | 868 | 868 | 0 | 0 | 133 | 0 |
| 2023 | early | 1536 | 868 | 254 | 614 | 614 | 0 | 0 | 133 | 0 |
| 2023 | late | 1536 | 868 | 653 | 215 | 215 | 0 | 0 | 133 | 0 |
| 2024 | preseason | 1620 | 874 | 0 | 874 | 874 | 0 | 0 | 134 | 0 |
| 2024 | early | 1620 | 874 | 242 | 632 | 632 | 0 | 0 | 134 | 0 |
| 2024 | late | 1620 | 874 | 632 | 242 | 241 | 1 | 1 | 132 | 2 |
| 2025 | preseason | 1637 | 888 | 0 | 888 | 888 | 0 | 0 | 136 | 0 |
| 2025 | early | 1637 | 888 | 251 | 637 | 637 | 0 | 0 | 136 | 0 |
| 2025 | late | 1637 | 888 | 688 | 200 | 200 | 0 | 0 | 136 | 0 |

The unavailable team-cutoffs are retained in the machine-readable summary with their actual target and unsupported game descriptors: 2024 late: App State (2026) — games 401640992; 2024 late: Liberty (2335) — games 401640992.

## Forecast metrics

Expected final-win MAE and CRPS are in wins. Coverage is the fraction of realized final totals inside the central predictive interval.

| Phase | Team forecasts | Expected-win MAE | CRPS | 50% coverage | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| preseason | 534 | 1.773 | 1.239 | 68.2% | 91.2% | 98.1% |
| early | 534 | 1.417 | 0.986 | 66.1% | 87.8% | 98.1% |
| late | 532 | 0.678 | 0.434 | 79.5% | 95.1% | 98.9% |
| all | 1600 | 1.290 | 0.887 | 71.2% | 91.4% | 98.4% |

## Threshold calibration

The table reports the mean predicted probability, observed frequency, and Brier score for selected final-win thresholds.

| Phase | Threshold | Predicted | Observed | Brier |
| --- | ---: | ---: | ---: | ---: |
| preseason | 6+ wins | 60.6% | 61.8% | 0.179 |
| preseason | 8+ wins | 37.7% | 36.9% | 0.180 |
| preseason | 10+ wins | 17.1% | 15.0% | 0.099 |
| early | 6+ wins | 61.5% | 61.8% | 0.136 |
| early | 8+ wins | 37.7% | 36.9% | 0.147 |
| early | 10+ wins | 16.3% | 15.0% | 0.078 |
| late | 6+ wins | 61.8% | 61.8% | 0.057 |
| late | 8+ wins | 36.2% | 36.8% | 0.054 |
| late | 10+ wins | 14.9% | 15.0% | 0.041 |
| all | 6+ wins | 61.3% | 61.8% | 0.124 |
| all | 8+ wins | 37.2% | 36.9% | 0.127 |
| all | 10+ wins | 16.1% | 15.0% | 0.073 |

## Convergence check

For the latest current Context snapshot `2026-weekly-2026-09-08T11-43-00.275833Z-context`, the report reran Ohio State (194) with seed 49049 at outer-draw counts 250, 500, 1000, 2000. The selected final-record rows are the three highest-probability records at the largest run; threshold and quality/game variance fractions are recomputed at each budget.

| Outer draws | Expected final wins | Selected final records | 8+ wins | 10+ wins | 11+ wins | Quality fraction | Game fraction |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 250 | 9.096 | 10-2 0.243; 9-3 0.242; 8-4 0.168 | 0.845 | 0.435 | 0.192 | 0.394 | 0.606 |
| 500 | 9.130 | 10-2 0.246; 9-3 0.235; 8-4 0.159 | 0.845 | 0.452 | 0.206 | 0.432 | 0.568 |
| 1000 | 9.182 | 10-2 0.252; 9-3 0.237; 8-4 0.158 | 0.856 | 0.461 | 0.209 | 0.406 | 0.594 |
| 2000 | 9.153 | 10-2 0.250; 9-3 0.239; 8-4 0.160 | 0.852 | 0.454 | 0.204 | 0.407 | 0.593 |

## Current single-game equivalence

The latest current Context snapshot `2026-weekly-2026-09-08T11-43-00.275833Z-context` contains 789 supported future games. Its outer latent-rank integration has maximum absolute error of 0.016304 in home-win probability and 0.850769 points in expected home margin versus the exact single-game V1 mixture.

## Published payload impact

The manifest measures the durable diagnostics and the browser-facing lazy artifact separately. The browser copy keeps team summaries and distributions, while stripping game marginals, cross-game dependence, validation, Monte Carlo, and event-decomposition diagnostics.

| Measure | Value |
| --- | ---: |
| Published snapshots | 13 |
| Durable season-simulation bytes | 16.8 MiB |
| Browser season-simulation bytes | 5.1 MiB |
| Mean browser lazy increment per snapshot | 405.1 KiB |
| Mean team-summary bytes per snapshot | 6.6 KiB |
| Mean game-marginals diagnostic bytes per snapshot | 354.5 KiB |
| Mean event-decomposition diagnostic bytes per snapshot | 548.0 KiB |
| Existing lazy team-season payload total | 21.6 MiB |

Validation is a historical diagnostic over already-frozen seasons, not independent confirmation: those seasons have been used by prior GippyRank research. The check is still leakage-safe with respect to each selected cutoff and does not alter production artifacts.
