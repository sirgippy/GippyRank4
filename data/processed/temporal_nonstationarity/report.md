# Within-season team-strength drift investigation

Issue #34 is a research diagnostic. Historical Likelihood V1, the Context/History prior families, Posterior V1, Performance V1, and publication artifacts are unchanged.

## Design

- Training: 2008–2017; development: 2018–2021; evaluation: 2022–2025. 2026 outcomes are not used.
- Context is the primary prior family. Existing 2022–2025 Context PMFs are consumed; missing 2018–2021 PMFs apply the repository's frozen, predeclared Context development fit trained through 2017 to outcome-free historical rank and preseason-context rows.
- For a cutoff `c`, only games with `game_time < c` enter inference. A target is strictly after `c`; the next-game panel uses the first future game for each team at four deterministic, evenly spaced completed-week cutoffs per season, deduplicated by game. An all-future panel was omitted because it repeatedly scores the same games at every cutoff and is not inexpensive on this loopy graph.
- Residual = oriented observed margin − the V1 likelihood location averaged over the paired historical rank-observation PMFs. Positive focal-team residual means better than expected. Demeaned residuals subtract each team-season mean.
- Recency candidates temper only the existing factor: `L_g_tempered = L_g ^ 2^(-age_days / h)`. Static V1 uses the unchanged production call semantics.

## Stage 0

The residual panel contains 52,140 focal-team games and 4,512 team-seasons with at least three games.

| Period | Dimension | Control | Lag | Pairs | Correlation |
|---|---|---|---:|---:|---:|
| development | demeaned | observed | 1 | 9789 | -0.07601619325779568 |
| development | demeaned | shuffle_order | 1 | 244725 | -0.10096388896026644 |
| development | demeaned | unrelated_team | 1 | 220421 | -0.0014009956271197265 |
| development | raw | observed | 1 | 9789 | 0.00595697662202859 |
| development | raw | shuffle_order | 1 | 244725 | -0.018431028572543244 |
| development | raw | unrelated_team | 1 | 220421 | -0.0018339882029197064 |
| evaluation | demeaned | observed | 1 | 11594 | -0.07322535229887353 |
| evaluation | demeaned | shuffle_order | 1 | 289850 | -0.09164892871887764 |
| evaluation | demeaned | unrelated_team | 1 | 271067 | -7.840711366878525e-05 |
| evaluation | raw | observed | 1 | 11594 | -0.009371172792994606 |
| evaluation | raw | shuffle_order | 1 | 289850 | -0.025988188792405414 |
| evaluation | raw | unrelated_team | 1 | 271067 | 0.0001823796037461787 |
| training | demeaned | observed | 1 | 26205 | -0.0679993276081468 |
| training | demeaned | shuffle_order | 1 | 655125 | -0.09623094222866803 |
| training | demeaned | unrelated_team | 1 | 597871 | -0.0016991986553841563 |
| training | raw | observed | 1 | 26205 | -0.0011741234411734283 |
| training | raw | shuffle_order | 1 | 655125 | -0.028067389336074427 |
| training | raw | unrelated_team | 1 | 597871 | -0.0011155817111629302 |

Elapsed-time, game-count, early/late, recent-history, and deterministic shuffle/unrelated-team controls are in `residual_lag_metrics.csv` and `summary.json`.

## Stage 1 development

The frozen gate requires aggregate next-game NLL improvement ≥ 0.010, margin MAE improvement ≥ 0.10, win Brier worsening ≤ 0.001, NLL improvement in at least 3 of 4 seasons, and no season NLL worsening above 0.015.
Selected half-life: **none**. Selection used only 2018–2021; 2022–2025 could not affect selection because it was reserved for the frozen post-selection evaluation. The gate result is **not passed**.

| Candidate | Target | N | Margin NLL | Margin MAE | Win Brier | ΔNLL vs static |
|---|---|---:|---:|---:|---:|---:|
| R112 | next_game 2018 | 459 | 4.3328 | 14.5429 | 0.1899 | -0.0046 |
| R112 | next_game 2019 | 473 | 4.2462 | 13.7010 | 0.1892 | -0.0025 |
| R112 | next_game 2020 | 196 | 4.3012 | 13.3631 | 0.2300 | +0.0025 |
| R112 | next_game 2021 | 463 | 4.2565 | 13.4010 | 0.1755 | -0.0015 |
| R14 | next_game 2018 | 459 | 4.3316 | 14.7430 | 0.1910 | -0.0058 |
| R14 | next_game 2019 | 473 | 4.2561 | 13.7863 | 0.1894 | +0.0073 |
| R14 | next_game 2020 | 196 | 4.3222 | 13.4468 | 0.2316 | +0.0235 |
| R14 | next_game 2021 | 463 | 4.2749 | 13.7498 | 0.1801 | +0.0169 |
| R28 | next_game 2018 | 459 | 4.3274 | 14.5839 | 0.1895 | -0.0100 |
| R28 | next_game 2019 | 473 | 4.2464 | 13.6941 | 0.1885 | -0.0024 |
| R28 | next_game 2020 | 196 | 4.3107 | 13.4163 | 0.2306 | +0.0120 |
| R28 | next_game 2021 | 463 | 4.2606 | 13.4902 | 0.1765 | +0.0026 |
| R56 | next_game 2018 | 459 | 4.3298 | 14.5356 | 0.1895 | -0.0076 |
| R56 | next_game 2019 | 473 | 4.2452 | 13.6814 | 0.1887 | -0.0035 |
| R56 | next_game 2020 | 196 | 4.3043 | 13.3798 | 0.2303 | +0.0057 |
| R56 | next_game 2021 | 463 | 4.2567 | 13.4158 | 0.1756 | -0.0013 |
| Static V1 | next_game 2018 | 459 | 4.3374 | 14.5648 | 0.1905 | +0.0000 |
| Static V1 | next_game 2019 | 473 | 4.2488 | 13.7450 | 0.1899 | +0.0000 |
| Static V1 | next_game 2020 | 196 | 4.2987 | 13.3817 | 0.2297 | +0.0000 |
| Static V1 | next_game 2021 | 463 | 4.2580 | 13.4074 | 0.1756 | +0.0000 |

## Stage 2 evaluation

Stage 2 was not triggered because no candidate cleared the frozen development gate.

## Recommendation: C — Neither residual persistence nor future-game validation provides compelling evidence of nonstationarity.

Any recency gain reported here is diagnostic only. It is not a production recommendation to down-weight old games. If the recommendation is A, the next step is a separate explicit latent-state/state-space Posterior V2 design investigation.

## Confounders and integrity

Opponent adjustment remains in every V1 rank-pair likelihood location; site indicators remain active; elapsed days, rather than game count, determine age; the static comparison naturally includes preseason-prior fade; no injury, quarterback, yards, plays, turnover, or play-by-play data are used.

Production artifact SHA-256 values before and after the run are identical; see `summary.json`. The 2022–2025 evaluation window is leakage-safe under the frozen cutoff construction but is not untouched independent confirmation because it has been used in earlier GippyRank research.

The completed deterministic run took 433.897 seconds. History-prior sensitivity was not run because Context is primary and no candidate cleared the development gate.

Artifacts: `residual_persistence.csv`, `residual_lag_metrics.csv`, `development_future_metrics.csv`, `development_season_metrics.csv`, `model_spec.json`, and `summary.json`; `evaluation_*` and `team_examples.csv` are present only when Stage 2 triggers.
