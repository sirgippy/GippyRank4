# Returning-production ablation study (issue 84)

## Primary result: frozen C evaluation

Both variants were fit through 2021 and scored unchanged on 2022–2025. C-full NLL was **4.5437**; C-minus-RP NLL was **4.5304**; RP contribution was **-0.0133** NLL, so returning production hurt on the established held-out panel.
C-full CRPS was **0.1120** versus **0.1112** without RP. Positive RP contribution means RP helped; negative means it hurt.
This frozen comparison is the primary answer to issue 84. It is separate from the rolling-origin diagnostic below.

## Exact experiment

- C-full uses the current Context C 1.2 location equation: rank-history features plus coach tenure, recruiting, Talent, and all returning-production features.
- C-minus-RP uses the same equation, penalty (0.25), production fitting path, optimizer defaults, and iteration-limit retry, with only the four returning-production features removed.
- Both variants use the full eligible historical FBS population and identical target-season keys. The existing training-only median imputation and missingness indicators are retained for the remaining features.
- Primary protocol: fit each variant once on rows through 2021, then score the unchanged fits across 2022–2025. 2026 is excluded.
- Secondary protocol: refit each target season using only earlier seasons, preserving the prior report’s rolling-origin diagnostic for temporal shape analysis.

Reproduce with:

```text
uv run python scripts/investigate_returning_production_ablation.py --source-root /path/to/cached-input-checkout
```

## Primary year-by-year results

RP contribution is `NLL(C-minus-RP) - NLL(C-full)`: positive means RP helped; negative means RP hurt. The complete established metric set is in `annual_metrics.csv`; the table shows the primary metrics.

| Season | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS | C-full rank MAE | C-minus-RP rank MAE | C-full 80% cov. | C-minus-RP 80% cov. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 130 | 4.4616 | 4.4821 | +0.0205 | 0.1065 | 0.1085 | 18.56 | 19.17 | 0.825 | 0.832 |
| 2023 | 131 | 4.4718 | 4.5048 | +0.0330 | 0.1043 | 0.1077 | 19.21 | 20.04 | 0.865 | 0.860 |
| 2024 | 133 | 4.5962 | 4.5658 | -0.0304 | 0.1187 | 0.1164 | 21.02 | 20.76 | 0.803 | 0.814 |
| 2025 | 134 | 4.6417 | 4.5672 | -0.0745 | 0.1184 | 0.1121 | 21.90 | 21.01 | 0.800 | 0.824 |

## Secondary rolling-origin diagnostic

The rolling-origin diagnostic is secondary and does not replace the frozen held-out comparison. It is consistent with declining value in the temporal pattern: RP contribution was 0.0145 in the RP-observable pre-2022 era (2015–2021) and -0.0034 in 2022–2025. Removing RP changed recent-vs-pre-2022 deterioration by -0.0104 NLL.

| Period | Seasons | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS |
|---|---|---:|---:|---:|---:|---:|---:|
| pre_2022 | 2007;2008;2009;2010;2011;2012;2013;2014;2015;2016;2017;2018;2019;2020;2021 | 1863 | 4.4572 | 4.4642 | +0.0070 | 0.1083 | 0.1096 |
| rp_observable_pre_2022 | 2015;2016;2017;2018;2019;2020;2021 | 896 | 4.4810 | 4.4955 | +0.0145 | 0.1092 | 0.1120 |
| recent_2022_2025 | 2022;2023;2024;2025 | 528 | 4.5329 | 4.5296 | -0.0034 | 0.1110 | 0.1112 |

## Interpretation

- Returning production is useful when its contribution is positive, but a worse absolute C-minus-RP score is expected and is not by itself evidence against feature drift.
- The primary frozen result is the relevant held-out answer: RP hurt by -0.0133 NLL on 2022–2025.
- The secondary rolling result compares 0.0070 before 2022, 0.0145 in 2015–2021, and -0.0034 in 2022–2025. This is consistent with the hypothesis that RP became less informative, but it is not causal evidence that omitted transfers caused the drift.

## Limitations

- Returning production is correlated with recruiting, Talent, coaching, and rank history; this is a conditional model ablation, not a causal estimate of transfers.
- The returning-production source excludes incoming transfers, but this study does not observe or reconstruct transfer-adjusted rosters. Other roster changes, source revisions, and changing data coverage are alternative explanations.
- The frozen 2022–2025 comparison is short, and the seasons are not independent. The rolling-origin period comparisons and linear slopes in `summary.json` are descriptive.
- The study evaluates historical retrospective source values under the repository's established preprocessing; it does not modify or reforecast the production 2026 artifacts.

## Artifacts

- `annual_metrics.csv` — primary frozen 2022–2025 model metrics and RP contribution for every metric.
- `per_team_losses.csv` — primary paired team-season NLL and CRPS losses.
- `heldout_summary.csv` — primary aggregate held-out metrics.
- `rolling_annual_metrics.csv` and `rolling_per_team_losses.csv` — secondary rolling-origin diagnostics.
- `period_summary.csv` — secondary pre-2022, RP-observable-era, and recent descriptive aggregates.
- `summary.json` — feature lists, hashes, configuration, period changes, and temporal slopes.
- `plots/heldout_rp_contribution_nll.png` — primary frozen held-out comparison.
- `plots/rp_contribution_nll.png` — secondary rolling-origin comparison.
