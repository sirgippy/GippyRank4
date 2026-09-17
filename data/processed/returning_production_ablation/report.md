# Returning-production ablation study (issue 84)

## Result

Removing RP reduced the recent-vs-pre-2022 NLL deterioration. C-full recent-minus-pre-2022 NLL change was **0.0758**; C-minus-RP was **0.0655**; the change in degradation after removing RP was **-0.0102**.
Returning production became less valuable on NLL: its ablation contribution moved toward zero or negative. The RP contribution moved **from 0.0070 pre-2022 to -0.0033 in 2022–2025**; its recent-minus-pre-2022 change was **-0.0102** NLL.
The direction is consistent with the transfer-portal drift hypothesis, but it cannot establish that missing transfer information caused the change.
These are descriptive rolling-origin results, not evidence of a causal transfer-portal break.

## Exact experiment

- C-full uses the current Context C 1.2 location equation: rank-history features plus coach tenure, recruiting, Talent, and all returning-production features.
- C-minus-RP uses the same equation, penalty (0.25), optimizer settings, rolling training windows, and preprocessing, with only the four returning-production features removed.
- Both variants use the full eligible historical FBS population and identical target-season keys. The existing training-only median imputation and missingness indicators are retained for the remaining features.
- For each target season, models are fit only on earlier seasons and scored on the target season. 2026 is excluded; the recent period is 2022–2025 because that is the established held-out Context evaluation period.

Reproduce with:

```text
uv run python scripts/investigate_returning_production_ablation.py --source-root /path/to/cached-input-checkout
```

## Year-by-year results

RP contribution is `NLL(C-minus-RP) - NLL(C-full)`: positive means RP helped; negative means RP hurt. The complete established metric set is in `annual_metrics.csv`; the table shows the primary metrics.

| Season | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS | C-full rank MAE | C-minus-RP rank MAE | C-full 80% cov. | C-minus-RP 80% cov. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2007 | 119 | 4.3856 | 4.3855 | -0.0001 | 0.1037 | 0.1037 | 17.12 | 17.12 | 0.862 | 0.862 |
| 2008 | 119 | 4.4669 | 4.4664 | -0.0005 | 0.1130 | 0.1129 | 18.23 | 18.21 | 0.836 | 0.837 |
| 2009 | 120 | 4.4155 | 4.4152 | -0.0003 | 0.1062 | 0.1062 | 17.39 | 17.39 | 0.867 | 0.867 |
| 2010 | 120 | 4.4906 | 4.4908 | +0.0002 | 0.1159 | 0.1159 | 19.15 | 19.16 | 0.819 | 0.818 |
| 2011 | 120 | 4.3927 | 4.3926 | -0.0001 | 0.1068 | 0.1068 | 17.46 | 17.46 | 0.851 | 0.851 |
| 2012 | 120 | 4.4894 | 4.4894 | -0.0001 | 0.1125 | 0.1125 | 18.47 | 18.46 | 0.853 | 0.853 |
| 2013 | 124 | 4.4393 | 4.4393 | +0.0000 | 0.1038 | 0.1038 | 17.66 | 17.66 | 0.874 | 0.874 |
| 2014 | 125 | 4.4007 | 4.4017 | +0.0009 | 0.0980 | 0.0981 | 17.28 | 17.38 | 0.879 | 0.877 |
| 2015 | 127 | 4.4899 | 4.4375 | -0.0525 | 0.1008 | 0.1015 | 17.05 | 17.57 | 0.873 | 0.878 |
| 2016 | 128 | 4.4747 | 4.4894 | +0.0147 | 0.1106 | 0.1116 | 19.37 | 19.46 | 0.843 | 0.838 |
| 2017 | 128 | 4.4724 | 4.5212 | +0.0488 | 0.1058 | 0.1117 | 18.45 | 19.81 | 0.856 | 0.842 |
| 2018 | 129 | 4.4386 | 4.4599 | +0.0213 | 0.1056 | 0.1070 | 18.42 | 18.85 | 0.853 | 0.845 |
| 2019 | 130 | 4.4293 | 4.4710 | +0.0417 | 0.1063 | 0.1128 | 18.30 | 19.94 | 0.852 | 0.836 |
| 2020 | 127 | 4.5603 | 4.5910 | +0.0308 | 0.1220 | 0.1263 | 19.30 | 20.30 | 0.789 | 0.783 |
| 2021 | 127 | 4.5043 | 4.4998 | -0.0046 | 0.1135 | 0.1132 | 20.37 | 20.47 | 0.849 | 0.841 |
| 2022 | 130 | 4.4612 | 4.4820 | +0.0208 | 0.1065 | 0.1085 | 18.56 | 19.16 | 0.828 | 0.832 |
| 2023 | 131 | 4.4682 | 4.5031 | +0.0349 | 0.1039 | 0.1075 | 19.14 | 19.97 | 0.869 | 0.860 |
| 2024 | 133 | 4.5912 | 4.5681 | -0.0231 | 0.1183 | 0.1167 | 20.98 | 20.83 | 0.793 | 0.812 |
| 2025 | 134 | 4.6079 | 4.5636 | -0.0443 | 0.1151 | 0.1120 | 21.44 | 21.04 | 0.810 | 0.824 |

## Period summary

The period split is a reporting comparison only; no breakpoint was fitted and no model choice used these results.

| Period | Seasons | N | C-full NLL | C-minus-RP NLL | RP contribution | C-full CRPS | C-minus-RP CRPS |
|---|---|---:|---:|---:|---:|---:|---:|
| pre_2022 | 2007;2008;2009;2010;2011;2012;2013;2014;2015;2016;2017;2018;2019;2020;2021 | 1863 | 4.4571 | 4.4641 | +0.0070 | 0.1083 | 0.1096 |
| recent_2022_2025 | 2022;2023;2024;2025 | 528 | 4.5329 | 4.5296 | -0.0033 | 0.1110 | 0.1112 |

## Interpretation

- Returning production is useful when its contribution is positive, but a worse absolute C-minus-RP score is expected and is not by itself evidence against feature drift.
- The primary test is whether the contribution moves toward zero or negative in recent seasons and whether removing RP changes the recent-vs-historical degradation pattern. The result lines above report both directly.
- This run shows that directional pattern: RP contributes 0.0070 before 2022 and -0.0033 in 2022–2025, while the change in recent-vs-pre-2022 degradation after removal is -0.0102. That is a concrete lead, not a causal transfer estimate.

## Limitations

- Returning production is correlated with recruiting, Talent, coaching, and rank history; this is a conditional model ablation, not a causal estimate of transfers.
- The returning-production source excludes incoming transfers, but this study does not observe or reconstruct transfer-adjusted rosters. Other roster changes, source revisions, and changing data coverage are alternative explanations.
- The 2022–2025 comparison is short, and the seasons are not independent. Season-level patterns and the linear slopes in `summary.json` are descriptive.
- The study evaluates historical retrospective source values under the repository's established preprocessing; it does not modify or reforecast the production 2026 artifacts.

## Artifacts

- `annual_metrics.csv` — complete year-by-year model metrics and RP contribution for every metric.
- `per_team_losses.csv` — paired team-season NLL and CRPS losses.
- `period_summary.csv` — pre-2022 versus recent descriptive aggregates.
- `summary.json` — feature lists, hashes, configuration, period changes, and temporal slopes.
- `plots/rp_contribution_nll.png` — annual model NLLs and marginal RP contribution.
