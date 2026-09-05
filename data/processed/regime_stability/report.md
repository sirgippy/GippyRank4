# Regime-stability investigation

## Scope and guardrails

This research refits historical rolling-origin models only. H 1.1, C 1.2, Historical Likelihood V1, and the frozen 2026 preseason PMFs were not modified. Every target-season fit uses outcomes only from earlier seasons; 2026 is excluded.

## Rolling-origin evidence

Targets span 2008–2025; paired H/C scoring uses identical regular FBS team keys. The best descriptive H recency candidate was **H_window_10** (mean ΔNLL -0.0003 versus H-static). The best descriptive C recency candidate was **C_context_hl_5** (mean ΔNLL 0.0105 versus C-static).

The machine-readable tables retain the annual NLL, CRPS, expected/median-rank MAE, interval coverage and width, and Top-5/10/25 Brier scores. Plots show raw annual points; no rule-change date was fit as a breakpoint.

## Prospective family-specific nested strategies

The H-only selector chose an adaptive H candidate in **7/18** target seasons. Its realized aggregate NLL was **4.5038** (ΔNLL **0.0004** versus always H-static), with ΔCRPS **-0.0000** and expected-rank-MAE change **-0.0149**.

The C-only selector chose an adaptive C candidate in **0/18** target seasons. Its realized aggregate NLL was **4.4785** (ΔNLL **0.0000** versus always C-static), with ΔCRPS **0.0000** and expected-rank-MAE change **0.0000**.

These are prospective strategy results: each target's choice uses only earlier rolling target forecasts within its own family. They are distinct from the descriptive hindsight candidate means above. The two-timescale experiment holds rank-history effects slow and fits a recent weighted context correction; it remains research-only, and a future production proposal requires a separate specification and validation PR.

## Artifacts

- `annual_metrics.csv` — paired annual scores for static and adaptive candidates.
- `candidate_results.csv` — descriptive aggregate results and family-specific selection counts.
- `nested_selection.csv` — the prospective selected candidate and realized score for every family/target.
- `coefficient_trajectories.csv` and `feature_distributions.csv` — coefficient/effect and covariate-shift diagnostics.
- `hc_disagreement.csv` and `decomposition_2025.csv` — forecast disagreement and the 2025 C-vs-H NLL decomposition.
- `plots/` — requested annual performance, feature/effect, interval, disagreement, and 2025 plots.

## Findings from this run

- Classification: **C — feature-specific drift**. Direct lag-1-to-target percentile persistence declined gradually from 0.727 (2004–09) to 0.661 (2020–25); it is not evidence of a discrete portal/NIL breakpoint.
- Long-run-history coefficients are variable and their simple linear trend is weak for H (p=0.190), while direct transition diagnostics show only modest recent weakening. A static long-run baseline remains defensible pending uncertainty-aware follow-up.
- H recency/window experiments provide no meaningful validated gain: the best descriptive result is H_window_10 at ΔNLL -0.00025, while the family-specific prospective selector chose adaptive H in 7/18 targets and realized ΔNLL 0.00039 versus always H-static.
- Fast context adaptation did not help: the best slow-H/fast-context half-life is C_context_hl_5 at ΔNLL 0.0105 versus C-static, and the family-specific prospective C selector chose no adaptive candidate (realized ΔNLL 0.0000). Do not introduce a recency-weighted C production model from this evidence.
- Coach-tenure effect is stable (linear coefficient-time p=0.608). Recruiting, Talent, and returning-production coefficient trajectories move substantially, but their early missingness/coverage changes make raw long-run trends descriptive rather than causal.
- Returning total and passing production remain directionally useful once observed; their standardized effects do not support the hypothesized modern weakening. Recruiting/Talent effects are small and unstable conditional on the other context inputs.
- Recent C performance deteriorated in 2025: C beat H by 0.0616 NLL in 2022, 0.0207 in 2023, and 0.0103 in 2024, then lost by 0.0420 in 2025. The 2025 loss sums exactly from team contributions; largest positive C-minus-H contributions were New Mexico (1.88), Utah (1.66), James Madison (1.61), North Texas (1.43).
- H/C disagreement increased modestly in the recent years (mean expected-rank gap 7.76 in 2022 to 8.71 in 2025), so large context adjustments merit audit rather than stronger automatic weighting.
- Recommended next step: preserve H 1.1/C 1.2 and 2026 priors; conduct a separate uncertainty-aware feature-ablation/interaction study using only years with observed context coverage before proposing any production V1.3/C1.3.
