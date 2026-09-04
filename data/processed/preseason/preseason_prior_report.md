# GippyRank4 Preseason Prior V1

## Recommendation

The leakage-safe production candidate is Model A2 (t-1+t-2+t-3): a direct heteroscedastic Normal distribution over the logit within-subdivision final-rank percentile. Its t-1 input is the full empirical constituent-rank distribution; additional lags improve untouched FBS scoring and remain rank-history-only. It does not infer a scalar team-strength state. Model B has no additional qualified feature in the current cached sources, so it is intentionally identical to the selected rank-history candidate rather than promoting retrospective fields.

## Target and uncertainty semantics

`CMP` is excluded and `MAS` remains an ordinary constituent. Each historical team-season has weight one: current constituent outcomes are averaged within a team-season, while previous-season constituents are equal-weight deterministic quadrature points in the conditional distribution. Thus disagreement in the previous season is conditioning uncertainty, and disagreement in the current season is outcome uncertainty; neither changes a team's total weight.

## Feature safety

Rank history is production-safe. CFBD recruiting final values, Team Talent Composite, returning production, and coaching payloads are exploratory because the cache lacks archival as-of preseason timing; they are not production inputs. Transfer data remains rejected because timing cannot be reconstructed. Missing exploratory values use training-only median imputation plus an explicit indicator. Numeric inputs are standardized using only the training partition and that metadata is written to the machine-readable report.

## Validation

All development choices use 2004–2017 training and 2018–2021 season holdouts. 2022–2025 is untouched until final fitting and evaluation. FBS and FCS rank universes are always fit and scored separately. `preseason_model_report.json` contains team-season NLL, CRPS, rank MAE, interval coverage and width, Top-5/10/25 calibration, ablations, preprocessing, and diagnostics; `rank_prior_predictions.csv` contains the full integrated PMFs.

- A_t1: FBS N=528, NLL=4.590, CRPS=0.1200, expected-rank MAE=22.4, 80% coverage=0.820.
- A2_t1_t2: FBS N=528, NLL=4.584, CRPS=0.1191, expected-rank MAE=22.2, 80% coverage=0.817.
- A2_t1_t2_t3: FBS N=528, NLL=4.573, CRPS=0.1178, expected-rank MAE=21.8, 80% coverage=0.839.
- B_safe_long_history: FBS N=528, NLL=4.573, CRPS=0.1178, expected-rank MAE=21.8, 80% coverage=0.839.
- C_exploratory_modern: FBS N=526, NLL=4.528, CRPS=0.1113, expected-rank MAE=20.3, 80% coverage=0.835.

Model A2 is the explicit lag ablation (t-1+t-2 and t-1+t-2+t-3); B has no currently qualified safe covariate beyond rank history. Model C is evaluated only on its high-coverage FBS modern subset and remains exploratory, so its score is not a production-selection comparison. The machine-readable report records all FBS/FCS breakdowns, feature coverage, Top-5/10/25 calibration gaps, and scale diagnostics.

## Discrete probabilities and 2026

PMFs integrate continuous Normal mass across transformed rank bins. The first and last bins have infinite exterior boundaries, preserving uncertainty at ranks 1 and N. No 2026 prior is created: the season has begun and the repository has no independently archived 2026 preseason snapshot. A valid reconstruction requires dated, pre-kickoff snapshots for every promoted feature and a roster of the eligible subdivision population.
