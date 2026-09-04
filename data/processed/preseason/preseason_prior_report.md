# GippyRank4 Preseason Prior V1

## Recommendation

The leakage-safe production candidate is Model A2 (t-1+t-2+t-3): a direct heteroscedastic Normal distribution over the logit within-subdivision final-rank percentile. Its t-1 input is the full empirical constituent-rank distribution; additional lags improve untouched FBS scoring and remain rank-history-only. It does not infer a scalar team-strength state. Model B has no additional qualified feature in the current cached sources, so it is intentionally identical to the selected rank-history candidate rather than promoting retrospective fields.

## Target and uncertainty semantics

`CMP` is excluded and `MAS` remains an ordinary constituent. Each historical team-season has weight one: current constituent outcomes are averaged within a team-season, while previous-season constituents are equal-weight deterministic quadrature points in the conditional distribution. Thus disagreement in the previous season is conditioning uncertainty, and disagreement in the current season is outcome uncertainty; neither changes a team's total weight.

## Feature safety

Rank history is production-safe. CFBD recruiting final values, Team Talent Composite, returning production, and coaching payloads are exploratory because the cache lacks archival as-of preseason timing; they are not production inputs. Transfer data remains rejected because timing cannot be reconstructed. Missing exploratory values use training-only median imputation plus an explicit indicator. Numeric inputs are standardized using only the training partition and that metadata is written to the machine-readable report.

## Validation

All development choices use 2004–2017 training and 2018–2021 season holdouts. 2022–2025 is untouched until final fitting and evaluation. FBS and FCS rank universes are always fit and scored separately. `preseason_model_report.json` contains team-season NLL, CRPS, rank MAE, interval coverage and width, Top-5/10/25 calibration, ablations, preprocessing, and diagnostics; `rank_prior_predictions.csv` contains the full integrated PMFs.

- A_t1: standard-lag FBS N=528, NLL=4.590, CRPS=0.1202, expected-rank MAE=22.4, 80% coverage=0.832.
- A2_t1_t2: standard-lag FBS N=528, NLL=4.584, CRPS=0.1193, expected-rank MAE=22.1, 80% coverage=0.833.
- A2_t1_t2_t3: standard-lag FBS N=528, NLL=4.569, CRPS=0.1174, expected-rank MAE=21.8, 80% coverage=0.832.
- B_safe_long_history: standard-lag FBS N=528, NLL=4.569, CRPS=0.1174, expected-rank MAE=21.8, 80% coverage=0.832.
- C_exploratory_modern: standard-lag FBS N=526, NLL=4.543, CRPS=0.1114, expected-rank MAE=20.2, 80% coverage=0.834.

The production A2 headline combines all 534 FBS target team-seasons: NLL=4.568, CRPS=0.1172, expected-rank MAE=21.7, and 80% coverage=0.834. The standard-lag A2 row above is a diagnostic subset, not the production headline.

Model A2 is the explicit lag ablation (t-1+t-2 and t-1+t-2+t-3); B has no currently qualified safe covariate beyond rank history. Model C is evaluated only on its high-coverage FBS modern subset and remains exploratory, so its score is not a production-selection comparison. The machine-readable report records all FBS/FCS breakdowns, feature coverage, Top-5/10/25 calibration gaps, and scale diagnostics.

## Conditional Top-N calibration

Top-N diagnostics now include coarse probability reliability bins with fractional empirical constituent outcome frequencies, in addition to aggregate gaps. Brier scores for the production FBS A2 model are:

- Top 5: Brier=0.0171; full coarse reliability bins are in `preseason_model_report.json`.
- Top 10: Brier=0.0330; full coarse reliability bins are in `preseason_model_report.json`.
- Top 25: Brier=0.0783; full coarse reliability bins are in `preseason_model_report.json`.

## Cold starts and fair exploratory comparison

Every FBS target team-season now receives a prior. The learned FCS-to-FBS transition fit uses 11 pre-2022 transitions. The untouched test population has 6 cold starts: 6 transition and 0 generic. Programs without any prior distribution use a broad analytical FBS no-prior fallback. FCS cold starts remain explicitly reported as omitted (161 historical team-seasons).

- Standard A2: N=528, NLL=4.569, CRPS=0.1174, expected-rank MAE=21.8, 80% coverage=0.832.
- FCS-to-FBS transition: N=6, NLL=4.417, CRPS=0.1027, expected-rank MAE=17.9, 80% coverage=0.983.
- Combined production: N=534, NLL=4.568, CRPS=0.1172, expected-rank MAE=21.7, 80% coverage=0.834.

On the exact 526-team-season modern FBS subset, A2 t-1+t-2+t-3 has NLL=4.569; exploratory Model C has NLL=4.543. This is incremental signal only: C remains timing-uncertain and is not promoted.

## Discrete probabilities and 2026

PMFs integrate continuous Normal mass across transformed rank bins. The first and last bins have infinite exterior boundaries, preserving uncertainty at ranks 1 and N. No 2026 prior is created: the season has begun and the repository has no independently archived 2026 preseason snapshot. A valid reconstruction requires dated, pre-kickoff snapshots for every promoted feature and a roster of the eligible subdivision population.
