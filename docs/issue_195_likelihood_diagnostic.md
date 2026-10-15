# Issue 195: rank-only likelihood Context diagnostic

## Question and status

This study asks whether the existing-policy rank-only game likelihood leaves a predictable regular-season margin association with the repaired Context inputs after conditioning on corrected final-rank measurements and allowing ordinary likelihood recalibration.

The historical run is **complete**: 2,998 common #201 FBS–FBS games were joined to the corrected #204 final-rank measurements and exact #200 Context inputs. The fixed decision rule classified the result as Context association beyond generic calibration. The Context arm improved held-out NLL by 0.004590 nats per game in the pooled comparison and by 0.004604 under equal-season weighting. This is a descriptive ex-post association, not evidence that the current forecast is defective.

The runner writes an input gate and deterministic synthetic-control results if required artifacts are absent; an unscored run is never labeled negative or inconclusive. The 2022–2025 seasons informed prior work and remain exploratory development data.

## Population and inputs

- Start from the common FBS–FBS game IDs in #201's preseason forecast table for 2022–2025. Verify the four #201 arms agree on each game's teams, score, neutral-site flag, and fixed phase label.
- Keep regular-season games, conference championships, and Army–Navy as labeled by #201. Exclude bowls/CFP through the #201 primary cohort. Do not revise #201's phase mapping or attach its earlier snapshot-timing mismatch to this final-rank diagnostic.
- Join the corrected #204 final target distributions and final constituent-rank-pair file. The selected #200 existing-policy likelihood's manifest must match the exact corrected final observations, target, and pair hashes.
- Use `scripts/build_historical_modeling.py`'s `build_joined`, `pseudo_data`, and design-matrix implementation. Keep every same-system pair. If a game has no same-system pair, use the builder's deterministic game-ID-seeded marginal fallback. Never replace a joint pair mixture by expected ranks or independent marginals when same-system pairs exist.
- Require the exact `context_feature_inputs.csv` export from the #200 builder. It contains the repaired Context 1.4 model feature values, explicit missing values, feature-row hash, and #200 predictor-input hash. Exclude a game only if a team-season feature row is absent; an observed missing value remains represented by its missingness indicator.
- Every scored arm uses exactly the same resulting game IDs. The game manifest records all #201 candidates, additional exclusions, phase labels, pair method/count, and pair-mixture hashes.

Corrected final rankings include regular-season outcomes, potentially including the game being diagnosed, and postseason information. They are noisy, endogenous rank measurements. The statistic therefore describes an **ex-post likelihood adequacy association**; it does not test conditional independence given true latent ranks.

## Fixed comparisons

For each game, let `mu_ij` be the frozen #200 conditional mean for corrected rank pair `j`, with `K_i` pairs and Student-t scale `s0`, degrees of freedom 15. A per-game mixture density is the equal-weight mean of the constituent Student-t densities. Each game contributes one log mixture density regardless of `K_i`.

1. **Rank only:** frozen #200 existing-policy likelihood, unchanged. Original V1 is retained by hash as a separately labeled historical reference and is not part of the primary comparison.
2. **Rank/site calibration:** `mu_ij*exp(a) + h*(1-neutral)` with common scale `s0*exp(q)`. Initialize `a=h=q=0`, the frozen baseline identity. The multiplier remains positive; `h` is an additional home term and is zero on neutral sites.
3. **Rank/site plus Context:** the same terms and fitting rules as arm 2, plus a ridge-regularized linear mean correction. Its predictors are training-standardized home-minus-away differences for every Context-only feature and its missingness indicator. Context does not affect scale. No feature selection, interactions, or alternate distributions are used.

Fit the sum of marginalized per-game NLL plus `lambda/2 * ||beta_context||^2`. Non-Context terms are unpenalized. Use L-BFGS-B with analytic gradients; constrain log multiplier to `[-4, 4]`, extra home field to `[-100, 100]` points, and log scale correction to `[-3, 3]`. Record optimizer status, iteration count, objective, and final gradient norm.

## Forward fitting and selection

- 2022 trains the residual controls for 2023 and is not reported as held-out evaluation.
- Fit 2022–2023 for 2024, and 2022–2024 for 2025. The base #200 likelihood stays fixed.
- For each target, choose Context ridge only from `[0.01, 0.1, 1]` using forward validation among earlier seasons. Each validation game receives weight one; folds are pooled by game count. Use `0.1` when only one earlier season is available. Exact ties prefer `0.1`.
- Fit preprocessing only on the current training seasons: for each feature, compute the mean and population standard deviation from both team sides in training games, impute missing values to that mean, standardize, then form home-minus-away differences. Apply the same fitted means/scales to the target season. Standardize missingness indicators with their training-side prevalence and standard deviation.
- Before scoring, preserve this protocol and the `[0.01, 0.1, 1]` grid. The runner records its objective, bounds, fit origin, preprocessing, selected penalty, and convergence checks.

## Evaluation and decision

The primary diagnostic is paired held-out-season game-mixture NLL for arm 3 minus arm 2. Report arm 2 minus arm 1 separately. Negative differences favor the candidate. Publish game and season scores, pooled equal-game and equal-season contrasts, phase summaries, Context coefficients, signed residuals, mixture PIT midpoint values, 50/80/95% interval coverage and widths, and gain/loss concentration.

For integer margins, midpoint PIT uses the predictive mixture CDF at the score-bin edges `y - 0.5` and `y + 0.5`. Central intervals are mixture quantiles. No game-independence standard errors or p-values are reported: teams and seasons repeat.

The descriptive decision rule is fixed before scoring:

- **Context association beyond calibration:** arm 3 improves on arm 2 in pooled equal-game and equal-season NLL and in at least two of the three held-out seasons.
- **Calibration explains the association:** arm 2 improves on arm 1 by those same pooled and seasonal criteria, while arm 3 does not improve on arm 2 in either pooled summary.
- **Inconclusive/no detectable Context association:** all other patterns.

A negative result only addresses this fixed linear Context mean correction. A positive association motivates a separate testable forecast experiment; it does not establish that current Context forecasts are defective. Future regular-season CRPS remains the operational model-comparison measure, with common pre-bowl ranking NLL secondary in a later prospective replay.

## Synthetic controls

The runner simulates bounded deterministic null and positive-control datasets from the frozen #200 rank/site likelihood. Each case uses 240 training games, 96 evaluation games, and four constituent pairs per game. The positive control adds a known five-point mean effect per training standard deviation of one Context contrast. The null passes only when both optimizers converge, the absolute fitted Context effect is at most one point per training standard deviation, and the Context model's held-out NLL is no more than 0.01 nats per game better than generic calibration. The positive control requires a positive fitted effect and lower held-out NLL than generic calibration. These fixed criteria check mixture integration, gradient fitting, null behavior, and recovery of a fixed signal; they do not tune the historical analysis or make a real-data causal claim.

## Reproduction

Restore the preserved #204 sources under `data/raw/`, then use the checked-in regeneration commands in `data/research/massey_rank_population_issue_204/report.md` to produce the corrected corpus. Rebuild the #200 pipeline with its documented `--corpus-root`, `--source-root`, `--game-root`, and `--reference-root` arguments; the updated builder emits `context_feature_inputs.csv` and records its hash in the #200 manifest. Finally run:

```sh
uv run python scripts/run_issue_195_likelihood_diagnostic.py \
  --corpus-root CORPUS_ROOT \
  --issue200-root ISSUE200_OUTPUT_ROOT \
  --issue201-root data/research/issue_201_prebowl_forecast \
  --output-root data/research/issue_195_likelihood_diagnostic
```

All roots are explicit and can point to the retained local research-data corpus. Raw data remain unchanged. No 2026 ranks are read or required.
