# Issue 195: rank-only likelihood Context diagnostic

**Decision: Context association beyond calibration.**

This is a fixed ex-post game-margin mechanism diagnostic. The primary comparison is the rank/site/Context arm against the same rank/site generic calibration control; generic calibration versus the frozen rank-only likelihood is reported separately.

- Corrected #204 final-rank/common #201 cohort: 2,998 games after 0 additional exclusions.
- Held-out evaluation seasons: 2023, 2024, 2025; 2022 is residual training only.
- Context versus generic calibration: pooled equal-game ΔNLL -0.004590, equal-season ΔNLL -0.004604; negative favors Context.
- Generic calibration versus frozen likelihood: pooled equal-game ΔNLL -0.002948, equal-season ΔNLL -0.002943; negative favors recalibration.

## Interpretation limits

Corrected final ranking measurements incorporate regular-season outcomes, potentially including the game being diagnosed, and postseason information. They are noisy measurements rather than latent rank truth. A Context association may reflect missing game-result structure, ordinary calibration, or rank measurement error/endogeneity; it does not directly test conditional independence given true ranks. These 2022–2025 seasons have informed earlier research and remain exploratory development data. Repeated teams and seasons are not independent confirmation. The negative result, if any, addresses only this fixed linear mean correction.

The ex-post margin diagnostic does not replace future-game regular-season CRPS as the operational comparison measure or common pre-bowl ranking NLL as a secondary prospective measure. No likelihood, ranking, inference setting, website default, or published snapshot is changed.

## Reproduction

```sh
uv run python scripts/run_issue_195_likelihood_diagnostic.py --corpus-root CORPUS_ROOT --issue200-root ISSUE200_OUTPUT_ROOT --issue201-root data/research/issue_201_prebowl_forecast --output-root data/research/issue_195_likelihood_diagnostic
```

The existing `scripts/build_historical_modeling.py` join and pseudo-data builders construct the corrected final-rank mixtures. `game_manifest.csv` binds the #201 game IDs, final-rank pair mixtures, phase labels, pair counts, and exclusions. `fit_records.json` records forward training seasons, regularization selection, preprocessing, coefficients, objective, and convergence checks.
