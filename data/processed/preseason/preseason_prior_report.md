# GippyRank4 preseason prior investigation

## Conclusion

The defensible production candidate is a direct distributional prior conditioned on the previous season's constituent Massey rank distribution. It preserves disagreement and does not introduce a scalar team-strength state. CFBD recruiting, talent, returning-production, and coaching payloads are cached for audit; their available historical values are retrospective or lack archived publication timestamps, so they are not approved as the leakage-safe baseline. Transfer history was not acquired because no usable season-wide endpoint with reconstructable preseason timing was available.

## Sources and leakage assessment

- CFBD `/recruiting/teams`: final class rank/points; useful but historical snapshots are not established as preseason-safe.
- CFBD `/talent`: team talent composite, documented minimum 2015; current values are not a 2026 preseason snapshot.
- CFBD `/player/returning`: PPA and usage returning-production fields; season-wide and preseason-oriented by name, but historical as-of snapshots are unavailable.
- CFBD `/coaches`: coaching identity and hire date; same-season records are not used. Coach-change indicators are exploratory only.
- CFBD transfer history: rejected for this iteration; no accessible endpoint provided a defensible, timestamped historical preseason reconstruction.

## Coverage and eras

The machine-readable per-season, per-subdivision matrix is `coverage_matrix.json`; normalized rows are in `team_season_features.csv`; raw responses are under `data/raw/cfbd/preseason/`. Long-history eligibility is prior outcome only. Modern enriched coverage is endpoint-dependent, with Talent beginning in 2015. FBS and FCS are modeled separately.

## Models and validation

Model A uses lagged constituent-rank distributions. Model B adds exploratory recruiting, Talent, returning production, and coaching features where present. Model C is the same enriched specification with modern coverage. Targets are constituent pseudo-observations weighted so each team-season contributes total weight one. A smooth percentile-normal distribution is mapped to the full discrete rank PMF; prior disagreement contributes to predictive width.

Validation is season-held-out for 2022–2025 and reported in `preseason_model_report.json` with NLL and CRPS by subdivision. `rank_prior_predictions.csv` contains PMFs and expected rank, median, 80% interval, and Top 5/10/25 probabilities for held-out historical team-seasons.

## 2026 feasibility

No 2026 preseason prior is emitted. The season has started, and live/current endpoint values cannot be treated as archived preseason values. This is an intentional no-leakage result, not a missing-data imputation.

## Limitations and next step

Final CFBD recruiting values may be revised after signing, returning-production availability may reflect a methodology change, and no raw publication timestamp is preserved by these endpoints. Acquire archived preseason snapshots before admitting richer features to production. The baseline should be recalibrated on additional seasons and compared with a production ranking-outcome simulator once the preseason model is integrated.
