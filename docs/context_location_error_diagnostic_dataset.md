# Context location-error diagnostic dataset

This artifact joins the frozen #147 rolling-origin Context 1.3 / History 1.1 comparison with exact Context location inputs and training support. It is an infrastructure artifact; no error pattern is interpreted here.

## Population

Expected: 534 team-seasons. Included: 534. Excluded: 0. Of these, 528 have fitted location decompositions and 6 retain native cold-start fallback priors with blank feature/contribution fields.

## #147 reproduction

All 700 retained #147 baseline checks passed (maximum absolute error 8.74e-10); this builder also checks all 534 Context and History prior NLL, CRPS, and expected-rank values against the final checkpoint rows, checks the team population, and requires SHA-256 parity for each rolling Context fit's metadata. Final posterior values are copied from #147's CC and HH rows at the final shared checkpoint.

## Feature coverage and contributions

15 location features come from the production fitting specification; all 15 are decomposed for each fitted team. The six fallback cases have no fitted Context location center. Missing diagnostic field counts: {'fitted_context_location_center': 6, 'observed_db_impact_sum': 34, 'incoming_offensive_observed_usage_sum': 6}. The fitted center includes the intercept, 15 standardized feature contributions, 15 missing-indicator contributions, and the mean t-1 rank-distribution quadrature contribution. Maximum / mean absolute reconstruction residual: 8.88e-16 / 1.31e-16.

## Transfer coverage

DB states: {'complete': 238, 'no incoming DB players': 80, 'partial': 182, 'unavailable': 34}. The 15 fitted-row frozen/corrected differences reconcile one-to-one with #151's historical feature-change inventory: {'ambiguous_usage_join_removed': 1, 'legitimate_zero_restoration': 13, 'name_normalization_join_added': 1}. Each `*_difference_reason` is the authoritative `change_class`; historical timing remains in the separate `*_difference_timing_status` and `transfer_checkpoint_status` fields. Repaired #151 historical evidence is descriptive and has unverified August 15 availability. The frozen #147 model-facing values remain in separate columns. Observed DB sums are reconstructed from retained player audit rows marked on or before the cutoff, with count parity against #151 and sum parity against complete corrected aggregates. Unavailable DB sums remain blank; neutral model zeros are never presented as observed partial sums. Unknown offensive applicability remains a separate count.

## Training support

Nearest-neighbor Euclidean distance uses the 15 training-standardized numeric features plus their 15 missing indicators and fixed k=5. Minimum / median / maximum nearest distance: {'min': 0.8713213049261718, 'median': 2.456458970585671, 'max': 18.674838286075136}. Feature range-violation counts: {0: 500, 1: 24, 2: 2, 3: 1, 4: 1}. Percentiles use midranks against the target season's rolling training rows only.

## Conventions and provenance

Signed expected-rank error is forecast minus target; positive means a worse (numerically larger) predicted rank. Positive Context-minus-History absolute-rank error means Context was farther from the target. Positive Context-minus-History final NLL means History assigned the target more probability. Preseason NLL, CRPS, absolute error, interval target mass, and final posterior scores are outcome-derived evaluation columns; they are never used as preseason inputs. Source paths and SHA-256 hashes are in `data/processed/context_location_error_diagnostics/provenance.json`; the #147 provenance records likelihood, evidence, and inference settings. Historical transfer inputs are retrospective reconstructions, not certified archived preseason snapshots.
