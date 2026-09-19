# Preseason Context 1.3 candidate

Context 1.3 is implemented as a validated, future-activation candidate. The
active production Context version remains 1.2, including the existing 2026
publication lineage.

## Frozen model contract

The model is a normal `DirectRankModel` with penalty `0.25`, the existing
quadrature, and the existing retry with `maxiter=2000` after an iteration-limit
failure. History features affect location and scale. Context features affect
location only.

The model feature order is:

```text
lag2_z_mean
lag3_z_mean
long_run_z_mean
coach_tenure_seasons
recruiting_class_rank
recruiting_class_points
recruiting_points_2y_mean
recruiting_points_3y_mean
recruiting_points_4y_mean
recruiting_points_trend
talent_composite
returning_pct_ppa
transfer_in_prior_usage_sum
transfer_in_prior_defensive_impact_db_sum
transfer_in_prior_defensive_impact_db_available
```

The decomposed returning passing, receiving, and rushing features are not part
of this candidate. Transfer counts, outgoing/net measures, ratings/stars,
position-specific non-DB measures, and aggregate defensive measures are also
excluded.

## Transfer boundary and provenance

The fit consumes only the three model-facing columns from the issue #114
`merge_preseason_transfer_features` boundary. It does not parse raw transfer,
roster, or game-player payloads. Historical 2021–2025 values are a
retrospective research reconstruction and are not archived August 15 snapshots.

For a future season, the annual inference boundary requires a validated
immutable snapshot manifest, exactly one canonical portal/usage/stats input,
canonical roster and game-player coverage, complete FBS team coverage, matching
snapshot identifiers and hashes in provenance, and an on-time August 15 cutoff.
Missing, duplicate, late, or unvalidated inputs fail closed. Context 1.3 is
explicitly ineligible for the existing 2026 artifact.

## Candidate artifacts

The separate artifact namespace is
`data/processed/preseason/context_v1_3_candidate/`. It contains the model
specification, fitted model metadata, predictions, coverage, provenance,
evaluation, annual and rolling metrics, coefficients, parity report, and a
human-readable validation report. The active `context/` artifacts are not
rewritten.

To reproduce the historical panel, provide the disposable retrospective oracle
from the transfer research work and run:

```bash
uv run python scripts/materialize_preseason_context_prior_v1_3_research_features.py \
  --transfer-root /path/to/retrospective-transfer-oracle \
  --output data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv
uv run python scripts/build_preseason_context_prior_v1_3.py
```

The builder verifies D5 and P3 aggregate metrics, P3 coefficients, target
population, H fallback population, and rolling-origin parity against the
frozen research artifacts before writing the candidate report.
