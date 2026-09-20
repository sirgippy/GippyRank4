# Preseason Context 1.3 production model

Context 1.3 is the active production Context model. Context 1.2 remains a
supported historical artifact, including the genuine 2026 preseason artifact
and its early-season publication lineage.

The 2026 Context 1.3 starting point is explicitly a retrospective
reconstruction. Its transfer inputs were retrieved after the August 15, 2026
cutoff because the immutable preseason transfer-snapshot process did not yet
exist. It must not be described as the information state published before
kickoff.

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
Missing, duplicate, or unvalidated inputs fail closed. Late inputs are
accepted only by the explicit retrospective 2026 reconstruction path; future
production seasons still require an on-time immutable transfer snapshot.

## Validation and activation artifacts

The separate artifact namespace is
`data/processed/preseason/context_v1_3_candidate/`. It contains the frozen
historical parity validation for the model
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
