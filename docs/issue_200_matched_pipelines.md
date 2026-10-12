# Issue 200: matched History and Context pre-bowl pipelines

## Scope

This research build pairs History 1.1 and repaired Context 1.4 with two FBS
response policies: corrected final targets (`existing`) and corrected pre-bowl
targets (`pre_bowl`). For each target season, both models train through the
previous season. The pre-bowl fit substitutes only FBS response distributions;
rank-lag and preseason feature inputs continue to use corrected final-rank
history. FCS target rows and FCS-only postseason evidence stay on the corrected
final policy.

The four reported prior arms are History/existing, Context/existing,
History/pre-bowl, and Context/pre-bowl. Each arm is paired with the likelihood
refit for its target policy. The existing likelihood keeps all eligible
postseason games. The pre-bowl likelihood excludes postseason games involving
an FBS team, retains Army-Navy, and leaves FCS-only postseason games unchanged.
The season type and inclusion decision for every joined game are recorded.

The fit window is 2003-2021. Results for 2022-2025 are exploratory development
outputs because earlier research used those seasons. This build records PMFs
and parity diagnostics; it does not score held-out outcomes, select a model, or
change production defaults.

## Data gate and lineage

The build requires the complete #198 collection and the corrected #204
observations, targets, and constituent-pair files. Before fitting, it checks
the #204 file sizes and SHA-256 values against its manifest, confirms the
corrected support audit passed with zero anomalies, and confirms the support
report contains only its header. The earlier #198 `downstream_ready: false`
value is preserved as a pre-repair record; it does not override the separately
validated #204 corrected corpus.

History inputs retain the issue #192 rank-history construction. Context uses
the same History features and lag distributions plus repaired Context 1.4
features, including observed incoming defensive-back impact and its coverage.
Rows without a repaired transfer-source record are retained with unknown
repaired transfer features. Missing source coverage is not changed to a zero
transfer impact. Per-season Context coverage counts and row identities are
included in `preseason_feature_manifest.json` and `fitted_models.json`.

The frozen #192 / PR #196 prior panel and likelihood are read only and hashed
in `baseline_parity.json`. The report separates those original-reference
differences, which include corrected-corpus repairs, from the pre-bowl
intervention.

## Rebuild

First prepare the validated corrected inputs described in
[`issue_198_prebowl_collection.md`](issue_198_prebowl_collection.md) and the
merged #204 population audit. By default the builder reads those versioned
artifacts under the repository's `data/` tree:

```sh
uv run python scripts/build_matched_history_context_issue_200.py
```

Use `--corpus-root`, `--source-root`, `--game-root`, and `--reference-root`
when the validated artifacts are retained in separate checkouts. The builder
fails before fitting when a required file or gate check is missing. Its
`manifest.json` records every input and output size/hash.

The versioned output is written to
`data/research/matched_history_context_issue_200/` and includes:

- `pipeline_bundle.json`: all four full prior PMF arms, both likelihoods, FCS
  fallback populations, and deterministic inference settings.
- `fitted_models.json` and `training_row_manifest.csv`: rolling cutoffs,
  optimizer details, cold-start fits, response-row hashes, and predictor-row
  hashes.
- `target_snapshot_manifest.csv`, `preseason_feature_manifest.json`,
  `likelihood_game_manifest.csv`, and `rank_pair_composition.csv`: snapshot
  dates, shared feature inputs, postseason decisions, and same-system versus
  fallback pair coverage.
- `baseline_parity.json`, `data_gate.json`, `report.md`, and `manifest.json`.

## Posterior replay

`gippyrank.research.matched_pipeline_replay` loads the serialized bundle and
runs the existing posterior engine with a selected prior family and its
matching policy likelihood. Supply the evidence games after applying the
selected policy's inclusion decisions:

```python
from pathlib import Path

from gippyrank.research.matched_pipeline_replay import (
    read_pipeline_bundle,
    replay_pipeline_bundle,
)

bundle = read_pipeline_bundle(
    Path("data/research/matched_history_context_issue_200/pipeline_bundle.json")
)
result = replay_pipeline_bundle(
    bundle,
    season=2025,
    policy="pre_bowl",
    prior_family="Context",
    games=pre_bowl_games,
)
```

FBS priors are loaded from the bundle. FCS teams that occur in the supplied
evidence receive the existing uniform prior over the bundle's season-specific
FCS population. The replay returns the standard `PosteriorResult`; callers can
inspect convergence and normalized team PMFs using the existing engine API.
