# History 1.1 annual provenance

`gippyrank.history_annual_v1_1` owns the annual source type, canonical builder,
reproducer, retained-legacy loader, research-fixture loader, fitted-model checks,
and cold-start row validation. Context 1.4 consumes that public History API;
History does not import the research candidate.

The retained 2026 History annual forecast is loaded as
`retained_legacy_history_artifact`. Its fitted model and prediction semantics
are pinned to the committed artifact under provenance schema 1. The published
predictions are unchanged. Future canonical and research source metadata uses
schema 2.

For a future season, run `uv run python scripts/build_history_annual_v1_1.py YEAR`
after the canonical processed historical rank distributions and target-season
team-feature artifact have been prepared. The builder reads those fixed inputs,
rejects target-season outcomes, derives the eligible historical FBS rows and
target FBS population, fits History 1.1 and its historical cold-start methods,
copies the exact input bytes into the season's `source_inputs/` directory, and
writes `predictions.csv`, `fitted_instance.json`, and
`fitted_model_source.json` together in the fixed annual output directory.

The annual source binds the input artifact hashes and derived-row identity,
History build semantics, fitted model and instance, prediction bytes and rows,
team IDs, population, prior methods, and PMFs. Loading a future annual source
reconstructs the model and every prediction from the retained canonical input
snapshots before accepting the sidecar. This remains possible when newer
seasons are added to the live input tables. A matching sidecar alone has no
authority. Loading fails closed if the retained inputs are unavailable.
Retain the `source_inputs/` files with the annual output when committing a
future build; the repository's general `data/processed/` ignore rule otherwise
omits newly generated files.
Synthetic files can be
inspected as `research_history_fixture`, which cannot supply a canonical
Context 1.3 cold-start fallback.

## Independent retained reference

The historical rank-distribution table needed to refit the published 2026
History model is not retained in this checkout. The 2026 prediction and fitted
instance artifacts therefore stay pinned legacy artifacts, and the future
builder cannot independently reproduce their fit from committed inputs.

The older History 1.1 backtest retains 534 FBS predictions for 2022–2025:
528 same-subdivision lag-1 rows and six FCS-to-FBS transition rows. There are
no generic no-prior FBS predictions in that panel. A separately retained
outcome-free Context input fixture supplies the exact lag-1 distributions for
the 528 ordinary teams, and the retained History model report supplies their
pre-builder fitted model. The golden test passes those inputs through the new
annual prediction arm and obtains identical serialized PMFs for all 528 rows
(maximum absolute difference zero). Team IDs, names, prior methods, population,
and the retained fitted-model metadata identity are checked. The six transition
rows are pinned and cross-checked against the earlier V1.1 prediction artifact;
their original FCS lag distributions are unavailable for an independent
recalculation. A separate future-season fixture compares the new annual builder
with the older annual procedure across all three arms, including a generic
no-prior team.

The History semantic specification records rank-coordinate transforms, PMF
integration, the main and transition fits, generic equal-team moments, and
prediction rounding. Named values are shared with the model implementation;
formula tests pin the estimators and integration behavior.

Context 1.4 remains a research candidate and Context 1.3 remains active.
Provenance metadata can advance independently of the published prediction
schema and prediction bytes.
