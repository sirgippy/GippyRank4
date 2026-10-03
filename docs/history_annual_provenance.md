# History 1.1 annual provenance

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

Context 1.4 remains a research candidate and Context 1.3 remains active.
Provenance metadata can advance independently of the published prediction
schema and prediction bytes.
