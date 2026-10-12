# Matched History and Context pre-bowl pipelines (issue 200)

The four priors use the same corrected final-rank predictor inputs within each policy comparison. The pre-bowl policy substitutes only FBS response targets; FCS response and postseason policies remain on the corrected final corpus.

All 2022-2025 results are exploratory development because prior research has informed these seasons. This build fits models and records parity; it does not compare held-out scores or select a model.

## Data gate

- Corrected corpus gate: `complete` after #198 collection completeness and #204 hash/support validation.
- The pre-repair #198 manifest remains preserved with its historical `downstream_ready: false` value; this build verifies the corrected #204 target and pair artifacts separately.
- Corrected target corpora and source game files are bound by input hashes. Response rows and predictor rows are keyed and hashed; the full predicted rank PMFs are serialized in `pipeline_bundle.json`.

## Likelihood training

| Policy | Games | Rank-pair pseudo-observations | Scale | df |
|---|---:|---:|---:|---:|
| existing | 26164 | 1887398 | 13.230607 | 15 |
| pre_bowl | 25504 | 1895100 | 13.188732 | 15 |

FBS postseason rows are excluded together from the pre-bowl likelihood. Regular-season games, conference championships, and Army-Navy are retained. FCS postseason games keep the existing inclusion policy. `likelihood_game_manifest.csv` records each decision; `rank_pair_composition.csv` records same-system versus fallback composition and the FBS snapshot date for cross-subdivision rows.

Cross-subdivision composition rows recorded: 38.

## Baseline parity

The frozen #192/#196 final-target references are retained by hash. The corrected final-control likelihood differs from the frozen likelihood by max absolute coefficient 6.308832; see `baseline_parity.json` for the History and Context prior PMF deltas. These differences are reported separately from the target/game-policy intervention.

## Reproduction

Run `uv run python scripts/build_matched_history_context_issue_200.py --help` and supply the corrected #204 corpus root, the retained source-feature root, the cached game root, and the #192 reference root. The command writes the same versioned bundle, fit records, game/feature/target manifests, and report.
