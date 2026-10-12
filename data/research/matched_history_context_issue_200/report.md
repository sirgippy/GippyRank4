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

Both likelihood fits retain the fixed eight-iteration weighted-pseudo IRLS recipe. No convergence criterion is evaluated; fit metadata records convergence as not assessed.

FBS postseason rows are excluded together from the pre-bowl likelihood. Regular-season games, conference championships, and Army-Navy are retained. FCS postseason games keep the existing inclusion policy. `likelihood_game_manifest.csv` records each decision; `rank_pair_composition.csv` records same-system versus fallback composition and the FBS snapshot date for cross-subdivision rows.

Cross-subdivision composition rows recorded: 38.

## Baseline parity

The frozen #192/#196 final-target references are retained by hash. The corrected final-control likelihood differs from the frozen likelihood by max absolute coefficient 6.308832; see `baseline_parity.json` for the History and Context prior PMF deltas. The prior comparison includes #204 corpus repairs and the restored coaching-tenure source; both fitted issue 200 policies use the same source inventory, so the comparisons below isolate the target/game-policy intervention.

## Pre-bowl intervention

The following comparisons are between the two newly fitted policies on the corrected #204 corpus. They describe the intervention only; data-repair effects remain in the separate frozen-reference section above. A materially changed prior is defined as total variation greater than 0.01.

### Prior PMFs by target season and model family

| Season | Family | Teams | Mean TV | Max TV | Teams above TV 0.01 |
|---:|---|---:|---:|---:|---:|
| 2022 | History | 131 | 0.003685 | 0.012300 | 1 |
| 2022 | Context | 131 | 0.018449 | 0.101140 | 82 |
| 2023 | History | 133 | 0.003132 | 0.012466 | 2 |
| 2023 | Context | 133 | 0.010953 | 0.044371 | 55 |
| 2024 | History | 134 | 0.003061 | 0.010429 | 1 |
| 2024 | Context | 134 | 0.008022 | 0.031607 | 43 |
| 2025 | History | 136 | 0.003784 | 0.019124 | 2 |
| 2025 | Context | 136 | 0.007820 | 0.027595 | 38 |

### FBS target distributions by season

| Season | Teams | Mean TV | Max TV | Changed targets | Different population support |
|---:|---:|---:|---:|---:|---:|
| 2022 | 131 | 0.377852 | 0.655028 | 131 | 0 |
| 2023 | 133 | 0.410906 | 0.811712 | 133 | 0 |
| 2024 | 134 | 0.414915 | 0.770864 | 134 | 0 |
| 2025 | 136 | 0.388139 | 0.754761 | 136 | 0 |

### Prior fitted coefficients

| Season | Family | Component | Coefficients | Mean absolute difference | Max absolute difference |
|---:|---|---|---:|---:|---:|
| 2022 | History | main_prior | 15 | 0.00471035 | 0.01622 |
| 2022 | Context | main_prior | 63 | 0.0123539 | 0.121497 |
| 2022 | History | transition_model | 3 | 0.016609 | 0.0279561 |
| 2023 | History | main_prior | 15 | 0.00412973 | 0.0152519 |
| 2023 | Context | main_prior | 63 | 0.0113223 | 0.0952026 |
| 2023 | History | transition_model | 3 | 0.0168362 | 0.0267255 |
| 2024 | History | main_prior | 15 | 0.00455208 | 0.013069 |
| 2024 | Context | main_prior | 63 | 0.0128564 | 0.103815 |
| 2024 | History | transition_model | 3 | 0.0142544 | 0.019351 |
| 2025 | History | main_prior | 15 | 0.00508158 | 0.0133863 |
| 2025 | Context | main_prior | 63 | 0.0123051 | 0.103099 |
| 2025 | History | transition_model | 3 | 0.0127868 | 0.0226 |

### Fitted likelihood differences

The fitted likelihood coefficient vectors have mean absolute difference 3.49827 and maximum absolute difference 27.3137. Scale changes by 0.041875 in absolute terms. Included game counts are 26164 (existing) and 25504 (pre-bowl); rank-pair pseudo-observation counts are 1,887,398 and 1,895,100, respectively. Pairing and rank-pair composition deltas are in `policy_intervention.json`.

## Reproduction

Run `uv run python scripts/build_matched_history_context_issue_200.py --help` and supply the corrected #204 corpus root, the retained source-feature root (including its non-empty `data/raw/cfbd/preseason/coach_tenures/` directory), the cached game root, and the #192 reference root. Coaching-tenure JSON inputs are validated and recorded individually by path, size, and SHA-256. The command writes the same versioned bundle, fit records, game/feature/target manifests, and report.
