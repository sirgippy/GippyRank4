# Issue 194 Part B: exact synthetic mechanism test

Part A is complete in the merged [Part A report](../issue_194_part_a/report.md). Its historical co-movement classification was **descriptively unsupported** under the preregistered four-season rule. Part B is a separate controlled synthetic check; it does not revise that historical finding.

## Frozen design

The fixed experiment uses four teams, five ordinal rank states, a three-edge tree and a four-edge cycle, 100 seeds per case/graph panel, and the pinned Historical Likelihood V1. The exact enumerator visits 625 assignments per trial. All inference arms share the same generated ranks, games, home/neutral assignments, and continuous V1 margins within each trial. The full protocol and seed ranges are in [the frozen protocol](../../../docs/issue_194_part_b_protocol.md).

The shared prior's two rank-distribution scenarios each have prior weight 0.5. Their component posterior weights are computed as prior weight times scenario evidence, then normalized. They are not averaged equally after observing games; per-draw evidence weights are retained in `scenario_weights.csv`.

## Posterior rank scores and calibration

Rank NLL and multicategory Brier score are lower-is-better. Expected-rank error is forecast expected rank minus simulated rank. Calibration MAE averages the absolute difference between forecast probability and observed frequency over each team/rank cell.

| Case | Graph | Inference arm | Draws | Rank NLL | Brier | Signed expected-rank error | Absolute expected-rank error | Calibration MAE | BP converged | Mean BP iterations |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent_null | cycle_4 | bp_product_marginals | 100 | 1.211823 | 0.648535 | 0.041587 | 0.694545 | 0.031693 | 1.000000 | 41.270000 |
| independent_null | cycle_4 | exact_product_marginals | 100 | 1.212988 | 0.649003 | 0.042752 | 0.694625 | 0.031976 |  |  |
| independent_null | cycle_4 | exact_true_joint | 100 | 1.212988 | 0.649003 | 0.042752 | 0.694625 | 0.031976 |  |  |
| independent_null | path_tree | bp_product_marginals | 100 | 1.357577 | 0.690221 | 0.012856 | 0.864907 | 0.022405 | 1.000000 | 30.930000 |
| independent_null | path_tree | exact_product_marginals | 100 | 1.357577 | 0.690221 | 0.012856 | 0.864907 | 0.022405 |  |  |
| independent_null | path_tree | exact_true_joint | 100 | 1.357577 | 0.690221 | 0.012856 | 0.864907 | 0.022405 |  |  |
| shared_context | cycle_4 | bp_product_marginals | 100 | 1.281997 | 0.676031 | -0.012007 | 0.817948 | 0.025390 | 1.000000 | 41.260000 |
| shared_context | cycle_4 | exact_product_marginals | 100 | 1.281437 | 0.675741 | -0.010622 | 0.817954 | 0.025310 |  |  |
| shared_context | cycle_4 | exact_true_joint | 100 | 1.281624 | 0.675744 | -0.010036 | 0.819613 | 0.025301 |  |  |
| shared_context | path_tree | bp_product_marginals | 100 | 1.257268 | 0.661758 | 0.131997 | 0.807830 | 0.033685 | 1.000000 | 30.930000 |
| shared_context | path_tree | exact_product_marginals | 100 | 1.257268 | 0.661758 | 0.131997 | 0.807830 | 0.033685 |  |  |
| shared_context | path_tree | exact_true_joint | 100 | 1.258840 | 0.662197 | 0.132139 | 0.809935 | 0.033748 |  |  |

Full team/rank calibration cells are in `rank_calibration.csv`; per-draw paired scores are in `draw_scores.csv`.

## Inference-arm differences

Total variation is averaged over teams within each draw, then over the 100 draws. The first pair isolates replacing the true joint prior with independent marginals. The second isolates BP approximation under the product prior. The third is the combined difference from the true-joint exact reference.

| Case | Graph | Product exact vs true-joint exact TV | Absolute expected-rank difference | BP vs product exact TV | Absolute expected-rank difference | BP vs true-joint exact TV | Absolute expected-rank difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| independent_null | cycle_4 | 0.000000 | 0.000000 | 0.002543 | 0.001933 | 0.002543 | 0.001933 |
| independent_null | path_tree | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| shared_context | cycle_4 | 0.006918 | 0.009902 | 0.002737 | 0.002044 | 0.009452 | 0.011474 |
| shared_context | path_tree | 0.005841 | 0.009280 | 0.000000 | 0.000000 | 0.005841 | 0.009280 |

The preregistered shared-prior NLL contrast (product exact minus true-joint exact) was path_tree: -0.001571; cycle_4: -0.000187. Under the frozen two-graph rule, dependence-related rank-NLL degradation is **unsupported** for this synthetic setup.

The product-prior exact and BP posterior differences on the cycle were independent_null: 0.00254325 TV; shared_context: 0.00273739 TV. BP approximation as a nonzero cyclic marginal difference is **supported** at the frozen 1e-10 criterion. The cycle BP-minus-product-exact rank-NLL changes were independent_null: -0.001165; shared_context: +0.000561. Their score direction is unresolved; a PMF difference alone is not treated as a rank-score effect. The tree exactness check passed in the focused tests.

### Shared-scenario evidence weights

| Graph | Draws | Mean posterior weight A | Mean posterior weight B |
| --- | ---: | ---: | ---: |
| path_tree | 100 | 0.502335 | 0.497665 |
| cycle_4 | 100 | 0.499994 | 0.500006 |

## Decision and limits

* Historical co-movement consistent with shared Context errors: **descriptively unsupported** by the completed Part A report. Part B supplies no historical evidence.
* Dependence causing rank-NLL degradation in the controlled synthetic setting: **unsupported** under the preregistered direction rule. This says only whether the fixed shared-prior example loses rank score when its joint prior is replaced by independent marginals.
* BP approximation as a source of a small-graph posterior difference: **supported** under the fixed cycle criterion. Its rank-NLL effect is **unresolved** across the two generative cases because the cycle contrasts have opposite signs. The measured BP-to-exact gap is specific to these four-node examples.

This experiment measures posterior rank prediction under known simulated ranks. It does not evaluate future-game forecasting and cannot establish that Context harms real-game forecasts or explain the historical rank-NLL contrast. Better game forecasts can coexist with worse rank NLL. The 2022–2025 historical seasons have already informed this research, so the results remain exploratory development.

No full-season joint-prior experiment is recommended from this result alone. The unresolved real-data question is whether dependence in existing Context prior errors contributes to its historical rank-NLL disadvantage despite its better future-game CRPS; Part A did not find descriptive support for shared opponent-error co-movement, while this five-state synthetic control tests only possibility under its specified scenario magnitude. A full-season model would require a separately justified historical dependence signal; these four-team BP discrepancies are not extrapolated into a causal share.

## Reproduce

From the repository root, run `uv run python scripts/run_issue_194_part_b.py --input-root . --output-root data/research/issue_194_part_b`. The runner requires the pinned `data/processed/posterior/historical_likelihood_v1.json` input and verifies its retained V1 hash.

No production priors, likelihoods, inference settings, snapshots, website defaults, or raw downloads were changed.
