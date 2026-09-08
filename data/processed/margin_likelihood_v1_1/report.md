# Historical Likelihood V1.1 margin-evidence investigation

## Research question

Can GippyRank extract better-calibrated evidence from the observed final score margin without adding another football statistic? The production Historical Likelihood V1 remains frozen.

## Frozen design

- Training: 2003–2017; development: 2018–2021; final historical evaluation: 2022–2025.
- Student-t df is fixed at 15 for every candidate.
- All candidates retain the 34-column V1 rank/site/pairing mean surface.
- Candidate A uses a positive pairing-intercept plus `log1p(abs(V1 expected margin))` scale.
- Candidate B adds standardized `log1p(total points)` to scale only; normalization is training-derived.
- Candidate C applies `k * asinh(margin / k)` for exactly one development-selected k in {14, 21, 28, 42}; original-scale NLL includes its log Jacobian.

## Stage 0 residual diagnostics

The diagnostics use 26137 training/development games (20743 training and 5394 development). Quantile edges for expected mismatch, total points, and observed margin magnitude were computed from training games only. Residual means are observed oriented margin minus the equal-rank-pair V1 expected margin. Full values are in `residual_diagnostics.csv`.

## Development selection

| Candidate | NLL | Δ NLL vs V1 | Margin MAE | 50% | 80% | 95% | Gate vs predecessor |
|:--|--:|--:|--:|--:|--:|--:|:--|
| V1 | 4.065 | 0.000 | 11.146 | 0.506 | 0.810 | 0.960 | False |
| A | 4.068 | 0.003 | 11.176 | 0.502 | 0.809 | 0.958 | False |
| B | 4.065 | 0.000 | 11.176 | 0.501 | 0.810 | 0.958 | False |
| C14 | 4.110 | 0.045 | 11.276 | 0.476 | 0.808 | 0.968 | False |
| C21 | 4.088 | 0.023 | 11.225 | 0.481 | 0.807 | 0.966 | False |
| C28 | 4.078 | 0.013 | 11.202 | 0.484 | 0.806 | 0.964 | False |
| C42 | 4.070 | 0.005 | 11.185 | 0.492 | 0.805 | 0.961 | False |

The development-only selection returned `none`. Selection metadata and all frozen parameters are in `model_spec.json`; no 2022–2025 row was used to choose the candidate or C k. The decision trace is in `summary.json`.

## Rolling direct-likelihood robustness

No candidate passed the development gate, so the expensive rolling/posterior stages were not triggered.

## Posterior stage and conclusion

Stage 2 posterior validation triggered: `False`.

Because 2022–2025 has already been examined by prior GippyRank research, it is leakage-safe historical evaluation but not independent confirmation for this hypothesis.

If no candidate is selected, the recommendation is C — retain Historical Likelihood V1. A selected candidate would be a frozen research candidate only, not an immediate production promotion.

The artifact directory is research-only. Production V1 artifacts and publication files are hash-checked before and after the build.
