# Context incremental information after games

## Scope and audit

- Seasons: 2022–2025; checkpoints: 1, 2, 3, 4, 5, 6, 7.
- FBS team-seasons per checkpoint: 534 (528 fitted Context rows; 6 native cold starts).
- Inference evidence rows: 28; target-free inference panel rows: 7,350 including FCS fallback nodes.
- Reconstructed fitted Context priors: 528; maximum absolute PMF error: 0.
- Retained History 1.1 score parity: 3,738 rows; maximum errors by metric: `{"crps": 5.273559366969494e-16, "expected_rank": 7.105427357601002e-14, "expected_rank_error": 7.105427357601002e-14, "interval_80_target_mass": 5.551115123125783e-16, "interval_80_width": 0.0, "nll": 2.6645352591003757e-15}` (tolerance 1e-08).
- Retained crossover CC is Context 1.3, so it is not a Context 1.4 posterior-score target. Its maximum prior-PMF difference from reconstructed Context 1.4 is 0.103 across 534 team-seasons.
- Pilot replay: passed before extending to checkpoints 3–6; pilot NLL tolerance 1e-05.
- Targets are held in `scoring_targets.jsonl.gz` and are not present in the inference panel. Probe outer folds evaluate 2023, 2024, and 2025 after training only on earlier seasons.
- KL uses exact `KL(History posterior || History prior)` on positive posterior support; positive posterior mass outside prior support fails. No smoothing is applied to inference PMFs or crossed messages.

## Checkpoint results

The table reports equal-team-season mean NLL across the three held-out years for each checkpoint separately. Repeated checkpoints are not pooled as independent observations.

| Checkpoint | History | Context | History control | Existing-signal Δ | Raw-feature Δ | Existing signal by year | Raw features by year |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 4.54758 | 4.51459 | 4.55202 | -0.02716 | -0.00931 | positive descriptive signal (-0.02821, -0.03086, -0.02249) | mixed/inconclusive across evaluation seasons (+0.00610, -0.02180, -0.01208) |
| 2 | 4.36388 | 4.38336 | 4.36007 | -0.00778 | +0.00368 | positive descriptive signal (-0.01218, -0.00631, -0.00493) | mixed/inconclusive across evaluation seasons (-0.00181, +0.00583, +0.00693) |
| 3 | 4.22166 | 4.25545 | 4.21199 | -0.00349 | +0.00739 | positive descriptive signal (-0.00461, -0.00058, -0.00526) | mixed/inconclusive across evaluation seasons (+0.02054, -0.00175, +0.00355) |
| 4 | 4.08976 | 4.12410 | 4.07485 | -0.00218 | +0.00104 | positive descriptive signal (-0.00461, -0.00179, -0.00017) | mixed/inconclusive across evaluation seasons (-0.00261, +0.00339, +0.00228) |
| 5 | 4.01418 | 4.06351 | 3.99434 | +0.00021 | +0.00423 | mixed/inconclusive across evaluation seasons (-0.00007, +0.00059, +0.00010) | mixed/inconclusive across evaluation seasons (-0.00406, +0.00907, +0.00756) |
| 6 | 3.93359 | 3.98682 | 3.90997 | +0.00035 | +0.00883 | negative descriptive signal (+0.00007, +0.00074, +0.00023) | mixed/inconclusive across evaluation seasons (-0.00770, +0.02667, +0.00742) |
| 7 | 3.83089 | 3.88793 | 3.81017 | +0.00048 | +0.00317 | mixed/inconclusive across evaluation seasons (-0.00005, +0.00111, +0.00037) | mixed/inconclusive across evaluation seasons (-0.00800, +0.01630, +0.00114) |

A negative delta means the Context addition lowered NLL relative to the same History calibration control. These are descriptive outer-fold results over development seasons, not an untouched confirmation set or a production recommendation. Mixed year signs are reported as mixed/inconclusive. Each checkpoint remains a separate comparison.

## Pilot crossed-message check

At checkpoint 7, the crossed-arm NLLs (mean over 2023–2025) were: CC 3.88793, CH 3.85329, HC 3.85739, HH 3.83089. The crosses freeze each model's incoming message product and therefore are a sensitivity analysis, not new converged joint posteriors or causal opponent attributions.

## Output files

- `inference_panel.jsonl.gz`: complete prior/posterior and crossed PMFs, prediction-time features, games played, and History information gain; contains no targets.
- `scoring_targets.jsonl.gz`: frozen final-rank target PMFs, isolated from inference.
- `probe_predictions.jsonl.gz`, `team_scores.csv`, `aggregate_scores.csv`, and `comparison_summary.csv`: per-team forecasts/scores and checkpoint/season comparisons, with fitted and cold-start strata.
- `probe_fits.json` and `probe_coefficients.csv`: nested regularization choices, preprocessing, optimizer details, coefficients, and training/evaluation fold sizes.
- `evidence.json` and `manifest.json`: cutoff/game evidence, source and output hashes, reconstruction/parity checks, and environment versions.

Reproduce with `uv run python scripts/build_context_incremental_information.py --input-root <repair-input-root> --targets <team_season_rank_distributions.csv> --raw-games <cfbd-games-cache>`.
