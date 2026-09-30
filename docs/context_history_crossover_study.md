# Context 1.3 / History 1.1 crossover diagnostic

## 1. Executive finding

The tested Context uncertainty bundle explains little of the 2023–2025 crossover. Context starts with better preseason probability scores and a sharper rank distribution. Swapping History's location-mixture offsets and residual scale onto Context's location center recovers only 2.0–3.4% of the December Context-versus-History NLL gap in each post-crossover season. Conversely, putting Context's uncertainty bundle around History's center moves History's December NLL only 1.0–5.9% of that gap in the unfavorable direction. The hybrid posteriors track the source of their location center much more strongly than the source of their uncertainty bundle. This motivates a separate diagnostic of Context's location signals and error structure; this study does not isolate location/error structure as the causal mechanism.

The confidence/error diagnostic adds a narrower association: Context's preseason expected-rank errors coincide with large downstream losses. Across 2023–2025, the 40 team-seasons in the per-season top quartile for both Context's relative sharpness and its expected-rank error gap account for 48.8% of the total December C−H NLL gap. Yet Context does move those teams' expected-rank estimates more toward their target than History does. This pattern motivates investigating Context's location signals and error structure, but it does not establish them as the cause of the crossover or reduce the result to a simple failure to let the evidence move Context enough.

This is an explanatory analysis of the already-observed 2022–2025 behavior, not independent confirmation or a new model selection result.

## 2. Reproduction of #140

The runner reproduces the #140 rolling-origin Context 1.3 and History 1.1 priors, checkpoint dates, target rank distributions, cutoff-safe game evidence, Historical Likelihood V1, and production inference settings. It records the same 934 near-cutoff exclusions in the cutoff audit. Across the four years and seven checkpoints per year, **700 baseline checks pass** at tolerance `1e-8`; the maximum absolute difference is `8.74e-10`. Checks cover team-level prior/posterior scores and the posterior expected-rank Spearman correlation.

There are 534 FBS team-seasons total and 528 with fitted Context/History component decompositions. The remaining six retain their native fallback priors in every arm and in the full inference networks; matched-primary summaries exclude them. All **112 primary posterior runs** (28 checkpoints × four arms) converged with `max_iterations=500`, `tolerance=1e-9`, and `damping=0.35`. Each checkpoint has matching game-row hashes, FCS fallback IDs, likelihood hash, and evaluation population across arms.

By December, Context-versus-History expected-rank Spearman correlations are `0.9964` (2023), `0.9965` (2024), and `0.9984` (2025), even as their matched-primary mean NLL gaps are `+0.0794`, `+0.0703`, and `+0.0674`. The ordering is nearly the same while the probability assigned across rank positions remains meaningfully different.

## 3. Preseason uncertainty anatomy

Matched fitted team-seasons are equally weighted. In the table, “target mass” is the final target PMF mass inside the forecast's central 80% interval.

| Preseason metric | Context 1.3 | History 1.1 |
| --- | ---: | ---: |
| NLL | 4.5079 | 4.5451 |
| CRPS | 0.0749 | 0.0807 |
| Expected-rank absolute error | 19.53 | 21.29 |
| Central 80% width, ranks | 70.60 | 76.82 |
| Target mass in central 80% interval | 0.8340 | 0.8383 |
| Entropy | 4.4730 | 4.5663 |
| Rank standard deviation | 27.08 | 29.09 |

Context is better on preseason NLL, CRPS, and expected-rank error, and has lower entropy and rank spread. Its central interval is narrower, with slightly less target mass.

The fitted latent decomposition is also narrower for Context. Across matched teams, mean location-mixture SD is `0.2113` for Context and `0.2245` for History; mean conditional residual scale is `1.2202` and `1.2754`; mean total latent variance is `1.5397` and `1.6848`. The average Context-minus-History total variance is `−0.1451`. Context's and History's mean location centers are `0.0468` and `−0.0157` in the model's latent coordinates; the mean absolute team-level center difference is `0.358`.

The stored fitted metadata confirms that Context 1.3's rich Context features enter its location model, while its scale model retains only the History features (`lag2_z_mean`, `lag3_z_mean`, and `long_run_z_mean`). Those features do not directly enter Context scale. Any C/H scale difference reflects the jointly fitted models and their differing location specifications.

## 4. Confidence versus error

Quartile cut points are computed separately within each target season, before posterior results are grouped. Positive relative sharpness means Context's prior 80% interval is narrower than History's. Positive expected-rank error gap means Context's prior expected rank is farther from the target expected rank. These are descriptive strata, not thresholds selected after seeing posterior scores.

The quartile analysis uses all FBS target team-seasons, including the six native-fallback cases; component-level latent decompositions and matched-primary summaries use fitted team-seasons only.

At the final checkpoint, the highest expected-rank-error quartile has mean C−H posterior NLL gaps of `+0.124` in 2022, `+0.259` in 2023, `+0.209` in 2024, and `+0.202` in 2025. Positive gaps favor History. In the lowest error quartile, the corresponding gaps are `−0.117`, `−0.047`, approximately `−0.001`, and `+0.022`. The pattern is strongest in the years with the crossover.

In the 2023–2025 intersection of the highest relative-sharpness and highest error quartiles, 40 of 403 team-seasons (9.9%) account for `14.32` of `29.33` total December C−H NLL-gap units (48.8%). The analogous intersection in 2022 has seven teams and a mean gap of `+0.191`, but Context remains slightly better overall that season; the 2022 low-error groups offset the expensive misses. This negative-control contrast supports concentration in a subset of teams, while also showing that the same descriptive stratum does not mechanically imply an aggregate crossover.

## 5. Hybrid counterfactuals

`CC` uses Context center, Context location offsets, and Context scale; `HH` uses all History components. `C_center_H_uncertainty` keeps Context's center and swaps in History's offsets and scale. `H_center_C_uncertainty` keeps History's center and swaps in Context's offsets and scale. These are synthetic priors, with no refitting.

The pooled preseason NLLs are `4.5079` (CC), `4.5451` (HH), `4.5136` (C center / H uncertainty), and `4.5427` (H center / C uncertainty). December matched-primary posterior NLL is:

| Target season | CC | HH | C center / H uncertainty | H center / C uncertainty |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 3.7950 | 3.8010 | 3.7927 | 3.8083 |
| 2023 | 3.9139 | 3.8345 | 3.9119 | 3.8367 |
| 2024 | 3.8850 | 3.8147 | 3.8826 | 3.8189 |
| 2025 | 3.9153 | 3.8480 | 3.9140 | 3.8487 |

In 2023–2025, changing Context uncertainty while keeping its center recovers only `2.0–3.4%` of the ordinary C−H December NLL gap. Changing History uncertainty while keeping its center gives back `1.0–5.9%` of History's ordinary advantage. The hybrid posteriors track the source of the location center much more strongly than the source of the uncertainty bundle. This motivates a location-signal diagnostic but does not identify location/error structure as the causal mechanism. September-through-December tables for all four arms are in the machine-readable summary.

The predeclared secondary decomposition trigger required the C-center/History-uncertainty arm to recover at least 25% of the ordinary post-September C/H gap, averaged over eligible checkpoints, in two or more of 2023–2025. It qualified in zero seasons, so residual-scale-versus-mixture-dispersion hybrids were not run.

## 6. Update responsiveness and reversals

History gains more NLL from preseason to December in every post-crossover season, even though Context begins with the better prior score:

| Season | Context NLL improvement | History NLL improvement | Context mean absolute rank shift | History mean absolute rank shift |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 0.554 | 0.654 | 15.32 | 17.42 |
| 2024 | 0.678 | 0.787 | 17.85 | 19.90 |
| 2025 | 0.633 | 0.718 | 17.39 | 19.47 |

By December, 33 teams in 2023, 46 in 2024, and 36 in 2025 move from Context having the better prior NLL to History having the better posterior NLL. The `reversal_categories.csv` file reports every category by season and checkpoint, with counts and contributions to the total gap.

The high-sharpness/high-error teams are not simply resistant to correction. Their Context expected-rank error improvement from prior to December averages `11.65`, `22.94`, and `16.84` ranks in 2023, 2024, and 2025; History's corresponding improvements are `1.25`, `14.78`, and `10.63`. Context moves farther on this diagnostic yet still has much worse NLL in those groups. This is consistent with a role for Context location/error structure and motivates investigating it alongside game-network and target-PMF interactions; it does not establish that mechanism as causal or show insufficient movement alone is the explanation.

## 8. Limits

- The 2022–2025 panel motivated this study and is not an untouched validation set.
- Historical Context transfer inputs are retrospective reconstructions, as documented by #140.
- Four target seasons are available, and 2022 is a useful but limited negative control.
- Belief propagation couples teams through the schedule network. The hybrid comparisons are controlled diagnostics, not a clean causal decomposition of an isolated team-level effect.
- Synthetic hybrids are counterfactuals for explanation, not proposed production priors.
- All reported differences are descriptive effect sizes. No significance tests, confidence intervals, or new holdout claims are used.

## 9. Recommendation for next research step

Open a separate, leakage-safe **Context location-signal error-structure diagnostic**. Examine which Context location signals and combinations mark the concentrated preseason error groups, with rolling-origin inputs and predeclared summaries. Do not tune features or change Context 1.3 in this issue. The current uncertainty swap does not justify a separate scale-calibration study.

## Reproduction and artifacts

Run from the repository root with the pinned `uv` environment and the local historical source corpus:

```bash
uv run python scripts/study_context_history_crossover.py \
  --targets /path/to/data/processed/modeling/team_season_rank_distributions.csv \
  --raw-games /path/to/data/raw/cfbd/games
```

The run requires the committed #140 study artifacts and the raw coach-tenure cache alongside the target corpus. It writes to `data/processed/context_history_crossover/` by default. In this worktree the target and raw-game corpus were read from the neighboring `/home/gippy/src/GippyRank4` checkout; the corpus was not copied or modified.

The committed artifacts include:

- `team_season_prior_decomposition.csv`: fitted conditional locations, centers, offsets, mixture dispersion, residual scale, total variance, rank-space metrics, and target hashes.
- `hybrid_prior_results.csv`: per-team PMFs and hashes, metrics, and center/offset/scale provenance for each arm.
- `hybrid_posterior_team_results.csv` and `hybrid_posterior_summary.csv`: team/checkpoint results and aggregates by season, checkpoint, arm, and period.
- `confidence_error_strata.csv`, `confidence_error_correlations.csv`, and `confidence_error_quartile_cutpoints.json`: predeclared within-season strata and descriptive relationships.
- `reversal_categories.csv`, `arm_evidence_and_convergence.csv`, `baseline_reproduction_checks.csv`, and `cutoff_audit.csv`: reversal contributions, evidence/inference audits, #140 reproduction checks, and cutoff exclusions.
- `provenance.json`: model and input hashes, rolling training origins, target seasons, evidence contract, inference settings, and the secondary-analysis trigger result.
