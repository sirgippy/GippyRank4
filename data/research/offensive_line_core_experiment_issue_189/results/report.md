# Issue 189: experience-selected OL core continuity

**Decision: Reject.**

The three additions are post-hoc: the 2022–2025 evaluation years were inspected before this experiment. A favorable result only supports refinement and future independent evaluation, not production promotion.

## Construction and population

Each target-season CFBD OL roster is retrospective. For each player, experience is the count of unique observed college roster seasons across all programs in T-4 through T-1. The five highest counts define the proxy core. Every valid five-player choice at a fifth-place tie is evaluated and the two features are averaged over those choices. Selection never uses shared history. Shared experience averages the ten pair counts for prior seasons on the current program's roster. The proxy is not an actual starting five.

The primary roster panel has 1691 team-seasons; 1686 have usable core features. The common model population has 1140 training and 528 held-out team-seasons. All arms have identical held-out keys. Exclusions from the previous common population: {"target_pool_unusable_or_too_small": 5}.

The target OL pool has p10/median/p90 sizes of [15.0, 19.0, 22.0]. Across the primary panel, 0 team-seasons have missing target OL IDs; unknown and ambiguous position row counts are 2353 and 0. Core feature statuses: {"observed": 1686, "prior_panel_unavailable": 5}.

Fifth-place ties produced multiple valid cores for 1230 observed team-seasons (73.0%); median and maximum valid realizations were 4 and 15504. The full count distribution and retained roster coverage fields are in the manifest and feature panel.

## Fixed held-out comparisons

Negative NLL differences favor the added feature. The restricted Context 1.3-style reference uses the same rank-history, recruiting, talent, returning-production, coaching, and transfer inputs, training-only preprocessing, rank-distribution likelihood, regularization, and 2013–2021 fit as issue 187. The interaction uses existing `talent_composite` and retains both main effects. Its sign was free in fitting.

| Model | NLL | Δ vs reference | Incremental Δ | CRPS | Expected-rank MAE | Improved vs reference | Improved vs previous | 95% season-cluster interval for incremental Δ |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| `context_1_3_style_restricted_refit` | 4.5191 | +0.0000 | +0.0000 | 0.1094 | 19.786 | — | — | — |
| `context_plus_core_individual_experience` | 4.5190 | -0.0001 | -0.0001 | 0.1094 | 19.778 | 260 (49.2%) | 260 (49.2%) | [-0.0025, +0.0024] |
| `context_plus_core_individual_and_shared_experience` | 4.5281 | +0.0090 | +0.0090 | 0.1104 | 19.944 | 238 (45.1%) | 247 (46.8%) | [-0.0019, +0.0228] |
| `context_plus_core_individual_shared_and_talent_interaction` | 4.5281 | +0.0090 | +0.0000 | 0.1104 | 19.952 | 235 (44.5%) | 255 (48.3%) | [-0.0002, +0.0003] |

The cluster intervals exhaustively resample four held-out seasons (4⁴ ordered draws). They are descriptive and imprecise with only four clusters. Per-team scores and both paired reference and incremental uncertainty summaries are in the CSV and manifest.

Reproducibility audit: held-out team keys match issue 187 exactly. Its committed reference reports 4.5170 NLL; rerunning its unchanged adapter on the pinned source hashes in this environment yields 4.5191. Its `--check` reproduces the feature panel but fails on model-score artifacts. The source of that numerical discrepancy is unresolved; all issue 189 deltas use the same-run refit above.

### Results by season

| Season | Model | N | NLL | Δ vs reference | Incremental Δ | CRPS | Expected-rank MAE |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 2022 | `context_1_3_style_restricted_refit` | 130 | 4.4784 | +0.0000 | +0.0000 | 0.1091 | 19.315 |
| 2023 | `context_1_3_style_restricted_refit` | 131 | 4.4435 | +0.0000 | +0.0000 | 0.1013 | 18.445 |
| 2024 | `context_1_3_style_restricted_refit` | 133 | 4.5853 | +0.0000 | +0.0000 | 0.1180 | 21.304 |
| 2025 | `context_1_3_style_restricted_refit` | 134 | 4.5668 | +0.0000 | +0.0000 | 0.1089 | 20.046 |
| 2022 | `context_plus_core_individual_experience` | 130 | 4.4764 | -0.0020 | -0.0020 | 0.1089 | 19.265 |
| 2023 | `context_plus_core_individual_experience` | 131 | 4.4447 | +0.0012 | +0.0012 | 0.1016 | 18.521 |
| 2024 | `context_plus_core_individual_experience` | 133 | 4.5889 | +0.0036 | +0.0036 | 0.1183 | 21.300 |
| 2025 | `context_plus_core_individual_experience` | 134 | 4.5637 | -0.0031 | -0.0031 | 0.1087 | 19.993 |
| 2022 | `context_plus_core_individual_and_shared_experience` | 130 | 4.4810 | +0.0025 | +0.0045 | 0.1093 | 19.201 |
| 2023 | `context_plus_core_individual_and_shared_experience` | 131 | 4.4524 | +0.0089 | +0.0077 | 0.1024 | 18.658 |
| 2024 | `context_plus_core_individual_and_shared_experience` | 133 | 4.5839 | -0.0014 | -0.0050 | 0.1181 | 21.373 |
| 2025 | `context_plus_core_individual_and_shared_experience` | 134 | 4.5924 | +0.0256 | +0.0287 | 0.1116 | 20.505 |
| 2022 | `context_plus_core_individual_shared_and_talent_interaction` | 130 | 4.4806 | +0.0022 | -0.0003 | 0.1093 | 19.192 |
| 2023 | `context_plus_core_individual_shared_and_talent_interaction` | 131 | 4.4523 | +0.0088 | -0.0001 | 0.1023 | 18.663 |
| 2024 | `context_plus_core_individual_shared_and_talent_interaction` | 133 | 4.5843 | -0.0010 | +0.0004 | 0.1182 | 21.395 |
| 2025 | `context_plus_core_individual_shared_and_talent_interaction` | 134 | 4.5925 | +0.0258 | +0.0002 | 0.1117 | 20.517 |

## Feature and coefficient behavior

| Feature | Mean | Median | P10 | P90 | r vs final-rank fraction |
|:--|--:|--:|--:|--:|--:|
| `ol_core_mean_prior_college_seasons_4y` | 3.104 | 3.200 | 2.400 | 3.800 | -0.107 |
| `ol_core_mean_shared_same_program_seasons_4y` | 2.252 | 2.300 | 0.964 | 3.300 | -0.074 |

Core individual and shared experience correlate at 0.293; the earlier returning-player and returning-group features correlate at 0.999 on this cohort. The core construction materially reduces the earlier near-redundancy, though that alone does not establish predictive value.

| Model | Feature | Standardized location coefficient |
|:--|:--|--:|
| `context_plus_core_individual_experience` | `ol_core_mean_prior_college_seasons_4y` | -0.0345 |
| `context_plus_core_individual_and_shared_experience` | `ol_core_mean_prior_college_seasons_4y` | +0.0189 |
| `context_plus_core_individual_and_shared_experience` | `ol_core_mean_shared_same_program_seasons_4y` | -0.0678 |
| `context_plus_core_individual_shared_and_talent_interaction` | `ol_core_mean_prior_college_seasons_4y` | +0.0170 |
| `context_plus_core_individual_shared_and_talent_interaction` | `ol_core_mean_shared_same_program_seasons_4y` | -0.0579 |
| `context_plus_core_individual_shared_and_talent_interaction` | `ol_core_shared_same_program_seasons_4y_x_talent_composite` | -0.0147 |

The individual-experience coefficient changes sign when shared experience enters; its separate interpretation is unstable. The shared-experience coefficient stays negative in the two nested arms, but held-out likelihood worsens when it enters. The interaction coefficient is small and its incremental predictive result is essentially zero. Coefficients use training-standardized inputs, are regularized, and are not causal effects.

## Concentration and decision

- `context_plus_core_individual_experience` vs its simpler arm: 260 of 528 team-seasons improved; 2/4 seasons improved. The top ten team-seasons account for 19.8% of gross gains and 18.2% of gross losses. Season total ΔNLL: 2022: -0.2602, 2023: +0.1534, 2024: +0.4748, 2025: -0.4101. Largest losses: Navy 2024 (+0.1816), Florida State 2024 (+0.1507), Sam Houston 2025 (+0.0887).
- `context_plus_core_individual_and_shared_experience` vs its simpler arm: 247 of 528 team-seasons improved; 1/4 seasons improved. The top ten team-seasons account for 20.3% of gross gains and 17.8% of gross losses. Season total ΔNLL: 2022: +0.5894, 2023: +1.0118, 2024: -0.6663, 2025: +3.8397. Largest losses: Kennesaw State 2025 (+0.4964), North Texas 2025 (+0.4737), Texas Tech 2025 (+0.4204).
- `context_plus_core_individual_shared_and_talent_interaction` vs its simpler arm: 255 of 528 team-seasons improved; 2/4 seasons improved. The top ten team-seasons account for 26.8% of gross gains and 19.9% of gross losses. Season total ΔNLL: 2022: -0.0427, 2023: -0.0159, 2024: +0.0550, 2025: +0.0245. Largest losses: Kent State 2023 (+0.0213), Texas Tech 2025 (+0.0188), Arizona State 2024 (+0.0162).

Neither core shared experience beyond individual core experience nor the fixed talent interaction met the stated credibility screen. **Reject** CFBD roster-based OL continuity until materially better evidence, such as point-in-time depth charts, starts, or reliable preseason OL membership, becomes available. No further feature search is proposed.

## Limits

Unobserved JUCO, lower-division, or other college seasons cannot contribute to measured experience. CFBD target OL rosters are retrospective, and player identities and historical memberships retain the limitations documented in issues 183, 185, and 187. Prior seasons are strictly before T, but retrospective target membership can itself reveal later availability. These post-hoc results do not validate preseason deployment or revise the conclusion of issue 187.
