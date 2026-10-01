# Context 1.3 location-error structure (issue 154)

## Executive finding

The concentrated misses are real, but one feature does not explain them. In 2023–2025, Context 1.3 was closer to the eventual expected rank than History 1.1 by **1.43 ranks per team-season preseason**, yet its final shared-checkpoint NLL was **0.0728 worse**. The worst 10% of team-seasons by final Context-minus-History NLL gap account for **76.2% of the net gap**. The predeclared expensive-miss group has 35 of 403 team-seasons and contributes 16.31 of 29.33 NLL-gap units (55.6%).

The strongest descriptive stratum is a large `context_only_positive_sum`: the sum of positive fitted Context-only terms before negative terms offset them. Its top within-season quartile has 20 expensive misses among 98 fitted 2023–2025 teams (20.4%), versus five among 101 (5.0%) in the bottom quartile; mean final NLL disadvantage is +0.178 versus +0.040. The **net** `context_only_subtotal` includes those opposing negative terms. Its top quartile has 19/98 expensive misses and mean final gap +0.168, versus 5/101 and +0.050 at the bottom. The two top quartiles overlap for only 74 of 98 teams, so they are materially different diagnostics. The next experiment targets positive values of **`context_only_subtotal`**, because this is the net Context-only term in Context's fitted location center. The positive-sum result remains supporting evidence. For the net term, the expensive-miss association is directionally consistent in 2023–2025, but the final-NLL Q4-versus-Q1 association reverses slightly in 2024; posterior benefit from moderation must be tested. The prominent contributing family is multiyear recruiting, especially the three-year recruiting-points contribution, but its direct final-NLL association is weak in 2024. This is a correlated signal family, not evidence that any coefficient is wrong or causal.

Large Context–History disagreement is not intrinsically harmful. The highest-disagreement quartile improves preseason absolute error for 64% of 2023–2025 teams and has the largest mean preseason improvement (4.22 ranks), while still contributing 9.62 NLL-gap units. Opposing fitted signals and training-space distance do **not** consistently identify the expensive cases. Transfer volume, offensive usage, DB coverage, and the 15 known repaired-input cases also do not explain the crossover. The 2022 control shows that positive `context_only_subtotal` values and large disagreements can help; the question for a later experiment is whether the *direction and reliability* of that net term changed after 2022.

These are retrospective associations. The corrected transfer evidence was not substituted for the frozen model inputs, and its August 15 historical availability remains unverified.

## Population, definitions, and reproducibility

The sole modeling input is the committed [`team_seasons.csv`](../data/processed/context_location_error_diagnostics/team_seasons.csv) from issue 152 / PR 153, with its feature inventory, summary, and provenance. The script [`study_context_location_errors.py`](../scripts/study_context_location_errors.py) validates its 534 unique team-seasons and metric orientations, uses the frozen raw location inputs and fitted contributions, and records SHA-256 hashes in the new study provenance. It does not refit or publish rankings. Every team-season has equal weight, as in the #147 comparison. The per-team NLL gap sums to the Context-minus-History aggregate difference; dividing each gap by the period's team count gives its contribution to the mean gap.

| Season | All | Fitted feature rows | Cold-start fallback | Expensive misses | Mean preseason absolute-error difference, C−H | Mean final NLL gap, C−H |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 131 | 130 | 1 | 12 | −2.729 | −0.0061 |
| 2023 | 133 | 131 | 2 | 11 | −0.887 | +0.0785 |
| 2024 | 134 | 133 | 1 | 12 | −2.158 | +0.0695 |
| 2025 | 136 | 134 | 2 | 12 | −1.255 | +0.0704 |
| 2023–2025 | 403 | 398 | 5 | 35 | −1.434 | +0.0728 |

Positive C−H absolute-rank error means Context was farther from the target; positive C−H final NLL favors History. Within each season, quartiles are balanced by ascending value and canonical team ID breaks exact ties. **Expensive miss** means top quartile Context absolute error *and* top quartile final NLL disadvantage. **Large miss recovered** means top quartile Context absolute error and final gap ≤ 0. **Material disagreement** means top quartile absolute preseason Context–History expected-rank difference; helpful/harmful refers to the sign of the preseason absolute-error difference. The artifact also provides an exclusive group label with precedence expensive, recovered, helpful, harmful, ordinary. Continuous measures are retained throughout. These groups describe outcomes; they are not usable preseason selectors.

Prediction comparisons use all 534 rows. Feature, contribution, support, and transfer diagnostics use only the 528 fitted rows (130/131/133/134 by season). Descriptive columns with missing values report their observed sample sizes. The 2023–2025 population has median C−H preseason error difference −1.365 ranks (10th/90th percentiles −13.458/+10.224) and median final gap +0.0259 (10th/90th −0.117/+0.324). The 2022 median final gap is −0.0071.

## Failure concentration

| Worst fraction by final C−H NLL gap | 2023–2025 teams | Gap sum | Share of net +29.326 gap |
|---|---:|---:|---:|
| 5% | 21 | +14.526 | 49.5% |
| 10% | 41 | +22.348 | 76.2% |
| 20% | 81 | +32.933 | 112.3% |
| 25% | 101 | +36.363 | 124.0% |

Shares above 100% are possible because Context wins elsewhere offset these losses. The top-10% share is 85.4% in 2023, 70.1% in 2024, and 74.8% in 2025. The aggregate 2022 gap is −0.793, so a share of its negative net total is not a useful concentration measure; 2022 nevertheless has twelve expensive misses and positive loss contributions offset by larger Context wins. This reproduces the phenomenon from #147 before examining features.

## Feature and fitted-contribution structure

[`feature_associations.csv`](../data/processed/context_location_error_study/feature_associations.csv) reports, for each of the 15 location features and each of three outcomes, frozen raw input and combined fitted numeric-plus-missing contribution: Pearson and Spearman correlations, within-season standardized Pearson correlation, observed/missing counts, quartile means/medians, and decile means. It gives each season, pooled 2023–2025, and 2022. It also covers training percentiles, support metrics, and descriptive transfer fields. Quartile means use within-season bins, including in the pooled period; decile summaries are descriptive pooled tails.

The clearest raw-value association with **Context absolute error** is the correlated talent/recruiting family, especially lower three-year recruiting points: Pearson r is +0.010 in 2022 (n=130), −0.193 in 2023 (n=131), −0.166 in 2024 (n=133), and −0.279 in 2025 (n=134); pooled r=−0.209, Spearman ρ=−0.149, and the within-season standardized r=−0.213 (n=398). The top-versus-bottom feature quartile differs by −8.12 Context error ranks pooled. The same feature has weaker associations with *relative* preseason error (pooled r=−0.115, quartile difference −2.37 ranks). Talent composite is similar (pooled n=396, r=−0.214 Context error and −0.126 relative error), so much of the raw association reflects generally harder lower-talent teams, not a unique Context defect. History's absolute error also correlates with three-year recruiting points (r=−0.140, 2023–2025).

For final NLL disadvantage, raw three-year recruiting points has pooled r=−0.221, but the per-year r is −0.283/ **+0.016** /−0.371 in 2023/2024/2025, so this specific link is not uniformly stable. The corresponding fitted contribution reverses sign (pooled r=+0.202 with NLL gap). Among 2023–2025 expensive fitted cases (n=34), its mean contribution is +0.166 versus −0.092 in other fitted cases; the difference is +0.311/+0.223/+0.248 by year, compared with +0.063 in 2022. Coach tenure and returning production also frequently make large contributions, but no individual feature dominates the expensive cases: their largest absolute contribution is three-year recruiting points for 11, coach tenure for 10, returning production for 10, long-run history for two, and incoming offensive usage for one. These are fitted terms, not causal attributions.

The contribution diagnostics sum each numeric and missing-indicator term, keep the History-derived subtotal (lag-1 plus lag-2, lag-3, and long-run terms) separate from Context-only terms, and verify they reconstruct every fitted Context center. Let each Context-only feature's combined numeric-plus-missing contribution be `cᵢ`. Then `context_only_positive_sum = Σ max(cᵢ, 0)`, while `context_only_subtotal = Σ cᵢ = context_only_positive_sum + context_only_negative_sum`. Thus `context_only_subtotal` is the signed **net** Context-only term relative to Context's own History-derived terms and intercept. It is not the full `context_minus_history_location_center`, because the two priors also differ in their baseline construction. The mean largest-feature share of total absolute contribution is **0.304** in expensive 2023–2025 cases versus **0.322** otherwise. Expensive cases are therefore not primarily one-feature dominance. Their mean `context_only_subtotal` is +0.517 versus +0.095 otherwise; mean `context_only_positive_sum` is +0.967 versus +0.655. Their cancellation fraction is lower (0.445 versus 0.543), and the high-cancellation Context-only quartile has only 2.0% expensive misses versus 12.9% in the low-cancellation quartile. This argues against major internal signal conflict as the main explanation.

| Diagnostic | Period | Within-season quartile | Fitted n | Expensive misses | Miss rate | Mean final NLL gap |
|---|---|---|---:|---:|---:|---:|
| `context_only_positive_sum` (supporting) | 2023–2025 | Bottom | 101 | 5 | 5.0% | +0.040 |
| `context_only_positive_sum` (supporting) | 2023–2025 | Top | 98 | 20 | 20.4% | +0.178 |
| `context_only_subtotal` (experiment target) | 2023–2025 | Bottom | 101 | 5 | 5.0% | +0.050 |
| `context_only_subtotal` (experiment target) | 2023–2025 | Top | 98 | 19 | 19.4% | +0.168 |
| `context_only_subtotal` | 2023 | Bottom | 33 | 1 | 3.0% | +0.013 |
| `context_only_subtotal` | 2023 | Top | 32 | 6 | 18.8% | +0.232 |
| `context_only_subtotal` | 2024 | Bottom | 34 | 3 | 8.8% | +0.116 |
| `context_only_subtotal` | 2024 | Top | 33 | 8 | 24.2% | +0.106 |
| `context_only_subtotal` | 2025 | Bottom | 34 | 1 | 2.9% | +0.021 |
| `context_only_subtotal` | 2025 | Top | 33 | 5 | 15.2% | +0.169 |
| `context_only_subtotal` (control) | 2022 | Bottom | 33 | 3 | 9.1% | −0.025 |
| `context_only_subtotal` (control) | 2022 | Top | 32 | 3 | 9.4% | −0.041 |

The `context_only_subtotal` Q4 expensive-miss rate exceeds Q1 in every primary season: 18.8% versus 3.0% in 2023, 24.2% versus 8.8% in 2024, and 15.2% versus 2.9% in 2025. The final-NLL relation is less consistent: Q4 exceeds Q1 by +0.218 in 2023 and +0.148 in 2025, while Q4 is **0.010 lower** in 2024 (+0.106 versus +0.116). The continuous Pearson correlation between `context_only_subtotal` and final NLL gap is +0.324/−0.014/+0.279 in 2023/2024/2025 (n=131/133/134 fitted rows). The pooled final-NLL association must therefore not be read as uniform across years.

The top `context_only_positive_sum` quartile contributes +17.448 of +28.778 fitted-row NLL-gap units and contains 20 of 34 fitted expensive misses. Its mean Context–History signed preseason expected-rank difference is +11.74 ranks (Context worse numerically), yet Context still improves preseason absolute error for 48% of these teams. The top **`context_only_subtotal`** quartile contributes +16.497 gap units and contains 19 of 34 fitted expensive misses. Both are interpretable descriptive strata, not preseason policies: their quartile boundaries were defined on the target seasons for description, and the large positive contributions are correlated with lower recruiting levels. A post hoc low-recruiting intersection adds little independent evidence.

## Context versus History disagreement

| Absolute preseason disagreement quartile | 2023–2025 n | Mean Context error | Mean History error | Mean C−H error | Context closer | Mean final NLL gap | Gap sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 102 | 18.07 | 18.19 | −0.12 | 52% | +0.041 | +4.22 |
| Q2 | 100 | 19.14 | 19.41 | −0.27 | 52% | +0.057 | +5.69 |
| Q3 | 101 | 20.59 | 21.74 | −1.15 | 59% | +0.097 | +9.80 |
| Q4 | 100 | 21.66 | 25.88 | −4.22 | 64% | +0.096 | +9.62 |

The top disagreement quartile's mean C−H preseason error is −3.36/−5.23/−4.09 ranks in 2023/2024/2025 and −6.87 in 2022. Disagreement often adds useful information. Among fitted material-disagreement cases in 2023–2025, the 63 preseason-helpful cases have mean final gap −0.016 and mean `context_only_subtotal` +0.265; the 36 harmful cases have mean final gap +0.289 and mean `context_only_subtotal` +0.474. Their mean Context-only cancellation fractions, 0.360 and 0.389, are close. Harmful cases have a larger signed Context–History rank departure (+9.3 versus +3.1 ranks on average), but many helpful departures also move to worse numeric rank. This comparison uses observed outcomes to define helpful/harmful; it diagnoses the shift rather than predicting which teams will benefit.

Large fitted center displacement is mixed: in 2023–2025 its top quartile has mean final gap +0.128 versus +0.035 in the bottom quartile, yet Context's preseason absolute-error advantage also grows (−3.34 versus −0.34 ranks). In 2022 the top-displacement quartile has a negative mean final gap (−0.046). Thus simply limiting Context–History disagreement would discard real preseason gains; any shrinkage experiment needs rolling-origin evaluation of the full predictive distribution.

## Training support and extrapolation

Rolling-origin support is measured by nearest and mean-five-nearest Euclidean distance in the fitted standardized 15-feature-plus-missing-indicator space. The 2023–2025 nearest-distance quartiles are non-monotone:

| Nearest-distance quartile | Fitted n | Mean Context absolute error | Mean C−H preseason error | Mean final NLL gap | Expensive-miss rate |
|---|---:|---:|---:|---:|---:|
| Q1 | 101 | 20.16 | −2.52 | +0.032 | 4.0% |
| Q2 | 99 | 21.46 | −0.67 | +0.098 | 13.1% |
| Q3 | 100 | 17.73 | −1.36 | +0.054 | 5.0% |
| Q4 | 98 | 20.35 | −1.15 | +0.107 | 12.2% |

The nearest-distance correlation with final gap is only r=+0.093 pooled; by season it is +0.046/+0.054/+0.260, versus −0.128 in 2022. Mean-five-nearest distance is likewise weak (pooled r=+0.073). Context absolute-error correlation with nearest distance is −0.011 pooled. These data do not establish sparse support as the main failure mechanism.

Only 22 of 398 fitted 2023–2025 rows violate any feature training range, and one of them is an expensive miss. By violation count, the 376 in-range rows average 20.03 Context error ranks and +0.070 final gap; 20 rows with one violation average 19.39 and +0.105; the one row with two violations has 3.01 and +0.342; the one with three has 6.43 and −0.070. In-range rows contribute +26.407 of the +28.778 fitted gap. The most distant case, UCF 2023 (distance 18.675, one range violation), has Context error 12.05 ranks and final gap +0.108, well below the largest misses. In 2022, Ole Miss (15.868, zero violations) has a slight Context win, and Arizona State (12.977, zero violations) has a larger one. Individual per-feature rolling training percentiles and range statuses are included in the machine-readable associations and grouped diagnostics. An extreme multivariate distance need not imply a marginal range violation.

## Transfer diagnostics and repair sensitivity

Transfer-facing frozen inputs and the separate repaired/coverage metadata were analyzed separately. The 2023–2025 top transfer-count quartile (n=98) has mean final gap +0.065 and 9.2% expensive misses, comparable with the bottom quartile's +0.088 and 9.9%. The top observed offensive-usage quartile (n=98) has +0.051 and 8.2%, versus +0.108 and 8.9% at the bottom. The highest observed DB-impact quartile (n=95 of 383 with observed sums) has +0.026 and 5.3% versus +0.111 and 7.3% at the bottom. Low returning production, unknown offensive applicability, and the fitted returning-production/incoming-usage opposition diagnostic show no consistent elevated loss across all three primary years. The fitted incoming-offensive-usage contribution has near-zero correlation with *relative* preseason error (pooled r=−0.003) and a weak final-gap correlation (r=+0.068). These comparisons do not implicate transfer magnitude as a leading failure condition.

DB coverage is **complete/partial/no incoming DB/unavailable** for 183/154/46/15 fitted 2023–2025 teams. Partial coverage has mean final gap +0.065 and 9.7% expensive misses, versus +0.062 and 7.7% for complete coverage; its association changes by year. Unavailable has mean gap +0.110 but only 15 rows, and 2024/2025 each have two. Coverage status is descriptive; the frozen model's availability flag and neutral zero semantics remain untouched. The diagnostic observed DB sum is missing when unavailable.

Exactly **15 fitted** team-seasons have frozen/repaired transfer-input differences: 13 legitimate-zero restorations (eight in 2022, five in 2023), one removed ambiguous usage join (North Texas 2023), and one added name-normalization join (Coastal Carolina 2023). One 2022 zero-restoration case, and none of the seven 2023 repair cases, is in the expensive-miss group. The eight 2022 restorations sum to **−0.598** final NLL-gap units; the five 2023 restorations sum to **+0.039**. North Texas 2023 has Context/History preseason errors 19.19/13.36 ranks and final gap +0.004 (error quartile 3, gap quartile 2). Coastal Carolina 2023 has errors 2.83/2.89 and gap +0.276 (error quartile 1, gap quartile 4). All seven 2023 cases together contribute +0.319 of the +29.326 pooled 2023–2025 gap, and there are no 2024–2025 known differences. This is a sensitivity description of frozen predictions, not a counterfactual repair effect. The repaired evidence's historical timing is unverified.

## 2022 negative control

Context wins the 2022 aggregate final NLL comparison despite twelve expensive misses. Its top `context_only_subtotal` quartile has nearly the same expensive-miss rate as the bottom (9.4% versus 9.1%) and a more favorable final mean NLL gap (−0.041 versus −0.025); the analogous 2023–2025 top quartile has 19.4% expensive misses and +0.168 final gap. The supporting `context_only_positive_sum` also reverses interpretation: its 2022 top quartile has 9.4% expensive misses and −0.023 final gap, versus 20.4% and +0.178 in 2023–2025. High disagreement improves 2022 preseason error more than low disagreement (−6.87 versus −0.25 ranks) and has a negative final gap (−0.039). These contrasts make the net positive Context-only term relevant to the crossover while showing it is not inherently bad. Training distance, contribution dominance, and transfer-volume patterns are too weak or inconsistent in the primary seasons to become crossover-specific explanations.

## Representative team-seasons

Full target ranks, both final NLLs and location centers, fitted top-five and opposing contributions, support fields, DB coverage, and transfer provenance for 19 selected cases are in [`case_studies.csv`](../data/processed/context_location_error_study/case_studies.csv). The compact examples below show why group-level evidence is needed:

| Season/team | Context / History / target expected rank | Context / History error | Final C / H NLL | C / H location center | Nearest distance; range violations | Leading fitted contributions; DB state |
|---|---:|---:|---:|---:|---:|---|
| 2023 James Madison | 109.1 / 56.0 / 26.6 | 82.5 / 29.5 | 5.404 / 4.002 | +1.895 / −0.439 | 5.04; 0 | 3y recruiting +0.932, returning +0.622, coach +0.506; opposing 2y recruiting −0.234; complete |
| 2024 Navy | 107.3 / 82.6 / 37.6 | 69.6 / 45.0 | 5.508 / 4.602 | +1.726 / +0.599 | 2.23; 0 | 3y recruiting +0.764, class rank +0.236; no incoming DB |
| 2025 North Texas | 105.4 / 92.7 / 25.1 | 80.3 / 67.6 | 4.385 / 3.921 | +1.567 / +0.991 | 1.57; 0 | coach +0.463, returning +0.441; partial DB |
| 2022 Tulsa | 91.5 / 67.2 / 91.8 | 0.3 / 24.6 | 3.633 / 3.829 | +1.067 / +0.050 | 2.08; 0 | coach +0.496, 3y recruiting +0.385; no incoming DB |
| 2024 Air Force | 83.8 / 58.2 / 105.3 | 21.5 / 47.1 | 3.477 / 3.437 | +0.644 / −0.371 | 3.34; 0 | 3y recruiting +0.504, returning +0.347; opposing coach −0.269; no incoming DB |
| 2023 UCF | 46.3 / 57.3 / 58.4 | 12.0 / 1.1 | 4.962 / 4.854 | −0.836 / −0.393 | 18.67; 1 | DB impact −0.192, 3y recruiting −0.155; complete DB |

James Madison and Navy illustrate expensive losses with aligned positive contributions. Tulsa shows that a similar direction can be highly helpful in 2022. Air Force shows a preseason-helpful move that does not guarantee a final NLL win; the posterior score evaluates the whole distribution. UCF is an extreme support-distance case without an extreme Context preseason error. North Texas and Coastal Carolina 2023 are documented above as transfer repair cases. Every selected case carries retrospective transfer timing marked `historical_timing_unverified`; none establishes a causal feature effect.

## Recommended next experiment

**One constrained Context 1.4 research hypothesis:** test whether rolling-origin, training-defined moderation of **positive values of `context_only_subtotal`** improves posterior NLL without losing the 2022 and preseason wins. `context_only_subtotal` is the signed sum of all Context-only fitted location contributions after negative terms offset positive ones. Its high quartile consistently contains more expensive misses in 2023–2025, but the final-NLL association is clear in 2023 and 2025 and weak/non-monotonic in 2024. The experiment must **test**, rather than assume, whether moderating the net term improves rolling-origin posterior NLL in each season and overall. The proposed moderation changes that term relative to Context's own History-derived location components and intercept; it does not directly moderate `context_only_positive_sum` or the full Context-minus-History center difference. Specify any moderation rule from each training fold only, preserve frozen transfer-input semantics, and compare preseason and shared-evidence posterior distributions against unchanged Context 1.3 and History 1.1 baselines. Lower multiyear recruiting is a correlated descriptive condition that may be examined as a predeclared secondary analysis; this study does not justify using it as an intervention selector. No threshold, coefficient, or production model is chosen here. If the rolling-origin test fails to preserve gains, the appropriate conclusion is that no compact reliable preseason selector has yet been found.

## Artifacts

Run `uv run python scripts/study_context_location_errors.py` from the repository root. The script writes [`feature_associations.csv`](../data/processed/context_location_error_study/feature_associations.csv), [`contribution_diagnostics.csv`](../data/processed/context_location_error_study/contribution_diagnostics.csv), [`grouped_diagnostics.csv`](../data/processed/context_location_error_study/grouped_diagnostics.csv), [`case_studies.csv`](../data/processed/context_location_error_study/case_studies.csv), [`summary.json`](../data/processed/context_location_error_study/summary.json), and [`provenance.json`](../data/processed/context_location_error_study/provenance.json). Repeated runs against identical inputs are byte-identical. No inferential p-values or feature-selection sweeps were used; feature and intersection comparisons are descriptive, and the low-recruiting/high-push intersection is exploratory.
