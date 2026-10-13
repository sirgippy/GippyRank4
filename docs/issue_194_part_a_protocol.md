# Issue 194 Part A: frozen historical protocol

## Scope and provenance

This run covers **Part A only**, selected by the user after issue 194's execution
gate. Part B remains pending and is not run here. This is exploratory development:
the 2022–2025 seasons and the source ranking measurements have already informed
project work. No result promotes a model or changes production settings.

Use the existing-policy History and Context priors and likelihood from the
matched issue 200 bundle. Do not refit either model. Use the corrected #204
pre-bowl rank target PMFs as the common secondary ranking measurement. Do not
change prior-season features, the FCS fallback population, the regular-season
or postseason evidence policy, or belief-propagation settings.

The frozen issue 201 artifacts define the safe original-checkpoint cohort and
event phases. This is the 24-checkpoint cohort from the #201 frozen-target
sensitivity, not the 62 weekly rolling checkpoints. Use the exact original
evidence IDs and common FBS team IDs audited by #201. Verify the source game
rows and keep each cutoff fixed. The four final original checkpoints are
excluded because their evidence is not target-safe; do not move cutoffs or call
the retained comparison a December replication. The #192 frozen PMFs are not
used as model outputs in this analysis; their evidence file is used only to
recover the already-reviewed evidence IDs.

### Fixed checkpoint set

All dates are UTC. `Other` is the #201 checkpoint phase label.

| Season | Other checkpoints | Pre-Thanksgiving | Rivalry window | Excluded final checkpoint |
| --- | --- | --- | --- | --- |
| 2022 | Aug 27, Sep 22, Oct 7, Oct 27 | Nov 9 | Nov 23 | Dec 10 |
| 2023 | Aug 26, Sep 17, Oct 6, Oct 21 | Nov 4 | Nov 18 | Dec 9 |
| 2024 | Aug 24, Sep 19, Oct 6, Oct 24 | Nov 8 | Nov 23 | Dec 14 |
| 2025 | Aug 23, Sep 19, Oct 8, Oct 24 | Nov 9 | Nov 23 | Dec 13 |

The 24 retained checkpoints have 16 `other_checkpoint`, four
`pre_thanksgiving`, and four `rivalry_window` observations. There is no
target-safe `conference_championship_period` checkpoint in this cohort. Keep
2022 visible in every table and summary.

## Crossed-message diagnostics

At every retained season/checkpoint, replay the same #201-audited evidence set
through the issue 200 existing-policy History and Context arms. Score only the
common FBS team IDs with identical rank support in both priors and the corrected
pre-bowl target. Require a converged replay and verify that each included game
kickoff precedes the target observation date.

For team `i`, let `p_H` and `p_C` be the existing-policy priors and `q_H` and
`q_C` their native replay posteriors. Extract each native model's normalized
message product from `q / p`, using the existing #196 positive-support guard.
Do not infer a message where its own prior is zero, smooth zeros, or change
support. Form four local PMFs:

- `HH = p_H × M_H` (native History)
- `CH = p_C × M_H` (Context prior with frozen History messages)
- `HC = p_H × M_C` (History prior with frozen Context messages)
- `CC = p_C × M_C` (native Context)

Each product is normalized on its existing positive support. Verify that `HH`
and `CC` reproduce the native posteriors within the #196 identity tolerance.
These crossed PMFs are frozen-message hybrids, not newly converged posteriors;
in loopy graphs their messages include focal-team feedback.

For each team/checkpoint and each reported metric `S`, compute the anchored
factorial decomposition:

```text
direct prior = S(CH) - S(HH)
transmitted  = S(HC) - S(HH)
interaction  = S(CC) - S(CH) - S(HC) + S(HH)
total        = S(CC) - S(HH)
```

The three components sum to the native Context-minus-History contrast. They are
descriptive sensitivities, not causal shares. Report means with equal team
weight within each checkpoint, preserving season and checkpoint rows.

For all four PMFs, report target cross-entropy/NLL, discrete rank CRPS, central
80% target mass and interval width, entropy, and Herfindahl concentration
`sum(p**2)` (with its reciprocal as effective rank support). Use the existing
#201 score convention for NLL, including its `1e-15` scoring floor; this floor
does not modify priors, messages, or posterior PMFs. Also report direct,
transmitted, interaction, and total contrasts for NLL, rank CRPS, 80% target
mass, 80% width, and concentration.

Place these ranking summaries beside the retained #201 existing-policy rolling
future-game CRPS for the same seasons and named phases. The game CRPS is a
contextual operational measure with a different forecast population and unit;
it is not paired with the checkpoint ranking score. Show 2022–2025 and preserve
missing phase coverage rather than filling it with a shifted cutoff.

Keep the ticket's original final-target December pilot values in a clearly
labeled reference note only: History NLL 3.83089, Context NLL 3.88793, Context
prior with History messages 3.85330, and History prior with Context messages
3.85739. They are not recomputed, are not the corrected-target results, and do
not identify correlated errors or BP error.

## Preseason shared-error comparison and matched null

Use the issue 200 existing-policy preseason prior PMFs with no current-season
games, and score them against the same corrected #204 pre-bowl targets. The
pairing universe is every unordered pair of common FBS teams in a season.
Scheduled opponent pairs come only from the outcome-blind #201 schedule plan:
regular-season FBS–FBS games, including conference championships and
Army–Navy. Collapse rematches to one team pair while retaining scheduled game
count. A team's schedule degree is its number of unique scheduled FBS
opponents. Do not use scores, realized losses, target ranks, forecast errors,
or posterior results to define the universe or matches.

For team `i` and each model, compute full-PMF NLL and signed midpoint-CDF
calibration residual:

```text
r_i = sum_r target_i[r] * (sum_{k < r} forecast_i[k]
                           + 0.5 * forecast_i[r]) - 0.5
```

Expected-rank residual (forecast expected rank minus target expected rank) is a
secondary diagnostic. The primary co-movement statistic is the mean product
`(r_i - mean_season(r)) * (r_j - mean_season(r))` over pairs. Positive values
mean the pair's signed residuals co-move in the same direction relative to the
season/model mean.

For each season, match every scheduled opponent pair to one unique
non-opponent pair without replacement. Build six matching features from
preseason forecasts and schedule structure only: mean pair expected-rank level;
absolute difference in pair expected-rank level; mean within-pair
History–Context expected-rank disagreement; absolute difference in that
disagreement between the two teams; mean schedule degree; and absolute schedule
degree difference. Expected ranks and model disagreement are divided by the
season's rank population. Standardize each feature by its standard deviation
over that season's complete pair universe (a zero standard deviation gives the
feature zero weight). Use Euclidean distance in this six-feature space.

Matching is deterministic and outcome-blind: prioritize opponent pairs in
descending order of their nearest available-control distance, then canonical
team IDs; assign the nearest remaining non-opponent pair, with canonical team
IDs as the distance tie-break. Use no caliper. Record every pair, match,
distance, multiplicity, and exclusion in the machine-readable pairing file.
This same matched pair set is used for History and Context. The matched
non-opponent set is the descriptive null; matching accounts for pair-level
schedule-degree level and difference, but does not enforce exact degree
balance. It does not constitute a randomized experiment.

Report opponent and matched-control means, their matched differences, full-PMF
NLL, signed residual, centered residual product, and expected-rank residual by
season/model. Give seasons equal weight in the across-season descriptive
summary. Do not report p-values or treat overlapping pairs as independent.
The four seasons are only four season clusters.

Classify historical co-movement as **descriptively consistent** with shared
Context errors only if Context's opponent-minus-matched-control residual
co-movement is positive and greater than History's in at least three of four
seasons. Classify it as **descriptively unsupported** if either condition fails
in at least three seasons; otherwise call it **unresolved**. This is a
directional summary, not proof of cause. Similar team levels, schedule
structure, common ranking-measurement error, and shared fitted coefficients
remain alternatives.

## Limits and stopping point

This part can describe existing priors, messages, score differences, and
preseason error co-movement. It cannot establish that Context causes shared
errors, that loopy BP is inaccurate at full season scale, or that Context harms
future-game forecasts. Better game CRPS can coexist with worse ranking NLL.
Any synthetic dependence or BP-approximation conclusion is pending Part B.

Stop after this historical analysis. Do not fit a replacement model, change
damping/tolerance, add prior decay, introduce a correlated full-season prior,
search conference/roster features, or alter production outputs. Any full-season
joint model is a separate research decision.
