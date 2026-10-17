# Issue 213: Context dependent likelihood forecast experiment

## Frozen protocol

This protocol is committed before scoring any candidate forecasts. The 2023–2025
seasons are exploratory development seasons because prior research has informed
them. A positive result supports a separately designed prospective confirmation
only; it does not authorize production changes or establish why the ranking
measurements disagree with game outcomes.

The experiment asks whether the fixed Context dependent margin correction from
#195 improves genuine future game forecasts beyond both unmodified Context and a
matched generic likelihood recalibration. The simpler likelihood remains the
preferred model unless the Context terms show a useful incremental benefit.

## Fixed arms

All arms use #200's repaired `Context_existing` prior, corrected #204 support,
existing policy, Student-t degrees of freedom 15, common scale form, ordinal-rank
mapping, and #200 belief propagation settings.

1. **Context existing:** unchanged #200 existing-policy likelihood.
2. **Context generic recalibrated:** the exact target-season
   `rank_site_calibration_fit` in #195.
3. **Context dependent recalibrated:** the exact target-season
   `rank_site_context_fit` in #195, including its jointly fitted generic terms,
   Context coefficients, and training-only preprocessing.

For arms 2 and 3, the #195 conditional margin surface is reconstructed as
`mu_new = multiplier * mu_base + home_adjustment * (1 - neutral) + context_delta`
and `scale_new = scale_base * exp(common_log_scale_correction)`. The Context
delta is the saved regularized linear correction applied to standardized
home-minus-away Context values and missingness indicators. Arm 2 has no Context
delta. Arm 3 uses its own joint generic terms; arm 2's coefficients are never
grafted onto it. All and only the saved #195 columns, imputation/scaling values,
coefficients, and selected penalties are used. No fit, feature search,
interaction, Context dependent scale, replacement prior, or scalar team-strength
state is introduced.

The corrections apply only to FBS–FBS factors and predictions. FBS–FCS and
FCS–FCS likelihoods remain exactly #200's existing likelihood in every arm.
Cross-subdivision evidence stays in posterior replay. The #195 FBS–FBS fit is
not extrapolated to FCS games.

## Frozen inputs and replay

- Reuse `data/research/issue_195_likelihood_diagnostic/fit_records.json` with
  SHA-256
  `a6bb5229c6991c5dd1661bc138f6c322b071c7591c0d7b4c75315b3966e59530` and verify
  the #195 manifest and every recorded input hash. Its base existing-policy
  #200 likelihood must have SHA-256
  `b64d237d4abf3121355a28505f805252cf32062b3fce66db6ca620ef35887952`.
- Use the exact #200 exported `context_feature_inputs.csv` and its feature
  manifest; verify both against the rebuilt #200 output manifest and the #195
  input hashes. Keep native cold starts and observed missingness. Missing rows
  are an input failure and cannot be replaced with zero vectors or silently
  removed from only one arm.
- Evaluate 2023, 2024, and 2025. Keep #195's parameters fixed: the 2023 fit uses
  2022; the 2024 fit uses 2022–2023; the 2025 fit uses 2022–2024. Do not refit,
  alter its loss, select another policy, or inspect evaluation scores to choose
  parameters.
- Freeze the #201 weekly schedule plan before reading result fields. Reuse its
  canonical IDs, regular-season FBS–FBS cohort, event phases, 48-hour evidence
  lag, outcome availability proxy, and existing-policy evidence rules. Include
  one rolling forecast per retained primary game. Keep preseason-only forecasts
  separate. Preserve rivalry, conference-championship, and Army–Navy labels;
  bowls and CFP games are excluded from primary scoring.
- Every arm rebuilds each cutoff posterior from the same #200 Context prior and
  allowed evidence using that arm's likelihood on every FBS–FBS evidence
  factor. The target game and its outcome must be absent from its own evidence.
  Apply the corresponding arm likelihood to its future FBS–FBS predictive
  mixture as well.
- Evaluation-season final and pre-bowl observations are never posterior inputs
  or game prediction features. Reconstruct predictions by mixing over complete
  rank PMFs under #201's Student-t predictive contract, with its numerical error
  checks. Preserve and hash the full posterior PMFs. Corrected pre-bowl target
  PMFs are a secondary ranking measurement only and must pass #201's strict
  target-date and evidence safety checks. #195's earlier-season final-rank
  measurements remain labeled as training measurements.
- Reproduce #201's `Context_existing` per-game metrics on the retained common
  cohort within absolute tolerance `1e-8`. Verify game IDs, cutoffs, evidence
  hashes, common teams, and baseline PMFs before interpreting candidate scores.
  Fail the run on a mismatch, non-converged posterior, or invalid candidate PMF.

## Measures and fixed decisions

The primary score is paired future-game margin CRPS, lower is better. Report the
pooled equal-game mean, equal-season mean, and each season separately. Apply
#193's practical-gain threshold: an arm must lower CRPS by at least 0.05 margin
points in both aggregate summaries and in at least two of the three seasons.
This is a descriptive rule, not a significance test. Compare arm 3 against both
arms 1 and 2, and report arm 2 against arm 1.

Also report margin NLL, home-win Brier score, median-based margin MAE, 50/80/95%
interval coverage and widths, phase counts, and gain/loss concentration by team
and game. For any candidate recommended for independent confirmation, enforce
#193's early guardrails against arm 1 separately for preseason and
pre-Thanksgiving forecasts: pooled CRPS deterioration may not exceed 0.10 and
season deterioration may not exceed 0.25; mean absolute coverage error may not
increase by more than 0.03 pooled or 0.05 in any season. Publish all values and
failures.

Secondary ranking scores use corrected pre-bowl target PMFs: NLL, rank CRPS,
expected-rank error, 80% target mass and interval width. Give each team-season
total weight one across valid checkpoints in the pooled aggregate; report
preseason separately. Ranking gains cannot substitute for improved future-game
CRPS. Repeated games, teams, and checkpoints are not independent confirmation.

Conclude **Context dependent correction merits independent confirmation** only
if arm 3 passes the practical-gain rule against both other arms and passes the
early guardrails against arm 1. Conclude **Prefer generic recalibration for
further confirmation** if arm 2 clears its rule against arm 1 and arm 3 fails to
establish the required incremental gain over arm 2. Otherwise conclude **Retain
unmodified Context / no useful replacement established**. Report all score
directions and mixed patterns regardless of threshold outcome.

## Deliverables

The research-only runner writes a hash manifest, complete per-game paired
scores, season/phase/calibration summaries, corrected pre-bowl rank scores,
losslessly reconstructible posterior and predictive PMFs, evidence IDs and row
hashes, convergence diagnostics, baseline parity audit, fit provenance, and a
concise decision report. It has explicit input and output roots and does not
change production defaults, priors, likelihood artifacts, inference settings,
published snapshots, website defaults, or raw downloads.

Any promising result needs a new prospective confirmation plan with predictions
frozen before outcomes become available. A positive development result does not
show that rankings are wrong, that Context belongs in production likelihoods, or
that ranking measurements recover latent true ranks. No further feature search
or production promotion is part of this experiment.
