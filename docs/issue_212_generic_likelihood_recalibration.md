# Issue 212: generic likelihood recalibration for Context forecasts

## Frozen protocol

This protocol is frozen before replaying candidate scores. The evaluation uses
2023, 2024, and 2025, which are exploratory development seasons. It does not
change production priors, likelihood artifacts, inference settings, or website
defaults.

The experiment has exactly two arms, both using the #200 existing-policy
Context prior and the same evidence schedule:

1. `Context_existing`: the unchanged #200 existing-policy rank/site likelihood.
2. `Context_generic_recalibrated`: the same prior with the #195
   `rank_site_calibration_fit` for that target season.

For FBS–FBS evidence and future forecasts only, the candidate uses
`mu_new = multiplier * mu_base + home_adjustment * (1 - neutral)` and
`scale_new = scale_base * exp(log_scale_correction)`. The #195 positive
multiplier, non-neutral home adjustment, and common log-scale correction are
read from the verified saved forward fit for each evaluation season. Student-t
degrees of freedom remain 15. Rank coordinates, priors, canonical teams, and
FBS population support remain fixed. The candidate adjustment is common to all
teams and uses no Context feature columns.

FBS–FCS and FCS–FCS conditional factors keep the unmodified #200 likelihood.
They remain in evidence with their existing priors; no correction is
extrapolated to them. FBS–FCS forecasts, if retained, are reported separately
because their FBS posterior may change through FBS–FBS evidence.

## Replay and scoring contract

Use the matched #200 repaired Context prior and existing-policy likelihood,
with the #204 corrected population/support policy. Verify the #195 manifest,
its declared input hashes, the exact saved fit-record SHA-256
`a6bb5229c6991c5dd1661bc138f6c322b071c7591c0d7b4c75315b3966e59530`, and the
base likelihood SHA-256
`b64d237d4abf3121355a28505f805252cf32062b3fce66db6ca620ef35887952`.
Use the saved 2023, 2024, and 2025 fits trained through 2022, 2023, and 2024,
respectively. Do not refit or select settings using evaluation scores.

Use #201's frozen score-blind schedule plan, Tuesday 00:00 UTC weekly cutoffs,
48-hour evidence lag, canonical game IDs, phase definitions, existing-policy
evidence rules, and corrected FBS–FBS forecast cohort. Rebuild each cutoff
posterior from the same Context prior and all eligible evidence. The target
game must be absent from its own evidence. Score exactly one rolling forecast
per retained regular-season FBS–FBS game. Keep preseason forecasts separate;
report FBS–FCS forecasts separately. Bowls and CFP games are excluded from
primary scoring. Retain rivalry, conference-championship, and Army–Navy phase
labels.

Use full inferred rank PMFs for prediction, not expected-rank plug-ins. Apply
the candidate likelihood during both posterior inference and future FBS–FBS
prediction. FBS–FCS and FCS–FCS evidence factors and predictive likelihoods
remain unchanged. Keep all evaluation-season final and pre-bowl rank outcomes
out of posterior inference and game prediction. Corrected pre-bowl target PMFs
are secondary ranking measurements only, scored at checkpoints strictly
before their target dates, with all included evidence before the target date.
2022 is residual-training support for #195 and is not evaluated in this
experiment.

Before interpreting the candidate, reproduce #201's `Context_existing`
per-game metrics on the common retained cohort within `1e-8`. Verify game IDs,
teams, cutoffs, evidence hashes, and baseline rank PMFs. Fail visibly on a
mismatch or non-converged posterior. Retain #200 belief-propagation settings.
The cache identity includes the candidate parameters.

## Measures and fixed decision rule

The primary measure is paired future-game margin CRPS. Report pooled equal-game,
equal-season, and season-specific values. Generic recalibration clears the
practical-gain rule only when it improves CRPS by at least 0.05 points in both
aggregate summaries and has lower CRPS in at least two of the three seasons.
This is descriptive, not a significance test.

Report margin NLL, win Brier, median-based margin MAE, 50/80/95% interval
coverage and widths, phase counts, and gains/losses by game and team. For a
candidate that clears the primary rule, retain #193's early guardrails against
unmodified Context separately for preseason and pre-Thanksgiving forecasts:
pooled CRPS deterioration no greater than 0.10 points or 0.25 points in any
season, and mean absolute 50/80/95% coverage-error increase no greater than
0.03 pooled or 0.05 in any season.

Secondary ranking measures are corrected pre-bowl target NLL, rank CRPS,
expected-rank error, 80% target mass and interval width. Each team-season has
total weight one across valid checkpoints in the aggregate; preseason is
reported separately. Ranking gains do not substitute for future-game CRPS.
Repeated games, teams, and checkpoints are not independent confirmation.

Conclude **Generic recalibration merits independent confirmation** only when
the primary practical-gain rule and both early guardrails pass. Otherwise
conclude **Retain unmodified Context / no useful gain established**. Quantify
ranking and calibration tradeoffs. Any promising result needs fresh future
confirmation with predictions frozen before outcomes become available. No
result authorizes production promotion, further feature searches, or a
Context-aware production likelihood.

## Reproduction

Run from the repository root with the preserved issue inputs and the raw game
cache used by #201:

```sh
uv run python scripts/run_issue_212_likelihood_recalibration.py \
  --issue195-root data/research/issue_195_likelihood_diagnostic \
  --issue200-root data/research/matched_history_context_issue_200 \
  --issue200-manifest <manifest-matching-the-#195-input-hash> \
  --issue201-root data/research/issue_201_prebowl_forecast \
  --issue204-final-root <corrected-final-corpus-root> \
  --issue204-root data/processed/massey_prebowl_issue_204_v1 \
  --game-root <preserved-cfbd-game-cache> \
  --output-root data/research/issue_212_generic_likelihood_recalibration
```

The runner records input, fit, source, and output hashes; full paired scores;
per-game evidence IDs and row hashes; convergence diagnostics; reconstructible
posterior and target PMFs; rank scores; and a concise report. Raw downloads are
read-only inputs. The supplied #200 manifest must match the exact manifest hash
recorded by #195; the runner stops with an incomplete input-provenance error
when only a different manifest is available.
