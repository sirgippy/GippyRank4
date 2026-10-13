# Issue 193: evidence-dependent Context integration

## Status and scope

This is an exploratory development experiment. The 2022–2025 seasons and the
pilot work have already informed this research sequence; the outer-season
results are not untouched confirmation. No result in this experiment changes
production defaults or authorizes a rollout.

The primary artifact policy is **existing** from the matched #200 bundle:
History 1.1 and repaired Context 1.4 priors trained for corrected final-rank
outputs, paired with the fixed existing-policy likelihood. Use the same
#200 priors, likelihood, FCS policy, inference configuration, rank population,
and #201 schedule/evidence rules in every arm. The corrected pre-bowl target
PMFs are the common correction-training target and secondary ranking target;
they do not replace the existing-policy prior or likelihood artifacts.

The four paths are:

1. `History_existing`, unchanged matched History.
2. `Context_existing`, unchanged matched Context.
3. `History_calibrated`, one History calibration model using the fixed evidence
   summaries below.
4. `History_calibrated_plus_Context`, that same calibration design with one
   evidence-dependent use of the existing Context prior signal.

Path 4 versus path 2 is the operational comparison. Path 4 versus path 3 tests
whether Context adds information beyond generic evidence-dependent History
calibration. Path 3 versus path 2 measures calibration alone. A rank-NLL gain
does not establish a better game forecast.

## Fixed model

For a team `i` at a cutoff, `qH_i(r)` is the matched History posterior,
`pH_i(r)` and `pC_i(r)` are the matched existing-policy History and Context
priors, and `N_i` is the rank population. The Context residual is

```text
D_i(r) = log(pC_i(r)) - log(pH_i(r)).
```

The only evidence features are:

- `e1_i = log1p(n_i)`, where `n_i` counts distinct included completed game IDs
  under the #201 48-hour evidence rule;
- `e2_i = KL(qH_i || pH_i)` with no smoothing and the shared-support policy
  below.

Standardize both features using weighted training-row means and population
standard deviations from the current fit only; use `1e-6` as the scale floor.
Do not use evaluation-season statistics. Write standardized features as
`x1_i`, `x2_i`, and let `z(r) = logit((r - 0.5) / N_i)`. The common History
calibration control is

```text
log u3_i(r) = log(qH_i(r))
               + a * centered(log(qH_i))(r)
               + z(r) * (b0 + b1*x1_i + b2*x2_i).
```

The candidate adds exactly one residual term:

```text
log u4_i(r) = log u3_i(r)
               + (w0 + w1*x1_i + w2*x2_i) * D_i(r).
```

Normalize each path over its supported ranks with stable log-softmax. The
Context weight and both slopes are fitted freely and may be negative; do not
force decay, clip, or otherwise constrain them after viewing results. This is
an empirical conditional distribution model, not an exact generative posterior
and not a weight restricted to `[0, 1]`.

Require Context and History priors to have identical positive support. The
History posterior may remove prior support but may not create it. Define `D`
only on shared positive prior support, and keep each prediction on positive
History-posterior support. If a target has positive mass outside prediction
support, fail the run; do not add epsilon, smooth, pad, or renormalize onto a
different support. `KL(qH || pH)` sums only positive `qH` entries and fails if
any such entry has zero `pH` support.

Use the #192 residual-probe optimizer as the starting implementation: mean
weighted target-PMF cross-entropy plus `0.5 * lambda * ||theta||^2`, including
the intercept; L-BFGS-B from zero with analytic gradients, `maxiter=2000`,
`ftol=1e-13`, and `gtol=1e-8`. Keep the #192 `a` bound `[-0.75, 3]`; all
other coefficients, including `w0`, `w1`, and `w2`, are unbounded. Verify the
analytic gradient numerically and require successful optimizer convergence.

Choose lambda separately for paths 3 and 4 from the fixed #192 grid
`[0.01, 0.1, 1.0]`. With one training season use `0.1`. Otherwise use nested
forward-chained validation over earlier training seasons only; score the
validation season by equal team-season-weighted target NLL, and break ties
within `1e-12` toward the larger lambda. Do not change the grid, fitting loss,
features, or candidate family after game scores are seen.

In every fit, a team-season's total training weight is one across all of its
valid ranking-training checkpoints: each checkpoint row receives the inverse
of that team's row count. Apply this weighting to preprocessing, fitting, and
inner validation. Repeated checkpoints are not independent observations.

## Learning and prospective evaluation

Use 2022 as the earliest correction-training season. For the outer evaluation
of 2023, 2024, and 2025, fit only on strictly earlier seasons. The 2022–2025
training and evaluation targets are corrected pre-bowl PMFs from #204. Use
#201's frozen weekly ranking-training checkpoints and retain only cutoffs
strictly before each season's pre-bowl target date; verify that each ranking
row's included evidence also precedes that target date. Preserve #201's phase
mapping. The first zero-game checkpoint is reported as preseason.

For game evaluation, use #201's fixed weekly prospective cutoffs, common
scoreable game population, phase labels, evidence IDs, and 48-hour safety lag.
At each held-out cutoff, obtain the History posterior once and apply the fitted
correction once. Do not feed corrected PMFs back through games. The forecast
game must be absent from its evidence. Apply each held-out path's rank PMFs
before constructing the full pairwise rank-mixture margin forecast with the
fixed existing-policy likelihood. Score the primary regular-season FBS–FBS
population with continuous-margin CRPS. Keep preseason-only forecasts
separate. Report FBS–FCS separately; do not mix it into the primary result.

Retain #201's margin NLL, 50/80/95% interval coverage and widths,
median-based margin MAE, and home-win Brier as supporting game measures. For
secondary ranking results, report NLL, rank CRPS, expected-rank error,
80% target mass and width by season and checkpoint, plus preseason and the
equal-checkpoint mean over valid checkpoints with games. Report pooled
team-season-weighted results, native cold-start/fallback and repaired-transfer
coverage strata, fitted `w(e)` against the evidence distribution, and where
team/game gains and losses concentrate. Persist full per-team rank PMFs or
losslessly reconstructible prediction records, per-game scores, coefficient
selection/fit traces, evidence IDs/hashes, and source/output hashes.

Before interpreting candidate results, reproduce #201's retained baseline
per-game scores for paths 1 and 2 on the same evaluation games/cutoffs within
`1e-8` for CRPS, margin NLL, median absolute error, win Brier, and interval
bounds/coverage. Fail on any population, evidence, team, cutoff, or baseline
parity mismatch.

## Guardrails and decision rule

Game CRPS is the primary result and lower is better. Compare the three outer
evaluation seasons with equal weight by season as well as pooled equally by
game. A practically useful aggregate improvement is at least `0.05` margin
points in both pooled summaries and a lower mean in at least two of the three
evaluation seasons. These are descriptive stability thresholds, not
significance tests.

Preserve early Context value with both of these preregistered guardrails for
path 4 versus path 2, evaluated separately for preseason and pre-Thanksgiving:

- pooled CRPS degradation is at most `0.10` margin points, and no season's
average degrades by more than `0.25` points;
- the increase in mean absolute deviation of 50/80/95% empirical interval
  coverage from nominal is at most `0.03` pooled and `0.05` in each season.

Report all phase counts and values even if a phase misses its guardrail. Do not
choose a gate from the conference-championship or Army–Navy slice.

Conclude with exactly one outcome:

- **Refine for independent confirmation** when path 4 meets the useful-gain
  threshold against both paths 2 and 3 and passes both early guardrails. State
  the ranking-NLL and game-calibration tradeoffs.
- **Prefer calibration-only for further confirmation** when path 3 meets the
  useful-gain threshold against path 2 and passes the same early guardrails,
  while path 4 does not meet the useful-gain threshold against path 3.
- **Retain unmodified Context / inconclusive** otherwise, including when the
  evidence or baseline parity gate fails or the evaluation cannot distinguish
  the proposed replacements.

Do not make an independence or significance claim from repeated team,
checkpoint, or game rows. Stop after this single candidate; do not search
features, gates, thresholds, or production settings.

## Reproduction inputs and outputs

The CLI takes explicit roots for the #200 bundle, #201 frozen comparison,
#204 corrected target corpus, and raw CFBD game cache. It validates the
#201 schedule-plan and source manifests, #204 readiness/hash manifest, #200
artifact hashes, protocol hash, and baseline score parity before writing a
complete results manifest. Output paths are portable labels plus byte counts
and SHA-256 values; machine-specific absolute paths are not part of the
results manifest.
