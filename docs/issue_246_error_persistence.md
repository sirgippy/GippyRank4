# Issue 246: persistence of sequential pregame forecast errors

## Frozen analysis protocol

This analysis reuses issue 201's retained leakage-safe rolling game forecasts for 2022–2025 and issue 200's retained preseason team-rank PMFs. Those seasons have already informed project research, so the results are exploratory historical evidence, not independent confirmation. Do not refit models, alter parameters or likelihoods, or rebuild the forecasting pipeline.

### Question

For a team's consecutive regular-season games, does the signed error in its earlier pregame margin forecast predict the signed error in its next game, after the earlier result is available to that next forecast?

Keep the four issue 201 arms separate: History and Context, each under the existing and pre-bowl policies. Policy arms are not pooled as independent observations.

### Inputs and eligibility

Read the retained issue 201 forecast scores, forecast summary, replay evidence, and manifest. Verify the recorded byte counts and SHA-256 hashes for the three output artifacts. Reproduce issue 201's rolling FBS–FBS per-season game counts and mean CRPS and margin NLL from the score rows against the summary before deriving errors.

Read the issue 200 manifest and pipeline bundle. Verify the bundle's recorded size and SHA-256, then verify each selected preseason rank PMF against its stored PMF hash. Use only FBS priors for the 2022–2025 target seasons and the matching model family and policy. Check that each PMF has the expected rank support, contains valid probabilities, and sums to one. Do not run the issue 200 fitting pipeline.

Use rolling regular-season FBS–FBS and FBS–FCS issue 201 forecasts. Build complete sequences for FBS focal teams: retain both team orientations for FBS–FBS matchups and only the FBS team's orientation for FBS–FCS matchups. The latter preserves the FBS team's schedule between two FBS games. Do not construct FCS-only histories because issue 201 does not retain FCS–FCS forecasts.

Sort each focal FBS team's games by retained UTC kickoff. For each adjacent pair, use the earlier and later forecasts from the same season and model arm. A pair is eligible only if the earlier game ID appears in the later forecast's exact issue 201 evidence list. Also verify that each forecast cutoff is before its own kickoff and its own game ID is absent from its evidence. Keep excluded adjacent pairs in the output with an explicit reason.

### Preseason rank-prior classification

Classify every issue 200 FBS team-season independently for each model arm using that team's preseason rank PMF:

- Compute P(top 10) by summing the PMF mass on ranks 1 through 10.
- Label the team-season a strong preseason rank prior when P(top 10) is at least 0.50; otherwise label it other.
- Assign this one team-season label to every game and transition for that team in the corresponding arm.

The classification is independent of opponents and game-specific win probabilities. Save the auditable team-season classifications, including P(top 10) and the source PMF hash.

### Error and persistence measures

Orient the retained expected margin and observed margin from the focal team's perspective. Define signed error as observed team margin minus expected team margin. Positive means the team performed better than the forecast expected; negative means worse. Use issue 201's predictive expected margin, not a collapsed ranking or win/loss outcome. Opponent strength and site are already represented in the retained matchup forecast.

Report Pearson lag-1 correlation between consecutive eligible signed errors. Report directional persistence as the share of pairs where both errors are nonzero and have the same sign. Do not assume its independence baseline is 0.50. For nonzero pairs, calculate the baseline from the observed previous- and next-error positive-sign shares:

    p(previous positive) * p(next positive)
      + p(previous negative) * p(next negative)

Report the two positive-sign shares, the observed same-direction rate, this independence baseline, and their difference. Bootstrap the difference from zero, recomputing both sign marginals and the baseline within every replicate. Exclude pairs with a zero error from this directional calculation but retain them for lag correlation and transition counts.

Report both measures overall and by season, prior family/policy, preseason rank-prior stratum, and season progress. Define progress by the destination team-game number: early games 2–4, middle games 5–8, and late games 9+.

### Dependence and uncertainty

Transitions reuse team-games. FBS–FBS matchups also appear from both teams' perspectives, with opposite error orientation. Do not treat transition rows as independent replicates. Use 95% percentile intervals from a deterministic multiway pigeonhole bootstrap that resamples focal team IDs and game IDs. Share each game-ID resample across previous and next positions to preserve both orientations of a matchup. Recompute the sign-marginal independence baseline for every replicate. Bootstrap the preseason-stratum and late-minus-early differences using the same team and game draws.

These intervals describe variation in this retained historical panel; they do not convert four previously studied seasons into independent confirmation. Report sample counts and season-level stability alongside pooled results.

### Interpretation

A positive residual association would indicate directional errors remain in sequential pregame forecasts after the earlier result was available to the later forecast. It does not identify a cause, prove likelihood weakness, or establish a prior-likelihood imbalance. An interval that includes zero is a failure to detect an effect at this resolution, not evidence that no meaningful effect exists; these are not equivalence tests.

Defer model changes whose rationale is sequential error persistence based on this exploratory retrospective analysis. This does not resolve the broader calibration epic or establish that prior-likelihood balance is correct; other calibration questions should be assessed using their own evidence. Do not tune parameters or add predictors here.

This differs from issue 34's residual-persistence study, which used historical ranking observations rather than sequential pregame forecast errors.

### Reproduction

From the repository root, run:

    uv run python scripts/run_issue_246_error_persistence.py

The deterministic outputs under data/research/issue_246_error_persistence include team-season prior classifications, per-team-game errors, all adjacent pairs and exclusions, aggregate summaries, contrasts, a report, and input/output hash manifests.
