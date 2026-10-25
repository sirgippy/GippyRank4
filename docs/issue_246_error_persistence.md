# Issue 246: persistence of sequential pregame forecast errors

## Frozen analysis protocol

This analysis reuses issue 201's retained leakage-safe rolling game forecasts for 2022–2025. Those seasons have already informed project research, so results are exploratory historical evidence, not independent confirmation. No models are refit, no parameters or likelihoods are changed, and no production outputs are touched.

### Question

For a team's consecutive regular-season games, does the signed error in the earlier pregame margin forecast predict the signed error in its next game, after the earlier result is available to that next forecast?

The four issue 201 arms are kept separate: History and Context, each under the existing and pre-bowl policies. Showing both policies checks that a result is not specific to one of issue 201's evidence policies. Policy arms are not pooled as independent observations.

### Inputs and eligible games

Read only these retained issue 201 outputs:

- `forecast_scores.csv`
- `forecast_summary.csv`
- `replay_evidence.jsonl.gz`
- `manifest.json`

Verify the recorded byte count and SHA-256 for the three output artifacts. Reproduce issue 201's rolling FBS–FBS per-season game counts, mean CRPS, and mean margin NLL from `forecast_scores.csv` against `forecast_summary.csv` before deriving errors.

Use rolling regular-season `fbs_fbs` and `fbs_fcs` forecasts. Build complete sequences for FBS focal teams: retain both team orientations for FBS–FBS matchups and only the FBS team's orientation for FBS–FCS matchups. The latter preserves the FBS schedule between two FBS games. Do not construct FCS-only histories because issue 201 does not retain FCS–FCS forecasts.

Sort each focal FBS team's games by the retained UTC kickoff. For each adjacent pair, use the earlier and later forecasts from the same season and model arm. A pair is eligible only if the earlier game ID appears in the later forecast's exact issue 201 evidence list. Also verify that each forecast cutoff is before its own kickoff and its own game ID is absent from its evidence. This preserves issue 201's 48-hour evidence safety lag and avoids treating a game as evidence for itself. Keep excluded adjacent pairs in the output with an explicit reason.

### Error and persistence measures

For a home forecast, orient the margin from the home team's view; for an away team, negate both actual and expected home margins. Define signed error as:

```text
observed team margin − expected team margin
```

Positive means the team performed better than the forecast expected; negative means worse. Use issue 201's predictive expected margin, not a collapsed ranking or win/loss outcome. Opponent strength and site are already represented by the retained matchup forecast.

Report:

- Pearson lag-1 correlation between earlier and next signed errors.
- Directional persistence: the share of nonzero adjacent error pairs with the same sign. The no-directional-persistence reference is 0.5.
- Both statistics by season, prior family, and policy arm.
- The same statistics for strong preseason favorites and other teams.
- Changes by destination team-game number: early games 2–4, middle games 5–8, and late games 9+.

A strong preseason favorite is a focal team with at least 0.80 win probability in that arm's preseason-only forecast for the destination game. This threshold is fixed before reading sequential error results. Transitions are assigned to a season-progress bin using the next game's ordinal for that team.

### Dependence and uncertainty

Transitions reuse team-games. FBS–FBS matchups also appear from both teams' perspectives, with opposite error orientation. Do not treat transition rows as independent replicates. Report 95% percentile intervals from a deterministic multiway pigeonhole bootstrap that resamples focal team IDs and game IDs; a single game-ID resample is shared across previous and next positions, preserving both orientations of a matchup. Bootstrap the favorite-stratum difference and late-minus-early difference on the same team/game draws.

These intervals describe variation in the retained historical panel; they do not convert four previously studied seasons into independent confirmation. Report sample counts and season-level stability alongside pooled results.

### Interpretation

A positive residual association means systematic directional errors remain in sequential pregame forecasts despite the earlier result entering the next posterior. It does not identify a cause, prove likelihood weakness, or justify a model change. A negative, weak, season-sensitive, or uncertain estimate is also a valid result.

Recommend a bounded posterior-responsiveness follow-up only if positive lag correlation and above-chance directional persistence recur across both prior families and both policies, with season-level support. Otherwise recommend deferring a model response or revising the analysis around the observed uncertainty. Do not tune parameters or add predictors here.

This differs from issue #34's residual-persistence study: issue 246 uses retained pregame game-margin forecasts and outcome-safe sequential evidence inclusion, rather than historical ranking observations or in-sample performance residuals.

### Reproduction

From the repository root:

```bash
uv run python scripts/run_issue_246_error_persistence.py
```

The deterministic output under `data/research/issue_246_error_persistence/` includes per-team-game errors, all adjacent pairs and exclusions, aggregate summaries, contrasts, a report, and input/output hash manifests.
