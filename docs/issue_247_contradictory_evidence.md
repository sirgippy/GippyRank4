# Issue 247: accumulated contradictory evidence and subsequent forecast errors

## Frozen analysis protocol

This study reuses issue 201's leakage-safe rolling game forecasts and issue 200's retained preseason rank distributions for 2022–2025. The seasons are exploratory historical data, not independent confirmation. Do not refit models, change inference, tune thresholds, or use 2026 results to select definitions.

### Question and scope

For each FBS team's future-game forecast, does the team's accumulated in-season performance relative to prior pregame expectations predict the signed error in that future game, after the forecast has incorporated those prior results? Keep History existing and Context existing separate. Do not include pre-bowl policy arms.

Use regular-season FBS–FBS and FBS–FCS forecasts, retaining both team perspectives for FBS–FBS games and only the FBS perspective for FBS–FCS games. This preserves each focal FBS team's schedule.

### Inputs and temporal integrity

Reuse issue 246's input readers and team-game orientation helpers. They verify issue 201's retained score/evidence hashes, score-summary parity, cutoff safety, evidence digests, and issue 200's bundle and preseason PMF hashes. Use the retained artifacts only; do not rebuild the forecasting pipeline or run the issue 200 fitting pipeline.

For every future team-game, form its history from earlier same-season games for that same FBS team. A prior game is eligible only when its game ID appears in that future forecast's exact issue 201 evidence set. Verify each included prior kickoff is before that forecast's cutoff. Do not include a forecast's own game. Keep the number of prior schedule games, number of evidence-eligible prior games, target team-game number, and season in the observation artifact. The primary analysis requires at least three evidence-eligible prior games.

### Frozen surprise summary and outcomes

For each oriented prior team-game, define signed margin error as observed margin minus the issue 201 expected margin from that team's perspective. Define the matchup's predictive uncertainty as half the width of its central 80% predictive margin interval. Its standardized surprise is signed margin error divided by that half-width. This uses the retained forecast's uncertainty without assuming a Normal predictive distribution.

The primary accumulated-surprise summary is the unweighted arithmetic mean of all evidence-eligible prior standardized surprises. Do not search over windows, decay factors, or alternative weighting schemes. Use the following descriptive groups: sustained negative when the mean is at most -0.5; sustained positive when it is at least +0.5; approximately neutral otherwise. These fixed thresholds are in 80% interval half-width units and are set before inspecting future errors.

The primary outcome is the next game's signed margin prediction error in points. Also report that next error in 80% interval half-width units. The target is the future forecast after all eligible past results have entered its evidence set; a relationship describes conditional predictive bias, not a lag-1 correlation and not a cause.

### Preseason expectations

From each matching issue 200 FBS preseason rank PMF, compute expected rank and rank standard deviation. Express strength as the reverse expected-rank percentile, `1 - (E[rank] - 1) / (population - 1)`, so larger values mean stronger preseason expectations. Express uncertainty as rank standard deviation divided by `population - 1`. Retain P(top 10), PMF hash, and these continuous summaries with every team-season. Do not rely only on a small strong-prior subgroup.

### Estimation and dependence

For each prior family separately, estimate the pooled association between accumulated surprise and the next signed error with OLS. Include continuous preseason strength, normalized preseason uncertainty, surprise-by-strength interaction, prior-game count, target team-game number, and season indicators. Report the surprise slope at mean preseason strength, the interaction, and model-implied error shifts comparing surprise -1 with 0 at the 25th and 75th percentiles of preseason strength. Fit the same adjusted surprise slope separately by season to show stability. Repeat these estimands for the standardized future error as a secondary scale check.

Report mean next errors in points and standardized units for the three frozen surprise groups, plus negative-versus-neutral and positive-versus-neutral differences. Show sample counts by season and model family.

Use 1,000 deterministic multiway pigeonhole bootstrap repetitions for 95% percentile intervals, resampling focal team IDs and target game IDs. Team resampling keeps repeated observations within a team's season history dependent; shared target game IDs keep the opposite orientations of an FBS–FBS matchup linked. Use the same bootstrap draws across coefficients and group contrasts. These intervals summarize this retrospective panel and do not make its seasons independent.

Select examples without using their next-game errors: for each prior family, among observations with at least five eligible prior games, show the most negative and most positive accumulated-surprise observations, with deterministic tie-breaking. This yields at most four examples.

### Interpretation and stopping point

Report whether accumulated surprise is associated with directional future signed error, its magnitude, uncertainty, season stability, and any modification by preseason expectations. Distinguish a detectable association from evidence about its cause. Do not modify models or production behavior. Recommend one next research question under #244 based on what these results establish, then stop.

### Reproduction

Run `uv run python scripts/run_issue_247_contradictory_evidence.py` from the repository root. The runner writes the team-game errors, all team-game history observations (including those below the minimum history), group summaries, pooled and season-specific model estimates, selected examples, a report, a JSON summary, and input/output hash manifests under `data/research/issue_247_contradictory_evidence/`.
