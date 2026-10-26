# Issue 253 posterior movement protocol

## Question and frozen inputs

Characterize how the retained History 1.1 and repaired Context 1.4 posterior PMFs move across the seven already-defined checkpoints in seasons 2022–2025. Use only FBS team-seasons, preserve the retained native cold starts, and keep the two model families separate. Do not rebuild inference or select checkpoints after inspecting outcomes.

Inputs are the hash-verified `inference_panel.jsonl.gz` and `evidence.json` from #192, and the hash-verified `team_game_errors.csv` from #247. The #247 rows supply only game-level standardized surprise values; the #192 PMFs supply every prior and posterior summary. For each #192 checkpoint, include a prior game’s #247 surprise only when its exact game ID is in that checkpoint’s #192 included-game set and its kickoff is strictly before that checkpoint cutoff. Validate that every FBS team’s joined game count equals #192 `games_played`, and that every joined game has a pregame forecast cutoff before kickoff. Record the total evidence-game count and focal-team game count separately.

Check #201’s and #247’s recorded hashes, model-arm labels, policies, and links to the frozen #192 panel/evidence before joining. Do not use the #247 accumulated-surprise values computed at #201’s different forecast cutoffs; recompute the same unweighted mean from its game-level surprises under each #192 cutoff. Apply #247’s ±0.5 group thresholds only after at least three joined prior games. Keep its existing-policy History and Context forecast families distinct from the #192 posterior inference configurations. If priors or likelihoods differ, record the differences and never substitute #247 prior summaries or forecasts into the #192 posterior movement calculations.

## Distribution summaries

For each family, team, season, and checkpoint, retain the complete #192 prior and posterior PMFs. Validate finite nonnegative probabilities, unit mass, and identical prior/posterior support. Treat rank support as `1..N`, where `N` is the retained checkpoint `rank_count`.

Report expected rank, rank standard deviation, and equal-tail central 80% interval bounds and width. For comparisons across rank populations, normalize expected-rank changes, standard deviations, and interval widths by `N - 1`. Define positive location movement as improvement: preseason expected rank minus posterior expected rank; checkpoint movement is the previous checkpoint’s posterior expected rank minus the current posterior expected rank. Report posterior rank SD and interval-width changes relative to preseason and the preceding checkpoint. Compute `KL(posterior || prior)` from full PMFs without smoothing, and total variation `0.5 * sum(abs(current_posterior - previous_posterior))` between adjacent checkpoint PMFs. The first checkpoint has no preceding-checkpoint change or TV value.

Preseason strength is `1 - (expected_rank - 1) / (N - 1)` and preseason uncertainty is rank SD divided by `N - 1`. Preserve raw expected ranks, raw dispersion, rank population, and support alongside normalized summaries. The evidence counts are descriptive; a posterior can change through games between other teams because inference is joint.

## Surprise comparisons

Retain every team-checkpoint movement row, including cold starts and checkpoints with fewer than three eligible prior games. For surprise analyses, require at least three joined games and checkpoint 2 or later. Compare the frozen #247 groups (mean standardized surprise ≤ -0.5, between -0.5 and +0.5, and ≥ +0.5) descriptively.

The primary directional outcome is normalized expected-rank improvement since the previous checkpoint. The primary magnitude outcome is its absolute value. The primary uncertainty outcome is normalized rank-SD change since the previous checkpoint. Also summarize cumulative movement from preseason and central-80%-width changes by surprise group. Fit separate descriptive linear models for History and Context with season and checkpoint fixed effects, continuous preseason strength and uncertainty, prior-game count, and a surprise-by-preseason-strength interaction for directional location and uncertainty. Fit absolute location movement against absolute accumulated surprise with the same controls. Compare families with paired team-checkpoint differences in movement, uncertainty, KL, and adjacent TV. Do not compare the family-specific surprise slopes: their #247 surprise predictors come from different forecast models and do not represent identical evidence under #192. Treat #247 surprises as external descriptive proxies; associations cannot establish causal responsiveness or precisely quantify how either posterior responds to the same evidence. Report two-way pigeonhole bootstrap percentile intervals, resampling team IDs and season-checkpoint snapshots independently, using a fixed seed. Repeated checkpoints are not independent. Because inference couples teams through shared game evidence, the intervals remain approximate and exploratory.

Summarize the full distribution of movement and uncertainty by season, checkpoint, family, and component status, with mean, median, and 10th/25th/75th/90th percentiles. Use two-way clustered bootstrap intervals for pooled surprise-group contrasts. Do not treat four historical seasons as independent confirmation or interpret associations causally.

## Trajectories and stopping point

Select two team-seasons using only checkpoint-7 accumulated surprise: the minimum and maximum of the mean History/Context surprise among cases with at least three joined games in both families. Use both families’ full seven-checkpoint PMFs for those two cases. Do not use posterior movement, eventual ranks, or future outcomes to select them.

Do not use 2026 data. Do not alter priors, likelihoods, inference, or production behavior. The report will recommend a bounded next Phase 1 question based on whether posterior movement is directionally related to accumulated surprise, while leaving evaluation of conditional predictive calibration as separate research.

## Reproduction

Run from the repository root:

```bash
uv run python scripts/run_issue_253_posterior_movement.py
```

The runner validates retained inputs, writes reproducible team-checkpoint summaries and full PMFs, aggregate tables, selected trajectories, the report, and an input/output hash manifest under `data/research/issue_253_posterior_movement/`.
