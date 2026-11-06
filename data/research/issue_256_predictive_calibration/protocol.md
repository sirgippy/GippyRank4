# Issue 256 frozen calibration protocol

## Question and lineage

Evaluate whether the retained issue 201 predictive distributions remain calibrated across 2022–2025 and conditional on season progression, preseason beliefs, and accumulated surprise. Evaluate only `History_existing` and `Context_existing`. This study evaluates the issue 201 game-margin forecast distributions; it does not validate the issue 253 rank-PMF uncertainty trajectories, whose retained models use a different configuration.

All results are exploratory historical evidence. No refits, new data acquisition, 2026 outcomes, model changes, or parameter searches are allowed.

## Frozen forecast populations

Use the hash-verified `issue_201_prebowl_forecast/forecast_scores.csv` and its retained score and interval fields. The primary population is one rolling forecast per physical FBS–FBS regular-season game, model family, and issue 201 cutoff. Keep the existing-policy History and Context arms separate. Use the full issue 201 regular-season forecast population, including its separately labeled late-season phases. Report FBS–FCS rolling forecasts separately, with the FBS team's orientation. Keep issue 201 preseason-only forecasts in a separately labeled reference; never combine them with rolling forecasts.

For each selected forecast, recalculate central 50%, 80%, and 95% interval coverage from the retained bounds and actual margin. Require exact indicator parity with issue 201's stored coverage fields. Retain the issue 201 mean margin CRPS and margin NLL without rebuilding predictive mixtures. Report nominal and observed coverage, coverage error in percentage points, upper- and lower-tail misses for the 80% interval, mean and median interval widths, CRPS, NLL, and counts.

## Team-oriented conditional observations

Join issue 247's retained observations to issue 201 by season, game ID, forecast ID, model arm/family, and exact cutoff. For FBS–FBS games, keep both team orientations, reverse margin and predictive bounds for the away-team perspective, retain the shared game ID, and treat the two perspectives as dependent. For FBS–FCS games, keep the FBS team's perspective. Compute each team's prior-game count from issue 247's count of earlier games included in that forecast's actual evidence set. Use these frozen groups: early 0–2, middle 3–6, and late 7 or more prior games.

Use issue 247's retained continuous preseason strength (`preseason_rank_strength_percentile`) and uncertainty (`preseason_rank_uncertainty_fraction`). Report family-specific empirical strength quartiles using distinct FBS team-season characteristics pooled across 2022–2025, with explicit sample counts; derive the 25th, 50th, and 75th percentile cut points from preseason strength alone using linear sample quantiles before examining coverage, and use the quartiles only as descriptive summaries. Assign values equal to a cut point to the lower quartile. Analyze 80% interval misses with a prespecified linear-probability model including season fixed effects, progression groups, continuous strength and uncertainty, and their interactions with progression.

## Accumulated surprise

Reuse issue 247's unweighted mean of prior standardized margin surprises, computed only from earlier team games in the exact issue 201 evidence set available at the forecast cutoff. Keep its minimum of three eligible prior games and frozen group thresholds: sustained negative at or below −0.5, approximately neutral between the thresholds, and sustained positive at or above +0.5. Verify every join and evidence history; the target game's result must not appear in its own history.

For surprise-group summaries, retain all three groups, counts, 80% coverage and tail-miss rates, widths, CRPS, and NLL. For the continuous analysis, use only observations meeting the frozen three-game minimum. Fit a descriptive linear-probability model for 80% interval misses with season effects, prior-game count, continuous preseason strength and uncertainty, signed and absolute accumulated surprise, and a continuous interaction between preseason strength and negative-surprise magnitude. Do not search alternative thresholds, interactions, or specifications.

## Dependence and uncertainty

Use 1,000 deterministic percentile-bootstrap repetitions and seed 256. For game-level estimates, use two-way pigeonhole weights from game-ID resampling and the mean of the two endpoint-team resampling multiplicities, so each game contributes once. For team-oriented estimates and regressions, use issue 247's two-way pigeonhole resampling of focal team IDs and target game IDs. Use the same dependence structure for History–Context paired differences. Report 95% intervals for 80% coverage and coverage error; report intervals for paired differences and model coefficients. Equal-season summaries are unweighted means of season-specific estimates, with intervals formed by independent season-specific clustered bootstrap draws.

## Decision rule

Treat ±5 percentage points around nominal 80% coverage as the prespecified global practical band. A potential global deficiency requires an absolute pooled error of at least 5 points, an interval excluding zero, and the same error direction in at least three of four seasons. A potential localized deficiency requires at least 10 points of pooled error in a prespecified, adequately sized progression, strength, or surprise group, an interval excluding zero, and recurrence in both families or in at least three seasons. Surprise extremes with fewer than 100 unique games are too small to trigger a Phase 2 recommendation.

Recommend concluding Phase 1 only when the pooled 80% coverage intervals for both families lie within the global practical band, the 95% coverage-error intervals for adequately sized pooled conditional groups lie within ±10 points, and no prespecified subgroup meets the localized-deficiency rule. Recommend a bounded Phase 2 investigation only when a potential deficiency meets the rules above. If the intervals remain too wide to rule out these practical deviations without identifying a deficiency, classify the evidence as inconclusive. These screens do not turn a nonsignificant estimate into proof of equivalence; report effect sizes, interval widths, subgroup counts, season-level results, and limitations.
