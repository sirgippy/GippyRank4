# Issue 140: Does History 1.1 still add value after Context 1.3?

## Recommendation

**A — keep History 1.1 as a first-class production predictive lineage.** The
rolling-origin Context 1.3 comparison supports this more directly than the
frozen-through-2021 comparison alone. Context has lower preseason NLL in all
four seasons. History first has lower posterior NLL at the **September 17, 2023,
September 19, 2024, and September 19, 2025** checkpoints, and keeps that lead at
every observed October, November, and postseason checkpoint in those years.
2022 is the exception: Context remains slightly ahead. The four-season mean
also changes sign at the September checkpoint, from a rolling Context advantage
of 0.034 nats per team in August to a History advantage of 0.013 in September.
This is evidence for useful in-season publication, not merely a postseason
comparison. No publication or model behavior changes in this study.

The rolling comparison improves Context's final pooled posterior NLL by 0.004
relative to the frozen comparison, but History still leads by 0.054. That lead
is 0.079, 0.072, and 0.070 nats per team in 2023–2025, with better target-mass
80% coverage in each. History also has lower final pooled CRPS and expected-rank
error. The high agreement in ranking order means its value lies mainly in the
probability distributions and calibration, rather than a wholly different
ordering.

Context 1.2 needs **retained-artifact reading and reconstruction support only**
for its 2026 preseason and early weekly publications. This compatibility
question is separate from whether History 1.1 merits active publication.

## Method and provenance

We replay seven actual-date checkpoints in each of 2022–2025, using the
retained rolling posterior backtest dates. The matched population is all FBS
teams with a frozen final Massey constituent-rank target PMF: 131, 133, 134,
and 136 teams. Each team's target distribution is reused at every checkpoint.
NLL is target-weighted cross entropy; 80% coverage is target probability mass
inside the forecast interval, not a binary hit against an assumed single true
rank. CRPS and expected-rank absolute error provide complementary scores.

The historical arms are:

- **Frozen Context 1.3:** validated P3 research PMFs from
  `data/processed/preseason/context_v1_3_candidate/predictions.csv`, fitted
  through 2021 and applied to every target season. This leakage-safe
  architecture comparison remains in the results.
- **Rolling-origin Context 1.3:** the same validated P3 specification refitted
  through 2021 for 2022, through 2022 for 2023, through 2023 for 2024, and
  through 2024 for 2025. The study calls the existing Context 1.3 fit and
  prediction functions, checks each season's prior scores against the retained
  validated `rolling_metrics.csv` at 1e-8 tolerance, and uses the resulting
  team PMFs in the posterior replay. It does not refit on a target season's
  outcomes.
- **History 1.1:** retained historical PMFs from
  `data/processed/preseason/history/predictions.csv`. The same History
  posterior is reused as the matched comparator for both Context arms.

Every checkpoint uses the pinned Historical Likelihood V1 (SHA-256
`89eff21a304939a62beed6fd58dd642e0e4dbba698d7bdaa9aaca9f09f2de153`),
identical included games, cutoff, FCS population and fallback rule, and FBS
team IDs. Inference uses the current production snapshot configuration: **500
maximum iterations, 1e-9 tolerance, 0.35 damping**. All 84 historical runs
converged, requiring at most 135 iterations. The independent History replay
stays within 0.00155 NLL of the older retained backtest at every checkpoint;
that diagnostic is recorded in
[`historical_evidence.csv`](../data/processed/history_context_posterior_study/historical_evidence.csv).

The processed historical CFBD game CSV is not checked into this worktree. The
study reconstructs it from cached raw FBS/FCS schedule responses and rejects
conflicting duplicate game IDs. The final-rank target corpus is another local
input. Input hashes, including the Context feature panel and coaching sources
used for rolling fits, are in
[`provenance.json`](../data/processed/history_context_posterior_study/provenance.json).
Historical Context transfer values remain **retrospective research
reconstructions**, not archived August 15 inputs. Rolling-origin fitting
addresses model updating as modern seasons enter training; it does not remove
this data-timing limitation. The 28 checkpoints and 7,476 arm-specific
team-checkpoint rows repeatedly score 534 team-seasons, so they are not 7,476
independent observations.

## Calendar crossover and matched checkpoints

Positive Δ means History has lower NLL; Δ = Context minus History, in nats per
team. Dates are UTC. The first History-leading checkpoint is September 17 in
2023 and September 19 in 2024 and 2025 **under both Context arms**. History
remains ahead at all later observed dates in those three seasons. In 2022,
Context leads at every checkpoint, though its edge narrows to 0.004 by
December 10. The four-season aggregate first turns History-leading at
checkpoint 2, which falls on September 17–22 by season.

| Season | Depth | Date | H NLL | Frozen C NLL | Frozen Δ | Rolling C NLL | Rolling Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| 2022 | 1 | Aug 27 | 4.519 | 4.448 | -0.071 | 4.448 | -0.071 |
| 2022 | 2 | Sep 22 | 4.311 | 4.282 | -0.029 | 4.282 | -0.029 |
| 2022 | 3 | Oct 07 | 4.183 | 4.158 | -0.026 | 4.158 | -0.026 |
| 2022 | 4 | Oct 27 | 4.078 | 4.068 | -0.010 | 4.068 | -0.010 |
| 2022 | 5 | Nov 09 | 3.994 | 3.980 | -0.014 | 3.980 | -0.014 |
| 2022 | 6 | Nov 23 | 3.861 | 3.852 | -0.009 | 3.852 | -0.009 |
| 2022 | 7 | Dec 10 | 3.801 | 3.797 | -0.004 | 3.797 | -0.004 |
| 2023 | 1 | Aug 26 | 4.492 | 4.474 | -0.018 | 4.476 | -0.017 |
| 2023 | 2 | Sep 17 | 4.306 | 4.337 | +0.032 | 4.339 | +0.034 |
| 2023 | 3 | Oct 06 | 4.244 | 4.274 | +0.030 | 4.277 | +0.033 |
| 2023 | 4 | Oct 21 | 4.049 | 4.077 | +0.028 | 4.081 | +0.032 |
| 2023 | 5 | Nov 04 | 3.945 | 4.005 | +0.060 | 4.010 | +0.065 |
| 2023 | 6 | Nov 18 | 3.873 | 3.929 | +0.057 | 3.934 | +0.062 |
| 2023 | 7 | Dec 09 | 3.836 | 3.908 | +0.072 | 3.915 | +0.079 |
| 2024 | 1 | Aug 24 | 4.585 | 4.548 | -0.036 | 4.553 | -0.032 |
| 2024 | 2 | Sep 19 | 4.375 | 4.410 | +0.035 | 4.403 | +0.028 |
| 2024 | 3 | Oct 06 | 4.193 | 4.232 | +0.039 | 4.225 | +0.033 |
| 2024 | 4 | Oct 24 | 4.113 | 4.171 | +0.058 | 4.164 | +0.051 |
| 2024 | 5 | Nov 08 | 4.037 | 4.116 | +0.079 | 4.106 | +0.069 |
| 2024 | 6 | Nov 23 | 3.897 | 3.981 | +0.083 | 3.974 | +0.077 |
| 2024 | 7 | Dec 14 | 3.798 | 3.876 | +0.078 | 3.870 | +0.072 |
| 2025 | 1 | Aug 23 | 4.564 | 4.547 | -0.017 | 4.545 | -0.019 |
| 2025 | 2 | Sep 19 | 4.340 | 4.364 | +0.024 | 4.358 | +0.018 |
| 2025 | 3 | Oct 08 | 4.195 | 4.247 | +0.052 | 4.238 | +0.043 |
| 2025 | 4 | Oct 24 | 4.067 | 4.117 | +0.050 | 4.109 | +0.042 |
| 2025 | 5 | Nov 09 | 3.967 | 4.037 | +0.070 | 4.027 | +0.060 |
| 2025 | 6 | Nov 23 | 3.947 | 4.022 | +0.075 | 4.008 | +0.061 |
| 2025 | 7 | Dec 13 | 3.860 | 3.945 | +0.085 | 3.929 | +0.070 |

The full prior, posterior, CRPS, error, coverage, width, and ranking agreement
metrics by season, checkpoint, Context arm, and transfer group are in
[`historical_checkpoints.csv`](../data/processed/history_context_posterior_study/historical_checkpoints.csv).
The team-level paired observations are in
[`historical_teams.csv`](../data/processed/history_context_posterior_study/historical_teams.csv).

## September, October, November, and postseason behavior

These are team-weighted means across the four seasons; October and November
each contain two checkpoints per season. August is the first checkpoint, before
most game evidence. Coverage is target mass in the forecast's 80% interval.

| Period | Frozen Δ NLL | Rolling Δ NLL | Rolling C / H coverage | Rolling C / H width |
|---|---:|---:|---:|---:|
| August | -0.035 | -0.034 | .833 / .839 | 70.1 / 76.4 |
| September | +0.016 | +0.013 | .777 / .793 | 49.2 / 51.7 |
| October | +0.028 | +0.025 | .766 / .774 | 38.5 / 39.9 |
| November | +0.051 | +0.047 | .771 / .788 | 31.4 / 32.3 |
| Postseason (December) | +0.058 | +0.054 | .778 / .802 | 28.6 / 29.4 |

The History advantage begins in September in 2023–2025, persists in October,
and grows in November and December. Rolling Context narrows the pooled gap by
roughly 0.003–0.004 NLL from September onward; it does not move the observed
crossover or erase the repeated coverage difference.

## Prior improvement, calibration, and uncertainty

At the final checkpoint, the team-weighted rolling Context prior/posterior NLL
is 4.507/3.878, an improvement of 0.629. History's is 4.544/3.824, an
improvement of 0.720. Frozen Context improves from 4.506 to 3.882, or 0.624.
Both priors therefore benefit substantially from the same game likelihood.

The rolling Context final 80% interval narrows from 70.7 to 28.6 ranks;
History's narrows from 77.0 to 29.4. History is slightly wider yet has more
target mass inside its interval, 0.802 versus 0.778 pooled. In 2023, 2024,
and 2025, rolling Context final coverage is 0.752, 0.794, and 0.762, versus
History's 0.794, 0.821, and 0.793. The 2022 exception is 0.805 for Context
versus 0.801 for History. Final pooled CRPS is 0.0223 for rolling Context
versus 0.0194 for History; expected-rank absolute error is 8.23 versus 7.66.

The team-level final rolling Δ NLL median is +0.017, with 10th/90th
percentiles -0.132/+0.304; History has lower NLL for 293 of 534 team-seasons.
The mean advantage therefore includes some large gains and substantial
heterogeneity. Expected-rank order remains similar: within-season Spearman
correlation between rolling Context and History is 0.932–0.951 in August and
0.996–0.998 by the final checkpoint. Adjacent-checkpoint mean absolute
expected-rank movement falls from 13.3 to 2.8 for rolling Context and from
15.1 to 2.9 for History. Similar rank stability does not imply equal
probability quality.

## Transfer availability and notable disagreements

Transfer completeness is based on the historical Context 1.3 coverage artifact
**before imputation**. Both incoming prior-usage and defensive DB impact sums
must be marked available. This says whether those feature sums were available,
not whether every real-world transfer was identified. Across 534
team-seasons, 488 are complete, 40 incomplete, and 6 use a History fallback
without a Context coverage record. The DB-sum availability flag is true
throughout this panel, so the incomplete split primarily reflects missing
prior usage. The small incomplete and fallback groups limit subgroup claims.

| Group | Team-seasons | Rolling August Δ | Rolling September Δ | Rolling final Δ | Rolling final C−H coverage |
|---|---:|---:|---:|---:|---:|
| Complete | 488 | -0.032 | +0.016 | +0.058 | -0.026 |
| Incomplete | 40 | -0.072 | -0.023 | +0.018 | -0.009 |
| History fallback | 6 | +0.043 | -0.008 | +0.033 | -0.029 |

The final History advantage is larger among transfer-complete teams, so it is
not concentrated in the missing-transfer group. For complete teams, rolling
final Δ NLL by year is -0.005, +0.086, +0.071, and +0.072. For incomplete
teams it is +0.001, +0.031, +0.137, and -0.094, on annual counts of 19, 14,
4, and 3. The 2024–2025 incomplete results are especially unstable because
those groups are tiny.

Examples of large final rolling team-level differences in target-weighted NLL
(C−H) are 2023 James Madison +1.398, 2025 Navy +1.008, and 2024 Buffalo
+0.820, all transfer-complete. Context instead does better for 2022 Nevada
-0.779 and 2023 East Carolina -0.680, also complete, and for 2025 Northern
Illinois -0.655, incomplete. These examples show why the mean should be read
alongside the distribution of team differences and the year-by-year pattern.

## Current 2026 checkpoints: descriptive only

Six retained Context 1.3/History 1.1 weekly snapshot pairs from September
8–27 use matched effective cutoffs and identical included-game rows. The table
describes 138 FBS teams. There is no final 2026 rank-distribution target, so
**no 2026 NLL or calibration claim** is made. Context 1.3's 2026 prior is a
retrospective reconstruction using transfer inputs retrieved after the
preseason cutoff. Full per-team values are in
[`current_2026_descriptive.csv`](../data/processed/history_context_posterior_study/current_2026_descriptive.csv).
Some older retained History snapshot metadata does not record its inference
configuration; any configuration that is recorded is checked against the
current 500 / 1e-9 / 0.35 settings.

| Date | Included games | C/H mean 80% width | Mean absolute C/H expected-rank gap | C/H rank Spearman | Top-25 overlap |
|---|---:|---:|---:|---:|---:|
| Sep 08 | 172 | 62.3 / 66.7 | 7.25 | .971 | 24 |
| Sep 13 | 291 | 55.0 / 58.6 | 6.09 | .985 | 23 |
| Sep 19 | 295 | 54.8 / 58.2 | 6.06 | .985 | 23 |
| Sep 20 | 410 | 50.0 / 52.9 | 5.51 | .988 | 22 |
| Sep 26 | 417 | 49.6 / 52.4 | 5.42 | .989 | 22 |
| Sep 27 | 530 | 46.2 / 48.4 | 5.11 | .991 | 21 |

The two rankings are already highly correlated, while the lineages still
assign different uncertainty and top-25 memberships. These snapshots show
current descriptive divergence; they cannot establish which lineage is better
calibrated in 2026.

## Reproduction

With the historical final-rank corpus and cached raw CFBD game responses
available locally, run:

```bash
uv run python scripts/study_history_context_posterior.py \
  --targets /path/to/team_season_rank_distributions.csv \
  --raw-games /path/to/raw/cfbd/games
```

The script writes only `data/processed/history_context_posterior_study/`.
The committed CSVs contain the arm-specific team results, checkpoint and
transfer-group summaries, evidence diagnostics, and the 2026 descriptive
sample. The retained historical posterior backtest supplies dates and a
History replay diagnostic, not an unmatched comparator. The validated rolling
P3 prior metrics are checked before posterior scoring.
