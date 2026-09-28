# Retrospective expectation holdout study

## Question and scope

This study measures how removing the evaluated game changes the retrospective
margin distribution shown for it. The primary sample is the default Sep. 27,
2026 publication slot: Context 1.3 and History each contribute the same 530
modeled games through Week 4. These are two prior-family sensitivity checks at
the same evidence boundary, not 1,060 independent games. Neither calculation
uses evidence after its selected snapshot.

For each game, the existing leave-one-game-out (LOO) result comes from that
snapshot's saved `retrospective_game_expectations`. The comparison uses the
same snapshot's saved full-posterior PMFs for both teams. Both sides use the
same frozen Historical Likelihood V1 margin surface, home-minus-away
orientation, product of team marginal PMFs, and exact finite Student-t mixture
for means and observed-margin CDFs. LOO removes the exact game ID and reruns
deterministic damped BP on the connected component; full posterior uses the
ordinary PMFs that include that game. The only intended difference is whether
the evaluated game's evidence contributes to those PMFs.

The signed margin change is `full-posterior expected home margin - LOO
expected home margin`. Percentile changes use the exact mixture CDF at the
observed margin. For tail summaries, the two-sided tail is `min(p, 1-p)`; “LOO
surprising, full ordinary” means LOO two-sided tail below 5% and full tail at
least 10%. Team residuals are actual minus expected margin oriented for each
focal team. Rank changes are full-posterior mean rank minus frozen prior mean
rank, so a negative value means the team moved toward a better rank. The
script also computes exact central 80% intervals for representative games.

The study script writes one row per game and per FBS team to the machine-readable
files linked below. It validates game IDs, LOO source snapshot, likelihood
version, prior hash, and full-posterior PMFs before comparing anything.

## Aggregate margin and percentile movement

| Prior family | Games | Median absolute margin shift | 90th percentile | Moved over 3 points | Over 5 points | Over 10 points | Median absolute percentile change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context 1.3 | 530 | 3.16 | 8.10 | 51.9% | 30.2% | 5.3% | 5.1 percentage points |
| History | 530 | 3.29 | 8.51 | 54.9% | 29.4% | 5.8% | 5.4 percentage points |

The signed mean margin shift is small (+0.45 points for Context and +0.59 for
History, home-oriented). The absolute shifts are not negligible for every
game: only 84 Context and 89 History games move by less than one point. The
largest absolute shift is 13.5 points in both models. Context's week-level
mean absolute shifts are 4.11, 3.80, 3.39, and 3.94 points for Weeks 1–4;
History is similar at 4.26, 3.99, 3.51, and 4.06. There is no monotonic early
versus late pattern in this sample.

The observed percentile moves closer to the 50th percentile in 96.6% of
Context games and 96.2% of History games. A shift of at least 10 percentile
points occurs in 6.2% and 6.6%, respectively; none shifts by 20 points. This
is the expected self-inclusion effect and makes the full-posterior percentile
an in-sample compatibility description, not an independent measure of
over- or under-performance.

The tail check shows a limited but visible interpretation change. Twelve
Context games (10 History games) move from an LOO two-sided tail below 5% to a
full-posterior two-sided tail of at least 10%. Fourteen Context games (12
History games) remain beyond a 5% two-sided tail under both methods. Thus the
full posterior does make some games ordinary, while a set of the most unusual
results remains unusual. It does not make almost every result ordinary.

## Representative games

Margins and intervals below are home-minus-away. Percentiles are the observed
margin's lower tail.

| Context game | Actual | LOO expected (80% interval) | Full expected (80% interval) | LOO → full percentile | Reading |
| --- | ---: | --- | --- | ---: | --- |
| Vanderbilt vs. Austin Peay | +19 | +18.65 `[-1.47, 38.81]` | +18.66 `[-1.06, 38.44]` | 50.9% → 51.0% | Expectations and uncertainty nearly agree. |
| Nicholls vs. Mississippi Valley State | +34 | +1.94 `[-22.04, 25.84]` | +15.33 `[-6.67, 36.76]` | 95.7% → 86.7% | A LOO tail surprise becomes ordinary by the stated 10% two-sided threshold; the result remains near the full 80% interval's upper edge. |
| UC Davis vs. Montana | −40 | +2.76 `[-19.84, 25.36]` | −10.73 `[-32.00, 10.88]` | 0.9% → 4.0% | A 13.5-point expectation shift still leaves the result outside the full 80% interval and below a 5% two-sided tail. |

The same change occurs in FBS-only games. Minnesota's home loss to Mississippi
State (Week 2) is −25 points: expected margin moves from +3.5 LOO to −5.5 full
posterior, and its two-sided tail moves from 4.8% to 11.2%. Tulsa's +14 home
win over Oklahoma State moves from a −16.0 to −6.9 expected margin, with its
tail moving from 3.8% to 10.9%.

Uncertainty changes are smaller than the mean shifts in these examples. For
Vanderbilt, the central 80% width changes from 40.3 to 39.5 points. For UC
Davis–Montana it narrows from 45.2 to 42.9 points while the center shifts by
13.5. The full result can therefore look less surprising from both a moved
center and modestly narrower uncertainty.

## Team-level narrative

The table uses the Context 1.3 snapshot. Team examples are selected
mechanically: Georgia; the FBS team with the smallest prior-to-posterior mean
rank movement among teams with at least three games; the largest improvement
and decline by the same measure; and the team with the largest posterior rank
standard deviation.

| Team | Prior → posterior mean rank (posterior SD) | Games above expectation, LOO → full | Mean focal residual, LOO → full |
| --- | --- | ---: | ---: |
| Georgia | 9.2 → 4.1 (3.4) | 4/4 → 4/4 | +12.4 → +10.3 |
| Middle Tennessee (nearest to prior) | 115.0 → 115.1 (15.2) | 3/4 → 3/4 | +1.8 → +1.0 |
| New Mexico (largest rank improvement) | 88.3 → 46.3 (21.8) | 4/4 → 4/4 | +9.5 → +6.9 |
| Western Kentucky (largest rank decline) | 63.1 → 114.5 (16.6) | 1/4 → 1/4 | −12.9 → −9.9 |
| Arkansas (widest posterior rank SD) | 68.5 → 68.0 (26.5) | 1/4 → 1/4 | +0.1 → −0.9 |

Georgia's four focal-team margins were +60, +50, +28, and +28. Its LOO
expected margins were +53.4, +33.5, +18.1, and +11.3; full-posterior margins
were +54.6, +35.5, +19.6, and +15.3. Every game remains above expectation
under both constructions. Georgia's posterior moves from a mean rank of 9.2
to 4.1, so the full-posterior story is that the model learned a stronger
Georgia and still leaves positive residuals; it does not erase the repeated
pattern.

Across the 138 FBS teams with at least three included games, 45 Context teams
have positive residuals in at least 75% of their games under LOO, versus 44
under full posterior. Thirty-two teams have negative residuals in at least
75% under either method. In History, positive-pattern counts are 45 versus
46, and negative-pattern counts are 30 under either method. Only a few
individual FBS game residual signs change: three Context team-game signs
across Colorado–Weber State and Michigan–Iowa. Their expected margins move by
less than one point and the observed margins are near expectation in both
cases. The repeated team-level over/under-performance counts barely move.

As a simple uncertainty slice for FBS-vs-FBS matchups, mean absolute shifts
are 3.62 points for 81 games when the two teams' average posterior rank SD is
at least 20, versus 3.07 across 134 games below 20 in Context; History is 3.82
across 98 games versus 3.19 across 117. The effect is a little larger for less
certain matchups, but not dramatically so. Mean shifts
for games with an absolute margin of at least 35 are 4.52 points in Context
and 4.90 in History.

## Recommendation

For a team-season claim that a team repeatedly **outperformed or
underperformed expectation**, keep the LOO expectation as the comparison. It
prevents the evaluated score from setting the expectation used to judge that
same score. The evidence does not show a broad LOO-created season narrative:
the counts of teams with consistently positive or negative residuals barely
change, and Georgia's four-game pattern survives in full.

Label that statistic as “compared with expectations from the selected
snapshot's other games.” It is retrospective because later games through the
selected cutoff can inform team beliefs; it is not a kickoff-time forecast.
If the product instead wants to describe how compatible a game is with the
team profiles learned from the complete selected snapshot, full-posterior
expectations answer that separate question. Their percentile should be
identified as in-sample compatibility: it moves most games toward the center
and normalizes a small set of otherwise tail-surprising games.

## Reproduction and limits

Run from the repository root:

```bash
uv run python scripts/analyze_retrospective_expectation_holdout.py
```

The default is the selected publication slot in `site/publish_config.json`.
Optional repeated `--source` arguments select explicit predictive snapshot
directories; `--output-dir` changes the output location.

The outputs are [per-game comparisons](../data/processed/retrospective_expectation_holdout/game_comparisons.csv),
[per-team summaries](../data/processed/retrospective_expectation_holdout/team_comparisons.csv),
and an [aggregate summary](../data/processed/retrospective_expectation_holdout/summary.json).

This is one unfinished 2026 season at a late-September snapshot, not a
cross-season calibration study. The full-posterior comparison is in-sample by
construction, and neither statistic is a pregame forecast. The study does not
change the published artifact, team-season UI, PR #137 semantics, or
production expectation implementation.
