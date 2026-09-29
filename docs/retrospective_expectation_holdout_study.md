# Retrospective expectation holdout study

## Question and scope

This study measures how removing the evaluated game changes the retrospective
margin distribution shown for it. The primary sample is the default Sep. 27,
2026 publication slot. Context 1.3 and History each have the same 530 modeled
games through Week 4. These are two prior-family checks at one evidence
boundary, not 1,060 independent games. Neither calculation uses evidence
later than its selected snapshot.

For each game, the existing leave-one-game-out (LOO) result comes from that
snapshot's saved `retrospective_game_expectations`. The comparison uses the
same snapshot's saved full-posterior PMFs for both teams. Both use the same
Historical Likelihood V1 margin surface, home-minus-away orientation, product
of team marginal PMFs, and exact finite Student-t mixture. LOO removes the
exact game ID and reruns deterministic damped BP on the connected component;
full posterior uses the ordinary PMFs that include that game. The intended
difference is whether the evaluated score contributes to those PMFs.

The signed margin change is `full-posterior expected home margin - LOO
expected home margin`. Percentile `p` is the observed margin's lower-tail
probability. The study's **observed-direction one-sided tail probability** is
`min(p, 1-p)`: the probability of a result at least as favorable or
unfavorable as the observed result, in the observed direction. This matches
the existing UI meaning. The reported probability is intentionally not
doubled. A value below 5% means the result is beyond the
5th or 95th percentile; a value at least 10% places it between the 10th and
90th percentiles.

For a game, absolute-residual reduction is
`abs(actual - LOO expected) - abs(actual - full expected)`. Positive values
mean full posterior reduced the residual magnitude. Relative reduction is
reported only when the LOO absolute residual is at least 5 points, avoiding
unstable ratios near zero. It is the absolute-residual reduction divided by
the LOO absolute residual. Rank changes are full-posterior mean rank minus
frozen prior mean rank; a negative value means movement toward a better rank.

The main product population is every modeled game with at least one FBS team,
since those games appear on FBS team-season pages. FBS-vs-FBS is reported
separately; the complete 530-game set remains a broader model diagnostic.

The reproducible script validates game IDs and row hashes, the LOO source
snapshot and likelihood version, the prior hash, and full-posterior PMF
coverage before comparing results. It writes per-game and per-team tables as
well as aggregate summaries.

## Product-population margin shifts

All values are from the home-minus-away expected margin. Threshold shares are
the fraction with an absolute shift strictly greater than the named threshold.

| Prior family | Population | Games | Median absolute shift | 90th percentile | >3 points | >5 points | >10 points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Context 1.3 | At least one FBS team | 331 | 2.81 | 6.91 | 46.8% | 24.8% | 3.0% |
| Context 1.3 | FBS vs. FBS | 215 | 2.48 | 6.61 | 43.3% | 23.7% | 3.7% |
| History | At least one FBS team | 331 | 3.06 | 7.49 | 52.0% | 23.6% | 3.6% |
| History | FBS vs. FBS | 215 | 2.79 | 7.33 | 47.4% | 21.9% | 4.2% |

The signed mean shift is small in these populations: +0.44 to +0.64 points for
games involving an FBS team, and +0.03 to +0.18 for FBS-vs-FBS games. Expected
margins move by a few points for many games, but large shifts are uncommon.

## Product-population percentile and tail movement

| Prior family | Population | Median absolute percentile change | Moved closer to 50th percentile | Changed by ≥10 percentile points | LOO tail <5%, full tail ≥10% | Tail <5% under both |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Context 1.3 | At least one FBS team | 4.5 pp | 97.0% | 0.3% | 4 / 331 | 8 / 331 |
| Context 1.3 | FBS vs. FBS | 4.4 pp | 97.2% | 0.5% | 3 / 215 | 4 / 215 |
| History | At least one FBS team | 5.0 pp | 97.0% | 0.6% | 4 / 331 | 6 / 331 |
| History | FBS vs. FBS | 4.8 pp | 95.3% | 0.9% | 2 / 215 | 3 / 215 |

A full-posterior percentile moves toward the 50th percentile for about 95–97%
of games in the product populations. Four FBS-involved games in each prior
family cross from an observed-direction tail below 5% under LOO to at least
10% under full posterior; two to four games, depending on family and
population, remain below 5% under both. Thus some individual interpretations
change, but the full posterior does not make every result ordinary.

## Residual-magnitude normalization

| Prior family | Population | Median reduction | 10th–90th percentile | Residual shrank | Shrunk ≥5 points | Same-sign shrunk ≥5 points | Median relative reduction when LOO residual ≥5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Context 1.3 | At least one FBS team | 3.06 | 0.65–7.49 | 97.6% | 24.8% | 82 / 331 | 29.3% (n=247) |
| Context 1.3 | FBS vs. FBS | 2.79 | 0.54–7.33 | 97.2% | 23.7% | 51 / 215 | 30.0% (n=156) |
| History | At least one FBS team | 3.06 | 0.65–7.49 | 97.0% | 23.6% | 78 / 331 | 30.4% (n=245) |
| History | FBS vs. FBS | 2.79 | 0.54–7.33 | 95.3% | 21.9% | 47 / 215 | 32.0% (n=153) |

The full posterior usually reduces residual magnitude while keeping its sign.
Among FBS-involved games, the median reduction is about 3.1 points; for cases
where the LOO residual starts at 5 points or more, the median relative
reduction is about 29–30%. FBS-vs-FBS results are similar. There are just two
Context and five History game-level sign changes in the FBS-involved
population; most of the change is in how large the residual looks, not which
side of expectation it falls on.

These FBS examples have substantial magnitude changes without sign changes.
Margins and residuals are home-oriented; tail probabilities use the
observed-direction one-sided definition above.

| Game | Actual margin | LOO expected | Full expected | LOO residual → full residual | Residual reduction | LOO → full tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nevada vs. Western Kentucky (FBS-FBS) | +35 | −6.8 | +5.8 | +41.8 → +29.2 | 12.6 | 0.9% → 4.1% |
| Maryland vs. UCLA (FBS-FBS) | −51 | +2.6 | −10.6 | −53.4 → −40.4 | 13.1 | 0.15% → 0.66% |
| Utah State vs. Idaho State (FBS-FCS) | −12 | +23.4 | +10.4 | −35.4 → −22.4 | 13.0 | 2.4% → 9.7% |

The tail probability increases in all three examples, but remains below 10%.
The expected margins move materially while the residual direction stays the
same.

## Broader model-population diagnostic

The full 530-game population includes games with no FBS participant. Its
median absolute margin shift is 3.16 points for Context and 3.29 for History;
90th percentiles are 8.10 and 8.51. Median absolute percentile changes are
5.1 and 5.4 points. Twelve Context and 10 History games move from a LOO
observed-direction tail below 5% to a full tail of at least 10%; 14 and 12,
respectively, remain below 5% under both. Median absolute-residual reductions
are 3.16 and 3.29 points, and residual magnitude decreases in 97.2% and 96.8%
of games.

## Team-level narrative

The following examples use Context 1.3. Team selection is mechanical: Georgia;
the FBS team with the smallest prior-to-posterior mean-rank movement among
teams with at least three games; the largest improvement and decline by that
measure; and the team with the largest posterior rank standard deviation.

| Team | Prior → posterior mean rank (posterior SD) | Games above expectation, LOO → full | Mean focal residual, LOO → full |
| --- | --- | ---: | ---: |
| Georgia | 9.2 → 4.1 (3.4) | 4/4 → 4/4 | +12.4 → +10.3 |
| Middle Tennessee (nearest to prior) | 115.0 → 115.1 (15.2) | 3/4 → 3/4 | +1.8 → +1.0 |
| New Mexico (largest rank improvement) | 88.3 → 46.3 (21.8) | 4/4 → 4/4 | +9.5 → +6.9 |
| Western Kentucky (largest rank decline) | 63.1 → 114.5 (16.6) | 1/4 → 1/4 | −12.9 → −9.9 |
| Arkansas (widest posterior rank SD) | 68.5 → 68.0 (26.5) | 1/4 → 1/4 | +0.1 → −0.9 |

Georgia's four focal-team margins were +60, +50, +28, and +28. Its LOO
expected margins were +53.4, +33.5, +18.1, and +11.3; full-posterior margins
were +54.6, +35.5, +19.6, and +15.3. All four remain above expectation.
Georgia's posterior moves from mean rank 9.2 to 4.1. The full posterior learns
a stronger Georgia, but still leaves positive residuals for these games.

Across the 138 FBS teams with at least three included games, 45 Context teams
have positive residuals in at least 75% of games under LOO, versus 44 under
full posterior. Thirty-two teams have negative residuals at that rate under
either method. In History, positive-pattern counts are 45 versus 46, and
negative-pattern counts are 30 under both methods. This agrees with the
game-level result: magnitudes usually shrink, while repeated team-level
patterns and residual signs rarely change.

## Product interpretation

The evidence does not select one statistic without first choosing what the
page is asking:

- If the page asks, “How did this game compare with expectations formed from
  all other available evidence in this snapshot?”, LOO is the matching
  statistic.
- If the page asks, “How compatible is this game with our complete hindsight
  understanding of the teams, including this game?”, full posterior is the
  matching statistic.

The measured effect of switching is usually a smaller apparent residual and a
percentile closer to the center, not a reversed season narrative. That
includes Georgia: its four positive residuals survive both methods. A small
number of individual games move materially in tail interpretation, so the
page's wording should make the chosen question clear.

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
cross-season calibration study. Full-posterior percentiles are in-sample by
construction; neither statistic is a pregame forecast. This study does not
change the published artifact, team-season UI, PR #137 semantics, or
production expectation implementation.
