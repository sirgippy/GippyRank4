# Retrospective completed-game expectations

The `retrospective_game_expectations` object in each supported predictive
snapshot's `team_seasons.json` is a versioned `1.0` completed-game artifact.
It is browser-facing static data, but it is produced during snapshot modeling;
the browser does not calculate probabilities or reconstruct summaries.

## Interpretation

This is a **retrospective leave-one-game-out posterior predictive
distribution**. It describes how the observed result compares with what
GippyRank would expect given all other evidence available at the selected
snapshot. It is **not** the prediction GippyRank would have made before the
game was played.

For completed modeled game `g`, the artifact answers:

```text
P(M_g | E_snapshot \ {g})
```

where `E_snapshot` is exactly the selected snapshot's included-game evidence
and `M_g` is `home points - away points`. Later games already included in that
snapshot remain useful evidence; games beyond its effective cutoff are never
read. This is consequently neither a kickoff-time forecast nor a belief-change
attribution for `g`.

## Construction

Each target game is removed by exact game ID, then GippyRank reruns the same
deterministic loopy BP inference for the connected factor-graph components
containing either participant. Disconnected components cannot affect either
team's marginal and are omitted only as an equivalent computational shortcut.
Every replay starts with deterministic uniform BP messages that are
independent of the held-out score, then converges after that target's removal.

This recomputation is necessary for the primary self-exclusion invariant. A
standard BP pair cavity from the full posterior can retain indirect feedback
from the held-out game through a schedule cycle. When teams have a rematch,
removing one game rebuilds their grouped pair factor from the other game(s), so
the other rematch is retained in `E_snapshot \ {g}`.

The evidence boundary is exact by game ID. The inferred rank PMFs are the
output of deterministic **loopy BP**, which is approximate on graphs with
cycles. As in GippyRank's future-game predictions, the margin mixture uses
the product of the two BP marginal PMFs rather than their joint posterior.
The scalar mixture integrations are exact given those recomputed marginals;
this artifact does not claim an exact joint Bayesian posterior predictive.

The artifact's `inference` object records implementation, component-size
bounds, BP-iteration bounds, worker count, and wall-clock runtime. Static-site
manifest entries expose per-snapshot expected-outcome and display-payload byte
counts, making runtime and storage impact measurable at publication time.
Each replay is summarized before the next one is retained, and production uses
one worker to bound peak memory on large connected schedules.

Retained publications are materialized with
`uv run python scripts/backfill_retrospective_game_expectations.py`. That
command reads each configured predictive source's frozen `included_games.csv`,
checks its prior hash, and adds the expectation field without changing the
saved posterior or rankings. It also refreshes paired Performance artifacts
and their Context metadata hashes before rebuilding static site data.

## Distribution and orientation

For each rank pair, the conditional distribution is the existing Historical
Likelihood V1 Student-t margin distribution with its established
home/away/neutral and FBS/FCS pairing semantics. The output integrates its
finite mixture over the two leave-one-out posterior rank PMFs; it does not use
an expected-rank plug-in or a separate scoring model.

Every scalar and chart value is canonically home-oriented:

```text
margin = home points - away points
```

An entry provides the observed margin, expected and median margin, central
50%, 80%, and 95% intervals, and an observed-margin percentile. The
Historical Likelihood mixture is continuous, so `lower_tail_probability =
P(M <= observed)` equals the percentile and `upper_tail_probability = P(M >=
observed)` is its complement. Scalar summaries are calculated from the mixture,
never from display bins.

`display_distribution` uses the same fixed `-40` to `+40` 40-bin margin axis
as future-game displays. It has deterministic fixed-scale integer mass,
explicit lower and upper off-axis tail mass, and a total encoded mass of 1000.
For large mixtures, the deterministic display approximation affects only those
bins; it cannot alter an entry's mixture summaries or tails.

## Shape

```json
{
  "retrospective_game_expectations_version": "1.0",
  "source_snapshot_id": "2026-weekly-...-context",
  "effective_cutoff": "2026-09-27T12:27:35+00:00",
  "historical_likelihood_version": "V1",
  "margin_orientation": "home_minus_away",
  "games": {
    "game-id": {
      "home_team_id": "61",
      "away_team_id": "201",
      "actual_home_margin": 28.0,
      "expected_home_margin": 11.4,
      "median_home_margin": 11.1,
      "margin_interval_50": [5.0, 17.2],
      "margin_interval_80": [-1.8, 24.1],
      "margin_interval_95": [-10.7, 33.5],
      "observed_margin_percentile": 0.96,
      "lower_tail_probability": 0.96,
      "upper_tail_probability": 0.04,
      "display_distribution": {"masses": [0], "lower_tail_probability": 0, "upper_tail_probability": 0}
    }
  }
}
```

The displayed values are illustrative. A later UI may reverse the sign for an
away focal team as presentation only; it must preserve the stored canonical
home orientation.

The source model artifact retains retrospective records for every included
game, including FCS/FCS evidence. Static publication validates that source
and publishes only games with an FBS participant in the browser-facing
team-season artifact. Its `included_game_ids` and `inference.games_evaluated`
still describe the full source evidence set; its `games` map contains the FBS
schedule records. Every published record has the required FBS participant
schedule reference or references.

## Production sanity check: 2026-09-27 Context 1.3

The selected source snapshot contains 530 modeled completed games; 331 have an
FBS participant and appear in the published team-season artifact. The table below
reads its committed leave-one-out records directly. Margins and intervals are
home minus away; the percentile is the observed-margin lower tail. Numbers are
rounded only for this table.

| Case (away at home) | Actual | Expected | 50% interval | 80% interval | 95% interval | Observed percentile |
| --- | ---: | ---: | --- | --- | --- | ---: |
| Near expectation: Hawai'i at Stanford | +10 | +10.0 | [-1.3, 21.4] | [-11.8, 31.6] | [-24.0, 43.3] | 49.6% |
| Strong favorite loses: Massachusetts at Rutgers | -16 | +24.1 | [13.3, 35.1] | [2.9, 44.8] | [-9.4, 56.2] | 1.1% |
| Large favorite wins: Ball State at Ohio State | +53 | +38.8 | [29.3, 48.2] | [20.5, 57.0] | [9.7, 67.6] | 84.4% |
| Close game: Iowa at Michigan | -1 | -0.7 | [-12.1, 10.6] | [-22.2, 21.1] | [-33.9, 33.2] | 49.5% |
| Early game with later evidence: UCLA at California | -21 | -7.8 | [-19.0, 3.3] | [-29.0, 13.8] | [-40.5, 26.1] | 21.3% |
| FBS/FCS: Maine at Boston College | +6 | +42.7 | [32.1, 53.4] | [22.1, 63.1] | [10.3, 74.4] | 1.4% |
| Neutral site: West Virginia at Virginia | -11 | +20.3 | [10.2, 30.6] | [0.6, 39.9] | [-11.0, 50.9] | 2.5% |

UCLA at California was played in Week 1. Its expected home margin changed
from +4.5 in the 2026-09-19 Context 1.3 snapshot to -7.8 here, with the
same prior family and held-out result. This is later-season evidence changing
the retrospective assessment, not a pregame forecast. The retained snapshot
has no repeated team pair; rematch self-exclusion is covered by the synthetic
regression test. The examples check orientation, numerical summaries, site and
subdivision handling, and observed tails; they do not establish calibration.

## Measured publication cost

The one-time backfill evaluated 6,941 game distributions across 30 configured
predictive sources (four preseason sources contain zero completed games). The
sum of the recorded replay runtimes was 5,049.9 seconds. At the current 530
game publication, Context took 757.8 seconds and History took 742.0 seconds
with one worker. These timings include inference and mixture summaries for
each held-out game; static site export reads the saved artifacts.

The production manifest has 41 publications, including paired Performance
copies. Across those publications it reports 6,214 browser-facing expectation
records and 6,147,679 bytes of compact serialized expectation payload, of
which 1,150,929 bytes are display distributions. The current Context 1.3
snapshot contributes 331 records and 328,296 bytes of expectation payload.
The four preseason distributions
are empty by design.
