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

The artifact's `inference` object records implementation, component-size
bounds, BP-iteration bounds, worker count, and wall-clock runtime. Static-site
manifest entries expose per-snapshot expected-outcome and display-payload byte
counts, making runtime and storage impact measurable at publication time.

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
observed)` is its complement. Exact scalars are calculated from the mixture,
never from display bins.

`display_distribution` uses the same fixed `-40` to `+40` 40-bin margin axis
as future-game displays. It has deterministic fixed-scale integer mass,
explicit lower and upper off-axis tail mass, and a total encoded mass of 1000.
For large mixtures, the deterministic display approximation affects only those
bins; it cannot alter an entry's exact summaries or tails.

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
