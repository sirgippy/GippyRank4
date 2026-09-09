# Team-season artifact contract

The static exporter publishes one lazy `data/team-seasons/<snapshot-id>.json`
artifact for each ranking snapshot. The rankings page loads only its compact
snapshot table; `team.html?team=<stable-id>&season=<season>&snapshot=<snapshot-id>`
loads this artifact on direct navigation.

## Game-rating definition

For a focal team `i`, opponent `j`, and game `g`, the rating PMF is:

```text
normalize( HistoricalLikelihood_g(i, j) × Cavity_j→g )
```

`Cavity_j→g` is the converged BP belief formed from the opponent prior and all
messages from neighboring factors except the grouped pair factor containing
`g`. The focal preseason prior is not multiplied into this PMF. The
`BeliefPropagationState` is exposed on `PosteriorResult` so the site artifact
builder reuses the production inference state instead of running a second
approximation.

Multiple games between the same pair use the same pair cavity and apply each
individual game factor separately. Individual rematch ratings are explanatory
single-game estimates; their product is not intended to reconstruct the
grouped pair factor.

The Context network is the production anchor for all game ratings, including
Predictive History and Performance pages. Under loopy BP, the focal prior is
absent as a direct factor, but indirect feedback through schedule cycles can
remain.

## Completed-game presentation data

Each modeled completed game keeps the exact game-evidence summaries above and
adds `display_pmf`, a deterministic 40-bin compression of that same rank PMF.
Display bins are fixed-scale integer weights from 0 through 1000; divide by
the shared axis `probability_encoding.scale` to recover chart probabilities.
The artifact-level `performance_axis` is shared by every game in the season:
rank 1 (best) is on the left and `max_rank` (worst) is on the right. The
display weights sum to 1000 and are not used to recompute any exact summary.

`performance_percentile` is an empirical percentile of
`game_rating.expected_rank` among all eligible FBS team-game performance
distributions included by the selected snapshot. Lower expected rank is
better; a value receives credit for every reference performance with a worse
expected rank and half credit for exact ties. Consequently, a historical
percentile never uses a later game. The frozen presentation grade mapping is
`A >= 90`, `B >= 70`, `C >= 30`, `D >= 10`, and `F < 10`; grades are display
sugar and never enter inference.

The artifact records the statistic, direction, reference population, tie
rule, and reference count in `performance_percentile`.

## Future-game prediction definition

An eligible scheduled game has one canonical entry in the artifact's
`future_predictions` object.  Its margin is home-oriented, even though the
Historical Likelihood V1 cross-subdivision response is FBS-minus-FCS.  For a
home/away rank pair `(r_h, r_a)`:

```text
P(M | snapshot) = Σ P(M | r_h, r_a, site, pairing)
                       P(r_h | snapshot) P(r_a | snapshot)
```

Each conditional component is the exact V1 Student-t margin distribution with
the frozen V1 location surface, scale, and 15 degrees of freedom.  The
published expected margin is the mixture mean; median and central 50%, 80%,
and 95% intervals are deterministic mixture quantiles found by bracketing and
solving the finite-mixture CDF.  No expected-rank plug-in or sampled Monte
Carlo distribution is used.

The continuous Student-t mixture has no point mass at zero.  The artifact
still records the explicit football tie convention:

```text
P(home wins) = P(M > 0) + 0.5 P(M = 0)
P(away wins) = P(M < 0) + 0.5 P(M = 0)
```

The two stored win probabilities are complements from that same distribution.

Future predictions use `predictive_context` on Predictive Context pages,
`predictive_history` on Predictive History pages, and the same-slot
`predictive_context` artifact on Performance pages.  A Performance page labels
that source in the browser.  Completed-game ratings remain Context-anchored.

Eligibility is strict: the scheduled kickoff must be after the selected
snapshot's effective cutoff, the game must not be in the included evidence,
and both teams must have supported V1 rank representations.  Existing FCS
fallback variables are reused when present; unsupported matchups remain
visible without a prediction.  Neutral games use the neutral V1 site row, and
FBS/FCS games use the stable FBS-first V1 coordinate orientation before the
stored home-oriented margin is restored.

## JSON shape

Each team contains schedule entries with stable IDs, date/week, opponent
metadata, site, result/score when known at the cutoff, and `modeled`. A modeled
completed game has a compact `game_rating` summary containing:

- expected rank, median, and mode;
- central 50%, 80%, and 95% intervals;
- Top 5, Top 10, and Top 25 probabilities;
- the fixed-scale integer 40-bin `display_pmf`, empirical `performance_percentile`, and
  presentation-only `performance_grade`.

No full per-game PMF is serialized. Games after the selected cutoff retain
schedule metadata but have null result, score, and rating fields. An eligible
future row has `future_prediction_id`; the ID resolves into the single
canonical `future_predictions` map entry. Ineligible completed games remain
visible as `modeled: false` and never receive a fabricated rating or
prediction.

The prediction map contains compact summaries, not sampled distributions:

```json
{
  "prediction_schema_version": "1.0",
  "prediction_source": "predictive_context",
  "prediction_provenance": {
    "source_snapshot_id": "2026-weekly-...-context",
    "effective_cutoff": "2026-09-08T11:43:00+00:00",
    "included_game_ids": ["..."],
    "historical_likelihood_version": "V1"
  },
  "future_predictions": {
    "401752680": {
      "home_team_id": "61",
      "away_team_id": "333",
      "expected_home_margin": 5.8,
      "median_home_margin": 5.6,
      "home_win_probability": 0.67,
      "away_win_probability": 0.33,
      "tie_probability": 0.0,
      "margin_interval_50": [-1.2, 12.8],
      "margin_interval_80": [-8.1, 19.9],
      "margin_interval_95": [-17.4, 29.3]
    }
  }
}
```

The values above are illustrative.  Static export validates the prediction
source, provenance, complementarity, nested interval ordering, strict cutoff,
and cross-team references before publishing.

The `future_margin_axis` is fixed at `-40` through `+40` points in 40 bins,
with the home-oriented convention `margin = home points - away points`.
Each prediction also contains `display_distribution` with the CDF mass in each
visible bin and explicit lower/upper tail weights. The display is clipped only
for chart space; its three encoded mass components sum to 1000 and are decoded
using `future_margin_axis.probability_encoding`.
The exact expected margin, median, win probabilities, and 50/80/95% intervals
remain authoritative and are not derived from display bins. For large
posterior mixtures, export aggregates nearby component locations into a
deterministic maximum-256-component representation before evaluating the
display CDF; this approximation affects presentation bins only, never the
exact predictive summaries.

Stable `team_id` and `opponent_id` values are resolved at render time through
the manifest-level `team_logos.handles` map. The team-season artifact does not
repeat logo handles on every team or schedule entry. Missing identities are
omitted from that map and listed in the manifest audit; the browser keeps the
text name when no logo is available.

The artifact repeats the historical inference provenance (`effective_cutoff`,
`game_corpus_sha256`, included game IDs, and source retrieval evidence). Because
backfilled historical artifacts may use a newer schedule corpus for display
metadata, `schedule_source` separately records that corpus's kind, repository
path, and SHA-256. Static export validates the provenance object's structure
and the paired Context provenance; it does not compare a retained artifact's
schedule hash with a later mutable schedule corpus.

Results and scores are shown only when the game ID is in the snapshot's durable
`included_game_ids` evidence. A kickoff before the cutoff is not enough: a game
that was in progress or otherwise absent from that evidence remains redacted,
even if the current schedule corpus now contains a final score.
