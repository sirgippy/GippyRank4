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

## JSON shape

Each team contains schedule entries with stable IDs, date/week, opponent
metadata, site, result/score when known at the cutoff, and `modeled`. A modeled
completed game has a compact `game_rating` summary containing:

- expected rank, median, and mode;
- central 50%, 80%, and 95% intervals;
- Top 5, Top 10, and Top 25 probabilities.

No full per-game PMF is serialized. Games after the selected cutoff retain
schedule metadata but have null result, score, and rating fields. Ineligible
completed games remain visible as `modeled: false` and never receive a
fabricated rating.

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
