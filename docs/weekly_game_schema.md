# Weekly game artifact contract

Issue 50 adds a lazy weekly artifact for every published ranking snapshot:

```text
data/week-games/<snapshot-id>.json
```

The manifest exposes its path, byte size, game count, and week count.  The
rankings page does not load this artifact; its size is reported under
`payload_stats.lazy_week_games_bytes`.

## Snapshot and week semantics

`week` is copied from the processed CFBD schedule.  It is never inferred from
the browser date.  Numeric weeks, including Week 0, sort before named or
missing weeks.  Games sort by week, UTC kickoff timestamp, stable game ID, and
team IDs, so a rebuild is deterministic.  Within a selected week the browser
groups cards by browser-local calendar date and preserves that order.  Kickoff
date/time labels and grouping use the same browser-local timezone; snapshot
cutoff timestamps remain explicitly UTC.
The manifest's `default_week` selects the earliest available week for preseason
snapshots, the publication-labeled week when one is present, and otherwise the
latest week represented by the snapshot cutoff.  An explicit browser `week`
query parameter always takes precedence.

The selected snapshot controls game state:

- `completed`: the score is known at that snapshot; eligible games carry both
  canonical completed performance summaries.
- `future`: the kickoff is strictly after the snapshot cutoff; an eligible
  game references one canonical future prediction.
- `unresolved`: the schedule did not provide durable completed evidence at the
  cutoff; no score, rating, or fabricated prediction is shown.
- `cancelled`: a cancellation or postponement is explicit and never receives a
  prediction.

Started-but-unincluded games remain redacted.  Games outside the Historical
Likelihood eligibility boundary remain visible as schedule context, but their
scores are shown only when the snapshot explicitly includes durable completion
evidence; otherwise they remain unresolved and marked not modeled.  This
prevents current-corpus lower-division results from leaking into older
snapshots.

## Canonical game shape

Each scheduled game occurs once in `weeks[].games[]` and retains:

```json
{
  "game_id": "...",
  "week": 2,
  "date": "...",
  "home_team": {"team_id": "...", "team_name": "...", "subdivision": "fbs"},
  "away_team": {"team_id": "...", "team_name": "...", "subdivision": "fbs"},
  "neutral_site": false,
  "state": "future",
  "future_prediction_id": "..."
}
```

Non-neutral games use the source home/away orientation.  Neutral games use a
stable team-ID ordering and are explicitly labeled neutral.  Future margins
remain canonical `home minus away`; the browser does not calculate a second
orientation.  Performance summaries are copied from the existing team-season
rating objects with `display_pmf_ref` pointing to the weekly artifact's single
keyed display array.  The future prediction map is the existing canonical map,
including its source provenance and fixed-grid display distribution.

FBS team links carry season, snapshot ID, ranking family, prior family where
applicable, and the selected week.  FCS or unsupported opponents stay visible
as text with no invented team page or probabilities.

## Browser presentation

`week.html` uses the same green completed-performance and gold future-margin
visual language as team pages.  Completed cards show both team assessments;
future cards show home/away win probability, expected margin, and the central
predictive ranges.  Details use semantic `<details>` controls, and every SVG
has an accessible text equivalent.  The desktop layout is a two-column card
grid; the mobile layout collapses to one column without horizontal scrolling.
