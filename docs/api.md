# GippyRank Static JSON API V1

GippyRank's public API is a static, read-only JSON interface hosted by the
same GitHub Pages deployment as the human-facing site. It is generated during
the reviewed site export from validated publication artifacts. Consumers do
not need repository access, a Python process, a database, model execution, or
a separately hosted service.

The API base path is:

```text
https://sirgippy.github.io/GippyRank4/api/v1/
```

The generated files are also available locally under `site/api/v1/` after:

```bash
uv run python scripts/build_site_data.py
```

## Discovery

Start with `index.json`, then read `publications.json`:

```bash
curl 'https://sirgippy.github.io/GippyRank4/api/v1/index.json'
curl 'https://sirgippy.github.io/GippyRank4/api/v1/publications.json'
```

`publications.json` contains a `publications` array and one entry per approved
snapshot. Each entry includes the season, exact snapshot ID, publication slot,
official or temporary status, display label, ranking family, predictive prior
when applicable, requested/effective cutoff, model metadata, and `links` to
the ranking, week, and season-outlook resources plus stable team/game path
templates. Links are relative to the API base path.

There is no implicit latest or fallback snapshot. Select the exact snapshot
and preserve the distinction between:

- Predictive Context: current predictive ranking using the Context prior.
- Predictive History: current predictive ranking using the History prior.
- Performance: observed-game performance ranking, Context-anchored as declared
  by the publication metadata.

Official and temporary publications are both explicit in the inventory.
Historical snapshots remain bounded by their effective cutoff; later scores,
game states, performance results, and predictions are not substituted into an
older publication.

## Resources

| Resource | Purpose |
| --- | --- |
| `index.json` | API version, schema, counts, and entry-point links |
| `publications.json` | Small publication inventory and resource links |
| `rankings/<snapshot-id>.json` | One published ranking table and metadata |
| `rankings/<snapshot-id>/weeks.json` | Week inventory and counts |
| `rankings/<snapshot-id>/weeks/<week>.json` | One canonical game-centric week |
| `rankings/<snapshot-id>/games/<game-id>.json` | One canonical game |
| `rankings/<snapshot-id>/teams/<team-id>.json` | One team ranking and rank PMF |
| `rankings/<snapshot-id>/teams/<team-id>/season.json` | One team's schedule and outlook |
| `rankings/<snapshot-id>/season-outlook.json` | Published regular-season outlooks |

All subordinate resources repeat the resolved `publication` metadata. Team
and game identifiers are stable IDs, not names. Every scheduled game appears
once in the weekly resource for its week, including games with an unsupported
or null prediction.

## Agent workflow: one week's expected margins

An automated consumer can retrieve one week's compact, canonical payload:

```bash
curl 'https://sirgippy.github.io/GippyRank4/api/v1/publications.json'
curl 'https://sirgippy.github.io/GippyRank4/api/v1/rankings/2026-weekly-2026-09-13T12-02-55.255941Z-context/weeks/3.json'
```

For a future game, `prediction.expected_home_margin` and
`prediction.median_home_margin` use the explicit, stable orientation **home
minus away**, in points. The payload also includes home/away win probabilities,
neutral-site state, kickoff, team IDs/names, intervals, and game state. A
future game without a supported prediction remains in `games` with
`prediction: null` and no invented margin.

Completed games expose scores and, where published, performance evidence.
That evidence describes the observed game and is not an intermediate team
quality or Elo-style rating.

## Snapshot and resource semantics

The ranking resource contains only ranking rows and normal public metadata; it
does not bundle rank PMFs, schedules, all games, or season simulations. Fetch a
team or game resource when that detail is needed. JSON files are deterministic
and are safe to cache as immutable publication artifacts.

The site exporter is the only publication step:

```text
reviewed publication artifacts
    -> scripts/build_site_data.py
    -> site/data and site/api/v1
    -> existing GitHub Pages deployment
```

The exporter performs no CFBD acquisition, model execution, live updates, or
runtime request handling. The human-facing site and static API are two views
over the same approved publication state.
