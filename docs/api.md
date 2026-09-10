# GippyRank Public API V1

The API is a thin publication interface over the validated JSON under
`site/data`. It does not acquire CFBD data, run Posterior V1, run Performance
V1, simulate seasons, or mutate a repository artifact. The static site and API
are two consumers of the same generated publication source.

## Framework and deployment

The service uses FastAPI because it fits the repository's Python/uv runtime,
generates OpenAPI from the typed Pydantic response models, and provides a small
ASGI process with no database requirement. The deployment target is a Render
Docker web service defined in [`render.yaml`](../render.yaml). Render deploys
reviewed commits from `main`, serves HTTPS, supplies the `PORT` environment
variable, and checks `/health`.

The public HTTPS base URL is assigned by Render when the `gippyrank-api` service
is provisioned. This repository does not claim a live URL until that external
service and DNS decision exist. The local base URL is:

```text
http://127.0.0.1:8000
```

Start locally from the repository root with:

```bash
uv run uvicorn gippyrank.api.app:app --host 127.0.0.1 --port 8000
```

Set `GIPPYRANK_PUBLICATION_DATA` only to choose a different, trusted publication
directory for a deployment or test. It must contain a manifest-approved data
set. `GIPPYRANK_API_CORS_ORIGINS` accepts a comma-separated allowlist; the
default is `*` for public read-only access. Credentials are never allowed.

## Routes

All public data routes are under `/api/v1/`. The exact `snapshot_id` is the
durable lookup key. The response's `publication.snapshot_id` always reports the
resolved ID.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Process and publication-data readiness |
| GET | `/api/v1/` | API identity, version, and resource links |
| GET | `/api/v1/publications` | Publication inventory and filters |
| GET | `/api/v1/rankings/{snapshot_id}` | Exact ranking table |
| GET | `/api/v1/rankings/{snapshot_id}/teams/{team_id}` | One team and its rank PMF |
| GET | `/api/v1/rankings/{snapshot_id}/teams/{team_id}/season` | Snapshot-safe team schedule and outlook |
| GET | `/api/v1/rankings/{snapshot_id}/season-outlook` | All published season-simulation summaries |
| GET | `/api/v1/rankings/{snapshot_id}/weeks` | Week inventory |
| GET | `/api/v1/rankings/{snapshot_id}/weeks/{week}` | One canonical game-centric week |
| GET | `/api/v1/rankings/{snapshot_id}/games/{game_id}` | One canonical snapshot-scoped game |
| GET | `/api/v1/seasons/{season}/publications/{publication_slot}/rankings` | Explicit slot convenience lookup |

`/api/v1/publications` supports `season`, `family`, `prior`, and `status`:

```bash
curl 'http://127.0.0.1:8000/api/v1/publications?season=2026&family=predictive&prior=context&status=official'
```

`family` is `predictive` or `performance`; `prior` is `context` or `history`
and applies only to Predictive; `status` is `official` or `temporary`. An
invalid value is a `400`. A Performance-plus-prior combination is a `422`.
An empty filtered inventory is returned as an empty list; it is not silently
replaced by another view.

The convenience route requires enough selectors to identify exactly one view.
With no family/prior it returns `422` when a slot contains multiple views.
`family=predictive&prior=context` and `family=predictive&prior=history` remain
distinct. `status` is an exact constraint, not a preference, and there is no
implicit `latest` alias.

## Response contract

Every publication response contains a `publication` object with:

- season, exact snapshot ID, snapshot type, publication slot, display label;
- official or temporary status;
- `predictive` versus `performance` family and, for Predictive, Context versus
  History prior;
- requested and effective cutoff, generation and source retrieval timestamps;
- public model-version metadata, anchor family and source-context snapshot,
  ranking support, publication counts, and the comparison baseline.

OpenAPI is available at `/openapi.json`; interactive Swagger UI is at `/docs`
and ReDoc is at `/redoc`. The generated models intentionally reject arbitrary
extra response fields. Nullable fields are present as JSON `null` where a
publication does not provide a value.

### Rankings

```bash
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-preseason-context'
```

`rankings` is already in website order: rated display ranks ascending, then
`display_rank: "NR"`. Each row includes the stable team ID, name, conference,
display rank, expected/median/modal rank, 50/80/95% intervals and widths,
Top-5/10/25 and rank-1 probabilities, modeled record, eligibility, rated/NR
state, and official-baseline movement fields. Expected rank is a distribution
mean; it is not a deterministic ordinal rank.

The team route includes the exact published rank PMF for that team:

```bash
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-preseason-context/teams/194'
```

No team-name lookup is used as the canonical identifier.

### Team season and games

```bash
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-weekly-2026-09-08T11-43-00.275833Z-context/teams/194/season'
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-weekly-2026-09-08T11-43-00.275833Z-context/weeks/2'
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-weekly-2026-09-08T11-43-00.275833Z-context/games/401858213'
```

The team-season schedule retains the site artifact's focal-team orientation:
stable opponent identity, home/away/neutral site, result and team/opponent
score when known, game state, completed-game performance summary, and one
canonical future prediction when supported. The weekly and direct game routes
use the game-centric representation, where `home_team` and `away_team` are
stable, neutral-site state is explicit, score is `{home, away}`, and future
margin is always home minus away. A weekly game appears exactly once.

Game states are `completed`, `future`, `unresolved`, `cancelled`, and
`out_of_scope`. Unsupported future games remain visible without an invented
prediction. Completed performance is a game-evidence summary, not a team
quality rating.

The completed-game display summary uses the published 40-bin performance PMF
and its fixed-scale axis when available. Future predictions expose the exact
published expected margin, median, win probabilities, nested intervals, and
the optional fixed-grid display distribution. Display bins are presentation
data only; the API does not reconstruct exact values from them.

### Season outlook

```bash
curl 'http://127.0.0.1:8000/api/v1/rankings/2026-weekly-2026-09-08T11-43-00.275833Z-context/season-outlook'
```

The response is absent/empty when the selected publication did not publish a
season simulation. When present, it exposes the simulation version, prediction
source, explicit configuration, schedule scope, and one typed outlook per FBS
team. Outlooks contain completed and remaining accounting, expected and median
wins, win distributions, record/threshold probabilities, variance
decomposition, and any unsupported-game reasons. The API never reruns the
simulation.

## Historical safety and provenance

The selected snapshot's effective cutoff controls what is returned. Results,
scores, completed performance, and season state are retained only when the
publication artifact has durable evidence for that snapshot. Future
predictions are returned only for games strictly after the cutoff and only when
the artifact contains a canonical prediction. A request for an older snapshot
cannot fall through to a newer artifact.

Predictive Context, Predictive History, and Performance remain separate. The
Performance publication's completed ratings are Context-anchored as declared
by the publication metadata; its future predictions use the paired Context
prediction artifact. The response says which source is being represented.

## Errors

Errors have one stable V1 shape:

```json
{
  "error": {
    "code": "snapshot_not_found",
    "message": "No published snapshot matches the requested identifier."
  }
}
```

The service uses `400` for malformed parameters, `404` for absent exact
season/snapshot/team/week/game resources, `422` for a semantically invalid or
ambiguous selector, `405` for non-read methods, and `500`/`503` when the
server's publication state is invalid or not loaded. Error messages never
include filesystem paths, credentials, or acquisition details.

## Caching and CORS

Exact snapshot, team, season, week, game, and outlook responses are immutable
publication resources and return:

```text
Cache-Control: public, max-age=31536000, immutable
ETag: "..."
Last-Modified: ...
```

They support `If-None-Match` and `If-Modified-Since` and return `304` when
appropriate. Inventory and the API root use a five-minute cache. The slot
convenience route uses a one-minute cache because its resolution is an alias.
Health is `no-store`.

CORS permits public GET/OPTIONS requests. The default origin policy is `*`,
with no credentials; deployments may replace it with an explicit comma-
separated allowlist via `GIPPYRANK_API_CORS_ORIGINS`. There are no write routes
in V1.

## Representative payload sizes

Sizes vary with the number of published games and distributions. In the
checked-in 2026 publication set, representative uncompressed API responses
are approximately:

| Resource | Approximate payload |
| --- | ---: |
| One ranking snapshot | 92 KiB |
| One team ranking and rank-distribution response | 5 KiB |
| One team-season response | 48 KiB |
| One weekly-game response | 122 KiB |
| One season-outlook response | 394 KiB |

These are uncompressed JSON response sizes; compression and HTTP headers are
not included. The manifest's `payload_stats` and per-publication byte fields
remain the authoritative generated-artifact measurements. Resource-specific
routes keep these payloads out of the ranking-table response.

## Security and source-of-truth boundary

Only manifest-listed relative paths are loaded. User identifiers are resolved
against in-memory indexes, not interpolated into filesystem paths. The API has
no arbitrary path, command, model, CFBD, authentication, or mutation feature.
Startup validates the manifest, methodology, snapshot tables, PMFs,
team-season artifacts, weekly canonical game fold, prediction references,
leakage boundaries, and season-outlook types. A malformed or internally
inconsistent publication set makes the service fail closed and causes health
to report unavailable.
