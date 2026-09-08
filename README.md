# GippyRank 4.0

GippyRank is an experimental probabilistic college-football ranking project.
The public site currently presents predictive ranking snapshots; it does not use
AP, Coaches, or CFP poll inputs.

## Static website

The GitHub Pages site in `site/` is deliberately a static consumer of compact
JSON. It knows nothing about posterior inference, historical model fitting, or
CFBD. The publication boundary is:

```text
existing validated snapshot bundles -> scripts/build_site_data.py -> site/data -> GitHub Pages
```

`site/publish_config.json` is the explicit list of snapshot directories approved
for publication. Its ordered `publication_slots` list is the publication
chronology, and each slot must be explicitly marked `official` or `temporary`.
All Context, History, and Performance snapshots in a slot inherit that one
status. The publisher refuses missing or incompatible metadata, copies
display-level summaries only, and derives records solely from each snapshot's
`included_games.csv`. Raw posterior PMFs remain research artifacts and are not
sent to the browser.

Team names link to a deep-linkable `site/team.html` view. It lazy-loads a
snapshot-scoped team-season artifact containing schedule metadata and compact,
Context-anchored game-rating summaries with uncertainty; future results are
redacted relative to the selected cutoff. See
[the team-season artifact contract](docs/team_season_schema.md) for the
mathematical definition and provenance checks.

Each approved item has a deterministic `publication_slot`. Context and History
entries in the same slot represent the same logical publication, so changing
views preserves (for example) Sep. 5 rather than jumping to Preseason. The
visible **Modeled record** likewise includes only FBS/FCS games eligible for the
ranking model, not a conventional all-games standings record.

Official snapshots are the historical comparison checkpoints. The static export
resolves each snapshot's movement baseline as the latest strictly earlier
official slot with the same season, ranking family, and predictive prior family.
Temporary snapshots remain selectable but never reset that baseline. The
current 2026 configuration marks Preseason and Week 2 official; Sep. 5, Sep. 6,
and Sep. 7 are temporary. To promote a future candidate, change only its slot
status to `official` in `publication_slots` and regenerate `site/data`; no model
rebuild is required.

Publication families are registered in `RANKING_FAMILIES` in
`src/gippyrank/site_data.py`. To publish a future family, add its label to that
registry and add compatible, explicitly approved bundles to the configuration;
unknown families are rejected.

### Regenerate public data

After manually building and reviewing a new snapshot, add it to
`site/publish_config.json`, then run:

```bash
uv run python scripts/build_site_data.py
```

Commit the resulting `site/data/` changes with the configuration change. This
command never creates a snapshot, calls CFBD, or runs Posterior V1.

### Preview locally

```bash
python -m http.server --directory site 8000
```

Open `http://localhost:8000`. Relative asset and data paths keep the site safe
under the GitHub Pages project-site subpath (`/GippyRank4/`).

### Deployment

`.github/workflows/deploy-pages.yml` deploys the `site/` directory after pushes
to `main`. It runs only the static export/validation step before uploading the
Pages artifact; it does not fetch CFBD, run inference, or require project
secrets beyond GitHub Pages' standard permissions.

For the posterior snapshot contract and math, see
[Posterior Snapshot V1](docs/posterior_v1.md).

## Test suite

The default developer/CI command runs every retained test:

```bash
uv run pytest -q
```

The test inventory, pruning rationale, and timing report are maintained in
[docs/test_suite.md](docs/test_suite.md).

## Weekly ranking update

The reviewed publication path is deliberately separate from Pages deployment:

```text
Actions → Update GippyRank rankings → Run workflow → inspect summary/PR → merge PR → Pages deploys
```

Set the repository Actions secret at **Settings → Secrets and variables →
Actions → New repository secret → `CFBD_API_KEY`**. The update workflow calls
only `GET /games?year=<season>&classification=fbs` and
`GET /games?year=<season>&classification=fcs`; it never calls `/games/teams`.
It writes each raw response under `data/raw/cfbd/games/` with an adjacent,
durable provenance file (endpoint, parameters, retrieval time, source kind, and
SHA-256).

The normal form needs only a season. The acquisition timestamp supplies the
requested/effective evidence boundary and a stable date publication slot; label
and slot are optional overrides. Both Context and History are built as `weekly`
snapshots from that exact corpus and must agree on cutoffs, source hashes, and
eligible game IDs before any site data is prepared. An invalid or nonconverged
snapshot fails closed.

The Action creates or updates `automation/rankings-<publication-slot>` and its
review PR; it never pushes to `main`, merges, or enables auto-merge. If your
repository restricts workflow-created PRs, enable **Settings → Actions →
General → Workflow permissions → Allow GitHub Actions to create and approve
pull requests**. Merging the candidate PR is the explicit publication decision.

For a local fallback (which prepares artifacts but deliberately does not do any
Git/PR operation), run:

```bash
CFBD_API_KEY=... uv run python scripts/update_rankings.py --season 2026
```

Repeated candidates use the same logical slot and replace that slot's Context/
History configuration entries while preserving older approved slots and
Preseason. The site opens its explicit `default_publication_slot` (normally the
newest weekly candidate). A repeat with identical rankings and eligible games
reports no publishable change rather than opening a timestamp-only PR.
