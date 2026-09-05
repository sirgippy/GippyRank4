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

`site/publish_config.json` is the explicit, ordered list of snapshot directories
approved for publication. The publisher refuses invalid or incompatible
snapshots, copies display-level summaries only, and derives records solely from
each snapshot's `included_games.csv`. Raw posterior PMFs remain research
artifacts and are not sent to the browser.

Each approved item has a deterministic `publication_slot`. Context and History
entries in the same slot represent the same logical publication, so changing
views preserves (for example) Sep. 5 rather than jumping to Preseason. The
visible **Modeled record** likewise includes only FBS/FCS games eligible for the
ranking model, not a conventional all-games standings record.

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
