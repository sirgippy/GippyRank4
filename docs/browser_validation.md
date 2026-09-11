# Browser/UI validation

The static site has a small Playwright harness under `tests/browser/`. It uses
Chromium only and runs each Schedule smoke test in both a desktop viewport and
a narrow Pixel 5-style viewport. The Playwright configuration starts the
existing static site with `uv run python -m http.server`; do not start a second
server first.

## Fresh-environment bootstrap

From the repository root on Linux or WSL:

```bash
uv sync
npm ci
npx playwright install --with-deps chromium
```

The `--with-deps` flag installs Chromium and the Linux libraries it needs.
After the one-time bootstrap, run the checks with:

```bash
npm run test:js
npm run test:browser
```

Or run both in one command:

```bash
npm run test:ui
```

`test:js` runs `node --check` over every JavaScript file under
`site/assets/`. `test:browser` exercises rendered Schedule controls, deep
links, responsive overflow, matchup geometry, team identity layout, and the
Miami intra-word wrapping regression.

When a browser test fails, Playwright writes failure screenshots, traces, and
videos under `test-results/`. CI also produces an HTML report under
`playwright-report/` and uploads both locations as a failure artifact.
