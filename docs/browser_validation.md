# Browser/UI validation

The static site has a small Playwright harness under `tests/browser/`. It uses
Chromium only and runs each Schedule smoke test in both a desktop viewport and
a narrow Pixel 5-style viewport. Each browser spec routes the synthetic
`http://gippyrank.test` origin directly to files in `site/`; the harness does
not start a local HTTP server.

## Fresh-environment bootstrap

From the repository root on Linux or WSL:

```bash
npm ci
npx playwright install --with-deps chromium
```

The `--with-deps` flag installs Chromium and the Linux libraries it needs.
After the one-time bootstrap, run the checks with:

```bash
npm run test:browser
```

`npm ci` followed by `npm run test:browser` is sufficient once Chromium and
its system libraries are present. Browser tests do not require Python, `uv`, a
writable uv cache, or permission to bind a listening socket.

Run the JavaScript syntax check separately with:

```bash
npm run test:js
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
