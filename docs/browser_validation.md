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

## Standard Codex WSL workers

Standard Codex WSL workers cannot launch Chromium themselves, but can connect
to a loopback-only Playwright browser server started by the host. The same
browser specs still route the worker's own `site/` files from its worktree.

In a host WSL terminal outside the Codex sandbox, run:

```bash
npm ci
npm run browser:server
```

The server prints its WebSocket endpoint and writes it to
`~/.cache/gippyrank/playwright-ws-endpoint`. Keep that terminal open. In the
standard worker, run the usual command:

```bash
npm ci
npm run test:browser
```

The Playwright configuration discovers the endpoint file automatically. An
explicit `PLAYWRIGHT_WS_ENDPOINT` takes precedence when a different endpoint is
needed. The endpoint is a local browser-control capability: do not commit it
or share it. `browser:server` binds only `127.0.0.1`, uses an unguessable
WebSocket path, and removes its endpoint file when stopped with Ctrl-C.

The cached endpoint file is disposable discovery state. Before using it, the
test configuration probes the endpoint. If the host server was killed and the
cache is stale, the file is removed and the command stops once with instructions
to restart `npm run browser:server`; it does not fall back to local Chromium.
When a sandbox may read but not remove the host cache, it records a temporary
invalidation marker instead, so that stale endpoint is still rejected once per
command without repeated connection attempts. A newly published endpoint clears
the outdated marker automatically.
An explicitly supplied `PLAYWRIGHT_WS_ENDPOINT` is authoritative and instead
fails directly if it cannot be reached.

Run `npm ci` in both the host and worker so their Playwright major/minor
versions match. For a standard Codex worker, enable network access without
using Full Access in the user configuration:

```toml
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
```

This permits the worker to reach the loopback browser server; it does not
require `danger-full-access`.

Run the JavaScript syntax check separately with:

```bash
npm run test:js
```

Or run both in one command:

```bash
npm run test:ui
```

`test:js` runs `node --check` over every JavaScript file under `site/assets/`
and verifies browser endpoint discovery/configuration without launching a
browser. `test:browser` exercises rendered Schedule controls, deep links,
responsive overflow, matchup geometry, team identity layout, and the Miami
intra-word wrapping regression.

When a browser test fails, Playwright writes failure screenshots, traces, and
videos under `test-results/`. CI also produces an HTML report under
`playwright-report/` and uploads both locations as a failure artifact.
