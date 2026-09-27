const assert = require("node:assert/strict");
const { mkdtempSync, rmSync, writeFileSync } = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const { discoverWsEndpoint } = require("../../scripts/playwright-ws-endpoint");
const { buildPlaywrightConfig } = require("../../scripts/playwright-test-config");

function withEndpointFile(callback) {
  const directory = mkdtempSync(path.join(os.tmpdir(), "gippyrank-playwright-"));
  const endpointFile = path.join(directory, "playwright-ws-endpoint");
  try {
    return callback(endpointFile);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test("an explicit endpoint takes precedence over the endpoint file", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "ws://127.0.0.1:37123/from-file\n");
    assert.equal(
      discoverWsEndpoint({
        endpointFile,
        env: { PLAYWRIGHT_WS_ENDPOINT: "ws://127.0.0.1:37123/from-environment" },
      }),
      "ws://127.0.0.1:37123/from-environment",
    );
  });
});

test("the endpoint file is used when no explicit endpoint is configured", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "  ws://127.0.0.1:37123/from-file  \n");
    assert.equal(
      discoverWsEndpoint({ endpointFile, env: {} }),
      "ws://127.0.0.1:37123/from-file",
    );
  });
});

test("normal configuration omits connectOptions", () => {
  withEndpointFile((endpointFile) => {
    const config = buildPlaywrightConfig({ endpointFile, env: {} });
    assert.equal(Object.hasOwn(config.use, "connectOptions"), false);
  });
});

test("endpoint configuration connects the standard fixtures to the remote browser", () => {
  withEndpointFile((endpointFile) => {
    const config = buildPlaywrightConfig({
      endpointFile,
      env: { PLAYWRIGHT_WS_ENDPOINT: "ws://127.0.0.1:37123/remote-browser" },
    });
    assert.deepEqual(config.use.connectOptions, {
      wsEndpoint: "ws://127.0.0.1:37123/remote-browser",
    });
  });
});

test("invalid configured endpoints fail before Playwright starts", () => {
  withEndpointFile((endpointFile) => {
    assert.throws(
      () => discoverWsEndpoint({ endpointFile, env: { PLAYWRIGHT_WS_ENDPOINT: "http://127.0.0.1:37123" } }),
      /ws:\/\/ or wss:\/\//,
    );
  });
});
