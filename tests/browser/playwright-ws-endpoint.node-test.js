const assert = require("node:assert/strict");
const { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const {
  CACHE_ENDPOINT_STALE_MESSAGE,
  discoverWsEndpoint,
} = require("../../scripts/playwright-ws-endpoint");
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

test("a live cached endpoint configures the remote browser", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "ws://127.0.0.1:37123/from-file\n");
    const config = buildPlaywrightConfig({ endpointFile, env: {}, probeEndpoint: () => true });
    assert.deepEqual(config.use.connectOptions, {
      wsEndpoint: "ws://127.0.0.1:37123/from-file",
    });
  });
});

test("a stale cached endpoint is removed and fails with recovery guidance", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "ws://127.0.0.1:37123/stale\n");
    assert.throws(
      () => buildPlaywrightConfig({ endpointFile, env: {}, probeEndpoint: () => false }),
      new RegExp(CACHE_ENDPOINT_STALE_MESSAGE),
    );
    assert.equal(existsSync(endpointFile), false);
  });
});

test("a stale read-only cache is invalidated when it cannot be removed", () => {
  withEndpointFile((endpointFile) => {
    const invalidationFile = path.join(path.dirname(endpointFile), "stale-endpoint");
    const endpoint = "ws://127.0.0.1:37123/stale";
    writeFileSync(endpointFile, `${endpoint}\n`);
    const removeEndpointFile = () => {
      throw new Error("read-only cache");
    };

    assert.throws(
      () => buildPlaywrightConfig({
        endpointFile,
        invalidationFile,
        env: {},
        probeEndpoint: () => false,
        removeEndpointFile,
      }),
      new RegExp(CACHE_ENDPOINT_STALE_MESSAGE),
    );
    assert.equal(existsSync(endpointFile), true);
    assert.match(readFileSync(invalidationFile, "utf8"), /^[a-f0-9]{64}\n$/);
    assert.throws(
      () => buildPlaywrightConfig({
        endpointFile,
        invalidationFile,
        env: {},
        probeEndpoint: () => {
          throw new Error("a known stale endpoint must not be probed again");
        },
        removeEndpointFile,
      }),
      new RegExp(CACHE_ENDPOINT_STALE_MESSAGE),
    );
  });
});

test("an explicit dead endpoint fails without changing cached discovery state", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "ws://127.0.0.1:37123/from-file\n");
    assert.throws(
      () => discoverWsEndpoint({
        endpointFile,
        env: { PLAYWRIGHT_WS_ENDPOINT: "ws://127.0.0.1:37123/from-environment" },
        probeEndpoint: () => false,
      }),
      /PLAYWRIGHT_WS_ENDPOINT is unreachable/,
    );
    assert.equal(existsSync(endpointFile), true);
  });
});

test("a configured endpoint file is normalized before it is probed", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "  ws://127.0.0.1:37123/from-file  \n");
    assert.equal(
      discoverWsEndpoint({ endpointFile, env: {}, probeEndpoint: () => true }),
      "ws://127.0.0.1:37123/from-file",
    );
  });
});

test("normal configuration omits connectOptions", () => {
  withEndpointFile((endpointFile) => {
    const config = buildPlaywrightConfig({ endpointFile, env: {}, probeEndpoint: () => true });
    assert.equal(Object.hasOwn(config.use, "connectOptions"), false);
  });
});

test("endpoint configuration connects the standard fixtures to the remote browser", () => {
  withEndpointFile((endpointFile) => {
    writeFileSync(endpointFile, "ws://127.0.0.1:37123/from-file\n");
    const config = buildPlaywrightConfig({
      endpointFile,
      env: { PLAYWRIGHT_WS_ENDPOINT: "ws://127.0.0.1:37123/remote-browser" },
      probeEndpoint: () => true,
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
