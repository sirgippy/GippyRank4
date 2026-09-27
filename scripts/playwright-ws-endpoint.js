const { spawnSync } = require("node:child_process");
const { createHash } = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DEFAULT_WS_ENDPOINT_FILE = path.join(
  os.homedir(),
  ".cache",
  "gippyrank",
  "playwright-ws-endpoint",
);
const ENDPOINT_PROBE_SCRIPT = path.join(__dirname, "probe-playwright-ws-endpoint.js");
const CACHE_ENDPOINT_STALE_MESSAGE = "Cached Playwright browser endpoint is stale. Start 'npm run browser:server' in the host WSL environment and retry.";

function parseWsEndpoint(value, source) {
  const endpoint = value.trim();
  if (!endpoint) return undefined;

  let url;
  try {
    url = new URL(endpoint);
  } catch {
    throw new Error(`${source} must contain a valid ws:// or wss:// endpoint.`);
  }
  if (url.protocol !== "ws:" && url.protocol !== "wss:") {
    throw new Error(`${source} must contain a ws:// or wss:// endpoint.`);
  }
  return url.toString();
}

function readWsEndpointFile(endpointFile = DEFAULT_WS_ENDPOINT_FILE) {
  let contents;
  try {
    contents = fs.readFileSync(endpointFile, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return undefined;
    throw new Error(`Unable to read Playwright endpoint file ${endpointFile}: ${error.message}`);
  }
  return parseWsEndpoint(contents, `Playwright endpoint file ${endpointFile}`);
}

function probeWsEndpoint(endpoint) {
  const result = spawnSync(process.execPath, [ENDPOINT_PROBE_SCRIPT, endpoint], {
    encoding: "utf8",
    timeout: 5_000,
  });
  return result.status === 0;
}

function removeWsEndpointFile(endpointFile) {
  try {
    fs.rmSync(endpointFile, { force: true });
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

function endpointFingerprint(endpoint) {
  return createHash("sha256").update(endpoint).digest("hex");
}

function defaultInvalidationFile(endpointFile) {
  const cacheFileFingerprint = createHash("sha256").update(path.resolve(endpointFile)).digest("hex");
  return path.join(os.tmpdir(), "gippyrank", `playwright-ws-endpoint-${cacheFileFingerprint}`);
}

function isInvalidatedEndpoint(endpoint, invalidationFile) {
  try {
    return fs.readFileSync(invalidationFile, "utf8").trim() === endpointFingerprint(endpoint);
  } catch (error) {
    if (error.code === "ENOENT") return false;
    return false;
  }
}

function invalidateEndpoint(endpoint, endpointFile, invalidationFile, removeEndpointFile) {
  try {
    removeEndpointFile(endpointFile);
    return;
  } catch {
    try {
      fs.mkdirSync(path.dirname(invalidationFile), { recursive: true });
      fs.writeFileSync(invalidationFile, `${endpointFingerprint(endpoint)}\n`, "utf8");
    } catch {
      // A read-only host cache still must produce the concise recovery error below.
    }
  }
}

function clearOutdatedInvalidation(endpoint, invalidationFile) {
  if (isInvalidatedEndpoint(endpoint, invalidationFile)) return true;
  try {
    fs.rmSync(invalidationFile, { force: true });
  } catch {
    // A leftover marker is harmless when it does not describe this endpoint.
  }
  return false;
}

function discoverWsEndpoint({
  env = process.env,
  endpointFile = DEFAULT_WS_ENDPOINT_FILE,
  probeEndpoint = probeWsEndpoint,
  removeEndpointFile = removeWsEndpointFile,
  invalidationFile = defaultInvalidationFile(endpointFile),
} = {}) {
  const explicitEndpoint = env.PLAYWRIGHT_WS_ENDPOINT;
  if (typeof explicitEndpoint === "string" && explicitEndpoint.trim()) {
    const endpoint = parseWsEndpoint(explicitEndpoint, "PLAYWRIGHT_WS_ENDPOINT");
    if (!probeEndpoint(endpoint)) {
      throw new Error("PLAYWRIGHT_WS_ENDPOINT is unreachable. Verify the endpoint and retry.");
    }
    return endpoint;
  }

  const cachedEndpoint = readWsEndpointFile(endpointFile);
  if (!cachedEndpoint) return undefined;
  if (clearOutdatedInvalidation(cachedEndpoint, invalidationFile)) {
    throw new Error(CACHE_ENDPOINT_STALE_MESSAGE);
  }
  if (!probeEndpoint(cachedEndpoint)) {
    invalidateEndpoint(cachedEndpoint, endpointFile, invalidationFile, removeEndpointFile);
    throw new Error(CACHE_ENDPOINT_STALE_MESSAGE);
  }
  return cachedEndpoint;
}

module.exports = {
  CACHE_ENDPOINT_STALE_MESSAGE,
  DEFAULT_WS_ENDPOINT_FILE,
  discoverWsEndpoint,
  defaultInvalidationFile,
  probeWsEndpoint,
  readWsEndpointFile,
};
