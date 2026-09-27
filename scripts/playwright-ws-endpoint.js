const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DEFAULT_WS_ENDPOINT_FILE = path.join(
  os.homedir(),
  ".cache",
  "gippyrank",
  "playwright-ws-endpoint",
);

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

function discoverWsEndpoint({ env = process.env, endpointFile = DEFAULT_WS_ENDPOINT_FILE } = {}) {
  const explicitEndpoint = env.PLAYWRIGHT_WS_ENDPOINT;
  if (typeof explicitEndpoint === "string" && explicitEndpoint.trim()) {
    return parseWsEndpoint(explicitEndpoint, "PLAYWRIGHT_WS_ENDPOINT");
  }
  return readWsEndpointFile(endpointFile);
}

module.exports = {
  DEFAULT_WS_ENDPOINT_FILE,
  discoverWsEndpoint,
  readWsEndpointFile,
};
