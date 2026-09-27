const { chmod, mkdir, readFile, rm, writeFile } = require("node:fs/promises");
const path = require("node:path");
const { randomBytes } = require("node:crypto");
const { chromium } = require("@playwright/test");
const { DEFAULT_WS_ENDPOINT_FILE } = require("./playwright-ws-endpoint");

async function writeEndpointFile(endpoint) {
  const endpointDirectory = path.dirname(DEFAULT_WS_ENDPOINT_FILE);
  await mkdir(endpointDirectory, { recursive: true, mode: 0o700 });
  await chmod(endpointDirectory, 0o700);
  await writeFile(DEFAULT_WS_ENDPOINT_FILE, `${endpoint}\n`, { encoding: "utf8", mode: 0o600 });
  await chmod(DEFAULT_WS_ENDPOINT_FILE, 0o600);
}

async function removeOwnEndpointFile(endpoint) {
  try {
    const currentEndpoint = (await readFile(DEFAULT_WS_ENDPOINT_FILE, "utf8")).trim();
    if (currentEndpoint === endpoint) await rm(DEFAULT_WS_ENDPOINT_FILE, { force: true });
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

async function main() {
  const browserServer = await chromium.launchServer({
    headless: true,
    host: "127.0.0.1",
    port: 0,
    wsPath: `gippyrank-${randomBytes(32).toString("hex")}`,
    handleSIGHUP: false,
    handleSIGINT: false,
    handleSIGTERM: false,
  });
  const endpoint = browserServer.wsEndpoint();
  await writeEndpointFile(endpoint);

  console.log("Playwright Chromium server is ready.");
  console.log(`Endpoint: ${endpoint}`);
  console.log(`Endpoint file: ${DEFAULT_WS_ENDPOINT_FILE}`);
  console.log("Press Ctrl-C to stop the server and remove the endpoint file.");

  let stop;
  let stopping = false;
  const stopped = new Promise((resolve) => {
    stop = resolve;
  });
  const shutdown = async () => {
    if (stopping) return;
    stopping = true;
    await removeOwnEndpointFile(endpoint);
    await browserServer.close();
    stop();
  };

  process.once("SIGINT", () => void shutdown());
  process.once("SIGTERM", () => void shutdown());
  browserServer.once("close", () => void shutdown());
  await stopped;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
