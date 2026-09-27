const { devices } = require("@playwright/test");
const { discoverWsEndpoint } = require("./playwright-ws-endpoint");

function buildPlaywrightConfig({
  env = process.env,
  endpointFile,
  probeEndpoint,
  removeEndpointFile,
  invalidationFile,
} = {}) {
  const isCi = Boolean(env.CI);
  const wsEndpoint = discoverWsEndpoint({
    env,
    endpointFile,
    probeEndpoint,
    removeEndpointFile,
    invalidationFile,
  });
  const use = {
    baseURL: "http://gippyrank.test",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  };
  if (wsEndpoint) use.connectOptions = { wsEndpoint };

  return {
    testDir: "./tests/browser",
    testMatch: "**/*.spec.js",
    outputDir: "./test-results",
    fullyParallel: true,
    workers: isCi ? 2 : undefined,
    forbidOnly: isCi,
    retries: isCi ? 2 : 0,
    timeout: 30_000,
    reporter: isCi
      ? [["line"], ["html", { outputFolder: "playwright-report", open: "never" }]]
      : "list",
    use,
    projects: [
      {
        name: "desktop",
        use: {
          ...devices["Desktop Chrome"],
          viewport: { width: 1280, height: 900 },
        },
      },
      {
        name: "mobile",
        use: {
          ...devices["Pixel 5"],
          viewport: { width: 390, height: 844 },
        },
      },
    ],
  };
}

module.exports = { buildPlaywrightConfig };
