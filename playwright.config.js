const { defineConfig, devices } = require("@playwright/test");

const port = Number(process.env.PLAYWRIGHT_PORT || 4173);
const isCi = Boolean(process.env.CI);

module.exports = defineConfig({
  testDir: "./tests/browser",
  outputDir: "./test-results",
  fullyParallel: true,
  workers: isCi ? 2 : undefined,
  forbidOnly: isCi,
  retries: isCi ? 2 : 0,
  timeout: 30_000,
  reporter: isCi
    ? [["line"], ["html", { outputFolder: "playwright-report", open: "never" }]]
    : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
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
  webServer: {
    command: `uv run python -m http.server ${port} --bind 127.0.0.1 --directory site`,
    url: `http://127.0.0.1:${port}/schedule.html`,
    reuseExistingServer: !isCi,
    timeout: 30_000,
  },
});
