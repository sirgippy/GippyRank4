const { defineConfig } = require("@playwright/test");
const { buildPlaywrightConfig } = require("./scripts/playwright-test-config");

module.exports = defineConfig(buildPlaywrightConfig());
