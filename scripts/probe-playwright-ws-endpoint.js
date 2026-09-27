const { chromium } = require("@playwright/test");

async function main() {
  const endpoint = process.argv[2];
  if (!endpoint) throw new Error("A Playwright WebSocket endpoint is required.");
  const browser = await chromium.connect(endpoint, { timeout: 3_000 });
  await browser.close();
}

main().catch(() => {
  process.exitCode = 1;
});
