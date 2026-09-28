const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("@playwright/test");
const { installStaticSiteRoute } = require("../tests/browser/static-site");

const output = path.resolve(__dirname, "../docs/visual-evidence/issue-136");
const current = "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3";
const historical = "2026-weekly-2026-09-26T12-09-11.589245Z-context-v1.3";
const logoResponse = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" rx="5" fill="#e0e8e2"/><path d="M12 26h36" stroke="#9bafa3" stroke-width="3"/></svg>';

async function capture(browser, name, team, snapshot, viewport, target, expandGame) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1, timezoneId: "America/Chicago" });
  const page = await context.newPage();
  await installStaticSiteRoute(page);
  await page.route("**/*", (route) => route.request().resourceType() === "image"
    ? route.fulfill({ status: 200, contentType: "image/svg+xml", body: logoResponse }) : route.fallback());
  await page.goto(`http://gippyrank.test/team.html?team=${team}&season=2026&family=predictive&prior=context&snapshot=${snapshot}`);
  await page.locator("#schedule-list .game-card").first().waitFor();
  if (expandGame) {
    await page.locator(`.game-card[data-game-id="${expandGame}"] .game-distribution-disclosure summary`).click();
  }
  await page.locator(target).screenshot({ path: path.join(output, `${name}.png`) });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  console.log(`${name}: overflow ${overflow}px`);
  if (overflow > 1) throw new Error(`${name} overflows the viewport`);
  await context.close();
}

async function main() {
  await fs.mkdir(output, { recursive: true });
  const endpointPath = path.join(process.env.HOME, ".cache/gippyrank/playwright-ws-endpoint");
  const endpoint = process.env.PLAYWRIGHT_WS_ENDPOINT || await fs.readFile(endpointPath, "utf8").then((value) => value.trim(), () => null);
  const browser = endpoint ? await chromium.connect(endpoint) : await chromium.launch();
  try {
    const desktop = { width: 1280, height: 900 };
    const mobile = { width: 390, height: 844 };
    await capture(browser, "georgia-desktop", "61", current, desktop, 'section[aria-labelledby="schedule-title"]', "401856700");
    await capture(browser, "georgia-mobile", "61", current, mobile, 'section[aria-labelledby="schedule-title"]', "401856700");
    await capture(browser, "georgia-offscale-intervals", "61", current, desktop, '.game-card[data-game-id="401856658"]', "401856658");
    await capture(browser, "georgia-offscale-intervals-mobile", "61", current, mobile, '.game-card[data-game-id="401856658"]', "401856658");
    await capture(browser, "lsu-future-intervals", "99", current, desktop, '.game-card[data-game-id="401856706"]', "401856706");
    await capture(browser, "massachusetts-surprising", "113", current, desktop, '.game-card[data-game-id="401858423"]', "401858423");
    await capture(browser, "colorado-near-expectation", "38", current, desktop, '.game-card[data-game-id="401856785"]');
    await capture(browser, "georgia-historical", "61", historical, desktop, "#schedule-list");
    await capture(browser, "massachusetts-season", "113", current, mobile, "#schedule-list");
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
