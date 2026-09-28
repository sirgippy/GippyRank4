const fs = require("node:fs");
const path = require("node:path");
const { test, expect } = require("@playwright/test");
const { installStaticSiteRoute } = require("./static-site");

const current = "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3";
const currentPerformance = `${current}-performance`;
const historical = "2026-weekly-2026-09-26T12-09-11.589245Z-context-v1.3";
const logoResponse = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" rx="5" fill="#e0e8e2"/><path d="M12 26h36" stroke="#9bafa3" stroke-width="3"/></svg>';

function teamUrl(team, snapshot = current) {
  return `/team.html?team=${team}&season=2026&family=predictive&prior=context&snapshot=${snapshot}`;
}

async function loadTeam(page, team, snapshot = current) {
  await page.goto(teamUrl(team, snapshot));
  await expect(page.locator("#team-page-status")).toHaveText(/scheduled games/);
}

test.beforeEach(async ({ page }) => {
  await installStaticSiteRoute(page);
  await page.route("**/*", (route) => route.request().resourceType() === "image"
    ? route.fulfill({ status: 200, contentType: "image/svg+xml", body: logoResponse }) : route.fallback());
});

test("keeps current Georgia results, checkpoint movement, and future predictions compact", async ({ page }, testInfo) => {
  await loadTeam(page, "61");
  const completed = page.locator('.game-card[data-game-id="401856700"]');
  const future = page.locator('.game-card[data-game-id="401856705"]');
  await expect(completed).toContainText("W 41–13");
  await expect(completed).toContainText("Expected Georgia by 11.3");
  await expect(completed).toContainText("Actual Georgia by 28.0");
  await expect(completed).toContainText("14% this favorable or better");
  await expect(completed.locator(".game-rating")).toHaveCount(0);
  await expect(page.locator(".schedule-checkpoint")).toHaveCount(4);
  await expect(page.locator(".schedule-checkpoint").last()).toContainText("Week 5 · published belief");
  await expect(future).toContainText("Upcoming");
  await expect(future.locator(".game-prediction-title")).toHaveText("Georgia 91% to win · Georgia by 20.7");
  await expect(page.locator('.game-card[data-game-id="401856712"] .game-prediction-title'))
    .toHaveText("Alabama 52% to win · Alabama by 0.6");
  await expect(future.locator(".game-retrospective")).toHaveCount(0);
  const maximumCollapsedHeight = testInfo.project.name === "mobile" ? 125 : 85;
  expect((await page.locator(".game-card").first().boundingBox()).height).toBeLessThan(maximumCollapsedHeight);
  expect((await future.boundingBox()).height).toBeLessThan(maximumCollapsedHeight);
});

test("shows Context-anchored retrospective expectations on a Performance page", async ({ page }) => {
  await page.goto(`/team.html?team=61&season=2026&family=performance&snapshot=${currentPerformance}`);
  await expect(page.locator("#team-page-status")).toHaveText(/scheduled games/);
  await expect(page.locator("#team-page-meta")).toContainText("Performance");
  const completed = page.locator('.game-card[data-game-id="401856700"]');
  await expect(completed).toContainText("W 41–13");
  await expect(completed.locator(".game-retrospective")).toContainText("Expected Georgia by 11.3");
  await expect(completed.locator(".game-retrospective")).toContainText("Actual Georgia by 28.0");
  await expect(completed).not.toContainText("No retrospective expectation available");
});

test("orients an away upset from the focal team and keeps an ordinary result quiet", async ({ page }) => {
  await loadTeam(page, "113");
  const upset = page.locator('.game-card[data-game-id="401858423"]');
  await expect(upset).toContainText("Rutgers");
  await expect(upset).toContainText("W 37–21");
  await expect(upset).toContainText("Expected Rutgers by 24.1");
  await expect(upset).toContainText("Actual Massachusetts by 16.0");
  await expect(upset).toContainText("1.1% this favorable or better");
  await upset.locator(".game-distribution-disclosure summary").click();
  await expect(upset.locator(".game-distribution-retrospective .distribution-tail-note"))
    .toContainText("16% of otherwise-expected outcomes lie beyond the visible ±40-point range");
  await expect(upset.locator(".game-distribution-retrospective .distribution-tail")).toHaveCount(0);
  const ordinary = page.locator('.game-card[data-game-id="401866424"]');
  await expect(ordinary).toContainText("37% this unfavorable or worse");
  await expect(ordinary).not.toContainText(/shocking|dominant|fluke|grade/i);
});

test("uses selected historical evidence and ends checkpoint movement at that snapshot", async ({ page }) => {
  await loadTeam(page, "61", historical);
  const arkansas = page.locator('.game-card[data-game-id="401856686"]');
  const oklahoma = page.locator('.game-card[data-game-id="401856700"]');
  await expect(arkansas).toContainText("Expected Georgia by 22.5");
  await expect(oklahoma).toContainText("Upcoming");
  await expect(oklahoma.locator(".game-prediction")).toBeVisible();
  await expect(page.locator(".schedule-checkpoint").last()).toContainText("Sep 26 interim");
  await expect(page.locator("#schedule-list")).not.toContainText("Week 5 · published belief");
});

test("keeps an unmodeled completed score when its expectation is unavailable", async ({ page }) => {
  const artifactPath = `site/data/team-seasons/${current}.json`;
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
  delete artifact.retrospective_game_expectations.games["401856700"];
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856700"]');
  await expect(game).toContainText("W 41–13");
  await expect(game).toContainText("No retrospective expectation available");
  await expect(page.locator('.game-card[data-game-id="401856686"] .game-retrospective')).toBeVisible();
});

test("opens distinct retrospective and predictive distributions from the keyboard without overflow", async ({ page }) => {
  await loadTeam(page, "61");
  const completed = page.locator('.game-card[data-game-id="401856700"]');
  const retrospective = completed.locator(".game-distribution-disclosure");
  const retroToggle = retrospective.locator("summary");
  await expect(retroToggle).toHaveAccessibleName(/Expected Georgia by 11\.3.*Actual Georgia by 28\.0.*14% this favorable or better.*Show Georgia completed-game retrospective distribution/);
  await expect(retrospective.locator(".retrospective-distribution-chart")).toBeHidden();
  await retroToggle.focus();
  await page.keyboard.press("Enter");
  await expect(retrospective).toHaveAttribute("open", "");
  await expect(retroToggle).toHaveAccessibleName(/Expected Georgia by 11\.3.*Actual Georgia by 28\.0.*14% this favorable or better.*Hide Georgia completed-game retrospective distribution/);
  const expectedMargin = Number(await retrospective.locator(".retrospective-marker-expected").getAttribute("data-margin"));
  expect(expectedMargin).toBeCloseTo(11.298375740164296, 10);
  await expect(retrospective.locator(".retrospective-marker-actual")).toHaveAttribute("data-margin", "28");
  await expect(retrospective.locator(".retrospective-chart-legend")).toContainText("Expected Georgia by 11.3");
  await expect(retrospective.locator(".retrospective-chart-legend")).toContainText("Actual Georgia by 28.0");
  await expect(retrospective.locator("dt")).toHaveText([
    "Observed percentile",
    "Median margin",
    "Central 50%",
    "Central 80%",
    "Central 95%",
  ]);
  await expect(retrospective.locator(".distribution-tail")).toHaveCount(0);
  await expect(retrospective.locator(".distribution-tail-note"))
    .toContainText("4% of otherwise-expected outcomes lie beyond the visible ±40-point range");
  const future = page.locator('.game-card[data-game-id="401856705"]');
  await expect(future.locator(".game-distribution-disclosure summary"))
    .toHaveAccessibleName(/Georgia 91% to win.*Georgia by 20\.7.*Show predictive margin distribution/);
  await future.locator(".game-distribution-disclosure summary").click();
  await expect(future.locator(".future-distribution-chart")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("keeps long mobile future predictions and their disclosure controls together with logos present", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "This is a narrow-layout regression.");
  await page.setViewportSize({ width: 320, height: 844 });
  await loadTeam(page, "113");
  let wrapped = false;
  for (const gameId of ["401866433", "401866455"]) {
    const game = page.locator(`.game-card[data-game-id="${gameId}"]`);
    await game.scrollIntoViewIfNeeded();
    await expect(game.locator(".team-logo")).toBeVisible();
    await expect.poll(() => game.locator(".team-logo").evaluate((image) => image.naturalWidth)).toBe(60);
    const layout = await game.locator(".game-distribution-disclosure summary").evaluate((summary) => {
      const copy = summary.querySelector(".game-summary-copy");
      const chevron = summary.querySelector(".distribution-chevron");
      const copyBox = copy.getBoundingClientRect();
      const chevronBox = chevron.getBoundingClientRect();
      return {
        copyHeight: copyBox.height,
        lineHeight: Number.parseFloat(getComputedStyle(copy).lineHeight),
        copyRight: copyBox.right,
        copyBottom: copyBox.bottom,
        chevronLeft: chevronBox.left,
        chevronTop: chevronBox.top,
      };
    });
    wrapped ||= layout.copyHeight > layout.lineHeight + 1;
    expect(layout.chevronLeft).toBeGreaterThanOrEqual(layout.copyRight - 1);
    expect(layout.chevronTop).toBeLessThan(layout.copyBottom);
  }
  expect(wrapped).toBeTruthy();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1);
});
