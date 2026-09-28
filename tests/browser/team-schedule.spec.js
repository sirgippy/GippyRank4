const fs = require("node:fs");
const path = require("node:path");
const { test, expect } = require("@playwright/test");
const { installStaticSiteRoute } = require("./static-site");

const current = "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3";
const currentPerformance = `${current}-performance`;
const currentHistory = "2026-weekly-2026-09-27T12-27-35.698895Z-history";
const historical = "2026-weekly-2026-09-26T12-09-11.589245Z-context-v1.3";
const logoResponse = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" rx="5" fill="#e0e8e2"/><path d="M12 26h36" stroke="#9bafa3" stroke-width="3"/></svg>';

function teamUrl(team, snapshot = current) {
  return `/team.html?team=${team}&season=2026&family=predictive&prior=context&snapshot=${snapshot}`;
}

async function loadTeam(page, team, snapshot = current) {
  await page.goto(teamUrl(team, snapshot));
  await expect(page.locator("#schedule-list .game-card").first()).toBeVisible();
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
  await expect(completed).toContainText("Actual Georgia by 28");
  await expect(completed.locator(".game-summary-copy")).toHaveText("Expected Georgia by 11.3 · Actual Georgia by 28 · 14% this favorable or better");
  await expect(completed).toContainText("14% this favorable or better");
  await expect(page.locator(".team-schedule-heading .team-page-help summary")).toHaveText("How to read games");
  await expect(page.locator("#team-page-status")).toBeHidden();
  await expect(page.getByRole("list", { name: "Scheduled games" }).getByRole("listitem")).toHaveCount(12);
  await expect(page.locator("#schedule-list").getByRole("note")).toHaveCount(4);
  await expect(page.locator(".schedule-checkpoint")).toHaveCount(4);
  await expect(page.locator(".schedule-checkpoint").last()).toContainText("Week 5 update");
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
  await expect(page.locator("#schedule-list .game-card").first()).toBeVisible();
  await expect(page.locator("#team-page-meta")).toContainText("Performance");
  const completed = page.locator('.game-card[data-game-id="401856700"]');
  await expect(completed).toContainText("W 41–13");
  await expect(completed.locator(".game-retrospective")).toContainText("Expected Georgia by 11.3");
  await expect(completed.locator(".game-retrospective")).toContainText("Actual Georgia by 28");
  await expect(completed).not.toContainText("No retrospective expectation available");
});

test("orients an away upset from the focal team and keeps an ordinary result quiet", async ({ page }) => {
  await loadTeam(page, "113");
  const upset = page.locator('.game-card[data-game-id="401858423"]');
  await expect(upset).toContainText("Rutgers");
  await expect(upset).toContainText("W 37–21");
  await expect(upset).toContainText("Expected Rutgers by 24.1");
  await expect(upset).toContainText("Actual Massachusetts by 16");
  await expect(upset).toContainText("1.1% this favorable or better");
  await upset.locator(".game-distribution-disclosure summary").click();
  await expect(upset.locator(".game-distribution-retrospective .distribution-tail-note"))
    .toContainText("16% of expected outcomes fall outside the chart's ±40-point range");
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
  await expect(page.locator("#schedule-list")).not.toContainText("Week 5 update");
  await expect(page.locator('.game-card[data-game-id="401856741"] .game-prediction-title'))
    .toHaveText("Near-even matchup · Georgia by 0.1");
  await page.locator('.game-card[data-game-id="401856741"] .game-distribution-disclosure summary').click();
  const nearEven = page.locator('.game-card[data-game-id="401856741"]');
  await expect(nearEven.locator(".game-facts")).toContainText("Georgia win probability50.1%");
  await expect(nearEven.locator(".game-facts")).toContainText("Ole Miss win probability49.9%");
});

test("uses History expectations, predictions, and checkpoint sequence for the selected History publication", async ({ page }) => {
  await page.goto(`/team.html?team=61&season=2026&family=predictive&prior=history&snapshot=${currentHistory}`);
  await expect(page.locator("#schedule-list .game-card").first()).toBeVisible();
  await expect(page.locator("#team-page-meta")).toContainText("Predictive History");
  await expect(page.locator('.game-card[data-game-id="401856700"]')).toContainText("Expected Georgia by 10.9");
  await expect(page.locator('.game-card[data-game-id="401856705"] .game-prediction-title'))
    .toHaveText("Georgia 92% to win · Georgia by 21.4");
  await expect(page.locator(".schedule-checkpoint")).toHaveCount(2);
  await expect(page.locator(".schedule-checkpoint").first()).toContainText("Week 4 update");
  await expect(page.locator(".schedule-checkpoint").last()).toContainText("Week 5 update");
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

test("does not render a retrospective record with a fractional actual margin", async ({ page }) => {
  const artifactPath = `site/data/team-seasons/${current}.json`;
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
  artifact.retrospective_game_expectations.games["401856700"].actual_home_margin = 28.4;
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856700"]');
  await expect(game).toContainText("No retrospective expectation available");
  await expect(game.locator(".game-retrospective")).toHaveCount(0);
});

test("renders cancelled and unresolved entries without fabricated analysis", async ({ page }) => {
  const artifactPath = `site/data/team-seasons/${current}.json`;
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
  for (const state of ["cancelled", "out_of_scope", "unresolved"]) {
    artifact.teams["61"].games.push({
      game_id: `${state}-test`, date: "2026-12-01T00:00:00Z", week: 14,
      opponent_id: "99999", opponent_name: "Example opponent", site: "home",
      game_state: state, result: null, score: null,
    });
  }
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  for (const [state, outcome] of [["cancelled", "Cancelled or postponed"], ["out_of_scope", "Outside model scope"], ["unresolved", "Pending"]]) {
    const game = page.locator(`.game-card[data-game-id="${state}-test"]`);
    await expect(game.locator(".game-outcome")).toHaveText(outcome);
    await expect(game.locator(".game-card-body")).toHaveCount(0);
  }
});

test("opens distinct retrospective and predictive distributions from the keyboard without overflow", async ({ page }) => {
  await loadTeam(page, "61");
  const completed = page.locator('.game-card[data-game-id="401856700"]');
  const retrospective = completed.locator(".game-distribution-disclosure");
  const retroToggle = retrospective.locator("summary");
  await expect(retroToggle).toHaveAccessibleName(/Expected Georgia by 11\.3.*Actual Georgia by 28.*14% this favorable or better.*Show Georgia completed-game retrospective distribution/);
  await expect(retrospective.locator(".retrospective-distribution-chart")).toBeHidden();
  await retroToggle.focus();
  await page.keyboard.press("Enter");
  await expect(retrospective).toHaveAttribute("open", "");
  await expect(retroToggle).toHaveAccessibleName(/Expected Georgia by 11\.3.*Actual Georgia by 28.*14% this favorable or better.*Hide Georgia completed-game retrospective distribution/);
  const expectedMargin = Number(await retrospective.locator(".retrospective-marker-expected").getAttribute("data-margin"));
  expect(expectedMargin).toBeCloseTo(11.298375740164296, 10);
  await expect(retrospective.locator(".retrospective-marker-actual")).toHaveAttribute("data-margin", "28");
  await expect(retrospective.locator(".retrospective-chart-legend")).toContainText("Expected Georgia by 11.3");
  await expect(retrospective.locator(".retrospective-chart-legend")).toContainText("Actual Georgia by 28");
  await expect(retrospective.locator(".retrospective-percentile-facts dt")).toHaveText("Observed percentile");
  await expect(retrospective.locator(".retrospective-margin-facts dt")).toHaveText(["Median margin", "Central 50%", "Central 80%", "Central 95%"]);
  await expect(retrospective.locator("dt")).toHaveText([
    "Observed percentile",
    "Median margin",
    "Central 50%",
    "Central 80%",
    "Central 95%",
  ]);
  await expect(retrospective.locator(".distribution-tail")).toHaveCount(0);
  await expect(retrospective.locator(".distribution-tail-note"))
    .toContainText("4% of expected outcomes fall outside the chart's ±40-point range");
  const future = page.locator('.game-card[data-game-id="401856705"]');
  await expect(future.locator(".game-distribution-disclosure summary"))
    .toHaveAccessibleName(/Georgia 91% to win.*Georgia by 20\.7.*Show predictive margin distribution/);
  await future.locator(".game-distribution-disclosure summary").click();
  await expect(future.locator(".future-distribution-chart")).toBeVisible();
  await expect(future.locator(".distribution-tail")).toHaveCount(0);
  await expect(future.locator(".distribution-tail-note")).toContainText("predicted outcomes fall outside the chart's ±40-point range");
  await expect(future.locator(".future-distribution-chart desc")).toContainText("Georgia win probability");
  await expect(future.locator(".future-distribution-chart desc")).toContainText("Vanderbilt win probability");
  await expect(future.locator(".game-facts dt").first()).toHaveText("Georgia win probability");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("uses a full interval view when most Tennessee State probability lies beyond the compact axis", async ({ page }) => {
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856658"]');
  await expect(game).toContainText("Expected Georgia by 53.4 · Actual Georgia by 60");
  await game.locator(".game-distribution-disclosure summary").click();
  await expect(game.locator(".retrospective-interval-chart")).toBeVisible();
  await expect(game.locator(".retrospective-distribution-chart")).toHaveCount(0);
  await expect(game.locator(".retrospective-interval")).toHaveCount(3);
  const expected = game.locator(".retrospective-marker-expected");
  const actual = game.locator(".retrospective-marker-actual");
  expect(Number(await expected.getAttribute("x1"))).toBeLessThan(Number(await actual.getAttribute("x1")));
  await expect(game.locator(".distribution-tail-note")).toContainText("82% of expected outcomes fall outside the standard ±40-point histogram");
});

test("orients an off-scale away interval view from the selected team's perspective", async ({ page }) => {
  await loadTeam(page, "189");
  const game = page.locator('.game-card[data-game-id="401858441"]');
  await expect(game).toContainText("Expected Nebraska by 32.7");
  await expect(game).toContainText("Actual Nebraska by 49");
  await game.locator(".game-distribution-disclosure summary").click();
  await expect(game.locator(".retrospective-interval-chart")).toBeVisible();
  const expected = Number(await game.locator(".retrospective-marker-expected").getAttribute("x1"));
  const actual = Number(await game.locator(".retrospective-marker-actual").getAttribute("x1"));
  expect(actual).toBeLessThan(expected);
  await expect(game.locator(".distribution-axis .axis-start")).toContainText("Nebraska by");
});

test("uses the same interval view for an off-scale future forecast", async ({ page }) => {
  await loadTeam(page, "99");
  const game = page.locator('.game-card[data-game-id="401856706"]');
  await expect(game.locator(".game-prediction-title")).toContainText("LSU 99.8% to win");
  await game.locator(".game-distribution-disclosure summary").click();
  await expect(game.locator(".future-interval-chart")).toBeVisible();
  await expect(game.locator(".future-distribution-chart")).toHaveCount(0);
  await expect(game.locator(".future-interval")).toHaveCount(3);
  await expect(game.locator(".distribution-tail-note")).toContainText("71% of predicted outcomes fall outside the standard ±40-point histogram");
  await expect(game.locator(".game-facts")).toContainText("LSU win probability99.8%");
  await expect(game.locator(".future-interval-chart desc")).toContainText("McNeese win probability 0.2%");
});

for (const [label, expectedMargin, actualMargin, score] of [
  ["upper", 40, 60, { team: 73, opponent: 13 }],
  ["lower", -40, -60, { team: 13, opponent: 73 }],
]) {
  test(`separates markers when one is at the ${label} bound and the other is beyond it`, async ({ page }) => {
    const artifactPath = `site/data/team-seasons/${current}.json`;
    const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
    const gameId = "401856700";
    const expectation = artifact.retrospective_game_expectations.games[gameId];
    expectation.expected_home_margin = expectedMargin;
    expectation.actual_home_margin = actualMargin;
    const gameRecord = artifact.teams["61"].games.find((game) => game.game_id === gameId);
    gameRecord.score = score;
    gameRecord.result = actualMargin > 0 ? "W" : "L";
    await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
    await loadTeam(page, "61");
    const game = page.locator(`.game-card[data-game-id="${gameId}"]`);
    await game.locator(".game-distribution-disclosure summary").click();
    const expected = game.locator(".retrospective-marker-expected");
    const actual = game.locator(".retrospective-marker-actual");
    expect(await expected.getAttribute("x1")).toBe(await actual.getAttribute("x1"));
    expect(Number(await expected.getAttribute("y2"))).toBeLessThan(Number(await actual.getAttribute("y1")));
    await expect(actual).toHaveAttribute("data-clipped", label);
    await expect(game.locator(".distribution-marker-note")).toContainText("Actual");
  });
}

test("keeps a retrospective chart quiet when less than 1% lies beyond its axis", async ({ page }) => {
  const artifactPath = `site/data/team-seasons/${current}.json`;
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
  const retrospective = artifact.retrospective_game_expectations;
  const scale = retrospective.margin_axis.probability_encoding.scale;
  retrospective.games["401856700"].display_distribution.lower_tail_probability = scale * 0.004;
  retrospective.games["401856700"].display_distribution.upper_tail_probability = scale * 0.004;
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856700"]');
  await game.locator(".game-distribution-disclosure summary").click();
  await expect(game.locator(".distribution-tail-note")).toHaveCount(0);
});

test("uses actual future axis bounds and omits sub-1% tail notes", async ({ page }) => {
  const artifactPath = `site/data/team-seasons/${current}.json`;
  const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "../..", artifactPath), "utf8"));
  artifact.future_margin_axis.min_margin = -30;
  const lowTail = artifact.future_predictions["401856712"].display_distribution;
  const originalTail = lowTail.lower_tail_probability + lowTail.upper_tail_probability;
  lowTail.lower_tail_probability = 4;
  lowTail.upper_tail_probability = 4;
  lowTail.masses[0] += originalTail - 8;
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const visibleTail = page.locator('.game-card[data-game-id="401856705"]');
  await visibleTail.locator(".game-distribution-disclosure summary").click();
  await expect(visibleTail.locator(".distribution-tail-note")).toContainText("-30 to 40-point range");
  await expect(visibleTail.locator(".distribution-tail")).toHaveCount(0);
  const tinyTail = page.locator('.game-card[data-game-id="401856712"]');
  await tinyTail.locator(".game-distribution-disclosure summary").click();
  await expect(tinyTail.locator(".distribution-tail-note")).toHaveCount(0);
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
      const margin = summary.querySelector(".game-prediction-margin");
      const marginRange = document.createRange();
      marginRange.selectNodeContents(margin);
      const copyBox = copy.getBoundingClientRect();
      const chevronBox = chevron.getBoundingClientRect();
      return {
        copyHeight: copyBox.height,
        lineHeight: Number.parseFloat(getComputedStyle(copy).lineHeight),
        copyRight: copyBox.right,
        copyBottom: copyBox.bottom,
        chevronLeft: chevronBox.left,
        chevronTop: chevronBox.top,
        marginLines: marginRange.getClientRects().length,
      };
    });
    wrapped ||= layout.copyHeight > layout.lineHeight + 1;
    expect(layout.chevronLeft).toBeGreaterThanOrEqual(layout.copyRight - 1);
    expect(layout.chevronTop).toBeLessThan(layout.copyBottom);
    expect(layout.marginLines).toBe(1);
  }
  expect(wrapped).toBeTruthy();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1);
});
