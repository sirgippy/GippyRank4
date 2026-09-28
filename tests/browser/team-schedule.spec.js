const fs = require("node:fs");
const path = require("node:path");
const { test, expect } = require("@playwright/test");
const { installStaticSiteRoute } = require("./static-site");

const current = "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3";
const currentPerformance = `${current}-performance`;
const currentHistory = "2026-weekly-2026-09-27T12-27-35.698895Z-history";
const historical = "2026-weekly-2026-09-26T12-09-11.589245Z-context-v1.3";
const logoResponse = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" rx="5" fill="#e0e8e2"/><path d="M12 26h36" stroke="#9bafa3" stroke-width="3"/></svg>';

function siteArtifact(folder, snapshot) {
  return JSON.parse(fs.readFileSync(path.join(__dirname, "../..", `site/data/${folder}/${snapshot}.json`), "utf8"));
}

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
  await expect(page.getByRole("group", { name: "Scheduled games and model updates" }).getByRole("article")).toHaveCount(12);
  await expect(page.getByRole("group", { name: "Scheduled games and model updates" }).getByRole("article", { name: "Oklahoma" })).toHaveCount(1);
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
    .toContainText("16% of expected outcomes extend beyond Rutgers by 40");
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
    .toHaveText("Georgia 50.1% to win · Georgia by 0.1");
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
  await expect(game).toContainText("Retrospective expectation unavailable because published game data conflict");
  await expect(game.locator(".game-retrospective")).toHaveCount(0);
});

test("refuses a future forecast whose matchup orientation conflicts with the schedule", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  const prediction = artifact.future_predictions["401856712"];
  [prediction.home_team_id, prediction.away_team_id] = [prediction.away_team_id, prediction.home_team_id];
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856712"]');
  await expect(game).toContainText("Prediction unavailable because published game data conflict");
  await expect(game.locator(".game-prediction")).toHaveCount(0);
});

test("refuses a future forecast with a conflicting displayed team name", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  artifact.future_predictions["401856712"].away_team_name = "Auburn";
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856712"]');
  await expect(game).toContainText("Prediction unavailable because published game data conflict");
  await expect(game.locator(".game-prediction")).toHaveCount(0);
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
  const expectedMargin = Number(await retrospective.locator(".margin-marker-expected").getAttribute("data-margin"));
  expect(expectedMargin).toBeCloseTo(11.298375740164296, 10);
  await expect(retrospective.locator(".margin-marker-actual")).toHaveAttribute("data-margin", "28");
  await expect(retrospective.locator(".margin-marker-legend")).toContainText("Expected Georgia by 11.3");
  await expect(retrospective.locator(".margin-marker-legend")).toContainText("Actual Georgia by 28");
  await expect(retrospective.locator(".retrospective-distribution-chart .distribution-zero-label")).toHaveCount(0);
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
    .toContainText("4% of expected outcomes extend beyond Georgia by 40");
  const future = page.locator('.game-card[data-game-id="401856705"]');
  await expect(future.locator(".game-distribution-disclosure summary"))
    .toHaveAccessibleName(/Georgia 91% to win.*Georgia by 20\.7.*Show predictive margin distribution/);
  await future.locator(".game-distribution-disclosure summary").click();
  await expect(future.locator(".future-distribution-chart")).toBeVisible();
  await expect(future.locator(".future-distribution-chart .distribution-zero-label")).toHaveCount(0);
  await expect(future.locator(".distribution-axis .axis-label")).toContainText("Even");
  await expect(future.locator(".distribution-tail")).toHaveCount(0);
  await expect(future.locator(".distribution-tail-note")).toContainText("predicted outcomes extend beyond Georgia by 40");
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
  const expected = game.locator(".margin-marker-expected");
  const actual = game.locator(".margin-marker-actual");
  expect(Number(await expected.getAttribute("x1"))).toBeLessThan(Number(await actual.getAttribute("x1")));
  await expect(game.locator(".distribution-tail-note")).toContainText("82% of expected outcomes extend beyond Georgia by 40");
  await expect(game.locator(".margin-interval-legend")).toHaveText("Central 95%Central 80%Central 50%");
});

test("orients an off-scale away interval view from the selected team's perspective", async ({ page }) => {
  await loadTeam(page, "189");
  const game = page.locator('.game-card[data-game-id="401858441"]');
  await expect(game).toContainText("Expected Nebraska by 32.7");
  await expect(game).toContainText("Actual Nebraska by 49");
  await game.locator(".game-distribution-disclosure summary").click();
  await expect(game.locator(".retrospective-interval-chart")).toBeVisible();
  const expected = Number(await game.locator(".margin-marker-expected").getAttribute("x1"));
  const actual = Number(await game.locator(".margin-marker-actual").getAttribute("x1"));
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
  await expect(game.locator(".distribution-tail-note")).toContainText("71% of predicted outcomes extend beyond LSU by 40");
  await expect(game.locator(".game-facts")).toContainText("LSU win probability99.8%");
  await expect(game.locator(".future-interval-chart desc")).toContainText("McNeese win probability 0.2%");
});

for (const [caseName, bounds, widened] of [
  ["inside", [-39.9, 39.9], false],
  ["touching", [-40, 40], false],
  ["outside", [-40, 40.001], true],
]) {
  test(`uses the standard-axis boundary contract when the central half is ${caseName}`, async ({ page }) => {
    const artifact = siteArtifact("team-seasons", current);
    artifact.retrospective_game_expectations.games["401856700"].margin_interval_50 = bounds;
    await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
    await loadTeam(page, "61");
    const game = page.locator('.game-card[data-game-id="401856700"]');
    await game.locator("summary").click();
    await expect(game.locator(widened ? ".retrospective-interval-chart" : ".retrospective-distribution-chart")).toBeVisible();
  });

  test(`uses the future standard-axis boundary when the central half is ${caseName}`, async ({ page }) => {
    const artifact = siteArtifact("team-seasons", current);
    artifact.future_predictions["401856712"].margin_interval_50 = bounds;
    await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
    await loadTeam(page, "61");
    const game = page.locator('.game-card[data-game-id="401856712"]');
    await game.locator("summary").click();
    await expect(game.locator(widened ? ".future-interval-chart" : ".future-distribution-chart")).toBeVisible();
  });
}

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
    const expected = game.locator(".margin-marker-expected");
    const actual = game.locator(".margin-marker-actual");
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
      const separator = margin.querySelector(".game-summary-separator");
      const copyBox = copy.getBoundingClientRect();
      const chevronBox = chevron.getBoundingClientRect();
      return {
        copyHeight: copyBox.height,
        lineHeight: Number.parseFloat(getComputedStyle(copy).lineHeight),
        copyRight: copyBox.right,
        copyBottom: copyBox.bottom,
        chevronLeft: chevronBox.left,
        chevronTop: chevronBox.top,
        separatorVisible: getComputedStyle(separator).display !== "none",
      };
    });
    wrapped ||= layout.copyHeight > layout.lineHeight + 1;
    expect(layout.chevronLeft).toBeGreaterThanOrEqual(layout.copyRight - 1);
    expect(layout.chevronLeft - layout.copyRight).toBeLessThanOrEqual(12);
    expect(layout.chevronTop).toBeLessThan(layout.copyBottom);
    expect(layout.separatorVisible).toBeTruthy();
  }
  expect(wrapped).toBeTruthy();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1);
});

test("keeps desktop disclosure copy beside its chevron and aligned with the matchup", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === "mobile", "This checks the desktop composition.");
  await loadTeam(page, "61");
  for (const gameId of ["401856658", "401856700", "401856705"]) {
    const card = page.locator(`.game-card[data-game-id="${gameId}"]`);
    const layout = await card.evaluate((element) => {
      const copy = element.querySelector(".game-summary-copy").getBoundingClientRect();
      const chevron = element.querySelector(".distribution-chevron").getBoundingClientRect();
      const heading = element.querySelector(".game-opponent-row").getBoundingClientRect();
      return { gap: chevron.left - copy.right, indent: copy.left - heading.left };
    });
    expect(layout.gap).toBeGreaterThanOrEqual(0);
    expect(layout.gap).toBeLessThanOrEqual(12);
    expect(layout.indent).toBeLessThanOrEqual(20);
  }
});

test("places a checkpoint after included evidence even when the next kickoff precedes its cutoff", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  const trajectory = siteArtifact("team-trajectories", current);
  const [first, second] = artifact.teams["61"].games;
  second.date = first.date;
  const before = structuredClone(trajectory.points[0]);
  const after = structuredClone(trajectory.points.at(-1));
  after.included_game_ids = [first.game_id];
  trajectory.points = [before, after];
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await page.route(`**/data/team-trajectories/${current}.json`, (route) => route.fulfill({ json: trajectory }));
  await loadTeam(page, "61");
  const entries = page.locator("#schedule-list > .schedule-entry");
  await expect(entries.nth(0).locator(".schedule-checkpoint")).toHaveCount(0);
  await expect(page.locator("#schedule-list > .schedule-checkpoint")).toHaveCount(1);
  await expect(page.locator("#schedule-list > :nth-child(2)")).toHaveClass(/schedule-checkpoint/);
  await expect(entries.nth(1)).toHaveAttribute("aria-labelledby", /schedule-game/);
});

test("orders multiple pregame checkpoints outside the named game articles", async ({ page }) => {
  const trajectory = siteArtifact("team-trajectories", current);
  const before = structuredClone(trajectory.points[0]);
  const middle = structuredClone(trajectory.points.at(-2));
  const after = structuredClone(trajectory.points.at(-1));
  middle.included_game_ids = [];
  after.included_game_ids = [];
  middle.display_label = "First interim";
  after.display_label = "Second interim";
  middle.effective_cutoff = null;
  after.effective_cutoff = null;
  trajectory.points = [before, middle, after];
  await page.route(`**/data/team-trajectories/${current}.json`, (route) => route.fulfill({ json: trajectory }));
  await loadTeam(page, "61");
  await expect(page.locator("#schedule-list > :nth-child(1)")).toContainText("First interim update");
  await expect(page.locator("#schedule-list > :nth-child(2)")).toContainText("Second interim update");
  await expect(page.locator("#schedule-list > :nth-child(3)")).toHaveClass(/schedule-entry/);
  await expect(page.locator(".schedule-entry .schedule-checkpoint")).toHaveCount(0);
});

test("describes evidence removed at a checkpoint", async ({ page }) => {
  const trajectory = siteArtifact("team-trajectories", current);
  const before = structuredClone(trajectory.points.at(-1));
  const after = structuredClone(before);
  before.snapshot_id = "before-removal-test";
  const gameId = "401856700";
  expect(before.included_game_ids).toContain(gameId);
  after.included_game_ids = before.included_game_ids.filter((id) => id !== gameId);
  after.snapshot_id = current;
  trajectory.points = [before, after];
  await page.route(`**/data/team-trajectories/${current}.json`, (route) => route.fulfill({ json: trajectory }));
  await loadTeam(page, "61");
  await expect(page.locator(".schedule-checkpoint")).toContainText("1 Georgia game removed from evidence");
});

test("does not attach an unrelated global removal to a team's checkpoint", async ({ page }) => {
  const trajectory = siteArtifact("team-trajectories", current);
  const before = structuredClone(trajectory.points[0]);
  const after = structuredClone(trajectory.points.at(-1));
  before.included_game_ids = ["unrelated-game"];
  after.included_game_ids = ["401856700"];
  trajectory.points = [before, after];
  await page.route(`**/data/team-trajectories/${current}.json`, (route) => route.fulfill({ json: trajectory }));
  await loadTeam(page, "61");
  const checkpoint = page.locator(".schedule-checkpoint");
  await expect(checkpoint).toHaveCount(1);
  await expect(checkpoint).not.toContainText("removed from evidence");
});

test("describes a checkpoint with no newly included games", async ({ page }) => {
  const trajectory = siteArtifact("team-trajectories", current);
  const before = structuredClone(trajectory.points[0]);
  const after = structuredClone(trajectory.points.at(-1));
  after.included_game_ids = [];
  trajectory.points = [before, after];
  await page.route(`**/data/team-trajectories/${current}.json`, (route) => route.fulfill({ json: trajectory }));
  await loadTeam(page, "61");
  const checkpoint = page.locator(".schedule-checkpoint");
  await expect(checkpoint).toHaveAttribute("data-evidence-change", "none");
  await expect(checkpoint).toContainText("No new games included");
  await expect(page.getByRole("group", { name: "Scheduled games and model updates" }).getByRole("article")).toHaveCount(12);
});

test.describe("schedule date semantics", () => {
  test.use({ timezoneId: "America/Los_Angeles" });
  test("uses the local calendar day across the UTC boundary", async ({ page }) => {
    const artifact = siteArtifact("team-seasons", current);
    artifact.teams["61"].games[0].date = "2026-09-06T00:30:00Z";
    await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
    await loadTeam(page, "61");
    await expect(page.locator(".game-card").first().locator(".game-date")).toContainText("Sep 5");
    await expect(page.locator('.game-card[data-game-id="401856712"] .game-date')).toContainText("Oct 10");
    await expect(page.locator("#team-page-meta")).toContainText("Through Sep 27");
  });
});

test("formats complementary matchup probabilities as a coherent pair", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  const close = artifact.future_predictions["401856712"];
  close.home_win_probability = 0.505;
  close.away_win_probability = 0.495;
  const nearCertain = artifact.future_predictions["401856705"];
  nearCertain.home_win_probability = 0.9996;
  nearCertain.away_win_probability = 0.0004;
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const closeGame = page.locator('.game-card[data-game-id="401856712"]');
  await closeGame.locator("summary").click();
  await expect(closeGame.locator(".game-prediction-title")).toContainText("50.5% to win");
  await expect(closeGame.locator(".game-facts")).toContainText("50.5%");
  await expect(closeGame.locator(".game-facts")).toContainText("49.5%");
  const certainGame = page.locator('.game-card[data-game-id="401856705"]');
  await expect(certainGame.locator(".game-prediction-title")).toContainText("99.96% to win");
  await certainGame.locator("summary").click();
  await expect(certainGame.locator(".game-facts")).not.toContainText("100.0%");
});

test("treats sub-display-precision matchup differences as near even", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  const prediction = artifact.future_predictions["401856712"];
  prediction.home_win_probability = 0.50000001;
  prediction.away_win_probability = 0.49999999;
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856712"]');
  await expect(game.locator(".game-prediction-title")).toContainText("Near-even matchup");
  await game.locator("summary").click();
  await expect(game.locator(".game-facts dd").first()).toHaveText("50%");
  await expect(game.locator(".game-facts dd").nth(1)).toHaveText("50%");
});

test("labels zero and separates equal markers in a widened crossing-zero interval", async ({ page }) => {
  const artifact = siteArtifact("team-seasons", current);
  const expectation = artifact.retrospective_game_expectations.games["401856658"];
  expectation.expected_home_margin = expectation.actual_home_margin;
  expectation.margin_interval_50 = [-10, 70];
  expectation.margin_interval_80 = [-20, 80];
  expectation.margin_interval_95 = [-30, 90];
  await page.route(`**/data/team-seasons/${current}.json`, (route) => route.fulfill({ json: artifact }));
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856658"]');
  await game.locator("summary").click();
  await expect(game.locator(".retrospective-interval-chart .distribution-zero")).toHaveCount(1);
  await expect(game.locator(".retrospective-interval-chart .distribution-zero-label")).toHaveText("Even");
  const expected = game.locator(".margin-marker-expected");
  const actual = game.locator(".margin-marker-actual");
  expect(await expected.getAttribute("x1")).toBe(await actual.getAttribute("x1"));
  expect(Number(await expected.getAttribute("y2"))).toBeLessThan(Number(await actual.getAttribute("y1")));
  await expect(game.locator(".margin-interval-legend")).toHaveText("Central 95%Central 80%Central 50%");
});

test("keeps the widened interval and facts within a 320px viewport", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile", "This is a narrow-layout regression.");
  await page.setViewportSize({ width: 320, height: 844 });
  await loadTeam(page, "61");
  const game = page.locator('.game-card[data-game-id="401856658"]');
  await game.locator("summary").click();
  await expect(game.locator(".retrospective-interval-chart")).toBeVisible();
  await expect(game.locator(".margin-interval-legend")).toBeVisible();
  await expect(game.locator(".retrospective-facts")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
});
