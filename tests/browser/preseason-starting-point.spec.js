const { test, expect } = require("@playwright/test");
const { SITE_ORIGIN, installStaticSiteRoute } = require("./static-site");

const contextSnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-context";
const historySnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-history";
const performanceSnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-performance";
const currentContextSnapshot = "2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.3";
const interimContextSnapshot = "2026-weekly-2026-09-26T12-09-11.589245Z-context-v1.3";

const logoResponse = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" fill="#d6dfda"/></svg>`;

test.beforeEach(async ({ page }) => {
  await installStaticSiteRoute(page);
  await page.route("**/*", async (route) => {
    if (route.request().resourceType() === "image") {
      await route.fulfill({
        status: 200,
        contentType: "image/svg+xml",
        body: logoResponse,
      });
      return;
    }
    if (new URL(route.request().url()).origin !== SITE_ORIGIN) {
      await route.fulfill({ status: 204 });
      return;
    }
    await route.fallback();
  });
});

async function loadTeam(page, snapshot, prior = null, team = { id: "194", name: "Ohio State" }) {
  const params = new URLSearchParams({
    team: team.id,
    season: "2026",
    snapshot,
    family: prior ? "predictive" : "performance",
  });
  if (prior) params.set("prior", prior);
  await page.goto(`/team.html?${params}`);
  await expect(page.locator("#team-page-title")).toHaveText(team.name);
  await expect(page.locator("#team-page-status")).toHaveText(/scheduled games/);
}

test.describe("Preseason starting point browser checks", () => {
  test("explains Context and History beside the ranking selector", async ({ page }) => {
    await page.goto("/?family=predictive&season=2026&prior=context");
    await expect(page.locator("#rankings-body tr").first()).toBeVisible();
    await expect(page.locator("#prior-select legend")).toHaveText("Preseason starting point");
    await expect(page.locator("#prior-explanation")).toContainText("Program history");

    await page.locator("#prior-help summary").click();
    await expect(page.locator("#prior-help-text")).toContainText("recruiting");
    await expect(page.locator("#prior-help-text")).toContainText("game evidence");

    await page.getByRole("button", { name: "History", exact: true }).click();
    await expect(page).toHaveURL(/[?&]prior=history(?:&|$)/);
    await expect(page.locator("#prior-explanation")).toContainText("longer historical windows");
    await expect(page.locator("#prior-help-text")).toContainText("ignores current roster talent");
  });

  test("shows a compact season story and preserves prior selection", async ({ page }) => {
    await loadTeam(page, contextSnapshot, "context");
    await expect(page.locator("#season-story-section")).toBeVisible();
    await expect(page.locator("#season-story-content")).toContainText("Preseason → Sep 11 interim");
    await expect(page.locator("#season-story-content")).toContainText("Central 80%");
    await expect(page.locator("#season-movement-section")).toBeVisible();
    await expect(page.locator("#season-movement")).toContainText("Preseason");
    await expect(page.locator("#season-movement")).toContainText("Width");
    await expect(page.locator("#season-movement")).toContainText("80%");
    await expect(page.locator("#season-movement svg")).toHaveAttribute("role", "img");
    await expect(page.locator("#season-movement svg")).toHaveAttribute("aria-labelledby", /season-trajectory/);
    await expect(page.locator(".trajectory-uncertainty-band")).toBeVisible();
    await expect(page.locator("#team-context-link")).toHaveAttribute("aria-current", "page");

    await page.locator("#team-history-link").click();
    await expect(page).toHaveURL(/[?&]prior=history(?:&|$)/);
    await expect(page.locator("#team-page-title")).toHaveText("Ohio State");
    await expect(page.locator("#season-movement-context")).toContainText("History");
    await expect(page.locator("#season-movement svg")).toBeVisible();
  });

  test("treats an explicit History snapshot as authoritative without prior", async ({ page }) => {
    await page.goto(`/team.html?team=194&season=2026&family=predictive&snapshot=${historySnapshot}`);
    await expect(page.locator("#team-page-title")).toHaveText("Ohio State");
    await expect(page.locator("#team-page-status")).toHaveText(/scheduled games/);
    await expect(page.locator("#team-page-meta")).toContainText("Predictive History");
    await expect(page.locator("#season-movement-context")).toContainText("History");
    await expect(page.locator("#season-story-content")).toContainText("Preseason → Sep 11 interim");
    await expect(page.locator("#team-history-link")).toHaveAttribute("aria-current", "page");
  });

  test("shows team-specific preseason evidence and an expandable distribution", async ({ page }) => {
    await loadTeam(page, "2026-preseason-context", "context");
    await expect(page.locator("#preseason-starting-point-section")).toBeVisible();
    await expect(page.locator("#preseason-starting-point-content")).toContainText("recruiting");
    await expect(page.locator("#preseason-starting-point-content")).toContainText("Team talent composite");
    const distribution = page.locator("#preseason-starting-point-content .preseason-distribution-disclosure svg");
    await expect(distribution).toBeHidden();
    await page.locator("#preseason-starting-point-content .preseason-distribution-disclosure .scalar-distribution-summary").click();
    await expect(distribution).toBeVisible();
    await expect(page.locator("#season-movement-section")).toBeHidden();
  });

  test("shows Context 1.3's published transfer inputs with their provenance caveat", async ({ page }) => {
    await loadTeam(page, "2026-preseason-context-v1.3", "context");
    const evidence = page.locator("#preseason-starting-point-content");
    await expect(evidence).toContainText("Incoming prior offensive usage");
    await expect(evidence).toContainText("Incoming DB defensive impact");
    await expect(evidence).toContainText("DB transfer-data availability");
    await expect(evidence.getByText("2026 transfer inputs were reconstructed after the Aug. 15 cutoff.", { exact: false })).toBeVisible();
  });

  test("keeps the dashboard compact while preserving belief and distribution detail", async ({ page }) => {
    await loadTeam(page, "2026-weekly-2026-09-20T11-00-14.294077Z-context-v1.3", "context");
    await expect(page.locator("#team-ranking-summary")).not.toContainText("Expected final wins");
    await expect(page.locator("#season-outlook")).toContainText("Expected final wins");
    await expect(page.locator("#season-story-content")).toContainText("Preseason → Week 4");
    await expect(page.locator("#season-movement")).toContainText("Week 4");
    await expect(page.locator("#preseason-starting-point-section")).toBeVisible();
    await expect(page.locator("#preseason-starting-point-content")).toContainText("Incoming prior offensive usage");
    const distribution = page.locator("#season-story-content .season-story-distribution svg");
    await expect(distribution).toBeHidden();
    await page.locator("#season-story-content .season-story-distribution .scalar-distribution-summary").click();
    await expect(distribution).toBeVisible();
    await expect(page.locator(".schedule-checkpoint").first()).toContainText("published belief");
    await expect(page.locator(".schedule-checkpoint").first()).toContainText("Expected rank");
    await expect(page.locator(".game-card").first()).not.toContainText("published belief");
    expect(await page.locator(".team-page").evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBeTruthy();
    if (page.viewportSize()?.width < 760) {
      const columnCount = await page.locator("#team-ranking-summary .team-summary-grid").evaluate(
        (element) => getComputedStyle(element).gridTemplateColumns.trim().split(/\s+/).length,
      );
      expect(columnCount).toBe(2);
    }
  });

  test("does not fabricate a preseason comparison for Performance", async ({ page }) => {
    await loadTeam(page, performanceSnapshot);
    await expect(page.locator("#preseason-starting-point-section")).toBeHidden();
    await expect(page.locator("#season-movement-section")).toBeVisible();
    await expect(page.locator("#season-movement")).toContainText("does not show a preseason-to-current comparison");
    await expect(page.locator("#team-prior-switch")).toBeHidden();
  });

  test("renders Georgia's current story with an adaptive uncertainty trajectory", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "This overview hierarchy check uses the desktop presentation.");
    await loadTeam(page, currentContextSnapshot, "context", { id: "61", name: "Georgia" });
    await expect(page.locator("#season-story-content")).toContainText("Preseason → Week 5");
    await expect(page.locator("#season-story-content")).toContainText("8 ranks wide");
    await expect(page.locator("#season-story-content")).toContainText("narrower than preseason");
    const probabilityMetric = page.locator(".season-story-metric").filter({
      has: page.getByText("Top 5 probability", { exact: true }),
    });
    await expect(probabilityMetric).toContainText("49%");
    await expect(probabilityMetric).toContainText("77%");
    await expect(page.getByText("Top 25 probability", { exact: true })).toHaveCount(0);
    await expect(page.locator("#season-movement")).toContainText("Week 2");
    await expect(page.locator("#season-movement")).not.toContainText("Context 1.3 retrospective");
    const chart = page.locator(".season-trajectory-chart");
    await expect(chart).toHaveAttribute("data-rank-domain-min", "1");
    expect(Number(await chart.getAttribute("data-rank-domain-max"))).toBeLessThan(40);
    await expect(page.locator(".preseason-input-groups")).toBeVisible();
    await expect(page.locator(".preseason-input-group-recruiting .preseason-input-value")).toHaveCount(6);
    await expect(page.locator("#season-outlook")).toContainText("Expected final wins");
  });

  test("keeps the current Georgia overview readable on mobile without horizontal overflow", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile", "This check is specific to the narrow presentation.");
    await loadTeam(page, currentContextSnapshot, "context", { id: "61", name: "Georgia" });
    await expect(page.locator("#season-story-section")).toBeVisible();
    const chevron = page.locator("#season-story-content .season-story-distribution .scalar-distribution-chevron");
    const relation = await chevron.evaluate((element) => ({
      summaryTop: element.closest("summary").getBoundingClientRect().top,
      chevronTop: element.getBoundingClientRect().top,
    }));
    expect(Math.abs(relation.chevronTop - relation.summaryTop)).toBeLessThan(3);
    const overflow = await page.evaluate(() => ({
      body: document.body.scrollWidth,
      document: document.documentElement.scrollWidth,
      viewport: window.innerWidth,
    }));
    expect(overflow.body).toBeLessThanOrEqual(overflow.viewport);
    expect(overflow.document).toBeLessThanOrEqual(overflow.viewport);
  });

  test("keeps a selected interim checkpoint while omitting older interim publications", async ({ page }) => {
    await loadTeam(page, interimContextSnapshot, "context", { id: "61", name: "Georgia" });
    const checkpoints = page.locator(".trajectory-point-item");
    await expect(checkpoints.last()).toContainText("Sep 26 interim");
    await expect(page.locator("#season-movement")).not.toContainText("Sep 19 interim");
    await expect(page.locator("#season-movement")).not.toContainText("Context 1.3 retrospective");
  });

  test("uses a wider adaptive domain for a materially different team", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "This adaptive-domain comparison uses desktop geometry.");
    await loadTeam(page, currentContextSnapshot, "context", { id: "113", name: "Massachusetts" });
    const domainMaximum = Number(await page.locator(".season-trajectory-chart").getAttribute("data-rank-domain-max"));
    expect(domainMaximum).toBeGreaterThan(100);
    await expect(page.locator(".trajectory-uncertainty-band")).toBeVisible();
    await expect(page.locator("#season-story-content .season-story-metric")).toHaveCount(2);
    await expect(page.locator("#season-story-content")).not.toContainText(/Top (5|10|25) probability/);
  });
});
