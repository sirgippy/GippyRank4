const { test, expect } = require("@playwright/test");

const contextSnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-context";
const historySnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-history";
const performanceSnapshot = "2026-weekly-2026-09-11T17-02-26.077461Z-performance";

const logoResponse = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" fill="#d6dfda"/></svg>`;

test.beforeEach(async ({ page }) => {
  await page.route("**/*", async (route) => {
    if (route.request().resourceType() === "image") {
      await route.fulfill({
        status: 200,
        contentType: "image/svg+xml",
        body: logoResponse,
      });
      return;
    }
    await route.continue();
  });
});

async function loadTeam(page, snapshot, prior = null) {
  const params = new URLSearchParams({
    team: "194",
    season: "2026",
    snapshot,
    family: prior ? "predictive" : "performance",
  });
  if (prior) params.set("prior", prior);
  await page.goto(`/team.html?${params}`);
  await expect(page.locator("#team-page-title")).toHaveText("Ohio State");
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

  test("shows exact preseason-to-current movement and preserves prior selection", async ({ page }) => {
    await loadTeam(page, contextSnapshot, "context");
    await expect(page.locator("#season-movement-section")).toBeVisible();
    await expect(page.locator("#season-movement")).toContainText("Preseason");
    await expect(page.locator("#season-movement")).toContainText("80% range");
    await expect(page.locator("#season-movement svg")).toHaveAttribute("role", "img");
    await expect(page.locator("#season-movement svg")).toHaveAttribute("aria-labelledby", /season-movement/);
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
    await expect(page.locator("#season-movement")).toContainText("Published Preseason History distribution");
    await expect(page.locator("#team-history-link")).toHaveAttribute("aria-current", "page");
  });

  test("shows the published preseason distribution without duplicating movement", async ({ page }) => {
    await loadTeam(page, "2026-preseason-context", "context");
    await expect(page.locator("#preseason-starting-point-section")).toBeVisible();
    await expect(page.locator("#preseason-starting-point-content")).toContainText("recruiting");
    await expect(page.locator("#preseason-starting-point-content svg")).toBeVisible();
    await expect(page.locator("#season-movement-section")).toBeHidden();
  });

  test("does not fabricate a preseason comparison for Performance", async ({ page }) => {
    await loadTeam(page, performanceSnapshot);
    await expect(page.locator("#preseason-starting-point-section")).toBeHidden();
    await expect(page.locator("#season-movement-section")).toBeVisible();
    await expect(page.locator("#season-movement")).toContainText("does not show a preseason-to-current comparison");
    await expect(page.locator("#team-prior-switch")).toBeHidden();
  });
});
