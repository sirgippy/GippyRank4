const { test, expect } = require("@playwright/test");
const scheduleLayoutFixture = require("./fixtures/schedule-layout.json");

const scheduleCards = ".weekly-game-card";
const logoResponse = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40"><rect width="60" height="40" fill="#d6dfda"/></svg>`;

test.beforeEach(async ({ page }) => {
  // Keep the tests focused on the rendered site layout instead of third-party logo availability.
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

async function loadSchedule(page, path = "/schedule.html") {
  await page.goto(path);
  await expect(page.locator("#schedule-page-title")).toHaveText("Schedule");
  await expect(page.locator("#schedule-status")).toHaveText("");
  await expect(page.locator("#weekly-schedule")).toBeVisible();
}

async function expectGamesRendered(page) {
  await expect(page.locator(scheduleCards).first()).toBeVisible();
}

async function selectWeek(page, value) {
  await page.locator("#week-select").selectOption(value);
  await expect(page.locator("#week-select")).toHaveValue(value);
  await expect(page).toHaveURL(new RegExp(`[?&]week=${value}(?:&|$)`));
}

async function useScheduleLayoutFixture(page) {
  await page.route("**/data/manifest.json", (route) => route.fulfill({ json: scheduleLayoutFixture.manifest }));
  await page.route("**/data/week-games/browser-layout-fixture.json", (route) => route.fulfill({ json: scheduleLayoutFixture.artifact }));
}

async function weekWithMarqueeGame(page) {
  return page.evaluate(async () => {
    const manifest = await fetch("./data/manifest.json").then((response) => response.json());
    const current = new URL(window.location.href).searchParams;
    const entry = manifest.snapshots.find((snapshot) => snapshot.snapshot_id === current.get("snapshot"))
      ?? manifest.snapshots.find((snapshot) => snapshot.publication_slot === manifest.default_publication_slot && snapshot.ranking_family === "predictive" && snapshot.prior_family === "context");
    if (!entry) return null;
    const artifact = await fetch(`./${entry.week_games_path}`).then((response) => response.json());
    return artifact.weeks.find((week) => week.games.some((game) => game.marquee))?.key ?? null;
  });
}

test.describe("Schedule browser smoke tests", () => {
  test("loads and changes the selected week through the rendered controls", async ({ page }) => {
    await loadSchedule(page);
    await expectGamesRendered(page);

    const options = await page.locator("#week-select option").evaluateAll((elements) => elements.map((element) => ({
      label: element.textContent,
      value: element.value,
    })));
    expect(options.length).toBeGreaterThan(1);
    const initial = await page.locator("#week-select").inputValue();
    const target = options.find((option) => option.value !== initial);
    expect(target).toBeDefined();

    await selectWeek(page, target.value);
    await expectGamesRendered(page);
    expect(await page.locator("#week-select").inputValue()).toBe(target.value);
  });

  test("filters marquee games and restores all games", async ({ page }) => {
    await loadSchedule(page);
    const marqueeWeek = await weekWithMarqueeGame(page);
    expect(marqueeWeek).not.toBeNull();
    if (await page.locator("#week-select").inputValue() !== marqueeWeek) {
      await selectWeek(page, marqueeWeek);
    }
    await expectGamesRendered(page);

    const allCount = await page.locator(scheduleCards).count();
    expect(allCount).toBeGreaterThan(0);
    await page.getByRole("button", { name: "Marquee", exact: true }).click();
    await expect(page).toHaveURL(/[?&]view=marquee(?:&|$)/);
    await expect(page.getByRole("button", { name: "Marquee", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expectGamesRendered(page);
    const marqueeCount = await page.locator(scheduleCards).count();
    expect(marqueeCount).toBeGreaterThan(0);
    expect(marqueeCount).toBeLessThanOrEqual(allCount);

    await page.getByRole("button", { name: "All games", exact: true }).click();
    await expect(page).not.toHaveURL(/[?&]view=marquee(?:&|$)/);
    await expect(page.getByRole("button", { name: "All games", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expectGamesRendered(page);
    expect(await page.locator(scheduleCards).count()).toBe(allCount);
  });

  test("restores Marquee mode from a deep link", async ({ page }) => {
    await loadSchedule(page, "/schedule.html?view=marquee");
    await expect(page.getByRole("button", { name: "Marquee", exact: true })).toHaveAttribute("aria-pressed", "true");
    expect(new URL(page.url()).searchParams.get("view")).toBe("marquee");

    const marqueeCount = await page.locator(scheduleCards).count();
    await page.getByRole("button", { name: "All games", exact: true }).click();
    await expect(page).not.toHaveURL(/[?&]view=marquee(?:&|$)/);
    await expect(page.getByRole("button", { name: "All games", exact: true })).toHaveAttribute("aria-pressed", "true");
    await expectGamesRendered(page);
    expect(await page.locator(scheduleCards).count()).toBeGreaterThanOrEqual(marqueeCount);
  });

  test("keeps short team identities readable on both sides of controlled matchups", async ({ page }) => {
    await useScheduleLayoutFixture(page);
    await loadSchedule(page);
    await expect(page.locator("#week-select")).toHaveValue("fixture-week");
    await expectGamesRendered(page);

    const expectedMatchups = [
      ["Miami", "Alabama"],
      ["UCF", "Miami"],
      ["Alabama", "UCF"],
    ];
    const matchupNames = await page.locator(".weekly-matchup").evaluateAll((matchups) => matchups.slice(0, 3).map((matchup) => (
      [...matchup.querySelectorAll(":scope > .weekly-team-row .weekly-team-link")].map((link) => link.textContent.trim())
    )));
    expect(matchupNames).toEqual(expectedMatchups);

    for (const [teamName, count] of [["Miami", 2], ["Alabama", 2], ["UCF", 3]]) {
      const links = page.locator(".weekly-team-link").filter({ hasText: new RegExp(`^${teamName}$`) });
      await expect(links).toHaveCount(count);
      for (let index = 0; index < count; index += 1) {
        const link = links.nth(index);
        await expect(link).toBeVisible();
        const identity = link.locator("..");
        await expect(identity.locator(".team-logo-frame")).toHaveCount(1);
        await expect(identity.locator(".weekly-team-rank")).toHaveCount(1);
        await expect(identity).toContainText(teamName);

        const textLayout = await link.evaluate((element) => {
          const range = document.createRange();
          range.selectNodeContents(element);
          const style = getComputedStyle(element);
          const box = element.getBoundingClientRect();
          return {
            lineCount: range.getClientRects().length,
            height: box.height,
            width: box.width,
            lineHeight: Number.parseFloat(style.lineHeight),
          };
        });
        expect(textLayout.lineCount).toBe(1);
        expect(textLayout.height).toBeLessThanOrEqual(textLayout.lineHeight + 1);
        expect(textLayout.width).toBeGreaterThan(0);
        await expect(identity).toHaveCSS("display", "flex");
      }
    }

    const overflow = await page.evaluate(() => ({
      body: document.body.scrollWidth,
      document: document.documentElement.scrollWidth,
      viewport: window.innerWidth,
    }));
    expect(overflow.body).toBeLessThanOrEqual(overflow.viewport);
    expect(overflow.document).toBeLessThanOrEqual(overflow.viewport);
  });

  test("keeps the matchup horizontal at desktop width", async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "desktop", "This invariant is specific to the desktop viewport.");
    await loadSchedule(page);
    await expectGamesRendered(page);

    const layout = await page.locator(".weekly-matchup").first().evaluate((element) => {
      const rows = [...element.querySelectorAll(":scope > .weekly-team-row")].map((row) => {
        const box = row.getBoundingClientRect();
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom };
      });
      const separator = element.querySelector(":scope > .weekly-at").getBoundingClientRect();
      return {
        display: getComputedStyle(element).display,
        rows,
        separator: { left: separator.left, right: separator.right, top: separator.top, bottom: separator.bottom },
      };
    });
    expect(layout.display).toBe("grid");
    expect(layout.rows).toHaveLength(2);
    expect(Math.abs(layout.rows[0].top - layout.rows[1].top)).toBeLessThan(4);
    expect(layout.separator.left).toBeGreaterThan(layout.rows[0].left);
    expect(layout.separator.right).toBeLessThan(layout.rows[1].right);
  });
});
