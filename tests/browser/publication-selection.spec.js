const { test, expect } = require("@playwright/test");
const manifest = require("../../site/data/manifest.json");
const { SITE_ORIGIN, installStaticSiteRoute } = require("./static-site");

const surfaces = ["rankings", "schedule", "team"];
const historyEntry = manifest.snapshots.find((entry) => entry.season === 2026 && entry.ranking_family === "predictive" && entry.prior_family === "history" && entry.publication_slot === manifest.default_publication_slot);
const contextEntry = manifest.snapshots.find((entry) => entry.season === 2026 && entry.ranking_family === "predictive" && entry.prior_family === "context" && entry.publication_slot === manifest.default_publication_slot);
const qualityOfPlayEntry = manifest.snapshots.find((entry) => entry.season === 2026 && entry.ranking_family === "performance" && (entry.performance_variant ?? "quality_of_play") === "quality_of_play" && entry.publication_slot === manifest.default_publication_slot);
const qualityOfRecordEntry = manifest.snapshots.find((entry) => entry.season === 2026 && entry.ranking_family === "performance" && entry.performance_variant === "quality_of_record" && entry.publication_slot === manifest.default_publication_slot);
const alternateQualityOfRecordEntry = manifest.snapshots.find((entry) => entry.season === 2026 && entry.ranking_family === "performance" && entry.performance_variant === "quality_of_record" && entry.publication_slot !== manifest.default_publication_slot);
const missingVariantSlot = manifest.default_publication_slot;

test.beforeEach(async ({ page }) => {
  await installStaticSiteRoute(page);
  await page.route("**/*", async (route) => {
    if (route.request().resourceType() === "image") {
      await route.fulfill({ status: 200, contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg"/>' });
      return;
    }
    if (new URL(route.request().url()).origin !== SITE_ORIGIN) {
      await route.fulfill({ status: 204 });
      return;
    }
    await route.fallback();
  });
});

function urlFor(surface, query) {
  const path = surface === "rankings" ? "/" : surface === "schedule" ? "/schedule.html" : "/team.html";
  const params = new URLSearchParams(query);
  if (surface === "team") params.set("team", "61");
  return `${path}?${params.toString()}`;
}

function statusFor(page, surface) {
  return page.locator(surface === "rankings" ? "#snapshot-status" : surface === "schedule" ? "#schedule-status" : "#team-page-status");
}

function rowsFor(page, surface) {
  return page.locator(surface === "rankings" ? "#rankings-body tr" : surface === "schedule" ? ".weekly-game-card" : "#schedule-list .game-card");
}

async function expectResolved(page, surface, entry, familyText) {
  if (surface === "rankings") {
    await expect(page.locator("#snapshot-title")).toContainText(familyText);
    await expect(page.locator("#rankings-body tr").first()).toBeVisible();
    await expect(page.locator("#snapshot-select")).toHaveValue(entry.publication_slot);
  } else if (surface === "schedule") {
    await expect(page.locator("#weekly-schedule .weekly-game-card").first()).toBeVisible();
    await expect(page.locator("#snapshot-select")).toHaveValue(entry.publication_slot);
    if (entry.ranking_family === "predictive") {
      await expect(page.locator(`[data-prior="${entry.prior_family}"]`)).toHaveClass(/is-active/);
    } else {
      await expect(page.locator(`[data-variant="${entry.performance_variant ?? "quality_of_play"}"]`)).toHaveAttribute("aria-pressed", "true");
    }
  } else {
    await expect(page.locator("#team-page-meta")).toContainText(familyText);
    await expect(page.locator("#schedule-list .game-card").first()).toBeVisible();
  }
  if (surface === "schedule") await expect(statusFor(page, surface)).toHaveText("");
  else await expect(statusFor(page, surface)).toBeHidden();
}

for (const surface of surfaces) {
  test(`${surface} resolves an exact snapshot ID without requiring omitted selectors`, async ({ page }) => {
    expect(historyEntry).toBeTruthy();
    await page.goto(urlFor(surface, { season: "2026", snapshot: historyEntry.snapshot_id }));
    await expectResolved(page, surface, historyEntry, "History");
  });

  test(`${surface} resolves an explicit Performance publication slot for the requested variant`, async ({ page }) => {
    expect(qualityOfRecordEntry).toBeTruthy();
    await page.goto(urlFor(surface, {
      season: "2026",
      family: "performance",
      variant: "quality_of_record",
      snapshot: qualityOfRecordEntry.publication_slot,
    }));
    await expectResolved(page, surface, qualityOfRecordEntry, "Quality of Record");
  });

  test(`${surface} resolves a legacy slot link and remains valid after reload`, async ({ page }) => {
    expect(qualityOfRecordEntry).toBeTruthy();
    await page.goto(urlFor(surface, {
      season: "2026",
      family: "performance",
      variant: "quality_of_record",
      slot: qualityOfRecordEntry.publication_slot,
    }));
    await expectResolved(page, surface, qualityOfRecordEntry, "Quality of Record");

    const currentUrl = new URL(page.url());
    if (surface === "team") {
      expect(currentUrl.searchParams.get("slot")).toBe(qualityOfRecordEntry.publication_slot);
    } else {
      expect(currentUrl.searchParams.has("slot")).toBe(false);
      expect(currentUrl.searchParams.get("snapshot")).toBe(qualityOfRecordEntry.snapshot_id);
    }

    await page.reload();
    await expectResolved(page, surface, qualityOfRecordEntry, "Quality of Record");
  });

  test(`${surface} shows an unavailable state for an unknown snapshot and keeps its URL`, async ({ page }) => {
    const requested = "issue-239-missing-snapshot";
    for (const parameter of ["snapshot", "slot"]) {
      await page.goto(urlFor(surface, { season: "2026", family: "predictive", prior: "context", [parameter]: requested }));
      await expect(statusFor(page, surface)).toBeVisible();
      await expect(statusFor(page, surface)).toContainText(requested);
      await expect(rowsFor(page, surface)).toHaveCount(0);
      await expect(page).toHaveURL(new RegExp(`${parameter}=${requested}`));
      if (surface !== "team") {
        await expect(page.locator("#snapshot-select")).toHaveValue(requested);
        await expect(page.locator("#snapshot-select option:checked")).toHaveText(`Unavailable request: ${requested}`);
      }
    }
  });

  test(`${surface} rejects mismatched family, prior, and variant selectors`, async ({ page }) => {
    const requests = [
      {
        query: { season: "2026", family: "performance", snapshot: historyEntry.snapshot_id },
        expected: "belongs to Predictive",
        entry: historyEntry,
      },
      {
        query: { season: "2026", family: "predictive", prior: "context", snapshot: historyEntry.snapshot_id },
        expected: "uses history starting point",
        entry: historyEntry,
      },
      {
        query: { season: "2026", family: "performance", variant: "quality_of_record", snapshot: qualityOfPlayEntry.snapshot_id },
        expected: "is Quality of Play",
        entry: qualityOfPlayEntry,
      },
      {
        query: { season: "2026", family: "predictive", variant: "quality_of_record", snapshot: contextEntry.publication_slot },
        expected: "applies to Performance publications",
        entry: contextEntry,
      },
    ];
    for (const request of requests) {
      await page.goto(urlFor(surface, request.query));
      await expect(statusFor(page, surface)).toBeVisible();
      await expect(statusFor(page, surface)).toContainText(request.expected);
      await expect(rowsFor(page, surface)).toHaveCount(0);
      await expect(page).toHaveURL(new RegExp(`snapshot=${request.query.snapshot.replaceAll(".", "\\.")}`));
    }
  });

  test(`${surface} explains an unavailable variant and offers an explicit available publication`, async ({ page }) => {
    const withoutRecord = structuredClone(manifest);
    withoutRecord.snapshots = withoutRecord.snapshots.filter((entry) => !(
      entry.season === 2026
      && entry.publication_slot === missingVariantSlot
      && entry.ranking_family === "performance"
      && entry.performance_variant === "quality_of_record"
    ));
    await page.route("**/data/manifest.json", (route) => route.fulfill({ json: withoutRecord }));
    await page.goto(urlFor(surface, {
      season: "2026",
      family: "performance",
      variant: "quality_of_record",
      snapshot: missingVariantSlot,
    }));
    const status = statusFor(page, surface);
    await expect(status).toBeVisible();
    await expect(status).toContainText("Quality of Record is unavailable");
    await expect(status).toContainText("Quality of Play");
    await expect(rowsFor(page, surface)).toHaveCount(0);
    await expect(page).toHaveURL(new RegExp(`snapshot=${missingVariantSlot}`));

    if (surface === "rankings") {
      await expect(page.locator('[data-variant="quality_of_play"]')).toBeEnabled();
      await page.locator('[data-variant="quality_of_play"]').click();
      await expect(page.locator("#snapshot-title")).toContainText("Quality of Play");
      await expect(page.locator("#rankings-body tr").first()).toBeVisible();
    } else {
      const alternative = status.locator("a").filter({ hasText: "Quality of Play" }).first();
      await expect(alternative).toBeVisible();
      await alternative.click();
      await expectResolved(page, surface, qualityOfPlayEntryForSlot(missingVariantSlot), "Quality of Play");
    }
  });
}

function qualityOfPlayEntryForSlot(slot) {
  return manifest.snapshots.find((entry) => entry.season === 2026 && entry.publication_slot === slot && entry.ranking_family === "performance" && (entry.performance_variant ?? "quality_of_play") === "quality_of_play");
}

test("rankings clears a legacy slot while switching variants and publications", async ({ page }) => {
  expect(qualityOfPlayEntry).toBeTruthy();
  expect(qualityOfRecordEntry).toBeTruthy();
  expect(alternateQualityOfRecordEntry).toBeTruthy();
  await page.goto(urlFor("rankings", {
    season: "2026",
    family: "performance",
    variant: "quality_of_play",
    slot: qualityOfPlayEntry.publication_slot,
  }));
  await expectResolved(page, "rankings", qualityOfPlayEntry, "Quality of Play");

  await page.getByRole("button", { name: "Quality of Record", exact: true }).click();
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  let currentUrl = new URL(page.url());
  expect(currentUrl.searchParams.has("slot")).toBe(false);
  expect(currentUrl.searchParams.get("snapshot")).toBe(qualityOfRecordEntry.snapshot_id);

  await page.locator("#snapshot-select").selectOption(alternateQualityOfRecordEntry.publication_slot);
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  currentUrl = new URL(page.url());
  expect(currentUrl.searchParams.has("slot")).toBe(false);
  expect(currentUrl.searchParams.get("snapshot")).toBe(alternateQualityOfRecordEntry.snapshot_id);
  await page.reload();
  await expectResolved(page, "rankings", alternateQualityOfRecordEntry, "Quality of Record");
});

test("Schedule clears legacy slot parameters when switching publications and variants", async ({ page }) => {
  expect(qualityOfRecordEntry).toBeTruthy();
  expect(qualityOfPlayEntry).toBeTruthy();
  expect(alternateQualityOfRecordEntry).toBeTruthy();
  const legacyScheduleQuery = {
    season: "2026",
    family: "performance",
    variant: "quality_of_record",
    slot: qualityOfRecordEntry.publication_slot,
    week: qualityOfRecordEntry.default_week,
    view: "all",
  };
  await page.goto(urlFor("schedule", legacyScheduleQuery));
  await expectResolved(page, "schedule", qualityOfRecordEntry, "Quality of Record");

  await page.locator("#snapshot-select").selectOption(alternateQualityOfRecordEntry.publication_slot);
  await expectResolved(page, "schedule", alternateQualityOfRecordEntry, "Quality of Record");
  let currentUrl = new URL(page.url());
  expect(currentUrl.searchParams.has("slot")).toBe(false);
  expect(currentUrl.searchParams.get("snapshot")).toBe(alternateQualityOfRecordEntry.publication_slot);
  await page.reload();
  await expectResolved(page, "schedule", alternateQualityOfRecordEntry, "Quality of Record");

  await page.goto(urlFor("schedule", legacyScheduleQuery));
  await expectResolved(page, "schedule", qualityOfRecordEntry, "Quality of Record");
  await page.locator('[data-variant="quality_of_play"]').click();
  await expectResolved(page, "schedule", qualityOfPlayEntry, "Quality of Play");
  currentUrl = new URL(page.url());
  expect(currentUrl.searchParams.has("slot")).toBe(false);
  expect(currentUrl.searchParams.get("variant")).toBe("quality_of_play");
});

test("team Performance variant links clear the legacy slot parameter", async ({ page }) => {
  expect(qualityOfPlayEntry).toBeTruthy();
  expect(qualityOfRecordEntry).toBeTruthy();
  await page.goto(urlFor("team", {
    season: "2026",
    family: "performance",
    variant: "quality_of_play",
    slot: qualityOfPlayEntry.publication_slot,
  }));
  await expectResolved(page, "team", qualityOfPlayEntry, "Quality of Play");
  const recordLink = page.locator('[data-variant-link="quality_of_record"]');
  const linkUrl = new URL(await recordLink.getAttribute("href"), page.url());
  expect(linkUrl.searchParams.has("slot")).toBe(false);

  await recordLink.click();
  await expectResolved(page, "team", qualityOfRecordEntry, "Quality of Record");
  let currentUrl = new URL(page.url());
  expect(currentUrl.searchParams.has("slot")).toBe(false);
  expect(currentUrl.searchParams.get("snapshot")).toBe(qualityOfRecordEntry.snapshot_id);
  await page.reload();
  await expectResolved(page, "team", qualityOfRecordEntry, "Quality of Record");
});
