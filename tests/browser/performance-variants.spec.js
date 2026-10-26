const { test, expect } = require("@playwright/test");
const { SITE_ORIGIN, installStaticSiteRoute } = require("./static-site");
const manifest = require("../../site/data/manifest.json");

test.beforeEach(async ({ page }) => {
  await installStaticSiteRoute(page);
  await page.route("**/*", async (route) => {
    if (route.request().resourceType() === "image") {
      await route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg"/>' });
    } else if (new URL(route.request().url()).origin !== SITE_ORIGIN) {
      await route.fulfill({ status: 204 });
    } else await route.fallback();
  });
});

test("Performance variants retain historical selection and team/Schedule links", async ({ page }) => {
  const slot = manifest.default_publication_slot;
  await page.goto(`/?family=performance&snapshot=${encodeURIComponent(slot)}`);
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Play");
  await page.getByRole("button", { name: "Quality of Record", exact: true }).click();
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  await expect(page.locator("#snapshot-select")).toHaveValue(slot);
  await expect(page).toHaveURL(/variant=quality_of_record/);
  const record = manifest.snapshots.find((entry) => entry.publication_slot !== slot && entry.performance_variant === "quality_of_record");
  expect(record).toBeTruthy();
  await page.locator("#snapshot-select").selectOption(record.publication_slot);
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  await expect(page.locator("#snapshot-select")).toHaveValue(record.publication_slot);
  await page.reload();
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  const teamLink = page.locator('#rankings-body a[href*="team.html"]').first();
  await teamLink.click();
  await expect(page.locator("#team-page-meta")).toContainText("Quality of Record");
  await expect(page.locator('[data-variant-link="quality_of_record"]')).toHaveAttribute("aria-current", "page");
  await page.locator("#back-to-schedule").click();
  await expect(page.locator('[data-variant="quality_of_record"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#snapshot-select")).toHaveValue(record.publication_slot);
  await page.locator('[data-variant="quality_of_play"]').click();
  await expect(page.locator('[data-variant="quality_of_play"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#snapshot-select")).toHaveValue(record.publication_slot);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  expect(overflow).toBe(false);
});

test("exact Quality of Record snapshot links work without a variant parameter", async ({ page }) => {
  const entry = manifest.snapshots.find((item) => item.publication_slot === manifest.default_publication_slot && item.performance_variant === "quality_of_record");
  expect(entry).toBeTruthy();
  await page.goto(`/?family=performance&snapshot=${encodeURIComponent(entry.snapshot_id)}`);
  await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
  await expect(page.locator("#snapshot-select")).toHaveValue(entry.publication_slot);
});

for (const slot of ["2026-09-08"]) {
  test(`Official QoR publication ${slot} remains available`, async ({ page }) => {
    const entry = manifest.snapshots.find((item) => item.publication_slot === slot && item.performance_variant === "quality_of_record");
    expect(entry).toBeTruthy();
    await page.goto(`/?family=performance&variant=quality_of_record&snapshot=${encodeURIComponent(entry.snapshot_id)}`);
    await expect(page.locator("#snapshot-title")).toContainText("Quality of Record");
    await expect(page.locator("#snapshot-select")).toHaveValue(slot);
    await expect(page.locator("#snapshot-kind")).toHaveText("Official publication");
    await expect(page.locator('[data-variant="quality_of_record"]')).toBeEnabled();
    await expect(page.locator("#rankings-body tr")).toHaveCount(25);
    await expect(page.locator("#snapshot-evidence")).toContainText(`${entry.included_game_count} eligible games included`);
    await expect(page.locator("#snapshot-evidence")).not.toContainText("malformed Campbell");
    await expect(page).toHaveURL(new RegExp(`snapshot=${entry.snapshot_id.replaceAll(".", "\\.")}`));
  });
}

for (const slot of ["2026-09-06", "2026-09-07"]) {
  test(`retired QoR interim ${slot} is unavailable`, async ({ page }) => {
    expect(manifest.snapshots.some((item) => item.publication_slot === slot)).toBe(false);
    await page.goto(`/?family=performance&variant=quality_of_record&snapshot=${encodeURIComponent(slot)}`);
    await expect(page.locator("#snapshot-status")).toContainText(slot);
    await expect(page.locator("#snapshot-status")).toContainText("is unavailable");
    await expect(page.locator("#rankings-body tr")).toHaveCount(0);
    await expect(page).toHaveURL(new RegExp(`snapshot=${slot}`));
  });
}
