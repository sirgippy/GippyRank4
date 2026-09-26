const { test, expect } = require("@playwright/test");
const { installStaticSiteRoute } = require("./static-site");

test.beforeEach(async ({ page }) => {
  await installStaticSiteRoute(page);
  await page.route("**/*", async (route) => {
    if (route.request().resourceType() === "image") {
      await route.fulfill({
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg"/>',
      });
      return;
    }
    await route.fallback();
  });
});

test("serves site files from the synthetic origin", async ({ page }) => {
  const response = await page.goto("/?route-test=1");
  expect(response.status()).toBe(200);
  await expect(page.locator("#rankings-body tr").first()).toBeVisible();

  const results = await page.evaluate(async () => {
    const [script, missing, traversal] = await Promise.all([
      fetch("/assets/app.js?cache-buster=1"),
      fetch("/missing-file.json"),
      fetch("/assets%2f..%2f..%2fpyproject.toml"),
    ]);
    return {
      script: {
        contentType: script.headers.get("content-type"),
        status: script.status,
      },
      missing: missing.status,
      traversal: traversal.status,
    };
  });

  expect(results.script).toEqual({
    contentType: "text/javascript; charset=utf-8",
    status: 200,
  });
  expect(results.missing).toBe(404);
  expect(results.traversal).toBe(404);
});
