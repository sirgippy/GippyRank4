const { readFile, stat } = require("node:fs/promises");
const path = require("node:path");

const SITE_ORIGIN = "http://gippyrank.test";
const SITE_ROOT = path.resolve(__dirname, "../../site");

const CONTENT_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".gif": "image/gif",
  ".htm": "text/html; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
  ".webp": "image/webp",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function contentTypeFor(filePath) {
  return CONTENT_TYPES[path.extname(filePath).toLowerCase()] ?? "application/octet-stream";
}

function filePathFor(requestUrl) {
  const url = new URL(requestUrl);
  if (url.origin !== SITE_ORIGIN) return null;

  let pathname;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    return null;
  }
  if (pathname.includes("\\") || pathname.includes("\0")) return null;

  const relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const filePath = path.resolve(SITE_ROOT, relativePath);
  if (!filePath.startsWith(`${SITE_ROOT}${path.sep}`)) return null;
  return filePath;
}

async function fulfillStaticFile(route) {
  const filePath = filePathFor(route.request().url());
  if (filePath === null) {
    await route.fulfill({
      status: 404,
      contentType: "text/plain; charset=utf-8",
      body: "Not Found",
    });
    return;
  }

  try {
    if (!(await stat(filePath)).isFile()) throw new Error("Not a file");
    await route.fulfill({
      contentType: contentTypeFor(filePath),
      body: await readFile(filePath),
    });
  } catch {
    await route.fulfill({
      status: 404,
      contentType: "text/plain; charset=utf-8",
      body: "Not Found",
    });
  }
}

async function installStaticSiteRoute(page) {
  await page.route(`${SITE_ORIGIN}/**`, fulfillStaticFile);
}

module.exports = { SITE_ORIGIN, installStaticSiteRoute };
