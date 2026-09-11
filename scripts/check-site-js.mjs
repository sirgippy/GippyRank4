import { readdirSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

const assetsDirectory = join(process.cwd(), "site", "assets");
const javascriptFiles = readdirSync(assetsDirectory)
  .filter((filename) => filename.endsWith(".js"))
  .sort();

if (javascriptFiles.length === 0) {
  console.error("No JavaScript assets found under site/assets.");
  process.exit(1);
}

let failed = false;
for (const filename of javascriptFiles) {
  const path = join(assetsDirectory, filename);
  console.log(`Checking ${path}`);
  const result = spawnSync(process.execPath, ["--check", path], { stdio: "inherit" });
  if (result.status !== 0) failed = true;
}

process.exitCode = failed ? 1 : 0;
