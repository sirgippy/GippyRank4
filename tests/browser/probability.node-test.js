const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.resolve(__dirname, "../../site/assets/probability.js"), "utf8");
const modulePromise = import(`data:text/javascript,${encodeURIComponent(source)}`);

for (const [label, first, second, expected] of [
  ["exactly even", 0.5, 0.5, ["50%", "50%"]],
  ["noise below display precision", 0.50000001, 0.49999999, ["50%", "50%"]],
  ["one decimal", 0.505, 0.495, ["50.5%", "49.5%"]],
  ["two decimals", 0.9996, 0.0004, ["99.96%", "0.04%"]],
  ["less than 0.01%", 0.9999999, 0.0000001, [">99.99%", "<0.01%"]],
  ["invalid sum", 0.5, 0.4, ["Unavailable", "Unavailable"]],
  ["out of bounds", -0.1, 1.1, ["Unavailable", "Unavailable"]],
  ["missing value", null, 1, ["Unavailable", "Unavailable"]],
  ["nonnumeric value", "0.5", 0.5, ["Unavailable", "Unavailable"]],
]) {
  test(`matchupProbabilityPair: ${label}`, async () => {
    const { matchupProbabilityPair } = await modulePromise;
    assert.deepEqual(matchupProbabilityPair(first, second), expected);
    assert.deepEqual(matchupProbabilityPair(second, first), [...expected].reverse());
  });
}
