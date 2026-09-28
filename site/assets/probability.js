export function percentage(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) return "Unavailable";
  if (numeric === 0) return "0%";
  if (numeric === 1) return "100%";
  const percent = numeric * 100;
  if (percent < 0.01) return "<0.01%";
  if (percent >= 99.95) return "<100%";
  if (percent < 1 || percent > 99) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

export function matchupProbability(value, other) {
  return matchupProbabilityPair(value, other)[0];
}

export function matchupProbabilityPair(value, other) {
  const first = Number(value);
  const second = Number(other);
  if (!Number.isFinite(first) || !Number.isFinite(second)
    || first < 0 || second < 0 || first > 1 || second > 1
    || Math.abs(first + second - 1) > 1e-6) return ["Unavailable", "Unavailable"];
  if (Math.abs(first - second) < 0.0001) return ["50%", "50%"];
  for (let digits = 0; digits <= 2; digits += 1) {
    const shown = (first * 100).toFixed(digits);
    const opposite = (second * 100).toFixed(digits);
    if (shown !== opposite && Math.abs(Number(shown) + Number(opposite) - 100) < 1e-9
      && (first === 1 || Number(shown) < 100) && (second === 1 || Number(opposite) < 100)) {
      return [`${shown}%`, `${opposite}%`];
    }
  }
  if (first > 0.99) return [">99.99%", "<0.01%"];
  if (second > 0.99) return ["<0.01%", ">99.99%"];
  return ["50%", "50%"];
}

// Rankings use finer precision near 0, 10, and 100 than schedule summaries.
export function rankingPercentage(value) {
  const percent = Number(value) * 100;
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) return "Unavailable";
  if (percent === 0) return "0%";
  if (percent === 100) return "100%";
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(percent < 0.1 ? 2 : 1)}%`;
  if (percent >= 99.95) return "<100%";
  if (percent < 10 || percent >= 95) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}
