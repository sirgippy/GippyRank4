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
  const first = Number(value);
  const second = Number(other);
  if (!Number.isFinite(first) || !Number.isFinite(second)) return "Unavailable";
  if (first === second) return "50% (even)";
  for (let digits = 0; digits <= 10; digits += 1) {
    const shown = (first * 100).toFixed(digits);
    const opposite = (second * 100).toFixed(digits);
    if (shown !== opposite && Math.abs(Number(shown) + Number(opposite) - 100) < 1e-9
      && (first === 1 || Number(shown) < 100) && (second === 1 || Number(opposite) < 100)) {
      return `${shown}%`;
    }
  }
  return percentage(first);
}
