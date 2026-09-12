const $ = (selector) => document.querySelector(selector);
const params = new URLSearchParams(window.location.search);
let logoUrlTemplate = null;
let logoHandles = {};

function node(name, className, text) {
  const value = document.createElement(name);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function svgElement(name, attributes = {}) {
  const value = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, attribute]) => value.setAttribute(key, attribute));
  return value;
}

function teamLogo(teamId, className = "team-logo") {
  const handle = logoHandles[teamId];
  if (!handle || !logoUrlTemplate || logoUrlTemplate.split("{handle}").length !== 2) return null;
  const frame = node("span", "team-logo-frame");
  frame.setAttribute("aria-hidden", "true");
  const image = node("img", className);
  image.src = logoUrlTemplate.replace("{handle}", encodeURIComponent(handle));
  image.alt = "";
  image.width = 60;
  image.height = 40;
  image.loading = "lazy";
  image.decoding = "async";
  image.addEventListener("error", () => frame.remove(), { once: true });
  frame.append(image);
  return frame;
}

function formatDate(value, includeYear = true) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  }).format(new Date(value));
}

function timestamp(value) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", year: "numeric", hour: "numeric",
    minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  }).format(new Date(value));
}

function percentage(value) {
  if (value === 1) return "100%";
  if (value === 0) return "0%";
  const percent = value * 100;
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

function marginValue(value) {
  return Math.abs(Number(value)).toFixed(1);
}

function marginSide(teamName, value) {
  const amount = Number(value);
  if (Math.abs(amount) < 0.05) return "Even";
  return `${teamName} by ${marginValue(amount)}`;
}

function percentileLabel(value) {
  const rounded = Math.round(Number(value));
  const suffix = rounded % 100 >= 11 && rounded % 100 <= 13
    ? "th"
    : ({ 1: "st", 2: "nd", 3: "rd" }[rounded % 10] || "th");
  return `${rounded}${suffix} percentile`;
}

function uncertaintyLabel(interval, rankCount) {
  const width = Number(interval[1]) - Number(interval[0]) + 1;
  const fraction = width / Math.max(Number(rankCount), 1);
  if (fraction <= 0.2) return "relatively narrow";
  if (fraction >= 0.45) return "relatively broad";
  return "moderate";
}

function displayScale(axis) {
  const scale = Number(axis?.probability_encoding?.scale);
  return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function densitySvg(masses, {
  className,
  label,
  description,
  zero = false,
  tail = false,
}) {
  const width = 340;
  const height = 90;
  const left = 5;
  const right = width - 5;
  const baseline = 65;
  const plotWidth = right - left;
  const values = masses.map((value) => Math.max(0, Number(value) || 0));
  const maximum = Math.max(...values, 1e-12);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: `game-distribution-chart ${className}`,
    role: "img",
    "aria-label": label,
  });
  const title = svgElement("title");
  title.textContent = label;
  const desc = svgElement("desc");
  desc.textContent = description;
  svg.append(title, desc);
  svg.append(svgElement("line", { x1: left, y1: baseline, x2: right, y2: baseline, class: "distribution-baseline" }));
  if (zero) {
    const zeroX = left + plotWidth / 2;
    svg.append(svgElement("line", { x1: zeroX, y1: 6, x2: zeroX, y2: baseline + 3, class: "distribution-zero" }));
  }
  const barWidth = plotWidth / values.length;
  values.forEach((value, index) => {
    const barHeight = (value / maximum) * 52;
    svg.append(svgElement("rect", {
      x: left + index * barWidth + 0.25,
      y: baseline - barHeight,
      width: Math.max(barWidth - 0.5, 0.5),
      height: barHeight,
      class: "distribution-bar",
    }));
  });
  if (tail) {
    svg.append(svgElement("path", { d: `M ${left - 2} 12 l 5 -5 l 5 5 M ${right + 2} 12 l -5 -5 l -5 5`, class: "distribution-tail" }));
  }
  return svg;
}

function performanceChart(rating, axis) {
  if (!axis || !Array.isArray(rating.display_pmf)) return null;
  const label = `Inferred performance distribution from rank 1 through rank ${axis.max_rank}; best performances are on the left.`;
  const description = `The distribution has ${rating.display_pmf.length} fixed rank bins. The central 80 percent interval is ranks ${rating.interval_80[0]} through ${rating.interval_80[1]}.`;
  const figure = node("figure", "game-distribution game-distribution-performance");
  figure.append(densitySvg(rating.display_pmf, { className: "performance-distribution-chart", label, description }));
  const caption = node("figcaption", "distribution-axis");
  caption.append(node("span", "axis-start", "#1 best"), node("span", "axis-label", "Inferred performance"), node("span", "axis-end", `#${axis.max_rank} worst`));
  figure.append(caption);
  return figure;
}

function orientedDisplayDistribution(prediction, team, axis) {
  const focalIsHome = prediction.home_team_id === team.team_id;
  const display = prediction.display_distribution;
  if (!axis || !display || !Array.isArray(display.masses)) return null;
  return {
    masses: focalIsHome ? display.masses : [...display.masses].reverse(),
    lowerTail: focalIsHome ? display.lower_tail_probability : display.upper_tail_probability,
    upperTail: focalIsHome ? display.upper_tail_probability : display.lower_tail_probability,
  };
}

function futureChart(prediction, team, axis, oriented) {
  const display = orientedDisplayDistribution(prediction, team, axis);
  if (!display) return null;
  const label = `Predictive margin distribution from ${oriented.opponentName} by ${Math.abs(axis.min_margin)} to ${oriented.focalName} by ${axis.max_margin}; zero is even.`;
  const tailProbability = (display.lowerTail + display.upperTail) / displayScale(axis);
  const description = `The distribution is oriented from ${oriented.focalName}'s perspective. ${percentage(oriented.focalWin)} focal-team win probability and ${percentage(oriented.opponentWin)} opponent win probability. ${percentage(tailProbability)} of mass is outside the visible ${axis.min_margin} to ${axis.max_margin} point range.`;
  const figure = node("figure", "game-distribution game-distribution-future");
  figure.append(densitySvg(display.masses, { className: "future-distribution-chart", label, description, zero: true, tail: tailProbability > 0.001 }));
  const caption = node("figcaption", "distribution-axis");
  caption.append(node("span", "axis-start", `${oriented.opponentName} by ${Math.abs(axis.min_margin)}`), node("span", "axis-label", "Predictive margin · Even"), node("span", "axis-end", `${oriented.focalName} by ${axis.max_margin}`));
  figure.append(caption);
  if (tailProbability > 0.001) figure.append(node("p", "distribution-tail-note", `${percentage(tailProbability)} of predictive mass is beyond the visible ±${Math.max(Math.abs(axis.min_margin), Math.abs(axis.max_margin))}-point range.`));
  return figure;
}

function seasonWinChart(summary) {
  const distribution = summary.final_win_distribution;
  if (!distribution || typeof distribution !== "object") return null;
  const wins = Object.keys(distribution).map(Number).sort((left, right) => left - right);
  const values = wins.map((value) => Number(distribution[String(value)]) || 0);
  const label = "Probability distribution over final regular-season wins.";
  const description = wins.map((value, index) => `${value} wins ${percentage(values[index])}`).join(", ");
  const figure = node("figure", "season-outlook-chart");
  figure.append(densitySvg(values, { className: "season-win-distribution-chart", label, description }));
  const caption = node("figcaption", "distribution-axis");
  caption.append(node("span", "axis-start", `${wins[0]} wins`), node("span", "axis-label", "Final regular-season wins"), node("span", "axis-end", `${wins[wins.length - 1]} wins`));
  figure.append(caption);
  return figure;
}

function renderSeasonOutlook(simulation, team) {
  const section = $("#season-outlook-section");
  const content = $("#season-outlook");
  const summary = team && simulation?.teams?.[team.team_id];
  if (!summary) {
    section.hidden = true;
    content.replaceChildren();
    return;
  }
  section.hidden = false;
  if (summary.forecast_status !== "available") {
    content.replaceChildren(node("p", "season-outlook-unavailable", "Season forecast unavailable because one or more remaining regular-season games lack a supported model representation."));
    return;
  }
  const records = Object.entries(summary.record_probabilities || {})
    .sort((left, right) => Number(right[1]) - Number(left[1]) || left[0].localeCompare(right[0]))
    .slice(0, 5);
  const mostLikely = records[0];
  const details = node("dl", "season-outlook-details");
  const thresholds = Object.entries(summary.threshold_probabilities || {})
    .filter(([label]) => label.startsWith("wins_"))
    .map(([label, probability]) => `${label.replace(/^wins_(\d+)_plus$/, "$1+")} ${percentage(probability)}`)
    .join(" · ");
  [
    ["Expected finish", `${Number(summary.expected_final_wins).toFixed(1)} wins`],
    ["Most likely record", mostLikely ? `${mostLikely[0]} (${percentage(mostLikely[1])})` : "Unavailable"],
    ["Central 50%", `${summary.final_win_interval_50[0]}–${summary.final_win_interval_50[1]} wins`],
    ["Central 80%", `${summary.final_win_interval_80[0]}–${summary.final_win_interval_80[1]} wins`],
    ...(thresholds ? [["Win milestones", thresholds]] : []),
  ].forEach(([label, value]) => details.append(node("dt", "", label), node("dd", "", value)));
  const quality = summary.variance_decomposition?.team_quality_fraction;
  const game = summary.variance_decomposition?.game_randomness_fraction;
  if (Number.isFinite(quality) && Number.isFinite(game)) {
    details.append(node("dt", "", "Uncertainty sources"), node("dd", "", `${percentage(quality)} team quality · ${percentage(game)} game randomness`));
  }
  const recordList = node("ul", "season-outlook-records");
  records.forEach(([record, probability]) => recordList.append(node("li", "", `${record} · ${percentage(probability)}`)));
  const chart = seasonWinChart(summary);
  content.replaceChildren(...(chart ? [chart] : []), details, recordList);
}

function orientedPrediction(prediction, team) {
  const focalIsHome = prediction.home_team_id === team.team_id;
  const opponentName = focalIsHome ? prediction.away_team_name : prediction.home_team_name;
  const focalWin = focalIsHome ? prediction.home_win_probability : prediction.away_win_probability;
  const opponentWin = focalIsHome ? prediction.away_win_probability : prediction.home_win_probability;
  const expected = focalIsHome ? prediction.expected_home_margin : -prediction.expected_home_margin;
  const median = focalIsHome ? prediction.median_home_margin : -prediction.median_home_margin;
  const orientInterval = (interval) => focalIsHome
    ? [interval[0], interval[1]]
    : [-interval[1], -interval[0]];
  return {
    focalName: team.team_name,
    opponentName,
    focalWin,
    opponentWin,
    expected,
    median,
    interval50: orientInterval(prediction.margin_interval_50),
    interval80: orientInterval(prediction.margin_interval_80),
    interval95: orientInterval(prediction.margin_interval_95),
  };
}

function predictionRange(oriented, interval) {
  const [low, high] = interval;
  if (high < 0) return `${marginSide(oriented.opponentName, -high)} to ${marginSide(oriented.opponentName, -low)}`;
  if (low > 0) return `${marginSide(oriented.focalName, low)} to ${marginSide(oriented.focalName, high)}`;
  const lower = low < 0 ? marginSide(oriented.opponentName, -low) : "Even";
  const upper = high > 0 ? marginSide(oriented.focalName, high) : "Even";
  return `${lower} to ${upper}`;
}

function predictionPanel(prediction, team, axis) {
  const oriented = orientedPrediction(prediction, team);
  const favorite = oriented.focalWin >= oriented.opponentWin
    ? [oriented.focalName, oriented.focalWin]
    : [oriented.opponentName, oriented.opponentWin];
  const panel = node("div", "game-prediction");
  const chart = futureChart(prediction, team, axis, oriented);
  if (chart) panel.append(chart);
  const expectedText = oriented.expected >= 0 ? marginSide(oriented.focalName, oriented.expected) : marginSide(oriented.opponentName, -oriented.expected);
  const expectedTeam = oriented.expected >= 0 ? oriented.focalName : oriented.opponentName;
  const expectedAmount = marginValue(oriented.expected);
  const expectedPrimary = Math.abs(oriented.expected) < 0.05
    ? "expected Even"
    : expectedTeam === favorite[0]
      ? `expected by ${expectedAmount}`
      : `expected ${expectedTeam} by ${expectedAmount}`;
  const title = node("strong", "game-prediction-title", `${favorite[0]} ${percentage(favorite[1])} · ${expectedPrimary}`);
  const accessible = node("p", "sr-only", `${oriented.focalName} has a ${percentage(oriented.focalWin)} win probability. ${oriented.opponentName} has a ${percentage(oriented.opponentWin)} win probability. Expected margin is ${oriented.expected >= 0 ? marginSide(oriented.focalName, oriented.expected) : marginSide(oriented.opponentName, -oriented.expected)}. The central 80 percent predictive interval ranges from ${predictionRange(oriented, oriented.interval80)}.`);
  const details = node("details", "game-prediction-details");
  const summary = node("summary", "", "More predictive detail");
  const list = node("dl", "game-rating-details");
  [["Focal win probability", percentage(oriented.focalWin)], ["Opponent win probability", percentage(oriented.opponentWin)], ["Expected margin", expectedText], ["Median margin", oriented.median >= 0 ? marginSide(oriented.focalName, oriented.median) : marginSide(oriented.opponentName, -oriented.median)], ["Central 50% range", predictionRange(oriented, oriented.interval50)], ["Central 80% range", predictionRange(oriented, oriented.interval80)], ["Central 95% range", predictionRange(oriented, oriented.interval95)], ["Prediction source", prediction.prediction_source === "predictive_history" ? "Predictive History" : "Predictive Context"]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  details.append(summary, list);
  panel.append(title, accessible, details);
  panel.setAttribute("aria-label", `${favorite[0]} has a ${percentage(favorite[1])} win probability. Expected margin: ${oriented.expected >= 0 ? marginSide(oriented.focalName, oriented.expected) : marginSide(oriented.opponentName, -oriented.expected)}. Central 80% range: ${predictionRange(oriented, oriented.interval80)}.`);
  return panel;
}

function rank(value) {
  return `#${Number(value).toFixed(1)}`;
}

function distributionTeam(distribution, expectedSnapshotId, expectedRankCount, teamId, label) {
  if (!distribution || distribution.snapshot_id !== expectedSnapshotId) {
    throw new Error(`${label} distribution does not match the published snapshot.`);
  }
  if (distribution.rank_count !== expectedRankCount) {
    throw new Error(`${label} distribution does not use the selected rank support.`);
  }
  const team = distribution.teams?.[teamId];
  if (!team || !Array.isArray(team.pmf) || team.pmf.length !== expectedRankCount || !team.summary) {
    throw new Error(`${label} distribution for this team is unavailable.`);
  }
  return team;
}

function rankSummaryLine(summary) {
  return `expected ${rank(summary.expected_rank)} · 80% range #${summary.interval_80[0]}–#${summary.interval_80[1]}`;
}

function movementChart(teamName, preseasonTeam, currentTeam, rankCount, currentLabel) {
  const width = 760;
  const height = 252;
  const left = 74;
  const right = 18;
  const top = 28;
  const bottom = 47;
  const plotWidth = width - left - right;
  const plotHeight = 56;
  const baselines = [105, 183];
  const distributions = [
    { label: "Preseason", team: preseasonTeam, className: "movement-preseason-bar", baseline: baselines[0] },
    { label: currentLabel, team: currentTeam, className: "movement-current-bar", baseline: baselines[1] },
  ];
  const maximum = Math.max(
    ...distributions.flatMap(({ team }) => team.pmf.map((value) => Math.max(0, Number(value) || 0))),
    1e-12,
  );
  const xFor = (value) => left + ((value - 0.5) / rankCount) * plotWidth;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "distribution-chart season-movement-chart",
    role: "img",
    "aria-labelledby": "season-movement-chart-title season-movement-chart-description",
  });
  const title = svgElement("title", { id: "season-movement-chart-title" });
  title.textContent = `${teamName} preseason to ${currentLabel} rank distributions`;
  const description = svgElement("desc", { id: "season-movement-chart-description" });
  description.textContent = `Preseason ${rankSummaryLine(preseasonTeam.summary)}. At ${currentLabel}, ${rankSummaryLine(currentTeam.summary)}. Both distributions use the same horizontal rank axis from rank 1 best to rank ${rankCount} worst.`;
  svg.append(title, description);

  distributions.forEach(({ label, team, className, baseline }) => {
    svg.append(svgElement("line", { x1: left, y1: baseline, x2: width - right, y2: baseline, class: "movement-baseline" }));
    const rowLabel = svgElement("text", { x: left - 10, y: baseline - plotHeight / 2 + 4, "text-anchor": "end", class: "movement-row-label" });
    rowLabel.textContent = label;
    svg.append(rowLabel);
    const barWidth = Math.max(1, (plotWidth / rankCount) * 0.82);
    team.pmf.forEach((probability, index) => {
      const value = Math.max(0, Number(probability) || 0);
      const barHeight = (value / maximum) * plotHeight;
      svg.append(svgElement("rect", {
        x: xFor(index + 1) - barWidth / 2,
        y: baseline - barHeight,
        width: barWidth,
        height: barHeight,
        class: className,
      }));
    });
    const expected = svgElement("line", {
      x1: xFor(team.summary.expected_rank),
      y1: baseline - plotHeight - 2,
      x2: xFor(team.summary.expected_rank),
      y2: baseline + 2,
      class: "movement-expected-marker",
    });
    expected.setAttribute("aria-hidden", "true");
    svg.append(expected);
  });

  const axisY = height - bottom;
  svg.append(svgElement("line", { x1: left, y1: axisY, x2: width - right, y2: axisY, class: "chart-axis" }));
  const ticks = [...new Set([1, Math.round(rankCount * 0.25), Math.round(rankCount * 0.5), Math.round(rankCount * 0.75), rankCount])].sort((a, b) => a - b);
  ticks.forEach((value) => {
    const x = xFor(value);
    const label = svgElement("text", { x, y: axisY + 17, "text-anchor": "middle", class: "chart-label" });
    label.textContent = value;
    svg.append(svgElement("line", { x1: x, y1: axisY, x2: x, y2: axisY + 4, class: "chart-axis" }), label);
  });
  const axisTitle = svgElement("text", { x: left + plotWidth / 2, y: height - 6, "text-anchor": "middle", class: "chart-axis-title" });
  axisTitle.textContent = "Shared rank axis (1 is best)";
  svg.append(axisTitle);

  const figure = node("figure", "season-movement-chart-figure");
  figure.append(svg);
  const caption = node("figcaption", "distribution-axis");
  caption.append(node("span", "axis-start", "#1 best"), node("span", "axis-label", "Rank position · shared scale"), node("span", "axis-end", `#${rankCount} worst`));
  figure.append(caption);
  return figure;
}

function preseasonDistributionChart(teamName, team, rankCount) {
  const label = `${teamName} preseason rank distribution from rank 1 through rank ${rankCount}; rank 1 is best.`;
  const description = `The published preseason distribution has ${rankCount} rank bins. ${rankSummaryLine(team.summary)}. It is not a fixed rank.`;
  const figure = node("figure", "preseason-distribution-figure");
  figure.append(densitySvg(team.pmf, { className: "preseason-distribution-chart", label, description }));
  const caption = node("figcaption", "distribution-axis");
  caption.append(node("span", "axis-start", "#1 best"), node("span", "axis-label", "Preseason rank distribution"), node("span", "axis-end", `#${rankCount} worst`));
  figure.append(caption);
  return figure;
}

function renderPreseasonStartingPoint(entry, distribution, row) {
  const section = $("#preseason-starting-point-section");
  const content = $("#preseason-starting-point-content");
  const visible = entry.ranking_family === "predictive" && entry.snapshot_type === "preseason";
  section.hidden = !visible;
  if (!visible) {
    content.replaceChildren();
    return;
  }
  const copy = preseasonCopy[entry.prior_family] ?? preseasonCopy.context;
  const team = distributionTeam(distribution, entry.snapshot_id, distribution.rank_count, row.team_id, "Preseason");
  const familyList = node("ul", "preseason-input-list");
  copy.families.forEach(([label, description]) => {
    const item = node("li");
    item.append(node("strong", "", label), node("span", "", description));
    familyList.append(item);
  });
  const explanation = node("p", "preseason-starting-point-explanation", copy.detail);
  const uncertainty = node("p", "preseason-starting-point-note", `This is an uncertain distribution, not a declaration that ${row.team_name} is exactly ${rank(team.summary.expected_rank)}. ${row.team_name}'s published preseason summary is ${rankSummaryLine(team.summary)}.`);
  content.replaceChildren(
    node("p", "preseason-starting-point-summary", copy.summary),
    node("h3", "preseason-input-heading", "What goes into this?"),
    explanation,
    familyList,
    ...(entry.prior_family === "history"
      ? [node("p", "preseason-starting-point-note", "History does not use current roster talent, recruiting, returning production, or coach tenure.")]
      : []),
    preseasonDistributionChart(row.team_name, team, distribution.rank_count),
    uncertainty,
  );
}

function renderSeasonMovement(entry, currentDistribution, preseasonDistribution, row) {
  const section = $("#season-movement-section");
  const content = $("#season-movement");
  const context = $("#season-movement-context");
  if (entry.ranking_family === "performance") {
    section.hidden = false;
    context.textContent = "Performance is a separate view of completed-game evidence.";
    content.replaceChildren(node("p", "season-movement-not-applicable", "Performance has no preseason starting point in the same semantic sense, so this page does not show a preseason-to-current comparison."));
    return;
  }
  if (entry.snapshot_type === "preseason") {
    section.hidden = true;
    content.replaceChildren();
    return;
  }
  section.hidden = false;
  context.textContent = `Published ${priorLabel(entry.prior_family)} distributions · exact selected snapshot, not today's latest view.`;
  if (!entry.preseason_snapshot_id || !entry.preseason_distribution_path || !preseasonDistribution) {
    content.replaceChildren(node("p", "season-movement-unavailable", "A matching published preseason distribution is unavailable, so no comparison is shown."));
    return;
  }
  const currentTeam = distributionTeam(currentDistribution, entry.snapshot_id, currentDistribution.rank_count, row.team_id, "Selected snapshot");
  const preseasonTeam = distributionTeam(preseasonDistribution, entry.preseason_snapshot_id, currentDistribution.rank_count, row.team_id, "Preseason");
  const intro = node("p", "season-movement-intro", `GippyRank started ${row.team_name} from an uncertain preseason belief and updates that distribution with game evidence. The change below is probabilistic movement, not arithmetic rank points.`);
  const summary = node("div", "season-movement-summary");
  const preseason = node("article", "season-movement-point season-movement-preseason");
  preseason.append(node("h3", "", "Preseason"), node("p", "season-movement-rank", rankSummaryLine(preseasonTeam.summary)), node("p", "season-movement-source", `Published ${entry.preseason_display_label || "Preseason"} ${priorLabel(entry.prior_family)} distribution`));
  const current = node("article", "season-movement-point season-movement-current");
  current.append(node("h3", "", entry.display_label), node("p", "season-movement-rank", rankSummaryLine(currentTeam.summary)), node("p", "season-movement-source", "Exact distribution at the selected snapshot"));
  summary.append(preseason, node("span", "season-movement-arrow", "→"), current);
  const accessible = node("p", "sr-only", `Preseason expected rank ${Number(preseasonTeam.summary.expected_rank).toFixed(1)} with a central 80 percent range from rank ${preseasonTeam.summary.interval_80[0]} to rank ${preseasonTeam.summary.interval_80[1]}. At the selected ${entry.display_label} snapshot, expected rank is ${Number(currentTeam.summary.expected_rank).toFixed(1)} with a central 80 percent range from rank ${currentTeam.summary.interval_80[0]} to rank ${currentTeam.summary.interval_80[1]}.`);
  content.replaceChildren(intro, movementChart(row.team_name, preseasonTeam, currentTeam, currentDistribution.rank_count, entry.display_label), summary, accessible);
}

function publicationStatusLabel(entry) {
  return entry.publication_status === "official" ? "Official" : "Interim";
}

const preseasonCopy = {
  context: {
    summary: "Context asks what we knew about this particular team before kickoff.",
    detail: "Context starts with program history and adds information about this year's team. Those inputs produce the preseason distribution; they are not feature-by-feature rank adjustments.",
    families: [
      ["Program history", "Previous program performance."],
      ["Recruiting", "Recent recruiting quality and trends."],
      ["Roster talent", "Current roster talent."],
      ["Returning production", "How much production returns from the previous team."],
      ["Coaching", "Coach tenure."],
    ],
  },
  history: {
    summary: "History asks what we would believe from the program's track record alone.",
    detail: "History uses previous program performance across recent and longer historical windows. It intentionally ignores current roster talent, recruiting, returning production, and coach tenure.",
    families: [["Program history", "Previous program performance across recent and longer historical windows."]],
  },
};

function priorLabel(prior) {
  return prior === "history" ? "History" : "Context";
}

function entryFor(manifest) {
  const family = params.get("family") === "performance" ? "performance" : "predictive";
  const season = Number(params.get("season")) || manifest.seasons[0];
  let entries = manifest.snapshots.filter((item) => item.season === season && item.ranking_family === family);
  const prior = params.get("prior") === "history" ? "history" : "context";
  if (family === "predictive") {
    entries = entries.filter((item) => item.prior_family === prior);
  }
  const snapshot = params.get("snapshot");
  return entries.find((item) => item.snapshot_id === snapshot)
    ?? entries.find((item) => item.publication_slot === snapshot)
    ?? entries.find((item) => item.publication_slot === manifest.default_publication_slot)
    ?? entries[0];
}

function teamPageUrl(entry, prior) {
  const url = new URL("./team.html", document.baseURI);
  url.searchParams.set("team", params.get("team"));
  url.searchParams.set("season", String(entry.season));
  url.searchParams.set("snapshot", entry.snapshot_id);
  url.searchParams.set("family", "predictive");
  url.searchParams.set("prior", prior);
  return `${url.pathname}${url.search}`;
}

function updatePriorSwitch(manifest, entry) {
  const nav = $("#team-prior-switch");
  if (!nav) return;
  const predictive = entry?.ranking_family === "predictive";
  nav.hidden = !predictive;
  if (!predictive) return;
  ["context", "history"].forEach((prior) => {
    const link = nav.querySelector(`[data-prior-link="${prior}"]`);
    if (!link) return;
    const counterpart = manifest.snapshots.find((candidate) => (
      candidate.season === entry.season
      && candidate.publication_slot === entry.publication_slot
      && candidate.ranking_family === "predictive"
      && candidate.prior_family === prior
    ));
    link.hidden = !counterpart;
    if (counterpart) link.href = teamPageUrl(counterpart, prior);
    if (prior === entry.prior_family) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function updateBackLink(entry) {
  const url = new URL("./", document.baseURI);
  if (!entry) {
    $("#back-to-rankings").href = url.pathname;
    $("#rankings-view-link").href = url.pathname;
    const schedulePath = new URL("./schedule.html", document.baseURI).pathname;
    $("#back-to-schedule").href = schedulePath;
    $("#schedule-view-link").href = schedulePath;
    return;
  }
  url.searchParams.set("season", String(entry.season));
  url.searchParams.set("snapshot", entry.snapshot_id);
  url.searchParams.set("family", entry.ranking_family);
  if (entry.ranking_family === "predictive") url.searchParams.set("prior", entry.prior_family);
  const rankingsPath = `${url.pathname}${url.search}`;
  $("#back-to-rankings").href = rankingsPath;
  $("#rankings-view-link").href = rankingsPath;
  const scheduleUrl = new URL("./schedule.html", document.baseURI);
  scheduleUrl.search = url.search;
  if (params.get("view") === "marquee") scheduleUrl.searchParams.set("view", "marquee");
  scheduleUrl.searchParams.set("week", params.get("week") ?? entry.default_week);
  const schedulePath = `${scheduleUrl.pathname}${scheduleUrl.search}`;
  $("#back-to-schedule").href = schedulePath;
  $("#schedule-view-link").href = schedulePath;
}

function renderSummary(entry, snapshot, row) {
  const rankLabel = row.rated === false ? "NR" : `#${row.display_rank}`;
  const logo = teamLogo(row.team_id, "team-logo team-logo-card");
  $("#team-page-logo").replaceChildren(...(logo ? [logo] : []));
  $("#team-page-kind").textContent = `${entry.season} ${entry.display_label} · ${publicationStatusLabel(entry)} publication`;
  $("#team-page-title").textContent = row.team_name;
  $("#team-page-meta").textContent = `${row.conference || "Independent"} · ${entry.snapshot_type === "preseason" ? "Before game evidence" : `Through ${formatDate(entry.effective_cutoff)}`} · ${rankLabel} ${entry.ranking_family === "performance" ? "Performance" : `Predictive ${entry.prior_family === "history" ? "History" : "Context"}`}`;
  $("#schedule-context").textContent = entry.snapshot_type === "preseason"
    ? "Schedule metadata is shown without any season results."
    : `Results and ratings are shown only through ${timestamp(entry.effective_cutoff)}.`;
  $("#prediction-source").textContent = `Future prediction source: ${entry.ranking_family === "performance" ? "Predictive Context" : entry.prior_family === "history" ? "Predictive History" : "Predictive Context"}. Predictions use the posterior available at this snapshot.`;
  const summary = node("div", "team-summary-grid");
  summary.append(
    node("div", "team-summary-item", `${rankLabel} · expected ${rank(row.expected_rank)}`),
    node("div", "team-summary-item", row.rank_change_accessible || "No earlier official ranking baseline"),
    node("div", "team-summary-item", `Median ${row.median_rank} · 80% ${row.interval_80[0]}–${row.interval_80[1]}`),
    node("div", "team-summary-item", `${row.record} modeled record · ${percentage(row.top25_probability)} Top 25 probability`),
  );
  $("#team-ranking-summary").replaceChildren(summary);
}

function ratingPanel(rating, axis) {
  const panel = node("div", "game-rating");
  const percentile = rating.performance_percentile === undefined ? null : percentileLabel(rating.performance_percentile);
  const grade = rating.performance_grade ? `${rating.performance_grade} · ` : "";
  const title = node("strong", "game-rating-title", percentile ? `${grade}${percentile}` : "Inferred performance");
  const interval = node("span", "game-rating-interval", percentile
    ? `${uncertaintyLabel(rating.interval_80, rating.rank_count)} uncertainty`
    : `80% interval: ${rating.interval_80[0]}–${rating.interval_80[1]}`);
  interval.setAttribute("aria-label", `central 80 percent interval from rank ${rating.interval_80[0]} through rank ${rating.interval_80[1]}`);
  const chart = performanceChart(rating, axis);
  if (chart) panel.append(chart);
  panel.append(title, interval);
  const accessible = percentile
    ? `Performance grade ${rating.performance_grade}. ${percentile} among eligible FBS team-game performances at this snapshot. The inferred performance distribution has ${uncertaintyLabel(rating.interval_80, rating.rank_count)} uncertainty.`
    : `Inferred performance distribution; central 80 percent interval is ranks ${rating.interval_80[0]} through ${rating.interval_80[1]}.`;
  panel.append(node("p", "sr-only", accessible));
  const details = node("details", "game-rating-details-disclosure");
  details.append(node("summary", "", "More performance detail"));
  const list = node("dl", "game-rating-details");
  [["Expected performance rank", rank(rating.expected_rank)], ["Median", `#${rating.median_rank}`], ["Mode", `#${rating.mode_rank}`], ["50% interval", `${rating.interval_50[0]}–${rating.interval_50[1]}`], ["80% interval", `${rating.interval_80[0]}–${rating.interval_80[1]}`], ["95% interval", `${rating.interval_95[0]}–${rating.interval_95[1]}`], ["Top 5", percentage(rating.top5_probability)], ["Top 10", percentage(rating.top10_probability)], ["Top 25", percentage(rating.top25_probability)]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  details.append(list);
  panel.append(details);
  return panel;
}

function gameCard(game, cutoff, artifact, team) {
  const item = node("li", "game-card");
  const header = node("div", "game-card-header");
  const week = game.week === null ? "" : `Week ${game.week} · `;
  header.append(node("p", "game-date", `${week}${formatDate(game.date, false)}`), node("span", "game-site", game.site));
  const opponent = node("h3", "game-opponent", game.opponent_name || game.opponent_id || "Unknown opponent");
  const opponentHeading = node("div", "game-opponent-row");
  const logo = teamLogo(game.opponent_id);
  if (logo) opponentHeading.append(logo);
  opponentHeading.append(opponent);
  const meta = node("p", "game-meta", `${game.opponent_classification.toUpperCase()}${game.opponent_conference ? ` · ${game.opponent_conference}` : ""}`);
  const outcome = node("p", "game-outcome");
  const future = game.game_state ? game.game_state === "future" : cutoff === null || new Date(game.date) > cutoff;
  if (game.result && game.score) outcome.textContent = `${game.result} ${game.score.team}–${game.score.opponent}`;
  else if (game.game_state === "cancelled") outcome.textContent = "Cancelled or postponed";
  else if (future) outcome.textContent = "Future at this snapshot";
  else outcome.textContent = "Not completed by this snapshot";
  const body = node("div", "game-card-body");
  if (game.game_rating) {
    body.append(ratingPanel(game.game_rating, artifact.performance_axis));
  } else if (future) {
    const prediction = artifact.future_predictions?.[game.future_prediction_id];
    if (prediction) body.append(predictionPanel(prediction, team, artifact.future_margin_axis));
    else body.append(node("p", "game-not-modeled", "Prediction unavailable — the matchup lacks sufficient supported model representation."));
  } else if (game.result || game.game_state === "completed") {
    body.append(node("p", "game-not-modeled", "Not modeled — this game is outside the eligible Historical Likelihood evidence."));
  } else {
    body.append(node("p", "game-not-modeled", "Not modeled — no completed evidence was available at the cutoff."));
  }
  item.append(header, opponentHeading, meta, outcome, body);
  return item;
}

function renderSchedule(artifact, entry) {
  const teamId = params.get("team");
  const team = artifact.teams?.[teamId];
  if (!team) throw new Error("This team is not available in the selected season snapshot.");
  const cutoff = artifact.effective_cutoff ? new Date(artifact.effective_cutoff) : null;
  const games = Array.isArray(team.games) ? team.games : [];
  $("#team-page-status").textContent = games.length ? `${games.length} scheduled games · performance and prediction distributions include uncertainty` : "No schedule entries are available for this team.";
  $("#schedule-list").replaceChildren(...games.map((game) => gameCard(game, cutoff, artifact, team)));
}

async function load() {
  const manifestResponse = await fetch("./data/manifest.json");
  if (!manifestResponse.ok) throw new Error("Could not load the published manifest.");
  const manifest = await manifestResponse.json();
  logoUrlTemplate = manifest.team_logos?.url_template || null;
  logoHandles = manifest.team_logos?.handles || {};
  const entry = entryFor(manifest);
  updateBackLink(entry);
  if (!entry) throw new Error("No published snapshot matches this team page.");
  const teamId = params.get("team");
  if (!teamId) throw new Error("A stable team ID is required.");
  const preseasonPath = entry.ranking_family === "predictive"
    && entry.snapshot_type !== "preseason"
    ? entry.preseason_distribution_path
    : null;
  const responses = await Promise.all([
    fetch(`./${entry.data_path}`),
    fetch(`./${entry.team_seasons_path || entry.team_season_path}`),
    fetch(`./${entry.distribution_path}`),
    ...(preseasonPath ? [fetch(`./${preseasonPath}`)] : []),
  ]);
  if (responses.some((response) => !response.ok)) throw new Error("The selected team-season artifact is unavailable.");
  const [snapshot, artifact, distribution, preseasonDistribution] = await Promise.all([
    responses[0].json(),
    responses[1].json(),
    responses[2].json(),
    responses[3]?.json() ?? Promise.resolve(null),
  ]);
  if (artifact.snapshot_id !== entry.snapshot_id || artifact.season !== entry.season) throw new Error("The team-season artifact does not match the selected snapshot.");
  const row = snapshot.rankings.find((item) => item.team_id === teamId);
  if (!row) throw new Error("This team is not available in the selected ranking snapshot.");
  distributionTeam(distribution, entry.snapshot_id, snapshot.rank_count, teamId, "Selected snapshot");
  updatePriorSwitch(manifest, entry);
  renderSummary(entry, snapshot, row);
  renderPreseasonStartingPoint(entry, distribution, row);
  renderSeasonMovement(entry, distribution, preseasonDistribution, row);
  renderSeasonOutlook(artifact.season_simulation, artifact.teams[teamId]);
  renderSchedule(artifact, entry);
}

load().catch((error) => {
  updateBackLink(null);
  $("#team-page-status").textContent = error.message;
  $("#team-page-status").className = "team-page-status team-page-error";
});
