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
  const percent = Number(value) * 100;
  if (!Number.isFinite(percent)) return "Unavailable";
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

function rank(value) {
  return `#${Number(value).toFixed(1)}`;
}

function rankRange(interval) {
  return `#${interval[0]}–#${interval[1]}`;
}

function rankSummaryLine(summary) {
  return `expected ${rank(summary.expected_rank)} · median #${summary.median_rank} · 80% ${rankRange(summary.interval_80)}`;
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

function disclosure(label, content, className = "distribution-disclosure") {
  const details = node("details", className);
  const summary = node("summary", "distribution-toggle", "▾");
  const updateLabel = () => {
    const verb = details.open ? "Hide" : "Show";
    summary.setAttribute("aria-label", `${verb} ${label}`);
    summary.title = `${verb} ${label}`;
  };
  updateLabel();
  details.addEventListener("toggle", updateLabel);
  details.append(summary, content);
  return details;
}

function rankDistributionChart(teamName, team, rankCount, label = "Rank distribution") {
  const description = `${rankSummaryLine(team.summary)}. Rank 1 is best and rank ${rankCount} is worst.`;
  const figure = node("figure", "rank-distribution-figure");
  figure.append(densitySvg(team.pmf, {
    className: "rank-distribution-chart",
    label: `${teamName} ${label.toLowerCase()}: rank 1 through ${rankCount}.`,
    description,
  }));
  const caption = node("figcaption", "distribution-axis");
  caption.append(
    node("span", "axis-start", "#1 best"),
    node("span", "axis-label", label),
    node("span", "axis-end", `#${rankCount} worst`),
  );
  figure.append(caption);
  return figure;
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

function lazyRankDistributionDisclosure(point, teamId, rankCount, teamName) {
  const body = node("div", "lazy-distribution-content");
  const details = disclosure(`${point.display_label} rank distribution`, body, "distribution-disclosure trajectory-distribution-disclosure");
  let requested = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || requested) return;
    requested = true;
    body.replaceChildren(node("p", "distribution-loading", "Loading distribution…"));
    try {
      const response = await fetch(`./${point.distribution_path}`);
      if (!response.ok) throw new Error("The published distribution is unavailable.");
      const distribution = await response.json();
      const team = distributionTeam(distribution, point.snapshot_id, rankCount, teamId, "Published snapshot");
      body.replaceChildren(rankDistributionChart(teamName, team, rankCount, `${point.display_label} rank distribution`));
    } catch (error) {
      body.replaceChildren(node("p", "distribution-error", error.message));
    }
  });
  return details;
}

function likelyRecord(summary) {
  const records = Object.entries(summary?.record_probabilities || {})
    .sort((left, right) => Number(right[1]) - Number(left[1]) || left[0].localeCompare(right[0]));
  return records[0] ? `${records[0][0]} (${percentage(records[0][1])})` : "Unavailable";
}

function summaryItem(label, value, note = null) {
  const item = node("dl", "team-summary-item");
  item.append(node("dt", "", label), node("dd", "", value));
  if (note) item.append(node("p", "team-summary-note", note));
  return item;
}

function renderSummary(entry, snapshot, row, simulation, distribution) {
  const rankLabel = row.rated === false ? "NR" : `#${row.display_rank}`;
  const logo = teamLogo(row.team_id, "team-logo team-logo-card");
  $("#team-page-logo").replaceChildren(...(logo ? [logo] : []));
  $("#team-page-kind").textContent = `${entry.season} ${entry.display_label} · ${publicationStatusLabel(entry)} publication`;
  $("#team-page-title").textContent = row.team_name;
  $("#team-page-meta").textContent = `${row.conference || "Independent"} · ${entry.snapshot_type === "preseason" ? "Before game evidence" : `Through ${formatDate(entry.effective_cutoff)}`} · ${entry.ranking_family === "performance" ? "Performance" : `Predictive ${priorLabel(entry.prior_family)}`}`;
  $("#schedule-context").textContent = entry.snapshot_type === "preseason"
    ? "Schedule metadata is shown without any season results."
    : `Results and ratings are shown only through ${timestamp(entry.effective_cutoff)}.`;
  $("#prediction-source").textContent = `Future prediction source: ${entry.ranking_family === "performance" ? "Predictive Context" : `Predictive ${priorLabel(entry.prior_family)}`}. Predictions use the posterior available at this snapshot.`;

  const forecast = simulation?.teams?.[row.team_id];
  const summary = node("div", "team-summary-grid");
  summary.append(
    summaryItem("Display rank", rankLabel),
    summaryItem("Expected rank", rank(row.expected_rank)),
    summaryItem("Median rank", `#${row.median_rank}`),
    summaryItem("Central 80%", rankRange(row.interval_80)),
    summaryItem("Top 25 probability", percentage(row.top25_probability)),
    summaryItem("Modeled record", row.record),
  );
  if (forecast?.forecast_status === "available") {
    summary.append(
      summaryItem("Expected final wins", `${Number(forecast.expected_final_wins).toFixed(1)}`),
      summaryItem("Likely final record", likelyRecord(forecast)),
    );
  }
  const currentTeam = distributionTeam(distribution, entry.snapshot_id, snapshot.rank_count, row.team_id, "Selected snapshot");
  const currentDistribution = disclosure(
    "current rank distribution",
    rankDistributionChart(row.team_name, currentTeam, snapshot.rank_count, "Current rank distribution"),
    "distribution-disclosure header-distribution-disclosure",
  );
  $("#team-ranking-summary").replaceChildren(summary, currentDistribution);
}

function comparisonText(comparison) {
  if (!comparison || typeof comparison !== "object") return null;
  const rankValue = comparison.fbs_rank;
  const count = comparison.count;
  const percentile = comparison.fbs_percentile;
  if (!Number.isFinite(Number(rankValue)) || !Number.isFinite(Number(count))) return null;
  const parts = [`FBS #${rankValue}/${count}`];
  if (Number.isFinite(Number(percentile))) parts.push(`${Math.round(Number(percentile))}th pct.`);
  return parts.join(" · ");
}

function preseasonEvidenceGroup(group) {
  const section = node("article", "preseason-input-group");
  section.append(node("h3", "", group.title));
  if (group.source) section.append(node("p", "preseason-input-source", group.source));
  const list = node("dl", "preseason-input-values");
  (group.fields || []).forEach((field) => {
    const term = node("dt", "", field.label);
    const description = node("dd", "");
    description.append(node("strong", "", field.display_value));
    const comparison = comparisonText(field.comparison);
    if (comparison) description.append(node("span", "preseason-input-comparison", comparison));
    if (field.detail) description.append(node("span", "preseason-input-detail", field.detail));
    list.append(term, description);
  });
  section.append(list);
  if (group.note) section.append(node("p", "preseason-input-note", group.note));
  return section;
}

function renderPreseasonStartingPoint(entry, distribution, sourceArtifact, row) {
  const section = $("#preseason-starting-point-section");
  const content = $("#preseason-starting-point-content");
  const visible = entry.ranking_family === "predictive" && Boolean(entry.preseason_snapshot_id);
  section.hidden = !visible;
  if (!visible) {
    content.replaceChildren();
    return;
  }
  const team = distributionTeam(distribution, distribution.snapshot_id, distribution.rank_count, row.team_id, "Preseason");
  const inputs = sourceArtifact?.preseason_inputs;
  const evidence = inputs?.teams?.[row.team_id];
  const children = [
    node(
      "p",
      "preseason-starting-point-summary",
      entry.prior_family === "history"
        ? "History uses program-history evidence only."
        : "Context combines program history with this team's published preseason context evidence.",
    ),
  ];
  if (evidence) {
    const groups = node("div", "preseason-input-groups");
    ["program_history", "coach", "recruiting", "talent", "transfers"].forEach((key) => {
      if (evidence[key]) groups.append(preseasonEvidenceGroup(evidence[key]));
    });
    children.push(groups);
    const caveat = inputs?.provenance?.transfer_caveat;
    if (caveat) {
      const body = node("p", "preseason-provenance-detail", caveat.detail || "Transfer evidence is not a literal archived cutoff information state.");
      children.push(disclosure(caveat.visible_label, body, "preseason-provenance"));
    }
  } else {
    children.push(node("p", "preseason-starting-point-note", "This retained publication does not include the detailed preseason-evidence payload."));
  }
  children.push(
    disclosure(
      "preseason rank distribution",
      rankDistributionChart(row.team_name, team, distribution.rank_count, "Preseason rank distribution"),
      "distribution-disclosure preseason-distribution-disclosure",
    ),
  );
  content.replaceChildren(...children);
}

function trajectoryChart(teamName, points, rankCount) {
  const width = 800;
  const height = 310;
  const left = 58;
  const right = 18;
  const top = 24;
  const bottom = 74;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const xFor = (index) => left + (points.length <= 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth);
  const yFor = (value) => top + ((Number(value) - 1) / Math.max(rankCount - 1, 1)) * plotHeight;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "season-trajectory-chart",
    role: "img",
    "aria-labelledby": "season-trajectory-chart-title season-trajectory-chart-description",
  });
  const title = svgElement("title", { id: "season-trajectory-chart-title" });
  title.textContent = `${teamName} published expected rank trajectory`;
  const description = svgElement("desc", { id: "season-trajectory-chart-description" });
  description.textContent = points.map((point) => `${point.display_label}: ${rankSummaryLine(point.teams)}.`).join(" ");
  svg.append(title, description);
  const ticks = [...new Set([1, Math.round(rankCount * 0.25), Math.round(rankCount * 0.5), Math.round(rankCount * 0.75), rankCount])].sort((a, b) => a - b);
  ticks.forEach((value) => {
    const y = yFor(value);
    svg.append(svgElement("line", { x1: left, y1: y, x2: width - right, y2: y, class: "trajectory-gridline" }));
    const label = svgElement("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "trajectory-axis-label" });
    label.textContent = `#${value}`;
    svg.append(label);
  });
  const path = points.map((point, index) => `${index ? "L" : "M"} ${xFor(index)} ${yFor(point.teams.expected_rank)}`).join(" ");
  svg.append(svgElement("path", { d: path, class: "trajectory-expected-line" }));
  points.forEach((point, index) => {
    const summary = point.teams;
    const x = xFor(index);
    svg.append(svgElement("line", {
      x1: x,
      y1: yFor(summary.interval_80[0]),
      x2: x,
      y2: yFor(summary.interval_80[1]),
      class: "trajectory-interval",
    }));
    svg.append(svgElement("circle", {
      cx: x,
      cy: yFor(summary.expected_rank),
      r: point.publication_status === "official" ? 4.5 : 3.5,
      class: point.publication_status === "official" ? "trajectory-point trajectory-point-official" : "trajectory-point",
    }));
    const label = svgElement("text", {
      x,
      y: height - bottom + 22,
      "text-anchor": "end",
      transform: `rotate(-35 ${x} ${height - bottom + 22})`,
      class: "trajectory-snapshot-label",
    });
    label.textContent = point.display_label;
    svg.append(label);
  });
  const axisTitle = svgElement("text", { x: left + plotWidth / 2, y: height - 7, "text-anchor": "middle", class: "trajectory-axis-title" });
  axisTitle.textContent = "Published snapshots · rank 1 is best";
  svg.append(axisTitle);
  const figure = node("figure", "season-trajectory-figure");
  figure.append(svg);
  const caption = node("figcaption", "trajectory-caption", "Line: expected rank · vertical bars: central 80% interval · filled points: official publications.");
  figure.append(caption);
  return figure;
}

function trajectoryPointList(points, row, rankCount) {
  const list = node("ol", "trajectory-point-list");
  points.forEach((point) => {
    const summary = point.teams;
    const item = node("li", "trajectory-point-item");
    const facts = node("div", "trajectory-point-facts");
    facts.append(
      node("strong", "", point.display_label),
      node("span", "", `Expected ${rank(summary.expected_rank)}`),
      node("span", "", `80% ${rankRange(summary.interval_80)}`),
      node("span", "", `${percentage(summary.top25_probability)} Top 25`),
    );
    item.append(facts, lazyRankDistributionDisclosure(point, row.team_id, rankCount, row.team_name));
    list.append(item);
  });
  return list;
}

function renderSeasonMovement(entry, trajectory, row, rankCount) {
  const section = $("#season-movement-section");
  const content = $("#season-movement");
  const context = $("#season-movement-context");
  if (entry.ranking_family === "performance") {
    section.hidden = false;
    context.textContent = "Performance is a separate view of completed-game evidence.";
    content.replaceChildren(node("p", "season-movement-not-applicable", "Performance has no preseason starting point in the same semantic sense, so this page does not show a preseason-to-current comparison."));
    return;
  }
  if (!trajectory?.points?.length) {
    section.hidden = false;
    context.textContent = `Published ${priorLabel(entry.prior_family)} belief`;
    content.replaceChildren(node("p", "season-movement-unavailable", "A published belief trajectory is unavailable for this retained snapshot."));
    return;
  }
  const currentIndex = trajectory.points.findIndex((point) => point.snapshot_id === entry.snapshot_id);
  const points = trajectory.points.slice(0, currentIndex + 1).filter((point) => point.teams?.[row.team_id]).map((point) => ({ ...point, teams: point.teams[row.team_id] }));
  if (currentIndex < 0 || !points.length) {
    section.hidden = false;
    context.textContent = `Published ${priorLabel(entry.prior_family)} belief`;
    content.replaceChildren(node("p", "season-movement-unavailable", "This selected snapshot is not present in its published trajectory."));
    return;
  }
  section.hidden = false;
  context.textContent = `Published ${priorLabel(entry.prior_family)} belief · expected rank with central 80% intervals.`;
  content.replaceChildren(
    node("p", "season-movement-intro", `Preseason through ${entry.display_label}. Each point is a published belief; adjacent points are not attributed to one specific game.`),
    trajectoryChart(row.team_name, points, rankCount),
    trajectoryPointList(points, row, rankCount),
  );
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
  const thresholds = Object.entries(summary.threshold_probabilities || {})
    .filter(([label]) => label.startsWith("wins_"))
    .map(([label, probability]) => `${label.replace(/^wins_(\d+)_plus$/, "$1+")} ${percentage(probability)}`)
    .join(" · ");
  const details = node("dl", "season-outlook-details");
  [
    ["Expected final wins", `${Number(summary.expected_final_wins).toFixed(1)}`],
    ["Likely record", likelyRecord(summary)],
    ["Central 80%", `${summary.final_win_interval_80[0]}–${summary.final_win_interval_80[1]} wins`],
    ...(thresholds ? [["Win milestones", thresholds]] : []),
  ].forEach(([label, value]) => details.append(node("dt", "", label), node("dd", "", value)));
  const contentDetail = node("div", "season-outlook-disclosure-content");
  const chart = seasonWinChart(summary);
  if (chart) contentDetail.append(chart);
  const records = Object.entries(summary.record_probabilities || {})
    .sort((left, right) => Number(right[1]) - Number(left[1]) || left[0].localeCompare(right[0]))
    .slice(0, 5);
  const recordList = node("ul", "season-outlook-records");
  records.forEach(([record, probability]) => recordList.append(node("li", "", `${record} · ${percentage(probability)}`)));
  contentDetail.append(recordList);
  content.replaceChildren(details, disclosure("final-win distribution", contentDetail, "distribution-disclosure season-outlook-disclosure"));
}

function orientedPrediction(prediction, team) {
  const focalIsHome = prediction.home_team_id === team.team_id;
  const opponentName = focalIsHome ? prediction.away_team_name : prediction.home_team_name;
  const focalWin = focalIsHome ? prediction.home_win_probability : prediction.away_win_probability;
  const opponentWin = focalIsHome ? prediction.away_win_probability : prediction.home_win_probability;
  const expected = focalIsHome ? prediction.expected_home_margin : -prediction.expected_home_margin;
  const median = focalIsHome ? prediction.median_home_margin : -prediction.median_home_margin;
  const orientInterval = (interval) => focalIsHome ? [interval[0], interval[1]] : [-interval[1], -interval[0]];
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
  const favorite = oriented.focalWin >= oriented.opponentWin ? [oriented.focalName, oriented.focalWin] : [oriented.opponentName, oriented.opponentWin];
  const panel = node("div", "game-prediction");
  const expectedText = oriented.expected >= 0 ? marginSide(oriented.focalName, oriented.expected) : marginSide(oriented.opponentName, -oriented.expected);
  const expectedTeam = oriented.expected >= 0 ? oriented.focalName : oriented.opponentName;
  const expectedPrimary = Math.abs(oriented.expected) < 0.05 ? "expected Even" : expectedTeam === favorite[0] ? `expected by ${marginValue(oriented.expected)}` : `expected ${expectedTeam} by ${marginValue(oriented.expected)}`;
  panel.append(
    node("strong", "game-prediction-title", `${favorite[0]} ${percentage(favorite[1])} · ${expectedPrimary}`),
    node("span", "game-prediction-interval", `80% ${predictionRange(oriented, oriented.interval80)}`),
  );
  const detail = node("div", "game-prediction-disclosure-content");
  const chart = futureChart(prediction, team, axis, oriented);
  if (chart) detail.append(chart);
  const list = node("dl", "game-rating-details");
  [["Focal win probability", percentage(oriented.focalWin)], ["Opponent win probability", percentage(oriented.opponentWin)], ["Expected margin", expectedText], ["Median margin", oriented.median >= 0 ? marginSide(oriented.focalName, oriented.median) : marginSide(oriented.opponentName, -oriented.median)], ["Central 50% range", predictionRange(oriented, oriented.interval50)], ["Central 80% range", predictionRange(oriented, oriented.interval80)], ["Central 95% range", predictionRange(oriented, oriented.interval95)], ["Prediction source", prediction.prediction_source === "predictive_history" ? "Predictive History" : "Predictive Context"]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  detail.append(list);
  panel.append(disclosure("predictive margin distribution", detail, "distribution-disclosure game-distribution-disclosure"));
  panel.setAttribute("aria-label", `${favorite[0]} has a ${percentage(favorite[1])} win probability. Expected margin: ${expectedText}. Central 80% range: ${predictionRange(oriented, oriented.interval80)}.`);
  return panel;
}

function ratingPanel(rating, axis) {
  const panel = node("div", "game-rating");
  const percentile = rating.performance_percentile === undefined ? null : percentileLabel(rating.performance_percentile);
  const grade = rating.performance_grade ? `${rating.performance_grade} · ` : "";
  const title = node("strong", "game-rating-title", percentile ? `${grade}${percentile}` : "Inferred performance");
  const interval = node("span", "game-rating-interval", percentile ? `${uncertaintyLabel(rating.interval_80, rating.rank_count)} uncertainty · 80% ${rankRange(rating.interval_80)}` : `80% ${rankRange(rating.interval_80)}`);
  interval.setAttribute("aria-label", `central 80 percent interval from rank ${rating.interval_80[0]} through rank ${rating.interval_80[1]}`);
  const detail = node("div", "game-rating-disclosure-content");
  const chart = performanceChart(rating, axis);
  if (chart) detail.append(chart);
  const list = node("dl", "game-rating-details");
  [["Expected performance rank", rank(rating.expected_rank)], ["Median", `#${rating.median_rank}`], ["Mode", `#${rating.mode_rank}`], ["50% interval", rankRange(rating.interval_50)], ["80% interval", rankRange(rating.interval_80)], ["95% interval", rankRange(rating.interval_95)], ["Top 5", percentage(rating.top5_probability)], ["Top 10", percentage(rating.top10_probability)], ["Top 25", percentage(rating.top25_probability)]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  detail.append(list);
  panel.append(title, interval, disclosure("inferred performance distribution", detail, "distribution-disclosure game-distribution-disclosure"));
  return panel;
}

function expectedDelta(before, after) {
  const delta = Number(before.expected_rank) - Number(after.expected_rank);
  if (Math.abs(delta) < 0.05) return "—";
  return `${delta > 0 ? "↑" : "↓"}${Math.abs(delta).toFixed(1)}`;
}

function beliefMovementForGame(game, trajectory, entry, teamId) {
  if (!trajectory?.points?.length) return null;
  const selectedIndex = trajectory.points.findIndex((point) => point.snapshot_id === entry.snapshot_id);
  if (selectedIndex < 1) return null;
  const visible = trajectory.points.slice(0, selectedIndex + 1);
  const afterIndex = visible.findIndex((point, index) => {
    if (!index || !Array.isArray(point.included_game_ids)) return false;
    const beforeIds = new Set(visible[index - 1].included_game_ids || []);
    return point.included_game_ids.includes(game.game_id) && !beforeIds.has(game.game_id);
  });
  if (afterIndex < 1) return null;
  const before = visible[afterIndex - 1];
  const after = visible[afterIndex];
  const beforeSummary = before.teams?.[teamId];
  const afterSummary = after.teams?.[teamId];
  if (!beforeSummary || !afterSummary) return null;
  const priorIds = new Set(before.included_game_ids || []);
  const addedGames = (after.included_game_ids || []).filter((gameId) => !priorIds.has(gameId)).length;
  return { before, after, beforeSummary, afterSummary, addedGames };
}

function beliefMovementPanel(game, trajectory, entry, teamId) {
  const movement = beliefMovementForGame(game, trajectory, entry, teamId);
  if (!movement) return null;
  const panel = node("section", "game-belief-movement");
  panel.append(node("h4", "", "Published belief movement"));
  panel.append(node("p", "game-belief-scope", "Shared published update — it can include several newly included results, not a causal attribution to this game alone."));
  const facts = node("dl", "game-belief-facts");
  facts.append(
    node("dt", "", "Before"), node("dd", "", `${movement.before.display_label} · ${rank(movement.beforeSummary.expected_rank)}`),
    node("dt", "", "After"), node("dd", "", `${movement.after.display_label} · ${rank(movement.afterSummary.expected_rank)}`),
    node("dt", "", "Change"), node("dd", "", expectedDelta(movement.beforeSummary, movement.afterSummary)),
  );
  const detail = node("div", "game-belief-detail-content");
  detail.append(
    node("p", "", `Top 25: ${percentage(movement.beforeSummary.top25_probability)} → ${percentage(movement.afterSummary.top25_probability)} · 80%: ${rankRange(movement.beforeSummary.interval_80)} → ${rankRange(movement.afterSummary.interval_80)}.`),
    node("p", "game-belief-caveat", `This is ${movement.before.display_label} published belief → ${movement.after.display_label} published belief. The transition incorporated ${movement.addedGames} newly included result${movement.addedGames === 1 ? "" : "s"}; it is not attribution to this game alone.`),
  );
  panel.append(facts, disclosure("published belief details", detail, "distribution-disclosure game-belief-disclosure"));
  return panel;
}

function gameCard(game, cutoff, artifact, team, trajectory, entry) {
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
    const movement = beliefMovementPanel(game, trajectory, entry, team.team_id);
    if (movement) body.append(movement);
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

function renderSchedule(artifact, entry, trajectory) {
  const teamId = params.get("team");
  const team = artifact.teams?.[teamId];
  if (!team) throw new Error("This team is not available in the selected season snapshot.");
  const cutoff = artifact.effective_cutoff ? new Date(artifact.effective_cutoff) : null;
  const games = Array.isArray(team.games) ? team.games : [];
  $("#team-page-status").textContent = games.length ? `${games.length} scheduled games · compact summaries reveal distributions on demand` : "No schedule entries are available for this team.";
  $("#schedule-list").replaceChildren(...games.map((game) => gameCard(game, cutoff, artifact, team, trajectory, entry)));
}

function publicationStatusLabel(entry) {
  return entry.publication_status === "official" ? "Official" : "Interim";
}

function priorLabel(prior) {
  return prior === "history" ? "History" : "Context";
}

function entryFor(manifest) {
  const family = params.get("family") === "performance" ? "performance" : "predictive";
  const season = Number(params.get("season")) || manifest.seasons[0];
  const seasonEntries = manifest.snapshots.filter((item) => item.season === season && item.ranking_family === family);
  const snapshot = params.get("snapshot");
  const exact = seasonEntries.find((item) => item.snapshot_id === snapshot);
  if (exact) return exact;
  let entries = seasonEntries;
  const prior = params.get("prior") === "history" ? "history" : "context";
  if (family === "predictive") entries = entries.filter((item) => item.prior_family === prior);
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

async function optionalJson(path) {
  if (!path) return null;
  const response = await fetch(`./${path}`);
  if (!response.ok) throw new Error("The selected team-season artifact is unavailable.");
  return response.json();
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
  const predictive = entry.ranking_family === "predictive";
  const samePreseason = entry.preseason_snapshot_id === entry.snapshot_id;
  const [snapshot, artifact, distribution, loadedPreseasonDistribution, loadedPreseasonArtifact, trajectory] = await Promise.all([
    optionalJson(entry.data_path),
    optionalJson(entry.team_seasons_path || entry.team_season_path),
    optionalJson(entry.distribution_path),
    predictive && !samePreseason ? optionalJson(entry.preseason_distribution_path) : Promise.resolve(null),
    predictive && !samePreseason ? optionalJson(entry.preseason_team_seasons_path) : Promise.resolve(null),
    predictive ? optionalJson(entry.season_trajectory_path) : Promise.resolve(null),
  ]);
  const preseasonDistribution = loadedPreseasonDistribution || (samePreseason ? distribution : null);
  const preseasonArtifact = loadedPreseasonArtifact || (samePreseason ? artifact : null);
  if (artifact.snapshot_id !== entry.snapshot_id || artifact.season !== entry.season) throw new Error("The team-season artifact does not match the selected snapshot.");
  const row = snapshot.rankings.find((item) => item.team_id === teamId);
  if (!row) throw new Error("This team is not available in the selected ranking snapshot.");
  distributionTeam(distribution, entry.snapshot_id, snapshot.rank_count, teamId, "Selected snapshot");
  updatePriorSwitch(manifest, entry);
  renderSummary(entry, snapshot, row, artifact.season_simulation, distribution);
  renderPreseasonStartingPoint(entry, preseasonDistribution || distribution, preseasonArtifact, row);
  renderSeasonMovement(entry, trajectory, row, snapshot.rank_count);
  renderSeasonOutlook(artifact.season_simulation, artifact.teams[teamId]);
  renderSchedule(artifact, entry, trajectory);
}

load().catch((error) => {
  updateBackLink(null);
  $("#team-page-status").textContent = error.message;
  $("#team-page-status").className = "team-page-status team-page-error";
});
