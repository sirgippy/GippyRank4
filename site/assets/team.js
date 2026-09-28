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

function percentage(value) {
  if (value === 1) return "100%";
  if (value === 0) return "0%";
  const percent = Number(value) * 100;
  if (!Number.isFinite(percent)) return "Unavailable";
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(1)}%`;
  return `${Math.round(percent)}%`;
}

function tailPercentage(value) {
  const percent = Number(value) * 100;
  if (!Number.isFinite(percent) || percent >= 10) return percentage(value);
  if (percent === 0) return "0%";
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(2)}%`;
  return `${percent.toFixed(1)}%`;
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

function ordinal(value) {
  const rounded = Math.round(Number(value));
  if (!Number.isFinite(rounded)) return "Unavailable";
  const suffix = rounded % 100 >= 11 && rounded % 100 <= 13
    ? "th"
    : ({ 1: "st", 2: "nd", 3: "rd" }[rounded % 10] || "th");
  return `${rounded}${suffix}`;
}

function percentileLabel(value) {
  const formatted = ordinal(value);
  return formatted === "Unavailable" ? formatted : `${formatted} percentile`;
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

function disclosure(label, content, className = "distribution-disclosure", summaryText = null) {
  const details = node("details", className);
  const summary = node("summary", "distribution-toggle", "▾");
  const updateLabel = () => {
    const verb = details.open ? "Hide" : "Show";
    summary.setAttribute("aria-label", summaryText ? `${summaryText}. ${verb} ${label}` : `${verb} ${label}`);
    summary.title = `${verb} ${label}`;
  };
  updateLabel();
  details.addEventListener("toggle", updateLabel);
  details.append(summary, content);
  return details;
}

function labeledDisclosure(label, content, className) {
  const details = node("details", className);
  const summary = node("summary", "preseason-provenance-summary");
  summary.append(
    node("span", "", label),
    node("span", "preseason-provenance-chevron", "▾"),
  );
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

function retrospectiveChart(expectation, team, axis, oriented) {
  const display = orientedDisplayDistribution(expectation, team, axis);
  if (!display) return null;
  const offAxis = (display.lowerTail + display.upperTail) / displayScale(axis);
  const figure = node("figure", "game-distribution game-distribution-retrospective");
  const label = `${team.team_name} retrospective margin distribution. Expected ${oriented.expectedText}; actual ${oriented.actualText}.`;
  const svg = densitySvg(display.masses, {
    className: "retrospective-distribution-chart",
    label,
    description: "Bars show otherwise expected outcomes; the gold line marks the expected margin and the dark line marks the actual margin. The horizontal axis is from the team's perspective.",
    zero: true,
  });
  const minimum = Number(axis.min_margin);
  const maximum = Number(axis.max_margin);
  for (const [kind, value] of [["expected", oriented.expected], ["actual", oriented.actual]]) {
    const x = 5 + 330 * (Math.max(minimum, Math.min(maximum, value)) - minimum) / (maximum - minimum);
    svg.append(svgElement("line", {
      x1: x, y1: 4, x2: x, y2: 69,
      class: `retrospective-marker retrospective-marker-${kind}`,
      "data-margin": value,
    }));
  }
  figure.append(svg);
  const caption = node("figcaption", "distribution-axis");
  caption.append(
    node("span", "axis-start", `${oriented.opponentName} by ${Math.abs(minimum)}`),
    node("span", "axis-label", "Even"),
    node("span", "axis-end", `${oriented.focalName} by ${maximum}`),
  );
  figure.append(caption);
  const legend = node("p", "retrospective-chart-legend");
  legend.append(
    node("span", "retrospective-legend-expected", `Expected ${oriented.expectedText}`),
    node("span", "retrospective-legend-actual", `Actual ${oriented.actualText}`),
  );
  figure.append(legend);
  if (offAxis > 0.001) {
    figure.append(node("p", "distribution-tail-note", `${percentage(offAxis)} of otherwise-expected outcomes lie beyond the visible ±${Math.max(Math.abs(minimum), Math.abs(maximum))}-point range.`));
  }
  if (Math.abs(oriented.actual) > maximum || Math.abs(oriented.expected) > maximum) {
    figure.append(node("p", "distribution-tail-note", "A marker at the edge is beyond the displayed margin range."));
  }
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
  const label = checkpointLabel(point);
  const details = disclosure(`${label} rank distribution`, body, "distribution-disclosure trajectory-distribution-disclosure");
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
      body.replaceChildren(rankDistributionChart(teamName, team, rankCount, `${label} rank distribution`));
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

function scalarDistributionDisclosure(label, summaryText, content, className) {
  const details = node("details", `${className} scalar-distribution-disclosure`);
  const summary = node("summary", "scalar-distribution-summary");
  const text = node("span", "scalar-distribution-text", summaryText);
  const chevron = node("span", "scalar-distribution-chevron", "▾");
  const updateLabel = () => {
    const verb = details.open ? "Hide" : "Show";
    summary.setAttribute("aria-label", `${summaryText}. ${verb} ${label}`);
    summary.title = `${verb} ${label}`;
  };
  updateLabel();
  details.addEventListener("toggle", updateLabel);
  summary.append(text, chevron);
  details.append(summary, content);
  return details;
}

function renderSummary(entry, row) {
  const rankLabel = row.rated === false ? "NR" : `#${row.display_rank}`;
  const logo = teamLogo(row.team_id, "team-logo team-logo-card");
  $("#team-page-logo").replaceChildren(...(logo ? [logo] : []));
  $("#team-page-kind").textContent = `${entry.season} ${entry.display_label} · ${publicationStatusLabel(entry)} publication`;
  $("#team-page-title").textContent = row.team_name;
  $("#team-page-meta").textContent = `${row.conference || "Independent"} · ${entry.snapshot_type === "preseason" ? "Before game evidence" : `Through ${formatDate(entry.effective_cutoff)}`} · ${entry.ranking_family === "performance" ? "Performance" : `Predictive ${priorLabel(entry.prior_family)}`}`;
  $("#schedule-context").textContent = entry.snapshot_type === "preseason"
    ? "Before the season"
    : `Through ${formatDate(entry.effective_cutoff, false)}`;
  $("#prediction-source").textContent = `Completed-game expectations compare each result with the selected snapshot's other evidence, excluding that game. They are retrospective, not pregame forecasts. Checkpoint rows show shared published belief changes. Future matchups use ${entry.ranking_family === "performance" ? "Predictive Context" : `Predictive ${priorLabel(entry.prior_family)}`} at this snapshot.`;

  const summary = node("div", "team-summary-grid");
  summary.append(summaryItem("Published rank", rankLabel), summaryItem("Record", row.record));
  if (entry.ranking_family === "performance") {
    summary.append(
      summaryItem("Expected rank", rank(row.expected_rank)),
      summaryItem("Central 80%", rankRange(row.interval_80)),
      summaryItem("Top 25 probability", percentage(row.top25_probability)),
    );
  }
  $("#team-ranking-summary").replaceChildren(summary);
}

function comparisonText(comparison) {
  if (!comparison || typeof comparison !== "object") return null;
  const rankValue = comparison.fbs_rank;
  const count = comparison.count;
  const percentile = comparison.fbs_percentile;
  if (!Number.isFinite(Number(rankValue)) || !Number.isFinite(Number(count))) return null;
  const parts = [`FBS #${rankValue}/${count}`];
  if (Number.isFinite(Number(percentile))) parts.push(`${ordinal(percentile)} pct.`);
  return parts.join(" · ");
}

function readableProvenanceValue(value) {
  if (value === null || value === undefined || value === "") return "Unavailable";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function preseasonEvidenceGroup(key, group) {
  const section = node("article", `preseason-input-group preseason-input-group-${key}`);
  section.append(node("h3", "", group.title));
  const list = node("div", "preseason-input-values");
  const provenance = [];
  if (group.source) provenance.push(["Source", group.source]);
  (group.fields || []).forEach((field) => {
    const fact = node("div", "preseason-input-value");
    fact.append(node("span", "preseason-input-label", field.label), node("strong", "", field.display_value));
    const comparison = comparisonText(field.comparison);
    if (comparison) fact.append(node("span", "preseason-input-comparison", comparison));
    list.append(fact);
    const detail = [field.source, field.detail, field.model_feature ? `Model feature: ${field.model_feature}` : null]
      .filter(Boolean)
      .join(" · ");
    if (detail) provenance.push([field.label, detail]);
  });
  section.append(list);
  if (group.note) provenance.push(["Note", group.note]);
  if (provenance.length) {
    const detail = node("dl", "preseason-input-provenance");
    provenance.forEach(([label, value]) => detail.append(node("dt", "", label), node("dd", "", value)));
    section.append(labeledDisclosure("Source & model detail", detail, "preseason-input-provenance-disclosure"));
  }
  return section;
}

function preseasonProvenanceDisclosure(provenance) {
  if (!provenance || typeof provenance !== "object") return null;
  const entries = Object.entries(provenance).filter(([key]) => key !== "transfer_caveat");
  if (!entries.length) return null;
  const detail = node("dl", "preseason-provenance-detail-list");
  entries.forEach(([key, value]) => {
    detail.append(
      node("dt", "", key.replaceAll("_", " ")),
      node("dd", "", readableProvenanceValue(value)),
    );
  });
  return labeledDisclosure("Evidence provenance", detail, "preseason-provenance");
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
    scalarDistributionDisclosure(
      "preseason rank distribution",
      `Preseason belief: expected ${rank(team.summary.expected_rank)} · 80% ${rankRange(team.summary.interval_80)}`,
      rankDistributionChart(row.team_name, team, distribution.rank_count, "Preseason rank distribution"),
      "preseason-distribution-disclosure",
    ),
  ];
  if (evidence) {
    const groups = node("div", "preseason-input-groups");
    ["program_history", "coach", "recruiting", "talent", "transfers"].forEach((key) => {
      if (evidence[key]) groups.append(preseasonEvidenceGroup(key, evidence[key]));
    });
    children.push(groups);
    const caveat = inputs?.provenance?.transfer_caveat;
    if (caveat) {
      const body = node("p", "preseason-provenance-detail", caveat.detail || "Transfer evidence is not a literal archived cutoff information state.");
      children.push(labeledDisclosure(caveat.visible_label, body, "preseason-provenance"));
    }
    const provenance = preseasonProvenanceDisclosure(inputs?.provenance);
    if (provenance) children.push(provenance);
  } else {
    children.push(node("p", "preseason-starting-point-note", "This retained publication does not include the detailed preseason-evidence payload."));
  }
  content.replaceChildren(...children);
}

function checkpointLabel(point) {
  if (point.snapshot_type === "preseason") return "Preseason";
  if (point.publication_status === "official") {
    return point.display_label.replace(/\s*\([^)]*retrospective[^)]*\)\s*/i, "").trim();
  }
  const date = point.effective_cutoff ? formatDate(point.effective_cutoff, false) : point.display_label;
  return `${date} interim`;
}

function intervalWidth(summary) {
  const explicit = Number(summary.interval_80_width);
  if (Number.isFinite(explicit)) return explicit;
  return Number(summary.interval_80[1]) - Number(summary.interval_80[0]) + 1;
}

function intervalWidthMovement(summary) {
  const change = Number(summary.interval_80_width_change);
  return intervalWidthChangeDescription(change);
}

function intervalWidthChangeDescription(change) {
  if (!Number.isFinite(change)) return "starting range";
  if (change === 0) return "same width";
  const ranks = `${Math.abs(change)} rank${Math.abs(change) === 1 ? "" : "s"}`;
  return `${ranks} ${change < 0 ? "narrower" : "wider"}`;
}

function adaptiveRankDomain(points, rankCount) {
  const bounds = points.flatMap((point) => point.teams.interval_80.map(Number));
  const low = Math.max(1, Math.min(...bounds));
  const high = Math.min(rankCount, Math.max(...bounds));
  const padding = Math.max(2, Math.ceil(Math.max(high - low, 1) * 0.16));
  let minimum = Math.max(1, low - padding);
  let maximum = Math.min(rankCount, high + padding);
  const targetSpan = Math.min(Math.max(rankCount - 1, 0), 6);
  if (maximum - minimum < targetSpan) {
    const extra = targetSpan - (maximum - minimum);
    minimum = Math.max(1, minimum - Math.ceil(extra / 2));
    maximum = Math.min(rankCount, maximum + Math.floor(extra / 2));
    if (maximum - minimum < targetSpan) {
      if (minimum === 1) maximum = Math.min(rankCount, minimum + targetSpan);
      else minimum = Math.max(1, maximum - targetSpan);
    }
  }
  return { minimum, maximum };
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
  const domain = adaptiveRankDomain(points, rankCount);
  const yFor = (value) => top + ((Number(value) - domain.minimum) / Math.max(domain.maximum - domain.minimum, 1)) * plotHeight;
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "season-trajectory-chart",
    role: "img",
    "aria-labelledby": "season-trajectory-chart-title season-trajectory-chart-description",
    "data-rank-domain-min": domain.minimum,
    "data-rank-domain-max": domain.maximum,
  });
  const title = svgElement("title", { id: "season-trajectory-chart-title" });
  title.textContent = `${teamName} published expected rank trajectory`;
  const description = svgElement("desc", { id: "season-trajectory-chart-description" });
  description.textContent = `Adaptive rank view #${domain.minimum} through #${domain.maximum}; rank 1 is best. ${points.map((point) => `${checkpointLabel(point)}: expected ${rank(point.teams.expected_rank)}, central 80% ${rankRange(point.teams.interval_80)}, width ${intervalWidth(point.teams)} ranks (${intervalWidthMovement(point.teams)}), Top 25 ${percentage(point.teams.top25_probability)}.`).join(" ")}`;
  svg.append(title, description);
  const ticks = [...new Set(Array.from({ length: 5 }, (_, index) => Math.round(domain.minimum + ((domain.maximum - domain.minimum) * index) / 4)))].sort((a, b) => a - b);
  ticks.forEach((value) => {
    const y = yFor(value);
    svg.append(svgElement("line", { x1: left, y1: y, x2: width - right, y2: y, class: "trajectory-gridline" }));
    const label = svgElement("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "trajectory-axis-label" });
    label.textContent = `#${value}`;
    svg.append(label);
  });
  const bandTop = points.map((point, index) => `${index ? "L" : "M"} ${xFor(index)} ${yFor(point.teams.interval_80[0])}`).join(" ");
  const bandBottom = [...points].reverse().map((point, reverseIndex) => {
    const index = points.length - reverseIndex - 1;
    return `L ${xFor(index)} ${yFor(point.teams.interval_80[1])}`;
  }).join(" ");
  svg.append(svgElement("path", { d: `${bandTop} ${bandBottom} Z`, class: "trajectory-uncertainty-band" }));
  const path = points.map((point, index) => `${index ? "L" : "M"} ${xFor(index)} ${yFor(point.teams.expected_rank)}`).join(" ");
  svg.append(svgElement("path", { d: path, class: "trajectory-expected-line" }));
  points.forEach((point, index) => {
    const summary = point.teams;
    const x = xFor(index);
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
    label.textContent = checkpointLabel(point);
    svg.append(label);
  });
  const axisTitle = svgElement("text", { x: left + plotWidth / 2, y: height - 7, "text-anchor": "middle", class: "trajectory-axis-title" });
  axisTitle.textContent = `Team view · rank 1 is best · #${domain.minimum}–#${domain.maximum}`;
  svg.append(axisTitle);
  const figure = node("figure", "season-trajectory-figure");
  figure.append(svg);
  const caption = node("figcaption", "trajectory-caption", "Line: expected rank · shaded band: central 80% interval. The axis adapts to this team's published range.");
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
      node("strong", "", checkpointLabel(point)),
      node("span", "", `Expected ${rank(summary.expected_rank)}`),
      node("span", "", `80% ${rankRange(summary.interval_80)}`),
      node("span", "", `Width ${intervalWidth(summary)} · ${intervalWidthMovement(summary)}`),
      node("span", "", `${percentage(summary.top25_probability)} Top 25`),
    );
    item.append(facts, lazyRankDistributionDisclosure(point, row.team_id, rankCount, row.team_name));
    list.append(item);
  });
  return list;
}

function seasonStoryMetric(label, preseasonValue, currentValue, currentDistribution = null) {
  const metric = node("div", "season-story-metric");
  metric.append(node("dt", "", label));
  const values = node("dd", "season-story-values");
  if (preseasonValue !== null) {
    values.append(node("span", "season-story-before", preseasonValue), node("span", "season-story-arrow", "→"));
  }
  if (currentDistribution) values.append(currentDistribution);
  else values.append(node("strong", "season-story-current", currentValue));
  metric.append(values);
  return metric;
}

function mostInformativeSeasonStoryProbability(preseason, current) {
  const candidates = [
    ["Top 5 probability", "top5_probability"],
    ["Top 10 probability", "top10_probability"],
    ["Top 25 probability", "top25_probability"],
  ].filter(([, key]) => Number.isFinite(preseason[key]) && Number.isFinite(current[key]))
    .map(([label, key]) => ({
      label,
      preseasonValue: preseason[key],
      currentValue: current[key],
      change: Math.abs(current[key] - preseason[key]),
    }))
    .filter((candidate) => candidate.change >= 0.05)
    .sort((left, right) => right.change - left.change);
  return candidates[0] ?? null;
}

function renderSeasonStory(entry, points, row, rankCount, distribution) {
  const section = $("#season-story-section");
  const content = $("#season-story-content");
  if (entry.ranking_family !== "predictive" || !points.length) {
    section.hidden = true;
    content.replaceChildren();
    return;
  }
  const preseason = points[0];
  const current = points.at(-1);
  const startsAtPreseason = preseason.snapshot_id === current.snapshot_id;
  const currentTeam = distributionTeam(distribution, entry.snapshot_id, rankCount, row.team_id, "Selected snapshot");
  const currentExpected = rank(current.teams.expected_rank);
  const currentDistribution = scalarDistributionDisclosure(
    "current rank distribution",
    currentExpected,
    rankDistributionChart(row.team_name, currentTeam, rankCount, "Current rank distribution"),
    "season-story-distribution",
  );
  const metrics = node("dl", "season-story-metrics");
  metrics.append(
    seasonStoryMetric("Expected rank", startsAtPreseason ? null : rank(preseason.teams.expected_rank), currentExpected, currentDistribution),
    seasonStoryMetric("Central 80%", startsAtPreseason ? null : rankRange(preseason.teams.interval_80), rankRange(current.teams.interval_80)),
  );
  const probability = startsAtPreseason
    ? null
    : mostInformativeSeasonStoryProbability(preseason.teams, current.teams);
  if (probability) {
    metrics.append(seasonStoryMetric(
      probability.label,
      percentage(probability.preseasonValue),
      percentage(probability.currentValue),
    ));
  }
  const heading = startsAtPreseason ? "Preseason belief" : `Preseason → ${checkpointLabel(current)}`;
  const widthText = startsAtPreseason
    ? `The central 80% range starts ${intervalWidth(current.teams)} ranks wide.`
    : (() => {
      const change = intervalWidth(current.teams) - intervalWidth(preseason.teams);
      const comparison = change === 0
        ? "the same width as preseason"
        : `${intervalWidthChangeDescription(change)} than preseason`;
      return `The central 80% range is now ${intervalWidth(current.teams)} ranks wide — ${comparison}.`;
    })();
  const title = node("h2", "", heading);
  title.id = "season-story-title";
  content.replaceChildren(
    node("p", "season-story-eyebrow", "Season story"),
    title,
    metrics,
    node("p", "season-story-uncertainty", widthText),
  );
  section.hidden = false;
}

function renderSeasonMovement(entry, trajectory, row, rankCount, distribution) {
  const section = $("#season-movement-section");
  const content = $("#season-movement");
  const context = $("#season-movement-context");
  if (entry.ranking_family === "performance") {
    renderSeasonStory(entry, [], row, rankCount, distribution);
    section.hidden = false;
    context.textContent = "Performance is a separate view of completed-game evidence.";
    content.replaceChildren(node("p", "season-movement-not-applicable", "Performance has no preseason starting point in the same semantic sense, so this page does not show a preseason-to-current comparison."));
    return;
  }
  if (!trajectory?.points?.length) {
    renderSeasonStory(entry, [], row, rankCount, distribution);
    section.hidden = false;
    context.textContent = `Published ${priorLabel(entry.prior_family)} belief`;
    content.replaceChildren(node("p", "season-movement-unavailable", "A published belief trajectory is unavailable for this retained snapshot."));
    return;
  }
  const currentIndex = trajectory.points.findIndex((point) => point.snapshot_id === entry.snapshot_id);
  const points = trajectory.points.slice(0, currentIndex + 1).filter((point) => point.teams?.[row.team_id]).map((point) => ({ ...point, teams: point.teams[row.team_id] }));
  if (currentIndex < 0 || !points.length) {
    renderSeasonStory(entry, [], row, rankCount, distribution);
    section.hidden = false;
    context.textContent = `Published ${priorLabel(entry.prior_family)} belief`;
    content.replaceChildren(node("p", "season-movement-unavailable", "This selected snapshot is not present in its published trajectory."));
    return;
  }
  if (points.length < 2) {
    renderSeasonStory(entry, points, row, rankCount, distribution);
    section.hidden = true;
    context.textContent = "";
    content.replaceChildren();
    return;
  }
  renderSeasonStory(entry, points, row, rankCount, distribution);
  section.hidden = false;
  context.textContent = `Published ${priorLabel(entry.prior_family)} belief · expected rank with central 80% intervals.`;
  content.replaceChildren(
    node("p", "season-movement-intro", "Meaningful weekly checkpoints show what changed in expected rank and in the width of the central 80% range. The adjacent points are not attributed to one specific game. Historical interim publications are omitted."),
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
    .map(([label, probability]) => {
      const match = /^wins_(\d+)_plus$/.exec(label);
      return match ? { wins: Number(match[1]), probability } : null;
    })
    .filter(Boolean)
    .sort((left, right) => left.wins - right.wins)
    .map(({ wins, probability }) => `${wins}+ ${percentage(probability)}`)
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
  content.replaceChildren(
    details,
    scalarDistributionDisclosure(
      "final-win distribution",
      "Final-win distribution",
      contentDetail,
      "season-outlook-disclosure",
    ),
  );
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
  const titleText = `${favorite[0]} ${percentage(favorite[1])} to win · ${expectedText}`;
  const title = node("strong", "game-prediction-title", titleText);
  const detail = node("div", "game-prediction-disclosure-content");
  const chart = futureChart(prediction, team, axis, oriented);
  if (chart) detail.append(chart);
  const list = node("dl", "game-rating-details");
  [["Focal win probability", percentage(oriented.focalWin)], ["Opponent win probability", percentage(oriented.opponentWin)], ["Expected margin", expectedText], ["Median margin", oriented.median >= 0 ? marginSide(oriented.focalName, oriented.median) : marginSide(oriented.opponentName, -oriented.median)], ["Central 50% range", predictionRange(oriented, oriented.interval50)], ["Central 80% range", predictionRange(oriented, oriented.interval80)], ["Central 95% range", predictionRange(oriented, oriented.interval95)], ["Prediction source", prediction.prediction_source === "predictive_history" ? "Predictive History" : "Predictive Context"]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  detail.append(list);
  const distribution = disclosure("predictive margin distribution", detail, "distribution-disclosure game-distribution-disclosure", titleText);
  const copy = node("span", "game-summary-copy");
  copy.append(title);
  distribution.querySelector("summary").replaceChildren(copy, node("span", "distribution-chevron", "▾"));
  panel.append(distribution);
  return panel;
}

function retrospectivePanel(expectation, team, axis, opponentName) {
  const focalIsHome = expectation.home_team_id === team.team_id;
  const orient = (value) => focalIsHome ? Number(value) : -Number(value);
  const orientInterval = (interval) => focalIsHome ? interval : [-interval[1], -interval[0]];
  const expected = orient(expectation.expected_home_margin);
  const actual = orient(expectation.actual_home_margin);
  const oriented = {
    focalName: team.team_name,
    opponentName,
    expected,
    actual,
    expectedText: marginSide(expected >= 0 ? team.team_name : opponentName, expected),
    actualText: marginSide(actual >= 0 ? team.team_name : opponentName, actual),
  };
  const favorable = actual >= expected;
  const tail = favorable
    ? (focalIsHome ? expectation.upper_tail_probability : expectation.lower_tail_probability)
    : (focalIsHome ? expectation.lower_tail_probability : expectation.upper_tail_probability);
  const percentile = focalIsHome
    ? expectation.observed_margin_percentile : expectation.upper_tail_probability;
  const panel = node("div", "game-retrospective");
  const expectedText = `Expected ${oriented.expectedText}`;
  const actualText = `Actual ${oriented.actualText}`;
  const tailText = `${tailPercentage(tail)} ${favorable ? "this favorable or better" : "this unfavorable or worse"}`;
  const detail = node("div", "game-retrospective-disclosure-content");
  const chart = retrospectiveChart(expectation, team, axis, oriented);
  if (chart) detail.append(chart);
  const facts = node("dl", "game-rating-details");
  [
    ["Observed percentile", percentileLabel(percentile * 100)],
    ["Median margin", marginSide(orient(expectation.median_home_margin) >= 0 ? team.team_name : opponentName, orient(expectation.median_home_margin))],
    ["Central 50%", predictionRange(oriented, orientInterval(expectation.margin_interval_50))],
    ["Central 80%", predictionRange(oriented, orientInterval(expectation.margin_interval_80))],
    ["Central 95%", predictionRange(oriented, orientInterval(expectation.margin_interval_95))],
  ].forEach(([label, value]) => facts.append(node("dt", "", label), node("dd", "", value)));
  detail.append(facts);
  const distribution = disclosure(`${team.team_name} completed-game retrospective distribution`, detail, "distribution-disclosure game-distribution-disclosure", `${expectedText}. ${actualText}. ${tailText}`);
  const copy = node("span", "game-summary-copy");
  copy.append(
    node("span", "game-retrospective-expected", expectedText),
    node("span", "game-retrospective-actual", actualText),
    node("span", "game-retrospective-tail", tailText),
  );
  distribution.querySelector("summary").replaceChildren(copy, node("span", "distribution-chevron", "▾"));
  panel.append(distribution);
  return panel;
}

function checkpointPanel(before, after, teamId) {
  const earlier = before.teams?.[teamId];
  const later = after.teams?.[teamId];
  if (!earlier || !later) return null;
  const item = node("li", "schedule-checkpoint");
  const title = node("span", "schedule-checkpoint-title", `${checkpointLabel(after)} · published belief`);
  const values = node("span", "schedule-checkpoint-values");
  const rankBefore = rank(earlier.expected_rank);
  const rankAfter = rank(later.expected_rank);
  const intervalBefore = rankRange(earlier.interval_80);
  const intervalAfter = rankRange(later.interval_80);
  values.append(
    node("span", "", `Expected rank ${rankBefore}${rankBefore === rankAfter ? "" : ` → ${rankAfter}`}`),
    node("span", "", `80% ${intervalBefore}${intervalBefore === intervalAfter ? "" : ` → ${intervalAfter}`}`),
  );
  item.append(title, values);
  return item;
}

function gameCard(game, cutoff, artifact, team) {
  const item = node("li", "game-card");
  item.dataset.gameId = game.game_id;
  const header = node("div", "game-card-header");
  header.append(node("p", "game-date", game.week === null ? formatDate(game.date, false) : `W${game.week} · ${formatDate(game.date, false)}`));
  const opponent = node("h3", "game-opponent", game.opponent_name || game.opponent_id || "Unknown opponent");
  const opponentHeading = node("div", "game-opponent-row");
  const logo = teamLogo(game.opponent_id);
  if (logo) opponentHeading.append(logo);
  opponentHeading.append(opponent);
  opponentHeading.append(node("span", "game-site", game.site));
  const outcome = node("p", "game-outcome");
  const future = game.game_state ? game.game_state === "future" : cutoff === null || new Date(game.date) > cutoff;
  if (game.result && game.score) outcome.textContent = `${game.result} ${game.score.team}–${game.score.opponent}`;
  else if (game.game_state === "cancelled") outcome.textContent = "Cancelled or postponed";
  else if (future) outcome.textContent = "Upcoming";
  else outcome.textContent = "Pending";
  const body = node("div", "game-card-body");
  if (future) {
    const prediction = artifact.future_predictions?.[game.future_prediction_id];
    if (prediction) body.append(predictionPanel(prediction, team, artifact.future_margin_axis));
    else body.append(node("p", "game-not-modeled", "Prediction unavailable"));
  } else if (game.result || game.game_state === "completed") {
    const expectations = artifact.retrospective_game_expectations;
    const expectation = expectations?.games?.[game.retrospective_expectation_id || game.game_id];
    if (expectation && expectation.source_snapshot_id === expectations.source_snapshot_id) {
      body.append(retrospectivePanel(expectation, team, expectations.margin_axis, game.opponent_name || "Opponent"));
    } else body.append(node("p", "game-not-modeled", "No retrospective expectation available"));
  } else {
    body.append(node("p", "game-not-modeled", "No result at this snapshot"));
  }
  item.classList.add(future ? "game-card-future" : "game-card-completed");
  item.append(header, opponentHeading, outcome, body);
  return item;
}

function renderSchedule(artifact, entry, trajectory) {
  const teamId = params.get("team");
  const team = artifact.teams?.[teamId];
  if (!team) throw new Error("This team is not available in the selected season snapshot.");
  const cutoff = artifact.effective_cutoff ? new Date(artifact.effective_cutoff) : null;
  const games = Array.isArray(team.games) ? team.games : [];
  $("#team-page-status").textContent = games.length ? `${games.length} scheduled games` : "No schedule entries are available for this team.";
  const items = games.map((game) => gameCard(game, cutoff, artifact, team));
  const selectedIndex = trajectory?.points?.findIndex((point) => point.snapshot_id === entry.snapshot_id) ?? -1;
  if (selectedIndex > 0) {
    const points = trajectory.points.slice(0, selectedIndex + 1);
    const insertions = [];
    points.slice(1).forEach((after, index) => {
      const checkpoint = checkpointPanel(points[index], after, teamId);
      if (!checkpoint) return;
      const cutoffTime = new Date(after.effective_cutoff).getTime();
      let afterGame = -1;
      games.forEach((game, gameIndex) => {
        if (new Date(game.date).getTime() <= cutoffTime) afterGame = gameIndex;
      });
      insertions.push({ afterGame, checkpoint });
    });
    insertions.reverse().forEach(({ afterGame, checkpoint }) => items.splice(afterGame + 1, 0, checkpoint));
  }
  $("#schedule-list").replaceChildren(...items);
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
  renderSummary(entry, row);
  renderPreseasonStartingPoint(entry, preseasonDistribution || distribution, preseasonArtifact, row);
  renderSeasonMovement(entry, trajectory, row, snapshot.rank_count, distribution);
  renderSeasonOutlook(artifact.season_simulation, artifact.teams[teamId]);
  renderSchedule(artifact, entry, trajectory);
}

load().catch((error) => {
  updateBackLink(null);
  $("#team-page-status").textContent = error.message;
  $("#team-page-status").className = "team-page-status team-page-error";
});
