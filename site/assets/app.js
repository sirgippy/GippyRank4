const pageParams = new URLSearchParams(window.location.search);
const requestedSeason = pageParams.has("season") ? Number(pageParams.get("season")) : null;
const state = {
  manifest: null,
  family: pageParams.get("family") === "performance" ? "performance" : "predictive",
  season: Number.isSafeInteger(requestedSeason) && requestedSeason > 0 ? requestedSeason : null,
  slot: pageParams.get("snapshot") || pageParams.get("slot") || null,
  prior: pageParams.get("prior") === "history" ? "history" : "context",
  depth: 25,
  notice: "",
  snapshot: null,
  selectedTeamId: null,
  renderVersion: 0,
  detailVersion: 0,
  distributionCache: new Map(),
};

const $ = (selector) => document.querySelector(selector);
const detailDialog = $("#team-detail");
const aboutRankings = $("#about-rankings");

if (window.location.hash === "#about-rankings") aboutRankings.open = true;
$(".about-link")?.addEventListener("click", () => { aboutRankings.open = true; });

function element(name, className, text) {
  const node = document.createElement(name);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function teamLogo(teamId, className = "team-logo") {
  const template = state.manifest?.team_logos?.url_template;
  const handle = state.manifest?.team_logos?.handles?.[teamId];
  if (!handle || !template || template.split("{handle}").length !== 2) return null;
  const frame = element("span", "team-logo-frame");
  frame.setAttribute("aria-hidden", "true");
  const image = element("img", className);
  image.src = template.replace("{handle}", encodeURIComponent(handle));
  image.alt = "";
  image.width = 60;
  image.height = 40;
  image.loading = "lazy";
  image.decoding = "async";
  image.addEventListener("error", () => frame.remove(), { once: true });
  frame.append(image);
  return frame;
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function publicationStatusLabel(entry) { return entry.publication_status === "official" ? "Official" : "Interim"; }
function labelFor(entry) { return `${entry.display_label} · ${publicationStatusLabel(entry)}`; }
function choices() { return state.manifest.snapshots.filter((entry) => entry.ranking_family === state.family && entry.season === state.season); }
function selectedEntry() { return choices().find((entry) => entry.publication_slot === state.slot && (state.family === "performance" || entry.prior_family === state.prior)); }
function familyLabel() { return state.family === "performance" ? "Performance" : "Predictive"; }
function formatDate(value) { return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(value)); }
function formatTimestamp(value) { return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", timeZone: "UTC", timeZoneName: "short" }).format(new Date(value)); }
function percentage(value) {
  const valueAsPercent = value * 100;
  if (value === 1) return "100%";
  if (valueAsPercent === 0) return "0%";
  if (valueAsPercent < 0.01) return "<0.01%";
  if (valueAsPercent < 1) return `${valueAsPercent.toFixed(valueAsPercent < 0.1 ? 2 : 1)}%`;
  if (valueAsPercent < 10) return `${valueAsPercent.toFixed(1)}%`;
  if (valueAsPercent >= 99.95) return "<100%";
  if (valueAsPercent >= 95) return `${valueAsPercent.toFixed(1)}%`;
  return `${Math.round(valueAsPercent)}%`;
}

function displayedRatedRows(snapshot) {
  return [...(snapshot?.rankings?.filter((row) => row.rated !== false) ?? [])].sort((left, right) => (
    left.expected_rank - right.expected_rank
    || left.display_rank - right.display_rank
    || String(left.team_id).localeCompare(String(right.team_id))
  ));
}

function ballotRows(snapshot) {
  return displayedRatedRows(snapshot).slice(0, 25);
}

function ballotEligibility(snapshot) {
  if (!snapshot) return { ok: false, message: "Load a published ranking snapshot to export a ballot." };
  const rated = displayedRatedRows(snapshot);
  if (rated.length < 25) {
    return {
      ok: false,
      message: `A ballot requires 25 rated teams. This ${familyLabel()} snapshot currently has only ${rated.length}.`,
    };
  }
  const handles = state.manifest?.redditcfb?.team_handles ?? {};
  const missing = ballotRows(snapshot).filter((row) => !handles[row.team_id]);
  if (missing.length) {
    return {
      ok: false,
      message: `Cannot export: canonical r/CFB handles are unavailable for ${missing.map((row) => row.team_name).join(", ")}.`,
    };
  }
  return { ok: true, message: "Ready to download the selected GippyRank Top 25." };
}

function updateBallotExport(entry, snapshot) {
  const button = $("#download-ballot");
  const status = $("#ballot-export-status");
  if (!button || !status) return;
  const eligibility = ballotEligibility(snapshot);
  button.disabled = !eligibility.ok;
  status.textContent = eligibility.message;
  status.classList.toggle("is-error", !eligibility.ok && Boolean(snapshot));
  button.title = entry ? `Export the ${entry.season} ${entry.display_label} ${familyLabel()} Top 25` : "Export the selected GippyRank Top 25";
}

function selectedViewLabel(entry) {
  const prior = state.family === "predictive" ? ` ${entry.prior_family === "history" ? "History" : "Context"}` : "";
  return `${entry.season} ${entry.display_label} ${familyLabel()}${prior}`;
}

function overallBallotRationale(entry) {
  const semantics = state.family === "predictive"
    ? "Predictive estimates current underlying team quality using preseason information plus games."
    : "Performance asks what quality is implied by games played, using Context estimates to interpret opponent quality. Performance is not standings, strength of record, or postseason deservingness.";
  return `Generated from GippyRank 4.0, a probabilistic college-football ranking model that estimates underlying team quality from game performance and expresses uncertainty rather than treating rank as perfectly known. This ballot uses the ${selectedViewLabel(entry)} rankings. ${semantics}\n\nExplore the rankings and uncertainty at: ${state.manifest.site_url}`;
}

function teamBallotRationale(row) {
  const expectedLabel = state.family === "performance"
    ? "GippyRank Performance-equivalent expected rank"
    : "GippyRank expected rank";
  const probability = Number.isFinite(row.top25_probability)
    ? ` Top-25 probability: ${percentage(row.top25_probability)}.`
    : "";
  return `${expectedLabel}: ${row.expected_rank.toFixed(1)}. Central 80% interval: ${row.interval_80[0]}–${row.interval_80[1]}.${probability}`;
}

function ballotFilename(entry) {
  const label = entry.snapshot_type === "preseason"
    ? "preseason"
    : String(entry.display_label || entry.snapshot_id)
      .toLowerCase()
      .replace(/\./g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  const family = state.family === "predictive" ? `predictive-${entry.prior_family}` : "performance";
  return `gippyrank-${entry.season}-${label || "snapshot"}-${family}.json`;
}

function downloadBallot() {
  const entry = selectedEntry();
  const eligibility = ballotEligibility(state.snapshot);
  if (!entry || !eligibility.ok) {
    updateBallotExport(entry, state.snapshot);
    return;
  }
  const handles = state.manifest.redditcfb.team_handles;
  const ballot = {
    poll_type: "computer",
    overall_rationale: overallBallotRationale(entry),
    entries: ballotRows(state.snapshot).map((row, index) => ({
      rank: index + 1,
      team_handle: handles[row.team_id],
      rationale: teamBallotRationale(row),
    })),
  };
  const blob = new Blob([`${JSON.stringify(ballot, null, 2)}\n`], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = ballotFilename(entry);
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function ordinal(rank) {
  const rounded = Math.round(rank);
  const remainder = rounded % 100;
  const suffix = remainder >= 11 && remainder <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[rounded % 10] ?? "th");
  return `${rounded}${suffix}`;
}

function chooseDefault() {
  const available = choices();
  const preferred = state.manifest.default_publication_slot;
  const requestedSnapshot = available.find((item) => item.snapshot_id === state.slot && (state.family === "performance" || item.prior_family === state.prior));
  const entry = requestedSnapshot
    ?? available.find((item) => item.publication_slot === state.slot && (state.family === "performance" || item.prior_family === state.prior))
    ?? available.find((item) => item.publication_slot === preferred && (state.family === "performance" || item.prior_family === state.prior))
    ?? available.find((item) => item.publication_slot === preferred)
    ?? available.find((item) => state.family === "performance" || item.prior_family === state.prior)
    ?? available[0];
  if (!entry) return null;
  state.slot = entry.publication_slot;
  if (state.family !== "performance") state.prior = entry.prior_family;
  return entry;
}

function changePrior(prior) {
  const current = selectedEntry() ?? chooseDefault();
  const counterpart = current && choices().find((entry) => entry.publication_slot === current.publication_slot && entry.prior_family === prior);
  if (counterpart) {
    state.prior = prior;
    state.slot = counterpart.publication_slot;
    state.notice = "";
  } else if (current) {
    state.notice = `${prior === "history" ? "History" : "Context"} is not published for ${current.display_label}; staying on ${current.prior_family === "history" ? "History" : "Context"}.`;
  }
}

function populate() {
  const families = state.manifest.ranking_families;
  $("#family-select").replaceChildren(...families.map((family) => new Option(family.label, family.id, family.id === state.family, family.id === state.family)));
  const seasons = state.manifest.seasons;
  if (!seasons.includes(state.season)) state.season = seasons[0];
  $("#season-select").replaceChildren(...seasons.map((season) => new Option(season, season, season === state.season, season === state.season)));
  const entry = selectedEntry() ?? chooseDefault();
  const available = choices();
  const snapshotGroups = [...new Map(available.map((entry) => [entry.publication_slot, entry])).values()];
  $("#snapshot-select").replaceChildren(...snapshotGroups.map((item) => new Option(labelFor(item), item.publication_slot, item.publication_slot === state.slot, item.publication_slot === state.slot)));
  $("#snapshot-select").value = entry?.publication_slot ?? "";
  const priorSelector = $("#prior-select");
  const performance = state.family === "performance";
  priorSelector.hidden = performance;
  priorSelector.disabled = performance;
  document.querySelectorAll("[data-prior]").forEach((button) => { button.classList.toggle("is-active", button.dataset.prior === state.prior); button.disabled = performance; });
  const weekLink = $("#week-view-link");
  if (weekLink) weekLink.href = weekPageUrl(entry, entry?.default_week);
  updateBallotExport(entry, state.snapshot);
}

function updateSnapshotSummary(entry, snapshot) {
  const current = entry.snapshot_type !== "preseason";
  $("#snapshot-kind").textContent = `${publicationStatusLabel(entry)} publication`;
  $("#snapshot-title").textContent = `${entry.season} ${entry.display_label} · ${familyLabel()}${state.family === "predictive" ? ` · ${entry.prior_family === "context" ? "Context" : "History"}` : ""}`;
  $("#snapshot-freshness").textContent = current ? `Rankings through ${formatDate(entry.effective_cutoff)}.` : "Frozen before any game evidence.";
  const baseline = entry.comparison_display_label
    ? `Change vs ${entry.comparison_display_label}.`
    : `No earlier official ${familyLabel()} baseline; movement is not shown.`;
  $("#snapshot-comparison").textContent = baseline;
  $("#section-context").textContent = `FBS rankings · sorted by expected rank · ${baseline}`;
  const evidence = current ? `Effective cutoff: ${formatTimestamp(snapshot.effective_cutoff)} · ${snapshot.included_game_count} eligible games included · ${snapshot.excluded_lower_division_games ?? 0} lower-division games excluded.${state.family === "performance" ? ` ${snapshot.rated_count} rated, ${snapshot.unrated_count} NR.` : ""}` : `Snapshot generated ${formatTimestamp(snapshot.generation_timestamp)} · 0 eligible games included.`;
  $("#snapshot-evidence").textContent = state.notice ? `${evidence} ${state.notice}` : evidence;
  updateBallotExport(entry, snapshot);
}

function sharedRankPosition(rank, rankCount) {
  return Math.max(0, Math.min(100, ((rank - 0.5) / rankCount) * 100));
}

function uncertaintyIndicator(row, rankCount) {
  const indicator = element("div", "uncertainty-indicator");
  const numeric = element("span", "interval-numeric", `${row.interval_80[0]}–${row.interval_80[1]}`);
  const whisker = element("span", "rank-whisker");
  whisker.setAttribute("aria-hidden", "true");
  const range = element("span", "rank-whisker-range");
  const marker = element("span", "rank-whisker-marker");
  const low = sharedRankPosition(row.interval_80[0], rankCount);
  const high = sharedRankPosition(row.interval_80[1], rankCount);
  range.style.left = `${low}%`;
  range.style.width = `${Math.max(high - low, 100 / rankCount)}%`;
  marker.style.left = `${sharedRankPosition(row.expected_rank, rankCount)}%`;
  whisker.append(range, marker);
  indicator.append(numeric, whisker, element("span", "sr-only", `80% interval ranks ${row.interval_80[0]} through ${row.interval_80[1]}; expected rank ${row.expected_rank.toFixed(1)} on a shared scale from 1 through ${rankCount}.`));
  return indicator;
}

function teamPageUrl(row, entry) {
  const url = new URL("./team.html", document.baseURI);
  url.searchParams.set("team", row.team_id);
  url.searchParams.set("season", String(entry.season));
  url.searchParams.set("snapshot", entry.snapshot_id);
  url.searchParams.set("family", state.family);
  if (state.family === "predictive") url.searchParams.set("prior", entry.prior_family);
  return `${url.pathname}${url.search}`;
}

function weekPageUrl(entry, week = null) {
  const url = new URL("./week.html", document.baseURI);
  if (!entry) return url.pathname;
  url.searchParams.set("season", String(entry.season));
  url.searchParams.set("snapshot", entry.snapshot_id);
  url.searchParams.set("family", state.family);
  if (state.family === "predictive") url.searchParams.set("prior", entry.prior_family);
  if (week !== null && week !== undefined) url.searchParams.set("week", String(week));
  return `${url.pathname}${url.search}`;
}

function teamCell(row, entry) {
  const wrapper = element("div", "team-cell-content");
  const link = element("a", "team-link", row.team_name);
  link.href = teamPageUrl(row, entry);
  link.setAttribute("aria-label", `Open ${row.team_name} season page`);
  const text = element("span", "team-cell-text");
  text.append(link, element("span", "team-conference", row.conference || "Independent"));
  if (row.rated === false) text.append(element("span", "team-status", "No eligible games played"));
  const logo = teamLogo(row.team_id);
  if (logo) wrapper.append(logo);
  wrapper.append(text);
  return wrapper;
}

function rankCell(row) {
  const cell = element("td", "rank");
  cell.append(element("span", "rank-value", row.display_rank));
  const inline = element("span", "rank-change-inline", row.rank_change_display ?? "—");
  inline.setAttribute("aria-hidden", "true");
  cell.append(inline, element("span", "sr-only", row.rank_change_accessible ?? "No earlier official ranking baseline"));
  return cell;
}

function rankChangeCell(row) {
  const cell = element("td", "change-column");
  const visual = element("span", "rank-change-visual", row.rank_change_display ?? "—");
  visual.setAttribute("aria-hidden", "true");
  cell.append(visual);
  return cell;
}

function detailsButton(row) {
  const button = element("button", "details-button", "View");
  button.type = "button";
  button.dataset.teamId = row.team_id;
  button.setAttribute("aria-label", `View ${row.team_name} rank uncertainty`);
  return button;
}

function renderRankings(snapshot) {
  const entry = selectedEntry();
  const rated = displayedRatedRows(snapshot);
  const rankings = state.depth === "all" ? snapshot.rankings : rated.slice(0, 25);
  $("#rankings-title").textContent = state.depth === "all" ? "All FBS rankings" : "Top 25";
  const rows = rankings.map((row) => {
    const tr = document.createElement("tr");
    const team = element("td", "team");
    team.append(teamCell(row, entry));
    const uncertainty = element("td", "uncertainty-column");
    uncertainty.append(uncertaintyIndicator(row, snapshot.rank_count));
    const details = element("td", "details-cell");
    details.append(detailsButton(row));
    tr.append(
      rankCell(row), rankChangeCell(row), team, element("td", "record-column", row.record),
      element("td", "expected-rank", row.expected_rank.toFixed(1)), element("td", "median-column", row.median_rank),
      element("td", "interval-column", `${row.interval_80[0]}–${row.interval_80[1]}`), uncertainty,
      element("td", "top25-column", percentage(row.top25_probability)), details,
    );
    return tr;
  });
  $("#rankings-body").replaceChildren(...rows);
}

function summaryText(row, summary) {
  const lead = state.family === "performance"
    ? `Based on the games ${row.team_name} has played, its expected Performance-equivalent rank is ${summary.expected_rank.toFixed(1)}.`
    : `Expected rank: ${summary.expected_rank.toFixed(1)}.`;
  return `${lead} Median: ${ordinal(summary.median_rank)}. Most likely rank: ${ordinal(summary.modal_rank)}. The central 50% interval spans ${ordinal(summary.interval_50[0])}–${ordinal(summary.interval_50[1])}. The central 80% interval spans ${ordinal(summary.interval_80[0])}–${ordinal(summary.interval_80[1])}. The central 95% interval spans ${ordinal(summary.interval_95[0])}–${ordinal(summary.interval_95[1])}. The model gives ${row.team_name} a ${percentage(summary.top10_probability)} chance of landing in the Top 10 and a ${percentage(summary.top25_probability)} chance of landing in the Top 25.`;
}

function rankX(rank, rankCount, left, width) { return left + ((rank - 0.5) / rankCount) * width; }

function makeDistributionChart(row, pmf, summary, rankCount) {
  const width = 700;
  const height = 300;
  const left = 48;
  const right = 18;
  const top = 24;
  const bottom = 48;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const baseline = height - bottom;
  const yMaximum = Math.max(0.01, Math.ceil(Math.max(...pmf) * 200) / 200);
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, class: "distribution-chart", role: "img", "aria-labelledby": "distribution-chart-title distribution-chart-description" });
  const title = svgElement("title", { id: "distribution-chart-title" });
  const description = svgElement("desc", { id: "distribution-chart-description" });
  title.textContent = `${row.team_name} posterior rank distribution`;
  description.textContent = `Discrete probabilities for every rank from 1 through ${rankCount}. ${summaryText(row, summary)}`;
  svg.append(title, description, svgElement("line", { x1: left, y1: top, x2: width - right, y2: top, class: "chart-grid" }), svgElement("line", { x1: left, y1: baseline, x2: width - right, y2: baseline, class: "chart-axis" }));
  const yTop = svgElement("text", { x: left - 8, y: top + 4, "text-anchor": "end", class: "chart-label" });
  yTop.textContent = percentage(yMaximum);
  const yBottom = svgElement("text", { x: left - 8, y: baseline + 4, "text-anchor": "end", class: "chart-label" });
  yBottom.textContent = "0%";
  const yTitle = svgElement("text", { x: 13, y: top + plotHeight / 2, transform: `rotate(-90 13 ${top + plotHeight / 2})`, "text-anchor": "middle", class: "chart-axis-title" });
  yTitle.textContent = "Posterior probability";
  svg.append(yTop, yBottom, yTitle);
  const barWidth = Math.max(1, (plotWidth / rankCount) * 0.82);
  pmf.forEach((probability, index) => {
    const x = rankX(index + 1, rankCount, left, plotWidth) - barWidth / 2;
    const barHeight = (probability / yMaximum) * plotHeight;
    svg.append(svgElement("rect", { x, y: baseline - barHeight, width: barWidth, height: barHeight, class: "distribution-bar" }));
  });
  [[summary.interval_95, "interval-95", baseline - 5], [summary.interval_80, "interval-80", baseline - 13], [summary.interval_50, "interval-50", baseline - 21]].forEach(([interval, className, y]) => svg.append(svgElement("line", { x1: rankX(interval[0], rankCount, left, plotWidth), y1: y, x2: rankX(interval[1], rankCount, left, plotWidth), y2: y, class: className })));
  const expectedX = rankX(summary.expected_rank, rankCount, left, plotWidth);
  const medianX = rankX(summary.median_rank, rankCount, left, plotWidth);
  const modeX = rankX(summary.modal_rank, rankCount, left, plotWidth);
  const modeY = baseline - (pmf[summary.modal_rank - 1] / yMaximum) * plotHeight;
  svg.append(svgElement("line", { x1: expectedX, y1: top, x2: expectedX, y2: baseline, class: "expected-marker" }), svgElement("line", { x1: medianX, y1: top, x2: medianX, y2: baseline, class: "median-marker" }), svgElement("circle", { cx: modeX, cy: modeY, r: 4, class: "mode-marker" }));
  const ticks = [...new Set([1, Math.round(rankCount * 0.25), Math.round(rankCount * 0.5), Math.round(rankCount * 0.75), rankCount])].sort((a, b) => a - b);
  ticks.forEach((rank) => {
    const x = rankX(rank, rankCount, left, plotWidth);
    const label = svgElement("text", { x, y: baseline + 17, "text-anchor": "middle", class: "chart-label" });
    label.textContent = rank;
    svg.append(svgElement("line", { x1: x, y1: baseline, x2: x, y2: baseline + 4, class: "chart-axis" }), label);
  });
  const xTitle = svgElement("text", { x: left + plotWidth / 2, y: height - 6, "text-anchor": "middle", class: "chart-axis-title" });
  xTitle.textContent = "Possible final / latent rank (1 is best)";
  svg.append(xTitle);
  return svg;
}

function intervalList(summary) {
  const list = element("dl", "interval-list");
  [["50% interval", summary.interval_50, summary.interval_widths["50"]], ["80% interval", summary.interval_80, summary.interval_widths["80"]], ["95% interval", summary.interval_95, summary.interval_widths["95"]]].forEach(([label, interval, width]) => list.append(element("dt", "", label), element("dd", "", `${ordinal(interval[0])}–${ordinal(interval[1])} · ${width} ranks`)));
  return list;
}

function probabilityList(summary) {
  const list = element("dl", "probability-list");
  [["National #1", summary.rank_1_probability], ["Top 5", summary.top5_probability], ["Top 10", summary.top10_probability], ["Top 25", summary.top25_probability]].forEach(([label, value]) => list.append(element("dt", "", label), element("dd", "", percentage(value))));
  return list;
}

function renderDetail(row, entry, distribution) {
  const team = distribution.teams[row.team_id];
  if (!team || !Array.isArray(team.pmf) || team.pmf.length !== distribution.rank_count) throw new Error("The selected team's distribution is unavailable.");
  const summary = team.summary;
  const rankLabel = row.rated === false ? "NR (not rated)" : `#${row.display_rank}`;
  const logo = teamLogo(row.team_id, "team-logo team-logo-selector");
  $("#detail-team-logo").replaceChildren(...(logo ? [logo] : []));
  $("#detail-team-name").textContent = row.team_name;
  $("#detail-team-meta").textContent = `${rankLabel} by expected rank · ${row.conference || "Independent"} · ${row.record} modeled record · ${familyLabel()}${state.family === "predictive" ? ` · ${entry.prior_family === "context" ? "Context" : "History"}` : ""}`;
  const heading = element("h3", "detail-section-title", "Rank uncertainty");
  const explainer = element("p", "detail-explainer", state.family === "performance"
    ? `Performance uses only ${row.team_name}'s eligible games. Predictive Context estimates still help interpret opponent quality, but ${row.team_name}'s own preseason prior does not directly contribute. Performance is not a predicted final rank, résumé, standings, or postseason deservingness.`
    : `GippyRank does not assign ${row.team_name} one certain rank. It maintains a probability distribution over possible ranks. The table orders teams by expected rank; this distribution shows how uncertain that estimate is.`);
  const currentBelief = element("p", "detail-current-belief", state.family === "performance" && row.rated === false
    ? "No eligible games have been played by this team in the selected snapshot. It is shown as NR with a neutral uniform distribution."
    : entry.snapshot_type === "preseason" ? "This preseason distribution is the model's belief before any eligible game evidence." : `This is the model's current belief after all eligible games through ${formatDate(entry.effective_cutoff)}. It is not a prediction of AP, Coaches, or CFP voters.`);
  const legend = element("ul", "chart-legend");
  [["distribution", "Discrete PMF"], ["expected", `Expected ${summary.expected_rank.toFixed(1)}`], ["median", `Median ${ordinal(summary.median_rank)}`], ["mode", `Mode ${ordinal(summary.modal_rank)}`], ["interval", "50% / 80% / 95% intervals"]].forEach(([kind, text]) => { const item = element("li", `legend-${kind}`); item.append(element("span", "legend-swatch"), document.createTextNode(text)); legend.append(item); });
  const chartPanel = element("div", "chart-panel");
  chartPanel.append(makeDistributionChart(row, team.pmf, summary, distribution.rank_count), legend, element("p", "chart-text-alternative", summaryText(row, summary)));
  const summaryPanel = element("aside", "uncertainty-summary");
  summaryPanel.setAttribute("aria-label", "Distribution summaries");
  summaryPanel.append(element("h4", "", "Rank summaries"), intervalList(summary), element("h4", "", "Probability of finishing"), probabilityList(summary));
  const layout = element("div", "detail-layout");
  layout.append(chartPanel, summaryPanel);
  $("#detail-content").replaceChildren(heading, explainer, currentBelief, layout);
}

async function distributionFor(entry) {
  if (!state.distributionCache.has(entry.snapshot_id)) state.distributionCache.set(entry.snapshot_id, fetch(`./${entry.distribution_path}`).then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load rank distribution data."))));
  try { return await state.distributionCache.get(entry.snapshot_id); }
  catch (error) { state.distributionCache.delete(entry.snapshot_id); throw error; }
}

function showDetail() { if (!detailDialog.open) detailDialog.showModal(); }
function closeDetail() { state.selectedTeamId = null; state.detailVersion += 1; if (detailDialog.open) detailDialog.close(); }

async function loadDetail(row, entry, snapshot) {
  const version = ++state.detailVersion;
  showDetail();
  const logo = teamLogo(row.team_id, "team-logo team-logo-selector");
  $("#detail-team-logo").replaceChildren(...(logo ? [logo] : []));
  $("#detail-team-name").textContent = row.team_name;
  $("#detail-team-meta").textContent = "Loading the selected publication's rank distribution…";
  $("#detail-content").replaceChildren(element("p", "detail-loading", "Loading rank uncertainty…"));
  try {
    const distribution = await distributionFor(entry);
    if (version !== state.detailVersion || state.selectedTeamId !== row.team_id || selectedEntry()?.snapshot_id !== entry.snapshot_id) return;
    if (distribution.snapshot_id !== entry.snapshot_id || distribution.rank_count !== snapshot.rank_count) throw new Error("Distribution data does not match the selected ranking snapshot.");
    renderDetail(row, entry, distribution);
  } catch (error) {
    if (version === state.detailVersion) $("#detail-content").replaceChildren(element("p", "detail-error", error.message));
  }
}

function prepareSnapshotChange() {
  state.snapshot = null;
  state.renderVersion += 1;
  if (state.selectedTeamId && detailDialog.open) {
    state.detailVersion += 1;
    $("#detail-team-meta").textContent = "Updating for the selected publication…";
    $("#detail-content").replaceChildren(element("p", "detail-loading", "Updating rank uncertainty…"));
  }
}

async function render() {
  populate();
  const entry = selectedEntry();
  if (!entry) return;
  const version = ++state.renderVersion;
  try {
    const snapshot = await fetch(`./${entry.data_path}`).then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load ranking data.")));
    if (version !== state.renderVersion || selectedEntry()?.snapshot_id !== entry.snapshot_id) return;
    state.snapshot = snapshot;
    updateSnapshotSummary(entry, snapshot);
    renderRankings(snapshot);
    if (state.selectedTeamId) {
      const selectedRow = snapshot.rankings.find((row) => row.team_id === state.selectedTeamId);
      if (selectedRow) void loadDetail(selectedRow, entry, snapshot);
      else closeDetail();
    }
  } catch (error) {
    if (version === state.renderVersion) $("#snapshot-title").textContent = error.message;
  }
}

$("#rankings-body").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-team-id]");
  if (!button || !state.snapshot) return;
  const row = state.snapshot.rankings.find((ranking) => ranking.team_id === button.dataset.teamId);
  const entry = selectedEntry();
  if (!row || !entry) return;
  state.selectedTeamId = row.team_id;
  void loadDetail(row, entry, state.snapshot);
});
$("#detail-close").addEventListener("click", closeDetail);
detailDialog.addEventListener("close", () => { state.selectedTeamId = null; });
$("#family-select").addEventListener("change", (event) => {
  const previousSlot = state.slot;
  state.family = event.target.value;
  state.notice = "";
  if (previousSlot && !choices().some((entry) => entry.publication_slot === previousSlot)) {
    state.slot = null;
    if (state.family === "performance") state.notice = "Performance begins after eligible games have been played; showing the nearest available Performance snapshot.";
  }
  prepareSnapshotChange();
  void render();
});
$("#season-select").addEventListener("change", (event) => { state.season = Number(event.target.value); state.slot = null; state.notice = ""; prepareSnapshotChange(); void render(); });
$("#snapshot-select").addEventListener("change", (event) => { state.slot = event.target.value; state.notice = ""; prepareSnapshotChange(); void render(); });
document.querySelectorAll("[data-prior]").forEach((button) => button.addEventListener("click", () => { changePrior(button.dataset.prior); prepareSnapshotChange(); void render(); }));
document.querySelectorAll("[data-depth]").forEach((button) => button.addEventListener("click", () => { state.depth = button.dataset.depth === "all" ? "all" : 25; document.querySelectorAll("[data-depth]").forEach((item) => item.classList.toggle("is-active", item === button)); if (state.snapshot) renderRankings(state.snapshot); }));
$("#download-ballot")?.addEventListener("click", downloadBallot);

fetch("./data/manifest.json").then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load manifest."))).then((manifest) => { state.manifest = manifest; return render(); }).catch((error) => { $("#snapshot-title").textContent = error.message; });
