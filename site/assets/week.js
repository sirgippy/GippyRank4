const $ = (selector) => document.querySelector(selector);
const params = new URLSearchParams(window.location.search);
const state = {
  manifest: null,
  family: params.get("family") === "performance" ? "performance" : "predictive",
  season: Number(params.get("season")) || null,
  slot: params.get("snapshot") || params.get("slot") || null,
  prior: params.get("prior") === "history" ? "history" : "context",
  view: params.get("view") === "marquee" ? "marquee" : "all",
  weekKey: params.get("week") || null,
};

const priorExplanations = {
  context: {
    short: "Program history plus information about this year's team.",
    detail: "Every Predictive ranking needs a starting estimate before games are played. As results arrive, GippyRank updates that starting distribution with game evidence. Context uses program history plus information about this year's team, including recruiting, roster talent, returning production, and coach tenure.",
  },
  history: {
    short: "Previous program performance across recent and longer historical windows.",
    detail: "Every Predictive ranking needs a starting estimate before games are played. As results arrive, GippyRank updates that starting distribution with game evidence. History uses previous program performance across recent and longer historical windows only. It intentionally ignores current roster talent, recruiting, returning production, and coach tenure.",
  },
};

function node(name, className, text) {
  const value = document.createElement(name);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function svgElement(name, attributes = {}) {
  const value = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, attribute]) => value.setAttribute(key, String(attribute)));
  return value;
}

function percentage(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "Unavailable";
  if (numeric === 1) return "100%";
  if (numeric === 0) return "0%";
  const percent = numeric * 100;
  if (percent < 0.01) return "<0.01%";
  if (percent < 1) return `${percent.toFixed(1)}%`;
  if (percent >= 99.95) return "<100%";
  return `${Math.round(percent)}%`;
}

function ordinal(rank) {
  const rounded = Math.round(Number(rank));
  const remainder = rounded % 100;
  const suffix = remainder >= 11 && remainder <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[rounded % 10] ?? "th");
  return `${rounded}${suffix}`;
}

function formatDate(value, includeYear = false) {
  if (!value) return "Date unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", ...(includeYear ? { year: "numeric" } : {}),
  }).format(date);
}

function formatTime(value) {
  if (!value) return "Kickoff time unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Kickoff time unavailable";
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric", minute: "2-digit", timeZoneName: "short",
  }).format(date);
}

function localDateKey(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

function publicationStatusLabel(entry) {
  return entry.publication_status === "official" ? "Official" : "Interim";
}

function labelFor(entry) {
  return `${entry.display_label} · ${publicationStatusLabel(entry)}`;
}

function choices() {
  return state.manifest.snapshots.filter((entry) => entry.season === state.season && entry.ranking_family === state.family);
}

function matchesContext(entry) {
  return state.family === "performance" || entry.prior_family === state.prior;
}

function selectedEntry() {
  return choices().find((entry) => matchesContext(entry) && (entry.snapshot_id === state.slot || entry.publication_slot === state.slot));
}

function chooseDefault() {
  const available = choices();
  const preferred = state.manifest.default_publication_slot;
  const entry = available.find((item) => matchesContext(item) && item.snapshot_id === state.slot)
    ?? available.find((item) => matchesContext(item) && item.publication_slot === state.slot)
    ?? available.find((item) => matchesContext(item) && item.publication_slot === preferred)
    ?? available.find((item) => matchesContext(item))
    ?? available[0];
  if (!entry) return null;
  state.slot = entry.publication_slot;
  if (state.family !== "performance") state.prior = entry.prior_family;
  return entry;
}

function contextParams(entry, week = null) {
  const next = new URLSearchParams();
  next.set("season", String(entry.season));
  next.set("snapshot", entry.snapshot_id);
  next.set("family", state.family);
  if (state.family === "predictive") next.set("prior", entry.prior_family);
  if (week !== null && week !== undefined) next.set("week", String(week));
  if (state.view === "marquee") next.set("view", "marquee");
  return next;
}

function navigateTo(next) {
  window.location.href = `${window.location.pathname}?${next.toString()}`;
}

function teamPageUrl(team, entry, week) {
  const next = contextParams(entry, week);
  next.set("team", team.team_id);
  const url = new URL("./team.html", document.baseURI);
  url.search = next.toString();
  return `${url.pathname}${url.search}`;
}

function teamLogo(teamId, className = "team-logo team-logo-week") {
  const handle = state.manifest?.team_logos?.handles?.[teamId];
  const template = state.manifest?.team_logos?.url_template;
  if (!handle || !template || template.split("{handle}").length !== 2) return null;
  const frame = node("span", "team-logo-frame");
  frame.setAttribute("aria-hidden", "true");
  const image = node("img", className);
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

function displayScale(axis) {
  const scale = Number(axis?.probability_encoding?.scale);
  return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function densitySvg(masses, { className, label, description, zero = false, tail = false }) {
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
  svg.append(title, desc, svgElement("line", { x1: left, y1: baseline, x2: right, y2: baseline, class: "distribution-baseline" }));
  if (zero) {
    const zeroX = left + plotWidth / 2;
    svg.append(svgElement("line", { x1: zeroX, y1: 6, x2: zeroX, y2: baseline + 3, class: "distribution-zero" }));
  }
  const barWidth = plotWidth / Math.max(values.length, 1);
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
  if (tail) svg.append(svgElement("path", { d: `M ${left - 2} 12 l 5 -5 l 5 5 M ${right + 2} 12 l -5 -5 l -5 5`, class: "distribution-tail" }));
  return svg;
}

function performancePanel(rating, display, axis, teamName) {
  if (!rating) return node("p", "game-not-modeled", `${teamName}: Performance rating unavailable for this game.`);
  const panel = node("div", "weekly-performance-panel");
  const grade = rating.performance_grade ? `${rating.performance_grade} · ` : "";
  panel.append(node("strong", "game-rating-title", `${teamName}: ${grade}${ordinal(rating.performance_percentile)} percentile`));
  if (Array.isArray(display) && axis) {
    const label = `${teamName} performance distribution from rank 1 through rank ${axis.max_rank}; rank 1 is best.`;
    const description = `The central 80 percent interval is ranks ${rating.interval_80[0]} through ${rating.interval_80[1]}.`;
    const figure = node("figure", "game-distribution game-distribution-performance");
    figure.append(densitySvg(display, { className: "performance-distribution-chart", label, description }));
    const caption = node("figcaption", "distribution-axis");
    caption.append(node("span", "axis-start", "#1 best"), node("span", "axis-label", "Performance"), node("span", "axis-end", `#${axis.max_rank} worst`));
    figure.append(caption);
    panel.append(figure);
  }
  panel.append(node("p", "sr-only", `${teamName} performance grade ${rating.performance_grade || "unavailable"}; ${ordinal(rating.performance_percentile)} percentile. Central 80 percent interval: ranks ${rating.interval_80[0]} through ${rating.interval_80[1]}.`));
  const details = node("details", "game-rating-details-disclosure");
  details.append(node("summary", "", "More performance detail"));
  const list = node("dl", "game-rating-details");
  [["Expected performance rank", `#${Number(rating.expected_rank).toFixed(1)}`], ["Median", `#${rating.median_rank}`], ["Mode", `#${rating.mode_rank}`], ["50% interval", `${rating.interval_50[0]}–${rating.interval_50[1]}`], ["80% interval", `${rating.interval_80[0]}–${rating.interval_80[1]}`], ["95% interval", `${rating.interval_95[0]}–${rating.interval_95[1]}`], ["Top 25 caliber", percentage(rating.top25_probability)]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  details.append(list);
  panel.append(details);
  return panel;
}

function marginSide(teamName, value) {
  const amount = Math.abs(Number(value));
  return amount < 0.05 ? "Even" : `${teamName} by ${amount.toFixed(1)}`;
}

function predictionRange(prediction, interval) {
  const [low, high] = interval;
  if (high < 0) return `${marginSide(prediction.away_team_name, -high)} to ${marginSide(prediction.away_team_name, -low)}`;
  if (low > 0) return `${marginSide(prediction.home_team_name, low)} to ${marginSide(prediction.home_team_name, high)}`;
  return `${low < 0 ? marginSide(prediction.away_team_name, -low) : "Even"} to ${high > 0 ? marginSide(prediction.home_team_name, high) : "Even"}`;
}

function predictionPanel(prediction, axis) {
  if (!prediction) return node("p", "game-not-modeled", "No prediction available for this matchup.");
  const panel = node("div", "weekly-prediction-panel");
  const favorite = prediction.home_win_probability >= prediction.away_win_probability
    ? [prediction.home_team_name, prediction.home_win_probability]
    : [prediction.away_team_name, prediction.away_win_probability];
  const expectedTeam = prediction.expected_home_margin >= 0 ? prediction.home_team_name : prediction.away_team_name;
  const expectedMargin = Math.abs(Number(prediction.expected_home_margin));
  panel.append(node("strong", "game-prediction-title", `${favorite[0]} ${percentage(favorite[1])} · ${expectedMargin < 0.05 ? "expected Even" : `expected ${expectedTeam} by ${expectedMargin.toFixed(1)}`}`));
  const display = prediction.display_distribution;
  if (display && Array.isArray(display.masses) && axis) {
    const tail = (Number(display.lower_tail_probability) + Number(display.upper_tail_probability)) / displayScale(axis);
    const label = `Predictive margin distribution from ${prediction.away_team_name} by ${Math.abs(axis.min_margin)} to ${prediction.home_team_name} by ${axis.max_margin}; even is centered.`;
    const description = `${percentage(prediction.home_win_probability)} home win probability and ${percentage(prediction.away_win_probability)} away win probability. Expected margin: ${expectedTeam} by ${expectedMargin.toFixed(1)}. ${percentage(tail)} of mass is outside the visible range.`;
    const figure = node("figure", "game-distribution game-distribution-future");
    figure.append(densitySvg(display.masses, { className: "future-distribution-chart", label, description, zero: true, tail: tail > 0.001 }));
    const caption = node("figcaption", "distribution-axis");
    caption.append(node("span", "axis-start", `${prediction.away_team_name} by ${Math.abs(axis.min_margin)}`), node("span", "axis-label", "Even"), node("span", "axis-end", `${prediction.home_team_name} by ${axis.max_margin}`));
    figure.append(caption);
    if (tail > 0.001) figure.append(node("p", "distribution-tail-note", `${percentage(tail)} of predictive mass is beyond the visible range.`));
    panel.append(figure);
  }
  panel.append(node("p", "sr-only", `${prediction.home_team_name} has a ${percentage(prediction.home_win_probability)} win probability and ${prediction.away_team_name} has a ${percentage(prediction.away_win_probability)} win probability. Expected margin is ${expectedTeam} by ${expectedMargin.toFixed(1)}. Central 80 percent range: ${predictionRange(prediction, prediction.margin_interval_80)}.`));
  const details = node("details", "game-prediction-details");
  details.append(node("summary", "", "More predictive detail"));
  const list = node("dl", "game-rating-details");
  [["Home win probability", percentage(prediction.home_win_probability)], ["Away win probability", percentage(prediction.away_win_probability)], ["Expected margin", expectedTeam === prediction.home_team_name ? marginSide(expectedTeam, prediction.expected_home_margin) : marginSide(expectedTeam, -prediction.expected_home_margin)], ["Median margin", prediction.median_home_margin >= 0 ? marginSide(prediction.home_team_name, prediction.median_home_margin) : marginSide(prediction.away_team_name, -prediction.median_home_margin)], ["Central 50% range", predictionRange(prediction, prediction.margin_interval_50)], ["Central 80% range", predictionRange(prediction, prediction.margin_interval_80)], ["Central 95% range", predictionRange(prediction, prediction.margin_interval_95)], ["Prediction source", prediction.prediction_source === "predictive_history" ? "Predictive History" : "Predictive Context"]].forEach(([label, value]) => list.append(node("dt", "", label), node("dd", "", value)));
  details.append(list);
  panel.append(details);
  return panel;
}

function teamLink(team, entry, week, rank, winner = false) {
  const rankLabel = rank?.rated ? `, GippyRank number ${rank.display_rank}` : rank ? ", GippyRank not rated" : "";
  if (team.subdivision === "fbs") {
    const link = node("a", "weekly-team-link", team.team_name);
    link.href = teamPageUrl(team, entry, week);
    link.setAttribute("aria-label", `Open ${team.team_name} season page${rankLabel}${winner ? "; winner" : ""}`);
    return link;
  } else {
    const label = node("span", "weekly-team-link", team.team_name);
    label.setAttribute("aria-label", `${team.team_name}, ${team.subdivision.toUpperCase()} opponent${winner ? "; winner" : ""}`);
    return label;
  }
}

function teamRow(team, score, artifact, entry, week, winner = false) {
  const row = node("div", `weekly-team-row${winner ? " is-winner" : ""}`);
  const identity = node("div", "weekly-team-identity");
  const logo = teamLogo(team.team_id);
  if (logo) identity.append(logo);
  const rank = team.subdivision === "fbs" ? artifact.team_rankings?.[team.team_id] : null;
  if (rank) identity.append(node("span", `weekly-team-rank${rank.rated ? "" : " is-unrated"}`, rank.rated ? `#${rank.display_rank}` : "NR"));
  identity.append(teamLink(team, entry, week, rank, winner));
  if (team.conference) identity.append(node("span", "team-conference", team.conference));
  if (winner) identity.append(node("span", "weekly-winner-label", "Winner"));
  row.append(identity);
  if (score !== null && score !== undefined) row.append(node("strong", "weekly-score", String(score)));
  return row;
}

function stateLabel(game) {
  return {
    completed: "Final",
    future: "Upcoming",
    cancelled: "Canceled / postponed",
    unresolved: "Result unavailable",
    out_of_scope: "Result unavailable",
  }[game.state] || "Status unavailable";
}

function gameCard(game, artifact, entry, week) {
  const article = node("article", `weekly-game-card weekly-game-${game.state}`);
  const header = node("header", "weekly-game-header");
  const date = node("div", "weekly-game-date");
  date.append(node("strong", "", formatDate(game.date)), node("span", "", formatTime(game.date)));
  header.append(date, node("span", "weekly-game-state", stateLabel(game)));
  const matchup = node("div", "weekly-matchup");
  const homeScore = game.score ? game.score.home : null;
  const awayScore = game.score ? game.score.away : null;
  const firstTeam = game.neutral_site ? game.home_team : game.away_team;
  const secondTeam = game.neutral_site ? game.away_team : game.home_team;
  const firstScore = game.neutral_site ? homeScore : awayScore;
  const secondScore = game.neutral_site ? awayScore : homeScore;
  const winnerId = game.state === "completed" ? game.winner_team_id : null;
  matchup.append(
    teamRow(firstTeam, firstScore, artifact, entry, week, firstTeam.team_id === winnerId),
    node("span", "weekly-at", game.neutral_site ? "vs." : "at"),
    teamRow(secondTeam, secondScore, artifact, entry, week, secondTeam.team_id === winnerId),
  );
  const context = node("p", "weekly-game-context", `${game.neutral_site ? "Neutral site · " : ""}${game.conference_game ? "Conference" : "Non-conference"}`);
  const body = node("div", "weekly-game-body");
  if (game.state === "completed") {
    if (!game.home_performance && !game.away_performance) {
      body.append(node("p", "game-not-modeled", "Performance rating unavailable for this game."));
    } else {
      const performances = node("div", "weekly-performance-grid");
      const homeDisplay = artifact.performance_displays?.[game.home_performance_ref];
      const awayDisplay = artifact.performance_displays?.[game.away_performance_ref];
      performances.append(performancePanel(game.home_performance, homeDisplay, artifact.performance_axis, game.home_team.team_name), performancePanel(game.away_performance, awayDisplay, artifact.performance_axis, game.away_team.team_name));
      body.append(performances);
    }
  } else if (game.state === "future") {
    body.append(predictionPanel(artifact.future_predictions?.[game.future_prediction_id], artifact.future_margin_axis));
  } else if (game.state === "cancelled") {
    // The concise state label is sufficient; no prediction is shown for this game.
  } else if (game.state === "unresolved" || game.state === "out_of_scope") {
    body.append(node("p", "game-not-modeled", "Result unavailable in this snapshot."));
  }
  article.append(header, matchup, context);
  if (body.childNodes.length) article.append(body);
  return article;
}

function updateBackLink(entry, week) {
  const link = $("#back-to-rankings");
  if (!entry) {
    const rankingsPath = new URL("./", document.baseURI).pathname;
    link.href = rankingsPath;
    $("#rankings-view-link").href = rankingsPath;
    $("#schedule-view-link").href = new URL("./schedule.html", document.baseURI).pathname;
    return;
  }
  const url = new URL("./", document.baseURI);
  const context = contextParams(entry, week).toString();
  url.search = contextParams(entry).toString();
  const rankingsPath = `${url.pathname}${url.search}`;
  link.href = rankingsPath;
  $("#rankings-view-link").href = rankingsPath;
  const scheduleUrl = new URL("./schedule.html", document.baseURI);
  scheduleUrl.search = context;
  $("#schedule-view-link").href = `${scheduleUrl.pathname}${scheduleUrl.search}`;
}

function populateControls(entry) {
  const families = state.manifest.ranking_families;
  $("#family-select").replaceChildren(...families.map((family) => new Option(family.label, family.id, family.id === state.family, family.id === state.family)));
  const seasons = state.manifest.seasons;
  if (!seasons.includes(state.season)) state.season = seasons[0];
  $("#season-select").replaceChildren(...seasons.map((season) => new Option(season, season, season === state.season, season === state.season)));
  const available = choices();
  const groups = [...new Map(available.map((item) => [item.publication_slot, item])).values()];
  $("#snapshot-select").replaceChildren(...groups.map((item) => new Option(labelFor(item), item.publication_slot, item.publication_slot === state.slot, item.publication_slot === state.slot)));
  $("#snapshot-select").value = entry?.publication_slot || "";
  const performance = state.family === "performance";
  $("#prior-select").hidden = performance;
  $("#prior-select").disabled = performance;
  document.querySelectorAll("[data-prior]").forEach((button) => { button.disabled = performance; button.classList.toggle("is-active", button.dataset.prior === state.prior); });
  const priorCopy = priorExplanations[state.prior] ?? priorExplanations.context;
  $("#prior-explanation").textContent = priorCopy.short;
  $("#prior-help-text").textContent = priorCopy.detail;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const selected = button.dataset.view === state.view;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function renderWeekPicker(artifact, entry) {
  const weeks = artifact.weeks || [];
  if (!weeks.length) {
    $("#week-select").replaceChildren(new Option("No weeks available", ""));
    return null;
  }
  const requested = weeks.find((week) => week.key === state.weekKey);
  const defaultWeek = weeks.find((week) => week.key === String(entry?.default_week));
  const selected = requested || defaultWeek || weeks[0];
  state.weekKey = selected.key;
  $("#week-select").replaceChildren(...weeks.map((week) => new Option(week.label, week.key, week.key === selected.key, week.key === selected.key)));
  $("#week-select").value = selected.key;
  $("#previous-week").disabled = weeks.indexOf(selected) <= 0;
  $("#next-week").disabled = weeks.indexOf(selected) >= weeks.length - 1;
  return selected;
}

function clearScheduleStatus() {
  const status = $("#schedule-status");
  status.textContent = "";
  status.className = "team-page-status sr-only";
}

function renderSchedule(week, artifact, entry) {
  const groups = [];
  const games = state.view === "marquee" ? (week.games || []).filter((game) => game.marquee) : (week.games || []);
  for (const game of games) {
    const key = game.date ? localDateKey(game.date) : "unknown";
    let group = groups.find((item) => item.key === key);
    if (!group) {
      group = { key, date: game.date, games: [] };
      groups.push(group);
    }
    group.games.push(game);
  }
  const sections = groups.map((group) => {
    const section = node("section", "weekly-date-group");
    section.append(node("h3", "weekly-date-heading", group.key === "unknown" ? "Date unavailable" : formatDate(group.date, true)));
    const cards = node("div", "weekly-game-grid");
    cards.append(...group.games.map((game) => gameCard(game, artifact, entry, week.key)));
    section.append(cards);
    return section;
  });
  if (!sections.length) {
    const empty = node("p", "schedule-empty", "No marquee games this week.");
    $("#weekly-schedule").replaceChildren(empty);
    return;
  }
  $("#weekly-schedule").replaceChildren(...sections);
}

async function load() {
  const response = await fetch("./data/manifest.json");
  if (!response.ok) throw new Error("Could not load the published manifest.");
  state.manifest = await response.json();
  if (!state.season || !state.manifest.seasons.includes(state.season)) state.season = state.manifest.seasons[0];
  const entry = selectedEntry() || chooseDefault();
  populateControls(entry);
  if (!entry) throw new Error("No published snapshot matches this Schedule view.");
  updateBackLink(entry, state.weekKey);
  const artifactResponse = await fetch(`./${entry.week_games_path}`);
  if (!artifactResponse.ok) throw new Error("The selected Schedule artifact is unavailable.");
  const artifact = await artifactResponse.json();
  if (artifact.snapshot_id !== entry.snapshot_id || artifact.season !== entry.season) throw new Error("The selected Schedule artifact does not match the snapshot.");
  const week = renderWeekPicker(artifact, entry);
  if (!week) throw new Error("No weeks are available for this Schedule.");
  if (!params.has("week") || !params.has("view")) {
    const canonicalUrl = new URL(window.location.href);
    canonicalUrl.search = contextParams(entry, week.key).toString();
    window.history.replaceState({}, "", canonicalUrl);
  }
  updateBackLink(entry, week.key);
  clearScheduleStatus();
  renderSchedule(week, artifact, entry);
}

$("#family-select").addEventListener("change", (event) => { const next = new URLSearchParams(params); next.set("family", event.target.value); next.delete("snapshot"); navigateTo(next); });
$("#season-select").addEventListener("change", (event) => { const next = new URLSearchParams(params); next.set("season", event.target.value); next.delete("snapshot"); navigateTo(next); });
$("#snapshot-select").addEventListener("change", (event) => { const next = new URLSearchParams(params); next.set("snapshot", event.target.value); navigateTo(next); });
document.querySelectorAll("[data-prior]").forEach((button) => button.addEventListener("click", () => { const next = new URLSearchParams(params); next.set("prior", button.dataset.prior); next.delete("snapshot"); navigateTo(next); }));
$("#week-select").addEventListener("change", (event) => { const next = new URLSearchParams(params); next.set("week", event.target.value); navigateTo(next); });
$("#previous-week").addEventListener("click", () => { const options = [...$("#week-select").options]; const index = options.findIndex((option) => option.selected); if (index > 0) { const next = new URLSearchParams(params); next.set("week", options[index - 1].value); navigateTo(next); } });
$("#next-week").addEventListener("click", () => { const options = [...$("#week-select").options]; const index = options.findIndex((option) => option.selected); if (index >= 0 && index < options.length - 1) { const next = new URLSearchParams(params); next.set("week", options[index + 1].value); navigateTo(next); } });
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => { state.view = button.dataset.view; const next = new URLSearchParams(params); if (state.view === "marquee") next.set("view", "marquee"); else next.delete("view"); navigateTo(next); }));

load().catch((error) => { $("#schedule-status").textContent = error.message; $("#schedule-status").className = "team-page-status team-page-error"; });
