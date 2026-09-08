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

function rank(value) {
  return `#${Number(value).toFixed(1)}`;
}

function publicationStatusLabel(entry) {
  return entry.publication_status === "official" ? "Official" : "Interim";
}

function entryFor(manifest) {
  const family = params.get("family") === "performance" ? "performance" : "predictive";
  const season = Number(params.get("season")) || manifest.seasons[0];
  let entries = manifest.snapshots.filter((item) => item.season === season && item.ranking_family === family);
  const prior = params.get("prior");
  if (family === "predictive" && (prior === "context" || prior === "history")) {
    entries = entries.filter((item) => item.prior_family === prior);
  }
  const snapshot = params.get("snapshot");
  return entries.find((item) => item.snapshot_id === snapshot)
    ?? entries.find((item) => item.publication_slot === snapshot)
    ?? entries.find((item) => item.publication_slot === manifest.default_publication_slot)
    ?? entries[0];
}

function updateBackLink(entry) {
  const url = new URL("./", document.baseURI);
  if (!entry) {
    $("#back-to-rankings").href = url.pathname;
    return;
  }
  url.searchParams.set("season", String(entry.season));
  url.searchParams.set("snapshot", entry.snapshot_id);
  url.searchParams.set("family", entry.ranking_family);
  if (entry.ranking_family === "predictive") url.searchParams.set("prior", entry.prior_family);
  $("#back-to-rankings").href = `${url.pathname}${url.search}`;
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
  const summary = node("div", "team-summary-grid");
  summary.append(
    node("div", "team-summary-item", `${rankLabel} · expected ${rank(row.expected_rank)}`),
    node("div", "team-summary-item", row.rank_change_accessible || "No earlier official ranking baseline"),
    node("div", "team-summary-item", `Median ${row.median_rank} · 80% ${row.interval_80[0]}–${row.interval_80[1]}`),
    node("div", "team-summary-item", `${row.record} modeled record · ${percentage(row.top25_probability)} Top 25 probability`),
  );
  $("#team-ranking-summary").replaceChildren(summary);
}

function ratingPanel(rating) {
  const panel = node("div", "game-rating");
  const title = node("strong", "game-rating-title", `Played like ${rank(rating.expected_rank)}`);
  const interval = node("span", "game-rating-interval", `80% interval: ${rating.interval_80[0]}–${rating.interval_80[1]}`);
  interval.setAttribute("aria-label", `central 80 percent interval from rank ${rating.interval_80[0]} through rank ${rating.interval_80[1]}`);
  panel.append(title, interval);
  const details = node("dl", "game-rating-details");
  [["Median", `#${rating.median_rank}`], ["Mode", `#${rating.mode_rank}`], ["50% interval", `${rating.interval_50[0]}–${rating.interval_50[1]}`], ["95% interval", `${rating.interval_95[0]}–${rating.interval_95[1]}`], ["Top 5", percentage(rating.top5_probability)], ["Top 10", percentage(rating.top10_probability)], ["Top 25", percentage(rating.top25_probability)]].forEach(([label, value]) => details.append(node("dt", "", label), node("dd", "", value)));
  panel.append(details);
  return panel;
}

function gameCard(game, cutoff) {
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
  const future = cutoff === null || new Date(game.date) > cutoff;
  if (game.result && game.score) outcome.textContent = `${game.result} ${game.score.team}–${game.score.opponent}`;
  else if (future) outcome.textContent = "Future at this snapshot";
  else outcome.textContent = "Not completed by this snapshot";
  const body = node("div", "game-card-body");
  if (game.game_rating) {
    body.append(ratingPanel(game.game_rating));
  } else if (future) {
    body.append(node("p", "game-not-modeled", "No result or game rating is shown after the selected snapshot cutoff."));
  } else if (game.result) {
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
  $("#team-page-status").textContent = games.length ? `${games.length} scheduled games · game ratings include uncertainty` : "No schedule entries are available for this team.";
  $("#schedule-list").replaceChildren(...games.map((game) => gameCard(game, cutoff)));
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
  const [snapshotResponse, teamResponse] = await Promise.all([
    fetch(`./${entry.data_path}`),
    fetch(`./${entry.team_seasons_path || entry.team_season_path}`),
  ]);
  if (!snapshotResponse.ok || !teamResponse.ok) throw new Error("The selected team-season artifact is unavailable.");
  const [snapshot, artifact] = await Promise.all([snapshotResponse.json(), teamResponse.json()]);
  if (artifact.snapshot_id !== entry.snapshot_id || artifact.season !== entry.season) throw new Error("The team-season artifact does not match the selected snapshot.");
  const row = snapshot.rankings.find((item) => item.team_id === teamId);
  if (!row) throw new Error("This team is not available in the selected ranking snapshot.");
  renderSummary(entry, snapshot, row);
  renderSchedule(artifact, entry);
}

load().catch((error) => {
  updateBackLink(null);
  $("#team-page-status").textContent = error.message;
  $("#team-page-status").className = "team-page-status team-page-error";
});
