const state = { manifest: null, family: "predictive", season: null, snapshotId: null, prior: "context", depth: 25 };
const $ = (selector) => document.querySelector(selector);

function labelFor(entry) { return `${entry.display_label} · ${entry.snapshot_type === "preseason" ? "Preseason" : "Current"}`; }
function choices() { return state.manifest.snapshots.filter((entry) => entry.ranking_family === state.family && entry.season === state.season); }
function selectedEntry() { return choices().find((entry) => entry.snapshot_id === state.snapshotId && entry.prior_family === state.prior); }
function formatDate(value) { return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(value)); }
function formatTimestamp(value) { return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", timeZone: "UTC", timeZoneName: "short" }).format(new Date(value)); }
function percentage(value) { return `${Math.round(value * 100)}%`; }

function populate() {
  const families = state.manifest.ranking_families;
  $("#family-select").replaceChildren(...families.map((family) => new Option(family.label, family.id, family.id === state.family, family.id === state.family)));
  const seasons = state.manifest.seasons;
  state.season ??= seasons[0];
  $("#season-select").replaceChildren(...seasons.map((season) => new Option(season, season, season === state.season, season === state.season)));
  const available = choices();
  const snapshotGroups = [...new Map(available.map((entry) => [entry.display_label, entry])).values()];
  if (!available.some((entry) => entry.snapshot_id === state.snapshotId && entry.prior_family === state.prior)) state.snapshotId = available.find((entry) => entry.prior_family === state.prior)?.snapshot_id ?? available[0]?.snapshot_id;
  $("#snapshot-select").replaceChildren(...snapshotGroups.map((entry) => new Option(labelFor(entry), entry.display_label, entry.snapshot_id === state.snapshotId || state.snapshotId?.includes(entry.display_label.toLowerCase()), entry.snapshot_id === state.snapshotId)));
  $("#snapshot-select").value = selectedEntry()?.display_label ?? snapshotGroups[0]?.display_label;
  document.querySelectorAll("[data-prior]").forEach((button) => button.classList.toggle("is-active", button.dataset.prior === state.prior));
}

async function render() {
  populate();
  const entry = selectedEntry();
  if (!entry) return;
  const snapshot = await fetch(`./${entry.data_path}`).then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load ranking data.")));
  const current = entry.snapshot_type !== "preseason";
  $("#snapshot-kind").textContent = current ? "IN-SEASON / CURRENT" : "PRESEASON";
  $("#snapshot-title").textContent = `${entry.season} ${entry.display_label} · ${entry.prior_family === "context" ? "Context" : "History"}`;
  $("#snapshot-freshness").textContent = current ? `Rankings through ${formatDate(entry.effective_cutoff)}.` : "Frozen before any game evidence.";
  $("#snapshot-evidence").textContent = current ? `Effective cutoff: ${formatTimestamp(snapshot.effective_cutoff)} · ${snapshot.included_game_count} eligible games included · ${snapshot.excluded_lower_division_games} lower-division games excluded.` : `Snapshot generated ${formatTimestamp(snapshot.generation_timestamp)} · 0 eligible games included.`;
  const rankings = state.depth === "all" ? snapshot.rankings : snapshot.rankings.slice(0, 25);
  $("#rankings-title").textContent = state.depth === "all" ? "All FBS rankings" : "Top 25";
  $("#rankings-body").replaceChildren(...rankings.map((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="rank">${row.display_rank}</td><td class="team">${row.team_name}<span>${row.conference || "Independent"}</span></td><td>${row.record}</td><td>${row.expected_rank.toFixed(1)}</td><td>${row.median_rank}</td><td>${row.interval_80[0]}–${row.interval_80[1]}</td><td>${percentage(row.top25_probability)}</td><td class="details-cell"><details><summary>Odds</summary><p>Top 5: <strong>${percentage(row.top5_probability)}</strong><br>Top 10: <strong>${percentage(row.top10_probability)}</strong><br>Top 25: <strong>${percentage(row.top25_probability)}</strong></p></details></td>`;
    return tr;
  }));
}

$("#family-select").addEventListener("change", (event) => { state.family = event.target.value; state.snapshotId = null; render(); });
$("#season-select").addEventListener("change", (event) => { state.season = Number(event.target.value); state.snapshotId = null; render(); });
$("#snapshot-select").addEventListener("change", (event) => { const match = choices().find((entry) => entry.display_label === event.target.value && entry.prior_family === state.prior); state.snapshotId = match?.snapshot_id; render(); });
document.querySelectorAll("[data-prior]").forEach((button) => button.addEventListener("click", () => { state.prior = button.dataset.prior; state.snapshotId = null; render(); }));
document.querySelectorAll("[data-depth]").forEach((button) => button.addEventListener("click", () => { state.depth = button.dataset.depth === "all" ? "all" : 25; document.querySelectorAll("[data-depth]").forEach((item) => item.classList.toggle("is-active", item === button)); render(); }));

fetch("./data/manifest.json").then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load manifest."))).then((manifest) => { state.manifest = manifest; return render(); }).catch((error) => { $("#snapshot-title").textContent = error.message; });
