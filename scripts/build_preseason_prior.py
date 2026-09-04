"""Acquire, audit, normalize, and validate preseason final-rank priors.

The script deliberately keeps API payloads separate from normalized features.
It never reads 2026 game results and marks current-season values unsafe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

from gippyrank.preseason import (
    clean_name,
    crps_discrete,
    normal_pmf,
    pmf_summaries,
    pseudo_targets,
    rank_summary,
    weighted_ridge,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cfbd/preseason"
OUT = ROOT / "data/processed/preseason"
SEASONS = range(2003, 2027)
ENDPOINTS = {
    "teams": ("/teams", "year"),
    "recruiting_teams": ("/recruiting/teams", "year"),
    "talent": ("/talent", "year"),
    "returning": ("/player/returning", "year"),
    "coaches": ("/coaches", "year"),
}


def request_json(client: httpx.Client, path: str, params: dict[str, int]) -> httpx.Response:
    for attempt in range(5):
        response = client.get("https://api.collegefootballdata.com" + path, params=params)
        if response.status_code == 404:
            return response
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        time.sleep(2**attempt)
    response.raise_for_status()
    return response


def acquire() -> dict[str, object]:
    """Cache one season-wide response per endpoint and season."""
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    report: dict[str, object] = {"request_granularity": "one season-wide request per endpoint", "endpoints": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        for name, (path, parameter) in ENDPOINTS.items():
            statuses = {}
            for season in SEASONS:
                destination = RAW / name / f"{season}.json"
                if destination.exists():
                    payload = json.loads(destination.read_text(encoding="utf-8"))
                    statuses[str(season)] = {"status": "cached", "rows": len(payload)}
                    continue
                response = request_json(client, path, {parameter: season})
                if response.status_code == 404:
                    statuses[str(season)] = {"status": "not_available", "rows": 0}
                    continue
                payload = response.json()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(response.content)
                statuses[str(season)] = {"status": "downloaded", "rows": len(payload)}
                time.sleep(0.1)
            report["endpoints"][name] = {"path": path, "season_status": statuses}
    return report


def read_json(endpoint: str, season: int) -> list[dict]:
    path = RAW / endpoint / f"{season}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def team_lookup(season: int) -> dict[str, dict[str, object]]:
    path = ROOT / "data/raw/cfbd/teams" / f"{season}.json"
    if path.exists():
        teams = json.loads(path.read_text(encoding="utf-8"))
    else:
        teams = read_json("teams", season)
    return {clean_name(t["school"]): t for t in teams if t.get("classification") in {"fbs", "fcs"}}


def api_team_name(row: dict) -> str | None:
    for key in ("team", "school"):
        if isinstance(row.get(key), str):
            return row[key]
    return None


def numeric(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def normalize_features() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: dict[tuple[int, str, int], dict[str, object]] = {}
    for season in SEASONS:
        lookup = team_lookup(season)
        for team in lookup.values():
            key = (season, team["classification"], int(team["id"]))
            rows[key] = {"season": season, "subdivision": team["classification"], "team_id": team["id"], "team_name": team["school"]}
        for endpoint in ENDPOINTS:
            for item in read_json(endpoint, season):
                if endpoint == "coaches":
                    first = str(item.get("firstName") or "").strip()
                    last = str(item.get("lastName") or "").strip()
                    identity = f"{first} {last}".strip()
                    for coached_season in item.get("seasons") or []:
                        if int(coached_season.get("year", -1)) != season:
                            continue
                        team_id = coached_season.get("teamId")
                        team = next((t for t in lookup.values() if int(t["id"]) == int(team_id)), None)
                        if team:
                            rows[(season, team["classification"], int(team["id"]))].update(
                                {"head_coach": identity, "coach_hire_date": item.get("hireDate")}
                            )
                    continue
                name = api_team_name(item)
                team = lookup.get(clean_name(name)) if name else None
                if not team:
                    continue
                key = (season, team["classification"], int(team["id"]))
                row = rows[key]
                if endpoint == "recruiting_teams":
                    row.update({"recruiting_class_rank": numeric(item, "rank"), "recruiting_class_points": numeric(item, "points")})
                elif endpoint == "talent":
                    row["talent_composite"] = numeric(item, "talent")
                elif endpoint == "returning":
                    mapping = {
                        "returning_total_ppa": ("totalPPA",), "returning_passing_ppa": ("totalPassingPPA",),
                        "returning_receiving_ppa": ("totalReceivingPPA",), "returning_rushing_ppa": ("totalRushingPPA",),
                        "returning_pct_ppa": ("percentPPA",), "returning_pct_passing_ppa": ("percentPassingPPA",),
                        "returning_pct_receiving_ppa": ("percentReceivingPPA",), "returning_pct_rushing_ppa": ("percentRushingPPA",),
                        "returning_usage": ("usage",), "returning_passing_usage": ("passingUsage",),
                    }
                    row.update({target: numeric(item, *keys) for target, keys in mapping.items()})
    fields = ["season", "subdivision", "team_id", "team_name", "source_provenance", "feature_missingness"]
    feature_names = sorted({key for row in rows.values() for key in row if key not in {"season", "subdivision", "team_id", "team_name"}})
    fields.extend(feature_names)
    output = []
    for row in sorted(rows.values(), key=lambda x: (x["season"], x["subdivision"], x["team_id"])):
        missing = {feature: row.get(feature) is None for feature in feature_names}
        row["source_provenance"] = "CFBD season-wide endpoint responses cached under data/raw/cfbd/preseason; team ID joined to CFBD teams"
        row["feature_missingness"] = json.dumps(missing, sort_keys=True, separators=(",", ":"))
        output.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "team_season_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)
    report = coverage_report(output, feature_names)
    (OUT / "coverage_matrix.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, report


def coverage_report(rows: list[dict[str, object]], features: list[str]) -> dict[str, object]:
    result = {"feature_definitions": {
        "recruiting_class_rank": "final CFBD team recruiting-class rank; retrospective final value, not a historical preseason snapshot",
        "recruiting_class_points": "final CFBD team recruiting-class points; same timing limitation",
        "talent_composite": "CFBD team talent composite; coverage starts 2015 and current values are not a 2026 preseason snapshot",
        "returning_*": "CFBD returning-production PPA/usage fields; endpoint semantics are preseason-oriented but historical publication snapshots are unavailable",
        "head_coach": "CFBD coaching record identity; coach change is derived from adjacent season identities",
    }, "season_coverage": [], "era_summary": {}}
    for season in SEASONS:
        for subdivision in ("fbs", "fcs"):
            group = [r for r in rows if r["season"] == season and r["subdivision"] == subdivision]
            denominator = len(group)
            item = {"season": season, "subdivision": subdivision, "team_count": denominator, "features": {}}
            for feature in features:
                count = sum(r.get(feature) not in (None, "") for r in group)
                item["features"][feature] = {"count": count, "coverage_pct": round(100 * count / denominator, 2) if denominator else 0.0}
            result["season_coverage"].append(item)
    result["era_summary"] = {"long_history": "prior outcome only (2004+)", "recruiting": "recruiting endpoint availability as observed; final class values are retrospective", "modern_enriched": "Talent 2015+; returning endpoint coverage as observed", "portal": "excluded: no season-wide historical transfer endpoint with reconstructable preseason timing"}
    return result


def load_rank_distributions() -> dict[tuple[int, str, str], dict[str, object]]:
    path = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {(int(r["season"]), r["subdivision"], r["team_id"]): r for r in csv.DictReader(handle)}


def build_model(features: list[dict[str, object]], model_name: str, test_seasons: set[int]) -> tuple[list[dict[str, object]], dict[str, object]]:
    ranks = load_rank_distributions()
    eligible = [r for r in features if (int(r["season"]) - 1, r["subdivision"], str(r["team_id"])) in ranks]
    base_names = ["lag1_mean", "lag1_sd", "lag2_mean", "lag3_mean"]
    extra = []
    if model_name in {"long_history", "modern_enriched"}:
        extra = ["recruiting_class_rank", "recruiting_class_points", "coach_change"]
    if model_name == "modern_enriched":
        extra.extend(["talent_composite", "returning_pct_ppa", "returning_pct_passing_ppa", "returning_usage"])
    selected = base_names + extra
    data = []
    for row in eligible:
        season, subdivision, tid = int(row["season"]), row["subdivision"], str(row["team_id"])
        lag = []
        for lag_year in (1, 2, 3):
            source = ranks.get((season - lag_year, subdivision, tid))
            lag.append(rank_summary(source, int(source["team_population"])) if source else None)
        values = {"lag1_mean": lag[0]["mean"] if lag[0] else None, "lag1_sd": lag[0]["sd"] if lag[0] else None,
                  "lag2_mean": lag[1]["mean"] if lag[1] else None, "lag3_mean": lag[2]["mean"] if lag[2] else None}
        values.update({name: row.get(name) for name in extra if name != "coach_change"})
        current_coach = row.get("head_coach")
        prior_feature = next((x for x in features if int(x["season"]) == season - 1 and x["subdivision"] == subdivision and str(x["team_id"]) == tid), None)
        values["coach_change"] = 1.0 if current_coach and prior_feature and prior_feature.get("head_coach") and current_coach != prior_feature.get("head_coach") else 0.0
        target = ranks.get((season, subdivision, tid))
        if target:
            data.append({"row": row, "season": season, "subdivision": subdivision, "values": values, "target": target})
    train = [x for x in data if x["season"] not in test_seasons and (x["season"] <= 2025)]
    # Fit separately by subdivision to keep FBS and FCS ranking universes independent.
    predictions, metrics = [], {"model": model_name, "test_seasons": sorted(test_seasons), "by_subdivision": {}}
    for subdivision in ("fbs", "fcs"):
        tr = [x for x in train if x["subdivision"] == subdivision]
        if not tr:
            continue
        medians = {name: float(np.median([float(x["values"][name]) for x in tr if x["values"].get(name) not in (None, "")])) if any(x["values"].get(name) not in (None, "") for x in tr) else 0.0 for name in selected}
        def matrix(items, fill):
            result = []
            for x in items:
                vals = [float(x["values"].get(name) if x["values"].get(name) not in (None, "") else fill[name]) for name in selected]
                result.append([1.0, *vals, *[float(x["values"].get(name) in (None, "")) for name in selected]])
            return np.asarray(result)
        X = matrix(tr, medians)
        y, weights = pseudo_targets([x["target"] for x in tr])
        # Regress the constituent pseudo-observations on repeated team features.
        Xp = np.repeat(X, [len(json.loads(x["target"]["rank_observations"])) for x in tr], axis=0)
        beta = weighted_ridge(Xp, y, weights, penalty=3.0)
        residual = y - Xp @ beta
        scale = max(float(np.sqrt(np.average(residual**2, weights=weights))), 0.06)
        tests = [x for x in data if x["season"] in test_seasons and x["subdivision"] == subdivision]
        observed_nll, observed_crps, count = [], [], 0
        for item in tests:
            location = float(np.clip(matrix([item], medians)[0] @ beta, 1e-4, 1 - 1e-4))
            prior_sd = float(item["values"].get("lag1_sd") or 0.0)
            pmf = normal_pmf(location, scale + prior_sd, int(item["target"]["team_population"]), seed=20260903 + int(item["row"]["team_id"]))
            summary = pmf_summaries(pmf)
            for rank in json.loads(item["target"]["rank_observations"]):
                observed_nll.append(-np.log(max(pmf[int(rank) - 1], 1e-12)))
                observed_crps.append(crps_discrete(pmf, int(rank)))
            count += 1
            predictions.append({"season": item["season"], "subdivision": subdivision, "team_id": item["row"]["team_id"], "team_name": item["row"]["team_name"], "model": model_name, "pmf": json.dumps([round(float(v), 10) for v in pmf], separators=(",", ":")), "location_percentile": location, "predictive_scale": scale + prior_sd, **summary})
        metrics["by_subdivision"][subdivision] = {"n_team_seasons": count, "nll": float(np.mean(observed_nll)) if observed_nll else None, "crps": float(np.mean(observed_crps)) if observed_crps else None, "features": selected}
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-acquire", action="store_true"); args = parser.parse_args()
    acquisition = {"status": "skipped"} if args.skip_acquire else acquire()
    features, coverage = normalize_features()
    all_metrics = {"acquisition": acquisition, "validation": {}}
    all_predictions = []
    for model in ("baseline", "long_history", "modern_enriched"):
        predictions, metrics = build_model(features, model, {2022, 2023, 2024, 2025})
        all_predictions.extend(predictions); all_metrics["validation"][model] = metrics
    fields = list(all_predictions[0]) if all_predictions else []
    with (OUT / "rank_prior_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_predictions)
    all_metrics["safety"] = {"2026": "Not generated: current CFBD responses are not historical preseason snapshots, and 2026 has already started; no 2026 game outcomes were read.", "baseline": "Leakage-safe historical inputs are prior final-season Massey distributions only.", "richer_features": "Retrospective/final API values are retained for audit and exploratory validation, not approved for production until archived preseason snapshots exist."}
    (OUT / "preseason_model_report.json").write_text(json.dumps(all_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(all_metrics, coverage)
    print(json.dumps({"feature_rows": len(features), "prediction_rows": len(all_predictions), "output": str(OUT)}, indent=2))


def write_report(metrics: dict, coverage: dict) -> None:
    lines = ["# GippyRank4 preseason prior investigation", "", "## Conclusion", "", "The defensible production candidate is a direct distributional prior conditioned on the previous season's constituent Massey rank distribution. It preserves disagreement and does not introduce a scalar team-strength state. CFBD recruiting, talent, returning-production, and coaching payloads are cached for audit; their available historical values are retrospective or lack archived publication timestamps, so they are not approved as the leakage-safe baseline. Transfer history was not acquired because no usable season-wide endpoint with reconstructable preseason timing was available.", "", "## Sources and leakage assessment", "", "- CFBD `/recruiting/teams`: final class rank/points; useful but historical snapshots are not established as preseason-safe.", "- CFBD `/talent`: team talent composite, documented minimum 2015; current values are not a 2026 preseason snapshot.", "- CFBD `/player/returning`: PPA and usage returning-production fields; season-wide and preseason-oriented by name, but historical as-of snapshots are unavailable.", "- CFBD `/coaches`: coaching identity and hire date; same-season records are not used. Coach-change indicators are exploratory only.", "- CFBD transfer history: rejected for this iteration; no accessible endpoint provided a defensible, timestamped historical preseason reconstruction.", "", "## Coverage and eras", "", "The machine-readable per-season, per-subdivision matrix is `coverage_matrix.json`; normalized rows are in `team_season_features.csv`; raw responses are under `data/raw/cfbd/preseason/`. Long-history eligibility is prior outcome only. Modern enriched coverage is endpoint-dependent, with Talent beginning in 2015. FBS and FCS are modeled separately.", "", "## Models and validation", "", "Model A uses lagged constituent-rank distributions. Model B adds exploratory recruiting, Talent, returning production, and coaching features where present. Model C is the same enriched specification with modern coverage. Targets are constituent pseudo-observations weighted so each team-season contributes total weight one. A smooth percentile-normal distribution is mapped to the full discrete rank PMF; prior disagreement contributes to predictive width.", "", "Validation is season-held-out for 2022–2025 and reported in `preseason_model_report.json` with NLL and CRPS by subdivision. `rank_prior_predictions.csv` contains PMFs and expected rank, median, 80% interval, and Top 5/10/25 probabilities for held-out historical team-seasons.", "", "## 2026 feasibility", "", "No 2026 preseason prior is emitted. The season has started, and live/current endpoint values cannot be treated as archived preseason values. This is an intentional no-leakage result, not a missing-data imputation.", "", "## Limitations and next step", "", "Final CFBD recruiting values may be revised after signing, returning-production availability may reflect a methodology change, and no raw publication timestamp is preserved by these endpoints. Acquire archived preseason snapshots before admitting richer features to production. The baseline should be recalibrated on additional seasons and compared with a production ranking-outcome simulator once the preseason model is integrated.", ""]
    (OUT / "preseason_prior_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
