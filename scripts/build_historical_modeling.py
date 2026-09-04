"""Build historical rank outcomes, joined games, and validate margin likelihoods."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from gippyrank.modeling import (
    PAIRINGS,
    build_distributions,
    coverage_table,
    design_matrix,
    fit_robust_surface,
    read_csv,
    student_t_nll,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/modeling"


def build_joined(games, distributions, rank_pairs, stats):
    d = {(int(r["season"]), r["subdivision"], r["team_id"]): r for r in distributions}
    stat = defaultdict(dict)
    for row in stats:
        stat[int(row["game_id"])][int(row["team_id"])] = row.get("yards_per_play")
    joined, missing = [], Counter()
    for game in games:
        if game["homeClassification"] not in {"fbs", "fcs"} or game["awayClassification"] not in {"fbs", "fcs"}:
            continue
        if game["homePoints"] in (None, "") or game["awayPoints"] in (None, ""):
            continue
        season = int(game["season"])
        hkey = (season, game["homeClassification"], game["homeId"])
        akey = (season, game["awayClassification"], game["awayId"])
        hr, ar = d.get(hkey), d.get(akey)
        if hr is None or ar is None:
            missing["home" if hr is None else "away"] += 1
            continue
        pairs = rank_pairs.get((int(game["id"]), hkey, akey), [])
        if not pairs:
            rng = np.random.default_rng(int(game["id"]) % (2**32))
            hp = json.loads(hr["rank_observations"]); ap = json.loads(ar["rank_observations"])
            pairs = [(int(rng.choice(hp)), int(rng.choice(ap))) for _ in range(min(12, max(len(hp), len(ap))))]
            pairing_method = "marginal_fallback"
        else:
            pairing_method = "same_system"
        ys = stat[int(game["id"])]
        ypp_h, ypp_a = ys.get(int(game["homeId"])), ys.get(int(game["awayId"]))
        joined.append({
            "game_id": game["id"], "season": season, "week": game["week"], "start_date": game["startDate"],
            "home_team_id": game["homeId"], "away_team_id": game["awayId"],
            "home_subdivision": game["homeClassification"], "away_subdivision": game["awayClassification"],
            "home_team_population": hr["team_population"], "away_team_population": ar["team_population"],
            "home_points": game["homePoints"], "away_points": game["awayPoints"],
            "margin": int(game["homePoints"]) - int(game["awayPoints"]),
            "neutral_site": game["neutralSite"], "pairing": "fbs-fcs" if {game["homeClassification"], game["awayClassification"]} == {"fbs", "fcs"} else f'{game["homeClassification"]}-{game["awayClassification"]}',
            "rank_pairs": json.dumps(pairs, separators=(",", ":")), "pairing_method": pairing_method,
            "home_rank_observations": hr["rank_observations"], "away_rank_observations": ar["rank_observations"],
            "home_ypp": ypp_h or "", "away_ypp": ypp_a or "",
        })
    return joined, missing


def pseudo_data(rows, seed=20260903):
    del seed  # Pair observations are already deterministic; fallback sampling is seeded upstream.
    x, y, z, pairing, home, neutral, weights, game_ids, ypp = [], [], [], [], [], [], [], [], []
    for row in rows:
        pairs = json.loads(row["rank_pairs"])
        n = len(pairs)
        for a, b in pairs:
            x.append((a - .5) / int(row["home_team_population"]))
            y.append((b - .5) / int(row["away_team_population"]))
            z.append(float(row["margin"])); pairing.append(row["pairing"]); home.append(1.0 - bool(row["neutral_site"])); neutral.append(bool(row["neutral_site"])); weights.append(1/n); game_ids.append(int(row["game_id"]))
            yh, ya = row["home_ypp"], row["away_ypp"]
            ypp.append(float(yh) - float(ya) if yh and ya else np.nan)
    return (np.array(x), np.array(y), np.array(z), np.array(pairing), np.array(home), np.array(neutral), np.array(weights), np.array(game_ids), np.array(ypp))


def evaluate(model, X, y, group, rows):
    location = X @ model["beta"]
    interval = student_t.ppf(.90, model["df"]) * model["scale"]
    out = {"n": len(y), "nll": student_t_nll(y, location, model["scale"], model["df"]), "mae": float(np.mean(np.abs(y-location))), "central_80pct_coverage": float(np.mean(np.abs(y-location) <= interval))}
    for p in PAIRINGS:
        m = group == p
        if m.any(): out[p] = {"n": int(m.sum()), "nll": student_t_nll(y[m], location[m], model["scale"], model["df"]), "mae": float(np.mean(np.abs(y[m]-location[m])))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--threshold", type=float, default=.90); args = parser.parse_args()
    observations = read_csv(ROOT / "data/processed/massey/final_observations.csv")
    games = read_csv(ROOT / "data/processed/cfbd/games.csv")
    stats = read_csv(ROOT / "data/processed/cfbd/team_game_stats.csv")
    depth = coverage_table(observations)
    distributions, by_team = build_distributions(observations, depth, args.threshold)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "system_depth.csv", list(depth[0]), depth)
    write_csv(OUT / "team_season_rank_distributions.csv", list(distributions[0]), distributions)
    rank_pairs = {}
    selected_codes = defaultdict(set)
    for item in depth:
        if float(item["coverage_ratio"]) >= args.threshold:
            selected_codes[(int(item["season"]), item["subdivision"])].add(item["system_code"])
    cross_overlap = {str(season): len(selected_codes[(season, "fbs")] & selected_codes[(season, "fcs")]) for season in range(2003, 2026)}
    for g in games:
        if g["homeClassification"] not in {"fbs", "fcs"} or g["awayClassification"] not in {"fbs", "fcs"}: continue
        hk=(int(g["season"]),g["homeClassification"],g["homeId"]); ak=(int(g["season"]),g["awayClassification"],g["awayId"])
        h = dict(by_team.get(hk, [])); a = dict(by_team.get(ak, [])); common = sorted(set(h) & set(a))
        if common: rank_pairs[(int(g["id"]),hk,ak)] = [(h[c],a[c]) for c in common]
    joined, missing = build_joined(games, distributions, rank_pairs, stats)
    fields = list(joined[0])
    write_csv(OUT / "historical_modeling_games.csv", fields, joined)
    x,y,z,pair,home,neutral,w,game_ids,_ypp = pseudo_data(joined)
    seasons = {int(r["game_id"]): int(r["season"]) for r in joined}
    train = np.array([seasons[gid] < 2022 for gid in game_ids])
    X = design_matrix(x,y,pair,home,neutral,True); B = design_matrix(x,y,pair,home,neutral,False)
    surface = fit_robust_surface(X[train],z[train],w[train]); benchmark = fit_robust_surface(B[train],z[train],w[train])
    ypp_model = None
    ypp_eval = {"available_games": 0}
    raw_ypp = _ypp
    ypp_train = train & np.isfinite(raw_ypp) & (pair != "fcs-fcs")
    ypp_test = ~train & np.isfinite(raw_ypp) & (pair != "fcs-fcs")
    if ypp_train.any() and ypp_test.any():
        ypp_model = fit_robust_surface(np.column_stack([X[ypp_train], raw_ypp[ypp_train]]), z[ypp_train], w[ypp_train])
        ypp_eval = {"available_games": len(set(game_ids[ypp_test])), "margin_surface": evaluate(surface, X[ypp_test], z[ypp_test], pair[ypp_test], joined), "margin_plus_ypp": evaluate(ypp_model, np.column_stack([X[ypp_test], raw_ypp[ypp_test]]), z[ypp_test], pair[ypp_test], joined)}
    result = {"specification": {"criterion": f"coverage >= {args.threshold:.2f}", "student_t_df": 5, "seed": 20260903, "pseudo_weight_per_game": 1.0, "surface": "tensor hinge basis with pairing-specific surfaces and home/neutral context"}, "coverage": {"usable_games": len(joined), "missing_rank_distribution_games": dict(missing), "by_pairing": dict(Counter(r["pairing"] for r in joined)), "same_system_games": sum(r["pairing_method"]=="same_system" for r in joined), "cross_subdivision_system_overlap_by_season": cross_overlap}, "validation": {"modern_holdout": {"n_games": len(set(game_ids[~train])), "surface": evaluate(surface,X[~train],z[~train],pair[~train],joined), "linear_benchmark": evaluate(benchmark,B[~train],z[~train],pair[~train],joined)}, "training": evaluate(surface,X[train],z[train],pair[train],joined)}, "ypp_exploration": ypp_eval, "model": {"scale": surface["scale"], "df": surface["df"], "beta": surface["beta"].tolist()}}
    (OUT / "margin_model_results.json").write_text(json.dumps(result, indent=2)+"\n")
    expected_team_seasons = len({(int(row["season"]), row["subdivision"], row["team_id"]) for row in observations if row["is_composite"] == "True"})
    excluded = [d for d in depth if float(d["coverage_ratio"]) < args.threshold]
    report = {"chosen_threshold": args.threshold, "criterion": "A constituent system is usable when it ranks at least 90% of the CMP team population for that season and subdivision. CMP is always excluded.", "sensitivity": {}, "systems": len(depth), "surviving_system_groups": sum(float(d["coverage_ratio"])>=args.threshold for d in depth), "team_seasons": len(distributions), "expected_team_seasons": expected_team_seasons, "missing_team_seasons": expected_team_seasons - len(distributions), "representative_excluded_systems": sorted(excluded, key=lambda d: float(d["coverage_ratio"]))[:20], "selected_systems_with_rank_gaps": [d for d in depth if float(d["coverage_ratio"]) >= args.threshold and d["has_rank_gaps"]][:20]}
    for threshold in (.90,.95):
        ds,_ = build_distributions(observations,depth,threshold); report["sensitivity"][str(threshold)] = {"team_seasons":len(ds), "usable_system_groups":sum(float(d["coverage_ratio"])>=threshold for d in depth)}
    (OUT / "historical_modeling_report.json").write_text(json.dumps(report, indent=2)+"\n")
    hold = result["validation"]["modern_holdout"]
    ypp_result = result["ypp_exploration"]
    (OUT / "historical_modeling_report.md").write_text(
        "# Historical modeling report\n\n"
        f"The baseline uses constituent systems with coverage >= {args.threshold:.0%} of the final CMP population. `CMP` is excluded; `MAS` is treated like every other constituent. Coverage has a natural separation between near-complete systems and censored top-N systems. A 95% sensitivity run is included below.\n\n"
        f"## Rank distributions\n\n- Team-seasons represented: {len(distributions)} / {expected_team_seasons}\n- Missing team-seasons: {expected_team_seasons-len(distributions)}\n- Usable system groups: {report['surviving_system_groups']} / {len(depth)}\n\n## Game join\n\n- Usable games: {len(joined)}\n- Pairings: {dict(Counter(r['pairing'] for r in joined))}\n- Same-system rank pairs: {sum(r['pairing_method']=='same_system' for r in joined)}\n- Marginal fallback games: {sum(r['pairing_method']=='marginal_fallback' for r in joined)}\n- Missing distributions: {dict(missing)}\n- Cross-subdivision usable system overlap by season is stored in `margin_model_results.json`.\n\n## Margin likelihood\n\nThe likelihood is a Student-t regression (df=5) over a tensor-product hinge surface of the two within-subdivision rank-percentile coordinates, with pairing-specific surfaces and home/neutral context. Rank-pair pseudo-observations are weighted so each game contributes one total unit. This is an observation model, not a scalar team-strength model.\n\nModern holdout (2022-2025) surface NLL: {hold['surface']['nll']:.3f}; linear benchmark NLL: {hold['linear_benchmark']['nll']:.3f}. Surface MAE: {hold['surface']['mae']:.3f}; benchmark MAE: {hold['linear_benchmark']['mae']:.3f}. Central 80% interval coverage: {hold['surface']['central_80pct_coverage']:.3f}.\n\n## YPP exploration\n\nYPP is used only where both teams have observed YPP, focusing on FBS-FBS and FBS-FCS. It is not part of the production margin model. Holdout comparison: {ypp_result}.\n\n## Sensitivity and limitations\n\n" + "\n".join(f"- {k}: {v}" for k,v in report["sensitivity"].items()) + "\n\nThis first model uses final-season ranks as historical outcome observations, so it is appropriate for learning the observation likelihood but not for current-season inference. Future work should improve rank-coordinate scaling by actual season population, quantify uncertainty in prediction intervals, and revisit calendar/season-specific effects before fitting a posterior ranking model.\n",
        encoding="utf-8",
    )
    print(json.dumps({"team_seasons":len(distributions),"usable_games":len(joined),"holdout_nll":result["validation"]["modern_holdout"]["surface"]["nll"]}, indent=2))


if __name__ == "__main__": main()
