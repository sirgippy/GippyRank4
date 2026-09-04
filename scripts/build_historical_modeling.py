"""Build and evaluate the historical score-margin likelihood."""

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
    fit_marginalized,
    fit_robust_surface,
    game_log_scores,
    read_csv,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/modeling"


def build_joined(games, distributions, rank_pairs, stats):
    dist = {
        (int(r["season"]), r["subdivision"], r["team_id"]): r for r in distributions
    }
    stat = defaultdict(dict)
    for row in stats:
        stat[int(row["game_id"])][int(row["team_id"])] = row.get("yards_per_play")
    joined, missing = [], Counter()
    for game in games:
        hc, ac = game["homeClassification"], game["awayClassification"]
        if (
            hc not in {"fbs", "fcs"}
            or ac not in {"fbs", "fcs"}
            or game["homePoints"] in (None, "")
            or game["awayPoints"] in (None, "")
        ):
            continue
        season = int(game["season"])
        hk = (season, hc, game["homeId"])
        ak = (season, ac, game["awayId"])
        hr, ar = dist.get(hk), dist.get(ak)
        if hr is None or ar is None:
            missing["home" if hr is None else "away"] += 1
            continue
        pairs = rank_pairs.get((int(game["id"]), hk, ak), [])
        if not pairs:
            rng = np.random.default_rng(int(game["id"]) % 2**32)
            hp = json.loads(hr["rank_observations"])
            ap = json.loads(ar["rank_observations"])
            pairs = [
                (int(rng.choice(hp)), int(rng.choice(ap)))
                for _ in range(min(12, max(len(hp), len(ap))))
            ]
            method = "marginal_fallback"
        else:
            method = "same_system"
        ys = stat[int(game["id"])]
        joined.append(
            {
                "game_id": game["id"],
                "season": season,
                "week": game["week"],
                "start_date": game["startDate"],
                "home_team_id": game["homeId"],
                "away_team_id": game["awayId"],
                "home_subdivision": hc,
                "away_subdivision": ac,
                "home_team_population": hr["team_population"],
                "away_team_population": ar["team_population"],
                "home_points": game["homePoints"],
                "away_points": game["awayPoints"],
                "margin": int(game["homePoints"]) - int(game["awayPoints"]),
                "neutral_site": game["neutralSite"],
                "pairing": "fbs-fcs" if {hc, ac} == {"fbs", "fcs"} else f"{hc}-{ac}",
                "rank_pairs": json.dumps(pairs, separators=(",", ":")),
                "pairing_method": method,
                "home_rank_observations": hr["rank_observations"],
                "away_rank_observations": ar["rank_observations"],
                "home_ypp": ys.get(int(game["homeId"]), "") or "",
                "away_ypp": ys.get(int(game["awayId"]), "") or "",
            }
        )
    return joined, missing


def pseudo_data(rows):
    values = [[] for _ in range(10)]
    for row in rows:
        pairs = json.loads(row["rank_pairs"])
        n = len(pairs)
        cross = row["pairing"] == "fbs-fcs"
        for a, b in pairs:
            hp = (a - 0.5) / int(row["home_team_population"])
            ap = (b - 0.5) / int(row["away_team_population"])
            if cross and row["home_subdivision"] == "fcs":
                hp, ap = ap, hp
            vals = [
                hp,
                ap,
                float(row["margin"]),
                row["pairing"],
                1.0 - bool(row["neutral_site"]),
                bool(row["neutral_site"]),
                1 / n,
                int(row["game_id"]),
                float(row["home_subdivision"] == "fbs")
                if cross and not bool(row["neutral_site"])
                else 0.0,
                float(row["season"]),
            ]
            for out, val in zip(values, vals):
                out.append(val)
    return tuple(np.asarray(v) for v in values)


def evaluate(model, X, y, pairing, game_ids):
    loc = X @ model["beta"]
    scores = game_log_scores(y, loc, game_ids, model["scale"], model["df"])
    actual = np.array([y[game_ids == g][0] for g in np.unique(game_ids)])
    pred = np.array([np.mean(loc[game_ids == g]) for g in np.unique(game_ids)])
    interval = student_t.ppf(0.90, model["df"]) * model["scale"]
    out = {
        **scores,
        "n_pseudo_observations": len(y),
        "mae_expected_margin": float(np.mean(np.abs(actual - pred))),
        "central_80pct_coverage": float(np.mean(np.abs(actual - pred) <= interval)),
    }
    for p in PAIRINGS:
        m = pairing == p
        if m.any():
            out[p] = evaluate(model, X[m], y[m], pairing[m], game_ids[m])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    observations = read_csv(ROOT / "data/processed/massey/final_observations.csv")
    games = read_csv(ROOT / "data/processed/cfbd/games.csv")
    stats = read_csv(ROOT / "data/processed/cfbd/team_game_stats.csv")
    depth = coverage_table(observations)
    distributions, by_team = build_distributions(observations, depth, args.threshold)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "system_depth.csv", list(depth[0]), depth)
    write_csv(
        OUT / "team_season_rank_distributions.csv",
        list(distributions[0]),
        distributions,
    )
    selected = defaultdict(set)
    for item in depth:
        if float(item["coverage_ratio"]) >= args.threshold:
            selected[(int(item["season"]), item["subdivision"])].add(
                item["system_code"]
            )
    rank_pairs = {}
    for g in games:
        if g["homeClassification"] not in {"fbs", "fcs"} or g[
            "awayClassification"
        ] not in {"fbs", "fcs"}:
            continue
        hk = (int(g["season"]), g["homeClassification"], g["homeId"])
        ak = (int(g["season"]), g["awayClassification"], g["awayId"])
        h = dict(by_team.get(hk, []))
        a = dict(by_team.get(ak, []))
        common = sorted(set(h) & set(a))
        if common:
            rank_pairs[(int(g["id"]), hk, ak)] = [(h[c], a[c]) for c in common]
    joined, missing = build_joined(games, distributions, rank_pairs, stats)
    write_csv(OUT / "historical_modeling_games.csv", list(joined[0]), joined)
    x, y, z, pair, home, neutral, weights, game_ids, _ypp, fbs_home = pseudo_data(
        joined
    )
    seasons = {int(r["game_id"]): int(r["season"]) for r in joined}
    train = np.array([seasons[g] < 2022 for g in game_ids])
    X = design_matrix(x, y, pair, home, neutral, True, fbs_home)
    B = design_matrix(x, y, pair, home, neutral, False, fbs_home)
    pseudo = fit_robust_surface(X[train], z[train], weights[train])
    marginal = fit_marginalized(X[train], z[train], game_ids[train], maxiter=12)
    benchmark = fit_robust_surface(B[train], z[train], weights[train])

    def all_metrics(model, mask, matrix=X):
        return evaluate(model, matrix[mask], z[mask], pair[mask], game_ids[mask])

    validation = {
        "modern_holdout_2022_2025": {
            "weighted_pseudo_fit": all_metrics(pseudo, ~train),
            "marginalized_fit": all_metrics(marginal, ~train),
            "linear_benchmark": all_metrics(benchmark, ~train, B),
        }
    }
    for season in sorted(set(seasons.values())):
        mask = np.array([seasons[g] == season for g in game_ids])
        validation.setdefault("season_holdouts", {})[str(season)] = {
            "weighted_pseudo_fit": all_metrics(pseudo, mask),
            "marginalized_fit": all_metrics(marginal, mask),
            "linear_benchmark": all_metrics(benchmark, mask, B),
        }
    result = {
        "specification": {
            "criterion": f"coverage >= {args.threshold:.2f}",
            "student_t_df": 5,
            "pseudo_weight_per_game": 1.0,
            "training_objectives": ["weighted_pseudo", "marginalized"],
            "surface": "same-subdivision odd smooth rank-difference basis; cross-subdivision FBS/FCS-oriented hinge surface; site indicators",
            "basis_features": int(X.shape[1]),
            "design_rank": int(np.linalg.matrix_rank(X[train].T @ X[train])),
            "design_condition_number": float(np.sqrt(np.linalg.cond(X[train].T @ X[train]))),
        },
        "coverage": {
            "usable_games": len(joined),
            "missing_rank_distribution_games": dict(missing),
            "by_pairing": dict(Counter(r["pairing"] for r in joined)),
            "same_system_games": sum(
                r["pairing_method"] == "same_system" for r in joined
            ),
        },
        "validation": validation,
        "models": {
            "weighted_pseudo": {"scale": pseudo["scale"], "df": pseudo["df"]},
            "marginalized": {"scale": marginal["scale"], "df": marginal["df"]},
        },
    }
    (OUT / "margin_model_results.json").write_text(json.dumps(result, indent=2) + "\n")
    report = {
        "chosen_threshold": args.threshold,
        "systems": len(depth),
        "surviving_system_groups": sum(
            float(d["coverage_ratio"]) >= args.threshold for d in depth
        ),
        "team_seasons": len(distributions),
        "usable_games": len(joined),
        "pseudo_observations": len(z),
        "validation_definition": "equal-weight whole-game metrics; no season is split",
        "results_file": "margin_model_results.json",
    }
    (OUT / "historical_modeling_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    h = validation["modern_holdout_2022_2025"]
    (OUT / "historical_modeling_report.md").write_text(
        f"# Historical modeling report\n\nThis report was regenerated with per-game scoring. CMP is excluded and MAS is treated as a constituent.\n\n- Usable games: {len(joined)}\n- Rank-pair pseudo-observations: {len(z)} (descriptive, not independent games)\n- Weighted pseudo fit marginalized NLL: {h['weighted_pseudo_fit']['marginalized_nll']:.3f}\n- Direct marginalized fit NLL: {h['marginalized_fit']['marginalized_nll']:.3f}\n- Linear benchmark marginalized NLL: {h['linear_benchmark']['marginalized_nll']:.3f}\n- Pseudo fit expected conditional NLL: {h['weighted_pseudo_fit']['expected_conditional_nll']:.3f}\n- Pseudo fit MAE: {h['weighted_pseudo_fit']['mae_expected_margin']:.3f}\n- Central 80% coverage: {h['weighted_pseudo_fit']['central_80pct_coverage']:.3f}\n- Basis: {X.shape[1]} pairing-block features, rank {np.linalg.matrix_rank(X[train].T @ X[train])}, condition number {np.sqrt(np.linalg.cond(X[train].T @ X[train])):.2g}\n- Student-t: df=5, scale={pseudo['scale']:.3f}; retained as a heavy-tailed baseline\n\nThe rich model and simple benchmark use the same oriented coordinates and equal-game scoring. See `margin_model_results.json` for pairing and season breakdowns, both training objectives, and fit diagnostics.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "team_seasons": len(distributions),
                "usable_games": len(joined),
                "holdout_marginalized_nll": h["weighted_pseudo_fit"][
                    "marginalized_nll"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
