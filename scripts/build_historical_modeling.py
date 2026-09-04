"""Build and evaluate the historical score-margin likelihood."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gippyrank.modeling import (
    PAIRINGS,
    build_distributions,
    coverage_table,
    design_matrix,
    fit_marginalized,
    fit_robust_surface,
    game_log_scores,
    mixture_central_interval,
    read_csv,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/modeling"


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


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
                "margin": (
                    int(game["homePoints"]) - int(game["awayPoints"])
                    if hc == ac
                    else (
                        (
                            int(game["homePoints"])
                            if hc == "fbs"
                            else int(game["awayPoints"])
                        )
                        - (
                            int(game["awayPoints"])
                            if hc == "fbs"
                            else int(game["homePoints"])
                        )
                    )
                ),
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


@dataclass(frozen=True)
class PseudoData:
    x: np.ndarray
    y: np.ndarray
    margin: np.ndarray
    pairing: np.ndarray
    home: np.ndarray
    neutral: np.ndarray
    weight: np.ndarray
    game_id: np.ndarray
    fbs_home: np.ndarray
    season: np.ndarray


def pseudo_data(rows) -> PseudoData:
    values = [[] for _ in range(10)]
    for row in rows:
        pairs = json.loads(row["rank_pairs"])
        n = len(pairs)
        cross = row["pairing"] == "fbs-fcs"
        neutral = _as_bool(row["neutral_site"])
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
                1.0 - neutral,
                neutral,
                1 / n,
                int(row["game_id"]),
                float(row["home_subdivision"] == "fbs")
                if cross and not neutral
                else 0.0,
                float(row["season"]),
            ]
            for out, val in zip(values, vals):
                out.append(val)
    arrays = tuple(np.asarray(v) for v in values)
    return PseudoData(*arrays)


def _evaluate_subset(model, X, y, game_ids, include_interval=True):
    loc = X @ model["beta"]
    scores = game_log_scores(y, loc, game_ids, model["scale"], model["df"])
    _, inverse = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(inverse)
    actual = np.bincount(inverse, weights=y) / counts
    pred = np.bincount(inverse, weights=loc) / counts
    covered = []
    for indexes in np.split(np.argsort(inverse), np.cumsum(counts)[:-1]):
        if include_interval:
            lower, upper = mixture_central_interval(
                loc[indexes], model["scale"], model["df"]
            )
            covered.append(lower <= actual[len(covered)] <= upper)
    out = {
        **scores,
        "n_pseudo_observations": len(y),
        "mae_expected_margin": float(np.mean(np.abs(actual - pred))),
        "central_80pct_coverage": float(np.mean(covered)) if include_interval else None,
    }
    return out


def evaluate(model, X, y, pairing, game_ids, include_interval=True):
    out = _evaluate_subset(model, X, y, game_ids, include_interval)
    for p in PAIRINGS:
        m = pairing == p
        if m.any():
            out[p] = _evaluate_subset(model, X[m], y[m], game_ids[m], include_interval)
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
    data = pseudo_data(joined)
    seasons = {int(r["game_id"]): int(r["season"]) for r in joined}
    train = data.season < 2022
    X = design_matrix(
        data.x, data.y, data.pairing, data.home, data.neutral, True, data.fbs_home
    )
    B = design_matrix(
        data.x, data.y, data.pairing, data.home, data.neutral, False, data.fbs_home
    )
    df_grid = (3.0, 5.0, 8.0, 15.0)
    df_fits = {
        df: fit_robust_surface(X[train], data.margin[train], data.weight[train], df=df)
        for df in df_grid
    }
    df_scores = {
        str(df): game_log_scores(
            data.margin[~train],
            X[~train] @ fit["beta"],
            data.game_id[~train],
            fit["scale"],
            fit["df"],
        )["marginalized_nll"]
        for df, fit in df_fits.items()
    }
    selected_df = min(df_grid, key=lambda df: df_scores[str(df)])
    pseudo = df_fits[selected_df]
    marginal = fit_marginalized(
        X[train], data.margin[train], data.game_id[train], df=selected_df
    )
    if not marginal["optimizer_success"]:
        raise RuntimeError(
            f"marginalized optimizer did not converge: {marginal['optimizer_message']}"
        )
    benchmark = fit_robust_surface(
        B[train], data.margin[train], data.weight[train], df=selected_df
    )

    def all_metrics(model, mask, matrix=X, include_interval=True):
        return evaluate(
            model,
            matrix[mask],
            data.margin[mask],
            data.pairing[mask],
            data.game_id[mask],
            include_interval,
        )

    validation = {
        "modern_holdout_2022_2025": {
            "weighted_pseudo_fit": all_metrics(pseudo, ~train),
            "marginalized_fit": all_metrics(marginal, ~train),
            "linear_benchmark": all_metrics(benchmark, ~train, B),
        }
    }
    cross = data.pairing == "fbs-fcs"
    site_masks = {
        "fbs_home": ~train & cross & (data.fbs_home == 1),
        "fcs_home": ~train & cross & (data.fbs_home == 0) & ~data.neutral,
        "neutral": ~train & cross & data.neutral,
    }
    validation["modern_holdout_2022_2025"]["fbs_fcs_site_context"] = {
        name: all_metrics(pseudo, mask)
        for name, mask in site_masks.items()
        if mask.any()
    }
    for season in sorted(set(seasons.values())):
        mask = np.array([seasons[g] == season for g in data.game_id])
        validation.setdefault("season_in_sample_diagnostics", {})[str(season)] = {
            "weighted_pseudo_fit": all_metrics(pseudo, mask, include_interval=False),
            "marginalized_fit": all_metrics(marginal, mask, include_interval=False),
            "linear_benchmark": all_metrics(benchmark, mask, B, include_interval=False),
        }
    result = {
        "specification": {
            "criterion": f"coverage >= {args.threshold:.2f}",
            "student_t_df": selected_df,
            "student_t_df_sensitivity": df_scores,
            "pseudo_weight_per_game": 1.0,
            "training_objectives": ["weighted_pseudo", "marginalized"],
            "surface": "same-subdivision odd smooth rank-difference basis; cross-subdivision FBS/FCS-oriented hinge surface; site indicators",
            "basis_features": int(X.shape[1]),
            "design_rank": int(np.linalg.matrix_rank(X[train].T @ X[train])),
            "design_condition_number": float(
                np.sqrt(np.linalg.cond(X[train].T @ X[train]))
            ),
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
            "marginalized": {
                "scale": marginal["scale"],
                "df": marginal["df"],
                "optimizer_success": marginal["optimizer_success"],
                "optimizer_message": marginal["optimizer_message"],
                "optimizer_status": marginal["optimizer_status"],
                "optimizer_nit": marginal["optimizer_nit"],
                "optimizer_nfev": marginal["optimizer_nfev"],
            },
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
        "pseudo_observations": len(data.margin),
        "validation_definition": "equal-weight whole-game metrics; no season is split",
        "results_file": "margin_model_results.json",
    }
    (OUT / "historical_modeling_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    h = validation["modern_holdout_2022_2025"]
    (OUT / "historical_modeling_report.md").write_text(
        f"# Historical modeling report\n\nThis report was regenerated with per-game scoring. CMP is excluded and MAS is treated as a constituent.\n\n- Usable games: {len(joined)}\n- Rank-pair pseudo-observations: {len(data.margin)} (descriptive, not independent games)\n- Weighted pseudo fit marginalized NLL: {h['weighted_pseudo_fit']['marginalized_nll']:.3f}\n- Direct marginalized fit NLL: {h['marginalized_fit']['marginalized_nll']:.3f}\n- Linear benchmark marginalized NLL: {h['linear_benchmark']['marginalized_nll']:.3f}\n- Pseudo fit expected conditional NLL: {h['weighted_pseudo_fit']['expected_conditional_nll']:.3f}\n- Pseudo fit MAE: {h['weighted_pseudo_fit']['mae_expected_margin']:.3f}\n- Central 80% mixture coverage: {h['weighted_pseudo_fit']['central_80pct_coverage']:.3f}\n- Basis: {X.shape[1]} pairing-block features, rank {np.linalg.matrix_rank(X[train].T @ X[train])}, condition number {np.sqrt(np.linalg.cond(X[train].T @ X[train])):.2g}\n- Student-t df sensitivity: {df_scores}; selected df={selected_df}; scale={pseudo['scale']:.3f}\n- Marginal optimizer: success={marginal['optimizer_success']}, status={marginal['optimizer_status']}, iterations={marginal['optimizer_nit']}, evaluations={marginal['optimizer_nfev']}\n\nThe rich model and simple benchmark use the same oriented coordinates and equal-game scoring. Training-era season values are in-sample diagnostics, not holdouts. See `margin_model_results.json` for pairing and season breakdowns.\n",
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
