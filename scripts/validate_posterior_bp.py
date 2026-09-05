"""Run a deterministic exact-enumeration stress audit of loopy BP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.validation import exact_marginals

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260905
CASE_COUNT = 72
# Predeclared before running: this deliberately varied small-graph suite must
# have p95 marginal TV <= 0.05 and worst marginal TV <= 0.15 to support use of
# loopy BP for routine snapshots. Exact tree cases are expected to be numeric.
P95_TV_LIMIT = 0.05
WORST_TV_LIMIT = 0.15


def likelihood() -> LikelihoodV1:
    beta = np.zeros(34)
    beta[0] = -24.0
    beta[10], beta[11] = -20.0, 20.0
    return LikelihoodV1(beta, scale=10.0, degrees_of_freedom=15.0)


def make_case(rng: np.random.Generator, case_id: int) -> tuple[list[Team], list[Game]]:
    team_count = 2 + case_id % 4
    ranks = 2 + (case_id // 4) % 3
    mixed = case_id % 6 == 0 and team_count >= 3
    teams = []
    for index in range(team_count):
        subdivision = "fcs" if mixed and index == team_count - 1 else "fbs"
        teams.append(
            Team(
                f"t{index}",
                f"Team {index}",
                subdivision,
                rng.dirichlet(np.full(ranks, 0.35 + (index % 3))),
            )
        )
    # A connected tree plus deterministic extra edges gives trees, cycles, and
    # denser graphs. Every ninth case includes a repeated pairing.
    pairs = [(index - 1, index) for index in range(1, team_count)]
    candidates = [
        (left, right)
        for left in range(team_count)
        for right in range(left + 1, team_count)
        if (left, right) not in pairs
    ]
    extra = min(len(candidates), case_id % (len(candidates) + 1))
    if extra:
        pairs.extend(
            rng.choice(np.asarray(candidates), size=extra, replace=False).tolist()
        )
    if case_id % 9 == 0:
        pairs.append(pairs[0])
    games = []
    for index, (left, right) in enumerate(pairs):
        home, away = (left, right) if (case_id + index) % 2 == 0 else (right, left)
        margin = int(rng.choice(np.array([-28, -14, -3, 3, 14, 28])))
        games.append(
            Game(
                f"case-{case_id}-game-{index}",
                f"t{home}",
                f"t{away}",
                teams[home].subdivision,
                teams[away].subdivision,
                28 + max(margin, 0),
                28 + max(-margin, 0),
                neutral_site=index % 5 == 0,
            )
        )
    return teams, games


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases, all_tv, all_expected_error = [], [], []
    for case_id in range(CASE_COUNT):
        teams, games = make_case(rng, case_id)
        exact = exact_marginals(teams, games, likelihood())
        approximate = infer_posterior(teams, games, likelihood(), tolerance=1e-9)
        tv = []
        expected_error = []
        for team in teams:
            ranks = np.arange(1, len(team.prior) + 1)
            tv.append(
                float(
                    0.5
                    * np.abs(exact[team.team_id] - approximate.pmfs[team.team_id]).sum()
                )
            )
            expected_error.append(
                float(
                    abs(
                        np.dot(ranks, exact[team.team_id])
                        - np.dot(ranks, approximate.pmfs[team.team_id])
                    )
                )
            )
        all_tv.extend(tv)
        all_expected_error.extend(expected_error)
        cases.append(
            {
                "case_id": case_id,
                "teams": len(teams),
                "raw_edges": len(games),
                "unique_pairwise_factors": approximate.unique_pair_factor_count,
                "mixed_fbs_fcs": any(team.subdivision == "fcs" for team in teams),
                "converged": approximate.converged,
                "iterations": approximate.iterations,
                "median_marginal_tv": float(np.median(tv)),
                "worst_marginal_tv": float(np.max(tv)),
                "median_expected_rank_error": float(np.median(expected_error)),
                "worst_expected_rank_error": float(np.max(expected_error)),
            }
        )
    summary = {
        "case_count": CASE_COUNT,
        "seed": SEED,
        "convergence_rate": float(np.mean([case["converged"] for case in cases])),
        "median_marginal_tv": float(np.median(all_tv)),
        "p90_marginal_tv": float(np.quantile(all_tv, 0.90)),
        "p95_marginal_tv": float(np.quantile(all_tv, 0.95)),
        "worst_marginal_tv": float(np.max(all_tv)),
        "median_expected_rank_error": float(np.median(all_expected_error)),
        "worst_expected_rank_error": float(np.max(all_expected_error)),
        "predeclared_acceptance": {
            "p95_marginal_tv_lte": P95_TV_LIMIT,
            "worst_marginal_tv_lte": WORST_TV_LIMIT,
        },
        "accepted": bool(
            np.quantile(all_tv, 0.95) <= P95_TV_LIMIT
            and np.max(all_tv) <= WORST_TV_LIMIT
        ),
    }
    output = ROOT / "data/processed/posterior_validation/bp_exact_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "specification": "72 deterministic 2-5 team cases; exact joint enumeration versus consolidated-factor loopy BP",
                "summary": summary,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)
    if not summary["accepted"]:
        raise SystemExit("BP exact-validation acceptance threshold failed")


if __name__ == "__main__":
    main()
