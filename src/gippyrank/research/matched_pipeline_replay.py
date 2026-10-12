"""Replay matched issue 200 priors against caller-supplied game evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    infer_posterior,
)

Policy = Literal["existing", "pre_bowl"]
PriorFamily = Literal["History", "Context"]


def read_pipeline_bundle(path: Path) -> dict[str, object]:
    """Read and validate the top-level issue 200 bundle contract."""
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("pipeline bundle must contain a JSON object")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported issue 200 pipeline bundle schema")
    if value.get("experiment") != "issue-200-matched-history-context":
        raise ValueError("pipeline bundle belongs to a different experiment")
    return value


def replay_pipeline_bundle(
    bundle: Mapping[str, object],
    *,
    season: int,
    policy: Policy,
    prior_family: PriorFamily,
    games: Sequence[Game],
    fcs_team_names: Mapping[str, str] | None = None,
) -> PosteriorResult:
    """Run posterior inference using one matched prior and likelihood arm.

    ``games`` must already follow the selected policy's evidence inclusion
    rules. FBS priors are read from the bundle; FCS participants receive the
    existing uniform full-subdivision fallback.
    """
    if policy not in {"existing", "pre_bowl"}:
        raise ValueError(f"unsupported issue 200 policy: {policy!r}")
    if prior_family not in {"History", "Context"}:
        raise ValueError(f"unsupported issue 200 prior family: {prior_family!r}")
    target_seasons = bundle.get("target_seasons")
    if not isinstance(target_seasons, list) or season not in target_seasons:
        raise ValueError(f"season {season} is not serialized in this bundle")

    prior_rows = bundle.get("priors")
    likelihood_rows = bundle.get("likelihoods")
    populations = bundle.get("fcs_population_by_season")
    if not isinstance(prior_rows, list) or not isinstance(likelihood_rows, dict):
        raise TypeError("pipeline bundle lacks serialized priors or likelihoods")
    if not isinstance(populations, dict) or str(season) not in populations:
        raise ValueError(f"pipeline bundle lacks the {season} FCS population")

    selected: dict[str, Mapping[str, object]] = {}
    for row in prior_rows:
        if not isinstance(row, dict):
            raise TypeError("serialized prior row is not an object")
        if (
            row.get("season") == season
            and row.get("policy") == policy
            and row.get("model_family") == prior_family
        ):
            team_id = str(row.get("team_id", ""))
            if not team_id or team_id in selected:
                raise ValueError(
                    f"{season}/{policy}/{prior_family}: duplicate or empty team ID"
                )
            selected[team_id] = row
    if not selected:
        raise ValueError(f"{season}/{policy}/{prior_family}: no bundled FBS priors")

    teams: list[Team] = []
    for team_id, row in sorted(
        selected.items(), key=lambda item: _team_id_key(item[0])
    ):
        population = int(row["population"])
        pmf = np.asarray(row["pmf"], dtype=float)
        if (
            pmf.ndim != 1
            or len(pmf) != population
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(pmf.sum(), 1.0, rtol=0.0, atol=1e-8)
        ):
            raise ValueError(f"{season}/{team_id}: invalid serialized FBS prior PMF")
        teams.append(Team(team_id, str(row["team_name"]), "fbs", pmf))

    fcs_population = int(populations[str(season)])
    if fcs_population < 1:
        raise ValueError(f"{season}: invalid FCS rank population")
    fcs_names = dict(fcs_team_names or {})
    for game in games:
        for team_id, subdivision in (
            (game.home_id, game.home_subdivision),
            (game.away_id, game.away_subdivision),
        ):
            if subdivision == "fbs" and team_id not in selected:
                raise ValueError(
                    f"{game.game_id}: FBS game team {team_id} lacks a bundled prior"
                )
            if subdivision == "fcs":
                fcs_names.setdefault(team_id, team_id)
    teams.extend(
        Team(
            team_id,
            name,
            "fcs",
            np.full(fcs_population, 1 / fcs_population, dtype=float),
        )
        for team_id, name in sorted(
            fcs_names.items(), key=lambda item: _team_id_key(item[0])
        )
    )

    raw_likelihood = likelihood_rows.get(policy)
    if not isinstance(raw_likelihood, dict):
        raise TypeError(f"pipeline bundle lacks the {policy} likelihood")
    likelihood = LikelihoodV1(
        beta=np.asarray(raw_likelihood["beta"], dtype=float),
        scale=float(raw_likelihood["scale"]),
        degrees_of_freedom=float(raw_likelihood["degrees_of_freedom"]),
        fit_kind=str(raw_likelihood.get("fit_kind", "weighted_pseudo")),
    )
    configuration = bundle.get("inference_configuration")
    if not isinstance(configuration, dict):
        raise TypeError("pipeline bundle lacks inference configuration")
    return infer_posterior(
        teams,
        list(games),
        likelihood,
        max_iterations=int(configuration["max_iterations"]),
        tolerance=float(configuration["tolerance"]),
        damping=float(configuration["damping"]),
    )


def _team_id_key(team_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(team_id)
    except ValueError:
        return 1, team_id
