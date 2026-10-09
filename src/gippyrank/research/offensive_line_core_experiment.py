"""Research-only experience-selected five-player offensive-line core.

Target OL membership is a retrospective CFBD observation. Prior experience
uses observed roster seasons at any college program; shared experience uses
only prior membership at the target program. Neither uses target-year outcomes.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gippyrank.research.offensive_line_continuity_experiment import (
    HISTORY_START_SEASON,
    LOOKBACK_SEASONS,
    player_identity,
)

CORE_INDIVIDUAL = "ol_core_mean_prior_college_seasons_4y"
CORE_SHARED = "ol_core_mean_shared_same_program_seasons_4y"
CORE_TALENT_INTERACTION = "ol_core_shared_same_program_seasons_4y_x_talent_composite"
CORE_SIZE = 5


def observed_college_seasons(
    roster_rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[int]]:
    """Count each observed college season once, including multi-program years."""
    seasons: dict[str, set[int]] = defaultdict(set)
    for row in roster_rows:
        identity = player_identity(row)
        if not identity:
            continue
        try:
            season = int(row["season"])
        except (KeyError, TypeError, ValueError):
            continue
        seasons[identity].add(season)
    return dict(seasons)


def core_values_for_pool(
    *,
    season: int,
    team_id: str,
    target_player_ids: Iterable[str],
    college_seasons: Mapping[str, set[int]],
    program_memberships: Mapping[tuple[str, int], set[str]],
    history_window_observed: bool = True,
) -> dict[str, float | int | None]:
    """Average both features across every experience-valid five-player core."""
    ids = {
        str(value).strip()
        for value in target_player_ids
        if value is not None and str(value).strip()
    }
    result: dict[str, float | int | None] = {
        CORE_INDIVIDUAL: None,
        CORE_SHARED: None,
        "core_tied_realizations": 0,
        "core_boundary_tied_players": 0,
        "core_players_strictly_above_boundary": 0,
    }
    if (
        not history_window_observed
        or season - LOOKBACK_SEASONS < HISTORY_START_SEASON
        or len(ids) < CORE_SIZE
    ):
        return result

    years = range(season - LOOKBACK_SEASONS, season)
    experience = {
        identity: len(college_seasons.get(identity, set()).intersection(years))
        for identity in ids
    }
    threshold = sorted(experience.values(), reverse=True)[CORE_SIZE - 1]
    above = {identity for identity, count in experience.items() if count > threshold}
    boundary = {
        identity for identity, count in experience.items() if count == threshold
    }
    needed = CORE_SIZE - len(above)
    result["core_boundary_tied_players"] = len(boundary)
    result["core_players_strictly_above_boundary"] = len(above)
    result["core_tied_realizations"] = math.comb(len(boundary), needed)

    same_program_years = {
        identity: {
            year
            for year in years
            if identity in program_memberships.get((team_id, year), set())
        }
        for identity in above | boundary
    }
    individual_values: list[float] = []
    shared_values: list[float] = []
    for chosen in itertools.combinations(sorted(boundary), needed):
        core = tuple(sorted(above | set(chosen)))
        individual_values.append(
            math.fsum(experience[identity] for identity in core) / 5
        )
        shared_values.append(
            math.fsum(
                len(same_program_years[left] & same_program_years[right])
                for left, right in itertools.combinations(core, 2)
            )
            / 10
        )
    # Sorting before fsum makes the result bitwise stable under source-row order.
    result[CORE_INDIVIDUAL] = math.fsum(sorted(individual_values)) / len(
        individual_values
    )
    result[CORE_SHARED] = math.fsum(sorted(shared_values)) / len(shared_values)
    return result


def build_core_panel(
    *,
    prior_panel: Sequence[Mapping[str, Any]],
    ol_rows: Iterable[Mapping[str, Any]],
    college_seasons: Mapping[str, set[int]],
    program_memberships: Mapping[tuple[str, int], set[str]],
) -> list[dict[str, Any]]:
    """Keep the prior experiment's coverage diagnostics and add core measures."""
    target_ids: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in ol_rows:
        if str(row.get("source_classification") or "").lower() != "fbs":
            continue
        identity = player_identity(row)
        if identity:
            target_ids[(int(row["season"]), str(row["team_id"]))].add(identity)

    output: list[dict[str, Any]] = []
    for prior in prior_panel:
        season = int(prior["season"])
        team_id = str(prior["team_id"])
        ids = target_ids.get((season, team_id), set())
        prior_status = str(prior["feature_status"])
        eligible = prior_status == "observed" and season >= 2013
        values = core_values_for_pool(
            season=season,
            team_id=team_id,
            target_player_ids=ids,
            college_seasons=college_seasons,
            program_memberships=program_memberships,
            history_window_observed=eligible,
        )
        status = (
            "prior_panel_unavailable"
            if not eligible
            else "fewer_than_five_usable_identities"
            if len(ids) < CORE_SIZE
            else "observed"
        )
        output.append(
            {
                **prior,
                "usable_target_ol_identities": len(ids),
                "core_feature_status": status,
                **values,
            }
        )
    return output
