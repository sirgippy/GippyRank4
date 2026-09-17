"""Leakage-safe year-over-year roster-talent features for research studies.

The feature in this module is deliberately a change in *season-relative*
roster talent.  Team Talent's raw scale drifts across seasons, so the current
and previous values are standardized within their respective seasons before
they are differenced.  The helper accepts only source feature values and
``TeamSeason`` keys; it has no access to target ranks or game outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np

from gippyrank.preseason import TeamSeason

TeamSeasonKey = tuple[int, str, str]


def season_talent_statistics(
    values: Mapping[TeamSeasonKey, float | None],
) -> dict[tuple[int, str], tuple[float, float]]:
    """Return season/subdivision means and standard deviations for talent.

    The population standard deviation is used because the feature describes
    the observed season universe rather than estimating a sample variance.
    A one-team or constant season receives a unit scale, making its
    standardized value zero instead of creating an unstable division.
    """

    grouped: dict[tuple[int, str], list[float]] = {}
    for (season, subdivision, _team_id), value in values.items():
        if value is not None and np.isfinite(value):
            grouped.setdefault((season, subdivision), []).append(float(value))
    return {
        key: (float(np.mean(observed)), max(float(np.std(observed)), 1e-8))
        for key, observed in grouped.items()
    }


def add_seasonal_talent_delta(
    rows: Iterable[TeamSeason],
    talent_values: Mapping[TeamSeasonKey, float | None],
    *,
    feature_name: str = "talent_delta",
) -> list[TeamSeason]:
    """Add season-relative current-minus-previous talent to ``rows``.

    For a target ``(season, subdivision, team_id)``, the value is

    ``z(talent_current within season) - z(talent_previous within season - 1)``.

    If either source value or either season's normalization statistics is
    unavailable, the feature is ``None``.  Downstream production-style
    preprocessing can then apply its existing training-only imputation and
    missingness indicator behavior.  The input mapping may contain future
    seasons, but callers should pass only the feature-source coverage intended
    for the study; no outcome data is used here.
    """

    materialized = list(rows)
    statistics = season_talent_statistics(talent_values)
    result: list[TeamSeason] = []
    for row in materialized:
        current_key = (row.season, row.subdivision, row.team_id)
        previous_key = (row.season - 1, row.subdivision, row.team_id)
        current = talent_values.get(current_key)
        previous = talent_values.get(previous_key)
        current_stats = statistics.get((row.season, row.subdivision))
        previous_stats = statistics.get((row.season - 1, row.subdivision))
        delta: float | None = None
        if (
            current is not None
            and previous is not None
            and current_stats is not None
            and previous_stats is not None
        ):
            current_mean, current_scale = current_stats
            previous_mean, previous_scale = previous_stats
            delta = (float(current) - current_mean) / current_scale - (
                float(previous) - previous_mean
            ) / previous_scale
        features = dict(row.features)
        features[feature_name] = delta
        result.append(replace(row, features=features))
    return result
