"""Leakage-safe utilities for observed-coverage context ablations.

These helpers are deliberately research-only.  They make the two important
contracts explicit: coverage is determined from raw values before imputation,
and a comparison can only be paired after its target keys have been checked.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gippyrank.preseason import Preprocessor, TeamSeason

Key = tuple[int, str, str]


def observed(row: TeamSeason, features: list[str]) -> bool:
    """Whether every requested raw feature was genuinely present."""
    return all(row.features.get(feature) is not None for feature in features)


def restrict_observed(rows: list[TeamSeason], features: list[str]) -> list[TeamSeason]:
    """Restrict before model preprocessing can impute absent source values."""
    return [row for row in rows if observed(row, features)]


def target_keys(rows: list[TeamSeason]) -> set[Key]:
    return {(row.season, row.subdivision, row.team_id) for row in rows}


def assert_same_population(left: list[TeamSeason], right: list[TeamSeason]) -> None:
    """Reject an ablation whose target populations differ."""
    if target_keys(left) != target_keys(right):
        raise ValueError("ablation reference and candidate must have identical keys")


def training_rows(rows: list[TeamSeason], target_season: int) -> list[TeamSeason]:
    """Return only eligible completed seasons for a rolling target."""
    result = [row for row in rows if row.season < target_season]
    if any(row.season >= target_season for row in result):
        raise ValueError("target-season outcomes entered ablation training")
    return result


def standardized_interaction_rows(
    train: list[TeamSeason],
    target: list[TeamSeason],
    left: str,
    right: str,
    *,
    name: str | None = None,
) -> tuple[list[TeamSeason], list[TeamSeason], str]:
    """Add a training-standardized product to train and target rows.

    The training preprocessor is fit once, before the product is formed.  It
    is then reused for target rows, so neither target values nor target outcomes
    can affect standardization.  Raw coverage restrictions normally mean the
    imputation path is unused, but it remains deterministic for diagnostics.
    """
    if not train:
        raise ValueError("interaction requires non-empty training rows")
    target_season = target[0].season if target else None
    if target_season is not None and any(row.season >= target_season for row in train):
        raise ValueError("interaction training includes target/future rows")
    interaction = name or f"interaction__{left}__{right}"
    prep = Preprocessor.fit([row.features for row in train], [left, right])

    def enrich(rows: list[TeamSeason]) -> list[TeamSeason]:
        values = prep.transform([row.features for row in rows])
        result = []
        for row, value in zip(rows, values, strict=True):
            features = dict(row.features)
            features[interaction] = float(value[0] * value[1])
            result.append(replace(row, features=features))
        return result

    return enrich(train), enrich(target), interaction


def training_only_impute_rows(
    train: list[TeamSeason], target: list[TeamSeason], features: list[str]
) -> tuple[list[TeamSeason], list[TeamSeason]]:
    """Impute raw values using training medians, leaving no absence indicators.

    This supports a diagnostic that separates the source's reporting pattern
    from the numerical feature itself; it is not a production preprocessing
    replacement.
    """
    prep = Preprocessor.fit([row.features for row in train], features)

    def apply(rows: list[TeamSeason]) -> list[TeamSeason]:
        result = []
        for row in rows:
            values = dict(row.features)
            for feature in features:
                if values.get(feature) is None:
                    values[feature] = prep.medians[feature]
            result.append(replace(row, features=values))
        return result

    return apply(train), apply(target)


def aggregate_difference(values: list[float]) -> float:
    """A small, testable aggregation used by stored per-team losses."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("ablation loss differences must be finite")
    return float(array.mean()) if len(array) else float("nan")
