"""Outcome-free positive-net moderation of a fitted Context location center."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gippyrank.context_prior_v1_3 import H_FEATURES, LOCATION_FEATURE_NAMES

ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
LOCATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class LocationParts:
    """Only rolling-origin fitted location inputs; no evaluated outcome fields."""

    intercept: float
    history_derived_subtotal: float
    context_only_subtotal: float
    conditional_location_points: np.ndarray

    def __post_init__(self) -> None:
        scalars = (
            self.intercept,
            self.history_derived_subtotal,
            self.context_only_subtotal,
        )
        points = np.asarray(self.conditional_location_points, dtype=float)
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("location components must be finite")
        if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
            raise ValueError("conditional location points must be a finite vector")
        expected = sum(scalars)
        if not np.isclose(points.mean(), expected, rtol=0, atol=LOCATION_TOLERANCE):
            raise ValueError("fitted location parts do not reconstruct the center")
        object.__setattr__(self, "conditional_location_points", points.copy())


def parts_from_fitted_contributions(
    row: Mapping[str, str], conditional_location_points: np.ndarray
) -> LocationParts:
    """Read only fitted, preseason-available terms from a diagnostic row."""
    history = float(row["lag1_contribution"])
    context_only = 0.0
    for name in LOCATION_FEATURE_NAMES:
        contribution = float(row[f"feature_{name}_contribution"]) + float(
            row[f"feature_{name}_missing_contribution"]
        )
        if name in H_FEATURES:
            history += contribution
        else:
            context_only += contribution
    return LocationParts(
        intercept=float(row["context_location_intercept"]),
        history_derived_subtotal=history,
        context_only_subtotal=context_only,
        conditional_location_points=conditional_location_points,
    )


def moderate_positive_net(value: float, alpha: float) -> float:
    """Keep a nonpositive net term; scale only its positive portion."""
    if not np.isfinite(value) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("net contribution must be finite and alpha within [0, 1]")
    return min(value, 0.0) + alpha * max(value, 0.0)


def moderated_location_points(parts: LocationParts, alpha: float) -> np.ndarray:
    """Shift the fitted mixture as a whole, retaining offsets and residual scale."""
    moderated = moderate_positive_net(parts.context_only_subtotal, alpha)
    points = parts.conditional_location_points + (
        moderated - parts.context_only_subtotal
    )
    expected = parts.intercept + parts.history_derived_subtotal + moderated
    if not np.isclose(points.mean(), expected, rtol=0, atol=LOCATION_TOLERANCE):
        raise ValueError("moderated location does not reconstruct the center")
    return points


def require_rolling_origin(
    season: int,
    context_training_cutoff: int,
    history_training_cutoff: int,
    context_instance: dict[str, object],
    history_instance: dict[str, object],
) -> None:
    """Reject any evaluated-season fit or provenance mismatch."""
    if season not in (2022, 2023, 2024, 2025):
        raise ValueError(f"unplanned evaluation season: {season}")
    through = season - 1
    if context_training_cutoff != through or history_training_cutoff != through:
        raise ValueError(f"{season}: fitted contribution is not rolling-origin")
    for family, instance, version in (
        ("context_prior", context_instance, "1.3"),
        ("history_prior", history_instance, "1.1"),
    ):
        if (
            instance.get("model_family") != family
            or instance.get("spec_version") != version
            or instance.get("target_season") != season
            or instance.get("trained_through_season") != through
        ):
            raise ValueError(f"{season}: {family} origin or version is invalid")
