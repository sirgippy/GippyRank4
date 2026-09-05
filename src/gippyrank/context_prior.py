"""Safe, explicitly dated inputs for the preseason Context Prior.

This module deliberately contains no statistical model selection.  It turns a
dated coaching-tenure source into the small set of factual observations that
were knowable at a requested preseason cutoff.  Keeping this boundary separate
from the fit prevents a convenient current-season endpoint from silently
becoming a model input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

PRODUCTION_SAFE_BY_CONSTRUCTION = "production-safe-by-construction"
PRODUCTION_SAFE_BY_SEMANTICS = "production-safe-by-semantic-definition"
PRODUCTION_SAFE_WITH_CAVEAT = "production-safe-with-retrospective-stability-caveat"
EXPLORATORY = "exploratory-unresolved"
REJECTED = "rejected"


@dataclass(frozen=True)
class ModelSpecification:
    """Durable methodology, deliberately independent of one annual fit."""

    model_family: str
    spec_version: str
    features: tuple[str, ...]
    distribution_family: str
    penalty: float
    quadrature: str

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnnualFittedInstance:
    """One refit of a specification for a named preseason target."""

    model_family: str
    spec_version: str
    trained_through_season: int
    target_season: int
    context_as_of: str | None = None

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceRow:
    """A target-season input deliberately incapable of carrying an outcome."""

    season: int
    subdivision: str
    team_id: str
    team_name: str
    population: int
    lag1_z: tuple[float, ...] | None
    lag_zs: tuple[tuple[float, ...], ...]
    features: dict[str, float | None]
    cold_start_reason: str | None = None

    def require_no_target(self) -> None:
        """Make the no-outcome contract explicit at the inference boundary."""
        forbidden = {"target_z", "target_ranks", "outcome", "final_rank"}
        if forbidden & set(self.features):
            raise ValueError(
                "inference features must not contain a target-season outcome"
            )


@dataclass(frozen=True)
class CoachContext:
    """A cutoff-safe coaching observation, or an explicit reason it is absent."""

    coach_id: int | None
    coach_name: str | None
    tenure_seasons: float | None
    known_by_cutoff: bool
    unavailable_reason: str | None

    def features(self, prior: CoachContext | None) -> dict[str, float | None]:
        """Return only values whose source state is known at both cutoffs."""
        change = None
        if self.known_by_cutoff and prior is not None and prior.known_by_cutoff:
            change = float(self.coach_id != prior.coach_id)
        return {
            "coach_change": change,
            "coach_tenure_seasons": self.tenure_seasons
            if self.known_by_cutoff
            else None,
        }


def parse_cutoff(value: str) -> date:
    """Reject an ambiguous cutoff rather than defaulting to a calendar date."""
    return date.fromisoformat(value)


def _as_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def coach_at_cutoff(
    tenures: list[dict[str, Any]], team: str, target_season: int, context_as_of: str
) -> CoachContext:
    """Find a continuous tenure proven active before ``context_as_of``.

    ``endYear == target_season`` is intentionally ambiguous without an effective
    end date: a midseason firing would otherwise leak an after-kickoff fact.
    Similarly, a tenure beginning in the target season requires a dated hire at
    or before the cutoff.  These conservative omissions are preferable to
    treating a season-attributed coaching record as a preseason snapshot.
    """
    cutoff = parse_cutoff(context_as_of)
    candidates = [
        tenure
        for tenure in tenures
        if (tenure.get("team") or {}).get("school") == team
        and int(tenure.get("startYear") or 10**9) <= target_season
        and (tenure.get("endYear") is None or int(tenure["endYear"]) >= target_season)
    ]
    if not candidates:
        return CoachContext(None, None, None, False, "no_matching_tenure")
    if len(candidates) != 1:
        return CoachContext(None, None, None, False, "ambiguous_overlapping_tenures")
    tenure = candidates[0]
    start_year = int(tenure["startYear"])
    end_year = tenure.get("endYear")
    hire_date = _as_date(tenure.get("hireDate")) or _as_date(
        tenure.get("effectiveStart")
    )
    effective_end = _as_date(tenure.get("effectiveEnd"))
    if effective_end is not None and effective_end <= cutoff:
        return CoachContext(None, None, None, False, "tenure_ended_by_cutoff")
    if (
        end_year is not None
        and int(end_year) == target_season
        and effective_end is None
    ):
        return CoachContext(None, None, None, False, "target_season_end_is_undated")
    if start_year == target_season and (hire_date is None or hire_date > cutoff):
        return CoachContext(
            None, None, None, False, "target_season_start_not_dated_by_cutoff"
        )
    coach = tenure.get("coach") or {}
    coach_id = coach.get("id")
    if coach_id is None:
        return CoachContext(None, None, None, False, "missing_canonical_coach_id")
    name = " ".join(
        str(value) for value in (coach.get("firstName"), coach.get("lastName")) if value
    )
    return CoachContext(
        int(coach_id),
        name or None,
        float(target_season - start_year + 1),
        True,
        None,
    )


def only_approved(
    features: dict[str, float | None], approved: set[str]
) -> dict[str, float | None]:
    """Fail closed: an unapproved feature never reaches a production fit."""
    unknown = set(features) - approved
    if unknown:
        raise ValueError(f"unapproved context features requested: {sorted(unknown)}")
    return dict(features)
