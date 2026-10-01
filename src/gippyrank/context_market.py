"""Provenance-aware helpers for model-versus-market retrospective studies.

The functions in this module deliberately keep four questions separate:
market alignment, subsequent market movement, accuracy against the final
margin, and any incremental relationship between a model/market edge and the
result. It supports both a strict pregame-publication study and an explicitly
labelled historical reconstruction. It has no dependency on the ranking
updater, and it never treats a line source's mutable ``spread`` field as a
verified closing line.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

EDGE_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("< 2", 0.0, 2.0),
    ("2–5", 2.0, 5.0),
    ("5–8", 5.0, 8.0),
    ("8+", 8.0, None),
)


class ContextMarketError(ValueError):
    """Raised when a source payload violates a study safety contract."""


@dataclass(frozen=True)
class ModelVariant:
    """An explicitly named prior-family/version comparison track."""

    key: str
    label: str
    prior_family: str
    model_version_field: str
    model_version: str
    anchor_context_version: str | None = None


COMPARISON_VARIANTS: tuple[ModelVariant, ...] = (
    ModelVariant(
        key="history-1.1",
        label="History 1.1",
        prior_family="history",
        model_version_field="history_prior",
        model_version="1.1",
        # The canonical History 1.1 artifacts are paired with Context 1.2.
        # Do not accidentally switch this baseline to a later hybrid rebuild.
        anchor_context_version="1.2",
    ),
    ModelVariant(
        key="context-1.2",
        label="Context 1.2",
        prior_family="context",
        model_version_field="context_prior",
        model_version="1.2",
    ),
    ModelVariant(
        key="context-1.3",
        label="Context 1.3",
        prior_family="context",
        model_version_field="context_prior",
        model_version="1.3",
    ),
)


@dataclass(frozen=True)
class PublicationChoice:
    """One variant/week snapshot decision, including its timing provenance."""

    variant_key: str
    week: int
    snapshot_id: str | None
    publication_slot: str | None
    publication_status: str | None
    snapshot_type: str | None
    generation_timestamp: str | None
    effective_cutoff: str | None
    requested_cutoff: str | None
    reason: str | None
    variant_label: str | None = None
    model_family: str | None = None
    model_version: str | None = None
    model_version_field: str | None = None
    selection_policy: str = "strict_pregame"
    is_reconstruction: bool = False
    is_retrospective_artifact: bool = False
    generated_after_target_week_start: bool | None = None
    timing_classification: str | None = None
    source_retrieved_at: str | None = None
    source_context_snapshot_id: str | None = None
    comparison_snapshot_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderChoice:
    """A deliberately single-book selection for opening-to-later comparisons."""

    provider: str | None
    comparable_game_count: int
    eligible_game_count: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_timestamp(value: object) -> datetime | None:
    """Parse an API timestamp as UTC, returning ``None`` for an absent value."""
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ContextMarketError(f"Invalid ISO timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def context_version(publication: Mapping[str, Any]) -> str | None:
    """Read the Context prior version from current or older API inventory forms."""
    return model_version_value(publication, "context_prior")


def model_version_value(
    publication: Mapping[str, Any], version_field: str
) -> str | None:
    """Read one versioned model component from an inventory row."""
    model_versions = publication.get("model_versions")
    if isinstance(model_versions, Mapping):
        value = model_versions.get(version_field)
        if value is not None:
            return str(value)
    if version_field == "context_prior":
        value = publication.get("prior_model_version")
        return str(value) if value is not None else None
    return None


def is_context_publication(publication: Mapping[str, Any]) -> bool:
    """Whether an inventory row belongs to the predictive Context family."""
    return (
        publication.get("ranking_family") == "predictive"
        and publication.get("prior_family") == "context"
    )


def is_variant_publication(
    publication: Mapping[str, Any], variant: ModelVariant
) -> bool:
    """Whether an inventory row belongs to one explicit comparison variant."""
    if (
        publication.get("ranking_family") != "predictive"
        or publication.get("prior_family") != variant.prior_family
        or model_version_value(publication, variant.model_version_field)
        != variant.model_version
    ):
        return False
    return (
        variant.anchor_context_version is None
        or model_version_value(publication, "context_prior")
        == variant.anchor_context_version
    )


def is_retrospective_publication(publication: Mapping[str, Any]) -> bool:
    """Identify an explicitly marked retrospective replay when metadata exposes it."""
    if publication.get("backfill") or publication.get("backfill_kind"):
        return True
    labels = (
        publication.get("display_label"),
        publication.get("comparison_display_label"),
    )
    return any(
        isinstance(label, str) and "retrospective" in label.casefold()
        for label in labels
    )


def select_model_publication(
    publications: Iterable[Mapping[str, Any]],
    *,
    season: int,
    week: int,
    week_start: datetime,
    variant: ModelVariant,
    selection_policy: str = "strict_pregame",
    allow_retrospective: bool = False,
) -> PublicationChoice:
    """Select a model artifact under a strict or reconstruction policy.

    ``strict_pregame`` requires actual publication before the first kickoff.
    ``reconstruction`` permits an artifact materialized later, but still
    requires a weekly/live effective cutoff before that first kickoff. A
    preseason artifact with no stated cutoff is allowed only as an explicitly
    classified unverified reconstruction.
    """
    if week_start.tzinfo is None:
        raise ContextMarketError("week_start must be timezone-aware")
    if selection_policy not in {"strict_pregame", "reconstruction"}:
        raise ContextMarketError(f"Unknown selection policy: {selection_policy!r}")
    start = week_start.astimezone(UTC)
    matching = [
        publication
        for publication in publications
        if publication.get("season") == season
        and is_variant_publication(publication, variant)
    ]
    if not matching:
        return PublicationChoice(
            variant.key,
            week,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"No {variant.label} publication exists in the inventory.",
            variant_label=variant.label,
            model_family=variant.prior_family,
            model_version=variant.model_version,
            model_version_field=variant.model_version_field,
            selection_policy=selection_policy,
            is_reconstruction=selection_policy == "reconstruction",
        )

    rejected: Counter[str] = Counter()
    candidates: list[tuple[datetime, datetime, str, Mapping[str, Any]]] = []
    for publication in matching:
        retrospective = is_retrospective_publication(publication)
        if (
            retrospective
            and selection_policy == "strict_pregame"
            and not allow_retrospective
        ):
            rejected["explicitly retrospective"] += 1
            continue
        generated = parse_timestamp(publication.get("generation_timestamp"))
        if generated is None:
            rejected["missing generation timestamp"] += 1
            continue
        if selection_policy == "strict_pregame" and generated >= start:
            rejected["generated after target-week start"] += 1
            continue
        snapshot_type = str(publication.get("snapshot_type", ""))
        cutoff = parse_timestamp(publication.get("effective_cutoff"))
        if snapshot_type in {"live", "weekly"} and cutoff is None:
            rejected["missing weekly/live evidence cutoff"] += 1
            continue
        if cutoff is not None and cutoff >= start:
            rejected["evidence cutoff after target-week start"] += 1
            continue
        snapshot_id = publication.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            rejected["missing snapshot id"] += 1
            continue
        candidates.append(
            (
                cutoff or datetime.min.replace(tzinfo=UTC),
                generated,
                snapshot_id,
                publication,
            )
        )

    if not candidates:
        details = "; ".join(
            f"{count} {reason}" for reason, count in sorted(rejected.items())
        )
        reason_prefix = (
            "No legitimate pregame publication"
            if selection_policy == "strict_pregame"
            else "No reconstruction-eligible publication"
        )
        return PublicationChoice(
            variant.key,
            week,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            reason_prefix + ": " + details,
            variant_label=variant.label,
            model_family=variant.prior_family,
            model_version=variant.model_version,
            model_version_field=variant.model_version_field,
            selection_policy=selection_policy,
            is_reconstruction=selection_policy == "reconstruction",
        )

    _, _, _, selected = max(candidates, key=lambda item: item[:3])
    generated = parse_timestamp(selected.get("generation_timestamp"))
    cutoff = parse_timestamp(selected.get("effective_cutoff"))
    generated_after_start = generated is not None and generated >= start
    if selection_policy == "strict_pregame":
        timing_classification = "published_before_target_week_start"
    elif cutoff is None:
        timing_classification = "reconstruction_without_effective_cutoff"
    elif generated_after_start:
        timing_classification = "postgame_generated_reconstruction_with_pregame_cutoff"
    else:
        timing_classification = "pregame_artifact_in_reconstruction_study"
    return PublicationChoice(
        variant_key=variant.key,
        week=week,
        snapshot_id=str(selected["snapshot_id"]),
        publication_slot=_optional_string(selected.get("publication_slot")),
        publication_status=_optional_string(selected.get("publication_status")),
        snapshot_type=_optional_string(selected.get("snapshot_type")),
        generation_timestamp=_optional_string(selected.get("generation_timestamp")),
        effective_cutoff=_optional_string(selected.get("effective_cutoff")),
        requested_cutoff=_optional_string(selected.get("requested_cutoff")),
        reason=None,
        variant_label=variant.label,
        model_family=variant.prior_family,
        model_version=variant.model_version,
        model_version_field=variant.model_version_field,
        selection_policy=selection_policy,
        is_reconstruction=selection_policy == "reconstruction",
        is_retrospective_artifact=is_retrospective_publication(selected),
        generated_after_target_week_start=generated_after_start,
        timing_classification=timing_classification,
        source_retrieved_at=_optional_string(selected.get("source_retrieved_at")),
        source_context_snapshot_id=_optional_string(
            selected.get("source_context_snapshot_id")
        ),
        comparison_snapshot_id=_optional_string(selected.get("comparison_snapshot_id")),
    )


def select_pregame_publication(
    publications: Iterable[Mapping[str, Any]],
    *,
    season: int,
    week: int,
    week_start: datetime,
    version: str,
    allow_retrospective: bool = False,
) -> PublicationChoice:
    """Select the newest legitimate pregame Context artifact (legacy wrapper)."""
    return select_model_publication(
        publications,
        season=season,
        week=week,
        week_start=week_start,
        variant=ModelVariant(
            key=version,
            label=f"Context {version}",
            prior_family="context",
            model_version_field="context_prior",
            model_version=version,
        ),
        selection_policy="strict_pregame",
        allow_retrospective=allow_retrospective,
    )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def weekly_resource_path(publication: Mapping[str, Any], week_key: str) -> str:
    """Derive a canonical week resource from a publication's API link template."""
    links = publication.get("links")
    if not isinstance(links, Mapping) or not isinstance(links.get("weeks"), str):
        raise ContextMarketError("Publication has no links.weeks path")
    weeks_path = str(links["weeks"])
    if not weeks_path.endswith("/weeks.json"):
        raise ContextMarketError(f"Unexpected links.weeks path: {weeks_path!r}")
    return f"{weeks_path.removesuffix('/weeks.json')}/weeks/{week_key}.json"


def prediction_rows(
    payload: Mapping[str, Any],
    *,
    choice: PublicationChoice,
    allow_nonfuture_state: bool = False,
) -> list[dict[str, object]]:
    """Extract validated predictions from a selected static weekly resource.

    A strict study accepts only source rows marked ``future``. Historical
    states are allowed only when the caller deliberately opted into an
    explicitly labelled reconstruction policy.
    """
    if choice.snapshot_id is None:
        return []
    payload_week = payload.get("week")
    if (
        not isinstance(payload_week, Mapping)
        or int(payload_week.get("week", -1)) != choice.week
    ):
        raise ContextMarketError("Weekly resource does not match selected target week")
    games = payload.get("games")
    if not isinstance(games, list):
        raise ContextMarketError("Weekly resource has no games array")

    rows: list[dict[str, object]] = []
    for game in games:
        if not isinstance(game, Mapping):
            raise ContextMarketError("Weekly resource includes a non-object game")
        prediction = game.get("prediction")
        if prediction is None:
            continue
        if not isinstance(prediction, Mapping):
            raise ContextMarketError("Weekly resource has malformed prediction")
        game_id = str(game.get("game_id", ""))
        if not game_id or str(prediction.get("game_id", "")) != game_id:
            raise ContextMarketError(
                "Prediction game id does not match containing game"
            )
        if prediction.get("source_snapshot_id") != choice.snapshot_id:
            raise ContextMarketError(
                "Prediction source snapshot does not match selection"
            )
        prediction_state = game.get("state")
        if prediction_state != "future" and not allow_nonfuture_state:
            raise ContextMarketError(
                f"Selected pregame resource contains non-future game {game_id}"
            )
        expected = finite_number(prediction.get("expected_home_margin"))
        if expected is None:
            raise ContextMarketError(
                f"Prediction has no finite expected margin: {game_id}"
            )
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        if not isinstance(home_team, Mapping) or not isinstance(away_team, Mapping):
            raise ContextMarketError(f"Prediction has malformed teams: {game_id}")
        if str(home_team.get("team_id", "")) != str(game.get("home_team_id", "")):
            raise ContextMarketError(f"Home team id mismatch for {game_id}")
        if str(away_team.get("team_id", "")) != str(game.get("away_team_id", "")):
            raise ContextMarketError(f"Away team id mismatch for {game_id}")
        row: dict[str, object] = {
            "game_id": game_id,
            "gippy_expected_home_margin": expected,
            "gippy_median_home_margin": finite_number(
                prediction.get("median_home_margin")
            ),
            "gippy_margin_interval_50": prediction.get("margin_interval_50"),
            "gippy_margin_interval_80": prediction.get("margin_interval_80"),
            "gippy_margin_interval_95": prediction.get("margin_interval_95"),
            "prediction_home_team_id": str(game.get("home_team_id")),
            "prediction_away_team_id": str(game.get("away_team_id")),
            "prediction_home_subdivision": home_team.get("subdivision"),
            "prediction_away_subdivision": away_team.get("subdivision"),
            "source_prediction_state": prediction_state,
        }
        rows.append(row)
    return rows


def finite_number(value: object) -> float | None:
    """Return a finite float or ``None`` without treating booleans as numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def sportsbook_spread_to_home_margin(spread: object) -> float | None:
    """Convert CFBD's home-oriented sportsbook spread to home-score-minus-away.

    CFBD's ``spread`` signs match its ``formattedSpread`` convention: a home
    favorite is negative (``Home -7``).  GippyRank's expected margin has the
    opposite sign convention, so it is the negation of that numeric spread.
    """
    numeric = finite_number(spread)
    return -numeric if numeric is not None else None


def actual_home_margin(home_score: object, away_score: object) -> float | None:
    """Compute actual final margin in the same home-minus-away orientation."""
    home = finite_number(home_score)
    away = finite_number(away_score)
    return home - away if home is not None and away is not None else None


def favorite_side(home_margin: object) -> str | None:
    """Return the favorite's side in canonical home-margin orientation."""
    value = finite_number(home_margin)
    if value is None:
        return None
    if value > 0:
        return "home"
    if value < 0:
        return "away"
    return "pickem"


def parse_market_lines(payload: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Flatten raw CFBD line observations without silently combining books."""
    rows: list[dict[str, object]] = []
    for game in payload:
        game_id = str(game.get("id", ""))
        if not game_id:
            raise ContextMarketError("CFBD lines payload contains a game without id")
        lines = game.get("lines")
        if lines is None:
            continue
        if not isinstance(lines, list):
            raise ContextMarketError(f"CFBD lines for {game_id} are not an array")
        for line in lines:
            if not isinstance(line, Mapping):
                raise ContextMarketError(f"CFBD line for {game_id} is not an object")
            provider = line.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "provider": provider.strip(),
                    "formatted_spread": line.get("formattedSpread"),
                    "raw_spread": finite_number(line.get("spread")),
                    "raw_spread_open": finite_number(line.get("spreadOpen")),
                    "later_home_margin": sportsbook_spread_to_home_margin(
                        line.get("spread")
                    ),
                    "opening_home_margin": sportsbook_spread_to_home_margin(
                        line.get("spreadOpen")
                    ),
                }
            )
    return rows


def provider_coverage(
    market_rows: Iterable[Mapping[str, Any]], game_ids: Iterable[str]
) -> list[dict[str, object]]:
    """Measure exact-provider open/later coverage over a fixed game population."""
    eligible = {str(game_id) for game_id in game_ids}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in market_rows:
        game_id = str(row.get("game_id", ""))
        provider = row.get("provider")
        if game_id in eligible and isinstance(provider, str):
            grouped.setdefault((provider, game_id), []).append(row)
    coverage: list[dict[str, object]] = []
    providers = sorted({provider for provider, _ in grouped})
    for provider in providers:
        observations = [
            _resolve_provider_observation(grouped[(provider, game_id)])
            for candidate, game_id in grouped
            if candidate == provider
        ]
        opening = sum(
            observation.get("opening_home_margin") is not None
            for observation in observations
        )
        later = sum(
            observation.get("later_home_margin") is not None
            for observation in observations
        )
        comparable = sum(
            observation.get("opening_home_margin") is not None
            and observation.get("later_home_margin") is not None
            for observation in observations
        )
        coverage.append(
            {
                "provider": provider,
                "eligible_games": len(eligible),
                "observed_games": len(observations),
                "opening_games": opening,
                "later_games": later,
                "same_provider_open_later_games": comparable,
            }
        )
    return coverage


def select_primary_provider(
    market_rows: Iterable[Mapping[str, Any]],
    game_ids: Iterable[str],
    *,
    preferred_provider: str = "DraftKings",
) -> ProviderChoice:
    """Choose one book globally; never fall back to a different book per game."""
    coverage = provider_coverage(market_rows, game_ids)
    eligible_count = len({str(game_id) for game_id in game_ids})
    if not coverage:
        return ProviderChoice(None, 0, eligible_count, "No provider observations.")
    best_count = max(int(row["same_provider_open_later_games"]) for row in coverage)
    best = [
        row
        for row in coverage
        if int(row["same_provider_open_later_games"]) == best_count
    ]
    preferred = next(
        (row for row in best if row["provider"] == preferred_provider), None
    )
    selected = preferred or min(best, key=lambda row: str(row["provider"]))
    provider = str(selected["provider"])
    if provider == preferred_provider:
        reason = "Preferred provider has the maximum comparable open/later coverage."
    else:
        reason = (
            f"{preferred_provider} is not the maximum-coverage provider; selected "
            f"{provider} deterministically from providers with {best_count} comparable games."
        )
    return ProviderChoice(provider, best_count, eligible_count, reason)


def select_same_provider_lines(
    market_rows: Iterable[Mapping[str, Any]], provider: str
) -> dict[str, dict[str, object]]:
    """Return exactly one unblended market record per game for one provider."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in market_rows:
        if row.get("provider") == provider:
            grouped.setdefault(str(row.get("game_id", "")), []).append(row)
    return {
        game_id: _resolve_provider_observation(observations)
        for game_id, observations in grouped.items()
        if game_id
    }


def _resolve_provider_observation(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Reject conflicting duplicate lines from an identically named provider."""
    if not observations:
        raise ContextMarketError("Cannot resolve an empty provider observation set")
    first = dict(observations[0])
    fields = (
        "opening_home_margin",
        "later_home_margin",
        "raw_spread",
        "raw_spread_open",
    )
    for observation in observations[1:]:
        if any(observation.get(field) != first.get(field) for field in fields):
            game_id = first.get("game_id")
            provider = first.get("provider")
            raise ContextMarketError(
                f"Conflicting duplicate market rows for {provider} game {game_id}"
            )
    return first


def schedule_by_game_id(
    games: Iterable[Mapping[str, Any]], *, season: int
) -> dict[str, Mapping[str, Any]]:
    """Index the CFBD schedule by its stable game id and reject duplicates."""
    index: dict[str, Mapping[str, Any]] = {}
    for game in games:
        if game.get("season") != season:
            continue
        game_id = str(game.get("id", ""))
        if not game_id:
            raise ContextMarketError("CFBD schedule has a game without id")
        if game_id in index and index[game_id] != game:
            raise ContextMarketError(f"Conflicting schedule records for game {game_id}")
        index[game_id] = game
    return index


def completed_weeks(games: Iterable[Mapping[str, Any]], *, season: int) -> list[int]:
    """Return regular-season weeks whose supplied schedule records are complete."""
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for game in games:
        if game.get("season") != season or game.get("seasonType") != "regular":
            continue
        week = game.get("week")
        if isinstance(week, bool):
            continue
        try:
            groups.setdefault(int(week), []).append(game)
        except (TypeError, ValueError):
            continue
    return sorted(
        week
        for week, rows in groups.items()
        if rows and all(bool(row.get("completed")) for row in rows)
    )


def week_start(
    games: Iterable[Mapping[str, Any]], *, season: int, week: int
) -> datetime:
    """Get the earliest kickoff in a CFBD week, in UTC."""
    kickoffs = [
        parsed
        for game in games
        if game.get("season") == season
        and game.get("seasonType") == "regular"
        and game.get("week") == week
        and (parsed := parse_timestamp(game.get("startDate"))) is not None
    ]
    if not kickoffs:
        raise ContextMarketError(
            f"No kickoff timestamp for season {season} week {week}"
        )
    return min(kickoffs)


def build_per_game_rows(
    predictions: Iterable[Mapping[str, Any]],
    *,
    schedule: Mapping[str, Mapping[str, Any]],
    market: Mapping[str, Mapping[str, Any]],
    choice: PublicationChoice,
    provider: str | None,
) -> list[dict[str, object]]:
    """Join a selected model/week to schedule outcomes and one market provider."""
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        game_id = str(prediction["game_id"])
        game = schedule.get(game_id)
        if game is None:
            raise ContextMarketError(
                f"Prediction game {game_id} is absent from CFBD schedule"
            )
        if game.get("week") != choice.week:
            raise ContextMarketError(
                f"Prediction game {game_id} does not belong to target week {choice.week}"
            )
        if str(game.get("homeId")) != str(prediction["prediction_home_team_id"]):
            raise ContextMarketError(f"Home-team mismatch for game {game_id}")
        if str(game.get("awayId")) != str(prediction["prediction_away_team_id"]):
            raise ContextMarketError(f"Away-team mismatch for game {game_id}")
        line = market.get(game_id, {})
        gippy = finite_number(prediction.get("gippy_expected_home_margin"))
        opening = finite_number(line.get("opening_home_margin"))
        later = finite_number(line.get("later_home_margin"))
        actual = actual_home_margin(game.get("homePoints"), game.get("awayPoints"))
        home_classification = _optional_string(game.get("homeClassification"))
        away_classification = _optional_string(game.get("awayClassification"))
        primary = (
            home_classification is not None
            and away_classification is not None
            and home_classification.casefold() == "fbs"
            and away_classification.casefold() == "fbs"
        )
        row: dict[str, object] = {
            "week": choice.week,
            "game_id": game_id,
            "kickoff": game.get("startDate"),
            "away_team": game.get("awayTeam"),
            "home_team": game.get("homeTeam"),
            "away_team_id": str(game.get("awayId")),
            "home_team_id": str(game.get("homeId")),
            "away_classification": away_classification,
            "home_classification": home_classification,
            "neutral_site": bool(game.get("neutralSite")),
            "variant_key": choice.variant_key,
            "variant_label": choice.variant_label,
            "model_family": choice.model_family,
            "model_version": choice.model_version,
            "model_version_field": choice.model_version_field,
            "snapshot_id": choice.snapshot_id,
            "publication_slot": choice.publication_slot,
            "publication_status": choice.publication_status,
            "generation_timestamp": choice.generation_timestamp,
            "effective_cutoff": choice.effective_cutoff,
            "source_retrieved_at": choice.source_retrieved_at,
            "source_context_snapshot_id": choice.source_context_snapshot_id,
            "comparison_snapshot_id": choice.comparison_snapshot_id,
            "selection_policy": choice.selection_policy,
            "is_reconstruction": choice.is_reconstruction,
            "is_retrospective_artifact": choice.is_retrospective_artifact,
            "generated_after_target_week_start": choice.generated_after_target_week_start,
            "timing_classification": choice.timing_classification,
            "source_prediction_state": prediction.get("source_prediction_state"),
            "gippy_expected_home_margin": gippy,
            "gippy_median_home_margin": prediction.get("gippy_median_home_margin"),
            "gippy_margin_interval_50": prediction.get("gippy_margin_interval_50"),
            "gippy_margin_interval_80": prediction.get("gippy_margin_interval_80"),
            "gippy_margin_interval_95": prediction.get("gippy_margin_interval_95"),
            "market_provider": provider,
            "market_formatted_spread": line.get("formatted_spread"),
            "opening_home_margin": opening,
            # CFBD documents no observation timestamp/history for this field.
            "closing_home_margin": later,
            "cfbd_later_home_margin": later,
            "actual_home_score": finite_number(game.get("homePoints")),
            "actual_away_score": finite_number(game.get("awayPoints")),
            "actual_home_margin": actual,
            "is_primary_fbs_vs_fbs": primary,
            "schedule_completed": bool(game.get("completed")),
        }
        _add_derived_columns(row)
        rows.append(row)
    return rows


def _add_derived_columns(row: dict[str, object]) -> None:
    """Add all arithmetic fields requested by the study in a null-safe form."""
    gippy = finite_number(row.get("gippy_expected_home_margin"))
    opening = finite_number(row.get("opening_home_margin"))
    closing = finite_number(row.get("closing_home_margin"))
    actual = finite_number(row.get("actual_home_margin"))
    row["gippy_minus_open"] = _subtract(gippy, opening)
    row["gippy_minus_close"] = _subtract(gippy, closing)
    row["abs_gippy_minus_open"] = _absolute(row["gippy_minus_open"])
    row["abs_gippy_minus_close"] = _absolute(row["gippy_minus_close"])
    row["market_move"] = _subtract(closing, opening)
    row["open_distance_to_gippy"] = row["abs_gippy_minus_open"]
    row["close_distance_to_gippy"] = row["abs_gippy_minus_close"]
    if gippy is not None and opening is not None and closing is not None:
        row["market_movement_toward_gippy"] = abs(gippy - opening) - abs(
            gippy - closing
        )
    else:
        row["market_movement_toward_gippy"] = None
    row["gippy_error_vs_actual"] = _subtract(gippy, actual)
    row["open_error_vs_actual"] = _subtract(opening, actual)
    row["close_error_vs_actual"] = _subtract(closing, actual)
    row["gippy_edge_vs_open"] = row["gippy_minus_open"]
    row["gippy_edge_vs_close"] = row["gippy_minus_close"]
    row["actual_residual_vs_open"] = _subtract(actual, opening)
    row["actual_residual_vs_close"] = _subtract(actual, closing)
    reasons: list[str] = []
    if not bool(row.get("is_primary_fbs_vs_fbs")):
        reasons.append("not_fbs_vs_fbs")
    if gippy is None:
        reasons.append("missing_gippy_prediction")
    if opening is None:
        reasons.append("missing_opening_line")
    if closing is None:
        reasons.append("missing_cfbd_later_line")
    if actual is None:
        reasons.append("missing_final_score")
    row["exclusion_reasons"] = ";".join(reasons)


def _subtract(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _absolute(value: object) -> float | None:
    numeric = finite_number(value)
    return abs(numeric) if numeric is not None else None


def primary_sample(frame: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    """Filter to FBS-v-FBS rows with every named field present."""
    if frame.empty:
        return frame.copy()
    mask = frame["is_primary_fbs_vs_fbs"].astype(bool)
    for column in required:
        mask &= frame[column].notna()
    return frame.loc[mask].copy()


def common_game_ids(
    frame: pd.DataFrame,
    versions: Sequence[str],
    *,
    required: Sequence[str],
) -> set[str]:
    """Return the exact paired game intersection across requested versions."""
    sets: list[set[str]] = []
    for version in versions:
        subset = frame.loc[frame["variant_key"] == version]
        if subset.empty:
            return set()
        valid = primary_sample(subset, required)
        sets.append(set(valid["game_id"].astype(str)))
    return set.intersection(*sets) if sets else set()


def alignment_metrics(frame: pd.DataFrame, market_column: str) -> dict[str, object]:
    """Summarize model-to-market alignment without making outcome claims."""
    paired = _paired_frame(frame, "gippy_expected_home_margin", market_column)
    if paired.empty:
        return _empty_metric_row()
    difference = paired["gippy_expected_home_margin"] - paired[market_column]
    gippy = paired["gippy_expected_home_margin"].to_numpy(dtype=float)
    market = paired[market_column].to_numpy(dtype=float)
    return {
        "n": len(paired),
        "mean_absolute_difference": _float(abs(difference).mean()),
        "median_absolute_difference": _float(abs(difference).median()),
        "rmse": _float(np.sqrt(np.mean(np.square(difference)))),
        "mean_signed_difference": _float(difference.mean()),
        "pearson_correlation": _correlation(gippy, market, pearsonr),
        "spearman_correlation": _correlation(gippy, market, spearmanr),
        "same_favorite_pct": _same_favorite_percentage(gippy, market),
        "within_3_pct": _float(float(np.mean(np.abs(difference) <= 3.0))),
        "within_5_pct": _float(float(np.mean(np.abs(difference) <= 5.0))),
        "within_7_pct": _float(float(np.mean(np.abs(difference) <= 7.0))),
    }


def calibration_metrics(frame: pd.DataFrame, market_column: str) -> dict[str, object]:
    """Fit market margin = intercept + slope * Gippy margin descriptively."""
    paired = _paired_frame(frame, "gippy_expected_home_margin", market_column)
    return linear_relationship(
        paired.get("gippy_expected_home_margin", pd.Series(dtype=float)),
        paired.get(market_column, pd.Series(dtype=float)),
    )


def movement_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize whether a same-book later line moved toward the model."""
    paired = _paired_frame(
        frame,
        "gippy_expected_home_margin",
        "opening_home_margin",
        "closing_home_margin",
    )
    if paired.empty:
        return _empty_movement_row()
    movement = paired["market_movement_toward_gippy"].astype(float)
    return {
        "n": len(paired),
        "mean_distance_from_open": _float(
            paired["open_distance_to_gippy"].astype(float).mean()
        ),
        "mean_distance_from_close": _float(
            paired["close_distance_to_gippy"].astype(float).mean()
        ),
        "median_distance_from_open": _float(
            paired["open_distance_to_gippy"].astype(float).median()
        ),
        "median_distance_from_close": _float(
            paired["close_distance_to_gippy"].astype(float).median()
        ),
        "mean_market_movement_toward_gippy": _float(movement.mean()),
        "moved_toward_pct": _float(float(np.mean(movement > 1e-9))),
        "moved_away_pct": _float(float(np.mean(movement < -1e-9))),
        "unchanged_pct": _float(float(np.mean(np.abs(movement) <= 1e-9))),
    }


def market_move_relationship(frame: pd.DataFrame) -> dict[str, object]:
    """Relate opening model edge to subsequent same-book market movement."""
    paired = _paired_frame(frame, "gippy_edge_vs_open", "market_move")
    if paired.empty:
        return {**_empty_relationship_row(), "n": 0}
    x = paired["gippy_edge_vs_open"].to_numpy(dtype=float)
    y = paired["market_move"].to_numpy(dtype=float)
    return {
        "n": len(paired),
        "pearson_correlation": _correlation(x, y, pearsonr),
        "spearman_correlation": _correlation(x, y, spearmanr),
        **linear_relationship(x, y),
    }


def actual_margin_metrics(
    frame: pd.DataFrame, prediction_column: str
) -> dict[str, object]:
    """Score one margin prediction against actual home-minus-away margin."""
    paired = _paired_frame(frame, prediction_column, "actual_home_margin")
    if paired.empty:
        return _empty_metric_row()
    error = paired[prediction_column] - paired["actual_home_margin"]
    prediction = paired[prediction_column].to_numpy(dtype=float)
    actual = paired["actual_home_margin"].to_numpy(dtype=float)
    return {
        "n": len(paired),
        "mae": _float(abs(error).mean()),
        "median_absolute_error": _float(abs(error).median()),
        "rmse": _float(np.sqrt(np.mean(np.square(error)))),
        "mean_signed_error": _float(error.mean()),
        "pearson_correlation": _correlation(prediction, actual, pearsonr),
        "spearman_correlation": _correlation(prediction, actual, spearmanr),
    }


def paired_mae_difference(
    frame: pd.DataFrame,
    benchmark_column: str,
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 125,
) -> dict[str, object]:
    """Compare Gippy's MAE to a benchmark on exactly the same games.

    Positive differences mean Gippy's absolute errors were larger (worse) than
    the benchmark's.  The bootstrap interval is descriptive, not a claim of a
    calibrated hypothesis test.
    """
    paired = _paired_frame(
        frame,
        "gippy_expected_home_margin",
        benchmark_column,
        "actual_home_margin",
    )
    if paired.empty:
        return {
            "n": 0,
            "mean_paired_mae_difference": None,
            "median_paired_mae_difference": None,
            "bootstrap_95_ci": [None, None],
        }
    actual = paired["actual_home_margin"].to_numpy(dtype=float)
    gippy_error = np.abs(
        paired["gippy_expected_home_margin"].to_numpy(dtype=float) - actual
    )
    benchmark_error = np.abs(paired[benchmark_column].to_numpy(dtype=float) - actual)
    difference = gippy_error - benchmark_error
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(difference), size=(bootstrap_samples, len(difference))
    )
    bootstrap_means = difference[indices].mean(axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "n": len(difference),
        "mean_paired_mae_difference": _float(float(np.mean(difference))),
        "median_paired_mae_difference": _float(float(np.median(difference))),
        "bootstrap_95_ci": [_float(float(interval[0])), _float(float(interval[1]))],
    }


def edge_signal_metrics(frame: pd.DataFrame, endpoint: str) -> dict[str, object]:
    """Measure whether model/market disagreement co-moves with game residuals."""
    if endpoint not in {"open", "close"}:
        raise ContextMarketError(f"Unsupported endpoint: {endpoint}")
    edge_column = f"gippy_edge_vs_{endpoint}"
    residual_column = f"actual_residual_vs_{endpoint}"
    paired = _paired_frame(frame, edge_column, residual_column)
    if paired.empty:
        return {**_empty_relationship_row(), "n": 0, "directional_agreement_pct": None}
    edge = paired[edge_column].to_numpy(dtype=float)
    residual = paired[residual_column].to_numpy(dtype=float)
    return {
        "n": len(paired),
        "pearson_correlation": _correlation(edge, residual, pearsonr),
        "spearman_correlation": _correlation(edge, residual, spearmanr),
        **linear_relationship(edge, residual),
        "directional_agreement_pct": _float(
            float(np.mean(np.sign(edge) == np.sign(residual)))
        ),
    }


def edge_bucket_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Describe market movement within transparent absolute opening-edge buckets."""
    paired = _paired_frame(
        frame,
        "gippy_edge_vs_open",
        "market_movement_toward_gippy",
        "close_distance_to_gippy",
    )
    rows: list[dict[str, object]] = []
    absolute_edge = (
        paired["gippy_edge_vs_open"].abs()
        if not paired.empty
        else pd.Series(dtype=float)
    )
    for label, lower, upper in EDGE_BUCKETS:
        mask = absolute_edge >= lower
        if upper is not None:
            mask &= absolute_edge < upper
        bucket = paired.loc[mask]
        rows.append(
            {
                "bucket": label,
                "n": len(bucket),
                "average_movement_toward_gippy": _float(
                    bucket["market_movement_toward_gippy"].mean()
                )
                if not bucket.empty
                else None,
                "moved_toward_pct": _float(
                    float(np.mean(bucket["market_movement_toward_gippy"] > 1e-9))
                )
                if not bucket.empty
                else None,
                "average_close_distance_to_gippy": _float(
                    bucket["close_distance_to_gippy"].mean()
                )
                if not bucket.empty
                else None,
            }
        )
    return rows


def edge_threshold_metrics(
    frame: pd.DataFrame, endpoint: str, thresholds: Sequence[float] = (3, 5, 7, 10)
) -> list[dict[str, object]]:
    """Repeat residual-signal summaries for progressively larger disagreements."""
    edge_column = f"gippy_edge_vs_{endpoint}"
    output: list[dict[str, object]] = []
    for threshold in thresholds:
        subset = frame.loc[frame[edge_column].abs() >= threshold].copy()
        output.append(
            {
                "minimum_absolute_edge": threshold,
                **edge_signal_metrics(subset, endpoint),
            }
        )
    return output


def linear_relationship(
    x: Sequence[float] | pd.Series, y: Sequence[float] | pd.Series
) -> dict[str, object]:
    """Return a simple descriptive intercept, slope, and R² safely."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if (
        len(x_values) < 2
        or len(y_values) < 2
        or not np.isfinite(x_values).all()
        or not np.isfinite(y_values).all()
        or np.allclose(x_values, x_values[0])
    ):
        return _empty_relationship_row()
    result = linregress(x_values, y_values)
    return {
        "intercept": _float(float(result.intercept)),
        "slope": _float(float(result.slope)),
        "r_squared": _float(float(result.rvalue**2)),
    }


def _paired_frame(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.dropna(subset=list(columns)).copy()


def _correlation(first: np.ndarray, second: np.ndarray, function: Any) -> float | None:
    if len(first) < 2 or np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return None
    try:
        result = function(first, second)
    except ValueError:
        return None
    statistic = getattr(result, "statistic", result[0])
    return _float(float(statistic))


def _same_favorite_percentage(first: np.ndarray, second: np.ndarray) -> float:
    left = np.sign(first)
    right = np.sign(second)
    return _float(float(np.mean(left == right))) or 0.0


def _empty_metric_row() -> dict[str, object]:
    return {
        "n": 0,
        "mean_absolute_difference": None,
        "median_absolute_difference": None,
        "rmse": None,
        "mean_signed_difference": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
        "same_favorite_pct": None,
        "within_3_pct": None,
        "within_5_pct": None,
        "within_7_pct": None,
    }


def _empty_movement_row() -> dict[str, object]:
    return {
        "n": 0,
        "mean_distance_from_open": None,
        "mean_distance_from_close": None,
        "median_distance_from_open": None,
        "median_distance_from_close": None,
        "mean_market_movement_toward_gippy": None,
        "moved_toward_pct": None,
        "moved_away_pct": None,
        "unchanged_pct": None,
    }


def _empty_relationship_row() -> dict[str, object]:
    return {"intercept": None, "slope": None, "r_squared": None}


def _float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
