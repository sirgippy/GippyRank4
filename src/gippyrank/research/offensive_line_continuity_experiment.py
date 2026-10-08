"""Research-only four-season offensive-line continuity features.

The target cohort comes from the retrospective CFBD OL panel. Prior program
membership is identity-linked and position-agnostic, but is restricted to the
same team ID and the four seasons before the target season. This module does
not change the production Context feature contract.
"""

from __future__ import annotations

import itertools
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

HISTORY_START_SEASON = 2009
PRIMARY_START_SEASON = 2013
PRIMARY_END_SEASON = 2025
LOOKBACK_SEASONS = 4

CONTINUITY_FEATURES = (
    "ol_mean_shared_roster_seasons_4y",
    "ol_shared_pair_share_4y",
    "ol_returning_group_share_4y",
)
INDIVIDUAL_EXPERIENCE_FEATURES = (
    "ol_returning_player_share_4y",
    "ol_mean_prior_same_program_seasons_4y",
)
DIAGNOSTIC_FEATURES = ("ol_prior_other_program_share_4y",)
MODEL_FEATURES = (*INDIVIDUAL_EXPERIENCE_FEATURES, *CONTINUITY_FEATURES)


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalized_person_name(value: str) -> str:
    """Normalize a source display name for exact, auditable roster joins."""
    ascii_name = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return "".join(char for char in ascii_name if char.isalnum())


def player_identity(row: Mapping[str, Any]) -> str | None:
    identity = str(row.get("normalized_player_identity") or "").strip()
    if identity:
        return identity
    source_id = str(row.get("source_player_id") or "").strip()
    return f"cfbd:{source_id}" if source_id else None


def build_program_memberships(
    roster_rows: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int], set[str]],
    dict[tuple[str, str], set[int]],
    dict[tuple[int, str], dict[str, set[str]]],
    dict[str, set[tuple[str, int]]],
]:
    """Index all-position roster memberships by program, season, and player.

    The third index is only for resolving validation names to provider IDs. It
    intentionally preserves ambiguity instead of choosing one duplicate name.
    """
    members: dict[tuple[str, int], set[str]] = defaultdict(set)
    player_seasons: dict[tuple[str, str], set[int]] = defaultdict(set)
    player_program_seasons: dict[str, set[tuple[str, int]]] = defaultdict(set)
    names: dict[tuple[int, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in roster_rows:
        team_id = str(row.get("team_id") or "").strip()
        identity = player_identity(row)
        if not team_id or not identity:
            continue
        try:
            season = int(row["season"])
        except (KeyError, TypeError, ValueError):
            continue
        members[(team_id, season)].add(identity)
        player_seasons[(team_id, identity)].add(season)
        player_program_seasons[identity].add((team_id, season))
        name = normalized_person_name(
            str(row.get("normalized_player_name") or row.get("player_name") or "")
        )
        if name:
            names[(season, team_id)][name].add(identity)
    return (
        dict(members),
        dict(player_seasons),
        dict(names),
        dict(player_program_seasons),
    )


def validate_pairwise_artifact(
    summary_rows: Iterable[Mapping[str, Any]],
    pair_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Aggregate and cross-check the committed all-history pairwise artifact."""
    summaries = {(int(row["season"]), str(row["team_id"])): row for row in summary_rows}
    observed: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in pair_rows:
        try:
            key = (int(row["target_season"]), str(row["team_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        value = row.get("shared_prior_season_count")
        if value not in (None, ""):
            count = int(value)
            if count < 0:
                raise ValueError("shared prior-season counts cannot be negative")
            observed[key].append(count)

    result: dict[tuple[int, str], dict[str, Any]] = {}
    for key, summary in summaries.items():
        if key[0] < HISTORY_START_SEASON:
            continue
        values = observed.get(key, [])
        expected_count = _as_int(summary.get("pair_count_evaluable"))
        if len(values) != expected_count:
            raise ValueError(
                f"pairwise artifact count mismatch for {key}: "
                f"expected {expected_count}, found {len(values)}"
            )
        if values:
            expected_total = _as_int(summary.get("total_pairwise_shared_seasons"))
            if sum(values) != expected_total:
                raise ValueError(f"pairwise artifact total mismatch for {key}")
            expected_nonzero = _as_int(
                summary.get("pairs_with_at_least_1_shared_season")
            )
            if sum(value > 0 for value in values) != expected_nonzero:
                raise ValueError(f"pairwise artifact nonzero-pair mismatch for {key}")
            mean = math.fsum(values) / len(values)
        else:
            mean = None
        result[key] = {
            "pair_count_evaluable": expected_count,
            "pair_mean_all_history": mean,
            "pair_nonzero_all_history": sum(value > 0 for value in values),
            "pair_values": tuple(values),
        }
    return result


def feature_values_for_pool(
    *,
    season: int,
    team_id: str,
    target_player_ids: Iterable[str],
    program_memberships: Mapping[tuple[str, int], set[str]],
    player_program_seasons: Mapping[str, set[tuple[str, int]]],
    lookback_seasons: int = LOOKBACK_SEASONS,
    history_window_observed: bool = True,
    identity_complete: bool = True,
) -> dict[str, float | None]:
    """Calculate pair, returning-group, and individual exposure summaries."""
    ids = tuple(sorted({str(value) for value in target_player_ids if value}))
    output: dict[str, float | None] = {name: None for name in MODEL_FEATURES}
    output.update({name: None for name in DIAGNOSTIC_FEATURES})
    if not history_window_observed or not identity_complete or len(ids) < 2:
        return output

    window_start = max(HISTORY_START_SEASON, season - lookback_seasons)
    years = range(window_start, season)
    prior_sets = {
        player_id: {
            prior_season
            for prior_season in years
            if player_id in program_memberships.get((team_id, prior_season), set())
        }
        for player_id in ids
    }
    n_pairs = len(ids) * (len(ids) - 1) // 2
    pair_counts = [
        len(prior_sets[left] & prior_sets[right])
        for left, right in itertools.combinations(ids, 2)
    ]
    if len(pair_counts) != n_pairs:
        raise AssertionError("pair construction lost a target-player pair")

    group_sizes = [
        sum(
            player_id in program_memberships.get((team_id, prior_season), set())
            for player_id in ids
        )
        for prior_season in years
    ]
    largest_group = max(group_sizes, default=0)
    prior_other_program = {
        player_id
        for player_id in ids
        if any(
            other_team != team_id and window_start <= prior_season < season
            for other_team, prior_season in player_program_seasons.get(player_id, set())
        )
    }

    output["ol_mean_shared_roster_seasons_4y"] = math.fsum(pair_counts) / n_pairs
    output["ol_shared_pair_share_4y"] = (
        sum(value > 0 for value in pair_counts) / n_pairs
    )
    output["ol_returning_group_share_4y"] = (
        largest_group / len(ids) if largest_group >= 2 else 0.0
    )
    output["ol_returning_player_share_4y"] = sum(
        bool(value) for value in prior_sets.values()
    ) / len(ids)
    output["ol_mean_prior_same_program_seasons_4y"] = math.fsum(
        len(value) for value in prior_sets.values()
    ) / len(ids)
    output["ol_prior_other_program_share_4y"] = len(prior_other_program) / len(ids)
    return output


def build_team_season_feature_panel(
    *,
    summary_rows: Iterable[Mapping[str, Any]],
    ol_rows: Iterable[Mapping[str, Any]],
    program_memberships: Mapping[tuple[str, int], set[str]],
    player_program_seasons: Mapping[str, set[tuple[str, int]]],
    pairwise_audit: Mapping[tuple[int, str], Mapping[str, Any]],
    start_season: int = HISTORY_START_SEASON,
    end_season: int = PRIMARY_END_SEASON,
) -> list[dict[str, Any]]:
    """Build a coverage-explicit panel from the existing #183 artifacts."""
    target_ids: dict[tuple[int, str], set[str]] = defaultdict(set)
    target_missing_ids: dict[tuple[int, str], int] = defaultdict(int)
    for row in ol_rows:
        if str(row.get("source_classification") or "").lower() != "fbs":
            continue
        try:
            key = (int(row["season"]), str(row["team_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        identity = player_identity(row)
        if identity:
            target_ids[key].add(identity)
        else:
            target_missing_ids[key] += 1

    output: list[dict[str, Any]] = []
    for summary in summary_rows:
        season = int(summary["season"])
        if not start_season <= season <= end_season:
            continue
        team_id = str(summary["team_id"])
        key = (season, team_id)
        pool_size = _as_int(summary.get("identifiable_ol_player_count"))
        source_rows = _as_int(summary.get("ol_source_rows"))
        missing_ids = max(
            _as_int(summary.get("ol_player_ids_missing")),
            target_missing_ids.get(key, 0),
        )
        ids = target_ids.get(key, set())
        id_complete = missing_ids == 0 and len(ids) == _as_int(
            summary.get("ol_player_ids_available"), len(ids)
        )
        roster_available = _as_bool(summary.get("roster_response_available"))
        roster_status = str(summary.get("roster_status") or "unknown")
        target_usable = (
            roster_available
            and roster_status == "roster_present_with_identifiable_ol"
            and pool_size >= 2
        )
        target_evaluable = _as_bool(summary.get("continuity_target_evaluable"))
        history_available = _as_int(summary.get("continuity_history_seasons_available"))
        history_start = max(HISTORY_START_SEASON, season - LOOKBACK_SEASONS)
        requested_lookback = max(0, season - history_start)
        history_observed = (
            target_evaluable
            and requested_lookback > 0
            and history_available >= requested_lookback
        )
        primary_window_complete = (
            requested_lookback == LOOKBACK_SEASONS and history_observed
        )
        measurement = feature_values_for_pool(
            season=season,
            team_id=team_id,
            target_player_ids=ids,
            program_memberships=program_memberships,
            player_program_seasons=player_program_seasons,
            history_window_observed=history_observed,
            identity_complete=id_complete,
        )
        pair_audit = pairwise_audit.get(key, {})
        possible_pairs = pool_size * (pool_size - 1) // 2
        pair_evaluable = _as_int(summary.get("pair_count_evaluable"))
        pair_history_censored = _as_int(summary.get("pair_count_history_censored"))
        pair_identity_unresolved = _as_int(
            summary.get("pair_count_identity_unresolved")
        )
        row: dict[str, Any] = {
            "season": season,
            "subdivision": "fbs",
            "team_id": team_id,
            "team_name": summary.get("team_name", ""),
            "roster_response_available": roster_available,
            "roster_status": roster_status,
            "target_ol_pool_usable": target_usable,
            "target_ol_player_count": pool_size,
            "target_ol_source_rows": source_rows,
            "target_ol_player_ids_available": _as_int(
                summary.get("ol_player_ids_available")
            ),
            "target_ol_player_ids_missing": missing_ids,
            "target_ol_identity_complete": id_complete,
            "target_ol_unknown_position_rows": _as_int(
                summary.get("unknown_position_rows")
            ),
            "target_ol_ambiguous_position_rows": _as_int(
                summary.get("ambiguous_ol_position_rows")
            ),
            "continuity_target_evaluable": target_evaluable,
            "continuity_history_status": summary.get("continuity_history_status", ""),
            "continuity_censor_reason": summary.get("continuity_censor_reason", ""),
            "lookback_start_season": history_start,
            "lookback_end_season": season - 1,
            "lookback_calendar_seasons": requested_lookback,
            "history_window_observed": history_observed,
            "primary_four_year_window_complete": primary_window_complete,
            "history_seasons_available": history_available,
            "pair_count_possible": possible_pairs,
            "pair_count_evaluable_all_history": pair_evaluable,
            "pair_count_history_censored_all_history": pair_history_censored,
            "pair_count_identity_unresolved_all_history": pair_identity_unresolved,
            "pair_coverage_all_history": (
                pair_evaluable / possible_pairs if possible_pairs else None
            ),
            "pairwise_artifact_mean_all_history": pair_audit.get(
                "pair_mean_all_history"
            ),
            "target_roster_temporal_status": "retrospective_target_season_roster_not_preseason_snapshot",
            "feature_status": _feature_status(
                roster_available=roster_available,
                target_usable=target_usable,
                history_observed=history_observed,
                primary_window_complete=primary_window_complete,
                identity_complete=id_complete,
                pool_size=pool_size,
            ),
            **measurement,
        }
        output.append(row)
    return sorted(output, key=lambda row: (int(row["season"]), str(row["team_id"])))


def _feature_status(
    *,
    roster_available: bool,
    target_usable: bool,
    history_observed: bool,
    primary_window_complete: bool,
    identity_complete: bool,
    pool_size: int,
) -> str:
    if not roster_available:
        return "target_roster_unavailable"
    if not target_usable:
        return "target_pool_unusable_or_too_small"
    if not identity_complete:
        return "target_identity_incomplete"
    if not history_observed:
        return "history_window_incomplete_or_censored"
    if pool_size < 2:
        return "insufficient_target_pairs"
    if not primary_window_complete:
        return "observed_secondary_partial_lookback"
    return "observed"


def build_official_pool_sensitivity(
    *,
    player_comparison_rows: Iterable[Mapping[str, Any]],
    team_season_rows: Iterable[Mapping[str, Any]],
    normalized_roster_names: Mapping[tuple[int, str], Mapping[str, set[str]]],
    program_memberships: Mapping[tuple[str, int], set[str]],
    player_program_seasons: Mapping[str, set[tuple[str, int]]],
    cfbd_feature_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    crosswalk_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Recompute issue #185 official-pool features where every identity maps.

    No ID is invented for an official player absent from the CFBD history.
    Such a team-season remains in the output with an explicit unmapped count.
    """
    teams = {
        (int(row["season"]), normalized_person_name(str(row["team_name"]))): row
        for row in team_season_rows
    }
    aliases: dict[tuple[str, str], str] = {}
    for row in crosswalk_rows:
        aliases[
            (
                str(row.get("sample_id") or ""),
                normalized_person_name(str(row.get("official_player_name") or "")),
            )
        ] = str(row.get("cfbd_player_name") or "")

    players_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in player_comparison_rows:
        if _as_bool(row.get("official_ol")):
            players_by_sample[str(row.get("sample_id") or "")].append(row)

    output: list[dict[str, Any]] = []
    for sample_id, player_rows in sorted(players_by_sample.items()):
        if not player_rows:
            continue
        first = player_rows[0]
        season = int(first["season"])
        team_name = str(first["team_name"])
        team = teams.get((season, normalized_person_name(team_name)))
        if team is None:
            output.append(
                {
                    "sample_id": sample_id,
                    "season": season,
                    "team_name": team_name,
                    "official_pool_mapping_status": "team_key_unmatched",
                    "official_pool_size": len(player_rows),
                    "official_pool_ids_mapped": 0,
                    "official_pool_ids_unmapped": len(player_rows),
                }
            )
            continue
        team_id = str(team["team_id"])
        if not team_id:
            output.append(
                {
                    "sample_id": sample_id,
                    "season": season,
                    "team_name": team_name,
                    "official_pool_mapping_status": "team_key_unmatched",
                    "official_pool_size": len(player_rows),
                    "official_pool_ids_mapped": 0,
                    "official_pool_ids_unmapped": len(player_rows),
                }
            )
            continue
        ids: set[str] = set()
        unmapped = 0
        ambiguous = 0
        mapped_names: list[str] = []
        unmapped_names: list[str] = []
        for player in player_rows:
            official_name = str(player.get("official_player_name") or "")
            lookup_name = aliases.get(
                (sample_id, normalized_person_name(official_name)), official_name
            )
            cfbd_name = str(player.get("cfbd_roster_player_name") or "")
            if not cfbd_name:
                cfbd_name = lookup_name
            normalized = normalized_person_name(cfbd_name)
            candidates = normalized_roster_names.get((season, team_id), {}).get(
                normalized, set()
            )
            if len(candidates) == 1:
                ids.update(candidates)
                mapped_names.append(f"{official_name}={next(iter(candidates))}")
            else:
                unmapped += 1
                ambiguous += int(len(candidates) > 1)
                unmapped_names.append(official_name)
        complete = unmapped == 0 and len(ids) == len(player_rows)
        base = cfbd_feature_rows.get((season, team_id), {})
        official_features = feature_values_for_pool(
            season=season,
            team_id=team_id,
            target_player_ids=ids,
            program_memberships=program_memberships,
            player_program_seasons=player_program_seasons,
            history_window_observed=_as_bool(base.get("history_window_observed")),
            identity_complete=complete,
        )
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "season": season,
            "team_name": team_name,
            "team_id": team_id,
            "official_pool_size": len(player_rows),
            "official_pool_ids_mapped": len(ids),
            "official_pool_ids_unmapped": unmapped,
            "official_pool_id_ambiguities": ambiguous,
            "official_player_identity_mapping": "; ".join(mapped_names),
            "official_unmapped_player_names": "; ".join(unmapped_names),
            "official_pool_mapping_status": "complete" if complete else "incomplete",
            "cfbd_target_pool_size": _as_int(team.get("cfbd_ol_count")),
            "cfbd_exact_player_set_match": team.get("exact_player_set_match", ""),
        }
        for name in MODEL_FEATURES:
            official_value = official_features.get(name)
            cfbd_value = base.get(name)
            result[f"cfbd_{name}"] = cfbd_value
            result[f"official_{name}"] = official_value if complete else None
            result[f"official_minus_cfbd_{name}"] = (
                float(official_value) - float(cfbd_value)
                if complete
                and official_value is not None
                and cfbd_value not in (None, "")
                else None
            )
        output.append(result)
    return output


def paired_season_bootstrap(
    deltas_by_season: Mapping[int, Sequence[float]],
    *,
    seed: int = 7,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    """Deterministic team-season weighted bootstrap clustered by season."""
    seasons = tuple(
        sorted(season for season, values in deltas_by_season.items() if values)
    )
    if not seasons:
        raise ValueError("bootstrap requires at least one non-empty season")
    values = {
        season: tuple(float(item) for item in deltas_by_season[season])
        for season in seasons
    }
    sizes = {season: len(values[season]) for season in seasons}
    combinations = len(seasons) ** len(seasons)
    if combinations <= 100_000:
        draws: Iterable[tuple[int, ...]] = itertools.product(
            seasons, repeat=len(seasons)
        )
        resampling = (
            f"exhaustive ordered season bootstrap: {len(seasons)}^{len(seasons)}"
        )
    else:
        import numpy as np

        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(seasons), size=(n_resamples, len(seasons)))
        draws = (tuple(seasons[int(index)] for index in row) for row in sampled)
        resampling = (
            f"deterministic season bootstrap ({n_resamples} draws; seed={seed})"
        )
    samples: list[float] = []
    for draw in draws:
        total = math.fsum(math.fsum(values[season]) for season in draw)
        count = sum(sizes[season] for season in draw)
        samples.append(total / count)
    ordered = sorted(samples)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    point_total = math.fsum(math.fsum(values[season]) for season in seasons)
    point_count = sum(sizes.values())
    return {
        "mean_delta": point_total / point_count,
        "central_95_interval": [quantile(0.025), quantile(0.975)],
        "fraction_bootstrap_better": sum(value < 0 for value in samples) / len(samples),
        "n_team_seasons": point_count,
        "n_season_clusters": len(seasons),
        "n_resamples": len(samples),
        "resampling": resampling,
    }
