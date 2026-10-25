"""Reproducible 2026 replay for the issue 242 Quality of Results study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from gippyrank.artifact_hashes import included_game_rows_sha256
from gippyrank.performance_snapshot import (
    _pmfs,
    validate_context_source,
    validate_performance_against_context,
)
from gippyrank.posterior.snapshots import (
    Snapshot,
    _load_frozen_evidence,
    load_teams,
    sha256,
)
from gippyrank.quality_of_record_snapshot import (
    _load_source_likelihood,
    validate_quality_of_record_against_context,
)
from gippyrank.research.quality_of_results import (
    infer_quality_of_results,
    validate_margin_weight,
)

MARGIN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MARSHALL_SWEEP_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
OREGON_COUNTERFACTUAL_MARGIN = 28
BAD_CAMPBELL_FIXTURE_ID = "401866625"
REPAIRED_CAMPBELL_GAME_ID = "401917058"
ENDPOINT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class PublishedInput:
    slot_id: str
    display_label: str
    publication_order: int
    context_path: Path
    quality_of_play_path: Path
    quality_of_record_path: Path


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_pmf_artifact(path: Path, per_weight: dict[float, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", compresslevel=9, mtime=0
        ) as compressed_handle,
        io.TextIOWrapper(compressed_handle, encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(
            handle,
            fieldnames=["margin_weight", "team_id", "rank", "probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for weight in MARGIN_WEIGHTS:
            for team_id, pmf in sorted(per_weight[weight]["pmfs"].items()):
                for rank, probability in enumerate(pmf, 1):
                    writer.writerow(
                        {
                            "margin_weight": weight,
                            "team_id": team_id,
                            "rank": rank,
                            "probability": float(probability),
                        }
                    )


def _published_inputs(
    root: Path, slots: list[str] | None = None
) -> list[PublishedInput]:
    config_path = root / "site/publish_config.json"
    config = _read_json(config_path)
    statuses = {
        str(slot["id"]): str(slot["status"])
        for slot in config.get("publication_slots", [])
    }
    order = {slot_id: index for index, slot_id in enumerate(statuses)}
    requested = set(slots) if slots is not None else None
    selected: dict[str, PublishedInput] = {}
    for publication in config.get("snapshots", []):
        slot_id = str(publication.get("publication_slot", ""))
        source = Path(str(publication.get("source", "")))
        if (
            statuses.get(slot_id) != "official"
            or (requested is not None and slot_id not in requested)
            or source.parts[-2:] != ("predictive", "context")
        ):
            continue
        context_path = source if source.is_absolute() else root / source
        metadata_path = context_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = _read_json(metadata_path)
        if metadata.get("snapshot_type") != "weekly":
            continue
        bundle_path = context_path.parent.parent
        candidate = PublishedInput(
            slot_id,
            str(publication.get("display_label", slot_id)),
            order[slot_id],
            context_path,
            bundle_path / "performance",
            bundle_path / "performance/quality_of_record",
        )
        if slot_id in selected:
            raise ValueError(
                f"official publication slot {slot_id} has multiple Context snapshots"
            )
        selected[slot_id] = candidate

    if requested is not None:
        missing = requested - selected.keys()
        if missing:
            raise ValueError(
                f"requested slots are not official weekly Context snapshots: {sorted(missing)}"
            )
    return sorted(selected.values(), key=lambda item: item.publication_order)


def _snapshot(path: Path) -> Snapshot:
    metadata = _read_json(path / "metadata.json")
    return Snapshot(str(metadata["snapshot_id"]), path, metadata)


def _pmf_summaries(pmf: np.ndarray) -> dict[str, float | int]:
    ranks = np.arange(1, len(pmf) + 1, dtype=float)
    expected_rank = float(np.dot(ranks, pmf))
    variance = float(np.dot((ranks - expected_rank) ** 2, pmf))
    positive = pmf > 0
    entropy_bits = float(-np.sum(pmf[positive] * np.log2(pmf[positive])))
    output: dict[str, float | int] = {
        "expected_rank": expected_rank,
        "median_rank": int(np.searchsorted(np.cumsum(pmf), 0.5, side="left") + 1),
        "mode_rank": int(np.argmax(pmf) + 1),
        "rank_sd": math.sqrt(max(variance, 0.0)),
        "entropy_bits": entropy_bits,
    }
    for level, tail in ((50, 0.25), (80, 0.10), (95, 0.025)):
        low = int(np.searchsorted(np.cumsum(pmf), tail, side="left") + 1)
        high = int(np.searchsorted(np.cumsum(pmf), 1 - tail, side="left") + 1)
        output[f"interval_{level}_low"] = low
        output[f"interval_{level}_high"] = high
        output[f"interval_{level}_width"] = high - low + 1
    return output


def _rank_order(pmfs: dict[str, np.ndarray], counts: dict[str, int]) -> list[str]:
    rated = [team_id for team_id in pmfs if counts.get(team_id, 0) > 0]
    summaries = {team_id: _pmf_summaries(pmfs[team_id]) for team_id in rated}
    return sorted(
        rated, key=lambda team_id: (summaries[team_id]["expected_rank"], team_id)
    )


def _display_ranks(order: list[str]) -> dict[str, int]:
    return {team_id: rank for rank, team_id in enumerate(order, 1)}


def _pairwise_disagreement(left: list[str], right: list[str]) -> tuple[int, float]:
    left_rank = _display_ranks(left)
    right_rank = _display_ranks(right)
    common = sorted(left_rank.keys() & right_rank.keys())
    total_pairs = len(common) * (len(common) - 1) // 2
    disagreements = 0
    for first, second in itertools.combinations(common, 2):
        if (left_rank[first] - left_rank[second]) * (
            right_rank[first] - right_rank[second]
        ) < 0:
            disagreements += 1
    return disagreements, disagreements / total_pairs if total_pairs else 0.0


def _ranking_comparison(
    candidate: list[str], baseline: list[str], threshold: int
) -> dict[str, int | float]:
    candidate_rank = _display_ranks(candidate)
    baseline_rank = _display_ranks(baseline)
    common = sorted(candidate_rank.keys() & baseline_rank.keys())
    disagreements, disagreement_rate = _pairwise_disagreement(candidate, baseline)
    candidate_top = set(candidate[:threshold])
    baseline_top = set(baseline[:threshold])
    return {
        "mean_abs_display_rank_change": float(
            np.mean(
                [abs(candidate_rank[team] - baseline_rank[team]) for team in common]
            )
        )
        if common
        else 0.0,
        "median_abs_display_rank_change": float(
            np.median(
                [abs(candidate_rank[team] - baseline_rank[team]) for team in common]
            )
        )
        if common
        else 0.0,
        "pairwise_disagreement_count": disagreements,
        "pairwise_disagreement_rate": disagreement_rate,
        f"top{threshold}_symmetric_difference": len(candidate_top ^ baseline_top),
        f"top{threshold}_overlap": len(candidate_top & baseline_top),
    }


def _mean_abs_expected_rank_change(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> float:
    common = sorted(left.keys() & right.keys())
    if not common:
        return 0.0
    left_summaries = {team_id: _pmf_summaries(left[team_id]) for team_id in common}
    right_summaries = {team_id: _pmf_summaries(right[team_id]) for team_id in common}
    return float(
        np.mean(
            [
                abs(
                    float(left_summaries[team_id]["expected_rank"])
                    - float(right_summaries[team_id]["expected_rank"])
                )
                for team_id in common
            ]
        )
    )


def _team_game_statistics(
    fbs_ids: set[str], games: list[Any], teams_by_id: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    stats: dict[str, dict[str, Any]] = {
        team_id: {
            "wins": 0,
            "losses": 0,
            "points_for": 0,
            "points_against": 0,
            "margins": [],
            "games": 0,
            "close_losses": 0,
            "elite_close_losses": 0,
            "opponent_qop_ranks": [],
        }
        for team_id in fbs_ids
    }
    by_team: dict[str, list[dict[str, Any]]] = {team_id: [] for team_id in fbs_ids}
    for game in games:
        home = game.home_id
        away = game.away_id
        margin = int(game.home_points) - int(game.away_points)
        for team_id, opponent, team_margin, points_for, points_against in (
            (home, away, margin, game.home_points, game.away_points),
            (away, home, -margin, game.away_points, game.home_points),
        ):
            if team_id not in fbs_ids:
                continue
            record = stats[team_id]
            record["games"] += 1
            record["wins"] += int(team_margin > 0)
            record["losses"] += int(team_margin < 0)
            record["points_for"] += int(points_for)
            record["points_against"] += int(points_against)
            record["margins"].append(team_margin)
            by_team[team_id].append(
                {
                    "game": game,
                    "opponent_id": opponent,
                    "opponent_name": teams_by_id[opponent].name,
                    "team_margin": team_margin,
                    "points_for": int(points_for),
                    "points_against": int(points_against),
                    "result": "W"
                    if team_margin > 0
                    else "L"
                    if team_margin < 0
                    else "T",
                }
            )
    return stats, by_team


def _select_case_studies(
    teams_by_id: dict[str, Any],
    games: list[Any],
    qop_pmfs: dict[str, np.ndarray],
    qor_pmfs: dict[str, np.ndarray],
    midpoint_pmfs: dict[str, np.ndarray],
    additional_cases: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fbs_ids = {
        team_id for team_id, team in teams_by_id.items() if team.subdivision == "fbs"
    }
    counts = Counter()
    for game in games:
        counts[game.home_id] += 1
        counts[game.away_id] += 1
    stats, by_team = _team_game_statistics(fbs_ids, games, teams_by_id)
    qop = {
        team_id: _pmf_summaries(pmf)["expected_rank"]
        for team_id, pmf in qop_pmfs.items()
    }
    qor = {
        team_id: _pmf_summaries(pmf)["expected_rank"]
        for team_id, pmf in qor_pmfs.items()
    }
    midpoint = {
        team_id: _pmf_summaries(pmf)["expected_rank"]
        for team_id, pmf in midpoint_pmfs.items()
    }
    for team_id, record in stats.items():
        record["record"] = f"{record['wins']}-{record['losses']}"
        record["average_margin"] = (
            float(np.mean(record["margins"])) if record["margins"] else 0.0
        )
        record["average_opponent_qop_rank"] = (
            float(
                np.mean(
                    [
                        qop[game["opponent_id"]]
                        for game in by_team[team_id]
                        if game["opponent_id"] in qop
                    ]
                )
            )
            if any(game["opponent_id"] in qop for game in by_team[team_id])
            else None
        )
        record["qop_expected_rank"] = qop.get(team_id)
        record["qor_expected_rank"] = qor.get(team_id)
        record["midpoint_expected_rank"] = midpoint.get(team_id)
        record["qor_minus_qop_expected_rank"] = (
            qor[team_id] - qop[team_id] if team_id in qor and team_id in qop else None
        )

    rated = sorted(team_id for team_id in fbs_ids if stats[team_id]["games"] > 0)
    selected: list[tuple[str, str]] = []

    strong_records = [
        team_id
        for team_id in rated
        if stats[team_id]["games"] >= 5
        and stats[team_id]["wins"] >= 5
        and stats[team_id]["average_margin"] <= 14
    ]
    if strong_records:
        selected.append(
            (
                "winning_record_vs_margin",
                max(
                    strong_records,
                    key=lambda team_id: (
                        (stats[team_id]["qop_expected_rank"] or 0)
                        - (stats[team_id]["qor_expected_rank"] or math.inf),
                        stats[team_id]["wins"],
                    ),
                ),
            )
        )

    elite_close_losses = []
    for team_id in rated:
        losses = [
            game
            for game in by_team[team_id]
            if game["team_margin"] < 0
            and abs(game["team_margin"]) <= 7
            and game["opponent_id"] in qop
            and qop[game["opponent_id"]] <= 25
        ]
        stats[team_id]["elite_close_losses"] = len(losses)
        if losses:
            elite_close_losses.append(team_id)
    if elite_close_losses:
        selected.append(
            (
                "competitive_losses_vs_elite",
                max(
                    elite_close_losses,
                    key=lambda team_id: (
                        stats[team_id]["elite_close_losses"],
                        -(stats[team_id]["qop_expected_rank"] or math.inf),
                    ),
                ),
            )
        )

    dominant_weaker_wins = [
        (team_id, game)
        for team_id in rated
        for game in by_team[team_id]
        if game["team_margin"] >= 28
        and game["opponent_id"] in qop
        and qop[game["opponent_id"]] >= 70
    ]
    if dominant_weaker_wins:
        team_id, _ = max(
            dominant_weaker_wins,
            key=lambda item: (
                item[1]["team_margin"],
                stats[item[0]]["average_opponent_qop_rank"] or 0,
            ),
        )
        selected.append(("dominant_win_vs_lower_ranked_opponent", team_id))

    outlier_candidates = []
    for team_id in rated:
        margins = [abs(value) for value in stats[team_id]["margins"]]
        if len(margins) > 1:
            maximum = max(margins)
            other_mean = (
                float(np.mean([value for value in margins if value != maximum]))
                if any(value != maximum for value in margins)
                else 0.0
            )
            stats[team_id]["largest_margin_outlier"] = maximum - other_mean
            outlier_candidates.append(team_id)
    if outlier_candidates:
        selected.append(
            (
                "single_margin_outlier",
                max(
                    outlier_candidates,
                    key=lambda team_id: (
                        stats[team_id].get("largest_margin_outlier", 0.0),
                        abs(stats[team_id]["qor_minus_qop_expected_rank"] or 0.0),
                    ),
                ),
            )
        )

    record_groups: dict[str, list[str]] = {}
    for team_id in rated:
        record_groups.setdefault(str(stats[team_id]["record"]), []).append(team_id)
    record_pairs = []
    for record, ids in record_groups.items():
        comparable = [
            team_id
            for team_id in ids
            if stats[team_id]["average_opponent_qop_rank"] is not None
        ]
        if len(comparable) < 2:
            continue
        pair = max(
            itertools.combinations(comparable, 2),
            key=lambda pair: abs(
                stats[pair[0]]["average_opponent_qop_rank"]
                - stats[pair[1]]["average_opponent_qop_rank"]
            ),
        )
        difference = abs(
            stats[pair[0]]["average_opponent_qop_rank"]
            - stats[pair[1]]["average_opponent_qop_rank"]
        )
        record_pairs.append((difference, record, pair))
    if record_pairs:
        _, record, pair = max(record_pairs)
        selected.extend(("same_record_schedule_quality", team_id) for team_id in pair)
    additional_cases = additional_cases or []
    selected.extend(
        (str(case["case"]), str(case["team_id"])) for case in additional_cases
    )
    additional_by_team = {str(case["team_id"]): case for case in additional_cases}

    case_rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    for case, team_id in selected:
        record = stats[team_id]
        if record["qop_expected_rank"] is None or record["qor_expected_rank"] is None:
            continue
        case_row: dict[str, object] = {
            "case": case,
            "team_id": team_id,
            "team_name": teams_by_id[team_id].name,
            "record": record["record"],
            "games": record["games"],
            "average_margin": record["average_margin"],
            "average_opponent_qop_rank": record["average_opponent_qop_rank"],
            "quality_of_record_expected_rank": record["qor_expected_rank"],
            "quality_of_results_050_expected_rank": record["midpoint_expected_rank"],
            "quality_of_play_expected_rank": record["qop_expected_rank"],
            "largest_margin_outlier": record.get("largest_margin_outlier"),
            "elite_close_losses": record["elite_close_losses"],
        }
        if case == "largest_adjacent_lambda_shift":
            case_row.update(additional_by_team[team_id])
        case_rows.append(case_row)
        for item in by_team[team_id]:
            game = item["game"]
            game_rows.append(
                {
                    "case": case,
                    "team_id": team_id,
                    "team_name": teams_by_id[team_id].name,
                    "opponent_id": item["opponent_id"],
                    "opponent_name": item["opponent_name"],
                    "opponent_subdivision": teams_by_id[
                        item["opponent_id"]
                    ].subdivision,
                    "result": item["result"],
                    "points_for": item["points_for"],
                    "points_against": item["points_against"],
                    "team_margin": item["team_margin"],
                    "opponent_quality_of_play_expected_rank": qop.get(
                        item["opponent_id"]
                    ),
                    "opponent_quality_of_record_expected_rank": qor.get(
                        item["opponent_id"]
                    ),
                    "game_id": game.game_id,
                }
            )
    return case_rows, game_rows


def _input_bundle(
    root: Path, published: PublishedInput
) -> tuple[
    Snapshot,
    Snapshot,
    Snapshot,
    list[Any],
    list[dict[str, str]],
    list[Any],
    dict[str, Any],
    dict[str, Any],
    list[Any],
    Any,
    Path,
]:
    context = _snapshot(published.context_path)
    quality_of_play = _snapshot(published.quality_of_play_path)
    quality_of_record = _snapshot(published.quality_of_record_path)
    validate_context_source(context, root)
    validate_performance_against_context(context, quality_of_play)
    validate_quality_of_record_against_context(context, quality_of_record, root=root)
    context_rows = _read_rows(context.directory / "included_games.csv")
    if context_rows != _read_rows(quality_of_play.directory / "included_games.csv"):
        raise ValueError(
            f"{published.slot_id}: Quality of Play evidence rows differ from Context"
        )

    source = context.metadata
    teams, team_rows, prior_path = load_teams(
        root, int(source["season"]), "context", str(source["prior_model_version"])
    )
    (
        source_metadata,
        games,
        included_rows,
        _excluded_lower,
        _provenance,
        _fcs_population,
        _fcs_population_source,
        _fcs_fallback_ids,
        _simulation_config,
        inference_config,
    ) = _load_frozen_evidence(
        source=context.directory,
        season=int(source["season"]),
        snapshot_type="weekly",
        teams=teams,
        team_rows=team_rows,
    )
    if included_rows != context_rows:
        raise ValueError(
            f"{published.slot_id}: frozen loader returned different evidence rows"
        )
    expected_rows_hash = source.get("included_game_rows_sha256")
    actual_rows_hash = included_game_rows_sha256(included_rows)
    if expected_rows_hash is not None and expected_rows_hash != actual_rows_hash:
        raise ValueError(f"{published.slot_id}: Context included-game hash mismatch")
    if source_metadata.get("snapshot_id") != context.snapshot_id:
        raise ValueError(f"{published.slot_id}: Context snapshot ID mismatch")
    if (
        inference_config.get("implementation")
        != "deterministic_damped_loopy_sum_product"
    ):
        raise ValueError(f"{published.slot_id}: unsupported inference implementation")
    likelihood = _load_source_likelihood(source, root)
    return (
        context,
        quality_of_play,
        quality_of_record,
        teams,
        included_rows,
        games,
        inference_config,
        source_metadata,
        [team for team in teams if team.subdivision == "fbs"],
        likelihood,
        prior_path,
    )


def _baseline_rankings(
    published: PublishedInput,
    teams: list[Any],
    games: list[Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    qop_pmfs = _pmfs(published.quality_of_play_path / "posterior_pmfs.csv")
    qor_pmfs = _pmfs(published.quality_of_record_path / "posterior_pmfs.csv")
    fbs_ids = {team.team_id for team in teams if team.subdivision == "fbs"}
    if set(qop_pmfs) != fbs_ids or set(qor_pmfs) != fbs_ids:
        raise ValueError(f"{published.slot_id}: baseline FBS PMF populations differ")
    counts: Counter[str] = Counter({team_id: 0 for team_id in fbs_ids})
    for game in games:
        counts[game.home_id] += 1
        counts[game.away_id] += 1
    frozen_counts = dict(counts)
    return qop_pmfs, qor_pmfs, frozen_counts


def _validate_endpoint(
    label: str,
    actual: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    tolerance: float = ENDPOINT_TOLERANCE,
) -> dict[str, object]:
    if actual.keys() != baseline.keys():
        raise ValueError(f"{label} endpoint has a different FBS population")
    errors = {
        team_id: float(np.max(np.abs(actual[team_id] - baseline[team_id])))
        for team_id in actual
    }
    maximum = max(errors.values(), default=0.0)
    if maximum > tolerance:
        worst = max(errors, key=errors.get)
        raise ValueError(
            f"{label} endpoint differs from the published baseline for {worst}: {maximum:g}"
        )
    return {
        "baseline": label,
        "max_absolute_pmf_difference": maximum,
        "mean_absolute_pmf_difference": float(np.mean(list(errors.values())))
        if errors
        else 0.0,
        "team_count": len(errors),
        "tolerance": tolerance,
        "passed": True,
    }


def _make_rank_rows(
    teams: list[Any],
    pmfs: dict[str, np.ndarray],
    counts: dict[str, int],
    snapshot: PublishedInput,
    weight: float,
) -> tuple[list[dict[str, object]], list[str]]:
    fbs = [team for team in teams if team.subdivision == "fbs"]
    rated_order = _rank_order(pmfs, counts)
    rated_ranks = _display_ranks(rated_order)
    rows: list[dict[str, object]] = []
    for team in fbs:
        summary = _pmf_summaries(pmfs[team.team_id])
        rows.append(
            {
                "publication_slot": snapshot.slot_id,
                "display_label": snapshot.display_label,
                "snapshot_id": snapshot.context_path.parent.parent.name,
                "margin_weight": weight,
                "team_id": team.team_id,
                "team_name": team.name,
                "eligible_games": counts.get(team.team_id, 0),
                "rated": counts.get(team.team_id, 0) > 0,
                "display_rank": rated_ranks.get(team.team_id, "NR"),
                **summary,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["rated"],
            row["display_rank"] if row["rated"] else math.inf,
            row["team_id"],
        )
    )
    return rows, rated_order


def _run_targeted_sensitivities(
    root: Path,
    published_inputs: list[PublishedInput],
    all_rank_rows: list[dict[str, object]],
    snapshot_summaries: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run two narrow follow-up experiments on frozen Week 5 and Week 9 inputs."""
    by_label = {item.display_label: item for item in published_inputs}
    if "Week 5" not in by_label or "Week 9" not in by_label:
        return [], []

    def load(published: PublishedInput) -> tuple[Any, ...]:
        return _input_bundle(root, published)

    oregon_input = load(by_label["Week 9"])
    (
        _,
        _,
        _,
        oregon_teams,
        _,
        oregon_games,
        oregon_config,
        _,
        oregon_fbs_teams,
        oregon_likelihood,
        _,
    ) = oregon_input
    oregon_published = by_label["Week 9"]
    oregon = next(team for team in oregon_teams if team.name.casefold() == "oregon")
    portland_state = next(
        team for team in oregon_teams if team.name.casefold() == "portland state"
    )
    blowout_index = next(
        index
        for index, game in enumerate(oregon_games)
        if {game.home_id, game.away_id} == {oregon.team_id, portland_state.team_id}
        and (
            game.home_points - game.away_points
            if game.home_id == oregon.team_id
            else game.away_points - game.home_points
        )
        == 84
    )
    original_game = oregon_games[blowout_index]
    oregon_is_home = original_game.home_id == oregon.team_id
    original_oregon_score = (
        original_game.home_points if oregon_is_home else original_game.away_points
    )
    original_opponent_score = (
        original_game.away_points if oregon_is_home else original_game.home_points
    )
    if original_oregon_score != 84 or original_opponent_score != 0:
        raise ValueError("Week 9 Oregon–Portland State source game is not 84–0")
    counterfactual_game = replace(
        original_game,
        home_points=35 if oregon_is_home else 7,
        away_points=7 if oregon_is_home else 35,
    )
    counterfactual_games = list(oregon_games)
    counterfactual_games[blowout_index] = counterfactual_game
    _, _, original_counts = _baseline_rankings(
        oregon_published, oregon_fbs_teams, oregon_games
    )
    _, _, counterfactual_counts = _baseline_rankings(
        oregon_published, oregon_fbs_teams, counterfactual_games
    )
    if original_counts != counterfactual_counts:
        raise AssertionError("Oregon counterfactual changed eligible-game counts")
    original_rows = {
        (
            str(row["publication_slot"]),
            float(row["margin_weight"]),
            str(row["team_id"]),
        ): row
        for row in all_rank_rows
    }
    oregon_rows: list[dict[str, object]] = []
    factor_surface_cache: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
    for weight in MARGIN_WEIGHTS:
        inference = infer_quality_of_results(
            oregon_teams,
            counterfactual_games,
            oregon_likelihood,
            weight,
            max_iterations=int(oregon_config["max_iterations"]),
            tolerance=float(oregon_config["tolerance"]),
            damping=float(oregon_config["damping"]),
            factor_surface_cache=factor_surface_cache,
        )
        cf_rows, _ = _make_rank_rows(
            oregon_teams,
            inference.pmfs,
            counterfactual_counts,
            oregon_published,
            weight,
        )
        original = original_rows[(oregon_published.slot_id, weight, oregon.team_id)]
        counterfactual = next(
            row for row in cf_rows if row["team_id"] == oregon.team_id
        )
        original_expected = float(original["expected_rank"])
        counterfactual_expected = float(counterfactual["expected_rank"])
        original_display = int(original["display_rank"])
        counterfactual_display = int(counterfactual["display_rank"])
        expected_change = counterfactual_expected - original_expected
        display_change = counterfactual_display - original_display
        if weight == 0.0 and (abs(expected_change) > 1e-10 or display_change != 0):
            raise AssertionError(
                "preserving Oregon's win must leave the Quality of Record baseline unchanged"
            )
        oregon_rows.append(
            {
                "experiment": "oregon_blowout_counterfactual",
                "publication_slot": oregon_published.slot_id,
                "display_label": oregon_published.display_label,
                "team_id": oregon.team_id,
                "team_name": oregon.name,
                "game_id": original_game.game_id,
                "original_score": f"{original_oregon_score}-{original_opponent_score}",
                "counterfactual_score": f"35-7 (margin {OREGON_COUNTERFACTUAL_MARGIN})",
                "margin_weight": weight,
                "original_expected_rank": original_expected,
                "counterfactual_expected_rank": counterfactual_expected,
                "expected_rank_change": expected_change,
                "original_display_rank": original_display,
                "counterfactual_display_rank": counterfactual_display,
                "display_rank_change": display_change,
                "expected_rank_change_pct_of_lambda1": None,
                "display_rank_change_pct_of_lambda1": None,
                "converged": inference.blended_context_posterior.converged,
                "iterations": inference.blended_context_posterior.iterations,
                "max_message_delta": inference.blended_context_posterior.max_message_delta,
            }
        )
    lambda1 = oregon_rows[-1]
    for row in oregon_rows:
        row["expected_rank_change_pct_of_lambda1"] = (
            100.0
            * abs(float(row["expected_rank_change"]))
            / abs(float(lambda1["expected_rank_change"]))
            if float(lambda1["expected_rank_change"]) != 0
            else None
        )
        row["display_rank_change_pct_of_lambda1"] = (
            100.0
            * abs(int(row["display_rank_change"]))
            / abs(int(lambda1["display_rank_change"]))
            if int(lambda1["display_rank_change"]) != 0
            else None
        )

    marshall_published = by_label["Week 5"]
    (
        _,
        _,
        _,
        marshall_teams,
        _,
        marshall_games,
        marshall_config,
        _,
        _,
        marshall_likelihood,
        _,
    ) = load(marshall_published)
    marshall = next(
        team for team in marshall_teams if team.name.casefold() == "marshall"
    )
    _, _, marshall_counts = _baseline_rankings(
        marshall_published,
        [team for team in marshall_teams if team.subdivision == "fbs"],
        marshall_games,
    )
    baseline_marshall = {
        float(row["margin_weight"]): row
        for row in all_rank_rows
        if row["publication_slot"] == marshall_published.slot_id
        and row["team_id"] == marshall.team_id
        and float(row["margin_weight"]) in {0.0, 0.25}
    }
    summary_by_weight = {
        float(row["margin_weight"]): row
        for row in snapshot_summaries
        if row["publication_slot"] == marshall_published.slot_id
    }
    marshall_rows: list[dict[str, object]] = []
    marshall_factor_cache: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
    previous_expected: float | None = None
    two_back_expected: float | None = None
    for weight in MARSHALL_SWEEP_WEIGHTS:
        if weight in baseline_marshall:
            rank_row = baseline_marshall[weight]
            summary = summary_by_weight[weight]
            converged = bool(summary["converged"])
            iterations = int(summary["iterations"])
            max_message_delta = float(summary["max_message_delta"])
        else:
            inference = infer_quality_of_results(
                marshall_teams,
                marshall_games,
                marshall_likelihood,
                weight,
                max_iterations=int(marshall_config["max_iterations"]),
                tolerance=float(marshall_config["tolerance"]),
                damping=float(marshall_config["damping"]),
                factor_surface_cache=marshall_factor_cache,
            )
            rank_rows, _ = _make_rank_rows(
                marshall_teams,
                inference.pmfs,
                marshall_counts,
                marshall_published,
                weight,
            )
            rank_row = next(
                row for row in rank_rows if row["team_id"] == marshall.team_id
            )
            converged = inference.blended_context_posterior.converged
            iterations = inference.blended_context_posterior.iterations
            max_message_delta = inference.blended_context_posterior.max_message_delta
        expected = float(rank_row["expected_rank"])
        increment = (
            expected - previous_expected if previous_expected is not None else None
        )
        previous_weight = (
            MARSHALL_SWEEP_WEIGHTS[len(marshall_rows) - 1] if marshall_rows else None
        )
        second_difference = (
            expected - 2 * previous_expected + two_back_expected
            if previous_expected is not None and two_back_expected is not None
            else None
        )
        marshall_rows.append(
            {
                "experiment": "marshall_fine_weight_sweep",
                "publication_slot": marshall_published.slot_id,
                "display_label": marshall_published.display_label,
                "team_id": marshall.team_id,
                "team_name": marshall.name,
                "margin_weight": weight,
                "expected_rank": expected,
                "display_rank": rank_row["display_rank"],
                "expected_rank_increment": increment,
                "expected_rank_slope_per_lambda": (
                    increment / (weight - previous_weight)
                    if increment is not None
                    else None
                ),
                "second_difference": second_difference,
                "converged": converged,
                "iterations": iterations,
                "max_message_delta": max_message_delta,
            }
        )
        two_back_expected, previous_expected = previous_expected, expected
    return oregon_rows, marshall_rows


def _snapshot_manifest(
    root: Path,
    published: PublishedInput,
    context: Snapshot,
    quality_of_play: Snapshot,
    quality_of_record: Snapshot,
    prior_path: Path,
    included_rows: list[dict[str, str]],
    source_metadata: dict[str, Any],
) -> dict[str, object]:
    game_ids = {row["id"] for row in included_rows}
    correction = source_metadata.get("historical_evidence_correction")
    correction_issue = correction.get("issue") if isinstance(correction, dict) else None
    return {
        "publication_slot": published.slot_id,
        "display_label": published.display_label,
        "context_snapshot_id": context.snapshot_id,
        "context_source": published.context_path.relative_to(root).as_posix(),
        "context_metadata_sha256": sha256(context.directory / "metadata.json"),
        "included_game_count": len(included_rows),
        "included_game_ids_sha256": source_metadata.get("included_game_ids_sha256"),
        "included_game_rows_sha256": included_game_rows_sha256(included_rows),
        "prior_artifact": prior_path.relative_to(root).as_posix(),
        "prior_artifact_sha256": sha256(prior_path),
        "historical_likelihood_version": source_metadata.get(
            "historical_likelihood_version"
        ),
        "historical_likelihood_sha256": source_metadata.get(
            "historical_likelihood_sha256"
        ),
        "posterior_inference_configuration": source_metadata.get(
            "posterior_inference_configuration"
        ),
        "quality_of_play_metadata_sha256": sha256(
            quality_of_play.directory / "metadata.json"
        ),
        "quality_of_record_metadata_sha256": sha256(
            quality_of_record.directory / "metadata.json"
        ),
        "campbell_fixture_absent": BAD_CAMPBELL_FIXTURE_ID not in game_ids,
        "campbell_replacement_game_present": REPAIRED_CAMPBELL_GAME_ID in game_ids,
        "historical_evidence_correction_issue": correction_issue,
    }


def _format_case_report(
    case_rows: list[dict[str, object]], game_rows: list[dict[str, object]]
) -> list[str]:
    lines: list[str] = []
    for case in case_rows:
        label = str(case.get("display_label", case.get("publication_slot", "")))
        source = f" at {label}" if label else ""
        opponent_rank = case["average_opponent_qop_rank"]
        opponent_rank_text = (
            f"{float(opponent_rank):.1f}"
            if opponent_rank is not None
            else "unavailable"
        )
        if case["case"] == "largest_adjacent_lambda_shift":
            lines.append(
                f"- **Largest adjacent-λ shift — {case['team_name']} ({case['record']}){source}:** "
                f"from λ={float(case['previous_margin_weight']):g} to "
                f"λ={float(case['margin_weight']):g}, expected rank moved from "
                f"{float(case['previous_expected_rank']):.1f} to "
                f"{float(case['current_expected_rank']):.1f} "
                f"({float(case['expected_rank_change']):+.1f}); display rank moved "
                f"from {case['previous_display_rank']} to {case['current_display_rank']} "
                f"({int(case['display_rank_change']):+d})."
            )
        else:
            lines.append(
                f"- **{case['case']} — {case['team_name']} ({case['record']}){source}:** "
                f"expected rank is {case['quality_of_record_expected_rank']:.1f} under Quality of Record, "
                f"{case['quality_of_results_050_expected_rank']:.1f} at λ=0.50, and "
                f"{case['quality_of_play_expected_rank']:.1f} under Quality of Play. "
                f"Average scoring margin is {case['average_margin']:.1f}; the mean opponent "
                f"Quality of Play rank is {opponent_rank_text}."
            )
        examples = [row for row in game_rows if row["team_id"] == case["team_id"]]
        if case["case"] == "winning_record_vs_margin":
            fbs_wins = [
                row
                for row in examples
                if row["result"] == "W" and row["opponent_subdivision"] == "fbs"
            ]
            losses = [row for row in examples if row["result"] == "L"]
            examples = []
            if fbs_wins:
                examples.append(min(fbs_wins, key=lambda row: row["team_margin"]))
            if losses:
                examples.append(min(losses, key=lambda row: row["team_margin"]))
        elif case["case"] == "competitive_losses_vs_elite":
            examples = [
                row
                for row in examples
                if row["result"] == "L"
                and row["opponent_quality_of_play_expected_rank"] is not None
                and row["opponent_quality_of_play_expected_rank"] <= 25
                and abs(row["team_margin"]) <= 7
            ]
        elif case["case"] == "dominant_win_vs_lower_ranked_opponent":
            examples = [
                row
                for row in examples
                if row["result"] == "W"
                and row["team_margin"] >= 28
                and row["opponent_quality_of_play_expected_rank"] is not None
                and row["opponent_quality_of_play_expected_rank"] >= 70
            ]
        elif case["case"] == "single_margin_outlier" and examples:
            examples = [max(examples, key=lambda row: abs(row["team_margin"]))]
        elif case["case"] == "same_record_schedule_quality":
            fbs_games = [
                row
                for row in examples
                if row["opponent_subdivision"] == "fbs"
                and row["opponent_quality_of_play_expected_rank"] is not None
            ]
            if fbs_games:
                examples = [
                    min(
                        fbs_games,
                        key=lambda row: row["opponent_quality_of_play_expected_rank"],
                    ),
                    max(
                        fbs_games,
                        key=lambda row: row["opponent_quality_of_play_expected_rank"],
                    ),
                ]
        examples = list({str(row["game_id"]): row for row in examples}.values())
        if examples:

            def opponent_rank_text(row: dict[str, object]) -> str:
                value = row["opponent_quality_of_play_expected_rank"]
                return (
                    f"{float(value):.1f}" if value is not None else "not ranked in FBS"
                )

            game_text = "; ".join(
                f"{row['result']} {row['points_for']}–{row['points_against']} vs "
                f"{row['opponent_name']} (opponent Quality of Play rank {opponent_rank_text(row)})"
                for row in examples
            )
            lines.append(f"  Game evidence: {game_text}.")
    return lines


def _render_report(
    snapshot_summaries: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    case_rows: list[dict[str, object]],
    game_rows: list[dict[str, object]],
    endpoint_parity: list[dict[str, object]],
    oregon_sensitivity_rows: list[dict[str, object]],
    marshall_sweep_rows: list[dict[str, object]],
) -> str:
    weights = sorted({float(row["margin_weight"]) for row in sensitivity_rows})
    aggregate: dict[float, dict[str, float]] = {}
    for weight in weights:
        rows = [
            row for row in sensitivity_rows if float(row["margin_weight"]) == weight
        ]
        aggregate[weight] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in (
                "qor_pairwise_disagreement_rate",
                "qop_pairwise_disagreement_rate",
                "qor_mean_abs_display_rank_change",
                "qop_mean_abs_display_rank_change",
                "qor_mean_abs_expected_rank_change",
                "qop_mean_abs_expected_rank_change",
                "qor_top10_symmetric_difference",
                "qop_top10_symmetric_difference",
                "qor_top25_symmetric_difference",
                "qop_top25_symmetric_difference",
                "adjacent_mean_abs_display_rank_change",
                "adjacent_mean_abs_expected_rank_change",
                "adjacent_max_abs_expected_rank_change",
                "adjacent_pairwise_disagreement_rate",
                "adjacent_top10_symmetric_difference",
                "adjacent_top25_symmetric_difference",
                "mean_rank_sd",
                "mean_entropy_bits",
            )
        }
    lines = [
        "# Issue 242: Quality of Results research",
        "",
        "## Scope and method",
        "",
        "This is a research replay only. It does not activate or publish a third Performance ranking. For each completed game, the experimental factor is the binary winner likelihood raised to `1 − λ` times the existing scoring-margin likelihood raised to `λ`. Each λ is inferred as its own network posterior with the frozen Historical Likelihood V1, site effects, subdivision handling, selected Context prior, deterministic damped belief propagation, and the existing Performance prior-stripping method. The endpoints use the existing Quality of Record and Quality of Play factors directly; no completed rankings or posterior distributions are averaged.",
        "",
        f"The replay used {len({row['publication_slot'] for row in snapshot_summaries})} official weekly Context snapshots and {len(snapshot_summaries)} candidate inference runs, from {snapshot_summaries[0]['publication_slot']} through {snapshot_summaries[-1]['publication_slot']}, with candidate weights {', '.join(f'{weight:g}' for weight in weights)}. Each snapshot retained its own frozen eligible-game evidence and inference settings.",
        "",
        "The September 8 snapshot is the repaired issue 238 lineage: malformed completed 0–0 fixture `401866625` is excluded and its actual replacement `401917058` is retained. The runner checks that the malformed ID is absent from every selected snapshot and records whether the replacement game is present. Quality of Play and Quality of Record baseline artifacts are checked against the exact Context evidence before endpoint parity is measured.",
        "",
        "## Endpoint parity",
        "",
        "| Endpoint | Snapshots | Largest absolute PMF difference | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for endpoint in ("Quality of Record (λ=0)", "Quality of Play (λ=1)"):
        rows = [row for row in endpoint_parity if row["baseline"] == endpoint]
        maximum = max(
            (float(row["max_absolute_pmf_difference"]) for row in rows), default=0.0
        )
        lines.append(
            f"| {endpoint} | {len(rows)} | {maximum:.3g} | pass at tolerance {ENDPOINT_TOLERANCE:g} |"
        )
    lines.extend(
        [
            "",
            "## Candidate sensitivity across official snapshots",
            "",
            "Pairwise disagreement is the fraction of rated FBS team pairs whose order differs from the named endpoint, using expected-rank ordering. Top-10 and Top-25 changes are the average symmetric difference in membership. Rank changes use display rank. Uncertainty reports the average posterior rank standard deviation and entropy across rated teams.",
            "",
            "| λ | Pairwise disagreement vs QOR | vs QOP | Mean display-rank change vs QOR / QOP | Mean expected-rank change vs QOR / QOP | Top-10 changes vs QOR / QOP | Top-25 changes vs QOR / QOP | Adjacent display / expected-rank change | Adjacent pair disagreement | Mean rank SD | Mean entropy (bits) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for weight in weights:
        row = aggregate[weight]
        lines.append(
            f"| {weight:g} | {row['qor_pairwise_disagreement_rate']:.3%} | {row['qop_pairwise_disagreement_rate']:.3%} | "
            f"{row['qor_mean_abs_display_rank_change']:.2f} / {row['qop_mean_abs_display_rank_change']:.2f} | "
            f"{row['qor_mean_abs_expected_rank_change']:.2f} / {row['qop_mean_abs_expected_rank_change']:.2f} | "
            f"{row['qor_top10_symmetric_difference']:.2f} / {row['qop_top10_symmetric_difference']:.2f} | "
            f"{row['qor_top25_symmetric_difference']:.2f} / {row['qop_top25_symmetric_difference']:.2f} | "
            f"{row['adjacent_mean_abs_display_rank_change']:.2f} / {row['adjacent_mean_abs_expected_rank_change']:.2f} | "
            f"{row['adjacent_pairwise_disagreement_rate']:.3%} | "
            f"{row['mean_rank_sd']:.2f} | {row['mean_entropy_bits']:.2f} |"
        )
    adjacent_rows = [
        row
        for row in sensitivity_rows
        if row.get("adjacent_max_abs_expected_rank_change_team_id")
    ]
    if adjacent_rows:
        largest = max(
            adjacent_rows,
            key=lambda row: float(row["adjacent_max_abs_expected_rank_change"]),
        )
        lines.extend(
            [
                "",
                f"The largest team-level expected-rank movement between neighboring weights was {float(largest['adjacent_max_abs_expected_rank_change']):.1f} places for team `{largest['adjacent_max_abs_expected_rank_change_team_id']}` at {largest['publication_slot']} (λ={largest['previous_margin_weight']} to {largest['margin_weight']}). The full per-team trajectories are in `rankings.csv`.",
            ]
        )
        sensitivity_case = next(
            (
                case
                for case in case_rows
                if case["case"] == "largest_adjacent_lambda_shift"
            ),
            None,
        )
        if sensitivity_case is not None:
            shift_games = [
                row
                for row in game_rows
                if row["case"] == "largest_adjacent_lambda_shift"
            ]
            losses = [row for row in shift_games if row["result"] == "L"]
            largest_loss = max(
                losses, key=lambda row: abs(row["team_margin"]), default=None
            )
            win_count = len(shift_games) - len(losses)
            explanation = (
                f"This is meaningful sensitivity, not numerical non-convergence: every candidate run converged. "
                f"{sensitivity_case['team_name']} had a {sensitivity_case['record']} record in this cutoff "
                f"({win_count} win{'s' if win_count != 1 else ''} and "
                f"{len(losses)} loss{'es' if len(losses) != 1 else ''})."
            )
            if largest_loss is not None:
                explanation += (
                    f" Its largest loss was by {abs(int(largest_loss['team_margin']))} points to "
                    f"{largest_loss['opponent_name']}."
                )
            explanation += " The margin-weighted factor distinguishes results by score margin, while the binary factor counts each win or loss once."
            lines.append(explanation)
    lines.extend(
        [
            "",
            "Rank changes across λ are fully reported in `rankings.csv`; per-cutoff convergence and pairwise metrics are in `snapshot_summary.csv` and `sensitivity_summary.csv`.",
            "",
            "## Team-level case studies",
            "",
            f"The primary examples use the latest official cutoff ({snapshot_summaries[-1]['display_label']}); the largest adjacent-λ sensitivity case uses its own source cutoff. Game-level evidence follows each case; FCS opponents are shown but do not receive an FBS expected rank.",
            "",
        ]
    )
    lines.extend(_format_case_report(case_rows, game_rows))
    if oregon_sensitivity_rows:
        lines.extend(
            [
                "",
                "## Oregon blowout counterfactual",
                "",
                "Using the frozen Week 9 inputs, Oregon's 84–0 Portland State win was changed only for this sensitivity run to a 35–7 win (28-point margin). The matchup, Oregon win, priors, likelihood, all other games, and inference settings were held fixed. This is not a proposed score correction or a production ranking change.",
                "",
                "| λ | Original expected rank | Counterfactual expected rank | Change | Original display rank | Counterfactual display rank | Change | Expected-rank change as % of λ=1 | Display-rank change as % of λ=1 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in oregon_sensitivity_rows:
            expected_pct = row["expected_rank_change_pct_of_lambda1"]
            display_pct = row["display_rank_change_pct_of_lambda1"]
            lines.append(
                f"| {float(row['margin_weight']):g} | {float(row['original_expected_rank']):.2f} | "
                f"{float(row['counterfactual_expected_rank']):.2f} | {float(row['expected_rank_change']):+.2f} | "
                f"{row['original_display_rank']} | {row['counterfactual_display_rank']} | {int(row['display_rank_change']):+d} | "
                f"{float(expected_pct):.1f}% | {float(display_pct):.1f}% |"
            )
        full_weight = oregon_sensitivity_rows[-1]
        lines.append(
            f"\nThe λ=0 result is unchanged because Oregon still wins; margin evidence is absent at that endpoint. Relative to the λ=1 effect, the counterfactual movement at λ=0.25 is {float(oregon_sensitivity_rows[1]['expected_rank_change_pct_of_lambda1']):.1f}% for expected rank and {float(oregon_sensitivity_rows[1]['display_rank_change_pct_of_lambda1']):.1f}% for display rank. At λ=0.50 those shares are {float(oregon_sensitivity_rows[2]['expected_rank_change_pct_of_lambda1']):.1f}% and {float(oregon_sensitivity_rows[2]['display_rank_change_pct_of_lambda1']):.1f}%; at λ=0.75 they are {float(oregon_sensitivity_rows[3]['expected_rank_change_pct_of_lambda1']):.1f}% and {float(oregon_sensitivity_rows[3]['display_rank_change_pct_of_lambda1']):.1f}%. The λ=1 expected-rank and display-rank changes are {float(full_weight['expected_rank_change']):+.2f} and {int(full_weight['display_rank_change']):+d}. Thus lower intermediate weights materially reduce this blowout's ranking effect, though λ=0.75 remains close to λ=1. These observed differences quantify this game's influence on these rankings; they do not decide how wins and margins ought to be valued."
        )
    if marshall_sweep_rows:
        lines.extend(
            [
                "",
                "## Marshall fine weight sweep",
                "",
                "Using the same frozen Week 5 evidence, the λ interval from 0 to 0.25 was sampled in 0.05 steps. The added runs leave the full-grid results untouched; the endpoints reuse their Week 5 results from the main replay.",
                "",
                "| λ | Expected rank | Expected-rank change from prior step | Display rank | Second difference | Converged |",
                "| ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in marshall_sweep_rows:
            increment = row["expected_rank_increment"]
            curvature = row["second_difference"]
            increment_text = (
                f"{float(increment):+.2f}" if increment is not None else "—"
            )
            curvature_text = (
                f"{float(curvature):+.2f}" if curvature is not None else "—"
            )
            lines.append(
                f"| {float(row['margin_weight']):.2f} | {float(row['expected_rank']):.2f} | "
                f"{increment_text} | {row['display_rank']} | {curvature_text} | {row['converged']} |"
            )
        increments = [
            float(row["expected_rank_increment"])
            for row in marshall_sweep_rows
            if row["expected_rank_increment"] is not None
        ]
        second_differences = [
            abs(float(row["second_difference"]))
            for row in marshall_sweep_rows
            if row["second_difference"] is not None
        ]
        monotone = all(value >= 0 for value in increments) or all(
            value <= 0 for value in increments
        )
        lines.append(
            f"\nExpected rank is {'monotone' if monotone else 'not monotone'} across this sweep. The five 0.05-step expected-rank changes range from {min(increments):+.2f} to {max(increments):+.2f} places; the largest absolute second difference is {max(second_differences):.2f} places. These adjacent changes and curvature values assess trajectory shape separately from convergence. Marshall was 3–1: a 45-point loss at Penn State and three close wins (36–35 over Gardner-Webb, 28–26 over Middle Tennessee, 30–24 over Missouri State). The sweep shows how increasing margin weight gives that blowout loss more relative influence than the binary record does. This is an empirical response of the model; it does not determine the intended normative tradeoff."
        )
    lines.extend(
        [
            "",
            "## Interpretation and recommendation",
            "",
            "The candidate is a coherent one-pass inference model: λ=0.50 retains all winner evidence while tempering how strongly the observed margin differentiates rank pairs. This gives a direct normative knob between rewarding the win and crediting strong play in a loss. The model continues to use the opponent network, so schedule context is not replaced with a record bonus or a named-opponent adjustment. Large margins still contribute at intermediate λ, though their conditional evidence is tempered.",
            "",
            "The historical replay describes rank behavior; it does not establish which balance of wins and margins best represents season accomplishment. The issue explicitly makes λ a judgment about that meaning, and there is no objective target here that can select it without changing the question into poll imitation or predictive-margin optimization.",
            "",
            "The Oregon counterfactual shows that λ=0.25 and λ=0.50 substantially temper this extreme margin's effect; the Week 5 Marshall sweep is smooth at the tested 0.05 resolution. Neither sensitivity result selects a preferred λ: the choice still depends on how the ranking should value wins against margin evidence.",
            "",
            "**Recommendation:** retain Quality of Results as a research candidate and continue with a separate implementation/publication proposal only after the product owner chooses the intended win-versus-margin tradeoff. Treat λ=0.50 as a transparent central candidate for that discussion, not an empirically selected winner. Do not promote a third production variant from this study alone. A follow-up should preregister the interpretation and compare the documented game-level cases with that interpretation before publication.",
            "",
            "## Reproducibility",
            "",
            "The machine-readable manifest pins each publication slot, Context metadata, included game rows, prior, likelihood, and published endpoint artifacts. `rankings.csv` preserves rank and uncertainty summaries for every FBS team and λ; `posterior_pmfs/` retains the full distributions with deterministic gzip timestamps (`mtime=0`). `case_studies.csv` and `case_study_games.csv` provide selected team cases and underlying game rows. The two targeted experiments are in `oregon_blowout_counterfactual.csv` and `marshall_fine_weight_sweep.csv`. All candidate runs converged; see `snapshot_summary.csv` and the targeted outputs for convergence details.",
            "",
        ]
    )
    if any(row["campbell_fixture_absent"] is not True for row in snapshot_summaries):
        raise ValueError(
            "cannot report results while a malformed Campbell fixture is present"
        )
    if any(row["converged"] is not True for row in snapshot_summaries):
        raise ValueError(
            "cannot report results while an inference run did not converge"
        )
    return "\n".join(lines)


def run_experiment(
    root: Path,
    output_dir: Path | None = None,
    *,
    publication_slots: list[str] | None = None,
    margin_weights: tuple[float, ...] = MARGIN_WEIGHTS,
) -> dict[str, object]:
    """Replay official 2026 weekly inputs and write auditable research outputs."""
    weights = tuple(validate_margin_weight(weight) for weight in margin_weights)
    if weights != MARGIN_WEIGHTS:
        raise ValueError(
            f"margin_weights must equal the preregistered set {MARGIN_WEIGHTS}"
        )
    root = root.resolve()
    output_dir = output_dir or root / "data/research/issue_242_quality_of_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    published_inputs = _published_inputs(root, publication_slots)
    if not published_inputs:
        raise ValueError("no official weekly Context inputs were selected")

    all_rank_rows: list[dict[str, object]] = []
    snapshot_summaries: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    endpoint_parity: list[dict[str, object]] = []
    input_manifest: list[dict[str, object]] = []
    pmf_artifacts: list[dict[str, str]] = []
    latest_case_payload: (
        tuple[list[dict[str, object]], list[dict[str, object]]] | None
    ) = None
    largest_sensitivity_case: dict[str, object] | None = None
    largest_sensitivity_context: (
        tuple[
            PublishedInput,
            dict[str, Any],
            list[Any],
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            dict[str, np.ndarray],
        ]
        | None
    ) = None

    for published in published_inputs:
        (
            context,
            quality_of_play,
            quality_of_record,
            teams,
            included_rows,
            games,
            inference_config,
            source_metadata,
            fbs_teams,
            likelihood,
            prior_path,
        ) = _input_bundle(root, published)
        qop_pmfs, qor_pmfs, counts = _baseline_rankings(published, fbs_teams, games)
        fbs_ids = {team.team_id for team in fbs_teams}
        teams_by_id = {team.team_id: team for team in teams}
        game_ids = {row["id"] for row in included_rows}
        if BAD_CAMPBELL_FIXTURE_ID in game_ids:
            raise ValueError(
                f"{published.slot_id}: malformed Campbell fixture remains in included evidence"
            )
        if REPAIRED_CAMPBELL_GAME_ID not in game_ids:
            raise ValueError(
                f"{published.slot_id}: repaired Campbell–Western Carolina game is absent"
            )
        if included_game_rows_sha256(included_rows) != source_metadata.get(
            "included_game_rows_sha256", included_game_rows_sha256(included_rows)
        ):
            raise ValueError(f"{published.slot_id}: frozen game evidence hash mismatch")

        input_manifest.append(
            _snapshot_manifest(
                root,
                published,
                context,
                quality_of_play,
                quality_of_record,
                prior_path,
                included_rows,
                source_metadata,
            )
        )
        per_weight: dict[float, dict[str, Any]] = {}
        factor_surface_cache: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
        for weight in weights:
            inference = infer_quality_of_results(
                teams,
                games,
                likelihood,
                weight,
                max_iterations=int(inference_config["max_iterations"]),
                tolerance=float(inference_config["tolerance"]),
                damping=float(inference_config["damping"]),
                factor_surface_cache=factor_surface_cache,
            )
            pmfs = inference.pmfs
            if set(pmfs) != fbs_ids:
                raise ValueError(
                    f"{published.slot_id}: candidate FBS PMF population differs"
                )
            for team in fbs_teams:
                if counts[team.team_id] == 0 and not np.allclose(
                    pmfs[team.team_id],
                    np.full(len(team.prior), 1 / len(team.prior)),
                    atol=1e-14,
                    rtol=0,
                ):
                    raise ValueError(
                        f"{published.slot_id}: zero-game team is not uniform"
                    )
            rows, order = _make_rank_rows(teams, pmfs, counts, published, weight)
            all_rank_rows.extend(rows)
            per_weight[weight] = {
                "pmfs": pmfs,
                "order": order,
                "rows": rows,
                "converged": inference.blended_context_posterior.converged,
                "iterations": inference.blended_context_posterior.iterations,
                "max_message_delta": inference.blended_context_posterior.max_message_delta,
                "raw_game_factor_count": inference.blended_context_posterior.raw_game_factor_count,
                "unique_pair_factor_count": inference.blended_context_posterior.unique_pair_factor_count,
            }
        for weight_index, weight in enumerate(weights):
            current_rows = per_weight[weight]["rows"]
            previous_weight = weights[weight_index - 1] if weight_index else None
            previous_rows = (
                {row["team_id"]: row for row in per_weight[previous_weight]["rows"]}
                if previous_weight is not None
                else {}
            )
            for row in current_rows:
                previous = previous_rows.get(row["team_id"])
                row["previous_margin_weight"] = (
                    previous_weight if previous_weight is not None else ""
                )
                row["display_rank_change_from_previous"] = (
                    int(row["display_rank"]) - int(previous["display_rank"])
                    if previous is not None and row["rated"] and previous["rated"]
                    else ""
                )
                row["expected_rank_change_from_previous"] = (
                    float(row["expected_rank"]) - float(previous["expected_rank"])
                    if previous is not None
                    else ""
                )

        qop_parity = _validate_endpoint(
            "Quality of Play (λ=1)", per_weight[1.0]["pmfs"], qop_pmfs
        )
        qor_parity = _validate_endpoint(
            "Quality of Record (λ=0)", per_weight[0.0]["pmfs"], qor_pmfs
        )
        qop_parity["publication_slot"] = published.slot_id
        qor_parity["publication_slot"] = published.slot_id
        endpoint_parity.extend((qor_parity, qop_parity))
        pmf_path = output_dir / "posterior_pmfs" / f"{published.slot_id}.csv.gz"
        _write_pmf_artifact(pmf_path, per_weight)
        pmf_artifacts.append(
            {
                "publication_slot": published.slot_id,
                "path": pmf_path.relative_to(output_dir).as_posix(),
                "sha256": sha256(pmf_path),
            }
        )

        baseline_orders = {
            0.0: _rank_order(qor_pmfs, counts),
            1.0: _rank_order(qop_pmfs, counts),
        }
        for weight_index, weight in enumerate(weights):
            data = per_weight[weight]
            order = data["order"]
            summaries = [_pmf_summaries(data["pmfs"][team_id]) for team_id in order]
            summary_row: dict[str, object] = {
                "publication_slot": published.slot_id,
                "display_label": published.display_label,
                "margin_weight": weight,
                "included_game_count": len(games),
                "campbell_fixture_absent": BAD_CAMPBELL_FIXTURE_ID not in game_ids,
                "campbell_replacement_game_present": REPAIRED_CAMPBELL_GAME_ID
                in game_ids,
                "rated_fbs_count": len(order),
                "unrated_fbs_count": len(fbs_ids - set(order)),
                "raw_game_factor_count": data["raw_game_factor_count"],
                "unique_pair_factor_count": data["unique_pair_factor_count"],
                "converged": data["converged"],
                "iterations": data["iterations"],
                "max_message_delta": data["max_message_delta"],
                "mean_rank_sd": float(np.mean([item["rank_sd"] for item in summaries]))
                if summaries
                else 0.0,
                "mean_entropy_bits": float(
                    np.mean([item["entropy_bits"] for item in summaries])
                )
                if summaries
                else 0.0,
                "mean_interval_80_width": float(
                    np.mean([item["interval_80_width"] for item in summaries])
                )
                if summaries
                else 0.0,
            }
            snapshot_summaries.append(summary_row)
            adjacent: dict[str, object] = {
                "previous_margin_weight": "",
                "adjacent_mean_abs_display_rank_change": 0.0,
                "adjacent_mean_abs_expected_rank_change": 0.0,
                "adjacent_max_abs_expected_rank_change": 0.0,
                "adjacent_max_abs_expected_rank_change_team_id": "",
                "adjacent_pairwise_disagreement_rate": 0.0,
                "adjacent_top10_symmetric_difference": 0,
                "adjacent_top25_symmetric_difference": 0,
            }
            if weight_index > 0:
                previous = per_weight[weights[weight_index - 1]]
                adjacent10 = _ranking_comparison(order, previous["order"], 10)
                adjacent25 = _ranking_comparison(order, previous["order"], 25)
                expected_rank_changes = {
                    team_id: abs(
                        float(_pmf_summaries(data["pmfs"][team_id])["expected_rank"])
                        - float(
                            _pmf_summaries(previous["pmfs"][team_id])["expected_rank"]
                        )
                    )
                    for team_id in sorted(data["pmfs"].keys() & previous["pmfs"].keys())
                }
                largest_change_team = max(
                    expected_rank_changes, key=expected_rank_changes.get
                )
                largest_change = expected_rank_changes[largest_change_team]
                if largest_sensitivity_case is None or largest_change > float(
                    largest_sensitivity_case["absolute_expected_rank_change"]
                ):
                    previous_row = next(
                        row
                        for row in previous["rows"]
                        if row["team_id"] == largest_change_team
                    )
                    current_row = next(
                        row
                        for row in data["rows"]
                        if row["team_id"] == largest_change_team
                    )
                    previous_display_rank = int(previous_row["display_rank"])
                    current_display_rank = int(current_row["display_rank"])
                    expected_rank_change = float(current_row["expected_rank"]) - float(
                        previous_row["expected_rank"]
                    )
                    largest_sensitivity_case = {
                        "case": "largest_adjacent_lambda_shift",
                        "team_id": largest_change_team,
                        "previous_margin_weight": weights[weight_index - 1],
                        "margin_weight": weight,
                        "previous_expected_rank": float(previous_row["expected_rank"]),
                        "current_expected_rank": float(current_row["expected_rank"]),
                        "expected_rank_change": expected_rank_change,
                        "absolute_expected_rank_change": largest_change,
                        "previous_display_rank": previous_display_rank,
                        "current_display_rank": current_display_rank,
                        "display_rank_change": current_display_rank
                        - previous_display_rank,
                    }
                    largest_sensitivity_context = (
                        published,
                        teams_by_id,
                        games,
                        qop_pmfs,
                        qor_pmfs,
                        per_weight[0.5]["pmfs"],
                    )
                adjacent.update(
                    {
                        "previous_margin_weight": weights[weight_index - 1],
                        "adjacent_mean_abs_display_rank_change": adjacent10[
                            "mean_abs_display_rank_change"
                        ],
                        "adjacent_mean_abs_expected_rank_change": _mean_abs_expected_rank_change(
                            data["pmfs"], previous["pmfs"]
                        ),
                        "adjacent_max_abs_expected_rank_change": expected_rank_changes[
                            largest_change_team
                        ],
                        "adjacent_max_abs_expected_rank_change_team_id": largest_change_team,
                        "adjacent_pairwise_disagreement_rate": adjacent10[
                            "pairwise_disagreement_rate"
                        ],
                        "adjacent_top10_symmetric_difference": adjacent10[
                            "top10_symmetric_difference"
                        ],
                        "adjacent_top25_symmetric_difference": adjacent25[
                            "top25_symmetric_difference"
                        ],
                    }
                )
            sensitivity_row: dict[str, object] = {
                "publication_slot": published.slot_id,
                "display_label": published.display_label,
                "margin_weight": weight,
                **adjacent,
                "mean_rank_sd": summary_row["mean_rank_sd"],
                "mean_entropy_bits": summary_row["mean_entropy_bits"],
            }
            for baseline_weight, baseline_name in ((0.0, "qor"), (1.0, "qop")):
                comparison = _ranking_comparison(
                    order, baseline_orders[baseline_weight], 10
                )
                comparison25 = _ranking_comparison(
                    order, baseline_orders[baseline_weight], 25
                )
                sensitivity_row.update(
                    {
                        f"{baseline_name}_mean_abs_display_rank_change": comparison[
                            "mean_abs_display_rank_change"
                        ],
                        f"{baseline_name}_median_abs_display_rank_change": comparison[
                            "median_abs_display_rank_change"
                        ],
                        f"{baseline_name}_pairwise_disagreement_count": comparison[
                            "pairwise_disagreement_count"
                        ],
                        f"{baseline_name}_pairwise_disagreement_rate": comparison[
                            "pairwise_disagreement_rate"
                        ],
                        f"{baseline_name}_top10_symmetric_difference": comparison[
                            "top10_symmetric_difference"
                        ],
                        f"{baseline_name}_top10_overlap": comparison["top10_overlap"],
                        f"{baseline_name}_top25_symmetric_difference": comparison25[
                            "top25_symmetric_difference"
                        ],
                        f"{baseline_name}_top25_overlap": comparison25["top25_overlap"],
                        f"{baseline_name}_mean_abs_expected_rank_change": _mean_abs_expected_rank_change(
                            data["pmfs"], per_weight[baseline_weight]["pmfs"]
                        ),
                    }
                )
            sensitivity_rows.append(sensitivity_row)
        latest = max(weights)
        latest_data = per_weight[0.5] if 0.5 in per_weight else per_weight[latest]
        latest_case_payload = _select_case_studies(
            teams_by_id,
            games,
            qop_pmfs,
            qor_pmfs,
            latest_data["pmfs"],
        )
        for case in latest_case_payload[0]:
            case["publication_slot"] = published.slot_id
            case["display_label"] = published.display_label
        for game in latest_case_payload[1]:
            game["publication_slot"] = published.slot_id
            game["display_label"] = published.display_label

    if len(endpoint_parity) != len(published_inputs) * 2:
        raise AssertionError("endpoint parity audit is incomplete")
    case_rows, game_rows = latest_case_payload or ([], [])
    if largest_sensitivity_case is not None and largest_sensitivity_context is not None:
        (
            source_publication,
            source_teams_by_id,
            source_games,
            source_qop_pmfs,
            source_qor_pmfs,
            source_midpoint_pmfs,
        ) = largest_sensitivity_context
        sensitivity_case_rows, sensitivity_game_rows = _select_case_studies(
            source_teams_by_id,
            source_games,
            source_qop_pmfs,
            source_qor_pmfs,
            source_midpoint_pmfs,
            additional_cases=[largest_sensitivity_case],
        )
        case_rows.extend(
            case
            for case in sensitivity_case_rows
            if case["case"] == "largest_adjacent_lambda_shift"
        )
        game_rows.extend(
            game
            for game in sensitivity_game_rows
            if game["case"] == "largest_adjacent_lambda_shift"
        )
        for case in case_rows:
            case.setdefault("publication_slot", source_publication.slot_id)
            case.setdefault("display_label", source_publication.display_label)
        for game in game_rows:
            if game["case"] == "largest_adjacent_lambda_shift":
                game["publication_slot"] = source_publication.slot_id
                game["display_label"] = source_publication.display_label
    oregon_sensitivity_rows, marshall_sweep_rows = _run_targeted_sensitivities(
        root, published_inputs, all_rank_rows, snapshot_summaries
    )
    _write_csv(
        output_dir / "rankings.csv",
        all_rank_rows,
        [
            "publication_slot",
            "display_label",
            "snapshot_id",
            "margin_weight",
            "team_id",
            "team_name",
            "eligible_games",
            "rated",
            "display_rank",
            "expected_rank",
            "median_rank",
            "mode_rank",
            "rank_sd",
            "entropy_bits",
            "interval_50_low",
            "interval_50_high",
            "interval_50_width",
            "interval_80_low",
            "interval_80_high",
            "interval_80_width",
            "interval_95_low",
            "interval_95_high",
            "interval_95_width",
            "previous_margin_weight",
            "display_rank_change_from_previous",
            "expected_rank_change_from_previous",
        ],
    )
    _write_csv(
        output_dir / "snapshot_summary.csv",
        snapshot_summaries,
        list(snapshot_summaries[0]),
    )
    sensitivity_fields = [
        "publication_slot",
        "display_label",
        "margin_weight",
        "previous_margin_weight",
        *[
            f"{baseline}_{metric}"
            for baseline in ("qor", "qop")
            for metric in (
                "mean_abs_display_rank_change",
                "median_abs_display_rank_change",
                "pairwise_disagreement_count",
                "pairwise_disagreement_rate",
                "top10_symmetric_difference",
                "top10_overlap",
                "top25_symmetric_difference",
                "top25_overlap",
                "mean_abs_expected_rank_change",
            )
        ],
        "adjacent_mean_abs_display_rank_change",
        "adjacent_mean_abs_expected_rank_change",
        "adjacent_max_abs_expected_rank_change",
        "adjacent_max_abs_expected_rank_change_team_id",
        "adjacent_pairwise_disagreement_rate",
        "adjacent_top10_symmetric_difference",
        "adjacent_top25_symmetric_difference",
        "mean_rank_sd",
        "mean_entropy_bits",
    ]
    _write_csv(
        output_dir / "sensitivity_summary.csv", sensitivity_rows, sensitivity_fields
    )
    _write_csv(
        output_dir / "case_studies.csv",
        case_rows,
        [
            "case",
            "publication_slot",
            "display_label",
            "team_id",
            "team_name",
            "record",
            "games",
            "average_margin",
            "average_opponent_qop_rank",
            "quality_of_record_expected_rank",
            "quality_of_results_050_expected_rank",
            "quality_of_play_expected_rank",
            "largest_margin_outlier",
            "elite_close_losses",
            "previous_margin_weight",
            "margin_weight",
            "previous_expected_rank",
            "current_expected_rank",
            "expected_rank_change",
            "absolute_expected_rank_change",
            "previous_display_rank",
            "current_display_rank",
            "display_rank_change",
        ],
    )
    _write_csv(
        output_dir / "case_study_games.csv",
        game_rows,
        [
            "case",
            "publication_slot",
            "display_label",
            "team_id",
            "team_name",
            "opponent_id",
            "opponent_name",
            "opponent_subdivision",
            "result",
            "points_for",
            "points_against",
            "team_margin",
            "opponent_quality_of_play_expected_rank",
            "opponent_quality_of_record_expected_rank",
            "game_id",
        ],
    )
    _write_csv(
        output_dir / "oregon_blowout_counterfactual.csv",
        oregon_sensitivity_rows,
        [
            "experiment",
            "publication_slot",
            "display_label",
            "team_id",
            "team_name",
            "game_id",
            "original_score",
            "counterfactual_score",
            "margin_weight",
            "original_expected_rank",
            "counterfactual_expected_rank",
            "expected_rank_change",
            "expected_rank_change_pct_of_lambda1",
            "original_display_rank",
            "counterfactual_display_rank",
            "display_rank_change",
            "display_rank_change_pct_of_lambda1",
            "converged",
            "iterations",
            "max_message_delta",
        ],
    )
    _write_csv(
        output_dir / "marshall_fine_weight_sweep.csv",
        marshall_sweep_rows,
        [
            "experiment",
            "publication_slot",
            "display_label",
            "team_id",
            "team_name",
            "margin_weight",
            "expected_rank",
            "display_rank",
            "expected_rank_increment",
            "expected_rank_slope_per_lambda",
            "second_difference",
            "converged",
            "iterations",
            "max_message_delta",
        ],
    )
    (output_dir / "endpoint_parity.json").write_text(
        json.dumps(endpoint_parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "issue": 242,
        "study": "research_only_quality_of_results",
        "schema_version": "1.0",
        "publication_config": "site/publish_config.json",
        "publication_config_sha256": sha256(root / "site/publish_config.json"),
        "publication_status": "official",
        "snapshot_type": "weekly",
        "candidate_margin_weights": list(weights),
        "candidate_margin_weights_sha256": _sha256_json(list(weights)),
        "factor_definition": "P(sign(m)|theta) ** (1-lambda) * p(m|theta) ** lambda",
        "inference": "one deterministic damped loopy sum-product posterior per weight and snapshot",
        "prior_method": "Performance prior stripping; uniform PMF for zero-game FBS teams",
        "convergence_required": True,
        "endpoint_tolerance": ENDPOINT_TOLERANCE,
        "campbell_fixture_id_excluded": BAD_CAMPBELL_FIXTURE_ID,
        "campbell_replacement_game_id": REPAIRED_CAMPBELL_GAME_ID,
        "snapshots": input_manifest,
        "posterior_pmf_artifacts": pmf_artifacts,
        "targeted_sensitivity_experiments": {
            "oregon_blowout_counterfactual": {
                "performed": bool(oregon_sensitivity_rows),
                "publication_slot": next(
                    (
                        item.slot_id
                        for item in published_inputs
                        if item.display_label == "Week 9"
                    ),
                    None,
                ),
                "actual_score": "84-0",
                "counterfactual_score": "35-7",
                "preserve_win": True,
                "margin_weights": list(MARGIN_WEIGHTS),
            },
            "marshall_fine_weight_sweep": {
                "performed": bool(marshall_sweep_rows),
                "publication_slot": next(
                    (
                        item.slot_id
                        for item in published_inputs
                        if item.display_label == "Week 5"
                    ),
                    None,
                ),
                "margin_weights": list(MARSHALL_SWEEP_WEIGHTS),
                "step_size": 0.05,
            },
        },
        "output_sha256": {
            name: sha256(output_dir / name)
            for name in (
                "rankings.csv",
                "snapshot_summary.csv",
                "sensitivity_summary.csv",
                "case_studies.csv",
                "case_study_games.csv",
                "oregon_blowout_counterfactual.csv",
                "marshall_fine_weight_sweep.csv",
                "endpoint_parity.json",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _render_report(
        snapshot_summaries,
        sensitivity_rows,
        case_rows,
        game_rows,
        endpoint_parity,
        oregon_sensitivity_rows,
        marshall_sweep_rows,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--publication-slot",
        action="append",
        dest="publication_slots",
        help="Limit replay to one or more official weekly publication slots; may be repeated.",
    )
    args = parser.parse_args()
    manifest = run_experiment(
        args.root, args.output_dir, publication_slots=args.publication_slots
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "snapshot_count": len(manifest["snapshots"]),
                "output_dir": str(
                    args.output_dir
                    or args.root / "data/research/issue_242_quality_of_results"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
