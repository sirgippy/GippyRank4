"""Run the provenance-labelled 2026 model-versus-CFBD-field retrospective.

The default is the requested historical reconstruction of History 1.1, Context
1.2, and Context 1.3. It discovers snapshots through the public static API,
keeps a durable raw cache of mutable CFBD line responses, and writes a derived
per-game table plus a Markdown report. CFBD's stored ``spread`` field is
intentionally labelled as a later/stored spread: the API has no per-observation
timestamp and this script does not upgrade it to a verified closing line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from textwrap import fill
from typing import Any

import httpx

# Codex workers commonly have a read-only user configuration directory.  Keep
# matplotlib's ephemeral font/cache files outside the repository while still
# respecting a caller-provided configuration location.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gippyrank-matplotlib")
)
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gippyrank.context_market import (
    COMPARISON_VARIANTS,
    ContextMarketError,
    ProviderChoice,
    PublicationChoice,
    actual_margin_metrics,
    alignment_metrics,
    build_per_game_rows,
    calibration_metrics,
    common_game_ids,
    completed_weeks,
    edge_bucket_metrics,
    edge_signal_metrics,
    edge_threshold_metrics,
    market_move_relationship,
    movement_metrics,
    paired_mae_difference,
    parse_market_lines,
    parse_timestamp,
    prediction_rows,
    primary_sample,
    provider_coverage,
    schedule_by_game_id,
    select_model_publication,
    select_primary_provider,
    select_same_provider_lines,
    week_start,
    weekly_resource_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "https://sirgippy.github.io/GippyRank4/api/v1"
DEFAULT_CFBD_LINES_URL = "https://api.collegefootballdata.com/lines"
VARIANT_BY_KEY = {variant.key: variant for variant in COMPARISON_VARIANTS}
VARIANT_KEYS = tuple(VARIANT_BY_KEY)

# These permanent pages were manually checked while establishing the source
# limitation below.  They are not imported as a second market feed: their role
# is limited to testing whether CFBD's stored provider line is reliably a true
# final pre-kickoff quote.  It is not.
CFBD_SPREAD_SPOT_CHECKS = (
    {
        "game": "Boston College at Cincinnati (401856777, Week 1)",
        "url": "https://theoddsgap.com/odds/college-football/boston-college-eagles-vs-cincinnati-bearcats",
        "finding": "CFBD's DraftKings and Bovada stored spreads both matched the archived last pre-kickoff quote.",
    },
    {
        "game": "SMU at Florida State (401858212, Week 1)",
        "url": "https://theoddsgap.com/odds/college-football/smu-mustangs-vs-florida-state-seminoles",
        "finding": "CFBD DraftKings showed SMU -3 while the archived DraftKings close was SMU -4.5; Bovada matched at SMU -3.",
    },
    {
        "game": "Oregon at Oklahoma State (401856782, Week 2)",
        "url": "https://theoddsgap.com/odds/college-football/oregon-ducks-vs-oklahoma-state-cowboys/2026-09-12",
        "finding": "CFBD DraftKings and Bovada values both differed from their archived per-book last pre-kickoff quotes.",
    },
    {
        "game": "Florida State at Alabama (401856685, Week 3)",
        "url": "https://theoddsgap.com/odds/college-football/florida-state-seminoles-vs-alabama-crimson-tide/2026-09-19",
        "finding": "CFBD DraftKings matched the archived close; CFBD Bovada differed by one point.",
    },
)

PER_GAME_COLUMNS = (
    "week",
    "game_id",
    "kickoff",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "away_classification",
    "home_classification",
    "neutral_site",
    "variant_key",
    "variant_label",
    "model_family",
    "model_version",
    "model_version_field",
    "snapshot_id",
    "publication_slot",
    "publication_status",
    "generation_timestamp",
    "effective_cutoff",
    "source_retrieved_at",
    "source_context_snapshot_id",
    "comparison_snapshot_id",
    "selection_policy",
    "is_reconstruction",
    "is_retrospective_artifact",
    "generated_after_target_week_start",
    "timing_classification",
    "source_prediction_state",
    "gippy_expected_home_margin",
    "gippy_median_home_margin",
    "gippy_margin_interval_50",
    "gippy_margin_interval_80",
    "gippy_margin_interval_95",
    "market_provider",
    "market_formatted_spread",
    "opening_home_margin",
    "closing_home_margin",
    "cfbd_later_home_margin",
    "actual_home_score",
    "actual_away_score",
    "actual_home_margin",
    "gippy_minus_open",
    "gippy_minus_close",
    "abs_gippy_minus_open",
    "abs_gippy_minus_close",
    "market_move",
    "open_distance_to_gippy",
    "close_distance_to_gippy",
    "market_movement_toward_gippy",
    "gippy_error_vs_actual",
    "open_error_vs_actual",
    "close_error_vs_actual",
    "gippy_edge_vs_open",
    "gippy_edge_vs_close",
    "actual_residual_vs_open",
    "actual_residual_vs_close",
    "is_primary_fbs_vs_fbs",
    "schedule_completed",
    "exclusion_reasons",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT / "data/raw/cfbd/games/2026.json",
        help="Cached CFBD schedule/results JSON with stable game ids.",
    )
    parser.add_argument(
        "--market-cache",
        type=Path,
        default=ROOT / "data/raw/cfbd/lines/2026.json",
        help="Raw CFBD /lines response. It is created only when absent.",
    )
    parser.add_argument(
        "--raw-api-dir",
        type=Path,
        default=ROOT / "data/raw/gippyrank/context_market",
        help="Cache root for static API inventory and selected weekly resources.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/context_market_2026",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/studies/context_market_2026.md",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="*",
        help="Completed regular-season weeks to analyze; defaults to all complete weeks.",
    )
    parser.add_argument(
        "--preferred-provider",
        default="DraftKings",
        help="Preferred book only if it has maximum same-provider open/later coverage.",
    )
    parser.add_argument(
        "--market-provider",
        help="Explicit one-book override; never causes per-game fallback selection.",
    )
    parser.add_argument(
        "--strict-pregame",
        action="store_true",
        help=(
            "Require publication before the first weekly kickoff and reject "
            "retrospective artifacts. The default is the requested labelled "
            "historical reconstruction."
        ),
    )
    parser.add_argument(
        "--refresh-api",
        action="store_true",
        help="Fetch new timestamped raw static-API files instead of the existing cache.",
    )
    parser.add_argument(
        "--refresh-market",
        action="store_true",
        help="Fetch a new timestamped CFBD /lines raw response instead of the cache.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule_payload = _read_json_array(args.schedule, label="CFBD schedule")
    schedule = schedule_by_game_id(schedule_payload, season=args.season)
    target_weeks = _target_weeks(args, schedule_payload)
    if not target_weeks:
        raise ContextMarketError("No completed regular-season weeks selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_url = f"{args.api_base.rstrip('/')}/publications.json"
    inventory_path = args.raw_api_dir / str(args.season) / "publications.json"
    inventory_payload, inventory_source = _load_or_fetch_json(
        inventory_url, inventory_path, refresh=args.refresh_api
    )
    publications = inventory_payload.get("publications")
    if not isinstance(publications, list):
        raise ContextMarketError(
            "Static publication inventory has no publications array"
        )
    publication_by_id = _index_publications(publications)

    market_payload, market_source = _load_or_fetch_market(args)
    market_rows = parse_market_lines(market_payload)
    selection_policy = "strict_pregame" if args.strict_pregame else "reconstruction"
    choices: list[PublicationChoice] = []
    selected_predictions: list[tuple[PublicationChoice, list[dict[str, object]]]] = []
    for week in target_weeks:
        starts = week_start(schedule_payload, season=args.season, week=week)
        for variant in COMPARISON_VARIANTS:
            choice = select_model_publication(
                publications,
                season=args.season,
                week=week,
                week_start=starts,
                variant=variant,
                selection_policy=selection_policy,
            )
            choices.append(choice)
            prediction_data: list[dict[str, object]] = []
            if choice.snapshot_id:
                publication = publication_by_id[choice.snapshot_id]
                weekly_payload, source = _load_week_payload(
                    args,
                    publication,
                    choice,
                    refresh=args.refresh_api,
                )
                inventory_source.setdefault("weekly_resources", []).append(source)
                prediction_data = prediction_rows(
                    weekly_payload,
                    choice=choice,
                    allow_nonfuture_state=choice.is_reconstruction,
                )
            selected_predictions.append((choice, prediction_data))

    primary_schedule_ids_by_week = {
        week: set(_primary_completed_game_ids(schedule.values(), [week]))
        for week in target_weeks
    }
    candidate_prediction_ids_by_week = {
        week: {
            str(prediction["game_id"])
            for choice, predictions in selected_predictions
            if choice.week == week
            for prediction in predictions
        }
        & primary_schedule_ids_by_week[week]
        for week in target_weeks
    }
    candidate_prediction_ids = set().union(*candidate_prediction_ids_by_week.values())
    coverage = provider_coverage(market_rows, candidate_prediction_ids)
    coverage_by_week = [
        {
            "week": week,
            "candidate_model_games": len(candidate_prediction_ids_by_week[week]),
            "providers": provider_coverage(
                market_rows, candidate_prediction_ids_by_week[week]
            ),
        }
        for week in target_weeks
    ]
    provider_choice = _choose_provider(
        args, coverage, market_rows, candidate_prediction_ids
    )
    selected_market = (
        select_same_provider_lines(market_rows, provider_choice.provider)
        if provider_choice.provider
        else {}
    )

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for choice, prediction_data in selected_predictions:
        rows.extend(
            build_per_game_rows(
                prediction_data,
                schedule=schedule,
                market=selected_market,
                choice=choice,
                provider=provider_choice.provider,
            )
        )
        diagnostics.append(
            _diagnostic_row(
                choice,
                schedule=schedule,
                predictions=prediction_data,
                provider_choice=provider_choice,
                market=selected_market,
            )
        )

    frame = _per_game_frame(rows)
    summary = _build_summary(
        season=args.season,
        target_weeks=target_weeks,
        choices=choices,
        diagnostics=diagnostics,
        coverage=coverage,
        coverage_by_week=coverage_by_week,
        provider_choice=provider_choice,
        frame=frame,
        inventory_source=inventory_source,
        market_source=market_source,
        bootstrap_samples=args.bootstrap_samples,
        selection_policy=selection_policy,
        candidate_prediction_ids_by_week=candidate_prediction_ids_by_week,
    )
    _write_outputs(args.output_dir, frame, diagnostics, choices, summary)
    plot_paths = _write_plots(args.output_dir / "plots", frame, summary)
    summary["plots"] = plot_paths
    _write_json(args.output_dir / "summary.json", summary)
    _write_report(args.report, summary, frame, args.output_dir)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "per_game_dataset": str(args.output_dir / "per_game.csv"),
                "summary": str(args.output_dir / "summary.json"),
                "primary_provider": provider_choice.provider,
                "selected_rows": len(frame),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _target_weeks(
    args: argparse.Namespace, schedule: list[Mapping[str, Any]]
) -> list[int]:
    available = completed_weeks(schedule, season=args.season)
    if args.weeks is None:
        return available
    requested = sorted(set(args.weeks))
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ContextMarketError(
            f"Requested weeks are not complete in the supplied schedule: {missing}"
        )
    return requested


def _index_publications(
    publications: Sequence[object],
) -> dict[str, Mapping[str, Any]]:
    """Index a static inventory without silently overwriting duplicate ids."""
    indexed: dict[str, Mapping[str, Any]] = {}
    for publication in publications:
        if not isinstance(publication, Mapping):
            continue
        snapshot_id = publication.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            continue
        existing = indexed.get(snapshot_id)
        if existing is not None and existing != publication:
            raise ContextMarketError(
                f"Inventory has conflicting records for snapshot {snapshot_id!r}"
            )
        indexed[snapshot_id] = publication
    return indexed


def _primary_completed_game_ids(
    games: Iterable[Mapping[str, Any]], weeks: Sequence[int]
) -> set[str]:
    selected: set[str] = set()
    for game in games:
        if game.get("week") not in weeks or not bool(game.get("completed")):
            continue
        home = str(game.get("homeClassification", "")).casefold()
        away = str(game.get("awayClassification", "")).casefold()
        if home == "fbs" and away == "fbs":
            selected.add(str(game["id"]))
    return selected


def _choose_provider(
    args: argparse.Namespace,
    coverage: list[dict[str, object]],
    market_rows: list[dict[str, object]],
    game_ids: set[str],
) -> ProviderChoice:
    if args.market_provider:
        selected = next(
            (row for row in coverage if row["provider"] == args.market_provider),
            None,
        )
        if selected is None:
            raise ContextMarketError(
                f"Requested provider {args.market_provider!r} has no primary-sample lines"
            )
        return ProviderChoice(
            provider=args.market_provider,
            comparable_game_count=int(selected["same_provider_open_later_games"]),
            eligible_game_count=len(game_ids),
            reason="User-selected provider override; no per-game provider fallback is used.",
        )
    return select_primary_provider(
        market_rows,
        game_ids,
        preferred_provider=args.preferred_provider,
    )


def _load_week_payload(
    args: argparse.Namespace,
    publication: Mapping[str, Any],
    choice: PublicationChoice,
    *,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, object]]:
    if choice.snapshot_id is None:
        raise ContextMarketError("Cannot fetch a week for a missing publication")
    weeks_relative = publication.get("links", {}).get("weeks")
    if not isinstance(weeks_relative, str):
        raise ContextMarketError(f"Publication {choice.snapshot_id} lacks a weeks link")
    weeks_url = _absolute_api_url(args.api_base, weeks_relative)
    safe_snapshot = _safe_path_component(choice.snapshot_id)
    weeks_cache = (
        args.raw_api_dir / str(args.season) / "weeks" / safe_snapshot / "index.json"
    )
    weeks_payload, index_source = _load_or_fetch_json(
        weeks_url, weeks_cache, refresh=refresh
    )
    summaries = weeks_payload.get("weeks")
    if not isinstance(summaries, list):
        raise ContextMarketError(
            f"Week index for {choice.snapshot_id} has no weeks array"
        )
    summary = next(
        (
            row
            for row in summaries
            if isinstance(row, Mapping) and int(row.get("week", -1)) == choice.week
        ),
        None,
    )
    if summary is None or not isinstance(summary.get("key"), (str, int)):
        raise ContextMarketError(
            f"Snapshot {choice.snapshot_id} has no resource for week {choice.week}"
        )
    relative = weekly_resource_path(publication, str(summary["key"]))
    weekly_url = _absolute_api_url(args.api_base, relative)
    weekly_cache = weeks_cache.with_name(f"week-{summary['key']}.json")
    weekly_payload, weekly_source = _load_or_fetch_json(
        weekly_url, weekly_cache, refresh=refresh
    )
    return weekly_payload, {"week_index": index_source, "week_resource": weekly_source}


def _absolute_api_url(base: str, relative: str) -> str:
    return f"{base.rstrip('/')}/{relative.lstrip('/')}"


def _load_or_fetch_market(
    args: argparse.Namespace,
) -> tuple[list[Mapping[str, Any]], dict[str, object]]:
    if args.market_cache.is_file() and not args.refresh_market:
        return _read_json_array(
            args.market_cache, label="CFBD lines"
        ), _read_provenance(args.market_cache)
    target = (
        _fresh_target(args.market_cache)
        if args.market_cache.exists()
        else args.market_cache
    )
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise ContextMarketError(
            "CFBD_API_KEY is required to acquire CFBD /lines; provide a raw cache instead."
        )
    params = {"year": args.season, "seasonType": "regular"}
    with httpx.Client(timeout=60) as client:
        response = client.get(
            DEFAULT_CFBD_LINES_URL,
            params=params,
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ContextMarketError("CFBD /lines response is not an array")
    metadata = {
        "content_sha256": hashlib.sha256(response.content).hexdigest(),
        "endpoint": "/lines",
        "parameters": params,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_kind": "cfbd_api_lines",
        "semantics": {
            "spread_open": "CFBD source-provided opening field; first-market semantics are not independently established.",
            "spread": "CFBD later/stored field; no observation history or timestamp establishes it as the final pre-kickoff closing quote.",
        },
    }
    _write_raw_response(target, response.content, metadata)
    return payload, {"path": _relative_or_absolute(target), **metadata}


def _load_or_fetch_json(
    url: str, path: Path, *, refresh: bool
) -> tuple[dict[str, Any], dict[str, object]]:
    if path.is_file() and not refresh:
        return _read_json_object(path, label=url), _read_provenance(path)
    target = _fresh_target(path) if path.exists() else path
    with httpx.Client(timeout=60) as client:
        response = client.get(url)
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ContextMarketError(f"Expected JSON object from {url}")
    metadata = {
        "content_sha256": hashlib.sha256(response.content).hexdigest(),
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_kind": "gippyrank_static_api",
    }
    _write_raw_response(target, response.content, metadata)
    return payload, {"path": _relative_or_absolute(target), **metadata}


def _fresh_target(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.stem}-{timestamp}{path.suffix}")


def _write_raw_response(
    path: Path, content: bytes, metadata: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    provenance_path = path.with_name(f"{path.name}.provenance.json")
    provenance_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_provenance(path: Path) -> dict[str, object]:
    provenance_path = path.with_name(f"{path.name}.provenance.json")
    if not provenance_path.is_file():
        return {
            "path": _relative_or_absolute(path),
            "provenance": "not available for this pre-existing cache",
        }
    value = _read_json_object(provenance_path, label="provenance")
    return {"path": _relative_or_absolute(path), **value}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContextMarketError(
            f"Could not read {label} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ContextMarketError(f"{label} at {path} must be a JSON object")
    return value


def _read_json_array(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContextMarketError(
            f"Could not read {label} at {path}: {error}"
        ) from error
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ContextMarketError(f"{label} at {path} must be an array of JSON objects")
    return value


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _diagnostic_row(
    choice: PublicationChoice,
    *,
    schedule: Mapping[str, Mapping[str, Any]],
    predictions: list[dict[str, object]],
    provider_choice: ProviderChoice,
    market: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    expected = [
        game
        for game in schedule.values()
        if game.get("week") == choice.week
        and str(game.get("homeClassification", "")).casefold() == "fbs"
        and str(game.get("awayClassification", "")).casefold() == "fbs"
        and bool(game.get("completed"))
    ]
    primary_predictions = [
        prediction
        for prediction in predictions
        if str(prediction.get("prediction_home_subdivision", "")).casefold() == "fbs"
        and str(prediction.get("prediction_away_subdivision", "")).casefold() == "fbs"
    ]
    predicted_ids = {str(prediction["game_id"]) for prediction in primary_predictions}
    opening_ids = {
        game_id
        for game_id in predicted_ids
        if market.get(game_id, {}).get("opening_home_margin") is not None
    }
    later_ids = {
        game_id
        for game_id in predicted_ids
        if market.get(game_id, {}).get("later_home_margin") is not None
    }
    same_provider_ids = opening_ids & later_ids
    final_ids = {
        str(game["id"])
        for game in expected
        if game.get("homePoints") is not None and game.get("awayPoints") is not None
    }
    return {
        **choice.as_dict(),
        "expected_fbs_vs_fbs_games": len(expected),
        "games_with_gippy_prediction": len(predicted_ids),
        "games_with_opening_line": len(opening_ids),
        "games_with_cfbd_later_line": len(later_ids),
        "games_with_same_provider_open_later": len(same_provider_ids),
        "games_with_final_score": len(final_ids & predicted_ids),
        "primary_retained": len(same_provider_ids & final_ids),
        "source_future_prediction_rows": sum(
            prediction.get("source_prediction_state") == "future"
            for prediction in predictions
        ),
        "source_nonfuture_prediction_rows": sum(
            prediction.get("source_prediction_state") != "future"
            for prediction in predictions
        ),
        "market_provider": provider_choice.provider,
        "excluded_no_pregame_publication": 0 if choice.snapshot_id else len(expected),
        "excluded_no_gippy_prediction": len(
            {str(game["id"]) for game in expected} - predicted_ids
        ),
        "excluded_open_or_later_line": len(predicted_ids - same_provider_ids),
        "excluded_no_final_score": len(predicted_ids - final_ids),
    }


def _per_game_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=PER_GAME_COLUMNS)
    if frame.empty:
        return frame
    for column in (
        "week",
        "gippy_expected_home_margin",
        "gippy_median_home_margin",
        "opening_home_margin",
        "closing_home_margin",
        "cfbd_later_home_margin",
        "actual_home_score",
        "actual_away_score",
        "actual_home_margin",
        "gippy_minus_open",
        "gippy_minus_close",
        "abs_gippy_minus_open",
        "abs_gippy_minus_close",
        "market_move",
        "open_distance_to_gippy",
        "close_distance_to_gippy",
        "market_movement_toward_gippy",
        "gippy_error_vs_actual",
        "open_error_vs_actual",
        "close_error_vs_actual",
        "gippy_edge_vs_open",
        "gippy_edge_vs_close",
        "actual_residual_vs_open",
        "actual_residual_vs_close",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_primary_fbs_vs_fbs"] = frame["is_primary_fbs_vs_fbs"].astype(bool)
    return frame.sort_values(["variant_key", "week", "game_id"]).reset_index(drop=True)


def _build_summary(
    *,
    season: int,
    target_weeks: list[int],
    choices: list[PublicationChoice],
    diagnostics: list[dict[str, object]],
    coverage: list[dict[str, object]],
    coverage_by_week: list[dict[str, object]],
    provider_choice: ProviderChoice,
    frame: pd.DataFrame,
    inventory_source: dict[str, object],
    market_source: dict[str, object],
    bootstrap_samples: int,
    selection_policy: str,
    candidate_prediction_ids_by_week: Mapping[int, set[str]],
) -> dict[str, object]:
    by_version: dict[str, object] = {}
    weekly: dict[str, dict[str, object]] = {}
    cutoff_backed_by_version: dict[str, object] = {}
    cutoff_backed_frame = frame.loc[frame["effective_cutoff"].notna()].copy()
    for version in VARIANT_KEYS:
        version_frame = frame.loc[frame["variant_key"] == version].copy()
        by_version[version] = _sample_summary(version_frame, bootstrap_samples)
        cutoff_backed_by_version[version] = _sample_summary(
            cutoff_backed_frame.loc[
                cutoff_backed_frame["variant_key"] == version
            ].copy(),
            bootstrap_samples,
        )
        weekly[version] = {
            str(week): _sample_summary(
                version_frame.loc[version_frame["week"] == week].copy(),
                bootstrap_samples,
            )
            for week in target_weeks
        }

    common = _common_sample_summary(frame, bootstrap_samples)
    cutoff_backed_common = _common_sample_summary(
        cutoff_backed_frame, bootstrap_samples
    )
    return {
        "study": {
            "season": season,
            "completed_weeks": target_weeks,
            "generated_at": datetime.now(UTC).isoformat(),
            "comparison_variants": [
                {
                    "key": variant.key,
                    "label": variant.label,
                    "prior_family": variant.prior_family,
                    "model_version_field": variant.model_version_field,
                    "model_version": variant.model_version,
                    "anchor_context_version": variant.anchor_context_version,
                }
                for variant in COMPARISON_VARIANTS
            ],
            "selection_policy": selection_policy,
            "publication_timing_policy": (
                "Historical reconstruction: retain postgame-materialized artifacts "
                "when their weekly/live effective cutoff precedes the target week; "
                "preseason artifacts without a cutoff are retained but classified "
                "as unverified reconstructions."
                if selection_policy == "reconstruction"
                else "Strict pregame: require a non-retrospective artifact generated "
                "before the target week's first kickoff."
            ),
            "market_endpoint_policy": {
                "opening": "CFBD spreadOpen normalized to expected home margin; source first-quote semantics are not independently established.",
                "later": "CFBD spread normalized to expected home margin; no timestamp/history supports calling it a verified closing line.",
                "analysis_label": "cfbd_stored_later_spread",
            },
        },
        "sources": {
            "gippyrank_static_api": inventory_source,
            "cfbd_lines": market_source,
            "cfbd_spread_spot_checks": list(CFBD_SPREAD_SPOT_CHECKS),
        },
        "publication_selections": [choice.as_dict() for choice in choices],
        "data_quality": {
            "by_version_week": diagnostics,
            "provider_coverage": coverage,
            "provider_coverage_by_week": coverage_by_week,
            "primary_provider": provider_choice.as_dict(),
            "primary_rows_without_effective_cutoff": len(
                primary_sample(frame.loc[frame["effective_cutoff"].isna()].copy(), [])
            ),
            "candidate_model_game_population": {
                "unique_fbs_vs_fbs_games": len(
                    set().union(*candidate_prediction_ids_by_week.values())
                ),
                "by_week": [
                    {
                        "week": week,
                        "unique_fbs_vs_fbs_games": len(
                            candidate_prediction_ids_by_week[week]
                        ),
                    }
                    for week in target_weeks
                ],
            },
            "per_game_rows": len(frame),
        },
        "full_available_sample": by_version,
        "cutoff_backed_sample": cutoff_backed_by_version,
        "weekly": weekly,
        "common_game_paired_sample": common,
        "cutoff_backed_common_game_paired_sample": cutoff_backed_common,
    }


def _sample_summary(frame: pd.DataFrame, bootstrap_samples: int) -> dict[str, object]:
    primary = primary_sample(frame, [])
    return {
        "row_count": len(primary),
        "alignment": {
            "cfbd_opening": alignment_metrics(primary, "opening_home_margin"),
            "cfbd_stored_later": alignment_metrics(primary, "closing_home_margin"),
        },
        "calibration": {
            "cfbd_opening": calibration_metrics(primary, "opening_home_margin"),
            "cfbd_stored_later": calibration_metrics(primary, "closing_home_margin"),
        },
        "market_movement": {
            "distance_and_direction": movement_metrics(primary),
            "open_edge_vs_market_move": market_move_relationship(primary),
            "edge_buckets": edge_bucket_metrics(primary),
        },
        "actual_margin_accuracy": {
            "gippy": actual_margin_metrics(primary, "gippy_expected_home_margin"),
            "cfbd_opening": actual_margin_metrics(primary, "opening_home_margin"),
            "cfbd_stored_later": actual_margin_metrics(primary, "closing_home_margin"),
            "gippy_minus_open_paired_mae": paired_mae_difference(
                primary,
                "opening_home_margin",
                bootstrap_samples=bootstrap_samples,
            ),
            "gippy_minus_later_paired_mae": paired_mae_difference(
                primary,
                "closing_home_margin",
                bootstrap_samples=bootstrap_samples,
            ),
        },
        "incremental_signal": {
            "vs_open": edge_signal_metrics(primary, "open"),
            "vs_cfbd_stored_later": edge_signal_metrics(primary, "close"),
            "open_edge_thresholds": edge_threshold_metrics(primary, "open"),
            "later_edge_thresholds": edge_threshold_metrics(primary, "close"),
        },
    }


def _common_sample_summary(
    frame: pd.DataFrame, bootstrap_samples: int
) -> dict[str, object]:
    versions = list(VARIANT_KEYS)
    required = (
        "gippy_expected_home_margin",
        "opening_home_margin",
        "closing_home_margin",
        "actual_home_margin",
    )
    game_ids = common_game_ids(frame, versions, required=required)
    if not game_ids:
        return {
            "versions": versions,
            "common_game_count": 0,
            "reason": "At least one requested model variant has no primary-sample observations.",
            "by_version": {
                version: _sample_summary(frame.iloc[0:0], bootstrap_samples)
                for version in versions
            },
        }
    paired = frame.loc[frame["game_id"].astype(str).isin(game_ids)].copy()
    return {
        "versions": versions,
        "common_game_count": len(game_ids),
        "reason": None,
        "by_version": {
            version: _sample_summary(
                paired.loc[paired["variant_key"] == version], bootstrap_samples
            )
            for version in versions
        },
    }


def _write_outputs(
    output_dir: Path,
    frame: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    choices: list[PublicationChoice],
    summary: dict[str, object],
) -> None:
    csv_frame = frame.copy()
    for column in (
        "gippy_margin_interval_50",
        "gippy_margin_interval_80",
        "gippy_margin_interval_95",
    ):
        if column in csv_frame:
            csv_frame[column] = csv_frame[column].map(_json_cell)
    csv_frame.to_csv(output_dir / "per_game.csv", index=False)
    _write_json(output_dir / "per_game.json", _records_for_json(frame))
    pd.DataFrame(diagnostics).to_csv(output_dir / "data_quality.csv", index=False)
    _write_json(
        output_dir / "publication_selections.json",
        [choice.as_dict() for choice in choices],
    )
    largest = _largest_disagreements(frame)
    pd.DataFrame(largest).to_csv(output_dir / "largest_disagreements.csv", index=False)
    pd.DataFrame(_notable_market_events(frame)).to_csv(
        output_dir / "notable_cfbd_field_events.csv", index=False
    )
    _write_json(output_dir / "summary.json", summary)


def _json_cell(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math_is_nan(value)):
        return None
    return json.dumps(value, separators=(",", ":"))


def math_is_nan(value: float) -> bool:
    return bool(np.isnan(value))


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def _largest_disagreements(frame: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if frame.empty:
        return output
    primary = primary_sample(frame, [])
    for endpoint, column in (
        ("opening", "abs_gippy_minus_open"),
        ("cfbd_stored_later", "abs_gippy_minus_close"),
    ):
        valid = primary.dropna(subset=[column])
        for (version, week), group in valid.groupby(["variant_key", "week"]):
            for _, row in group.nlargest(10, column).iterrows():
                output.append(
                    {
                        "variant_key": version,
                        "week": int(week),
                        "endpoint": endpoint,
                        "game": f"{row['away_team']} at {row['home_team']}",
                        "game_id": row["game_id"],
                        "gippy_home_margin": row["gippy_expected_home_margin"],
                        "opening_home_margin": row["opening_home_margin"],
                        "cfbd_stored_later_home_margin": row["closing_home_margin"],
                        "actual_home_margin": row["actual_home_margin"],
                        "gippy_minus_open": row["gippy_minus_open"],
                        "gippy_minus_later": row["gippy_minus_close"],
                        "market_move": row["market_move"],
                        "market_movement_toward_gippy": row[
                            "market_movement_toward_gippy"
                        ],
                        "flags": _disagreement_flags(row),
                    }
                )
    return output


def _disagreement_flags(row: pd.Series) -> str:
    flags: list[str] = []
    gippy = row.get("gippy_expected_home_margin")
    opening = row.get("opening_home_margin")
    closing = row.get("closing_home_margin")
    movement = row.get("market_movement_toward_gippy")
    if _opposite_favorites(gippy, opening):
        flags.append("model/CFBD-opening favorite inversion")
    if _opposite_favorites(gippy, closing):
        flags.append("model/CFBD-stored favorite inversion")
    if _crossed_zero(opening, closing):
        if movement is not None and movement > 0:
            flags.append("CFBD fields crossed zero toward model")
        elif movement is not None and movement < 0:
            flags.append("CFBD fields crossed zero away from model")
        else:
            flags.append("CFBD fields crossed zero")
    if movement is not None and movement <= -3:
        flags.append("CFBD stored field moved ≥3 points away from model")
    return "; ".join(flags)


def _notable_market_events(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return every flagged endpoint event, rather than only top disagreements."""
    if frame.empty:
        return []
    events: list[dict[str, object]] = []
    for _, row in primary_sample(frame, []).iterrows():
        flags = _disagreement_flags(row)
        if not flags:
            continue
        events.append(
            {
                "variant_key": row["variant_key"],
                "variant_label": row["variant_label"],
                "week": int(row["week"]),
                "game_id": row["game_id"],
                "game": f"{row['away_team']} at {row['home_team']}",
                "kickoff": row["kickoff"],
                "timing_classification": row["timing_classification"],
                "gippy_expected_home_margin": row["gippy_expected_home_margin"],
                "opening_home_margin": row["opening_home_margin"],
                "cfbd_stored_later_home_margin": row["closing_home_margin"],
                "cfbd_field_delta": row["market_move"],
                "cfbd_field_change_toward_model": row["market_movement_toward_gippy"],
                "actual_home_margin": row["actual_home_margin"],
                "flags": flags,
            }
        )
    return events


def _opposite_favorites(left: object, right: object) -> bool:
    return left is not None and right is not None and float(left) * float(right) < 0


def _crossed_zero(left: object, right: object) -> bool:
    return left is not None and right is not None and float(left) * float(right) < 0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_plots(
    plot_dir: Path, frame: pd.DataFrame, summary: Mapping[str, object]
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_sample(frame, [])
    if primary.empty:
        return []
    all_rows_scope = (
        "all selected rows; includes no-cutoff reconstructions"
        if primary["effective_cutoff"].isna().any()
        else "all selected rows"
    )
    paths: list[Path] = []
    paths.extend(
        _scatter_plot(
            primary,
            x="opening_home_margin",
            y="gippy_expected_home_margin",
            xlabel="CFBD spreadOpen → expected home margin",
            ylabel="Gippy expected home margin",
            title=f"GippyRank vs CFBD opening field ({all_rows_scope})",
            output=plot_dir / "gippy_vs_open.png",
            identity=True,
        )
    )
    paths.extend(
        _scatter_plot(
            primary,
            x="closing_home_margin",
            y="gippy_expected_home_margin",
            xlabel="CFBD stored spread → expected home margin",
            ylabel="Gippy expected home margin",
            title=(
                "GippyRank vs CFBD stored later spread "
                f"(not verified close; {all_rows_scope})"
            ),
            output=plot_dir / "gippy_vs_cfbd_stored_later.png",
            identity=True,
        )
    )
    paths.extend(
        _scatter_plot(
            primary,
            x="gippy_edge_vs_open",
            y="market_move",
            xlabel="Gippy − CFBD opening field",
            ylabel="CFBD stored later − opening",
            title=f"Opening model edge vs later CFBD field ({all_rows_scope})",
            output=plot_dir / "open_edge_vs_market_move.png",
            identity=False,
            horizontal_zero=True,
            vertical_zero=True,
        )
    )
    paths.extend(
        _scatter_plot(
            primary,
            x="gippy_edge_vs_close",
            y="actual_residual_vs_close",
            xlabel="Gippy − CFBD stored later spread",
            ylabel="Actual margin − CFBD stored later spread",
            title=(
                f"Model/later-field edge vs actual margin residual ({all_rows_scope})"
            ),
            output=plot_dir / "later_edge_vs_actual_residual.png",
            identity=True,
        )
    )
    cutoff_backed = primary.loc[primary["effective_cutoff"].notna()].copy()
    if not cutoff_backed.empty:
        paths.extend(
            _scatter_plot(
                cutoff_backed,
                x="opening_home_margin",
                y="gippy_expected_home_margin",
                xlabel="CFBD spreadOpen → expected home margin",
                ylabel="Gippy expected home margin",
                title="GippyRank vs CFBD opening field (cutoff-backed Weeks 2–3)",
                output=plot_dir / "gippy_vs_open_cutoff_backed.png",
                identity=True,
            )
        )
        paths.extend(
            _scatter_plot(
                cutoff_backed,
                x="closing_home_margin",
                y="gippy_expected_home_margin",
                xlabel="CFBD stored spread → expected home margin",
                ylabel="Gippy expected home margin",
                title=(
                    "GippyRank vs CFBD stored later spread "
                    "(cutoff-backed Weeks 2–3; not verified close)"
                ),
                output=plot_dir / "gippy_vs_cfbd_stored_later_cutoff_backed.png",
                identity=True,
            )
        )
    weekly = summary.get("weekly")
    if isinstance(weekly, Mapping):
        path = _weekly_mae_plot(weekly, plot_dir / "weekly_margin_mae.png")
        if path:
            paths.append(path)
    return [_relative_or_absolute(path) for path in paths]


def _scatter_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    identity: bool,
    horizontal_zero: bool = False,
    vertical_zero: bool = False,
) -> list[Path]:
    data = frame.dropna(subset=[x, y])
    if data.empty:
        return []
    figure, axis = plt.subplots(figsize=(7, 5))
    for version, group in data.groupby("variant_key"):
        axis.scatter(group[x], group[y], alpha=0.75, label=_variant_label(str(version)))
    if identity:
        low = min(float(data[x].min()), float(data[y].min()))
        high = max(float(data[x].max()), float(data[y].max()))
        padding = max(1.0, (high - low) * 0.05)
        axis.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            "--",
            color="0.35",
        )
    if horizontal_zero:
        axis.axhline(0, color="0.45", linewidth=0.8)
    if vertical_zero:
        axis.axvline(0, color="0.45", linewidth=0.8)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=fill(title, width=62))
    axis.legend(title="Model variant")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return [output]


def _weekly_mae_plot(weekly: Mapping[str, object], output: Path) -> Path | None:
    records: list[dict[str, object]] = []
    for version, weeks in weekly.items():
        if not isinstance(weeks, Mapping):
            continue
        for week, value in weeks.items():
            if not isinstance(value, Mapping):
                continue
            accuracy = value.get("actual_margin_accuracy")
            if not isinstance(accuracy, Mapping):
                continue
            for label, key in (
                ("Gippy", "gippy"),
                ("CFBD opening", "cfbd_opening"),
                ("CFBD stored later", "cfbd_stored_later"),
            ):
                metrics = accuracy.get(key)
                if isinstance(metrics, Mapping) and metrics.get("mae") is not None:
                    records.append(
                        {
                            "variant_key": version,
                            "week": int(week),
                            "series": label,
                            "mae": float(metrics["mae"]),
                        }
                    )
    if not records:
        return None
    data = pd.DataFrame(records)
    figure, axis = plt.subplots(figsize=(7, 5))
    for (version, series), group in data.groupby(["variant_key", "series"]):
        group = group.sort_values("week")
        axis.plot(
            group["week"],
            group["mae"],
            marker="o",
            label=f"{_variant_label(str(version))} {series}",
        )
    axis.set(
        xlabel="Week",
        ylabel="Actual-margin MAE",
        title="Weekly actual-margin MAE by model variant and CFBD field (all selected)",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _write_report(
    report_path: Path,
    summary: Mapping[str, object],
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    study = _mapping(summary, "study")
    quality = _mapping(summary, "data_quality")
    selections = _sequence(summary, "publication_selections")
    full = _mapping(summary, "full_available_sample")
    cutoff_backed = _mapping(summary, "cutoff_backed_sample")
    weekly = _mapping(summary, "weekly")
    common = _mapping(summary, "common_game_paired_sample")
    cutoff_backed_common = _mapping(summary, "cutoff_backed_common_game_paired_sample")
    unverified_primary_rows = _number(
        quality.get("primary_rows_without_effective_cutoff"), digits=0
    )
    reconstruction = study.get("selection_policy") == "reconstruction"
    study_description = (
        "This is an explicitly labelled historical reconstruction. It asks how "
        "History 1.1, Context 1.2, and Context 1.3 artifacts would have scored "
        "against final margins, even when an artifact was materialized after the "
        "games. Generation timestamps are retained in every output and are not "
        "silently treated as pregame publication times."
        if reconstruction
        else "This is a strict pregame-publication study. It accepts only "
        "non-retrospective artifacts generated before the first kickoff in each "
        "target week."
    )
    timing_description = (
        "For reconstruction, weekly/live artifacts still need an effective cutoff "
        "before the target week's first kickoff. Week 1 preseason artifacts have "
        "no effective-cutoff timestamp, so they are retained as **unverified "
        "historical reconstructions**, not as contemporaneous predictions."
        if reconstruction
        else "Weekly/live artifacts must have an effective cutoff before the "
        "target week's first kickoff."
    )
    lines = [
        f"# 2026 model-variant versus CFBD-field retrospective ({study.get('season')})",
        "",
        "## Bottom line",
        "",
        study_description,
        "",
        "There is no Context 1.1 artifact: the baseline is correctly named **History 1.1** and selected from `prior_family=history`, `history_prior=1.1`, with its canonical Context 1.2 lineage. Context 1.3 Week 2/3 replays are included by request and visibly labelled as postgame-generated reconstructions.",
        "",
        "This analysis cannot establish true historical opening-to-closing market movement. CFBD supplies a mutable, timestamp-free `spread` per provider; spot checks show it does not reliably equal the last pre-kickoff quote. Thus `CFBD opening` and `CFBD stored later` are endpoint fields for diagnostics, not validated historical opening and closing lines.",
        "",
        "## Compact summary",
        "",
        "### All selected rows",
        "",
        _summary_table(full),
        "",
        f"The all-selected table reports FBS-v-FBS games only and includes {unverified_primary_rows} Week 1 variant-game rows (51 shared games across three variants) without an effective-cutoff timestamp. `CFBD later` is a stored-line diagnostic, not a closing-market claim; paired MAE is model absolute error minus the benchmark on identical games (negative favors the model).",
        "",
        "### Cutoff-backed sensitivity (Weeks 2–3)",
        "",
        _summary_table(cutoff_backed),
        "",
        "This sensitivity excludes the Week 1 preseason reconstruction rows with no effective-cutoff timestamp. It still includes Context 1.3’s postgame-materialized artifacts, which have an earlier effective cutoff and are labelled as reconstructions rather than contemporaneous publications.",
        "",
        "## Direct answers",
        "",
        *_direct_answer_lines(full, common),
        "",
        "## Artifact selection and timing provenance",
        "",
        timing_description,
        "",
        _publication_table(selections),
        "",
        "### Interpretation of timing classes",
        "",
        "- `published_before_target_week_start` is a contemporaneously generated source artifact.",
        "- `postgame_generated_reconstruction_with_pregame_cutoff` has a pregame evidence cutoff but was generated later; use it only for this counterfactual reconstruction.",
        "- `reconstruction_without_effective_cutoff` is the Week 1 preseason case: it is retained at the user's request but cannot prove what information was available before the first Week 1 kickoff.",
        "",
        "## CFBD endpoint fields and provider policy",
        "",
        "CFBD `/lines` records `formattedSpread`, `spreadOpen`, and `spread` under an exact provider display string but provides no provider id, observation timestamps, or historical sequence. `spread` is home-oriented in CFBD's sign convention (`Home -7` is `-7`), so this study negates it to the canonical expected home margin (`+7`). Pick'em remains zero. `DraftKings` and `Draft Kings` are retained as distinct raw provider names; they are never silently merged.",
        "",
        "The source-provided `spreadOpen` is called the **CFBD opening field** and `spread` the **CFBD stored later field**. Neither supplies enough evidence to answer a genuine first-quote, closing-line, or market-response question.",
        "",
        "For compatibility with the original requested schema, the per-game `closing_home_margin` column is a duplicate alias of `cfbd_later_home_margin`; despite that legacy name, it is **not** a verified closing line.",
        "",
        "### Candidate model-game population used for provider selection",
        "",
        _candidate_population_table(
            _mapping(quality, "candidate_model_game_population")
        ),
        "",
        _provider_table(
            _sequence(quality, "provider_coverage"),
            _mapping(quality, "primary_provider"),
        ),
        "",
        "### Provider coverage by model-game week",
        "",
        _provider_coverage_by_week_table(
            _sequence(quality, "provider_coverage_by_week"),
            _mapping(quality, "primary_provider"),
        ),
        "",
        "### Manual, non-reproducible CFBD spread observations",
        "",
        "The following manual observations were used only to test whether a closing-line interpretation was safe. The study did not capture archival quote payloads or retrieval timestamps, so they are **not** reproducible semantic verification and are not used as a second market feed. Mixed observations are enough to leave both CFBD field semantics unverified:",
        "",
        *_spot_check_lines(_mapping(summary, "sources")),
        "",
        "## Data-quality diagnostics",
        "",
        _diagnostic_table(_sequence(quality, "by_version_week")),
        "",
        "### Exclusion screens",
        "",
        _exclusion_table(_sequence(quality, "by_version_week")),
        "",
        "`primary retained` requires an FBS-v-FBS model prediction, same-book CFBD opening/stored-later fields, and a final score. Lower-division games remain outside the headline sample; all joins use stable CFBD game ids. The source game state and timing classification are retained per row for auditability.",
        "",
        "## CFBD-field alignment (a similarity question, not accuracy)",
        "",
        _alignment_table(full),
        "",
        _calibration_table(full),
        "",
        "## CFBD-field change diagnostic (not historical market movement)",
        "",
        _movement_table(full),
        "",
        _movement_relationship_table(full),
        "",
        _edge_bucket_tables(full),
        "",
        "A positive movement-toward value means the CFBD stored later field was closer to the model than the CFBD opening field. This is descriptive association only; it is not evidence of an actual market move, causal influence, or a verified open/close sequence.",
        "",
        "## Actual-margin predictive performance",
        "",
        _accuracy_table(full),
        "",
        "### Requested-variant common-game paired sample",
        "",
        _common_sample_table(common),
        "",
        _common_sample_note(common),
        "",
        "### Cutoff-backed common-game sensitivity",
        "",
        _common_sample_table(cutoff_backed_common),
        "",
        _common_sample_note(cutoff_backed_common),
        "",
        "## Did model/CFBD-field disagreement contain residual signal?",
        "",
        _signal_table(full),
        "",
        _threshold_tables(full),
        "",
        "A positive slope/correlation would mean that, when a model put the home team higher than a CFBD endpoint field, actual margins tended to exceed that endpoint too. These are descriptive checks, not profitability estimates or evidence about a true historical market line.",
        "",
        "## Weekly evolution",
        "",
        _weekly_table(weekly),
        "",
        "## Largest model/CFBD-field disagreements",
        "",
        *_largest_disagreement_sections(frame),
        "",
        "## All flagged CFBD-field events",
        "",
        "This exhaustive table is deduplicated by model variant and CFBD game id. It includes every favorite inversion, CFBD-field zero crossing, and stored-field change at least three points away from the model. These remain endpoint-field diagnostics, not claims about historical market movement.",
        "",
        _notable_event_table(frame),
        "",
        "## Plots",
        "",
        "Plots labelled `all selected` include the Week 1 no-cutoff reconstruction rows. Files labelled `cutoff_backed` exclude those rows and provide the Weeks 2–3 sensitivity view.",
        "",
        *_plot_lines(report_path, _sequence(summary, "plots")),
        "",
        "## Reproducibility",
        "",
        "The analysis script discovers publications from the public static API, caches raw selected API responses under `data/raw/gippyrank/context_market/`, caches the mutable CFBD line response with a SHA-256 provenance sidecar under `data/raw/cfbd/lines/`, and writes the derived table and figures under `data/processed/context_market_2026/`. No credential is written to any output.",
        "",
        "```bash",
        "uv run python scripts/analyze_context_market.py --season 2026",
        "```",
        "",
        "To refresh a mutable source without overwriting a raw file, pass `--refresh-market` or `--refresh-api`; each refresh gets a timestamped raw filename. The default run is the explicitly labelled reconstruction requested here; pass `--strict-pregame` to require contemporaneous publication instead.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _variant_label(key: str) -> str:
    variant = VARIANT_BY_KEY.get(key)
    return variant.label if variant is not None else key


def _summary_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        sample = _mapping(full, version)
        alignment = _mapping(sample, "alignment")
        movement = _mapping(sample, "market_movement")
        accuracy = _mapping(sample, "actual_margin_accuracy")
        opening = _mapping(alignment, "cfbd_opening")
        later = _mapping(alignment, "cfbd_stored_later")
        gippy = _mapping(accuracy, "gippy")
        paired_open = _mapping(accuracy, "gippy_minus_open_paired_mae")
        paired_later = _mapping(accuracy, "gippy_minus_later_paired_mae")
        distance = _mapping(movement, "distance_and_direction")
        rows.append(
            [
                _variant_label(version),
                _number(opening.get("n"), digits=0),
                _number(opening.get("mean_absolute_difference")),
                _number(later.get("mean_absolute_difference")),
                _number(distance.get("mean_market_movement_toward_gippy")),
                _number(gippy.get("mae")),
                _number(paired_open.get("mean_paired_mae_difference")),
                _number(paired_later.get("mean_paired_mae_difference")),
            ]
        )
    return _table(
        [
            "Variant",
            "N",
            "MAD vs CFBD open",
            "MAD vs CFBD later",
            "Mean move toward Gippy",
            "Gippy actual MAE",
            "Paired MAE − open",
            "Paired MAE − later",
        ],
        rows,
    )


def _direct_answer_lines(
    full: Mapping[str, object], common: Mapping[str, object]
) -> list[str]:
    lines = [
        "- **Historical opening/closing question:** not answered with this CFBD feed. The endpoint fields lack the timestamps and history needed to prove true market opening, closing, or movement.",
        "- **Outcome accuracy:** model and CFBD-field errors below are calculated against the final home margin on identical FBS-v-FBS games.",
    ]
    for version in VARIANT_KEYS:
        sample = _mapping(full, version)
        accuracy = _mapping(sample, "actual_margin_accuracy")
        movement = _mapping(sample, "market_movement")
        signal = _mapping(sample, "incremental_signal")
        lines.append(
            f"- **{_variant_label(version)}:** model actual-margin MAE "
            f"{_number(_mapping(accuracy, 'gippy').get('mae'))}; CFBD opening-field "
            f"MAE {_number(_mapping(accuracy, 'cfbd_opening').get('mae'))}; CFBD "
            f"stored-later-field MAE {_number(_mapping(accuracy, 'cfbd_stored_later').get('mae'))}; "
            f"stored-field distance change toward model {_number(_mapping(movement, 'distance_and_direction').get('mean_market_movement_toward_gippy'))}; "
            f"opening-edge/actual-residual Pearson {_number(_mapping(signal, 'vs_open').get('pearson_correlation'))}."
        )
    reason = common.get("reason")
    common_note = (
        _text(reason) if reason is not None else "all requested variants are present"
    )
    lines.append(
        f"- **Common-game comparison:** {_number(common.get('common_game_count'), digits=0)} games; {common_note}."
    )
    return lines


def _publication_table(selections: Sequence[object]) -> str:
    rows: list[list[str]] = []
    for selection in selections:
        if not isinstance(selection, Mapping):
            continue
        rows.append(
            [
                _text(selection.get("variant_label")),
                _text(selection.get("model_family")),
                _text(selection.get("model_version")),
                _text(selection.get("week")),
                _code_or_dash(selection.get("snapshot_id")),
                _text(selection.get("publication_status")),
                _short_timestamp(selection.get("generation_timestamp")),
                _short_timestamp(selection.get("effective_cutoff")),
                _text(selection.get("timing_classification")),
                _text(selection.get("source_retrieved_at")),
                _code_or_dash(selection.get("source_context_snapshot_id")),
                _code_or_dash(selection.get("comparison_snapshot_id")),
                _text(selection.get("reason")),
            ]
        )
    return _table(
        [
            "Variant",
            "Family",
            "Model version",
            "Week",
            "Selected snapshot",
            "Status",
            "Generated",
            "Cutoff",
            "Timing class",
            "Source retrieved",
            "Source Context snapshot",
            "Comparison snapshot",
            "Missing reason",
        ],
        rows,
    )


def _provider_table(coverage: Sequence[object], choice: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    selected = choice.get("provider")
    for row in coverage:
        if not isinstance(row, Mapping):
            continue
        provider = _text(row.get("provider"))
        rows.append(
            [
                provider + (" (selected)" if provider == selected else ""),
                _number(row.get("eligible_games"), digits=0),
                _number(row.get("opening_games"), digits=0),
                _number(row.get("later_games"), digits=0),
                _number(row.get("same_provider_open_later_games"), digits=0),
            ]
        )
    detail = _text(choice.get("reason"))
    return (
        _table(
            [
                "Provider",
                "Eligible",
                "Open",
                "Stored later",
                "Same-provider open/later",
            ],
            rows,
        )
        + f"\n\nSelection rationale: {detail}"
    )


def _candidate_population_table(population: Mapping[str, object]) -> str:
    rows = [
        [
            _number(row.get("week"), digits=0),
            _number(row.get("unique_fbs_vs_fbs_games"), digits=0),
        ]
        for row in _sequence(population, "by_week")
        if isinstance(row, Mapping)
    ]
    rows.append(
        [
            "All weeks (unique)",
            _number(population.get("unique_fbs_vs_fbs_games"), digits=0),
        ]
    )
    return _table(["Week", "Unique FBS-v-FBS model games"], rows)


def _provider_coverage_by_week_table(
    coverage_by_week: Sequence[object], choice: Mapping[str, object]
) -> str:
    rows: list[list[str]] = []
    selected = choice.get("provider")
    for entry in coverage_by_week:
        if not isinstance(entry, Mapping):
            continue
        week = _number(entry.get("week"), digits=0)
        candidate_games = _number(entry.get("candidate_model_games"), digits=0)
        for coverage in _sequence(entry, "providers"):
            if not isinstance(coverage, Mapping):
                continue
            provider = _text(coverage.get("provider"))
            rows.append(
                [
                    week,
                    candidate_games,
                    provider + (" (selected)" if provider == selected else ""),
                    _number(coverage.get("eligible_games"), digits=0),
                    _number(coverage.get("opening_games"), digits=0),
                    _number(coverage.get("later_games"), digits=0),
                    _number(coverage.get("same_provider_open_later_games"), digits=0),
                ]
            )
    return _table(
        [
            "Week",
            "Candidate model games",
            "Provider",
            "Eligible",
            "Open",
            "Stored later",
            "Same-provider open/later",
        ],
        rows,
    )


def _spot_check_lines(sources: Mapping[str, object]) -> list[str]:
    checks = _sequence(sources, "cfbd_spread_spot_checks")
    output: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        output.append(
            f"- [{_text(check.get('game'))}]({_text(check.get('url'))}): {_text(check.get('finding'))}"
        )
    return output or ["- No independent spot checks were recorded."]


def _diagnostic_table(rows: Sequence[object]) -> str:
    output: list[list[str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        output.append(
            [
                _text(row.get("variant_label")),
                _number(row.get("week"), digits=0),
                _text(row.get("timing_classification")),
                _number(row.get("expected_fbs_vs_fbs_games"), digits=0),
                _number(row.get("games_with_gippy_prediction"), digits=0),
                _number(row.get("games_with_opening_line"), digits=0),
                _number(row.get("games_with_cfbd_later_line"), digits=0),
                _number(row.get("games_with_same_provider_open_later"), digits=0),
                _number(row.get("games_with_final_score"), digits=0),
                _number(row.get("primary_retained"), digits=0),
                _number(row.get("source_nonfuture_prediction_rows"), digits=0),
                _text(row.get("market_provider")),
            ]
        )
    return _table(
        [
            "Variant",
            "Week",
            "Timing class",
            "Expected",
            "Gippy",
            "Open",
            "Later",
            "Same book",
            "Final",
            "Retained",
            "Nonfuture source rows",
            "Book",
        ],
        output,
    )


def _exclusion_table(rows: Sequence[object]) -> str:
    output: list[list[str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        output.append(
            [
                _text(row.get("variant_label")),
                _number(row.get("week"), digits=0),
                _number(row.get("expected_fbs_vs_fbs_games"), digits=0),
                _number(row.get("excluded_no_pregame_publication"), digits=0),
                _number(row.get("excluded_no_gippy_prediction"), digits=0),
                _number(row.get("excluded_open_or_later_line"), digits=0),
                _number(row.get("excluded_no_final_score"), digits=0),
                _number(row.get("primary_retained"), digits=0),
            ]
        )
    table = _table(
        [
            "Variant",
            "Week",
            "Expected FBS",
            "No selected artifact",
            "No Gippy prediction",
            "No same-book fields",
            "No final score",
            "Retained",
        ],
        output,
    )
    return (
        table
        + "\n\nExclusion screens are diagnostic and may overlap; they are not meant to sum to the expected count."
    )


def _alignment_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        metrics = _mapping(_mapping(full, version), "alignment")
        for endpoint, value in (
            ("CFBD opening", _mapping(metrics, "cfbd_opening")),
            ("CFBD stored later", _mapping(metrics, "cfbd_stored_later")),
        ):
            rows.append(
                [
                    _variant_label(version),
                    endpoint,
                    _number(value.get("n"), digits=0),
                    _number(value.get("mean_absolute_difference")),
                    _number(value.get("median_absolute_difference")),
                    _number(value.get("rmse")),
                    _number(value.get("mean_signed_difference")),
                    _number(value.get("pearson_correlation")),
                    _number(value.get("spearman_correlation")),
                    _percent(value.get("same_favorite_pct")),
                    _percent(value.get("within_3_pct")),
                    _percent(value.get("within_5_pct")),
                    _percent(value.get("within_7_pct")),
                ]
            )
    return _table(
        [
            "Variant",
            "Endpoint",
            "N",
            "Mean abs",
            "Median abs",
            "RMSE",
            "Signed",
            "Pearson",
            "Spearman",
            "Same favorite",
            "≤3",
            "≤5",
            "≤7",
        ],
        rows,
    )


def _calibration_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        calibration = _mapping(_mapping(full, version), "calibration")
        for endpoint, value in (
            ("CFBD opening", _mapping(calibration, "cfbd_opening")),
            ("CFBD stored later", _mapping(calibration, "cfbd_stored_later")),
        ):
            rows.append(
                [
                    _variant_label(version),
                    endpoint,
                    _number(value.get("intercept")),
                    _number(value.get("slope")),
                    _number(value.get("r_squared")),
                ]
            )
    return _table(["Variant", "Endpoint", "Intercept", "Slope", "R²"], rows)


def _movement_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        value = _mapping(
            _mapping(_mapping(full, version), "market_movement"),
            "distance_and_direction",
        )
        rows.append(
            [
                _variant_label(version),
                _number(value.get("n"), digits=0),
                _number(value.get("mean_distance_from_open")),
                _number(value.get("mean_distance_from_close")),
                _number(value.get("median_distance_from_open")),
                _number(value.get("median_distance_from_close")),
                _number(value.get("mean_market_movement_toward_gippy")),
                _percent(value.get("moved_toward_pct")),
                _percent(value.get("moved_away_pct")),
                _percent(value.get("unchanged_pct")),
            ]
        )
    return _table(
        [
            "Variant",
            "N",
            "Mean open dist",
            "Mean later dist",
            "Median open dist",
            "Median later dist",
            "Mean toward",
            "Toward",
            "Away",
            "Unchanged",
        ],
        rows,
    )


def _movement_relationship_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        value = _mapping(
            _mapping(_mapping(full, version), "market_movement"),
            "open_edge_vs_market_move",
        )
        rows.append(
            [
                _variant_label(version),
                _number(value.get("n"), digits=0),
                _number(value.get("pearson_correlation")),
                _number(value.get("spearman_correlation")),
                _number(value.get("intercept")),
                _number(value.get("slope")),
                _number(value.get("r_squared")),
            ]
        )
    return _table(
        ["Variant", "N", "Pearson", "Spearman", "Intercept", "Slope", "R²"], rows
    )


def _edge_bucket_tables(full: Mapping[str, object]) -> str:
    blocks: list[str] = []
    for version in VARIANT_KEYS:
        buckets = _sequence(
            _mapping(_mapping(full, version), "market_movement"), "edge_buckets"
        )
        rows = [
            [
                _text(row.get("bucket")),
                _number(row.get("n"), digits=0),
                _number(row.get("average_movement_toward_gippy")),
                _percent(row.get("moved_toward_pct")),
                _number(row.get("average_close_distance_to_gippy")),
            ]
            for row in buckets
            if isinstance(row, Mapping)
        ]
        blocks.extend(
            [
                f"### {_variant_label(version)} opening-edge buckets",
                "",
                _table(
                    [
                        "abs(edge)",
                        "N",
                        "Mean toward",
                        "Moved toward",
                        "Mean later distance",
                    ],
                    rows,
                ),
                "",
            ]
        )
    return "\n".join(blocks).rstrip()


def _accuracy_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        accuracy = _mapping(_mapping(full, version), "actual_margin_accuracy")
        for name, value in (
            ("Gippy", _mapping(accuracy, "gippy")),
            ("CFBD opening", _mapping(accuracy, "cfbd_opening")),
            ("CFBD stored later", _mapping(accuracy, "cfbd_stored_later")),
        ):
            rows.append(
                [
                    _variant_label(version),
                    name,
                    _number(value.get("n"), digits=0),
                    _number(value.get("mae")),
                    _number(value.get("median_absolute_error")),
                    _number(value.get("rmse")),
                    _number(value.get("mean_signed_error")),
                    _number(value.get("pearson_correlation")),
                    _number(value.get("spearman_correlation")),
                ]
            )
        for name, value in (
            (
                "Gippy − CFBD open paired",
                _mapping(accuracy, "gippy_minus_open_paired_mae"),
            ),
            (
                "Gippy − CFBD later paired",
                _mapping(accuracy, "gippy_minus_later_paired_mae"),
            ),
        ):
            interval = value.get("bootstrap_95_ci")
            interval_text = (
                f"[{_number(interval[0])}, {_number(interval[1])}]"
                if isinstance(interval, list) and len(interval) == 2
                else "—"
            )
            rows.append(
                [
                    _variant_label(version),
                    name,
                    _number(value.get("n"), digits=0),
                    _number(value.get("mean_paired_mae_difference")),
                    _number(value.get("median_paired_mae_difference")),
                    interval_text,
                    "—",
                    "—",
                    "—",
                ]
            )
    return _table(
        [
            "Variant",
            "Prediction/comparison",
            "N",
            "MAE / mean Δ",
            "Median abs / Δ",
            "RMSE / CI",
            "Signed",
            "Pearson",
            "Spearman",
        ],
        rows,
    )


def _common_sample_table(common: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    by_version = _mapping(common, "by_version")
    for version in VARIANT_KEYS:
        sample = _mapping(by_version, version)
        accuracy = _mapping(sample, "actual_margin_accuracy")
        rows.append(
            [
                _variant_label(version),
                _number(_mapping(accuracy, "gippy").get("n"), digits=0),
                _number(_mapping(accuracy, "gippy").get("mae")),
                _number(_mapping(accuracy, "cfbd_opening").get("mae")),
                _number(_mapping(accuracy, "cfbd_stored_later").get("mae")),
                _number(
                    _mapping(accuracy, "gippy_minus_open_paired_mae").get(
                        "mean_paired_mae_difference"
                    )
                ),
                _number(
                    _mapping(accuracy, "gippy_minus_later_paired_mae").get(
                        "mean_paired_mae_difference"
                    )
                ),
            ]
        )
    return _table(
        [
            "Variant",
            "Common N",
            "Gippy MAE",
            "Open MAE",
            "Later MAE",
            "Gippy − open",
            "Gippy − later",
        ],
        rows,
    )


def _common_sample_note(common: Mapping[str, object]) -> str:
    count = _number(common.get("common_game_count"), digits=0)
    reason = common.get("reason")
    if reason is not None:
        return f"Common-game count: {count}. {_text(reason)}"
    return (
        f"Common-game count: {count}. Every row above uses the same intersection "
        "of requested model variants, FBS-v-FBS games, same-provider fields, and final scores."
    )


def _signal_table(full: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        signal = _mapping(_mapping(full, version), "incremental_signal")
        for endpoint, value in (
            ("CFBD opening", _mapping(signal, "vs_open")),
            ("CFBD stored later", _mapping(signal, "vs_cfbd_stored_later")),
        ):
            rows.append(
                [
                    _variant_label(version),
                    endpoint,
                    _number(value.get("n"), digits=0),
                    _number(value.get("pearson_correlation")),
                    _number(value.get("spearman_correlation")),
                    _number(value.get("intercept")),
                    _number(value.get("slope")),
                    _number(value.get("r_squared")),
                    _percent(value.get("directional_agreement_pct")),
                ]
            )
    return _table(
        [
            "Variant",
            "Endpoint",
            "N",
            "Pearson",
            "Spearman",
            "Intercept",
            "Slope",
            "R²",
            "Directional agreement",
        ],
        rows,
    )


def _threshold_tables(full: Mapping[str, object]) -> str:
    blocks: list[str] = []
    for version in VARIANT_KEYS:
        signal = _mapping(_mapping(full, version), "incremental_signal")
        for title, key in (
            ("opening", "open_edge_thresholds"),
            ("CFBD stored later", "later_edge_thresholds"),
        ):
            values = _sequence(signal, key)
            rows = [
                [
                    f"≥ {_number(row.get('minimum_absolute_edge'))}",
                    _number(row.get("n"), digits=0),
                    _number(row.get("pearson_correlation")),
                    _number(row.get("spearman_correlation")),
                    _number(row.get("slope")),
                    _percent(row.get("directional_agreement_pct")),
                ]
                for row in values
                if isinstance(row, Mapping)
            ]
            blocks.extend(
                [
                    f"### {_variant_label(version)} edge thresholds vs {title}",
                    "",
                    _table(
                        [
                            "abs(edge)",
                            "N",
                            "Pearson",
                            "Spearman",
                            "Slope",
                            "Directional",
                        ],
                        rows,
                    ),
                    "",
                ]
            )
    return "\n".join(blocks).rstrip()


def _weekly_table(weekly: Mapping[str, object]) -> str:
    rows: list[list[str]] = []
    for version in VARIANT_KEYS:
        values = _mapping(weekly, version)
        for week, sample in sorted(values.items(), key=lambda item: int(item[0])):
            if not isinstance(sample, Mapping):
                continue
            alignment = _mapping(sample, "alignment")
            movement = _mapping(sample, "market_movement")
            accuracy = _mapping(sample, "actual_margin_accuracy")
            signal = _mapping(sample, "incremental_signal")
            rows.append(
                [
                    _variant_label(version),
                    week,
                    _number(_mapping(alignment, "cfbd_opening").get("n"), digits=0),
                    _number(
                        _mapping(alignment, "cfbd_opening").get(
                            "mean_absolute_difference"
                        )
                    ),
                    _number(
                        _mapping(alignment, "cfbd_stored_later").get(
                            "mean_absolute_difference"
                        )
                    ),
                    _number(_mapping(accuracy, "gippy").get("mae")),
                    _number(
                        _mapping(movement, "distance_and_direction").get(
                            "mean_market_movement_toward_gippy"
                        )
                    ),
                    _number(
                        _mapping(signal, "vs_cfbd_stored_later").get(
                            "pearson_correlation"
                        )
                    ),
                ]
            )
    return _table(
        [
            "Variant",
            "Week",
            "N",
            "MAD open",
            "MAD later",
            "Gippy MAE",
            "Mean toward",
            "Later edge/result Pearson",
        ],
        rows,
    )


def _largest_disagreement_sections(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No eligible model predictions were available."]
    output: list[str] = []
    primary = primary_sample(frame, [])
    for (version, week), group in primary.groupby(["variant_key", "week"]):
        for endpoint, column in (
            ("CFBD opening", "abs_gippy_minus_open"),
            ("CFBD stored later", "abs_gippy_minus_close"),
        ):
            selected = group.dropna(subset=[column]).nlargest(10, column)
            output.extend(
                [
                    f"### {_variant_label(str(version))} Week {int(week)} — largest gaps vs {endpoint}",
                    "",
                    _table(
                        [
                            "Game",
                            "Gippy",
                            "CFBD open",
                            "CFBD later",
                            "Actual",
                            "Gippy/open",
                            "Gippy/later",
                            "Move",
                            "Flags",
                        ],
                        [
                            [
                                f"{row['away_team']} at {row['home_team']}",
                                _football_line(
                                    row["gippy_expected_home_margin"],
                                    row["home_team"],
                                    row["away_team"],
                                ),
                                _football_line(
                                    row["opening_home_margin"],
                                    row["home_team"],
                                    row["away_team"],
                                ),
                                _football_line(
                                    row["closing_home_margin"],
                                    row["home_team"],
                                    row["away_team"],
                                ),
                                _football_line(
                                    row["actual_home_margin"],
                                    row["home_team"],
                                    row["away_team"],
                                ),
                                _signed(row["gippy_minus_open"]),
                                _signed(row["gippy_minus_close"]),
                                _signed(row["market_move"]),
                                _disagreement_flags(row),
                            ]
                            for _, row in selected.iterrows()
                        ],
                    ),
                    "",
                ]
            )
    return output


def _notable_event_table(frame: pd.DataFrame) -> str:
    events = _notable_market_events(frame)
    if not events:
        return "No flagged CFBD-field events."
    rows = [
        [
            _text(event.get("variant_label")),
            _number(event.get("week"), digits=0),
            _text(event.get("game")),
            _number(event.get("gippy_expected_home_margin")),
            _number(event.get("opening_home_margin")),
            _number(event.get("cfbd_stored_later_home_margin")),
            _number(event.get("cfbd_field_delta")),
            _number(event.get("cfbd_field_change_toward_model")),
            _text(event.get("flags")),
        ]
        for event in events
    ]
    return _table(
        [
            "Variant",
            "Week",
            "Game",
            "Model",
            "CFBD open",
            "CFBD later",
            "Field delta",
            "Toward model",
            "Flags",
        ],
        rows,
    )


def _plot_lines(report_path: Path, paths: Sequence[object]) -> list[str]:
    if not paths:
        return ["No plots were generated because there were no retained rows."]
    output: list[str] = []
    for path_value in paths:
        if not isinstance(path_value, str):
            continue
        absolute = ROOT / path_value
        relative = os.path.relpath(absolute, start=report_path.parent).replace(
            os.sep, "/"
        )
        label = Path(path_value).stem.replace("_", " ")
        output.extend([f"![{label}]({relative})", ""])
    return output


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    candidate = value.get(key)
    return candidate if isinstance(candidate, Mapping) else {}


def _sequence(value: Mapping[str, object], key: str) -> Sequence[object]:
    candidate = value.get(key)
    return candidate if isinstance(candidate, list) else []


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: object, *, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math_is_nan(value)):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _percent(value: object) -> str:
    if value is None or (isinstance(value, float) and math_is_nan(value)):
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _signed(value: object) -> str:
    if value is None or (isinstance(value, float) and math_is_nan(value)):
        return "—"
    try:
        return f"{float(value):+.1f}"
    except (TypeError, ValueError):
        return "—"


def _football_line(value: object, home: object, away: object) -> str:
    if value is None or (isinstance(value, float) and math_is_nan(value)):
        return "—"
    number = float(value)
    if abs(number) < 1e-9:
        return "Pick'em"
    team = _text(home) if number > 0 else _text(away)
    return f"{team} -{abs(number):.1f}"


def _short_timestamp(value: object) -> str:
    timestamp = parse_timestamp(value)
    return timestamp.isoformat().replace("+00:00", "Z") if timestamp else "—"


def _code_or_dash(value: object) -> str:
    return f"`{value}`" if value else "—"


def _text(value: object) -> str:
    return str(value) if value is not None else "—"


if __name__ == "__main__":
    try:
        main()
    except (ContextMarketError, httpx.HTTPError) as error:
        print(f"context-market analysis failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
