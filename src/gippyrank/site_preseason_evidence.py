"""Build browser-facing preseason evidence for static publication.

The website must show the evidence a preseason prior consumed without asking
browser code to rediscover values from research CSVs.  This module is a
build-time projection only: it reads immutable/processed inputs, preserves
their availability and provenance, and emits compact team-level display data.
It never fits a model or calculates feature attribution.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

PRESEASON_INPUT_SCHEMA_VERSION = "1.0"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: object) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:,.{digits}f}"


def _format_rank(value: float | None) -> str:
    return "Unavailable" if value is None else f"#{value:,.1f}"


def _format_percent(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value * 100:.1f}%"


def _format_trend(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:+,.2f}"


def _field(
    field_id: str,
    label: str,
    value: object,
    display_value: str,
    *,
    model_feature: str | None = None,
    source: str | None = None,
    comparison: dict[str, object] | None = None,
    detail: str | None = None,
) -> dict[str, object]:
    observed = value is not None and str(value).strip() not in {"", "None", "null"}
    result: dict[str, object] = {
        "id": field_id,
        "label": label,
        "raw_value": value if observed else None,
        "display_value": display_value if observed else "Unavailable",
        "availability": "observed" if observed else "missing",
    }
    if model_feature is not None:
        result["model_feature"] = model_feature
    if source is not None:
        result["source"] = source
    if comparison is not None:
        result["comparison"] = comparison
    if detail is not None:
        result["detail"] = detail
    return result


def _group(
    title: str,
    source: str,
    fields: list[dict[str, object]],
    *,
    note: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "title": title,
        "source": source,
        "availability": (
            "observed"
            if any(field["availability"] == "observed" for field in fields)
            else "missing"
        ),
        "fields": fields,
    }
    if note is not None:
        result["note"] = note
    return result


def _comparison(
    values: Mapping[str, float | None],
    team_id: str,
    *,
    direction: str,
) -> dict[str, object] | None:
    value = values.get(team_id)
    observed = [item for item in values.values() if item is not None]
    if value is None or not observed:
        return None
    if direction == "higher_is_better":
        better = sum(item > value for item in observed)
        below = sum(item < value for item in observed)
    elif direction == "lower_is_better":
        better = sum(item < value for item in observed)
        below = sum(item > value for item in observed)
    else:
        return None
    equal = sum(item == value for item in observed)
    return {
        "population": "target-season FBS teams with observed values",
        "count": len(observed),
        "direction": direction,
        "fbs_rank": better + 1,
        "fbs_percentile": 100.0 * (below + 0.5 * equal) / len(observed),
    }


def _program_history_records(root: Path, season: int) -> dict[str, dict[str, str]]:
    path = root / f"data/processed/preseason/program_history_evidence/{season}.csv"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        if (
            row.get("target_season") != str(season)
            or row.get("subdivision") != "fbs"
            or not row.get("team_id")
        ):
            continue
        result[str(row["team_id"])] = row
    return result


def _coach_tenure_records(root: Path, season: int) -> dict[str, dict[str, str]]:
    path = root / f"data/processed/preseason/coach_tenure_evidence/{season}.csv"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        if (
            row.get("target_season") != str(season)
            or row.get("subdivision") != "fbs"
            or not row.get("team_id")
        ):
            continue
        result[str(row["team_id"])] = row
    return result


def _rank_summary_field(
    *,
    field_id: str,
    label: str,
    row: Mapping[str, str] | None,
    rank_values: Mapping[str, float | None],
    team_id: str,
    model_feature: str | None,
    source: str,
) -> dict[str, object]:
    value = _number(row.get("rank_mean") if row is not None else None)
    median = _number(row.get("rank_median") if row is not None else None)
    systems = _number(row.get("usable_systems") if row is not None else None)
    description = _format_rank(value)
    if median is not None:
        description += f" mean · #{median:.0f} median"
    if systems is not None:
        description += f" · {systems:.0f} systems"
    return _field(
        field_id,
        label,
        value,
        description,
        model_feature=model_feature,
        source=source,
        comparison=_comparison(rank_values, team_id, direction="lower_is_better"),
    )


def _program_history(
    history_records: Mapping[str, Mapping[str, str]],
    *,
    season: int,
    team_id: str,
) -> dict[str, object]:
    source = (
        "Frozen preseason program-history evidence (Massey final constituent ranks)"
    )
    fields: list[dict[str, object]] = []
    for lag, label, model_feature in (
        (1, f"{season - 1} consensus rank", "lag1_rank_distribution"),
        (2, f"{season - 2} consensus rank", "lag2_z_mean"),
        (3, f"{season - 3} consensus rank", "lag3_z_mean"),
    ):
        rows = {
            key: {
                "rank_mean": item.get(f"lag{lag}_rank_mean", ""),
                "rank_median": item.get(f"lag{lag}_rank_median", ""),
                "usable_systems": item.get(f"lag{lag}_usable_systems", ""),
            }
            for key, item in history_records.items()
        }
        ranks = {key: _number(item["rank_mean"]) for key, item in rows.items()}
        fields.append(
            _rank_summary_field(
                field_id=f"season_{season - lag}_consensus_rank",
                label=label,
                row=rows.get(team_id),
                rank_values=ranks,
                team_id=team_id,
                model_feature=model_feature,
                source=source,
            )
        )

    record = history_records.get(team_id)
    raw_value = _number(
        record.get("long_run_rank_percentile_mean") if record is not None else None
    )
    model_input = _number(record.get("long_run_z_mean") if record is not None else None)
    history_seasons = _number(
        record.get("history_seasons") if record is not None else None
    )
    long_run_display = "Unavailable"
    if raw_value is not None:
        long_run_display = f"{raw_value * 100:.1f}% mean rank percentile"
        if history_seasons is not None:
            long_run_display += f" across {history_seasons:.0f} seasons"
    fields.append(
        _field(
            "long_run_program_level",
            "Long-run program level",
            raw_value,
            long_run_display,
            model_feature="long_run_z_mean",
            source=source,
            detail=(
                f"Model-input transformed-rank mean: {model_input:.3f}"
                if model_input is not None
                else None
            ),
        )
    )
    return _group(
        "Program history",
        source,
        fields,
        note="Historical summaries describe pre-target-season rank evidence; they are not a feature-attribution decomposition.",
    )


def _transfer_paths(root: Path, season: int) -> tuple[Path | None, Path | None]:
    candidates = (
        root
        / f"data/processed/preseason/context_v1_3_{season}_reconstruction/transfer_features.csv",
        root
        / f"data/processed/preseason/context_v1_3/annual/{season}/transfer_features.csv",
    )
    feature_path = next((path for path in candidates if path.is_file()), None)
    if feature_path is None:
        return None, None
    audit_path = feature_path.with_name("transfer_team_audit.csv")
    return feature_path, audit_path if audit_path.is_file() else None


def _transfer_provenance(root: Path, season: int) -> dict[str, object]:
    path = (
        root
        / f"data/processed/preseason/context_v1_3/annual/{season}/feature_provenance.json"
    )
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    transfer = payload.get("transfer") if isinstance(payload, dict) else None
    return transfer if isinstance(transfer, dict) else {}


def _input_state(root: Path, metadata: Mapping[str, object]) -> str | None:
    relative_path = metadata.get("prior_artifact_path")
    if not isinstance(relative_path, str):
        return None
    path = root / relative_path
    if not path.is_file():
        return None
    rows = _read_csv(path)
    return next(
        (
            str(row["starting_point_status"])
            for row in rows
            if row.get("starting_point_status")
        ),
        None,
    )


def _context_groups(
    *,
    feature_rows: Mapping[str, Mapping[str, str]],
    historical_recruiting_points: Mapping[str, list[str]],
    team_id: str,
    coach_records: Mapping[str, Mapping[str, str]],
    transfer_rows: Mapping[str, Mapping[str, str]],
    transfer_audits: Mapping[str, Mapping[str, str]],
    include_transfers: bool,
) -> dict[str, dict[str, object]]:
    source = "CFBD preseason team inputs"
    current = feature_rows.get(team_id, {})
    all_rows = feature_rows

    point_values = {
        candidate_id: _number(row.get("recruiting_class_points"))
        for candidate_id, row in all_rows.items()
    }
    rank_values = {
        candidate_id: _number(row.get("recruiting_class_rank"))
        for candidate_id, row in all_rows.items()
    }
    talent_values = {
        candidate_id: _number(row.get("talent_composite"))
        for candidate_id, row in all_rows.items()
    }
    returning_values = {
        candidate_id: _number(row.get("returning_pct_ppa"))
        for candidate_id, row in all_rows.items()
    }
    history = historical_recruiting_points.get(team_id, [])
    current_points = _number(current.get("recruiting_class_points"))
    point_series = [current_points, *[_number(value) for value in history]]
    means = {years: _mean(point_series[:years]) for years in (2, 3, 4)}
    trend = (
        current_points - point_series[2]
        if current_points is not None
        and len(point_series) > 2
        and point_series[2] is not None
        else None
    )

    coach_record = coach_records.get(team_id, {})
    coach_known_by_cutoff = coach_record.get("known_by_cutoff") == "true"
    coach_tenure = (
        _number(coach_record.get("coach_tenure_seasons"))
        if coach_known_by_cutoff
        else None
    )
    coach_name = coach_record.get("coach_name") or current.get("head_coach") or None
    coach_note = (
        None
        if coach_known_by_cutoff
        else str(
            coach_record.get("unavailable_reason") or "no_cutoff_safe_tenure_evidence"
        )
    )

    transfer = transfer_rows.get(team_id, {})
    audit = transfer_audits.get(team_id, {})
    usage_values = {
        candidate_id: _number(row.get("transfer_in_prior_usage_sum"))
        for candidate_id, row in transfer_rows.items()
    }
    db_values = {
        candidate_id: _number(row.get("transfer_in_prior_defensive_impact_db_sum"))
        for candidate_id, row in transfer_rows.items()
    }
    usage = _number(transfer.get("transfer_in_prior_usage_sum"))
    db_impact = _number(transfer.get("transfer_in_prior_defensive_impact_db_sum"))
    db_available = _number(
        transfer.get("transfer_in_prior_defensive_impact_db_available")
    )
    resolved_db = _number(audit.get("audit_resolved_db_transfers"))
    incoming_db = _number(audit.get("audit_incoming_db_transfers"))
    db_status = audit.get("audit_db_feature_status") or (
        "complete" if db_available == 1 else "incomplete" if db_available == 0 else None
    )

    groups = {
        "coach": _group(
            "Coaching",
            "Frozen cutoff-safe coach-tenure evidence",
            [
                _field(
                    "reported_head_coach",
                    "Reported head coach (supplemental)",
                    coach_name,
                    str(coach_name or "Unavailable"),
                    source="Frozen cutoff-safe coach-tenure evidence",
                    detail="Supplemental context; not a model feature.",
                ),
                _field(
                    "coach_tenure_seasons",
                    "Coach tenure",
                    coach_tenure,
                    f"{coach_tenure:.0f} seasons"
                    if coach_tenure is not None
                    else "Unavailable",
                    model_feature="coach_tenure_seasons",
                    source="Frozen cutoff-safe coach-tenure evidence",
                    detail=coach_note,
                ),
            ],
            note=(
                "Coach tenure is the only coaching model feature and is shown only "
                "when cutoff-safe tenure evidence was available."
            ),
        ),
        "recruiting": _group(
            "Recruiting",
            "CFBD /recruiting/teams",
            [
                _field(
                    "recruiting_class_rank",
                    "Current recruiting class rank",
                    _number(current.get("recruiting_class_rank")),
                    _format_rank(_number(current.get("recruiting_class_rank"))),
                    model_feature="recruiting_class_rank",
                    source="CFBD /recruiting/teams",
                    comparison=_comparison(
                        rank_values, team_id, direction="lower_is_better"
                    ),
                ),
                _field(
                    "recruiting_class_points",
                    "Current recruiting class points",
                    current_points,
                    _format_number(current_points),
                    model_feature="recruiting_class_points",
                    source="CFBD /recruiting/teams",
                    comparison=_comparison(
                        point_values, team_id, direction="higher_is_better"
                    ),
                ),
                *[
                    _field(
                        f"recruiting_points_{years}y_mean",
                        f"{years}-year recruiting-points mean",
                        means[years],
                        _format_number(means[years]),
                        model_feature=f"recruiting_points_{years}y_mean",
                        source="CFBD /recruiting/teams",
                    )
                    for years in (2, 3, 4)
                ],
                _field(
                    "recruiting_points_trend",
                    "Recruiting-points trend",
                    trend,
                    _format_trend(trend),
                    model_feature="recruiting_points_trend",
                    source="CFBD /recruiting/teams",
                    detail="Current class points minus the class from two seasons earlier.",
                ),
            ],
        ),
        "talent": _group(
            "Talent / continuity",
            source,
            [
                _field(
                    "talent_composite",
                    "Team talent composite",
                    _number(current.get("talent_composite")),
                    _format_number(_number(current.get("talent_composite"))),
                    model_feature="talent_composite",
                    source="CFBD /talent (247Sports Team Talent Composite)",
                    comparison=_comparison(
                        talent_values, team_id, direction="higher_is_better"
                    ),
                ),
                _field(
                    "returning_pct_ppa",
                    "Returning PPA",
                    _number(current.get("returning_pct_ppa")),
                    _format_percent(_number(current.get("returning_pct_ppa"))),
                    model_feature="returning_pct_ppa",
                    source="CFBD /player/returning",
                    comparison=_comparison(
                        returning_values, team_id, direction="higher_is_better"
                    ),
                ),
            ],
        ),
    }
    if include_transfers:
        groups["transfers"] = _group(
            "Transfers",
            "Frozen preseason transfer-feature artifact",
            [
                _field(
                    "transfer_in_prior_usage_sum",
                    "Incoming prior offensive usage",
                    usage,
                    _format_number(usage, 3),
                    model_feature="transfer_in_prior_usage_sum",
                    source="Frozen preseason transfer-feature artifact",
                    comparison=_comparison(
                        usage_values, team_id, direction="higher_is_better"
                    ),
                ),
                _field(
                    "transfer_in_prior_defensive_impact_db_sum",
                    "Incoming DB defensive impact",
                    db_impact,
                    _format_number(db_impact, 3),
                    model_feature="transfer_in_prior_defensive_impact_db_sum",
                    source="Frozen preseason transfer-feature artifact",
                    comparison=_comparison(
                        db_values, team_id, direction="higher_is_better"
                    ),
                ),
                _field(
                    "transfer_in_prior_defensive_impact_db_available",
                    "DB transfer-data availability",
                    db_available,
                    (
                        f"{str(db_status).replace('_', ' ')}"
                        + (
                            f" · {resolved_db:.0f}/{incoming_db:.0f} incoming DB transfers resolved"
                            if resolved_db is not None and incoming_db is not None
                            else ""
                        )
                        if db_available is not None
                        else "Unavailable"
                    ),
                    model_feature="transfer_in_prior_defensive_impact_db_available",
                    source="Frozen preseason transfer-feature artifact",
                ),
            ],
        )
    return groups


def build_preseason_input_projection(
    *,
    root: Path,
    metadata: Mapping[str, object],
    team_ids: Iterable[str],
) -> dict[str, object] | None:
    """Return stable display inputs for a published predictive preseason view.

    Small fixture roots intentionally omit the original preseason source
    corpus.  In that case the exporter keeps its existing artifact contract
    and returns ``None``; a production root with that corpus always emits the
    projection.
    """
    if metadata.get("ranking_family") != "predictive":
        return None
    season_value = metadata.get("season")
    try:
        season = int(season_value)
    except (TypeError, ValueError):
        return None
    prior_family = str(metadata.get("prior_family") or "context")
    prior_model_version = str(metadata.get("prior_model_version") or "")
    feature_path = root / "data/processed/preseason/team_season_features.csv"
    if not feature_path.is_file():
        return None

    all_features = _read_csv(feature_path)
    feature_rows_by_season: dict[int, dict[str, dict[str, str]]] = {}
    for row in all_features:
        if row.get("subdivision") != "fbs":
            continue
        try:
            row_season = int(str(row["season"]))
        except (KeyError, ValueError):
            continue
        feature_rows_by_season.setdefault(row_season, {})[str(row["team_id"])] = row
    current_rows = feature_rows_by_season.get(season, {})
    if not current_rows:
        return None
    historical_recruiting_points = {
        team_id: [
            prior_row["recruiting_class_points"]
            for prior_season in range(season - 1, season - 4, -1)
            if (prior_row := feature_rows_by_season.get(prior_season, {}).get(team_id))
            is not None
        ]
        for team_id in current_rows
    }

    transfer_rows: dict[str, dict[str, str]] = {}
    transfer_audits: dict[str, dict[str, str]] = {}
    transfer_path, audit_path = _transfer_paths(root, season)
    if transfer_path is not None:
        transfer_rows = {
            str(row["team_id"]): row
            for row in _read_csv(transfer_path)
            if row.get("season") == str(season) and row.get("subdivision") == "fbs"
        }
    if audit_path is not None:
        transfer_audits = {
            str(row["team_id"]): row
            for row in _read_csv(audit_path)
            if row.get("season") == str(season) and row.get("subdivision") == "fbs"
        }

    history_records = _program_history_records(root, season)
    coach_records = _coach_tenure_records(root, season)
    projected_teams: dict[str, dict[str, object]] = {}
    for team_id in sorted({str(value) for value in team_ids}):
        current = current_rows.get(team_id)
        if current is None:
            continue
        groups: dict[str, dict[str, object]] = {
            "program_history": _program_history(
                history_records, season=season, team_id=team_id
            )
        }
        if prior_family == "context":
            groups.update(
                _context_groups(
                    feature_rows=current_rows,
                    historical_recruiting_points=historical_recruiting_points,
                    team_id=team_id,
                    coach_records=coach_records,
                    transfer_rows=transfer_rows,
                    transfer_audits=transfer_audits,
                    include_transfers=prior_model_version == "1.3",
                )
            )
        projected_teams[team_id] = groups

    transfer_provenance = _transfer_provenance(root, season)
    retrospective = (
        transfer_provenance.get("provenance_class")
        == "retrospective_2026_reconstruction"
    )
    provenance: dict[str, object] = {
        "prior_artifact_path": metadata.get("prior_artifact_path"),
        "prior_artifact_sha256": metadata.get("prior_artifact_sha256"),
        "input_state": _input_state(root, metadata),
        "evidence_scope": (
            "program_history_only"
            if prior_family == "history"
            else f"context_{prior_model_version or 'unknown'}"
        ),
    }
    if transfer_path is not None:
        provenance["transfer_feature_path"] = transfer_path.relative_to(root).as_posix()
    if retrospective and prior_family == "context" and prior_model_version == "1.3":
        provenance["transfer_caveat"] = {
            "status": "retrospective_reconstruction",
            "visible_label": "2026 transfer inputs were reconstructed after the Aug. 15 cutoff.",
            "detail": transfer_provenance.get("provenance_statement"),
            "cutoff": transfer_provenance.get("cutoff"),
            "derivation_timestamp": transfer_provenance.get("derivation_timestamp"),
        }

    return {
        "schema_version": PRESEASON_INPUT_SCHEMA_VERSION,
        "artifact_kind": "preseason_input_projection",
        "season": season,
        "prior_family": prior_family,
        "prior_model_version": metadata.get("prior_model_version"),
        "prior_lineage": metadata.get("prior_lineage"),
        "provenance": provenance,
        "teams": projected_teams,
    }
