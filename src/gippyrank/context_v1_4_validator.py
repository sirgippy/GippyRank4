"""Outcome-gated execution of the frozen Context 1.4 confirmation protocol.

The registered September 27 Week 5 *snapshot* forecasts game Week 5. The
merged #164 text calls that prospective game slice Week 6; this module keeps
its frozen origins, sample floors, bootstrap, and decision rule while using the
correct game-week name required by #166.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from gippyrank.artifact_hashes import posterior_pmfs_sha256
from gippyrank.context_prior import InferenceRow
from gippyrank.context_prior_v1_3 import (
    load_validated_committed_2026_reconstruction,
    load_validated_context13_fitted_model,
    moderated_location_points,
)
from gippyrank.context_prior_v1_4_candidate import (
    Context13FallbackSource,
    Context13PriorInput,
    construct_candidate_prior,
    load_candidate_spec,
    sha256_json,
)
from gippyrank.context_v1_4_validation_protocol import (
    ProtocolError,
    RegisteredProtocol,
    classify_primary,
    paired_mean_bootstrap,
)
from gippyrank.history_annual_v1_1 import load_validated_history_annual_artifact
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    infer_posterior,
)
from gippyrank.posterior.predictive import (
    ScheduledGame,
    posterior_prediction_teams,
    predict_game,
    predictive_components,
)
from gippyrank.posterior.snapshots import load_likelihood
from gippyrank.preseason import conditional_rank_mixture_pmf

VALIDATOR_INPUTS_PATH = "data/processed/context_v1_4_validation/validator_inputs.jsonl"
VALIDATOR_INPUTS_SHA256 = (
    "50be291962af424b337b848555080a53fe5b8dfde5030f00e502fdaf6251bfd2"
)
PROSPECTIVE_LABEL = "prospective_week_5"
MODELS = ("candidate", "context_1_3", "history_1_1")
GAME_FIELDS = (
    "id",
    "season",
    "week",
    "seasonType",
    "startDate",
    "completed",
    "neutralSite",
    "homeId",
    "awayId",
    "homeClassification",
    "awayClassification",
    "homePoints",
    "awayPoints",
)
SCORE_FIELDS = (
    "game_id",
    "origin",
    "stratum",
    "week",
    "home_id",
    "away_id",
    "model",
    "actual_home_margin",
    "margin_nll",
    "expected_home_margin",
    "margin_mae",
    "home_win_probability",
    "win_brier",
    "interval_50_low",
    "interval_50_high",
    "interval_80_low",
    "interval_80_high",
    "interval_95_low",
    "interval_95_high",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)
OUTPUT_PRODUCTS = (
    "game_results.csv",
    "by_origin.json",
    "evidence_audit.csv",
    "exclusions.csv",
    "summary.json",
    "provenance.json",
    "failure.json",
    "source_audit.json",
    "context_v1_4_validation_result.md",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    if value.strip().casefold() in {"true", "1", "yes"}:
        return True
    if value.strip().casefold() in {"false", "0", "no"}:
        return False
    raise ProtocolError(f"invalid Boolean game field: {value!r}")


def _kickoff(row: dict[str, str]) -> datetime:
    try:
        result = datetime.fromisoformat(row["startDate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(f"game {row.get('id')}: unknown kickoff") from error
    if result.utcoffset() is None:
        raise ProtocolError(f"game {row.get('id')}: kickoff lacks timezone")
    return result


def _score_int(value: str, game_id: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"game {game_id}: invalid final score") from error
    if score < 0 or str(score) != value.strip():
        raise ProtocolError(f"game {game_id}: invalid final score")
    return score


@dataclass(frozen=True)
class SourceAudit:
    status: str
    games: tuple[dict[str, str], ...]
    assignments: dict[str, str]
    expected_ids: tuple[str, ...]
    unresolved_ids: tuple[str, ...]
    exclusions: tuple[dict[str, str], ...]
    exact_duplicate_count: int
    source_sha256: str

    @property
    def completed_count(self) -> int:
        return len(self.games)


@dataclass(frozen=True)
class ForecastState:
    origin: str
    model: str
    teams: dict[str, Team]
    posterior: dict[str, np.ndarray]
    audit: dict[str, object]


def load_validator_inputs(root: Path, protocol: RegisteredProtocol) -> dict[str, Any]:
    """Verify the separately frozen, outcome-free reconstruction inputs."""
    path = root / VALIDATOR_INPUTS_PATH
    if _sha(path) != VALIDATOR_INPUTS_SHA256:
        raise ProtocolError("validator forecast/schedule input hash changed")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not records or records[0].get("kind") != "metadata":
        raise ProtocolError("validator input metadata is missing")
    value = {
        key: records[0][key]
        for key in ("schema_version", "purpose", "season", "source_sha256")
    }
    value["forecast_rows"] = [
        record["value"]
        for record in records[1:]
        if record.get("kind") == "forecast_row"
    ]
    value["registered_schedule"] = [
        record["value"]
        for record in records[1:]
        if record.get("kind") == "schedule_game"
    ]
    if any(
        record.get("kind") not in {"forecast_row", "schedule_game"}
        for record in records[1:]
    ):
        raise ProtocolError("validator input record type changed")
    if value.get("schema_version") != 1 or value.get("season") != 2026:
        raise ProtocolError("validator input schema changed")
    hashes = value["source_sha256"]
    context = protocol.data["models"]["context_1_3"]
    if (
        hashes["context_1_3_predictions"] != context["predictions_file_sha256"]
        or hashes["transfer_features"] != context["transfer_features_file_sha256"]
    ):
        raise ProtocolError(
            "validator input source does not match registered Context 1.3"
        )
    schedule = protocol.data["evidence"]["fcs_fallback"]["raw_schedule_files"]
    if (
        hashes["fbs_schedule"] != schedule[0]["sha256"]
        or hashes["fcs_schedule"] != schedule[1]["sha256"]
    ):
        raise ProtocolError("validator expected schedule does not match registration")
    rows = value["forecast_rows"]
    ids = [row["team_id"] for row in rows]
    if ids != protocol.data["population"]["expected_team_ids"]:
        raise ProtocolError("validator inference team population changed")
    expected = value["registered_schedule"]
    game_ids = [row["id"] for row in expected]
    if game_ids != sorted(set(game_ids)) or not any(
        row["week"] == "5" for row in expected
    ):
        raise ProtocolError("validator expected schedule is invalid")
    return value


def load_source_manifest(
    path: Path, games_path: Path, *, exceptions_path: Path | None = None
) -> dict[str, Any]:
    """Bind a completed canonical corpus to its preserved acquisition files."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("games_sha256") != _sha(games_path):
        raise ProtocolError("game source differs from frozen source manifest")
    files = value.get("source_files")
    if (
        value.get("source_kind") != "cfbd_api_schedule"
        or not isinstance(files, list)
        or any(not isinstance(entry, dict) for entry in files)
        or {entry.get("coverage") for entry in files} != {"fbs", "fcs"}
        or len(files) != 2
    ):
        raise ProtocolError("source manifest must name FBS and FCS CFBD raw files")
    try:
        retrieved_at = datetime.fromisoformat(value["retrieved_at_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("source manifest has invalid retrieval time") from error
    if retrieved_at.utcoffset() is None:
        raise ProtocolError("source manifest retrieval time lacks timezone")
    raw_games: dict[str, dict[str, Any]] = {}
    exceptions = _load_exceptions(exceptions_path)
    for entry in files:
        raw = Path(entry["path"])
        raw = raw if raw.is_absolute() else path.parent / raw
        if not raw.is_file() or _sha(raw) != entry["sha256"]:
            raise ProtocolError(f"preserved source file hash mismatch: {raw}")
        payload = json.loads(raw.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ProtocolError(f"preserved source is not a CFBD game list: {raw}")
        for item in payload:
            if not isinstance(item, dict):
                raise ProtocolError(f"preserved source has a non-object game: {raw}")
            if (
                item.get("season") != 2026
                or item.get("seasonType") != "regular"
                or item.get("week") not in {1, 2, 3, 4, 5}
                or item.get("homeClassification") not in {"fbs", "fcs"}
                or item.get("awayClassification") not in {"fbs", "fcs"}
            ):
                continue
            game_id = str(item["id"])
            comparable = {
                field: item.get(field)
                for field in (
                    "season",
                    "week",
                    "seasonType",
                    "startDate",
                    "completed",
                    "neutralSite",
                    "homeId",
                    "awayId",
                    "homeClassification",
                    "awayClassification",
                    "homePoints",
                    "awayPoints",
                    "startTimeTBD",
                )
            }
            previous = raw_games.setdefault(game_id, comparable)
            if previous != comparable:
                raise ProtocolError(
                    f"conflicting duplicate preserved raw game: {game_id}"
                )
    canonical, _ = _canonical_source_rows(games_path)
    in_scope = {
        game_id: row
        for game_id, row in canonical.items()
        if row.get("season") == "2026"
        and row.get("seasonType") == "regular"
        and row.get("week") in {"1", "2", "3", "4", "5"}
        and row.get("homeClassification", "").casefold() in {"fbs", "fcs"}
        and row.get("awayClassification", "").casefold() in {"fbs", "fcs"}
    }
    if set(in_scope) - set(raw_games) or (set(raw_games) - set(in_scope)) - set(
        exceptions
    ):
        raise ProtocolError(
            "canonical game scope differs from preserved CFBD raw source"
        )
    for game_id, source in raw_games.items():
        if game_id not in in_scope:
            continue
        row = in_scope[game_id]
        if any(
            type(source[field]) is not bool
            for field in ("completed", "neutralSite", "startTimeTBD")
        ):
            raise ProtocolError(f"game {game_id}: raw status flags are ambiguous")
        if source["startTimeTBD"] is True and game_id not in exceptions:
            raise ProtocolError(
                f"game {game_id}: ambiguous actual kickoff in raw source"
            )
        for field in (
            "season",
            "week",
            "seasonType",
            "homeId",
            "awayId",
            "homeClassification",
            "awayClassification",
        ):
            if str(source[field]).casefold() != row[field].casefold():
                raise ProtocolError(
                    f"game {game_id}: canonical {field} differs from raw source"
                )
        for field in ("completed", "neutralSite"):
            if bool(source[field]) != _bool(row[field]):
                raise ProtocolError(
                    f"game {game_id}: canonical {field} differs from raw source"
                )
        if game_id not in exceptions and _kickoff(row) != _kickoff(
            {"id": game_id, "startDate": source["startDate"]}
        ):
            raise ProtocolError(
                f"game {game_id}: actual kickoff differs from raw source"
            )
        if _bool(row["completed"]):
            for field in ("homePoints", "awayPoints"):
                if type(source[field]) is not int or source[field] != _score_int(
                    row[field], game_id
                ):
                    raise ProtocolError(
                        f"game {game_id}: canonical score differs from raw source"
                    )
    return value


def _canonical_source_rows(path: Path) -> tuple[dict[str, dict[str, str]], int]:
    rows = _read_csv(path)
    if not rows:
        raise ProtocolError("game source is empty")
    if set(GAME_FIELDS) - set(rows[0]):
        raise ProtocolError("game source lacks canonical fields")
    unique: dict[str, dict[str, str]] = {}
    duplicate_count = 0
    for row in rows:
        game_id = row.get("id", "")
        if not game_id:
            raise ProtocolError("game source has a blank canonical ID")
        if game_id in unique:
            if row != unique[game_id]:
                raise ProtocolError(f"conflicting duplicate source rows: {game_id}")
            duplicate_count += 1
        else:
            unique[game_id] = row
    return unique, duplicate_count


def _load_exceptions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ProtocolError("exception audit must be a JSON list")
    result: dict[str, dict[str, str]] = {}
    for item in value:
        if (
            not isinstance(item, dict)
            or item.get("disposition") not in {"terminal_exception", "out_of_scope"}
            or not isinstance(item.get("game_id"), str)
            or not item.get("reason")
            or not item.get("evidence")
            or item["game_id"] in result
        ):
            raise ProtocolError("exception audit has an invalid or duplicate entry")
        result[item["game_id"]] = item
    return result


def audit_game_source(
    games_path: Path,
    protocol: RegisteredProtocol,
    inputs: dict[str, Any],
    root: Path,
    *,
    exceptions_path: Path | None = None,
) -> SourceAudit:
    """Complete and assign the registered games before any candidate access."""
    source, duplicates = _canonical_source_rows(games_path)
    expected = {row["id"]: row for row in inputs["registered_schedule"]}
    exceptions = _load_exceptions(exceptions_path)
    unknown_exceptions = set(exceptions) - set(expected) - set(source)
    if unknown_exceptions:
        raise ProtocolError(
            f"exception audit names unknown games: {sorted(unknown_exceptions)}"
        )
    fbs = set(protocol.data["population"]["expected_team_ids"])
    fcs = set(protocol.data["evidence"]["fcs_fallback"]["population_team_ids"])
    origins = protocol.data["forecast_origins"]
    included = {
        origin["publication_slot"]: {
            row["id"]
            for row in _read_csv(
                root / origin["context_files"]["included_games.csv"]["path"]
            )
        }
        for origin in origins
    }
    selected: list[dict[str, str]] = []
    assigned: dict[str, str] = {}
    unresolved: set[str] = set()
    exclusions: list[dict[str, str]] = []
    candidate_ids = set(expected)
    for game_id, row in source.items():
        if row.get("season") == "2026" and row.get("week") in {"1", "2", "3", "4", "5"}:
            candidate_ids.add(game_id)
    for game_id in sorted(candidate_ids):
        row = source.get(game_id)
        registered = expected.get(game_id)
        exception = exceptions.get(game_id)
        if row is None:
            if exception is None:
                unresolved.add(game_id)
            else:
                exclusions.append({"game_id": game_id, **exception})
            continue
        if row.get("season") != "2026" or row.get("seasonType") != "regular":
            if exception and exception["disposition"] == "out_of_scope":
                exclusions.append({"game_id": game_id, **exception})
                continue
            raise ProtocolError(f"game {game_id}: unexpected season classification")
        try:
            week = int(row["week"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(
                f"game {game_id}: invalid week classification"
            ) from error
        if not 1 <= week <= 5 or (registered and week != int(registered["week"])):
            if exception and exception["disposition"] == "out_of_scope":
                exclusions.append({"game_id": game_id, **exception})
                continue
            raise ProtocolError(
                f"game {game_id}: unexpected week or reschedule classification"
            )
        subdivisions = tuple(
            row.get(f"{side}Classification", "").casefold() for side in ("home", "away")
        )
        if any(value not in {"fbs", "fcs"} for value in subdivisions):
            if exception and exception["disposition"] == "out_of_scope":
                exclusions.append({"game_id": game_id, **exception})
                continue
            if registered:
                raise ProtocolError(
                    f"game {game_id}: unexpected participant classification"
                )
            exclusions.append(
                {
                    "game_id": game_id,
                    "disposition": "out_of_scope",
                    "reason": "lower_division",
                    "evidence": "canonical_source",
                }
            )
            continue
        if registered and any(
            str(row.get(field, "")).casefold() != registered[field].casefold()
            for field in (
                "homeId",
                "awayId",
                "homeClassification",
                "awayClassification",
            )
        ):
            raise ProtocolError(
                f"game {game_id}: participant identity changed from registered schedule"
            )
        for side, subdivision in zip(("home", "away"), subdivisions, strict=True):
            team_id = row.get(f"{side}Id", "")
            if team_id not in (fbs if subdivision == "fbs" else fcs):
                raise ProtocolError(
                    f"game {game_id}: team outside locked {subdivision} population"
                )
        if row["homeId"] == row["awayId"]:
            raise ProtocolError(f"game {game_id}: same team on both sides")
        if exception is not None:
            if _bool(row["completed"]):
                raise ProtocolError(
                    f"game {game_id}: terminal exception conflicts with completed score"
                )
            exclusions.append({"game_id": game_id, **exception})
            continue
        if not _bool(row["completed"]):
            if row.get("homePoints") or row.get("awayPoints"):
                raise ProtocolError(
                    f"game {game_id}: unresolved result has score fields"
                )
            unresolved.add(game_id)
            continue
        _score_int(row["homePoints"], game_id)
        _score_int(row["awayPoints"], game_id)
        kickoff = _kickoff(row)
        eligible = [
            origin
            for origin in origins
            if datetime.fromisoformat(origin["logical_cutoff_utc"]) < kickoff
        ]
        if not eligible:
            raise ProtocolError(f"game {game_id}: no earlier registered origin")
        origin = eligible[-1]["publication_slot"]
        if week == 5 and origin != protocol.data["evidence"]["week_6_origin"]:
            raise ProtocolError(f"prospective Week 5 game {game_id}: wrong origin")
        if game_id in included[origin]:
            raise ProtocolError(
                f"game {game_id}: forecast evidence already contains result"
            )
        selected.append(row)
        assigned[game_id] = origin
    status = "SOURCE_COMPLETE" if not unresolved else "SOURCE_INCOMPLETE"
    return SourceAudit(
        status=status,
        games=tuple(sorted(selected, key=lambda row: row["id"])),
        assignments=dict(sorted(assigned.items())),
        expected_ids=tuple(sorted(expected)),
        unresolved_ids=tuple(sorted(unresolved)),
        exclusions=tuple(sorted(exclusions, key=lambda row: row["game_id"])),
        exact_duplicate_count=duplicates,
        source_sha256=_sha(games_path),
    )


def construct_frozen_priors(
    root: Path, protocol: RegisteredProtocol, inputs: dict[str, Any]
) -> dict[str, dict[str, Team]]:
    """Reconstruct all three preseason arms from pinned outcome-free inputs."""
    context = protocol.data["models"]["context_1_3"]
    history = protocol.data["models"]["history_1_1"]
    annual = root / context["predictions_path"]
    model, instance, fit_source = load_validated_context13_fitted_model(
        annual.parent / "fitted_model.json",
        annual.parent / "fitted_instance.json",
        fitted_source_path=root / context["fit_source_path"],
        prediction_artifact_path=annual,
    )
    _, transfer = load_validated_committed_2026_reconstruction(
        (root / context["transfer_features_path"]).parent
    )
    history_source = load_validated_history_annual_artifact(
        root / history["predictions_path"],
        root / history["fitted_instance_path"],
        target_season=2026,
        trained_through_season=2025,
    )
    context_rows = {row["team_id"]: row for row in _read_csv(annual)}
    history_rows = {
        row["team_id"]: row for row in _read_csv(root / history["predictions_path"])
    }
    ids = protocol.data["population"]["expected_team_ids"]
    if set(context_rows) != set(ids) or set(history_rows) != set(ids):
        raise ProtocolError("candidate and baselines have different FBS populations")
    candidate_teams: dict[str, Team] = {}
    baseline_teams: dict[str, Team] = {}
    history_teams: dict[str, Team] = {}
    spec = load_candidate_spec(
        json.loads(
            (root / protocol.data["models"]["candidate"]["audit_spec_path"]).read_text()
        )
    )
    for saved in inputs["forecast_rows"]:
        row = InferenceRow(
            season=int(saved["season"]),
            subdivision=saved["subdivision"],
            team_id=saved["team_id"],
            team_name=saved["team_name"],
            population=int(saved["population"]),
            lag1_z=None
            if saved["lag1_z"] is None
            else tuple(float(v) for v in saved["lag1_z"]),
            lag_zs=tuple(tuple(float(v) for v in values) for values in saved["lag_zs"]),
            features=saved["features"],
            cold_start_reason=saved["cold_start_reason"],
        )
        row.require_no_target()
        team_id = row.team_id
        published = np.asarray(json.loads(context_rows[team_id]["pmf"]), dtype=float)
        if context_rows[team_id]["team_name"] != row.team_name or row.population != len(
            ids
        ):
            raise ProtocolError(f"Context 1.3 team identity changed: {team_id}")
        if team_id in protocol.data["population"]["cold_start_team_ids"]:
            fallback = Context13FallbackSource.from_history_annual_source(
                source=history_source,
                target_season=2026,
                trained_through_season=2025,
                team_id=team_id,
                team_name=row.team_name,
                population=row.population,
                cold_start_reason=row.cold_start_reason or "fcs_to_fbs_transition",
            )
            if not np.array_equal(fallback.pmf, published):
                raise ProtocolError(f"cold-start Context 1.3 PMF changed: {team_id}")
            prior = Context13PriorInput.cold_start(
                fitted_instance=instance,
                fitted_model_source=fit_source,
                transfer_provenance=transfer,
                fallback_source=fallback,
            )
            exact = fallback.pmf
        else:
            if row.lag1_z is None or row.lag_zs:
                raise ProtocolError(
                    f"fitted Context 1.3 inference row is invalid: {team_id}"
                )
            exact = model.pmf(
                {name: row.features[name] for name in model.feature_names},
                np.asarray(row.lag1_z),
                row.population,
            )
            # The retained CSV rounds each probability to 12 decimals. Its
            # serialization difference is checked before using the exact
            # model PMF as the shared C1.3/candidate source prior.
            if not np.allclose(exact, published, rtol=0, atol=5.01e-13):
                raise ProtocolError(f"Context 1.3 source PMF parity failed: {team_id}")
            prior = Context13PriorInput.fitted(
                model=model,
                fitted_instance=instance,
                fitted_model_source=fit_source,
                inference_row=row,
                transfer_provenance=transfer,
                prior_pmf=exact,
                expected_team_id=team_id,
                expected_target_season=2026,
            )
        candidate = construct_candidate_prior(prior)
        if (
            candidate.candidate_semantics_sha256
            != protocol.data["models"]["candidate"]["candidate_semantics_sha256"]
            or candidate.source_prior_pmf_sha256 != sha256_json(exact.tolist())
            or not np.allclose(
                prior.source_prior_pmf(),
                exact,
                rtol=0,
                atol=protocol.data["guardrails"]["source_prior_pmf_absolute_tolerance"],
            )
        ):
            raise ProtocolError(f"candidate source or semantics mismatch: {team_id}")
        decomposition = prior.location_decomposition()
        if decomposition is None or decomposition.context_only_subtotal <= 0:
            if not np.array_equal(candidate.pmf, exact):
                raise ProtocolError(f"nonpositive/cold candidate changed: {team_id}")
        else:
            points = moderated_location_points(decomposition, float(spec["alpha"]))
            offset = (float(spec["alpha"]) - 1.0) * decomposition.context_only_subtotal
            if not np.allclose(
                points - decomposition.conditional_location_points,
                offset,
                rtol=0,
                atol=1e-14,
            ):
                raise ProtocolError(
                    f"candidate centered mixture shifted incorrectly: {team_id}"
                )
            expected_pmf = conditional_rank_mixture_pmf(
                points, decomposition.residual_scale, decomposition.population
            )
            if not np.array_equal(candidate.pmf, expected_pmf):
                raise ProtocolError(
                    f"candidate positive moderation mismatch: {team_id}"
                )
        candidate_teams[team_id] = Team(team_id, row.team_name, "fbs", candidate.pmf)
        baseline_teams[team_id] = Team(team_id, row.team_name, "fbs", exact)
        history_pmf = np.asarray(json.loads(history_rows[team_id]["pmf"]), dtype=float)
        history_teams[team_id] = Team(team_id, row.team_name, "fbs", history_pmf)
    if set(candidate_teams) != set(ids):
        raise ProtocolError("candidate prior construction is incomplete")
    # Preserve the retained CSV team order. Loopy message updates are
    # deterministic in that order, and this also gives History exact replay.
    return {
        "candidate": {team_id: candidate_teams[team_id] for team_id in context_rows},
        "context_1_3": {team_id: baseline_teams[team_id] for team_id in context_rows},
        "history_1_1": {team_id: history_teams[team_id] for team_id in history_rows},
    }


def construct_forecast_states(
    root: Path, protocol: RegisteredProtocol, inputs: dict[str, Any]
) -> dict[tuple[str, str], ForecastState]:
    """Infer matched deterministic posteriors at every official origin."""
    priors = construct_frozen_priors(root, protocol, inputs)
    evidence = protocol.data["evidence"]
    likelihood_path = root / evidence["likelihood_path"]
    if _sha(likelihood_path) != evidence["likelihood_sha256"]:
        raise ProtocolError("Historical Likelihood V1 identity changed")
    likelihood = load_likelihood(likelihood_path)
    locked_fcs = set(evidence["fcs_fallback"]["population_team_ids"])
    fcs_population = evidence["fcs_fallback"]["population_size"]
    result: dict[tuple[str, str], ForecastState] = {}
    for origin in protocol.data["forecast_origins"]:
        slot = origin["publication_slot"]
        context_rows = _read_csv(
            root / origin["context_files"]["included_games.csv"]["path"]
        )
        history_rows = _read_csv(
            root / origin["history_files"]["included_games.csv"]["path"]
        )
        if (
            context_rows != history_rows
            or len(context_rows) != origin["included_game_count"]
        ):
            raise ProtocolError(f"{slot}: included-game evidence differs across arms")
        cutoff = datetime.fromisoformat(origin["logical_cutoff_utc"])
        games: list[Game] = []
        fcs_names: dict[str, str] = {}
        for row in context_rows:
            if not _bool(row["completed"]) or _kickoff(row) > cutoff:
                raise ProtocolError(
                    f"{slot}: future or unfinished result in forecast evidence"
                )
            game_id = row["id"]
            for side in ("home", "away"):
                if row[f"{side}Classification"].casefold() == "fcs":
                    fcs_names[row[f"{side}Id"]] = row[f"{side}Team"]
            games.append(
                Game(
                    game_id,
                    row["homeId"],
                    row["awayId"],
                    row["homeClassification"].casefold(),
                    row["awayClassification"].casefold(),
                    _score_int(row["homePoints"], game_id),
                    _score_int(row["awayPoints"], game_id),
                    _bool(row["neutralSite"]),
                )
            )
        snapshot_fcs: tuple[str, ...] | None = None
        for arm, metadata_name in (
            ("candidate", "context"),
            ("context_1_3", "context"),
            ("history_1_1", "history"),
        ):
            files = origin[f"{metadata_name}_files"]
            metadata = json.loads((root / files["metadata.json"]["path"]).read_text())
            fcs_ids = tuple(str(value) for value in metadata["fcs_fallback_team_ids"])
            if (
                len(fcs_ids) != len(set(fcs_ids))
                or not set(fcs_ids) <= locked_fcs
                or metadata["fcs_population_size"] != fcs_population
                or (snapshot_fcs is not None and fcs_ids != snapshot_fcs)
            ):
                raise ProtocolError(f"{slot}: FCS fallback network changed")
            snapshot_fcs = fcs_ids
            team_map = dict(priors[arm])
            for team_id in fcs_ids:
                team_map[team_id] = Team(
                    team_id,
                    fcs_names.get(team_id, f"FCS {team_id}"),
                    "fcs",
                    np.full(fcs_population, 1.0 / fcs_population),
                )
            if any(
                game.home_id not in team_map or game.away_id not in team_map
                for game in games
            ):
                raise ProtocolError(f"{slot}: included game outside inference network")
            posterior = infer_registered_posterior(
                list(team_map.values()),
                games,
                likelihood,
                protocol,
                origin=slot,
                model=arm,
            )
            retained_error: float | None = None
            if arm != "candidate":
                retained_error = _retained_posterior_parity(
                    root / files["posterior_pmfs.csv"]["path"], posterior.pmfs
                )
            audit = {
                "origin": slot,
                "model": arm,
                "included_game_count": len(games),
                "team_count": len(team_map),
                "fcs_fallback_count": len(fcs_ids),
                "iterations": posterior.iterations,
                "converged": posterior.converged,
                "included_games_sha256": files["included_games.csv"]["sha256"],
                "snapshot_metadata_sha256": files["metadata.json"]["sha256"],
                "source_prior_sha256": (
                    protocol.data["models"]["context_1_3"]["predictions_file_sha256"]
                    if arm in {"candidate", "context_1_3"}
                    else protocol.data["models"]["history_1_1"][
                        "predictions_file_sha256"
                    ]
                ),
                "prior_pmfs_sha256": posterior_pmfs_sha256(
                    {team_id: team.prior for team_id, team in team_map.items()}
                ),
                "posterior_sha256": posterior_pmfs_sha256(posterior.pmfs),
                "retained_posterior_sha256": (
                    metadata["posterior_pmfs_sha256"] if arm != "candidate" else None
                ),
                "retained_posterior_max_abs_error": retained_error,
            }
            result[slot, arm] = ForecastState(
                slot, arm, team_map, posterior.pmfs, audit
            )
    return result


def _retained_posterior_parity(
    path: Path, reconstructed: dict[str, np.ndarray]
) -> float:
    """Check numerical parity with the registered serialized baseline state."""
    rows: dict[str, list[float]] = defaultdict(list)
    ranks: dict[str, list[int]] = defaultdict(list)
    for row in _read_csv(path):
        rows[row["team_id"]].append(float(row["probability"]))
        ranks[row["team_id"]].append(int(row["rank"]))
    if set(rows) != set(reconstructed) or any(
        ranks[team_id] != list(range(1, len(pmf) + 1))
        for team_id, pmf in reconstructed.items()
    ):
        raise ProtocolError(f"retained posterior population or ranks changed: {path}")
    error = max(
        float(np.max(np.abs(np.asarray(rows[team_id]) - pmf)))
        for team_id, pmf in reconstructed.items()
    )
    # Annual C1.3 PMFs were serialized to twelve decimal places before the
    # retained posterior run. Reconstruction uses their exact model values.
    if error > 1e-9:
        raise ProtocolError(f"retained posterior parity failed: {path} ({error})")
    return error


def infer_registered_posterior(
    teams: list[Team],
    games: list[Game],
    likelihood: LikelihoodV1,
    protocol: RegisteredProtocol,
    *,
    origin: str,
    model: str,
) -> PosteriorResult:
    """Run the registered inference settings and require convergence."""
    config = protocol.data["evidence"]["inference"]
    posterior = infer_posterior(
        teams,
        games,
        likelihood,
        max_iterations=config["max_iterations"],
        tolerance=config["tolerance"],
        damping=config["damping"],
    )
    if not posterior.converged:
        raise ProtocolError(f"{origin} {model}: posterior inference did not converge")
    return posterior


def score_predictive_game(
    game: Game,
    *,
    week: int,
    state: ForecastState,
    likelihood: LikelihoodV1,
    locked_fcs_ids: set[str],
    fcs_population: int,
    prediction_teams: dict[str, Team] | None = None,
) -> dict[str, object]:
    """Score a supplied final against the existing exact rank mixture."""
    teams = (
        prediction_teams
        if prediction_teams is not None
        else posterior_prediction_teams(list(state.teams.values()), state.posterior)
    )
    for team_id, subdivision in (
        (game.home_id, game.home_subdivision),
        (game.away_id, game.away_subdivision),
    ):
        if team_id not in teams:
            if subdivision != "fcs" or team_id not in locked_fcs_ids:
                raise ProtocolError(
                    f"game {game.game_id}: team absent from forecast population"
                )
            teams[team_id] = Team(
                team_id,
                f"FCS {team_id}",
                "fcs",
                np.full(fcs_population, 1.0 / fcs_population),
            )
    scheduled = ScheduledGame(
        game.game_id,
        game.home_id,
        game.away_id,
        game.home_subdivision,
        game.away_subdivision,
        game.neutral_site,
    )
    home, away = teams[game.home_id], teams[game.away_id]
    prediction = predict_game(scheduled, home, away, likelihood)
    locations, weights = predictive_components(scheduled, home, away, likelihood)
    actual = game.home_points - game.away_points
    density = float(
        np.dot(
            weights,
            student_t.pdf(
                (actual - locations) / likelihood.scale,
                likelihood.degrees_of_freedom,
            )
            / likelihood.scale,
        )
    )
    nll = -float(np.log(max(density, 1e-300)))
    actual_win = 1.0 if actual > 0 else 0.0 if actual < 0 else 0.5
    intervals = {
        size: getattr(prediction, f"margin_interval_{size}") for size in (50, 80, 95)
    }
    result: dict[str, object] = {
        "game_id": game.game_id,
        "origin": state.origin,
        "stratum": PROSPECTIVE_LABEL
        if week == 5
        else "candidate_score_unseen_prior_weeks",
        "week": week,
        "home_id": game.home_id,
        "away_id": game.away_id,
        "model": state.model,
        "actual_home_margin": actual,
        "margin_nll": nll,
        "expected_home_margin": prediction.expected_home_margin,
        "margin_mae": abs(prediction.expected_home_margin - actual),
        "home_win_probability": prediction.home_win_probability,
        "win_brier": (prediction.home_win_probability - actual_win) ** 2,
    }
    for size, interval in intervals.items():
        result[f"interval_{size}_low"] = interval[0]
        result[f"interval_{size}_high"] = interval[1]
        result[f"coverage_{size}"] = int(interval[0] <= actual <= interval[1])
    return result


def score_complete_source(
    audit: SourceAudit,
    states: dict[tuple[str, str], ForecastState],
    protocol: RegisteredProtocol,
    root: Path,
) -> list[dict[str, object]]:
    """Open results only after the source gate and all states have passed."""
    if audit.status != "SOURCE_COMPLETE":
        raise ProtocolError("candidate scoring is forbidden for an incomplete source")
    evidence = protocol.data["evidence"]
    likelihood = load_likelihood(root / evidence["likelihood_path"])
    fcs_ids = set(evidence["fcs_fallback"]["population_team_ids"])
    fcs_population = evidence["fcs_fallback"]["population_size"]
    prediction_teams = {
        key: posterior_prediction_teams(list(state.teams.values()), state.posterior)
        for key, state in states.items()
    }
    rows: list[dict[str, object]] = []
    for source in audit.games:
        game_id = source["id"]
        game = Game(
            game_id,
            source["homeId"],
            source["awayId"],
            source["homeClassification"].casefold(),
            source["awayClassification"].casefold(),
            _score_int(source["homePoints"], game_id),
            _score_int(source["awayPoints"], game_id),
            _bool(source["neutralSite"]),
        )
        origin = audit.assignments[game_id]
        for model in MODELS:
            rows.append(
                score_predictive_game(
                    game,
                    week=int(source["week"]),
                    state=states[origin, model],
                    likelihood=likelihood,
                    locked_fcs_ids=fcs_ids,
                    fcs_population=fcs_population,
                    prediction_teams=prediction_teams[origin, model],
                )
            )
    return sorted(rows, key=lambda row: (str(row["game_id"]), str(row["model"])))


def _group_summary(
    rows: list[dict[str, object]], protocol: RegisteredProtocol
) -> dict[str, Any]:
    if not rows:
        return {"eligible_game_count": 0, "primary": None, "secondary": None}
    by_game: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        game_id, model = str(row["game_id"]), str(row["model"])
        if model not in MODELS or model in by_game[game_id]:
            raise ProtocolError(f"game {game_id}: duplicate or unknown model score")
        by_game[game_id][model] = row
    for game_id, models in by_game.items():
        if set(models) != set(MODELS):
            raise ProtocolError(f"game {game_id}: score arms are not paired")
        for field in (
            "origin",
            "stratum",
            "week",
            "home_id",
            "away_id",
            "actual_home_margin",
        ):
            if len({str(item[field]) for item in models.values()}) != 1:
                raise ProtocolError(f"game {game_id}: {field} differs across arms")
    ids = sorted(by_game)
    deltas = {
        game_id: float(by_game[game_id]["candidate"]["margin_nll"])
        - float(by_game[game_id]["context_1_3"]["margin_nll"])
        for game_id in ids
    }
    mean_delta, low, high = paired_mean_bootstrap(deltas, protocol)
    primary = {
        "candidate_mean_margin_nll": math.fsum(
            float(by_game[game_id]["candidate"]["margin_nll"]) for game_id in ids
        )
        / len(ids),
        "context_1_3_mean_margin_nll": math.fsum(
            float(by_game[game_id]["context_1_3"]["margin_nll"]) for game_id in ids
        )
        / len(ids),
        "paired_mean_delta": mean_delta,
        "paired_median_delta": float(np.median([deltas[game_id] for game_id in ids])),
        "paired_mean_95_percent_interval": [low, high],
    }
    secondary = {}
    for model in protocol.data["analysis"]["secondary"]["models"]:
        secondary[model] = {
            metric: math.fsum(float(by_game[game_id][model][field]) for game_id in ids)
            / len(ids)
            for metric, field in (
                ("margin_mae", "margin_mae"),
                ("win_brier", "win_brier"),
                ("interval_coverage_50", "coverage_50"),
                ("interval_coverage_80", "coverage_80"),
                ("interval_coverage_95", "coverage_95"),
            )
        }
    return {"eligible_game_count": len(ids), "primary": primary, "secondary": secondary}


def decide_validation(
    full: dict[str, Any],
    prospective: dict[str, Any],
    protocol: RegisteredProtocol,
    *,
    source_complete: bool,
    guardrails_passed: bool,
) -> str:
    """Apply only the frozen asymmetric rule to validated statistics."""
    if not source_complete or not guardrails_passed:
        raise ProtocolError("source or structural guardrail forbids a decision")
    if full["primary"] is None:
        return "inconclusive"
    primary = full["primary"]
    return classify_primary(
        float(primary["paired_mean_delta"]),
        float(primary["paired_mean_95_percent_interval"][0]),
        float(primary["paired_mean_95_percent_interval"][1]),
        int(full["eligible_game_count"]),
        int(prospective["eligible_game_count"]),
        protocol,
    )


def aggregate_scores(
    rows: list[dict[str, object]], protocol: RegisteredProtocol
) -> dict[str, Any]:
    """Produce paired full, origin, and prospective Week 5 summaries."""
    allowed_origins = {
        origin["publication_slot"] for origin in protocol.data["forecast_origins"]
    }
    if any(row["origin"] not in allowed_origins for row in rows):
        raise ProtocolError("score row names an unregistered forecast origin")
    full = _group_summary(rows, protocol)
    origins = {
        origin["publication_slot"]: _group_summary(
            [row for row in rows if row["origin"] == origin["publication_slot"]],
            protocol,
        )
        for origin in protocol.data["forecast_origins"]
    }
    prospective = _group_summary(
        [row for row in rows if row["stratum"] == PROSPECTIVE_LABEL], protocol
    )
    return {
        "full_sample": full,
        "by_origin": origins,
        PROSPECTIVE_LABEL: prospective,
        "decision": decide_validation(
            full,
            prospective,
            protocol,
            source_complete=True,
            guardrails_passed=True,
        ),
    }


def _csv_bytes(rows: list[dict[str, object]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def failure_object(
    audit: SourceAudit,
    protocol: RegisteredProtocol,
    inputs: dict[str, Any],
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "status": audit.status,
        "reason": reason,
        "decision": "unavailable",
        "candidate_scores_opened": False,
        "missing_or_unresolved_game_ids": list(audit.unresolved_ids),
        "registered_expected_scope": {
            "season": 2026,
            "game_weeks": "1_through_5",
            "prospective_game_week": 5,
            "prospective_forecast_origin": protocol.data["evidence"]["week_6_origin"],
            "expected_game_count": len(audit.expected_ids),
            "expected_week_5_game_count": sum(
                row["week"] == "5" for row in inputs["registered_schedule"]
            ),
        },
        "observed_completed_count": audit.completed_count,
        "source_sha256": audit.source_sha256,
        "source_hashes": {
            "canonical_games_csv": audit.source_sha256,
            **inputs["source_sha256"],
        },
        "protocol_sha256": protocol.sha256,
        "validator_inputs_sha256": VALIDATOR_INPUTS_SHA256,
    }


def validation_abort_object(
    *,
    reason: str,
    abort_stage: str,
    candidate_scores_opened: bool,
    protocol: RegisteredProtocol | None,
    inputs: dict[str, Any] | None,
    audit: SourceAudit | None,
) -> dict[str, object]:
    """Describe a failed execution without leaving a success-shaped result."""
    registered_schedule = inputs["registered_schedule"] if inputs else []
    expected_scope: dict[str, object] = {
        "season": 2026,
        "game_weeks": "1_through_5",
        "prospective_game_week": 5,
        "expected_game_count": len(registered_schedule),
        "expected_week_5_game_count": sum(
            row["week"] == "5" for row in registered_schedule
        ),
    }
    if protocol is not None:
        expected_scope["prospective_forecast_origin"] = protocol.data["evidence"][
            "week_6_origin"
        ]
    return {
        "status": "VALIDATION_ABORTED",
        "reason": reason,
        "abort_stage": abort_stage,
        "decision": "unavailable",
        "candidate_scores_opened": candidate_scores_opened,
        "missing_or_unresolved_game_ids": (
            list(audit.unresolved_ids) if audit is not None else []
        ),
        "registered_expected_scope": expected_scope,
        "observed_completed_count": (
            audit.completed_count if audit is not None else None
        ),
        "source_sha256": audit.source_sha256 if audit is not None else None,
        "source_hashes": {
            **({"canonical_games_csv": audit.source_sha256} if audit else {}),
            **(inputs["source_sha256"] if inputs else {}),
        },
        "protocol_sha256": protocol.sha256 if protocol is not None else None,
        "validator_inputs_sha256": VALIDATOR_INPUTS_SHA256,
    }


def clear_output_products(path: Path, report_path: Path | None = None) -> None:
    """Remove products from any previous invocation before a new one starts."""
    path.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_PRODUCTS:
        (path / name).unlink(missing_ok=True)
    if report_path is not None:
        report_path.unlink(missing_ok=True)


def write_failure(
    path: Path, value: dict[str, object], *, report_path: Path | None = None
) -> None:
    clear_output_products(path, report_path)
    (path / "failure.json").write_bytes(_json_bytes(value))


def write_source_audit(
    path: Path, value: dict[str, object], *, report_path: Path | None = None
) -> None:
    clear_output_products(path, report_path)
    (path / "source_audit.json").write_bytes(_json_bytes(value))


def write_success(
    output: Path,
    rows: list[dict[str, object]],
    states: dict[tuple[str, str], ForecastState],
    audit: SourceAudit,
    summary: dict[str, Any],
    protocol: RegisteredProtocol,
    source_manifest: dict[str, Any],
    *,
    report_path: Path | None = None,
) -> None:
    """Write only sorted deterministic products after a complete run."""
    if audit.status != "SOURCE_COMPLETE":
        raise ProtocolError("cannot write a success artifact for an incomplete source")
    clear_output_products(output, report_path)
    root = Path(__file__).resolve().parents[2]
    provenance = {
        "protocol_sha256": protocol.sha256,
        "preregistration_commit": "74818f43426609b5d16b4d3e160f24af1dbb8c79",
        "validator_module_sha256": _sha(Path(__file__)),
        "validator_script_sha256": _sha(
            root / protocol.data["artifacts_for_later_validator"]["script"]
        ),
        "candidate_semantics_sha256": protocol.data["models"]["candidate"][
            "candidate_semantics_sha256"
        ],
        "candidate_builder_sha256": protocol.data["models"]["candidate"][
            "builder_file_sha256"
        ],
        "context_1_3_source_sha256": protocol.data["models"]["context_1_3"][
            "predictions_file_sha256"
        ],
        "history_1_1_source_sha256": protocol.data["models"]["history_1_1"][
            "predictions_file_sha256"
        ],
        "historical_likelihood_sha256": protocol.data["evidence"]["likelihood_sha256"],
        "fbs_population_sha256": protocol.data["population"]["team_ids_sha256"],
        "fcs_population_sha256": protocol.data["evidence"]["fcs_fallback"][
            "population_team_ids_sha256"
        ],
        "inference_configuration": protocol.data["evidence"]["inference"],
        "validator_inputs_sha256": VALIDATOR_INPUTS_SHA256,
        "game_source_sha256": audit.source_sha256,
        "source_manifest": source_manifest,
        "forecast_origins": [state.audit for _, state in sorted(states.items())],
    }
    (output / "game_results.csv").write_bytes(_csv_bytes(rows, SCORE_FIELDS))
    (output / "by_origin.json").write_bytes(_json_bytes(summary["by_origin"]))
    fields = tuple(next(iter(states.values())).audit) if states else ()
    (output / "evidence_audit.csv").write_bytes(
        _csv_bytes([state.audit for _, state in sorted(states.items())], fields)
    )
    (output / "exclusions.csv").write_bytes(
        _csv_bytes(
            list(audit.exclusions), ("game_id", "disposition", "reason", "evidence")
        )
    )
    (output / "summary.json").write_bytes(_json_bytes(summary))
    (output / "provenance.json").write_bytes(_json_bytes(provenance))
    primary = summary["full_sample"]["primary"]
    report = "\n".join(
        [
            "# Context 1.4 in-season validation",
            "",
            f"Decision: **{summary['decision']}**",
            "",
            f"Scored games: {summary['full_sample']['eligible_game_count']}",
            f"Prospective Week 5 games: {summary[PROSPECTIVE_LABEL]['eligible_game_count']}",
            f"Paired mean margin NLL delta (1.4 − 1.3): {primary['paired_mean_delta'] if primary else 'unavailable'}",
            "",
            (
                "The September 27 Week 5 snapshot forecasts the prospective Week 5 games. "
                "The merged #164 preregistration called that game slice Week 6; its "
                "frozen origins, sample floors, bootstrap, and decision rule were retained."
            ),
            "",
            (
                "The retained Context 1.3 fit is a legacy attestation without a verified "
                "training-corpus digest."
            ),
            "",
        ]
    )
    target = report_path or output / "context_v1_4_validation_result.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
