"""Durable, consumer-oriented posterior snapshot bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from gippyrank.methodology import (
    CONTEXT_PRIOR_VERSION,
    HISTORICAL_LIKELIHOOD_VERSION,
    HISTORY_PRIOR_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    TEAM_SEASON_SCHEMA_VERSION,
)
from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.game_evidence import build_team_season_artifact
from gippyrank.posterior.season_simulation import (
    SEASON_SIMULATION_SCHEMA_VERSION,
    SeasonSimulationConfig,
)
from gippyrank.preseason import pmf_summaries

SCHEMA_VERSION = SNAPSHOT_SCHEMA_VERSION
RANKING_FAMILY = "predictive"
PriorFamily = Literal["context", "history"]
SnapshotType = Literal["preseason", "weekly", "live"]

INCLUDED_GAME_FIELDS = (
    "id",
    "season",
    "week",
    "seasonType",
    "startDate",
    "completed",
    "neutralSite",
    "conferenceGame",
    "homeId",
    "homeTeam",
    "homeClassification",
    "homeConference",
    "homePoints",
    "awayId",
    "awayTeam",
    "awayClassification",
    "awayConference",
    "awayPoints",
)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    directory: Path
    metadata: dict[str, object]


@dataclass(frozen=True)
class CorpusProvenance:
    source_mode: str
    source_kind: str
    source_retrieved_at: datetime | None
    source_retrieval_times: dict[str, datetime]
    source_response_hashes: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_values_sha256(values: list[str]) -> str:
    """Hash an ordered canonical list without depending on JSON formatting."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def included_game_rows_sha256(rows: list[dict[str, str]]) -> str:
    """Hash the ordered game evidence consumed by posterior inference.

    The source CSV is intentionally not hashed byte-for-byte: line endings and
    CSV quoting are serialization details, while the fixed field/value surface
    below is the evidence contract shared by source and replay snapshots.
    """
    digest = hashlib.sha256()
    for row in rows:
        canonical = {
            field: str(row.get(field, ""))
            for field in INCLUDED_GAME_FIELDS
        }
        digest.update(
            json.dumps(
                canonical,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    """Return a repository-relative artifact path when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def snapshot_id(
    season: int,
    snapshot_type: SnapshotType,
    prior_family: PriorFamily,
    cutoff: datetime | date | None = None,
    prior_model_version: str | None = None,
    lineage_suffix: str | None = None,
) -> str:
    suffix = (
        f"-v{prior_model_version}"
        if prior_model_version in {"1.2", "1.3"}
        else ""
    )
    suffix += f"-{lineage_suffix}" if lineage_suffix else ""
    if snapshot_type == "preseason":
        base = f"{season}-preseason-{prior_family}"
        return f"{base}{suffix}"
    if cutoff is None:
        raise ValueError("weekly and live snapshots require an explicit cutoff")
    if snapshot_type == "weekly":
        stamp = cutoff.isoformat().replace("+00:00", "Z").replace(":", "-")
        base = f"{season}-weekly-{stamp}-{prior_family}"
        return f"{base}{suffix}"
    stamp = cutoff.isoformat().replace("+00:00", "Z").replace(":", "-")
    base = f"{season}-live-{stamp}-{prior_family}"
    return f"{base}{suffix}"


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prior_path(
    root: Path,
    family: PriorFamily,
    season: int,
    prior_model_version: str | None = None,
) -> Path:
    if family == "context":
        version = prior_model_version or CONTEXT_PRIOR_VERSION
        if version == "1.3":
            family_root = "context_v1_3"
        elif version == "1.2":
            family_root = "context"
        else:
            raise ValueError(f"unsupported Context prior version: {version}")
    else:
        family_root = family
    annual = root / f"data/processed/preseason/{family_root}/annual/{season}/predictions.csv"
    # Historical prior rows are retained in the frozen multi-season artifact;
    # 2026 additionally has its explicit annually frozen instance.
    return (
        annual
        if annual.exists()
        else root / f"data/processed/preseason/{family_root}/predictions.csv"
    )


def load_teams(
    root: Path,
    season: int,
    family: PriorFamily,
    prior_model_version: str | None = None,
) -> tuple[list[Team], dict[str, dict[str, str]], Path]:
    path = _prior_path(root, family, season, prior_model_version)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen {family} prior is unavailable for {season}: {path}"
        )
    metadata: dict[str, dict[str, str]] = {}
    teams: list[Team] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["season"]) != season:
                continue
            subdivision = row["subdivision"].lower()
            if subdivision not in {"fbs", "fcs"}:
                continue
            team = Team(
                row["team_id"],
                row["team_name"],
                subdivision,
                np.asarray(json.loads(row["pmf"]), dtype=float),
            )
            teams.append(team)
            metadata[team.team_id] = row
    return teams, metadata, path


def _parse_completed(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def _as_utc_datetime(cutoff: datetime | date) -> datetime:
    result = (
        datetime.combine(cutoff, datetime.max.time(), tzinfo=UTC)
        if isinstance(cutoff, date) and not isinstance(cutoff, datetime)
        else cutoff
    )
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def corpus_provenance(root: Path, season: int) -> CorpusProvenance:
    """Read an explicit current-season CFBD acquisition manifest when present."""
    directory = root / "data/raw/cfbd/games"
    manifests = [
        directory / f"{season}{suffix}.json.provenance.json"
        for suffix in ("", "-fcs")
    ]
    if not all(path.exists() for path in manifests):
        return CorpusProvenance("historical_frozen", "frozen_game_corpus", None, {}, {})
    values = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    retrieval_times = {
        classification: datetime.fromisoformat(value["retrieved_at"]).astimezone(UTC)
        for classification, value in zip(("fbs", "fcs"), values)
    }
    return CorpusProvenance(
        "current_cached_cfbd",
        "cfbd_api_schedule",
        min(retrieval_times.values()),
        retrieval_times,
        {path.name: value["content_sha256"] for path, value in zip(manifests, values)},
    )


def filter_games(
    root: Path, season: int, cutoff: datetime | date | None, snapshot_type: SnapshotType
) -> tuple[list[Game], list[dict[str, str]], int, Path]:
    path = root / "data/processed/cfbd/games.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    games: list[Game] = []
    included_rows: list[dict[str, str]] = []
    lower_division = 0
    cutoff_dt = _as_utc_datetime(cutoff) if cutoff is not None else None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["season"]) != season or not _parse_completed(row["completed"]):
                continue
            try:
                when = datetime.fromisoformat(row["startDate"])
            except (KeyError, TypeError, ValueError):
                # Keep malformed completed rows out of inference. The team
                # season builder will classify them as unresolved schedule
                # state rather than silently dropping them.
                continue
            if snapshot_type == "preseason" or (
                cutoff_dt is not None and when > cutoff_dt
            ):
                continue
            if cutoff_dt is None:
                raise ValueError("non-preseason snapshot needs cutoff")
            home, away = (
                row["homeClassification"].lower(),
                row["awayClassification"].lower(),
            )
            if home not in {"fbs", "fcs"} or away not in {"fbs", "fcs"}:
                lower_division += 1
                continue
            game = Game(
                str(row["id"]),
                row["homeId"],
                row["awayId"],
                home,
                away,
                int(row["homePoints"]),
                int(row["awayPoints"]),
                _parse_completed(row["neutralSite"]),
            )
            games.append(game)
            included_rows.append(row)
    return games, included_rows, lower_division, path


def _read_included_game_rows(path: Path) -> list[dict[str, str]]:
    """Read the fixed game-evidence surface from a snapshot artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(INCLUDED_GAME_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: included game evidence is missing {sorted(missing)}"
            )
        rows: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for raw in reader:
            row = {
                field: str(raw.get(field) or "") for field in INCLUDED_GAME_FIELDS
            }
            game_id = row["id"]
            if not game_id:
                raise ValueError(f"{path}: included game evidence has a blank ID")
            if game_id in seen_ids:
                raise ValueError(f"{path}: included game ID appears more than once: {game_id}")
            seen_ids.add(game_id)
            rows.append(row)
    return rows


def historical_schedule_rows_from_site_artifact(
    path: Path,
    *,
    season: int,
    expected_snapshot_id: str,
    expected_included_game_ids: list[str],
    required_included_game_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """Convert a frozen weekly site schedule into builder-compatible rows.

    A retrospective replay has two intentionally different inputs: the
    included-game CSV is the immutable inference evidence, while this frozen
    site artifact supplies the historical schedule surface used for team
    seasons and future predictions.  Keep the conversion strict so a mutable
    or mismatched publication cannot silently become presentation input.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("artifact_kind") != "weekly_games":
        raise ValueError(f"{path}: expected a weekly_games artifact")
    if value.get("snapshot_id") != expected_snapshot_id:
        raise ValueError(
            f"{path}: snapshot ID does not match frozen evidence: "
            f"{value.get('snapshot_id')!r} != {expected_snapshot_id!r}"
        )
    declared_included = [str(item) for item in value.get("included_game_ids", [])]
    expected_included = [str(item) for item in expected_included_game_ids]
    if declared_included != expected_included:
        raise ValueError(f"{path}: included game IDs do not match frozen evidence")
    weeks = value.get("weeks")
    if not isinstance(weeks, list):
        raise TypeError(f"{path}: weekly schedule is missing weeks")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for week in weeks:
        if not isinstance(week, dict) or not isinstance(week.get("games"), list):
            raise TypeError(f"{path}: weekly schedule contains an invalid week")
        for game in week["games"]:
            if not isinstance(game, dict):
                raise TypeError(f"{path}: weekly schedule contains an invalid game")
            game_id = str(game.get("game_id", ""))
            if not game_id:
                raise ValueError(f"{path}: weekly schedule contains a blank game ID")
            if game_id in seen_ids:
                raise ValueError(f"{path}: game ID appears more than once: {game_id}")
            seen_ids.add(game_id)
            home = game.get("home_team")
            away = game.get("away_team")
            if not isinstance(home, dict) or not isinstance(away, dict):
                raise TypeError(f"{path}: {game_id} is missing team identities")
            home_id = str(home.get("team_id", ""))
            away_id = str(away.get("team_id", ""))
            if not home_id or not away_id:
                raise ValueError(f"{path}: {game_id} is missing a team ID")
            score = game.get("score")
            home_points = ""
            away_points = ""
            if score is not None:
                if not isinstance(score, dict) or "home" not in score or "away" not in score:
                    raise ValueError(f"{path}: {game_id} has an invalid score")
                home_points = str(score["home"])
                away_points = str(score["away"])
            state = str(game.get("state", ""))
            rows.append(
                {
                    "id": game_id,
                    "season": str(season),
                    "week": str(game.get("week", "")),
                    "seasonType": str(game.get("season_type", "regular") or "regular"),
                    "startDate": str(game.get("date", "")),
                    "completed": "True" if state == "completed" and score is not None else "False",
                    "neutralSite": "True" if bool(game.get("neutral_site")) else "False",
                    "conferenceGame": "True" if bool(game.get("conference_game")) else "False",
                    "homeId": home_id,
                    "homeTeam": str(home.get("team_name", "")),
                    "homeClassification": str(home.get("subdivision", "")),
                    "homeConference": str(home.get("conference", "")),
                    "homePoints": home_points,
                    "awayId": away_id,
                    "awayTeam": str(away.get("team_name", "")),
                    "awayClassification": str(away.get("subdivision", "")),
                    "awayConference": str(away.get("conference", "")),
                    "awayPoints": away_points,
                    "status": "cancelled" if state == "cancelled" else "",
                }
            )

    required_ids = (
        expected_included
        if required_included_game_ids is None
        else [str(game_id) for game_id in required_included_game_ids]
    )
    missing = [game_id for game_id in required_ids if game_id not in seen_ids]
    if missing:
        raise ValueError(f"{path}: frozen schedule is missing included games: {missing}")
    return rows


def _game_from_included_row(row: dict[str, str]) -> Game:
    """Convert one frozen CSV row into the exact inference game object."""
    home_subdivision = row["homeClassification"].casefold()
    away_subdivision = row["awayClassification"].casefold()
    if home_subdivision not in {"fbs", "fcs"} or away_subdivision not in {
        "fbs",
        "fcs",
    }:
        raise ValueError(
            f"included game {row['id']} has unsupported subdivisions: "
            f"{home_subdivision!r}, {away_subdivision!r}"
        )
    try:
        home_points = int(row["homePoints"])
        away_points = int(row["awayPoints"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"included game {row['id']} has invalid scores") from error
    return Game(
        row["id"],
        row["homeId"],
        row["awayId"],
        home_subdivision,
        away_subdivision,
        home_points,
        away_points,
        _parse_completed(row["neutralSite"]),
    )


def _metadata_datetime(value: object, *, field: str, source: Path) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{source}: invalid {field}: {value!r}") from error
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _frozen_provenance(metadata: dict[str, object], source: Path) -> CorpusProvenance:
    retrieval_values = metadata.get("source_retrieval_times", {})
    response_values = metadata.get("source_response_hashes", {})
    if not isinstance(retrieval_values, dict) or not isinstance(response_values, dict):
        raise TypeError(f"{source}: frozen source provenance maps are invalid")
    return CorpusProvenance(
        str(metadata.get("source_mode", "historical_frozen")),
        str(metadata.get("source_kind", "frozen_game_corpus")),
        _metadata_datetime(
            metadata.get("source_retrieved_at"),
            field="source_retrieved_at",
            source=source,
        ),
        {
            str(key): value
            for key, raw in retrieval_values.items()
            if (value := _metadata_datetime(raw, field="source_retrieval_times", source=source))
            is not None
        },
        {str(key): str(value) for key, value in response_values.items()},
    )


def _source_inference_configuration(
    source: Path, metadata: dict[str, object]
) -> dict[str, object]:
    value = metadata.get("posterior_inference_configuration")
    if isinstance(value, dict):
        configuration = dict(value)
    else:
        diagnostics_path = source / "diagnostics.json"
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.is_file()
            else {}
        )
        configuration = {
            "implementation": diagnostics.get(
                "inference", "deterministic_damped_loopy_sum_product"
            ),
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
        }
    required = {"implementation", "max_iterations", "tolerance", "damping"}
    missing = required - configuration.keys()
    if missing:
        raise ValueError(
            f"{source}: posterior inference configuration is missing {sorted(missing)}"
        )
    return configuration


def _source_season_simulation_config(
    metadata: dict[str, object]
) -> SeasonSimulationConfig:
    value = metadata.get("season_simulation_configuration")
    values = value if isinstance(value, dict) else {}
    defaults = SeasonSimulationConfig().as_dict()
    return SeasonSimulationConfig(
        outer_draw_count=int(values.get("outer_draw_count", defaults["outer_draw_count"])),
        inner_rollout_count=int(
            values.get("inner_rollout_count", defaults["inner_rollout_count"])
        ),
        seed=int(values.get("seed", defaults["seed"])),
        simulation_version=str(
            values.get("simulation_version", defaults["simulation_version"])
        ),
        likelihood_version=str(
            values.get("likelihood_version", defaults["likelihood_version"])
        ),
    )


def _source_lower_division_value(
    metadata: dict[str, object], field: str
) -> object:
    value = metadata.get(field)
    if value is not None:
        return value
    handling = metadata.get("lower_division_handling")
    if isinstance(handling, dict):
        return handling.get(field)
    return None


def _frozen_fcs_fallbacks(
    *,
    source: Path,
    source_metadata: dict[str, object],
    included_rows: list[dict[str, str]],
    teams: list[Team],
    team_rows: dict[str, dict[str, str]],
) -> tuple[list[Team], tuple[str, ...], int | None, str | None]:
    raw_ids = source_metadata.get("fcs_fallback_team_ids")
    if raw_ids is None:
        raw_ids = _source_lower_division_value(source_metadata, "fcs_fallback_team_ids")
    if not isinstance(raw_ids, list):
        raise TypeError(f"{source}: frozen FCS fallback IDs are missing")
    fallback_ids = tuple(str(value) for value in raw_ids)
    if len(fallback_ids) != len(set(fallback_ids)):
        raise ValueError(f"{source}: frozen FCS fallback IDs contain duplicates")
    population_value = source_metadata.get("fcs_population_size")
    if population_value is None:
        population_value = _source_lower_division_value(
            source_metadata, "fcs_population_size"
        )
    population = int(population_value) if population_value is not None else None
    population_source = source_metadata.get("fcs_population_source")
    if population_source is None:
        population_source = _source_lower_division_value(
            source_metadata, "fcs_population_source"
        )
    population_source = str(population_source) if population_source is not None else None
    if fallback_ids and (population is None or population < 1):
        raise ValueError(f"{source}: frozen FCS fallbacks need a positive population")

    ranking_rows: dict[str, dict[str, str]] = {}
    rankings_path = source / "rankings.csv"
    if rankings_path.is_file():
        with rankings_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("team_id"):
                    ranking_rows[str(row["team_id"])] = {
                        "team_name": str(row.get("team_name") or row["team_id"]),
                        "conference": str(row.get("conference") or ""),
                        "subdivision": str(row.get("subdivision") or ""),
                    }
    included_team_rows: dict[str, dict[str, str]] = {}
    for row in included_rows:
        for side in ("home", "away"):
            if row[f"{side}Classification"].casefold() == "fcs":
                included_team_rows[row[f"{side}Id"]] = {
                    "team_name": row[f"{side}Team"],
                    "conference": row[f"{side}Conference"],
                    "subdivision": "fcs",
                }

    known = {team.team_id for team in teams}
    if known.intersection(fallback_ids):
        overlap = sorted(known.intersection(fallback_ids))
        raise ValueError(
            f"{source}: frozen FCS fallback IDs already exist in the selected prior: {overlap}"
        )
    for team_id in fallback_ids:
        row = ranking_rows.get(team_id) or included_team_rows.get(team_id)
        if row is None or row.get("subdivision", "").casefold() != "fcs":
            raise ValueError(
                f"{source}: no FCS identity row for frozen fallback team {team_id}"
            )
        teams.append(
            Team(
                team_id,
                row["team_name"],
                "fcs",
                np.full(population, 1 / population),
            )
        )
        team_rows[team_id] = {"conference": row.get("conference", "")}
    return teams, fallback_ids, population, population_source


def _load_frozen_evidence(
    *,
    source: Path,
    season: int,
    teams: list[Team],
    team_rows: dict[str, dict[str, str]],
) -> tuple[
    dict[str, object],
    list[Game],
    list[dict[str, str]],
    int,
    CorpusProvenance,
    int | None,
    str | None,
    tuple[str, ...],
    SeasonSimulationConfig,
    dict[str, object],
]:
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{metadata_path} must contain a JSON object")
    source_metadata: dict[str, object] = value
    if int(source_metadata.get("season", -1)) != season:
        raise ValueError(f"{source}: frozen evidence season does not match {season}")
    if source_metadata.get("snapshot_type") != "weekly":
        raise ValueError(f"{source}: frozen evidence must be a weekly snapshot")
    rows = _read_included_game_rows(source / "included_games.csv")
    declared_count = source_metadata.get("included_game_count")
    if declared_count is not None and int(declared_count) != len(rows):
        raise ValueError(f"{source}: included game count does not match its rows")
    declared_ids = source_metadata.get("included_game_ids")
    ids = [row["id"] for row in rows]
    if declared_ids is not None and [str(value) for value in declared_ids] != ids:
        raise ValueError(f"{source}: included game IDs do not match its rows")
    games = [_game_from_included_row(row) for row in rows]
    fbs_ids = {
        team.team_id for team in teams if team.subdivision == "fbs"
    }
    with (source / "rankings.csv").open(newline="", encoding="utf-8") as handle:
        source_fbs_ids = {
            str(row["team_id"])
            for row in csv.DictReader(handle)
            if row.get("team_id")
            and row.get("subdivision", "").casefold() == "fbs"
        }
    if source_fbs_ids != fbs_ids:
        raise ValueError(
            f"{source}: selected prior FBS population does not match frozen source"
        )
    excluded_lower = int(source_metadata.get("excluded_lower_division_games", 0))
    provenance = _frozen_provenance(source_metadata, source)
    teams, fallback_ids, population, population_source = _frozen_fcs_fallbacks(
        source=source,
        source_metadata=source_metadata,
        included_rows=rows,
        teams=teams,
        team_rows=team_rows,
    )
    return (
        source_metadata,
        games,
        rows,
        excluded_lower,
        provenance,
        population,
        population_source,
        fallback_ids,
        _source_season_simulation_config(source_metadata),
        _source_inference_configuration(source, source_metadata),
    )


def add_fcs_fallbacks(
    teams: list[Team],
    team_rows: dict[str, dict[str, str]],
    included: list[dict[str, str]],
    fcs_population_size: int | None,
    scheduled_future: list[dict[str, str]] | None = None,
) -> tuple[list[Team], tuple[str, ...]]:
    """Add weak FCS PMFs when the frozen upstream prior has no FCS rows.

    The current frozen H/C artifacts contain only FBS forecasts. A uniform FCS
    PMF is deliberately conservative but remains a real graph variable rather
    than a fixed anonymous-strength placeholder. It is created for FCS teams
    that appeared by this snapshot's cutoff or are scheduled future opponents.
    """
    known = {team.team_id for team in teams}
    fcs_rows: dict[str, tuple[str, dict[str, str], str]] = {}
    for row in [*included, *(scheduled_future or [])]:
        for side in ("home", "away"):
            if row[f"{side}Classification"].lower() == "fcs":
                fcs_rows[row[f"{side}Id"]] = (row[f"{side}Team"], row, side)
    missing_fcs_rows = {
        team_id: value for team_id, value in fcs_rows.items() if team_id not in known
    }
    if missing_fcs_rows and fcs_population_size is None:
        raise ValueError(
            "Cannot create an FCS fallback without an authoritative full-season "
            "FCS population size. Add durable season/subdivision coverage first."
        )
    fallbacks = []
    for team_id, (name, row, side) in sorted(missing_fcs_rows.items()):
        teams.append(
            Team(
                team_id,
                name,
                "fcs",
                np.full(fcs_population_size, 1 / fcs_population_size),
            )
        )
        team_rows[team_id] = {
            "conference": row.get(f"{side}Conference", ""),
        }
        fallbacks.append(team_id)
    return teams, tuple(fallbacks)


def _scheduled_future_fcs_rows(
    root: Path,
    season: int,
    cutoff: datetime | date | None,
    snapshot_type: SnapshotType,
) -> list[dict[str, str]]:
    """Return scheduled FCS opponents visible from this snapshot onward."""
    path = root / "data/processed/cfbd/games.csv"
    if not path.is_file():
        return []
    cutoff_dt = _as_utc_datetime(cutoff) if cutoff is not None else None
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("season", -1)) != season:
                continue
            if str(row.get("seasonType", "regular") or "regular").casefold() not in {
                "",
                "regular",
            }:
                continue
            if not any(
                str(row.get(f"{side}Classification", "")).casefold() == "fcs"
                for side in ("home", "away")
            ):
                continue
            status = str(row.get("status", row.get("gameStatus", "")) or "").casefold()
            if "cancel" in status or "postpon" in status:
                continue
            if snapshot_type == "preseason":
                rows.append(row)
                continue
            try:
                when = datetime.fromisoformat(row["startDate"])
            except (KeyError, TypeError, ValueError):
                rows.append(row)
                continue
            when = when if when.tzinfo is not None else when.replace(tzinfo=UTC)
            if cutoff_dt is None or when > cutoff_dt:
                rows.append(row)
    return rows


def subdivision_population_size(
    root: Path, season: int, subdivision: str
) -> int | None:
    """Read a full ordinal universe size without depending on the cutoff.

    Historical seasons use the durable Massey team-season population.  For a
    current season not yet represented there, the cached CFBD full-season
    schedule is the authoritative enumeration: it includes scheduled teams
    that have not appeared by a particular snapshot's cutoff.
    """
    path = root / "data/processed/modeling/team_season_rank_distributions.csv"
    populations = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    int(row["season"]) == season
                    and row["subdivision"] == subdivision
                ):
                    populations.add(int(row["team_population"]))
    if populations:
        if len(populations) != 1:
            raise ValueError(
                f"Inconsistent {subdivision} population values for {season}: {populations}"
            )
        return populations.pop()

    participants: set[str] = set()
    processed_schedule = root / "data/processed/cfbd/games.csv"
    if processed_schedule.exists():
        with processed_schedule.open(newline="", encoding="utf-8") as handle:
            for game in csv.DictReader(handle):
                if int(game["season"]) != season:
                    continue
                for side in ("home", "away"):
                    if game.get(f"{side}Classification") == subdivision:
                        team_id = game.get(f"{side}Id")
                        if team_id:
                            participants.add(str(team_id))
        if participants:
            return len(participants)

    games_directory = root / "data/raw/cfbd/games"
    for suffix in ("", "-fcs"):
        schedule_path = games_directory / f"{season}{suffix}.json"
        if not schedule_path.exists():
            continue
        with schedule_path.open(encoding="utf-8") as handle:
            for game in json.load(handle):
                for side in ("home", "away"):
                    if game.get(f"{side}Classification") == subdivision:
                        team_id = game.get(f"{side}Id")
                        if team_id is not None:
                            participants.add(str(team_id))
    return len(participants) or None


def subdivision_population_source(
    root: Path, season: int, subdivision: str
) -> str | None:
    """Name the durable source used for a season/subdivision universe."""
    path = root / "data/processed/modeling/team_season_rank_distributions.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            if any(
                int(row["season"]) == season and row["subdivision"] == subdivision
                for row in csv.DictReader(handle)
            ):
                return "massey_team_season_rank_distributions"
    processed_schedule = root / "data/processed/cfbd/games.csv"
    if processed_schedule.exists():
        with processed_schedule.open(newline="", encoding="utf-8") as handle:
            if any(
                int(row["season"]) == season
                and any(
                    row.get(f"{side}Classification") == subdivision
                    for side in ("home", "away")
                )
                for row in csv.DictReader(handle)
            ):
                return "cfbd_full_season_schedule"

    games_directory = root / "data/raw/cfbd/games"
    if any(
        (games_directory / f"{season}{suffix}.json").exists()
        for suffix in ("", "-fcs")
    ):
        return "cfbd_full_season_schedule"
    return None


def load_likelihood(path: Path) -> LikelihoodV1:
    if not path.exists():
        raise FileNotFoundError(
            "Historical Likelihood V1 coefficient artifact is absent. Run "
            "`uv run python scripts/materialize_historical_likelihood_v1.py` first; "
            "the legacy report only contains scale and df, not coefficients."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    return LikelihoodV1(
        np.asarray(value["beta"], dtype=float),
        float(value["scale"]),
        float(value["degrees_of_freedom"]),
        value.get("fit_kind", "weighted_pseudo"),
    )


def _write_csv(
    path: Path, rows: list[dict[str, object]], fields: list[str] | None = None
) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_presentation_schedule(
    rows: list[dict[str, str]], *, season: int
) -> None:
    schedule_ids = [str(row.get("id", "")) for row in rows]
    if any(not game_id for game_id in schedule_ids):
        raise ValueError("presentation schedule contains a blank game ID")
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("presentation schedule contains duplicate game IDs")
    if any(str(row.get("season", "")) != str(season) for row in rows):
        raise ValueError("presentation schedule contains a different season")


def build_snapshot(
    *,
    season: int,
    cutoff: datetime | date | None,
    prior_family: PriorFamily,
    snapshot_type: SnapshotType,
    prior_model_version: str | None = None,
    lineage_suffix: str | None = None,
    root: Path | None = None,
    output_root: Path | None = None,
    likelihood: LikelihoodV1 | None = None,
    inference_max_iterations: int = 500,
    inference_tolerance: float = 1e-9,
    inference_damping: float = 0.35,
    season_simulation_config: SeasonSimulationConfig | None = None,
    generation_timestamp: datetime | None = None,
    evidence_snapshot: Path | None = None,
    presentation_schedule_rows: list[dict[str, str]] | None = None,
    presentation_schedule_source: dict[str, str] | None = None,
) -> Snapshot:
    """Build an atomic-on-success schema-v1 bundle without any publishing logic."""
    started = time.perf_counter()
    root = _root() if root is None else root
    supplied_season_simulation_config = season_simulation_config
    season_simulation_config = season_simulation_config or SeasonSimulationConfig()
    output_root = (
        root / "data/processed/snapshots" if output_root is None else output_root
    )
    selected_prior_version = (
        prior_model_version or CONTEXT_PRIOR_VERSION
        if prior_family == "context"
        else HISTORY_PRIOR_VERSION
    )
    teams, team_rows, prior_path = load_teams(
        root, season, prior_family, selected_prior_version
    )
    replay_source: Path | None = None
    replay_metadata: dict[str, object] | None = None
    replay_schedule_source: dict[str, str] | None = None
    replay_schedule_rows: list[dict[str, str]] | None = None
    replay_inference_configuration: dict[str, object] | None = None
    source_game_corpus_sha256: str | None = None

    if presentation_schedule_rows is not None and evidence_snapshot is None:
        raise ValueError(
            "presentation schedule rows are only supported for frozen evidence replay"
        )

    if evidence_snapshot is not None:
        replay_source = evidence_snapshot
        if not replay_source.is_absolute():
            replay_source = root / replay_source
        if cutoff is not None:
            requested_from_source = _metadata_datetime(
                json.loads((replay_source / "metadata.json").read_text(encoding="utf-8")).get(
                    "requested_cutoff"
                ),
                field="requested_cutoff",
                source=replay_source,
            )
            if requested_from_source != _as_utc_datetime(cutoff):
                raise ValueError(
                    f"{replay_source}: requested cutoff does not match replay cutoff"
                )
        (
            replay_metadata,
            games,
            included,
            excluded_lower,
            provenance,
            fcs_population,
            fcs_population_source,
            fcs_fallbacks,
            frozen_simulation_config,
            replay_inference_configuration,
        ) = _load_frozen_evidence(
            source=replay_source,
            season=season,
            teams=teams,
            team_rows=team_rows,
        )
        requested_cutoff = _metadata_datetime(
            replay_metadata.get("requested_cutoff"),
            field="requested_cutoff",
            source=replay_source,
        )
        effective_cutoff = _metadata_datetime(
            replay_metadata.get("effective_cutoff"),
            field="effective_cutoff",
            source=replay_source,
        )
        if requested_cutoff is None or effective_cutoff is None:
            raise ValueError(f"{replay_source}: weekly replay needs both cutoff values")
        if supplied_season_simulation_config is not None and (
            supplied_season_simulation_config.as_dict()
            != frozen_simulation_config.as_dict()
        ):
            raise ValueError(
                f"{replay_source}: supplied season simulation configuration differs"
            )
        season_simulation_config = frozen_simulation_config
        if presentation_schedule_rows is None:
            replay_schedule_rows = included
            replay_schedule_source = {
                "kind": "frozen_included_games",
                "path": relative_path(replay_source / "included_games.csv", root),
                "sha256": sha256(replay_source / "included_games.csv"),
            }
        else:
            _validate_presentation_schedule(
                presentation_schedule_rows,
                season=season,
            )
            if (
                not isinstance(presentation_schedule_source, dict)
                or presentation_schedule_source.get("kind")
                != "frozen_historical_schedule"
            ):
                raise ValueError(
                    "presentation schedule source must be frozen_historical_schedule"
                )
            replay_schedule_rows = [dict(row) for row in presentation_schedule_rows]
            replay_schedule_source = dict(presentation_schedule_source)
        source_game_corpus_value = replay_metadata.get("game_corpus_sha256")
        if not isinstance(source_game_corpus_value, str) or len(
            source_game_corpus_value
        ) != 64:
            raise ValueError(f"{replay_source}: frozen game corpus hash is invalid")
        source_game_corpus_sha256 = source_game_corpus_value
        if snapshot_type != "weekly" or prior_family != "context":
            raise ValueError("frozen evidence replay currently supports weekly Context snapshots")
        if selected_prior_version == str(replay_metadata.get("prior_model_version")):
            raise ValueError("frozen evidence replay must change the prior model version")
        if selected_prior_version != "1.3":
            raise ValueError("frozen evidence replay requires Context prior version 1.3")
    else:
        requested_cutoff = _as_utc_datetime(cutoff) if cutoff is not None else None
        provenance = (
            CorpusProvenance("preseason_prior_only", "none", None, {}, {})
            if snapshot_type == "preseason"
            else corpus_provenance(root, season)
        )
        effective_cutoff = requested_cutoff
        if provenance.source_retrieved_at is not None and requested_cutoff is not None:
            effective_cutoff = min(requested_cutoff, provenance.source_retrieved_at)
        games, included, excluded_lower, corpus_path = filter_games(
            root, season, effective_cutoff, snapshot_type
        )
        scheduled_future_fcs = _scheduled_future_fcs_rows(
            root, season, effective_cutoff, snapshot_type
        )
        has_included_fcs = any(
            row[f"{side}Classification"].lower() == "fcs"
            for row in included
            for side in ("home", "away")
        ) or bool(scheduled_future_fcs)
        fcs_population = (
            subdivision_population_size(root, season, "fcs") if has_included_fcs else None
        )
        fcs_population_source = (
            subdivision_population_source(root, season, "fcs") if has_included_fcs else None
        )
        teams, fcs_fallbacks = add_fcs_fallbacks(
            teams,
            team_rows,
            included,
            fcs_population,
            scheduled_future_fcs,
        )
    likelihood_path = root / "data/processed/posterior/historical_likelihood_v1.json"
    if replay_inference_configuration is not None:
        inference_max_iterations = int(
            replay_inference_configuration["max_iterations"]
        )
        inference_tolerance = float(replay_inference_configuration["tolerance"])
        inference_damping = float(replay_inference_configuration["damping"])
    if likelihood is None and likelihood_path.exists():
        # A preseason snapshot still needs the frozen likelihood to publish
        # future-game predictions.  Small fixture roots used for prior-only
        # tests may intentionally omit the artifact, so retain the historical
        # prior-only fallback when no game evidence or prediction is possible.
        likelihood = load_likelihood(likelihood_path)
    if games:
        likelihood = likelihood or load_likelihood(likelihood_path)
        result = infer_posterior(
            teams,
            games,
            likelihood,
            max_iterations=inference_max_iterations,
            tolerance=inference_tolerance,
            damping=inference_damping,
        )
    else:
        result = infer_posterior(teams, [], LikelihoodV1(np.zeros(34), 1.0, 1.0))
    sid = snapshot_id(
        season,
        snapshot_type,
        prior_family,
        cutoff,
        selected_prior_version,
        lineage_suffix,
    )
    directory = output_root / str(season) / sid / RANKING_FAMILY / prior_family
    directory.mkdir(parents=True, exist_ok=True)
    rankings, pmf_rows = [], []
    for team in teams:
        pmf = result.pmfs[team.team_id]
        summary = pmf_summaries(pmf)
        rankings.append(
            {
                "team_id": team.team_id,
                "team_name": team.name,
                "subdivision": team.subdivision,
                "conference": team_rows[team.team_id].get("conference", ""),
                "record_through_cutoff": "",
                "movement_from_previous": None,
                **summary,
            }
        )
        pmf_rows.extend(
            {"team_id": team.team_id, "rank": rank, "probability": float(probability)}
            for rank, probability in enumerate(pmf, 1)
        )
    rankings.sort(
        key=lambda row: (
            row["subdivision"] != "fbs",
            float(row["expected_rank"]),
            str(row["team_name"]),
        )
    )
    for rank, row in enumerate(
        (row for row in rankings if row["subdivision"] == "fbs"), 1
    ):
        row["display_rank"] = rank
    for row in rankings:
        row.setdefault("display_rank", None)
    diagnostics = {
        "converged": result.converged,
        "iterations": result.iterations,
        "max_message_delta": result.max_message_delta,
        "objective_surrogate": result.objective,
        "warnings": list(result.warnings),
        "team_variable_count": len(teams),
        "raw_game_likelihood_count": result.raw_game_factor_count,
        "unique_pairwise_factor_count": result.unique_pair_factor_count,
        "maximum_team_degree": result.max_team_degree,
        "runtime_seconds": time.perf_counter() - started,
        "inference": "deterministic_damped_loopy_sum_product"
        if games
        else "prior_passthrough",
    }
    fbs_team_ids = sorted(
        team.team_id for team in teams if team.subdivision == "fbs"
    )
    included_game_ids = [row["id"] for row in included]
    posterior_inference_configuration = {
        "implementation": diagnostics["inference"],
        "max_iterations": inference_max_iterations,
        "tolerance": inference_tolerance,
        "damping": inference_damping,
    }
    lower_division_handling = {
        "excluded_lower_division_games": excluded_lower,
        "fcs_fallback_team_ids": list(fcs_fallbacks),
        "fcs_fallback_count": len(fcs_fallbacks),
        "fcs_population_size": fcs_population,
        "fcs_population_source": fcs_population_source,
        "fcs_fallback_kind": "uniform_full_subdivision_rank"
        if fcs_fallbacks
        else None,
        "fcs_fallback_pmf_semantics": "uniform ranks 1..N_FCS"
        if fcs_fallbacks
        else None,
    }
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": sid,
        "season": season,
        "snapshot_type": snapshot_type,
        "ranking_family": RANKING_FAMILY,
        "prior_family": prior_family,
        "prior_model_version": selected_prior_version,
        "prior_lineage": (
            "context_1_3_reconstructed_2026"
            if prior_family == "context"
            and selected_prior_version == "1.3"
            and season == 2026
            else f"{prior_family}_{selected_prior_version}"
        ),
        "posterior_rebuilt_from_prior": True,
        "prior_artifact_path": relative_path(prior_path, root),
        "prior_artifact_sha256": sha256(prior_path),
        "historical_likelihood_version": HISTORICAL_LIKELIHOOD_VERSION,
        "posterior_inference_configuration": posterior_inference_configuration,
        "season_simulation_schema_version": SEASON_SIMULATION_SCHEMA_VERSION,
        "season_simulation_version": season_simulation_config.simulation_version,
        "season_simulation_configuration": season_simulation_config.as_dict(),
        "requested_cutoff": requested_cutoff.isoformat() if requested_cutoff else None,
        "effective_cutoff": effective_cutoff.isoformat() if effective_cutoff else None,
        "source_mode": provenance.source_mode,
        "source_kind": provenance.source_kind,
        "source_retrieved_at": (
            provenance.source_retrieved_at.isoformat()
            if provenance.source_retrieved_at
            else None
        ),
        "source_retrieval_times": {
            source: retrieved_at.isoformat()
            for source, retrieved_at in provenance.source_retrieval_times.items()
        },
        "source_response_hashes": provenance.source_response_hashes,
        "generation_timestamp": (
            _as_utc_datetime(generation_timestamp).isoformat()
            if generation_timestamp is not None
            else datetime.now(UTC).isoformat()
        ),
        "game_corpus_sha256": source_game_corpus_sha256
        if source_game_corpus_sha256 is not None
        else sha256(corpus_path),
        "included_game_count": len(included_game_ids),
        "included_game_ids": included_game_ids,
        "included_game_ids_sha256": stable_values_sha256(included_game_ids),
        "included_game_rows_sha256": included_game_rows_sha256(included),
        "fbs_team_count": len(fbs_team_ids),
        "fbs_team_keys_sha256": stable_values_sha256(fbs_team_ids),
        "excluded_lower_division_games": excluded_lower,
        "fcs_fallback_team_ids": list(fcs_fallbacks),
        "fcs_fallback_count": len(fcs_fallbacks),
        "fcs_fallback_kind": "uniform_full_subdivision_rank" if fcs_fallbacks else None,
        "fcs_fallback_pmf_semantics": "uniform ranks 1..N_FCS"
        if fcs_fallbacks
        else None,
        "fcs_population_size": fcs_population,
        "fcs_population_source": fcs_population_source,
        "display_statistic": "expected_rank",
        "valid": result.converged,
        "model_versions": {
            "context_prior": selected_prior_version
            if prior_family == "context"
            else CONTEXT_PRIOR_VERSION,
            "history_prior": HISTORY_PRIOR_VERSION,
            "historical_likelihood": HISTORICAL_LIKELIHOOD_VERSION,
        },
        "team_season_path": "team_seasons.json" if prior_family == "context" else None,
        "team_season_schema_version": (
            TEAM_SEASON_SCHEMA_VERSION if prior_family == "context" else None
        ),
        "lower_division_handling": lower_division_handling,
    }
    if replay_source is not None and replay_metadata is not None:
        evidence_hash = included_game_rows_sha256(included)
        metadata.update(
            {
                "backfill": True,
                "backfill_kind": "retrospective_prior_replay",
                "source_evidence_snapshot_id": replay_metadata["snapshot_id"],
                "source_evidence_snapshot_path": relative_path(replay_source, root),
                "source_evidence_game_corpus_sha256": replay_metadata.get(
                    "game_corpus_sha256"
                ),
                "source_evidence_included_game_count": len(included),
                "source_evidence_included_game_ids_sha256": stable_values_sha256(
                    included_game_ids
                ),
                "source_evidence_included_game_rows_sha256": evidence_hash,
                "source_evidence_hash": evidence_hash,
                "presentation_schedule_game_count": len(replay_schedule_rows or []),
                "presentation_schedule_source": replay_schedule_source,
            }
        )
    if likelihood is not None:
        team_season = build_team_season_artifact(
            root=root,
            metadata=metadata,
            teams=teams,
            team_rows=team_rows,
            games=games,
            included_rows=included,
            posterior=result,
            likelihood=likelihood,
            season_simulation_config=season_simulation_config,
            prediction_source=(
                "predictive_history" if prior_family == "history" else "predictive_context"
            ),
            schedule_rows=replay_schedule_rows,
            schedule_source=replay_schedule_source,
        )
        metadata["team_season_path"] = "team_seasons.json"
        metadata["team_season_schema_version"] = team_season["schema_version"]
        (directory / "team_seasons.json").write_text(
            json.dumps(team_season, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "rankings.json").write_text(
        json.dumps(rankings, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(directory / "rankings.csv", rankings)
    _write_csv(directory / "posterior_pmfs.csv", pmf_rows)
    _write_csv(
        directory / "included_games.csv",
        included,
        [
            "id",
            "season",
            "week",
            "seasonType",
            "startDate",
            "completed",
            "neutralSite",
            "conferenceGame",
            "homeId",
            "homeTeam",
            "homeClassification",
            "homeConference",
            "homePoints",
            "awayId",
            "awayTeam",
            "awayClassification",
            "awayConference",
            "awayPoints",
        ],
    )
    return Snapshot(sid, directory, metadata)
