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

from gippyrank.posterior.engine import Game, LikelihoodV1, Team, infer_posterior
from gippyrank.posterior.game_evidence import build_team_season_artifact
from gippyrank.preseason import pmf_summaries

SCHEMA_VERSION = "1.0"
RANKING_FAMILY = "predictive"
PriorFamily = Literal["context", "history"]
SnapshotType = Literal["preseason", "weekly", "live"]


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


def snapshot_id(
    season: int,
    snapshot_type: SnapshotType,
    prior_family: PriorFamily,
    cutoff: datetime | date | None = None,
) -> str:
    if snapshot_type == "preseason":
        return f"{season}-preseason-{prior_family}"
    if cutoff is None:
        raise ValueError("weekly and live snapshots require an explicit cutoff")
    if snapshot_type == "weekly":
        stamp = cutoff.isoformat().replace("+00:00", "Z").replace(":", "-")
        return f"{season}-weekly-{stamp}-{prior_family}"
    stamp = cutoff.isoformat().replace("+00:00", "Z").replace(":", "-")
    return f"{season}-live-{stamp}-{prior_family}"


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prior_path(root: Path, family: PriorFamily, season: int) -> Path:
    annual = root / f"data/processed/preseason/{family}/annual/{season}/predictions.csv"
    # Historical prior rows are retained in the frozen multi-season artifact;
    # 2026 additionally has its explicit annually frozen instance.
    return (
        annual
        if annual.exists()
        else root / f"data/processed/preseason/{family}/predictions.csv"
    )


def load_teams(
    root: Path, season: int, family: PriorFamily
) -> tuple[list[Team], dict[str, dict[str, str]], Path]:
    path = _prior_path(root, family, season)
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
            when = datetime.fromisoformat(row["startDate"])
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


def add_fcs_fallbacks(
    teams: list[Team],
    team_rows: dict[str, dict[str, str]],
    included: list[dict[str, str]],
    fcs_population_size: int | None,
) -> tuple[list[Team], tuple[str, ...]]:
    """Add weak FCS PMFs when the frozen upstream prior has no FCS rows.

    The current frozen H/C artifacts contain only FBS forecasts. A uniform FCS
    PMF is deliberately conservative but remains a real graph variable rather
    than a fixed anonymous-strength placeholder. It is created only for FCS
    teams that have appeared by this snapshot's cutoff.
    """
    known = {team.team_id for team in teams}
    fcs_rows: dict[str, tuple[str, dict[str, str]]] = {}
    for row in included:
        for side in ("home", "away"):
            if row[f"{side}Classification"].lower() == "fcs":
                fcs_rows[row[f"{side}Id"]] = (row[f"{side}Team"], row)
    if fcs_rows and fcs_population_size is None:
        raise ValueError(
            "Cannot create an FCS fallback without an authoritative full-season "
            "FCS population size. Add durable season/subdivision coverage first."
        )
    fallbacks = []
    for team_id, (name, row) in sorted(fcs_rows.items()):
        if team_id in known:
            continue
        teams.append(
            Team(
                team_id,
                name,
                "fcs",
                np.full(fcs_population_size, 1 / fcs_population_size),
            )
        )
        team_rows[team_id] = {"conference": row.get("homeConference", "")}
        fallbacks.append(team_id)
    return teams, tuple(fallbacks)


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


def build_snapshot(
    *,
    season: int,
    cutoff: datetime | date | None,
    prior_family: PriorFamily,
    snapshot_type: SnapshotType,
    root: Path | None = None,
    output_root: Path | None = None,
    likelihood: LikelihoodV1 | None = None,
    inference_max_iterations: int = 500,
    inference_tolerance: float = 1e-9,
    inference_damping: float = 0.35,
    generation_timestamp: datetime | None = None,
) -> Snapshot:
    """Build an atomic-on-success schema-v1 bundle without any publishing logic."""
    started = time.perf_counter()
    root = _root() if root is None else root
    output_root = (
        root / "data/processed/snapshots" if output_root is None else output_root
    )
    teams, team_rows, prior_path = load_teams(root, season, prior_family)
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
    has_included_fcs = any(
        row[f"{side}Classification"].lower() == "fcs"
        for row in included
        for side in ("home", "away")
    )
    fcs_population = (
        subdivision_population_size(root, season, "fcs") if has_included_fcs else None
    )
    fcs_population_source = (
        subdivision_population_source(root, season, "fcs") if has_included_fcs else None
    )
    teams, fcs_fallbacks = add_fcs_fallbacks(teams, team_rows, included, fcs_population)
    likelihood_path = root / "data/processed/posterior/historical_likelihood_v1.json"
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
    sid = snapshot_id(season, snapshot_type, prior_family, cutoff)
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
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": sid,
        "season": season,
        "snapshot_type": snapshot_type,
        "ranking_family": RANKING_FAMILY,
        "prior_family": prior_family,
        "prior_model_version": "1.2" if prior_family == "context" else "1.1",
        "prior_artifact_sha256": sha256(prior_path),
        "historical_likelihood_version": "V1",
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
        "game_corpus_sha256": sha256(corpus_path),
        "included_game_count": len(included),
        "included_game_ids": [row["id"] for row in included],
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
            "context_prior": "1.2",
            "history_prior": "1.1",
            "historical_likelihood": "V1",
        },
        "team_season_path": "team_seasons.json" if prior_family == "context" else None,
        "team_season_schema_version": "1.0" if prior_family == "context" else None,
    }
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
            prediction_source=(
                "predictive_history" if prior_family == "history" else "predictive_context"
            ),
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
