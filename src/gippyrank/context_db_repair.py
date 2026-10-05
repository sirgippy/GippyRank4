"""Coverage-aware incoming DB evidence for corrected production Context 1.4."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from gippyrank import preseason
from gippyrank.context_prior_v1_3 import (
    CONTEXT13_DEGREES_OF_FREEDOM,
    CONTEXT13_DISTRIBUTION_FAMILY,
    CONTEXT13_LAG_COUNT,
    CONTEXT13_MINIMUM_SCALE,
    CONTEXT13_OPTIMIZER_ITERATION_LIMIT_MESSAGE,
    CONTEXT13_OPTIMIZER_RETRY_MAXITER,
    CONTEXT13_TEAM_SEASON_WEIGHT,
    D5_CONTEXT_FEATURES,
    FROZEN_PENALTY,
    H_FEATURES,
    SCALE_FEATURE_NAMES,
)
from gippyrank.preseason import (
    DirectRankModel,
    TeamSeason,
    conditional_rank_mixture_pmf,
)

REPAIR = Path("data/processed/transfer_data_repair")
HISTORICAL_FEATURES = REPAIR / "historical_transfer_features.csv"
HISTORICAL_COVERAGE = REPAIR / "team_seasons.csv"
HISTORICAL_PLAYERS = Path(
    "data/processed/defensive_transfer_audit/transfer_player_audit.csv"
)
CURRENT_COVERAGE = REPAIR / "db_coverage_2026.csv"
CURRENT_OFFENSE = Path(
    "data/processed/preseason/context_v1_3_2026_reconstruction/transfer_features.csv"
)
DB_SUM = "observed_db_impact_sum"
DB_COVERAGE = "db_impact_coverage_fraction"
DB_MISSING = "missing_db_impact_count"
CONTEXT_FEATURES = (*D5_CONTEXT_FEATURES, DB_SUM, DB_COVERAGE)
MODEL_FEATURES = (*H_FEATURES, *CONTEXT_FEATURES)
ALPHA = 0.75
GOOD_IMPACT_STATUSES = frozenset(
    {"resolved", "zero_recorded_defensive_box_score_games"}
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DBEvidence:
    incoming_db_count: int
    observed_db_impact_count: int
    observed_db_impact_sum: float

    def __post_init__(self) -> None:
        if (
            self.incoming_db_count < 0
            or not 0 <= self.observed_db_impact_count <= self.incoming_db_count
            or not math.isfinite(self.observed_db_impact_sum)
        ):
            raise ValueError("invalid DB transfer coverage")
        if self.observed_db_impact_count == 0 and self.observed_db_impact_sum != 0:
            raise ValueError("no observed DB players cannot have an observed sum")

    @property
    def missing_db_impact_count(self) -> int:
        return self.incoming_db_count - self.observed_db_impact_count

    @property
    def db_impact_coverage_fraction(self) -> float:
        if not self.incoming_db_count:
            return 1.0
        return self.observed_db_impact_count / self.incoming_db_count

    @property
    def status(self) -> str:
        if not self.incoming_db_count:
            return "natural_zero"
        if self.observed_db_impact_count == self.incoming_db_count:
            return "complete"
        if self.observed_db_impact_count:
            return "partial"
        return "no_observed"

    def model_features(self) -> dict[str, float]:
        return {
            DB_SUM: self.observed_db_impact_sum,
            DB_COVERAGE: self.db_impact_coverage_fraction,
        }

    def audit_fields(self) -> dict[str, int | float | str]:
        return {
            "incoming_db_count": self.incoming_db_count,
            "observed_db_impact_count": self.observed_db_impact_count,
            DB_MISSING: self.missing_db_impact_count,
            DB_SUM: self.observed_db_impact_sum,
            DB_COVERAGE: self.db_impact_coverage_fraction,
            "db_impact_coverage_status": self.status,
        }


def _unique(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, str]]:
    result: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["season"]), row["team_id"])
        if key in result:
            raise ValueError(f"duplicate repaired transfer team: {key}")
        result[key] = row
    return result


def historical_repaired_features(
    root: Path,
) -> tuple[
    dict[tuple[int, str], dict[str, float | None]], dict[tuple[int, str], DBEvidence]
]:
    """Join #150's corrected transfer panel and availability to player DB sums."""
    features = _unique(read_csv(root / HISTORICAL_FEATURES))
    coverage = _unique(
        [
            row
            for row in read_csv(root / HISTORICAL_COVERAGE)
            if int(row["season"]) <= 2025
        ]
    )
    players: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(root / HISTORICAL_PLAYERS):
        if (
            row["in_model_relevant_population"] == "True"
            and row["portal_position_group"] == "db"
        ):
            players[(int(row["season"]), row["destination_team_id"])].append(row)
    if set(coverage) - set(features):
        raise ValueError(
            "historical DB coverage has teams absent from corrected transfer panel"
        )
    result: dict[tuple[int, str], dict[str, float | None]] = {}
    db_evidence: dict[tuple[int, str], DBEvidence] = {}
    for key, row in features.items():
        if row["subdivision"] != "fbs":
            continue
        usage = row["transfer_in_prior_usage_sum"]
        values: dict[str, float | None] = {
            "transfer_in_prior_usage_sum": None if usage == "" else float(usage),
            DB_SUM: None,
            DB_COVERAGE: None,
        }
        audit = coverage.get(key)
        if audit is not None:
            player_rows = players.get(key, [])
            observed = [
                float(player["prior_defensive_impact"])
                for player in player_rows
                if player["impact_status"] in GOOD_IMPACT_STATUSES
            ]
            incoming = int(audit["db_incoming"])
            observed_count = int(audit["db_resolved"])
            if len(player_rows) != incoming or len(observed) != observed_count:
                raise ValueError(f"historical DB audit/player counts disagree: {key}")
            evidence = DBEvidence(incoming, observed_count, math.fsum(observed))
            if evidence.status in {"complete", "natural_zero"}:
                old = float(row["transfer_in_prior_defensive_impact_db_sum"])
                if not math.isclose(old, evidence.observed_db_impact_sum, abs_tol=1e-9):
                    raise ValueError(f"complete historical DB sum disagrees: {key}")
            values.update(evidence.model_features())
            db_evidence[key] = evidence
        result[key] = values
    if len(db_evidence) != len(coverage):
        raise ValueError("historical repaired DB coverage population is incomplete")
    return result, db_evidence


def current_repaired_features(
    root: Path,
) -> tuple[dict[str, dict[str, float | None]], dict[str, DBEvidence]]:
    """Use the #151 reacquired DB artifact, retaining the separate offense field."""
    offense_rows = _unique(read_csv(root / CURRENT_OFFENSE))
    coverage_rows = _unique(read_csv(root / CURRENT_COVERAGE))
    if set(offense_rows) != set(coverage_rows) or {
        season for season, _ in coverage_rows
    } != {2026}:
        raise ValueError("2026 repaired DB and offense populations differ")
    values: dict[str, dict[str, float | None]] = {}
    evidence: dict[str, DBEvidence] = {}
    for (season, team_id), row in coverage_rows.items():
        db = DBEvidence(
            int(row["total_count"]),
            int(row["observed_count"]),
            float(row["observed_db_impact_sum"]),
        )
        source_fraction = row["coverage_fraction"]
        source_status = row["coverage_status"]
        if (
            int(row["missing_count"]) != db.missing_db_impact_count
            or (
                source_fraction != ""
                and not math.isclose(
                    float(source_fraction),
                    db.db_impact_coverage_fraction,
                    abs_tol=1e-12,
                )
            )
            or (source_fraction == "" and db.status != "natural_zero")
            or source_status
            != (
                "no_incoming_db_transfers" if db.status == "natural_zero" else db.status
            )
        ):
            raise ValueError(f"2026 repaired DB audit disagrees: {team_id}")
        usage = offense_rows[(season, team_id)]["transfer_in_prior_usage_sum"]
        values[team_id] = {
            "transfer_in_prior_usage_sum": None if usage == "" else float(usage),
            **db.model_features(),
        }
        evidence[team_id] = db
    if len(values) != 138:
        raise ValueError("2026 repaired DB coverage must include 138 FBS teams")
    return values, evidence


def attach_historical(
    rows: list[TeamSeason],
    features: dict[tuple[int, str], dict[str, float | None]],
) -> list[TeamSeason]:
    attached = []
    for row in rows:
        key = (row.season, row.team_id)
        if key not in features:
            raise ValueError(
                f"historical Context row lacks repaired transfer source: {key}"
            )
        values = {name: row.features.get(name) for name in MODEL_FEATURES}
        values.update(features[key])
        attached.append(replace(row, features=values))
    return attached


def fit_model(rows: list[TeamSeason], trained_through: int) -> DirectRankModel:
    training = [row for row in rows if row.season <= trained_through]
    if not training or any(row.season > trained_through for row in training):
        raise ValueError("invalid corrected Context training population")
    options = {
        "maxiter": preseason.DIRECT_RANK_OPTIMIZER_MAXITER,
        "ftol": preseason.DIRECT_RANK_OPTIMIZER_FTOL,
        "gtol": preseason.DIRECT_RANK_OPTIMIZER_GTOL,
    }

    def fit() -> DirectRankModel:
        return DirectRankModel.fit(
            training,
            list(MODEL_FEATURES),
            penalty=FROZEN_PENALTY,
            minimum_scale=CONTEXT13_MINIMUM_SCALE,
            lag_count=CONTEXT13_LAG_COUNT,
            family=CONTEXT13_DISTRIBUTION_FAMILY,
            degrees_of_freedom=CONTEXT13_DEGREES_OF_FREEDOM,
            location_feature_names=list(MODEL_FEATURES),
            scale_feature_names=list(SCALE_FEATURE_NAMES),
            row_weights=np.full(len(training), CONTEXT13_TEAM_SEASON_WEIGHT),
            preprocessor_scale_floor=preseason.PREPROCESSOR_SCALE_FLOOR,
            preprocessor_std_ddof=preseason.PREPROCESSOR_STD_DDOF,
            optimizer_options=options,
        )

    try:
        return fit()
    except RuntimeError as error:
        if CONTEXT13_OPTIMIZER_ITERATION_LIMIT_MESSAGE not in str(error):
            raise
        options["maxiter"] = CONTEXT13_OPTIMIZER_RETRY_MAXITER
        return fit()


def production_pmf(
    model: DirectRankModel,
    features: dict[str, float | None],
    lag1_z: np.ndarray,
    population: int,
) -> tuple[np.ndarray, float, float]:
    """Moderate only a positive fitted Context-only location subtotal."""
    if (
        tuple(model.feature_names) != MODEL_FEATURES
        or tuple(model.scale_feature_names or ()) != SCALE_FEATURE_NAMES
    ):
        raise ValueError("production Context 1.4 fit has a different feature contract")
    if set(features) != set(MODEL_FEATURES):
        raise ValueError("production Context 1.4 row has a different feature contract")
    locations, scale = model.conditional_parameters(features, lag1_z)
    design = model.design_vector(features)
    coefficients = model.beta[model.lag_count :]
    n = len(MODEL_FEATURES)
    context_only = math.fsum(
        float(
            design[1 + index] * coefficients[1 + index]
            + design[1 + n + index] * coefficients[1 + n + index]
        )
        for index, name in enumerate(MODEL_FEATURES)
        if name not in H_FEATURES
    )
    moderated = locations - (1.0 - ALPHA) * max(context_only, 0.0)
    pmf = conditional_rank_mixture_pmf(
        moderated,
        scale,
        population,
        family=model.family,
        degrees_of_freedom=model.degrees_of_freedom,
    )
    return pmf, float(np.mean(moderated)), scale
