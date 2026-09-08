"""Build the research-only primitive box-score likelihood investigation.

This script consumes the frozen historical modeling rows, the merged audited
primitive artifact from PR #20, frozen priors, and the frozen Historical
Likelihood V1.  It writes only to
``data/processed/primitive_box_score_likelihood`` and never changes production
ranking or publication artifacts.

The candidate family is fixed in source before the evaluation is run:

* V1: frozen margin-only Historical Likelihood V1;
* A: V1 times a conditional yards factor with plays as context;
* B: V1 times conditional plays and yards factors;
* C: B's factors conditioned on separate interceptions-thrown and
  fumbles-lost context, with an exact per-game fallback to B when that
  context is missing; and
* ``ypp_supported``: the fixed supported-pairing YPP comparator from PR #17.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t

from gippyrank.modeling import read_csv
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_margin_parameters,
)
from gippyrank.posterior.snapshots import load_likelihood, load_teams
from gippyrank.preseason import pmf_summaries
from gippyrank.research.primitive_box_score_likelihood import (
    DF_GRID,
    PRIMITIVE_FIELDS,
    SUPPORTED_PRIMITIVE_PAIRINGS,
    PrimitiveData,
    _model_locations,
    _student_t_logpdf,
    build_primitive_data,
    component_mask,
    fit_primitive_model,
    infer_posterior_with_primitives,
    oriented_difference,
    oriented_margin,
    oriented_rank_coordinates,
    pairing_for,
    primitive_model_scores,
)
from gippyrank.research.ypp_likelihood import (
    SUPPORTED_YPP_PAIRINGS,
    build_ypp_data,
    fit_ypp_model,
    infer_posterior_with_ypp,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/primitive_box_score_likelihood"
HISTORICAL_ROWS_PATH = ROOT / "data/processed/modeling/historical_modeling_games.csv"
RANK_DISTRIBUTIONS_PATH = ROOT / "data/processed/modeling/team_season_rank_distributions.csv"
AUDIT_ROWS_PATH = ROOT / "data/processed/box_score_audit/team_game_audit.csv"
AUDIT_SUMMARY_PATH = ROOT / "data/processed/box_score_audit/summary.json"
GAMES_PATH = ROOT / "data/processed/cfbd/games.csv"
LIKELIHOOD_PATH = ROOT / "data/processed/posterior/historical_likelihood_v1.json"

TRAIN_YEARS = tuple(range(2004, 2018))
DEVELOPMENT_YEARS = tuple(range(2018, 2022))
FINAL_YEARS = (2022, 2023, 2024, 2025)
ROLLING_YEARS = tuple(range(2008, 2026))
FINAL_CANDIDATES = ("v1", "a", "b", "c", "ypp_supported")
PRIMITIVE_CANDIDATES = ("a", "b", "c")
SELECTION_CANDIDATES = ("v1", "a", "b", "c")
YPP_FIXED_DF = 15.0
CUTOFF_FRACTIONS = (0.0, 0.20, 0.35, 0.55, 0.72, 0.87, 1.0)
RECONSTRUCTED_PRIOR_FAMILIES = ("context_reconstructed", "history_reconstructed")

# These thresholds are declared before the final 2022--2025 evaluation.  The
# simpler candidate wins unless the added component clears every development
# gate in the sequential A-vs-V1, B-vs-A, C-vs-B comparison.
PROMOTION_CRITERIA: dict[str, object] = {
    "aggregate_nll_improvement_nats_per_team": 0.02,
    "maximum_aggregate_crps_degradation": 0.002,
    "maximum_aggregate_80pct_coverage_drop": 0.03,
    "minimum_evaluation_seasons_with_nll_improvement": 3,
    "maximum_single_season_nll_degradation": 0.10,
    "maximum_single_season_crps_degradation": 0.02,
    "maximum_future_margin_mae_degradation_points": 0.50,
    "maximum_future_margin_nll_degradation_nats": 0.02,
    "minimum_rolling_seasons_with_nll_improvement": 9,
}

DEVELOPMENT_SELECTION_RULE = {
    "comparison": "sequential A vs V1, B vs A, C vs B",
    "aggregate_nll_gain_nats_per_team": 0.02,
    "maximum_crps_degradation": 0.002,
    "maximum_80pct_coverage_drop": 0.03,
    "minimum_of_four_development_seasons_with_nll_improvement": 3,
    "maximum_single_season_nll_degradation": 0.10,
    "tie_break": "simpler candidate",
    "forced_reporting_candidate": "A when no A/B/C clears its predecessor; this does not imply promotion",
}


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> dict[str, str]:
    if path.is_file():
        return {str(path.relative_to(ROOT)): _sha256(path)}
    if not path.exists():
        return {}
    return {
        str(item.relative_to(ROOT)): _sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_date(value: object) -> datetime:
    text = str(value).replace("Z", "+00:00")
    result = datetime.fromisoformat(text)
    return result if result.tzinfo is not None else result.replace(tzinfo=UTC)


def _period(season: int) -> str:
    if season in TRAIN_YEARS:
        return "training_2004_2017"
    if season in DEVELOPMENT_YEARS:
        return "development_2018_2021"
    if season in FINAL_YEARS:
        return "final_2022_2025"
    return "outside_evaluation"


def _margin_bin(margin: float) -> str:
    if margin <= -28:
        return "<=-28"
    if margin <= -14:
        return "(-28,-14]"
    if margin <= 0:
        return "(-14,0]"
    if margin <= 14:
        return "(0,14]"
    if margin <= 28:
        return "(14,28]"
    return ">28"


def load_historical_rows() -> list[dict[str, object]]:
    rows = [dict(row) for row in read_csv(HISTORICAL_ROWS_PATH)]
    games = {
        str(row["id"]): row
        for row in read_csv(GAMES_PATH)
        if str(row.get("homeClassification", "")).casefold() in {"fbs", "fcs"}
        and str(row.get("awayClassification", "")).casefold() in {"fbs", "fcs"}
    }
    for row in rows:
        game = games.get(str(row["game_id"]), {})
        row["season"] = int(row["season"])
        row["game_id"] = str(row["game_id"])
        row["home_team_id"] = str(row["home_team_id"])
        row["away_team_id"] = str(row["away_team_id"])
        row["home_points"] = int(row["home_points"])
        row["away_points"] = int(row["away_points"])
        row["neutral_site"] = _bool(row["neutral_site"])
        row["start_date"] = str(row["start_date"])
        row["season_type"] = str(game.get("seasonType") or "regular")
        row["home_team"] = str(game.get("homeTeam") or row["home_team_id"])
        row["away_team"] = str(game.get("awayTeam") or row["away_team_id"])
        row["pairing"] = pairing_for(
            str(row["home_subdivision"]), str(row["away_subdivision"])
        )
    return rows


def load_rank_targets() -> dict[tuple[int, str, str], dict[str, object]]:
    targets: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in read_csv(RANK_DISTRIBUTIONS_PATH):
        population = int(row["team_population"])
        items = json.loads(row["pmf"])
        support = max([population, *(int(item["rank"]) for item in items)])
        pmf = np.zeros(support, dtype=float)
        for item in items:
            pmf[int(item["rank"]) - 1] = float(item["probability"])
        targets[(int(row["season"]), row["subdivision"], row["team_id"])] = {
            "population": population,
            "mean_rank": float(row["rank_mean"]),
            "pmf": pmf,
            "team_name": row["team_name"],
        }
    return targets


def load_audit_evidence() -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, object]]:
    """Load only the merged PR #20 audit artifact as primitive source data."""

    evidence: dict[tuple[str, str], dict[str, object]] = {}
    status = Counter()
    for row in read_csv(AUDIT_ROWS_PATH):
        key = (str(row["game_id"]), str(row["team_id"]))
        impossible = {
            value for value in str(row.get("impossible_fields", "")).split(";") if value
        }
        values: dict[str, object] = {
            "team_id": str(row["team_id"]),
            "team": str(row.get("team") or row["team_id"]),
            "home_away": str(row.get("home_away") or ""),
            "audit_pairing": str(row.get("pairing") or "unknown").casefold(),
            "game_found": _bool(row.get("game_found")),
            "completed": _bool(row.get("completed")),
            "impossible_fields": sorted(impossible),
        }
        for field in PRIMITIVE_FIELDS:
            raw = row.get(field)
            if field in impossible:
                values[field] = None
                values[f"{field}_status"] = "excluded_impossible"
                status[f"{field}:excluded_impossible"] += 1
            else:
                parsed = _int(raw)
                values[field] = parsed
                values[f"{field}_status"] = "observed" if parsed is not None else "missing"
                status[f"{field}:{values[f'{field}_status']}"] += 1
        evidence[key] = values
    audit_summary = json.loads(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    return evidence, {
        "source": "data/processed/box_score_audit/team_game_audit.csv",
        "source_sha256": _sha256(AUDIT_ROWS_PATH),
        "summary_sha256": _sha256(AUDIT_SUMMARY_PATH),
        "rows": len(evidence),
        "field_status_counts": dict(sorted(status.items())),
        "support_statement": audit_summary.get("scope", {}),
    }


def attach_primitive_evidence(
    rows: Sequence[Mapping[str, object]],
    audit: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Attach audited values without repairing or zero-filling missing data."""

    enriched: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        game_id = str(row["game_id"])
        for side in ("home", "away"):
            team_id = str(row[f"{side}_team_id"])
            audited = audit.get((game_id, team_id))
            for field in PRIMITIVE_FIELDS:
                row[f"{side}_{field}"] = None if audited is None else audited.get(field)
                row[f"{side}_{field}_status"] = (
                    "missing_audit_row" if audited is None else audited.get(f"{field}_status")
                )
            row[f"{side}_audit_team"] = (
                team_id if audited is None else audited.get("team", team_id)
            )
            if audited is not None:
                for field in PRIMITIVE_FIELDS:
                    if audited.get(f"{field}_status") == "excluded_impossible":
                        excluded.append(
                            {
                                "game_id": game_id,
                                "season": row["season"],
                                "team_id": team_id,
                                "team": audited.get("team", team_id),
                                "field": field,
                                "reason": "prospective exclusion of audited impossible field; game remains in V1 graph",
                                "status": audited.get(f"{field}_status"),
                            }
                        )
        enriched.append(row)
    return enriched, excluded


def build_evidence_by_game(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    result: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for row in rows:
        home = {field: row.get(f"home_{field}") for field in PRIMITIVE_FIELDS}
        away = {field: row.get(f"away_{field}") for field in PRIMITIVE_FIELDS}
        result[str(row["game_id"])] = (home, away)
    return result


def row_to_game(row: Mapping[str, object]) -> Game:
    return Game(
        str(row["game_id"]),
        str(row["home_team_id"]),
        str(row["away_team_id"]),
        str(row["home_subdivision"]),  # type: ignore[arg-type]
        str(row["away_subdivision"]),  # type: ignore[arg-type]
        int(row["home_points"]),
        int(row["away_points"]),
        bool(row["neutral_site"]),
    )


def _team_names(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        result[str(row["home_team_id"])] = str(row.get("home_team") or row["home_team_id"])
        result[str(row["away_team_id"])] = str(row.get("away_team") or row["away_team_id"])
    return result


def _fcs_population(
    targets: Mapping[tuple[int, str, str], Mapping[str, object]], season: int
) -> int:
    values = {
        int(value["population"])
        for (target_season, subdivision, _team_id), value in targets.items()
        if target_season == season and subdivision == "fcs"
    }
    if len(values) != 1:
        raise ValueError(f"expected one FCS population for {season}, found {values}")
    return values.pop()


def make_uniform_teams(
    season: int,
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> list[Team]:
    """Build a leakage-safe development/rolling selection prior.

    Frozen H/C prior predictions are available only for 2022--2025 in this
    checkout.  Development selection therefore uses a transparent uniform
    ordinal support, while target PMFs remain the held-out labels.
    """

    names = _team_names(rows)
    teams: list[Team] = []
    ids = {
        (str(row[f"{side}_team_id"]), str(row[f"{side}_subdivision"]))
        for row in rows
        for side in ("home", "away")
    }
    for team_id, subdivision in sorted(ids):
        target = targets.get((season, subdivision, team_id))
        if target is None:
            continue
        support = len(np.asarray(target["pmf"], dtype=float))
        teams.append(
            Team(team_id, names.get(team_id, team_id), subdivision, np.full(support, 1 / support))
        )
    return teams


def make_frozen_teams(
    season: int,
    family: str,
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> list[Team]:
    """Load the exact frozen prior and add explicit uniform FCS graph nodes."""

    teams, _metadata, _path = load_teams(ROOT, season, family)  # type: ignore[arg-type]
    known = {team.team_id for team in teams}
    names = _team_names(rows)
    population = _fcs_population(targets, season)
    fcs_ids = {
        str(row[f"{side}_team_id"])
        for row in rows
        for side in ("home", "away")
        if str(row[f"{side}_subdivision"]) == "fcs"
    }
    for team_id in sorted(fcs_ids - known):
        teams.append(
            Team(team_id, names.get(team_id, team_id), "fcs", np.full(population, 1 / population))
        )
    return teams


def make_reconstructed_teams(
    season: int,
    family: str,
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    reconstructed_pmfs: Mapping[str, np.ndarray],
) -> list[Team]:
    """Build graph teams from an outcome-free historical preseason PMF set."""

    if family not in {"context_reconstructed", "history_reconstructed"}:
        raise ValueError(f"unknown reconstructed prior family: {family}")
    names = _team_names(rows)
    teams: list[Team] = []
    fbs_ids = {
        str(row[f"{side}_team_id"])
        for row in rows
        for side in ("home", "away")
        if str(row[f"{side}_subdivision"]) == "fbs"
    }
    missing = sorted(team_id for team_id in fbs_ids if team_id not in reconstructed_pmfs)
    if missing:
        raise ValueError(f"reconstructed {family} prior missing FBS teams for {season}: {missing[:5]}")
    for team_id in sorted(fbs_ids):
        pmf = np.asarray(reconstructed_pmfs[team_id], dtype=float)
        if len(pmf) != len(np.asarray(targets[(season, "fbs", team_id)]["pmf"])):
            raise ValueError(f"reconstructed {family} support mismatch for {season}/{team_id}")
        teams.append(Team(team_id, names.get(team_id, team_id), "fbs", pmf))
    fcs_ids = {
        str(row[f"{side}_team_id"])
        for row in rows
        for side in ("home", "away")
        if str(row[f"{side}_subdivision"]) == "fcs"
    }
    if fcs_ids:
        population = _fcs_population(targets, season)
        teams.extend(
            Team(team_id, names.get(team_id, team_id), "fcs", np.full(population, 1 / population))
            for team_id in sorted(fcs_ids)
        )
    return teams


def reconstructed_prior_audit_row(
    target_season: int, family: str, pmf_count: int, context_features: Sequence[str],
    *, supported: bool = True, reason: str | None = None,
    cold_start_reasons: Mapping[str, int] | None = None,
    eligible_promotion_rows: int | None = None,
    eligible_generic_rows: int | None = None,
) -> dict[str, object]:
    """Record the temporal boundary used by one reconstructed prior family."""

    return {
        "target_season": target_season,
        "prior_family": family,
        "trained_through_season": target_season - 1,
        "target_outcomes_used": False,
        "supported": supported,
        "reason": reason,
        "pmf_count": pmf_count,
        "context_features": list(context_features),
        "cold_start_reasons": dict(cold_start_reasons or {}),
        "eligible_promotion_rows": eligible_promotion_rows,
        "eligible_generic_rows": eligible_generic_rows,
    }


def cold_start_requirements(
    inference: Sequence[object], cold: Sequence[object], trained_through: int
) -> tuple[dict[str, int], list[object], list[object]]:
    """Return target requirements and independently eligible H fallback rows."""

    reasons = Counter(
        str(row.cold_start_reason)
        for row in inference
        if row.cold_start_reason is not None
    )
    promotion_rows = [
        row for row in cold_start_teams(cold) if row.season <= trained_through
    ]
    generic_rows = [
        row
        for row in cold
        if row.subdivision == "fbs"
        and row.reason == "no_prior_rank_distribution"
        and row.season <= trained_through
    ]
    return dict(reasons), promotion_rows, generic_rows


def cold_start_teams(cold: Sequence[object]) -> list[object]:
    """Use H 1.1's promotion population without fitting either fallback."""

    return [row for row in cold if getattr(row, "reason", None) == "fcs_to_fbs_transition"]


def reconstruct_historical_priors(
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    seasons: Sequence[int],
    families: Sequence[str] = RECONSTRUCTED_PRIOR_FAMILIES,
) -> tuple[dict[tuple[int, str], list[Team]], list[dict[str, object]]]:
    """Rebuild H 1.1/C 1.2 PMFs using only information available pre-target.

    The target-season rank distributions are used only by the caller as held-out
    labels.  Every model fit and every target-season feature row here is built
    from completed outcomes through ``target_season - 1`` and preseason-
    semantic feature fields.
    """

    scripts_path = str(ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    import build_preseason_context_prior_v1_2 as context_prior
    import build_preseason_prior as history_prior

    context_features = [
        *context_prior.COACH_FEATURES,
        *context_prior.RECRUITING_FEATURES,
        *context_prior.TALENT_FEATURES,
        *context_prior.RETURNING_FEATURES,
    ]
    result: dict[tuple[int, str], list[Team]] = {}
    audit: list[dict[str, object]] = []
    for target_season in seasons:
        trained_through = target_season - 1
        historical, cold, _coverage = history_prior.load_rows(max_season=trained_through)
        fbs = [row for row in historical if row.subdivision == "fbs"]
        index, tenures = context_prior.feature_index(), context_prior.cached_tenures()
        contextual, _context_coverage = context_prior.attach_context(fbs, index, tenures)
        h_model, _h_fit = context_prior.build_history_prior(
            fbs,
            target_season=target_season,
            trained_through_season=trained_through,
        )
        c_model = None
        if "context_reconstructed" in families:
            c_model, _c_fit = context_prior.build_context_prior(
                contextual,
                target_season=target_season,
                trained_through_season=trained_through,
                context_features=context_features,
                mode="both",
            )
        inference = context_prior.inference_rows(
            target_season, trained_through, index, tenures
        )
        inference = [
            replace(
                row,
                population=int(
                    targets[(target_season, "fbs", row.team_id)]["population"]
                ),
            )
            for row in inference
            if (target_season, "fbs", row.team_id) in targets
        ]
        reasons, eligible_promotion, eligible_generic = cold_start_requirements(
            inference, cold, trained_through
        )
        needs_promotion = reasons.get("fcs_to_fbs_transition", 0) > 0
        needs_generic = reasons.get("no_prior_rank_distribution", 0) > 0
        if needs_promotion and not eligible_promotion:
            for family in families:
                audit.append(reconstructed_prior_audit_row(target_season, family, 0, context_features if family == "context_reconstructed" else [], supported=False, reason="required FCS-to-FBS promotion fallback has zero pre-target training rows", cold_start_reasons={"fcs_to_fbs_transition": sum(row.cold_start_reason == "fcs_to_fbs_transition" for row in inference)}, eligible_promotion_rows=0, eligible_generic_rows=len(eligible_generic)))
            continue
        if needs_generic and not eligible_generic:
            for family in families:
                audit.append(reconstructed_prior_audit_row(target_season, family, 0, context_features if family == "context_reconstructed" else [], supported=False, reason="required generic FBS cold-start fallback has zero pre-target training rows", cold_start_reasons={"no_prior_rank_distribution": sum(row.cold_start_reason == "no_prior_rank_distribution" for row in inference)}, eligible_promotion_rows=len(eligible_promotion), eligible_generic_rows=0))
            continue
        promotion = generic = None
        if needs_promotion:
            promotion = context_prior.DirectRankModel.fit(
                eligible_promotion, [], penalty=0.25
            )
        if needs_generic:
            generic = context_prior.GenericRankPrior.fit(
                [
                    context_prior.TeamSeason(
                        row.season, row.subdivision, row.team_id, row.team_name,
                        row.population, np.asarray([0.0]), row.target_z,
                        row.target_ranks, {},
                    )
                    for row in eligible_generic
                ]
            )
        history_predictions, context_predictions = context_prior.future_predictions(
            inference,
            h_model,
            c_model,
            trained_through_season=trained_through,
            promotion_model=promotion,
            generic_prior=generic,
        )
        predictions_by_family = {
            "history_reconstructed": history_predictions,
            "context_reconstructed": context_predictions,
        }
        for family in families:
            predictions = predictions_by_family[family]
            pmfs = {
                str(row["team_id"]): np.asarray(
                    json.loads(row["pmf"]) if isinstance(row["pmf"], str) else row["pmf"],
                    dtype=float,
                )
                for row in predictions
            }
            result[(target_season, family)] = make_reconstructed_teams(
                target_season,
                family,
                [row for row in rows if int(row["season"]) == target_season],
                targets,
                pmfs,
            )
            audit.append(
                reconstructed_prior_audit_row(
                    target_season,
                    family,
                    len(pmfs),
                    context_features if family == "context_reconstructed" else [],
                )
            )
    return result, audit


def standard_cutoffs(
    rows: Sequence[Mapping[str, object]], season: int
) -> list[tuple[str, datetime]]:
    dates = sorted(
        {
            str(row["start_date"])[:10]
            for row in rows
            if int(row["season"]) == season and str(row.get("season_type")) == "regular"
        }
    )
    if not dates:
        dates = sorted({str(row["start_date"])[:10] for row in rows if int(row["season"]) == season})
    if not dates:
        raise ValueError(f"no dates available for {season}")
    indexes = [round(fraction * (len(dates) - 1)) for fraction in CUTOFF_FRACTIONS]
    result: list[tuple[str, datetime]] = []
    seen: set[str] = set()
    for ordinal, index in enumerate(indexes, 1):
        date = dates[index]
        if date in seen:
            continue
        seen.add(date)
        result.append((f"q{ordinal}_{date}", datetime.fromisoformat(f"{date}T23:59:59+00:00")))
    return result


def _game_population_hash(rows: Sequence[Mapping[str, object]]) -> str:
    value = "\n".join(sorted(str(row["game_id"]) for row in rows)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _team_population_hash(targets: Mapping[tuple[int, str, str], Mapping[str, object]], season: int) -> str:
    value = "\n".join(
        sorted(
            team_id
            for (target_season, subdivision, team_id) in targets
            if target_season == season and subdivision == "fbs"
        )
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _target_pmfs(
    targets: Mapping[tuple[int, str, str], Mapping[str, object]], season: int
) -> dict[str, np.ndarray]:
    return {
        team_id: np.asarray(value["pmf"], dtype=float)
        for (target_season, subdivision, team_id), value in targets.items()
        if target_season == season and subdivision == "fbs"
    }


def posterior_metrics(
    pmfs: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> dict[str, float | int]:
    keys = sorted(set(pmfs) & set(targets))
    if not keys:
        raise ValueError("no common FBS posterior/target keys")
    nll: list[float] = []
    crps: list[float] = []
    expected_mae: list[float] = []
    median_mae: list[float] = []
    coverage: list[float] = []
    width: list[float] = []
    entropy: list[float] = []
    max_probability: list[float] = []
    brier: dict[int, list[float]] = {5: [], 10: [], 25: []}
    for key in keys:
        prediction = np.asarray(pmfs[key], dtype=float)
        target = np.asarray(targets[key], dtype=float)
        if len(prediction) != len(target):
            raise ValueError(f"rank support mismatch for common key {key}")
        ranks = np.arange(1, len(prediction) + 1)
        p_summary = pmf_summaries(prediction)
        q_summary = pmf_summaries(target)
        nll.append(float(-np.sum(target * np.log(np.maximum(prediction, 1e-15)))))
        crps.append(float(np.mean((np.cumsum(prediction) - np.cumsum(target)) ** 2)))
        expected_mae.append(abs(p_summary["expected_rank"] - q_summary["expected_rank"]))
        median_mae.append(abs(p_summary["median_rank"] - q_summary["median_rank"]))
        low, high = p_summary["interval_80_low"], p_summary["interval_80_high"]
        coverage.append(float(target[(ranks >= low) & (ranks <= high)].sum()))
        width.append(high - low)
        entropy.append(float(-np.sum(prediction * np.log(np.maximum(prediction, 1e-15)))))
        max_probability.append(float(np.max(prediction)))
        for threshold, values in brier.items():
            values.append(
                float((p_summary[f"top{threshold}_probability"] - target[:threshold].sum()) ** 2)
            )
    return {
        "matched_fbs_teams": len(keys),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "expected_rank_mae": float(np.mean(expected_mae)),
        "median_rank_mae": float(np.mean(median_mae)),
        "interval_80_coverage": float(np.mean(coverage)),
        "interval_80_width": float(np.mean(width)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_probability": float(np.mean(max_probability)),
        **{f"top{threshold}_brier": float(np.mean(values)) for threshold, values in brier.items()},
    }


def _student_t_density(value: float, locations: np.ndarray, scale: float, df: float) -> np.ndarray:
    return student_t.pdf((value - locations) / scale, df) / scale


_FUTURE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def future_margin_score(
    game: Game,
    home: Team,
    away: Team,
    pmfs: Mapping[str, np.ndarray],
    likelihood: LikelihoodV1,
) -> dict[str, float]:
    actual_margin = oriented_margin(
        home.subdivision, away.subdivision, game.home_points, game.away_points
    )
    key = (
        likelihood.beta.tobytes(),
        likelihood.scale,
        likelihood.degrees_of_freedom,
        home.subdivision,
        away.subdivision,
        len(home.prior),
        len(away.prior),
        game.neutral_site,
        actual_margin,
    )
    cached = _FUTURE_CACHE.get(key)
    if cached is None:
        locations, _ = game_margin_parameters(game, home, away, likelihood)
        density = _student_t_density(actual_margin, locations, likelihood.scale, likelihood.degrees_of_freedom)
        win_density = student_t.sf((0.0 - locations) / likelihood.scale, likelihood.degrees_of_freedom)
        cached = (locations, density, win_density)
        _FUTURE_CACHE[key] = cached
    locations, density, win_density = cached
    weights = np.asarray(pmfs[home.team_id])[:, None] * np.asarray(pmfs[away.team_id])[None, :]
    predictive_density = float(np.sum(weights * density))
    expected_margin = float(np.sum(weights * locations))
    win_probability = float(np.sum(weights * win_density))
    actual_win = float(actual_margin > 0)
    return {
        "margin_mae": abs(expected_margin - actual_margin),
        "win_brier": (win_probability - actual_win) ** 2,
        "margin_nll": -math.log(max(predictive_density, 1e-300)),
    }


def _aggregate_future(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int | None]:
    if not rows:
        return {"n_games": 0, "margin_mae": None, "win_brier": None, "margin_nll": None}
    return {
        "n_games": len(rows),
        "margin_mae": float(np.mean([float(row["margin_mae"]) for row in rows])),
        "win_brier": float(np.mean([float(row["win_brier"]) for row in rows])),
        "margin_nll": float(np.mean([float(row["margin_nll"]) for row in rows])),
    }


def _next_game_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    by_team: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_team[str(row["home_team_id"])].append(row)
        by_team[str(row["away_team_id"])].append(row)
    selected: set[str] = set()
    for team_rows in by_team.values():
        first = min(team_rows, key=lambda row: (_parse_date(row["start_date"]), str(row["game_id"])))
        selected.add(str(first["game_id"]))
    return selected


def _future_rows(rows: Sequence[Mapping[str, object]], cutoff: datetime) -> list[Mapping[str, object]]:
    return [row for row in rows if _parse_date(row["start_date"]) > cutoff]


def _component_specs(candidate: str) -> tuple[tuple[str, str, bool], ...]:
    if candidate == "a":
        return (("a_yards", "yards", False),)
    if candidate == "b":
        return (("b_plays", "plays", False), ("b_yards", "yards", False))
    if candidate == "c":
        return (("c_plays", "plays", True), ("c_yards", "yards", True))
    raise ValueError(f"unknown primitive candidate: {candidate}")


def fit_candidate_models(
    data: PrimitiveData,
    seasons: Iterable[int],
    candidate: str,
    degrees_of_freedom: float,
) -> dict[str, dict[str, object]]:
    models: dict[str, dict[str, object]] = {}
    seasons = tuple(int(value) for value in seasons)
    for name, response_kind, turnover_context in _component_specs(candidate):
        mask = component_mask(
            data,
            seasons,
            response_kind=response_kind,
            turnover_context=turnover_context,
        )
        models[name] = fit_primitive_model(
            data,
            mask,
            response_kind=response_kind,
            turnover_context=turnover_context,
            rank_signal=True,
            degrees_of_freedom=degrees_of_freedom,
        )
    return models


def _component_score(
    data: PrimitiveData,
    model: Mapping[str, object],
    seasons: Iterable[int],
) -> dict[str, float | int]:
    mask = component_mask(
        data,
        seasons,
        response_kind=str(model["response_kind"]),
        turnover_context=bool(model["turnover_context"]),
    )
    return primitive_model_scores(model, data, mask)


def select_primitive_models(
    data: PrimitiveData,
) -> tuple[
    dict[str, dict[str, dict[str, object]]],
    dict[str, dict[str, dict[str, object]]],
    list[dict[str, object]],
]:
    """Select df only from 2018--2021 conditional development density.

    The returned first mapping contains train-only models for development
    posterior selection; the second contains train+development final models.
    """

    development_models: dict[str, dict[str, dict[str, object]]] = {}
    final_models: dict[str, dict[str, dict[str, object]]] = {}
    selection_rows: list[dict[str, object]] = []
    selected_df: dict[str, float] = {}
    for candidate in PRIMITIVE_CANDIDATES:
        options: list[tuple[float, float, dict[str, dict[str, object]], list[dict[str, object]]]] = []
        for df in DF_GRID:
            models = fit_candidate_models(data, TRAIN_YEARS, candidate, df)
            component_rows: list[dict[str, object]] = []
            component_scores: list[float] = []
            for name, model in models.items():
                score = _component_score(data, model, DEVELOPMENT_YEARS)
                component_rows.append(
                    {
                        "component": name,
                        "response_kind": model["response_kind"],
                        "turnover_context": model["turnover_context"],
                        "development_games": score["n_games"],
                        "development_marginalized_nll": score["marginalized_nll"],
                        "development_conditional_nll": score["expected_conditional_nll"],
                    }
                )
                if score["n_games"]:
                    component_scores.append(float(score["marginalized_nll"]))
            aggregate = float(np.mean(component_scores))
            train_games = sorted(
                {
                    str(game_id)
                    for name, model in models.items()
                    for game_id in data.game_id[
                        component_mask(
                            data,
                            TRAIN_YEARS,
                            response_kind=str(model["response_kind"]),
                            turnover_context=bool(model["turnover_context"]),
                        )
                    ]
                }
            )
            selection_rows.append(
                {
                    "candidate": candidate,
                    "student_t_df": df,
                    "training_seasons": "2004-2017",
                    "development_seasons": "2018-2021",
                    "training_game_count_union": len(train_games),
                    "development_marginalized_nll_mean": aggregate,
                    "development_components": json.dumps(component_rows, sort_keys=True),
                    "selected": False,
                }
            )
            options.append((aggregate, df, models, component_rows))
        best_score, best_df, best_models, _component_rows = min(options, key=lambda item: (item[0], item[1]))
        selected_df[candidate] = best_df
        development_models[candidate] = best_models
        final_models[candidate] = fit_candidate_models(
            data,
            (*TRAIN_YEARS, *DEVELOPMENT_YEARS),
            candidate,
            best_df,
        )
        for row in selection_rows:
            if row["candidate"] == candidate and row["student_t_df"] == best_df:
                row["selected"] = True
                row["selected_development_marginalized_nll_mean"] = best_score
                row["final_fit_seasons"] = "2004-2021"
    # Add a compact, machine-readable selected row for each candidate after the
    # per-df rows so no downstream selection needs to parse JSON strings.
    for candidate in PRIMITIVE_CANDIDATES:
        selection_rows.append(
            {
                "candidate": candidate,
                "student_t_df": selected_df[candidate],
                "training_seasons": "2004-2017",
                "development_seasons": "2018-2021",
                "selected": True,
                "selection_row_type": "selected_summary",
                "final_fit_seasons": "2004-2021",
            }
        )
    return development_models, final_models, selection_rows


def _uniform_models_for_development(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    season: int,
) -> list[Team]:
    return make_uniform_teams(season, rows, targets)


def _infer_variant(
    variant: str,
    teams: list[Team],
    eligible: Sequence[Mapping[str, object]],
    likelihood: LikelihoodV1,
    primitive_models: Mapping[str, Mapping[str, object] | None],
    ypp_model: Mapping[str, object] | None,
) -> PosteriorResult:
    games = [row_to_game(row) for row in eligible]
    evidence = build_evidence_by_game(eligible)
    if variant == "ypp_supported":
        if ypp_model is None:
            raise ValueError("fixed YPP model is required for ypp inference")
        ypp_by_game = {
            str(row["game_id"]): (row.get("home_ypp"), row.get("away_ypp"))
            for row in eligible
        }
        result = infer_posterior_with_ypp(
            teams,
            games,
            likelihood,
            ypp_by_game,
            ypp_model,
            allowed_pairings=SUPPORTED_YPP_PAIRINGS,
            max_iterations=100,
            tolerance=1e-6,
            damping=0.35,
        )
    else:
        result = infer_posterior_with_primitives(
            teams,
            games,
            likelihood,
            evidence,
            primitive_models,
            variant=variant,
            max_iterations=100,
            tolerance=1e-6,
            damping=0.35,
        )
    if not result.converged:
        raise RuntimeError(
            f"posterior did not converge for {variant}: delta={result.max_message_delta} iterations={result.iterations}"
        )
    return result


def _future_key_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return _game_population_hash(rows)


def validate_common_comparison_keys(
    rows: Sequence[Mapping[str, object]], candidates: Sequence[str]
) -> dict[str, object]:
    """Require identical game/team support for every candidate comparison."""

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["season"],
                row["cutoff_index"],
                row["prior_family"],
                row.get("population_view", "full"),
            )
        ].append(row)
    audits: list[dict[str, object]] = []
    expected = set(candidates)
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        hashes = {(row["game_key_sha256"], row["team_key_sha256"]) for row in group}
        seen = {str(row["candidate"]) for row in group}
        if len(hashes) != 1 or seen != expected:
            raise ValueError(f"comparison support mismatch for {key}: {seen}")
        audits.append(
            {
                "season": key[0],
                "cutoff_index": key[1],
                "prior_family": key[2],
                "population_view": key[3],
                "candidate_count": len(group),
                "game_key_sha256": group[0]["game_key_sha256"],
                "team_key_sha256": group[0]["team_key_sha256"],
            }
        )
    return {"groups": audits, "group_count": len(audits)}


def run_posterior_panel(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    likelihood: LikelihoodV1,
    primitive_models: Mapping[str, Mapping[str, Mapping[str, object]]],
    ypp_model: Mapping[str, object] | None,
    *,
    seasons: Sequence[int],
    candidates: Sequence[str],
    prior_families: Sequence[str],
    frozen_priors: bool,
    include_future: bool,
    future_candidates: Sequence[str] | None = None,
    final_only: bool = False,
    reconstructed_teams: Mapping[tuple[int, str], Sequence[Team]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Evaluate matched candidates on identical rows, cutoffs, and targets."""

    candidate_rows: list[dict[str, object]] = []
    season_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    future_candidates = tuple(future_candidates or candidates)
    for season in seasons:
        season_source = [row for row in rows if int(row["season"]) == season]
        cutoffs = standard_cutoffs(rows, season)
        for cutoff_index, (cutoff_label, cutoff) in enumerate(cutoffs):
            if final_only and cutoff_index != len(cutoffs) - 1:
                continue
            eligible = [
                row
                for row in season_source
                if _parse_date(row["start_date"]) <= cutoff
            ]
            if reconstructed_teams is not None:
                team_by_family = {
                    family: list(reconstructed_teams[(season, family)])
                    for family in prior_families
                }
            elif frozen_priors:
                # All final-season candidate rows use exactly the same frozen
                # family/team support.  Development uses uniform supports for
                # the explicit selection-prior reason documented above.
                team_by_family = {
                    family: make_frozen_teams(season, family, eligible, targets)
                    for family in prior_families
                }
            else:
                team_by_family = {
                    # Development selection uses an explicit uniform ordinal
                    # prior over the full season's target support.  The
                    # full-season support is needed even though the games are
                    # still restricted to the cutoff, otherwise early cutoffs
                    # can have no common target keys.  Selection calls use the
                    # final cutoff only (``final_only=True``).
                    family: _uniform_models_for_development(season_source, targets, season)
                    for family in prior_families
                }
            game_hash = _game_population_hash(eligible)
            team_hash = _team_population_hash(targets, season)
            for family in prior_families:
                teams = team_by_family[family]
                results: dict[str, PosteriorResult] = {}
                for variant in candidates:
                    model_map = primitive_models.get(variant, primitive_models.get("a", {}))
                    result = _infer_variant(
                        variant,
                        teams,
                        eligible,
                        likelihood,
                        model_map,
                        ypp_model,
                    )
                    results[variant] = result
                    metrics = posterior_metrics(result.pmfs, _target_pmfs(targets, season))
                    row = {
                        "season": season,
                        "period": _period(season),
                        "cutoff_label": cutoff_label,
                        "cutoff": cutoff.isoformat(),
                        "cutoff_index": cutoff_index,
                        "is_final_cutoff": cutoff_index == len(cutoffs) - 1,
                        "prior_family": family,
                        "candidate": variant,
                        "population_view": "full",
                        "game_count": len(eligible),
                        "primitive_complete_game_count": sum(
                            all(
                                row.get(f"{side}_{field}") is not None
                                for side in ("home", "away")
                                for field in ("total_yards", "offensive_plays_derived")
                            )
                            and str(row["pairing"]) in SUPPORTED_PRIMITIVE_PAIRINGS
                            for row in eligible
                        ),
                        "game_key_sha256": game_hash,
                        "team_key_sha256": team_hash,
                        **metrics,
                        "posterior_converged": result.converged,
                        "posterior_iterations": result.iterations,
                        "posterior_max_message_delta": result.max_message_delta,
                    }
                    candidate_rows.append(row)
                    calibration_rows.append(
                        {
                            "season": season,
                            "period": _period(season),
                            "cutoff_label": cutoff_label,
                            "cutoff_index": cutoff_index,
                            "is_final_cutoff": cutoff_index == len(cutoffs) - 1,
                            "prior_family": family,
                            "candidate": variant,
                            "n_teams": metrics["matched_fbs_teams"],
                            "nll": metrics["nll"],
                            "crps": metrics["crps"],
                            "interval_80_coverage": metrics["interval_80_coverage"],
                            "interval_80_width": metrics["interval_80_width"],
                            "mean_entropy": metrics["mean_entropy"],
                            "mean_max_probability": metrics["mean_max_probability"],
                        }
                    )
                    if cutoff_index == len(cutoffs) - 1:
                        season_rows.append(row.copy())

                if include_future:
                    future = _future_rows(season_source, cutoff)
                    next_ids = _next_game_ids(future)
                    team_lookup = {team.team_id: team for team in teams}
                    for variant in future_candidates:
                        if variant not in results:
                            # A future-only diagnostic still needs its posterior.
                            model_map = primitive_models.get(variant, primitive_models.get("a", {}))
                            results[variant] = _infer_variant(
                                variant,
                                teams,
                                eligible,
                                likelihood,
                                model_map,
                                ypp_model,
                            )
                        result = results[variant]
                        scores: list[dict[str, float]] = []
                        next_scores: list[dict[str, float]] = []
                        scored_future: list[Mapping[str, object]] = []
                        scored_next: list[Mapping[str, object]] = []
                        for future_row in future:
                            game = row_to_game(future_row)
                            if game.home_id not in result.pmfs or game.away_id not in result.pmfs:
                                continue
                            if game.home_id not in team_lookup or game.away_id not in team_lookup:
                                continue
                            score = future_margin_score(
                                game,
                                team_lookup[game.home_id],
                                team_lookup[game.away_id],
                                result.pmfs,
                                likelihood,
                            )
                            scores.append(score)
                            scored_future.append(future_row)
                            if str(future_row["game_id"]) in next_ids:
                                next_scores.append(score)
                                scored_next.append(future_row)
                        aggregate = _aggregate_future(scores)
                        next_aggregate = _aggregate_future(next_scores)
                        future_rows.append(
                            {
                                "season": season,
                                "period": _period(season),
                                "cutoff_label": cutoff_label,
                                "cutoff_index": cutoff_index,
                                "is_final_cutoff": cutoff_index == len(cutoffs) - 1,
                                "prior_family": family,
                                "candidate": variant,
                                "future_game_count": aggregate["n_games"],
                                "future_margin_mae": aggregate["margin_mae"],
                                "future_win_brier": aggregate["win_brier"],
                                "future_margin_nll": aggregate["margin_nll"],
                                "future_game_key_sha256": _future_key_hash(scored_future),
                                "next_game_count": next_aggregate["n_games"],
                                "next_margin_mae": next_aggregate["margin_mae"],
                                "next_win_brier": next_aggregate["win_brier"],
                                "next_margin_nll": next_aggregate["margin_nll"],
                                "next_game_key_sha256": _future_key_hash(scored_next),
                            }
                        )
    # Strict comparison-key checks are done on every candidate group.
    comparison_audit = validate_common_comparison_keys(candidate_rows, candidates)["groups"]
    future_groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in future_rows:
        future_groups[(row["season"], row["cutoff_index"], row["prior_family"])].append(row)
    future_audit: list[dict[str, object]] = []
    for key, group in sorted(future_groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        future_hashes = {row["future_game_key_sha256"] for row in group}
        next_hashes = {row["next_game_key_sha256"] for row in group}
        if len(future_hashes) != 1 or len(next_hashes) != 1:
            raise ValueError(f"future comparison support mismatch for {key}")
        future_audit.append(
            {
                "season": key[0],
                "cutoff_index": key[1],
                "prior_family": key[2],
                "candidate_count": len(group),
                "future_game_key_sha256": group[0]["future_game_key_sha256"],
                "next_game_key_sha256": group[0]["next_game_key_sha256"],
            }
        )
    return candidate_rows, season_rows, calibration_rows, {
        "future_game_metrics": future_rows,
        "comparison_key_audit": comparison_audit,
        "future_key_audit": future_audit,
        "standard_cutoff_definition": "seven actual-date regular-season quantiles: 0, .20, .35, .55, .72, .87, 1.0",
        "posterior_engine": "frozen V1 BP margin factor plus research factor; 100 iterations, tolerance 1e-6, damping .35",
    }


def _final_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [row for row in rows if _bool(row.get("is_final_cutoff"))]


def _metric_mean(
    rows: Sequence[Mapping[str, object]], candidate: str, metric: str
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if str(row["candidate"]) == candidate and row.get(metric) is not None
    ]
    if not values:
        raise ValueError(f"no {metric} values for {candidate}")
    return float(np.mean(values))


def _season_metric_means(
    rows: Sequence[Mapping[str, object]], candidate: str, metric: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if str(row["candidate"]) == candidate and row.get(metric) is not None:
            grouped[int(row["season"])].append(float(row[metric]))
    return {season: float(np.mean(values)) for season, values in grouped.items()}


def _comparison_gate(
    newer: str,
    predecessor: str,
    development_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    newer_nll = _metric_mean(development_rows, newer, "nll")
    predecessor_nll = _metric_mean(development_rows, predecessor, "nll")
    newer_crps = _metric_mean(development_rows, newer, "crps")
    predecessor_crps = _metric_mean(development_rows, predecessor, "crps")
    newer_coverage = _metric_mean(development_rows, newer, "interval_80_coverage")
    predecessor_coverage = _metric_mean(development_rows, predecessor, "interval_80_coverage")
    newer_season = _season_metric_means(development_rows, newer, "nll")
    predecessor_season = _season_metric_means(development_rows, predecessor, "nll")
    deltas = [newer_season[season] - predecessor_season[season] for season in sorted(newer_season)]
    nll_gain = predecessor_nll - newer_nll
    crps_delta = newer_crps - predecessor_crps
    coverage_delta = newer_coverage - predecessor_coverage
    seasons_improved = sum(delta < 0 for delta in deltas)
    max_season_nll_degradation = max(deltas, default=float("nan"))
    checks = {
        "aggregate_nll_gain": nll_gain >= float(DEVELOPMENT_SELECTION_RULE["aggregate_nll_gain_nats_per_team"]),
        "aggregate_crps": crps_delta <= float(DEVELOPMENT_SELECTION_RULE["maximum_crps_degradation"]),
        "aggregate_coverage": coverage_delta >= -float(DEVELOPMENT_SELECTION_RULE["maximum_80pct_coverage_drop"]),
        "majority_seasons": seasons_improved >= int(DEVELOPMENT_SELECTION_RULE["minimum_of_four_development_seasons_with_nll_improvement"]),
        "no_catastrophic_season": max_season_nll_degradation <= float(DEVELOPMENT_SELECTION_RULE["maximum_single_season_nll_degradation"]),
    }
    return {
        "newer": newer,
        "predecessor": predecessor,
        "development_nll_newer": newer_nll,
        "development_nll_predecessor": predecessor_nll,
        "development_nll_gain_predecessor_minus_newer": nll_gain,
        "development_crps_delta_newer_minus_predecessor": crps_delta,
        "development_coverage_delta_newer_minus_predecessor": coverage_delta,
        "development_seasons_with_nll_improvement": seasons_improved,
        "development_max_season_nll_degradation": max_season_nll_degradation,
        "checks": checks,
        "passed": all(checks.values()),
        "season_nll_deltas": {
            str(season): newer_season[season] - predecessor_season[season]
            for season in sorted(newer_season)
        },
    }


def select_candidate_from_development(
    development_rows: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Select exactly one A/B/C candidate without reading final metrics."""

    final_development = _final_rows(development_rows)
    predecessor = "v1"
    path: list[str] = []
    comparisons: list[dict[str, object]] = []
    for candidate in PRIMITIVE_CANDIDATES:
        comparison = _comparison_gate(candidate, predecessor, final_development)
        comparison["selection_stage"] = f"{candidate}_vs_{predecessor}"
        comparison["selected_after_comparison"] = bool(comparison["passed"])
        comparisons.append(comparison)
        if comparison["passed"]:
            path.append(candidate)
            predecessor = candidate
        else:
            # The next stage is still labeled B-vs-A or C-vs-B, but an
            # unpromoted predecessor cannot carry the selection path.
            predecessor = candidate
    selected = path[-1] if path else "a"
    return selected, comparisons, {
        "selected_candidate": selected,
        "selection_path": path,
        "selection_rule": DEVELOPMENT_SELECTION_RULE,
        "final_metrics_used": False,
        "final_seasons_used": [],
    }


def _quality_coordinates(
    row: Mapping[str, object],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
) -> tuple[float, float] | None:
    home = targets.get((int(row["season"]), str(row["home_subdivision"]), str(row["home_team_id"])))
    away = targets.get((int(row["season"]), str(row["away_subdivision"]), str(row["away_team_id"])))
    if home is None or away is None:
        return None
    return oriented_rank_coordinates(
        str(row["home_subdivision"]),
        str(row["away_subdivision"]),
        float(home["mean_rank"]),
        float(away["mean_rank"]),
        int(row["home_team_population"]),
        int(row["away_team_population"]),
    )


def _oriented_primitive_values(row: Mapping[str, object]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for field in PRIMITIVE_FIELDS:
        difference = oriented_difference(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            row.get(f"home_{field}"),
            row.get(f"away_{field}"),
        )
        home = _float(row.get(f"home_{field}"))
        away = _float(row.get(f"away_{field}"))
        if difference is None or home is None or away is None:
            return None
        values[f"{field}_diff"] = difference
        values[f"{field}_total"] = home + away
    return values


def _oriented_values_for_model(
    row: Mapping[str, object], model: Mapping[str, object]
) -> dict[str, float] | None:
    """Return model-required values while preserving unneeded missing fields."""

    values: dict[str, float] = {}
    for field in PRIMITIVE_FIELDS:
        difference = oriented_difference(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            row.get(f"home_{field}"),
            row.get(f"away_{field}"),
        )
        home = _float(row.get(f"home_{field}"))
        away = _float(row.get(f"away_{field}"))
        values[f"{field}_diff"] = float("nan") if difference is None else difference
        values[f"{field}_total"] = float("nan") if home is None or away is None else home + away
    required = ["offensive_plays_derived_diff", "offensive_plays_derived_total"]
    if str(model["response_kind"]) == "yards":
        required.extend(("total_yards_diff", "total_yards_total"))
    if bool(model["turnover_context"]):
        required.extend(
            (
                "interceptions_thrown_diff",
                "interceptions_thrown_total",
                "fumbles_lost_diff",
                "fumbles_lost_total",
            )
        )
    return values if all(np.isfinite(values[name]) for name in required) else None


def _location_for_row(
    model: Mapping[str, object],
    row: Mapping[str, object],
    x: float,
    y: float,
) -> float | None:
    # The model's required context determines missingness.  A plays-only
    # signal does not discard a row solely because turnover context is absent.
    values = _oriented_values_for_model(row, model)
    if values is None:
        return None
    try:
        locations = _model_locations(
            model,
            margin=np.asarray([
                oriented_margin(
                    str(row["home_subdivision"]),
                    str(row["away_subdivision"]),
                    int(row["home_points"]),
                    int(row["away_points"]),
                )
            ]),
            pairing=np.asarray([str(row["pairing"])]),
            neutral=np.asarray([float(_bool(row["neutral_site"]))]),
            fbs_home=np.asarray([
                float(
                    str(row["home_subdivision"]) != str(row["away_subdivision"])
                    and str(row["home_subdivision"]) == "fbs"
                    and not _bool(row["neutral_site"])
                )
            ]),
            x=np.asarray([x]),
            y=np.asarray([y]),
            plays_diff=np.asarray([values["offensive_plays_derived_diff"]]),
            plays_total=np.asarray([values["offensive_plays_derived_total"]]),
            yards_total=np.asarray([values["total_yards_total"]]),
            interceptions_diff=np.asarray([values["interceptions_thrown_diff"]]),
            interceptions_total=np.asarray([values["interceptions_thrown_total"]]),
            fumbles_lost_diff=np.asarray([values["fumbles_lost_diff"]]),
            fumbles_lost_total=np.asarray([values["fumbles_lost_total"]]),
        )
    except (ValueError, FloatingPointError):
        return None
    return float(locations[0])


def _response_value(row: Mapping[str, object], response_kind: str) -> float | None:
    home_subdivision = str(row["home_subdivision"])
    away_subdivision = str(row["away_subdivision"])
    if response_kind == "plays":
        difference = oriented_difference(
            home_subdivision,
            away_subdivision,
            row.get("home_offensive_plays_derived"),
            row.get("away_offensive_plays_derived"),
        )
        return None if difference is None else float(np.arcsinh(difference / 10.0))
    if response_kind == "yards":
        difference = oriented_difference(
            home_subdivision,
            away_subdivision,
            row.get("home_total_yards"),
            row.get("away_total_yards"),
        )
        return None if difference is None else float(np.arcsinh(difference / 100.0))
    raise ValueError(response_kind)


def _slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.var(xa) <= 1e-14:
        return None
    return float(np.cov(xa, ya, ddof=0)[0, 1] / np.var(xa))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _attach_next_margin(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_team: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_team[str(row["home_team_id"])].append(row)
        by_team[str(row["away_team_id"])].append(row)
    for values in by_team.values():
        values.sort(key=lambda item: (_parse_date(item["start_date"]), str(item["game_id"])))

    output: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        current = _parse_date(row["start_date"])
        home_id, away_id = str(row["home_team_id"]), str(row["away_team_id"])
        first_id, second_id = (
            (away_id, home_id)
            if str(row["home_subdivision"]) == "fcs" and str(row["away_subdivision"]) == "fbs"
            else (home_id, away_id)
        )

        def next_game(
            team_id: str, before: datetime = current
        ) -> Mapping[str, object] | None:
            return next(
                (
                    candidate
                    for candidate in by_team[team_id]
                    if _parse_date(candidate["start_date"]) > before
                ),
                None,
            )

        first = next_game(first_id)
        second = next_game(second_id)
        if first is None or second is None:
            row["next_margin_diff"] = None
        else:
            first_margin = oriented_margin(
                str(first["home_subdivision"]),
                str(first["away_subdivision"]),
                int(first["home_points"]),
                int(first["away_points"]),
            ) if first_id == str(first["home_team_id"]) else -oriented_margin(
                str(first["home_subdivision"]),
                str(first["away_subdivision"]),
                int(first["home_points"]),
                int(first["away_points"]),
            )
            second_margin = oriented_margin(
                str(second["home_subdivision"]),
                str(second["away_subdivision"]),
                int(second["home_points"]),
                int(second["away_points"]),
            ) if second_id == str(second["home_team_id"]) else -oriented_margin(
                str(second["home_subdivision"]),
                str(second["away_subdivision"]),
                int(second["home_points"]),
                int(second["away_points"]),
            )
            row["next_margin_diff"] = first_margin - second_margin
        output.append(row)
    return output


def build_play_signal(
    rows: Sequence[Mapping[str, object]],
    data: PrimitiveData,
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    model: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Describe play-differential residuals after fixed context conditioning."""

    enriched = _attach_next_margin(rows)
    observations: list[dict[str, object]] = []
    for row in enriched:
        if str(row["pairing"]) not in SUPPORTED_PRIMITIVE_PAIRINGS:
            continue
        values = _oriented_values_for_model(row, model)
        quality = _quality_coordinates(row, targets)
        if values is None or quality is None:
            continue
        response = _response_value(row, "plays")
        if response is None:
            continue
        x, y = quality
        full_location = _location_for_row(model, row, x, y)
        null_location = _location_for_row(model, row, 0.0, 0.0)
        if full_location is None or null_location is None:
            continue
        observations.append(
            {
                "period": _period(int(row["season"])),
                "season": int(row["season"]),
                "pairing": row["pairing"],
                "margin_band": _margin_bin(
                    oriented_margin(
                        str(row["home_subdivision"]),
                        str(row["away_subdivision"]),
                        int(row["home_points"]),
                        int(row["away_points"]),
                    )
                ),
                "site": "neutral" if _bool(row["neutral_site"]) else "home_site",
                "plays_diff": values["offensive_plays_derived_diff"],
                "plays_total": values["offensive_plays_derived_total"],
                "play_response": response,
                "play_residual_after_margin_site_volume": response - null_location,
                "rank_signal_location_shift": full_location - null_location,
                "quality_advantage": y - x,
                "next_margin_diff": row.get("next_margin_diff"),
            }
        )
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in observations:
        # Site is a conditioning variable, not a requested reporting axis; an
        # all-site row prevents site imbalance from being mistaken for signal.
        grouped[(row["period"], row["season"], row["pairing"], row["margin_band"])].append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        residual = [float(row["play_residual_after_margin_site_volume"]) for row in values]
        quality = [float(row["quality_advantage"]) for row in values]
        next_pairs = [
            (float(row["play_residual_after_margin_site_volume"]), float(row["next_margin_diff"]))
            for row in values
            if row.get("next_margin_diff") is not None
        ]
        output.append(
            {
                "period": key[0],
                "season": key[1],
                "pairing": key[2],
                "margin_band": key[3],
                "site_control": "home/neutral included as a model covariate",
                "overall_volume_control": "plays_total included as a model covariate",
                "n_games": len(values),
                "mean_plays_diff": float(np.mean([float(row["plays_diff"]) for row in values])),
                "mean_plays_total": float(np.mean([float(row["plays_total"]) for row in values])),
                "more_plays_rate": float(np.mean(np.asarray([float(row["plays_diff"]) for row in values]) > 0)),
                "mean_play_residual": float(np.mean(residual)),
                "mean_rank_signal_location_shift": float(np.mean([float(row["rank_signal_location_shift"]) for row in values])),
                "quality_slope": _slope(residual, quality),
                "quality_spearman": _spearman(residual, quality),
                "next_margin_slope": _slope(
                    [pair[0] for pair in next_pairs], [pair[1] for pair in next_pairs]
                ),
                "next_margin_spearman": _spearman(
                    [pair[0] for pair in next_pairs], [pair[1] for pair in next_pairs]
                ),
            }
        )
    quality_rows = [row for row in output if row["quality_slope"] is not None]
    return output, {
        "rows": len(observations),
        "positive_quality_slope_rows": sum(float(row["quality_slope"]) > 0 for row in quality_rows),
        "quality_slope_rows": len(quality_rows),
        "interpretation": "positive quality slope means more play differential is associated with the better final-rank side after the fixed margin/site/overall-volume conditioning model; it is descriptive, not a direct play-count rating factor",
    }


def _turnover_pattern(values: Mapping[str, float]) -> str:
    labels: list[str] = []
    if values["interceptions_thrown_diff"] > 0:
        labels.append("A_more_interceptions_thrown")
    elif values["interceptions_thrown_diff"] < 0:
        labels.append("B_more_interceptions_thrown")
    else:
        labels.append("interceptions_equal")
    if values["fumbles_lost_diff"] > 0:
        labels.append("A_more_fumbles_lost")
    elif values["fumbles_lost_diff"] < 0:
        labels.append("B_more_fumbles_lost")
    else:
        labels.append("fumbles_lost_equal")
    return "+".join(labels)


def _log_density_at_row(
    model: Mapping[str, object], row: Mapping[str, object], x: float, y: float
) -> float | None:
    location = _location_for_row(model, row, x, y)
    response = _response_value(row, str(model["response_kind"]))
    if location is None or response is None:
        return None
    return float(
        _student_t_logpdf(
            response,
            np.asarray([location]),
            float(model["scale"]),
            float(model["df"]),
        )[0]
    )


def build_turnover_context_table(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize separate INT/fumble context without direct turnover evidence."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["pairing"]) not in SUPPORTED_PRIMITIVE_PAIRINGS:
            continue
        values = _oriented_primitive_values(row)
        quality = _quality_coordinates(row, targets)
        if values is None or quality is None:
            continue
        if any(
            _float(row.get(f"{side}_{field}")) is None
            for side in ("home", "away")
            for field in ("interceptions_thrown", "fumbles_lost")
        ):
            continue
        x, y = quality
        b_plays = _log_density_at_row(models["b_plays"], row, x, y)
        b_yards = _log_density_at_row(models["b_yards"], row, x, y)
        c_plays = _log_density_at_row(models["c_plays"], row, x, y)
        c_yards = _log_density_at_row(models["c_yards"], row, x, y)
        if None in (b_plays, b_yards, c_plays, c_yards):
            continue
        margin = oriented_margin(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            int(row["home_points"]),
            int(row["away_points"]),
        )
        key = (
            _period(int(row["season"])),
            int(row["season"]),
            str(row["pairing"]),
            _margin_bin(margin),
            _turnover_pattern(values),
        )
        grouped[key].append(
            {
                "values": values,
                "delta": float(c_plays + c_yards - b_plays - b_yards),
            }
        )
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        values = [item["values"] for item in group]
        output.append(
            {
                "period": key[0],
                "season": key[1],
                "pairing": key[2],
                "margin_band": key[3],
                "turnover_pattern": key[4],
                "n_games": len(group),
                "mean_interceptions_diff": float(np.mean([row["interceptions_thrown_diff"] for row in values])),
                "mean_interceptions_total": float(np.mean([row["interceptions_thrown_total"] for row in values])),
                "mean_fumbles_lost_diff": float(np.mean([row["fumbles_lost_diff"] for row in values])),
                "mean_fumbles_lost_total": float(np.mean([row["fumbles_lost_total"] for row in values])),
                "mean_yards_diff": float(np.mean([row["total_yards_diff"] for row in values])),
                "mean_plays_diff": float(np.mean([row["offensive_plays_derived_diff"] for row in values])),
                "mean_log_density_c_minus_b_at_final_rank": float(np.mean([item["delta"] for item in group])),
                "context_role": "separate INT/fumbles-lost conditioning only; no p(INT,FL|ranks) factor",
            }
        )
    return output


def build_disagreement_games(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Select deterministic real corpus examples of score/primitive disagreement."""

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["season"]) not in FINAL_YEARS or str(row["pairing"]) not in SUPPORTED_PRIMITIVE_PAIRINGS:
            continue
        values = _oriented_primitive_values(row)
        quality = _quality_coordinates(row, targets)
        if values is None or quality is None:
            continue
        margin = oriented_margin(
            str(row["home_subdivision"]),
            str(row["away_subdivision"]),
            int(row["home_points"]),
            int(row["away_points"]),
        )
        pattern = None
        if -14 < margin < 0 and values["total_yards_diff"] >= 100 and values["interceptions_thrown_diff"] >= 2:
            pattern = "close_loss_large_yard_advantage_multiple_interceptions"
        elif 0 < margin <= 14 and values["total_yards_diff"] <= -100 and values["fumbles_lost_diff"] <= -1:
            pattern = "close_win_yard_deficit_opponent_fumbles_lost"
        if pattern is None:
            continue
        x, y = quality
        b_plays = _log_density_at_row(models["b_plays"], row, x, y)
        b_yards = _log_density_at_row(models["b_yards"], row, x, y)
        c_plays = _log_density_at_row(models["c_plays"], row, x, y)
        c_yards = _log_density_at_row(models["c_yards"], row, x, y)
        if None in (b_plays, b_yards, c_plays, c_yards):
            continue
        candidates[pattern].append(
            {
                "disagreement_type": pattern,
                "season": row["season"],
                "game_id": row["game_id"],
                "home_team": row.get("home_team", row["home_team_id"]),
                "away_team": row.get("away_team", row["away_team_id"]),
                "home_score": row["home_points"],
                "away_score": row["away_points"],
                "pairing": row["pairing"],
                "oriented_margin": margin,
                "oriented_yards_diff": values["total_yards_diff"],
                "oriented_yards_total": values["total_yards_total"],
                "oriented_plays_diff": values["offensive_plays_derived_diff"],
                "oriented_plays_total": values["offensive_plays_derived_total"],
                "interceptions_thrown_a": row.get("home_interceptions_thrown") if str(row["home_subdivision"]) == "fbs" or row["home_subdivision"] == row["away_subdivision"] else row.get("away_interceptions_thrown"),
                "interceptions_thrown_b": row.get("away_interceptions_thrown") if str(row["home_subdivision"]) == "fbs" or row["home_subdivision"] == row["away_subdivision"] else row.get("home_interceptions_thrown"),
                "fumbles_lost_a": row.get("home_fumbles_lost") if str(row["home_subdivision"]) == "fbs" or row["home_subdivision"] == row["away_subdivision"] else row.get("away_fumbles_lost"),
                "fumbles_lost_b": row.get("away_fumbles_lost") if str(row["home_subdivision"]) == "fbs" or row["home_subdivision"] == row["away_subdivision"] else row.get("home_fumbles_lost"),
                "b_conditional_log_density_at_final_rank": b_plays + b_yards,
                "c_conditional_log_density_at_final_rank": c_plays + c_yards,
                "c_minus_b_context_log_density": c_plays + c_yards - b_plays - b_yards,
                "note": "turnover counts alter conditional yards/plays interpretation only; no direct turnover rating factor",
            }
        )
    output: list[dict[str, object]] = []
    for pattern in sorted(candidates):
        output.append(
            min(
                candidates[pattern],
                key=lambda row: (-abs(float(row["oriented_yards_diff"])), int(row["season"]), str(row["game_id"])),
            )
        )
    return output


def run_rolling_robustness(
    rows: Sequence[Mapping[str, object]],
    targets: Mapping[tuple[int, str, str], Mapping[str, object]],
    data: PrimitiveData,
    likelihood: LikelihoodV1,
    selected_candidate: str,
    selected_df: float,
    reconstructed_teams: Mapping[tuple[int, str], Sequence[Team]] | None = None,
    prior_family: str = "uniform_selection_prior",
) -> list[dict[str, object]]:
    """Fit the selected architecture only on seasons before each target."""

    output: list[dict[str, object]] = []
    for target_season in ROLLING_YEARS:
        prior_seasons = tuple(year for year in range(2004, target_season) if year <= 2025)
        if not prior_seasons:
            continue
        models = fit_candidate_models(data, prior_seasons, selected_candidate, selected_df)
        primitive_models = {selected_candidate: models}
        candidate_rows, season_rows, _calibration, _metadata = run_posterior_panel(
            rows,
            targets,
            likelihood,
            primitive_models,
            None,
            seasons=(target_season,),
            candidates=("v1", selected_candidate),
            prior_families=(prior_family,),
            frozen_priors=False,
            include_future=False,
            final_only=True,
            reconstructed_teams=reconstructed_teams,
        )
        del candidate_rows
        for row in _final_rows(season_rows):
            output.append(
                {
                    "target_season": target_season,
                    "fit_seasons": f"{prior_seasons[0]}-{prior_seasons[-1]}",
                    "prior_family": prior_family,
                    "candidate": row["candidate"],
                    "game_count": row["game_count"],
                    "nll": row["nll"],
                    "crps": row["crps"],
                    "expected_rank_mae": row["expected_rank_mae"],
                    "interval_80_coverage": row["interval_80_coverage"],
                    "interval_80_width": row["interval_80_width"],
                    "game_key_sha256": row["game_key_sha256"],
                    "team_key_sha256": row["team_key_sha256"],
                    "is_final_cutoff": True,
                }
            )
    return output


def _aggregate_final_metrics(
    season_rows: Sequence[Mapping[str, object]], candidates: Sequence[str]
) -> list[dict[str, object]]:
    final = _final_rows(season_rows)
    output: list[dict[str, object]] = []
    metric_names = (
        "nll",
        "crps",
        "expected_rank_mae",
        "median_rank_mae",
        "interval_80_coverage",
        "interval_80_width",
        "top5_brier",
        "top10_brier",
        "top25_brier",
        "mean_entropy",
        "mean_max_probability",
    )
    for family in ("context", "history"):
        for candidate in candidates:
            values = [
                row
                for row in final
                if row["prior_family"] == family and row["candidate"] == candidate
            ]
            if not values:
                continue
            output.append(
                {
                    "summary_type": "final_aggregate",
                    "prior_family": family,
                    "candidate": candidate,
                    "season": "all",
                    "n_seasons": len(values),
                    **{metric: float(np.mean([float(row[metric]) for row in values])) for metric in metric_names},
                }
            )
    return output


def build_metric_deltas(
    season_rows: Sequence[Mapping[str, object]],
    candidates: Sequence[str],
    baseline: str = "v1",
) -> list[dict[str, object]]:
    final = _final_rows(season_rows)
    metric_names = (
        "nll",
        "crps",
        "expected_rank_mae",
        "median_rank_mae",
        "interval_80_coverage",
        "interval_80_width",
        "top5_brier",
        "top10_brier",
        "top25_brier",
        "mean_entropy",
        "mean_max_probability",
    )
    grouped: dict[tuple[int, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in final:
        grouped[(int(row["season"]), str(row["prior_family"]))][str(row["candidate"])] = row
    output: list[dict[str, object]] = []
    for (season, family), group in sorted(grouped.items()):
        if baseline not in group:
            continue
        base = group[baseline]
        for candidate in candidates:
            if candidate == baseline or candidate not in group:
                continue
            row: dict[str, object] = {
                "season": season,
                "prior_family": family,
                "candidate": candidate,
                "baseline": baseline,
            }
            for metric in metric_names:
                row[f"delta_{metric}"] = float(group[candidate][metric]) - float(base[metric])
            output.append(row)
    return output


def _promotion_assessment(
    season_rows: Sequence[Mapping[str, object]],
    future_rows: Sequence[Mapping[str, object]],
    rolling_rows: Sequence[Mapping[str, object]],
    selected_candidate: str,
) -> dict[str, object]:
    final = _final_rows(season_rows)
    baseline = "v1"
    aggregate = {
        "nll_gain": _metric_mean(final, baseline, "nll") - _metric_mean(final, selected_candidate, "nll"),
        "crps_delta": _metric_mean(final, selected_candidate, "crps") - _metric_mean(final, baseline, "crps"),
        "coverage_delta": _metric_mean(final, selected_candidate, "interval_80_coverage") - _metric_mean(final, baseline, "interval_80_coverage"),
        "interval_width_delta": _metric_mean(final, selected_candidate, "interval_80_width") - _metric_mean(final, baseline, "interval_80_width"),
        "entropy_delta": _metric_mean(final, selected_candidate, "mean_entropy") - _metric_mean(final, baseline, "mean_entropy"),
    }
    by_season: dict[int, dict[str, float]] = {}
    for season in FINAL_YEARS:
        current = [row for row in final if int(row["season"]) == season]
        by_season[season] = {
            "nll_delta": _metric_mean(current, selected_candidate, "nll") - _metric_mean(current, baseline, "nll"),
            "crps_delta": _metric_mean(current, selected_candidate, "crps") - _metric_mean(current, baseline, "crps"),
            "coverage_delta": _metric_mean(current, selected_candidate, "interval_80_coverage") - _metric_mean(current, baseline, "interval_80_coverage"),
        }
    # The final cutoff has no games after it by construction.  Future-game
    # validation therefore uses every non-empty standard cutoff, with the
    # same candidate/game keys for the selected model and V1.
    future_scored = [
        row for row in future_rows if row.get("future_margin_mae") is not None
    ]
    selected_future = [row for row in future_scored if row["candidate"] == selected_candidate]
    baseline_future = [row for row in future_scored if row["candidate"] == baseline]
    future = {
        "margin_mae_delta": _metric_mean(selected_future, selected_candidate, "future_margin_mae") - _metric_mean(baseline_future, baseline, "future_margin_mae"),
        "margin_nll_delta": _metric_mean(selected_future, selected_candidate, "future_margin_nll") - _metric_mean(baseline_future, baseline, "future_margin_nll"),
        "win_brier_delta": _metric_mean(selected_future, selected_candidate, "future_win_brier") - _metric_mean(baseline_future, baseline, "future_win_brier"),
    }
    rolling_final = defaultdict(dict)
    for row in _final_rows(rolling_rows):
        rolling_final[int(row["target_season"])][str(row["candidate"])] = row
    rolling_deltas = [
        float(values[selected_candidate]["nll"]) - float(values[baseline]["nll"])
        for values in rolling_final.values()
        if selected_candidate in values and baseline in values
    ]
    checks = {
        "aggregate_nll": aggregate["nll_gain"] >= float(PROMOTION_CRITERIA["aggregate_nll_improvement_nats_per_team"]),
        "aggregate_crps": aggregate["crps_delta"] <= float(PROMOTION_CRITERIA["maximum_aggregate_crps_degradation"]),
        "aggregate_coverage": aggregate["coverage_delta"] >= -float(PROMOTION_CRITERIA["maximum_aggregate_80pct_coverage_drop"]),
        "most_evaluation_seasons": sum(value["nll_delta"] < 0 for value in by_season.values()) >= int(PROMOTION_CRITERIA["minimum_evaluation_seasons_with_nll_improvement"]),
        "no_catastrophic_nll_season": max((value["nll_delta"] for value in by_season.values()), default=float("nan")) <= float(PROMOTION_CRITERIA["maximum_single_season_nll_degradation"]),
        "no_catastrophic_crps_season": max((value["crps_delta"] for value in by_season.values()), default=float("nan")) <= float(PROMOTION_CRITERIA["maximum_single_season_crps_degradation"]),
        "future_margin_mae_direction": future["margin_mae_delta"] <= float(PROMOTION_CRITERIA["maximum_future_margin_mae_degradation_points"]),
        "future_margin_nll_direction": future["margin_nll_delta"] <= float(PROMOTION_CRITERIA["maximum_future_margin_nll_degradation_nats"]),
        "rolling_robustness": sum(value < 0 for value in rolling_deltas) >= int(PROMOTION_CRITERIA["minimum_rolling_seasons_with_nll_improvement"]),
    }
    passed = all(checks.values())
    return {
        "selected_candidate": selected_candidate,
        "promotion_thresholds_met": passed,
        "recommendation_code": selected_candidate if passed else "not_promoted",
        "recommendation": (
            f"{selected_candidate.upper()} primitive decomposition clears the predeclared promotion gates; freeze only as research Likelihood V2 pending independent confirmation."
            if passed
            else "Primitive evidence is not strong enough to promote a Likelihood V2 candidate on this leakage-safe historical evaluation."
        ),
        "aggregate_final_deltas_selected_minus_v1": aggregate,
        "per_season_final_deltas_selected_minus_v1": by_season,
        "future_final_deltas_selected_minus_v1": future,
        "future_comparison_scope": "all non-empty standard cutoffs; final cutoff has no future games by construction",
        "rolling_nll_deltas_selected_minus_v1": rolling_deltas,
        "checks": checks,
    }


def frozen_model_spec() -> dict[str, object]:
    """Return the predeclared specification recorded before evaluation."""

    return {
        "experiment": "primitive_box_score_likelihood",
        "research_question": "Does retaining yards and plays separately recover useful latent FBS quality information that YPP collapses?",
        "factorization": {
            "v1": "p(margin | ranks, site)",
            "a": "p(margin | ranks, site) * p(yards_diff | plays_diff, plays_total, yards_total, margin, ranks, site, pairing)",
            "b": "p(margin | ranks, site) * p(plays_diff | plays_total, margin, ranks, site, pairing) * p(yards_diff | plays_diff, plays_total, yards_total, margin, ranks, site, pairing)",
            "c": "B with separate interceptions_thrown_diff/total and fumbles_lost_diff/total as context covariates in both conditional components; no p(INT,FL | ranks)",
        },
        "primitive_representation": {
            "selected_fields": list(PRIMITIVE_FIELDS),
            "team_pair_coordinates": ["yards_diff", "yards_total", "plays_diff", "plays_total"],
            "turnover_coordinates": [
                "interceptions_thrown_diff",
                "interceptions_thrown_total",
                "fumbles_lost_diff",
                "fumbles_lost_total",
            ],
            "orientation": "same-subdivision home-minus-away; cross-subdivision FBS-minus-FCS regardless of listing order",
            "lossless_claim": "diff plus total is an invertible representation of each selected pair of team values",
        },
        "response_transforms": {
            "yards_diff": "asinh(yards_diff / 100.0)",
            "plays_diff": "asinh(plays_diff / 10.0)",
            "yards_total_context": "asinh(yards_total / 200.0)",
            "plays_total_context": "asinh(plays_total / 80.0)",
            "turnover_diff_and_total_context": "asinh(value / 1.0), separately for INT thrown and fumbles lost",
            "no_winsorization": True,
        },
        "rank_basis": {
            "coordinates": "V1 percentile coordinates: same-subdivision rank surface uses odd d, d*mean, d*mean^2, d*abs(d), and d*hinges; cross-subdivision uses FBS/FCS percentile surface and site indicators",
            "rank_signal_null": "rank coordinates are set to zero in the reserved V1 rank columns; context and margin/site columns remain",
            "rank_support": "the same rank supports supplied by historical_modeling_games.csv and frozen final PMF supports for scoring",
        },
        "conditioning_variables": {
            "all_components": ["oriented margin", "pairing", "site"],
            "plays_component": ["overall plays_total context"],
            "yards_component": ["plays_diff", "plays_total", "yards_total context"],
            "candidate_c_only": ["interceptions_thrown_diff", "interceptions_thrown_total", "fumbles_lost_diff", "fumbles_lost_total"],
        },
        "distribution": {
            "family": "Student-t",
            "fit": "equal-total-weight rank-pair pseudo-observations per game; robust IRLS",
            "df_grid": list(DF_GRID),
            "scale": "one fitted robust residual scale per conditional component; no outcome-dependent retuning",
            "iterations": 8,
        },
        "temporal_design": {
            "training": "2004-2017",
            "development": "2018-2021",
            "evaluation": "2022-2025 leakage-safe historical evaluation/prospective-style validation; not independent confirmation",
            "rolling": "target seasons 2008-2025 fit primitive components only on earlier seasons; no 2026 outcomes",
        },
        "pairing_support": {
            "supported": sorted(SUPPORTED_PRIMITIVE_PAIRINGS),
            "unsupported": ["fcs-fcs"],
            "fit": "primitive parameters are fit only on FBS-FBS and FBS-FCS rows",
            "inference": "FCS-FCS and unsupported/missing primitive evidence use the exact V1 margin factor for that game",
        },
        "missing_data": {
            "rule": "missing is not zero; each factor requires the complete fields it conditions on",
            "a": "missing yards or plays -> V1",
            "b": "missing plays disables play factor; missing yards disables yards factor; if no usable factor remains -> V1",
            "c": "missing either INT-thrown or fumbles-lost value for either team falls back exactly to B for that game",
        },
        "candidate_selection": DEVELOPMENT_SELECTION_RULE,
        "promotion_thresholds": PROMOTION_CRITERIA,
        "ypp_comparator": {
            "name": "ypp_supported",
            "source": "PR #17 final supported-pairing YPP candidate",
            "pairings": sorted(SUPPORTED_YPP_PAIRINGS),
            "student_t_df": YPP_FIXED_DF,
            "retuning": "none; fit specification and df are fixed to the merged PR #17 result",
        },
        "scope_exclusions": [
            "passing/rushing splits",
            "sacks",
            "total fumbles",
            "first downs",
            "penalties",
            "PPA/EPA",
            "success rate",
            "explosiveness",
            "play-by-play",
            "direct turnover rating likelihood",
            "production Likelihood V2 and website/publication changes",
        ],
    }


def _report_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_plots(
    season_rows: Sequence[Mapping[str, object]],
    play_signal: Sequence[Mapping[str, object]],
    selected_candidate: str,
) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    final = _final_rows(season_rows)
    labels = ["a", "b", "c", "ypp_supported"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, family in enumerate(("context", "history")):
        baseline = _metric_mean(
            [row for row in final if row["prior_family"] == family],
            "v1",
            "nll",
        )
        values = []
        for label in labels:
            metric = _metric_mean(
                [row for row in final if row["prior_family"] == family],
                label,
                "nll",
            )
            values.append(baseline - metric)
        axis.plot(labels, values, marker="o", label=family)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("NLL improvement over V1 (nats/team)")
    axis.set_title(f"Final-rank NLL deltas; selected={selected_candidate.upper()}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "rank_nll_deltas.png", dpi=120, metadata={"Date": None})
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for pairing in sorted({str(row["pairing"]) for row in play_signal}):
        values = [
            row
            for row in play_signal
            if row["pairing"] == pairing and row["margin_band"] == "(-14,0]"
        ]
        values.sort(key=lambda row: int(row["season"]))
        axis.plot(
            [int(row["season"]) for row in values],
            [float(row["quality_slope"]) if row["quality_slope"] is not None else np.nan for row in values],
            marker="o",
            label=pairing,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Season")
    axis.set_ylabel("Play residual → final-quality slope")
    axis.set_title("Play-count signal in the close-loss margin band")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "play_signal_close_loss.png", dpi=120, metadata={"Date": None})
    plt.close(figure)


def write_report(
    *,
    spec: Mapping[str, object],
    audit_metadata: Mapping[str, object],
    excluded_rows: Sequence[Mapping[str, object]],
    selection_summary: Mapping[str, object],
    selection_comparisons: Sequence[Mapping[str, object]],
    prior_sensitivity: Mapping[str, object],
    selection_rows: Sequence[Mapping[str, object]],
    season_rows: Sequence[Mapping[str, object]],
    aggregate_metrics: Sequence[Mapping[str, object]],
    metric_deltas: Sequence[Mapping[str, object]],
    play_signal: Sequence[Mapping[str, object]],
    play_summary: Mapping[str, object],
    turnover_context: Sequence[Mapping[str, object]],
    disagreement_games: Sequence[Mapping[str, object]],
    rolling_rows: Sequence[Mapping[str, object]],
    future_rows: Sequence[Mapping[str, object]],
    promotion: Mapping[str, object],
    production_unchanged: bool,
) -> None:
    lines = [
        "# Primitive box-score likelihood research",
        "",
        "## Conclusion",
        "",
        f"Selected before the 2022--2025 evaluation: **{str(selection_summary['selected_candidate']).upper()}**. {promotion['recommendation']}",
        "",
        "The central question is whether YPP lost signal by collapsing two primitives into one ratio. This experiment keeps yards and plays as separate oriented differences with totals as context, and compares the preselected candidate against the frozen margin-only V1 and fixed supported-pairing YPP comparator.",
        "",
        "2022--2025 is a leakage-safe historical evaluation/prospective-style validation, not independent confirmation: the hypothesis was motivated by reviewing the earlier YPP study that already examined those seasons.",
        "",
        "## Frozen specification",
        "",
        "The exact machine-readable specification is in `model_spec.json` and was fixed in source before the final evaluation:",
        "",
        "* Factorization: V1 margin × optional `p(plays_diff | margin, ranks, site, plays_total, pairing)` × `p(yards_diff | plays, margin, ranks, site, yards_total, pairing)`.",
        "* Responses: `asinh(yards_diff / 100)`, `asinh(plays_diff / 10)`; totals and turnover diff/total pairs are context covariates, not separate rating factors.",
        "* Rank basis: frozen V1 percentile surface, with same-subdivision odd rank terms and FBS/FCS percentile coordinates; site and fixed margin basis are pairing-specific.",
        "* Distribution: robust Student-t IRLS, equal total weight per game over rank-pair pseudo-observations, df grid `3, 5, 8, 15`; component scales are fit on the declared training sample.",
        "* Pairing: fit and apply primitives only for FBS-FBS and FBS-FCS; FCS-FCS remains in the graph with an exact V1 factor.",
        "* Missingness: missing is never zero; incomplete evidence falls to the lower-level factor and ultimately V1. C falls exactly to B when either separate turnover context is incomplete.",
        "",
        "## Audit source and exclusions",
        "",
        f"The only primitive source was the merged PR #20 audit artifact (`team_game_audit.csv`, SHA-256 `{audit_metadata['source_sha256']}`). No passing/rushing split, sacks, total fumbles, first downs, penalties, or advanced statistics enter the model.",
        f"The joined historical population is the exact historical-modeling game population; primitive evidence is supported only on the declared pairings. {len(excluded_rows)} audited field-level impossible values were excluded from primitive factors and retained as V1 games; every exclusion is in `excluded_rows.csv`.",
        "Extreme-but-not-impossible audit flags were not winsorized or silently repaired.",
        "",
        "## Development selection",
        "",
        "Selection used only 2018--2021. Because frozen H/C prior predictions in this checkout begin in 2022, development selection used an explicit uniform ordinal selection prior; the final evaluation uses identical frozen Context and History priors. The final seasons were not read by the selection function.",
        "",
        "| comparison | NLL gain | CRPS delta | 80% coverage delta | seasons improved | passed |",
        "|:--|--:|--:|--:|--:|:--:|",
    ]
    for row in selection_comparisons:
        lines.append(
            f"| {row['selection_stage']} | {_report_number(row['development_nll_gain_predecessor_minus_newer'])} | {_report_number(row['development_crps_delta_newer_minus_predecessor'])} | {_report_number(row['development_coverage_delta_newer_minus_predecessor'])} | {row['development_seasons_with_nll_improvement']} | {row['passed']} |"
        )
    lines.extend(["", "## Development prior sensitivity", "", "The original uniform choice was explicit: frozen H/C PMFs in this checkout begin in 2022, while the 2018--2021 posterior labels remain available. A uniform ordinal PMF therefore supplied a reproducible, model-neutral starting state without pretending that a later frozen artifact was historically available. This matters because BP combines each team's prior with the V1 and optional primitive likelihood factors; an informative prior can change both the posterior rank weights and the nonlinear message-passing state, so it can change relative candidate scores even when candidate likelihood factors are identical.", "", "For the sensitivity, H 1.1 and C 1.2 were reconstructed separately for each target season using only completed rank distributions through the prior season and preseason-semantic feature fields. No 2022--2025 row or outcome was read. The candidate definitions, fitted component df choices, comparison keys, and gates are unchanged.", "", "| prior family | comparison | NLL gain | CRPS delta | 80% coverage delta | seasons improved | passed |", "|:--|:--|--:|--:|--:|--:|:--|"])
    for family, rows_for_family in prior_sensitivity["comparisons"].items():
        for row in rows_for_family:
            lines.append(f"| {family} | {row['selection_stage']} | {_report_number(row['development_nll_gain_predecessor_minus_newer'])} | {_report_number(row['development_crps_delta_newer_minus_predecessor'])} | {_report_number(row['development_coverage_delta_newer_minus_predecessor'])} | {row['development_seasons_with_nll_improvement']} | {row['passed']} |")
    lines.extend(["", f"Informative-prior selection paths: {json.dumps(prior_sensitivity['selection_paths'], sort_keys=True)}. Stability classification: **{prior_sensitivity['stability']}**. This is a sensitivity of the pre-2022 selection question, not a reselection using the final evaluation."])
    lines.extend(
        [
            "",
            f"The selected reporting candidate is **{str(selection_summary['selected_candidate']).upper()}**; selection path `{', '.join(selection_summary['selection_path']) or 'none'}`. Candidate df selection details are in `candidate_selection.csv`.",
            "",
            "## 2022--2025 final-rank evaluation",
            "",
            "All candidates use the same FBS scoring teams, frozen Context/History priors, final-rank target PMFs, full game populations, cutoffs, and common comparison keys.",
            "",
            "| prior | candidate | NLL | CRPS | expected MAE | median MAE | 80% coverage | 80% width | top5 Brier | top10 Brier | top25 Brier | entropy |",
            "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in aggregate_metrics:
        lines.append(
            f"| {row['prior_family']} | {row['candidate']} | {_report_number(row['nll'])} | {_report_number(row['crps'])} | {_report_number(row['expected_rank_mae'])} | {_report_number(row['median_rank_mae'])} | {_report_number(row['interval_80_coverage'])} | {_report_number(row['interval_80_width'], 1)} | {_report_number(row['top5_brier'])} | {_report_number(row['top10_brier'])} | {_report_number(row['top25_brier'])} | {_report_number(row['mean_entropy'])} |"
        )
    lines.extend(["", "Per-season selected-candidate deltas versus V1:", "", "| season | prior | ΔNLL | ΔCRPS | Δexpected MAE | Δmedian MAE | Δ80% coverage | Δ80% width |", "|--:|:--|--:|--:|--:|--:|--:|--:|"])
    for row in metric_deltas:
        if row["candidate"] == selection_summary["selected_candidate"]:
            lines.append(
                f"| {row['season']} | {row['prior_family']} | {_report_number(row['delta_nll'])} | {_report_number(row['delta_crps'])} | {_report_number(row['delta_expected_rank_mae'])} | {_report_number(row['delta_median_rank_mae'])} | {_report_number(row['delta_interval_80_coverage'])} | {_report_number(row['delta_interval_80_width'], 1)} |"
            )
    lines.extend(
        [
            "",
            "The fixed `ypp_supported` rows in the same table are the PR #17 comparator; they were not part of candidate selection and were not retuned.",
            "",
            "## Direct answer: do plays contain quality signal?",
            "",
            f"The dedicated `play_signal.csv` analysis has {play_summary['rows']} supported-pairing rows and {play_summary['positive_quality_slope_rows']} positive season/pairing/margin cells out of {play_summary['quality_slope_rows']} with estimable slopes. It conditions on oriented margin, site, opponent-quality rank coordinates, and overall `plays_total`; positive slopes mean more plays than the opponent align with the better final-rank side in that descriptive residual analysis. The season/pairing/margin-band table is the evidence, not a hidden aggregate candidate result.",
            "",
            "| period | season | pairing | margin band | n | mean plays diff | quality slope | quality Spearman | next-margin slope |",
            "|:--|--:|:--|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for row in play_signal:
        lines.append(
            f"| {row['period']} | {row['season']} | {row['pairing']} | {row['margin_band']} | {row['n_games']} | {_report_number(row['mean_plays_diff'], 1)} | {_report_number(row['quality_slope'])} | {_report_number(row['quality_spearman'])} | {_report_number(row['next_margin_slope'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation is deliberately descriptive: the presence of a positive slope is evidence of possible play-count signal beyond the scoreboard, while unstable or near-zero slopes do not justify a direct plays rating factor.",
            "",
            "## Turnover context without turnover rating",
            "",
            "Candidate C retains interceptions thrown and fumbles lost separately and only changes the conditional yards/plays interpretation. It never multiplies `p(INT, FL | ranks)` and missing turnover context falls exactly to B.",
            "",
            "| period | season | pairing | margin band | turnover pattern | n | mean INT diff | mean fumbles-lost diff | C−B conditional log density |",
            "|:--|--:|:--|:--|:--|--:|--:|--:|--:|",
        ]
    )
    for row in turnover_context:
        lines.append(
            f"| {row['period']} | {row['season']} | {row['pairing']} | {row['margin_band']} | {row['turnover_pattern']} | {row['n_games']} | {_report_number(row['mean_interceptions_diff'], 2)} | {_report_number(row['mean_fumbles_lost_diff'], 2)} | {_report_number(row['mean_log_density_c_minus_b_at_final_rank'])} |"
        )
    lines.extend(["", "Representative real corpus disagreement games:", "", "| type | season | game | matchup | score | margin | yards diff | plays diff | INT A/B | fumbles lost A/B | C−B context log density |", "|:--|--:|--:|:--|:--|--:|--:|--:|:--|:--|--:|"])
    for row in disagreement_games:
        lines.append(
            f"| {row['disagreement_type']} | {row['season']} | {row['game_id']} | {row['home_team']}–{row['away_team']} | {row['home_score']}–{row['away_score']} | {_report_number(row['oriented_margin'], 0)} | {_report_number(row['oriented_yards_diff'], 0)} | {_report_number(row['oriented_plays_diff'], 0)} | {row['interceptions_thrown_a']}/{row['interceptions_thrown_b']} | {row['fumbles_lost_a']}/{row['fumbles_lost_b']} | {_report_number(row['c_minus_b_context_log_density'])} |"
        )
    lines.extend(
        [
            "",
            "These examples are observations of conditional disagreement, not claims that turnovers are luck or skill and not direct turnover ratings.",
            "",
            "## Rolling robustness",
            "",
            "The original rolling panel used a uniform ordinal selection prior. It is therefore not directly comparable to the final H/C evaluation. The corrected History-prior rolling rows below use reconstructed H 1.1 PMFs from information through each target's prior season; no target-season outcomes enter prior construction.",
            "",
            "| target season | fit through | candidate | NLL | CRPS | 80% coverage |",
            "|--:|:--|:--|--:|--:|--:|",
        ]
    )
    for row in rolling_rows:
        lines.append(
            f"| {row['target_season']} | {row['fit_seasons']} | {row['candidate']} | {_report_number(row['nll'])} | {_report_number(row['crps'])} | {_report_number(row['interval_80_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Calibration and future-game validation",
            "",
            f"Promotion checks: `{json.dumps(promotion['checks'], sort_keys=True)}`. Selected-minus-V1 aggregate deltas are `{json.dumps(promotion['aggregate_final_deltas_selected_minus_v1'], sort_keys=True)}`; final posterior width, 80% coverage, NLL, entropy, and concentration are in `calibration.csv` and `candidate_metrics.csv`.",
            "",
            "Future-game scoring uses the same frozen V1 margin likelihood for every posterior; only posterior state estimation differs. `future_game_metrics.csv` reports future margin MAE, margin NLL, win Brier, and next-game counterparts on identical keys.",
            "",
            "## Data integrity and artifacts",
            "",
            f"Production artifacts unchanged during the build: `{production_unchanged}`. No production Likelihood V2, Posterior V1, H/C, Performance, weekly publication, acquisition, or website files were edited.",
            "",
            "Artifacts: `report.md`, `summary.json`, `model_spec.json`, `candidate_selection.csv`, `candidate_metrics.csv`, `season_metrics.csv`, `future_game_metrics.csv`, `play_signal.csv`, `turnover_context.csv`, `disagreement_games.csv`, `calibration.csv`, `rolling_metrics.csv`, `excluded_rows.csv`, and deterministic PNGs under `plots/`.",
            "",
            "The final evaluation is intentionally historical and leakage-safe, but it is not independent confirmation of the hypothesis because 2022--2025 motivated the investigation through the earlier YPP work.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _production_paths() -> list[Path]:
    return [
        ROOT / "src/gippyrank/posterior",
        ROOT / "src/gippyrank/preseason.py",
        ROOT / "src/gippyrank/performance_v1.py",
        ROOT / "src/gippyrank/performance_snapshot.py",
        ROOT / "src/gippyrank/weekly_update.py",
        ROOT / "data/processed/posterior",
        ROOT / "data/processed/posterior_backtest",
        ROOT / "data/processed/preseason",
        ROOT / "data/processed/performance_v1",
        ROOT / "data/processed/weekly_updates",
        ROOT / "site",
    ]


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--prior-preflight", action="store_true")
    args = parser.parse_args()
    OUT = args.output
    required = [HISTORICAL_ROWS_PATH, RANK_DISTRIBUTIONS_PATH, AUDIT_ROWS_PATH, LIKELIHOOD_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required frozen research inputs are missing: " + ", ".join(missing))

    if args.prior_preflight:
        started = datetime.now(UTC)
        if not RANK_DISTRIBUTIONS_PATH.exists():
            raise FileNotFoundError(str(RANK_DISTRIBUTIONS_PATH))
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_preseason_context_prior_v1_2 as context_prior
        import build_preseason_prior as history_prior
        index, tenures = context_prior.feature_index(), context_prior.cached_tenures()
        rows_out: list[dict[str, object]] = []
        requests = [(season, RECONSTRUCTED_PRIOR_FAMILIES) for season in DEVELOPMENT_YEARS]
        requests.extend((season, ("history_reconstructed",)) for season in ROLLING_YEARS)
        for target, families in requests:
            _historical, cold, _ = history_prior.load_rows(max_season=target - 1)
            inference = context_prior.inference_rows(target, target - 1, index, tenures)
            reasons, promotion, generic = cold_start_requirements(inference, cold, target - 1)
            for family in families:
                unsupported_promotion = reasons.get("fcs_to_fbs_transition", 0) > 0 and not promotion
                unsupported_generic = reasons.get("no_prior_rank_distribution", 0) > 0 and not generic
                unsupported = unsupported_promotion or unsupported_generic
                reason = None
                if unsupported_promotion:
                    reason = "required FCS-to-FBS promotion fallback has zero pre-target training rows"
                elif unsupported_generic:
                    reason = "required generic FBS cold-start fallback has zero pre-target training rows"
                rows_out.append({"target_season": target, "family": family, "inference_team_count": len(inference), "generic_cold_start_count": reasons.get("no_prior_rank_distribution", 0), "promotion_cold_start_count": reasons.get("fcs_to_fbs_transition", 0), "generic_training_row_count": len(generic) if reasons.get("no_prior_rank_distribution", 0) else None, "promotion_training_row_count": len(promotion) if reasons.get("fcs_to_fbs_transition", 0) else None, "supported": not unsupported, "reason": reason, "target_outcomes_used": False})
        print(json.dumps({"runtime_seconds": (datetime.now(UTC) - started).total_seconds(), "rows": rows_out}, indent=2, sort_keys=True))
        return

    production_before: dict[str, str] = {}
    for path in _production_paths():
        production_before.update(_tree_hash(path))

    print("loading audited primitive source")
    historical = load_historical_rows()
    audit, audit_metadata = load_audit_evidence()
    enriched, excluded_rows = attach_primitive_evidence(historical, audit)
    targets = load_rank_targets()
    data = build_primitive_data(enriched)
    likelihood = load_likelihood(LIKELIHOOD_PATH)
    spec = frozen_model_spec()
    _write_json(OUT / "model_spec.json", spec)

    print("selecting conditional models from training/development only")
    development_models, final_models, selection_rows = select_primitive_models(data)
    development_primitive_models = {
        candidate: development_models[candidate] for candidate in PRIMITIVE_CANDIDATES
    }
    development_candidate_rows, _development_season_rows, _development_calibration, _development_meta = run_posterior_panel(
        enriched,
        targets,
        likelihood,
        development_primitive_models,
        None,
        seasons=DEVELOPMENT_YEARS,
        candidates=SELECTION_CANDIDATES,
        prior_families=("uniform",),
        frozen_priors=False,
        include_future=False,
        final_only=True,
    )
    selected_candidate, selection_comparisons, selection_summary = select_candidate_from_development(
        development_candidate_rows
    )
    print("reconstructing leakage-safe 2018-2021 H/C priors")
    development_reconstructed_teams, prior_audit = reconstruct_historical_priors(
        targets, enriched, DEVELOPMENT_YEARS
    )
    sensitivity_rows: dict[str, list[dict[str, object]]] = {}
    sensitivity_paths: dict[str, str] = {}
    for family in ("context_reconstructed", "history_reconstructed"):
        sensitivity_candidate_rows, _season_rows, _calibration, _metadata = run_posterior_panel(
            enriched,
            targets,
            likelihood,
            development_primitive_models,
            None,
            seasons=DEVELOPMENT_YEARS,
            candidates=SELECTION_CANDIDATES,
            prior_families=(family,),
            frozen_priors=False,
            include_future=False,
            final_only=True,
            reconstructed_teams=development_reconstructed_teams,
        )
        path, comparisons, _summary = select_candidate_from_development(
            sensitivity_candidate_rows
        )
        sensitivity_rows[family] = comparisons
        sensitivity_paths[family] = path
    original_path = selection_summary["selected_candidate"]
    stability = (
        "Stable"
        if all(path == original_path for path in sensitivity_paths.values())
        else "Sensitive but inconclusive"
    )
    prior_sensitivity = {
        "method": "prospective reconstruction of H 1.1 and C 1.2 for each 2018-2021 target season",
        "comparisons": sensitivity_rows,
        "selection_paths": sensitivity_paths,
        "original_uniform_selection": original_path,
        "stability": stability,
        "prior_audit": prior_audit,
        "final_seasons_used_for_reconstruction": [],
        "candidate_definitions_unchanged": True,
        "thresholds_unchanged": True,
    }
    for row in selection_rows:
        if row.get("selection_row_type") == "selected_summary":
            row["development_posterior_selection_candidate"] = selected_candidate
    spec = {
        **spec,
        "selected_after_development": selection_summary,
        "selected_component_df": {
            candidate: next(
                float(row["student_t_df"])
                for row in selection_rows
                if row.get("candidate") == candidate and row.get("selected") and row.get("selection_row_type") != "selected_summary"
            )
            for candidate in PRIMITIVE_CANDIDATES
        },
    }
    _write_json(OUT / "model_spec.json", spec)

    print("running matched 2022-2025 posterior and future panels")
    final_primitive_models = {candidate: final_models[candidate] for candidate in PRIMITIVE_CANDIDATES}
    ypp_data = build_ypp_data(enriched, allowed_pairings=SUPPORTED_YPP_PAIRINGS)
    ypp_mask = np.isin(ypp_data.season, (*TRAIN_YEARS, *DEVELOPMENT_YEARS))
    ypp_model = fit_ypp_model(
        ypp_data,
        ypp_mask,
        rank_signal=True,
        include_margin=True,
        degrees_of_freedom=YPP_FIXED_DF,
        allowed_pairings=SUPPORTED_YPP_PAIRINGS,
    )
    final_candidate_rows, season_rows, calibration_rows, evaluation_metadata = run_posterior_panel(
        enriched,
        targets,
        likelihood,
        final_primitive_models,
        ypp_model,
        seasons=FINAL_YEARS,
        candidates=FINAL_CANDIDATES,
        prior_families=("context", "history"),
        frozen_priors=True,
        include_future=True,
        future_candidates=FINAL_CANDIDATES,
    )

    print("building rolling, play-signal, and turnover diagnostics")
    selected_df = float(
        next(
            row["student_t_df"]
            for row in selection_rows
            if row.get("candidate") == selected_candidate
            and row.get("selected")
            and row.get("selection_row_type") != "selected_summary"
        )
    )
    print("rerunning rolling panel with reconstructed History priors")
    rolling_reconstructed_teams, rolling_prior_audit = reconstruct_historical_priors(
        targets, enriched, ROLLING_YEARS, families=("history_reconstructed",)
    )
    rolling_rows = run_rolling_robustness(
        enriched,
        targets,
        data,
        likelihood,
        selected_candidate,
        selected_df,
        reconstructed_teams=rolling_reconstructed_teams,
        prior_family="history_reconstructed",
    )
    prior_sensitivity["rolling_prior_audit"] = rolling_prior_audit
    play_signal, play_summary = build_play_signal(
        enriched,
        data,
        targets,
        final_models["b"]["b_plays"],
    )
    turnover_context = build_turnover_context_table(
        enriched,
        targets,
        final_models["b"] | final_models["c"],
    )
    disagreement_games = build_disagreement_games(
        enriched,
        targets,
        final_models["b"] | final_models["c"],
    )
    metric_deltas = build_metric_deltas(season_rows, FINAL_CANDIDATES)
    aggregate_metrics = _aggregate_final_metrics(season_rows, FINAL_CANDIDATES)
    future_rows = evaluation_metadata["future_game_metrics"]
    promotion = _promotion_assessment(
        season_rows,
        future_rows,
        rolling_rows,
        selected_candidate,
    )
    production_after: dict[str, str] = {}
    for path in _production_paths():
        production_after.update(_tree_hash(path))
    production_unchanged = production_before == production_after

    summary = {
        **spec,
        "audit": audit_metadata,
        "population": {
            "historical_rows": len(historical),
            "enriched_rows": len(enriched),
            "historical_games": len({str(row["game_id"]) for row in historical}),
            "primitive_pseudo_observations": len(data),
            "excluded_field_rows": len(excluded_rows),
        },
        "model_selection": {
            "candidate_rows": selection_rows,
            "development_comparisons": selection_comparisons,
            "selection": selection_summary,
            "prior_sensitivity": prior_sensitivity,
        },
        "evaluation": {
            "final_candidate_metrics": aggregate_metrics,
            "final_season_rows": season_rows,
            "metric_deltas": metric_deltas,
            "future_key_audit": evaluation_metadata["future_key_audit"],
            "comparison_key_audit": evaluation_metadata["comparison_key_audit"],
            "ypp_fixed_model": {
                "df": YPP_FIXED_DF,
                "scale": ypp_model["scale"],
                "fit_game_count": ypp_model["fit_game_count"],
                "fit_pseudo_observation_count": ypp_model["fit_pseudo_observation_count"],
                "beta_sha256": hashlib.sha256(np.asarray(ypp_model["beta"], dtype=float).tobytes()).hexdigest(),
            },
        },
        "play_signal": play_summary,
        "promotion_assessment": promotion,
        "rolling": {
            "target_seasons": list(ROLLING_YEARS),
            "rows": len(rolling_rows),
            "no_2026_outcomes": True,
        },
        "frozen_input_sha256": {
            "historical_modeling_games.csv": _sha256(HISTORICAL_ROWS_PATH),
            "team_season_rank_distributions.csv": _sha256(RANK_DISTRIBUTIONS_PATH),
            "box_score_audit/team_game_audit.csv": _sha256(AUDIT_ROWS_PATH),
            "historical_likelihood_v1.json": _sha256(LIKELIHOOD_PATH),
        },
        "frozen_production_integrity": {
            "hashes_before": production_before,
            "hashes_after": production_after,
            "unchanged": production_unchanged,
            "scope": "production posterior, priors/H-C, performance, weekly updates, and website trees",
        },
        "artifact_inventory": {
            "report.md": "human-readable research findings",
            "summary.json": "machine-readable configuration, metrics, selection, and integrity",
            "model_spec.json": "exact frozen model specification and development selection record",
            "candidate_selection.csv": "df grid and development component-density selection",
            "candidate_metrics.csv": "matched cutoff V1/A/B/C/YPP posterior metrics",
            "season_metrics.csv": "final-cutoff per-season metrics",
            "future_game_metrics.csv": "frozen-V1 future margin validation",
            "play_signal.csv": "season/pairing/margin-band descriptive play signal",
            "turnover_context.csv": "separate INT/fumbles-lost context summaries",
            "disagreement_games.csv": "real score/primitive disagreement examples",
            "calibration.csv": "coverage, interval width, entropy, concentration",
            "rolling_metrics.csv": "earlier-season-only robustness panel",
            "excluded_rows.csv": "all audited impossible primitive fields excluded from primitive factors",
            "plots/": "deterministic PNG diagnostics",
        },
    }
    _write_json(OUT / "summary.json", summary)
    _write_csv(OUT / "excluded_rows.csv", excluded_rows)
    _write_csv(OUT / "candidate_selection.csv", selection_rows)
    _write_csv(OUT / "candidate_metrics.csv", final_candidate_rows)
    _write_csv(OUT / "season_metrics.csv", season_rows)
    _write_csv(OUT / "future_game_metrics.csv", future_rows)
    _write_csv(OUT / "play_signal.csv", play_signal)
    _write_csv(OUT / "turnover_context.csv", turnover_context)
    _write_csv(OUT / "disagreement_games.csv", disagreement_games)
    _write_csv(OUT / "calibration.csv", calibration_rows)
    _write_csv(OUT / "rolling_metrics.csv", rolling_rows)
    _write_csv(OUT / "metric_deltas.csv", metric_deltas)
    if not args.skip_plots:
        write_plots(season_rows, play_signal, selected_candidate)
    write_report(
        spec=spec,
        audit_metadata=audit_metadata,
        excluded_rows=excluded_rows,
        selection_summary=selection_summary,
        selection_comparisons=selection_comparisons,
        selection_rows=selection_rows,
        season_rows=season_rows,
        aggregate_metrics=aggregate_metrics,
        metric_deltas=metric_deltas,
        play_signal=play_signal,
        play_summary=play_summary,
        turnover_context=turnover_context,
        disagreement_games=disagreement_games,
        rolling_rows=rolling_rows,
        future_rows=future_rows,
        promotion=promotion,
        prior_sensitivity=prior_sensitivity,
        production_unchanged=production_unchanged,
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "selected_candidate": selected_candidate,
                "promotion": promotion["recommendation_code"],
                "candidate_rows": len(final_candidate_rows),
                "future_rows": len(future_rows),
                "production_unchanged": production_unchanged,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
