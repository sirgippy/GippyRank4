"""Issue 212 fixed replay of generic FBS–FBS likelihood recalibration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.posterior.predictive import ScheduledGame, predictive_components
from gippyrank.research.issue_201_prebowl_forecast import (
    PRIMARY_POPULATION,
    SECONDARY_POPULATION,
    _build_posterior_teams,
    _common_target_teams,
    _evidence_record,
    _fcs_names,
    _game_id_key,
    _load_arm_likelihood,
    _load_issue200_inputs,
    _posterior_diagnostics,
    _prediction_row,
    _predictive_summary_from_components,
    _read_target_snapshot_dates,
    _scoreable_game_rows,
    _target_window_rows,
    filter_evidence_rows,
    game_from_outcome,
    load_pre_bowl_targets,
    mixture_margin_crps,
    mixture_margin_nll,
    score_rank_pmf,
    sha256_file,
    sha256_json,
    summarize_forecast_scores,
    verify_score_blind_plan,
)
from gippyrank.research.matched_pipeline_replay import replay_pipeline_bundle

BASELINE_ARM = "Context_existing"
CANDIDATE_ARM = "Context_generic_recalibrated"
EVALUATION_SEASONS = (2023, 2024, 2025)
EXPECTED_FIT_RECORD_SHA256 = (
    "a6bb5229c6991c5dd1661bc138f6c322b071c7591c0d7b4c75315b3966e59530"
)
EXPECTED_BASE_LIKELIHOOD_SHA256 = (
    "b64d237d4abf3121355a28505f805252cf32062b3fce66db6ca620ef35887952"
)
IDENTITY_TOLERANCE = 1e-8
PRIMARY_GAIN_THRESHOLD = 0.05
GUARDRAIL_POOLED_CRPS = 0.10
GUARDRAIL_SEASON_CRPS = 0.25
GUARDRAIL_POOLED_COVERAGE_ERROR = 0.03
GUARDRAIL_SEASON_COVERAGE_ERROR = 0.05
RANK_METRICS = (
    "nll",
    "crps",
    "expected_rank_error",
    "interval_80_width",
    "interval_80_target_mass",
)


@dataclass(frozen=True)
class CalibrationParameters:
    season: int
    multiplier: float
    home_adjustment: float
    log_scale_correction: float
    fit_record: dict[str, object]

    @property
    def scale_multiplier(self) -> float:
        return math.exp(self.log_scale_correction)


class RecalibratedLikelihood:
    """Research-only adapter that adjusts only FBS–FBS factors."""

    def __init__(self, base: LikelihoodV1, parameters: CalibrationParameters) -> None:
        self.base = base
        self.parameters = parameters
        self.factor_cache_key = sha256_json(
            {
                "adapter": "issue212_fbs_fbs_rank_site_calibration_v1",
                "base_beta_sha256": hashlib.sha256(base.beta.tobytes()).hexdigest(),
                "base_scale": base.scale,
                "degrees_of_freedom": base.degrees_of_freedom,
                "season": parameters.season,
                "multiplier": parameters.multiplier,
                "home_adjustment": parameters.home_adjustment,
                "log_scale_correction": parameters.log_scale_correction,
            }
        )

    def scale_for(self, home: Team, away: Team) -> float:
        if home.subdivision == away.subdivision == "fbs":
            return self.base.scale * self.parameters.scale_multiplier
        return self.base.scale

    def locations_for(self, locations: np.ndarray, game: Game) -> np.ndarray:
        if game.home_subdivision == game.away_subdivision == "fbs":
            return (
                self.parameters.multiplier * locations
                + self.parameters.home_adjustment * (1.0 - float(game.neutral_site))
            )
        return locations

    def factor(self, game: Game, home: Team, away: Team) -> np.ndarray:
        if home.subdivision != "fbs" or away.subdivision != "fbs":
            return game_factor(game, home, away, self.base)

        if (
            self.parameters.multiplier == 1.0
            and self.parameters.home_adjustment == 0.0
            and self.parameters.log_scale_correction == 0.0
        ):
            return game_factor(game, home, away, self.base)

        base_locations, margin = game_margin_parameters(game, home, away, self.base)
        locations = self.locations_for(base_locations, game)
        scale = self.scale_for(home, away)
        log_density = student_t.logpdf(
            (margin - locations) / scale, df=self.base.degrees_of_freedom
        ) - math.log(scale)
        density = np.exp(log_density - float(np.max(log_density)))
        return density


def calibration_parameters_from_record(
    record: dict[str, object], season: int
) -> CalibrationParameters:
    if record.get("target_season") != season:
        raise ValueError(f"#195 fit record does not target {season}")
    expected_training = list(range(2022, season))
    if record.get("training_seasons") != expected_training:
        raise ValueError(f"{season}: #195 fit training seasons differ from protocol")
    if record.get("role") != "held_out_evaluation" or not record.get(
        "target_season_excluded_from_fit_and_regularization_selection"
    ):
        raise ValueError(f"{season}: #195 calibration fit is not forward-only")
    baseline = record.get("frozen_baseline")
    if not isinstance(baseline, dict) or baseline.get("refit") is not False:
        raise ValueError(f"{season}: #195 fit does not use the frozen baseline")
    if float(baseline.get("degrees_of_freedom", 0)) != 15.0:
        raise ValueError(f"{season}: #195 fit Student-t df is not 15")
    raw = record.get("rank_site_calibration_fit")
    if not isinstance(raw, dict):
        raise TypeError(f"{season}: #195 record is missing generic calibration")
    optimizer = raw.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("success") is not True:
        raise ValueError(f"{season}: #195 generic calibration did not converge")
    if raw.get("context_coefficients") != {}:
        raise ValueError(f"{season}: #195 generic fit contains Context coefficients")
    multiplier = float(raw["overall_mean_multiplier"])
    home_adjustment = float(raw["additional_non_neutral_home_field_points"])
    log_scale = float(raw["common_log_scale_correction"])
    if not all(map(math.isfinite, (multiplier, home_adjustment, log_scale))):
        raise ValueError(f"{season}: #195 calibration parameters are non-finite")
    if multiplier <= 0:
        raise ValueError(f"{season}: #195 mean multiplier must be positive")
    return CalibrationParameters(season, multiplier, home_adjustment, log_scale, raw)


def load_frozen_fits(
    issue195_root: Path,
) -> tuple[dict[int, CalibrationParameters], dict[str, object]]:
    manifest_path = issue195_root / "manifest.json"
    fit_path = issue195_root / "fit_records.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("#195 manifest is not marked complete")
    if sha256_file(fit_path) != EXPECTED_FIT_RECORD_SHA256:
        raise ValueError("#195 fit_records.json does not match the frozen SHA-256")
    output_record = manifest.get("outputs", {}).get(fit_path.name)
    if (
        not isinstance(output_record, dict)
        or output_record.get("sha256") != EXPECTED_FIT_RECORD_SHA256
    ):
        raise ValueError("#195 manifest does not bind the required fit records")
    records = json.loads(fit_path.read_text(encoding="utf-8"))
    selected: dict[int, CalibrationParameters] = {}
    for season in EVALUATION_SEASONS:
        matches = [
            row
            for row in records
            if row.get("role") == "held_out_evaluation"
            and row.get("target_season") == season
        ]
        if len(matches) != 1:
            raise ValueError(f"#195 has {len(matches)} fit records for {season}")
        selected[season] = calibration_parameters_from_record(matches[0], season)
    return selected, {
        "manifest_sha256": sha256_file(manifest_path),
        "fit_records_sha256": EXPECTED_FIT_RECORD_SHA256,
        "fit_records_commit": "2b0cfc2 (required source commit recorded by issue 212)",
        "fit_record_by_season": {
            str(season): {
                "training_seasons": list(range(2022, season)),
                "overall_mean_multiplier": fit.multiplier,
                "additional_non_neutral_home_field_points": fit.home_adjustment,
                "common_log_scale_correction": fit.log_scale_correction,
                "optimizer": fit.fit_record.get("optimizer"),
            }
            for season, fit in selected.items()
        },
    }


def _verify_issue195_inputs(
    *,
    repository_root: Path,
    issue195_root: Path,
    issue200_root: Path,
    issue200_manifest_path: Path,
    issue201_root: Path,
    issue204_final_root: Path,
    issue204_root: Path,
) -> dict[str, object]:
    manifest = json.loads((issue195_root / "manifest.json").read_text(encoding="utf-8"))
    mapping = {
        "issue200_existing_likelihood": issue200_root
        / "historical_likelihood_existing.json",
        "issue200_manifest": issue200_manifest_path,
        "issue200_pipeline_bundle": issue200_root / "pipeline_bundle.json",
        "issue200_preseason_feature_manifest": issue200_root
        / "preseason_feature_manifest.json",
        "issue200_context_feature_inputs": issue200_root / "context_feature_inputs.csv",
        "issue201_manifest": issue201_root / "manifest.json",
        "issue201_forecast_scores": issue201_root / "forecast_scores.csv",
        "issue204_corrected_final_observations": issue204_final_root
        / "final_observations.csv",
        "issue204_corrected_final_targets": issue204_root
        / "final_target_distributions.csv",
        "issue204_corrected_final_rank_pairs": issue204_root
        / "final_constituent_rank_pairs.csv",
    }
    verified: dict[str, object] = {}
    mismatches: list[str] = []
    seen_logical_names: set[str] = set()
    for record in manifest.get("input_records", []):
        logical_name = str(record["logical_name"])
        if logical_name in seen_logical_names:
            mismatches.append(f"#195 input manifest repeats {logical_name}")
        seen_logical_names.add(logical_name)
        path = mapping.get(logical_name)
        if path is None:
            path = repository_root / str(record["path"])
        if not path.is_file():
            mismatches.append(f"{logical_name}: missing at {path}")
            continue
        actual_sha = sha256_file(path)
        if (
            not record.get("exists")
            or actual_sha != record.get("sha256")
            or path.stat().st_size != int(record.get("bytes", -1))
        ):
            mismatches.append(
                f"{logical_name}: expected sha256={record.get('sha256')} "
                f"bytes={record.get('bytes')}, got sha256={actual_sha} "
                f"bytes={path.stat().st_size}"
            )
        verified[logical_name] = {
            "path_label": logical_name,
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
        }
    if (
        "issue200_existing_likelihood" in verified
        and verified["issue200_existing_likelihood"]["sha256"]
        != EXPECTED_BASE_LIKELIHOOD_SHA256  # type: ignore[index]
    ):
        mismatches.append("#195 input baseline likelihood differs from frozen SHA-256")
    if (
        "issue200_manifest" in verified
        and manifest.get("issue200_manifest_sha256")
        != verified["issue200_manifest"]["sha256"]  # type: ignore[index]
    ):
        mismatches.append("#195 recorded #200 manifest hash does not match")
    if (
        "issue201_manifest" in verified
        and manifest.get("issue201_manifest_sha256")
        != verified["issue201_manifest"]["sha256"]  # type: ignore[index]
    ):
        mismatches.append("#195 recorded #201 manifest hash does not match")
    if mismatches:
        raise ValueError(
            "#195 input provenance verification failed: " + "; ".join(mismatches)
        )
    return verified


def _deduplicate_games(games: Sequence[Game]) -> list[Game]:
    """Keep one exact game ID and reject conflicting duplicate evidence."""
    by_id: dict[str, Game] = {}
    for game in games:
        prior = by_id.get(game.game_id)
        if prior is not None and prior != game:
            raise ValueError(f"conflicting duplicate evidence game {game.game_id}")
        by_id[game.game_id] = game
    return [by_id[key] for key in sorted(by_id, key=_game_id_key)]


def _validate_target_exclusion(
    *, game_id: str, cutoff: datetime, kickoff: datetime, evidence_ids: set[str]
) -> None:
    if game_id in evidence_ids:
        raise ValueError(f"{game_id}: forecast target appears in its own evidence")
    if kickoff <= cutoff:
        raise ValueError(f"{game_id}: forecast cutoff is not strictly before kickoff")


def _context_prior_teams(
    bundle: dict[str, object],
    season: int,
    prior_rows: dict[str, dict[str, object]],
    fcs_names: dict[str, str],
) -> list[Team]:
    teams = [
        Team(
            team_id,
            str(row["team_name"]),
            "fbs",
            np.asarray(row["pmf"], dtype=np.float64),
        )
        for team_id, row in sorted(
            prior_rows.items(), key=lambda item: _game_id_key(item[0])
        )
    ]
    populations = bundle.get("fcs_population_by_season")
    if not isinstance(populations, dict) or str(season) not in populations:
        raise ValueError(f"#200 bundle misses the {season} FCS support")
    fcs_population = int(populations[str(season)])
    teams.extend(
        Team(team_id, name, "fcs", np.full(fcs_population, 1 / fcs_population))
        for team_id, name in sorted(
            fcs_names.items(), key=lambda item: _game_id_key(item[0])
        )
    )
    return teams


def _replay_candidate(
    *,
    bundle: dict[str, object],
    season: int,
    prior_rows: dict[str, dict[str, object]],
    fcs_names: dict[str, str],
    games: Sequence[Game],
    base_likelihood: LikelihoodV1,
    adapter: RecalibratedLikelihood,
) -> PosteriorResult:
    teams = _context_prior_teams(bundle, season, prior_rows, fcs_names)
    configuration = bundle.get("inference_configuration")
    if not isinstance(configuration, dict):
        raise TypeError("#200 bundle lacks belief-propagation configuration")
    return infer_posterior(
        teams,
        list(games),
        base_likelihood,
        factor_builder=adapter.factor,
        factor_cache_key=adapter.factor_cache_key,
        max_iterations=int(configuration["max_iterations"]),
        tolerance=float(configuration["tolerance"]),
        damping=float(configuration["damping"]),
    )


def _pmf_digest(pmfs: dict[str, np.ndarray]) -> str:
    return sha256_json(
        [
            {"team_id": team_id, "pmf": np.asarray(pmf, dtype=float).tolist()}
            for team_id, pmf in sorted(
                pmfs.items(), key=lambda item: _game_id_key(item[0])
            )
        ]
    )


def _validate_pmf(pmf: np.ndarray, *, context: str) -> None:
    if (
        pmf.ndim != 1
        or not len(pmf)
        or not np.isfinite(pmf).all()
        or np.any(pmf < 0)
        or not np.isclose(float(pmf.sum()), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError(f"{context}: posterior PMF is not normalized")


def _scheduled_game(
    metadata: dict[str, object], home: Team, away: Team
) -> ScheduledGame:
    return ScheduledGame(
        game_id=str(metadata["game_id"]),
        home_id=home.team_id,
        away_id=away.team_id,
        home_subdivision=str(metadata["home_subdivision"]),  # type: ignore[arg-type]
        away_subdivision=str(metadata["away_subdivision"]),  # type: ignore[arg-type]
        neutral_site=bool(metadata["neutral_site"]),
        season_type=str(metadata["season_type"]),
        date=str(metadata["kickoff_utc"]),
        schedule_status="completed_source_schedule",
    )


def _candidate_prediction_row(
    *,
    metadata: dict[str, object],
    outcome: dict[str, object],
    posterior_teams: dict[str, Team],
    base_likelihood: LikelihoodV1,
    adapter: RecalibratedLikelihood,
    evidence: dict[str, object],
    stage: str,
    cutoff: str,
    pmf_identity: str,
) -> dict[str, object]:
    home = posterior_teams[str(metadata["home_id"])]
    away = posterior_teams[str(metadata["away_id"])]
    scheduled = _scheduled_game(metadata, home, away)
    locations, weights = predictive_components(scheduled, home, away, base_likelihood)
    game = Game(
        str(metadata["game_id"]),
        home.team_id,
        away.team_id,
        home.subdivision,
        away.subdivision,
        int(outcome["home_points"]),
        int(outcome["away_points"]),
        bool(metadata["neutral_site"]),
    )
    locations = adapter.locations_for(locations, game)
    scale = adapter.scale_for(home, away)
    scoring_likelihood = LikelihoodV1(
        base_likelihood.beta, scale, base_likelihood.degrees_of_freedom
    )
    summary = _predictive_summary_from_components(
        locations, weights, scoring_likelihood
    )
    actual_margin = int(outcome["home_points"]) - int(outcome["away_points"])
    crps, error, order, tail_multiple = mixture_margin_crps(
        locations,
        weights,
        float(actual_margin),
        scale,
        base_likelihood.degrees_of_freedom,
    )
    margin_nll = mixture_margin_nll(
        locations,
        weights,
        float(actual_margin),
        scale,
        base_likelihood.degrees_of_freedom,
    )
    actual_home_win = 1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
    return {
        "forecast_id": sha256_json(
            {
                "stage": stage,
                "season": metadata["season"],
                "game_id": metadata["game_id"],
                "cutoff": cutoff,
                "model_arm": CANDIDATE_ARM,
            }
        ),
        "forecast_stage": stage,
        "season": metadata["season"],
        "cutoff_utc": cutoff,
        "game_id": metadata["game_id"],
        "phase": metadata.get("phase"),
        "forecast_population": metadata.get("forecast_population"),
        "home_id": home.team_id,
        "home_team": metadata["home_team"],
        "home_subdivision": metadata["home_subdivision"],
        "away_id": away.team_id,
        "away_team": metadata["away_team"],
        "away_subdivision": metadata["away_subdivision"],
        "kickoff_source": metadata["kickoff_source"],
        "kickoff_utc": metadata["kickoff_utc"],
        "kickoff_precision": metadata["kickoff_precision"],
        "schedule_availability": metadata["schedule_availability"],
        "neutral_site": metadata["neutral_site"],
        "conference_game": metadata["conference_game"],
        "actual_home_points": outcome["home_points"],
        "actual_away_points": outcome["away_points"],
        "actual_home_margin": actual_margin,
        "model_arm": CANDIDATE_ARM,
        "policy": "existing",
        "prior_family": "Context",
        "evidence_game_count": evidence["included_game_count"],
        "evidence_ids_sha256": evidence["included_game_ids_sha256"],
        "evidence_rows_sha256": evidence["included_game_rows_sha256"],
        "expected_home_margin": summary.expected_home_margin,
        "median_home_margin": summary.median_home_margin,
        "home_win_probability": summary.home_win_probability,
        "away_win_probability": summary.away_win_probability,
        "tie_probability": summary.tie_probability,
        "margin_interval_50_low": summary.margin_interval_50[0],
        "margin_interval_50_high": summary.margin_interval_50[1],
        "margin_interval_80_low": summary.margin_interval_80[0],
        "margin_interval_80_high": summary.margin_interval_80[1],
        "margin_interval_95_low": summary.margin_interval_95[0],
        "margin_interval_95_high": summary.margin_interval_95[1],
        "crps": crps,
        "crps_numerical_error_estimate": error,
        "crps_quadrature_order": order,
        "crps_tail_multiple": tail_multiple,
        "margin_nll": margin_nll,
        "median_absolute_error": abs(summary.median_home_margin - actual_margin),
        "home_win_brier": (summary.home_win_probability - actual_home_win) ** 2,
        "coverage_50": int(
            summary.margin_interval_50[0]
            <= actual_margin
            <= summary.margin_interval_50[1]
        ),
        "coverage_80": int(
            summary.margin_interval_80[0]
            <= actual_margin
            <= summary.margin_interval_80[1]
        ),
        "coverage_95": int(
            summary.margin_interval_95[0]
            <= actual_margin
            <= summary.margin_interval_95[1]
        ),
        "predictive_scale": scale,
        "posterior_pmf_pair_sha256": sha256_json(
            {
                "home": home.prior.tolist(),
                "away": away.prior.tolist(),
                "posterior_identity": pmf_identity,
            }
        ),
    }


def _pmf_rows(
    *,
    season: int,
    cutoff: str,
    phase: str,
    arms: dict[str, tuple[dict[str, Team], dict[str, np.ndarray]]],
    prior_pmfs: dict[str, np.ndarray],
    evidence: dict[str, object],
    targets: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm, (teams, pmfs) in arms.items():
        for team_id, team in sorted(
            teams.items(), key=lambda item: _game_id_key(item[0])
        ):
            posterior = np.asarray(pmfs[team_id], dtype=np.float64)
            prior = np.asarray(prior_pmfs[team_id], dtype=np.float64)
            _validate_pmf(posterior, context=f"{season}/{cutoff}/{arm}/{team_id}")
            _validate_pmf(prior, context=f"{season}/{cutoff}/prior/{team_id}")
            if posterior.shape != prior.shape:
                raise ValueError(
                    f"{season}/{cutoff}/{arm}/{team_id}: prior/posterior support differs"
                )
            target = None if targets is None else targets.get(team_id)
            target_pmf = (
                None if target is None else np.asarray(target["pmf"], dtype=float)
            )
            rows.append(
                {
                    "season": season,
                    "cutoff_utc": cutoff,
                    "checkpoint_phase": phase,
                    "model_arm": arm,
                    "team_id": team_id,
                    "team_name": team.name,
                    "subdivision": team.subdivision,
                    "prior_pmf": prior.tolist(),
                    "prior_pmf_sha256": sha256_json(prior.tolist()),
                    "posterior_pmf": posterior.tolist(),
                    "target_pmf": None if target_pmf is None else target_pmf.tolist(),
                    "posterior_pmf_sha256": sha256_json(posterior.tolist()),
                    "target_pmf_sha256": (
                        None if target_pmf is None else sha256_json(target_pmf.tolist())
                    ),
                    "evidence_game_count": evidence["included_game_count"],
                    "evidence_ids_sha256": evidence["included_game_ids_sha256"],
                    "evidence_rows_sha256": evidence["included_game_rows_sha256"],
                }
            )
    return rows


def _rank_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate in-season checkpoints and preseason separately by team-season."""

    def is_preseason(row: dict[str, object]) -> bool:
        return (
            row.get("checkpoint_phase") == "preseason"
            or int(row.get("evidence_game_count", 1)) == 0
        )

    result: list[dict[str, object]] = []
    for stage, stage_rows in (
        (
            "in_season",
            [row for row in rows if not is_preseason(row)],
        ),
        (
            "preseason",
            [row for row in rows if is_preseason(row)],
        ),
    ):
        if not stage_rows:
            continue
        by_team: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in stage_rows:
            by_team[
                (int(row["season"]), str(row["model_arm"]), str(row["team_id"]))
            ].append(row)
        team_rows = [
            {
                "season": season,
                "model_arm": arm,
                "team_id": team_id,
                "checkpoints": len(members),
                **{
                    f"mean_{metric}": float(
                        np.mean([float(member[metric]) for member in members])
                    )
                    for metric in RANK_METRICS
                },
            }
            for (season, arm, team_id), members in sorted(by_team.items())
        ]
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in team_rows:
            grouped[(str(row["season"]), str(row["model_arm"]))].append(row)
        season_aggregation = (
            "season_team_season_weight_one"
            if stage == "in_season"
            else "season_preseason_team_season"
        )
        pooled_aggregation = (
            "pooled_equal_team_season"
            if stage == "in_season"
            else "pooled_preseason_equal_team_season"
        )
        for (season, arm), members in sorted(grouped.items()):
            result.append(
                {
                    "aggregation": season_aggregation,
                    "checkpoint_stage": stage,
                    "season": season,
                    "model_arm": arm,
                    "team_seasons": len(members),
                    "checkpoints": sum(
                        int(member["checkpoints"]) for member in members
                    ),
                    **{
                        metric: float(
                            np.mean(
                                [float(member[f"mean_{metric}"]) for member in members]
                            )
                        )
                        for metric in RANK_METRICS
                    },
                }
            )
        for arm in sorted({str(row["model_arm"]) for row in team_rows}):
            members = [row for row in team_rows if row["model_arm"] == arm]
            result.append(
                {
                    "aggregation": pooled_aggregation,
                    "checkpoint_stage": stage,
                    "season": "2023-2025",
                    "model_arm": arm,
                    "team_seasons": len(members),
                    "checkpoints": sum(
                        int(member["checkpoints"]) for member in members
                    ),
                    **{
                        metric: float(
                            np.mean(
                                [float(member[f"mean_{metric}"]) for member in members]
                            )
                        )
                        for metric in RANK_METRICS
                    },
                }
            )
    return result


def _paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    by_arm_key = {
        (
            str(row["forecast_stage"]),
            str(row["forecast_population"]),
            int(row["season"]),
            str(row["game_id"]),
            str(row["model_arm"]),
        ): row
        for row in rows
    }
    keys = sorted({key[:4] for key in by_arm_key})
    for key in keys:
        baseline = by_arm_key.get((*key, BASELINE_ARM))
        candidate = by_arm_key.get((*key, CANDIDATE_ARM))
        if baseline is None or candidate is None:
            raise ValueError(f"{key}: forecast arm is missing from common cohort")
        if baseline["cutoff_utc"] != candidate["cutoff_utc"]:
            raise ValueError(f"{key}: cutoff differs across forecast arms")
        row: dict[str, object] = {
            "forecast_stage": key[0],
            "forecast_population": key[1],
            "season": key[2],
            "game_id": key[3],
            "cutoff_utc": baseline["cutoff_utc"],
            "phase": baseline["phase"],
            "home_id": baseline["home_id"],
            "away_id": baseline["away_id"],
            "evidence_ids_sha256": baseline["evidence_ids_sha256"],
            "evidence_rows_sha256": baseline["evidence_rows_sha256"],
        }
        for metric in (
            "crps",
            "margin_nll",
            "home_win_brier",
            "median_absolute_error",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "margin_interval_50_low",
            "margin_interval_50_high",
            "margin_interval_80_low",
            "margin_interval_80_high",
            "margin_interval_95_low",
            "margin_interval_95_high",
        ):
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_candidate"] = candidate[metric]
            row[f"{metric}_delta"] = float(candidate[metric]) - float(baseline[metric])
        output.append(row)
    return output


def _add_phase_pooled_summaries(
    forecast_rows: list[dict[str, object]], summaries: list[dict[str, object]]
) -> list[dict[str, object]]:
    metrics = (
        "crps",
        "margin_nll",
        "median_absolute_error",
        "home_win_brier",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "margin_interval_50_width",
        "margin_interval_80_width",
        "margin_interval_95_width",
    )
    selected = [
        row
        for row in forecast_rows
        if row["forecast_stage"] == "rolling"
        and row["forecast_population"] == PRIMARY_POPULATION
        and row["phase"] == "pre_thanksgiving"
    ]
    for arm in (BASELINE_ARM, CANDIDATE_ARM):
        members = [row for row in selected if row["model_arm"] == arm]
        if not members:
            raise ValueError(f"no pre-Thanksgiving forecasts for {arm}")

        def value(row: dict[str, object], metric: str) -> float:
            if metric.endswith("_width"):
                interval = metric.removeprefix("margin_interval_").removesuffix(
                    "_width"
                )
                return float(row[f"margin_interval_{interval}_high"]) - float(
                    row[f"margin_interval_{interval}_low"]
                )
            return float(row[metric])

        summaries.append(
            {
                "forecast_stage": "rolling",
                "forecast_population": PRIMARY_POPULATION,
                "model_arm": arm,
                "season": "pooled_equal_game",
                "phase": "pre_thanksgiving",
                "games": len(members),
                **{
                    metric: float(np.mean([value(row, metric) for row in members]))
                    for metric in metrics
                },
            }
        )
    return summaries


def _gain_concentration(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    by_team: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in paired:
        if (
            row["forecast_stage"] != "rolling"
            or row["forecast_population"] != PRIMARY_POPULATION
        ):
            continue
        delta = float(row["crps_delta"])
        for team_id in (str(row["home_id"]), str(row["away_id"])):
            by_team[(team_id, int(row["season"]), str(row["phase"]))].append(delta)
    return [
        {
            "team_id": team_id,
            "season": season,
            "phase": phase,
            "game_appearances": len(values),
            "mean_crps_delta": float(np.mean(values)),
            "sum_crps_delta": float(np.sum(values)),
            "candidate_worse_games": sum(value > 0 for value in values),
            "candidate_better_games": sum(value < 0 for value in values),
        }
        for (team_id, season, phase), values in sorted(by_team.items())
    ]


def _coverage_error(row: dict[str, object]) -> float:
    levels = (50, 80, 95)
    return float(
        np.mean(
            [abs(float(row[f"coverage_{level}"]) - level / 100) for level in levels]
        )
    )


def _summary_lookup(
    summaries: list[dict[str, object]], arm: str, *, stage: str, season: str, phase: str
) -> dict[str, object]:
    matches = [
        row
        for row in summaries
        if row["model_arm"] == arm
        and row["forecast_stage"] == stage
        and row["forecast_population"] == PRIMARY_POPULATION
        and str(row["season"]) == season
        and row["phase"] == phase
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing or duplicate forecast summary {arm}/{stage}/{season}/{phase}"
        )
    return matches[0]


def _decision(summaries: list[dict[str, object]]) -> dict[str, object]:
    baseline_pooled = _summary_lookup(
        summaries,
        BASELINE_ARM,
        stage="rolling",
        season="pooled_equal_game",
        phase="all_phases",
    )
    candidate_pooled = _summary_lookup(
        summaries,
        CANDIDATE_ARM,
        stage="rolling",
        season="pooled_equal_game",
        phase="all_phases",
    )
    baseline_equal = _summary_lookup(
        summaries,
        BASELINE_ARM,
        stage="rolling",
        season="equal_season_sensitivity",
        phase="all_phases",
    )
    candidate_equal = _summary_lookup(
        summaries,
        CANDIDATE_ARM,
        stage="rolling",
        season="equal_season_sensitivity",
        phase="all_phases",
    )
    season_delta = {
        season: float(
            _summary_lookup(
                summaries,
                CANDIDATE_ARM,
                stage="rolling",
                season=str(season),
                phase="all_phases",
            )["crps"]
        )
        - float(
            _summary_lookup(
                summaries,
                BASELINE_ARM,
                stage="rolling",
                season=str(season),
                phase="all_phases",
            )["crps"]
        )
        for season in EVALUATION_SEASONS
    }
    pooled_delta = float(candidate_pooled["crps"]) - float(baseline_pooled["crps"])
    equal_delta = float(candidate_equal["crps"]) - float(baseline_equal["crps"])
    wins = sum(delta < 0 for delta in season_delta.values())
    primary_pass = (
        pooled_delta <= -PRIMARY_GAIN_THRESHOLD
        and equal_delta <= -PRIMARY_GAIN_THRESHOLD
        and wins >= 2
    )

    guardrails: dict[str, object] = {}
    for stratum, stage, phase, summary_season, season_phase in (
        ("preseason", "preseason", "all_phases", "pooled_equal_game", "all_phases"),
        (
            "pre_thanksgiving",
            "rolling",
            "pre_thanksgiving",
            "pooled_equal_game",
            "pre_thanksgiving",
        ),
    ):
        base_pooled = _summary_lookup(
            summaries, BASELINE_ARM, stage=stage, season=summary_season, phase=phase
        )
        candidate_pooled_stratum = _summary_lookup(
            summaries, CANDIDATE_ARM, stage=stage, season=summary_season, phase=phase
        )
        pooled_crps_delta = float(candidate_pooled_stratum["crps"]) - float(
            base_pooled["crps"]
        )
        pooled_coverage_delta = _coverage_error(
            candidate_pooled_stratum
        ) - _coverage_error(base_pooled)
        per_season = {}
        for season in EVALUATION_SEASONS:
            base_row = _summary_lookup(
                summaries,
                BASELINE_ARM,
                stage=stage,
                season=str(season),
                phase=season_phase,
            )
            candidate_row = _summary_lookup(
                summaries,
                CANDIDATE_ARM,
                stage=stage,
                season=str(season),
                phase=season_phase,
            )
            per_season[str(season)] = {
                "crps_delta": float(candidate_row["crps"]) - float(base_row["crps"]),
                "coverage_error_delta": _coverage_error(candidate_row)
                - _coverage_error(base_row),
                "crps_pass": float(candidate_row["crps"]) - float(base_row["crps"])
                <= GUARDRAIL_SEASON_CRPS,
                "coverage_pass": _coverage_error(candidate_row)
                - _coverage_error(base_row)
                <= GUARDRAIL_SEASON_COVERAGE_ERROR,
            }
        guardrails[stratum] = {
            "pooled_crps_delta": pooled_crps_delta,
            "pooled_coverage_error_delta": pooled_coverage_delta,
            "pooled_crps_pass": pooled_crps_delta <= GUARDRAIL_POOLED_CRPS,
            "pooled_coverage_pass": pooled_coverage_delta
            <= GUARDRAIL_POOLED_COVERAGE_ERROR,
            "per_season": per_season,
            "pass": (
                pooled_crps_delta <= GUARDRAIL_POOLED_CRPS
                and pooled_coverage_delta <= GUARDRAIL_POOLED_COVERAGE_ERROR
                and all(
                    bool(row["crps_pass"]) and bool(row["coverage_pass"])
                    for row in per_season.values()
                )
            ),
        }
    guardrails_pass = all(bool(row["pass"]) for row in guardrails.values())
    confirmation = primary_pass and guardrails_pass
    return {
        "conclusion": (
            "Generic recalibration merits independent confirmation"
            if confirmation
            else "Retain unmodified Context / no useful gain established"
        ),
        "recommend_independent_confirmation": confirmation,
        "primary_rule": {
            "pooled_equal_game_crps_delta": pooled_delta,
            "equal_season_crps_delta": equal_delta,
            "season_crps_delta": {
                str(key): value for key, value in season_delta.items()
            },
            "seasons_won": wins,
            "pooled_equal_game_threshold": -PRIMARY_GAIN_THRESHOLD,
            "equal_season_threshold": -PRIMARY_GAIN_THRESHOLD,
            "minimum_season_wins": 2,
            "pass": primary_pass,
        },
        "early_guardrails": guardrails,
        "early_guardrails_pass": guardrails_pass,
        "decision_scope": "descriptive exploratory development seasons; not a significance test",
    }


def _input_hash_record(path: Path, label: str) -> dict[str, object]:
    return {
        "path_label": label,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_common_forecasts(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for stage in ("rolling", "preseason"):
        for population in (PRIMARY_POPULATION, SECONDARY_POPULATION):
            selected = [
                row
                for row in rows
                if row["forecast_stage"] == stage
                and row["forecast_population"] == population
            ]
            by_arm: dict[str, set[tuple[int, str]]] = defaultdict(set)
            for row in selected:
                key = (int(row["season"]), str(row["game_id"]))
                if key in by_arm[str(row["model_arm"])]:
                    raise ValueError(
                        f"{stage}/{population}: duplicate game forecast {key}"
                    )
                by_arm[str(row["model_arm"])].add(key)
            if selected and set(by_arm) != {BASELINE_ARM, CANDIDATE_ARM}:
                raise ValueError(
                    f"{stage}/{population}: incomplete model-arm forecast cohort"
                )
            if selected and by_arm[BASELINE_ARM] != by_arm[CANDIDATE_ARM]:
                raise ValueError(
                    f"{stage}/{population}: forecast arms have different game IDs"
                )
            result[f"{stage}/{population}"] = {
                "forecast_rows": len(selected),
                "game_count_per_arm": len(by_arm.get(BASELINE_ARM, set())),
                "common_cohort": not selected
                or by_arm[BASELINE_ARM] == by_arm[CANDIDATE_ARM],
            }
    return result


def _verify_baseline_parity(
    baseline_rows: list[dict[str, object]], issue201_forecast_path: Path
) -> dict[str, object]:
    with issue201_forecast_path.open(newline="", encoding="utf-8-sig") as handle:
        reference = list(csv.DictReader(handle))
    expected = {
        (int(row["season"]), str(row["game_id"]), str(row["cutoff_utc"])): row
        for row in reference
        if row.get("model_arm") == BASELINE_ARM
        and row.get("forecast_stage") == "rolling"
        and row.get("forecast_population") == PRIMARY_POPULATION
        and int(row["season"]) in EVALUATION_SEASONS
    }
    actual = {
        (int(row["season"]), str(row["game_id"]), str(row["cutoff_utc"])): row
        for row in baseline_rows
        if row["model_arm"] == BASELINE_ARM
        and row["forecast_stage"] == "rolling"
        and row["forecast_population"] == PRIMARY_POPULATION
        and int(row["season"]) in EVALUATION_SEASONS
    }
    if set(actual) != set(expected):
        raise ValueError(
            "#201 baseline forecast cohort mismatch: "
            f"missing={len(set(expected) - set(actual))}, extra={len(set(actual) - set(expected))}"
        )
    metrics = (
        "crps",
        "margin_nll",
        "median_absolute_error",
        "home_win_brier",
        "expected_home_margin",
        "median_home_margin",
        "home_win_probability",
        "away_win_probability",
        "margin_interval_50_low",
        "margin_interval_50_high",
        "margin_interval_80_low",
        "margin_interval_80_high",
        "margin_interval_95_low",
        "margin_interval_95_high",
    )
    errors = []
    for key, row in actual.items():
        ref = expected[key]
        for field in metrics:
            error = abs(float(row[field]) - float(ref[field]))
            if error > IDENTITY_TOLERANCE:
                raise ValueError(
                    f"#201 baseline parity failed for {key}/{field}: {error}"
                )
            errors.append(error)
        for field in ("evidence_ids_sha256", "evidence_rows_sha256"):
            if row[field] != ref[field]:
                raise ValueError(
                    f"#201 baseline evidence hash mismatch for {key}/{field}"
                )
        if row["home_id"] != ref["home_id"] or row["away_id"] != ref["away_id"]:
            raise ValueError(f"#201 baseline team mismatch for {key}")
    return {
        "passed": True,
        "forecast_rows_compared": len(actual),
        "metrics_compared_per_row": list(metrics),
        "max_absolute_metric_difference": max(errors, default=0.0),
        "tolerance": IDENTITY_TOLERANCE,
        "cutoff_game_keys_sha256": sha256_json(sorted(actual)),
        "baseline_pmfs": (
            "Reconstructed from the verified #200 Context priors and #201 replay API; "
            "#201 did not persist per-cutoff posterior PMF files. PMFs are included "
            "in this run's lossless compressed PMF artifact."
        ),
    }


def run_issue_212_replay(
    *,
    repository_root: Path,
    issue195_root: Path,
    issue200_root: Path,
    issue201_root: Path,
    issue204_final_root: Path,
    issue204_root: Path,
    game_root: Path,
    protocol_path: Path,
    issue200_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Run the frozen 2023–2025 two-arm replay and return output rows."""
    fits, fit_provenance = load_frozen_fits(issue195_root)
    issue200_manifest_path = issue200_manifest_path or issue200_root / "manifest.json"
    issue195_inputs = _verify_issue195_inputs(
        repository_root=repository_root,
        issue195_root=issue195_root,
        issue200_root=issue200_root,
        issue200_manifest_path=issue200_manifest_path,
        issue201_root=issue201_root,
        issue204_final_root=issue204_final_root,
        issue204_root=issue204_root,
    )
    issue201_manifest_path = issue201_root / "manifest.json"
    issue201_plan_path = issue201_root / "schedule_plan.json"
    issue201_plan_manifest_path = issue201_root / "schedule_plan.manifest.json"
    schedule_plan = json.loads(issue201_plan_path.read_text(encoding="utf-8"))
    issue204_manifest_path = issue204_root / "manifest.json"
    target_path = issue204_root / "pre_bowl_target_distributions.csv"
    target_snapshot_manifest_path = issue200_root / "target_snapshot_manifest.csv"
    schedule_windows_path = (
        repository_root / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
    )
    issue200_bundle_path = issue200_root / "pipeline_bundle.json"
    issue200_fitted_models_path = issue200_root / "fitted_models.json"
    protocol_sha = sha256_file(protocol_path)

    outcome_rows_by_season, game_source_files = verify_score_blind_plan(
        schedule_plan,
        issue201_plan_path,
        issue201_plan_manifest_path,
        game_root,
        protocol_sha256=sha256_file(
            repository_root / "docs/issue_201_prebowl_forecast_comparison.md"
        ),
        target_snapshot_manifest_path=target_snapshot_manifest_path,
        schedule_windows_path=schedule_windows_path,
    )
    bundle, priors_by_season, issue200_provenance = _load_issue200_inputs(
        bundle_path=issue200_bundle_path,
        manifest_path=issue200_manifest_path,
        fitted_models_path=issue200_fitted_models_path,
    )
    base_likelihood = _load_arm_likelihood(bundle, "existing")
    if (
        sha256_file(issue200_root / "historical_likelihood_existing.json")
        != EXPECTED_BASE_LIKELIHOOD_SHA256
    ):
        raise ValueError("#200 existing likelihood differs from required SHA-256")
    targets, target_provenance = load_pre_bowl_targets(
        target_path, issue204_manifest_path
    )
    target_dates = _read_target_snapshot_dates(target_snapshot_manifest_path)
    _target_window_rows(schedule_windows_path)

    all_forecasts: list[dict[str, object]] = []
    all_rank_scores: list[dict[str, object]] = []
    all_pmf_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    forecast_exclusions: list[dict[str, object]] = []
    rank_exclusions: list[dict[str, object]] = []
    baseline_pmf_hashes: dict[str, str] = {}
    candidate_pmf_hashes: dict[str, str] = {}

    schedule_seasons = {int(row["season"]): row for row in schedule_plan["seasons"]}
    for season in EVALUATION_SEASONS:
        print(f"Issue 212: replaying season {season}", flush=True)
        season_plan = schedule_seasons[season]
        season_outcomes = outcome_rows_by_season[season]
        scoreable, exclusions = _scoreable_game_rows(season_plan, season_outcomes)
        forecast_exclusions.extend(exclusions)
        common_teams, excluded_targets = _common_target_teams(
            season, priors_by_season[str(season)], targets
        )
        rank_exclusions.extend(excluded_targets)
        if not common_teams:
            raise ValueError(f"{season}: no common Context prior/pre-bowl target teams")
        fcs_names = _fcs_names(season_outcomes)
        prior_rows = priors_by_season[str(season)][BASELINE_ARM]
        parameters = fits[season]
        adapter = RecalibratedLikelihood(base_likelihood, parameters)
        likelihood_identity = {
            "season": season,
            "baseline_likelihood_sha256": EXPECTED_BASE_LIKELIHOOD_SHA256,
            "candidate_parameters": {
                "multiplier": parameters.multiplier,
                "home_adjustment": parameters.home_adjustment,
                "log_scale_correction": parameters.log_scale_correction,
                "degrees_of_freedom": 15.0,
            },
        }

        # Preseason predictions use identical Context priors and no current-season evidence.
        empty_replay = replay_pipeline_bundle(
            bundle,
            season=season,
            policy="existing",
            prior_family="Context",
            games=[],
            fcs_team_names=fcs_names,
        )
        empty_candidate = _replay_candidate(
            bundle=bundle,
            season=season,
            prior_rows=prior_rows,
            fcs_names=fcs_names,
            games=[],
            base_likelihood=base_likelihood,
            adapter=adapter,
        )
        baseline_pre_diag = _posterior_diagnostics(empty_replay)
        candidate_pre_diag = _posterior_diagnostics(empty_candidate)
        baseline_pre_teams = _build_posterior_teams(
            empty_replay.pmfs, prior_rows, fcs_names
        )
        candidate_pre_teams = _build_posterior_teams(
            empty_candidate.pmfs, prior_rows, fcs_names
        )
        pre_evidence = _evidence_record(
            season,
            datetime(season, 1, 1, tzinfo=UTC),
            "existing",
            [],
            [],
            {"Context_existing": baseline_pre_diag, CANDIDATE_ARM: candidate_pre_diag},
            len(empty_replay.pmfs),
            len(fcs_names),
        )
        pre_evidence["cutoff_utc"] = "preseason_no_evidence"
        pre_evidence["included_game_row_hashes"] = []
        evidence_rows.append(pre_evidence)
        pre_key = f"{season}/preseason_no_evidence"
        baseline_pre_hash = _pmf_digest(empty_replay.pmfs)
        candidate_pre_hash = _pmf_digest(empty_candidate.pmfs)
        if baseline_pre_hash != candidate_pre_hash:
            raise ValueError(
                f"{season}: baseline and candidate preseason priors differ"
            )
        baseline_pmf_hashes[pre_key] = baseline_pre_hash
        candidate_pmf_hashes[pre_key] = candidate_pre_hash
        pre_arms = {
            BASELINE_ARM: (baseline_pre_teams, empty_replay.pmfs),
            CANDIDATE_ARM: (candidate_pre_teams, empty_candidate.pmfs),
        }
        all_pmf_rows.extend(
            _pmf_rows(
                season=season,
                cutoff="preseason_no_evidence",
                phase="preseason",
                arms=pre_arms,
                prior_pmfs=empty_replay.pmfs,
                evidence=pre_evidence,
            )
        )
        convergence_rows.extend(
            [
                {
                    "season": season,
                    "cutoff_utc": "preseason_no_evidence",
                    "model_arm": BASELINE_ARM,
                    **baseline_pre_diag,
                },
                {
                    "season": season,
                    "cutoff_utc": "preseason_no_evidence",
                    "model_arm": CANDIDATE_ARM,
                    **candidate_pre_diag,
                },
            ]
        )

        for metadata, outcome in scoreable.values():
            if metadata["forecast_population"] not in {
                PRIMARY_POPULATION,
                SECONDARY_POPULATION,
            }:
                continue
            for arm, teams, pmf_hash in (
                (BASELINE_ARM, baseline_pre_teams, baseline_pre_hash),
                (CANDIDATE_ARM, candidate_pre_teams, candidate_pre_hash),
            ):
                if arm == BASELINE_ARM:
                    row = _prediction_row(
                        metadata=metadata,
                        outcome=outcome,
                        arm=arm,
                        policy="existing",
                        family="Context",
                        posterior_teams=teams,
                        likelihood=base_likelihood,
                        evidence=pre_evidence,
                        stage="preseason",
                        cutoff="preseason_no_evidence",
                    )
                else:
                    row = _candidate_prediction_row(
                        metadata=metadata,
                        outcome=outcome,
                        posterior_teams=teams,
                        base_likelihood=base_likelihood,
                        adapter=adapter,
                        evidence=pre_evidence,
                        stage="preseason",
                        cutoff="preseason_no_evidence",
                        pmf_identity=pmf_hash,
                    )
                row["posterior_pmf_pair_sha256"] = sha256_json(
                    {
                        "home": teams[str(metadata["home_id"])].prior.tolist(),
                        "away": teams[str(metadata["away_id"])].prior.tolist(),
                        "posterior_identity": pmf_hash,
                    }
                )
                row["predictive_scale"] = (
                    adapter.scale_for(
                        teams[str(metadata["home_id"])],
                        teams[str(metadata["away_id"])],
                    )
                    if arm == CANDIDATE_ARM
                    else base_likelihood.scale
                )
                row["likelihood_identity_sha256"] = sha256_json(likelihood_identity)
                all_forecasts.append(row)

        plan_games_by_cutoff: dict[str, list[str]] = defaultdict(list)
        for game_id, (metadata, _) in scoreable.items():
            plan_games_by_cutoff[str(metadata["assigned_cutoff"])].append(game_id)
        plan_by_id = {str(row["game_id"]): row for row in season_plan["games"]}
        for ids in plan_games_by_cutoff.values():
            ids.sort(
                key=lambda game_id: (
                    str(plan_by_id[game_id]["kickoff_utc"]),
                    _game_id_key(game_id),
                )
            )

        cutoff_phases = {
            str(row["cutoff_utc"]): str(row["phase"])
            for row in season_plan["cutoff_phases"]
        }
        target_start = datetime.combine(target_dates[season], time.min, tzinfo=UTC)
        for cutoff_index, cutoff_text in enumerate(season_plan["cutoffs_utc"], start=1):
            cutoff = datetime.fromisoformat(str(cutoff_text))
            future_ids = set(plan_games_by_cutoff.get(str(cutoff_text), []))
            included, excluded = filter_evidence_rows(
                season_outcomes, cutoff, "existing"
            )
            evidence_games = _deduplicate_games(
                [game_from_outcome(row) for row in included]
            )
            included_ids = {str(row["game_id"]) for row in included}
            overlap = future_ids & included_ids
            if overlap:
                raise ValueError(
                    f"{season}/{cutoff_text}: scheduled target game entered evidence: {sorted(overlap)[:5]}"
                )
            for game_id in future_ids:
                game_meta = plan_by_id[game_id]
                _validate_target_exclusion(
                    game_id=game_id,
                    cutoff=cutoff,
                    kickoff=datetime.fromisoformat(str(game_meta["kickoff_utc"])),
                    evidence_ids=included_ids,
                )

            baseline_replay = replay_pipeline_bundle(
                bundle,
                season=season,
                policy="existing",
                prior_family="Context",
                games=evidence_games,
                fcs_team_names=fcs_names,
            )
            candidate_replay = _replay_candidate(
                bundle=bundle,
                season=season,
                prior_rows=prior_rows,
                fcs_names=fcs_names,
                games=evidence_games,
                base_likelihood=base_likelihood,
                adapter=adapter,
            )
            baseline_diag = _posterior_diagnostics(baseline_replay)
            candidate_diag = _posterior_diagnostics(candidate_replay)
            baseline_teams = _build_posterior_teams(
                baseline_replay.pmfs, prior_rows, fcs_names
            )
            candidate_teams = _build_posterior_teams(
                candidate_replay.pmfs, prior_rows, fcs_names
            )
            evidence = _evidence_record(
                season,
                cutoff,
                "existing",
                included,
                excluded,
                {BASELINE_ARM: baseline_diag, CANDIDATE_ARM: candidate_diag},
                len(baseline_replay.pmfs),
                len(fcs_names),
            )
            evidence["included_game_row_hashes"] = [
                {
                    "game_id": row["game_id"],
                    "kickoff_utc": row.get("kickoff_utc"),
                    "game_row_sha256": row["game_row_sha256"],
                }
                for row in sorted(
                    included, key=lambda item: _game_id_key(str(item["game_id"]))
                )
            ]
            evidence["excluded_game_details"] = [
                {
                    "game_id": row["game_id"],
                    "kickoff_utc": row.get("kickoff_utc"),
                    "reason": row["evidence_exclusion_reason"],
                    "game_row_sha256": row["game_row_sha256"],
                }
                for row in excluded
            ]
            evidence["applies_to_arms"] = [BASELINE_ARM, CANDIDATE_ARM]
            evidence_rows.append(evidence)
            cutoff_key = f"{season}/{cutoff_text}"
            baseline_hash = _pmf_digest(baseline_replay.pmfs)
            candidate_hash = _pmf_digest(candidate_replay.pmfs)
            baseline_pmf_hashes[cutoff_key] = baseline_hash
            candidate_pmf_hashes[cutoff_key] = candidate_hash
            arms = {
                BASELINE_ARM: (baseline_teams, baseline_replay.pmfs),
                CANDIDATE_ARM: (candidate_teams, candidate_replay.pmfs),
            }
            rank_targets = targets[season] if cutoff < target_start else None
            all_pmf_rows.extend(
                _pmf_rows(
                    season=season,
                    cutoff=str(cutoff_text),
                    phase=cutoff_phases[str(cutoff_text)],
                    arms=arms,
                    prior_pmfs=empty_replay.pmfs,
                    evidence=evidence,
                    targets=rank_targets,
                )
            )
            convergence_rows.extend(
                [
                    {
                        "season": season,
                        "cutoff_utc": cutoff_text,
                        "model_arm": BASELINE_ARM,
                        **baseline_diag,
                    },
                    {
                        "season": season,
                        "cutoff_utc": cutoff_text,
                        "model_arm": CANDIDATE_ARM,
                        **candidate_diag,
                    },
                ]
            )
            if cutoff < target_start:
                late_ids = [
                    str(row["game_id"])
                    for row in included
                    if datetime.fromisoformat(str(row["kickoff_utc"])) >= target_start
                ]
                if late_ids:
                    raise ValueError(
                        f"{season}/{cutoff_text}: ranking evidence includes post-target games"
                    )
                for arm, pmfs in (
                    (BASELINE_ARM, baseline_replay.pmfs),
                    (CANDIDATE_ARM, candidate_replay.pmfs),
                ):
                    for team_id in common_teams:
                        forecast = np.asarray(pmfs[team_id], dtype=np.float64)
                        target = np.asarray(
                            targets[season][team_id]["pmf"], dtype=np.float64
                        )
                        if forecast.shape != target.shape:
                            raise ValueError(
                                f"{season}/{team_id}/{arm}: ranking support differs"
                            )
                        all_rank_scores.append(
                            {
                                "season": season,
                                "cutoff_utc": cutoff_text,
                                "checkpoint_phase": cutoff_phases[str(cutoff_text)],
                                "team_id": team_id,
                                "team_name": targets[season][team_id]["team_name"],
                                "model_arm": arm,
                                "target_observation_date": target_dates[
                                    season
                                ].isoformat(),
                                "rank_count": len(target),
                                "target_pmf_sha256": sha256_json(target.tolist()),
                                "forecast_pmf_sha256": sha256_json(forecast.tolist()),
                                "evidence_game_count": evidence["included_game_count"],
                                "evidence_ids_sha256": evidence[
                                    "included_game_ids_sha256"
                                ],
                                **score_rank_pmf(forecast, target),
                            }
                        )
            for game_id in sorted(future_ids, key=_game_id_key):
                metadata, outcome = scoreable[game_id]
                if metadata["forecast_population"] not in {
                    PRIMARY_POPULATION,
                    SECONDARY_POPULATION,
                }:
                    continue
                for arm in (BASELINE_ARM, CANDIDATE_ARM):
                    teams = baseline_teams if arm == BASELINE_ARM else candidate_teams
                    if (
                        str(metadata["home_id"]) not in teams
                        or str(metadata["away_id"]) not in teams
                    ):
                        forecast_exclusions.append(
                            {
                                **metadata,
                                "exclusion_reason": "team_missing_from_context_posterior",
                            }
                        )
                        continue
                    if arm == BASELINE_ARM:
                        row = _prediction_row(
                            metadata=metadata,
                            outcome=outcome,
                            arm=arm,
                            policy="existing",
                            family="Context",
                            posterior_teams=teams,
                            likelihood=base_likelihood,
                            evidence=evidence,
                            stage="rolling",
                            cutoff=str(cutoff_text),
                        )
                    else:
                        row = _candidate_prediction_row(
                            metadata=metadata,
                            outcome=outcome,
                            posterior_teams=teams,
                            base_likelihood=base_likelihood,
                            adapter=adapter,
                            evidence=evidence,
                            stage="rolling",
                            cutoff=str(cutoff_text),
                            pmf_identity=candidate_hash,
                        )
                    row["posterior_pmf_pair_sha256"] = sha256_json(
                        {
                            "home": teams[str(metadata["home_id"])].prior.tolist(),
                            "away": teams[str(metadata["away_id"])].prior.tolist(),
                            "posterior_identity": (
                                baseline_hash if arm == BASELINE_ARM else candidate_hash
                            ),
                        }
                    )
                    row["predictive_scale"] = (
                        adapter.scale_for(
                            teams[str(metadata["home_id"])],
                            teams[str(metadata["away_id"])],
                        )
                        if arm == CANDIDATE_ARM
                        else base_likelihood.scale
                    )
                    row["likelihood_identity_sha256"] = sha256_json(likelihood_identity)
                    all_forecasts.append(row)

            if cutoff_index == 1 or cutoff_index % 4 == 0:
                print(
                    f"Issue 212: {season} cutoff {cutoff_index}/{len(season_plan['cutoffs_utc'])}",
                    flush=True,
                )

    common_cohorts = _validate_common_forecasts(all_forecasts)
    paired = _paired_rows(all_forecasts)
    forecast_summaries = _add_phase_pooled_summaries(
        all_forecasts,
        (
            summarize_forecast_scores(
                all_forecasts, stage="rolling", population=PRIMARY_POPULATION
            )
            + summarize_forecast_scores(
                all_forecasts, stage="rolling", population=SECONDARY_POPULATION
            )
            + summarize_forecast_scores(
                all_forecasts, stage="preseason", population=PRIMARY_POPULATION
            )
            + summarize_forecast_scores(
                all_forecasts, stage="preseason", population=SECONDARY_POPULATION
            )
        ),
    )
    decision = _decision(forecast_summaries)
    rank_summaries = _rank_summary(all_rank_scores)
    parity = _verify_baseline_parity(
        all_forecasts, issue201_root / "forecast_scores.csv"
    )

    issue204_manifest = json.loads(issue204_manifest_path.read_text(encoding="utf-8"))
    issue201_input_hashes = {
        name: _input_hash_record(path, name)
        for name, path in {
            "issue201_schedule_plan": issue201_plan_path,
            "issue201_schedule_plan_manifest": issue201_plan_manifest_path,
            "issue201_manifest": issue201_manifest_path,
            "issue201_forecast_scores": issue201_root / "forecast_scores.csv",
            "issue200_pipeline_bundle": issue200_bundle_path,
            "issue200_manifest": issue200_manifest_path,
            "issue200_fitted_models": issue200_fitted_models_path,
            "issue204_target_file": target_path,
            "issue204_manifest": issue204_manifest_path,
            "issue200_target_snapshot_manifest": target_snapshot_manifest_path,
            "issue198_schedule_windows": schedule_windows_path,
            "protocol": protocol_path,
            "issue195_fit_records": issue195_root / "fit_records.json",
            "issue195_manifest": issue195_root / "manifest.json",
        }.items()
    }
    return {
        "forecast_scores": all_forecasts,
        "forecast_summaries": forecast_summaries,
        "paired_game_differences": paired,
        "rank_scores": all_rank_scores,
        "rank_summaries": rank_summaries,
        "posterior_pmf_rows": all_pmf_rows,
        "evidence_rows": evidence_rows,
        "convergence_diagnostics": convergence_rows,
        "forecast_exclusions": forecast_exclusions,
        "rank_exclusions": rank_exclusions,
        "gain_concentration": _gain_concentration(paired),
        "decision": decision,
        "baseline_parity": parity,
        "common_cohorts": common_cohorts,
        "baseline_pmf_hashes": baseline_pmf_hashes,
        "candidate_pmf_hashes": candidate_pmf_hashes,
        "fit_provenance": fit_provenance,
        "issue195_verified_inputs": issue195_inputs,
        "issue200_provenance": issue200_provenance,
        "target_provenance": target_provenance,
        "raw_game_files": game_source_files,
        "input_hashes": issue201_input_hashes,
        "protocol_sha256": protocol_sha,
        "issue204_target_support": {
            "status": issue204_manifest.get("status"),
            "downstream_ready": issue204_manifest.get("downstream_ready"),
            "support_validation": issue204_manifest.get("target_support_validation"),
        },
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle,
    ):
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=_json_default,
                ).encode()
                + b"\n"
            )


def _render_report(result: dict[str, object]) -> str:
    decision = result["decision"]
    parity = result["baseline_parity"]
    summaries = result["forecast_summaries"]
    rank_summaries = result["rank_summaries"]
    assert (
        isinstance(decision, dict)
        and isinstance(parity, dict)
        and isinstance(summaries, list)
        and isinstance(rank_summaries, list)
    )
    find = lambda arm, stage, season, phase="all_phases": next(
        row
        for row in summaries
        if row["model_arm"] == arm
        and row["forecast_stage"] == stage
        and row["forecast_population"] == PRIMARY_POPULATION
        and str(row["season"]) == str(season)
        and row["phase"] == phase
    )
    find_rank = lambda arm, aggregation, season: next(
        row
        for row in rank_summaries
        if row["model_arm"] == arm
        and row["aggregation"] == aggregation
        and str(row["season"]) == str(season)
    )
    lines = [
        "# Issue 212: generic likelihood recalibration",
        "",
        f"**Conclusion:** {decision['conclusion']}",
        "",
        "This is a fixed, exploratory replay of the verified #195 generic calibration fits. It does not establish an untouched prospective confirmation or authorize production promotion.",
        "",
        "## Primary FBS–FBS future-game CRPS",
        "",
        "| Evaluation | Unmodified Context | Recalibrated Context | Candidate minus baseline | Games |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, season, phase, label in (
        ("rolling", "pooled_equal_game", "all_phases", "Pooled equal game"),
        ("rolling", "equal_season_sensitivity", "all_phases", "Equal season"),
        *(
            ("rolling", str(season), "all_phases", str(season))
            for season in EVALUATION_SEASONS
        ),
        ("preseason", "pooled_equal_game", "all_phases", "Preseason pooled equal game"),
        ("rolling", "pooled_equal_game", "pre_thanksgiving", "Pre-Thanksgiving pooled"),
    ):
        try:
            baseline = find(BASELINE_ARM, stage, season, phase)
            candidate = find(CANDIDATE_ARM, stage, season, phase)
        except StopIteration:
            continue
        lines.append(
            f"| {label} | {float(baseline['crps']):.4f} | {float(candidate['crps']):.4f} | {float(candidate['crps']) - float(baseline['crps']):+.4f} | {baseline['games']} |"
        )
    lines.extend(
        [
            "",
            "The fixed gain rule requires at least 0.05 lower CRPS in both pooled equal-game and equal-season summaries, with gains in at least two of three seasons. The separate preseason and pre-Thanksgiving guardrails apply only to a candidate that clears that primary rule.",
            "",
            "## Decision and parity",
            "",
            f"- Primary practical-gain rule passed: `{decision['primary_rule']['pass']}`.",
            f"- Early guardrails passed: `{decision['early_guardrails_pass']}`.",
            f"- #201 baseline parity passed: `{parity['passed']}` across {parity['forecast_rows_compared']} per-game forecasts; maximum metric difference `{parity['max_absolute_metric_difference']:.3g}` against tolerance `{parity['tolerance']}`.",
            f"- Forecast-cohort audit: `{json.dumps(result['common_cohorts'], sort_keys=True)}`.",
            "",
            "## Season-specific calibration parameters",
            "",
            "| Season | Mean multiplier | Extra non-neutral home term | Log scale correction |",
            "|---:|---:|---:|---:|",
        ]
    )
    fit_provenance = result["fit_provenance"]
    assert isinstance(fit_provenance, dict)
    records = fit_provenance["fit_record_by_season"]
    assert isinstance(records, dict)
    for season, row in records.items():
        lines.append(
            f"| {season} | {row['overall_mean_multiplier']:.6f} | {row['additional_non_neutral_home_field_points']:.6f} | {row['common_log_scale_correction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary rank results",
            "",
            "Corrected pre-bowl rank scores use only checkpoints strictly before the target observation date. Each team-season contributes total weight one across its valid checkpoints.",
            "",
            "| Evaluation | Rank CRPS (baseline → recalibrated) | Δ NLL | Δ expected-rank error | Δ 80% target mass | Δ 80% interval width |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for aggregation, season, label in (
        ("pooled_equal_team_season", "2023-2025", "In-season pooled"),
        *(
            ("season_team_season_weight_one", str(season), str(season))
            for season in EVALUATION_SEASONS
        ),
        (
            "pooled_preseason_equal_team_season",
            "2023-2025",
            "Preseason pooled",
        ),
        *(
            ("season_preseason_team_season", str(season), f"{season} preseason")
            for season in EVALUATION_SEASONS
        ),
    ):
        baseline = find_rank(BASELINE_ARM, aggregation, season)
        candidate = find_rank(CANDIDATE_ARM, aggregation, season)
        lines.append(
            f"| {label} | {float(baseline['crps']):.5f} → {float(candidate['crps']):.5f} | {float(candidate['nll']) - float(baseline['nll']):+.5f} | {float(candidate['expected_rank_error']) - float(baseline['expected_rank_error']):+.4f} | {float(candidate['interval_80_target_mass']) - float(baseline['interval_80_target_mass']):+.4f} | {float(candidate['interval_80_width']) - float(baseline['interval_80_width']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "In-season aggregates exclude preseason checkpoints; preseason scores are shown separately. Each aggregate gives every team-season equal weight after averaging that team's valid checkpoints.",
            "",
            "## Pooled game calibration tradeoffs",
            "",
            "Changes are recalibrated minus baseline. The 80% coverage target is 0.80.",
            "",
            "| Evaluation | Δ margin NLL | Δ win Brier | Δ median absolute error | Δ 80% coverage | Δ 80% interval width |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, season, phase, label in (
        ("rolling", "pooled_equal_game", "all_phases", "Rolling all phases"),
        ("preseason", "pooled_equal_game", "all_phases", "Preseason"),
        ("rolling", "pooled_equal_game", "pre_thanksgiving", "Pre-Thanksgiving"),
    ):
        try:
            baseline = find(BASELINE_ARM, stage, season, phase)
            candidate = find(CANDIDATE_ARM, stage, season, phase)
        except StopIteration:
            continue
        metrics = (
            "margin_nll",
            "home_win_brier",
            "median_absolute_error",
            "coverage_80",
            "margin_interval_80_width",
        )
        deltas = [float(candidate[key]) - float(baseline[key]) for key in metrics]
        lines.append(
            f"| {label} | {deltas[0]:+.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} | {deltas[3]:+.4f} | {deltas[4]:+.3f} |"
        )
    lines.extend(
        [
            "",
            "`forecast_summaries.csv` contains level metrics plus season-, phase-, and population-specific values. FBS–FCS forecasts remain a separate population and use the unchanged conditional likelihood; their FBS rank posterior can change indirectly through FBS–FBS evidence.",
            "",
            "## Limitations and next confirmation",
            "",
            "The 2022–2025 seasons informed prior development and are not fresh confirmation. A promising result needs a separately registered future-season replay with model settings and predictions frozen before outcomes are available. Repeated teams, games, and checkpoints are not independent observations.",
            "",
            "The #201 artifact did not retain per-cutoff posterior PMFs. This run regenerates the baseline and candidate PMFs from the hash-verified #200 bundle and stores both losslessly in `posterior_pmfs.jsonl.gz`; baseline metric parity against #201 is checked per game.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(result: dict[str, object], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    csv_outputs = {
        "forecast_scores.csv": result["forecast_scores"],
        "forecast_summaries.csv": result["forecast_summaries"],
        "paired_game_differences.csv": result["paired_game_differences"],
        "rank_scores.csv": result["rank_scores"],
        "rank_summaries.csv": result["rank_summaries"],
        "convergence_diagnostics.csv": result["convergence_diagnostics"],
        "forecast_exclusions.csv": result["forecast_exclusions"],
        "rank_exclusions.csv": result["rank_exclusions"],
        "gain_concentration.csv": result["gain_concentration"],
    }
    for name, rows in csv_outputs.items():
        path = output_root / name
        write_csv(path, rows)  # type: ignore[arg-type]
        outputs[name] = path
    for name, key in (
        ("evidence.jsonl.gz", "evidence_rows"),
        ("posterior_pmfs.jsonl.gz", "posterior_pmf_rows"),
    ):
        path = output_root / name
        write_jsonl_gzip(path, result[key])  # type: ignore[arg-type]
        outputs[name] = path
    for name, key in (
        ("decision.json", "decision"),
        ("baseline_parity.json", "baseline_parity"),
        ("fit_provenance.json", "fit_provenance"),
        ("input_hashes.json", "input_hashes"),
    ):
        path = output_root / name
        write_json(path, result[key])
        outputs[name] = path
    report = output_root / "report.md"
    report.write_text(_render_report(result), encoding="utf-8")
    outputs[report.name] = report
    output_records = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(outputs.items())
    }
    manifest = {
        "schema_version": 1,
        "experiment": "issue-212-generic-likelihood-recalibration",
        "status": "complete",
        "conclusion": result["decision"]["conclusion"],  # type: ignore[index]
        "seasons": list(EVALUATION_SEASONS),
        "arms": [BASELINE_ARM, CANDIDATE_ARM],
        "protocol_sha256": result["protocol_sha256"],
        "issue195_fit_provenance": result["fit_provenance"],
        "issue195_verified_inputs": result["issue195_verified_inputs"],
        "issue200_provenance": result["issue200_provenance"],
        "corrected_target_provenance": result["target_provenance"],
        "issue204_support_status": result["issue204_target_support"],
        "raw_game_files": result["raw_game_files"],
        "input_hashes": result["input_hashes"],
        "common_cohorts": result["common_cohorts"],
        "baseline_parity": result["baseline_parity"],
        "posterior_pmf_hashes": {
            "baseline": result["baseline_pmf_hashes"],
            "candidate": result["candidate_pmf_hashes"],
        },
        "decision": result["decision"],
        "outputs": output_records,
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def write_blocked_result(
    *, output_root: Path, reason: str, input_roots: dict[str, Path]
) -> dict[str, object]:
    """Record a failed input, inference, or parity gate without a result claim."""
    output_root.mkdir(parents=True, exist_ok=True)
    root_labels = {
        name: f"<external>/{path.name}" if path.is_absolute() else str(path)
        for name, path in input_roots.items()
    }
    gate = {
        "status": "blocked",
        "scientific_decision": None,
        "reason": reason,
        "required_fit_records_sha256": EXPECTED_FIT_RECORD_SHA256,
        "input_roots": root_labels,
    }
    gate_path = output_root / "input_gate.json"
    write_json(gate_path, gate)
    report_path = output_root / "report.md"
    report_path.write_text(
        "# Issue 212: replay incomplete\n\n"
        "No candidate scores were interpreted because a frozen input, inference, "
        "or parity check failed. This is not a negative or inconclusive scientific "
        "result. Restore the required artifact through the documented workflow, "
        "verify its hash, and rerun the fixed protocol.\n\n"
        f"Blocking reason: `{reason}`\n",
        encoding="utf-8",
    )
    output_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (gate_path, report_path)
    }
    manifest = {
        "schema_version": 1,
        "experiment": "issue-212-generic-likelihood-recalibration",
        "status": "incomplete",
        "scientific_decision": None,
        "blocking_reason": reason,
        "input_roots": gate["input_roots"],
        "required_fit_records_sha256": EXPECTED_FIT_RECORD_SHA256,
        "outputs": output_records,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
