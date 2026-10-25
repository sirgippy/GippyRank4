"""Build the reproducible issue 192 Context incremental-information benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_context_history_crossover as crossover
import study_history_context_posterior as historical

from gippyrank.context_db_repair import (
    H_FEATURES,
    MODEL_FEATURES,
    historical_repaired_features,
    production_pmf,
)
from gippyrank.posterior.engine import Team, infer_posterior
from gippyrank.posterior.snapshots import (
    _scheduled_future_fcs_rows,
    add_fcs_fallbacks,
    filter_games,
    load_pinned_likelihood,
)
from gippyrank.preseason import DirectRankModel, Preprocessor
from gippyrank.research.context_incremental_information import (
    LAMBDA_GRID,
    ProbeExample,
    cross_message_arms,
    fit_probe,
    kl_posterior_from_prior,
    normalized_pmf,
    predict_probe,
    select_regularization,
    validate_evidence_parity,
    write_deterministic_gzip_jsonl,
)

SEASONS = (2022, 2023, 2024, 2025)
REPRODUCTION_CHECKPOINTS = (1, 2, 7)
ALL_CHECKPOINTS = tuple(range(1, 8))
INFERENCE = {"max_iterations": 500, "tolerance": 1e-9, "damping": 0.35}
SCORE_METRICS = (
    "nll",
    "crps",
    "expected_rank",
    "expected_rank_error",
    "interval_80_width",
    "interval_80_target_mass",
)
BASELINE_SCORE_TOLERANCE = 1e-8
PILOT_SCORE_TOLERANCE = 1e-5
PILOT_PROTOCOL_SHA256 = (
    "327d2d301d59668898dff89046e617ddfc5f0d36dd42e282d2fecd71b0d67101"
)
EXPECTED_PILOT_NLL = {
    1: {
        "history_posterior": 4.54758,
        "context_posterior": 4.51459,
        "history_calibration": 4.55202,
        "existing_context_signal": 4.52486,
        "raw_context_features": 4.54271,
    },
    2: {
        "history_posterior": 4.36388,
        "context_posterior": 4.38336,
        "history_calibration": 4.36007,
        "existing_context_signal": 4.35228,
        "raw_context_features": 4.36374,
    },
    7: {
        "history_posterior": 3.83089,
        "context_posterior": 3.88793,
        "history_calibration": 3.81017,
        "existing_context_signal": 3.81065,
        "raw_context_features": 3.81333,
    },
}
EXPECTED_DECEMBER_CROSSES = {
    "HH": 3.83089,
    "CH": 3.85330,
    "HC": 3.85739,
    "CC": 3.88793,
}
CONTROL_FEATURES = (
    "mean_lag1_z",
    "lag2_z_mean",
    "lag3_z_mean",
    "long_run_z_mean",
    "history_posterior_expected_rank_percentile",
    "history_posterior_rank_sd_ratio",
)
RAW_CONTEXT_FEATURES = tuple(name for name in MODEL_FEATURES if name not in H_FEATURES)
PROBE_FEATURES = {
    "history_calibration": CONTROL_FEATURES,
    "existing_context_signal": CONTROL_FEATURES,
    "raw_context_features": (*CONTROL_FEATURES, *RAW_CONTEXT_FEATURES),
}
PROBE_SIGNAL = {
    "history_calibration": False,
    "existing_context_signal": True,
    "raw_context_features": False,
}
MODEL_NAME_BY_KIND = {
    "history_calibration": "History calibration control",
    "existing_context_signal": "Control + existing Context signal",
    "raw_context_features": "Control + raw Context features",
}
CROSS_ARM_LABELS = {
    "HH": "History prior × History message",
    "CH": "Context prior × History message",
    "HC": "History prior × Context message",
    "CC": "Context prior × Context message",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write stable gzip JSONL with no wall-clock timestamp in the header."""
    write_deterministic_gzip_jsonl(path, rows)


def checked_hash(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {label} ({path.name})")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"retained hash mismatch for {label}: expected {expected}, observed {actual}"
        )
    return actual


def verify_repair_inputs(input_root: Path, repo_root: Path) -> dict[str, str]:
    """Verify every retained source hash used to repair Context 1.4 inputs."""
    comparison = read_json(
        repo_root / "data/processed/context_db_repair_172/model_comparison.json"
    )
    expected = comparison["source_sha256"]
    actual: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        if relative == "data/processed/modeling/team_season_rank_distributions.csv":
            continue
        path = input_root / relative
        actual[relative] = checked_hash(path, str(digest), label=relative)
    return actual


def verify_scoring_target_source(repo_root: Path, target_path: Path) -> str:
    """Verify the target corpus only after target-free inference is complete."""
    comparison = read_json(
        repo_root / "data/processed/context_db_repair_172/model_comparison.json"
    )
    relative = "data/processed/modeling/team_season_rank_distributions.csv"
    return checked_hash(
        target_path, str(comparison["source_sha256"][relative]), label=relative
    )


def verify_crossover_sources(
    repo_root: Path,
    raw_games: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Verify retained History/evidence artifacts and the raw game cache."""
    provenance = read_json(
        repo_root / "data/processed/context_positive_net_moderation/provenance.json"
    )
    recorded = provenance["source_hashes"]
    actual: dict[str, str] = {}
    raw_expected: dict[str, str] = {}
    for logical, digest in sorted(recorded.items()):
        if logical.startswith("raw_games/"):
            filename = Path(logical).name
            raw_expected[filename] = str(digest)
            actual[logical] = checked_hash(
                raw_games / filename, str(digest), label=logical
            )
        elif logical == "external_input/team_season_rank_distributions.csv":
            continue
        elif logical.startswith("data/"):
            actual[logical] = checked_hash(
                repo_root / logical, str(digest), label=logical
            )
    return actual, raw_expected


def _direct_rank_model(metadata: dict[str, Any]) -> DirectRankModel:
    preprocessing = metadata["preprocessing"]
    preprocessor = Preprocessor(
        tuple(preprocessing["feature_names"]),
        dict(preprocessing["medians"]),
        dict(preprocessing["means"]),
        dict(preprocessing["scales"]),
    )
    return DirectRankModel(
        feature_names=list(metadata["feature_names"]),
        preprocessor=preprocessor,
        beta=np.asarray(metadata["location_coefficients"], dtype=float),
        gamma=np.asarray(metadata["log_scale_coefficients"], dtype=float),
        minimum_scale=float(metadata["minimum_scale"]),
        penalty=float(metadata["penalty"]),
        optimizer=metadata.get("optimizer"),
        lag_count=int(metadata["lag_count"]),
        family=str(metadata["family"]),
        degrees_of_freedom=metadata.get("degrees_of_freedom"),
        location_feature_names=list(metadata["location_feature_names"]),
        scale_feature_names=list(metadata["scale_feature_names"]),
    )


def load_context_priors(
    repo_root: Path,
    input_root: Path,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, dict[str, float | None]]],
    dict[int, dict[str, str]],
    dict[tuple[int, str], int],
    dict[str, str],
    float,
]:
    """Reconstruct repaired Context 1.4 priors from retained model/input rows."""
    repair_hashes = verify_repair_inputs(input_root, repo_root)
    fitted_input = (
        repo_root
        / "data/processed/context_v1_4_candidate/development_model_inputs.json"
    )
    inputs_hash = sha256_file(fitted_input)
    model_inputs = read_json(fitted_input)
    if model_inputs.get("format_version") != 1:
        raise ValueError("unsupported Context 1.4 development-input schema")
    repaired_features, _ = historical_repaired_features(input_root)
    priors: dict[int, dict[str, np.ndarray]] = {}
    features_by_season: dict[int, dict[str, dict[str, float | None]]] = {}
    names_by_season: dict[int, dict[str, str]] = {}
    populations: dict[tuple[int, str], int] = {}
    max_error = 0.0
    fitted_count = 0

    for season in SEASONS:
        season_text = str(season)
        block = model_inputs["seasons"][season_text]
        model_path = (
            repo_root
            / f"data/processed/context_db_repair_172/historical_models/{season}.json"
        )
        model_record = read_json(model_path)
        if (
            int(model_record["target_season"]) != season
            or int(model_record["trained_through_season"]) != season - 1
            or int(block["fitted_instance"]["trained_through_season"]) != season - 1
        ):
            raise ValueError(f"{season}: repaired Context rolling origin changed")
        model = _direct_rank_model(model_record["model"])
        if tuple(model.feature_names) != tuple(MODEL_FEATURES):
            raise ValueError(f"{season}: repaired Context feature contract changed")

        retained_rows = read_csv(
            repo_root
            / f"data/processed/context_db_repair_172/historical_priors/{season}.csv"
        )
        retained: dict[str, np.ndarray] = {}
        names: dict[str, str] = {}
        for row in retained_rows:
            team_id = str(row["team_id"])
            if team_id in retained:
                raise ValueError(
                    f"{season}: duplicate repaired Context prior {team_id}"
                )
            retained[team_id] = normalized_pmf(
                json.loads(row["pmf"]), name=f"{season} Context prior {team_id}"
            )
            names[team_id] = row["team_name"]

        season_features: dict[str, dict[str, float | None]] = {}
        reconstructed_ids: set[str] = set()
        for row in block["fitted_rows"]:
            team_id = str(row["team_id"])
            key = (season, team_id)
            if key not in repaired_features:
                raise ValueError(f"{season} {team_id}: repaired feature row is missing")
            features = dict(row["features"])
            features.pop("transfer_in_prior_defensive_impact_db_sum", None)
            features.pop("transfer_in_prior_defensive_impact_db_available", None)
            features.update(repaired_features[key])
            features = {name: features.get(name) for name in MODEL_FEATURES}
            if set(features) != set(MODEL_FEATURES):
                raise ValueError(f"{season} {team_id}: feature boundary is incomplete")
            pmf, _, _ = production_pmf(
                model,
                features,
                np.asarray(row["lag1_z"], dtype=float),
                int(row["population"]),
            )
            pmf = normalized_pmf(pmf, name=f"{season} reconstructed Context {team_id}")
            if team_id not in retained:
                raise ValueError(
                    f"{season} {team_id}: reconstructed prior is unretained"
                )
            error = float(np.max(np.abs(pmf - retained[team_id])))
            max_error = max(max_error, error)
            if error > 1e-12:
                raise ValueError(
                    f"{season} {team_id}: repaired fitted prior differs by {error}"
                )
            reconstructed_ids.add(team_id)
            season_features[team_id] = {
                **features,
                "mean_lag1_z": float(np.mean(np.asarray(row["lag1_z"], dtype=float))),
            }
            populations[season, team_id] = int(row["population"])
            fitted_count += 1

        if len(reconstructed_ids) != len(block["fitted_rows"]):
            raise ValueError(f"{season}: duplicate fitted Context rows")
        if (
            len(retained)
            != len(reconstructed_ids)
            + {
                2022: 1,
                2023: 2,
                2024: 1,
                2025: 2,
            }[season]
        ):
            raise ValueError(f"{season}: native Context fallback population changed")
        for team_id, retained_pmf in retained.items():
            if team_id not in reconstructed_ids:
                season_features[team_id] = {
                    **{name: None for name in MODEL_FEATURES},
                    "mean_lag1_z": None,
                }
                populations[season, team_id] = len(retained_pmf)
        priors[season] = retained
        features_by_season[season] = season_features
        names_by_season[season] = names

    if fitted_count != 528 or max_error > 1e-12:
        raise ValueError(
            f"repaired Context prior reconstruction failed: {fitted_count} rows, error {max_error}"
        )
    source_hashes = {
        **repair_hashes,
        "data/processed/context_v1_4_candidate/development_model_inputs.json": inputs_hash,
    }
    for season in SEASONS:
        for path in (
            repo_root
            / f"data/processed/context_db_repair_172/historical_models/{season}.json",
            repo_root
            / f"data/processed/context_db_repair_172/historical_priors/{season}.csv",
        ):
            source_hashes[path.relative_to(repo_root).as_posix()] = sha256_file(path)
    return (
        priors,
        features_by_season,
        names_by_season,
        populations,
        source_hashes,
        max_error,
    )


def load_history_priors(
    repo_root: Path,
    context_priors: dict[int, dict[str, np.ndarray]],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, str]],
    dict[int, dict[str, str]],
    dict[str, str],
]:
    rows = read_csv(
        repo_root / "data/processed/context_history_crossover/hybrid_prior_results.csv"
    )
    history_priors: dict[int, dict[str, np.ndarray]] = {
        season: {} for season in SEASONS
    }
    statuses: dict[int, dict[str, str]] = {season: {} for season in SEASONS}
    names: dict[int, dict[str, str]] = {season: {} for season in SEASONS}
    for row in rows:
        if row["arm"] != "HH":
            continue
        season = int(row["season"])
        team_id = str(row["team_id"])
        if season not in history_priors or team_id in history_priors[season]:
            raise ValueError(
                f"duplicate or unplanned History prior: {season} {team_id}"
            )
        raw_pmf = np.asarray(json.loads(row["prior_pmf"]), dtype=float)
        if crossover.sha256_json(raw_pmf.tolist()) != row["prior_pmf_sha256"]:
            raise ValueError(f"{season} {team_id}: retained History PMF hash mismatch")
        pmf = normalized_pmf(raw_pmf, name=f"{season} History prior {team_id}")
        if team_id not in context_priors[season]:
            raise ValueError(f"{season} {team_id}: History/Context populations differ")
        if len(pmf) != len(context_priors[season][team_id]):
            raise ValueError(
                f"{season} {team_id}: History/Context rank support differs"
            )
        history_priors[season][team_id] = pmf
        statuses[season][team_id] = row["component_status"]
        names[season][team_id] = row["team_name"]
    for season in SEASONS:
        if set(history_priors[season]) != set(context_priors[season]):
            raise ValueError(f"{season}: retained rolling History population differs")
    source_hash = sha256_file(
        repo_root / "data/processed/context_history_crossover/hybrid_prior_results.csv"
    )
    return (
        history_priors,
        statuses,
        names,
        {
            "data/processed/context_history_crossover/hybrid_prior_results.csv": source_hash
        },
    )


def _team_id_sort_key(team_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(team_id))
    except ValueError:
        return (1, team_id)


def _expected_evidence(
    repo_root: Path,
) -> tuple[dict[tuple[int, int, str], dict[str, str]], dict[int, int]]:
    rows = read_csv(
        repo_root
        / "data/processed/context_history_crossover/arm_evidence_and_convergence.csv"
    )
    expected = {
        (int(row["season"]), int(row["checkpoint"]), row["arm"]): row
        for row in rows
        if row["arm"] in {"CC", "HH"}
    }
    fcs_support = {}
    for season in SEASONS:
        context = expected[season, 1, "CC"]
        history = expected[season, 1, "HH"]
        if context["fcs_fallback_count"] != history["fcs_fallback_count"]:
            raise ValueError(f"{season}: preseason History/Context FCS support differs")
        fcs_support[season] = int(context["fcs_fallback_count"])
    if len(expected) != len(SEASONS) * len(ALL_CHECKPOINTS) * 2:
        raise ValueError("retained arm evidence does not cover all planned checkpoints")
    return expected, fcs_support


def _inference_inputs(
    priors: dict[str, np.ndarray],
    names: dict[str, str],
    *,
    subdivision: str,
    included: list[dict[str, str]],
    fcs_population: int,
    future_fcs: list[dict[str, str]],
) -> tuple[list[Team], set[str]]:
    teams = [
        Team(team_id, names[team_id], subdivision, pmf.copy())
        for team_id, pmf in sorted(
            priors.items(), key=lambda item: _team_id_sort_key(item[0])
        )
    ]
    metadata = {team_id: {"team_name": name} for team_id, name in names.items()}
    return add_fcs_fallbacks(teams, metadata, included, fcs_population, future_fcs)


def build_inference_panel(
    *,
    repo_root: Path,
    game_root: Path,
    context_priors: dict[int, dict[str, np.ndarray]],
    context_features: dict[int, dict[str, dict[str, float | None]]],
    context_names: dict[int, dict[str, str]],
    history_priors: dict[int, dict[str, np.ndarray]],
    history_statuses: dict[int, dict[str, str]],
    evidence_checkpoints: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    """Infer both baseline arms without accepting or loading outcome targets."""
    if crossover.INFERENCE != INFERENCE:
        raise ValueError("posterior inference settings differ from issue 192 protocol")
    expected, fcs_support = _expected_evidence(repo_root)
    likelihood_path = (
        repo_root / "data/processed/posterior/historical_likelihood_v1.json"
    )
    likelihood = load_pinned_likelihood(likelihood_path)
    panel_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    message_errors = {
        "max_history_identity_error": 0.0,
        "max_context_identity_error": 0.0,
    }

    for season in SEASONS:
        points = crossover.checkpoint_inputs(game_root, season)
        if tuple(int(point["checkpoint"]) for point in points) != ALL_CHECKPOINTS:
            raise ValueError(f"{season}: retained cutoff checkpoints changed")
        for point in points:
            checkpoint = int(point["checkpoint"])
            if checkpoint not in evidence_checkpoints:
                continue
            cutoff = datetime.fromisoformat(str(point["cutoff"]))
            games, included, excluded_lower, _ = filter_games(
                game_root, season, cutoff, "weekly"
            )
            games, included, cutoff_audit = historical.cutoff_safe_games(
                games, included, cutoff
            )
            future_fcs = _scheduled_future_fcs_rows(game_root, season, cutoff, "weekly")
            cc = expected[season, checkpoint, "CC"]
            hh = expected[season, checkpoint, "HH"]
            same_fields = (
                "cutoff",
                "included_games",
                "included_game_ids_sha256",
                "included_game_rows_sha256",
                "excluded_near_cutoff_games",
                "excluded_lower_division_games",
                "fcs_fallback_count",
                "fcs_fallback_ids_sha256",
                "total_inference_team_count",
                "likelihood_sha256",
                "inference_max_iterations",
                "inference_tolerance",
                "inference_damping",
            )
            included_ids = tuple(sorted((str(row["id"]) for row in included), key=int))
            observed = {
                "cutoff": cutoff.isoformat(),
                "included_games": str(len(included)),
                "included_game_ids_sha256": crossover.sha256_json(included_ids),
                "included_game_rows_sha256": crossover.stable_evidence_hash(included),
                "excluded_near_cutoff_games": str(len(cutoff_audit)),
                "excluded_lower_division_games": str(excluded_lower),
                "likelihood_sha256": sha256_file(likelihood_path),
            }
            try:
                validate_evidence_parity(cc, hh, observed, tuple(observed))
            except ValueError as error:
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: {error}"
                ) from error

            c_teams, c_fallback_ids = _inference_inputs(
                context_priors[season],
                context_names[season],
                subdivision="fbs",
                included=included,
                fcs_population=fcs_support[season],
                future_fcs=future_fcs,
            )
            h_teams, h_fallback_ids = _inference_inputs(
                history_priors[season],
                {
                    team_id: context_names[season][team_id]
                    for team_id in history_priors[season]
                },
                subdivision="fbs",
                included=included,
                fcs_population=fcs_support[season],
                future_fcs=future_fcs,
            )
            fallback_hash = crossover.sha256_json(sorted(c_fallback_ids))
            if (
                c_fallback_ids != h_fallback_ids
                or str(len(c_fallback_ids)) != cc["fcs_fallback_count"]
                or fallback_hash != cc["fcs_fallback_ids_sha256"]
                or str(len(c_teams)) != cc["total_inference_team_count"]
                or str(len(h_teams)) != cc["total_inference_team_count"]
            ):
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: FCS fallbacks or population differ"
                )
            replayed_evidence = {
                **observed,
                "fcs_fallback_count": str(len(c_fallback_ids)),
                "fcs_fallback_ids_sha256": fallback_hash,
                "total_inference_team_count": str(len(c_teams)),
                "inference_max_iterations": str(INFERENCE["max_iterations"]),
                "inference_tolerance": str(INFERENCE["tolerance"]),
                "inference_damping": str(INFERENCE["damping"]),
            }
            try:
                validate_evidence_parity(cc, hh, replayed_evidence, same_fields)
            except ValueError as error:
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: {error}"
                ) from error
            c_result = infer_posterior(c_teams, games, likelihood, **INFERENCE)
            h_result = infer_posterior(h_teams, games, likelihood, **INFERENCE)
            for arm, result in (("Context", c_result), ("History", h_result)):
                if not result.converged or set(result.pmfs) != {
                    team.team_id for team in c_teams
                }:
                    raise RuntimeError(
                        f"{season} checkpoint {checkpoint}: {arm} posterior did not converge"
                    )

            c_team_by_id = {team.team_id: team for team in c_teams}
            h_team_by_id = {team.team_id: team for team in h_teams}
            if set(c_team_by_id) != set(h_team_by_id):
                raise ValueError(
                    "Context and History inference team populations differ"
                )
            game_counts: dict[str, int] = defaultdict(int)
            for row in included:
                game_counts[str(row["homeId"])] += 1
                game_counts[str(row["awayId"])] += 1

            fitted_feature_rows = context_features[season]
            crosses_by_id: dict[str, dict[str, np.ndarray]] = {}
            for team_id in context_priors[season]:
                c_prior = c_team_by_id[team_id].prior
                h_prior = h_team_by_id[team_id].prior
                c_post = normalized_pmf(
                    c_result.pmfs[team_id],
                    name=f"{season} {checkpoint} Context posterior",
                )
                h_post = normalized_pmf(
                    h_result.pmfs[team_id],
                    name=f"{season} {checkpoint} History posterior",
                )
                crosses_by_id[team_id], errors = cross_message_arms(
                    c_prior, h_prior, c_post, h_post
                )
                message_errors["max_context_identity_error"] = max(
                    message_errors["max_context_identity_error"],
                    errors["context_identity_error"],
                )
                message_errors["max_history_identity_error"] = max(
                    message_errors["max_history_identity_error"],
                    errors["history_identity_error"],
                )

            for team_id in sorted(context_priors[season], key=_team_id_sort_key):
                c_prior = c_team_by_id[team_id].prior
                h_prior = h_team_by_id[team_id].prior
                c_post = normalized_pmf(
                    c_result.pmfs[team_id], name="Context posterior"
                )
                h_post = normalized_pmf(
                    h_result.pmfs[team_id], name="History posterior"
                )
                rank_count = len(h_prior)
                ranks = np.arange(1, rank_count + 1, dtype=float)
                history_expected = float(np.dot(ranks, h_post))
                history_sd = float(
                    np.sqrt(np.dot((ranks - history_expected) ** 2, h_post))
                )
                raw_features = fitted_feature_rows.get(
                    team_id, {name: None for name in MODEL_FEATURES}
                )
                control_features: dict[str, float | None] = {
                    "mean_lag1_z": raw_features.get("mean_lag1_z"),
                    "lag2_z_mean": raw_features.get("lag2_z_mean"),
                    "lag3_z_mean": raw_features.get("lag3_z_mean"),
                    "long_run_z_mean": raw_features.get("long_run_z_mean"),
                    "history_posterior_expected_rank_percentile": history_expected
                    / rank_count,
                    "history_posterior_rank_sd_ratio": history_sd / rank_count,
                }
                all_features = {
                    **{name: raw_features.get(name) for name in MODEL_FEATURES},
                    **control_features,
                }
                row: dict[str, object] = {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff": cutoff.isoformat(),
                    "subdivision": "fbs",
                    "team_id": team_id,
                    "team_name": context_names[season][team_id],
                    "component_status": history_statuses[season][team_id],
                    "rank_count": rank_count,
                    "games_played": game_counts.get(team_id, 0),
                    "history_kl_posterior_from_prior": kl_posterior_from_prior(
                        h_post, h_prior
                    ),
                    "context_prior_pmf": c_prior.tolist(),
                    "history_prior_pmf": h_prior.tolist(),
                    "context_posterior_pmf": c_post.tolist(),
                    "history_posterior_pmf": h_post.tolist(),
                    "crossed_message_pmfs": {
                        arm: pmf.tolist() for arm, pmf in crosses_by_id[team_id].items()
                    },
                    "features": all_features,
                }
                panel_rows.append(row)

            fcs_ids = sorted(c_fallback_ids, key=_team_id_sort_key)
            for team_id in fcs_ids:
                c_team = c_team_by_id[team_id]
                h_team = h_team_by_id[team_id]
                if not np.array_equal(c_team.prior, h_team.prior):
                    raise ValueError(
                        f"{season} {team_id}: FCS fallback prior differs by arm"
                    )
                c_post = normalized_pmf(
                    c_result.pmfs[team_id], name="FCS Context posterior"
                )
                h_post = normalized_pmf(
                    h_result.pmfs[team_id], name="FCS History posterior"
                )
                crosses, errors = cross_message_arms(
                    c_team.prior, h_team.prior, c_post, h_post
                )
                message_errors["max_context_identity_error"] = max(
                    message_errors["max_context_identity_error"],
                    errors["context_identity_error"],
                )
                message_errors["max_history_identity_error"] = max(
                    message_errors["max_history_identity_error"],
                    errors["history_identity_error"],
                )
                rank_count = len(h_team.prior)
                ranks = np.arange(1, rank_count + 1, dtype=float)
                history_expected = float(np.dot(ranks, h_post))
                history_sd = float(
                    np.sqrt(np.dot((ranks - history_expected) ** 2, h_post))
                )
                panel_rows.append(
                    {
                        "season": season,
                        "checkpoint": checkpoint,
                        "cutoff": cutoff.isoformat(),
                        "subdivision": "fcs",
                        "team_id": team_id,
                        "team_name": h_team.name,
                        "component_status": "fcs_uniform_fallback",
                        "rank_count": rank_count,
                        "games_played": game_counts.get(team_id, 0),
                        "history_kl_posterior_from_prior": kl_posterior_from_prior(
                            h_post, h_team.prior
                        ),
                        "context_prior_pmf": c_team.prior.tolist(),
                        "history_prior_pmf": h_team.prior.tolist(),
                        "context_posterior_pmf": c_post.tolist(),
                        "history_posterior_pmf": h_post.tolist(),
                        "crossed_message_pmfs": {
                            arm: pmf.tolist() for arm, pmf in crosses.items()
                        },
                        "features": {},
                    }
                )

            evidence_rows.append(
                {
                    "season": season,
                    "checkpoint": checkpoint,
                    "cutoff": cutoff.isoformat(),
                    "included_game_count": len(included),
                    "included_game_ids": list(included_ids),
                    "included_game_ids_sha256": observed["included_game_ids_sha256"],
                    "included_game_rows_sha256": observed["included_game_rows_sha256"],
                    "excluded_near_cutoff_games": len(cutoff_audit),
                    "excluded_lower_division_games": excluded_lower,
                    "fcs_rank_support": fcs_support[season],
                    "fcs_fallback_count": len(c_fallback_ids),
                    "fcs_fallback_ids": sorted(c_fallback_ids, key=_team_id_sort_key),
                    "fcs_fallback_ids_sha256": fallback_hash,
                    "total_inference_team_count": len(c_teams),
                    "likelihood_sha256": sha256_file(likelihood_path),
                    "inference": dict(INFERENCE),
                    "context_iterations": c_result.iterations,
                    "history_iterations": h_result.iterations,
                    "context_max_message_delta": c_result.max_message_delta,
                    "history_max_message_delta": h_result.max_message_delta,
                    "converged": bool(c_result.converged and h_result.converged),
                    "message_identity_max_error": max(
                        message_errors["max_context_identity_error"],
                        message_errors["max_history_identity_error"],
                    ),
                }
            )
            print(
                f"inferred {season} checkpoint {checkpoint}",
                flush=True,
            )
    panel_rows.sort(
        key=lambda row: (
            int(row["season"]),
            int(row["checkpoint"]),
            0 if row["subdivision"] == "fbs" else 1,
            _team_id_sort_key(str(row["team_id"])),
        )
    )
    evidence_rows.sort(key=lambda row: (int(row["season"]), int(row["checkpoint"])))
    return panel_rows, evidence_rows, message_errors


def load_scoring_targets(
    target_path: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, int]]:
    """Load scoring/training targets after inference has finished."""
    source_rows = read_csv(target_path)
    targets: dict[int, dict[str, np.ndarray]] = {}
    fcs_populations: dict[int, int] = {}
    for season in SEASONS:
        fbs_ids = [
            row["team_id"]
            for row in source_rows
            if int(row["season"]) == season and row["subdivision"] == "fbs"
        ]
        if len(fbs_ids) != len(set(fbs_ids)):
            raise ValueError(f"{season}: duplicate scoring target rows")
        season_targets = historical.load_targets(season, target_path)
        targets[season] = {
            str(team_id): normalized_pmf(pmf, name=f"{season} target {team_id}")
            for team_id, pmf in season_targets.items()
        }
        fcs_populations[season] = historical.fcs_population(season, target_path)
    return targets, fcs_populations


def validate_scoring_population(
    panel_rows: list[dict[str, object]],
    targets: dict[int, dict[str, np.ndarray]],
    expected_fcs_populations: dict[int, int],
    observed_fcs_populations: dict[int, int],
) -> None:
    for season in SEASONS:
        fbs_rows = {
            str(row["team_id"]): row
            for row in panel_rows
            if int(row["season"]) == season and row["subdivision"] == "fbs"
        }
        if set(fbs_rows) != set(targets[season]):
            raise ValueError(f"{season}: inference and scoring FBS populations differ")
        for team_id, row in fbs_rows.items():
            if int(row["rank_count"]) != len(targets[season][team_id]):
                raise ValueError(f"{season} {team_id}: target rank support differs")
        if expected_fcs_populations[season] != observed_fcs_populations[season]:
            raise ValueError(
                f"{season}: FCS rank support differs from retained evidence"
            )


def make_probe_examples(
    panel_rows: list[dict[str, object]],
    targets: dict[int, dict[str, np.ndarray]],
) -> dict[int, list[ProbeExample]]:
    examples: dict[int, list[ProbeExample]] = {season: [] for season in SEASONS}
    seen: set[tuple[int, str, int]] = set()
    for row in panel_rows:
        if row["subdivision"] != "fbs":
            continue
        season = int(row["season"])
        team_id = str(row["team_id"])
        checkpoint = int(row["checkpoint"])
        key = (season, team_id, checkpoint)
        if key in seen:
            raise ValueError(f"duplicate inference panel row: {key}")
        seen.add(key)
        if team_id not in targets[season]:
            raise ValueError(f"{season} {team_id}: scoring target is missing")
        features = row["features"]
        if not isinstance(features, dict):
            raise TypeError(f"{season} {team_id}: prediction features are malformed")
        examples[season].append(
            ProbeExample(
                season=season,
                team_id=team_id,
                team_name=str(row["team_name"]),
                history_posterior=np.asarray(row["history_posterior_pmf"], dtype=float),
                context_prior=np.asarray(row["context_prior_pmf"], dtype=float),
                history_prior=np.asarray(row["history_prior_pmf"], dtype=float),
                target=targets[season][team_id],
                features={
                    str(name): None if value is None else float(value)
                    for name, value in features.items()
                },
            )
        )
    for season in SEASONS:
        examples[season].sort(key=lambda example: _team_id_sort_key(example.team_id))
    return examples


def _team_score_row(
    *,
    season: int,
    checkpoint: int,
    team_id: str,
    team_name: str,
    component_status: str,
    games_played: int,
    history_kl: float,
    model: str,
    model_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    score_type: str,
) -> dict[str, object]:
    metrics = crossover.score_pmf(prediction, target)
    return {
        "season": season,
        "checkpoint": checkpoint,
        "team_id": team_id,
        "team_name": team_name,
        "component_status": component_status,
        "games_played": games_played,
        "history_kl_posterior_from_prior": history_kl,
        "score_type": score_type,
        "model": model,
        "model_name": model_name,
        **metrics,
    }


def score_panel(
    panel_rows: list[dict[str, object]],
    targets: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    for row in panel_rows:
        if row["subdivision"] != "fbs":
            continue
        season = int(row["season"])
        checkpoint = int(row["checkpoint"])
        team_id = str(row["team_id"])
        target = targets[season][team_id]
        predictions = {
            "history_posterior": row["history_posterior_pmf"],
            "context_posterior": row["context_posterior_pmf"],
            **row["crossed_message_pmfs"],
        }
        names = {
            "history_posterior": "History posterior",
            "context_posterior": "Context posterior",
            **CROSS_ARM_LABELS,
        }
        for model, prediction in predictions.items():
            score_type = "crossed_arm" if model in CROSS_ARM_LABELS else "baseline"
            scores.append(
                _team_score_row(
                    season=season,
                    checkpoint=checkpoint,
                    team_id=team_id,
                    team_name=str(row["team_name"]),
                    component_status=str(row["component_status"]),
                    games_played=int(row["games_played"]),
                    history_kl=float(row["history_kl_posterior_from_prior"]),
                    model=model,
                    model_name=names[model],
                    prediction=np.asarray(prediction, dtype=float),
                    target=target,
                    score_type=score_type,
                )
            )
    return scores


def fit_and_score_probes(
    panel_rows: list[dict[str, object]],
    targets: dict[int, dict[str, np.ndarray]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run outer forward-chained probe evaluation for one panel's checkpoints."""
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    panel_by_checkpoint: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in panel_rows:
        if row["subdivision"] == "fbs":
            panel_by_checkpoint[int(row["checkpoint"])].append(row)

    for checkpoint, checkpoint_rows in sorted(panel_by_checkpoint.items()):
        examples_by_season = make_probe_examples(checkpoint_rows, targets)
        for model, feature_names in PROBE_FEATURES.items():
            include_signal = PROBE_SIGNAL[model]
            for evaluation_season in SEASONS[1:]:
                training_seasons = tuple(
                    season for season in SEASONS if season < evaluation_season
                )
                training_examples = [
                    example
                    for season in training_seasons
                    for example in examples_by_season[season]
                ]
                training_examples_by_season = {
                    season: examples_by_season[season] for season in training_seasons
                }
                selected_lambda, selection = select_regularization(
                    training_examples_by_season,
                    training_seasons,
                    feature_names,
                    include_signal=include_signal,
                )
                fitted = fit_probe(
                    training_examples,
                    feature_names,
                    regularization=selected_lambda,
                    include_signal=include_signal,
                    training_seasons=training_seasons,
                )
                evaluation_examples = examples_by_season[evaluation_season]
                predictions = [
                    predict_probe(example, fitted) for example in evaluation_examples
                ]
                selection_record = {
                    "model": model,
                    "checkpoint": checkpoint,
                    "evaluation_season": evaluation_season,
                    "training_seasons": list(training_seasons),
                    "training_team_seasons": len(training_examples),
                    "training_cold_starts": sum(
                        example.features.get("mean_lag1_z") is None
                        for example in training_examples
                    ),
                    "evaluation_team_seasons": len(evaluation_examples),
                    "evaluation_cold_starts": sum(
                        example.features.get("mean_lag1_z") is None
                        for example in evaluation_examples
                    ),
                    "selected_lambda": selected_lambda,
                    "selection": selection,
                    "fit": {
                        "optimizer": dict(fitted.optimizer),
                        "coefficient_names": list(fitted.coefficient_names),
                        "coefficients": fitted.coefficients.tolist(),
                        "preprocessor": fitted.preprocessor.as_dict(),
                        "regularization": fitted.regularization,
                        "include_signal": fitted.include_signal,
                        "feature_names": list(feature_names),
                        "training_seasons": list(fitted.training_seasons),
                    },
                }
                fit_records.append(selection_record)
                row_by_id = {
                    str(row["team_id"]): row
                    for row in checkpoint_rows
                    if int(row["season"]) == evaluation_season
                }
                for example, prediction in zip(
                    evaluation_examples, predictions, strict=True
                ):
                    panel_row = row_by_id[example.team_id]
                    prediction_rows.append(
                        {
                            "season": evaluation_season,
                            "checkpoint": checkpoint,
                            "team_id": example.team_id,
                            "team_name": example.team_name,
                            "component_status": panel_row["component_status"],
                            "model": model,
                            "model_name": MODEL_NAME_BY_KIND[model],
                            "training_seasons": list(training_seasons),
                            "selected_lambda": selected_lambda,
                            "prediction_pmf": prediction.tolist(),
                        }
                    )
                    score_rows.append(
                        _team_score_row(
                            season=evaluation_season,
                            checkpoint=checkpoint,
                            team_id=example.team_id,
                            team_name=example.team_name,
                            component_status=str(panel_row["component_status"]),
                            games_played=int(panel_row["games_played"]),
                            history_kl=float(
                                panel_row["history_kl_posterior_from_prior"]
                            ),
                            model=model,
                            model_name=MODEL_NAME_BY_KIND[model],
                            prediction=prediction,
                            target=example.target,
                            score_type="residual_probe",
                        )
                    )
    fit_records.sort(
        key=lambda row: (
            int(row["checkpoint"]),
            str(row["model"]),
            int(row["evaluation_season"]),
        )
    )
    prediction_rows.sort(
        key=lambda row: (
            int(row["checkpoint"]),
            int(row["season"]),
            str(row["model"]),
            _team_id_sort_key(str(row["team_id"])),
        )
    )
    return prediction_rows, score_rows, fit_records


def validate_retained_baseline_parity(
    repo_root: Path,
    score_rows: list[dict[str, object]],
    checkpoints: tuple[int, ...],
) -> dict[str, object]:
    retained_rows = read_csv(
        repo_root
        / "data/processed/context_history_crossover/hybrid_posterior_team_results.csv"
    )
    retained = {
        (int(row["season"]), int(row["checkpoint"]), row["team_id"], row["arm"]): row
        for row in retained_rows
        if row["arm"] == "HH" and int(row["checkpoint"]) in checkpoints
    }
    local = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
            row["model"],
        ): row
        for row in score_rows
        if row["model"] == "history_posterior" and int(row["checkpoint"]) in checkpoints
    }
    expected_count = len(retained)
    metric_pairs = {
        "nll": "posterior_nll",
        "crps": "posterior_crps",
        "expected_rank": "posterior_expected_rank",
        "expected_rank_error": "posterior_expected_rank_error",
        "interval_80_width": "posterior_interval_80_width",
        "interval_80_target_mass": "posterior_interval_80_target_mass",
    }
    max_error = {metric: 0.0 for metric in metric_pairs}
    checked = 0
    for key, expected in retained.items():
        season, checkpoint, team_id, arm = key
        if arm != "HH":
            raise ValueError(f"unexpected retained baseline arm: {arm}")
        model = "history_posterior"
        observed = local.get((season, checkpoint, team_id, model))
        if observed is None:
            raise ValueError(f"retained baseline score is missing: {key}")
        for metric, retained_field in metric_pairs.items():
            error = abs(float(observed[metric]) - float(expected[retained_field]))
            max_error[metric] = max(max_error[metric], error)
            if error > BASELINE_SCORE_TOLERANCE:
                raise ValueError(f"{key} {metric}: retained score differs by {error}")
        checked += 1
    if checked != len(retained) or checked != len(local) or checked != expected_count:
        raise ValueError(
            f"baseline parity population differs: checked={checked}, retained={len(retained)}, local={len(local)}"
        )
    return {
        "model_family": "rolling History 1.1",
        "score_rows_compared": checked,
        "metrics": max_error,
        "tolerance": BASELINE_SCORE_TOLERANCE,
        "checkpoints": list(checkpoints),
    }


def measure_legacy_context_prior_difference(
    repo_root: Path,
    context_priors: dict[int, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Describe, but do not treat as parity, retained Context 1.3 crossover priors."""
    rows = read_csv(
        repo_root / "data/processed/context_history_crossover/hybrid_prior_results.csv"
    )
    retained = {
        (int(row["season"]), row["team_id"]): normalized_pmf(
            json.loads(row["prior_pmf"]), name="retained Context 1.3 prior"
        )
        for row in rows
        if row["arm"] == "CC"
    }
    differences = []
    for season in SEASONS:
        for team_id, pmf in context_priors[season].items():
            old = retained.get((season, team_id))
            if old is None or old.shape != pmf.shape:
                raise ValueError(
                    f"{season} {team_id}: legacy Context prior population differs"
                )
            differences.append(float(np.max(np.abs(pmf - old))))
    return {
        "legacy_family": "Context 1.3 retained crossover CC",
        "team_seasons_compared": len(differences),
        "max_absolute_pmf_difference": max(differences, default=0.0),
        "use_as_context_1_4_parity_target": False,
        "reason": "retained crossover CC was built from Context 1.3 features; this study reconstructs repaired production Context 1.4",
    }


def validate_pilot_gate(
    score_rows: list[dict[str, object]],
    checkpoints: tuple[int, ...],
) -> dict[str, object]:
    def subset(
        model: str, checkpoint: int, seasons: tuple[int, ...]
    ) -> list[dict[str, object]]:
        return [
            row
            for row in score_rows
            if row["model"] == model
            and int(row["checkpoint"]) == checkpoint
            and int(row["season"]) in seasons
        ]

    heldout = (2023, 2024, 2025)
    means: dict[str, dict[str, float]] = {}
    checks: list[dict[str, object]] = []
    for checkpoint in REPRODUCTION_CHECKPOINTS:
        if checkpoint not in checkpoints:
            continue
        observed = {}
        for model, expected in EXPECTED_PILOT_NLL[checkpoint].items():
            rows = subset(model, checkpoint, heldout)
            if len(rows) != 403:
                raise ValueError(
                    f"checkpoint {checkpoint} {model}: expected 403 held-out team-seasons, got {len(rows)}"
                )
            value = float(np.mean([float(row["nll"]) for row in rows]))
            observed[model] = value
            error = abs(value - expected)
            checks.append(
                {
                    "checkpoint": checkpoint,
                    "model": model,
                    "observed_mean_nll": value,
                    "expected_mean_nll": expected,
                    "absolute_error": error,
                    "tolerance": PILOT_SCORE_TOLERANCE,
                    "passed": error <= PILOT_SCORE_TOLERANCE,
                }
            )
            if error > PILOT_SCORE_TOLERANCE:
                raise ValueError(
                    f"checkpoint {checkpoint} {model}: pilot mean NLL differs by {error}"
                )
        means[str(checkpoint)] = observed

    cross_means = {}
    if 7 in checkpoints:
        for arm, expected in EXPECTED_DECEMBER_CROSSES.items():
            rows = subset(arm, 7, heldout)
            if len(rows) != 403:
                raise ValueError(
                    f"December crossed arm {arm}: unexpected held-out population"
                )
            value = float(np.mean([float(row["nll"]) for row in rows]))
            error = abs(value - expected)
            cross_means[arm] = value
            checks.append(
                {
                    "checkpoint": 7,
                    "model": f"cross_{arm}",
                    "observed_mean_nll": value,
                    "expected_mean_nll": expected,
                    "absolute_error": error,
                    "tolerance": PILOT_SCORE_TOLERANCE,
                    "passed": error <= PILOT_SCORE_TOLERANCE,
                }
            )
            if error > PILOT_SCORE_TOLERANCE:
                raise ValueError(
                    f"December crossed arm {arm}: pilot NLL differs by {error}"
                )

    september_deltas: dict[str, float] = {}
    if 2 in checkpoints:
        expected_deltas = {2023: -0.01218, 2024: -0.00631, 2025: -0.00493}
        for season, expected in expected_deltas.items():
            control = subset("history_calibration", 2, (season,))
            signal = subset("existing_context_signal", 2, (season,))
            delta = float(np.mean([float(row["nll"]) for row in signal])) - float(
                np.mean([float(row["nll"]) for row in control])
            )
            september_deltas[str(season)] = delta
            if abs(delta - expected) > PILOT_SCORE_TOLERANCE:
                raise ValueError(
                    f"September {season}: existing-signal delta differs from pilot by {abs(delta - expected)}"
                )
    return {
        "expected_table_means": means,
        "december_crossed_arm_means": cross_means,
        "september_existing_signal_delta_by_season": september_deltas,
        "checks": checks,
        "tolerance": PILOT_SCORE_TOLERANCE,
        "heldout_seasons": list(heldout),
        "heldout_team_seasons_per_checkpoint": 403,
        "passed": True,
    }


def aggregate_scores(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in score_rows:
        key = (
            str(row["score_type"]),
            int(row["checkpoint"]),
            int(row["season"]),
            str(row["model"]),
            str(row["component_status"]),
        )
        groups[key].append(row)
    result = []
    for (score_type, checkpoint, season, model, component_status), members in sorted(
        groups.items()
    ):
        aggregate: dict[str, object] = {
            "score_type": score_type,
            "checkpoint": checkpoint,
            "season": season,
            "model": model,
            "model_name": str(members[0]["model_name"]),
            "component_status": component_status,
            "team_seasons": len(members),
        }
        for metric in SCORE_METRICS:
            aggregate[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in members])
            )
        result.append(aggregate)
    return result


def primary_comparisons(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (
            str(row["model"]),
            int(row["checkpoint"]),
            int(row["season"]),
            str(row["component_status"]),
        ): row
        for row in aggregate_scores(score_rows)
    }
    comparisons = []
    for checkpoint in sorted({int(row["checkpoint"]) for row in score_rows}):
        for season in (2023, 2024, 2025):
            for component_status in ("all", "fitted", "cold_start_fallback"):
                control = None
                if component_status == "all":
                    members = [
                        row
                        for row in score_rows
                        if row["model"] == "history_calibration"
                        and int(row["checkpoint"]) == checkpoint
                        and int(row["season"]) == season
                    ]
                    if members:
                        control = {
                            metric: float(
                                np.mean([float(row[metric]) for row in members])
                            )
                            for metric in SCORE_METRICS
                        }
                else:
                    grouped = by_key.get(
                        (
                            "history_calibration",
                            checkpoint,
                            season,
                            component_status,
                        )
                    )
                    if grouped:
                        control = {
                            metric: float(grouped[f"mean_{metric}"])
                            for metric in SCORE_METRICS
                        }
                if control is None:
                    continue
                for model in ("existing_context_signal", "raw_context_features"):
                    if component_status == "all":
                        members = [
                            row
                            for row in score_rows
                            if row["model"] == model
                            and int(row["checkpoint"]) == checkpoint
                            and int(row["season"]) == season
                        ]
                        if not members:
                            continue
                        observed = {
                            metric: float(
                                np.mean([float(row[metric]) for row in members])
                            )
                            for metric in SCORE_METRICS
                        }
                        count = len(members)
                    else:
                        grouped = by_key.get(
                            (model, checkpoint, season, component_status)
                        )
                        if not grouped:
                            continue
                        observed = {
                            metric: float(grouped[f"mean_{metric}"])
                            for metric in SCORE_METRICS
                        }
                        count = int(grouped["team_seasons"])
                    comparisons.append(
                        {
                            "checkpoint": checkpoint,
                            "season": season,
                            "component_status": component_status,
                            "context_addition": model,
                            "model_name": MODEL_NAME_BY_KIND[model],
                            "team_seasons": count,
                            **{
                                f"delta_{metric}": observed[metric] - control[metric]
                                for metric in SCORE_METRICS
                            },
                        }
                    )
    return comparisons


def _residual_classification(deltas: list[float]) -> str:
    if len(deltas) != 3:
        return "inconclusive: incomplete outer-fold coverage"
    if all(value < -1e-12 for value in deltas):
        return "positive descriptive signal"
    if all(value > 1e-12 for value in deltas):
        return "negative descriptive signal"
    return "mixed/inconclusive across evaluation seasons"


def build_report(
    *,
    output_path: Path,
    panel_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    reconstruction_error: float,
    pilot_gate: dict[str, object],
    parity: dict[str, object],
    legacy_context_difference: dict[str, object],
    pilot_only: bool,
) -> None:
    total_fbs_team_seasons = sum(
        1
        for row in panel_rows
        if row["subdivision"] == "fbs" and int(row["checkpoint"]) == 1
    )
    fitted = sum(
        1
        for row in panel_rows
        if row["subdivision"] == "fbs"
        and int(row["checkpoint"]) == 1
        and row["component_status"] == "fitted"
    )
    cold = total_fbs_team_seasons - fitted
    checkpoints = sorted({int(row["checkpoint"]) for row in panel_rows})

    def mean_nll(model: str, checkpoint: int, season: int | None = None) -> float:
        selected = [
            row
            for row in score_rows
            if row["model"] == model
            and int(row["checkpoint"]) == checkpoint
            and int(row["season"]) in (2023, 2024, 2025)
            and (season is None or int(row["season"]) == season)
        ]
        if not selected:
            return float("nan")
        return float(np.mean([float(row["nll"]) for row in selected]))

    lines = [
        "# Context incremental information after games",
        "",
        "## Scope and audit",
        "",
        f"- Seasons: 2022–2025; checkpoints: {', '.join(map(str, checkpoints))}.",
        f"- FBS team-seasons per checkpoint: {total_fbs_team_seasons:,} ({fitted:,} fitted Context rows; {cold:,} native cold starts).",
        f"- Inference evidence rows: {len(evidence_rows):,}; target-free inference panel rows: {len(panel_rows):,} including FCS fallback nodes.",
        f"- Reconstructed fitted Context priors: 528; maximum absolute PMF error: {reconstruction_error:.3g}.",
        f"- Retained History 1.1 score parity: {parity['score_rows_compared']:,} rows; maximum errors by metric: `{json.dumps(parity['metrics'], sort_keys=True)}` (tolerance {BASELINE_SCORE_TOLERANCE:g}).",
        f"- Retained crossover CC is Context 1.3, so it is not a Context 1.4 posterior-score target. Its maximum prior-PMF difference from reconstructed Context 1.4 is {float(legacy_context_difference['max_absolute_pmf_difference']):.3g} across {legacy_context_difference['team_seasons_compared']} team-seasons.",
        f"- Pilot replay: {'pilot only; bounded extension not run' if pilot_only else 'passed before extending to checkpoints 3–6'}; pilot NLL tolerance {PILOT_SCORE_TOLERANCE:g}.",
        "- Targets are held in `scoring_targets.jsonl.gz` and are not present in the inference panel. Probe outer folds evaluate 2023, 2024, and 2025 after training only on earlier seasons.",
        "- KL uses exact `KL(History posterior || History prior)` on positive posterior support; positive posterior mass outside prior support fails. No smoothing is applied to inference PMFs or crossed messages.",
        "",
        "## Checkpoint results",
        "",
        "The table reports equal-team-season mean NLL across the three held-out years for each checkpoint separately. Repeated checkpoints are not pooled as independent observations.",
        "",
        "| Checkpoint | History | Context | History control | Existing-signal Δ | Raw-feature Δ | Existing signal by year | Raw features by year |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for checkpoint in checkpoints:
        control = mean_nll("history_calibration", checkpoint)
        existing = mean_nll("existing_context_signal", checkpoint)
        raw = mean_nll("raw_context_features", checkpoint)
        history = mean_nll("history_posterior", checkpoint)
        context = mean_nll("context_posterior", checkpoint)
        annual_signal = [
            mean_nll("existing_context_signal", checkpoint, season)
            - mean_nll("history_calibration", checkpoint, season)
            for season in (2023, 2024, 2025)
        ]
        annual_raw = [
            mean_nll("raw_context_features", checkpoint, season)
            - mean_nll("history_calibration", checkpoint, season)
            for season in (2023, 2024, 2025)
        ]
        signal_class = _residual_classification(annual_signal)
        raw_class = _residual_classification(annual_raw)
        lines.append(
            f"| {checkpoint} | {history:.5f} | {context:.5f} | {control:.5f} | {existing - control:+.5f} | {raw - control:+.5f} | {signal_class} ({', '.join(f'{value:+.5f}' for value in annual_signal)}) | {raw_class} ({', '.join(f'{value:+.5f}' for value in annual_raw)}) |"
        )
    lines.extend(
        [
            "",
            "A negative delta means the Context addition lowered NLL relative to the same History calibration control. These are descriptive outer-fold results over development seasons, not an untouched confirmation set or a production recommendation. Mixed year signs are reported as mixed/inconclusive. Each checkpoint remains a separate comparison.",
            "",
            "## Pilot crossed-message check",
            "",
            "At checkpoint 7, the crossed-arm NLLs (mean over 2023–2025) were: "
            + ", ".join(
                f"{arm} {float(value):.5f}"
                for arm, value in sorted(
                    pilot_gate["december_crossed_arm_means"].items()
                )
            )
            + ". The crosses freeze each model's incoming message product and therefore are a sensitivity analysis, not new converged joint posteriors or causal opponent attributions.",
            "",
            "## Output files",
            "",
            "- `inference_panel.jsonl.gz`: complete prior/posterior and crossed PMFs, prediction-time features, games played, and History information gain; contains no targets.",
            "- `scoring_targets.jsonl.gz`: frozen final-rank target PMFs, isolated from inference.",
            "- `probe_predictions.jsonl.gz`, `team_scores.csv`, `aggregate_scores.csv`, and `comparison_summary.csv`: per-team forecasts/scores and checkpoint/season comparisons, with fitted and cold-start strata.",
            "- `probe_fits.json` and `probe_coefficients.csv`: nested regularization choices, preprocessing, optimizer details, coefficients, and training/evaluation fold sizes.",
            "- `evidence.json` and `manifest.json`: cutoff/game evidence, source and output hashes, reconstruction/parity checks, and environment versions.",
            "",
            "Reproduce with `uv run python scripts/build_context_incremental_information.py --input-root <repair-input-root> --targets <team_season_rank_distributions.csv> --raw-games <cfbd-games-cache>`.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _source_file_hashes(
    repo_root: Path,
    input_root: Path,
    target_path: Path,
    raw_games: Path,
    repair_hashes: dict[str, str],
    history_hashes: dict[str, str],
    crossover_hashes: dict[str, str],
    raw_expected: dict[str, str],
) -> dict[str, str]:
    sources = {
        **repair_hashes,
        **history_hashes,
        **crossover_hashes,
    }
    sources.update(
        {f"raw_games/{name}": digest for name, digest in raw_expected.items()}
    )
    sources["external_input/team_season_rank_distributions.csv"] = sha256_file(
        target_path
    )
    sources["docs/context_incremental_information_protocol.md"] = sha256_file(
        repo_root / "docs/context_incremental_information_protocol.md"
    )
    sources["scripts/build_context_incremental_information.py"] = sha256_file(
        repo_root / "scripts/build_context_incremental_information.py"
    )
    sources["src/gippyrank/research/context_incremental_information.py"] = sha256_file(
        repo_root / "src/gippyrank/research/context_incremental_information.py"
    )
    for relative in (
        "scripts/study_context_history_crossover.py",
        "scripts/study_history_context_posterior.py",
        "src/gippyrank/context_db_repair.py",
        "src/gippyrank/posterior/engine.py",
        "src/gippyrank/posterior/snapshots.py",
        "src/gippyrank/preseason.py",
    ):
        sources[relative] = sha256_file(repo_root / relative)
    for key, digest in list(sources.items()):
        if (
            key.startswith("data/")
            and key not in repair_hashes
            and key not in history_hashes
            and key not in crossover_hashes
        ):
            if not (repo_root / key).is_file():
                raise FileNotFoundError(f"manifest source is absent: {key}")
            if sha256_file(repo_root / key) != digest:
                raise ValueError(f"manifest source changed while building: {key}")
    if not input_root.is_dir() or not raw_games.is_dir():
        raise FileNotFoundError(
            "explicit repair and raw game input roots must be directories"
        )
    return dict(sorted(sources.items()))


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.system().lower(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }


def write_outputs(
    *,
    repo_root: Path,
    output: Path,
    panel_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    fit_records: list[dict[str, object]],
    evidence: dict[str, object],
    sources: dict[str, str],
    pilot_only: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "inference_panel.jsonl.gz"
    targets_path = output / "scoring_targets.jsonl.gz"
    predictions_path = output / "probe_predictions.jsonl.gz"
    team_scores_path = output / "team_scores.csv"
    aggregate_path = output / "aggregate_scores.csv"
    comparisons_path = output / "comparison_summary.csv"
    probe_fits_path = output / "probe_fits.json"
    coefficients_path = output / "probe_coefficients.csv"
    evidence_path = output / "evidence.json"
    report_path = output / "report.md"
    manifest_path = output / "manifest.json"

    write_gzip_jsonl(panel_path, panel_rows)
    write_gzip_jsonl(targets_path, target_rows)
    write_gzip_jsonl(predictions_path, prediction_rows)
    write_csv(team_scores_path, score_rows)
    aggregates = aggregate_scores(score_rows)
    write_csv(aggregate_path, aggregates)
    comparisons = primary_comparisons(score_rows)
    write_csv(comparisons_path, comparisons)
    write_json(probe_fits_path, fit_records)
    coefficient_rows = []
    for fit in fit_records:
        values = dict(
            zip(
                fit["fit"]["coefficient_names"], fit["fit"]["coefficients"], strict=True
            )
        )
        for name, value in values.items():
            coefficient_rows.append(
                {
                    "model": fit["model"],
                    "checkpoint": fit["checkpoint"],
                    "evaluation_season": fit["evaluation_season"],
                    "training_seasons": json.dumps(fit["training_seasons"]),
                    "selected_lambda": fit["selected_lambda"],
                    "coefficient": name,
                    "value": value,
                }
            )
    write_csv(coefficients_path, coefficient_rows)
    write_json(evidence_path, evidence)
    build_report(
        output_path=report_path,
        panel_rows=panel_rows,
        evidence_rows=evidence["checkpoints"],
        score_rows=score_rows,
        comparisons=comparisons,
        reconstruction_error=float(evidence["context_reconstruction_max_error"]),
        pilot_gate=evidence["pilot_gate"],
        parity=evidence["baseline_parity"],
        legacy_context_difference=evidence["legacy_context_prior_difference"],
        pilot_only=pilot_only,
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in (
            panel_path,
            targets_path,
            predictions_path,
            team_scores_path,
            aggregate_path,
            comparisons_path,
            probe_fits_path,
            coefficients_path,
            evidence_path,
            report_path,
        )
    }
    write_json(
        manifest_path,
        {
            "format_version": 1,
            "study": "issue-192-context-incremental-information",
            "pilot_protocol_sha256": PILOT_PROTOCOL_SHA256,
            "checked_in_protocol_sha256": sources[
                "docs/context_incremental_information_protocol.md"
            ],
            "source_sha256": sources,
            "output_sha256": output_hashes,
            "runtime": _runtime_versions(),
            "scope": {
                "seasons": list(SEASONS),
                "checkpoints": sorted({int(row["checkpoint"]) for row in panel_rows}),
                "pilot_only": pilot_only,
                "inference_settings": dict(INFERENCE),
                "residual_probe_lambda_grid": list(LAMBDA_GRID),
                "targets_separate_from_inference_panel": True,
            },
            "manifest_hash_policy": "source and output files are hashed; this manifest does not hash itself",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--raw-games", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/context_incremental_information",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="stop after the checkpoint 1/2/7 parity-gated reproduction",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    input_root = args.input_root.resolve()
    target_path = args.targets.resolve()
    raw_games = args.raw_games.resolve()
    output = args.output.resolve()
    if target_path.name != "team_season_rank_distributions.csv":
        raise ValueError("--targets must be the frozen final-rank corpus")

    repair_hashes = verify_repair_inputs(input_root, repo_root)
    crossover_hashes, raw_expected = verify_crossover_sources(repo_root, raw_games)
    (
        context_priors,
        context_features,
        context_names,
        _populations,
        context_hashes,
        reconstruction_error,
    ) = load_context_priors(repo_root, input_root)
    repair_hashes.update(context_hashes)
    legacy_context_difference = measure_legacy_context_prior_difference(
        repo_root, context_priors
    )
    history_priors, history_statuses, history_names, history_hashes = (
        load_history_priors(repo_root, context_priors)
    )
    for season in SEASONS:
        if history_statuses[season] != {
            team_id: (
                "cold_start_fallback"
                if team_id not in context_features[season]
                or context_features[season][team_id].get("mean_lag1_z") is None
                else "fitted"
            )
            for team_id in history_priors[season]
        }:
            raise ValueError(f"{season}: retained History/Context cold starts differ")
        for team_id in history_priors[season]:
            if history_names[season][team_id] != context_names[season][team_id]:
                raise ValueError(
                    f"{season} {team_id}: History and Context names differ"
                )

    print("verified repair, retained evidence, and raw-game hashes", flush=True)
    with historical.historical_game_root(raw_games) as (
        game_root,
        _game_path,
        _game_hash,
        _raw_hashes,
    ):
        pilot_panel, pilot_evidence, pilot_message_errors = build_inference_panel(
            repo_root=repo_root,
            game_root=game_root,
            context_priors=context_priors,
            context_features=context_features,
            context_names=context_names,
            history_priors=history_priors,
            history_statuses=history_statuses,
            evidence_checkpoints=REPRODUCTION_CHECKPOINTS,
        )

    # The scoring target corpus is intentionally opened only after the pilot's
    # target-free inference panel has been fully constructed.
    target_hash = verify_scoring_target_source(repo_root, target_path)
    targets, observed_fcs_populations = load_scoring_targets(target_path)
    _, expected_fcs_populations = _expected_evidence(repo_root)
    validate_scoring_population(
        pilot_panel, targets, expected_fcs_populations, observed_fcs_populations
    )
    target_rows = [
        {
            "season": season,
            "team_id": team_id,
            "rank_count": len(pmf),
            "target_pmf": pmf.tolist(),
        }
        for season in SEASONS
        for team_id, pmf in sorted(
            targets[season].items(), key=lambda item: _team_id_sort_key(item[0])
        )
    ]
    pilot_baseline_scores = score_panel(pilot_panel, targets)
    pilot_predictions, pilot_probe_scores, pilot_fit_records = fit_and_score_probes(
        pilot_panel, targets
    )
    pilot_scores = pilot_baseline_scores + pilot_probe_scores
    validate_retained_baseline_parity(
        repo_root, pilot_baseline_scores, REPRODUCTION_CHECKPOINTS
    )
    validate_pilot_gate(pilot_scores, REPRODUCTION_CHECKPOINTS)
    print(
        "pilot checkpoints 1/2/7 passed retained-score and rounded-NLL gates",
        flush=True,
    )

    panel_rows = list(pilot_panel)
    evidence_rows = list(pilot_evidence)
    prediction_rows = list(pilot_predictions)
    probe_scores = list(pilot_probe_scores)
    fit_records = list(pilot_fit_records)
    if not args.pilot_only:
        with historical.historical_game_root(raw_games) as (
            game_root,
            _game_path,
            _game_hash,
            _raw_hashes,
        ):
            added_panel, added_evidence, added_message_errors = build_inference_panel(
                repo_root=repo_root,
                game_root=game_root,
                context_priors=context_priors,
                context_features=context_features,
                context_names=context_names,
                history_priors=history_priors,
                history_statuses=history_statuses,
                evidence_checkpoints=tuple(
                    checkpoint
                    for checkpoint in ALL_CHECKPOINTS
                    if checkpoint not in REPRODUCTION_CHECKPOINTS
                ),
            )
        added_predictions, added_probe_scores, added_fit_records = fit_and_score_probes(
            added_panel, targets
        )
        panel_rows.extend(added_panel)
        evidence_rows.extend(added_evidence)
        prediction_rows.extend(added_predictions)
        probe_scores.extend(added_probe_scores)
        fit_records.extend(added_fit_records)
        for key in pilot_message_errors:
            pilot_message_errors[key] = max(
                pilot_message_errors[key], added_message_errors[key]
            )

    panel_rows.sort(
        key=lambda row: (
            int(row["season"]),
            int(row["checkpoint"]),
            0 if row["subdivision"] == "fbs" else 1,
            _team_id_sort_key(str(row["team_id"])),
        )
    )
    evidence_rows.sort(key=lambda row: (int(row["season"]), int(row["checkpoint"])))
    prediction_rows.sort(
        key=lambda row: (
            int(row["checkpoint"]),
            int(row["season"]),
            str(row["model"]),
            _team_id_sort_key(str(row["team_id"])),
        )
    )
    baseline_scores = score_panel(panel_rows, targets)
    score_rows = baseline_scores + probe_scores
    selected_checkpoints = tuple(sorted({int(row["checkpoint"]) for row in panel_rows}))
    baseline_parity = validate_retained_baseline_parity(
        repo_root, baseline_scores, selected_checkpoints
    )
    final_pilot_gate = validate_pilot_gate(score_rows, REPRODUCTION_CHECKPOINTS)
    source_hashes = _source_file_hashes(
        repo_root,
        input_root,
        target_path,
        raw_games,
        repair_hashes,
        history_hashes,
        crossover_hashes,
        raw_expected,
    )
    evidence = {
        "format_version": 1,
        "study": "issue-192-context-incremental-information",
        "seasons": list(SEASONS),
        "checkpoint_role": {
            str(checkpoint): (
                "retained_reproduction"
                if checkpoint in REPRODUCTION_CHECKPOINTS
                else "bounded_new_exploratory_diagnostic"
            )
            for checkpoint in selected_checkpoints
        },
        "inference_settings": dict(INFERENCE),
        "score_contract": {
            "rank_nll": "study_context_history_crossover.score_pmf; forecast floor 1e-15",
            "probe_lambda_selection": "exact target cross-entropy on supported ranks; no smoothing",
        },
        "pilot_protocol_sha256": PILOT_PROTOCOL_SHA256,
        "pilot_gate": final_pilot_gate,
        "baseline_parity": baseline_parity,
        "legacy_context_prior_difference": legacy_context_difference,
        "context_reconstruction_fitted_rows": 528,
        "context_reconstruction_max_error": reconstruction_error,
        "context_native_cold_start_count": 6,
        "history_native_cold_start_count": sum(
            status == "cold_start_fallback"
            for season in history_statuses.values()
            for status in season.values()
        ),
        "message_identity_max_errors": pilot_message_errors,
        "kl_support_policy": "exact KL(posterior || prior); posterior mass outside prior support is an error; no smoothing",
        "target_boundary": {
            "targets_loaded_after_pilot_inference": True,
            "target_sha256": target_hash,
            "target_file_logical_name": "data/processed/modeling/team_season_rank_distributions.csv",
            "target_pmfs_in_inference_panel": False,
        },
        "checkpoints": evidence_rows,
        "pilot_only": bool(args.pilot_only),
        "source_sha256": source_hashes,
        "raw_game_file_count": len(raw_expected),
        "repair_source_hash_count": len(repair_hashes),
        "retained_evidence_source_hash_count": len(crossover_hashes),
        "score_row_count": len(score_rows),
        "probe_prediction_count": len(prediction_rows),
    }
    write_outputs(
        repo_root=repo_root,
        output=output,
        panel_rows=panel_rows,
        target_rows=target_rows,
        prediction_rows=prediction_rows,
        score_rows=score_rows,
        fit_records=fit_records,
        evidence=evidence,
        sources=source_hashes,
        pilot_only=args.pilot_only,
    )
    print(f"wrote reproducible study to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
