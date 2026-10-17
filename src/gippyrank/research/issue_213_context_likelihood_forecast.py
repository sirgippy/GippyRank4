"""Frozen three-arm prospective Context likelihood replay for issue 213."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import stdtr

from gippyrank.posterior import engine as posterior_engine
from gippyrank.posterior import predictive as predictive_engine
from gippyrank.posterior.engine import (
    Game,
    LikelihoodV1,
    PosteriorResult,
    Team,
    _student_t_logpdf,
    game_factor,
    game_margin_parameters,
    infer_posterior,
)
from gippyrank.posterior.predictive import (
    ScheduledGame,
    conditional_margin_location_surface,
    predictive_components,
    win_probabilities,
)
from gippyrank.research import issue_201_prebowl_forecast as issue201
from gippyrank.research.issue_195_likelihood_diagnostic import (
    CONTEXT_FEATURES,
    MISSINGNESS_FEATURES,
    ContextPreprocessor,
    _read_context_features,
    sha256_json,
)
from gippyrank.research.matched_pipeline_replay import (
    read_pipeline_bundle,
    replay_pipeline_bundle,
)

EVALUATION_SEASONS = (2023, 2024, 2025)
FIT_RECORDS_SHA256 = "a6bb5229c6991c5dd1661bc138f6c322b071c7591c0d7b4c75315b3966e59530"
BASE_LIKELIHOOD_SHA256 = "b64d237d4abf3121355a28505f805252cf32062b3fce66db6ca620ef35887952"
BASE_BUNDLE_SHA256 = "c5a80a6e5aa1415bacfe3cc373cfbbc6cfe02465f33be202d8ce26962cb8419c"
EXPECTED_CONTEXT_FEATURE_SHA256 = "f909d4cc4eb7d72f96d2ee107d85b2e9f4926178310342bc30b2635fd52666ef"
EXPECTED_FIT_TRAINING = {2023: [2022], 2024: [2022, 2023], 2025: [2022, 2023, 2024]}
BASELINE_ARM = "Context_existing"
GENERIC_ARM = "Context_generic_recalibrated"
CONTEXT_ARM = "Context_context_recalibrated"
MODEL_ARMS = (BASELINE_ARM, GENERIC_ARM, CONTEXT_ARM)
ARM_DEFINITIONS = {
    BASELINE_ARM: "#200 Context_existing prior and existing likelihood",
    GENERIC_ARM: "#200 Context_existing prior plus #195 rank_site_calibration_fit",
    CONTEXT_ARM: "#200 Context_existing prior plus #195 rank_site_context_fit",
}
IDENTITY_TOLERANCE = 1e-8
PRACTICAL_GAIN = 0.05
GUARDRAIL_POOLED_CRPS = 0.10
GUARDRAIL_SEASON_CRPS = 0.25
GUARDRAIL_POOLED_COVERAGE = 0.03
GUARDRAIL_SEASON_COVERAGE = 0.05
CRPS_TOLERANCE = 2e-6
FBS_FBS_BLOCK = slice(0, 9)
FBS_FBS_HOME_INDEX = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: issue201.canonical_json(value)
                if isinstance(value, (dict, list, tuple))
                else ""
                if value is None
                else value
                for key, value in row.items()
            }
        )
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "".join(issue201.canonical_json(row) + "\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as zipped:
        zipped.write(payload)


def _append_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    """Append one deterministic gzip member of canonical JSONL records."""
    with path.open("ab") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as zipped:
        for row in rows:
            zipped.write((issue201.canonical_json(row) + "\n").encode())


def _record(path: Path, label: str) -> dict[str, object]:
    return {"label": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


@dataclass(frozen=True)
class FrozenFit:
    season: int
    multiplier: float
    home_adjustment: float
    log_scale_correction: float
    context_coefficients: np.ndarray
    preprocessor: ContextPreprocessor
    training_seasons: tuple[int, ...]
    context_penalty: float
    source_record: dict[str, object]


def _preprocessor_from_record(record: dict[str, object]) -> ContextPreprocessor:
    names = tuple(str(value) for value in record.get("feature_names", []))
    if names != tuple(CONTEXT_FEATURES):
        raise ValueError("#195 preprocessing feature order differs from its frozen schema")
    arrays = {
        "means": np.asarray(record.get("value_means"), dtype=np.float64),
        "scales": np.asarray(record.get("value_scales"), dtype=np.float64),
        "missing_means": np.asarray(record.get("missingness_means"), dtype=np.float64),
        "missing_scales": np.asarray(record.get("missingness_scales"), dtype=np.float64),
    }
    if any(array.shape != (len(names),) for array in arrays.values()):
        raise ValueError("#195 training-only preprocessing has the wrong dimensions")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("#195 training-only preprocessing contains non-finite values")
    if np.any(arrays["scales"] <= 0) or np.any(arrays["missing_scales"] <= 0):
        raise ValueError("#195 preprocessing scales must be positive")
    return ContextPreprocessor(
        feature_names=names,
        means=arrays["means"],
        scales=arrays["scales"],
        missing_means=arrays["missing_means"],
        missing_scales=arrays["missing_scales"],
    )


def load_frozen_fits(issue195_root: Path) -> tuple[dict[int, FrozenFit], dict[str, object]]:
    manifest_path = issue195_root / "manifest.json"
    records_path = issue195_root / "fit_records.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("#195 manifest is not marked complete")
    if sha256_file(records_path) != FIT_RECORDS_SHA256:
        raise ValueError("#195 fit_records.json differs from the frozen SHA-256")
    output = manifest.get("outputs", {}).get("fit_records.json")
    if not isinstance(output, dict) or output.get("sha256") != FIT_RECORDS_SHA256:
        raise ValueError("#195 manifest does not bind the exact frozen fit records")
    all_records = json.loads(records_path.read_text(encoding="utf-8"))
    selected: dict[int, FrozenFit] = {}
    fit_metadata: dict[str, object] = {}
    for season in EVALUATION_SEASONS:
        matches = [
            row
            for row in all_records
            if row.get("role") == "held_out_evaluation"
            and row.get("target_season") == season
        ]
        if len(matches) != 1:
            raise ValueError(f"#195 contains {len(matches)} forward fit rows for {season}")
        row = matches[0]
        training = tuple(int(value) for value in row.get("training_seasons", []))
        if list(training) != EXPECTED_FIT_TRAINING[season]:
            raise ValueError(f"{season}: #195 fit training seasons differ from the protocol")
        if not row.get("target_season_excluded_from_fit_and_regularization_selection"):
            raise ValueError(f"{season}: #195 fit is not forward-only")
        if row.get("role") != "held_out_evaluation":
            raise ValueError(f"{season}: #195 fit is not marked as an evaluation fit")
        baseline = row.get("frozen_baseline")
        if not isinstance(baseline, dict) or baseline.get("refit") is not False:
            raise ValueError(f"{season}: #195 fit refit the frozen likelihood")
        if float(baseline.get("degrees_of_freedom", 0)) != 15.0:
            raise ValueError(f"{season}: #195 fit degrees of freedom are not 15")
        generic = row.get("rank_site_calibration_fit")
        context = row.get("rank_site_context_fit")
        prep_row = row.get("context_preprocessor")
        if not all(isinstance(value, dict) for value in (generic, context, prep_row)):
            raise TypeError(f"{season}: #195 fit record is missing frozen parameters")
        for name, fit in (("generic", generic), ("Context", context)):
            optimizer = fit.get("optimizer")  # type: ignore[union-attr]
            if not isinstance(optimizer, dict) or optimizer.get("success") is not True:
                raise ValueError(f"{season}: #195 {name} optimizer did not converge")
        if generic.get("context_coefficients") != {}:  # type: ignore[union-attr]
            raise ValueError(f"{season}: generic fit includes Context coefficients")
        names = (*CONTEXT_FEATURES, *MISSINGNESS_FEATURES)
        coefficients = context.get("context_coefficients")  # type: ignore[union-attr]
        if not isinstance(coefficients, dict) or set(coefficients) != set(names):
            raise ValueError(f"{season}: #195 Context coefficient schema changed")
        coefficient_array = np.asarray([coefficients[name] for name in names], dtype=float)
        if coefficient_array.shape != (len(names),) or not np.isfinite(coefficient_array).all():
            raise ValueError(f"{season}: invalid #195 Context coefficients")
        multiplier = float(context["overall_mean_multiplier"])  # type: ignore[index]
        home_adjustment = float(context["additional_non_neutral_home_field_points"])  # type: ignore[index]
        log_scale = float(context["common_log_scale_correction"])  # type: ignore[index]
        if not all(map(math.isfinite, (multiplier, home_adjustment, log_scale))) or multiplier <= 0:
            raise ValueError(f"{season}: invalid #195 Context joint parameters")
        generic_multiplier = float(generic["overall_mean_multiplier"])  # type: ignore[index]
        generic_home = float(generic["additional_non_neutral_home_field_points"])  # type: ignore[index]
        generic_log_scale = float(generic["common_log_scale_correction"])  # type: ignore[index]
        if not all(map(math.isfinite, (generic_multiplier, generic_home, generic_log_scale))) or generic_multiplier <= 0:
            raise ValueError(f"{season}: invalid #195 generic parameters")
        preprocessor = _preprocessor_from_record(prep_row)  # type: ignore[arg-type]
        selected[season] = FrozenFit(
            season=season,
            multiplier=multiplier,
            home_adjustment=home_adjustment,
            log_scale_correction=log_scale,
            context_coefficients=coefficient_array,
            preprocessor=preprocessor,
            training_seasons=training,
            context_penalty=float(context.get("context_ridge_penalty", 0.0)),  # type: ignore[union-attr]
            source_record=row,
        )
        fit_metadata[str(season)] = {
            "training_seasons": list(training),
            "rank_site_calibration_fit": generic,
            "rank_site_context_fit": context,
            "context_preprocessor": prep_row,
        }
    return selected, {
        "manifest_sha256": sha256_file(manifest_path),
        "fit_records_sha256": FIT_RECORDS_SHA256,
        "base_likelihood_sha256": BASE_LIKELIHOOD_SHA256,
        "fit_record_by_season": fit_metadata,
    }


def _verify_frozen_inputs(
    *,
    corpus_root: Path,
    issue195_root: Path,
    issue200_root: Path,
    issue201_root: Path,
) -> dict[str, object]:
    issue195 = json.loads((issue195_root / "manifest.json").read_text(encoding="utf-8"))
    records = issue195.get("input_records", [])
    mapping = {
        "issue200_existing_likelihood": issue200_root / "historical_likelihood_existing.json",
        "issue200_manifest": issue200_root / "manifest.json",
        "issue200_pipeline_bundle": issue200_root / "pipeline_bundle.json",
        "issue200_preseason_feature_manifest": issue200_root / "preseason_feature_manifest.json",
        "issue200_context_feature_inputs": issue200_root / "context_feature_inputs.csv",
        "issue201_manifest": issue201_root / "manifest.json",
        "issue201_forecast_scores": issue201_root / "forecast_scores.csv",
        "issue204_corrected_final_observations": corpus_root / "data/processed/massey_issue_204_v1/final_observations.csv",
        "issue204_corrected_final_targets": corpus_root / "data/processed/massey_prebowl_issue_204_v1/final_target_distributions.csv",
        "issue204_corrected_final_rank_pairs": corpus_root / "data/processed/massey_prebowl_issue_204_v1/final_constituent_rank_pairs.csv",
    }
    verified: dict[str, object] = {}
    mismatches: list[str] = []
    for record in records:
        logical = str(record.get("logical_name", ""))
        path = mapping.get(logical)
        if path is None:
            mismatches.append(f"unrecognized #195 input {logical!r}")
            continue
        if not path.is_file():
            mismatches.append(f"{logical}: missing at {path}")
            continue
        actual = sha256_file(path)
        size = path.stat().st_size
        if not record.get("exists") or actual != record.get("sha256") or size != int(record.get("bytes", -1)):
            mismatches.append(f"{logical}: expected {record.get('sha256')} / {record.get('bytes')}; got {actual} / {size}")
        verified[logical] = {"bytes": size, "sha256": actual}
    if len(records) != len(verified):
        mismatches.append("#195 input records are missing, duplicated, or unmapped")
    if verified.get("issue200_existing_likelihood", {}).get("sha256") != BASE_LIKELIHOOD_SHA256:  # type: ignore[union-attr]
        mismatches.append("#195 baseline likelihood SHA-256 is not the frozen value")
    if verified.get("issue200_pipeline_bundle", {}).get("sha256") != BASE_BUNDLE_SHA256:  # type: ignore[union-attr]
        mismatches.append("#200 bundle SHA-256 is not the frozen value")
    feature = verified.get("issue200_context_feature_inputs", {})
    if feature.get("sha256") != EXPECTED_CONTEXT_FEATURE_SHA256:  # type: ignore[union-attr]
        mismatches.append("#200 exact Context feature export SHA-256 is not the frozen value")
    if issue195.get("issue200_manifest_sha256") != verified.get("issue200_manifest", {}).get("sha256"):  # type: ignore[union-attr]
        mismatches.append("#195 recorded #200 manifest SHA-256 differs")
    if issue195.get("issue201_manifest_sha256") != verified.get("issue201_manifest", {}).get("sha256"):  # type: ignore[union-attr]
        mismatches.append("#195 recorded #201 manifest SHA-256 differs")

    issue200_manifest = json.loads((issue200_root / "manifest.json").read_text(encoding="utf-8"))
    if issue200_manifest.get("fitting_status") != "complete":
        mismatches.append("#200 manifest does not mark the pipeline complete")
    for name in (
        "pipeline_bundle.json",
        "fitted_models.json",
        "preseason_feature_manifest.json",
        "context_feature_inputs.csv",
        "historical_likelihood_existing.json",
    ):
        path = issue200_root / name
        expected = issue200_manifest.get("outputs", {}).get(name)
        if not isinstance(expected, dict) or path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256"):
            mismatches.append(f"#200 manifest output hash mismatch: {name}")
    if sha256_file(issue200_root / "pipeline_bundle.json") != BASE_BUNDLE_SHA256:
        mismatches.append("rebuilt #200 bundle does not match #195/#201")
    if mismatches:
        raise ValueError("frozen issue 195/#200/#201 input verification failed: " + "; ".join(mismatches))
    return {
        "verified_issue195_inputs": verified,
        "issue200_manifest_sha256": sha256_file(issue200_root / "manifest.json"),
        "issue200_manifest_output_records": issue200_manifest["outputs"],
    }


class ContextLikelihoodAdapter:
    """Apply one saved #195 fit to FBS–FBS factors only."""

    def __init__(
        self,
        base: LikelihoodV1,
        fit: FrozenFit,
        arm: str,
        feature_rows: dict[tuple[int, str], dict[str, object]],
    ) -> None:
        if arm not in {GENERIC_ARM, CONTEXT_ARM}:
            raise ValueError(f"unsupported candidate arm {arm}")
        self.base = base
        self.fit = fit
        self.arm = arm
        self.feature_rows = feature_rows
        if arm == GENERIC_ARM:
            generic = fit.source_record["rank_site_calibration_fit"]
            self.multiplier = float(generic["overall_mean_multiplier"])
            self.home_adjustment = float(generic["additional_non_neutral_home_field_points"])
            self.log_scale_correction = float(generic["common_log_scale_correction"])
            self.context_coefficients = np.zeros(len(CONTEXT_FEATURES) + len(MISSINGNESS_FEATURES))
        else:
            self.multiplier = fit.multiplier
            self.home_adjustment = fit.home_adjustment
            self.log_scale_correction = fit.log_scale_correction
            self.context_coefficients = fit.context_coefficients
        self.scale_multiplier = math.exp(self.log_scale_correction)
        self.base_digest = hashlib.sha256(base.beta.tobytes()).hexdigest()
        self.parameter_digest = sha256_json(
            {
                "adapter": "issue213_fbs_fbs_likelihood_v1",
                "arm": arm,
                "season": fit.season,
                "base_beta_sha256": self.base_digest,
                "base_scale": base.scale,
                "degrees_of_freedom": base.degrees_of_freedom,
                "multiplier": self.multiplier,
                "home_adjustment": self.home_adjustment,
                "log_scale_correction": self.log_scale_correction,
                "context_coefficients": self.context_coefficients.tolist(),
                "preprocessor": fit.preprocessor.metadata(),
            }
        )

    def context_delta(self, game: Game) -> tuple[float, str, str]:
        if self.arm != CONTEXT_ARM:
            return 0.0, "", ""
        try:
            home = self.feature_rows[(self.fit.season, game.home_id)]
            away = self.feature_rows[(self.fit.season, game.away_id)]
        except KeyError as error:
            raise ValueError(
                f"{game.game_id}: exact #200 Context feature row is missing for {error.args[0]}"
            ) from error
        home_values = np.asarray(
            [home["features"][name] for name in CONTEXT_FEATURES], dtype=np.float64
        )[None, :]
        away_values = np.asarray(
            [away["features"][name] for name in CONTEXT_FEATURES], dtype=np.float64
        )[None, :]
        design = self.fit.preprocessor.transform(home_values, away_values)
        if design.shape != (1, len(self.context_coefficients)) or not np.isfinite(design).all():
            raise ValueError(f"{game.game_id}: invalid saved #195 Context difference vector")
        return (
            float(design[0] @ self.context_coefficients),
            str(home["context_feature_sha256"]),
            str(away["context_feature_sha256"]),
        )

    def likelihood_for(self, game: Game, home: Team, away: Team) -> LikelihoodV1:
        """Reconstruct #195's shared multiplier, home, and scale terms."""
        if home.subdivision != "fbs" or away.subdivision != "fbs":
            return self.base
        beta = self.base.beta.copy()
        beta[FBS_FBS_BLOCK] *= self.multiplier
        beta[FBS_FBS_HOME_INDEX] += self.home_adjustment
        return LikelihoodV1(
            beta=beta,
            scale=self.base.scale * self.scale_multiplier,
            degrees_of_freedom=self.base.degrees_of_freedom,
            fit_kind=self.base.fit_kind,
        )

    def factor(self, game: Game, home: Team, away: Team) -> np.ndarray:
        likelihood = self.likelihood_for(game, home, away)
        if home.subdivision != "fbs" or away.subdivision != "fbs":
            return game_factor(game, home, away, likelihood)
        delta, _home_hash, _away_hash = self.context_delta(game)
        locations, margin = game_margin_parameters(game, home, away, likelihood)
        log_values = _student_t_logpdf(
            margin, locations + delta, likelihood.scale, likelihood.degrees_of_freedom
        )
        values = np.exp(log_values - float(np.max(log_values)))
        return values

    def predictive_components(
        self,
        scheduled: ScheduledGame,
        game: Game,
        home: Team,
        away: Team,
    ) -> tuple[np.ndarray, np.ndarray, LikelihoodV1]:
        """Return the saved-fit rank mixture with its game-level Context shift."""
        likelihood = self.likelihood_for(game, home, away)
        if home.subdivision != "fbs" or away.subdivision != "fbs":
            locations, weights = predictive_components(scheduled, home, away, likelihood)
            return locations, weights, likelihood
        delta, _home_hash, _away_hash = self.context_delta(game)
        locations = conditional_margin_location_surface(
            scheduled, home, away, likelihood
        ).ravel() + delta
        weights = np.multiply.outer(home.prior, away.prior).ravel()
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("posterior rank PMFs must produce positive finite mixture weight")
        return locations, weights / total, likelihood

    def factor_cache_key(self, game: Game, home: Team, away: Team) -> str:
        if home.subdivision != "fbs" or away.subdivision != "fbs":
            return sha256_json(
                {
                    "adapter": "issue213_unchanged_non_fbs_factor_v1",
                    "base_beta_sha256": self.base_digest,
                    "base_scale": self.base.scale,
                    "degrees_of_freedom": self.base.degrees_of_freedom,
                }
            )
        delta, home_hash, away_hash = self.context_delta(game)
        return sha256_json(
            {
                "parameter_digest": self.parameter_digest,
                "game_id": game.game_id,
                "context_delta": delta,
                "home_context_feature_sha256": home_hash,
                "away_context_feature_sha256": away_hash,
            }
        )


def _predictive_model_parameters(
    base: LikelihoodV1, fits: dict[int, FrozenFit]
) -> dict[str, object]:
    """Serialize exact location surfaces and Context adjustments for replaying mixtures."""
    models: list[dict[str, object]] = []
    for season in EVALUATION_SEASONS:
        fit = fits[season]
        for arm in MODEL_ARMS:
            if arm == BASELINE_ARM:
                multiplier, home_adjustment, log_scale = 1.0, 0.0, 0.0
                context_coefficients = np.zeros(
                    len(CONTEXT_FEATURES) + len(MISSINGNESS_FEATURES), dtype=np.float64
                )
                preprocessor: dict[str, object] | None = None
                source = "#200 unmodified existing likelihood"
            elif arm == GENERIC_ARM:
                generic = fit.source_record["rank_site_calibration_fit"]
                multiplier = float(generic["overall_mean_multiplier"])
                home_adjustment = float(generic["additional_non_neutral_home_field_points"])
                log_scale = float(generic["common_log_scale_correction"])
                context_coefficients = np.zeros(
                    len(CONTEXT_FEATURES) + len(MISSINGNESS_FEATURES), dtype=np.float64
                )
                preprocessor = None
                source = "#195 rank_site_calibration_fit"
            else:
                multiplier = fit.multiplier
                home_adjustment = fit.home_adjustment
                log_scale = fit.log_scale_correction
                context_coefficients = fit.context_coefficients
                preprocessor = fit.preprocessor.metadata()
                source = "#195 rank_site_context_fit"

            beta = base.beta.copy()
            beta[FBS_FBS_BLOCK] *= multiplier
            beta[FBS_FBS_HOME_INDEX] += home_adjustment
            scale = base.scale * math.exp(log_scale)
            model: dict[str, object] = {
                "season": season,
                "model_arm": arm,
                "source": source,
                "base_likelihood_sha256": BASE_LIKELIHOOD_SHA256,
                "rank_location_beta": beta.tolist(),
                "scale": scale,
                "degrees_of_freedom": base.degrees_of_freedom,
                "fit_kind": base.fit_kind,
                "overall_mean_multiplier": multiplier,
                "additional_non_neutral_home_field_points": home_adjustment,
                "common_log_scale_correction": log_scale,
                "context_feature_names": list(CONTEXT_FEATURES),
                "missingness_feature_names": list(MISSINGNESS_FEATURES),
                "context_coefficients": context_coefficients.tolist(),
                "context_preprocessor": preprocessor,
                "context_ridge_penalty": fit.context_penalty if arm == CONTEXT_ARM else None,
                "context_delta_rule": (
                    "dot(training_standardized_home_minus_away_values_and_missingness, context_coefficients)"
                    if arm == CONTEXT_ARM
                    else "zero"
                ),
            }
            model["model_parameters_sha256"] = sha256_json(model)
            models.append(model)
    return {
        "schema_version": 1,
        "distribution_family": "mixture of equal-scale Student-t rank-pair components",
        "degrees_of_freedom": base.degrees_of_freedom,
        "rank_coordinate": "zero-based rank position r maps to percentile (r + 0.5) / population",
        "surface_contract": "gippyrank.posterior.predictive.conditional_margin_location_surface with design_matrix(surface=True); add context_feature_mean_adjustment_points to every location",
        "mixture_weight_contract": "outer(home_posterior_pmf, away_posterior_pmf), flattened in C order (home rank major)",
        "models": models,
    }


def _predictive_mixture_rows(
    forecasts: list[dict[str, object]],
    posterior_rows: list[dict[str, object]],
    model_parameters: dict[str, object],
) -> list[dict[str, object]]:
    """Bind each exact predictive mixture to its posterior PMFs and frozen surface."""
    posterior_by_key = {
        (
            int(row["season"]),
            str(row["forecast_stage"]),
            str(row["cutoff_utc"]),
            str(row["model_arm"]),
            str(row["team_id"]),
        ): row
        for row in posterior_rows
    }
    parameters_by_key = {
        (int(row["season"]), str(row["model_arm"])): row
        for row in model_parameters["models"]  # type: ignore[index]
    }
    output: list[dict[str, object]] = []
    ordered_forecasts = sorted(
        forecasts,
        key=lambda row: (
            int(row["season"]),
            str(row["forecast_stage"]),
            str(row["cutoff_utc"]),
            issue201._game_id_key(str(row["game_id"])),
            str(row["model_arm"]),
        ),
    )
    for forecast in ordered_forecasts:
        season = int(forecast["season"])
        stage = str(forecast["forecast_stage"])
        cutoff = str(forecast["cutoff_utc"])
        arm = str(forecast["model_arm"])
        home_id, away_id = str(forecast["home_id"]), str(forecast["away_id"])
        try:
            home = posterior_by_key[(season, stage, cutoff, arm, home_id)]
            away = posterior_by_key[(season, stage, cutoff, arm, away_id)]
            parameters = parameters_by_key[(season, arm)]
        except KeyError as error:
            raise ValueError(
                f"{season}/{stage}/{cutoff}/{arm}: predictive mixture reference is missing {error.args[0]}"
            ) from error
        home_hash = str(home["posterior_pmf_sha256"])
        away_hash = str(away["posterior_pmf_sha256"])
        for field, actual in (
            ("home_posterior_pmf_sha256", home_hash),
            ("away_posterior_pmf_sha256", away_hash),
        ):
            if forecast.get(field) not in (None, "", actual):
                raise ValueError(
                    f"{forecast['game_id']}/{arm}: forecast {field} differs from its posterior artifact"
                )
        row: dict[str, object] = {
            "forecast_id": str(forecast["forecast_id"]),
            "forecast_stage": stage,
            "season": season,
            "cutoff_utc": cutoff,
            "game_id": str(forecast["game_id"]),
            "model_arm": arm,
            "home_id": home_id,
            "away_id": away_id,
            "neutral_site": issue201._truthy(forecast["neutral_site"]),
            "home_posterior_pmf_sha256": home_hash,
            "away_posterior_pmf_sha256": away_hash,
            "home_rank_population": int(home["rank_population"]),
            "away_rank_population": int(away["rank_population"]),
            "model_parameters_sha256": parameters["model_parameters_sha256"],
            "context_feature_mean_adjustment_points": float(
                forecast.get("context_feature_mean_adjustment_points", 0.0) or 0.0
            ),
            "home_context_feature_sha256": forecast.get("home_context_feature_sha256", ""),
            "away_context_feature_sha256": forecast.get("away_context_feature_sha256", ""),
            "student_t_scale": float(
                forecast.get("predictive_scale") or parameters["scale"]
            ),
            "student_t_degrees_of_freedom": float(parameters["degrees_of_freedom"]),
            "mixture_weight_rule": "outer product of referenced home and away posterior PMFs; home rank major",
            "location_rule": "conditional_margin_location_surface plus context_feature_mean_adjustment_points",
        }
        row["mixture_recipe_sha256"] = sha256_json(row)
        output.append(row)
    return output


def _verify_protocol(protocol_path: Path) -> str:
    protocol = protocol_path.read_text(encoding="utf-8")
    required_phrases = (
        "Frozen protocol",
        "rank_site_calibration_fit",
        "rank_site_context_fit",
        "2023, 2024, and 2025",
        "48-hour evidence",
        "0.05",
        "independent confirmation",
    )
    missing = [phrase for phrase in required_phrases if phrase not in protocol]
    if missing:
        raise ValueError("issue 213 protocol is incomplete: " + ", ".join(missing))
    return sha256_file(protocol_path)


def _validate_posterior(result: PosteriorResult, label: str) -> dict[str, object]:
    diagnostics = issue201._posterior_diagnostics(result)
    if not diagnostics["converged"]:
        raise RuntimeError(f"{label}: posterior did not converge")
    return diagnostics


def _deduplicate_games(games: list[Game]) -> list[Game]:
    """Keep exact duplicate IDs once and reject conflicting outcomes."""
    by_id: dict[str, Game] = {}
    for game in games:
        prior = by_id.get(game.game_id)
        if prior is not None and prior != game:
            raise ValueError(f"conflicting duplicate evidence game {game.game_id}")
        by_id[game.game_id] = game
    return [by_id[key] for key in sorted(by_id, key=issue201._game_id_key)]


def _validate_target_exclusion(
    *,
    game_id: str,
    cutoff: datetime,
    kickoff: datetime,
    evidence_ids: set[str],
) -> None:
    if game_id in evidence_ids:
        raise ValueError(f"{game_id}: forecast target appears in its own evidence")
    if kickoff <= cutoff:
        raise ValueError(f"{game_id}: forecast cutoff is not strictly before kickoff")


def _context_prior_teams(
    bundle: dict[str, object],
    season: int,
    fbs_prior_rows: dict[str, dict[str, object]],
    fcs_names: dict[str, str],
) -> list[Team]:
    teams: list[Team] = []
    for team_id, row in sorted(fbs_prior_rows.items(), key=lambda item: issue201._team_id_key(item[0])):
        teams.append(Team(team_id, str(row["team_name"]), "fbs", np.asarray(row["pmf"], dtype=float)))
    populations = bundle.get("fcs_population_by_season")
    if not isinstance(populations, dict) or str(season) not in populations:
        raise ValueError(f"#200 bundle lacks the {season} FCS fallback population")
    population = int(populations[str(season)])
    teams.extend(
        Team(team_id, name, "fcs", np.full(population, 1.0 / population))
        for team_id, name in sorted(fcs_names.items(), key=lambda item: issue201._team_id_key(item[0]))
    )
    return teams


def _replay_candidate(
    *,
    bundle: dict[str, object],
    season: int,
    prior_rows: dict[str, dict[str, object]],
    fcs_names: dict[str, str],
    games: list[Game],
    base_likelihood: LikelihoodV1,
    adapter: ContextLikelihoodAdapter,
) -> PosteriorResult:
    teams = _context_prior_teams(bundle, season, prior_rows, fcs_names)
    configuration = bundle.get("inference_configuration")
    if not isinstance(configuration, dict):
        raise TypeError("#200 bundle has no inference settings")
    return infer_posterior(
        teams,
        games,
        base_likelihood,
        factor_builder=adapter.factor,
        factor_cache_key=adapter.factor_cache_key,
        max_iterations=int(configuration["max_iterations"]),
        tolerance=float(configuration["tolerance"]),
        damping=float(configuration["damping"]),
    )


def _pmf_digest(pmfs: dict[str, np.ndarray]) -> str:
    ordered = {
        team_id: np.asarray(pmf, dtype=np.float64).tolist()
        for team_id, pmf in sorted(pmfs.items(), key=lambda item: issue201._team_id_key(item[0]))
    }
    return sha256_json(ordered)


def _posterior_rows(
    *,
    season: int,
    cutoff: str,
    stage: str,
    arms: dict[str, tuple[dict[str, Team], PosteriorResult, dict[str, object]]],
    evidence: dict[str, object],
    fbs_prior_rows: dict[str, dict[str, object]],
    fcs_population: int,
) -> list[dict[str, object]]:
    output = []
    for arm in MODEL_ARMS:
        teams, result, _diagnostics = arms[arm]
        for team_id, team in sorted(teams.items(), key=lambda item: issue201._team_id_key(item[0])):
            posterior = np.asarray(result.pmfs[team_id], dtype=np.float64)
            if posterior.ndim != 1 or not np.isfinite(posterior).all() or np.any(posterior < 0) or not np.isclose(float(posterior.sum()), 1.0, atol=1e-8, rtol=0.0):
                raise ValueError(f"{season}/{cutoff}/{arm}/{team_id}: invalid full posterior PMF")
            posterior = posterior / float(posterior.sum())
            prior = (
                np.asarray(fbs_prior_rows[team_id]["pmf"], dtype=np.float64)
                if team.subdivision == "fbs"
                else np.full(fcs_population, 1.0 / fcs_population, dtype=np.float64)
            )
            prior = prior / float(prior.sum())
            output.append(
                {
                    "season": season,
                    "forecast_stage": stage,
                    "cutoff_utc": cutoff,
                    "model_arm": arm,
                    "team_id": team_id,
                    "team_name": team.name,
                    "subdivision": team.subdivision,
                    "rank_population": len(posterior),
                    "prior_pmf": prior.tolist(),
                    "posterior_pmf": posterior.tolist(),
                    "posterior_pmf_sha256": sha256_json(posterior.tolist()),
                    "evidence_ids_sha256": evidence["included_game_ids_sha256"],
                    "evidence_rows_sha256": evidence["included_game_rows_sha256"],
                }
            )
    return output


def _forecast_row(
    *,
    metadata: dict[str, object],
    outcome: dict[str, object],
    arm: str,
    teams: dict[str, Team],
    likelihood: LikelihoodV1,
    adapter: ContextLikelihoodAdapter | None,
    evidence: dict[str, object],
    stage: str,
    cutoff: str,
) -> dict[str, object]:
    home_id, away_id = str(metadata["home_id"]), str(metadata["away_id"])
    if home_id not in teams or away_id not in teams:
        raise KeyError(f"{metadata['game_id']}: a FBS–FBS forecast team is absent")
    home, away = teams[home_id], teams[away_id]
    if home.subdivision != "fbs" or away.subdivision != "fbs":
        raise ValueError(f"{metadata['game_id']}: primary cohort includes a non-FBS team")
    scheduled = ScheduledGame(
        game_id=str(metadata["game_id"]),
        home_id=home_id,
        away_id=away_id,
        home_subdivision="fbs",
        away_subdivision="fbs",
        neutral_site=bool(metadata["neutral_site"]),
        season_type=str(metadata["season_type"]),
        date=str(metadata["kickoff_utc"]),
        schedule_status="completed_source_schedule",
    )
    forecast_game = Game(
        str(metadata["game_id"]),
        home_id,
        away_id,
        "fbs",
        "fbs",
        0,
        0,
        bool(metadata["neutral_site"]),
    )
    if adapter is None:
        scoring_likelihood = likelihood
        locations, weights = predictive_components(
            scheduled, home, away, scoring_likelihood
        )
        context_shift = 0.0
        home_context_hash = away_context_hash = ""
    else:
        locations, weights, scoring_likelihood = adapter.predictive_components(
            scheduled, forecast_game, home, away
        )
        context_shift, home_context_hash, away_context_hash = adapter.context_delta(
            forecast_game
        )
    scoring_locations = locations - context_shift
    summary = _predictive_summary_with_shift(
        scoring_locations, weights, scoring_likelihood, context_shift
    )
    actual_margin = int(outcome["home_points"]) - int(outcome["away_points"])
    # CRPS and NLL are translation invariant. Centering both locations and the
    # observed margin keeps quadrature surfaces reusable across Context shifts.
    scoring_margin = float(actual_margin) - context_shift
    crps, error, order, tail_multiple = issue201.mixture_margin_crps(
        scoring_locations, weights, scoring_margin, scoring_likelihood.scale,
        scoring_likelihood.degrees_of_freedom, tolerance=CRPS_TOLERANCE,
    )
    margin_nll = issue201.mixture_margin_nll(
        scoring_locations, weights, scoring_margin, scoring_likelihood.scale,
        scoring_likelihood.degrees_of_freedom,
    )
    actual_home_win = 1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
    return {
        "forecast_id": sha256_json({"stage": stage, "season": metadata["season"], "game_id": metadata["game_id"], "cutoff": cutoff, "model_arm": arm}),
        "forecast_stage": stage,
        "season": int(metadata["season"]),
        "cutoff_utc": cutoff,
        "game_id": str(metadata["game_id"]),
        "phase": metadata.get("phase"),
        "forecast_population": issue201.PRIMARY_POPULATION,
        "home_id": home_id,
        "home_team": metadata["home_team"],
        "home_subdivision": "fbs",
        "away_id": away_id,
        "away_team": metadata["away_team"],
        "away_subdivision": "fbs",
        "kickoff_source": metadata["kickoff_source"],
        "kickoff_utc": metadata["kickoff_utc"],
        "kickoff_precision": metadata["kickoff_precision"],
        "schedule_availability": metadata["schedule_availability"],
        "neutral_site": bool(metadata["neutral_site"]),
        "conference_game": bool(metadata["conference_game"]),
        "actual_home_points": int(outcome["home_points"]),
        "actual_away_points": int(outcome["away_points"]),
        "actual_home_margin": actual_margin,
        "model_arm": arm,
        "policy": "existing",
        "prior_family": "Context",
        "evidence_game_count": evidence["included_game_count"],
        "evidence_ids_sha256": evidence["included_game_ids_sha256"],
        "evidence_rows_sha256": evidence["included_game_rows_sha256"],
        "home_posterior_pmf_sha256": sha256_json(home.prior.tolist()),
        "away_posterior_pmf_sha256": sha256_json(away.prior.tolist()),
        "context_feature_mean_adjustment_points": context_shift,
        "home_context_feature_sha256": home_context_hash,
        "away_context_feature_sha256": away_context_hash,
        "predictive_scale": scoring_likelihood.scale,
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
        "coverage_50": int(summary.margin_interval_50[0] <= actual_margin <= summary.margin_interval_50[1]),
        "coverage_80": int(summary.margin_interval_80[0] <= actual_margin <= summary.margin_interval_80[1]),
        "coverage_95": int(summary.margin_interval_95[0] <= actual_margin <= summary.margin_interval_95[1]),
    }


def _predictive_summary_with_shift(
    centered_locations: np.ndarray,
    weights: np.ndarray,
    likelihood: LikelihoodV1,
    location_shift: float,
) -> issue201.PredictiveMarginSummary:
    """Translate mixture quantiles and recompute win probability at zero."""
    summary = issue201._predictive_summary_from_components(
        centered_locations, weights, likelihood
    )
    if location_shift == 0.0:
        return summary
    negative_probability = float(
        np.dot(
            weights,
            stdtr(
                likelihood.degrees_of_freedom,
                (-location_shift - centered_locations) / likelihood.scale,
            ),
        )
    )
    home_win, away_win = win_probabilities(negative_probability)
    return replace(
        summary,
        expected_home_margin=summary.expected_home_margin + location_shift,
        median_home_margin=summary.median_home_margin + location_shift,
        home_win_probability=home_win,
        away_win_probability=away_win,
        margin_interval_50=(
            summary.margin_interval_50[0] + location_shift,
            summary.margin_interval_50[1] + location_shift,
        ),
        margin_interval_80=(
            summary.margin_interval_80[0] + location_shift,
            summary.margin_interval_80[1] + location_shift,
        ),
        margin_interval_95=(
            summary.margin_interval_95[0] + location_shift,
            summary.margin_interval_95[1] + location_shift,
        ),
    )


def _paired_forecasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["forecast_stage"]), int(row["season"]), str(row["game_id"]), str(row["model_arm"])): row
        for row in rows
    }
    keys = sorted({key[:3] for key in by_key})
    comparisons = ((GENERIC_ARM, BASELINE_ARM), (CONTEXT_ARM, BASELINE_ARM), (CONTEXT_ARM, GENERIC_ARM))
    metrics = ("crps", "margin_nll", "home_win_brier", "median_absolute_error", "coverage_50", "coverage_80", "coverage_95")
    output = []
    for key in keys:
        members = {arm: by_key.get((*key, arm)) for arm in MODEL_ARMS}
        if any(row is None for row in members.values()):
            raise ValueError(f"{key}: forecast arms do not share a complete cohort")
        baseline = members[BASELINE_ARM]
        for row in members.values():
            if row["cutoff_utc"] != baseline["cutoff_utc"] or row["evidence_ids_sha256"] != baseline["evidence_ids_sha256"] or row["evidence_rows_sha256"] != baseline["evidence_rows_sha256"]:
                raise ValueError(f"{key}: forecast arms differ in cutoff or evidence")
        common = {
            "forecast_stage": key[0], "season": key[1], "game_id": key[2],
            "cutoff_utc": baseline["cutoff_utc"], "phase": baseline["phase"],
            "home_id": baseline["home_id"], "away_id": baseline["away_id"],
            "evidence_ids_sha256": baseline["evidence_ids_sha256"],
            "evidence_rows_sha256": baseline["evidence_rows_sha256"],
        }
        for candidate, reference in comparisons:
            output.append({
                **common,
                "candidate_arm": candidate,
                "reference_arm": reference,
                **{f"{metric}_candidate": members[candidate][metric] for metric in metrics},
                **{f"{metric}_reference": members[reference][metric] for metric in metrics},
                **{f"{metric}_delta": float(members[candidate][metric]) - float(members[reference][metric]) for metric in metrics},
            })
    return output


def _gain_concentration(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    gain_by_team: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    for row in paired:
        if row["forecast_stage"] != "rolling":
            continue
        for team_id in (str(row["home_id"]), str(row["away_id"])):
            gain_by_team[
                (
                    str(row["candidate_arm"]),
                    str(row["reference_arm"]),
                    int(row["season"]),
                    str(row["phase"]),
                    team_id,
                )
            ].append(float(row["crps_delta"]))
    return [
        {
            "candidate_arm": candidate_arm,
            "reference_arm": reference_arm,
            "season": season,
            "phase": phase,
            "team_id": team_id,
            "game_appearances": len(values),
            "mean_crps_delta": float(np.mean(values)),
            "sum_crps_delta": float(np.sum(values)),
            "candidate_better_games": sum(value < 0 for value in values),
            "candidate_worse_games": sum(value > 0 for value in values),
        }
        for (
            candidate_arm,
            reference_arm,
            season,
            phase,
            team_id,
        ), values in sorted(gain_by_team.items())
    ]


def _rank_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ("nll", "crps", "expected_rank_error", "interval_80_width", "interval_80_target_mass")
    by_team: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_team[(str(row["forecast_stage"]), str(row["model_arm"]), int(row["season"]), str(row["team_id"]))].append(row)
    team_seasons = []
    for (stage, arm, season, team_id), members in sorted(by_team.items()):
        team_seasons.append({
            "forecast_stage": stage, "model_arm": arm, "season": season,
            "team_id": team_id, "checkpoints": len(members),
            **{metric: float(np.mean([float(row[metric]) for row in members])) for metric in metrics},
        })
    output = []
    for stage in ("preseason", "rolling"):
        for arm in MODEL_ARMS:
            for season in EVALUATION_SEASONS:
                members = [row for row in team_seasons if row["forecast_stage"] == stage and row["model_arm"] == arm and row["season"] == season]
                if members:
                    output.append({
                        "aggregation": "season_team_season_weight_one", "forecast_stage": stage,
                        "season": season, "model_arm": arm, "team_seasons": len(members),
                        "checkpoints": sum(int(row["checkpoints"]) for row in members),
                        **{metric: float(np.mean([float(row[metric]) for row in members])) for metric in metrics},
                    })
            members = [row for row in team_seasons if row["forecast_stage"] == stage and row["model_arm"] == arm]
            if members:
                output.append({
                    "aggregation": "pooled_equal_team_season", "forecast_stage": stage,
                    "season": "2023-2025", "model_arm": arm, "team_seasons": len(members),
                    "checkpoints": sum(int(row["checkpoints"]) for row in members),
                    **{metric: float(np.mean([float(row[metric]) for row in members])) for metric in metrics},
                })
    return output


def _summary_row(summaries: list[dict[str, object]], arm: str, stage: str, season: str, phase: str = "all_phases") -> dict[str, object]:
    matches = [row for row in summaries if row["model_arm"] == arm and row["forecast_stage"] == stage and str(row["season"]) == season and row["phase"] == phase and row["forecast_population"] == issue201.PRIMARY_POPULATION]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicate forecast summary {arm}/{stage}/{season}/{phase}")
    return matches[0]


def _add_phase_aggregates(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add pooled and equal-season rows for each forecast phase.

    The shared #201 summarizer emits phase rows by season and pooled rows for
    all phases. Issue 213 also needs phase-specific pooled rows for its early
    guardrails, so construct those here without changing the shared reference.
    """
    result = list(summaries)
    phase_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        phase = str(row["phase"])
        if phase == "all_phases" or str(row["season"]) not in {
            str(season) for season in EVALUATION_SEASONS
        }:
            continue
        phase_groups[
            (str(row["forecast_stage"]), str(row["model_arm"]), phase)
        ].append(row)

    for (stage, arm, phase), members in sorted(phase_groups.items()):
        games = sum(int(row["games"]) for row in members)
        if games <= 0:
            continue
        metric_names = [
            key
            for key, value in members[0].items()
            if key != "games" and isinstance(value, (int, float))
        ]
        result.extend(
            [
                {
                    "forecast_stage": stage,
                    "forecast_population": issue201.PRIMARY_POPULATION,
                    "model_arm": arm,
                    "season": "pooled_equal_game",
                    "phase": phase,
                    "games": games,
                    **{
                        metric: float(
                            sum(
                                float(row[metric]) * int(row["games"])
                                for row in members
                            )
                            / games
                        )
                        for metric in metric_names
                    },
                },
                {
                    "forecast_stage": stage,
                    "forecast_population": issue201.PRIMARY_POPULATION,
                    "model_arm": arm,
                    "season": "equal_season_sensitivity",
                    "phase": phase,
                    "games": games,
                    **{
                        metric: float(np.mean([float(row[metric]) for row in members]))
                        for metric in metric_names
                    },
                },
            ]
        )
    return result


def _coverage_error(row: dict[str, object]) -> float:
    return float(np.mean([abs(float(row[f"coverage_{level}"]) - level / 100) for level in (50, 80, 95)]))


def _primary_comparison(summaries: list[dict[str, object]], candidate: str, reference: str) -> dict[str, object]:
    pooled = float(_summary_row(summaries, candidate, "rolling", "pooled_equal_game")["crps"]) - float(_summary_row(summaries, reference, "rolling", "pooled_equal_game")["crps"])
    equal = float(_summary_row(summaries, candidate, "rolling", "equal_season_sensitivity")["crps"]) - float(_summary_row(summaries, reference, "rolling", "equal_season_sensitivity")["crps"])
    by_season = {
        str(season): float(_summary_row(summaries, candidate, "rolling", str(season))["crps"]) - float(_summary_row(summaries, reference, "rolling", str(season))["crps"])
        for season in EVALUATION_SEASONS
    }
    wins = sum(value < 0 for value in by_season.values())
    return {
        "candidate_arm": candidate, "reference_arm": reference,
        "pooled_equal_game_crps_delta": pooled,
        "equal_season_crps_delta": equal,
        "season_crps_delta": by_season,
        "seasons_with_lower_crps": wins,
        "practical_gain_pass": pooled <= -PRACTICAL_GAIN and equal <= -PRACTICAL_GAIN and wins >= 2,
        "rule": "pooled and equal-season CRPS each at least 0.05 lower; lower CRPS in at least two of three seasons",
    }


def _guardrail(summaries: list[dict[str, object]], candidate: str) -> dict[str, object]:
    results = {}
    passed = True
    for stage, phase in (("preseason", "all_phases"), ("rolling", "pre_thanksgiving")):
        base = _summary_row(summaries, BASELINE_ARM, stage, "pooled_equal_game", phase)
        cand = _summary_row(summaries, candidate, stage, "pooled_equal_game", phase)
        pooled_crps_delta = float(cand["crps"]) - float(base["crps"])
        pooled_coverage_delta = _coverage_error(cand) - _coverage_error(base)
        season_crps = {}
        season_coverage = {}
        for season in EVALUATION_SEASONS:
            base_s = _summary_row(summaries, BASELINE_ARM, stage, str(season), phase)
            cand_s = _summary_row(summaries, candidate, stage, str(season), phase)
            season_crps[str(season)] = float(cand_s["crps"]) - float(base_s["crps"])
            season_coverage[str(season)] = _coverage_error(cand_s) - _coverage_error(base_s)
        stage_pass = (
            pooled_crps_delta <= GUARDRAIL_POOLED_CRPS
            and max(season_crps.values()) <= GUARDRAIL_SEASON_CRPS
            and pooled_coverage_delta <= GUARDRAIL_POOLED_COVERAGE
            and max(season_coverage.values()) <= GUARDRAIL_SEASON_COVERAGE
        )
        passed = passed and stage_pass
        results["preseason" if stage == "preseason" else "pre_thanksgiving"] = {
            "pooled_crps_deterioration": pooled_crps_delta,
            "pooled_crps_limit": GUARDRAIL_POOLED_CRPS,
            "season_crps_deterioration": season_crps,
            "season_crps_limit": GUARDRAIL_SEASON_CRPS,
            "pooled_mean_absolute_coverage_error_increase": pooled_coverage_delta,
            "pooled_coverage_error_limit": GUARDRAIL_POOLED_COVERAGE,
            "season_mean_absolute_coverage_error_increase": season_coverage,
            "season_coverage_error_limit": GUARDRAIL_SEASON_COVERAGE,
            "passed": stage_pass,
        }
    return {"candidate_arm": candidate, "passed": passed, "by_phase": results}


def _decision(summaries: list[dict[str, object]]) -> dict[str, object]:
    generic_vs_base = _primary_comparison(summaries, GENERIC_ARM, BASELINE_ARM)
    context_vs_base = _primary_comparison(summaries, CONTEXT_ARM, BASELINE_ARM)
    context_vs_generic = _primary_comparison(summaries, CONTEXT_ARM, GENERIC_ARM)
    generic_guardrail = _guardrail(summaries, GENERIC_ARM)
    context_guardrail = _guardrail(summaries, CONTEXT_ARM)
    if context_vs_base["practical_gain_pass"] and context_vs_generic["practical_gain_pass"] and context_guardrail["passed"]:
        conclusion = "Context-dependent correction merits independent confirmation"
    elif generic_vs_base["practical_gain_pass"] and not context_vs_generic["practical_gain_pass"] and generic_guardrail["passed"]:
        conclusion = "Prefer generic recalibration for further confirmation"
    else:
        conclusion = "Retain unmodified Context / no useful replacement established"
    return {
        "conclusion": conclusion,
        "comparisons": {
            "generic_vs_unmodified_context": generic_vs_base,
            "context_correction_vs_unmodified_context": context_vs_base,
            "context_correction_vs_generic": context_vs_generic,
        },
        "early_guardrails_against_unmodified_context": {
            GENERIC_ARM: generic_guardrail,
            CONTEXT_ARM: context_guardrail,
        },
        "interpretation": "descriptive exploratory development results; repeated games, teams, and checkpoints are not independent confirmation",
        "confirmation_plan": "If a candidate is promising, freeze its inputs, parameters, and weekly predictions before the next season's outcomes are available; then score the untouched season against the same unmodified Context and generic controls under a separately preregistered protocol.",
    }


def _verify_baseline_parity(
    rows: list[dict[str, object]],
    issue201_root: Path,
    *,
    baseline_pmf_checks: int,
) -> dict[str, object]:
    reference_rows = [
        row for row in _read_csv(issue201_root / "forecast_scores.csv")
        if row.get("model_arm") == BASELINE_ARM
        and int(row.get("season", "0")) in EVALUATION_SEASONS
        and row.get("forecast_population") == issue201.PRIMARY_POPULATION
        and row.get("forecast_stage") in {"preseason", "rolling"}
    ]
    current = [row for row in rows if row["model_arm"] == BASELINE_ARM]
    key = lambda row: (str(row["forecast_stage"]), int(row["season"]), str(row["game_id"]))
    reference_by_key = {key(row): row for row in reference_rows}
    current_by_key = {key(row): row for row in current}
    if len(reference_by_key) != len(reference_rows) or len(current_by_key) != len(current):
        raise ValueError("#201 baseline forecast rows contain duplicate keys")
    if set(reference_by_key) != set(current_by_key):
        missing = sorted(set(reference_by_key) - set(current_by_key))[:10]
        extra = sorted(set(current_by_key) - set(reference_by_key))[:10]
        raise ValueError(f"#201 baseline cohort mismatch; missing={missing}, extra={extra}")
    numeric = (
        "expected_home_margin", "median_home_margin", "home_win_probability", "away_win_probability", "tie_probability",
        "margin_interval_50_low", "margin_interval_50_high", "margin_interval_80_low", "margin_interval_80_high",
        "margin_interval_95_low", "margin_interval_95_high", "crps", "crps_numerical_error_estimate",
        "crps_quadrature_order", "crps_tail_multiple", "margin_nll", "median_absolute_error", "home_win_brier", "coverage_50", "coverage_80", "coverage_95",
    )
    max_difference = 0.0
    checks = 0
    for identity, row in current_by_key.items():
        reference = reference_by_key[identity]
        for field in (
            "cutoff_utc", "phase", "home_id", "away_id", "forecast_population",
            "actual_home_points", "actual_away_points", "evidence_ids_sha256", "evidence_rows_sha256",
        ):
            if str(row[field]) != str(reference[field]):
                raise ValueError(f"#201 baseline {identity}/{field} mismatch")
        for field in numeric:
            difference = abs(float(row[field]) - float(reference[field]))
            max_difference = max(max_difference, difference)
            checks += 1
            if difference > IDENTITY_TOLERANCE:
                raise ValueError(f"#201 baseline parity failed for {identity}/{field}: {difference}")
    return {
        "passed": True,
        "forecast_rows_compared": len(current),
        "numeric_values_compared": checks,
        "max_absolute_metric_difference": max_difference,
        "tolerance": IDENTITY_TOLERANCE,
        "baseline_pmf_team_game_checks": baseline_pmf_checks,
        "baseline_pmfs": "Fresh #200 Context_existing replay PMFs were checked for all baseline forecast team-game references, normalized, and serialized with per-team hashes in posterior_pmfs.jsonl.gz. #201 did not retain the PMF arrays or their hashes, so direct cross-artifact PMF hash comparison is unavailable.",
        "baseline_score_source": "Freshly replayed Context_existing posterior states were scored through the issue 213 prediction path and compared with the hash-verified #201 per-game forecast_scores.csv under the frozen tolerance.",
    }


def _verify_baseline_checkpoint(
    *,
    season: int,
    stage: str,
    cutoff: str,
    teams: dict[str, Team],
    target_games: dict[str, tuple[dict[str, object], dict[str, object]]],
    evidence: dict[str, object],
    reference_by_key: dict[tuple[str, int, str], dict[str, str]],
) -> int:
    """Bind the fresh baseline posterior teams to #201's frozen forecast rows."""
    checks = 0
    for game_id, (metadata, outcome) in target_games.items():
        key = (stage, season, game_id)
        reference = reference_by_key.get(key)
        if reference is None:
            raise ValueError(f"#201 baseline is missing {stage}/{season}/{game_id}")
        for field, expected in (
            ("cutoff_utc", cutoff),
            ("home_id", str(metadata["home_id"])),
            ("away_id", str(metadata["away_id"])),
            ("evidence_ids_sha256", str(evidence["included_game_ids_sha256"])),
            ("evidence_rows_sha256", str(evidence["included_game_rows_sha256"])),
            ("actual_home_points", str(outcome["home_points"])),
            ("actual_away_points", str(outcome["away_points"])),
        ):
            if str(reference.get(field)) != expected:
                raise ValueError(
                    f"#201 baseline {stage}/{season}/{game_id}/{field} mismatch"
                )
        for team_id in (str(metadata["home_id"]), str(metadata["away_id"])):
            if team_id not in teams:
                raise ValueError(f"{season}/{cutoff}: baseline team {team_id} is absent")
            pmf = np.asarray(teams[team_id].prior, dtype=np.float64)
            if (
                pmf.ndim != 1
                or not np.isfinite(pmf).all()
                or np.any(pmf < 0)
                or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
            ):
                raise ValueError(f"{season}/{cutoff}/{team_id}: baseline PMF is invalid")
            checks += 1
    return checks


def _empty_evidence_record(
    season: int,
    diagnostics: dict[str, dict[str, object]],
    team_count: int,
    fcs_count: int,
) -> dict[str, object]:
    return {
        "season": season,
        "cutoff_utc": "preseason_no_evidence",
        "policy": "existing",
        "safety_lag_hours": 48,
        "included_game_ids": [],
        "included_game_ids_sha256": sha256_json([]),
        "included_game_rows_sha256": sha256_json([]),
        "included_game_count": 0,
        "excluded_game_ids": [],
        "excluded_game_ids_sha256": sha256_json([]),
        "excluded_game_count": 0,
        "latest_included_kickoff_utc": None,
        "replay_diagnostics": diagnostics,
        "posterior_team_count": team_count,
        "fcs_fallback_team_count": fcs_count,
    }


def _run_arm_replays(
    *,
    bundle: dict[str, object],
    season: int,
    priors: dict[str, dict[str, dict[str, object]]],
    fcs_names: dict[str, str],
    games: list[Game],
    base: LikelihoodV1,
    adapters: dict[str, ContextLikelihoodAdapter],
) -> dict[str, tuple[dict[str, Team], PosteriorResult, dict[str, object]]]:
    results: dict[str, tuple[dict[str, Team], PosteriorResult, dict[str, object]]] = {}
    baseline = replay_pipeline_bundle(
        bundle,
        season=season,
        policy="existing",
        prior_family="Context",
        games=games,
        fcs_team_names=fcs_names,
    )
    baseline_diag = _validate_posterior(baseline, f"{season}/{BASELINE_ARM}")
    results[BASELINE_ARM] = (
        issue201._build_posterior_teams(
            baseline.pmfs, priors[BASELINE_ARM], fcs_names
        ),
        baseline,
        baseline_diag,
    )
    for arm in (GENERIC_ARM, CONTEXT_ARM):
        result = _replay_candidate(
            bundle=bundle,
            season=season,
            prior_rows=priors[BASELINE_ARM],
            fcs_names=fcs_names,
            games=games,
            base_likelihood=base,
            adapter=adapters[arm],
        )
        diagnostics = _validate_posterior(result, f"{season}/{arm}")
        if set(result.pmfs) != set(baseline.pmfs):
            raise ValueError(f"{season}/{arm}: candidate posterior team population changed")
        results[arm] = (
            issue201._build_posterior_teams(
                result.pmfs, priors[BASELINE_ARM], fcs_names
            ),
            result,
            diagnostics,
        )
    return results


def _parallel_forecasts(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    if not jobs:
        return []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        return list(pool.map(lambda job: _forecast_row(**job), jobs))  # type: ignore[arg-type]


def _rank_scores_from_pmfs(
    *,
    posterior_rows: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    targets: dict[int, dict[str, dict[str, object]]],
    target_dates: dict[int, Any],
    outcome_rows_by_season: dict[int, list[dict[str, object]]],
    checkpoint_phases: dict[tuple[int, str], str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pmfs_by_checkpoint: dict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in posterior_rows:
        pmfs_by_checkpoint[
            (
                int(row["season"]),
                str(row["forecast_stage"]),
                str(row["cutoff_utc"]),
                str(row["model_arm"]),
            )
        ].append(row)
    evidence_by_checkpoint = {
        (int(row["season"]), str(row["cutoff_utc"])): row for row in evidence_rows
    }
    kickoff_by_game = {
        (season, str(row["game_id"])): row.get("kickoff_utc")
        for season, rows in outcome_rows_by_season.items()
        for row in rows
    }
    score_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    checked: set[tuple[int, str, str]] = set()
    checkpoints = sorted(pmfs_by_checkpoint)
    for season, stage, cutoff, arm in checkpoints:
        if arm != BASELINE_ARM or (season, stage, cutoff) in checked:
            continue
        checked.add((season, stage, cutoff))
        if cutoff == "preseason_no_evidence":
            cutoff_time = datetime.min.replace(tzinfo=UTC)
        else:
            cutoff_time = datetime.fromisoformat(cutoff).astimezone(UTC)
        target_start = datetime.combine(target_dates[season], time.min, tzinfo=UTC)
        evidence = evidence_by_checkpoint[(season, cutoff)]
        if cutoff_time >= target_start:
            exclusions.append(
                {
                    "season": season,
                    "forecast_stage": stage,
                    "cutoff_utc": cutoff,
                    "reason": "cutoff_not_strictly_before_pre_bowl_target_date",
                    "target_observation_date": target_dates[season].isoformat(),
                    "latest_included_kickoff_utc": evidence["latest_included_kickoff_utc"] or "",
                }
            )
            continue
        unsafe = [
            game_id
            for game_id in evidence["included_game_ids"]
            if (kickoff := kickoff_by_game.get((season, str(game_id)))) is None
            or datetime.fromisoformat(str(kickoff)).astimezone(UTC) >= target_start
        ]
        if unsafe:
            raise ValueError(f"{season}/{cutoff}: ranking evidence includes a game on/after target date: {unsafe[:5]}")
        arm_rows = {
            other_arm: pmfs_by_checkpoint[(season, stage, cutoff, other_arm)]
            for other_arm in MODEL_ARMS
        }
        by_arm_team = {
            other_arm: {str(row["team_id"]): row for row in rows if row["subdivision"] == "fbs"}
            for other_arm, rows in arm_rows.items()
        }
        base_ids = set(by_arm_team[BASELINE_ARM])
        if any(set(rows) != base_ids for rows in by_arm_team.values()):
            raise ValueError(f"{season}/{cutoff}: posterior FBS team sets differ across arms")
        for team_id in sorted(base_ids, key=issue201._team_id_key):
            target = targets[season].get(team_id)
            if target is None:
                exclusions.append(
                    {"season": season, "forecast_stage": stage, "cutoff_utc": cutoff, "team_id": team_id, "reason": "missing_corrected_pre_bowl_target"}
                )
                continue
            for other_arm in MODEL_ARMS:
                row = by_arm_team[other_arm][team_id]
                pmf = np.asarray(row["posterior_pmf"], dtype=np.float64)
                target_pmf = np.asarray(target["pmf"], dtype=np.float64)
                if len(pmf) != int(target["population"]) or pmf.shape != target_pmf.shape:
                    exclusions.append(
                        {
                            "season": season,
                            "forecast_stage": stage,
                            "cutoff_utc": cutoff,
                            "team_id": team_id,
                            "model_arm": other_arm,
                            "reason": "posterior_and_target_rank_support_differ",
                            "posterior_population": len(pmf),
                            "target_population": target["population"],
                        }
                    )
                    continue
                score = issue201.score_rank_pmf(pmf, target_pmf)
                score_rows.append(
                    {
                        "season": season,
                        "forecast_stage": stage,
                        "checkpoint": "preseason" if stage == "preseason" else "weekly",
                        "cutoff_utc": cutoff,
                        "checkpoint_phase": "preseason" if stage == "preseason" else checkpoint_phases[(season, cutoff)],
                        "team_id": team_id,
                        "model_arm": other_arm,
                        "target_policy": "issue204_corrected_pre_bowl",
                        "target_observation_date": target_dates[season].isoformat(),
                        "rank_count": len(pmf),
                        "evidence_game_count": evidence["included_game_count"],
                        "evidence_ids_sha256": evidence["included_game_ids_sha256"],
                        "evidence_rows_sha256": evidence["included_game_rows_sha256"],
                        "posterior_pmf_sha256": row["posterior_pmf_sha256"],
                        "target_pmf_sha256": target["row_sha256"],
                        **score,
                    }
                )
    return score_rows, exclusions


def _rank_pmf_concentration(
    posterior_rows: list[dict[str, object]],
    checkpoint_phases: dict[tuple[int, str], str],
) -> list[dict[str, object]]:
    """Summarize complete FBS rank PMFs at every season/checkpoint/arm."""
    grouped: dict[tuple[int, str, str, str], list[np.ndarray]] = defaultdict(list)
    for row in posterior_rows:
        if row["subdivision"] != "fbs":
            continue
        season = int(row["season"])
        stage = str(row["forecast_stage"])
        cutoff = str(row["cutoff_utc"])
        arm = str(row["model_arm"])
        pmf = np.asarray(row["posterior_pmf"], dtype=np.float64)
        if (
            pmf.ndim != 1
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(float(pmf.sum()), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError(f"{season}/{cutoff}/{arm}: invalid rank PMF for concentration")
        grouped[(season, stage, cutoff, arm)].append(pmf)
    output = []
    for (season, stage, cutoff, arm), pmfs in sorted(grouped.items()):
        entropy = np.asarray(
            [
                -float(np.sum(pmf[pmf > 0] * np.log(pmf[pmf > 0])))
                for pmf in pmfs
            ],
            dtype=np.float64,
        )
        effective = np.exp(entropy)
        output.append(
            {
                "season": season,
                "forecast_stage": stage,
                "checkpoint": "preseason" if stage == "preseason" else "weekly",
                "cutoff_utc": cutoff,
                "checkpoint_phase": "preseason"
                if stage == "preseason"
                else checkpoint_phases[(season, cutoff)],
                "model_arm": arm,
                "fbs_team_count": len(pmfs),
                "mean_entropy_nats": float(np.mean(entropy)),
                "mean_effective_rank_count": float(np.mean(effective)),
                "mean_max_rank_probability": float(np.mean([pmf.max() for pmf in pmfs])),
                "mean_top_10_rank_mass": float(
                    np.mean([np.sort(pmf)[-10:].sum() for pmf in pmfs])
                ),
            }
        )
    return output


def _write_report(
    path: Path,
    *,
    decision: dict[str, object],
    summaries: list[dict[str, object]],
    baseline_parity: dict[str, object],
    forecast_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    protocol_sha256: str,
) -> None:
    lines = [
        "# Issue 213: Context dependent likelihood forecast experiment",
        "",
        f"**Conclusion: {decision['conclusion']}**",
        "",
        "This is a fixed exploratory replay of 2023–2025 using #195's saved forward fits. It is not an untouched confirmation and does not authorize production promotion.",
        "",
        f"Frozen protocol SHA-256: `{protocol_sha256}`.",
        "",
        "## Primary margin CRPS",
        "",
        "CRPS is lower when forecasts improve. Deltas below are candidate minus reference; negative favors the candidate.",
        "",
        "| Comparison | Pooled equal-game delta | Equal-season delta | Seasons lower | Practical rule |",
        "|---|---:|---:|---:|---|",
    ]
    comparisons = decision["comparisons"]
    for key, label in (
        ("generic_vs_unmodified_context", "Generic vs unmodified Context"),
        ("context_correction_vs_unmodified_context", "Context correction vs unmodified Context"),
        ("context_correction_vs_generic", "Context correction vs generic"),
    ):
        row = comparisons[key]
        lines.append(
            f"| {label} | {float(row['pooled_equal_game_crps_delta']):.5f} | {float(row['equal_season_crps_delta']):.5f} | {row['seasons_with_lower_crps']}/3 | {'pass' if row['practical_gain_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The comparisons share the same retained games, Context priors, and cutoff evidence. The experiment uses the saved 2023 fit trained on 2022, 2024 fit trained on 2022–2023, and 2025 fit trained on 2022–2024.",
            "",
            "## Replay and calibration coverage",
            "",
            f"Primary forecasts: {sum(row['forecast_stage'] == 'rolling' and row['model_arm'] == BASELINE_ARM for row in forecast_rows)} rolling game rows per arm; {sum(row['forecast_stage'] == 'preseason' and row['model_arm'] == BASELINE_ARM for row in forecast_rows)} separate preseason rows per arm.",
            f"Corrected pre-bowl team-checkpoint scores: {len(rank_rows)} rows across arms and stages.",
            f"#201 baseline parity: {'passed' if baseline_parity['passed'] else 'failed'} across {baseline_parity['forecast_rows_compared']} per-game rows; max numeric difference {float(baseline_parity['max_absolute_metric_difference']):.3g} (limit {IDENTITY_TOLERANCE:g}).",
            "",
            "Season, phase, margin NLL, win Brier, median absolute error, 50/80/95% coverage and widths, secondary ranking scores, and team/game gain concentration are in the accompanying CSV and compressed JSONL artifacts.",
            "The complete normalized rank PMFs are losslessly preserved in `posterior_pmfs.jsonl.gz`; per-season and per-cutoff entropy, effective rank count, maximum rank mass, and top-10 rank mass are in `ranking_pmf_concentration.csv`.",
            "Each future-game Student-t rank mixture is losslessly reconstructible from `predictive_mixtures.jsonl.gz`, its referenced team PMFs, and `predictive_model_parameters.json`.",
            "",
            "## Guardrails and interpretation",
            "",
            "Early preseason and pre-Thanksgiving CRPS and coverage-error guardrails are reported in `decision.json`. They are descriptive safeguards, not significance tests. Repeated games, teams, and checkpoints are dependent.",
            "",
            f"The frozen decision rule yields: **{decision['conclusion']}**. Any promising result needs a new protocol that freezes predictions before the next season's outcomes are available.",
            "",
            "A score improvement would not prove that rankings are wrong, that Context belongs in production likelihoods, or that the ranking measurements recover latent true ranks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_issue_213_replay(args: Any) -> dict[str, object]:
    protocol_sha = _verify_protocol(args.protocol)
    roots = _verify_frozen_inputs(
        corpus_root=args.corpus_root,
        issue195_root=args.issue195_root,
        issue200_root=args.issue200_root,
        issue201_root=args.issue201_root,
    )
    fits, fit_provenance = load_frozen_fits(args.issue195_root)
    bundle_path = args.issue200_root / "pipeline_bundle.json"
    bundle = read_pipeline_bundle(bundle_path)
    if sha256_file(bundle_path) != BASE_BUNDLE_SHA256:
        raise ValueError("issue 213 requires the exact #200 pipeline bundle used by #195/#201")
    issue200_manifest_path = args.issue200_root / "manifest.json"
    fitted_models_path = args.issue200_root / "fitted_models.json"
    bundle_checked, priors_by_season, issue200_provenance = issue201._load_issue200_inputs(
        bundle_path=bundle_path,
        manifest_path=issue200_manifest_path,
        fitted_models_path=fitted_models_path,
    )
    if sha256_json(bundle_checked) != sha256_json(bundle):
        raise ValueError("issue 200 bundle changed between provenance checks")
    feature_path = args.issue200_root / "context_feature_inputs.csv"
    feature_manifest_path = args.issue200_root / "preseason_feature_manifest.json"
    feature_rows = _read_context_features(
        feature_path, feature_manifest_path, target_seasons=tuple(range(2022, 2026))
    )
    feature_provenance = {
        "feature_input": _record(feature_path, "exact_issue200_context_feature_rows"),
        "feature_manifest": _record(
            feature_manifest_path, "exact_issue200_context_feature_lineage"
        ),
        "indexed_fbs_rows": len(feature_rows),
        "target_seasons": list(range(2022, 2026)),
    }
    schedule_plan_path = args.issue201_root / "schedule_plan.json"
    schedule_manifest_path = args.issue201_root / "schedule_plan.manifest.json"
    schedule_plan = json.loads(schedule_plan_path.read_text(encoding="utf-8"))
    issue201_protocol_sha = sha256_file(args.issue201_protocol)
    target_snapshot_path = args.target_snapshot_manifest or args.issue200_root / "target_snapshot_manifest.csv"
    schedule_windows_path = args.schedule_windows or args.corpus_root / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
    outcome_rows_by_season, game_source_files = issue201.verify_score_blind_plan(
        schedule_plan,
        schedule_plan_path,
        schedule_manifest_path,
        args.game_root,
        protocol_sha256=issue201_protocol_sha,
        target_snapshot_manifest_path=target_snapshot_path,
        schedule_windows_path=schedule_windows_path,
    )
    issue201_manifest = json.loads((args.issue201_root / "manifest.json").read_text(encoding="utf-8"))
    if issue201_manifest.get("outputs", {}).get("forecast_scores.csv", {}).get("sha256") != sha256_file(args.issue201_root / "forecast_scores.csv"):
        raise ValueError("#201 forecast scores differ from their manifest")
    if issue201_manifest.get("input_artifacts", {}).get("issue200_pipeline_bundle", {}).get("sha256") != BASE_BUNDLE_SHA256:
        raise ValueError("#201 was not run against the frozen #200 bundle")
    if schedule_plan.get("protocol_sha256") != issue201_protocol_sha:
        raise ValueError("#201 schedule plan was frozen under a different protocol")
    target_dates = issue201._read_target_snapshot_dates(target_snapshot_path)
    baseline_reference_rows = [
        row
        for row in _read_csv(args.issue201_root / "forecast_scores.csv")
        if row.get("model_arm") == BASELINE_ARM
        and int(row.get("season", "0")) in EVALUATION_SEASONS
        and row.get("forecast_population") == issue201.PRIMARY_POPULATION
        and row.get("forecast_stage") in {"preseason", "rolling"}
    ]
    reference_by_key = {
        (str(row["forecast_stage"]), int(row["season"]), str(row["game_id"])): row
        for row in baseline_reference_rows
    }
    if len(reference_by_key) != len(baseline_reference_rows):
        raise ValueError("#201 baseline forecast artifact contains duplicate game keys")
    schedule_seasons = {int(row["season"]): row for row in schedule_plan["seasons"]}
    base_likelihood = issue201._load_arm_likelihood(bundle, "existing")
    predictive_model_parameters = _predictive_model_parameters(base_likelihood, fits)
    forecasts: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    posterior_pmf_row_count = 0
    replay_evidence: list[dict[str, object]] = []
    forecast_exclusions: list[dict[str, object]] = []
    baseline_pmf_checks = 0
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    posterior_pmf_path = output_root / "posterior_pmfs.jsonl.gz"
    posterior_pmf_path.write_bytes(b"")
    prior_digests: dict[str, str] = {}
    missing_feature_keys: list[tuple[int, str]] = []

    for season in EVALUATION_SEASONS:
        # Keep the feature-aware factor and location caches bounded to one season.
        posterior_engine._GAME_FACTOR_CACHE.clear()
        predictive_engine._LOCATION_SURFACE_CACHE.clear()
        season_plan = schedule_seasons[season]
        season_outcomes = outcome_rows_by_season[season]
        scoreable, outcome_exclusions = issue201._scoreable_game_rows(season_plan, season_outcomes)
        primary_games = {
            game_id: item
            for game_id, item in scoreable.items()
            if item[0].get("forecast_population") == issue201.PRIMARY_POPULATION
        }
        forecast_exclusions.extend(outcome_exclusions)
        forecast_exclusions.extend(
            {**row, "exclusion_reason": row["plan_exclusion_reason"]}
            for row in season_plan["excluded_schedule_rows"]
        )
        fcs_names = issue201._fcs_names(season_outcomes)
        arm_priors = priors_by_season[str(season)]
        context_priors = arm_priors[BASELINE_ARM]
        if any(set(arm_priors[arm]) != set(context_priors) for arm in arm_priors):
            raise ValueError(f"{season}: #200 prior team sets differ across arms")
        prior_digests[str(season)] = sha256_json(
            {team_id: np.asarray(row["pmf"], dtype=np.float64).tolist() for team_id, row in sorted(context_priors.items(), key=lambda item: issue201._team_id_key(item[0]))}
        )
        for game in season_plan["games"]:
            if game.get("forecast_population") != issue201.PRIMARY_POPULATION:
                continue
            for team_id in (str(game["home_id"]), str(game["away_id"])):
                if (season, team_id) not in feature_rows:
                    missing_feature_keys.append((season, team_id))
        if missing_feature_keys:
            raise ValueError(f"#200 exact Context feature rows are missing for FBS game teams: {missing_feature_keys[:12]}")
        adapters = {
            arm: ContextLikelihoodAdapter(base_likelihood, fits[season], arm, feature_rows)
            for arm in (GENERIC_ARM, CONTEXT_ARM)
        }
        preseason_states = _run_arm_replays(
            bundle=bundle,
            season=season,
            priors=arm_priors,
            fcs_names=fcs_names,
            games=[],
            base=base_likelihood,
            adapters=adapters,
        )
        prior_pmfs = {arm: _pmf_digest(state[1].pmfs) for arm, state in preseason_states.items()}
        if len(set(prior_pmfs.values())) != 1:
            raise ValueError(f"{season}: arms changed the shared preseason Context PMFs")
        empty_diagnostics = {arm: state[2] for arm, state in preseason_states.items()}
        empty_evidence = _empty_evidence_record(
            season,
            empty_diagnostics,
            len(preseason_states[BASELINE_ARM][1].pmfs),
            len(fcs_names),
        )
        baseline_pmf_checks += _verify_baseline_checkpoint(
            season=season,
            stage="preseason",
            cutoff="preseason_no_evidence",
            teams=preseason_states[BASELINE_ARM][0],
            target_games=primary_games,
            evidence=empty_evidence,
            reference_by_key=reference_by_key,
        )
        empty_evidence["posterior_pmf_sha256_by_arm"] = prior_pmfs
        replay_evidence.append(empty_evidence)
        preseason_posterior_rows = _posterior_rows(
            season=season,
            cutoff="preseason_no_evidence",
            stage="preseason",
            arms=preseason_states,
            evidence=empty_evidence,
            fbs_prior_rows=context_priors,
            fcs_population=int(bundle["fcs_population_by_season"][str(season)]),
        )
        _append_jsonl_gzip(posterior_pmf_path, preseason_posterior_rows)
        posterior_pmf_row_count += len(preseason_posterior_rows)
        posterior_rows.extend(
            row for row in preseason_posterior_rows if row["subdivision"] == "fbs"
        )
        preseason_jobs: list[dict[str, object]] = []
        for arm in MODEL_ARMS:
            teams, _result, _diag = preseason_states[arm]
            for metadata, outcome in primary_games.values():
                preseason_jobs.append(
                    {
                        "metadata": metadata,
                        "outcome": outcome,
                        "arm": arm,
                        "teams": teams,
                        "likelihood": base_likelihood,
                        "adapter": None if arm == BASELINE_ARM else adapters[arm],
                        "evidence": empty_evidence,
                        "stage": "preseason",
                        "cutoff": "preseason_no_evidence",
                    }
                )
        forecasts.extend(_parallel_forecasts(preseason_jobs))

        games_by_cutoff: dict[str, list[str]] = defaultdict(list)
        plan_game_by_id = {str(row["game_id"]): row for row in season_plan["games"]}
        for game_id, (metadata, _outcome) in primary_games.items():
            games_by_cutoff[str(metadata["assigned_cutoff"])].append(game_id)
        for game_ids in games_by_cutoff.values():
            game_ids.sort(key=lambda game_id: (str(plan_game_by_id[game_id]["kickoff_utc"]), issue201._game_id_key(game_id)))
        for cutoff_index, cutoff_text in enumerate(season_plan["cutoffs_utc"], start=1):
            if cutoff_index == 1 or cutoff_index % 4 == 0:
                print(f"Issue 213: replaying {season} cutoff {cutoff_index}/{len(season_plan['cutoffs_utc'])}", flush=True)
            cutoff = datetime.fromisoformat(str(cutoff_text)).astimezone(UTC)
            future_ids = set(games_by_cutoff.get(str(cutoff_text), []))
            included, excluded = issue201.filter_evidence_rows(season_outcomes, cutoff, "existing")
            included_ids = {str(row["game_id"]) for row in included}
            if included_ids & future_ids:
                raise ValueError(f"{season}/{cutoff_text}: target game entered its own evidence")
            evidence_games = _deduplicate_games(
                [issue201.game_from_outcome(row) for row in included]
            )
            states = _run_arm_replays(
                bundle=bundle,
                season=season,
                priors=arm_priors,
                fcs_names=fcs_names,
                games=evidence_games,
                base=base_likelihood,
                adapters=adapters,
            )
            diagnostics = {arm: state[2] for arm, state in states.items()}
            evidence = issue201._evidence_record(
                season,
                cutoff,
                "existing",
                included,
                excluded,
                diagnostics,
                len(states[BASELINE_ARM][1].pmfs),
                len(fcs_names),
            )
            target_games = {
                game_id: primary_games[game_id]
                for game_id in sorted(future_ids, key=issue201._game_id_key)
            }
            baseline_pmf_checks += _verify_baseline_checkpoint(
                season=season,
                stage="rolling",
                cutoff=str(cutoff_text),
                teams=states[BASELINE_ARM][0],
                target_games=target_games,
                evidence=evidence,
                reference_by_key=reference_by_key,
            )
            evidence["posterior_pmf_sha256_by_arm"] = {
                arm: _pmf_digest(state[1].pmfs) for arm, state in states.items()
            }
            replay_evidence.append(evidence)
            checkpoint_posterior_rows = _posterior_rows(
                season=season,
                cutoff=str(cutoff_text),
                stage="rolling",
                arms=states,
                evidence=evidence,
                fbs_prior_rows=context_priors,
                fcs_population=int(bundle["fcs_population_by_season"][str(season)]),
            )
            _append_jsonl_gzip(posterior_pmf_path, checkpoint_posterior_rows)
            posterior_pmf_row_count += len(checkpoint_posterior_rows)
            posterior_rows.extend(
                row for row in checkpoint_posterior_rows if row["subdivision"] == "fbs"
            )
            jobs: list[dict[str, object]] = []
            for game_id in sorted(future_ids, key=issue201._game_id_key):
                metadata, outcome = primary_games[game_id]
                _validate_target_exclusion(
                    game_id=game_id,
                    cutoff=cutoff,
                    kickoff=datetime.fromisoformat(
                        str(metadata["kickoff_utc"])
                    ).astimezone(UTC),
                    evidence_ids=included_ids,
                )
                for arm in MODEL_ARMS:
                    jobs.append(
                        {
                            "metadata": metadata,
                            "outcome": outcome,
                            "arm": arm,
                            "teams": states[arm][0],
                            "likelihood": base_likelihood,
                            "adapter": None if arm == BASELINE_ARM else adapters[arm],
                            "evidence": evidence,
                            "stage": "rolling",
                            "cutoff": str(cutoff_text),
                        }
                    )
            forecasts.extend(_parallel_forecasts(jobs))

    baseline_parity = _verify_baseline_parity(
        forecasts,
        args.issue201_root,
        baseline_pmf_checks=baseline_pmf_checks,
    )
    predictive_mixture_rows = _predictive_mixture_rows(
        forecasts, posterior_rows, predictive_model_parameters
    )
    target_path = args.pre_bowl_targets or args.corpus_root / "data/processed/massey_prebowl_issue_204_v1/pre_bowl_target_distributions.csv"
    issue204_manifest_path = args.issue204_manifest or args.corpus_root / "data/processed/massey_prebowl_issue_204_v1/manifest.json"
    pre_bowl_targets, target_provenance = issue201.load_pre_bowl_targets(target_path, issue204_manifest_path)
    checkpoint_phases = {
        (int(season_row["season"]), str(row["cutoff_utc"])): str(row["phase"])
        for season_row in schedule_plan["seasons"]
        for row in season_row["cutoff_phases"]
    }
    rank_rows, rank_exclusions = _rank_scores_from_pmfs(
        posterior_rows=posterior_rows,
        evidence_rows=replay_evidence,
        targets=pre_bowl_targets,
        target_dates=target_dates,
        outcome_rows_by_season=outcome_rows_by_season,
        checkpoint_phases=checkpoint_phases,
    )
    pmf_concentration = _rank_pmf_concentration(posterior_rows, checkpoint_phases)
    summaries = []
    for stage in ("rolling", "preseason"):
        summaries.extend(issue201.summarize_forecast_scores(forecasts, stage=stage, population=issue201.PRIMARY_POPULATION))
    summaries = _add_phase_aggregates(summaries)
    paired = _paired_forecasts(forecasts)
    decision = _decision(summaries)
    rank_summaries = _rank_summary(rank_rows)
    gain_concentration = _gain_concentration(paired)

    outputs: dict[str, object] = {
        "forecast_scores.csv": forecasts,
        "predictive_mixtures.jsonl.gz": predictive_mixture_rows,
        "predictive_model_parameters.json": predictive_model_parameters,
        "paired_game_scores.csv": paired,
        "forecast_summary.csv": summaries,
        "ranking_scores.csv": rank_rows,
        "ranking_summary.csv": rank_summaries,
        "ranking_pmf_concentration.csv": pmf_concentration,
        "forecast_exclusions.csv": forecast_exclusions,
        "ranking_exclusions.csv": rank_exclusions,
        "gain_concentration.csv": gain_concentration,
        "baseline_parity.json": baseline_parity,
        "decision.json": decision,
        "replay_evidence.jsonl.gz": replay_evidence,
        "fit_provenance.json": fit_provenance,
    }
    for name, value in outputs.items():
        path = output_root / name
        if name.endswith(".jsonl.gz"):
            _write_jsonl_gzip(path, value)  # type: ignore[arg-type]
        elif name.endswith(".json"):
            _write_json(path, value)
        else:
            _write_csv(path, value)  # type: ignore[arg-type]
    report_path = output_root / "report.md"
    _write_report(
        report_path,
        decision=decision,
        summaries=summaries,
        baseline_parity=baseline_parity,
        forecast_rows=forecasts,
        rank_rows=rank_rows,
        protocol_sha256=protocol_sha,
    )
    inputs = {
        "issue195_manifest": _record(args.issue195_root / "manifest.json", "issue195_manifest"),
        "issue195_fit_records": _record(args.issue195_root / "fit_records.json", "frozen_forward_fits"),
        "issue200_manifest": _record(issue200_manifest_path, "issue200_rebuilt_manifest"),
        "issue200_bundle": _record(bundle_path, "matched_context_prior_and_likelihood_bundle"),
        "issue200_fitted_models": _record(fitted_models_path, "forward_prior_fit_origins"),
        "issue200_context_feature_inputs": _record(feature_path, "exact_context_predictor_rows"),
        "issue200_feature_manifest": _record(feature_manifest_path, "exact_context_feature_lineage"),
        "issue201_manifest": _record(args.issue201_root / "manifest.json", "frozen_forecast_reference"),
        "issue201_forecast_scores": _record(args.issue201_root / "forecast_scores.csv", "baseline_per_game_scores"),
        "issue201_schedule_plan": {
            **_record(schedule_plan_path, "score_blind_schedule_assignments"),
            "manifest_sha256": sha256_file(schedule_manifest_path),
        },
        "issue201_protocol": _record(args.issue201_protocol, "fixed_weekly_cutoff_contract"),
        "issue213_protocol": _record(args.protocol, "frozen_experiment_protocol"),
        "issue204_pre_bowl_targets": target_provenance,
        "raw_game_files": game_source_files,
        "corpus_inputs": roots["verified_issue195_inputs"],
    }
    manifest = {
        "artifact_kind": "issue_213_context_likelihood_forecast",
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "residual_training_seasons_by_target": {str(k): v for k, v in EXPECTED_FIT_TRAINING.items()},
        "arms": ARM_DEFINITIONS,
        "base_likelihood_sha256": BASE_LIKELIHOOD_SHA256,
        "issue200_bundle_sha256": BASE_BUNDLE_SHA256,
        "issue195_fit_records_sha256": FIT_RECORDS_SHA256,
        "issue195_feature_input_sha256": EXPECTED_CONTEXT_FEATURE_SHA256,
        "issue200_provenance": issue200_provenance,
        "context_feature_provenance": feature_provenance,
        "prior_pmf_sha256_by_season": prior_digests,
        "input_artifacts": inputs,
        "outputs": {},
    }
    manifest["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(output_root / "manifest.json", manifest)
    return {
        "status": "complete",
        "output_root": str(output_root),
        "forecast_rows": len(forecasts),
        "posterior_pmf_rows": posterior_pmf_row_count,
        "ranking_score_rows": len(rank_rows),
        "conclusion": decision["conclusion"],
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
    }
