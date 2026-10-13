"""Evidence-dependent History calibration and Context integration for issue 193."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gippyrank.research.context_incremental_information import (
    LAMBDA_GRID,
    OPTIMIZER_OPTIONS,
    PREPROCESSOR_SCALE_FLOOR,
    SINGLE_SEASON_LAMBDA,
    TIE_TOLERANCE,
    forward_chained_folds,
    kl_posterior_from_prior,
    normalized_pmf,
    rank_logit,
    target_cross_entropy,
)

EVIDENCE_FEATURE_NAMES = ("log1p_team_games", "kl_history_from_prior")
HISTORY_COEFFICIENT_NAMES = ("a", "b0", "b::log1p_team_games", "b::kl_history")
CONTEXT_COEFFICIENT_NAMES = (
    *HISTORY_COEFFICIENT_NAMES,
    "w0",
    "w::log1p_team_games",
    "w::kl_history",
)
ZERO_SUPPORT_TOLERANCE = 0.0


def _team_id_key(team_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(team_id)
    except ValueError:
        return 1, team_id


def count_distinct_game_ids(game_ids: Sequence[str]) -> int:
    """Count distinct nonempty game IDs, independent of input order."""
    normalized = [str(game_id).strip() for game_id in game_ids]
    if any(not game_id for game_id in normalized):
        raise ValueError("included evidence contains an empty game ID")
    return len(set(normalized))


def team_evidence_game_ids(
    included_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[str, ...]]:
    """Index unique included games by participant for team-specific n_i."""
    game_ids: set[str] = set()
    ids_by_team: dict[str, set[str]] = defaultdict(set)
    for row in included_rows:
        game_id = str(row.get("game_id", "")).strip()
        home_id = str(row.get("home_id", "")).strip()
        away_id = str(row.get("away_id", "")).strip()
        if not game_id or not home_id or not away_id:
            raise ValueError("included evidence row lacks game or team ID")
        if game_id in game_ids:
            raise ValueError(f"duplicate included evidence game ID: {game_id}")
        game_ids.add(game_id)
        ids_by_team[home_id].add(game_id)
        ids_by_team[away_id].add(game_id)
    return {
        team_id: tuple(sorted(ids, key=_team_id_key))
        for team_id, ids in ids_by_team.items()
    }


@dataclass(frozen=True)
class EvidenceInput:
    """Target-free inputs for one team at one frozen evidence cutoff."""

    season: int
    team_id: str
    checkpoint: str
    history_posterior: np.ndarray
    context_prior: np.ndarray
    history_prior: np.ndarray
    team_game_ids: tuple[str, ...]

    @property
    def evidence_count(self) -> int:
        return count_distinct_game_ids(self.team_game_ids)

    @property
    def raw_evidence_features(self) -> np.ndarray:
        history = normalized_pmf(self.history_posterior, name="History posterior")
        prior = normalized_pmf(self.history_prior, name="History prior")
        return np.asarray(
            [np.log1p(self.evidence_count), kl_posterior_from_prior(history, prior)],
            dtype=float,
        )


@dataclass(frozen=True)
class RankTrainingExample:
    """A target-bearing example kept separate from prediction-time inputs."""

    inputs: EvidenceInput
    target: np.ndarray
    team_name: str = ""


@dataclass(frozen=True)
class WeightedObservation:
    """Numerical optimizer row with training-only standardized evidence."""

    inputs: EvidenceInput
    target: np.ndarray
    standardized_features: tuple[float, float]
    weight: float


@dataclass(frozen=True)
class EvidenceStandardizer:
    """Training-only weighted means and scales for the two evidence variables."""

    means: tuple[float, float]
    scales: tuple[float, float]
    scale_floor: float = PREPROCESSOR_SCALE_FLOOR

    @classmethod
    def fit(
        cls,
        inputs: Sequence[EvidenceInput],
        weights: Sequence[float],
        *,
        scale_floor: float = PREPROCESSOR_SCALE_FLOOR,
    ) -> EvidenceStandardizer:
        if not inputs:
            raise ValueError("cannot standardize without training inputs")
        weight_array = np.asarray(weights, dtype=float)
        if (
            weight_array.shape != (len(inputs),)
            or not np.isfinite(weight_array).all()
            or np.any(weight_array <= 0)
        ):
            raise ValueError("standardization weights must be positive and finite")
        raw = np.stack([item.raw_evidence_features for item in inputs])
        total_weight = float(weight_array.sum())
        means = np.sum(raw * weight_array[:, None], axis=0) / total_weight
        variance = (
            np.sum(np.square(raw - means[None, :]) * weight_array[:, None], axis=0)
            / total_weight
        )
        scales = np.maximum(np.sqrt(variance), scale_floor)
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("evidence standardization produced nonfinite values")
        return cls(
            means=(float(means[0]), float(means[1])),
            scales=(float(scales[0]), float(scales[1])),
            scale_floor=float(scale_floor),
        )

    def transform(self, inputs: EvidenceInput) -> tuple[float, float]:
        raw = inputs.raw_evidence_features
        values = (raw - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(values).all():
            raise ValueError("standardized evidence is nonfinite")
        return float(values[0]), float(values[1])

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(EVIDENCE_FEATURE_NAMES),
            "means": list(self.means),
            "scales": list(self.scales),
            "scale_floor": self.scale_floor,
            "standard_deviation_ddof": 0,
            "weighting": "each training team-season has total weight one across checkpoints",
        }


@dataclass(frozen=True)
class CorrectionFit:
    """Fitted full-PMF correction and its training-only transform."""

    include_context_signal: bool
    regularization: float
    coefficients: np.ndarray
    standardizer: EvidenceStandardizer
    optimizer: Mapping[str, object]
    training_seasons: tuple[int, ...]

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return (
            CONTEXT_COEFFICIENT_NAMES
            if self.include_context_signal
            else HISTORY_COEFFICIENT_NAMES
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "include_context_signal": self.include_context_signal,
            "regularization": self.regularization,
            "coefficient_names": list(self.coefficient_names),
            "coefficients": self.coefficients.tolist(),
            "standardizer": self.standardizer.as_dict(),
            "optimizer": dict(self.optimizer),
            "training_seasons": list(self.training_seasons),
        }


def team_season_sample_weights(
    examples: Sequence[RankTrainingExample],
) -> np.ndarray:
    """Give every team-season total weight one over its checkpoint rows."""
    if not examples:
        raise ValueError("cannot weight an empty example set")
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for example in examples:
        counts[(int(example.inputs.season), str(example.inputs.team_id))] += 1
    return np.asarray(
        [
            1.0 / counts[(int(row.inputs.season), str(row.inputs.team_id))]
            for row in examples
        ],
        dtype=float,
    )


def _validated_inputs(
    inputs: EvidenceInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_history = normalized_pmf(inputs.history_posterior, name="History posterior")
    p_context = normalized_pmf(inputs.context_prior, name="Context prior")
    p_history = normalized_pmf(inputs.history_prior, name="History prior")
    if q_history.shape != p_history.shape or p_context.shape != p_history.shape:
        raise ValueError(
            "Context, History prior, and History posterior supports differ"
        )
    history_support = p_history > ZERO_SUPPORT_TOLERANCE
    context_support = p_context > ZERO_SUPPORT_TOLERANCE
    if not np.array_equal(history_support, context_support):
        raise ValueError("Context and History priors have different positive support")
    posterior_support = q_history > ZERO_SUPPORT_TOLERANCE
    if np.any(posterior_support & ~history_support):
        raise ValueError(
            "History posterior has positive mass outside History prior support"
        )
    if not posterior_support.any():
        raise ValueError("History posterior has no positive support")
    d = np.zeros_like(q_history)
    d[history_support] = np.log(p_context[history_support]) - np.log(
        p_history[history_support]
    )
    return q_history, p_context, p_history, d


def _weighted_observations(
    examples: Sequence[RankTrainingExample],
    standardizer: EvidenceStandardizer,
) -> list[WeightedObservation]:
    weights = team_season_sample_weights(examples)
    return [
        WeightedObservation(
            inputs=example.inputs,
            target=normalized_pmf(example.target, name="pre-bowl target"),
            standardized_features=standardizer.transform(example.inputs),
            weight=float(weight),
        )
        for example, weight in zip(examples, weights, strict=True)
    ]


def _prepared_batches(
    observations: Sequence[WeightedObservation], *, include_context_signal: bool
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = (
        defaultdict(list)
    )
    for observation in observations:
        q_history, _p_context, _p_history, signal = _validated_inputs(
            observation.inputs
        )
        target = normalized_pmf(observation.target, name="target")
        if target.shape != q_history.shape:
            raise ValueError("target and History posterior rank supports differ")
        posterior_support = q_history > ZERO_SUPPORT_TOLERANCE
        if np.any((target > ZERO_SUPPORT_TOLERANCE) & ~posterior_support):
            raise ValueError(
                "target has positive mass outside History posterior support"
            )
        x1, x2 = observation.standardized_features
        if (
            not np.isfinite([x1, x2, observation.weight]).all()
            or observation.weight <= 0
        ):
            raise ValueError(
                "optimizer features and weights must be finite and positive"
            )
        centered_log_history = np.zeros_like(q_history)
        centered_log_history[posterior_support] = np.log(q_history[posterior_support])
        centered_log_history[posterior_support] -= float(
            np.mean(centered_log_history[posterior_support])
        )
        z = rank_logit(len(q_history))
        basis_columns = [
            centered_log_history,
            z * posterior_support,
            z * x1 * posterior_support,
            z * x2 * posterior_support,
        ]
        if include_context_signal:
            basis_columns.extend(
                [
                    signal * posterior_support,
                    signal * x1 * posterior_support,
                    signal * x2 * posterior_support,
                ]
            )
        base_log = np.full(q_history.shape, -np.inf, dtype=float)
        base_log[posterior_support] = np.log(q_history[posterior_support])
        grouped[len(q_history)].append(
            (
                base_log,
                np.column_stack(basis_columns),
                target,
                float(observation.weight),
            )
        )
    batches = []
    for members in grouped.values():
        batches.append(
            (
                np.stack([row[0] for row in members]),
                np.stack([row[1] for row in members]),
                np.stack([row[2] for row in members]),
                np.asarray([row[3] for row in members], dtype=float),
            )
        )
    return batches


def objective_and_gradient(
    coefficients: Sequence[float] | np.ndarray,
    observations: Sequence[WeightedObservation],
    *,
    regularization: float,
    include_context_signal: bool,
) -> tuple[float, np.ndarray]:
    """Mean weighted target NLL plus ridge penalty with analytic gradient."""
    theta = np.asarray(coefficients, dtype=float)
    expected_count = 7 if include_context_signal else 4
    if (
        not observations
        or theta.shape != (expected_count,)
        or not np.isfinite(theta).all()
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError(
            "objective requires finite parameters, positive lambda, and rows"
        )
    batches = _prepared_batches(
        observations, include_context_signal=include_context_signal
    )
    loss_total = 0.0
    weight_total = 0.0
    gradient = np.zeros_like(theta)
    for base_logs, bases, targets, weights in batches:
        logits = base_logs + np.einsum("nrp,p->nr", bases, theta)
        normalizers = logsumexp(logits, axis=1, keepdims=True)
        if not np.isfinite(normalizers).all():
            raise ValueError("correction logits have no finite rank support")
        log_predictions = logits - normalizers
        weighted_log_probability = np.zeros_like(targets)
        np.multiply(
            targets,
            log_predictions,
            out=weighted_log_probability,
            where=targets > ZERO_SUPPORT_TOLERANCE,
        )
        per_row_loss = -np.sum(weighted_log_probability, axis=1)
        loss_total += float(np.dot(weights, per_row_loss))
        predictions = np.exp(log_predictions)
        gradient += np.einsum("nrp,nr,n->p", bases, predictions - targets, weights)
        weight_total += float(weights.sum())
    objective = loss_total / weight_total + 0.5 * regularization * float(theta @ theta)
    gradient = gradient / weight_total + regularization * theta
    return float(objective), gradient


def fit_correction(
    examples: Sequence[RankTrainingExample],
    *,
    regularization: float,
    include_context_signal: bool,
    training_seasons: Sequence[int],
) -> CorrectionFit:
    """Fit one calibration or Context-residual arm on supplied rows only."""
    if not examples:
        raise ValueError("cannot fit a correction without training rows")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    weights = team_season_sample_weights(examples)
    standardizer = EvidenceStandardizer.fit(
        [example.inputs for example in examples], weights
    )
    observations = _weighted_observations(examples, standardizer)
    parameter_count = 7 if include_context_signal else 4
    bounds: list[tuple[float | None, float | None]] = [
        (-0.75, 3.0),
        *[(None, None)] * (parameter_count - 1),
    ]
    result = minimize(
        lambda theta: objective_and_gradient(
            theta,
            observations,
            regularization=regularization,
            include_context_signal=include_context_signal,
        ),
        np.zeros(parameter_count, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=dict(OPTIMIZER_OPTIONS),
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or not np.isfinite(result.jac).all()
    ):
        raise RuntimeError(
            f"L-BFGS-B failed for lambda={regularization}: {result.message}"
        )
    return CorrectionFit(
        include_context_signal=include_context_signal,
        regularization=float(regularization),
        coefficients=np.asarray(result.x, dtype=float),
        standardizer=standardizer,
        optimizer={
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective": float(result.fun),
            "gradient_inf_norm": float(np.max(np.abs(result.jac))),
            "method": "L-BFGS-B",
            "options": dict(OPTIMIZER_OPTIONS),
            "bounds": {"a": [-0.75, 3.0], "other_coefficients": "unbounded"},
            "ridge_includes_intercept": True,
        },
        training_seasons=tuple(sorted({int(season) for season in training_seasons})),
    )


def predict_rank_distribution(inputs: EvidenceInput, fit: CorrectionFit) -> np.ndarray:
    """Apply a fitted full-PMF correction without access to a target value."""
    q_history, _p_context, _p_history, signal = _validated_inputs(inputs)
    standardized = np.asarray(fit.standardizer.transform(inputs), dtype=float)
    x1, x2 = float(standardized[0]), float(standardized[1])
    support = q_history > ZERO_SUPPORT_TOLERANCE
    centered_log_history = np.zeros_like(q_history)
    centered_log_history[support] = np.log(q_history[support])
    centered_log_history[support] -= float(np.mean(centered_log_history[support]))
    z = rank_logit(len(q_history))
    coefficients = fit.coefficients
    logits = np.full(q_history.shape, -np.inf, dtype=float)
    logits[support] = np.log(q_history[support])
    logits[support] += coefficients[0] * centered_log_history[support]
    logits[support] += z[support] * (
        coefficients[1] + coefficients[2] * x1 + coefficients[3] * x2
    )
    if fit.include_context_signal:
        weight = coefficients[4] + coefficients[5] * x1 + coefficients[6] * x2
        logits[support] += weight * signal[support]
    normalizer = float(logsumexp(logits))
    if not np.isfinite(normalizer):
        raise ValueError("correction output has no finite rank support")
    prediction = np.zeros_like(q_history)
    prediction[support] = np.exp(logits[support] - normalizer)
    if not np.isfinite(prediction).all() or not np.isclose(
        float(prediction.sum()), 1.0, atol=1e-10, rtol=0.0
    ):
        raise ValueError("correction output is not a normalized full-rank PMF")
    return prediction


def select_regularization(
    examples_by_season: Mapping[int, Sequence[RankTrainingExample]],
    training_seasons: Sequence[int],
    *,
    include_context_signal: bool,
) -> tuple[float, dict[str, object]]:
    """Select lambda by earlier-season forward-chained target-NLL folds."""
    ordered = tuple(sorted({int(season) for season in training_seasons}))
    if not ordered:
        raise ValueError("outer fit has no earlier training seasons")
    if any(season not in examples_by_season for season in ordered):
        raise KeyError("training season examples are missing")
    if len(ordered) == 1:
        return SINGLE_SEASON_LAMBDA, {
            "method": "fixed_single_training_season",
            "training_seasons": list(ordered),
            "selected_lambda": SINGLE_SEASON_LAMBDA,
            "candidates": [],
        }
    folds = forward_chained_folds(ordered)
    candidate_results: list[dict[str, object]] = []
    for regularization in LAMBDA_GRID:
        loss_total = 0.0
        weight_total = 0.0
        fold_results = []
        for inner_training_seasons, validation_season in folds:
            inner_training = [
                example
                for season in inner_training_seasons
                for example in examples_by_season[season]
            ]
            validation = list(examples_by_season[validation_season])
            fitted = fit_correction(
                inner_training,
                regularization=regularization,
                include_context_signal=include_context_signal,
                training_seasons=inner_training_seasons,
            )
            validation_weights = team_season_sample_weights(validation)
            losses = [
                target_cross_entropy(
                    predict_rank_distribution(example.inputs, fitted), example.target
                )
                for example in validation
            ]
            fold_loss = float(np.dot(validation_weights, np.asarray(losses)))
            fold_weight = float(validation_weights.sum())
            loss_total += fold_loss
            weight_total += fold_weight
            fold_results.append(
                {
                    "training_seasons": list(inner_training_seasons),
                    "validation_season": validation_season,
                    "team_seasons": len(
                        {
                            (example.inputs.season, example.inputs.team_id)
                            for example in validation
                        }
                    ),
                    "checkpoint_rows": len(validation),
                    "mean_team_season_nll": fold_loss / fold_weight,
                    "optimizer": dict(fitted.optimizer),
                }
            )
        if weight_total <= 0:
            raise ValueError("forward-chained validation produced no weighted rows")
        candidate_results.append(
            {
                "lambda": float(regularization),
                "mean_team_season_nll": loss_total / weight_total,
                "team_seasons": round(weight_total),
                "folds": fold_results,
            }
        )
    best = min(float(row["mean_team_season_nll"]) for row in candidate_results)
    tied = [
        row
        for row in candidate_results
        if abs(float(row["mean_team_season_nll"]) - best) <= TIE_TOLERANCE
    ]
    selected = max(tied, key=lambda row: float(row["lambda"]))
    return float(selected["lambda"]), {
        "method": "forward_chained_inner_validation",
        "training_seasons": list(ordered),
        "selected_lambda": float(selected["lambda"]),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": "stronger regularization",
        "weighting": "equal total weight per validation team-season across checkpoints",
        "candidates": candidate_results,
    }


def fit_outer_fold(
    examples_by_season: Mapping[int, Sequence[RankTrainingExample]],
    training_seasons: Sequence[int],
    *,
    include_context_signal: bool,
) -> tuple[CorrectionFit, dict[str, object]]:
    """Select lambda on prior seasons, then refit on all outer training rows."""
    training = tuple(sorted({int(season) for season in training_seasons}))
    selected, selection = select_regularization(
        examples_by_season,
        training,
        include_context_signal=include_context_signal,
    )
    examples = [
        example for season in training for example in examples_by_season[season]
    ]
    fit = fit_correction(
        examples,
        regularization=selected,
        include_context_signal=include_context_signal,
        training_seasons=training,
    )
    return fit, selection


def evidence_weight(inputs: EvidenceInput, fit: CorrectionFit) -> float:
    """Return the fitted affine Context weight for a candidate row."""
    if not fit.include_context_signal:
        return 0.0
    x1, x2 = fit.standardizer.transform(inputs)
    return float(
        fit.coefficients[4] + fit.coefficients[5] * x1 + fit.coefficients[6] * x2
    )


def input_signal_diagnostics(inputs: EvidenceInput) -> dict[str, object]:
    """Return support and evidence diagnostics without a target distribution."""
    q_history, p_context, p_history, signal = _validated_inputs(inputs)
    return {
        "evidence_count": inputs.evidence_count,
        "history_kl_from_prior": kl_posterior_from_prior(q_history, p_history),
        "context_history_prior_l1": float(np.sum(np.abs(p_context - p_history))),
        "context_signal_l1": float(np.sum(np.abs(signal))),
        "rank_count": len(q_history),
        "positive_history_posterior_support": int(np.count_nonzero(q_history)),
    }


def grouped_examples(
    examples: Sequence[RankTrainingExample],
) -> dict[int, list[RankTrainingExample]]:
    """Group examples by season in deterministic team/checkpoint order."""
    result: dict[int, list[RankTrainingExample]] = defaultdict(list)
    for example in examples:
        result[int(example.inputs.season)].append(example)
    return {
        season: sorted(
            rows,
            key=lambda row: (
                row.inputs.checkpoint,
                _team_id_key(row.inputs.team_id),
            ),
        )
        for season, rows in sorted(result.items())
    }


def input_rows_hash(examples: Sequence[RankTrainingExample]) -> str:
    """Stable digest for the exact full-PMF rows entering a fit."""
    import hashlib
    import json

    digest = hashlib.sha256()
    rows = sorted(
        examples,
        key=lambda row: (
            int(row.inputs.season),
            row.inputs.checkpoint,
            _team_id_key(str(row.inputs.team_id)),
        ),
    )
    for example in rows:
        payload = {
            "season": int(example.inputs.season),
            "team_id": str(example.inputs.team_id),
            "checkpoint": example.inputs.checkpoint,
            "history_posterior": np.asarray(example.inputs.history_posterior).tolist(),
            "context_prior": np.asarray(example.inputs.context_prior).tolist(),
            "history_prior": np.asarray(example.inputs.history_prior).tolist(),
            "team_game_ids": list(example.inputs.team_game_ids),
            "target": np.asarray(example.target).tolist(),
        }
        digest.update(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()
