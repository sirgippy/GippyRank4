"""Reproducible benchmark for Context information beyond History and games.

This module owns only the residual-probe and probability-support contracts.
Production prior construction and game inference stay in the existing
context_db_repair and posterior modules.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp

LAMBDA_GRID = (0.01, 0.1, 1.0)
SINGLE_SEASON_LAMBDA = 0.1
PREPROCESSOR_SCALE_FLOOR = 1e-6
OPTIMIZER_OPTIONS = {
    "maxiter": 2000,
    "ftol": 1e-13,
    "gtol": 1e-8,
}
ZERO_SUPPORT_TOLERANCE = 0.0
IDENTITY_TOLERANCE = 1e-10
TIE_TOLERANCE = 1e-12


def normalized_pmf(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Validate and production-normalize one serialized discrete PMF."""
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or not len(result)
        or not np.isfinite(result).all()
        or np.any(result < 0)
    ):
        raise ValueError(f"{name} must be a finite, nonnegative nonempty PMF")
    total = float(result.sum())
    if total <= 0 or not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"{name} must sum to one")
    return result / total


def apply_log_message(
    prior: Sequence[float] | np.ndarray,
    log_message: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply an exact incoming message, preserving impossible rank positions."""
    q = normalized_pmf(prior, name="prior")
    message = np.asarray(log_message, dtype=float)
    if (
        message.shape != q.shape
        or np.isnan(message).any()
        or np.isposinf(message).any()
    ):
        raise ValueError(
            "log message must match the prior support and contain no NaN/+inf"
        )
    log_product = np.full(q.shape, -np.inf, dtype=float)
    support = q > ZERO_SUPPORT_TOLERANCE
    finite_message = np.isfinite(message)
    usable = support & finite_message
    if not usable.any():
        raise ValueError("message and prior have no shared positive support")
    log_product[usable] = np.log(q[usable]) + message[usable]
    log_normalizer = float(logsumexp(log_product))
    if not np.isfinite(log_normalizer):
        raise ValueError("message product cannot be normalized")
    result = np.zeros_like(q)
    result[usable] = np.exp(log_product[usable] - log_normalizer)
    return result


def extract_log_message(
    prior: Sequence[float] | np.ndarray,
    posterior: Sequence[float] | np.ndarray,
    *,
    tolerance: float = IDENTITY_TOLERANCE,
) -> tuple[np.ndarray, float]:
    """Recover posterior/prior with explicit zero-support checks.

    No epsilon or smoothing is used. A posterior may remove support, but it
    may not introduce mass at a rank the prior ruled out.
    """
    q = normalized_pmf(prior, name="prior")
    p = normalized_pmf(posterior, name="posterior")
    if p.shape != q.shape:
        raise ValueError("prior and posterior rank supports differ")
    prior_support = q > ZERO_SUPPORT_TOLERANCE
    escaped = (~prior_support) & (p > ZERO_SUPPORT_TOLERANCE)
    if escaped.any():
        raise ValueError("posterior has positive mass outside prior support")
    message = np.full(q.shape, -np.inf, dtype=float)
    posterior_support = p > ZERO_SUPPORT_TOLERANCE
    retained = prior_support & posterior_support
    if not retained.any():
        raise ValueError("posterior and prior have no shared positive support")
    message[retained] = np.log(p[retained]) - np.log(q[retained])
    rebuilt = apply_log_message(q, message)
    error = float(np.max(np.abs(rebuilt - p)))
    if error > tolerance:
        raise ValueError(
            f"prior times its message does not reconstruct posterior: {error}"
        )
    return message, error


def cross_message_arms(
    context_prior: Sequence[float] | np.ndarray,
    history_prior: Sequence[float] | np.ndarray,
    context_posterior: Sequence[float] | np.ndarray,
    history_posterior: Sequence[float] | np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Cross each own-team prior with each model's exact frozen message.

    The Context and History priors must have identical support. A message is
    not identifiable where its own prior is zero, so crossing it onto a prior
    that assigns mass there would invent a hard exclusion.
    """
    context_q = normalized_pmf(context_prior, name="Context prior")
    history_q = normalized_pmf(history_prior, name="History prior")
    if context_q.shape != history_q.shape:
        raise ValueError("Context and History priors have different rank supports")
    if not np.array_equal(
        context_q > ZERO_SUPPORT_TOLERANCE,
        history_q > ZERO_SUPPORT_TOLERANCE,
    ):
        raise ValueError("Context and History priors have different positive support")
    context_message, context_error = extract_log_message(
        context_prior, context_posterior
    )
    history_message, history_error = extract_log_message(
        history_prior, history_posterior
    )
    arms = {
        "HH": apply_log_message(history_prior, history_message),
        "CH": apply_log_message(context_prior, history_message),
        "HC": apply_log_message(history_prior, context_message),
        "CC": apply_log_message(context_prior, context_message),
    }
    for arm, posterior in (("HH", history_posterior), ("CC", context_posterior)):
        error = float(
            np.max(
                np.abs(arms[arm] - normalized_pmf(posterior, name=f"{arm} posterior"))
            )
        )
        if error > IDENTITY_TOLERANCE:
            raise ValueError(f"{arm}: own prior/message identity failed: {error}")
    return arms, {
        "context_identity_error": context_error,
        "history_identity_error": history_error,
    }


def validate_evidence_parity(
    context_record: Mapping[str, object],
    history_record: Mapping[str, object],
    observed: Mapping[str, object],
    fields: Sequence[str],
) -> None:
    """Require both retained arms and the replay to share exact evidence."""
    for field in fields:
        if context_record.get(field) != history_record.get(field):
            raise ValueError(f"retained Context/History evidence differs for {field}")
        if observed.get(field) != context_record.get(field):
            raise ValueError(f"replayed evidence differs for {field}")


def write_deterministic_gzip_jsonl(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write canonical gzip JSONL with a fixed header timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed,
    ):
        for row in rows:
            encoded = json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            compressed.write(encoded + b"\n")


def kl_posterior_from_prior(
    posterior: Sequence[float] | np.ndarray,
    prior: Sequence[float] | np.ndarray,
) -> float:
    """Compute KL(posterior || prior) without smoothing zero support."""
    p = normalized_pmf(posterior, name="posterior")
    q = normalized_pmf(prior, name="prior")
    if p.shape != q.shape:
        raise ValueError("prior and posterior rank supports differ")
    positive_p = p > ZERO_SUPPORT_TOLERANCE
    if np.any(positive_p & (q <= ZERO_SUPPORT_TOLERANCE)):
        raise ValueError("posterior has positive mass outside prior support")
    return float(
        np.sum(p[positive_p] * (np.log(p[positive_p]) - np.log(q[positive_p])))
    )


def rank_logit(population: int) -> np.ndarray:
    """Return z(r)=logit((r-0.5)/N) for ranks 1 through N."""
    if population < 1:
        raise ValueError("rank population must be positive")
    probability = (np.arange(1, population + 1, dtype=float) - 0.5) / population
    return logit(probability)


def centered_context_signal(
    context_prior: Sequence[float] | np.ndarray,
    history_prior: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return centered log Context-prior / History-prior on common support."""
    p_context = normalized_pmf(context_prior, name="Context prior")
    p_history = normalized_pmf(history_prior, name="History prior")
    if p_context.shape != p_history.shape:
        raise ValueError("Context and History priors have different rank supports")
    context_support = p_context > ZERO_SUPPORT_TOLERANCE
    history_support = p_history > ZERO_SUPPORT_TOLERANCE
    if not np.array_equal(context_support, history_support):
        raise ValueError("existing Context signal requires identical prior support")
    signal = np.zeros_like(p_context)
    signal[context_support] = np.log(p_context[context_support]) - np.log(
        p_history[history_support]
    )
    signal[context_support] -= float(np.mean(signal[context_support]))
    return signal


@dataclass(frozen=True)
class ProbeExample:
    """One team-season at one checkpoint in the target-scoring boundary."""

    season: int
    team_id: str
    team_name: str
    history_posterior: np.ndarray
    context_prior: np.ndarray
    history_prior: np.ndarray
    target: np.ndarray
    features: Mapping[str, float | None]


@dataclass(frozen=True)
class ProbeInputs:
    """Target-free inputs for one prediction-time residual probe."""

    history_posterior: np.ndarray
    covariates: np.ndarray
    context_signal: np.ndarray | None = None


@dataclass(frozen=True)
class ProbeObservation:
    """Numerical, target-bearing observation passed only to probe fitting."""

    inputs: ProbeInputs
    target: np.ndarray


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Training-only median imputation, standardization, and missing flags."""

    feature_names: tuple[str, ...]
    medians: Mapping[str, float]
    means: Mapping[str, float]
    scales: Mapping[str, float]

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, float | None]],
        feature_names: Sequence[str],
        *,
        scale_floor: float = PREPROCESSOR_SCALE_FLOOR,
    ) -> FeaturePreprocessor:
        if not rows:
            raise ValueError("cannot fit preprocessing without training rows")
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in feature_names:
            observed = []
            for row in rows:
                value = row.get(name)
                if value is None:
                    continue
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(f"nonfinite training feature {name}")
                observed.append(number)
            median = float(np.median(observed)) if observed else 0.0
            imputed = np.asarray(
                [median if row.get(name) is None else float(row[name]) for row in rows],
                dtype=float,
            )
            medians[name] = median
            means[name] = float(np.mean(imputed))
            scales[name] = max(float(np.std(imputed, ddof=0)), scale_floor)
        return cls(tuple(feature_names), medians, means, scales)

    @property
    def design_names(self) -> tuple[str, ...]:
        numeric = self.feature_names
        missing = tuple(f"{name}__missing" for name in self.feature_names)
        return (*numeric, *missing)

    def transform(self, rows: Sequence[Mapping[str, float | None]]) -> np.ndarray:
        result = []
        for row in rows:
            values: list[float] = []
            missing: list[float] = []
            for name in self.feature_names:
                raw = row.get(name)
                missing.append(float(raw is None))
                number = self.medians[name] if raw is None else float(raw)
                if not np.isfinite(number):
                    raise ValueError(f"nonfinite feature {name}")
                values.append((number - self.means[name]) / self.scales[name])
            result.append([*values, *missing])
        if not result:
            return np.empty((0, len(self.design_names)), dtype=float)
        return np.asarray(result, dtype=float)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "medians": dict(self.medians),
            "means": dict(self.means),
            "scales": dict(self.scales),
            "design_names": list(self.design_names),
            "scale_floor": PREPROCESSOR_SCALE_FLOOR,
            "standard_deviation_ddof": 0,
        }


@dataclass(frozen=True)
class ProbeFit:
    """Fitted residual-probe coefficients and training-only preprocessing."""

    include_signal: bool
    regularization: float
    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    preprocessor: FeaturePreprocessor
    optimizer: Mapping[str, object]
    training_seasons: tuple[int, ...]


def _probe_basis(
    inputs: ProbeInputs, *, include_signal: bool
) -> tuple[np.ndarray, np.ndarray]:
    q_history = normalized_pmf(inputs.history_posterior, name="History posterior")
    covariates = np.asarray(inputs.covariates, dtype=float)
    if covariates.ndim != 1 or not np.isfinite(covariates).all():
        raise ValueError("probe covariates must be a finite vector")
    support = q_history > ZERO_SUPPORT_TOLERANCE
    centered_log_history = np.zeros_like(q_history)
    log_history = np.log(q_history[support])
    centered_log_history[support] = log_history - float(np.mean(log_history))
    z = rank_logit(len(q_history))
    columns = [centered_log_history, z]
    columns.extend(z * covariates[index] for index in range(len(covariates)))
    if include_signal:
        if inputs.context_signal is None:
            raise ValueError("existing-signal probe requires a Context signal")
        signal = np.asarray(inputs.context_signal, dtype=float)
        if signal.shape != q_history.shape or not np.isfinite(signal).all():
            raise ValueError("Context signal must be finite and match rank support")
        signal_support = signal[support]
        centered_signal = np.zeros_like(signal)
        centered_signal[support] = signal_support - float(np.mean(signal_support))
        columns.append(centered_signal)
    base_log = np.full(q_history.shape, -np.inf, dtype=float)
    base_log[support] = np.log(q_history[support])
    return base_log, np.column_stack(columns)


def probe_objective_and_gradient(
    coefficients: Sequence[float] | np.ndarray,
    observations: Sequence[ProbeObservation],
    *,
    regularization: float,
    include_signal: bool,
) -> tuple[float, np.ndarray]:
    """Mean target-PMF cross entropy plus ridge penalty and analytic gradient."""
    theta = np.asarray(coefficients, dtype=float)
    if not observations or not np.isfinite(theta).all() or regularization < 0:
        raise ValueError("probe objective requires finite parameters and training rows")
    expected_size = 2 + len(observations[0].inputs.covariates) + int(include_signal)
    if theta.shape != (expected_size,):
        raise ValueError("probe coefficient count does not match design")
    prepared = _prepare_probe_objective(observations, include_signal=include_signal)
    return _probe_objective_prepared(theta, prepared, regularization=regularization)


def _prepare_probe_objective(
    observations: Sequence[ProbeObservation], *, include_signal: bool
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for observation in observations:
        base_log, basis = _probe_basis(
            observation.inputs, include_signal=include_signal
        )
        target = normalized_pmf(observation.target, name="target")
        if target.shape != base_log.shape:
            raise ValueError("target and History posterior rank supports differ")
        positive_target = target > ZERO_SUPPORT_TOLERANCE
        if np.any(positive_target & ~np.isfinite(base_log)):
            raise ValueError(
                "target has positive mass outside History posterior support"
            )
        grouped.setdefault(len(target), []).append((base_log, basis, target))
    prepared = []
    for members in grouped.values():
        prepared.append(
            (
                np.stack([member[0] for member in members]),
                np.stack([member[1] for member in members]),
                np.stack([member[2] for member in members]),
            )
        )
    return prepared


def _probe_objective_prepared(
    theta: np.ndarray,
    prepared: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss_total = 0.0
    gradient = np.zeros_like(theta)
    count = 0
    for base_logs, bases, targets in prepared:
        if bases.shape[2] != len(theta):
            raise ValueError("probe observations have inconsistent covariate widths")
        logits = base_logs + np.einsum("nrf,f->nr", bases, theta)
        log_normalizers = logsumexp(logits, axis=1, keepdims=True)
        if not np.isfinite(log_normalizers).all():
            raise ValueError("probe logits have no finite support")
        log_prediction = logits - log_normalizers
        positive_target = targets > ZERO_SUPPORT_TOLERANCE
        weighted_log_probability = np.zeros_like(targets)
        np.multiply(
            targets,
            log_prediction,
            out=weighted_log_probability,
            where=positive_target,
        )
        loss_total -= float(np.sum(weighted_log_probability))
        prediction = np.exp(log_prediction)
        gradient += np.einsum("nrf,nr->f", bases, prediction - targets)
        count += len(targets)
    objective = float(loss_total / count + 0.5 * regularization * np.dot(theta, theta))
    gradient = gradient / count + regularization * theta
    return objective, gradient


def _make_observations(
    examples: Sequence[ProbeExample],
    preprocessor: FeaturePreprocessor,
    *,
    include_signal: bool,
) -> list[ProbeObservation]:
    covariates = preprocessor.transform([example.features for example in examples])
    observations = []
    for index, example in enumerate(examples):
        signal = (
            centered_context_signal(example.context_prior, example.history_prior)
            if include_signal
            else None
        )
        inputs = ProbeInputs(
            history_posterior=np.asarray(example.history_posterior, dtype=float),
            covariates=covariates[index],
            context_signal=signal if include_signal else None,
        )
        observations.append(
            ProbeObservation(
                inputs=inputs, target=np.asarray(example.target, dtype=float)
            )
        )
    return observations


def fit_probe(
    examples: Sequence[ProbeExample],
    feature_names: Sequence[str],
    *,
    regularization: float,
    include_signal: bool,
    training_seasons: Sequence[int],
) -> ProbeFit:
    """Fit one probe using only the supplied training examples."""
    if not examples:
        raise ValueError("cannot fit a probe without training examples")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    preprocessor = FeaturePreprocessor.fit(
        [example.features for example in examples], feature_names
    )
    observations = _make_observations(
        examples, preprocessor, include_signal=include_signal
    )
    prepared = _prepare_probe_objective(observations, include_signal=include_signal)
    parameter_count = 2 + len(preprocessor.design_names) + int(include_signal)
    bounds: list[tuple[float | None, float | None]] = [
        (-0.75, 3.0),
        *[(None, None)] * (parameter_count - 1),
    ]
    result = minimize(
        lambda theta: _probe_objective_prepared(
            theta, prepared, regularization=regularization
        ),
        np.zeros(parameter_count, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=dict(OPTIMIZER_OPTIONS),
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(
            f"L-BFGS-B failed for lambda={regularization}: {result.message}"
        )
    names = (
        "a",
        "b0",
        *(f"b::{name}" for name in preprocessor.design_names),
        *(("w",) if include_signal else ()),
    )
    return ProbeFit(
        include_signal=include_signal,
        regularization=float(regularization),
        coefficients=np.asarray(result.x, dtype=float),
        coefficient_names=tuple(names),
        preprocessor=preprocessor,
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
        },
        training_seasons=tuple(sorted({int(season) for season in training_seasons})),
    )


def predict_probe(
    example: ProbeExample,
    fit: ProbeFit,
) -> np.ndarray:
    """Apply a fitted probe using only prediction-time inputs."""
    covariates = fit.preprocessor.transform([example.features])[0]
    signal = (
        centered_context_signal(example.context_prior, example.history_prior)
        if fit.include_signal
        else None
    )
    inputs = ProbeInputs(
        history_posterior=np.asarray(example.history_posterior, dtype=float),
        covariates=covariates,
        context_signal=signal if fit.include_signal else None,
    )
    base_log, basis = _probe_basis(inputs, include_signal=fit.include_signal)
    logits = base_log + basis @ fit.coefficients
    log_normalizer = float(logsumexp(logits))
    if not np.isfinite(log_normalizer):
        raise ValueError("held-out probe prediction has no finite support")
    prediction = np.zeros_like(logits)
    finite = np.isfinite(logits)
    prediction[finite] = np.exp(logits[finite] - log_normalizer)
    return prediction


def target_cross_entropy(
    prediction: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> float:
    """Score a target PMF without adding probability to unsupported ranks."""
    q = normalized_pmf(prediction, name="prediction")
    p = normalized_pmf(target, name="target")
    if q.shape != p.shape:
        raise ValueError("prediction and target rank supports differ")
    positive = p > ZERO_SUPPORT_TOLERANCE
    if np.any(positive & (q <= ZERO_SUPPORT_TOLERANCE)):
        raise ValueError("prediction has zero support where target is positive")
    return float(-np.sum(p[positive] * np.log(q[positive])))


def forward_chained_folds(
    seasons: Sequence[int],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return earlier-season training folds for forward-chained validation."""
    ordered = tuple(sorted({int(season) for season in seasons}))
    if len(ordered) < 2:
        return ()
    return tuple((ordered[:index], ordered[index]) for index in range(1, len(ordered)))


def select_regularization(
    examples_by_season: Mapping[int, Sequence[ProbeExample]],
    training_seasons: Sequence[int],
    feature_names: Sequence[str],
    *,
    include_signal: bool,
) -> tuple[float, dict[str, object]]:
    """Choose lambda using only forward-chained folds inside the training era."""
    ordered = tuple(sorted({int(season) for season in training_seasons}))
    if not ordered:
        raise ValueError("outer fold has no training seasons")
    if len(ordered) == 1:
        return SINGLE_SEASON_LAMBDA, {
            "method": "fixed_single_training_season",
            "training_seasons": list(ordered),
            "selected_lambda": SINGLE_SEASON_LAMBDA,
            "candidates": [],
        }
    folds = forward_chained_folds(ordered)
    candidate_results = []
    for regularization in LAMBDA_GRID:
        total_loss = 0.0
        total_examples = 0
        fold_results = []
        for inner_training_seasons, validation_season in folds:
            inner_training = [
                example
                for season in inner_training_seasons
                for example in examples_by_season[season]
            ]
            validation = list(examples_by_season[validation_season])
            fitted = fit_probe(
                inner_training,
                feature_names,
                regularization=regularization,
                include_signal=include_signal,
                training_seasons=inner_training_seasons,
            )
            losses = [
                target_cross_entropy(predict_probe(example, fitted), example.target)
                for example in validation
            ]
            total_loss += float(sum(losses))
            total_examples += len(losses)
            fold_results.append(
                {
                    "training_seasons": list(inner_training_seasons),
                    "validation_season": validation_season,
                    "team_seasons": len(losses),
                    "mean_nll": float(np.mean(losses)),
                    "optimizer": dict(fitted.optimizer),
                }
            )
        if total_examples == 0:
            raise ValueError("forward-chained validation produced no held-out rows")
        candidate_results.append(
            {
                "lambda": regularization,
                "mean_team_season_nll": total_loss / total_examples,
                "team_seasons": total_examples,
                "folds": fold_results,
            }
        )
    best_score = min(float(row["mean_team_season_nll"]) for row in candidate_results)
    tied = [
        row
        for row in candidate_results
        if abs(float(row["mean_team_season_nll"]) - best_score) <= TIE_TOLERANCE
    ]
    selected = max(tied, key=lambda row: float(row["lambda"]))
    return float(selected["lambda"]), {
        "method": "forward_chained_inner_validation",
        "training_seasons": list(ordered),
        "selected_lambda": float(selected["lambda"]),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": "stronger regularization",
        "candidates": candidate_results,
    }
