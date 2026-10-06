"""Capture the raw History 1.1 optimizer result from the issue 178 test fixture.

This script is diagnostic only. Its first test invocation calls the production
fit without changing its options or failure behavior. If that fit returns an
ABNORMAL result, a separate second invocation measures a manual restart from
the captured iterate with a larger line-search limit; the library code is
never patched with a retry policy.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import gippyrank.history_annual_v1_1 as history
from gippyrank import preseason

ISSUE = 178
TEST_MODULE = "test_context_prior_v1_4_candidate"
TEST_FIXTURE = "_write_canonical_history_2027_inputs"
THREAD_ENVIRONMENT_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NPY_NUM_THREADS",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dense = getattr(value, "todense", None)
    if callable(to_dense):
        try:
            return _json_value(np.asarray(to_dense()))
        except Exception:  # noqa: BLE001 - keep an opaque SciPy field as its repr.
            return repr(value)
    return repr(value)


def _capture_text(function: Callable[[], Any]) -> str:
    output = io.StringIO()
    errors = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        returned = function()
    text = output.getvalue().strip()
    text = text or (repr(returned) if returned is not None else "")
    warning = errors.getvalue().strip()
    return f"{text}\nSTDERR:\n{warning}" if warning else text


def _runtime_details() -> dict[str, Any]:
    try:
        lscpu = subprocess.run(
            ["lscpu"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        lscpu_output = lscpu.stdout.strip()
        lscpu_error = lscpu.stderr.strip() or None
    except (OSError, subprocess.TimeoutExpired) as error:
        lscpu_output = ""
        lscpu_error = str(error)

    runtime_report = getattr(np, "show_runtime", None)
    return {
        "mode": os.environ.get("ISSUE178_MODE", "unspecified"),
        "python": platform.python_version(),
        "python_full_version": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "github_actions": {
            name: os.environ.get(name)
            for name in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "RUNNER_NAME",
                "RUNNER_OS",
                "ImageOS",
                "ImageVersion",
            )
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES
        },
        "numpy_show_config": _capture_text(np.show_config),
        "numpy_show_runtime": (
            _capture_text(runtime_report) if callable(runtime_report) else None
        ),
        "scipy_show_config": _capture_text(scipy.show_config),
        "lscpu": lscpu_output,
        "lscpu_error": lscpu_error,
    }


def _load_test_module(repo_root: Path) -> Any:
    tests_dir = str(repo_root / "tests")
    sys.path.insert(0, tests_dir)
    try:
        test_module = importlib.import_module(TEST_MODULE)
    finally:
        sys.path.remove(tests_dir)
    fixture_writer = getattr(test_module, TEST_FIXTURE)
    if not callable(fixture_writer):
        raise TypeError(f"{TEST_FIXTURE} must be callable")
    return test_module


def _array_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return _sha256_bytes(contiguous.tobytes())


def _gradient_check(
    objective: Callable[[np.ndarray], float],
    analytic: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, Any]:
    numeric = np.empty_like(parameters, dtype=float)
    for index, value in enumerate(parameters):
        step = 1e-5 * (1.0 + abs(value))
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += step
        backward[index] -= step
        numeric[index] = (objective(forward) - objective(backward)) / (2.0 * step)
    difference = np.abs(analytic - numeric)
    return {
        "method": "central difference; h = 1e-5 * (1 + abs(parameter))",
        "analytic_gradient": analytic.tolist(),
        "numeric_gradient": numeric.tolist(),
        "absolute_error_by_parameter": difference.tolist(),
        "max_absolute_analytic_gradient": float(np.max(np.abs(analytic))),
        "max_absolute_numeric_gradient": float(np.max(np.abs(numeric))),
        "max_absolute_error": float(np.max(difference)),
        "l2_error": float(np.linalg.norm(difference)),
        "all_finite": bool(
            np.isfinite(analytic).all()
            and np.isfinite(numeric).all()
            and np.isfinite(difference).all()
        ),
    }


def _bound_diagnostics(
    parameters: np.ndarray, gradient: np.ndarray, bounds: Sequence[Any] | None
) -> dict[str, Any]:
    if bounds is None:
        return {"available": False}

    lower = np.asarray(
        [-np.inf if bound[0] is None else bound[0] for bound in bounds], dtype=float
    )
    upper = np.asarray(
        [np.inf if bound[1] is None else bound[1] for bound in bounds], dtype=float
    )
    tolerance = 1e-8 * np.maximum(1.0, np.abs(parameters))
    at_lower = np.isfinite(lower) & (parameters - lower <= tolerance)
    at_upper = np.isfinite(upper) & (upper - parameters <= tolerance)
    projected = gradient.copy()
    projected[(at_lower & (gradient > 0)) | (at_upper & (gradient < 0))] = 0.0
    projected[lower == upper] = 0.0
    close_indices = np.flatnonzero(at_lower | at_upper).tolist()
    return {
        "available": True,
        "tolerance_rule": "1e-8 * max(1, abs(parameter))",
        "at_lower_bound_indices": np.flatnonzero(at_lower).tolist(),
        "at_upper_bound_indices": np.flatnonzero(at_upper).tolist(),
        "near_any_bound_indices": close_indices,
        "max_absolute_projected_gradient": float(np.max(np.abs(projected))),
        "projected_gradient_l2_norm": float(np.linalg.norm(projected)),
    }


def _result_fields(result: Any) -> dict[str, Any]:
    fields = dict(result.items()) if hasattr(result, "items") else vars(result)
    normalized: dict[str, Any] = {}
    for name, value in fields.items():
        if name == "hess_inv" and callable(getattr(value, "todense", None)):
            try:
                value = value.todense()
            except Exception:  # noqa: BLE001 - keep every available result field.
                value = repr(value)
        normalized[str(name)] = _json_value(value)
    return normalized


def _run_reproduction(
    root: Path,
    test_module: Any,
    *,
    restart_call_index: int | None = None,
    restart_parameters: np.ndarray | None = None,
) -> dict[str, Any]:
    original_minimize = preseason.minimize
    original_fit_descriptor = preseason.DirectRankModel.__dict__["fit"]
    original_fit = preseason.DirectRankModel.fit
    original_reproduce = history.reproduce_canonical_history_annual
    events: list[dict[str, Any]] = []
    fit_contexts: list[dict[str, Any]] = []
    captured_history_output: dict[str, Any] = {}

    def instrumented_fit(cls: Any, *args: Any, **kwargs: Any) -> Any:
        rows = args[0] if args else kwargs["rows"]
        feature_names = args[1] if len(args) > 1 else kwargs["feature_names"]
        fit_contexts.append(
            {
                "rows": rows,
                "feature_names": list(feature_names),
                "lag_count": kwargs.get("lag_count", 1),
                "preprocessor_scale_floor": kwargs.get(
                    "preprocessor_scale_floor", preseason.PREPROCESSOR_SCALE_FLOOR
                ),
                "preprocessor_std_ddof": kwargs.get(
                    "preprocessor_std_ddof", preseason.PREPROCESSOR_STD_DDOF
                ),
            }
        )
        return original_fit(*args, **kwargs)

    def capture_history_output(*args: Any, **kwargs: Any) -> Any:
        result = original_reproduce(*args, **kwargs)
        prediction_bytes, fitted, source = result
        captured_history_output.update(
            {
                "prediction_bytes": prediction_bytes,
                "fitted": fitted,
                "source_identity": source.source_identity_sha256,
            }
        )
        return result

    def instrumented_minimize(*args: Any, **kwargs: Any) -> Any:
        if len(args) < 2:
            raise TypeError("the History reproducer must pass fun and x0 positionally")
        event_index = len(events)
        objective = args[0]
        initial = np.asarray(args[1], dtype=float)
        analytic = kwargs.get("jac")
        bounds = kwargs.get("bounds")
        method = kwargs.get("method")
        options = dict(kwargs.get("options") or {})
        fit_context = fit_contexts[event_index]
        is_restart = restart_call_index == event_index
        start = (
            np.asarray(restart_parameters, dtype=float)
            if is_restart and restart_parameters is not None
            else initial
        )
        observed: dict[str, Any] = {
            "objective_evaluations": 0,
            "finite_objective_evaluations": 0,
            "gradient_evaluations": 0,
            "finite_gradient_evaluations": 0,
            "all_evaluated_iterates_finite": True,
            "non_finite_objective_samples": [],
            "non_finite_gradient_samples": [],
            "callback_iterations": [],
        }

        def checked_objective(parameters: np.ndarray) -> float:
            values = np.asarray(parameters, dtype=float)
            observed["all_evaluated_iterates_finite"] &= _array_finite(values)
            value = float(objective(parameters))
            observed["objective_evaluations"] += 1
            finite = bool(np.isfinite(value))
            observed["finite_objective_evaluations"] += int(finite)
            if not finite and len(observed["non_finite_objective_samples"]) < 3:
                observed["non_finite_objective_samples"].append(
                    {"x": values.tolist(), "value": value}
                )
            return value

        call_options = dict(kwargs)
        call_options["jac"] = analytic
        if callable(analytic):

            def checked_gradient(parameters: np.ndarray) -> np.ndarray:
                values = np.asarray(parameters, dtype=float)
                observed["all_evaluated_iterates_finite"] &= _array_finite(values)
                gradient = np.asarray(analytic(parameters), dtype=float)
                observed["gradient_evaluations"] += 1
                finite = bool(np.isfinite(gradient).all())
                observed["finite_gradient_evaluations"] += int(finite)
                if not finite and len(observed["non_finite_gradient_samples"]) < 3:
                    observed["non_finite_gradient_samples"].append(
                        {"x": values.tolist(), "gradient": gradient.tolist()}
                    )
                return gradient

            call_options["jac"] = checked_gradient

        previous_callback = call_options.get("callback")

        def callback(xk: Any) -> None:
            parameters = np.asarray(xk, dtype=float)
            observed["all_evaluated_iterates_finite"] &= _array_finite(parameters)
            observed["callback_iterations"].append(
                {
                    "iteration": len(observed["callback_iterations"]) + 1,
                    "x_sha256": _array_hash(parameters),
                    "all_finite": _array_finite(parameters),
                }
            )
            if callable(previous_callback):
                previous_callback(xk)

        call_options["callback"] = callback
        call_args = (checked_objective, start, *args[2:])
        effective_options = dict(options)
        if method == "L-BFGS-B":
            effective_options.setdefault("maxls", 20)
        if is_restart:
            call_options["options"] = dict(options)
            call_options["options"]["maxls"] = max(
                100, int(call_options["options"].get("maxls", 20))
            )
            effective_options = dict(call_options["options"])
        model_roles = (
            "legacy_main_history_model",
            "legacy_fcs_to_fbs_transition_model",
            "canonical_main_history_model",
            "canonical_fcs_to_fbs_transition_model",
        )
        event: dict[str, Any] = {
            "call_index": event_index,
            "model_role": (
                model_roles[event_index]
                if event_index < len(model_roles)
                else "additional_history_fit"
            ),
            "restart_from_captured_iterate": is_restart,
            "method": method,
            "x0": initial.tolist(),
            "starting_parameters": start.tolist(),
            "bounds": _json_value(bounds),
            "options_passed_to_optimizer": _json_value(
                call_options.get("options", options)
            ),
            "effective_maxls": effective_options.get("maxls"),
        }
        events.append(event)
        try:
            result = original_minimize(*call_args, **call_options)
        except Exception as error:
            event["minimize_exception"] = f"{type(error).__name__}: {error}"
            raise

        parameters = np.asarray(getattr(result, "x", []), dtype=float)
        try:
            reevaluated_objective = float(objective(parameters))
        except Exception as error:  # noqa: BLE001 - retain other diagnostics if reevaluation fails.
            reevaluated_objective = None
            event["objective_reevaluation_error"] = f"{type(error).__name__}: {error}"
        try:
            reevaluated_gradient = (
                np.asarray(analytic(parameters), dtype=float)
                if callable(analytic)
                else np.asarray([], dtype=float)
            )
        except Exception as error:  # noqa: BLE001 - retain other diagnostics if reevaluation fails.
            reevaluated_gradient = np.asarray([], dtype=float)
            event["gradient_reevaluation_error"] = f"{type(error).__name__}: {error}"

        feature_names = fit_context["feature_names"]
        lag_count = fit_context["lag_count"]
        beta_count = 1 + 2 * len(feature_names) + lag_count
        beta_gamma = {
            "beta": parameters[:beta_count].tolist(),
            "gamma": parameters[beta_count:].tolist(),
        }
        fit_rows = fit_context["rows"]
        processor = preseason.Preprocessor.fit(
            [row.features for row in fit_rows],
            feature_names,
            scale_floor=fit_context["preprocessor_scale_floor"],
            std_ddof=fit_context["preprocessor_std_ddof"],
        )
        design = np.column_stack(
            [
                np.ones(len(fit_rows)),
                processor.transform([row.features for row in fit_rows]),
            ]
        )
        eta = design @ parameters[beta_count:]
        clip_low, clip_high = preseason.DIRECT_RANK_LOG_SCALE_CLIP_BOUNDS
        low_indices = np.flatnonzero(eta <= clip_low).tolist()
        high_indices = np.flatnonzero(eta >= clip_high).tolist()
        result_success = bool(getattr(result, "success", False))
        event.update(
            {
                "optimize_result_fields": _result_fields(result),
                "result_x": parameters.tolist(),
                "result_coefficients": beta_gamma,
                "objective_reevaluated_at_result_x": reevaluated_objective,
                "analytic_gradient_reevaluated_at_result_x": reevaluated_gradient.tolist(),
                "max_absolute_analytic_gradient": (
                    float(np.max(np.abs(reevaluated_gradient)))
                    if reevaluated_gradient.size
                    else None
                ),
                "analytic_gradient_l2_norm": (
                    float(np.linalg.norm(reevaluated_gradient))
                    if reevaluated_gradient.size
                    else None
                ),
                "all_iterates_objectives_gradients_finite": bool(
                    observed["all_evaluated_iterates_finite"]
                    and observed["finite_objective_evaluations"]
                    == observed["objective_evaluations"]
                    and observed["finite_gradient_evaluations"]
                    == observed["gradient_evaluations"]
                    and _array_finite(parameters)
                    and reevaluated_objective is not None
                    and np.isfinite(reevaluated_objective)
                    and _array_finite(reevaluated_gradient)
                ),
                "observed_evaluations": observed,
                "bound_diagnostics": _bound_diagnostics(
                    parameters, reevaluated_gradient, bounds
                ),
                "scale_predictor_clipping": {
                    "clip_bounds": [clip_low, clip_high],
                    "row_count": len(eta),
                    "eta_min": float(np.min(eta)),
                    "eta_max": float(np.max(eta)),
                    "clipped_low_row_indices": low_indices,
                    "clipped_high_row_indices": high_indices,
                    "clipped_row_count": len(set(low_indices + high_indices)),
                },
                "gradient_check": _gradient_check(
                    objective, reevaluated_gradient, parameters
                ),
                "success": result_success,
                "status": int(getattr(result, "status", -1)),
                "message": str(getattr(result, "message", "")),
            }
        )
        return result

    preseason.minimize = instrumented_minimize
    preseason.DirectRankModel.fit = classmethod(instrumented_fit)
    history.reproduce_canonical_history_annual = capture_history_output
    test_exception: str | None = None
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        test_module.test_future_history_builder_matches_the_history_1_1_annual_procedure(
            root, monkeypatch
        )
    except Exception as error:  # noqa: BLE001 - raw fit errors are expected evidence.
        test_exception = f"{type(error).__name__}: {error}"
    finally:
        monkeypatch.undo()
        preseason.minimize = original_minimize
        preseason.DirectRankModel.fit = original_fit_descriptor
        history.reproduce_canonical_history_annual = original_reproduce

    output: dict[str, Any] = {
        "test_succeeded": test_exception is None,
        "test_exception": test_exception,
        "optimizer_runs": events,
        "training_input_source_identity_sha256": captured_history_output.get(
            "source_identity"
        ),
        "prediction_sha256": None,
        "pmf_map_sha256": None,
        "fitted_model": None,
    }
    if captured_history_output:
        prediction_bytes = captured_history_output["prediction_bytes"]
        fitted = captured_history_output["fitted"]
        rows = list(csv.DictReader(io.StringIO(prediction_bytes.decode("utf-8"))))
        pmfs = {row["team_id"]: row["pmf"] for row in rows}
        model = fitted["model"]
        output.update(
            {
                "prediction_sha256": _sha256_bytes(prediction_bytes),
                "pmf_map_sha256": _sha256_json(pmfs),
                "pmf_team_count": len(pmfs),
                "fitted_model": {
                    "beta": model["location_coefficients"],
                    "gamma": model["log_scale_coefficients"],
                    "model_metadata_sha256": _sha256_json(model),
                    "transition_model_beta": (
                        fitted.get("transition_model", {}).get("location_coefficients")
                        if fitted.get("transition_model")
                        else None
                    ),
                    "transition_model_gamma": (
                        fitted.get("transition_model", {}).get("log_scale_coefficients")
                        if fitted.get("transition_model")
                        else None
                    ),
                },
            }
        )
    return output


def _run_experiment(repo_root: Path) -> dict[str, Any]:
    test_module = _load_test_module(repo_root)
    runtime = _runtime_details()
    with tempfile.TemporaryDirectory(prefix="issue178-history-") as temp_dir:
        root = Path(temp_dir)
        baseline = _run_reproduction(root, test_module)
        rank_input = root / "data/processed/modeling/team_season_rank_distributions.csv"
        feature_input = root / "data/processed/preseason/team_season_features.csv"
        inputs = {
            "rank_fixture_sha256": _sha256_bytes(rank_input.read_bytes()),
            "feature_fixture_sha256": _sha256_bytes(feature_input.read_bytes()),
            "rank_fixture_path": "data/processed/modeling/team_season_rank_distributions.csv",
            "feature_fixture_path": "data/processed/preseason/team_season_features.csv",
        }
        failing = next(
            (
                run
                for run in baseline["optimizer_runs"]
                if not run.get("success", False)
            ),
            None,
        )
        manual_restart = None
        if (
            failing is not None
            and failing.get("status") == 2
            and "ABNORMAL" in str(failing.get("message", ""))
            and _array_finite(failing.get("result_x", []))
        ):
            failed_index = int(failing["call_index"])
            failed_parameters = np.asarray(failing["result_x"], dtype=float)
            with tempfile.TemporaryDirectory(
                prefix="issue178-history-restart-"
            ) as restart_dir:
                manual_restart = _run_reproduction(
                    Path(restart_dir),
                    test_module,
                    restart_call_index=failed_index,
                    restart_parameters=failed_parameters,
                )
            manual_restart["diagnostic_only"] = True
            manual_restart["restart_source_call_index"] = failed_index
            manual_restart["restart_maxls"] = 100

    return {
        "schema_version": 1,
        "issue": ISSUE,
        "reproducer": {
            "test": "tests/test_context_prior_v1_4_candidate.py",
            "fixture": f"{TEST_FIXTURE}(include_generic=True)",
            "operation": "test_future_history_builder_matches_the_history_1_1_annual_procedure",
            "threading_mode": runtime["mode"],
        },
        "runtime": runtime,
        "inputs": inputs,
        "baseline": baseline,
        "manual_restart": manual_restart,
    }


def _compact_build(build: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in build["optimizer_runs"]:
        result_fields = run.get("optimize_result_fields", {})
        runs.append(
            {
                key: run.get(key)
                for key in (
                    "call_index",
                    "model_role",
                    "restart_from_captured_iterate",
                    "success",
                    "status",
                    "message",
                    "result_coefficients",
                    "objective_reevaluated_at_result_x",
                    "max_absolute_analytic_gradient",
                    "all_iterates_objectives_gradients_finite",
                    "gradient_check",
                    "bound_diagnostics",
                )
            }
            | {
                "iterations": result_fields.get("nit"),
                "function_evaluations": result_fields.get("nfev"),
                "result_x_sha256": _array_hash(
                    np.asarray(run.get("result_x", []), dtype=float)
                ),
            }
        )
    fitted_model = build.get("fitted_model")
    return {
        "test_succeeded": build.get("test_succeeded"),
        "test_exception": build.get("test_exception"),
        "optimizer_runs": runs,
        "training_input_source_identity_sha256": build.get(
            "training_input_source_identity_sha256"
        ),
        "prediction_sha256": build.get("prediction_sha256"),
        "pmf_map_sha256": build.get("pmf_map_sha256"),
        "pmf_team_count": build.get("pmf_team_count"),
        "fitted_model": fitted_model,
    }


def _repeat_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    compact_runs = []
    for index, report in enumerate(reports, start=1):
        compact_runs.append(
            {
                "run": index,
                "baseline": _compact_build(report["baseline"]),
                "manual_restart": (
                    _compact_build(report["manual_restart"])
                    if report["manual_restart"] is not None
                    else None
                ),
            }
        )

    baseline_builds = [run["baseline"] for run in compact_runs]
    prediction_hashes = {
        build["prediction_sha256"]
        for build in baseline_builds
        if build["prediction_sha256"] is not None
    }
    pmf_map_hashes = {
        build["pmf_map_sha256"]
        for build in baseline_builds
        if build["pmf_map_sha256"] is not None
    }
    model_hashes = {
        build["fitted_model"]["model_metadata_sha256"]
        for build in baseline_builds
        if build["fitted_model"] is not None
    }
    status_counts: dict[str, int] = {}
    failure_count = 0
    for build in baseline_builds:
        if not build["test_succeeded"]:
            failure_count += 1
        for run in build["optimizer_runs"]:
            key = f"{run.get('model_role')}:{run.get('status')}:{run.get('message')}"
            status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "schema_version": 1,
        "issue": ISSUE,
        "reproducer": reports[0]["reproducer"],
        "runtime": reports[0]["runtime"],
        "inputs": reports[0]["inputs"],
        "repeat_count": len(reports),
        "summary": {
            "baseline_test_failures": failure_count,
            "baseline_test_successes": len(reports) - failure_count,
            "optimizer_status_counts": status_counts,
            "unique_model_metadata_hashes": sorted(model_hashes),
            "unique_prediction_hashes": sorted(prediction_hashes),
            "unique_pmf_map_hashes": sorted(pmf_map_hashes),
            "model_metadata_identical": len(model_hashes) <= 1,
            "prediction_identical": len(prediction_hashes) <= 1,
            "pmf_map_identical": len(pmf_map_hashes) <= 1,
        },
        "runs": compact_runs,
    }


def _summary_line(report: Mapping[str, Any]) -> dict[str, Any]:
    if "baseline" in report:
        baseline = report["baseline"]
        return {
            "mode": report["reproducer"]["threading_mode"],
            "test_succeeded": baseline["test_succeeded"],
            "optimizer_runs": [
                {
                    "role": item.get("model_role"),
                    "success": item.get("success"),
                    "status": item.get("status"),
                    "message": item.get("message"),
                }
                for item in baseline["optimizer_runs"]
            ],
            "prediction_sha256": baseline["prediction_sha256"],
            "pmf_map_sha256": baseline["pmf_map_sha256"],
        }
    return {
        "mode": report["reproducer"]["threading_mode"],
        "repeat_count": report["repeat_count"],
        "summary": report["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON artifact to this path; stdout is always emitted too",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat identical builds in this process and summarize each run",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    if args.repeat == 1:
        report = _run_experiment(repo_root)
    else:
        reports = [_run_experiment(repo_root) for _ in range(args.repeat)]
        report = _repeat_summary(reports)
    rendered = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(json.dumps(_summary_line(report), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
