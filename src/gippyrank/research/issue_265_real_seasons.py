"""Execute only the 32 frozen issue 265 baseline/reference/order solves.

Research outputs are isolated from production. No discovery of new publications,
perturbation experiments, warm starts, or fitting occurs here.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import json
import platform
import resource
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from gippyrank.artifact_hashes import included_game_rows_sha256, posterior_pmfs_sha256
from gippyrank.performance_snapshot import _pmfs, strip_context_prior
from gippyrank.performance_v1 import uniform_pmf
from gippyrank.posterior.engine import (
    _GAME_FACTOR_CACHE,
    Game,
    LikelihoodV1,
    Team,
    infer_posterior,
)
from gippyrank.posterior.snapshots import load_pinned_likelihood, sha256
from gippyrank.quality_of_record import infer_quality_of_record
from gippyrank.research.issue_242_quality_of_results import (
    MARGIN_WEIGHTS,
    PublishedInput,
    _input_bundle,
    _read_json,
    _read_rows,
    _sha256_json,
)
from gippyrank.research.issue_263_historical_evidence import load_historical_network
from gippyrank.research.issue_263_real_season_design import (
    ARTIFACT_DIR,
    check_design,
    validate_rows,
)
from gippyrank.research.quality_of_results import infer_quality_of_results

OUTPUT = "data/research/issue_265_real_seasons"
CONFIG = {"max_iterations": 500, "tolerance": 1e-9, "damping": 0.35}
# Audited execution path: frozen loaders/validators, factors/BP, prior stripping,
# shared likelihood/hash primitives and this runner. Imported research programs
# used only during design or reporting are not baseline inference dependencies.
BASELINE_SOURCES = (
    "src/gippyrank/artifact_hashes.py",
    "src/gippyrank/data/cfbd.py",
    "src/gippyrank/data/cfbd_games.py",
    "src/gippyrank/data/game_evidence.py",
    "src/gippyrank/likelihood_provenance.py",
    "src/gippyrank/methodology.py",
    "src/gippyrank/modeling.py",
    "src/gippyrank/performance_snapshot.py",
    "src/gippyrank/performance_v1.py",
    "src/gippyrank/posterior/engine.py",
    "src/gippyrank/posterior/predictive.py",
    "src/gippyrank/posterior/season_simulation.py",
    "src/gippyrank/posterior/snapshots.py",
    "src/gippyrank/quality_of_record.py",
    "src/gippyrank/quality_of_record_snapshot.py",
    "src/gippyrank/research/issue_242_quality_of_results.py",
    "src/gippyrank/research/issue_263_historical_evidence.py",
    "src/gippyrank/research/issue_263_real_season_design.py",
    "src/gippyrank/research/quality_of_results.py",
    "src/gippyrank/research/issue_265_real_seasons.py",
    "scripts/run_issue_265_real_seasons.py",
    "uv.lock",
    ".python-version",
)

# One immutable, already completed run. These anchors identify the original
# artifacts, not a replacement execution identity or rewritten fingerprints.
HISTORICAL_FINGERPRINT = (
    "dc7ab5703b9a7d464c7782b4d0ab75d8167f9cedd7a03d533e6d21f50addc0fa"
)
HISTORICAL_REVISION = "aba5f265a7bbca3bb3f6014d7c5500d1e9fdcd5a"
HISTORICAL_MANIFEST_SHA256 = (
    "6c6454697b19109ad76b3cf4dfd8cde8afa5aa7162882e9a81cff692b03c8f37"
)
HISTORICAL_RUNNER_SHA256 = (
    "b30fd16de9a26437d1a2f379cc759fae9b8d12a233d537a52e3c9706bc0eb3d8"
)
HISTORICAL_RUNNER_DEFINITIONS = {
    "Network",
    "load_network",
    "validate_pmfs",
    "mean_rank",
    "parity",
    "cell_fingerprint",
    "current_rss_bytes",
    "resource_guard",
    "solve",
}


def write_json(path: Path, value: object) -> None:
    """Atomic execution metadata, independent of other research runners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass
class Network:
    snapshot_id: str
    specification: dict[str, Any]
    teams: list[Team]
    games: list[Game]
    rows: list[dict[str, str]]
    likelihood: LikelihoodV1
    references: dict[str, dict[str, np.ndarray]]
    context: dict[str, np.ndarray]

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter({t.team_id: 0 for t in self.teams})
        for game in self.games:
            counts.update((game.home_id, game.away_id))
        return dict(counts)

    @property
    def fbs(self) -> dict[str, int]:
        return {t.team_id: len(t.prior) for t in self.teams if t.subdivision == "fbs"}


def load_network(root: Path, spec: dict, snapshot_id: str) -> Network:
    historical = next(
        (s for s in spec["historical_snapshots"] if s["snapshot_id"] == snapshot_id),
        None,
    )
    if historical:
        frozen = historical
        directory = root / ARTIFACT_DIR / frozen["input_directory"]
        teams, games, metadata = load_historical_network(directory)
        rows = _read_rows(directory / "included_games.csv")
        likelihood = load_pinned_likelihood(
            root / "data/processed/posterior/historical_likelihood_v1.json"
        )
        references, context = {}, {}
        if metadata != frozen:
            raise ValueError("historical metadata differs from protocol")
    else:
        frozen = next(
            s for s in spec["snapshot_audit"] if s["publication_slot"] == snapshot_id
        )
        if not frozen["selected"]:
            raise ValueError("unselected snapshot")
        published = PublishedInput(
            snapshot_id,
            frozen["display_label"],
            0,
            root / frozen["context_source"],
            root / frozen["qop_source"],
            root / frozen["qor_source"],
        )
        bundle = _input_bundle(root, published)
        source, qop, qor, teams, rows, games, config, metadata, _, likelihood, _ = (
            bundle
        )
        validate_rows(rows, metadata)
        if config != frozen["inference_configuration"]:
            raise ValueError("loaded inference configuration drift")
        references = {
            "qop": _pmfs(qop.directory / "posterior_pmfs.csv"),
            "qor": _pmfs(qor.directory / "posterior_pmfs.csv"),
        }
        context = _pmfs(source.directory / "posterior_pmfs.csv")
        for field in (
            "requested_cutoff",
            "effective_cutoff",
            "prior_model_version",
            "historical_likelihood_sha256",
            "source_response_hashes",
        ):
            if metadata[field] != frozen[field]:
                raise ValueError(f"source identity drift: {field}")
    expected_config = (
        CONFIG
        if historical
        else {"implementation": "deterministic_damped_loopy_sum_product", **CONFIG}
    )
    if frozen["inference_configuration"] != expected_config:
        raise ValueError("frozen numerical configuration changed")
    if (
        sorted(t.team_id for t in teams) != frozen["network_team_ids"]
        or sorted(t.team_id for t in teams if t.subdivision == "fbs")
        != frozen["fbs_team_ids"]
        or len(games) != frozen["included_game_count"]
        or included_game_rows_sha256(rows) != frozen["included_game_rows_sha256"]
        or posterior_pmfs_sha256({t.team_id: t.prior for t in teams})
        != frozen["prior_pmfs_sha256"]
    ):
        raise ValueError("frozen population/evidence/prior mismatch")
    network = Network(
        snapshot_id,
        frozen,
        sorted(teams, key=lambda t: t.team_id),
        sorted(games, key=lambda g: g.game_id),
        rows,
        likelihood,
        references,
        context,
    )
    validate_pmfs(
        {t.team_id: t.prior for t in teams}, {t.team_id: len(t.prior) for t in teams}
    )
    for pmfs in references.values():
        validate_pmfs(pmfs, network.fbs)
    return network


def validate_pmfs(pmfs: dict[str, np.ndarray], support: dict[str, int]) -> None:
    if pmfs.keys() != support.keys():
        raise ValueError("PMF population mismatch")
    for team_id, p in pmfs.items():
        if (
            p.shape != (support[team_id],)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or abs(float(p.sum()) - 1) > 1e-9
        ):
            raise ValueError(f"invalid PMF: {team_id}")


def mean_rank(p: np.ndarray) -> float:
    return float(p @ np.arange(1, len(p) + 1))


def parity(
    actual: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    counts: dict[str, int],
) -> dict:
    """Check complete PMFs, means and ordering; enumerate numerical ties."""
    if actual.keys() != reference.keys():
        raise ValueError("parity population mismatch")
    cell = max(float(np.max(np.abs(actual[t] - reference[t]))) for t in actual)
    mean_errors = {
        t: abs(mean_rank(actual[t]) - mean_rank(reference[t])) for t in actual
    }
    tolerance = len(next(iter(actual.values()))) * 1e-8
    rated = sorted(t for t in actual if counts[t])
    ties, inversions = [], []
    for i, a in enumerate(rated):
        for b in rated[i + 1 :]:
            da = mean_rank(actual[a]) - mean_rank(actual[b])
            dr = mean_rank(reference[a]) - mean_rank(reference[b])
            if min(abs(da), abs(dr)) <= tolerance:
                ties.append(
                    {"first": a, "second": b, "actual_gap": da, "reference_gap": dr}
                )
            elif da * dr < 0:
                inversions.append([a, b])
    return {
        "passed": cell <= 1e-8
        and max(mean_errors.values()) <= tolerance
        and not inversions,
        "max_absolute_pmf_difference": cell,
        "max_expected_rank_difference": max(mean_errors.values()),
        "pmf_tolerance": 1e-8,
        "expected_rank_tolerance": tolerance,
        "numerical_ties": ties,
        "non_tied_order_inversions": inversions,
        "team_count": len(actual),
    }


def frozen_input_hashes(root: Path, spec: dict) -> dict[str, str]:
    """The protocol's exact inventory, validated rather than relabeled on drift."""
    inputs = {**spec["input_sha256"]}
    for snapshot in spec["snapshot_audit"]:
        inputs.update(snapshot["artifact_sha256"])
    for name, digest in _read_json(root / ARTIFACT_DIR / "manifest.json")[
        "artifact_sha256"
    ].items():
        inputs[f"{ARTIFACT_DIR}/{name}"] = digest
    inputs[f"{ARTIFACT_DIR}/manifest.json"] = sha256(
        root / ARTIFACT_DIR / "manifest.json"
    )
    for path, expected in inputs.items():
        if sha256(root / path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    return inputs


def provenance(root: Path, spec: dict) -> dict:
    sources = {path: sha256(root / path) for path in BASELINE_SOURCES}
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    value = {
        "source_sha256": sources,
        "input_sha256": frozen_input_hashes(root, spec),
        "environment": environment,
        "configuration": CONFIG,
        "scope": "25 baselines + 4 independent references + 3 order reversals",
    }
    return {**value, "fingerprint": _sha256_json(value), "execution_revision": revision}


def cell_fingerprint(network: Network, origin: dict, kind: str, weight: float) -> str:
    return _sha256_json(
        {
            "execution": origin["fingerprint"],
            "snapshot": network.specification,
            "kind": kind,
            "weight": weight,
            "cold_start": True,
            "team_order": [t.team_id for t in network.teams],
            "game_order": [g.game_id for g in network.games],
        }
    )


def write_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(raw, mtime=0))
    temporary.replace(path)
    write_json(path.with_suffix(".sha256.json"), {"sha256": sha256(path)})


def read_checkpoint(path: Path, fingerprint: str, network: Network) -> dict:
    if sha256(path) != _read_json(path.with_suffix(".sha256.json"))["sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    value = json.loads(gzip.decompress(path.read_bytes()))
    if value["fingerprint"] != fingerprint or not value["diagnostics"]["converged"]:
        raise ValueError("checkpoint input/configuration/source mismatch")
    diagnostics = value["diagnostics"]
    if (
        not 0 <= diagnostics["iterations"] <= CONFIG["max_iterations"]
        or not 0 <= diagnostics["max_message_delta"] <= CONFIG["tolerance"]
    ):
        raise ValueError("checkpoint convergence diagnostics invalid")
    validate_pmfs({t: np.array(p) for t, p in value["pmfs"].items()}, network.fbs)
    validate_pmfs(
        {t: np.array(p) for t, p in value["context"].items()},
        {t.team_id: len(t.prior) for t in network.teams},
    )
    if value["counts"] != network.counts:
        raise ValueError("checkpoint evidence counts mismatch")
    return value


@dataclass
class VerifiedExecution:
    """Read-only handoff of validated cells, retaining their original identity."""

    specification: dict[str, Any]
    origin: dict[str, Any]
    execution: dict[str, Any]
    networks: dict[str, Network]
    cells: dict[str, dict[tuple[str, float], dict[str, Any]]]
    historical: bool


def historical_runner_signature(root: Path) -> str:
    """Pin the original inference path while allowing the provenance repair.

    This single-run compatibility check covers the original inference/loader
    definitions and project imports. It does not exempt arbitrary edits to solve
    or silently declare a newly generated checkpoint historically compatible.
    """
    tree = ast.parse(
        (root / "src/gippyrank/research/issue_265_real_seasons.py").read_text()
    )
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in HISTORICAL_RUNNER_DEFINITIONS
        )
        or (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("gippyrank.")
            and node.module != "gippyrank.research.issue_261_quality_of_results"
        )
    ]
    return hashlib.sha256(
        ast.dump(
            ast.Module(body=nodes, type_ignores=[]), include_attributes=False
        ).encode()
    ).hexdigest()


def verify_completed_execution(root: Path, output: Path) -> VerifiedExecution:
    """Consume all 32 cells without solving, writing, or changing their identity.

    Only the explicitly pinned original run has historical compatibility. Every
    newly generated run still requires its exact current execution fingerprint.
    Historical runtime versions remain audit data; reading immutable PMFs on a
    different platform does not assert that inference ran on that platform.
    """
    spec = check_design(root, root / ARTIFACT_DIR)
    origin = _read_json(output / "provenance.json")
    historical = origin["fingerprint"] == HISTORICAL_FINGERPRINT
    if historical:
        if sha256(output / "manifest.json") != HISTORICAL_MANIFEST_SHA256:
            raise ValueError("historical manifest integrity mismatch")
        manifest = _read_json(output / "manifest.json")
        payload = {
            k: v
            for k, v in origin.items()
            if k not in {"execution_revision", "fingerprint"}
        }
        if (
            manifest["execution_provenance"] != origin
            or origin["execution_revision"] != HISTORICAL_REVISION
            or _sha256_json(payload) != HISTORICAL_FINGERPRINT
        ):
            raise ValueError("historical execution identity mismatch")
        if origin["configuration"] != CONFIG or origin[
            "input_sha256"
        ] != frozen_input_hashes(root, spec):
            raise ValueError("historical inputs/configuration mismatch")
        for path in BASELINE_SOURCES:
            if path == "src/gippyrank/research/issue_265_real_seasons.py":
                continue  # Its inference definitions/imports are checked below.
            if sha256(root / path) != origin["source_sha256"][path]:
                raise ValueError(f"historical inference dependency changed: {path}")
        if historical_runner_signature(root) != HISTORICAL_RUNNER_SHA256:
            raise ValueError("historical runner inference path changed")
        for path, expected in manifest["output_sha256"].items():
            if sha256(output / path) != expected:
                raise ValueError(f"historical output integrity mismatch: {path}")
    elif origin["fingerprint"] != provenance(root, spec)["fingerprint"]:
        raise ValueError("checkpoint input/configuration/source mismatch")
    execution = _read_json(output / "execution.json")
    gates = _read_json(output / "endpoint_parity.json")
    if (
        not execution["complete"]
        or execution["failures"]
        or execution["expected_cells"] != 32
        or len(execution["cells"]) != 32
        or set(gates) != set(spec["selected_snapshot_ids"])
    ):
        raise ValueError("incomplete execution cannot be consumed")
    if not all(g["passed"] for snapshot in gates.values() for g in snapshot.values()):
        raise ValueError("numerical gates block consumption")
    networks, cells = {}, {}
    for snapshot_id in spec["selected_snapshot_ids"]:
        network = load_network(root, spec, snapshot_id)
        networks[snapshot_id] = network
        jobs = [("baseline", w) for w in MARGIN_WEIGHTS]
        if network.specification["season"] == 2025:
            jobs += [("reference", 0.0), ("reference", 1.0)]
        if snapshot_id == spec["game_diagnostics"]["publication_slot"]:
            jobs += [
                ("reversed", w)
                for w in spec["candidate_configuration"]["order_reversal_check_weights"]
            ]
        recorded = [c for c in execution["cells"] if c["snapshot_id"] == snapshot_id]
        if len(recorded) != len(jobs) or {
            (c["kind"], c["margin_weight"]) for c in recorded
        } != set(jobs):
            raise ValueError("execution coverage mismatch")
        validated = {}
        for kind, weight in jobs:
            relative = f"checkpoints/{snapshot_id}/{kind}-{weight:g}.json.gz"
            fingerprint = cell_fingerprint(network, origin, kind, weight)
            cell = read_checkpoint(output / relative, fingerprint, network)
            if (
                cell["snapshot_id"] != snapshot_id
                or cell["kind"] != kind
                or cell["margin_weight"] != weight
                or (historical and cell["execution_revision"] != HISTORICAL_REVISION)
            ):
                raise ValueError("checkpoint execution identity mismatch")
            entry = next(
                c for c in recorded if (c["kind"], c["margin_weight"]) == (kind, weight)
            )
            if (
                entry["checkpoint"] != relative
                or entry["fingerprint"] != fingerprint
                or any(entry[k] != v for k, v in cell["diagnostics"].items())
            ):
                raise ValueError("checkpoint execution inventory mismatch")
            validated[kind, weight] = cell
        recomputed = {}
        for weight, family in ((0.0, "qor"), (1.0, "qop")):
            reference = (
                network.references[family]
                if network.references
                else {
                    t: np.array(p)
                    for t, p in validated["reference", weight]["pmfs"].items()
                }
            )
            recomputed[family] = parity(
                {
                    t: np.array(p)
                    for t, p in validated["baseline", weight]["pmfs"].items()
                },
                reference,
                network.counts,
            )
        for kind, weight in jobs:
            if kind == "reversed":
                recomputed[f"order_{weight:g}"] = parity(
                    {
                        t: np.array(p)
                        for t, p in validated[kind, weight]["pmfs"].items()
                    },
                    {
                        t: np.array(p)
                        for t, p in validated["baseline", weight]["pmfs"].items()
                    },
                    network.counts,
                )
        # Recheck the frozen tolerances, not bitwise equality of floating-point
        # error summaries across BLAS/platforms. The original gates are anchored.
        if set(recomputed) != set(gates[snapshot_id]) or not all(
            g["passed"] for g in recomputed.values()
        ):
            raise ValueError("recomputed numerical gates failed")
        cells[snapshot_id] = validated
    return VerifiedExecution(spec, origin, execution, networks, cells, historical)


def current_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("RSS measurement unavailable")


@contextmanager
def resource_guard(
    overall_deadline: float, timeout: float = 600, rss_limit: int = 4 * 1024**3
):
    """Poll process RSS and frozen wall limits without changing inference settings."""
    deadline = min(overall_deadline, time.monotonic() + timeout)
    old_handler = signal.getsignal(signal.SIGALRM)

    def check(_signum, _frame):
        if time.monotonic() > deadline:
            raise TimeoutError("frozen execution/solve wall budget exceeded")
        if current_rss_bytes() > rss_limit:
            raise MemoryError("frozen 4 GiB RSS budget exceeded")

    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
    try:
        check(None, None)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def solve(
    network: Network,
    kind: str,
    weight: float,
    fingerprint: str,
    surfaces: dict,
    deadline: float,
    revision: str,
) -> dict:
    teams, games = network.teams, network.games
    if kind == "reversed":
        teams, games = list(reversed(teams)), list(reversed(games))
    _GAME_FACTOR_CACHE.clear()  # Bound cache memory across candidates; BP still cold starts.
    gc.collect()
    started = time.monotonic()
    with resource_guard(deadline):
        if kind == "reference" and weight == 0:
            inferred = infer_quality_of_record(
                teams, games, network.likelihood, **CONFIG
            )
            result, pmfs = inferred.binary_context_posterior, inferred.pmfs
        elif kind == "reference" and weight == 1:
            result = infer_posterior(teams, games, network.likelihood, **CONFIG)
            pmfs = {
                t.team_id: strip_context_prior(result.pmfs[t.team_id], t.prior)
                if network.counts[t.team_id]
                else uniform_pmf(len(t.prior))
                for t in teams
                if t.subdivision == "fbs"
            }
        else:
            inferred = infer_quality_of_results(
                teams,
                games,
                network.likelihood,
                weight,
                factor_surface_cache=surfaces,
                **CONFIG,
            )
            result, pmfs = inferred.blended_context_posterior, inferred.pmfs
        if not result.converged:
            raise RuntimeError("reference did not converge")
        validate_pmfs(pmfs, network.fbs)
        validate_pmfs(result.pmfs, {t.team_id: len(t.prior) for t in teams})
    elapsed = time.monotonic() - started
    return {
        "fingerprint": fingerprint,
        "execution_revision": revision,
        "snapshot_id": network.snapshot_id,
        "kind": kind,
        "margin_weight": weight,
        "counts": network.counts,
        "pmfs": {t: p.tolist() for t, p in pmfs.items()},
        "context": {t: p.tolist() for t, p in result.pmfs.items()},
        "diagnostics": {
            "converged": result.converged,
            "iterations": result.iterations,
            "max_message_delta": result.max_message_delta,
            "runtime_seconds": elapsed,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "raw_game_factor_count": result.raw_game_factor_count,
            "unique_pair_factor_count": result.unique_pair_factor_count,
            "max_pmf_sum_error": max(
                abs(float(p.sum()) - 1) for p in result.pmfs.values()
            ),
            "minimum_pmf_cell": min(float(p.min()) for p in result.pmfs.values()),
            "all_pmfs_finite_nonnegative_normalized": True,
        },
    }


def execute(root: Path, output: Path) -> dict:
    started = time.monotonic()
    spec = check_design(root, root / ARTIFACT_DIR)
    saved = (
        _read_json(output / "provenance.json")
        if (output / "provenance.json").exists()
        else None
    )
    if saved and saved["fingerprint"] == HISTORICAL_FINGERPRINT:
        return verify_completed_execution(root, output).execution
    origin = provenance(root, spec)
    actual_revision = origin["execution_revision"]
    if saved:
        if saved["fingerprint"] != origin["fingerprint"]:
            raise ValueError("existing execution input/configuration/source mismatch")
        origin = saved  # Resume preserves the original run identity.
    else:
        write_json(output / "provenance.json", origin)
    deadline = started + spec["execution"]["wall_budget_hours"] * 3600
    cells, failures, gates = [], [], {}
    for snapshot_id in spec["selected_snapshot_ids"]:
        if time.monotonic() >= deadline:
            failures.append(
                {"snapshot_id": snapshot_id, "error": "overall wall budget exhausted"}
            )
            continue
        network = load_network(root, spec, snapshot_id)
        jobs = [("baseline", w) for w in MARGIN_WEIGHTS]
        if network.specification["season"] == 2025:
            jobs += [("reference", 0.0), ("reference", 1.0)]
        if snapshot_id == spec["game_diagnostics"]["publication_slot"]:
            jobs += [
                ("reversed", w)
                for w in spec["candidate_configuration"]["order_reversal_check_weights"]
            ]
        surfaces, results = {}, {}
        try:
            for kind, weight in jobs:
                cell_id = f"{snapshot_id}/{kind}-{weight:g}"
                path = output / "checkpoints" / (cell_id + ".json.gz")
                fingerprint = cell_fingerprint(network, origin, kind, weight)
                print(f"{cell_id}: start", flush=True)
                cell_started = time.monotonic()
                try:
                    if path.exists():
                        value = read_checkpoint(path, fingerprint, network)
                    else:
                        value = solve(
                            network,
                            kind,
                            weight,
                            fingerprint,
                            surfaces,
                            deadline,
                            actual_revision,
                        )
                        write_checkpoint(path, value)
                except Exception as error:
                    failure = {
                        "snapshot_id": snapshot_id,
                        "kind": kind,
                        "margin_weight": weight,
                        "fingerprint": fingerprint,
                        "error": f"{type(error).__name__}: {error}",
                        "runtime_seconds": time.monotonic() - cell_started,
                    }
                    write_json(
                        output / "failures" / f"{snapshot_id}-{kind}-{weight:g}.json",
                        failure,
                    )
                    raise
                results[kind, weight] = value
                cells.append(
                    {
                        "snapshot_id": snapshot_id,
                        "kind": kind,
                        "margin_weight": weight,
                        "checkpoint": str(path.relative_to(output)),
                        "fingerprint": fingerprint,
                        **value["diagnostics"],
                    }
                )
                print(
                    f"{cell_id}: converged in {value['diagnostics']['iterations']} iterations",
                    flush=True,
                )
            snapshot_gates = {}
            for weight, family in ((0.0, "qor"), (1.0, "qop")):
                baseline = {
                    t: np.array(p)
                    for t, p in results["baseline", weight]["pmfs"].items()
                }
                reference = (
                    network.references[family]
                    if network.references
                    else {
                        t: np.array(p)
                        for t, p in results["reference", weight]["pmfs"].items()
                    }
                )
                snapshot_gates[family] = parity(baseline, reference, network.counts)
            for (kind, weight), result in results.items():
                if kind == "reversed":
                    snapshot_gates[f"order_{weight:g}"] = parity(
                        {t: np.array(p) for t, p in result["pmfs"].items()},
                        {
                            t: np.array(p)
                            for t, p in results["baseline", weight]["pmfs"].items()
                        },
                        network.counts,
                    )
            gates[snapshot_id] = snapshot_gates
            if not all(g["passed"] for g in snapshot_gates.values()):
                raise ValueError(
                    "endpoint or input-order gate failed; snapshot interpretation blocked"
                )
        except Exception as error:  # noqa: BLE001 - preserve failed experimental cells
            failures.append(
                {
                    "snapshot_id": snapshot_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{snapshot_id}: FAILED {error}", flush=True)
        finally:
            _GAME_FACTOR_CACHE.clear()
            surfaces.clear()
            del network, results
            gc.collect()
            write_json(output / "endpoint_parity.json", gates)
            write_json(
                output / "execution.json",
                {
                    "cells": cells,
                    "failures": failures,
                    "expected_cells": 32,
                    "complete": len(cells) == 32 and not failures and len(gates) == 5,
                    "invocation_runtime_seconds": time.monotonic() - started,
                },
            )
    return _read_json(output / "execution.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read and verify saved results without inference or writes",
    )
    args = parser.parse_args()
    output = args.output_dir or args.root / OUTPUT
    if args.verify_only or (
        not args.analyze_only
        and (output / "provenance.json").exists()
        and _read_json(output / "provenance.json")["fingerprint"]
        == HISTORICAL_FINGERPRINT
    ):
        verified = verify_completed_execution(args.root, output)
        print(
            json.dumps(
                {
                    "status": "verified_historical"
                    if verified.historical
                    else "verified",
                    "cells": sum(len(c) for c in verified.cells.values()),
                    "execution_revision": verified.origin["execution_revision"],
                    "fingerprint": verified.origin["fingerprint"],
                },
                sort_keys=True,
            )
        )
        return
    invocation_deadline = time.monotonic() + 3 * 3600
    if not args.analyze_only:
        result = execute(args.root, output)
        if not result["complete"]:
            raise SystemExit(
                "incomplete experiment; saved successful cells and failures"
            )
    from gippyrank.research.issue_265_population import analyze

    with resource_guard(invocation_deadline, timeout=3 * 3600):
        analyze(args.root, output)


if __name__ == "__main__":
    main()
