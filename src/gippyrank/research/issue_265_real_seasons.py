"""Execute only the 32 frozen issue 265 baseline/reference/order solves.

Research outputs are isolated from production. No discovery of new publications,
perturbation experiments, warm starts, or fitting occurs here.
"""

from __future__ import annotations

import argparse
import gc
import gzip
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
from gippyrank.research.issue_261_quality_of_results import write_json
from gippyrank.research.issue_263_historical_evidence import load_historical_network
from gippyrank.research.issue_263_real_season_design import (
    ARTIFACT_DIR,
    check_design,
    validate_rows,
)
from gippyrank.research.quality_of_results import infer_quality_of_results

OUTPUT = "data/research/issue_265_real_seasons"
CONFIG = {"max_iterations": 500, "tolerance": 1e-9, "damping": 0.35}


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


def provenance(root: Path, spec: dict) -> dict:
    # Include transitive implementation sources, not just this entry point.
    paths = sorted(root.glob("src/gippyrank/**/*.py"))
    paths = [p for p in paths if p.name != "issue_265_population.py"]
    paths += [
        root / p
        for p in ("uv.lock", ".python-version", "scripts/run_issue_265_real_seasons.py")
    ]
    sources = {str(p.relative_to(root)): sha256(p) for p in paths}
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
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
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    value = {
        "source_sha256": sources,
        "input_sha256": inputs,
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
    origin = provenance(root, spec)
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
                            origin["execution_revision"],
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
    args = parser.parse_args()
    output = args.output_dir or args.root / OUTPUT
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
