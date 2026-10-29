"""Bounded λ=.40 extension of the frozen 252/261/265/267 experiments.

No original artifact is written. Only the new weight is solved; historical
neighbors are authenticated and consumed through their existing handoffs.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

from gippyrank.posterior.engine import _GAME_FACTOR_CACHE, Team
from gippyrank.posterior.snapshots import load_pinned_likelihood, sha256
from gippyrank.research.issue_242_quality_of_results import _sha256_json
from gippyrank.research.issue_252_quality_of_results import (
    OpponentSlot,
    _infer_trial,
    _set_focal_margin,
    generate_trial,
)
from gippyrank.research.issue_261_quality_of_results import (
    PANELS,
    PROFILES,
    SOURCE_FILES,
    evidence_hash,
    load_design,
    read_csv,
)
from gippyrank.research.issue_265_real_seasons import (
    BASELINE_SOURCES,
    CONFIG,
    read_checkpoint,
    resource_guard,
    solve,
    validate_pmfs,
    verify_completed_execution,
    write_checkpoint,
    write_json,
)
from gippyrank.research.issue_265_real_seasons import (
    OUTPUT as BASELINE,
)
from gippyrank.research.issue_267_analysis import verify_completed_diagnostics
from gippyrank.research.issue_267_diagnostics import (
    OUTPUT as DIAGNOSTICS,
)
from gippyrank.research.issue_267_diagnostics import (
    cell_identity,
    inventory,
    perturb,
)

OUTPUT = "data/research/issue_269_lambda_040"
WEIGHT = 0.4
NEIGHBORS = (0.25, 0.4, 0.5)
ORIGINALS = {
    "data/research/issue_252_quality_of_results": "f395bcf131f1c34b58ba91398cdf3a1972ee5d11f46af73a1b11d7fac3fe9c5f",
    "data/research/issue_261_quality_of_results": "ca4eae32cacea20de10932ed3dcd3c75f4e98ac748c8c656a8bd17cc30d1b208",
}
SOURCES = tuple(
    sorted(
        {
            *BASELINE_SOURCES,
            *SOURCE_FILES,
            "src/gippyrank/research/issue_267_diagnostics.py",
            "src/gippyrank/research/issue_267_analysis.py",
            "src/gippyrank/research/issue_269_candidate.py",
            "scripts/run_issue_269_candidate.py",
        }
    )
)


def authenticate(directory: Path, expected: str | None = None) -> dict:
    """Authenticate the complete retained inventory, including its manifest."""
    path = directory / "manifest.json"
    if expected is not None and sha256(path) != expected:
        raise ValueError(f"original manifest changed: {directory}")
    manifest = json.loads(path.read_text())
    hashes = manifest.get("artifact_sha256", manifest.get("output_sha256"))
    if not hashes:
        raise ValueError("missing output inventory")
    for name, digest in hashes.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"artifact integrity mismatch: {directory}/{name}")
    return manifest


def references(root: Path):
    for name, digest in ORIGINALS.items():
        authenticate(root / name, digest)
    verified = verify_completed_execution(root, root / BASELINE)
    authenticate(root / DIAGNOSTICS)
    _, jobs, diagnostics = verify_completed_diagnostics(
        root, root / DIAGNOSTICS, verified
    )
    return verified, jobs, diagnostics


def provenance(root: Path, verified) -> dict:
    manifests = (*ORIGINALS, BASELINE, DIAGNOSTICS)
    payload = {
        "issue": 269,
        "margin_weight": WEIGHT,
        "configuration": CONFIG,
        "source_sha256": {p: sha256(root / p) for p in SOURCES},
        "input_sha256": verified.origin["input_sha256"],
        "original_manifest_sha256": {
            p: sha256(root / p / "manifest.json") for p in manifests
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "resources": {
            "solve_timeout_seconds": 600,
            "rss_limit_bytes": 4 * 1024**3,
            "invocation_limit_seconds": 10800,
        },
        "scope": "3600 frozen trials; 5 baselines; 13 diagnostics; 8 issue252 states",
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        **payload,
        "fingerprint": _sha256_json(payload),
        "execution_revision": revision,
    }


def origin_for(root: Path, output: Path, verified, *, portable=False) -> dict:
    current = provenance(root, verified)
    path = output / "provenance.json"
    if not path.exists():
        if portable:
            raise ValueError("missing extension provenance")
        write_json(path, current)
        return current
    saved = json.loads(path.read_text())
    payload = {
        k: v for k, v in saved.items() if k not in ("fingerprint", "execution_revision")
    }
    if _sha256_json(payload) != saved["fingerprint"]:
        raise ValueError("extension provenance integrity mismatch")
    if portable:
        current["environment"] = saved["environment"]
        current["fingerprint"] = _sha256_json(
            {
                k: v
                for k, v in current.items()
                if k not in ("fingerprint", "execution_revision")
            }
        )
    if current["fingerprint"] != saved["fingerprint"]:
        raise ValueError("extension source/input/runtime mismatch")
    return saved


def synthetic_identity(
    origin: dict, profile: str, panel: str, seed: int, evidence: str
) -> dict:
    return {
        "execution_fingerprint": origin["fingerprint"],
        "profile": profile,
        "panel": panel,
        "seed": seed,
        "evidence_sha256": evidence,
        "margin_weight": WEIGHT,
        "configuration": CONFIG,
    }


def read_hashed(path: Path) -> dict:
    if (
        sha256(path)
        != json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    ):
        raise ValueError("synthetic checkpoint integrity mismatch")
    return json.loads(gzip.decompress(path.read_bytes()))


def validate_synthetic(cell: dict, identity: dict, origin: dict) -> dict:
    if (
        cell["identity"] != identity
        or cell["fingerprint"] != _sha256_json(identity)
        or cell["execution_revision"] != origin["execution_revision"]
    ):
        raise ValueError("synthetic checkpoint identity mismatch")
    groups = (
        {"same_schedule_a", "same_schedule_b"}
        if identity["panel"] == "control_conditioned"
        else {
            "approximately_even",
            "weaker_opponents",
            "same_schedule_a",
            "same_schedule_b",
        }
    )
    validate_pmfs(
        {t: np.array(p) for t, p in cell["pmfs"].items()}, {t: 138 for t in groups}
    )
    if (
        len(cell["teams"]) != len(groups)
        or {r["focal_group"] for r in cell["teams"]} != groups
        or len(cell["pairs"]) != len(groups) // 2
    ):
        raise ValueError("synthetic checkpoint coverage mismatch")
    for row in cell["teams"]:
        if (
            row["margin_weight"] != WEIGHT
            or not row["converged"]
            or not 0 <= row["iterations"] <= CONFIG["max_iterations"]
            or not 0 <= row["max_message_delta"] <= CONFIG["tolerance"]
        ):
            raise ValueError("synthetic convergence mismatch")
    return cell


def synthetic_trials(root: Path, profile: str):
    """Use retained membership, never repeat a seed search or condition on .40."""
    directory = root / "data/research/issue_261_quality_of_results"
    manifest = authenticate(directory, ORIGINALS[str(directory.relative_to(root))])
    design = load_design(root)
    likelihood = load_pinned_likelihood(
        design.directory / "historical_likelihood_v1.json"
    )
    originals = {}
    for row in read_csv(directory / "team_trial_summaries.csv.gz"):
        if row["profile"] == profile and float(row["margin_weight"]) == 0.25:
            originals[row["panel"], int(row["seed"]), row["focal_group"]] = row
    for panel in PANELS:
        selected = manifest["completed_panels"][profile][panel]
        expected_count = 500 if panel == "unconditional" else 200
        seeds = selected["seeds"]
        if (
            not selected["complete"]
            or len(seeds) != expected_count
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("frozen sample membership mismatch")
        for seed in seeds:
            trial = generate_trial(
                seed, design.focal[profile].prior, design.schedules[profile], likelihood
            )
            if panel == "control_conditioned":
                trial = replace(
                    trial,
                    groups={
                        k: v
                        for k, v in trial.groups.items()
                        if k.startswith("same_schedule_")
                    },
                )
            digest = evidence_hash(trial)
            if panel == "primary_conditioned" and (
                tuple(
                    trial.groups[g].wins
                    for g in ("approximately_even", "weaker_opponents")
                )
                != design.records[profile]
            ):
                raise ValueError("primary record conditioning changed")
            if panel == "control_conditioned" and any(
                g.wins != design.records[profile][0] for g in trial.groups.values()
            ):
                raise ValueError("control record conditioning changed")
            for group, component in trial.groups.items():
                old = originals[panel, seed, group]
                if (
                    old["evidence_sha256"] != digest
                    or int(old["wins"]) != component.wins
                ):
                    raise ValueError(
                        "regenerated trial differs from frozen observations"
                    )
            yield panel, seed, digest, trial, likelihood


def run_synthetic(
    root: Path, output: Path, origin: dict, profile: str, *, verify_only=False
) -> dict:
    started = time.monotonic()
    complete, reused, inference_seconds = 0, 0, 0.0
    archive = output / "synthetic" / f"{profile}.json.gz"
    cells = read_hashed(archive) if archive.exists() else {}
    for panel, seed, digest, trial, likelihood in synthetic_trials(root, profile):
        identity = synthetic_identity(origin, profile, panel, seed, digest)
        key = f"{panel}/{seed}"
        path = (
            output / ".checkpoints" / "synthetic" / profile / panel / f"{seed}.json.gz"
        )
        if key not in cells and not path.exists() and not verify_only:
            _GAME_FACTOR_CACHE.clear()
            before = time.monotonic()
            with resource_guard(started + 10800):
                teams, pairs, pmfs = _infer_trial(
                    trial, likelihood, margin_weights=(WEIGHT,), **CONFIG
                )
            metadata = {
                "profile": profile,
                "panel": panel,
                "focal_true_rank": trial.focal_rank,
                "evidence_sha256": digest,
            }
            for row in (*teams, *pairs):
                row.update(metadata)
            for row in pairs:
                row["comparison_population"] = (
                    "primary_seed_control"
                    if panel == "primary_conditioned"
                    and row["comparison"] == "same_schedule_control"
                    else panel
                )
            value = {
                "identity": identity,
                "fingerprint": _sha256_json(identity),
                "execution_revision": origin["execution_revision"],
                "teams": teams,
                "pairs": pairs,
                "pmfs": {t: p.tolist() for t, p in pmfs[WEIGHT].items()},
                "runtime_seconds": time.monotonic() - before,
            }
            write_checkpoint(path, value)
        else:
            reused += 1
        cell = validate_synthetic(
            cells[key] if key in cells else read_hashed(path), identity, origin
        )
        cells[key] = cell
        inference_seconds += cell["runtime_seconds"]
        complete += 1
        if complete % 100 == 0:
            print(f"{profile}: {complete}/900 trials validated", flush=True)
        _GAME_FACTOR_CACHE.clear()
    if len(cells) != 900:
        raise ValueError("unexpected synthetic archive membership")
    result = {
        "complete": complete == 900,
        "trials": complete,
        "reused_trials": reused,
        "inference_seconds": inference_seconds,
        "invocation_seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    if not verify_only:
        write_checkpoint(archive, cells)
        write_json(output / "synthetic" / profile / "execution.json", result)
    return result


def network_identity(network, origin: dict, job=None) -> dict:
    if job is not None:
        return cell_identity(network, job, WEIGHT, origin)
    return {
        "execution_fingerprint": origin["fingerprint"],
        "source_snapshot": network.specification,
        "margin_weight": WEIGHT,
        "configuration": CONFIG,
        "cold_start": True,
        "kind": "baseline",
    }


def network_jobs(verified):
    for sid, network in verified.networks.items():
        yield f"baseline-{sid}", network, None
    for job in inventory(verified):
        yield job.key, perturb(verified.networks[job.snapshot_id], job), job


def validate_network(path: Path, network, identity: dict, origin: dict) -> dict:
    cell = read_checkpoint(path, _sha256_json(identity), network)
    if (
        cell["identity"] != identity
        or cell["margin_weight"] != WEIGHT
        or cell["execution_revision"] != origin["execution_revision"]
        or cell["snapshot_id"] != network.snapshot_id
        or cell["kind"]
        != identity.get("kind", identity.get("perturbation", {}).get("kind"))
    ):
        raise ValueError("extension network checkpoint identity mismatch")
    pairs = {tuple(sorted((g.home_id, g.away_id))) for g in network.games}
    if cell["diagnostics"]["raw_game_factor_count"] != len(network.games) or cell[
        "diagnostics"
    ]["unique_pair_factor_count"] != len(pairs):
        raise ValueError("extension factor counts mismatch")
    return cell


def run_networks(
    root: Path, output: Path, verified, origin: dict, *, verify_only=False
) -> dict:
    started = time.monotonic()
    entries = []
    for key, network, job in network_jobs(verified):
        identity = network_identity(network, origin, job)
        path = output / "networks" / f"{key}.json.gz"
        if not path.exists() and not verify_only:
            print(f"{key}: λ=.40 start", flush=True)
            surfaces = {}
            cell = solve(
                network,
                job.kind if job else "baseline",
                WEIGHT,
                _sha256_json(identity),
                surfaces,
                started + 10800,
                origin["execution_revision"],
            )
            cell["identity"] = identity
            write_checkpoint(path, cell)
            surfaces.clear()
        cell = validate_network(path, network, identity, origin)
        entries.append({"key": key, **cell["diagnostics"]})
        if not verify_only:
            write_json(
                output / "networks" / "execution.json",
                {
                    "complete": len(entries) == 18,
                    "cells": entries,
                    "invocation_seconds": time.monotonic() - started,
                },
            )
        _GAME_FACTOR_CACHE.clear()
        gc.collect()
    return {"complete": len(entries) == 18, "cells": entries}


def frozen_252(root: Path):
    directory = root / "data/research/issue_252_quality_of_results"
    manifest = authenticate(directory, ORIGINALS[str(directory.relative_to(root))])
    grouped = {}
    for row in read_csv(directory / "prior_profiles.csv.gz"):
        grouped.setdefault(row["team_id"], []).append(row)
    teams = {}
    for tid, rows in grouped.items():
        rows.sort(key=lambda r: int(r["rank"]))
        teams[tid] = Team(
            tid,
            rows[0]["team_name"],
            "fbs",
            np.array([float(r["probability"]) for r in rows]),
        )
    schedules = {}
    for row in read_csv(directory / "schedule_definitions.csv"):
        schedules.setdefault(row["schedule_role"], []).append(
            OpponentSlot(
                teams[row["opponent_team_id"]],
                row["venue"],
                float(row["v1_expected_focal_win_probability"]),
                float(row["selection_target_win_probability"]),
            )
        )
    schedules = {k: tuple(v) for k, v in schedules.items()}
    likelihood = load_pinned_likelihood(directory / "historical_likelihood_v1.json")
    rows = read_csv(directory / "sensitivity_representative_games.csv.gz")
    seeds = {int(r["seed"]) for r in rows}
    if len(seeds) != 1:
        raise ValueError("unexpected frozen representative membership")
    trial = generate_trial(
        seeds.pop(),
        teams[manifest["inputs"]["focal_team_id"]].prior,
        schedules,
        likelihood,
        include_control=False,
        include_strong=True,
    )
    actual = [r for g in trial.groups.values() for r in g.game_rows]

    def row_key(row):
        return row["focal_group"], int(row["game_number"])

    if sorted(
        ({k: str(v) for k, v in r.items()} for r in actual), key=row_key
    ) != sorted(rows, key=row_key):
        raise ValueError("issue252 regenerated observations changed")
    return trial, schedules, likelihood


def run_252(root: Path, output: Path, origin: dict, *, verify_only=False) -> list:
    from gippyrank.research.quality_of_results import infer_quality_of_results

    trial, schedules, likelihood = frozen_252(root)
    result = []
    for role in ("approximately_even", "weaker_opponents"):
        component = trial.groups[role]
        target = 0.5 if role == "approximately_even" else 0.8
        index = min(
            range(10),
            key=lambda i: (
                abs(schedules[role][i].expected_win_probability - target),
                i,
            ),
        )
        for margin in (-3, 3, 10, 35):
            original = component.games[index]
            replacement = _set_focal_margin(
                original, original.home_id == component.focal.team_id, margin
            )
            games = [
                replacement if i == index else g for i, g in enumerate(component.games)
            ]
            identity = {
                "execution_fingerprint": origin["fingerprint"],
                "role": role,
                "seed": trial.seed,
                "game_index": index,
                "margin": margin,
                "evidence_sha256": evidence_hash(trial),
                "margin_weight": WEIGHT,
            }
            path = output / "issue252" / f"{role}-{margin:+d}.json.gz"
            if not path.exists() and not verify_only:
                _GAME_FACTOR_CACHE.clear()
                before = time.monotonic()
                inferred = infer_quality_of_results(
                    list(component.teams), games, likelihood, WEIGHT, **CONFIG
                )
                context = inferred.blended_context_posterior
                write_checkpoint(
                    path,
                    {
                        "identity": identity,
                        "fingerprint": _sha256_json(identity),
                        "execution_revision": origin["execution_revision"],
                        "pmfs": {role: inferred.pmfs[component.focal.team_id].tolist()},
                        "diagnostics": {
                            "converged": context.converged,
                            "iterations": context.iterations,
                            "max_message_delta": context.max_message_delta,
                        },
                        "runtime_seconds": time.monotonic() - before,
                    },
                )
            if (
                sha256(path)
                != json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
            ):
                raise ValueError("issue252 checkpoint integrity mismatch")
            cell = json.loads(gzip.decompress(path.read_bytes()))
            d = cell["diagnostics"]
            if (
                cell["identity"] != identity
                or cell["fingerprint"] != _sha256_json(identity)
                or cell["execution_revision"] != origin["execution_revision"]
                or not d["converged"]
                or not 0 <= d["iterations"] <= CONFIG["max_iterations"]
                or not 0 <= d["max_message_delta"] <= CONFIG["tolerance"]
            ):
                raise ValueError("issue252 identity/convergence mismatch")
            validate_pmfs(
                {k: np.array(p) for k, p in cell["pmfs"].items()}, {role: 138}
            )
            result.append(cell)
    _GAME_FACTOR_CACHE.clear()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("networks", "synthetic", "issue252", "all"), default="all"
    )
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / OUTPUT).resolve()
    # A follow-up cannot overwrite any prior research, source, site or processed data.
    protected = [
        root / p
        for p in (
            *ORIGINALS,
            BASELINE,
            DIAGNOSTICS,
            "data/research/issue_258_quality_of_results",
            "data/research/issue_263_real_season_design",
            "src",
            "site",
            "data/processed",
        )
    ]
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in protected
    ):
        raise ValueError("extension output overlaps protected artifacts")
    verified, _, _ = references(root)
    origin = origin_for(root, output, verified, portable=args.verify_only)
    if args.stage in ("all", "networks"):
        print(
            json.dumps(
                run_networks(
                    root, output, verified, origin, verify_only=args.verify_only
                )
            )
        )
    if args.stage in ("all", "issue252"):
        print(
            f"issue252: {len(run_252(root, output, origin, verify_only=args.verify_only))} states validated"
        )
    if args.stage in ("all", "synthetic"):
        for profile in (args.profile,) if args.profile else PROFILES:
            print(
                json.dumps(
                    run_synthetic(
                        root, output, origin, profile, verify_only=args.verify_only
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
