"""The 65 frozen Phase 4 evidence perturbations; immutable baseline handoff.

This is a bounded research runner, not a publication or attribution service.
Each solve uses the original priors and a deterministic cold start.
"""

from __future__ import annotations

import argparse
import gc
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy

from gippyrank.artifact_hashes import included_game_rows_sha256, posterior_pmfs_sha256
from gippyrank.posterior.engine import _GAME_FACTOR_CACHE
from gippyrank.posterior.snapshots import sha256
from gippyrank.research.issue_242_quality_of_results import (
    MARGIN_WEIGHTS,
    _read_json,
    _sha256_json,
)
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR
from gippyrank.research.issue_265_real_seasons import (
    BASELINE_SOURCES,
    CONFIG,
    Network,
    VerifiedExecution,
    parity,
    read_checkpoint,
    resource_guard,
    solve,
    verify_completed_execution,
    write_checkpoint,
    write_json,
)
from gippyrank.research.issue_265_real_seasons import (
    OUTPUT as BASELINE_OUTPUT,
)

OUTPUT = "data/research/issue_267_real_season_diagnostics"
SOURCES = (
    *BASELINE_SOURCES,
    "src/gippyrank/research/issue_267_diagnostics.py",
    "scripts/run_issue_267_diagnostics.py",
)


@dataclass(frozen=True)
class Perturbation:
    snapshot_id: str
    game_id: str
    kind: str
    focal_id: str
    opponent_id: str
    profile: str
    margin: int | None
    original_observation: dict
    replacement_observation: dict | None

    @property
    def key(self) -> str:
        suffix = f"-{self.margin:+d}" if self.margin is not None else ""
        return f"{self.kind}-{self.game_id}{suffix}"


def inventory(verified: VerifiedExecution) -> list[Perturbation]:
    """Read the frozen inventory, never select games using candidate results."""
    plan = verified.specification["game_diagnostics"]
    sid = plan["publication_slot"]
    network = verified.networks[sid]
    rows = {r["id"]: r for r in network.rows}
    if len(rows) != len(network.rows):
        raise ValueError("duplicate frozen observation identity")
    probes, result = {}, []
    for probe in plan["counterfactuals"]:
        gid, focal = probe["game_id"], probe["team_id"]
        row = rows[gid]
        opponent = row["awayId"] if focal == row["homeId"] else row["homeId"]
        if (
            focal not in (row["homeId"], row["awayId"])
            or opponent != probe["opponent_id"]
        ):
            raise ValueError("frozen counterfactual orientation mismatch")
        probes[gid] = (focal, opponent, probe["profile"])
        home = focal == row["homeId"]
        opponent_points = max(int(row["awayPoints" if home else "homePoints"]), 3)
        for margin in probe["margins"]:
            replacement = {
                **row,
                "homePoints": str(
                    opponent_points + margin if home else opponent_points
                ),
                "awayPoints": str(
                    opponent_points if home else opponent_points + margin
                ),
            }
            result.append(
                Perturbation(
                    sid,
                    gid,
                    "replacement",
                    focal,
                    opponent,
                    probe["profile"],
                    margin,
                    dict(row),
                    replacement,
                )
            )
    # The remaining three focal identities are fixed by the frozen case categories.
    categories = {
        "401858496": ("130", "competitive_elite_loss"),
        "401858455": ("2483", "dominant_win_overmatched_opponent"),
        "401862807": ("235", "narrow_win_weaker_opponent"),
    }
    for gid in plan["removal_game_ids"]:
        row = rows[gid]
        if gid in probes:
            focal, opponent, profile = probes[gid]
        else:
            focal, profile = categories[gid]
            if focal not in (row["homeId"], row["awayId"]):
                raise ValueError("frozen removal focal identity mismatch")
            opponent = row["awayId"] if focal == row["homeId"] else row["homeId"]
        result.append(
            Perturbation(
                sid, gid, "removal", focal, opponent, profile, None, dict(row), None
            )
        )
    if len(result) != 13 or len(result) * len(MARGIN_WEIGHTS) != 65:
        raise ValueError("frozen diagnostic coverage changed")
    return result


def perturb(network: Network, job: Perturbation) -> Network:
    """Change precisely one score pair or remove one observation, including rematches."""
    originals = [r for r in network.rows if r["id"] == job.game_id]
    games = [g for g in network.games if g.game_id == job.game_id]
    if (
        network.snapshot_id != job.snapshot_id
        or originals != [job.original_observation]
        or len(games) != 1
    ):
        raise ValueError("perturbation source observation mismatch")
    if job.kind == "replacement":
        row = job.replacement_observation
        if row is None or any(
            row[k] != originals[0][k]
            for k in row
            if k not in ("homePoints", "awayPoints")
        ):
            raise ValueError("replacement changed non-score evidence")
        expected = dict(originals[0])
        home = job.focal_id == expected["homeId"]
        op = max(int(expected["awayPoints" if home else "homePoints"]), 3)
        expected.update(
            homePoints=str(op + job.margin if home else op),
            awayPoints=str(op if home else op + job.margin),
        )
        if row != expected:
            raise ValueError("replacement violates frozen score rule")
        changed = replace(
            games[0],
            home_points=int(row["homePoints"]),
            away_points=int(row["awayPoints"]),
        )
        return replace(
            network,
            games=[changed if g.game_id == job.game_id else g for g in network.games],
            rows=[
                dict(row) if r["id"] == job.game_id else dict(r) for r in network.rows
            ],
        )
    if (
        job.kind != "removal"
        or job.replacement_observation is not None
        or job.margin is not None
    ):
        raise ValueError("invalid removal specification")
    return replace(
        network,
        games=[g for g in network.games if g.game_id != job.game_id],
        rows=[dict(r) for r in network.rows if r["id"] != job.game_id],
    )


def provenance(root: Path, baseline: Path, verified: VerifiedExecution) -> dict:
    value = {
        "source_sha256": {p: sha256(root / p) for p in SOURCES},
        "input_sha256": verified.origin["input_sha256"],
        "baseline": {
            "directory": str(baseline.relative_to(root)),
            "fingerprint": verified.origin["fingerprint"],
            "execution_revision": verified.origin["execution_revision"],
            "manifest_sha256": sha256(baseline / "manifest.json"),
            "provenance_sha256": sha256(baseline / "provenance.json"),
            "historical": verified.historical,
        },
        "configuration": CONFIG,
        "resources": {
            "wall_budget_seconds": 10800,
            "rss_limit_bytes": 4 * 1024**3,
            "solve_timeout_seconds": 600,
            "serial": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "scope": "40 frozen score replacements + 25 single-observation removals",
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {**value, "fingerprint": _sha256_json(value), "execution_revision": revision}


def cell_identity(
    network: Network, job: Perturbation, weight: float, origin: dict
) -> dict:
    return {
        "execution_fingerprint": origin["fingerprint"],
        "source_snapshot": network.specification,
        "perturbation": asdict(job),
        "margin_weight": weight,
        "configuration": CONFIG,
        "cold_start": True,
        "input_rows_sha256": included_game_rows_sha256(network.rows),
        "game_objects_sha256": _sha256_json([asdict(g) for g in network.games]),
        "prior_pmfs_sha256": posterior_pmfs_sha256(
            {t.team_id: t.prior for t in network.teams}
        ),
        "team_population": [
            {"id": t.team_id, "subdivision": t.subdivision, "support": len(t.prior)}
            for t in network.teams
        ],
        "game_population": [g.game_id for g in network.games],
    }


def validate_cell(path: Path, network: Network, identity: dict, origin: dict) -> dict:
    cell = read_checkpoint(path, _sha256_json(identity), network)
    if (
        cell.get("identity") != identity
        or cell["execution_revision"] != origin["execution_revision"]
        or cell["snapshot_id"] != network.snapshot_id
        or cell["kind"] != identity["perturbation"]["kind"]
        or cell["margin_weight"] != identity["margin_weight"]
    ):
        raise ValueError("diagnostic checkpoint identity mismatch")
    diagnostics = cell["diagnostics"]
    pairs = {tuple(sorted((g.home_id, g.away_id))) for g in network.games}
    if diagnostics["raw_game_factor_count"] != len(network.games) or diagnostics[
        "unique_pair_factor_count"
    ] != len(pairs):
        raise ValueError("diagnostic factor evidence mismatch")
    return cell


def checkpoint_path(output: Path, job: Perturbation, weight: float) -> Path:
    return output / "checkpoints" / f"{job.key}-{weight:g}.json.gz"


def endpoint_gates(jobs: list[Perturbation], cells: dict, network: Network) -> dict:
    gates = {}
    for gid in sorted({j.game_id for j in jobs if j.kind == "replacement"}):
        wins = [
            j
            for j in jobs
            if j.game_id == gid and j.kind == "replacement" and j.margin > 0
        ]
        for other in wins[1:]:
            a, b = (cells[j.key, 0.0] for j in (wins[0], other))
            gate = parity(
                {t: np.array(p) for t, p in a["pmfs"].items()},
                {t: np.array(p) for t, p in b["pmfs"].items()},
                network.counts,
            )
            # Check the entire Context population as well as stripped FBS PMFs.
            difference = max(
                float(np.max(np.abs(np.array(a["context"][t]) - b["context"][t])))
                for t in a["context"]
            )
            gate["max_context_pmf_difference"] = difference
            gate["passed"] = gate["passed"] and difference <= 1e-8
            gates[f"{gid}/+3-vs-{other.margin:+d}"] = gate
    return gates


def verify_cells(
    root: Path, output: Path, verified: VerifiedExecution
) -> tuple[dict, list[Perturbation], dict]:
    baseline = root / BASELINE_OUTPUT
    origin = _read_json(output / "provenance.json")
    if origin["fingerprint"] != provenance(root, baseline, verified)["fingerprint"]:
        raise ValueError("diagnostic execution input/configuration/source mismatch")
    if (
        _sha256_json(
            {
                k: v
                for k, v in origin.items()
                if k not in ("fingerprint", "execution_revision")
            }
        )
        != origin["fingerprint"]
    ):
        raise ValueError("diagnostic provenance integrity mismatch")
    jobs, cells = inventory(verified), {}
    for job in jobs:
        n = perturb(verified.networks[job.snapshot_id], job)
        for weight in MARGIN_WEIGHTS:
            identity = cell_identity(n, job, weight, origin)
            cells[job.key, weight] = validate_cell(
                checkpoint_path(output, job, weight), n, identity, origin
            )
    execution = _read_json(output / "execution.json")
    if (
        not execution["complete"]
        or execution["failures"]
        or execution["completed_cells"] != 65
    ):
        raise ValueError("diagnostic execution is incomplete")
    gates = endpoint_gates(jobs, cells, verified.networks[jobs[0].snapshot_id])
    if len(gates) != 4 or not all(g["passed"] for g in gates.values()):
        raise ValueError("winning-margin endpoint gate failed")
    if gates != _read_json(output / "endpoint_parity.json"):
        raise ValueError("endpoint gate inventory mismatch")
    return origin, jobs, cells


def execute(root: Path, output: Path, verified: VerifiedExecution) -> dict:
    started = time.monotonic()
    origin = provenance(root, root / BASELINE_OUTPUT, verified)
    if (output / "provenance.json").exists():
        saved = _read_json(output / "provenance.json")
        if saved["fingerprint"] != origin["fingerprint"]:
            raise ValueError("existing diagnostic input/configuration/source mismatch")
        origin = saved
    else:
        write_json(output / "provenance.json", origin)
    old = (
        _read_json(output / "execution.json")
        if (output / "execution.json").exists()
        else {}
    )
    if old.get("complete"):
        verify_cells(root, output, verified)
        return old
    elapsed_before = old.get("elapsed_seconds", 0.0)
    deadline = started + 10800 - elapsed_before
    jobs = inventory(verified)
    network = verified.networks[jobs[0].snapshot_id]
    surfaces, cells, entries, failures = {}, {}, [], []
    state = {
        "complete": False,
        "expected_cells": 65,
        "completed_cells": 0,
        "cells": entries,
        "failures": failures,
    }
    try:
        for job in jobs:
            n = perturb(network, job)
            for weight in MARGIN_WEIGHTS:
                path = checkpoint_path(output, job, weight)
                identity = cell_identity(n, job, weight, origin)
                fingerprint = _sha256_json(identity)
                cell_started = time.monotonic()
                print(f"{job.key}/{weight:g}: start", flush=True)
                try:
                    if path.exists():
                        cell = validate_cell(path, n, identity, origin)
                    else:
                        cell = solve(
                            n,
                            job.kind,
                            weight,
                            fingerprint,
                            surfaces,
                            deadline,
                            origin["execution_revision"],
                        )
                        cell["identity"] = identity
                        write_checkpoint(path, cell)
                        cell = validate_cell(path, n, identity, origin)
                except Exception as error:
                    failure = {
                        "job": asdict(job),
                        "margin_weight": weight,
                        "fingerprint": fingerprint,
                        "runtime_seconds": time.monotonic() - cell_started,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    failures.append(failure)
                    attempt = len(list((output / "failures").glob("*.json"))) + 1
                    write_json(
                        output / "failures" / f"attempt-{attempt:04d}.json", failure
                    )
                    raise
                cells[job.key, weight] = cell
                entries.append(
                    {
                        "perturbation_key": job.key,
                        "margin_weight": weight,
                        "checkpoint": str(path.relative_to(output)),
                        "fingerprint": fingerprint,
                        **cell["diagnostics"],
                    }
                )
                state.update(
                    completed_cells=len(entries),
                    elapsed_seconds=elapsed_before + time.monotonic() - started,
                )
                write_json(output / "execution.json", state)
                print(
                    f"{job.key}/{weight:g}: converged in {cell['diagnostics']['iterations']} iterations",
                    flush=True,
                )
        gates = endpoint_gates(jobs, cells, network)
        write_json(output / "endpoint_parity.json", gates)
        if not all(g["passed"] for g in gates.values()):
            raise ValueError("winning-margin endpoint gate failed")
        state["complete"] = len(entries) == 65
    except Exception as error:  # noqa: BLE001 - fail closed, retain earlier successes
        if not failures:
            failures.append({"error": f"{type(error).__name__}: {error}"})
    finally:
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        write_json(output / "execution.json", state)
        _GAME_FACTOR_CACHE.clear()
        surfaces.clear()
        gc.collect()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / OUTPUT).resolve()
    protected = (
        root / BASELINE_OUTPUT,
        root / ARTIFACT_DIR,
        root / "site",
        root / "data/processed",
    )
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in protected
    ):
        raise ValueError("diagnostic output overlaps protected artifacts")
    started = time.monotonic()
    with resource_guard(started + 10800, timeout=10800):
        verified = verify_completed_execution(root, root / BASELINE_OUTPUT)
        if args.verify_only:
            verify_cells(root, output, verified)
            manifest = _read_json(output / "manifest.json")
            for path, expected in manifest["output_sha256"].items():
                if sha256(output / path) != expected:
                    raise ValueError(f"diagnostic output integrity mismatch: {path}")
            print(
                "Verified frozen design, immutable baselines, all 65 diagnostics and output hashes."
            )
            return
        if not args.analyze_only:
            result = execute(root, output, verified)
            if not result["complete"]:
                raise SystemExit(
                    "Incomplete diagnostics; failures and successful cells preserved"
                )
        from gippyrank.research.issue_267_analysis import analyze

        analyze(root, output, verified)


if __name__ == "__main__":
    main()
