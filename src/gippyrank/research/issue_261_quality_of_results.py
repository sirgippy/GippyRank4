"""Execute the frozen issue 258 study with the issue 252 research machinery.

No production inference or model parameters are changed. Trial checkpoints are
atomic, fingerprinted, and reusable only with identical inputs and source code.

The committed manifest describes the original published experiment. New runs
use their actual source hashes and record their revision when smoke checkpoints
are created. Aggregating historical checkpoints requires their original source
and environment; an existing manifest never overrides checkpoint validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy

from gippyrank.posterior.engine import _GAME_FACTOR_CACHE, Team
from gippyrank.posterior.snapshots import load_pinned_likelihood
from gippyrank.research.issue_252_quality_of_results import (
    MARGIN_WEIGHTS,
    SITE_SLOTS,
    OpponentSlot,
    Trial,
    _infer_trial,
    _mean_ci,
    _write_csv,
    _write_gzip_csv,
    expected_focal_win_probability,
    generate_trial,
    sha256_file,
    summarize_pairs,
)

PROFILES = ("elite", "strong", "middle", "weak")
PANELS = ("unconditional", "primary_conditioned", "control_conditioned")
# Reference to the original published experiment, not a reproduction's revision.
PINNED_EXECUTION_REVISION = "ec422581b537ce3d83190794485b8ee61ba09fb2"
SOURCE_FILES = (
    "src/gippyrank/research/issue_261_quality_of_results.py",
    "src/gippyrank/research/issue_252_quality_of_results.py",
    "src/gippyrank/research/quality_of_results.py",
    "src/gippyrank/posterior/engine.py",
    "src/gippyrank/performance_snapshot.py",
    "src/gippyrank/quality_of_record.py",
    "uv.lock",
    ".python-version",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _pmfs(path: Path) -> dict[tuple[str, str], Team]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["profile"], row["team_id"]].append(row)
    teams = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in rows] != list(range(1, 139)):
            raise ValueError(f"{path}: incomplete rank support for {key}")
        teams[key] = Team(
            key[1],
            rows[0]["team_name"],
            "fbs",
            np.array([float(row["probability"]) for row in rows]),
        )
    return teams


@dataclass(frozen=True)
class Design:
    directory: Path
    manifest: dict
    focal: dict[str, Team]
    schedules: dict[str, dict[str, tuple[OpponentSlot, ...]]]
    records: dict[str, tuple[int, int]]
    provenance: dict
    source_revision: str


def load_design(root: Path) -> Design:
    """Verify every committed design hash before reading any experiment input."""
    directory = root / "data/research/issue_258_quality_of_results"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {name}")
    extra = {
        "src/gippyrank/research/issue_258_quality_of_results_design.py": manifest[
            "design_code_sha256"
        ],
        manifest["documentation_path"]: manifest["documentation_sha256"],
        "docs/issue_252_quality_of_results_protocol.md": manifest[
            "issue_252_protocol_sha256"
        ],
    }
    for name, expected in extra.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"frozen contract hash mismatch: {name}")
    plan = manifest["follow_up_plan"]
    if tuple(plan["margin_weights"]) != MARGIN_WEIGHTS:
        raise ValueError("candidate weights differ from the frozen contract")
    focal_teams = _pmfs(directory / "focal_prior_pmfs.csv.gz")
    opponents = _pmfs(directory / "opponent_prior_pmfs.csv.gz")
    focal = {profile: team for (profile, _), team in focal_teams.items()}
    likelihood = load_pinned_likelihood(directory / "historical_likelihood_v1.json")
    schedule_rows = read_csv(directory / "schedule_games.csv")
    schedules = {}
    for profile in PROFILES:
        schedules[profile] = {}
        for arm, role in (
            ("lower_win_probability", "approximately_even"),
            ("higher_win_probability", "weaker_opponents"),
        ):
            rows = sorted(
                [
                    row
                    for row in schedule_rows
                    if row["profile"] == profile and row["schedule_arm"] == arm
                ],
                key=lambda row: int(row["game_number"]),
            )
            if [row["venue"] for row in rows] != list(SITE_SLOTS) or len(
                {row["opponent_team_id"] for row in rows}
            ) != 10:
                raise ValueError(f"invalid frozen schedule: {profile}/{arm}")
            slots = []
            for row in rows:
                opponent = opponents[profile, row["opponent_team_id"]]
                probability = float(row["v1_expected_focal_win_probability"])
                actual = expected_focal_win_probability(
                    focal[profile].prior, opponent.prior, row["venue"], likelihood
                )
                if not np.isclose(actual, probability, rtol=0, atol=1e-12):
                    raise ValueError("schedule probability disagrees with frozen PMFs")
                slots.append(
                    OpponentSlot(
                        opponent,
                        row["venue"],
                        probability,
                        float(row["target_probability"]),
                    )
                )
            schedules[profile][role] = tuple(slots)
    records = {
        row["profile"]: (
            int(row["lower_schedule_target_wins"]),
            int(row["higher_schedule_target_wins"]),
        )
        for row in read_csv(directory / "generation_pilot.csv")
        if row["profile"] in PROFILES
    }
    provenance = {
        "frozen_manifest_path": str(directory.relative_to(root) / "manifest.json"),
        "frozen_manifest_sha256": sha256_file(directory / "manifest.json"),
        "frozen_artifact_sha256": manifest["artifact_sha256"],
        "frozen_contract_sha256": extra,
        "source_sha256": {name: sha256_file(root / name) for name in SOURCE_FILES},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return Design(
        directory, manifest, focal, schedules, records, provenance, source_revision
    )


def fingerprint(design: Design) -> str:
    return hashlib.sha256(
        json.dumps(design.provenance, sort_keys=True).encode()
    ).hexdigest()


def select_seeds(design: Design, profile: str, panel: str) -> dict:
    """Search only the frozen seed interval, before inspecting candidate ranks."""
    plan = design.manifest["follow_up_plan"]
    if panel == "unconditional":
        start, stop = plan["unconditional_shared_seed_range"]
        return {
            "seeds": list(range(start, stop + 1)),
            "attempts": stop - start + 1,
            "target": plan["unconditional_pairs_per_profile"],
            "seed_range": [start, stop],
            "complete": True,
        }
    key = (
        "primary_record_conditioned_pairs_per_profile"
        if panel == "primary_conditioned"
        else "self_record_conditioned_control_pairs_per_profile"
    )
    target = plan[key]
    ranges = (
        "primary_conditioned_seed_ranges"
        if panel == "primary_conditioned"
        else "control_conditioned_seed_ranges"
    )
    start, stop = plan[ranges][profile]
    stop = min(stop, start + plan["conditioning_search_ceiling_per_panel"] - 1)
    likelihood = load_pinned_likelihood(
        design.directory / "historical_likelihood_v1.json"
    )
    seeds = []
    lower, higher = design.records[profile]
    for seed in range(start, stop + 1):
        trial = generate_trial(
            seed, design.focal[profile].prior, design.schedules[profile], likelihood
        )
        if panel == "primary_conditioned":
            accepted = (
                trial.groups["approximately_even"].wins == lower
                and trial.groups["weaker_opponents"].wins == higher
            )
        else:
            accepted = (
                trial.groups["same_schedule_a"].wins == lower
                and trial.groups["same_schedule_b"].wins == lower
            )
        if accepted:
            seeds.append(seed)
        if len(seeds) == target:
            break
    return {
        "seeds": seeds,
        "attempts": seed - start + 1,
        "target": target,
        "seed_range": [start, stop],
        "complete": len(seeds) == target,
    }


def evidence_hash(trial: Trial) -> str:
    rows = [row for group in trial.groups.values() for row in group.game_rows]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def evaluate_trial(design: Design, profile: str, panel: str, seed: int) -> dict:
    likelihood = load_pinned_likelihood(
        design.directory / "historical_likelihood_v1.json"
    )
    trial = generate_trial(
        seed, design.focal[profile].prior, design.schedules[profile], likelihood
    )
    if panel == "control_conditioned":
        trial = replace(
            trial,
            groups={
                name: group
                for name, group in trial.groups.items()
                if name.startswith("same_schedule_")
            },
        )
    evidence_sha256 = evidence_hash(trial)
    # The engine cache is unbounded. Trial-local clearing prevents research runs
    # from retaining factors for thousands of unique continuous margins.
    _GAME_FACTOR_CACHE.clear()
    started = time.perf_counter()
    team_rows, pair_rows, pmfs = _infer_trial(trial, likelihood)
    seconds = time.perf_counter() - started
    _GAME_FACTOR_CACHE.clear()
    if evidence_hash(trial) != evidence_sha256:
        raise RuntimeError("candidate inference mutated trial evidence")
    for weight, groups in pmfs.items():
        for pmf in groups.values():
            if (
                len(pmf) != 138
                or not np.isfinite(pmf).all()
                or np.any(pmf < 0)
                or not np.isclose(pmf.sum(), 1, atol=1e-12)
            ):
                raise RuntimeError(f"invalid focal posterior at weight {weight}")
    metadata = {
        "profile": profile,
        "panel": panel,
        "focal_true_rank": trial.focal_rank,
        "evidence_sha256": evidence_sha256,
    }
    for row in team_rows:
        group = trial.groups[str(row["focal_group"])]
        margins = [float(game["synthetic_focal_margin"]) for game in group.game_rows]
        row.update(metadata)
        row["mean_focal_margin"] = float(np.mean(margins))
        row["mean_absolute_focal_margin"] = float(np.mean(np.abs(margins)))
    for row in pair_rows:
        row.update(metadata)
        row["comparison_population"] = (
            "primary_seed_control"
            if panel == "primary_conditioned"
            and row["comparison"] == "same_schedule_control"
            else panel
        )
    return {
        "fingerprint": fingerprint(design),
        "seed": seed,
        "inference_seconds": seconds,
        "teams": team_rows,
        "pairs": pair_rows,
    }


def checkpoint_path(output: Path, profile: str, panel: str, seed: int) -> Path:
    return output / ".checkpoints" / profile / panel / f"{seed}.json"


def load_checkpoint(path: Path, expected_fingerprint: str, seed: int) -> dict:
    value = json.loads(path.read_text())
    if value["fingerprint"] != expected_fingerprint or value["seed"] != seed:
        raise ValueError(f"checkpoint provenance differs: {path}")
    return value


def run_profile(design: Design, output: Path, profile: str) -> None:
    expected_fingerprint = fingerprint(design)
    selections = {panel: select_seeds(design, profile, panel) for panel in PANELS}
    selection_path = output / ".checkpoints" / profile / "selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        if existing != {"fingerprint": expected_fingerprint, "panels": selections}:
            raise ValueError("checkpoint selection provenance differs")
    write_json(
        selection_path, {"fingerprint": expected_fingerprint, "panels": selections}
    )
    if not all(selection["complete"] for selection in selections.values()):
        raise RuntimeError(
            f"{profile}: frozen conditioning search exhausted; see {selection_path}"
        )
    print(
        f"{profile}: seed search complete "
        + str(
            {panel: selection["attempts"] for panel, selection in selections.items()}
        ),
        flush=True,
    )
    for panel, selection in selections.items():
        for index, seed in enumerate(selection["seeds"], start=1):
            path = checkpoint_path(output, profile, panel, seed)
            if path.exists():
                load_checkpoint(path, expected_fingerprint, seed)
            else:
                write_json(path, evaluate_trial(design, profile, panel, seed))
            if index % 25 == 0:
                print(
                    f"{profile}/{panel}: {index}/{len(selection['seeds'])}", flush=True
                )


def smoke(design: Design, output: Path) -> dict:
    """Exercise all four profiles and all candidates without changing budgets."""
    rows = []
    for profile in PROFILES:
        seed = design.manifest["follow_up_plan"]["unconditional_shared_seed_range"][0]
        path = checkpoint_path(output, profile, "unconditional", seed)
        value = (
            load_checkpoint(path, fingerprint(design), seed)
            if path.exists()
            else evaluate_trial(design, profile, "unconditional", seed)
        )
        write_json(path, value)
        rows.append(
            {
                "profile": profile,
                "seed": seed,
                "inference_seconds": value["inference_seconds"],
                "team_candidate_count": len(value["teams"]),
                "pair_candidate_count": len(value["pairs"]),
            }
        )
    seconds = sum(row["inference_seconds"] for row in rows) / len(rows)
    result = {
        "fingerprint": fingerprint(design),
        "source_revision": design.source_revision,
        "trials": rows,
        "estimated_serial_inference_minutes": seconds * 3200 / 60,
        "all_candidates_share_immutable_evidence": True,
        "all_focal_pmfs_valid": True,
    }
    write_json(output / ".checkpoints/smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def aggregate(design: Design, output: Path) -> None:
    """Regenerate portable results from completed, provenance-checked trials."""
    teams, pairs, games, selections, outcome_rows = [], [], [], {}, []
    inference_seconds = 0.0
    for profile in PROFILES:
        saved = json.loads(
            (output / ".checkpoints" / profile / "selection.json").read_text()
        )
        if saved["fingerprint"] != fingerprint(design):
            raise ValueError("selection fingerprint differs")
        selections[profile] = saved["panels"]
        likelihood = load_pinned_likelihood(
            design.directory / "historical_likelihood_v1.json"
        )
        for panel, selection in saved["panels"].items():
            if not selection["complete"]:
                raise ValueError("incomplete conditioning panel")
            for seed in selection["seeds"]:
                value = load_checkpoint(
                    checkpoint_path(output, profile, panel, seed),
                    fingerprint(design),
                    seed,
                )
                teams.extend(value["teams"])
                pairs.extend(value["pairs"])
                inference_seconds += value["inference_seconds"]
                trial = generate_trial(
                    seed,
                    design.focal[profile].prior,
                    design.schedules[profile],
                    likelihood,
                )
                if panel == "control_conditioned":
                    trial = replace(
                        trial,
                        groups={
                            name: group
                            for name, group in trial.groups.items()
                            if name.startswith("same_schedule_")
                        },
                    )
                if any(
                    row["evidence_sha256"] != evidence_hash(trial)
                    for row in value["teams"]
                ):
                    raise ValueError(
                        "checkpoint evidence differs from regenerated evidence"
                    )
                for group in trial.groups.values():
                    games.extend(
                        {"profile": profile, "panel": panel, **row}
                        for row in group.game_rows
                    )
    summaries = []
    for profile in PROFILES:
        schedule = design.schedules[profile]
        lower = float(
            np.mean(
                [
                    slot.expected_win_probability
                    for slot in schedule["approximately_even"]
                ]
            )
        )
        higher = float(
            np.mean(
                [slot.expected_win_probability for slot in schedule["weaker_opponents"]]
            )
        )
        for panel in PANELS:
            subset = [
                row
                for row in pairs
                if row["profile"] == profile and row["panel"] == panel
            ]
            for row in summarize_pairs(subset, panel):
                deltas = [
                    float(pair["rank_difference"])
                    for pair in subset
                    if pair["comparison"] == row["comparison"]
                    and pair["margin_weight"] == row["margin_weight"]
                ]
                row.update(
                    {
                        "profile": profile,
                        "lower_expected_win_probability": lower,
                        "higher_expected_win_probability": higher,
                        "achieved_win_probability_separation": higher - lower,
                        "rank_difference_sd": float(np.std(deltas, ddof=1)),
                        "comparison_population": "primary_seed_control"
                        if panel == "primary_conditioned"
                        and row["comparison"] == "same_schedule_control"
                        else panel,
                    }
                )
                summaries.append(row)
    grouped = defaultdict(list)
    for row in teams:
        grouped[
            row["profile"], row["panel"], row["focal_group"], row["margin_weight"]
        ].append(row)
    for (profile, panel, group, weight), rows in sorted(grouped.items()):
        summary = {
            "profile": profile,
            "panel": panel,
            "focal_group": group,
            "margin_weight": weight,
            "draw_count": len(rows),
        }
        for field in (
            "wins",
            "focal_true_rank",
            "mean_focal_margin",
            "mean_absolute_focal_margin",
            "expected_rank",
            "rank_sd",
            "entropy_bits",
            "interval_80_width",
        ):
            values = [float(row[field]) for row in rows]
            mean, low, high = _mean_ci(values)
            summary.update(
                {
                    f"{field}_mean": mean,
                    f"{field}_mean_ci95_low": low,
                    f"{field}_mean_ci95_high": high,
                    f"{field}_p05": float(np.quantile(values, 0.05)),
                    f"{field}_p50": float(np.quantile(values, 0.5)),
                    f"{field}_p95": float(np.quantile(values, 0.95)),
                }
            )
        summary["record_counts"] = json.dumps(
            dict(sorted(Counter(str(row["wins"]) for row in rows).items()))
        )
        outcome_rows.append(summary)
    _write_gzip_csv(output / "team_trial_summaries.csv.gz", teams)
    _write_gzip_csv(output / "pair_trial_summaries.csv.gz", pairs)
    _write_gzip_csv(output / "simulated_games.csv.gz", games)
    _write_csv(output / "simulation_summary.csv", summaries)
    _write_csv(output / "arm_summary.csv", outcome_rows)
    smoke_result = json.loads((output / ".checkpoints/smoke.json").read_text())
    if smoke_result["fingerprint"] != fingerprint(design):
        raise ValueError("smoke fingerprint differs")
    manifest = {
        "issue": 261,
        "schema_version": "1.0",
        "study": "research_only_multistrength_quality_of_results_schedule_equivalence",
        "source_revision": smoke_result["source_revision"],
        "published_execution_revision": PINNED_EXECUTION_REVISION,
        "provenance": design.provenance,
        "fingerprint": fingerprint(design),
        "frozen_follow_up_plan": design.manifest["follow_up_plan"],
        "record_targets": design.records,
        "candidate_margin_weights": MARGIN_WEIGHTS,
        "inference": {
            "implementation": "issue242_quality_of_results_v1",
            "trial_runner": "issue252._infer_trial",
            "max_iterations": 500,
            "tolerance": 1e-9,
            "damping": 0.35,
            "rank_support": 138,
            "prior_stripping": "normalize(context_posterior / context_prior)",
            "convergence_required": True,
            "all_candidates_share_evidence": True,
            "network": "four disconnected stars; two for independently conditioned controls",
        },
        "generation": {
            "implementation": "issue252.generate_trial",
            "random_generator": "numpy.random.default_rng / PCG64",
            "margin_draws": "Generator.standard_t; shared primary draws; independent control draws",
            "wins": "unmodified continuous margin sign",
            "matched": "shared focal rank and opponent quantiles across all four arms",
        },
        "completed_panels": selections,
        "smoke": smoke_result,
        "total_inference_seconds": inference_seconds,
        "counts": {
            "team_candidate_rows": len(teams),
            "pair_candidate_rows": len(pairs),
            "simulated_game_rows": len(games),
        },
        "deviations": [],
        "execution_limitations": [],
        "reference_profile": "Colorado retained in frozen design; no fifth simulation stratum",
        "artifact_sha256": {
            name: sha256_file(output / name)
            for name in (
                "team_trial_summaries.csv.gz",
                "pair_trial_summaries.csv.gz",
                "simulated_games.csv.gz",
                "simulation_summary.csv",
                "arm_summary.csv",
            )
        },
        "execution_environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.root / "data/research/issue_261_quality_of_results"
    design = load_design(args.root)
    if args.smoke:
        smoke(design, output)
    elif args.aggregate:
        aggregate(design, output)
    elif args.profile:
        run_profile(design, output, args.profile)
    else:
        parser.error("specify --smoke, --profile, or --aggregate")


if __name__ == "__main__":
    main()
