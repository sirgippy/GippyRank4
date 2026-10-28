"""Matched three-candidate analysis of the bounded issue 269 extension."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from gippyrank.posterior.snapshots import sha256
from gippyrank.research.issue_252_quality_of_results import (
    _infer_trial,
    _mean_ci,
    summarize_pairs,
)
from gippyrank.research.issue_261_quality_of_results import PANELS, PROFILES, read_csv
from gippyrank.research.issue_265_population import (
    compare,
    contrast_summary,
    distribution,
    order,
    summary,
    top_changes,
    write_csv,
)
from gippyrank.research.issue_265_real_seasons import (
    parity,
    solve,
    write_checkpoint,
    write_json,
)
from gippyrank.research.issue_269_candidate import (
    BASELINE,
    DIAGNOSTICS,
    NEIGHBORS,
    OUTPUT,
    WEIGHT,
    authenticate,
    network_identity,
    network_jobs,
    origin_for,
    read_hashed,
    references,
    run_252,
    run_networks,
    run_synthetic,
    synthetic_trials,
    validate_network,
)


def numeric(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if key.endswith("_id") or key in (
            "profile",
            "panel",
            "focal_group",
            "comparison",
            "left_group",
            "right_group",
            "comparison_population",
            "evidence_sha256",
        ):
            result[key] = value
        elif value in ("True", "False"):
            result[key] = value == "True"
        else:
            try:
                result[key] = float(value)
            except (ValueError, TypeError):
                result[key] = value
    return result


def matched_rows(rows: list[dict]) -> dict:
    """Reject omissions/duplicates before calculating within-seed contrasts."""
    indexed = {w: {} for w in NEIGHBORS}
    for row in rows:
        weight, seed = row["margin_weight"], int(row["seed"])
        if weight not in indexed or seed in indexed[weight]:
            raise ValueError("duplicate or unexpected synthetic candidate observation")
        indexed[weight][seed] = row
    if not indexed[0.25] or not all(
        set(v) == set(indexed[0.25]) for v in indexed.values()
    ):
        raise ValueError("unmatched synthetic candidate memberships")
    return indexed


def repeat_checks(root: Path, output: Path, verified, origin: dict, *, execute=False):
    """One new-weight order check and one independent synthetic repeat, off CI."""
    sid = verified.specification["game_diagnostics"]["publication_slot"]
    network = verified.networks[sid]
    baseline = read_hashed(output / "networks" / f"baseline-{sid}.json.gz")
    identity = {**network_identity(network, origin), "kind": "reversed"}
    from gippyrank.research.issue_242_quality_of_results import _sha256_json

    path = output / "validation" / "week9-reversed.json.gz"
    if not path.exists() and execute:
        cell = solve(
            network,
            "reversed",
            WEIGHT,
            _sha256_json(identity),
            {},
            time.monotonic() + 600,
            origin["execution_revision"],
        )
        cell["identity"] = identity
        write_checkpoint(path, cell)
    cell = validate_network(path, network, identity, origin)
    gate = parity(
        {t: np.array(p) for t, p in cell["pmfs"].items()},
        {t: np.array(p) for t, p in baseline["pmfs"].items()},
        network.counts,
    )
    gate["max_context_difference"] = max(
        float(np.max(np.abs(np.array(p) - baseline["context"][t])))
        for t, p in cell["context"].items()
    )
    if not gate["passed"] or gate["max_context_difference"] > 1e-8:
        raise ValueError("new-weight input-order gate failed")
    panel, seed, digest, trial, likelihood = next(synthetic_trials(root, "elite"))
    archive = output / "synthetic" / "elite.json.gz"
    original = (
        read_hashed(archive)[f"{panel}/{seed}"]
        if archive.exists()
        else read_hashed(
            output / ".checkpoints/synthetic/elite" / panel / f"{seed}.json.gz"
        )
    )
    path = output / "validation" / "elite-repeat.json.gz"
    if not path.exists() and execute:
        before = time.monotonic()
        teams, _, pmfs = _infer_trial(trial, likelihood, margin_weights=(WEIGHT,))
        write_checkpoint(
            path,
            {
                "identity": original["identity"],
                "evidence_sha256": digest,
                "execution_revision": origin["execution_revision"],
                "pmfs": {t: p.tolist() for t, p in pmfs[WEIGHT].items()},
                "convergence": teams,
                "runtime_seconds": time.monotonic() - before,
            },
        )
    repeated = read_hashed(path)
    if (
        repeated["identity"] != original["identity"]
        or repeated["execution_revision"] != origin["execution_revision"]
        or repeated["evidence_sha256"] != digest
        or set(repeated["pmfs"]) != set(original["pmfs"])
    ):
        raise ValueError("synthetic repeat identity mismatch")
    error = max(
        float(np.max(np.abs(np.array(p) - original["pmfs"][t])))
        for t, p in repeated["pmfs"].items()
    )
    if error != 0 or any(
        not r["converged"] or r["max_message_delta"] > 1e-9
        for r in repeated["convergence"]
    ):
        raise ValueError("synthetic repeat gate failed")
    result = {
        "order_gate": gate,
        "synthetic_repeat_max_pmf_difference": error,
        "synthetic_repeat_bit_identical": True,
        "passed": True,
    }
    if execute:
        write_json(output / "repeat_validation.json", result)
    elif result != json.loads((output / "repeat_validation.json").read_text()):
        raise ValueError("repeat validation summary mismatch")
    return result


def synthetic(root: Path, output: Path, origin: dict) -> list:
    old = root / "data/research/issue_261_quality_of_results"
    teams, pairs = [], []
    for target, filename in (
        (teams, "team_trial_summaries.csv.gz"),
        (pairs, "pair_trial_summaries.csv.gz"),
    ):
        target.extend(
            numeric(r)
            for r in read_csv(old / filename)
            if float(r["margin_weight"]) in (0.25, 0.5)
        )
    for profile in PROFILES:
        run_synthetic(root, output, origin, profile, verify_only=True)
        archive = read_hashed(output / "synthetic" / f"{profile}.json.gz")
        for cell in archive.values():
            teams.extend(cell["teams"])
            pairs.extend(cell["pairs"])
    summaries, contrasts, arms = [], [], []
    for profile in PROFILES:
        for panel in PANELS:
            subset = [
                r for r in pairs if r["profile"] == profile and r["panel"] == panel
            ]
            for row in summarize_pairs(subset, panel, margin_weights=NEIGHBORS):
                values = [
                    r["rank_difference"]
                    for r in subset
                    if r["comparison"] == row["comparison"]
                    and r["margin_weight"] == row["margin_weight"]
                ]
                row.update(
                    profile=profile,
                    rank_difference_sd=float(np.std(values, ddof=1)),
                    comparison_population=(
                        "primary_seed_control"
                        if panel == "primary_conditioned"
                        and row["comparison"] == "same_schedule_control"
                        else panel
                    ),
                )
                summaries.append(row)
            for comparison in sorted({r["comparison"] for r in subset}):
                by_weight = matched_rows(
                    [r for r in subset if r["comparison"] == comparison]
                )
                for low, high in combinations(NEIGHBORS, 2):
                    row = {
                        "profile": profile,
                        "panel": panel,
                        "comparison": comparison,
                        "low_weight": low,
                        "high_weight": high,
                        "n": len(by_weight[low]),
                    }
                    for field in (
                        "rank_difference",
                        "rank_sd_difference",
                        "entropy_difference_bits",
                        "interval_80_width_difference",
                    ):
                        values = [
                            by_weight[high][s][field] - by_weight[low][s][field]
                            for s in sorted(by_weight[low])
                        ]
                        mean, left, right = _mean_ci(values)
                        row.update(
                            {
                                f"{field}_mean": mean,
                                f"{field}_ci95_low": left,
                                f"{field}_ci95_high": right,
                                f"{field}_sd": float(np.std(values, ddof=1)),
                                f"{field}_p05": float(np.quantile(values, 0.05)),
                                f"{field}_p95": float(np.quantile(values, 0.95)),
                            }
                        )
                    contrasts.append(row)
    grouped = defaultdict(list)
    for row in teams:
        grouped[
            row["profile"], row["panel"], row["focal_group"], row["margin_weight"]
        ].append(row)
    for (profile, panel, group, weight), rows in sorted(grouped.items()):
        row = {
            "profile": profile,
            "panel": panel,
            "focal_group": group,
            "margin_weight": weight,
            "n": len(rows),
        }
        for field in ("expected_rank", "rank_sd", "entropy_bits", "interval_80_width"):
            mean, low, high = _mean_ci([r[field] for r in rows])
            row.update(
                {
                    f"{field}_mean": mean,
                    f"{field}_ci95_low": low,
                    f"{field}_ci95_high": high,
                }
            )
        arms.append(row)
    write_csv(output / "synthetic_pairs.csv.gz", pairs)
    write_csv(output / "synthetic_teams.csv.gz", teams)
    write_csv(output / "schedule_summary.csv", summaries)
    write_csv(output / "schedule_candidate_contrasts.csv", contrasts)
    write_csv(output / "schedule_arms.csv", arms)
    return summaries


def seasons(root: Path, output: Path, verified, origin: dict):
    descriptors = {
        (r["snapshot_id"], r["team_id"]): r
        for r in read_csv(root / BASELINE / "accomplishments.csv")
    }
    cells, pmfs, rankings, by_team = {}, {}, [], {}
    for key, network, job in network_jobs(verified):
        path = output / "networks" / f"{key}.json.gz"
        cell = validate_network(
            path, network, network_identity(network, origin, job), origin
        )
        cells[key] = cell
    populations, changes, exceptional = [], [], []
    unions = {}
    for sid, network in verified.networks.items():
        names = {t.team_id: t.name for t in network.teams}
        ranks = {}
        for w in NEIGHBORS:
            cell = (
                cells[f"baseline-{sid}"]
                if w == WEIGHT
                else verified.cells[sid]["baseline", w]
            )
            p = {t: np.array(v) for t, v in cell["pmfs"].items()}
            pmfs[sid, w] = p
            ranks[w] = order(p, cell["counts"])
        union25 = {t for w in (0.25, 0.5) for t, rank in ranks[w].items() if rank <= 25}
        union18 = {t for w in (0.25, 0.5) for t, rank in ranks[w].items() if rank <= 18}
        unions[sid] = union25
        for w in NEIGHBORS:
            for t, p in pmfs[sid, w].items():
                row = {
                    **descriptors[sid, t],
                    "margin_weight": w,
                    "ordinal_rank": ranks[w].get(t),
                    **summary(p),
                    "neighbor_top25_union": t in union25,
                    "neighbor_top18_union": t in union18,
                }
                rankings.append(row)
                by_team[sid, w, t] = row
        for low, high in combinations(NEIGHBORS, 2):
            subset = []
            for t in sorted(ranks[low]):
                row = {
                    "snapshot_id": sid,
                    "team_id": t,
                    "team_name": names[t],
                    "low_weight": low,
                    "high_weight": high,
                    "before_ordinal": ranks[low][t],
                    "after_ordinal": ranks[high][t],
                    "delta_ordinal_rank": ranks[high][t] - ranks[low][t],
                    "neighbor_top25_union": t in union25,
                    "neighbor_top18_union": t in union18,
                    **compare(pmfs[sid, low][t], pmfs[sid, high][t]),
                }
                subset.append(row)
            changes.extend(subset)
            row = {
                "snapshot_id": sid,
                "low_weight": low,
                "high_weight": high,
                **contrast_summary(subset),
            }
            for boundary in (10, 12, 25):
                row.update(top_changes(ranks[low], ranks[high], boundary, names))
                delta = [r[f"delta_mass_le_{boundary}"] for r in subset]
                row.update(
                    {
                        f"mass_le_{boundary}_{k}": v
                        for k, v in distribution(delta).items()
                    }
                )
            row["largest_expected_rank_changes"] = sorted(
                subset, key=lambda r: -abs(r["delta_expected_rank"])
            )[:10]
            populations.append(row)
        for t in sorted(ranks[0.25]):
            a, b, c = (by_team[sid, w, t] for w in NEIGHBORS)
            outside = {
                field: b[field]
                - np.clip(b[field], min(a[field], c[field]), max(a[field], c[field]))
                for field in (
                    "expected_rank",
                    "rank_sd",
                    "entropy_bits",
                    "mass_le_12",
                    "mass_le_25",
                )
            }
            if any(abs(v) > 1e-8 for v in outside.values()):
                exceptional.append(
                    {
                        "snapshot_id": sid,
                        "team_id": t,
                        "team_name": names[t],
                        "neighbor_top25_union": t in union25,
                        **outside,
                    }
                )
    write_csv(output / "rankings.csv", rankings)
    contender_appendix(output, rankings)
    write_csv(output / "team_contrasts.csv.gz", changes)
    write_json(output / "population_comparisons.json", populations)
    write_csv(output / "nonmonotonic.csv", exceptional)
    # Preserve the entire frozen case inventory and the neighbors' union, not a .40 selection.
    cases = []
    for old in read_csv(root / DIAGNOSTICS / "case_results.csv"):
        if float(old["margin_weight"]) != 0.25:
            continue
        sid, t = old["snapshot_id"], old["team_id"]
        for w in NEIGHBORS:
            cases.append(
                {
                    **by_team[sid, w, t],
                    **{
                        k: old[k]
                        for k in (
                            "frozen_selection",
                            "categories",
                            "questions",
                            "actual_games",
                        )
                    },
                }
            )
    write_csv(output / "cases.csv.gz", cases)
    games = read_csv(root / BASELINE / "actual_game_descriptors.csv.gz")
    write_csv(
        output / "contender_games.csv.gz",
        [r for r in games if r["team_id"] in unions[r["snapshot_id"]]],
    )
    return cells, pmfs, by_team


def contender_appendix(output: Path, rankings: list[dict]) -> None:
    """Every neighbor-selected Top 18 team, including the Top 12 neighborhood."""
    lines = [
        "# Neighbor-selected contender comparison",
        "",
        (
            "Selection is the union of λ=.25/.50 Top 18 teams. Each cell is "
            "ordinal / expected rank / latent Top 12 mass / rank SD. "
            "Mass is not a CFP selection probability. Full Top 25 union flags, "
            "50/80/95% intervals and Top 10/25 masses are in rankings.csv."
        ),
        "",
    ]
    for sid in sorted({r["snapshot_id"] for r in rankings}):
        subset = [
            r for r in rankings if r["snapshot_id"] == sid and r["neighbor_top18_union"]
        ]
        ids = sorted(
            {r["team_id"] for r in subset},
            key=lambda t: min(r["ordinal_rank"] for r in subset if r["team_id"] == t),
        )
        lines.extend(
            [
                f"## {sid}",
                "",
                "| Team | Record | λ=.25 | λ=.40 | λ=.50 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for t in ids:
            rows = {r["margin_weight"]: r for r in subset if r["team_id"] == t}
            cells = [
                f"{rows[w]['ordinal_rank']} / {rows[w]['expected_rank']:.2f} / "
                f"{rows[w]['mass_le_12']:.3f} / {rows[w]['rank_sd']:.2f}"
                for w in NEIGHBORS
            ]
            base = rows[0.25]
            lines.append(
                f"| {base['team_name']} | {base['wins']}–{base['losses']} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    (output / "contenders.md").write_text("\n".join(lines).rstrip() + "\n")


def diagnostics(
    output: Path,
    verified,
    jobs: list,
    old_cells: dict,
    new_cells: dict,
    baseline_pmfs: dict,
) -> list:
    responses, summaries, sensitivities, states = [], [], [], []
    sid = jobs[0].snapshot_id
    network = verified.networks[sid]
    names = {t.team_id: t.name for t in network.teams}
    for job in jobs:
        for w in NEIGHBORS:
            cell = new_cells[job.key] if w == WEIGHT else old_cells[job.key, w]
            p = {t: np.array(v) for t, v in cell["pmfs"].items()}
            ranks = order(p, cell["counts"])
            base = baseline_pmfs[sid, w]
            before = order(base, verified.cells[sid]["baseline", 0.25]["counts"])
            subset = []
            for t in sorted(base):
                row = {
                    "snapshot_id": sid,
                    "key": job.key,
                    "kind": job.kind,
                    "game_id": job.game_id,
                    "margin_weight": w,
                    "team_id": t,
                    "team_name": names[t],
                    "focal": t == job.focal_id,
                    "opponent": t == job.opponent_id,
                    "delta_ordinal_rank": ranks[t] - before[t],
                    **compare(base[t], p[t]),
                }
                subset.append(row)
            responses.extend(subset)
            focal = next(r for r in subset if r["focal"])
            summaries.append(
                {
                    **focal,
                    **{f"after_{k}": v for k, v in summary(p[job.focal_id]).items()},
                    **{
                        f"before_{k}": v for k, v in summary(base[job.focal_id]).items()
                    },
                    "ordinal_before": before[job.focal_id],
                    "ordinal_after": ranks[job.focal_id],
                    "network_mean_absolute_response": float(
                        np.mean([abs(r["delta_expected_rank"]) for r in subset])
                    ),
                    "network_max_absolute_response": max(
                        abs(r["delta_expected_rank"]) for r in subset
                    ),
                }
            )
            if job.kind == "replacement":
                states.append(
                    {
                        "source": "real",
                        "game_id": job.game_id,
                        "team_name": names[job.focal_id],
                        "margin_weight": w,
                        "margin": job.margin,
                        "ordinal_rank": ranks[job.focal_id],
                        **summary(p[job.focal_id]),
                    }
                )
    for gid in sorted({j.game_id for j in jobs if j.kind == "replacement"}):
        for w in NEIGHBORS:
            rows = {
                r["margin"]: r
                for r in states
                if r["game_id"] == gid and r["margin_weight"] == w
            }
            boundary = rows[-3]["expected_rank"] - rows[3]["expected_rank"]
            margin = rows[10]["expected_rank"] - rows[35]["expected_rank"]
            sensitivities.append(
                {
                    "source": "real",
                    "game_id": gid,
                    "team_name": rows[-3]["team_name"],
                    "margin_weight": w,
                    "outcome_boundary_benefit": boundary,
                    "within_win_margin_benefit": margin,
                    "outcome_minus_margin_benefit": boundary - margin,
                }
            )
    write_csv(output / "network_responses.csv.gz", responses)
    write_csv(output / "game_diagnostics.csv", summaries)
    return states, sensitivities


def sensitivity_252(
    root: Path, output: Path, origin: dict, states: list, sensitivities: list
):
    cells = run_252(root, output, origin, verify_only=True)
    retained = defaultdict(list)
    for row in read_csv(
        root / "data/research/issue_252_quality_of_results/counterfactual_pmfs.csv.gz"
    ):
        if (
            row["schedule_role"] in ("approximately_even", "weaker_opponents")
            and float(row["margin_weight"]) in (0.25, 0.5)
            and row["counterfactual_family"]
            in ("win_loss_boundary", "within_win_margin")
        ):
            retained[
                row["schedule_role"],
                float(row["margin_weight"]),
                int(float(row["focal_margin"])),
            ].append(row)
    for (role, w, margin), rows in sorted(retained.items()):
        rows.sort(key=lambda r: int(r["rank"]))
        if [int(r["rank"]) for r in rows] != list(range(1, 139)):
            raise ValueError("incomplete retained issue252 counterfactual support")
        states.append(
            {
                "source": "issue252",
                "game_id": role,
                "team_name": "Colorado synthetic",
                "margin_weight": w,
                "margin": margin,
                **summary(np.array([float(r["probability"]) for r in rows])),
            }
        )
    for cell in cells:
        role = cell["identity"]["role"]
        states.append(
            {
                "source": "issue252",
                "game_id": role,
                "team_name": "Colorado synthetic",
                "margin_weight": WEIGHT,
                "margin": cell["identity"]["margin"],
                **summary(np.array(cell["pmfs"][role])),
            }
        )
    old = read_csv(
        root / "data/research/issue_252_quality_of_results/outcome_sensitivity.csv"
    )
    for role in ("approximately_even", "weaker_opponents"):
        for w in NEIGHBORS:
            if w == WEIGHT:
                rows = {
                    r["margin"]: r
                    for r in states
                    if r["source"] == "issue252"
                    and r["game_id"] == role
                    and r["margin_weight"] == w
                }
                boundary = rows[-3]["expected_rank"] - rows[3]["expected_rank"]
                margin = rows[10]["expected_rank"] - rows[35]["expected_rank"]
            else:
                rows = {
                    r["counterfactual_family"]: r
                    for r in old
                    if r["schedule_role"] == role and float(r["margin_weight"]) == w
                }
                boundary = float(rows["win_loss_boundary"]["rank_benefit_of_b_over_a"])
                margin = float(rows["within_win_margin"]["rank_benefit_of_b_over_a"])
            sensitivities.append(
                {
                    "source": "issue252",
                    "game_id": role,
                    "team_name": "Colorado synthetic",
                    "margin_weight": w,
                    "outcome_boundary_benefit": boundary,
                    "within_win_margin_benefit": margin,
                    "outcome_minus_margin_benefit": boundary - margin,
                }
            )
    write_csv(output / "counterfactual_states.csv", states)
    write_csv(output / "outcome_sensitivity.csv", sensitivities)


def figure(output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(output / "schedule_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = ("#315b8b", "#b45822", "#328268")
    for axis, panel in zip(axes, ("unconditional", "primary_conditioned"), strict=True):
        for i, (w, color) in enumerate(zip(NEIGHBORS, colors, strict=True)):
            subset = [
                next(
                    r
                    for r in rows
                    if r["profile"] == profile
                    and r["panel"] == panel
                    and r["comparison"] == "schedule_pair"
                    and float(r["margin_weight"]) == w
                )
                for profile in PROFILES
            ]
            means = np.array([float(r["rank_mean"]) for r in subset])
            bounds = np.array(
                [
                    [float(r["rank_mean_ci95_low"]), float(r["rank_mean_ci95_high"])]
                    for r in subset
                ]
            )
            axis.errorbar(
                np.arange(4) + (i - 1) * 0.18,
                means,
                yerr=np.array([means - bounds[:, 0], bounds[:, 1] - means]),
                fmt="o",
                capsize=3,
                color=color,
                label=f"λ={w:.2f}",
            )
        axis.axhline(0, color="#777", linewidth=0.8)
        axis.set_xticks(range(4), PROFILES)
        axis.set_title(panel.replace("_", " "))
        axis.set_ylabel("Easy − difficult expected rank (+ favors difficult)")
        axis.legend()
    fig.suptitle("Frozen schedules: pointwise 95% Monte Carlo intervals")
    fig.tight_layout()
    fig.savefig(
        output / "schedule_comparison.png",
        dpi=160,
        metadata={"Software": "GippyRank research"},
    )
    plt.close(fig)


def seal(root: Path, output: Path, origin: dict):
    source = "src/gippyrank/research/issue_269_analysis.py"
    files = sorted(
        p
        for p in output.rglob("*")
        if p.is_file() and ".checkpoints" not in p.parts and p.name != "manifest.json"
    )
    write_json(
        output / "manifest.json",
        {
            "issue": 269,
            "research_only": True,
            "margin_weights": NEIGHBORS,
            "execution_provenance": origin,
            "analysis_source_sha256": {source: sha256(root / source)},
            "coverage": {
                "synthetic_trials": 3600,
                "new_network_solves": 18,
                "issue252_states": 8,
            },
            "validation": {
                "originals_authenticated": True,
                "frozen_membership_verified": True,
                "full_support_and_normalization": True,
                "all_converged": True,
                "existing_results_rerun": False,
                "production_changed": False,
            },
            "output_sha256": {str(p.relative_to(output)): sha256(p) for p in files},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repeat-check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / OUTPUT
    verified, jobs, old = references(root)
    origin = origin_for(root, output, verified, portable=True)
    run_networks(root, output, verified, origin, verify_only=True)
    if args.repeat_check:
        print(repeat_checks(root, output, verified, origin, execute=True))
        return
    repeat_checks(root, output, verified, origin)
    if args.verify_only:
        for profile in PROFILES:
            run_synthetic(root, output, origin, profile, verify_only=True)
        run_252(root, output, origin, verify_only=True)
        manifest = authenticate(output)
        if manifest["execution_provenance"] != origin or manifest["coverage"] != {
            "synthetic_trials": 3600,
            "new_network_solves": 18,
            "issue252_states": 8,
        }:
            raise ValueError("extension manifest coverage/provenance mismatch")
        for path, digest in manifest["analysis_source_sha256"].items():
            if sha256(root / path) != digest:
                raise ValueError("extension analysis source changed")
        print(
            "Verified all extension inputs, checkpoints, coverage and output hashes without inference or writes."
        )
        return
    synthetic(root, output, origin)
    cells, pmfs, _ = seasons(root, output, verified, origin)
    states, sensitivities = diagnostics(output, verified, jobs, old, cells, pmfs)
    sensitivity_252(root, output, origin, states, sensitivities)
    figure(output)
    seal(root, output, origin)
    print("Validated and analyzed the complete λ=.40 extension.")


if __name__ == "__main__":
    main()
