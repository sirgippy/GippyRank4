"""Descriptions of frozen cases and full-network evidence perturbations.

Baseline rankings and game descriptions are read from their verified original
files. Only the new perturbation rankings are computed here.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gippyrank.posterior.snapshots import sha256
from gippyrank.research.issue_242_quality_of_results import MARGIN_WEIGHTS, _sha256_json
from gippyrank.research.issue_263_real_season_design import ARTIFACT_DIR
from gippyrank.research.issue_265_population import (
    compare,
    order,
    summary,
    top_changes,
    write_csv,
)
from gippyrank.research.issue_265_real_seasons import (
    VerifiedExecution,
    resource_guard,
    write_json,
)
from gippyrank.research.issue_267_diagnostics import (
    BASELINE_OUTPUT,
    checkpoint_path,
    endpoint_gates,
    inventory,
    perturb,
    provenance,
    validate_cell,
)

SUMMARY_FIELDS = tuple(summary(np.array([0.5, 0.5])))


def verify_completed_diagnostics(
    root: Path, output: Path, verified: VerifiedExecution
) -> tuple[dict, list, dict]:
    """Read-only portable consumption, preserving the original execution identity.

    Runtime versions describe where inference actually ran. A reader on CI may
    differ, but every scientific input, implementation source and configuration
    must still match. Strict runtime equality remains required by runner resume.
    No checkpoint or provenance is relabeled and no inference is invoked.
    """
    origin = json.loads((output / "provenance.json").read_text())
    payload = {
        k: v
        for k, v in origin.items()
        if k not in ("fingerprint", "execution_revision")
    }
    if _sha256_json(payload) != origin["fingerprint"]:
        raise ValueError("diagnostic provenance integrity mismatch")
    expected = provenance(root, root / BASELINE_OUTPUT, verified)
    expected = {
        k: v
        for k, v in expected.items()
        if k not in ("fingerprint", "execution_revision")
    }
    expected["environment"] = origin[
        "environment"
    ]  # Original runtime remains audit evidence.
    if payload != expected:
        raise ValueError("completed diagnostic input/configuration/source mismatch")
    jobs, cells = inventory(verified), {}
    from gippyrank.research.issue_267_diagnostics import cell_identity

    for job in jobs:
        n = perturb(verified.networks[job.snapshot_id], job)
        for w in MARGIN_WEIGHTS:
            identity = cell_identity(n, job, w, origin)
            cells[job.key, w] = validate_cell(
                checkpoint_path(output, job, w), n, identity, origin
            )
    execution = json.loads((output / "execution.json").read_text())
    entries = execution["cells"]
    if (
        not execution["complete"]
        or execution["failures"]
        or execution["expected_cells"] != 65
        or execution["completed_cells"] != 65
        or len(entries) != 65
        or {(e["perturbation_key"], e["margin_weight"]) for e in entries} != set(cells)
    ):
        raise ValueError("completed diagnostic coverage mismatch")
    for e in entries:
        cell = cells[e["perturbation_key"], e["margin_weight"]]
        if e["fingerprint"] != cell["fingerprint"] or any(
            e[k] != v for k, v in cell["diagnostics"].items()
        ):
            raise ValueError("diagnostic execution inventory mismatch")
    gates = endpoint_gates(jobs, cells, verified.networks[jobs[0].snapshot_id])
    if len(gates) != 4 or not all(g["passed"] for g in gates.values()):
        raise ValueError("winning-margin endpoint gate failed")
    if gates != json.loads((output / "endpoint_parity.json").read_text()):
        raise ValueError("endpoint gate inventory mismatch")
    return origin, jobs, cells


def read_csv(path: Path) -> list[dict[str, str]]:
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    return list(csv.DictReader(io.StringIO(raw.decode())))


def numeric_rank(row: dict) -> dict:
    ints = {
        "eligible_games",
        "wins",
        "losses",
        "ordinal_rank",
        "season",
        "median_rank",
        "modal_rank",
    }
    ints.update(k for k in SUMMARY_FIELDS if k.startswith(("low_", "high_", "width_")))
    result = dict(row)
    for key in (*SUMMARY_FIELDS, "margin_weight", *ints):
        if key in row:
            result[key] = (
                (int if key in ints else float)(row[key]) if row[key] != "" else None
            )
    return result


def rich_compare(p: np.ndarray, q: np.ndarray) -> dict:
    left, right = summary(p), summary(q)
    return {
        **compare(p, q),
        **{f"delta_{k}": right[k] - left[k] for k in SUMMARY_FIELDS},
    }


def adjacent_gaps(t: str, ranks: dict[str, int], means: dict[str, float]) -> dict:
    if t not in ranks:
        return {"gap_to_better": None, "gap_to_worse": None}
    inverse = {r: team for team, r in ranks.items()}
    rank = ranks[t]
    return {
        "gap_to_better": means[t] - means[inverse[rank - 1]] if rank > 1 else None,
        "gap_to_worse": means[inverse[rank + 1]] - means[t]
        if rank + 1 in inverse
        else None,
    }


def analyze_diagnostics(
    output: Path, verified: VerifiedExecution, jobs: list, cells: dict, ranks: dict
) -> tuple[list, list, list]:
    network = verified.networks[jobs[0].snapshot_id]
    names = {t.team_id: t.name for t in network.teams}
    effects, summaries, focused, sensitivity = [], [], [], []
    for job in jobs:
        n = perturb(network, job)
        for weight in MARGIN_WEIGHTS:
            cell = cells[job.key, weight]
            pmfs = {t: np.array(p) for t, p in cell["pmfs"].items()}
            baseline = verified.cells[job.snapshot_id]["baseline", weight]
            before = {
                t: ranks[job.snapshot_id, weight, t]["ordinal_rank"]
                for t in pmfs
                if network.counts[t]
            }
            after = order(pmfs, n.counts)
            means = {t: summary(p)["expected_rank"] for t, p in pmfs.items()}
            base_means = {
                t: ranks[job.snapshot_id, weight, t]["expected_rank"] for t in pmfs
            }
            own = []
            for t in sorted(pmfs):
                old = np.array(baseline["pmfs"][t])
                s, b = summary(pmfs[t]), ranks[job.snapshot_id, weight, t]
                role = (
                    "focal"
                    if t == job.focal_id
                    else "opponent"
                    if t == job.opponent_id
                    else "other"
                )
                row = {
                    "perturbation_key": job.key,
                    "snapshot_id": job.snapshot_id,
                    "kind": job.kind,
                    "game_id": job.game_id,
                    "hypothetical_margin": job.margin,
                    "profile": job.profile,
                    "margin_weight": weight,
                    "team_id": t,
                    "team_name": names[t],
                    "role": role,
                    "subdivision": "fbs",
                    "posterior_family": "prior_stripped_performance",
                    "baseline_eligible_games": network.counts[t],
                    "eligible_games": n.counts[t],
                    "delta_eligible_games": n.counts[t] - network.counts[t],
                    "baseline_ordinal_rank": before.get(t),
                    "ordinal_rank": after.get(t),
                    "delta_ordinal_rank": after[t] - before[t]
                    if t in after and t in before
                    else None,
                    **s,
                    **{f"baseline_{k}": b[k] for k in SUMMARY_FIELDS},
                    **rich_compare(old, pmfs[t]),
                    **adjacent_gaps(t, after, means),
                    **{
                        f"baseline_{k}": v
                        for k, v in adjacent_gaps(t, before, base_means).items()
                    },
                }
                for k in (10, 12, 25):
                    row[f"top{k}_entry"] = (
                        t in after
                        and after[t] <= k
                        and (t not in before or before[t] > k)
                    )
                    row[f"top{k}_exit"] = (
                        t in before
                        and before[t] <= k
                        and (t not in after or after[t] > k)
                    )
                effects.append(row)
                own.append(row)
                if role != "other":
                    focused.append(row)
            # The dominant-win probe's opponent is FCS. Its engine posterior is
            # a Context marginal on FCS support; there is no published FCS
            # Performance distribution or FBS display ordinal to fabricate.
            if job.opponent_id not in pmfs:
                t = job.opponent_id
                old = np.array(baseline["context"][t])
                new = np.array(cell["context"][t])
                focused.append(
                    {
                        "perturbation_key": job.key,
                        "snapshot_id": job.snapshot_id,
                        "kind": job.kind,
                        "game_id": job.game_id,
                        "hypothetical_margin": job.margin,
                        "profile": job.profile,
                        "margin_weight": weight,
                        "team_id": t,
                        "team_name": names[t],
                        "role": "opponent",
                        "subdivision": "fcs",
                        "posterior_family": "context_on_fcs_support",
                        "baseline_eligible_games": network.counts[t],
                        "eligible_games": n.counts[t],
                        "delta_eligible_games": n.counts[t] - network.counts[t],
                        "baseline_ordinal_rank": None,
                        "ordinal_rank": None,
                        "delta_ordinal_rank": None,
                        **summary(new),
                        **{f"baseline_{k}": v for k, v in summary(old).items()},
                        **rich_compare(old, new),
                        "gap_to_better": None,
                        "gap_to_worse": None,
                        "baseline_gap_to_better": None,
                        "baseline_gap_to_worse": None,
                        **{
                            f"top{k}_{action}": None
                            for k in (10, 12, 25)
                            for action in ("entry", "exit")
                        },
                    }
                )
            other = [r for r in own if r["role"] == "other"]
            worst = max(other, key=lambda r: abs(r["delta_expected_rank"]))
            summaries.append(
                {
                    "perturbation_key": job.key,
                    "kind": job.kind,
                    "profile": job.profile,
                    "game_id": job.game_id,
                    "margin_weight": weight,
                    "hypothetical_margin": job.margin,
                    "original_observation": job.original_observation,
                    "replacement_observation": job.replacement_observation,
                    "baseline_game_count": len(network.games),
                    "game_count": len(n.games),
                    "fbs_team_count": len(pmfs),
                    "network_team_count": len(n.teams),
                    "other_team_count": len(other),
                    "other_mean_absolute_expected_rank_change": float(
                        np.mean([abs(r["delta_expected_rank"]) for r in other])
                    ),
                    "other_max_absolute_expected_rank_change": abs(
                        worst["delta_expected_rank"]
                    ),
                    "other_max_change_team": worst["team_name"],
                    "other_max_total_variation": max(
                        r["total_variation"] for r in other
                    ),
                    "other_teams_changed_above_mean_tolerance": sum(
                        abs(r["delta_expected_rank"]) > len(pmfs) * 1e-8 for r in other
                    ),
                    **{
                        k: v
                        for boundary in (10, 12, 25)
                        for k, v in top_changes(before, after, boundary, names).items()
                    },
                    **cell["diagnostics"],
                }
            )
    lookup = {
        (r["perturbation_key"], r["margin_weight"], r["role"]): r for r in focused
    }
    for probe in verified.specification["game_diagnostics"]["counterfactuals"]:
        gid = probe["game_id"]
        for weight in MARGIN_WEIGHTS:
            states = {
                margin: lookup[f"replacement-{gid}-{margin:+d}", weight, "focal"]
                for margin in probe["margins"]
            }
            benefit = states[-3]["expected_rank"] - states[3]["expected_rank"]
            margin_benefit = states[10]["expected_rank"] - states[35]["expected_rank"]
            sensitivity.append(
                {
                    "game_id": gid,
                    "profile": probe["profile"],
                    "margin_weight": weight,
                    "focal_team_id": probe["team_id"],
                    "focal_team_name": names[probe["team_id"]],
                    "outcome_boundary_benefit": benefit,
                    "within_win_margin_benefit": margin_benefit,
                    "outcome_minus_margin_benefit": benefit - margin_benefit,
                    "outcome_boundary_ordinal_benefit": states[-3]["ordinal_rank"]
                    - states[3]["ordinal_rank"],
                    "within_win_ordinal_benefit": states[10]["ordinal_rank"]
                    - states[35]["ordinal_rank"],
                    **{
                        f"margin_{m:+d}_{k}": r[k]
                        for m, r in states.items()
                        for k in (*SUMMARY_FIELDS, "ordinal_rank")
                    },
                }
            )
    write_csv(output / "network_changes.csv.gz", effects)
    write_csv(output / "game_diagnostics.csv", focused)
    write_csv(output / "diagnostic_summary.csv", summaries)
    write_csv(output / "outcome_sensitivity.csv", sensitivity)
    write_csv(
        output / "convergence.csv",
        [
            {
                "perturbation_key": j.key,
                "margin_weight": w,
                "fingerprint": cells[j.key, w]["fingerprint"],
                "checkpoint": str(checkpoint_path(output, j, w).relative_to(output)),
                **cells[j.key, w]["diagnostics"],
            }
            for j in jobs
            for w in MARGIN_WEIGHTS
        ],
    )
    return focused, summaries, sensitivity


def evidence(root: Path, baseline: Path, verified: VerifiedExecution) -> dict:
    """Join saved actual-game descriptors to original dated evidence and frozen references."""
    result = defaultdict(list)
    primary = {}
    for file in ("reference_games.csv", "historical_reference_games.csv"):
        for r in read_csv(root / ARTIFACT_DIR / file):
            sid = r.get("snapshot_id", r.get("publication_slot"))
            primary[sid, r["team_id"], r["game_id"]] = r
    for r in read_csv(baseline / "actual_game_descriptors.csv.gz"):
        sid, t, gid = r["snapshot_id"], r["team_id"], r["game_id"]
        n = verified.networks[sid]
        original = next(row for row in n.rows if row["id"] == gid)
        g = next(g for g in n.games if g.game_id == gid)
        p = primary[sid, t, gid]
        is_home = t == g.home_id
        value = {
            **r,
            "margin": int(r["margin"]),
            "original_observation": original,
            "focal_points": g.home_points if is_home else g.away_points,
            "opponent_points": g.away_points if is_home else g.home_points,
            "frozen_primary_win_probability": float(
                p.get("prior_win_probability", p.get("context_win_probability"))
            ),
            "frozen_primary_schedule_probability": float(
                p.get(
                    "schedule_reference_win_probability",
                    p.get("reference_win_probability"),
                )
            ),
            "reference_kind": "preseason_context_v1.2"
            if n.specification["season"] == 2025
            else "retrospective_context_v1.4",
        }
        for k in (
            "context_win_probability",
            "expected_margin",
            "schedule_reference_win_probability",
            "opponent_context_expected_rank",
            "opponent_context_rank_sd",
        ):
            value[k] = float(value[k])
        result[sid, t].append(value)
    return {
        key: sorted(
            rows, key=lambda r: (r["original_observation"]["startDate"], r["game_id"])
        )
        for key, rows in result.items()
    }


def case_answer(
    question: str, row: dict, peer: dict | None, diagnostic: dict | None
) -> str:
    """Post-observation interpretation, kept separate from the frozen question."""
    common = (
        f"At lambda={row['margin_weight']:g}, E[rank]={row['expected_rank']:.2f}, ordinal={row['ordinal_rank']}, "
        f"SD={row['rank_sd']:.2f}, entropy={row['entropy_bits']:.3f} bits, central 80%={row['low_80']}-{row['high_80']}; "
        f"versus Record/Play the mean-rank changes are {row['vs_record_delta_expected_rank']:+.2f}/{row['vs_play_delta_expected_rank']:+.2f}. "
    )
    q = question.lower()
    if "postseason" in q:
        return (
            common
            + "The pre-bowl-to-final comparison is retained for every lambda in case_evolution.csv, including own new games and the 55 added network observations; movement includes opponent updates and cannot be assigned solely to the team's postseason games."
        )
    if peer is not None and any(
        word in q for word in ("schedule", "identical", "uncertainty")
    ):
        gap = row["expected_rank"] - peer["expected_rank"]
        return common + (
            f"The frozen peer {peer['team_name']} is {peer['wins']}-{peer['losses']} with standardized schedule probability "
            f"{peer['schedule_reference_mean_win_probability']:.3f}, E[rank]={peer['expected_rank']:.2f}, SD={peer['rank_sd']:.2f}, "
            f"entropy={peer['entropy_bits']:.3f}, widths 50/80/95={peer['width_50']}/{peer['width_80']}/{peer['width_95']}; "
            f"this team's expected-rank gap is {gap:+.2f} and SD gap {row['rank_sd'] - peer['rank_sd']:+.2f}. "
            "Accomplishments, reference strength and bounded posterior locations differ; this selected comparison cannot isolate a causal schedule effect or establish equivalence."
        )
    if "boundary" in q:
        return common + (
            f"Latent top10/12/25 mass is {row['mass_le_10']:.3f}/{row['mass_le_12']:.3f}/{row['mass_le_25']:.3f}. "
            f"Adjacent expected-rank gaps are {row['gap_to_better']}/{row['gap_to_worse']}. "
            "Ordinal crossings are consequential display changes, while these diffuse marginals do not establish sharp separation. Historical boundary categories refer to preseason references."
        )
    if diagnostic is not None and any(
        word in q for word in ("loss", "narrow", "dominant", "blowout")
    ):
        return common + (
            f"Removing the frozen {diagnostic['game_id']} observation changes this team's mean rank by "
            f"{diagnostic['delta_expected_rank']:+.2f}, ordinal by {diagnostic['delta_ordinal_rank']:+d}, "
            f"SD by {diagnostic['delta_rank_sd']:+.2f}, TV={diagnostic['total_variation']:.3f}. "
            "This measures the total network response to removing one observation, not an additive contribution."
        )
    return common + (
        f"Actual evidence is {row['wins']}-{row['losses']}, mean margin {row['average_margin']:+.2f}, "
        f"{row['close_wins']} close wins and {row['close_losses']} close losses, standardized schedule probability "
        f"{row['schedule_reference_mean_win_probability']:.3f}, expected wins {row['expected_wins']:.2f}, "
        f"wins minus expectation {row['wins_minus_expectation']:+.2f}. "
        "These describe the record/margin disagreement and opponent-aware accomplishment, without independently validating a candidate. "
        + (
            "The expectation is retrospective and reuses these games."
            if row["season"] == 2026
            else "The primary expectation uses frozen preseason priors; original prospective availability remains unknown."
        )
    )


def analyze_cases(
    root: Path,
    output: Path,
    verified: VerifiedExecution,
    ranks: dict,
    games: dict,
    focused: list,
) -> tuple[list, list]:
    cases = [
        *json.loads((root / ARTIFACT_DIR / "cases.json").read_text()),
        *json.loads((root / ARTIFACT_DIR / "historical_cases.json").read_text()),
    ]
    if len(cases) != 70 or len({(c["season"], c["team_id"]) for c in cases}) != 27:
        raise ValueError("frozen case coverage changed")
    descriptions = {
        (r["snapshot_id"], r["team_id"]): r
        for r in read_csv(root / BASELINE_OUTPUT / "accomplishments.csv")
    }
    rows = []
    for c in cases:
        sid, t = c.get("snapshot_id", c.get("publication_slot")), c["team_id"]
        n = verified.networks[sid]
        own = games[sid, t]
        if len(own) != c["games"] or sum(g["margin"] > 0 for g in own) != c["wins"]:
            raise ValueError("case evidence differs from frozen record")
        d = descriptions[sid, t]
        expected = c.get("prior_expected_wins", c.get("context_expected_wins"))
        if (
            abs(sum(g["frozen_primary_win_probability"] for g in own) - expected)
            > 1e-10
        ):
            raise ValueError("case frozen expectation mismatch")
        if (
            abs(
                np.mean([g["frozen_primary_schedule_probability"] for g in own])
                - c["schedule_reference_mean_win_probability"]
            )
            > 1e-10
        ):
            raise ValueError("case frozen schedule mismatch")
        for weight in MARGIN_WEIGHTS:
            b = ranks[sid, weight, t]
            pmfs = verified.cells[sid]["baseline", weight]["pmfs"]
            p = np.array(pmfs[t])
            ordinals = {
                team: ranks[sid, weight, team]["ordinal_rank"]
                for team in pmfs
                if n.counts[team]
            }
            means = {team: ranks[sid, weight, team]["expected_rank"] for team in pmfs}
            rows.append(
                {
                    **b,
                    "frozen_selection": c,
                    "categories": c["categories"],
                    "questions": c["questions"],
                    "selection_snapshot": c.get(
                        "selection_snapshot_id", c.get("selection_publication_slot")
                    ),
                    "reference_kind": d["reference_kind"],
                    "reference_strength_ordinal": int(d["reference_ordinal"]),
                    "strength_band": d["strength_band"],
                    "schedule_tercile": d["schedule_tercile"],
                    "schedule_reference_mean_win_probability": c[
                        "schedule_reference_mean_win_probability"
                    ],
                    "expected_wins": expected,
                    "wins_minus_expectation": c["wins"] - expected,
                    "near_expected_record": abs(c["wins"] - expected) <= 1,
                    "average_margin": c["average_margin"],
                    "close_wins": sum(0 < g["margin"] <= 7 for g in own),
                    "close_losses": sum(-7 <= g["margin"] < 0 for g in own),
                    "dominant_wins": sum(g["margin"] >= 28 for g in own),
                    "actual_games": own,
                    **{
                        f"vs_record_{k}": v
                        for k, v in rich_compare(
                            np.array(verified.cells[sid]["baseline", 0.0]["pmfs"][t]), p
                        ).items()
                    },
                    **{
                        f"vs_play_{k}": v
                        for k, v in rich_compare(
                            np.array(verified.cells[sid]["baseline", 1.0]["pmfs"][t]), p
                        ).items()
                    },
                    "vs_record_delta_ordinal_rank": b["ordinal_rank"]
                    - ranks[sid, 0.0, t]["ordinal_rank"],
                    "vs_play_delta_ordinal_rank": b["ordinal_rank"]
                    - ranks[sid, 1.0, t]["ordinal_rank"],
                    **adjacent_gaps(t, ordinals, means),
                    **{f"in_top{k}": b["ordinal_rank"] <= k for k in (10, 12, 25)},
                }
            )
    lookup = {(r["snapshot_id"], r["margin_weight"], r["team_id"]): r for r in rows}
    interpretations = []
    for r in rows:
        peers = [
            v
            for v in rows
            if v["snapshot_id"] == r["snapshot_id"]
            and v["margin_weight"] == r["margin_weight"]
            and v["team_id"] != r["team_id"]
            and set(v["categories"])
            & set(r["categories"])
            & {"weak_schedule_pair", "middle_schedule_pair"}
        ]
        diagnostics = [
            d
            for d in focused
            if d["snapshot_id"] == r["snapshot_id"]
            and d["margin_weight"] == r["margin_weight"]
            and d["team_id"] == r["team_id"]
            and d["kind"] == "removal"
            and d["role"] == "focal"
        ]
        r["answers_after_observation"] = [
            {
                "question": q,
                "answer": case_answer(
                    q,
                    r,
                    peers[0] if peers else None,
                    diagnostics[0] if diagnostics else None,
                ),
            }
            for q in r["questions"]
        ]
        interpretations.append(
            {
                k: r[k]
                for k in (
                    "snapshot_id",
                    "season",
                    "team_id",
                    "team_name",
                    "margin_weight",
                    "categories",
                    "questions",
                    "answers_after_observation",
                )
            }
        )
    transitions = [
        ("2025-prebowl", "2025-final"),
        ("2026-09-08", "2026-09-27"),
        ("2026-09-27", "2026-10-25T12-59-22.246522Z"),
    ]
    evolution = []
    for old, new in transitions:
        old_network, new_network = verified.networks[old], verified.networks[new]
        old_ids = {g.game_id for g in old_network.games}
        added = [g.game_id for g in new_network.games if g.game_id not in old_ids]
        ids = sorted({r["team_id"] for r in rows if r["snapshot_id"] == old})
        for t in ids:
            for w in MARGIN_WEIGHTS:
                a, b = lookup[old, w, t], lookup[new, w, t]
                evolution.append(
                    {
                        "season": a["season"],
                        "team_id": t,
                        "team_name": a["team_name"],
                        "earlier_snapshot": old,
                        "later_snapshot": new,
                        "margin_weight": w,
                        "earlier_record": f"{a['wins']}-{a['losses']}",
                        "later_record": f"{b['wins']}-{b['losses']}",
                        "added_network_game_count": len(added),
                        "added_own_games": [
                            g for g in b["actual_games"] if g["game_id"] in added
                        ],
                        "earlier_expected_rank": a["expected_rank"],
                        "later_expected_rank": b["expected_rank"],
                        "earlier_ordinal_rank": a["ordinal_rank"],
                        "later_ordinal_rank": b["ordinal_rank"],
                        "delta_ordinal_rank": b["ordinal_rank"] - a["ordinal_rank"],
                        **rich_compare(
                            np.array(verified.cells[old]["baseline", w]["pmfs"][t]),
                            np.array(verified.cells[new]["baseline", w]["pmfs"][t]),
                        ),
                    }
                )
    # Answer the historical evolution question with its measured transition,
    # rather than leaving readers to join a second table themselves.
    for r in rows:
        r["evolution_across_snapshots"] = [
            e
            for e in evolution
            if e["team_id"] == r["team_id"]
            and e["season"] == r["season"]
            and e["margin_weight"] == r["margin_weight"]
        ]
        if r["season"] == 2025:
            e = r["evolution_across_snapshots"][0]
            for answer in r["answers_after_observation"]:
                if "postseason" in answer["question"].lower():
                    answer["answer"] = (
                        f"At lambda={r['margin_weight']:g}, pre-bowl-to-final E[rank] moves "
                        f"{e['earlier_expected_rank']:.2f} to {e['later_expected_rank']:.2f} "
                        f"({e['delta_expected_rank']:+.2f}); ordinal changes {e['delta_ordinal_rank']:+d}, "
                        f"SD changes {e['delta_rank_sd']:+.2f}, entropy {e['delta_entropy_bits']:+.3f}, "
                        f"80% width {e['delta_width_80']:+d}, TV={e['total_variation']:.3f}. "
                        f"Record changes {e['earlier_record']} to {e['later_record']}, with "
                        f"{len(e['added_own_games'])} added own observations and 55 network observations. "
                        "These are total network changes, including opponent updates; even unchanged own "
                        "records can move, and own games do not supply an additive explanation."
                    )
    write_csv(output / "case_results.csv", rows)
    write_csv(output / "case_evolution.csv", evolution)
    write_json(output / "case_interpretations.json", interpretations)
    selected = {(r["snapshot_id"], r["team_id"]) for r in rows}
    write_csv(
        output / "case_games.csv", [g for key in sorted(selected) for g in games[key]]
    )
    return rows, evolution


def write_case_studies(output: Path, rows: list, evolution: list) -> None:
    lines = [
        "# Frozen team-season case studies",
        "",
        (
            "All 27 team seasons and 70 snapshot-case observations are retained. Each table uses the original verified baseline rankings. "
            "Categories and questions were frozen before intermediate candidates were solved; the answers below are interpretations after observation. "
            "Rank intervals describe latent marginal coordinates, while ordinal ranks sort means. "
            "The 2026 expected wins reuse cutoff Context evidence; the 2025 primary expectations use preseason priors."
        ),
        "",
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[r["season"], r["team_id"]].append(r)
    for (season, t), team_rows in sorted(
        groups.items(), key=lambda x: (x[0][0], x[1][0]["team_name"])
    ):
        name = team_rows[0]["team_name"]
        lines += [
            f"## {season} {name} ({t})",
            "",
            "Frozen categories: " + ", ".join(team_rows[0]["categories"]) + ".",
            "",
        ]
        for sid in dict.fromkeys(r["snapshot_id"] for r in team_rows):
            selected = sorted(
                [r for r in team_rows if r["snapshot_id"] == sid],
                key=lambda r: r["margin_weight"],
            )
            a = selected[0]
            lines += [
                f"### {sid}",
                "",
                (
                    f"Actual record **{a['wins']}–{a['losses']}**, mean margin **{a['average_margin']:+.2f}**; "
                    f"close wins/losses {a['close_wins']}/{a['close_losses']}, dominant wins {a['dominant_wins']}. "
                    f"Frozen standardized schedule probability **{a['schedule_reference_mean_win_probability']:.3f}** (larger is easier), "
                    f"expected wins **{a['expected_wins']:.2f}**, wins minus expectation **{a['wins_minus_expectation']:+.2f}**. "
                    f"Reference: {a['reference_kind']}, ordinal {a['reference_strength_ordinal']}, band {a['strength_band']}, schedule tercile {a['schedule_tercile']}."
                ),
                "",
                "| λ | E[rank] | Ordinal | SD | Entropy bits | 50% interval | 80% interval | 95% interval | Top10/12/25 mass | ΔE vs Record / Play |",
                "|---|---:|---:|---:|---:|---|---|---|---|---|",
            ]
            for r in selected:
                lines += [
                    f"| {r['margin_weight']:g} | {r['expected_rank']:.2f} | {r['ordinal_rank']} | {r['rank_sd']:.2f} | {r['entropy_bits']:.3f} | {r['low_50']}–{r['high_50']} | {r['low_80']}–{r['high_80']} | {r['low_95']}–{r['high_95']} | {r['mass_le_10']:.3f}/{r['mass_le_12']:.3f}/{r['mass_le_25']:.3f} | {r['vs_record_delta_expected_rank']:+.2f} / {r['vs_play_delta_expected_rank']:+.2f} |"
                ]
            lines += [
                "",
                "| Game ID/date | Venue | Opponent | Actual score | Margin | Primary expected win | Opponent Context E/SD | Expected margin (posterior) |",
                "|---|---|---|---:|---:|---:|---|---:|",
            ]
            for g in a["actual_games"]:
                lines += [
                    f"| {g['game_id']} / {g['original_observation']['startDate'][:10]} | {g['venue']} | {g['opponent_name']} ({g['opponent_subdivision']}) | {g['focal_points']}–{g['opponent_points']} | {g['margin']:+d} | {g['frozen_primary_win_probability']:.3f} | {g['opponent_context_expected_rank']:.2f}/{g['opponent_context_rank_sd']:.2f} | {g['expected_margin']:+.2f} |"
                ]
            lines += ["", "Answers to the original questions at every candidate:", ""]
            for i, q in enumerate(a["questions"]):
                lines += [f"**{q}**", ""]
                for r in selected:
                    lines += ["- " + r["answers_after_observation"][i]["answer"]]
                lines += [""]
        e = [r for r in evolution if r["season"] == season and r["team_id"] == t]
        lines += [
            "### Evolution across available snapshots",
            "",
            "| Transition | λ | Own added games / network added games | ΔE[rank] | Δordinal | ΔSD | TV |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
        for r in e:
            lines += [
                f"| {r['earlier_snapshot']} → {r['later_snapshot']} | {r['margin_weight']:g} | {len(r['added_own_games'])} / {r['added_network_game_count']} | {r['delta_expected_rank']:+.2f} | {r['delta_ordinal_rank']:+d} | {r['delta_rank_sd']:+.2f} | {r['total_variation']:.3f} |"
            ]
        lines += [
            "",
            "These changes reflect the full network and snapshot-specific frozen priors. Own added games are evidence descriptions, not an additive explanation of movement.",
            "",
        ]
    (output / "case_studies.md").write_text("\n".join(lines).rstrip() + "\n")


def seal(root: Path, output: Path, verified: VerifiedExecution, origin: dict) -> None:
    files = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        {
            "issue": 267,
            "execution_provenance": origin,
            "baseline_verification": origin["baseline"],
            "analysis_source_sha256": {
                p: sha256(root / p)
                for p in (
                    "src/gippyrank/research/issue_267_analysis.py",
                    "src/gippyrank/research/issue_267_report.py",
                    "src/gippyrank/research/issue_265_population.py",
                )
            },
            "coverage": {
                "team_seasons": 27,
                "snapshot_cases": 70,
                "candidate_case_rows": 350,
                "counterfactual_solves": 40,
                "removal_solves": 25,
                "network_change_rows": 65 * 138,
                "case_evolution_rows": 215,
            },
            "output_sha256": files,
            "frozen_design_manifest_sha256": sha256(
                root / ARTIFACT_DIR / "manifest.json"
            ),
            "production_methodology_selected": False,
        },
    )


def _analyze(root: Path, output: Path, verified: VerifiedExecution) -> None:
    origin, jobs, cells = verify_completed_diagnostics(root, output, verified)
    baseline = root / BASELINE_OUTPUT
    ranks = {
        (r["snapshot_id"], r["margin_weight"], r["team_id"]): r
        for r in (numeric_rank(r) for r in read_csv(baseline / "rankings.csv"))
    }
    focused, diagnostics, sensitivity = analyze_diagnostics(
        output, verified, jobs, cells, ranks
    )
    games = evidence(root, baseline, verified)
    cases, evolution = analyze_cases(root, output, verified, ranks, games, focused)
    write_case_studies(output, cases, evolution)
    from gippyrank.research.issue_267_report import write_report

    write_report(root, output, cases, evolution, focused, diagnostics, sensitivity)
    seal(root, output, verified, origin)


def analyze(root: Path, output: Path, verified: VerifiedExecution) -> None:
    # The reused solve guard restores its signal handler but disables its timer.
    # Establish an independent analysis timer after inference as well; retain
    # the original execution's consumed wall time when budgeting this stage.
    execution = json.loads((output / "execution.json").read_text())
    remaining = 10800 - execution["elapsed_seconds"]
    with resource_guard(time.monotonic() + remaining, timeout=10800):
        _analyze(root, output, verified)
