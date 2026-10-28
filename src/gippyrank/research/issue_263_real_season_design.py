"""Freeze real-season inputs and reference-selected cases without candidate inference."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from gippyrank.artifact_hashes import included_game_rows_sha256, posterior_pmfs_sha256
from gippyrank.performance_snapshot import _pmfs
from gippyrank.posterior.engine import Game, LikelihoodV1, Team
from gippyrank.posterior.predictive import (
    ScheduledGame,
    conditional_margin_location_surface,
)
from gippyrank.posterior.snapshots import sha256
from gippyrank.research.issue_242_quality_of_results import (
    BAD_CAMPBELL_FIXTURE_ID,
    MARGIN_WEIGHTS,
    REPAIRED_CAMPBELL_GAME_ID,
    _baseline_rankings,
    _input_bundle,
    _pmf_summaries,
    _published_inputs,
    _read_json,
    _read_rows,
    _write_csv,
)

BASE_REVISION = "95401c6b1699ee9750884fab2163772420951a6e"
SELECTED_SLOTS = ("2026-09-08", "2026-09-27", "2026-10-25T12-59-22.246522Z")
ARTIFACT_DIR = "data/research/issue_263_real_season_design"
DOCUMENT = "docs/issue_263_real_season_protocol.md"
CASE_QUESTIONS = {
    "elite": "How do outcomes versus margins distinguish elite accomplishments?",
    "strong": "Does a strong team's evaluation depend on wins or dominant margins?",
    "middle": "Is any easier-schedule advantage visible amid middle-team uncertainty?",
    "weak": "Does weak-team uncertainty decrease on easier schedules, and how much does winning matter?",
    "excellent_record_easy_schedule": "Does this excellent record receive disproportionate easier-schedule credit?",
    "middling_record_difficult_schedule": "Does opponent evidence explain this team's ordinary record?",
    "competitive_elite_loss": "How does a competitive loss to an excellent opponent affect each candidate?",
    "narrow_win_weaker_opponent": "How much credit does a narrow win against a weaker opponent receive?",
    "dominant_win_overmatched_opponent": "Does one dominant win against an overmatched opponent drive the ranking?",
    "endpoint_disagreement": "Which games and uncertainty explain the Quality of Record/Play disagreement?",
    "middle_schedule_pair": "For identical middle-team records, how do schedule and uncertainty differ across candidates?",
    "weak_schedule_pair": "For identical weak-team records, does the easier schedule narrow uncertainty?",
}


def validate_rows(rows: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    """Reject duplicate, future, malformed or known superseded evidence."""
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or BAD_CAMPBELL_FIXTURE_ID in ids:
        raise ValueError("duplicate or superseded game evidence")
    cutoff = datetime.fromisoformat(metadata["effective_cutoff"])
    for row in rows:
        if (
            row["completed"].casefold() not in {"true", "1", "yes"}
            or int(row["season"]) != int(metadata["season"])
            or datetime.fromisoformat(row["startDate"]) > cutoff
            or int(row["homePoints"]) == int(row["awayPoints"])
            or min(int(row["homePoints"]), int(row["awayPoints"])) < 0
        ):
            raise ValueError(f"ineligible game evidence: {row['id']}")
    if metadata.get("included_game_rows_sha256", included_game_rows_sha256(rows)) != (
        included_game_rows_sha256(rows)
    ):
        raise ValueError("included-game rows hash mismatch")
    correction = metadata.get("historical_evidence_correction")
    if correction and (
        correction["source_requested_cutoff"] != metadata["requested_cutoff"]
        or correction["source_effective_cutoff"] != metadata["effective_cutoff"]
        or correction["source_prior_model_version"] != metadata["prior_model_version"]
    ):
        raise ValueError("repair changed cutoff or prior version")
    replacement = [row for row in rows if row["id"] == REPAIRED_CAMPBELL_GAME_ID]
    if len(replacement) != 1 or (
        replacement[0]["homeId"],
        replacement[0]["awayId"],
        replacement[0]["homePoints"],
        replacement[0]["awayPoints"],
    ) != ("2097", "2717", "28", "19"):
        raise ValueError("corrected Campbell result is missing or changed")


def integrated_win_probability(
    game: Game, home: Team, away: Team, likelihood: LikelihoodV1
) -> float:
    """Integrate the existing home-oriented V1 surface over complete marginals."""
    scheduled = ScheduledGame(
        game.game_id,
        game.home_id,
        game.away_id,
        game.home_subdivision,
        game.away_subdivision,
        game.neutral_site,
    )
    locations = conditional_margin_location_surface(scheduled, home, away, likelihood)
    probabilities = student_t.sf(
        -locations / likelihood.scale, likelihood.degrees_of_freedom
    )
    return float(home.prior @ probabilities @ away.prior)


def reference_descriptors(
    teams: list[Team],
    games: list[Game],
    context_pmfs: dict[str, np.ndarray],
    qop: dict[str, np.ndarray],
    qor: dict[str, np.ndarray],
    counts: dict[str, int],
    likelihood: LikelihoodV1,
    ranks: dict[str, int],
    qor_ranks: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe actual evidence with fixed reference beliefs, never candidate outputs."""
    by_id = {team.team_id: team for team in teams}
    fbs = sorted(qop)
    # Same full preseason PMF for every schedule: no scalar strength estimate.
    reference = min(
        (by_id[team_id] for team_id in fbs),
        key=lambda team: (
            abs(float(_pmf_summaries(team.prior)["expected_rank"]) - 65),
            team.team_id,
        ),
    )
    output: dict[str, dict[str, Any]] = {}
    game_rows: list[dict[str, Any]] = []
    for team_id in fbs:
        output[team_id] = {
            "team_id": team_id,
            "team_name": by_id[team_id].name,
            "games": counts[team_id],
            "wins": 0,
            "losses": 0,
            "qop_ordinal": ranks.get(team_id),
            "qor_ordinal": qor_ranks.get(team_id),
            "qop_expected_rank": _pmf_summaries(qop[team_id])["expected_rank"],
            "qor_expected_rank": _pmf_summaries(qor[team_id])["expected_rank"],
            "reference_team_id": reference.team_id,
        }
    for game in games:
        home, away = by_id[game.home_id], by_id[game.away_id]
        posterior_home = replace(home, prior=context_pmfs[home.team_id])
        posterior_away = replace(away, prior=context_pmfs[away.team_id])
        actual_p = integrated_win_probability(
            game, posterior_home, posterior_away, likelihood
        )
        for focal, opponent, margin, is_home in (
            (home, posterior_away, game.home_points - game.away_points, True),
            (away, posterior_home, game.away_points - game.home_points, False),
        ):
            if focal.team_id not in output:
                continue
            standardized = replace(focal, prior=reference.prior)
            ref_home = standardized if is_home else opponent
            ref_away = opponent if is_home else standardized
            ref_p = integrated_win_probability(game, ref_home, ref_away, likelihood)
            p = actual_p if is_home else 1 - actual_p
            ref_p = ref_p if is_home else 1 - ref_p
            output[focal.team_id]["wins"] += int(margin > 0)
            output[focal.team_id]["losses"] += int(margin < 0)
            game_rows.append(
                {
                    "team_id": focal.team_id,
                    "game_id": game.game_id,
                    "opponent_id": opponent.team_id,
                    "opponent_name": opponent.name,
                    "opponent_subdivision": opponent.subdivision,
                    "margin": margin,
                    "venue": "neutral"
                    if game.neutral_site
                    else "home"
                    if is_home
                    else "away",
                    "reference_win_probability": ref_p,
                    "context_win_probability": p,
                    "opponent_qop_ordinal": ranks.get(opponent.team_id),
                }
            )
    for team_id, row in output.items():
        own = [game for game in game_rows if game["team_id"] == team_id]
        row["win_fraction"] = row["wins"] / row["games"] if own else None
        row["schedule_reference_mean_win_probability"] = (
            float(np.mean([game["reference_win_probability"] for game in own]))
            if own
            else None
        )
        row["context_expected_wins"] = sum(
            game["context_win_probability"] for game in own
        )
        row["wins_minus_context_expectation"] = (
            row["wins"] - row["context_expected_wins"]
        )
        row["average_margin"] = (
            float(np.mean([game["margin"] for game in own])) if own else None
        )
        row["qor_minus_qop_expected_rank"] = (
            row["qor_expected_rank"] - row["qop_expected_rank"]
        )
    return list(output.values()), game_rows


def select_cases(
    rows: list[dict[str, Any]], games: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply frozen rules to endpoint references and eligible outcomes only."""
    rated = sorted((row for row in rows if row["games"]), key=lambda r: r["team_id"])
    selections: dict[str, list[str]] = {}
    anchors: dict[str, str] = {}

    def add(label: str, row: dict[str, Any]) -> None:
        selections.setdefault(row["team_id"], []).append(label)
        anchors[label] = row["team_id"]

    for label, target in (("elite", 5), ("strong", 20), ("middle", 65), ("weak", 120)):
        add(
            label,
            min(rated, key=lambda r: (abs(r["qop_ordinal"] - target), r["team_id"])),
        )
    excellent = [r for r in rated if r["win_fraction"] >= 0.8 and r["games"] >= 5]
    if excellent:
        add(
            "excellent_record_easy_schedule",
            min(
                excellent,
                key=lambda r: (
                    -r["schedule_reference_mean_win_probability"],
                    r["team_id"],
                ),
            ),
        )
    middling = [r for r in rated if 0.3 <= r["win_fraction"] <= 0.6 and r["games"] >= 5]
    if middling:
        add(
            "middling_record_difficult_schedule",
            min(
                middling,
                key=lambda r: (
                    r["schedule_reference_mean_win_probability"],
                    r["team_id"],
                ),
            ),
        )
    rules = {
        "competitive_elite_loss": lambda g: (
            -7 <= g["margin"] < 0
            and g["opponent_qop_ordinal"] is not None
            and g["opponent_qop_ordinal"] <= 10
        ),
        "narrow_win_weaker_opponent": lambda g: (
            0 < g["margin"] <= 7 and g["context_win_probability"] >= 0.8
        ),
        "dominant_win_overmatched_opponent": lambda g: (
            g["margin"] >= 28 and g["context_win_probability"] >= 0.8
        ),
    }
    for label, rule in rules.items():
        eligible = [g for g in games if rule(g)]
        if eligible:
            chosen = min(
                eligible, key=lambda g: (-abs(g["margin"]), g["team_id"], g["game_id"])
            )
            add(label, next(r for r in rated if r["team_id"] == chosen["team_id"]))
            anchors[label + "_game"] = chosen["game_id"]
    add(
        "endpoint_disagreement",
        min(
            rated, key=lambda r: (-abs(r["qor_minus_qop_expected_rank"]), r["team_id"])
        ),
    )
    for boundary in (10, 12, 25):
        add(
            f"boundary_{boundary}",
            min(rated, key=lambda r: (abs(r["qor_ordinal"] - boundary), r["team_id"])),
        )
    # Two matched-record pairs to examine middle ambiguity and weak uncertainty.
    for label, low, high in (
        ("middle_schedule_pair", 41, 95),
        ("weak_schedule_pair", 96, 138),
    ):
        group = [r for r in rated if low <= r["qop_ordinal"] <= high]
        pairs = [
            (a, b)
            for a, b in itertools.combinations(group, 2)
            if a["wins"] == b["wins"] and a["losses"] == b["losses"]
        ]
        if pairs:
            pair = min(
                pairs,
                key=lambda p: (
                    -abs(
                        p[0]["schedule_reference_mean_win_probability"]
                        - p[1]["schedule_reference_mean_win_probability"]
                    ),
                    p[0]["team_id"],
                    p[1]["team_id"],
                ),
            )
            for row in pair:
                add(label, row)
    return [
        dict(next(r for r in rated if r["team_id"] == team_id), categories=categories)
        for team_id, categories in sorted(selections.items())
    ], anchors


def audit_snapshot(
    root: Path, published: Any
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    bundle = _input_bundle(root, published)
    context, qop, qor, teams, rows, games, config, metadata, fbs, likelihood, prior = (
        bundle
    )
    validate_rows(rows, metadata)
    qop_pmfs, qor_pmfs, counts = _baseline_rankings(published, fbs, games)
    # Validate reference distributions, including all retained FCS variables.
    context_pmfs = _pmfs(context.directory / "posterior_pmfs.csv")
    prior_digest = posterior_pmfs_sha256({t.team_id: t.prior for t in teams})
    if metadata.get("prior_pmfs_sha256", prior_digest) != prior_digest:
        raise ValueError("frozen prior PMFs hash mismatch")
    for team in teams:
        for pmf in (context_pmfs[team.team_id],):
            if (
                len(pmf) != len(team.prior)
                or not np.isfinite(pmf).all()
                or np.min(pmf) < 0
                or not np.isclose(pmf.sum(), 1, atol=1e-9, rtol=0)
            ):
                raise ValueError("invalid reference PMF")
    for values in (qop_pmfs, qor_pmfs):
        for pmf in values.values():
            if (
                not np.isfinite(pmf).all()
                or np.min(pmf) < 0
                or not np.isclose(pmf.sum(), 1, atol=1e-9, rtol=0)
            ):
                raise ValueError("invalid endpoint PMF")
    diagnostics = _read_json(context.directory / "diagnostics.json")
    if not diagnostics["converged"]:
        raise ValueError("reference did not converge")
    if metadata["historical_likelihood_version"] != "V1":
        raise ValueError("unsupported likelihood")
    retrievals = metadata.get("source_retrieval_times", {})
    cutoff = datetime.fromisoformat(metadata["effective_cutoff"])
    late = {
        key: max(0, (datetime.fromisoformat(value) - cutoff).total_seconds())
        for key, value in retrievals.items()
    }
    artifact_paths = [prior]
    for snapshot in (context, qop, qor):
        artifact_paths.extend(
            snapshot.directory / name
            for name in (
                "metadata.json",
                "included_games.csv",
                "rankings.csv",
                "posterior_pmfs.csv",
                "diagnostics.json",
            )
        )
    audit = {
        "season": int(metadata["season"]),
        "publication_slot": published.slot_id,
        "display_label": published.display_label,
        "publication_status": "official",
        "context_snapshot_id": context.snapshot_id,
        "context_source": str(context.directory.relative_to(root)),
        "qop_source": str(qop.directory.relative_to(root)),
        "qor_source": str(qor.directory.relative_to(root)),
        "selected": published.slot_id in SELECTED_SLOTS,
        "information_phase": "regular_season_partial",
        "suitable_for_fixed_input_comparison": True,
        "requested_cutoff": metadata["requested_cutoff"],
        "effective_cutoff": metadata["effective_cutoff"],
        "source_retrieval_times": retrievals,
        "retrieval_seconds_after_cutoff": late,
        "source_retrieved_at_contract": metadata.get("source_retrieved_at_contract"),
        "source_response_hashes": metadata["source_response_hashes"],
        "game_corpus_sha256": metadata["game_corpus_sha256"],
        "included_game_count": len(games),
        "included_game_rows_sha256": included_game_rows_sha256(rows),
        "prior_artifact": str(prior.relative_to(root)),
        "prior_model_version": metadata["prior_model_version"],
        "prior_pmfs_sha256": posterior_pmfs_sha256({t.team_id: t.prior for t in teams}),
        "historical_likelihood_version": metadata["historical_likelihood_version"],
        "historical_likelihood_sha256": metadata["historical_likelihood_sha256"],
        "inference_configuration": config,
        "reference_runtime_seconds": diagnostics["runtime_seconds"],
        "reference_iterations": diagnostics["iterations"],
        "fbs_team_ids": sorted(t.team_id for t in fbs),
        "rated_fbs_team_ids": sorted(t.team_id for t in fbs if counts[t.team_id]),
        "network_team_ids": sorted(t.team_id for t in teams),
        "fcs_population_size": metadata.get("fcs_population_size"),
        "fcs_fallback_team_ids": metadata.get("fcs_fallback_team_ids"),
        "historical_evidence_correction": metadata.get(
            "historical_evidence_correction"
        ),
        "campbell_corrected": True,
        "artifact_sha256": {
            str(p.relative_to(root)): sha256(p) for p in artifact_paths
        },
    }
    reference_orders = []
    for snapshot in (qop, qor):
        order = {
            row["team_id"]: int(row["display_rank"])
            for row in _read_rows(snapshot.directory / "rankings.csv")
            if row["display_rank"]
        }
        rated = {t.team_id for t in fbs if counts[t.team_id]}
        if set(order) != rated or sorted(order.values()) != list(
            range(1, len(rated) + 1)
        ):
            raise ValueError("invalid retained reference display order")
        reference_orders.append(order)
    return audit, (
        teams,
        games,
        context_pmfs,
        qop_pmfs,
        qor_pmfs,
        counts,
        likelihood,
        *reference_orders,
    )


def build_design(root: Path, output: Path) -> dict[str, Any]:
    """Audit inputs and materialize the frozen specification; never infer candidates."""
    output.mkdir(parents=True, exist_ok=True)
    audits = []
    descriptors = []
    evidence = []
    latest_rows = latest_games = None
    for published in _published_inputs(root):
        audit, payload = audit_snapshot(root, published)
        audits.append(audit)
        if not audit["selected"]:
            continue
        rows, game_rows = reference_descriptors(*payload)
        for collection in (rows, game_rows):
            for row in collection:
                row.update(season=2026, publication_slot=published.slot_id)
        descriptors.extend(rows)
        evidence.extend(game_rows)
        latest_rows, latest_games = rows, game_rows
    if tuple(a["publication_slot"] for a in audits if a["selected"]) != SELECTED_SLOTS:
        raise ValueError("missing frozen selected snapshot")
    assert latest_rows is not None and latest_games is not None
    cases, anchors = select_cases(latest_rows, latest_games)
    # Follow the same frozen IDs at all three cutoffs, including sparse early evidence.
    case_inventory = [
        dict(
            row,
            categories=case["categories"],
            questions=[
                CASE_QUESTIONS.get(
                    category,
                    "Does this boundary crossing reflect meaningful evidence or close teams?",
                )
                for category in case["categories"]
            ],
            selection_publication_slot=SELECTED_SLOTS[-1],
        )
        for case in cases
        for row in descriptors
        if row["team_id"] == case["team_id"]
    ]
    diagnostics = []
    for profile in ("elite", "weak"):
        team_id = anchors[profile]
        eligible = [
            g
            for g in latest_games
            if g["team_id"] == team_id and g["opponent_subdivision"] == "fbs"
        ]
        chosen = min(
            eligible,
            key=lambda g: (abs(g["context_win_probability"] - 0.5), g["game_id"]),
        )
        diagnostics.append(
            dict(
                chosen,
                procedure="four_margin_states",
                profile=profile,
                margins=[-3, 3, 10, 35],
            )
        )
    removals = {g["game_id"] for g in diagnostics}
    removals.update(anchors[key] for key in anchors if key.endswith("_game"))
    # Two margin-state games and three category games give at most five removals.
    removal_ids = sorted(removals)
    for name, rows in (
        ("reference_descriptors.csv", descriptors),
        ("reference_games.csv", evidence),
    ):
        _write_csv(output / name, rows, list(rows[0]))
    (output / "publication_config.json").write_bytes(
        (root / "site/publish_config.json").read_bytes()
    )
    (output / "cases.json").write_text(
        json.dumps(case_inventory, indent=2, sort_keys=True) + "\n"
    )
    historical_paths = [
        "data/processed/cfbd/games.csv",
        "data/processed/preseason/context/predictions.csv",
        "data/processed/preseason/context_v1_3_candidate/predictions.csv",
        "data/processed/context_db_repair_172/historical_priors/2025.csv",
        "data/processed/context_db_repair_172/historical_models/2025.json",
        "data/processed/posterior_backtest/2025_rolling.json",
        "data/research/matched_history_context_issue_200/target_snapshot_manifest.csv",
    ]
    corpus_seasons = sorted(
        {r["season"] for r in _read_rows(root / historical_paths[0])}
    )
    if corpus_seasons != ["2026"] or list(
        (root / "data/processed/snapshots/2025").glob("**/metadata.json")
    ):
        raise ValueError("historical availability changed; amend the design explicitly")
    common_paths = [
        "data/processed/posterior/historical_likelihood_v1.json",
        "data/processed/preseason/context_v1_4/annual/2026/fitted_instance.json",
        "data/processed/issue_238_historical_repair/lineage_audit.json",
        "src/gippyrank/research/issue_263_real_season_design.py",
        "scripts/design_issue_263_real_seasons.py",
        DOCUMENT,
        "src/gippyrank/research/quality_of_results.py",
        "src/gippyrank/posterior/engine.py",
        "src/gippyrank/posterior/snapshots.py",
        "src/gippyrank/posterior/predictive.py",
        "src/gippyrank/quality_of_record.py",
        "src/gippyrank/quality_of_record_snapshot.py",
        "src/gippyrank/performance_snapshot.py",
        "src/gippyrank/performance_v1.py",
        "src/gippyrank/research/issue_242_quality_of_results.py",
        ".python-version",
        "uv.lock",
        "pyproject.toml",
    ]
    spec = {
        "schema_version": "1.0",
        "issue": 263,
        "base_source_revision": BASE_REVISION,
        "design_only": True,
        "candidate_inference_performed": False,
        "selected_publication_slots": list(SELECTED_SLOTS),
        "snapshot_audit": audits,
        "unavailable_snapshots": [
            {
                "season": 2025,
                "phase": phase,
                "selected": False,
                "reason": "No cutoff-local game rows or publication/endpoint bundles retained; priors and summaries alone are insufficient",
                "fallback": "Execute only frozen partial 2026 study; completed-season validation requires a separate provenance prerequisite",
            }
            for phase in ("pre_bowl", "final")
        ],
        "historical_audit": {
            "game_corpus_seasons": corpus_seasons,
            "context_2025_prior_rows": sum(
                r["season"] == "2025" for r in _read_rows(root / historical_paths[1])
            ),
            "snapshot_2025_metadata_count": 0,
        },
        "candidate_configuration": {
            "margin_weights": list(MARGIN_WEIGHTS),
            "factor": "P(sign(m)|ranks)^(1-lambda) * p(m|ranks)^lambda",
            "inference_function": "gippyrank.research.quality_of_results.infer_quality_of_results",
            "full_connected_network": True,
            "prior_stripping": "existing Performance ratio; zero-game FBS uniform",
            "convergence_required": True,
            "endpoint_pmf_max_abs_tolerance": 1e-8,
            "endpoint_expected_rank_tolerance": 138e-8,
            "pmf_sum_tolerance": 1e-9,
            "endpoint_ordinal_mismatch": "report ties within mean tolerance; otherwise fail",
            "order_reversal_check_weights": [0, 0.5, 1],
            "order_reversal_pmf_tolerance": 1e-8,
        },
        "case_selection": {
            "reference_slot": SELECTED_SLOTS[-1],
            "anchors": anchors,
            "case_team_ids": sorted(c["team_id"] for c in cases),
            "inventory_path": "cases.json",
            "selection_function": "select_cases",
            "intermediate_candidate_outputs_read": False,
            "longitudinal_slots": list(SELECTED_SLOTS),
        },
        "game_diagnostics": {
            "publication_slot": SELECTED_SLOTS[-1],
            "counterfactuals": diagnostics,
            "replacement_scores": "opponent=max(actual_opponent_points,3); focal=opponent+margin",
            "removal_game_ids": removal_ids,
            "recompute_entire_network": True,
            "all_weights_for_every_diagnostic": True,
            "additive_attribution_claim": False,
        },
        "metrics": {
            "candidate_pairs": "all 10 unordered pairs; difference=lambda_high minus lambda_low; positive rank difference is worse",
            "ranking": [
                "expected and ordinal rank changes",
                "pairwise inversions",
                "top10/top25 overlap and entries/exits",
                "boundary 10/12/25 bands +/-3",
            ],
            "distributions": [
                "rank SD",
                "central 50/80/95 interval inclusive widths",
                "entropy bits",
                "TV",
                "Wasserstein1=sum absolute CDF differences",
            ],
            "population_summaries": [
                "signed mean",
                "median",
                "mean absolute",
                "p05/p95",
                "maximum absolute with team ID",
            ],
            "schedule": "standardized full preseason PMF nearest expected rank65 integrated against frozen Context opponent PMFs through V1 at actual venue; lower mean win probability is harder",
            "comparisons": "within snapshot and endpoint QoP ordinal bands 1:10,11:40,41:95,96:138; schedule terciles with ID tie break; record and near-expected-record panels",
            "uncertainty_interpretation": "descriptive marginals; no independent-team posterior superiority probability or sampling CI",
            "details": DOCUMENT,
        },
        "execution": {
            "baseline_solves": 15,
            "counterfactual_solves": 40,
            "removal_solves": 5 * len(removal_ids),
            "order_reversal_solves": 3,
            "maximum_network_solves": 58 + 5 * len(removal_ids),
            "strategy": "serial snapshots and solves, reuse factor surfaces within snapshot, cold-start BP, checkpoint by full input+configuration hash",
            "wall_budget_hours": 2,
            "single_solve_timeout_seconds": 600,
            "rss_budget_gib": 4,
            "on_failure": "retain diagnostic failure, stop affected snapshot; no relaxed tolerance, changed weight or silent replacement",
            "planned_outputs": [
                "manifest.json",
                "rankings.csv",
                "posterior_pmfs/*.csv.gz",
                "population_contrasts.csv",
                "accomplishment_panels.csv",
                "case_results.csv",
                "game_diagnostics.csv",
                "convergence.csv",
                "endpoint_parity.json",
                "report.md",
            ],
            "implementation_plan": DOCUMENT,
        },
        "input_sha256": {p: sha256(root / p) for p in common_paths},
        # These mutable inventories document what existed at design time. The
        # frozen snapshot rows, not today's inventories, drive execution.
        "audit_only_input_sha256": {
            p: sha256(root / p) for p in ["site/publish_config.json", *historical_paths]
        },
        "artifact_sha256": {
            p.name: sha256(p)
            for p in sorted(output.iterdir())
            if p.name
            in {
                "reference_descriptors.csv",
                "reference_games.csv",
                "cases.json",
                "publication_config.json",
            }
        },
    }
    (output / "protocol.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "issue": 263,
        "artifact_sha256": {
            name: sha256(output / name)
            for name in (
                "protocol.json",
                "cases.json",
                "reference_descriptors.csv",
                "reference_games.csv",
                "publication_config.json",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return spec


def check_design(root: Path, output: Path) -> dict[str, Any]:
    """Fail closed on input or inventory drift without consulting newer publications."""
    manifest = _read_json(output / "manifest.json")
    for path, expected in manifest["artifact_sha256"].items():
        if not (output / path).is_file() or sha256(output / path) != expected:
            raise ValueError(f"frozen design artifact hash mismatch: {path}")
    spec = _read_json(output / "protocol.json")
    for path, expected in spec["input_sha256"].items():
        if not (root / path).is_file() or sha256(root / path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    for snapshot in spec["snapshot_audit"]:
        for path, expected in snapshot["artifact_sha256"].items():
            if not (root / path).is_file() or sha256(root / path) != expected:
                raise ValueError(f"frozen snapshot hash mismatch: {path}")
    for path, expected in spec["artifact_sha256"].items():
        if not (output / path).is_file() or sha256(output / path) != expected:
            raise ValueError(f"frozen design artifact hash mismatch: {path}")
    if spec["candidate_configuration"]["margin_weights"] != list(MARGIN_WEIGHTS):
        raise ValueError("candidate population changed")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Explicitly regenerate design; default only verifies hashes",
    )
    args = parser.parse_args()
    output = args.output_dir or args.root / ARTIFACT_DIR
    spec = (
        build_design(args.root, output)
        if args.write
        else check_design(args.root, output)
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "selected_snapshots": len(spec["selected_publication_slots"]),
                "case_teams": len(spec["case_selection"]["case_team_ids"]),
                "candidate_inference_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
