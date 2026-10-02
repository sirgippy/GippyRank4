"""Replay frozen rolling Context 1.3 with positive net location moderation.

The alpha grid is fixed in code. No evaluated outcome enters prior construction,
and every baseline parity check completes before moderated posterior inference.
Run with the historical corpus in a local WSL checkout, for example::

    uv run python scripts/experiment_context_positive_net_moderation.py \
      --targets /home/gippy/src/GippyRank4/data/processed/modeling/team_season_rank_distributions.csv \
      --raw-games /home/gippy/src/GippyRank4/data/raw/cfbd/games
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study_context_history_crossover as crossover
import study_history_context_posterior as historical

from gippyrank.context_positive_net_moderation import (
    ALPHAS,
    LocationParts,
    moderated_location_points,
    parts_from_fitted_contributions,
    require_rolling_origin,
)
from gippyrank.posterior.engine import Team, infer_posterior
from gippyrank.posterior.snapshots import (
    _scheduled_future_fcs_rows,
    add_fcs_fallbacks,
    filter_games,
    load_pinned_likelihood,
)

OUT = ROOT / "data/processed/context_positive_net_moderation"
BASE = ROOT / "data/processed/context_history_crossover"
DIAGNOSTIC = ROOT / "data/processed/context_location_error_diagnostics"
STUDY = ROOT / "data/processed/context_location_error_study"
SEASONS = (2022, 2023, 2024, 2025)
TOLERANCE = 1e-8
PMF_TOLERANCE = 1e-12
DELTA_TOLERANCE = 1e-10
METRICS = (
    "nll",
    "crps",
    "expected_rank",
    "expected_rank_error",
    "interval_80_width",
    "interval_80_target_mass",
    "entropy",
    "rank_standard_deviation",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_index(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in result:
            raise ValueError(f"duplicate source row for {keys}: {key}")
        result[key] = row
    return result


def check_hash(path: Path, expected: str) -> str:
    actual = crossover.sha256(path)
    if actual != expected:
        raise ValueError(f"source lineage mismatch for {path}: {actual} != {expected}")
    return actual


def load_frozen_priors() -> tuple[
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[tuple[int, str], LocationParts],
    dict[tuple[int, str], str],
    dict[str, object],
    dict[str, str],
    dict[str, float],
]:
    """Construct all candidate priors from outcome-free fitted fields only."""
    base_provenance = json.loads((BASE / "provenance.json").read_text())
    diagnostic_provenance = json.loads((DIAGNOSTIC / "provenance.json").read_text())
    study_provenance = json.loads((STUDY / "provenance.json").read_text())
    if tuple(base_provenance["target_seasons"]) != SEASONS:
        raise ValueError("baseline seasons drifted")
    if base_provenance["inference"] != crossover.INFERENCE:
        raise ValueError("inference settings drifted from retained baseline")
    if any(
        row["passed"] != "True"
        for row in read_csv(BASE / "baseline_reproduction_checks.csv")
    ):
        raise ValueError("retained Context/History baseline reproduction failed")

    source_hashes = {}
    for relative, expected in (
        (
            "data/processed/context_history_crossover/team_season_prior_decomposition.csv",
            diagnostic_provenance["source_hashes"][
                "data/processed/context_history_crossover/team_season_prior_decomposition.csv"
            ],
        ),
        (
            "data/processed/context_location_error_diagnostics/team_seasons.csv",
            study_provenance["source_hashes"][
                "data/processed/context_location_error_diagnostics/team_seasons.csv"
            ],
        ),
        (
            "data/processed/context_location_error_study/contribution_diagnostics.csv",
            study_provenance["output_hashes"]["contribution_diagnostics.csv"],
        ),
    ):
        source_hashes[relative] = check_hash(ROOT / relative, expected)
    for relative in (
        "data/processed/context_history_crossover/hybrid_prior_results.csv",
        "data/processed/context_history_crossover/hybrid_posterior_team_results.csv",
        "data/processed/context_history_crossover/hybrid_posterior_summary.csv",
        "data/processed/context_history_crossover/arm_evidence_and_convergence.csv",
        "data/processed/context_history_crossover/provenance.json",
        "data/processed/context_location_error_diagnostics/provenance.json",
        "data/processed/context_location_error_study/provenance.json",
        "data/processed/context_history_crossover/baseline_reproduction_checks.csv",
        "data/processed/posterior/historical_likelihood_v1.json",
    ):
        source_hashes[relative] = crossover.sha256(ROOT / relative)
    if (
        source_hashes["data/processed/context_history_crossover/provenance.json"]
        != diagnostic_provenance["source_hashes"][
            "data/processed/context_history_crossover/provenance.json"
        ]
    ):
        raise ValueError(
            "baseline provenance changed since contribution reconstruction"
        )
    if (
        source_hashes["data/processed/posterior/historical_likelihood_v1.json"]
        != base_provenance["likelihood"]["sha256"]
    ):
        raise ValueError("retained likelihood changed")
    for season in SEASONS:
        relative = f"data/processed/posterior_backtest/{season}_rolling.json"
        source_hashes[relative] = check_hash(
            ROOT / relative, base_provenance["sources_sha256"][relative]
        )

    decomposition = checked_index(
        read_csv(BASE / "team_season_prior_decomposition.csv"), ("season", "team_id")
    )
    diagnostic = checked_index(
        read_csv(DIAGNOSTIC / "team_seasons.csv"), ("season", "team_id")
    )
    contribution = checked_index(
        read_csv(STUDY / "contribution_diagnostics.csv"), ("season", "team_id")
    )
    prior_rows = checked_index(
        [
            row
            for row in read_csv(BASE / "hybrid_prior_results.csv")
            if row["arm"] in {"CC", "HH"}
        ],
        ("season", "team_id", "arm"),
    )
    if len(decomposition) != 534 or set(decomposition) != set(diagnostic):
        raise ValueError("frozen prior/diagnostic populations differ from 534 teams")
    if len(prior_rows) != 2 * len(decomposition):
        raise ValueError("retained CC/HH priors are incomplete")

    priors: dict[int, dict[str, dict[str, np.ndarray]]] = {
        season: {
            "context_1_3": {},
            "history_1_1": {},
            **{f"alpha_{alpha:.2f}": {} for alpha in ALPHAS},
        }
        for season in SEASONS
    }
    parts: dict[tuple[int, str], LocationParts] = {}
    names: dict[tuple[int, str], str] = {}
    parity = {
        "max_location_reconstruction_error": 0.0,
        "max_prior_pmf_error": 0.0,
        "max_contribution_study_error": 0.0,
    }
    fitted_keys = set()
    for season_text, team_id in sorted(
        decomposition, key=lambda key: (int(key[0]), key[1])
    ):
        season = int(season_text)
        if season not in SEASONS:
            raise ValueError(f"unplanned source season: {season}")
        key = season_text, team_id
        prior = decomposition[key]
        diag = diagnostic[key]
        status = prior["component_status"]
        if status != diag["component_status"]:
            raise ValueError(f"{key}: fitted/fallback status differs")
        names[season, team_id] = prior["team_name"]
        for arm, model in (("CC", "context_1_3"), ("HH", "history_1_1")):
            retained = prior_rows[season_text, team_id, arm]
            values = np.asarray(json.loads(retained["prior_pmf"]), dtype=float)
            if crossover.sha256_json(values.tolist()) != retained["prior_pmf_sha256"]:
                raise ValueError(f"{key} {arm}: retained PMF hash mismatch")
            if len(values) != int(prior["target_population"]):
                raise ValueError(f"{key} {arm}: retained rank support differs")
            priors[season][model][team_id] = values
        context_pmf = priors[season]["context_1_3"][team_id]
        if status == "fitted":
            fitted_keys.add(key)
            model = base_provenance["models"][season_text]
            require_rolling_origin(
                season,
                int(diag["context_training_cutoff"]),
                int(diag["history_training_cutoff"]),
                model["context_instance"],
                model["history_instance"],
            )
            if (
                diag["context_model_sha256"] != model["context_model_sha256"]
                or diag["history_model_sha256"] != model["history_model_sha256"]
            ):
                raise ValueError(f"{key}: fitted model hashes differ")
            if (
                diag["context_prior_pmf_sha256"] != prior["context_prior_pmf_sha256"]
                or diag["history_prior_pmf_sha256"] != prior["history_prior_pmf_sha256"]
            ):
                raise ValueError(f"{key}: frozen prior lineage differs")
            locations = np.asarray(
                json.loads(prior["context_conditional_location_points"]), dtype=float
            )
            fitted = parts_from_fitted_contributions(diag, locations)
            parts[season, team_id] = fitted
            for observed in (
                float(prior["context_location_center"]),
                float(diag["context_location_center"]),
                float(diag["context_fitted_location_center"]),
            ):
                error = abs(float(locations.mean()) - observed)
                parity["max_location_reconstruction_error"] = max(
                    parity["max_location_reconstruction_error"], error
                )
                if error > TOLERANCE:
                    raise ValueError(f"{key}: source centers do not reconstruct")
            study_row = contribution[key]
            for name, observed in (
                ("context_only_subtotal", fitted.context_only_subtotal),
                ("history_derived_subtotal", fitted.history_derived_subtotal),
            ):
                error = abs(observed - float(study_row[name]))
                parity["max_contribution_study_error"] = max(
                    parity["max_contribution_study_error"], error
                )
                if error > TOLERANCE:
                    raise ValueError(f"{key}: issue 154 contribution differs: {name}")
            rebuilt = crossover.normal_mixture_pmf(
                locations,
                float(prior["context_conditional_residual_scale"]),
                int(prior["target_population"]),
            )
            error = float(np.max(np.abs(rebuilt - context_pmf)))
            parity["max_prior_pmf_error"] = max(parity["max_prior_pmf_error"], error)
            if error > PMF_TOLERANCE:
                raise ValueError(f"{key}: alpha 1 does not reconstruct Context PMF")
        elif status != "cold_start_fallback":
            raise ValueError(f"{key}: unknown component status {status}")
        for alpha in ALPHAS:
            model = f"alpha_{alpha:.2f}"
            if (
                status == "cold_start_fallback"
                or alpha == 1.0
                or parts[season, team_id].context_only_subtotal <= 0
            ):
                values = context_pmf.copy()
            else:
                values = crossover.normal_mixture_pmf(
                    moderated_location_points(parts[season, team_id], alpha),
                    float(prior["context_conditional_residual_scale"]),
                    int(prior["target_population"]),
                )
            priors[season][model][team_id] = values
    if len(fitted_keys) != 528 or set(contribution) != fitted_keys:
        raise ValueError("fitted contribution population differs from 528")
    for season in SEASONS:
        expected_count = base_provenance["component_population"][str(season)][
            "all_fbs_target_teams"
        ]
        if any(len(arm) != expected_count for arm in priors[season].values()):
            raise ValueError(f"{season}: candidate prior populations differ")
        if any(
            not np.array_equal(
                priors[season]["context_1_3"][team_id],
                priors[season]["alpha_1.00"][team_id],
            )
            for team_id in priors[season]["context_1_3"]
        ):
            raise ValueError(f"{season}: alpha 1 prior is not bit-identical to Context")
    return priors, parts, names, base_provenance, source_hashes, parity


@dataclass(frozen=True)
class CheckpointEvidence:
    season: int
    checkpoint: int
    cutoff: datetime
    games: list
    included: list[dict[str, str]]
    future_fcs: list[dict[str, str]]
    fcs_population: int
    expected: dict[str, str]


def make_evidence(
    game_root: Path, targets_path: Path, base_provenance: dict[str, object]
) -> list[CheckpointEvidence]:
    retained = checked_index(
        [
            row
            for row in read_csv(BASE / "arm_evidence_and_convergence.csv")
            if row["arm"] in {"CC", "HH"}
        ],
        ("season", "checkpoint", "arm"),
    )
    result = []
    for season in SEASONS:
        fcs_population = historical.fcs_population(season, targets_path)
        for point in crossover.checkpoint_inputs(game_root, season):
            checkpoint = int(point["checkpoint"])
            cutoff = datetime.fromisoformat(str(point["cutoff"]))
            games, included, excluded_lower, _ = filter_games(
                game_root, season, cutoff, "weekly"
            )
            games, included, audit = historical.cutoff_safe_games(
                games, included, cutoff
            )
            cc = retained[str(season), str(checkpoint), "CC"]
            hh = retained[str(season), str(checkpoint), "HH"]
            fields = (
                "cutoff",
                "included_games",
                "included_game_ids_sha256",
                "included_game_rows_sha256",
                "excluded_near_cutoff_games",
                "excluded_lower_division_games",
                "fcs_fallback_count",
                "fcs_fallback_ids_sha256",
                "total_inference_team_count",
                "likelihood_sha256",
                "inference_max_iterations",
                "inference_tolerance",
                "inference_damping",
            )
            if any(cc[field] != hh[field] for field in fields):
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: retained evidence differs by baseline"
                )
            observed = {
                "cutoff": cutoff.isoformat(),
                "included_games": str(len(included)),
                "included_game_ids_sha256": crossover.sha256_json(
                    tuple(sorted((str(row["id"]) for row in included), key=int))
                ),
                "included_game_rows_sha256": crossover.stable_evidence_hash(included),
                "excluded_near_cutoff_games": str(len(audit)),
                "excluded_lower_division_games": str(excluded_lower),
                "likelihood_sha256": base_provenance["likelihood"]["sha256"],
            }
            if any(observed[field] != cc[field] for field in observed):
                raise ValueError(
                    f"{season} checkpoint {checkpoint}: evidence replay differs in {observed}"
                )
            result.append(
                CheckpointEvidence(
                    season,
                    checkpoint,
                    cutoff,
                    games,
                    included,
                    _scheduled_future_fcs_rows(game_root, season, cutoff, "weekly"),
                    fcs_population,
                    cc,
                )
            )
    if len(result) != 28:
        raise ValueError("expected seven shared checkpoints in four seasons")
    return result


def infer_arm(
    evidence: CheckpointEvidence,
    model: str,
    prior_map: dict[str, np.ndarray],
    names: dict[tuple[int, str], str],
    targets: dict[str, np.ndarray],
    parts: dict[tuple[int, str], LocationParts],
    likelihood,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    season = evidence.season
    teams = [
        Team(team_id, names[season, team_id], "fbs", pmf.copy())
        for team_id, pmf in sorted(prior_map.items())
    ]
    meta = {team_id: {"team_name": names[season, team_id]} for team_id in prior_map}
    teams, fallback_ids = add_fcs_fallbacks(
        teams, meta, evidence.included, evidence.fcs_population, evidence.future_fcs
    )
    expected = evidence.expected
    if (
        str(len(fallback_ids)) != expected["fcs_fallback_count"]
        or crossover.sha256_json(sorted(fallback_ids))
        != expected["fcs_fallback_ids_sha256"]
        or str(len(teams)) != expected["total_inference_team_count"]
    ):
        raise ValueError(
            f"{season} checkpoint {evidence.checkpoint} {model}: FCS network differs"
        )
    result = infer_posterior(teams, evidence.games, likelihood, **crossover.INFERENCE)
    if not result.converged or set(result.pmfs) != {team.team_id for team in teams}:
        raise ValueError(
            f"{season} checkpoint {evidence.checkpoint} {model}: posterior failed"
        )
    rows = []
    for team_id, prior in sorted(prior_map.items()):
        if team_id not in targets:
            raise ValueError(f"{season} {team_id}: no target")
        truth = targets[team_id]
        prior_score = crossover.score_pmf(prior, truth)
        post_score = crossover.score_pmf(result.pmfs[team_id], truth)
        rows.append(
            {
                "season": season,
                "checkpoint": evidence.checkpoint,
                "cutoff": evidence.cutoff.isoformat(),
                "period": evidence.cutoff.strftime("%B").lower(),
                "team_id": team_id,
                "team_name": names[season, team_id],
                "model": model,
                "alpha": model.removeprefix("alpha_")
                if model.startswith("alpha_")
                else "",
                "component_status": "fitted"
                if (season, team_id) in parts
                else "cold_start_fallback",
                "context_only_subtotal": parts[season, team_id].context_only_subtotal
                if (season, team_id) in parts
                else "",
                "posterior_pmf_sha256": crossover.sha256_json(
                    result.pmfs[team_id].tolist()
                ),
                **{f"prior_{name}": prior_score[name] for name in METRICS},
                **{f"posterior_{name}": post_score[name] for name in METRICS},
            }
        )
    return rows, result.pmfs


def compare_retained(
    rows: list[dict[str, object]],
    retained: dict[tuple[str, ...], dict[str, str]],
    expected_arm: str,
) -> float:
    maximum = 0.0
    for row in rows:
        key = (
            str(row["season"]),
            str(row["checkpoint"]),
            str(row["team_id"]),
            expected_arm,
        )
        old = retained[key]
        for stage in ("prior", "posterior"):
            for name in METRICS:
                error = abs(
                    float(row[f"{stage}_{name}"]) - float(old[f"{stage}_{name}"])
                )
                maximum = max(maximum, error)
                if error > TOLERANCE:
                    raise ValueError(
                        f"{key}: retained {stage}_{name} parity failed: {error}"
                    )
    return maximum


def retained_history_rows(
    evidence: CheckpointEvidence,
    retained: dict[tuple[str, ...], dict[str, str]],
    team_ids: set[str],
) -> list[dict[str, object]]:
    rows = []
    for team_id in sorted(team_ids):
        old = retained[str(evidence.season), str(evidence.checkpoint), team_id, "HH"]
        rows.append(
            {
                "season": evidence.season,
                "checkpoint": evidence.checkpoint,
                "cutoff": evidence.cutoff.isoformat(),
                "period": evidence.cutoff.strftime("%B").lower(),
                "team_id": team_id,
                "team_name": old["team_name"],
                "model": "history_1_1",
                "alpha": "",
                "component_status": old["component_status"],
                "context_only_subtotal": "",
                "posterior_pmf_sha256": "",
                **{
                    f"{stage}_{name}": float(old[f"{stage}_{name}"])
                    for stage in ("prior", "posterior")
                    for name in METRICS
                },
            }
        )
    return rows


def attach_deltas(rows: list[dict[str, object]]) -> None:
    indexed = {
        (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
            str(row["model"]),
        ): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("duplicate modeled checkpoint/team")
    for row in rows:
        season, checkpoint, team_id = (
            int(row["season"]),
            int(row["checkpoint"]),
            str(row["team_id"]),
        )
        context = indexed[season, checkpoint, team_id, "context_1_3"]
        history = indexed[season, checkpoint, team_id, "history_1_1"]
        row["prior_nll_delta_vs_context"] = float(row["prior_nll"]) - float(
            context["prior_nll"]
        )
        row["prior_expected_rank_error_delta_vs_context"] = float(
            row["prior_expected_rank_error"]
        ) - float(context["prior_expected_rank_error"])
        row["posterior_nll_delta_vs_context"] = float(row["posterior_nll"]) - float(
            context["posterior_nll"]
        )
        row["posterior_nll_delta_vs_history"] = float(row["posterior_nll"]) - float(
            history["posterior_nll"]
        )


def summarize_checkpoint(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Use one equal-weight row per team-season, then pool 2023-25 only."""
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        season = int(row["season"])
        checkpoint = int(row["checkpoint"])
        model = str(row["model"])
        groups[str(season), checkpoint, model].append(row)
        if season in (2023, 2024, 2025):
            groups["2023-2025", checkpoint, model].append(row)
    output = []
    for (cohort, checkpoint, model), group in sorted(groups.items()):
        group.sort(key=lambda row: (int(row["season"]), str(row["team_id"])))
        deltas = np.asarray(
            [float(row["posterior_nll_delta_vs_context"]) for row in group]
        )
        output.append(
            {
                "cohort": cohort,
                "checkpoint": checkpoint,
                "period": group[0]["period"]
                if len({row["period"] for row in group}) == 1
                else "pooled",
                "cutoff": group[0]["cutoff"]
                if len({row["cutoff"] for row in group}) == 1
                else "",
                "model": model,
                "alpha": group[0]["alpha"],
                "team_seasons": len(group),
                "weighting": "equal team-season",
                **{
                    f"mean_{stage}_{name}": float(
                        np.mean([float(row[f"{stage}_{name}"]) for row in group])
                    )
                    for stage in ("prior", "posterior")
                    for name in (
                        "nll",
                        "crps",
                        "expected_rank_error",
                        "interval_80_width",
                        "interval_80_target_mass",
                    )
                },
                "mean_prior_nll_delta_vs_context": float(
                    np.mean([float(row["prior_nll_delta_vs_context"]) for row in group])
                ),
                "mean_prior_expected_rank_error_delta_vs_context": float(
                    np.mean(
                        [
                            float(row["prior_expected_rank_error_delta_vs_context"])
                            for row in group
                        ]
                    )
                ),
                "mean_posterior_nll_delta_vs_context": float(deltas.mean()),
                "mean_posterior_nll_delta_vs_history": float(
                    np.mean(
                        [float(row["posterior_nll_delta_vs_history"]) for row in group]
                    )
                ),
                "improved_vs_context_count": int(np.sum(deltas < -DELTA_TOLERANCE)),
                "worsened_vs_context_count": int(np.sum(deltas > DELTA_TOLERANCE)),
                "tied_vs_context_count": int(np.sum(np.abs(deltas) <= DELTA_TOLERANCE)),
                "improved_vs_context_fraction": float(
                    np.mean(deltas < -DELTA_TOLERANCE)
                ),
                "worsened_vs_context_fraction": float(
                    np.mean(deltas > DELTA_TOLERANCE)
                ),
            }
        )
    return output


def concentration(
    rows: list[dict[str, object]], cohort: str, model: str
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and (
            str(row["season"]) == cohort
            if cohort != "2023-2025"
            else int(row["season"]) in (2023, 2024, 2025)
        )
    ]
    gains = np.asarray(
        [-float(row["posterior_nll_delta_vs_context"]) for row in selected]
    )
    top_count = math.ceil(len(selected) * 0.1)
    positive = float(np.maximum(gains, 0).sum())
    negative = float(np.maximum(-gains, 0).sum())
    largest = sorted(
        selected, key=lambda row: -float(row["posterior_nll_delta_vs_context"])
    )[:5]
    worst = sorted(
        selected, key=lambda row: float(row["posterior_nll_delta_vs_context"])
    )[:5]
    return {
        "team_seasons": len(selected),
        "mean_nll_gain_vs_context": float(gains.mean()),
        "median_nll_gain_vs_context": float(np.median(gains)),
        "gross_positive_gain": positive,
        "gross_worsening": negative,
        "top_10_percent_share_of_positive_gain": float(
            np.sort(np.maximum(gains, 0))[-top_count:].sum() / positive
        )
        if positive
        else 0.0,
        "worst_10_percent_share_of_gross_worsening": float(
            np.sort(np.maximum(-gains, 0))[-top_count:].sum() / negative
        )
        if negative
        else 0.0,
        "largest_gains": [
            {
                "season": row["season"],
                "team": row["team_name"],
                "nll_gain": -float(row["posterior_nll_delta_vs_context"]),
            }
            for row in worst
        ],
        "largest_losses": [
            {
                "season": row["season"],
                "team": row["team_name"],
                "nll_loss": float(row["posterior_nll_delta_vs_context"]),
            }
            for row in largest
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=historical.TARGETS)
    parser.add_argument("--raw-games", type=Path, default=ROOT / "data/raw/cfbd/games")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.targets.name != "team_season_rank_distributions.csv":
        raise ValueError("--targets must be the frozen final-rank corpus")
    priors, parts, names, base_provenance, source_hashes, parity = load_frozen_priors()
    source_hashes["external_input/team_season_rank_distributions.csv"] = check_hash(
        args.targets,
        base_provenance["sources_sha256"][
            "external_input/team_season_rank_distributions.csv"
        ],
    )
    targets = {
        season: historical.load_targets(season, args.targets) for season in SEASONS
    }
    for season in SEASONS:
        if set(targets[season]) != set(priors[season]["context_1_3"]):
            raise ValueError(
                f"{season}: final target and frozen prior populations differ"
            )
    retained = checked_index(
        [
            row
            for row in read_csv(BASE / "hybrid_posterior_team_results.csv")
            if row["arm"] in {"CC", "HH"}
        ],
        ("season", "checkpoint", "team_id", "arm"),
    )
    retained_summary = checked_index(
        [
            row
            for row in read_csv(BASE / "hybrid_posterior_summary.csv")
            if row["summary_type"] == "season_checkpoint"
            and row["population"] == "all_fbs"
            and row["arm"] == "CC"
        ],
        ("season", "checkpoint"),
    )
    if len(retained) != 2 * 534 * 7 or len(retained_summary) != 28:
        raise ValueError("retained baseline checkpoint population is incomplete")
    likelihood = load_pinned_likelihood(historical.LIKELIHOOD)
    rows: list[dict[str, object]] = []
    with historical.historical_game_root(args.raw_games) as (
        game_root,
        _,
        _,
        raw_hashes,
    ):
        for source, observed in raw_hashes.items():
            name = Path(source).name
            expected = base_provenance["sources_sha256"][f"raw_games/{name}"]
            if observed != expected:
                raise ValueError(f"raw game source differs from baseline: {name}")
            source_hashes[f"raw_games/{name}"] = observed
        evidence = make_evidence(game_root, args.targets, base_provenance)
        # Phase 1: run unchanged Context and alpha=1 independently at every
        # checkpoint. No alpha<1 posterior is evaluated until parity passes.
        parity["max_retained_team_metric_error"] = 0.0
        parity["max_posterior_pmf_error"] = 0.0
        parity["max_aggregate_metric_error"] = 0.0
        for item in evidence:
            season = item.season
            c_rows, c_post = infer_arm(
                item,
                "context_1_3",
                priors[season]["context_1_3"],
                names,
                targets[season],
                parts,
                likelihood,
            )
            a_rows, a_post = infer_arm(
                item,
                "alpha_1.00",
                priors[season]["alpha_1.00"],
                names,
                targets[season],
                parts,
                likelihood,
            )
            if set(c_post) != set(a_post):
                raise ValueError("alpha 1 posterior population differs")
            for team_id in c_post:
                error = float(np.max(np.abs(c_post[team_id] - a_post[team_id])))
                parity["max_posterior_pmf_error"] = max(
                    parity["max_posterior_pmf_error"], error
                )
                if error > PMF_TOLERANCE:
                    raise ValueError(
                        f"{season} checkpoint {item.checkpoint} {team_id}: alpha 1 posterior differs"
                    )
            for batch in (c_rows, a_rows):
                error = compare_retained(batch, retained, "CC")
                parity["max_retained_team_metric_error"] = max(
                    parity["max_retained_team_metric_error"], error
                )
            old_summary = retained_summary[str(season), str(item.checkpoint)]
            for stage in ("prior", "posterior"):
                for metric in METRICS:
                    field = f"mean_{stage}_{metric}"
                    if field not in old_summary:
                        # The retained aggregate omits some distribution
                        # fields, which still have team-level parity above.
                        continue
                    expected = float(old_summary[field])
                    observed = float(
                        np.mean([float(row[f"{stage}_{metric}"]) for row in c_rows])
                    )
                    error = abs(expected - observed)
                    parity["max_aggregate_metric_error"] = max(
                        parity["max_aggregate_metric_error"], error
                    )
                    if error > TOLERANCE:
                        raise ValueError(
                            f"{season} checkpoint {item.checkpoint}: aggregate parity failed for {stage}_{metric}"
                        )
            rows.extend(c_rows)
            rows.extend(a_rows)
            rows.extend(
                retained_history_rows(
                    item, retained, set(priors[season]["history_1_1"])
                )
            )
            print(
                f"baseline parity passed {season} checkpoint {item.checkpoint}",
                flush=True,
            )
        # Phase 2: the only model changes are the four fixed positive-only
        # location shifts. Evidence, likelihood, scale, and network stay fixed.
        for item in evidence:
            for alpha in ALPHAS[1:]:
                model = f"alpha_{alpha:.2f}"
                batch, _ = infer_arm(
                    item,
                    model,
                    priors[item.season][model],
                    names,
                    targets[item.season],
                    parts,
                    likelihood,
                )
                rows.extend(batch)
            print(
                f"moderation completed {item.season} checkpoint {item.checkpoint}",
                flush=True,
            )
    attach_deltas(rows)
    checkpoints = summarize_checkpoint(rows)
    final_rows = [row for row in rows if int(row["checkpoint"]) == 7]
    if len(final_rows) != 534 * 7:
        raise ValueError("final checkpoint rows are incomplete")
    summary = {
        "alpha_grid": list(ALPHAS),
        "selection_rule": "none; fixed predeclared grid, no learned moderation parameters or retrospective winner",
        "parity": parity,
        "seasons": list(SEASONS),
        "weighting": "equal team-season; pooled 2023-2025 concatenates team-seasons",
        "preseason_and_final": [row for row in checkpoints if row["checkpoint"] == 7],
        "final_concentration": {
            cohort: {
                f"alpha_{alpha:.2f}": concentration(
                    final_rows, cohort, f"alpha_{alpha:.2f}"
                )
                for alpha in ALPHAS[1:]
            }
            for cohort in ("2022", "2023", "2024", "2025", "2023-2025")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint_results.csv"
    team_path = args.output / "team_season_results.csv"
    summary_path = args.output / "summary.json"
    write_csv(checkpoint_path, checkpoints)
    write_csv(
        team_path,
        sorted(
            final_rows,
            key=lambda row: (
                int(row["season"]),
                str(row["team_id"]),
                str(row["model"]),
            ),
        ),
    )
    write_json(summary_path, summary)
    provenance = {
        "study": "issue-158 positive Context-only net location moderation",
        "baseline_identifiers": {"context": "Context 1.3", "history": "History 1.1"},
        "seasons": list(SEASONS),
        "rolling_training_origins": {
            str(season): {
                "trained_through_season": season - 1,
                "context_model_sha256": base_provenance["models"][str(season)][
                    "context_model_sha256"
                ],
                "history_model_sha256": base_provenance["models"][str(season)][
                    "history_model_sha256"
                ],
                "training_population": "all eligible pre-target-season rows in the retained rolling Context 1.3 / History 1.1 fits",
            }
            for season in SEASONS
        },
        "moderation_family": "x<=0: x; x>0: alpha*x; entire fitted location mixture shifted by moderated_x - x",
        "alpha_candidates": list(ALPHAS),
        "selection_rule": "none; fixed predeclared grid; no parameters learned from outcomes",
        "evidence": "same seven cutoff-safe game checkpoints, FCS fallbacks, likelihood and BP settings as issue 147",
        "inference": crossover.INFERENCE,
        "source_hashes": dict(sorted(source_hashes.items())),
        "script_sha256": crossover.sha256(Path(__file__)),
        "output_hashes": {
            path.name: crossover.sha256(path)
            for path in (checkpoint_path, team_path, summary_path)
        },
    }
    write_json(args.output / "provenance.json", provenance)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parity": parity,
                "checkpoint_rows": len(checkpoints),
                "final_team_rows": len(final_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
