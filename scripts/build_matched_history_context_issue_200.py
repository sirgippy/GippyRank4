"""Build issue 200's matched final/pre-bowl History and Context pipelines.

The builder consumes the corrected #204 corpora through explicit roots. It
never changes production defaults or writes target outcomes into prediction
rows. Example usage is documented in ``docs/issue_200_matched_pipelines.md``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_historical_modeling as historical_builder
import build_preseason_context_prior_v1_2 as c12
import build_preseason_context_prior_v1_3 as c13
import build_preseason_prior as v1

from gippyrank.context_db_repair import (
    HISTORICAL_COVERAGE,
    HISTORICAL_FEATURES,
    HISTORICAL_PLAYERS,
    MODEL_FEATURES,
    historical_repaired_features,
)
from gippyrank.data.cfbd_games import read_cfbd_game_rows
from gippyrank.history_annual_v1_1 import HISTORY_1_1_FEATURES
from gippyrank.modeling import validate_distribution_row
from gippyrank.preseason import rank_to_z
from gippyrank.research.matched_history_context_issue_200 import (
    apply_fbs_cold_response_targets,
    apply_fbs_response_targets,
    attach_repaired_context_rows,
    common_fbs_keys,
    fit_context14_rolling,
    fit_history11,
    fit_history_cold_starts,
    inference_payload,
    predict_context14,
    predict_history11,
    read_response_targets,
    sha256_json,
    validate_fcs_targets_unchanged,
    validate_prior_pmf,
)

TARGET_SEASONS = tuple(range(2022, 2026))
TRAINING_SEASONS = tuple(range(2003, 2022))
POLICIES = ("existing", "pre_bowl")
LIKELIHOOD_DF = 15.0
INFERENCE_CONFIGURATION = {
    "max_iterations": 500,
    "tolerance": 1e-9,
    "damping": 0.35,
}
OUT = ROOT / "data/research/matched_history_context_issue_200"
GAME_FILE = re.compile(r"^(?P<season>\d{4})(?:-fcs)?\.json$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external-input/{path.name}"


def file_record(logical_name: str, path: Path, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{logical_name} is missing: {path}")
    return {
        "logical_name": logical_name,
        "path": relative_ref(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def coaching_tenure_paths(source_root: Path) -> list[Path]:
    """Return the exact coaching inputs used by Context, failing on absent data."""
    tenure_root = source_root / "data/raw/cfbd/preseason/coach_tenures"
    if not tenure_root.is_dir():
        raise FileNotFoundError(
            f"required coaching-tenure source directory is missing: {tenure_root}"
        )
    paths = sorted(tenure_root.glob("*.json"))
    if not paths:
        raise ValueError(
            f"required coaching-tenure source directory has no JSON inputs: {tenure_root}"
        )
    return paths


def load_required_coaching_tenures(
    source_root: Path,
) -> dict[str, list[dict[str, object]]]:
    """Load and validate the exact coaching files consumed by this fit."""
    result: dict[str, list[dict[str, object]]] = {}
    for path in coaching_tenure_paths(source_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            if path.name == "manifest.json" and isinstance(payload, dict):
                # cached_tenures scans this sidecar JSON but does not treat it
                # as a school-tenure payload; its hash still belongs in inputs.
                continue
            raise TypeError(f"coaching-tenure input must be a JSON list: {path}")
        for item in payload:
            team = (item.get("team") or {}).get("school")
            if team:
                result.setdefault(str(team), []).append(item)
    if not result:
        raise ValueError(
            "required coaching-tenure source directory contains no team records: "
            f"{source_root / 'data/raw/cfbd/preseason/coach_tenures'}"
        )
    return result


def input_paths(
    corpus_root: Path,
    source_root: Path,
    game_root: Path,
    reference_root: Path,
) -> dict[str, Path]:
    paths = {
        "corrected_final_observations": corpus_root
        / "data/processed/massey_issue_204_v1/final_observations.csv",
        "corrected_final_targets": corpus_root
        / "data/processed/massey_prebowl_issue_204_v1/final_target_distributions.csv",
        "corrected_final_rank_pairs": corpus_root
        / "data/processed/massey_prebowl_issue_204_v1/final_constituent_rank_pairs.csv",
        "corrected_pre_bowl_observations": corpus_root
        / "data/processed/massey_prebowl_issue_204_v1/pre_bowl_observations.csv",
        "corrected_pre_bowl_targets": corpus_root
        / "data/processed/massey_prebowl_issue_204_v1/pre_bowl_target_distributions.csv",
        "corrected_pre_bowl_rank_pairs": corpus_root
        / "data/processed/massey_prebowl_issue_204_v1/pre_bowl_constituent_rank_pairs.csv",
        "issue_204_manifest": corpus_root
        / "data/research/massey_rank_population_issue_204/manifest.json",
        "issue_204_support_report": corpus_root
        / "data/research/massey_rank_population_issue_204/target_support_anomalies.csv",
        "issue_204_raw_source_manifest": corpus_root
        / "data/research/massey_rank_population_issue_204/raw_source_manifest.json",
        "issue_198_collection_manifest": corpus_root
        / "data/research/massey_prebowl_issue_198/manifest.json",
        "issue_198_schedule_windows": corpus_root
        / "data/research/massey_prebowl_issue_198/schedule_windows.csv",
        "team_season_features": source_root
        / "data/processed/preseason/team_season_features.csv",
        "historical_transfer_features": source_root / HISTORICAL_FEATURES,
        "historical_transfer_coverage": source_root / HISTORICAL_COVERAGE,
        "historical_transfer_players": source_root / HISTORICAL_PLAYERS,
        "baseline_inference_panel": reference_root
        / "data/processed/context_incremental_information/inference_panel.jsonl.gz",
        "baseline_historical_likelihood": reference_root
        / "data/processed/posterior/historical_likelihood_v1.json",
    }
    paths.update(
        {
            f"coaching_tenure_file_{index:03d}": path
            for index, path in enumerate(coaching_tenure_paths(source_root), start=1)
        }
    )
    return paths


def validate_corrected_corpus_gate(paths: dict[str, Path]) -> dict[str, object]:
    """Combine #198 collection coverage with #204's corrected support audit."""
    issue198 = json.loads(paths["issue_198_collection_manifest"].read_text())
    issue204 = json.loads(paths["issue_204_manifest"].read_text())
    raw204 = json.loads(paths["issue_204_raw_source_manifest"].read_text())
    if issue198.get("collection_status") != "complete":
        raise ValueError("#198 collection gate is not complete")
    if issue198.get("missing_fbs_seasons") or issue198.get("missing_fcs_seasons"):
        raise ValueError("#198 collection manifest has missing required seasons")
    if issue204.get("corpus_version") != "massey-rank-population-v1":
        raise ValueError("#204 corrected corpus version is unavailable")
    if issue204.get("model_fitting_performed") is not False:
        raise ValueError("#204 audit must remain separate from issue 200 fitting")
    if raw204.get("target_support_validation", {}).get("status") != "passed":
        raise ValueError("#204 corrected target-support gate has not passed")
    if raw204.get("target_support_validation", {}).get("anomaly_count") != 0:
        raise ValueError("#204 corrected target-support audit contains anomalies")

    support_rows = read_csv(paths["issue_204_support_report"])
    if support_rows:
        raise ValueError("#204 support report is not empty after its header")
    entries = {item["name"]: item for item in issue204.get("inputs", [])}
    expected = {
        "corrected_final_observations": paths["corrected_final_observations"],
        "corrected_final_targets": paths["corrected_final_targets"],
        "corrected_final_rank_pairs": paths["corrected_final_rank_pairs"],
        "corrected_pre_bowl_observations": paths["corrected_pre_bowl_observations"],
        "corrected_pre_bowl_targets": paths["corrected_pre_bowl_targets"],
        "corrected_pre_bowl_rank_pairs": paths["corrected_pre_bowl_rank_pairs"],
    }
    checked = {}
    for name, path in expected.items():
        record = entries.get(name)
        if record is None:
            raise ValueError(f"#204 manifest lacks required input {name}")
        actual_sha = sha256_file(path)
        if actual_sha != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"#204 manifest hash/size mismatch for {name}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": actual_sha}

    return {
        "data_gate_ready": True,
        "collection_status": issue198["collection_status"],
        "collection_missing_fbs_seasons": issue198.get("missing_fbs_seasons", []),
        "collection_missing_fcs_seasons": issue198.get("missing_fcs_seasons", []),
        "issue_198_downstream_ready_before_repair": issue198.get("downstream_ready"),
        "issue_198_pre_repair_anomalies": issue198.get(
            "target_support_validation", {}
        ).get("anomaly_count"),
        "issue_204_corpus_version": issue204["corpus_version"],
        "issue_204_support_status": raw204["target_support_validation"]["status"],
        "issue_204_support_anomalies": len(support_rows),
        "issue_204_checked_inputs": checked,
        "readiness_rule": (
            "Use #198's complete collection plus #204's hash-verified corrected "
            "final/pre-bowl observations, targets, rank pairs, and empty support audit. "
            "The #198 downstream_ready=false value is retained as its pre-repair record."
        ),
    }


def load_game_phase_map(game_root: Path) -> tuple[dict[str, str], list[Path]]:
    """Load raw seasonType fields without losing them in the rank-join reader."""
    directory = game_root / "data/raw/cfbd/games"
    if not directory.is_dir():
        raise FileNotFoundError(f"CFBD game cache is missing: {directory}")
    phases: dict[str, str] = {}
    inputs: list[Path] = []
    expected = {
        directory / f"{season}{suffix}.json"
        for season in TRAINING_SEASONS
        for suffix in ("", "-fcs")
    }
    missing = sorted(path.name for path in expected if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "CFBD game cache lacks required files: " + ", ".join(missing)
        )
    for path in sorted(directory.glob("*.json")):
        match = GAME_FILE.fullmatch(path.name)
        if match is None or int(match.group("season")) not in TRAINING_SEASONS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"CFBD game cache is not a list: {path}")
        inputs.append(path)
        for item in payload:
            game_id = str(item.get("id") or "")
            if not game_id:
                raise ValueError(f"CFBD game row has no ID: {path}")
            phase = str(item.get("seasonType") or "").strip().casefold()
            if not phase:
                raise ValueError(f"CFBD game {game_id} has no seasonType")
            prior = phases.get(game_id)
            if prior is not None and prior != phase:
                raise ValueError(
                    f"CFBD duplicate game {game_id} has conflicting seasonType"
                )
            phases[game_id] = phase
    return phases, inputs


def load_rank_pair_lookup(
    path: Path,
) -> dict[
    tuple[int, tuple[int, str, str], tuple[int, str, str]], list[tuple[int, int]]
]:
    """Load corrected #204 constituent pairs only for the fixed fit window."""
    result: dict[
        tuple[int, tuple[int, str, str], tuple[int, str, str]], list[tuple[int, int]]
    ] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            season = int(row["season"])
            if season not in TRAINING_SEASONS:
                continue
            home = (season, row["home_subdivision"], row["home_team_id"])
            away = (season, row["away_subdivision"], row["away_team_id"])
            key = (int(row["game_id"]), home, away)
            result[key].append((int(row["home_rank"]), int(row["away_rank"])))
    return dict(result)


def _distribution_rows_for_policy(
    existing_rows: list[dict[str, str]],
    pre_bowl_rows: list[dict[str, str]],
    policy: str,
) -> list[dict[str, str]]:
    if policy == "existing":
        return existing_rows
    return [row for row in pre_bowl_rows if row["subdivision"] == "fbs"] + [
        row for row in existing_rows if row["subdivision"] == "fcs"
    ]


def _game_is_postseason(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "").replace("_", "")
    return normalized in {"postseason", "post", "bowl", "playoff"}


def _is_army_navy(game: dict[str, str]) -> bool:
    return {
        game.get("homeTeam", "").casefold(),
        game.get("awayTeam", "").casefold(),
    } == {
        "army",
        "navy",
    }


def _game_exclusion_reason(
    policy: str,
    game: dict[str, str],
    phase: str,
    *,
    has_joined_rank_row: bool,
) -> str:
    if not has_joined_rank_row:
        return "missing_rank_target_or_score"
    if (
        policy == "pre_bowl"
        and _game_is_postseason(phase)
        and (game["homeClassification"] == "fbs" or game["awayClassification"] == "fbs")
        and not _is_army_navy(game)
    ):
        return "fbs_postseason_excluded"
    return ""


def build_likelihood_policy(
    *,
    policy: str,
    game_rows: list[dict[str, str]],
    game_phases: dict[str, str],
    distributions: list[dict[str, str]],
    rank_pairs: dict[
        tuple[int, tuple[int, str, str], tuple[int, str, str]], list[tuple[int, int]]
    ],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Join, gate, and refit the frozen rank/site Student-t surface."""
    stats: list[dict[str, str]] = []
    joined, missing = historical_builder.build_joined(
        game_rows, distributions, rank_pairs, stats
    )
    joined = sorted(joined, key=lambda row: int(row["game_id"]))
    joined_by_id = {str(row["game_id"]): row for row in joined}
    game_manifest: list[dict[str, object]] = []
    included: list[dict[str, str]] = []
    for game in game_rows:
        game_id = str(game["id"])
        season = int(game["season"])
        phase = game_phases.get(game_id, "")
        home_div, away_div = game["homeClassification"], game["awayClassification"]
        postseason = _game_is_postseason(phase)
        army_navy = _is_army_navy(game)
        exclusion = _game_exclusion_reason(
            policy,
            game,
            phase,
            has_joined_rank_row=game_id in joined_by_id,
        )
        if not exclusion:
            included.append(joined_by_id[game_id])
        joined_row = joined_by_id.get(game_id)
        game_manifest.append(
            {
                "policy": policy,
                "game_id": game_id,
                "season": season,
                "season_type": phase,
                "home_team_id": game["homeId"],
                "away_team_id": game["awayId"],
                "home_subdivision": home_div,
                "away_subdivision": away_div,
                "pairing": (
                    joined_row["pairing"]
                    if joined_row
                    else "fbs-fcs"
                    if {home_div, away_div} == {"fbs", "fcs"}
                    else f"{home_div}-{away_div}"
                ),
                "pairing_method": joined_row["pairing_method"] if joined_row else "",
                "rank_pair_count": (
                    len(json.loads(str(joined_row["rank_pairs"]))) if joined_row else 0
                ),
                "postseason": postseason,
                "army_navy": army_navy,
                "included_for_fit": not exclusion,
                "exclusion_reason": exclusion,
            }
        )

    if not included:
        raise ValueError(f"{policy}: no games remain for likelihood fitting")
    data = historical_builder.pseudo_data(included)
    matrix = historical_builder.design_matrix(
        data.x,
        data.y,
        data.pairing,
        data.home,
        data.neutral,
        surface=True,
        fbs_home=data.fbs_home,
    )
    if set(np.unique(data.season).astype(int)) - set(TRAINING_SEASONS):
        raise ValueError(f"{policy}: likelihood contains a season outside 2003-2021")
    model = historical_builder.fit_robust_surface(
        matrix, data.margin, data.weight, df=LIKELIHOOD_DF
    )
    games_used = sorted({str(row["game_id"]) for row in included}, key=int)
    by_pairing = Counter(row["pairing"] for row in included)
    by_pair_method = Counter(row["pairing_method"] for row in included)
    result: dict[str, object] = {
        "artifact_kind": "matched_historical_likelihood_issue_200",
        "fit_kind": "weighted_pseudo",
        "policy": policy,
        "training_seasons": "2003-2021",
        "included_game_ids_sha256": sha256_json(games_used),
        "included_game_count": len(games_used),
        "rank_pair_pseudo_observations": len(data.margin),
        "pairing_game_counts": dict(sorted(by_pairing.items())),
        "pairing_method_game_counts": dict(sorted(by_pair_method.items())),
        "student_t_df": LIKELIHOOD_DF,
        "degrees_of_freedom": float(model["df"]),
        "scale": float(model["scale"]),
        "beta": np.asarray(model["beta"], dtype=float).tolist(),
        "feature_count": int(matrix.shape[1]),
        "fitting_recipe": "eight weighted-pseudo Student-t IRLS iterations; same surface, pair weights, and fit semantics as historical likelihood V1",
        "optimizer": {
            "method": "iteratively reweighted least squares",
            "iterations": 8,
            "convergence_assessed": False,
            "convergence_status": "not_assessed_fixed_iteration_recipe",
            "objective": "weighted_pseudo",
        },
        "missing_rank_or_score_games": dict(sorted(missing.items())),
    }
    return result, game_manifest, included


def load_snapshot_rows(corpus_root: Path) -> list[dict[str, object]]:
    manifest = json.loads(
        (
            corpus_root / "data/research/massey_prebowl_issue_198/manifest.json"
        ).read_text()
    )
    windows = {
        int(row["season"]): row
        for row in read_csv(
            corpus_root / "data/research/massey_prebowl_issue_198/schedule_windows.csv"
        )
    }
    result = []
    for source in manifest.get("collection", []):
        season = int(source["season"])
        window = windows[season]
        result.append(
            {
                "season": season,
                "subdivision": "fbs",
                "snapshot_date": source["snapshot_date"],
                "snapshot_source": source["snapshot_source"],
                "source_url": source["source_url"],
                "last_fbs_conference_championship_date": window[
                    "last_fbs_conference_championship_date"
                ],
                "first_fbs_postseason_game_date": window[
                    "first_fbs_postseason_game_date"
                ],
                "phase_exception_decision": source.get("phase_exception_decision", ""),
                "phase_exception_rationale": source.get(
                    "phase_exception_rationale", ""
                ),
                "army_navy_timing_policy": "retain even when it follows the accepted FBS snapshot",
                "fcs_source_policy": "corrected final corpus; copied unchanged into the pre-bowl corpus",
            }
        )
    if {row["season"] for row in result} != set(range(2003, 2026)):
        raise ValueError("pre-bowl snapshot manifest does not cover 2003-2025")
    return result


def load_baseline_panel(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["checkpoint"]) != 1 or row["subdivision"] != "fbs":
                continue
            key = (int(row["season"]), str(row["team_id"]))
            if key in result:
                raise ValueError(f"#192 baseline has duplicate preseason row {key}")
            result[key] = row
    return result


def baseline_pmf_diagnostics(
    priors: list[dict[str, object]],
    panel: dict[tuple[int, str], dict[str, Any]],
    baseline_likelihood_path: Path,
    likelihoods: dict[str, dict[str, object]],
) -> dict[str, object]:
    reference_field = {"history": "history_prior_pmf", "context": "context_prior_pmf"}
    diagnostics: dict[str, object] = {
        "reference": "#192 / PR #196 preseason checkpoint 1; frozen original controls",
        "prior_comparison_interpretation": (
            "Corrected final-control prior differences include #204 corpus repair and the restored, hash-bound coaching-tenure inputs; these are shared by both issue 200 policies and are not the pre-bowl intervention."
        ),
        "prior_comparisons": {},
        "likelihood_comparison": {},
    }
    for family, field in reference_field.items():
        deltas: list[float] = []
        per_season: dict[str, dict[str, object]] = {}
        for season in TARGET_SEASONS:
            values = []
            for row in priors:
                if (
                    int(row["season"]) != season
                    or row["model_family"].casefold() != family
                    or row["policy"] != "existing"
                ):
                    continue
                source = panel.get((season, str(row["team_id"])))
                if source is None:
                    continue
                actual = np.asarray(row["pmf"], dtype=float)
                baseline = np.asarray(source[field], dtype=float)
                if len(actual) != len(baseline):
                    raise ValueError(
                        f"{season} {row['team_id']}: corrected and #192 baseline supports differ"
                    )
                difference = np.abs(actual - baseline)
                values.extend(float(value) for value in difference)
            if not values:
                raise ValueError(f"#192 baseline has no {family} rows for {season}")
            deltas.extend(values)
            per_season[str(season)] = {
                "team_season_count": sum(
                    1
                    for row in priors
                    if int(row["season"]) == season
                    and row["model_family"].casefold() == family
                    and row["policy"] == "existing"
                ),
                "max_abs_probability_difference": max(values),
                "mean_abs_probability_difference": float(np.mean(values)),
            }
        diagnostics["prior_comparisons"][family] = {
            "baseline_field": field,
            "season_results": per_season,
            "max_abs_probability_difference": max(deltas),
            "mean_abs_probability_difference": float(np.mean(deltas)),
        }

    original = json.loads(baseline_likelihood_path.read_text(encoding="utf-8"))
    corrected = likelihoods["existing"]
    old_beta = np.asarray(original["beta"], dtype=float)
    new_beta = np.asarray(corrected["beta"], dtype=float)
    if old_beta.shape != new_beta.shape:
        raise ValueError("corrected final likelihood changed the frozen surface shape")
    diagnostics["likelihood_comparison"] = {
        "baseline_artifact_sha256": sha256_file(baseline_likelihood_path),
        "baseline_training_seasons": original["training_seasons"],
        "baseline_df": float(original["degrees_of_freedom"]),
        "corrected_final_df": float(corrected["degrees_of_freedom"]),
        "max_abs_coefficient_difference": float(np.max(np.abs(old_beta - new_beta))),
        "mean_abs_coefficient_difference": float(np.mean(np.abs(old_beta - new_beta))),
        "scale_difference": float(corrected["scale"] - original["scale"]),
        "interpretation": "The #192 likelihood is an immutable original-corpus reference; changes here reflect #204 data repair and refitting, not the pre-bowl intervention.",
    }
    return diagnostics


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Total variation for rank PMFs after zero-padding to shared support."""
    support = max(len(left), len(right))
    left_padded = np.pad(np.asarray(left, dtype=float), (0, support - len(left)))
    right_padded = np.pad(np.asarray(right, dtype=float), (0, support - len(right)))
    return float(0.5 * np.abs(left_padded - right_padded).sum())


def policy_intervention_diagnostics(
    priors: list[dict[str, object]],
    final_targets: dict[tuple[int, str, str], Any],
    pre_bowl_targets: dict[tuple[int, str, str], Any],
    fitted_models: dict[str, Any],
    likelihoods: dict[str, dict[str, object]],
    composition: list[dict[str, object]],
) -> dict[str, object]:
    """Describe fitted policy differences independently of frozen baseline parity."""
    material_tv_threshold = 0.01
    prior_by_arm: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in priors:
        key = (int(row["season"]), str(row["model_family"]), str(row["team_id"]))
        arm = str(row["policy"])
        if arm not in POLICIES:
            raise ValueError(f"unknown prior policy in diagnostics: {arm}")
        arms = prior_by_arm.setdefault(key, {})
        if arm in arms:
            raise ValueError(f"duplicate prior policy row in diagnostics: {key}/{arm}")
        arms[arm] = row

    prior_results = []
    for season in TARGET_SEASONS:
        for family in ("History", "Context"):
            values = []
            for (row_season, row_family, _team_id), arms in sorted(
                prior_by_arm.items(),
                key=lambda item: (item[0][0], item[0][1], int(item[0][2])),
            ):
                if row_season != season or row_family != family:
                    continue
                if set(arms) != set(POLICIES):
                    raise ValueError(
                        f"{season}/{family}: intervention PMF policy arms differ"
                    )
                existing, pre_bowl = arms["existing"], arms["pre_bowl"]
                values.append(
                    total_variation(
                        np.asarray(existing["pmf"], dtype=float),
                        np.asarray(pre_bowl["pmf"], dtype=float),
                    )
                )
            if not values:
                raise ValueError(f"{season}/{family}: no matched intervention PMFs")
            prior_results.append(
                {
                    "target_season": season,
                    "model_family": family,
                    "team_count": len(values),
                    "mean_total_variation_distance": float(np.mean(values)),
                    "maximum_total_variation_distance": max(values),
                    "material_change_threshold_total_variation": material_tv_threshold,
                    "materially_changed_team_count": sum(
                        value > material_tv_threshold for value in values
                    ),
                }
            )

    target_results = []
    for season in TARGET_SEASONS:
        values = []
        changed_support = 0
        for key in sorted(
            set(final_targets) & set(pre_bowl_targets),
            key=lambda value: (value[0], value[1], int(value[2])),
        ):
            if key[0] != season or key[1] != "fbs":
                continue
            final_target, pre_target = final_targets[key], pre_bowl_targets[key]
            final_pmf = _target_pmf(
                {
                    "team_population": str(final_target.population),
                    "pmf": json.dumps(
                        [
                            {
                                "rank": rank,
                                "probability": count / len(final_target.ranks),
                            }
                            for rank, count in sorted(
                                Counter(final_target.ranks).items()
                            )
                        ]
                    ),
                }
            )
            pre_bowl_pmf = _target_pmf(
                {
                    "team_population": str(pre_target.population),
                    "pmf": json.dumps(
                        [
                            {
                                "rank": rank,
                                "probability": count / len(pre_target.ranks),
                            }
                            for rank, count in sorted(Counter(pre_target.ranks).items())
                        ]
                    ),
                }
            )
            values.append(total_variation(final_pmf, pre_bowl_pmf))
            changed_support += final_target.population != pre_target.population
        if not values:
            raise ValueError(f"{season}: no common FBS response distributions")
        target_results.append(
            {
                "target_season": season,
                "team_count": len(values),
                "mean_total_variation_distance": float(np.mean(values)),
                "maximum_total_variation_distance": max(values),
                "changed_distribution_team_count": sum(value > 0 for value in values),
                "different_population_support_team_count": changed_support,
            }
        )

    rolling_models = fitted_models["rolling_models"]
    coefficient_results = []
    for season in TARGET_SEASONS:
        season_fits = rolling_models[str(season)]
        for family, fit_key, model_key in (
            ("History", "history", "model"),
            ("Context", "context", "model"),
            ("History", "history", "transition_model"),
        ):
            existing_model = season_fits["existing"][fit_key].get(model_key)
            pre_bowl_model = season_fits["pre_bowl"][fit_key].get(model_key)
            component = (
                "transition_model" if model_key == "transition_model" else "main_prior"
            )
            if existing_model is None or pre_bowl_model is None:
                if existing_model is not pre_bowl_model:
                    raise ValueError(
                        f"{season}/{family}/{component}: fitted policy components differ"
                    )
                continue
            vector_differences = {}
            differences = []
            for name in ("location_coefficients", "log_scale_coefficients"):
                old = np.asarray(existing_model[name], dtype=float)
                new = np.asarray(pre_bowl_model[name], dtype=float)
                if old.shape != new.shape:
                    raise ValueError(
                        f"{season}/{family}/{component}/{name}: coefficient shapes differ"
                    )
                absolute = np.abs(old - new)
                differences.extend(absolute.tolist())
                vector_differences[name] = {
                    "coefficient_count": int(absolute.size),
                    "mean_absolute_difference": float(np.mean(absolute)),
                    "maximum_absolute_difference": float(np.max(absolute)),
                }
            coefficient_results.append(
                {
                    "target_season": season,
                    "model_family": family,
                    "component": component,
                    "coefficient_count": len(differences),
                    "mean_absolute_difference": float(np.mean(differences)),
                    "maximum_absolute_difference": max(differences),
                    "coefficient_vectors": vector_differences,
                }
            )

    existing_beta = np.asarray(likelihoods["existing"]["beta"], dtype=float)
    pre_bowl_beta = np.asarray(likelihoods["pre_bowl"]["beta"], dtype=float)
    if existing_beta.shape != pre_bowl_beta.shape:
        raise ValueError("policy likelihood coefficient vector shapes differ")
    likelihood_beta_difference = np.abs(existing_beta - pre_bowl_beta)
    composition_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"games": 0, "rank_pair_rows": 0}
    )
    for row in composition:
        arm = str(row["policy"])
        key = (arm, str(row["pairing"]), str(row["pairing_method"]))
        composition_counts[key]["games"] += int(row["games"])
        composition_counts[key]["rank_pair_rows"] += int(row["rank_pair_rows"])
    composition_differences = []
    for pairing, method in sorted({(key[1], key[2]) for key in composition_counts}):
        existing = composition_counts[("existing", pairing, method)]
        pre_bowl = composition_counts[("pre_bowl", pairing, method)]
        composition_differences.append(
            {
                "pairing": pairing,
                "pairing_method": method,
                "existing_game_count": existing["games"],
                "pre_bowl_game_count": pre_bowl["games"],
                "pre_bowl_minus_existing_game_count": pre_bowl["games"]
                - existing["games"],
                "existing_rank_pair_rows": existing["rank_pair_rows"],
                "pre_bowl_rank_pair_rows": pre_bowl["rank_pair_rows"],
                "pre_bowl_minus_existing_rank_pair_rows": pre_bowl["rank_pair_rows"]
                - existing["rank_pair_rows"],
            }
        )
    existing_likelihood, pre_bowl_likelihood = (
        likelihoods["existing"],
        likelihoods["pre_bowl"],
    )
    return {
        "comparison": "pre_bowl versus existing fitted policies on corrected issue 204 inputs",
        "data_repair_effects": "Corrected-corpus and restored coaching-source differences from frozen #192/#196 controls remain in baseline_parity.json. Both fitted policies share the same hash-bound coaching inputs, so these deltas describe the policy intervention.",
        "material_prior_change_threshold_total_variation": material_tv_threshold,
        "prior_pmf_differences": prior_results,
        "fbs_target_distribution_differences": target_results,
        "fitted_coefficient_differences": coefficient_results,
        "likelihood_differences": {
            "mean_absolute_coefficient_difference": float(
                np.mean(likelihood_beta_difference)
            ),
            "maximum_absolute_coefficient_difference": float(
                np.max(likelihood_beta_difference)
            ),
            "scale_existing": float(existing_likelihood["scale"]),
            "scale_pre_bowl": float(pre_bowl_likelihood["scale"]),
            "absolute_scale_difference": abs(
                float(existing_likelihood["scale"])
                - float(pre_bowl_likelihood["scale"])
            ),
            "included_game_count_existing": int(
                existing_likelihood["included_game_count"]
            ),
            "included_game_count_pre_bowl": int(
                pre_bowl_likelihood["included_game_count"]
            ),
            "pre_bowl_minus_existing_included_game_count": int(
                pre_bowl_likelihood["included_game_count"]
            )
            - int(existing_likelihood["included_game_count"]),
            "rank_pair_pseudo_observations_existing": int(
                existing_likelihood["rank_pair_pseudo_observations"]
            ),
            "rank_pair_pseudo_observations_pre_bowl": int(
                pre_bowl_likelihood["rank_pair_pseudo_observations"]
            ),
            "pre_bowl_minus_existing_rank_pair_pseudo_observations": int(
                pre_bowl_likelihood["rank_pair_pseudo_observations"]
            )
            - int(existing_likelihood["rank_pair_pseudo_observations"]),
            "rank_pair_composition": composition_differences,
        },
    }


def _target_rows_by_key(
    rows: list[dict[str, str]],
) -> dict[tuple[int, str, str], dict[str, str]]:
    result = {}
    for row in rows:
        key = (int(row["season"]), row["subdivision"], row["team_id"])
        if key in result:
            raise ValueError(f"duplicate target row {key}")
        validate_distribution_row(row)
        result[key] = row
    return result


def _target_pmf(target: dict[str, str]) -> np.ndarray:
    result = np.zeros(int(target["team_population"]), dtype=float)
    for point in json.loads(target["pmf"]):
        result[int(point["rank"]) - 1] = float(point["probability"])
    return result


def build_prior_pipelines(
    *,
    final_targets: list[dict[str, str]],
    pre_bowl_targets: list[dict[str, str]],
    final_target_path: Path,
    feature_path: Path,
    source_root: Path,
    output_root: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    dict[int, dict[str, object]],
]:
    """Fit rolling prior models and emit their shared prediction input manifest."""
    tenures = load_required_coaching_tenures(source_root)
    target_maps = {
        "existing": read_response_targets(final_targets, label="corrected final"),
        "pre_bowl": read_response_targets(pre_bowl_targets, label="corrected pre-bowl"),
    }
    validate_fcs_targets_unchanged(target_maps["existing"], target_maps["pre_bowl"])
    matched_keys = common_fbs_keys(target_maps["existing"], target_maps["pre_bowl"])
    if not matched_keys:
        raise ValueError("corrected corpora have no common FBS team-seasons")

    history_rows, history_cold, _ = v1.load_rows(
        max_season=max(TARGET_SEASONS),
        corpus_policy="corrected",
        outcomes_path=final_target_path,
        feature_path=feature_path,
    )
    base_context_rows, _, _ = c13.base_context_rows(
        max_season=max(TARGET_SEASONS),
        outcomes_path=final_target_path,
        feature_path=feature_path,
        tenure_root=source_root / "data/raw/cfbd/preseason/coach_tenures",
        tenures=tenures,
        corpus_policy="corrected",
    )
    repaired_features, db_evidence = historical_repaired_features(source_root)
    context_rows, missing_repaired_feature_keys = attach_repaired_context_rows(
        base_context_rows, repaired_features
    )
    history_rows = [row for row in history_rows if row.subdivision == "fbs"]
    history_cold = [row for row in history_cold if row.subdivision == "fbs"]
    context_by_key = {
        (row.season, row.subdivision, row.team_id): row for row in context_rows
    }
    history_by_key = {
        (row.season, row.subdivision, row.team_id): row for row in history_rows
    }
    if set(context_by_key) != set(history_by_key):
        raise ValueError("History and Context prior training opportunities differ")
    for key, history_row in history_by_key.items():
        context_row = context_by_key[key]
        for feature in HISTORY_1_1_FEATURES:
            if history_row.features.get(feature) != context_row.features.get(feature):
                raise ValueError(
                    f"{key}: History and Context rank-history inputs differ"
                )
        if not np.array_equal(history_row.lag1_z, context_row.lag1_z):
            raise ValueError(f"{key}: History and Context lag-one distributions differ")

    history_response = {
        policy: apply_fbs_response_targets(
            history_rows, target_maps[policy], matched_keys=matched_keys
        )
        for policy in POLICIES
    }
    context_response = {
        policy: apply_fbs_response_targets(
            context_rows, target_maps[policy], matched_keys=matched_keys
        )
        for policy in POLICIES
    }
    cold_response = {
        policy: apply_fbs_cold_response_targets(
            history_cold, target_maps[policy], matched_keys=matched_keys
        )
        for policy in POLICIES
    }

    feature_index = c12.feature_index(feature_path)
    inference_by_season: dict[int, dict[str, Any]] = {}
    for season in TARGET_SEASONS:
        inference = c12.inference_rows(
            season,
            season - 1,
            feature_index,
            tenures,
            outcomes_path=final_target_path,
            corpus_policy="corrected",
        )
        inference_by_season[season] = {row.team_id: row for row in inference}
        if len(inference_by_season[season]) != len(inference):
            raise ValueError(f"{season}: duplicate outcome-free prediction inputs")

    fitted_models: dict[str, object] = {
        "spec_version": "issue-200-v1",
        "history_semantics": {
            "features": list(HISTORY_1_1_FEATURES),
            "source": "src/gippyrank/history_annual_v1_1.py",
            "cold_start": "learned FCS-to-FBS transition, otherwise generic FBS prior",
        },
        "context_semantics": {
            "version": "1.4",
            "source": "src/gippyrank/context_db_repair.py",
            "feature_names": list(MODEL_FEATURES),
            "db_boundary": "repaired observed incoming DB impact sum and coverage",
            "cold_start": "matched History 1.1 fallback",
            "missing_repaired_feature_team_seasons": [
                [season, team_id]
                for season, team_id in sorted(missing_repaired_feature_keys)
            ],
            "missingness_policy": "Uncovered transfer inputs remain missing; they are not interpreted as zero transfer impact.",
        },
        "rolling_models": {},
    }
    training_manifest: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    shared_inputs: dict[str, dict[str, object]] = {}
    missing_target_repaired_feature_keys: set[tuple[int, str]] = set()

    for season in TARGET_SEASONS:
        trained_through = season - 1
        target_ids = {
            key[2] for key in matched_keys if key[0] == season and key[1] == "fbs"
        }
        missing_input_ids = target_ids - set(inference_by_season[season])
        if missing_input_ids:
            raise ValueError(
                f"{season}: outcome-free feature rows are missing target teams "
                f"{sorted(missing_input_ids)}"
            )
        season_fits: dict[str, object] = {}
        prediction_inputs: dict[str, dict[str, object]] = {}

        for policy in POLICIES:
            fit_history_rows = [
                row for row in history_response[policy] if row.season <= trained_through
            ]
            fit_context_rows = [
                row for row in context_response[policy] if row.season <= trained_through
            ]
            fit_cold_rows = [
                row
                for row in cold_response[policy]
                if int(row.season) <= trained_through
            ]
            h_keys = {(row.season, row.team_id) for row in fit_history_rows}
            c_keys = {(row.season, row.team_id) for row in fit_context_rows}
            if h_keys != c_keys:
                raise ValueError(
                    f"{season}/{policy}: History/Context training rows differ"
                )
            if not fit_history_rows:
                raise ValueError(f"{season}/{policy}: no History training rows")

            h_model = fit_history11(fit_history_rows)
            h_cold_models = fit_history_cold_starts(
                fit_cold_rows, trained_through_season=trained_through
            )
            c_model = fit_context14_rolling(
                fit_context_rows, trained_through_season=trained_through
            )
            season_fits[policy] = {
                "trained_through_season": trained_through,
                "history": {
                    "training_row_count": len(fit_history_rows),
                    "training_row_keys_sha256": sha256_json(
                        sorted([list(key) for key in h_keys])
                    ),
                    "model": h_model.metadata(),
                    "cold_start_training_rows": len(fit_cold_rows),
                    "transition_model": (
                        h_cold_models.transition.metadata()
                        if h_cold_models.transition is not None
                        else None
                    ),
                    "generic_prior": (
                        {
                            "location": h_cold_models.generic.location,
                            "scale": h_cold_models.generic.scale,
                        }
                        if h_cold_models.generic is not None
                        else None
                    ),
                },
                "context": {
                    "training_row_count": len(fit_context_rows),
                    "training_row_keys_sha256": sha256_json(
                        sorted([list(key) for key in c_keys])
                    ),
                    "model": c_model.metadata(),
                },
            }

            policy_predictions: dict[str, dict[str, dict[str, object]]] = {
                "history": {},
                "context": {},
            }
            for team_id in sorted(target_ids, key=int):
                final_target = target_maps["existing"][(season, "fbs", team_id)]
                pre_target = target_maps["pre_bowl"][(season, "fbs", team_id)]
                target = final_target if policy == "existing" else pre_target
                base_input = inference_by_season[season][team_id]
                if (
                    base_input.lag1_z is None
                    and base_input.cold_start_reason == "fcs_to_fbs_transition"
                ):
                    cross_target = target_maps["existing"].get(
                        (trained_through, "fcs", team_id)
                    )
                    if cross_target is None:
                        raise ValueError(
                            f"{season} {team_id}: transition fallback lacks corrected FCS lag"
                        )
                    cross_lag = rank_to_z(
                        np.asarray(cross_target.ranks, dtype=int),
                        cross_target.population,
                    )
                else:
                    cross_lag = None

                history_input = replace(base_input, population=target.population)
                h_pmf, h_method, h_location, h_scale = predict_history11(
                    h_model,
                    h_cold_models,
                    history_input,
                    cross_subdivision_lag_z=cross_lag,
                )
                repaired = repaired_features.get((season, team_id))
                c_features = {
                    name: base_input.features.get(name) for name in MODEL_FEATURES
                }
                if repaired is None:
                    missing_target_repaired_feature_keys.add((season, team_id))
                    for name in (
                        "transfer_in_prior_usage_sum",
                        "observed_db_impact_sum",
                        "db_impact_coverage_fraction",
                    ):
                        c_features[name] = None
                else:
                    c_features.update(repaired)
                if set(c_features) != set(MODEL_FEATURES):
                    raise ValueError(
                        f"{season} {team_id}: Context feature schema differs"
                    )
                context_input = replace(
                    base_input, population=target.population, features=c_features
                )
                c_pmf, c_method, c_location, c_scale = predict_context14(
                    c_model, context_input, history_pmf=h_pmf
                )
                validate_prior_pmf(
                    h_pmf, target.population, context=f"{season}/{team_id}/History"
                )
                validate_prior_pmf(
                    c_pmf, target.population, context=f"{season}/{team_id}/Context"
                )

                predictor_input = inference_payload(context_input)
                common_input = {
                    **predictor_input,
                    "active_history_features": list(HISTORY_1_1_FEATURES),
                    "active_context_features": list(MODEL_FEATURES),
                }
                input_sha = sha256_json(common_input)
                prediction_inputs[team_id] = {
                    "season": season,
                    "team_id": team_id,
                    "team_name": base_input.team_name,
                    "population_existing": final_target.population,
                    "population_pre_bowl": pre_target.population,
                    "target_supports_equal": final_target.population
                    == pre_target.population,
                    "predictor_input_sha256": input_sha,
                    "rank_lag_input_sha256": sha256_json(
                        {
                            "lag1_z": base_input.lag1_z,
                            "history_features": {
                                name: base_input.features.get(name)
                                for name in HISTORY_1_1_FEATURES
                            },
                        }
                    ),
                    "feature_source_class": "existing preseason features with corrected final-rank lags and repaired Context 1.4 DB transfer boundary",
                    "repaired_transfer_feature_available": repaired is not None,
                }
                for family, pmf, method, location, scale in (
                    ("history", h_pmf, h_method, h_location, h_scale),
                    ("context", c_pmf, c_method, c_location, c_scale),
                ):
                    pmf_list = [float(value) for value in np.asarray(pmf, dtype=float)]
                    policy_predictions[family][team_id] = {
                        "season": season,
                        "trained_through_season": trained_through,
                        "policy": policy,
                        "model_family": family.title(),
                        "team_id": team_id,
                        "team_name": base_input.team_name,
                        "subdivision": "fbs",
                        "population": target.population,
                        "prior_method": method,
                        "conditional_location_mean": location,
                        "predictive_scale": scale,
                        "predictor_input_sha256": input_sha,
                        "pmf_sha256": sha256_json(pmf_list),
                        "pmf": pmf_list,
                    }

            for family in ("history", "context"):
                training_rows = (
                    fit_history_rows if family == "history" else fit_context_rows
                )
                for row in training_rows:
                    key = (row.season, "fbs", row.team_id)
                    response = target_maps[policy][key]
                    training_manifest.append(
                        {
                            "target_season": season,
                            "trained_through_season": trained_through,
                            "policy": policy,
                            "model_family": family.title(),
                            "component": "main_prior",
                            "response_season": row.season,
                            "team_id": row.team_id,
                            "response_population": response.population,
                            "response_rank_count": len(response.ranks),
                            "response_row_sha256": response.row_sha256,
                            "predictor_row_sha256": sha256_json(
                                {
                                    "rank_lag_input": {
                                        "lag1_z": np.asarray(row.lag1_z).tolist(),
                                        "history_features": {
                                            name: row.features.get(name)
                                            for name in HISTORY_1_1_FEATURES
                                        },
                                    },
                                    "model_features": row.features,
                                }
                            ),
                        }
                    )
            for row in fit_cold_rows:
                response = target_maps[policy][
                    (int(row.season), "fbs", str(row.team_id))
                ]
                training_manifest.append(
                    {
                        "target_season": season,
                        "trained_through_season": trained_through,
                        "policy": policy,
                        "model_family": "History",
                        "component": f"cold_start_{row.reason}",
                        "response_season": row.season,
                        "team_id": row.team_id,
                        "response_population": response.population,
                        "response_rank_count": len(response.ranks),
                        "response_row_sha256": response.row_sha256,
                        "predictor_row_sha256": sha256_json(
                            {
                                "reason": row.reason,
                                "cross_subdivision_lag_z": (
                                    None
                                    if row.cross_subdivision_lag_z is None
                                    else np.asarray(
                                        row.cross_subdivision_lag_z, dtype=float
                                    ).tolist()
                                ),
                            }
                        ),
                    }
                )
            for family in ("history", "context"):
                prior_rows.extend(policy_predictions[family].values())

        for team_id in prediction_inputs:
            for family in ("history", "context"):
                rows_for_team = [
                    row
                    for row in prior_rows
                    if int(row["season"]) == season
                    and str(row["team_id"]) == team_id
                    and row["model_family"].casefold() == family
                ]
                if {str(row["policy"]) for row in rows_for_team} != set(POLICIES):
                    raise AssertionError(
                        f"{season}/{team_id}/{family}: missing policy arm"
                    )
                if len({row["predictor_input_sha256"] for row in rows_for_team}) != 1:
                    raise AssertionError(
                        f"{season}/{team_id}/{family}: policy inputs differ"
                    )
        shared_inputs[str(season)] = {
            "target_team_count": len(target_ids),
            "target_team_ids_sha256": sha256_json(sorted(target_ids, key=int)),
            "predictor_rows": [
                prediction_inputs[key] for key in sorted(prediction_inputs, key=int)
            ],
            "repaired_context_db_coverage": {
                "team_count": sum(
                    1 for year, _team_id in db_evidence if year == season
                ),
                "incoming_db_transfers": sum(
                    evidence.incoming_db_count
                    for (year, _team_id), evidence in db_evidence.items()
                    if year == season
                ),
                "observed_db_impact_count": sum(
                    evidence.observed_db_impact_count
                    for (year, _team_id), evidence in db_evidence.items()
                    if year == season
                ),
                "status_team_counts": dict(
                    sorted(
                        Counter(
                            evidence.status
                            for (year, _team_id), evidence in db_evidence.items()
                            if year == season
                        ).items()
                    )
                ),
            },
        }
        fitted_models["rolling_models"][str(season)] = season_fits

    prior_rows.sort(
        key=lambda row: (
            int(row["season"]),
            str(row["policy"]),
            str(row["model_family"]),
            int(str(row["team_id"])),
        )
    )
    training_manifest.sort(
        key=lambda row: (
            int(row["target_season"]),
            str(row["policy"]),
            str(row["model_family"]),
            str(row["component"]),
            int(str(row["response_season"])),
            int(str(row["team_id"])),
        )
    )
    shared_inputs["context_feature_coverage"] = {
        "available_repaired_feature_team_seasons": len(repaired_features),
        "missing_repaired_feature_team_seasons": [
            [season, team_id]
            for season, team_id in sorted(missing_repaired_feature_keys)
        ],
        "missing_repaired_feature_target_team_seasons": [
            [season, team_id]
            for season, team_id in sorted(missing_target_repaired_feature_keys)
        ],
    }
    fitted_models["context_semantics"][
        "missing_repaired_feature_target_team_seasons"
    ] = [
        [season, team_id]
        for season, team_id in sorted(missing_target_repaired_feature_keys)
    ]
    return prior_rows, fitted_models, training_manifest, shared_inputs


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = input_paths(
        args.corpus_root, args.source_root, args.game_root, args.reference_root
    )
    for logical_name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{logical_name} is missing: {path}")
    gate = validate_corrected_corpus_gate(paths)
    # Validate response corpora before any model-fitting call.
    final_target_rows = read_csv(paths["corrected_final_targets"])
    pre_bowl_target_rows = read_csv(paths["corrected_pre_bowl_targets"])
    final_response_map = read_response_targets(
        final_target_rows, label="corrected final"
    )
    pre_bowl_response_map = read_response_targets(
        pre_bowl_target_rows, label="corrected pre-bowl"
    )
    validate_fcs_targets_unchanged(final_response_map, pre_bowl_response_map)
    gate["validated_final_target_rows"] = len(final_response_map)
    gate["validated_pre_bowl_target_rows"] = len(pre_bowl_response_map)
    gate["common_fbs_target_team_seasons"] = len(
        common_fbs_keys(final_response_map, pre_bowl_response_map)
    )
    if not gate["data_gate_ready"]:
        raise ValueError("corrected Massey corpus gate is not ready")
    write_json(args.output_root / "data_gate.json", gate)

    input_records = []
    roots = {
        "corpus": args.corpus_root,
        "features": args.source_root,
        "games": args.game_root,
        "reference": args.reference_root,
    }
    for logical_name, path in paths.items():
        owner = (
            "corpus"
            if logical_name.startswith(("corrected_", "issue_"))
            else "reference"
            if logical_name.startswith("baseline_")
            else "features"
        )
        input_records.append(file_record(logical_name, path, roots[owner]))

    feature_path = paths["team_season_features"]
    prior_rows, fitted_models, training_manifest, shared_inputs = build_prior_pipelines(
        final_targets=final_target_rows,
        pre_bowl_targets=pre_bowl_target_rows,
        final_target_path=paths["corrected_final_targets"],
        feature_path=feature_path,
        source_root=args.source_root,
        output_root=args.output_root,
    )

    # Record all source schedules used for the same-system pair joins.
    game_phases, game_source_paths = load_game_phase_map(args.game_root)
    game_rows, _ = read_cfbd_game_rows(
        args.game_root / "data/raw/cfbd/games", apply_cohort_decisions=True
    )
    game_rows = [row for row in game_rows if int(row["season"]) in TRAINING_SEASONS]
    game_rows.sort(key=lambda row: int(row["id"]))
    if not game_rows:
        raise ValueError("CFBD schedule cache has no games in 2003-2021")
    input_records.extend(
        file_record(f"cfbd_games_{path.stem}", path, args.game_root)
        for path in game_source_paths
    )

    distribution_rows_by_policy = {
        "existing": final_target_rows,
        "pre_bowl": _distribution_rows_for_policy(
            final_target_rows, pre_bowl_target_rows, "pre_bowl"
        ),
    }
    pair_lookups = {
        "existing": load_rank_pair_lookup(paths["corrected_final_rank_pairs"]),
        "pre_bowl": load_rank_pair_lookup(paths["corrected_pre_bowl_rank_pairs"]),
    }
    likelihoods: dict[str, dict[str, object]] = {}
    game_manifests = []
    included_games_by_policy: dict[str, list[dict[str, str]]] = {}
    for policy in POLICIES:
        likelihood, game_manifest, included = build_likelihood_policy(
            policy=policy,
            game_rows=game_rows,
            game_phases=game_phases,
            distributions=distribution_rows_by_policy[policy],
            rank_pairs=pair_lookups[policy],
        )
        likelihoods[policy] = likelihood
        game_manifests.extend(game_manifest)
        included_games_by_policy[policy] = included

    final_joined_by_id = {
        str(row["game_id"]): row for row in included_games_by_policy["existing"]
    }
    pre_joined_by_id = {
        str(row["game_id"]): row for row in included_games_by_policy["pre_bowl"]
    }
    fcs_game_changes = []
    for game_id in sorted(set(final_joined_by_id) | set(pre_joined_by_id), key=int):
        final = final_joined_by_id.get(game_id)
        pre = pre_joined_by_id.get(game_id)
        representative = final or pre
        if representative is None:
            continue
        if representative["pairing"] == "fcs-fcs" and (
            final is None
            or pre is None
            or (
                final["rank_pairs"] != pre["rank_pairs"]
                or final["home_team_population"] != pre["home_team_population"]
                or final["away_team_population"] != pre["away_team_population"]
            )
        ):
            fcs_game_changes.append(game_id)
    if fcs_game_changes:
        raise ValueError(
            "FCS-FCS corrected rank-pair inputs changed between policies: "
            + ", ".join(fcs_game_changes[:20])
        )

    final_regular = {
        str(row["game_id"])
        for row in included_games_by_policy["existing"]
        if not _game_is_postseason(game_phases[str(row["game_id"])])
    }
    pre_regular = {
        str(row["game_id"])
        for row in included_games_by_policy["pre_bowl"]
        if not _game_is_postseason(game_phases[str(row["game_id"])])
    }
    if final_regular != pre_regular:
        raise ValueError(
            "regular-season likelihood game opportunities differ by target policy"
        )

    composition: list[dict[str, object]] = []
    for policy in POLICIES:
        counts: dict[tuple[int, str, str], dict[str, int]] = defaultdict(
            lambda: {"games": 0, "rank_pair_rows": 0}
        )
        for row in included_games_by_policy[policy]:
            key = (
                int(row["season"]),
                str(row["pairing"]),
                str(row["pairing_method"]),
            )
            counts[key]["games"] += 1
            counts[key]["rank_pair_rows"] += len(json.loads(str(row["rank_pairs"])))
        snapshots = {
            int(row["season"]): row for row in load_snapshot_rows(args.corpus_root)
        }
        for (season, pairing, method), values in sorted(counts.items()):
            composition.append(
                {
                    "policy": policy,
                    "season": season,
                    "pairing": pairing,
                    "pairing_method": method,
                    "games": values["games"],
                    "rank_pair_rows": values["rank_pair_rows"],
                    "fbs_target_snapshot_date": snapshots[season]["snapshot_date"],
                    "fcs_target_policy": "corrected final corpus unchanged",
                    "cross_subdivision_dates_differ": pairing == "fbs-fcs",
                }
            )

    baseline_panel = load_baseline_panel(paths["baseline_inference_panel"])
    baseline_parity = baseline_pmf_diagnostics(
        prior_rows, baseline_panel, paths["baseline_historical_likelihood"], likelihoods
    )
    baseline_parity["frozen_reference_inputs"] = {
        "prior_panel_sha256": sha256_file(paths["baseline_inference_panel"]),
        "prior_panel_checkpoint": 1,
        "prior_panel_lineage": "#192 / PR #196; original final-corpus controls retained unchanged",
        "likelihood_sha256": sha256_file(paths["baseline_historical_likelihood"]),
    }
    intervention = policy_intervention_diagnostics(
        prior_rows,
        final_response_map,
        pre_bowl_response_map,
        fitted_models,
        likelihoods,
        composition,
    )

    snapshots = load_snapshot_rows(args.corpus_root)
    fbs_target_ids = {
        season: sorted(
            [
                key[2]
                for key in common_fbs_keys(final_response_map, pre_bowl_response_map)
                if key[0] == season and key[1] == "fbs"
            ],
            key=int,
        )
        for season in TARGET_SEASONS
    }
    fcs_populations = {}
    for season in TARGET_SEASONS:
        sizes = {
            target.population
            for (year, subdivision, _team_id), target in final_response_map.items()
            if year == season and subdivision == "fcs"
        }
        if len(sizes) != 1:
            raise ValueError(f"{season}: corrected FCS population is not unique")
        fcs_populations[str(season)] = sizes.pop()

    artifact_files: dict[str, object] = {
        "data_gate.json": gate,
        "target_snapshot_manifest.csv": snapshots,
        "training_row_manifest.csv": training_manifest,
        "preseason_feature_manifest.json": shared_inputs,
        "likelihood_game_manifest.csv": game_manifests,
        "rank_pair_composition.csv": composition,
        "fitted_models.json": fitted_models,
        "historical_likelihood_existing.json": likelihoods["existing"],
        "historical_likelihood_pre_bowl.json": likelihoods["pre_bowl"],
        "baseline_parity.json": baseline_parity,
        "policy_intervention.json": intervention,
    }
    for name, value in artifact_files.items():
        path = args.output_root / name
        if name.endswith(".csv"):
            write_csv(path, value)  # type: ignore[arg-type]
        else:
            write_json(path, value)

    bundle = {
        "schema_version": 1,
        "experiment": "issue-200-matched-history-context",
        "fit_spec_version": "issue-200-v1",
        "inference_configuration": INFERENCE_CONFIGURATION,
        "target_seasons": list(TARGET_SEASONS),
        "fcs_population_by_season": fcs_populations,
        "priors": prior_rows,
        "likelihoods": likelihoods,
        "source_manifest_sha256": sha256_json(input_records),
    }
    write_json(args.output_root / "pipeline_bundle.json", bundle)
    manifest = {
        "artifact_kind": "issue_200_matched_pipeline_bundle",
        "fit_spec_version": "issue-200-v1",
        "target_seasons": list(TARGET_SEASONS),
        "training_seasons": list(TRAINING_SEASONS),
        "policies": {
            "existing": "corrected final FBS targets and regular-season plus postseason games",
            "pre_bowl": "corrected pre-bowl FBS targets, regular-season FBS games, and unchanged FCS game policy",
        },
        "fitted_prior_arms": [
            "history_existing",
            "context_existing",
            "history_pre_bowl",
            "context_pre_bowl",
        ],
        "likelihood_arms": ["existing", "pre_bowl"],
        "fbs_target_population_by_season": {
            str(season): len(ids) for season, ids in fbs_target_ids.items()
        },
        "fcs_population_by_season": fcs_populations,
        "inference_configuration": INFERENCE_CONFIGURATION,
        "fitting_status": "complete",
        "interpretation": "2022-2025 are exploratory development seasons because prior research used them; no held-out model selection was performed.",
        "input_artifacts": input_records,
        "outputs": {},
    }
    output_paths = sorted(
        path
        for path in args.output_root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_paths
    }
    write_json(args.output_root / "manifest.json", manifest)
    write_report(
        args.output_root / "report.md",
        manifest,
        likelihoods,
        baseline_parity,
        intervention,
        composition,
    )
    # Include the report itself in the final output digest table.
    manifest["outputs"]["report.md"] = {
        "bytes": (args.output_root / "report.md").stat().st_size,
        "sha256": sha256_file(args.output_root / "report.md"),
    }
    write_json(args.output_root / "manifest.json", manifest)
    return manifest


def write_report(
    path: Path,
    manifest: dict[str, object],
    likelihoods: dict[str, dict[str, object]],
    baseline_parity: dict[str, object],
    intervention: dict[str, object],
    composition: list[dict[str, object]],
) -> None:
    lines = [
        "# Matched History and Context pre-bowl pipelines (issue 200)",
        "",
        "The four priors use the same corrected final-rank predictor inputs within each policy comparison. The pre-bowl policy substitutes only FBS response targets; FCS response and postseason policies remain on the corrected final corpus.",
        "",
        "All 2022-2025 results are exploratory development because prior research has informed these seasons. This build fits models and records parity; it does not compare held-out scores or select a model.",
        "",
        "## Data gate",
        "",
        f"- Corrected corpus gate: `{manifest.get('fitting_status', 'ready')}` after #198 collection completeness and #204 hash/support validation.",
        "- The pre-repair #198 manifest remains preserved with its historical `downstream_ready: false` value; this build verifies the corrected #204 target and pair artifacts separately.",
        "- Corrected target corpora and source game files are bound by input hashes. Response rows and predictor rows are keyed and hashed; the full predicted rank PMFs are serialized in `pipeline_bundle.json`.",
        "",
        "## Likelihood training",
        "",
        "| Policy | Games | Rank-pair pseudo-observations | Scale | df |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        fit = likelihoods[policy]
        lines.append(
            f"| {policy} | {fit['included_game_count']} | {fit['rank_pair_pseudo_observations']} | {float(fit['scale']):.6f} | {float(fit['degrees_of_freedom']):g} |"
        )
    lines.extend(
        [
            "",
            "Both likelihood fits retain the fixed eight-iteration weighted-pseudo IRLS recipe. No convergence criterion is evaluated; fit metadata records convergence as not assessed.",
        ]
    )
    composition_rows = [row for row in composition if row["pairing"] == "fbs-fcs"]
    lines.extend(
        [
            "",
            "FBS postseason rows are excluded together from the pre-bowl likelihood. Regular-season games, conference championships, and Army-Navy are retained. FCS postseason games keep the existing inclusion policy. `likelihood_game_manifest.csv` records each decision; `rank_pair_composition.csv` records same-system versus fallback composition and the FBS snapshot date for cross-subdivision rows.",
            "",
            f"Cross-subdivision composition rows recorded: {len(composition_rows)}.",
            "",
            "## Baseline parity",
            "",
            f"The frozen #192/#196 final-target references are retained by hash. The corrected final-control likelihood differs from the frozen likelihood by max absolute coefficient {baseline_parity['likelihood_comparison']['max_abs_coefficient_difference']:.6f}; see `baseline_parity.json` for the History and Context prior PMF deltas. The prior comparison includes #204 corpus repairs and the restored coaching-tenure source; both fitted issue 200 policies use the same source inventory, so the comparisons below isolate the target/game-policy intervention.",
            "",
            "## Pre-bowl intervention",
            "",
            "The following comparisons are between the two newly fitted policies on the corrected #204 corpus. They describe the intervention only; data-repair effects remain in the separate frozen-reference section above. A materially changed prior is defined as total variation greater than 0.01.",
            "",
            "### Prior PMFs by target season and model family",
            "",
            "| Season | Family | Teams | Mean TV | Max TV | Teams above TV 0.01 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in intervention["prior_pmf_differences"]:
        lines.append(
            f"| {row['target_season']} | {row['model_family']} | {row['team_count']} | {float(row['mean_total_variation_distance']):.6f} | {float(row['maximum_total_variation_distance']):.6f} | {row['materially_changed_team_count']} |"
        )
    lines.extend(
        [
            "",
            "### FBS target distributions by season",
            "",
            "| Season | Teams | Mean TV | Max TV | Changed targets | Different population support |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in intervention["fbs_target_distribution_differences"]:
        lines.append(
            f"| {row['target_season']} | {row['team_count']} | {float(row['mean_total_variation_distance']):.6f} | {float(row['maximum_total_variation_distance']):.6f} | {row['changed_distribution_team_count']} | {row['different_population_support_team_count']} |"
        )
    lines.extend(
        [
            "",
            "### Prior fitted coefficients",
            "",
            "| Season | Family | Component | Coefficients | Mean absolute difference | Max absolute difference |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in intervention["fitted_coefficient_differences"]:
        lines.append(
            f"| {row['target_season']} | {row['model_family']} | {row['component']} | {row['coefficient_count']} | {float(row['mean_absolute_difference']):.6g} | {float(row['maximum_absolute_difference']):.6g} |"
        )
    likelihood_diff = intervention["likelihood_differences"]
    lines.extend(
        [
            "",
            "### Fitted likelihood differences",
            "",
            f"The fitted likelihood coefficient vectors have mean absolute difference {float(likelihood_diff['mean_absolute_coefficient_difference']):.6g} and maximum absolute difference {float(likelihood_diff['maximum_absolute_coefficient_difference']):.6g}. Scale changes by {float(likelihood_diff['absolute_scale_difference']):.6f} in absolute terms. Included game counts are {likelihood_diff['included_game_count_existing']} (existing) and {likelihood_diff['included_game_count_pre_bowl']} (pre-bowl); rank-pair pseudo-observation counts are {likelihood_diff['rank_pair_pseudo_observations_existing']:,} and {likelihood_diff['rank_pair_pseudo_observations_pre_bowl']:,}, respectively. Pairing and rank-pair composition deltas are in `policy_intervention.json`.",
            "",
            "## Reproduction",
            "",
            "Run `uv run python scripts/build_matched_history_context_issue_200.py --help` and supply the corrected #204 corpus root, the retained source-feature root (including its non-empty `data/raw/cfbd/preseason/coach_tenures/` directory), the cached game root, and the #192 reference root. Coaching-tenure JSON inputs are validated and recorded individually by path, size, and SHA-256. The command writes the same versioned bundle, fit records, game/feature/target manifests, and report.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--game-root", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "fitting_status": result["fitting_status"],
                "output_count": len(result["outputs"]),
                "priors": list(result["fitted_prior_arms"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
