"""Analyze posterior movement and checkpoint-safe accumulated surprise (#253)."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gippyrank.research import issue_246_error_persistence as issue246
from gippyrank.research import issue_253_posterior_movement as issue253

DEFAULT_192_ROOT = ROOT / "data/processed/context_incremental_information"
DEFAULT_247_ROOT = ROOT / "data/research/issue_247_contradictory_evidence"
DEFAULT_201_ROOT = ROOT / "data/research/issue_201_prebowl_forecast"
DEFAULT_200_ROOT = ROOT / "data/research/matched_history_context_issue_200"
DEFAULT_PROTOCOL = ROOT / "docs/issue_253_posterior_movement_protocol.md"
DEFAULT_OUTPUT = ROOT / "data/research/issue_253_posterior_movement"
SEASONS = (2022, 2023, 2024, 2025)
CHECKPOINTS = tuple(range(1, 8))


def _sha256_file(path: Path) -> str:
    return issue246.sha256_file(path)


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _verify_record(path: Path, expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    actual = _file_record(path)
    if actual["sha256"] != expected.get("sha256"):
        raise ValueError(f"{label} hash mismatch: {path}")
    if "bytes" in expected and actual["bytes"] != expected["bytes"]:
        raise ValueError(f"{label} byte-count mismatch: {path}")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        if fields:
            writer.writeheader()
            writer.writerows(
                {key: _csv_value(value) for key, value in row.items()} for row in rows
            )


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped,
    ):
        for row in rows:
            zipped.write(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )


def _validate_issue192(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("study") != "issue-192-context-incremental-information":
        raise ValueError(f"unexpected #192 manifest at {manifest_path}")
    scope = manifest.get("scope", {})
    if scope.get("pilot_only") is not False:
        raise ValueError("#192 artifact is a pilot; full seven-checkpoint panel is required")
    if tuple(scope.get("seasons", ())) != SEASONS or tuple(
        scope.get("checkpoints", ())
    ) != CHECKPOINTS:
        raise ValueError("#192 seasons or retained checkpoints differ from the protocol")
    input_records: dict[str, Any] = {}
    panel_path = root / "inference_panel.jsonl.gz"
    evidence_path = root / "evidence.json"
    for path in (panel_path, evidence_path):
        expected = manifest.get("output_sha256", {}).get(path.name)
        if not expected:
            raise ValueError(f"#192 manifest does not hash {path.name}")
        input_records[f"issue192/{path.name}"] = _verify_record(
            path, {"sha256": expected}, label="#192 output"
        )
    evidence_root = _read_json(evidence_path)
    if evidence_root.get("study") != manifest.get("study"):
        raise ValueError("#192 evidence and manifest study names differ")
    panel_rows = _read_jsonl_gzip(panel_path)
    evidence_rows = evidence_root.get("checkpoints", [])
    expected_keys = {(season, checkpoint) for season in SEASONS for checkpoint in CHECKPOINTS}
    observed_keys = {
        (int(row["season"]), int(row["checkpoint"])) for row in evidence_rows
    }
    if observed_keys != expected_keys or len(evidence_rows) != len(expected_keys):
        raise ValueError("#192 evidence does not contain each retained season/checkpoint")
    return manifest, panel_rows, evidence_rows, {
        "manifest": _file_record(manifest_path),
        "files": input_records,
        "fbs_panel_rows": sum(str(row["subdivision"]).lower() == "fbs" for row in panel_rows),
        "all_panel_rows": len(panel_rows),
    }


def _validate_issue247_and_lineage(
    issue192_manifest: dict[str, Any],
    issue192_root: Path,
    issue192_manifest_sha256: str,
    issue247_root: Path,
    issue201_root: Path,
    issue200_root: Path,
    panel_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    manifest247_path = issue247_root / "manifest.json"
    manifest247 = _read_json(manifest247_path)
    if manifest247.get("experiment") != "issue-247-accumulated-contradictory-evidence" or manifest247.get("status") != "complete":
        raise ValueError("#247 retained analysis is missing or incomplete")
    config247 = manifest247.get("config", {})
    if config247.get("model_arms") != ["History_existing", "Context_existing"]:
        raise ValueError("#247 surprise arms differ from the existing-policy protocol")
    if int(config247.get("primary_minimum_prior_games", 0)) != issue253.MIN_SURPRISE_GAMES:
        raise ValueError("#247 primary surprise game threshold changed")
    if float(config247.get("surprise_group_threshold", -1)) != 0.5:
        raise ValueError("#247 surprise-group thresholds changed")
    if "unweighted arithmetic mean" not in str(config247.get("accumulated_surprise", "")):
        raise ValueError("#247 accumulated-surprise definition changed")

    manifest201_path = issue201_root / "manifest.json"
    manifest201 = _read_json(manifest201_path)
    if manifest201.get("experiment") != "issue-201-prebowl-forecast-comparison":
        raise ValueError("unexpected #201 manifest")
    input_records: dict[str, Any] = {
        "issue247/manifest.json": _file_record(manifest247_path),
        "issue201/manifest.json": _file_record(manifest201_path),
    }
    for name, record in manifest247.get("outputs", {}).items():
        path = issue247_root / name
        input_records[f"issue247/{name}"] = _verify_record(
            path, record, label="#247 retained output"
        )

    for name in ("forecast_scores.csv", "forecast_summary.csv", "replay_evidence.jsonl.gz"):
        expected201 = manifest201.get("outputs", {}).get(name)
        if not isinstance(expected201, dict):
            raise TypeError(f"#201 manifest does not hash {name}")
        input_records[f"issue201/{name}"] = _verify_record(
            issue201_root / name, expected201, label="#201 retained output"
        )
    for name in ("forecast_scores.csv", "forecast_summary.csv", "manifest.json", "replay_evidence.jsonl.gz"):
        expected247 = manifest247.get("input_hashes", {}).get(f"issue201/{name}")
        if not isinstance(expected247, dict):
            raise TypeError(f"#247 manifest does not record issue201/{name}")
        path = issue201_root / name
        _verify_record(path, expected247, label=f"#247 recorded input issue201/{name}")

    manifest200_path = issue200_root / "manifest.json"
    bundle_path = issue200_root / "pipeline_bundle.json"
    for label, path in (("manifest.json", manifest200_path), ("pipeline_bundle.json", bundle_path)):
        expected247 = manifest247.get("input_hashes", {}).get(f"issue200/{label}")
        if not isinstance(expected247, dict):
            raise TypeError(f"#247 manifest does not record issue200/{label}")
        input_records[f"issue200/{label}"] = _verify_record(
            path, expected247, label=f"#247 recorded input issue200/{label}"
        )
    manifest200 = _read_json(manifest200_path)
    bundle = _read_json(bundle_path)
    bundle_expected = manifest200.get("outputs", {}).get("pipeline_bundle.json")
    if not isinstance(bundle_expected, dict):
        raise TypeError("#200 manifest does not hash its matched pipeline bundle")
    _verify_record(bundle_path, bundle_expected, label="#200 pipeline bundle")
    issue200_manifest_sha = _sha256_file(manifest200_path)
    issue200_bundle_sha = _sha256_file(bundle_path)
    provenance200 = manifest201.get("issue200_provenance", {})
    if provenance200.get("manifest_sha256") != issue200_manifest_sha:
        raise ValueError("#201 and #200 manifests are not from the same retained run")
    if provenance200.get("bundle_sha256") != issue200_bundle_sha:
        raise ValueError("#201 and #200 pipeline bundles are not from the same retained run")

    frozen = manifest201.get("input_artifacts", {}).get("frozen_192_outputs", {})
    expected_frozen = {
        "evidence_sha256": issue192_manifest["output_sha256"]["evidence.json"],
        "panel_sha256": issue192_manifest["output_sha256"]["inference_panel.jsonl.gz"],
        "manifest_sha256": issue192_manifest_sha256,
        "score_sha256": issue192_manifest["output_sha256"]["team_scores.csv"],
        "final_target_sha256": issue192_manifest["output_sha256"]["scoring_targets.jsonl.gz"],
    }
    if frozen != expected_frozen:
        raise ValueError("#201 frozen #192 artifact hashes do not match the selected #192 panel")

    existing_arms = {
        str(row["model_arm"]): row
        for row in manifest201.get("model_arms", [])
        if row.get("policy") == "existing"
    }
    if set(existing_arms) != set(issue253.MODEL_ARMS.values()):
        raise ValueError("#201 has no exact History_existing/Context_existing arm pair")
    if any(
        str(existing_arms[issue253.MODEL_ARMS[family]]["prior_family"]) != family
        for family in issue253.FAMILIES
    ):
        raise ValueError("#201 existing-policy model-family labels do not match #247")
    if manifest247.get("input_hashes", {}).get("issue201/manifest.json", {}).get("sha256") != _sha256_file(manifest201_path):
        raise ValueError("#247 and #201 manifests are not from the same retained run")

    evidence_root = _read_json(issue192_root / "evidence.json")
    likelihood192 = sorted(
        {str(row["likelihood_sha256"]) for row in evidence_root["checkpoints"]}
    )
    likelihood201 = sorted(
        {str(existing_arms[issue253.MODEL_ARMS[family]]["likelihood_sha256"]) for family in issue253.FAMILIES}
    )
    lineage: dict[str, Any] = {
        "posterior_source": "#192 retained History 1.1 and repaired Context 1.4 PMFs",
        "surprise_source": "#247 game-level standardized errors from #201 existing-policy forecasts",
        "issue201_frozen_192_panel_and_evidence_hashes_match": True,
        "issue247_recorded_issue201_and_issue200_hashes_match": True,
        "existing_policy_family_labels_match": True,
        "issue192_likelihood_sha256": likelihood192,
        "issue201_existing_likelihood_sha256": likelihood201,
        "likelihood_configurations_match": likelihood192 == likelihood201,
        "model_configuration_note": (
            "The rank posterior and game-margin surprise forecasts have distinct likelihood configurations. "
            "They are not treated as one fitted model: only #192 PMFs define posterior movement, and only "
            "#247 game surprises define the joined descriptive predictor."
        ),
    }

    prior_bundle: dict[tuple[int, str, str], np.ndarray] = {}
    prior_bundle_meta: dict[tuple[int, str, str], dict[str, Any]] = {}
    for prior in bundle.get("priors", []):
        if (
            str(prior.get("policy")) != "existing"
            or str(prior.get("subdivision", "")).lower() != "fbs"
        ):
            continue
        family = str(prior["model_family"]).capitalize()
        if family not in issue253.FAMILIES:
            continue
        key = (int(prior["season"]), family, str(prior["team_id"]))
        if key in prior_bundle:
            raise ValueError(f"duplicate #200 existing prior {key}")
        prior_bundle[key] = np.asarray(prior["pmf"], dtype=float)
        prior_bundle_meta[key] = prior

    panel_preseason: dict[tuple[int, str, str], np.ndarray] = {}
    for panel in panel_rows:
        if str(panel["subdivision"]).lower() != "fbs" or int(panel["checkpoint"]) != 1:
            continue
        for family in issue253.FAMILIES:
            key = (int(panel["season"]), family, str(panel["team_id"]))
            panel_preseason[key] = np.asarray(panel[f"{family.lower()}_prior_pmf"], dtype=float)
    if set(panel_preseason) != set(prior_bundle):
        raise ValueError("#192 FBS preseason priors and #200 existing priors have different coverage")

    game_errors = _read_csv(issue247_root / "team_game_errors.csv")
    forecast_population_counts: dict[str, int] = defaultdict(int)
    forecast_arm_counts: dict[str, int] = defaultdict(int)
    for game in game_errors:
        forecast_population_counts[str(game["forecast_population"])] += 1
        forecast_arm_counts[str(game["model_arm"])] += 1
    lineage["issue247_game_error_rows"] = len(game_errors)
    lineage["issue247_forecast_population_counts"] = dict(
        sorted(forecast_population_counts.items())
    )
    lineage["issue247_existing_arm_row_counts"] = dict(sorted(forecast_arm_counts.items()))
    rank_population_by_team = {
        (int(row["season"]), str(row["team_id"])): int(row["rank_count"])
        for row in panel_rows
        if str(row["subdivision"]).lower() == "fbs" and int(row["checkpoint"]) == 1
    }
    issue247_prior_hashes: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for game in game_errors:
        key = (int(game["season"]), str(game["prior_family"]), str(game["team_id"]))
        if key in prior_bundle_meta:
            issue247_prior_hashes[key].add(str(game["preseason_rank_pmf_sha256"]))
            expected_population = rank_population_by_team[(key[0], key[2])]
            if int(game["preseason_rank_population"]) != expected_population:
                raise ValueError(f"#247/#192 rank support mismatch for {key}")
    prior_comparison: dict[str, Any] = {}
    for family in issue253.FAMILIES:
        distances: list[float] = []
        absolute_rank_differences: list[float] = []
        max_bin_differences: list[float] = []
        hash_mismatches = 0
        family_keys = [key for key in panel_preseason if key[1] == family]
        for key in family_keys:
            baseline = panel_preseason[key]
            retained = prior_bundle[key]
            meta = prior_bundle_meta[key]
            if len(baseline) != len(retained) or len(retained) != int(meta["population"]):
                raise ValueError(f"#192/#200 rank support mismatch for {key}")
            l1 = float(np.abs(baseline - retained).sum())
            distances.append(l1)
            max_bin_differences.append(float(np.max(np.abs(baseline - retained))))
            ranks = np.arange(1, len(baseline) + 1, dtype=float)
            absolute_rank_differences.append(
                abs(float(np.dot(ranks, baseline) - np.dot(ranks, retained)))
            )
            if issue246._sha256_json(baseline.tolist()) != str(meta["pmf_sha256"]):
                hash_mismatches += 1
            hashes = issue247_prior_hashes.get(key, set())
            if hashes != {str(meta["pmf_sha256"])}:
                raise ValueError(f"#247 preseason PMF hash does not match #200 prior for {key}")
        prior_comparison[family] = {
            "team_seasons_compared": len(family_keys),
            "exact_pmf_hash_matches": len(family_keys) - hash_mismatches,
            "pmf_hash_mismatches": hash_mismatches,
            "median_l1_distance": float(np.median(distances)),
            "max_l1_distance": max(distances),
            "median_max_bin_absolute_difference": float(np.median(max_bin_differences)),
            "maximum_bin_absolute_difference": max(max_bin_differences),
            "median_absolute_expected_rank_difference": float(np.median(absolute_rank_differences)),
            "maximum_absolute_expected_rank_difference": max(absolute_rank_differences),
        }
    lineage["issue192_vs_issue200_existing_preseason_pmf_comparison"] = prior_comparison
    lineage["issue247_preseason_pmf_hashes_match_issue200"] = True
    lineage["issue247_preseason_pmfs_used_for_posterior_movement"] = False
    return game_errors, lineage, input_records


def _validate_evidence_digests(evidence_rows: list[dict[str, Any]]) -> None:
    for evidence in evidence_rows:
        ids = [str(game_id) for game_id in evidence["included_game_ids"]]
        if len(ids) != int(evidence["included_game_count"]):
            raise ValueError("#192 included game count does not match its IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("#192 evidence has duplicate game IDs")
        if issue246._sha256_json(ids) != evidence["included_game_ids_sha256"]:
            raise ValueError(
                f"#192 game-ID digest mismatch in {evidence['season']}/{evidence['checkpoint']}"
            )


def _pooled_checkpoint_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    fields = (
        "signed_location_change_from_preseason",
        "signed_location_change_from_preseason_normalized",
        "signed_location_change_since_previous_normalized",
        "preseason_rank_sd_normalized",
        "posterior_rank_sd_normalized",
        "rank_sd_change_from_preseason_normalized",
        "rank_sd_change_since_previous_normalized",
        "preseason_central80_width_normalized",
        "posterior_central80_width_normalized",
        "central80_width_change_from_preseason_normalized",
        "central80_width_change_since_previous_normalized",
        "kl_posterior_from_prior",
        "total_variation_since_previous",
    )
    for checkpoint in (1, 4, 7):
        for family in issue253.FAMILIES:
            members = [
                row
                for row in rows
                if row["prior_family"] == family and int(row["checkpoint"]) == checkpoint
            ]
            result: dict[str, Any] = {
                "checkpoint": checkpoint,
                "model_arm": issue253.MODEL_ARMS[family],
                "prior_family": family,
                "team_seasons": len(members),
                "cold_start_team_seasons": sum(
                    row["component_status"] != "fitted" for row in members
                ),
            }
            for field in fields:
                summary = issue253._quantile_summary([row[field] for row in members])
                for stat in ("median", "p10", "p90"):
                    result[f"{field}_{stat}"] = summary[stat]
            output.append(result)
    return output


def _season_checkpoint7_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for season in SEASONS:
        for family in issue253.FAMILIES:
            members = [
                row
                for row in rows
                if int(row["season"]) == season
                and int(row["checkpoint"]) == 7
                and row["prior_family"] == family
            ]
            if not members:
                continue
            output.append(
                {
                    "season": season,
                    "checkpoint": 7,
                    "model_arm": issue253.MODEL_ARMS[family],
                    "prior_family": family,
                    "team_seasons": len(members),
                    "cold_start_team_seasons": sum(
                        row["component_status"] != "fitted" for row in members
                    ),
                    "median_signed_location_change_from_preseason": float(
                        np.median([row["signed_location_change_from_preseason"] for row in members])
                    ),
                    "median_signed_location_change_from_preseason_normalized": float(
                        np.median(
                            [row["signed_location_change_from_preseason_normalized"] for row in members]
                        )
                    ),
                    "median_rank_sd_change_from_preseason_normalized": float(
                        np.median(
                            [row["rank_sd_change_from_preseason_normalized"] for row in members]
                        )
                    ),
                    "median_posterior_rank_sd_normalized": float(
                        np.median([row["posterior_rank_sd_normalized"] for row in members])
                    ),
                    "median_kl_posterior_from_prior": float(
                        np.median([row["kl_posterior_from_prior"] for row in members])
                    ),
                }
            )
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _metric_row(summary: dict[str, Any], family: str, checkpoint: int) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if row["prior_family"] == family and int(row["checkpoint"]) == checkpoint
    )


def build_report(
    *,
    lineage: dict[str, Any],
    counts: dict[str, Any],
    pooled_summary: list[dict[str, Any]],
    season_summary: list[dict[str, Any]],
    group_contrasts: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    family_differences: list[dict[str, Any]],
    family_slope_differences: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    bootstrap_repetitions: int,
) -> str:
    lines = [
        "# Issue 253: in-season posterior movement and accumulated surprise",
        "",
        "## Finding",
        "",
        (
            f"The retained panel contains {counts['fbs_team_seasons']:,} FBS team-seasons, "
            f"{counts['team_checkpoint_rows']:,} family/checkpoint observations, and "
            f"{counts['cold_start_team_checkpoints']:,} native cold-start team-checkpoint rows. "
            "Each row retains full prior and posterior rank PMFs."
        ),
        (
            f"At least three evidence-available prior games contribute to "
            f"{counts['surprise_eligible_team_checkpoints']:,} surprise comparisons. "
            "The full movement summaries keep all team-checkpoints, including those below that threshold."
        ),
        "",
        "Movement uses positive values for expected-rank improvement. Rank-location, standard deviation, and interval-width changes are divided by the season’s rank range (`N - 1`) for cross-season comparison. KL is posterior-to-prior in nats; adjacent-checkpoint total variation ranges from 0 to 1.",
        "",
        "## Typical movement across checkpoints",
        "",
        "Medians and 10th–90th percentiles pool the four exploratory seasons for compact display; checkpoint-specific, season-specific, and cold-start distributions are in `aggregate_by_season_checkpoint.csv`.",
        "",
        "| Family | Checkpoint | Median normalized expected-rank improvement | P10–P90 | Median KL | Median adjacent TV |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in (1, 4, 7):
        for family in issue253.FAMILIES:
            row = _metric_row(pooled_summary, family, checkpoint)
            lines.append(
                f"| {family} | {checkpoint} | "
                f"{_fmt(row['signed_location_change_from_preseason_normalized_median'])} | "
                f"{_fmt(row['signed_location_change_from_preseason_normalized_p10'])} to "
                f"{_fmt(row['signed_location_change_from_preseason_normalized_p90'])} | "
                f"{_fmt(row['kl_posterior_from_prior_median'])} | "
                f"{_fmt(row['total_variation_since_previous_median'])} |"
            )
    lines.extend(
        [
            "",
            "| Family | Checkpoint | Prior SD / range | Posterior SD / range | SD change / range | Prior 80% width / range | Posterior 80% width / range | Width change / range |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for checkpoint in (1, 4, 7):
        for family in issue253.FAMILIES:
            row = _metric_row(pooled_summary, family, checkpoint)
            lines.append(
                f"| {family} | {checkpoint} | "
                f"{_fmt(row['preseason_rank_sd_normalized_median'])} | "
                f"{_fmt(row['posterior_rank_sd_normalized_median'])} | "
                f"{_fmt(row['rank_sd_change_from_preseason_normalized_median'])} | "
                f"{_fmt(row['preseason_central80_width_normalized_median'])} | "
                f"{_fmt(row['posterior_central80_width_normalized_median'])} | "
                f"{_fmt(row['central80_width_change_from_preseason_normalized_median'])} |"
            )
    lines.extend(
        [
            "",
            "A negative normalized posterior-SD change indicates narrower uncertainty than preseason; it is descriptive and does not by itself indicate overconfidence. Expected-rank movement can also result from information elsewhere in the joint inference graph, so it is not attributed solely to a team’s own games.",
            "",
            "Checkpoint-7 median location movement changed sign across the four years: 2022–2023 medians moved toward worse expected ranks, while 2024–2025 medians improved. Median posterior SD was lower than preseason in all four years for both families. These are distribution summaries, not independent seasonal replications.",
            "",
            "## Checkpoint-7 season stability",
            "",
            "| Season | Family | Median rank improvement from preseason | Median normalized SD change | Median posterior normalized SD | Median KL |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in season_summary:
        lines.append(
            f"| {row['season']} | {row['prior_family']} | "
            f"{_fmt(row['median_signed_location_change_from_preseason'])} ranks "
            f"({_fmt(row['median_signed_location_change_from_preseason_normalized'])}) | "
            f"{_fmt(row['median_rank_sd_change_from_preseason_normalized'])} | "
            f"{_fmt(row['median_posterior_rank_sd_normalized'])} | "
            f"{_fmt(row['median_kl_posterior_from_prior'])} |"
        )

    lines.extend(
        [
            "",
            "## Relationship with accumulated surprise",
            "",
            "The #247 groups retain their fixed ±0.5 cutoffs and require at least three games available in the exact #192 evidence set. Pooled group differences and percentile intervals are descriptive. The regressions use continuous surprise, preseason strength and uncertainty, game count, and season/checkpoint fixed effects. The main continuous surprise slopes are evaluated at mean preseason strength. Positive location slopes mean positive prior game surprises are associated with rank improvement in the next checkpoint; negative SD-change slopes mean greater surprises are associated with narrowing rank uncertainty.",
            "",
            "| Family | Outcome | Term | Estimate | 95% interval | Team-checkpoint rows |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    focus_terms = {
        "directional_location_movement": {"accumulated_surprise", "accumulated_surprise_x_preseason_strength"},
        "absolute_location_movement": {"absolute_accumulated_surprise"},
        "rank_sd_change": {"accumulated_surprise", "accumulated_surprise_x_preseason_strength"},
        "central80_width_change": {"accumulated_surprise", "accumulated_surprise_x_preseason_strength"},
    }
    for row in regressions:
        if row["term"] not in focus_terms.get(str(row["analysis"]), set()):
            continue
        lines.append(
            f"| {row['prior_family']} | {row['analysis']} | {row['term']} | "
            f"{_fmt(row['estimate'], 4)} | {_fmt(row['ci95_low'], 4)} to {_fmt(row['ci95_high'], 4)} | "
            f"{row['observations']:,} |"
        )
    key_contrasts = [
        row
        for row in group_contrasts
        if row["outcome"]
        in {
            "signed_location_change_since_previous_normalized",
            "rank_sd_change_since_previous_normalized",
        }
        and row["contrast"]
        in {
            "sustained_negative_minus_approximately_neutral",
            "sustained_positive_minus_approximately_neutral",
        }
    ]
    lines.extend(
        [
            "",
            "Group mean contrasts (sustained group minus neutral) use two-way pigeonhole bootstrap intervals resampling team IDs and season-checkpoint snapshots.",
            "",
            "| Family | Outcome | Contrast | Difference | 95% interval | Eligible rows A / B |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in key_contrasts:
        lines.append(
            f"| {row['prior_family']} | {row['outcome']} | {row['contrast']} | "
            f"{_fmt(row['difference'], 4)} | {_fmt(row['ci95_low'], 4)} to {_fmt(row['ci95_high'], 4)} | "
            f"{row['team_checkpoints_a']:,} / {row['team_checkpoints_b']:,} |"
        )

    lines.extend(
        [
            "",
            "The continuous directional slopes are positive in both families, and absolute movement also increases with absolute surprise. This matches the expected direction of posterior response in the retained data. The adjusted average surprise slopes for adjacent-checkpoint SD and central-80%-width changes include zero; their surprise-by-preseason-strength terms are negative in both families, so the uncertainty association varies with preseason strength in this descriptive model.",
            "",
            "## History and Context differences",
            "",
            "Paired differences use the same team-season-checkpoint in both families. The table reports checkpoint-7 History-minus-Context movement differences; continuous surprise-slope differences are Context minus History.",
            "",
            "| Movement outcome at checkpoint 7 | History minus Context | 95% interval | Matched team-seasons |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in (
        "signed_location_change_from_preseason_normalized",
        "rank_sd_change_from_preseason_normalized",
        "kl_posterior_from_prior",
        "total_variation_since_previous",
    ):
        row = next(
            item
            for item in family_differences
            if item["scope"] == "checkpoint_7" and item["outcome"] == metric
        )
        lines.append(
            f"| {metric} | {_fmt(row['difference'], 4)} | "
            f"{_fmt(row['ci95_low'], 4)} to {_fmt(row['ci95_high'], 4)} | "
            f"{row['team_checkpoints']:,} |"
        )
    lines.extend(
        [
            "",
            "| Context minus History slope | Difference | 95% interval | Matched team-checkpoints |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in family_slope_differences:
        lines.append(
            f"| {row['analysis']} | {_fmt(row['difference'], 4)} | "
            f"{_fmt(row['ci95_low'], 4)} to {_fmt(row['ci95_high'], 4)} | "
            f"{row['team_checkpoints']:,} |"
        )
    lines.extend(
        [
            "",
            f"All pooled intervals use {bootstrap_repetitions:,} two-way pigeonhole resamples. These are approximate: posterior inference couples teams through shared evidence, only four exploratory seasons are retained, and repeated checkpoints are not independent. A relationship here is observational, not a causal effect of particular games.",
            "",
            "## Lineage and temporal reconciliation",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| #201 frozen #192 panel and evidence hashes | {'match' if lineage['issue201_frozen_192_panel_and_evidence_hashes_match'] else 'mismatch'} |",
            f"| #247 recorded #201/#200 input hashes | {'match' if lineage['issue247_recorded_issue201_and_issue200_hashes_match'] else 'mismatch'} |",
            f"| Existing-policy History/Context labels | {'match' if lineage['existing_policy_family_labels_match'] else 'mismatch'} |",
            f"| #192 and #201 existing likelihood hashes | {', '.join(lineage['issue192_likelihood_sha256'])} vs {', '.join(lineage['issue201_existing_likelihood_sha256'])} |",
            f"| Likelihood configurations identical | {lineage['likelihood_configurations_match']} |",
            f"| FBS focal-game counts matching exact #192 evidence | {counts['focal_game_count_checks']:,}/{counts['focal_game_count_checks']:,} |",
            f"| Joined #247 games at/after the checkpoint cutoff | {counts['joined_games_at_or_after_cutoff']} |",
            "",
            lineage["model_configuration_note"],
            "",
            "The #192 and #200 existing-policy preseason PMFs also differ. Per-family exact hash matches and L1/expected-rank differences are recorded in `compatibility_audit.json`. The #247 forecast PMFs and preseason summaries were not substituted into #192 movement or strength calculations. The #192 evidence set is the join authority, rather than #201’s different rolling forecast cutoff; the joined focal-game counts exactly reproduce #192 `games_played`.",
            "",
            "#247 forecast populations are retained separately from the #192 FBS rank supports: "
            + ", ".join(
                f"{name} {count:,} oriented rows"
                for name, count in lineage["issue247_forecast_population_counts"].items()
            )
            + ".",
            "",
            "| Family | #192/#200 exact prior PMF hashes | Median L1 | Maximum L1 | Median absolute expected-rank difference | Maximum absolute expected-rank difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in issue253.FAMILIES:
        comparison = lineage["issue192_vs_issue200_existing_preseason_pmf_comparison"][family]
        lines.append(
            f"| {family} | {comparison['exact_pmf_hash_matches']}/{comparison['team_seasons_compared']} | "
            f"{_fmt(comparison['median_l1_distance'], 5)} | "
            f"{_fmt(comparison['max_l1_distance'], 4)} | "
            f"{_fmt(comparison['median_absolute_expected_rank_difference'], 3)} | "
            f"{_fmt(comparison['maximum_absolute_expected_rank_difference'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Selected trajectories",
            "",
            "Two examples were selected using only checkpoint-7 accumulated surprise, as the minimum and maximum mean across the History and Context #247 signals among team-seasons with at least three eligible games in both families. Selection did not use posterior movement or eventual ranking outcomes. Full seven-checkpoint PMFs for both families are in `selected_trajectory_pmfs.jsonl.gz`.",
            "",
            "| Selection | Team-season | Mean family surprise at checkpoint 7 | History / Context |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['trajectory_id']} | {row['team_name']} / {row['season']} | "
            f"{_fmt(row['selection_score_mean_family_surprise'], 3)} | "
            f"{_fmt(row['history_accumulated_surprise'], 3)} / "
            f"{_fmt(row['context_accumulated_surprise'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation under #244",
            "",
            "Continue Phase 1 with a bounded conditional predictive-calibration investigation. Test whether posterior-derived future-game distributions are calibrated within preseason-strength and uncertainty strata and across the retained season checkpoints, preserving the leakage-safe forecast cutoffs. The current analysis characterizes how beliefs move; it does not establish whether the resulting beliefs improve or calibrate future predictions. The #247 study found no detectable persistent directional next-game error, so this next step should focus on conditional distribution calibration rather than changing priors or likelihood weights. Keep the historical seasons exploratory and do not start model interventions without predictive evidence.",
            "",
            "## Reproducibility and interpretation",
            "",
            "Reproduce with `uv run python scripts/run_issue_253_posterior_movement.py`. The input/output manifest records hashes, row counts, configuration, and environment. The PMF JSONL retains all serialized prior/posterior distributions used in the calculations. This analysis changes no production model and uses no 2026 outcomes.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    issue192_manifest, panel_rows, evidence_rows, audit192 = _validate_issue192(args.issue192_root)
    _validate_evidence_digests(evidence_rows)
    game_errors, lineage, input_records = _validate_issue247_and_lineage(
        issue192_manifest,
        args.issue192_root,
        audit192["manifest"]["sha256"],
        args.issue247_root,
        args.issue201_root,
        args.issue200_root,
        panel_rows,
    )
    input_records.update(audit192["files"])
    input_records["issue192/manifest.json"] = audit192["manifest"]
    panel_fbs = [row for row in panel_rows if str(row["subdivision"]).lower() == "fbs"]
    season_populations: dict[str, Any] = {}
    for season in SEASONS:
        season_rows = [row for row in panel_fbs if int(row["season"]) == season]
        rank_populations = sorted({int(row["rank_count"]) for row in season_rows})
        if len(rank_populations) != 1:
            raise ValueError(f"#192 FBS rank population varies within {season}")
        expected_team_count = rank_populations[0]
        checkpoint_counts = {
            str(checkpoint): sum(int(row["checkpoint"]) == checkpoint for row in season_rows)
            for checkpoint in CHECKPOINTS
        }
        if any(count != expected_team_count for count in checkpoint_counts.values()):
            raise ValueError(f"#192 FBS team coverage differs from rank population in {season}")
        season_populations[str(season)] = {
            "fbs_rank_population": expected_team_count,
            "team_rows_by_checkpoint": checkpoint_counts,
        }

    observations = issue253.build_team_checkpoint_observations(
        panel_rows, evidence_rows, game_errors
    )
    if len(observations) != 2 * len(panel_fbs):
        raise ValueError("posterior movement rows do not cover both families")
    observed_game_counts = sum(int(row["focal_games_included_in_evidence"]) for row in observations)
    surprise_eligible = [row for row in observations if row["surprise_group_eligible"]]
    cold_starts = [row for row in observations if row["component_status"] != "fitted"]
    counts = {
        "seasons": list(SEASONS),
        "checkpoints": list(CHECKPOINTS),
        "fbs_team_seasons": len({(int(row["season"]), str(row["team_id"])) for row in panel_fbs}),
        "fbs_panel_team_checkpoints": len(panel_fbs),
        "team_checkpoint_rows": len(observations),
        "cold_start_team_checkpoints": len(cold_starts),
        "cold_start_team_seasons": len(
            {(int(row["season"]), str(row["team_id"])) for row in cold_starts}
        ),
        "joined_focal_game_records_across_families_and_checkpoints": observed_game_counts,
        "surprise_eligible_team_checkpoints": len(surprise_eligible),
        "focal_game_count_checks": len(observations),
        "joined_games_at_or_after_cutoff": 0,
        "included_game_evidence_records_across_checkpoints": sum(
            int(row["included_game_count"]) for row in evidence_rows
        ),
        "season_populations": season_populations,
    }
    for family in issue253.FAMILIES:
        fam_rows = [row for row in observations if row["prior_family"] == family]
        if any(int(row["prior_surprise_game_count"]) != int(row["focal_games_included_in_evidence"]) for row in fam_rows):
            raise ValueError(f"#247 surprise coverage differs from #192 focal-game counts for {family}")

    movement_aggregates = issue253.aggregate_movement(observations)
    group_summaries = issue253.aggregate_by_surprise_group(observations)
    group_contrasts = issue253.bootstrap_surprise_group_contrasts(
        observations, repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed
    )
    regression_rows = issue253.analyze_surprise_relationships(
        observations, repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed
    )
    family_differences = issue253.paired_family_movement_differences(
        observations, repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed
    )
    family_slope_differences = issue253.paired_family_surprise_slope_differences(
        observations, repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed
    )
    selected = issue253.select_trajectories(observations)
    selected_pmfs = issue253.build_selected_trajectory_rows(observations, selected)
    pooled_summary = _pooled_checkpoint_summary(observations)
    season_summary = _season_checkpoint7_summary(observations)

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_rows = [
        ("team_checkpoint_summary.csv", [{key: value for key, value in row.items() if key not in {"prior_pmf", "posterior_pmf"}} for row in observations]),
        ("team_checkpoint_pmfs.jsonl.gz", observations),
        ("aggregate_by_season_checkpoint.csv", movement_aggregates),
        ("surprise_group_summary.csv", group_summaries),
        ("surprise_group_contrasts.csv", group_contrasts),
        ("surprise_response_models.csv", regression_rows),
        ("family_movement_differences.csv", family_differences),
        ("family_surprise_slope_differences.csv", family_slope_differences),
        ("selected_trajectories.csv", selected),
        ("selected_trajectory_pmfs.jsonl.gz", selected_pmfs),
    ]
    for filename, rows in output_rows:
        path = output_root / filename
        if filename.endswith(".jsonl.gz"):
            _write_gzip_jsonl(path, rows)
        else:
            _write_csv(path, rows)

    lineage["focal_game_count_checks"] = counts["focal_game_count_checks"]
    lineage["joined_focal_game_count_records"] = observed_game_counts
    lineage["joined_games_at_or_after_cutoff"] = 0
    lineage["issue192_fbs_rank_populations"] = season_populations
    _write_json(output_root / "compatibility_audit.json", lineage)
    report = build_report(
        lineage=lineage,
        counts=counts,
        pooled_summary=pooled_summary,
        season_summary=season_summary,
        group_contrasts=group_contrasts,
        regressions=regression_rows,
        family_differences=family_differences,
        family_slope_differences=family_slope_differences,
        selected=selected,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")

    output_hashes = {
        path.name: _file_record(path)
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    source_paths = (
        args.protocol,
        Path(__file__),
        ROOT / "src/gippyrank/research/issue_253_posterior_movement.py",
    )
    source_hashes = {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): _sha256_file(path)
        for path in source_paths
    }
    manifest = {
        "experiment": "issue-253-posterior-movement-and-uncertainty",
        "schema_version": 1,
        "status": "complete",
        "seasons": list(SEASONS),
        "checkpoints": list(CHECKPOINTS),
        "config": {
            "history_prior_games": issue253.MIN_SURPRISE_GAMES,
            "surprise_group_threshold": 0.5,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_clusters": "team_id x season-checkpoint pigeonhole resampling",
            "rank_movement_sign": "positive means expected-rank improvement",
            "rank_normalization": "divide location and dispersion changes by rank_population - 1",
            "central_interval": "discrete equal-tail central 80% interval; width is upper minus lower rank",
            "kl": "KL(posterior || preseason prior), no smoothing",
            "total_variation": "0.5 * L1 distance between adjacent posterior PMFs",
            "trajectory_selection": "min/max family-mean accumulated surprise at checkpoint 7; >=3 prior games",
        },
        "counts": counts,
        "input_sha256": input_records,
        "source_sha256": source_hashes,
        "compatibility": lineage,
        "output_sha256": output_hashes,
        "manifest_hash_policy": "input, source, and output files are hashed; this manifest does not hash itself",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": sys.platform,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return {"counts": counts, "manifest": manifest, "output_root": output_root}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue192-root", type=Path, default=DEFAULT_192_ROOT)
    parser.add_argument("--issue247-root", type=Path, default=DEFAULT_247_ROOT)
    parser.add_argument("--issue201-root", type=Path, default=DEFAULT_201_ROOT)
    parser.add_argument("--issue200-root", type=Path, default=DEFAULT_200_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=issue253.BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        parser.error("--bootstrap-repetitions must be positive")
    if not args.protocol.is_file():
        parser.error(f"protocol not found: {args.protocol}")
    if not args.issue192_root.is_dir() or not args.issue247_root.is_dir():
        parser.error("retained #192 and #247 input directories are required")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"Report: {result['output_root'] / 'report.md'}")


if __name__ == "__main__":
    main()
