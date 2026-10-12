"""Build an explicit, possibly incomplete, FBS pre-bowl target corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

from gippyrank.data.cfbd_games import read_cfbd_game_rows
from gippyrank.data.massey import (
    NORMALIZED_FIELDS,
    build_team_matches,
    parse_cfbd_teams,
)
from gippyrank.data.massey_prebowl import (
    APPROVED_PHASE_EXCEPTIONS,
    SEASONS,
    PreBowlWindow,
    audit_target_rank_support,
    latest_cmp_date_in_window,
    normalized_rows,
    read_dated_fbs_export,
    read_kaggle_snapshot,
    snapshot_date_is_accepted,
    snapshot_filename,
    validate_raw_provenance,
    validate_snapshot,
)
from gippyrank.modeling import (
    CONSTITUENT_RANK_PAIR_FIELDS,
    audit_population_coverage,
    build_constituent_rank_pairs,
    build_distributions,
    coverage_table,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = (
    ROOT / "data/research/massey_prebowl_issue_198/collection_checklist.csv"
)
DEFAULT_REPORT_ROOT = ROOT / "data/research/massey_prebowl_issue_198"
TARGET_SUPPORT_FIELDS = (
    "season",
    "subdivision",
    "team_id",
    "team_name",
    "team_population",
    "usable_systems",
    "out_of_support_ranks",
    "out_of_support_observations",
    "out_of_support_probability_mass",
    "present_in_final_corpus",
    "final_out_of_support_ranks",
    "final_out_of_support_probability_mass",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalidate_readiness(path: Path) -> None:
    """Atomically make an existing readiness manifest unusable during rebuild."""
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest["status"] = "rebuild_in_progress"
    manifest["downstream_ready"] = False
    manifest["readiness_invalidation_reason"] = (
        "A rebuild started; readiness remains false until validated artifacts "
        "and a replacement manifest are published."
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_csv(
    path: Path, fields: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NORMALIZED_FIELDS:
            raise ValueError(f"Unexpected normalized Massey schema in {path}")
        return list(reader)


def load_team_matches(
    raw_root: Path, season: int, names: set[str]
) -> tuple[dict, dict, list[str]]:
    team_path = raw_root / "cfbd" / "teams" / f"{season}.json"
    with team_path.open(encoding="utf-8") as handle:
        teams = parse_cfbd_teams(json.load(handle))

    matches, ambiguous, unmapped = build_team_matches(names, teams)
    fallback_seasons: list[int] = []
    if season == 2020 and (ambiguous or unmapped):
        for fallback_season in (2019, 2021):
            fallback_seasons.append(fallback_season)
            fallback_path = raw_root / "cfbd" / "teams" / f"{fallback_season}.json"
            with fallback_path.open(encoding="utf-8") as handle:
                teams.extend(parse_cfbd_teams(json.load(handle)))
            matches, ambiguous, unmapped = build_team_matches(names, teams)
            if not ambiguous and not unmapped:
                break
    return matches, ambiguous, unmapped


def kaggle_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[2].strip() for row in csv.reader(handle) if len(row) >= 3}


def export_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        return {row[0].strip() for row in reader if row and row[0].strip()}


def load_checklist(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "season",
        "last_fbs_conference_championship_date",
        "first_fbs_postseason_game_date",
        "window_first_full_date",
        "window_last_full_date",
        "snapshot_date",
        "snapshot_source",
        "snapshot_file",
        "phase_exception_decision",
        "phase_exception_rationale",
        "source_url",
        "collection_status",
        "notes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Checklist {path} is missing required columns")
    by_season = {int(row["season"]): row for row in rows}
    if set(by_season) != set(SEASONS):
        raise ValueError("Checklist must contain each season from 2003 through 2025")
    return by_season


def build(args: argparse.Namespace) -> None:
    invalidate_readiness(args.output_root / "manifest.json")
    invalidate_readiness(args.report_root / "manifest.json")
    checklist = load_checklist(args.checklist)
    final_rows = read_csv(args.final_observations)
    fcs_rows = [row.copy() for row in final_rows if row["subdivision"] == "fcs"]
    fcs_seasons = {
        int(row["season"])
        for row in fcs_rows
        if row["system_code"] == "CMP" and row["is_composite"] == "True"
    }

    output_rows: list[dict[str, str]] = []
    inventory: list[dict[str, object]] = []
    input_hashes: list[dict[str, object]] = []
    included_fbs: set[int] = set()

    for season in SEASONS:
        item = checklist[season]
        source_name = item["snapshot_source"]
        target_date = (
            date.fromisoformat(item["snapshot_date"])
            if item["snapshot_date"].strip()
            else None
        )
        rows: list[dict[str, str]] = []
        status = item["collection_status"]
        input_path: Path | None = None
        mapping_issue: dict[str, object] | None = None
        provenance: dict[str, object] | None = None

        if source_name == "massey_kaggle":
            relative_source = Path(item["snapshot_file"]).relative_to("data/raw")
            input_path = args.raw_root / relative_source
            if input_path.exists():
                provenance = validate_raw_provenance(
                    input_path,
                    season=season,
                    snapshot_date=target_date,
                    source_url=item["source_url"],
                )
                names = kaggle_names(input_path)
                matches, ambiguous, unmapped = load_team_matches(
                    args.raw_root, season, names
                )
                if ambiguous or unmapped:
                    mapping_issue = {
                        "ambiguous": {
                            name: [team.school for team in teams]
                            for name, teams in ambiguous.items()
                        },
                        "unmapped": unmapped,
                    }
                    status = "mapping_review_required"
                else:
                    window = PreBowlWindow(
                        season=season,
                        last_fbs_conference_championship=date.fromisoformat(
                            item["last_fbs_conference_championship_date"]
                        ),
                        first_fbs_postseason_game=date.fromisoformat(
                            item["first_fbs_postseason_game_date"]
                        ),
                    )
                    observed_date = latest_cmp_date_in_window(
                        input_path, window, matches=matches
                    )
                    if observed_date is None:
                        status = "no_qualifying_snapshot_in_existing_source"
                    elif target_date is None or observed_date != target_date:
                        status = "snapshot_date_mismatch_review_required"
                        mapping_issue = {
                            "expected_snapshot_date": (
                                target_date.isoformat() if target_date else None
                            ),
                            "observed_latest_cmp_date": observed_date.isoformat(),
                        }
                    else:
                        rows = read_kaggle_snapshot(
                            input_path,
                            matches,
                            season=season,
                            snapshot_date=observed_date,
                            source_file=item["snapshot_file"],
                        )
                        status = "available"
                        included_fbs.add(season)
            else:
                status = "missing_raw_kaggle_source"

        elif source_name == "massey_prebowl_export":
            if target_date is None:
                status = "snapshot_date_missing"
            else:
                export_name = snapshot_filename(season, target_date)
                input_path = args.raw_root / "massey_prebowl" / export_name
                if input_path.exists():
                    provenance = validate_raw_provenance(
                        input_path,
                        season=season,
                        snapshot_date=target_date,
                        source_url=item["source_url"],
                    )
                    names = export_names(input_path)
                    matches, ambiguous, unmapped = load_team_matches(
                        args.raw_root, season, names
                    )
                    if ambiguous or unmapped:
                        mapping_issue = {
                            "ambiguous": {
                                name: [team.school for team in teams]
                                for name, teams in ambiguous.items()
                            },
                            "unmapped": unmapped,
                        }
                        status = "mapping_review_required"
                    else:
                        lower = date.fromisoformat(
                            item["last_fbs_conference_championship_date"]
                        )
                        upper = date.fromisoformat(
                            item["first_fbs_postseason_game_date"]
                        )
                        accepted_date = snapshot_date_is_accepted(
                            season,
                            target_date,
                            PreBowlWindow(
                                season,
                                lower,
                                upper,
                            ),
                            item["phase_exception_decision"],
                        )
                        approved_exception = not lower < target_date < upper and accepted_date
                        if not accepted_date:
                            status = "snapshot_outside_accepted_window"
                        else:
                            observations, unmapped = read_dated_fbs_export(
                                input_path,
                                matches,
                                expected_season=season,
                                expected_date=target_date,
                                source_file=f"data/raw/massey_prebowl/{export_name}",
                            )
                            if unmapped:
                                mapping_issue = {"unmapped": unmapped}
                                status = "mapping_review_required"
                            else:
                                rows = normalized_rows(observations)
                                validate_snapshot(
                                    rows, season=season, snapshot_date=target_date
                                )
                                status = (
                                    "available_with_approved_phase_exception"
                                    if approved_exception
                                    else "available"
                                )
                                included_fbs.add(season)
                else:
                    status = "missing_dated_massey_export"

        elif source_name == "exception_required":
            status = "explicit_phase_exception_decision_required"
        else:
            status = "unsupported_snapshot_source"

        output_rows.extend(rows)
        inventory_item: dict[str, object] = {
            "season": season,
            "snapshot_date": target_date.isoformat() if target_date else None,
            "snapshot_source": source_name,
            "expected_file": item["snapshot_file"],
            "source_url": item["source_url"],
            "phase_exception_decision": item["phase_exception_decision"],
            "phase_exception_rationale": item["phase_exception_rationale"],
            "status": status,
            "observations": len(rows),
            "distinct_teams": len({row["team_id"] for row in rows}),
            "constituent_systems": len(
                {row["system_code"] for row in rows if row["system_code"] != "CMP"}
            ),
            "mapping_issue": mapping_issue,
            "raw_provenance": provenance,
        }
        if input_path is not None and input_path.exists():
            inventory_item["input_sha256"] = sha256(input_path)
            inventory_item["input_bytes"] = input_path.stat().st_size
            input_hashes.append(
                {
                    "season": season,
                    "path": item["snapshot_file"],
                    "sha256": inventory_item["input_sha256"],
                    "bytes": inventory_item["input_bytes"],
                    "snapshot_date": target_date.isoformat() if target_date else None,
                    "source_url": item["source_url"],
                    "imported_at": provenance.get("imported_at")
                    if provenance
                    else None,
                }
            )
        inventory.append(inventory_item)

    combined = output_rows + fcs_rows
    if [row for row in combined if row["subdivision"] == "fcs"] != fcs_rows:
        raise AssertionError("FCS observations changed during pre-bowl assembly")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_root / "pre_bowl_observations.csv",
        list(NORMALIZED_FIELDS),
        combined,
    )

    target_depth = coverage_table(combined)
    final_depth = coverage_table(final_rows)
    depth_fields = list(target_depth[0]) if target_depth else list(final_depth[0])
    write_csv(
        args.output_root / "pre_bowl_system_depth.csv", depth_fields, target_depth
    )
    write_csv(
        args.output_root / "final_system_depth.csv", depth_fields, final_depth
    )
    target_distributions, target_by_team = build_distributions(
        combined, target_depth, args.coverage_threshold
    )
    target_support_anomalies = audit_target_rank_support(target_distributions)
    final_distributions, final_by_team = build_distributions(
        final_rows, final_depth, args.coverage_threshold
    )
    final_support_anomalies = audit_target_rank_support(final_distributions)
    target_population_coverage = audit_population_coverage(
        combined,
        target_depth,
        target_distributions,
        args.coverage_threshold,
    )
    final_population_coverage = audit_population_coverage(
        final_rows,
        final_depth,
        final_distributions,
        args.coverage_threshold,
    )
    final_support_by_key = {
        (row["season"], row["subdivision"], row["team_id"]): row
        for row in audit_target_rank_support(final_distributions)
    }
    for anomaly in target_support_anomalies:
        baseline = final_support_by_key.get(
            (anomaly["season"], anomaly["subdivision"], anomaly["team_id"])
        )
        anomaly["present_in_final_corpus"] = baseline is not None
        anomaly["final_out_of_support_ranks"] = (
            baseline["out_of_support_ranks"] if baseline else "[]"
        )
        anomaly["final_out_of_support_probability_mass"] = (
            baseline["out_of_support_probability_mass"] if baseline else 0.0
        )
    write_csv(
        args.output_root / "target_support_anomalies.csv",
        list(TARGET_SUPPORT_FIELDS),
        target_support_anomalies,
    )
    write_csv(
        args.output_root / "final_target_support_anomalies.csv",
        list(TARGET_SUPPORT_FIELDS),
        final_support_anomalies,
    )
    coverage_fields = list(target_population_coverage[0]) if target_population_coverage else [
        "season",
        "subdivision",
        "team_population",
        "teams_with_selected_constituents",
        "teams_missing_selected_constituents",
        "coverage_ratio",
        "selected_system_count",
        "missing_team_ids",
    ]
    write_csv(
        args.output_root / "target_population_coverage.csv",
        coverage_fields,
        target_population_coverage,
    )
    write_csv(
        args.output_root / "final_population_coverage.csv",
        coverage_fields,
        final_population_coverage,
    )
    write_csv(
        args.output_root / "pre_bowl_target_distributions.csv",
        list(target_distributions[0]) if target_distributions else [],
        target_distributions,
    )
    write_csv(
        args.output_root / "final_target_distributions.csv",
        list(final_distributions[0]) if final_distributions else [],
        final_distributions,
    )
    pair_artifacts: dict[str, object] = {"status": "not_requested"}
    if args.games is not None:
        games, game_sources = read_cfbd_game_rows(args.games)
        target_pairs = build_constituent_rank_pairs(
            games, target_by_team, target_depth
        )
        final_pairs = build_constituent_rank_pairs(games, final_by_team, final_depth)
        pair_fields = list(CONSTITUENT_RANK_PAIR_FIELDS)
        write_csv(
            args.output_root / "pre_bowl_constituent_rank_pairs.csv",
            pair_fields,
            target_pairs,
        )
        write_csv(
            args.output_root / "final_constituent_rank_pairs.csv",
            pair_fields,
            final_pairs,
        )
        pair_artifacts = {
            "status": "built",
            "games_path": (
                "data/raw/cfbd/games"
                if args.games.is_dir()
                else args.games.as_posix()
            ),
            "game_sources": [
                {
                    "path": (
                        path.relative_to(args.games).as_posix()
                        if args.games.is_dir()
                        else path.as_posix()
                    ),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in game_sources
            ],
            "game_count": len(games),
            "pre_bowl_pair_rows": len(target_pairs),
            "final_pair_rows": len(final_pairs),
            "pre_bowl_file": "pre_bowl_constituent_rank_pairs.csv",
            "final_file": "final_constituent_rank_pairs.csv",
        }

    final_lookup = {
        (row["season"], row["subdivision"], row["system_code"]): row
        for row in final_depth
    }
    target_lookup = {
        (row["season"], row["subdivision"], row["system_code"]): row
        for row in target_depth
    }
    comparison_rows = []
    for key in sorted(set(final_lookup) | set(target_lookup)):
        final = final_lookup.get(key, {})
        prebowl = target_lookup.get(key, {})
        comparison_rows.append(
            {
                "season": key[0],
                "subdivision": key[1],
                "system_code": key[2],
                "final_distinct_teams_ranked": final.get("distinct_teams_ranked", 0),
                "pre_bowl_distinct_teams_ranked": prebowl.get(
                    "distinct_teams_ranked", 0
                ),
                "final_team_population": final.get("team_population", 0),
                "pre_bowl_team_population": prebowl.get("team_population", 0),
                "final_coverage_ratio": final.get("coverage_ratio", 0),
                "pre_bowl_coverage_ratio": prebowl.get("coverage_ratio", 0),
            }
        )
    coverage_fields = [
        "season",
        "subdivision",
        "system_code",
        "final_distinct_teams_ranked",
        "pre_bowl_distinct_teams_ranked",
        "final_team_population",
        "pre_bowl_team_population",
        "final_coverage_ratio",
        "pre_bowl_coverage_ratio",
    ]
    write_csv(
        args.output_root / "coverage_comparison.csv", coverage_fields, comparison_rows
    )

    missing_fbs = sorted(set(SEASONS) - included_fbs)
    missing_fcs = sorted(set(SEASONS) - fcs_seasons)
    unresolved_exceptions = [
        row["season"]
        for row in inventory
        if row["status"] == "explicit_phase_exception_decision_required"
    ]
    approved_exceptions = sorted(
        {
            row["season"]
            for row in inventory
            if row["snapshot_date"]
            and APPROVED_PHASE_EXCEPTIONS.get(
                (row["season"], date.fromisoformat(row["snapshot_date"]))
            )
            == row["phase_exception_decision"]
        }
    )
    collection_complete = not missing_fbs and not missing_fcs and not unresolved_exceptions
    expected_cohorts = {
        (season, subdivision)
        for season in SEASONS
        for subdivision in ("fbs", "fcs")
    }
    target_coverage_by_key = {
        (int(row["season"]), str(row["subdivision"])): row
        for row in target_population_coverage
    }
    final_coverage_by_key = {
        (int(row["season"]), str(row["subdivision"])): row
        for row in final_population_coverage
    }
    missing_target_coverage = sorted(
        key
        for key in expected_cohorts
        if key not in target_coverage_by_key
        or int(target_coverage_by_key[key]["teams_missing_selected_constituents"])
    )
    missing_final_coverage = sorted(
        key
        for key in expected_cohorts
        if key not in final_coverage_by_key
        or int(final_coverage_by_key[key]["teams_missing_selected_constituents"])
    )
    coverage_passed = not missing_target_coverage and not missing_final_coverage
    support_passed = not target_support_anomalies and not final_support_anomalies
    downstream_ready = collection_complete and support_passed and coverage_passed
    manifest = {
        "schema_version": 2,
        "corpus_version": "massey-rank-population-v1",
        "status": (
            "incomplete"
            if not collection_complete
            else "complete_with_validation_warnings"
            if not support_passed or not coverage_passed
            else "complete"
        ),
        "collection_status": "complete" if collection_complete else "incomplete",
        "downstream_ready": downstream_ready,
        "target_support_validation": {
            "status": "passed" if support_passed else "anomalies_found",
            "anomaly_count": len(target_support_anomalies),
            "final_anomaly_count": len(final_support_anomalies),
            "present_in_final_corpus_count": sum(
                bool(row["present_in_final_corpus"])
                for row in target_support_anomalies
            ),
            "new_anomaly_count": sum(
                not bool(row["present_in_final_corpus"])
                for row in target_support_anomalies
            ),
            "report": "target_support_anomalies.csv",
        },
        "population_coverage_validation": {
            "status": "passed" if coverage_passed else "teams_without_selected_constituents",
            "coverage_threshold": args.coverage_threshold,
            "missing_target_cohorts": [list(key) for key in missing_target_coverage],
            "missing_final_cohorts": [list(key) for key in missing_final_coverage],
            "target_report": "target_population_coverage.csv",
            "final_report": "final_population_coverage.csv",
        },
        "required_fbs_seasons": list(SEASONS),
        "included_fbs_seasons": sorted(included_fbs),
        "missing_fbs_seasons": missing_fbs,
        "missing_fcs_seasons": missing_fcs,
        "unresolved_phase_exception_seasons": unresolved_exceptions,
        "approved_phase_exception_seasons": approved_exceptions,
        "fcs_source": {
            "path": "data/processed/massey_issue_204_v1/final_observations.csv",
            "sha256": sha256(args.final_observations),
            "source_rows": len(final_rows),
            "carried_rows": len(fcs_rows),
            "ordered_fcs_rows_unchanged": True,
        },
        "inputs": input_hashes,
        "collection": inventory,
        "coverage_threshold": args.coverage_threshold,
        "constituent_rank_pairs": pair_artifacts,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output_root.glob("*.csv"))
        },
        "notes": [
            "No final-season FBS observation is substituted for a missing pre-bowl snapshot.",
            "FCS rows are copied from the corrected final-observation corpus without field changes.",
            "Only same-date constituent system observations are retained for each snapshot.",
            "An incomplete collection is not ready for downstream fitting; approved phase exceptions do not change target-support audit results.",
            "The 2020-12-21 snapshot is included under the user's explicit approval despite overlap with the first bowl date.",
            "Downstream readiness requires complete collection, zero unsupported final and pre-bowl target mass, and at least one selected constituent rank for every CMP population team in both corpora.",
            "The three historical cohort corrections are identity-and-season overrides in normalized source data; source ranks are retained unchanged.",
        ],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_source_manifest = {
        "schema_version": 1,
        "team_mapping_sources": [
            {
                "season": season,
                "path": f"data/raw/cfbd/teams/{season}.json",
                "source_url": f"https://api.collegefootballdata.com/teams?year={season}",
                "sha256": (
                    sha256(args.raw_root / "cfbd" / "teams" / f"{season}.json")
                    if (args.raw_root / "cfbd" / "teams" / f"{season}.json").exists()
                    else None
                ),
                "bytes": (
                    (args.raw_root / "cfbd" / "teams" / f"{season}.json").stat().st_size
                    if (args.raw_root / "cfbd" / "teams" / f"{season}.json").exists()
                    else None
                ),
                "provenance": (
                    json.loads(
                        (args.raw_root / "cfbd" / "teams" / f"{season}.json")
                        .with_name(f"{season}.json.provenance.json")
                        .read_text(encoding="utf-8")
                    )
                    if (args.raw_root / "cfbd" / "teams" / f"{season}.json")
                    .with_name(f"{season}.json.provenance.json")
                    .exists()
                    else None
                ),
            }
            for season in SEASONS
        ],
        "schedule_sources": [
            {
                "season": season,
                "source_url": checklist[season]["schedule_source_url"],
                "raw_path": checklist[season]["schedule_raw_path"],
                "sha256": checklist[season]["schedule_source_sha256"],
                "retrieved_at": checklist[season]["schedule_retrieved_at"],
            }
            for season in SEASONS
        ],
        "ranking_sources": [
            {
                "season": item["season"],
                "snapshot_date": item["snapshot_date"],
                "source": item["snapshot_source"],
                "path": item["expected_file"],
                "source_url": item["source_url"],
                "status": item["status"],
                "sha256": item.get("input_sha256"),
                "bytes": item.get("input_bytes"),
                "provenance": item["raw_provenance"],
            }
            for item in inventory
        ],
        "existing_source_inventory": [
            {
                "season": season,
                "path": item["existing_source_file"],
                "coverage": item["existing_source_coverage"],
                "sha256": item["existing_source_sha256"] or None,
                "required_snapshot_path": item["snapshot_file"],
                "required_snapshot_url": item["source_url"],
            }
            for season, item in checklist.items()
        ],
        "unchanged_fcs_reference": manifest["fcs_source"],
        "target_support_validation": manifest["target_support_validation"],
        "missing_or_unresolved_seasons": [
            {
                "season": item["season"],
                "status": item["status"],
                "expected_file": item["expected_file"],
                "source_url": item["source_url"],
            }
            for item in inventory
            if item["status"]
            not in {"available", "available_with_approved_phase_exception"}
        ],
        "notes": [
            "Existing historical ranking exports are referenced by their canonical repository paths and whole-file SHA-256 hashes.",
            "New dated exports are copied byte-for-byte into data/raw/massey_prebowl/ and verified against adjacent provenance sidecars.",
            "A missing or unresolved source remains explicit; no post-bowl or final ranking is substituted.",
        ],
    }
    (args.output_root / "raw_source_manifest.json").write_text(
        json.dumps(raw_source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.report_root / "coverage_comparison.csv", coverage_fields, comparison_rows
    )
    write_csv(
        args.report_root / "target_support_anomalies.csv",
        list(TARGET_SUPPORT_FIELDS),
        target_support_anomalies,
    )
    (args.report_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.report_root / "raw_source_manifest.json").write_text(
        json.dumps(raw_source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"pre-bowl target status: {manifest['status']}")
    print(f"included FBS seasons: {len(included_fbs)}/{len(SEASONS)}")
    print(f"missing FBS seasons: {missing_fbs}")
    print(f"target support anomalies: {len(target_support_anomalies)}")
    print(f"downstream ready: {downstream_ready}")
    print(f"output: {args.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw",
        help="Directory containing massey_rankings/, massey_prebowl/, and cfbd/.",
    )
    parser.add_argument(
        "--final-observations",
        type=Path,
        default=ROOT / "data/processed/massey/final_observations.csv",
        help="Existing normalized final corpus; its FCS rows are copied unchanged.",
    )
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/massey_prebowl",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    parser.add_argument(
        "--games",
        type=Path,
        help="Optional processed games CSV or cached raw game directory for no-fit pairs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
