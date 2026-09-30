"""Build canonical preseason transfer features from an immutable manifest.

No endpoint is contacted by this command.  The manifest is validated before
any parser or feature code runs, including every raw-file hash and cutoff
flag.  Derived model fields and audit/provenance artifacts are written
separately so the latter cannot accidentally become Context features.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gippyrank.preseason_transfer import (
    CANONICAL_FEATURE_COLUMNS,
    DB_COVERAGE_COLUMNS,
    MANIFEST_VERSION,
    MODEL_FEATURE_COLUMNS,
    derive_preseason_transfer_features,
    read_player_aliases,
    read_team_aliases,
    source_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/cfbd/preseason/transfers/manifest.json"
DEFAULT_TEAM_FILE = ROOT / "data/processed/preseason/team_season_features.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/preseason/transfer_features"
DEFAULT_TEAM_ALIASES = ROOT / "data/reference/preseason_team_aliases.csv"
DEFAULT_PLAYER_ALIASES = ROOT / "data/reference/preseason_player_aliases.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    values = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _render_report(result: Mapping[str, Any]) -> str:
    research_only = bool(result.get("allow_late_snapshots"))
    heading = (
        "# Reacquired preseason transfer research features"
        if research_only
        else "# Production-safe preseason transfer features"
    )
    provenance_text = (
        "This beta research artifact was derived offline from current/reacquired provider responses. These payloads are not the lost original snapshots and do not establish what was available at the historical cutoff."
        if research_only
        else "This artifact was derived offline from the validated immutable snapshot manifest."
    )
    cutoff_text = (
        "The source responses were retrieved after the target-season August 15 cutoff. Retrieval timestamps and request hashes are retained for provenance, but these current values are research evidence only and do not prove earlier availability."
        if research_only
        else "Every canonical raw snapshot is required to be captured on or before the target-season August 15 cutoff. Hashes, request parameters, raw filenames, contributing player records, and unresolved reasons are retained in the machine-readable manifest and provenance artifacts."
    )
    lines = [
        heading,
        "",
        provenance_text,
        "",
        "## Model-facing fields",
        "",
        *[f"- `{field}`" for field in MODEL_FEATURE_COLUMNS],
        "",
        "The DB numeric value is neutral-imputed to zero when a known incoming DB transfer is unresolved; `transfer_in_prior_defensive_impact_db_available` remains zero in that case. A season with no incoming DB transfers is a natural zero with availability one.",
        "",
        "The diagnostic output also records `incoming_db_count`, `observed_db_impact_count`, `observed_db_impact_sum`, missing-count, coverage-fraction, status, source-season, and per-player provenance fields. These describe observed evidence only; Context 1.3 continues to consume only the three model-facing fields listed above.",
        "",
        "## Data-quality summary",
        "",
        "| Season | Incoming FBS | Applicable offense | Resolved offense | Unresolved offense | Undetermined | Incoming DB | Resolved DB | Unresolved DB | DB unavailable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["quality_report"]["seasons"]:
        lines.append(
            "| {season} | {incoming_fbs_transfers} | {applicable_offensive_transfers} | "
            "{resolved_applicable_transfers} | {unresolved_applicable} | "
            "{undetermined_applicability} | {incoming_db_transfers} | "
            "{resolved_db_transfers} | {unresolved_db_transfers} | "
            "{teams_with_unavailable_db_feature} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Cutoff and provenance",
            "",
            cutoff_text,
            "",
            "> Historical research results use retrospective oracle data and are not evidence that historical feature values were available as-of those preseason cutoffs.",
            "",
            (
                "> This reacquired evidence is not eligible for production use."
                if research_only
                else "> Production safety begins only for seasons captured by the immutable preseason snapshot process."
            ),
            "",
            "## Source inventory",
            "",
        ]
    )
    for row in source_inventory():
        lines.append(
            f"- `{row['source']}` (`{row['kind']}`): {row['required_fields']}. "
            f"{row['cutoff_rule']}."
        )
    lines.append("")
    return "\n".join(lines)


def run(
    *,
    manifest: Path = DEFAULT_MANIFEST,
    team_file: Path = DEFAULT_TEAM_FILE,
    team_aliases: Path | None = DEFAULT_TEAM_ALIASES,
    player_aliases: Path | None = DEFAULT_PLAYER_ALIASES,
    output: Path = DEFAULT_OUTPUT,
    seasons: Iterable[int] | None = None,
    allow_late_snapshots: bool = False,
) -> dict[str, Any]:
    team_rows = _read_csv(team_file)
    aliases = read_team_aliases(
        team_aliases if team_aliases and team_aliases.exists() else None
    )
    player_resolver = read_player_aliases(
        player_aliases if player_aliases and player_aliases.exists() else None
    )
    result = derive_preseason_transfer_features(
        manifest,
        team_rows,
        team_aliases=aliases,
        player_aliases=player_resolver,
        required_seasons=seasons,
        allow_late_snapshots=allow_late_snapshots,
    )
    result["allow_late_snapshots"] = allow_late_snapshots
    feature_fields = list(CANONICAL_FEATURE_COLUMNS) + [
        field
        for field in result["features"][0]
        if field.startswith("audit_") or field in DB_COVERAGE_COLUMNS
    ]
    _write_csv(
        output / "preseason_transfer_features.csv",
        result["features"],
        fieldnames=feature_fields,
    )
    _write_csv(output / "transfer_team_audit.csv", result["audit"])
    _write_csv(output / "transfer_player_audit.csv", result["player_audit"])
    _write_csv(
        output / "offensive_player_join_records.csv",
        result["offensive_player_audit"],
    )
    _write_csv(output / "identity_mapping.csv", result["identity_mapping"])
    _write_json(output / "data_quality_report.json", result["quality_report"])
    _write_json(output / "feature_provenance.json", result["provenance"])
    _write_json(
        output / "source_manifest.json",
        {
            "manifest": str(manifest),
            "evidence_class": (
                "reacquired_current_provider_response_beta_research"
                if allow_late_snapshots
                else "immutable_snapshot_manifest"
            ),
            "claims_original_snapshot_equivalence": False
            if allow_late_snapshots
            else None,
            "manifest_version": MANIFEST_VERSION,
            "snapshots": [
                snapshot.as_dict() for snapshot in result["manifest"].snapshots
            ],
        },
    )
    _write_json(
        output / "source_inventory.json",
        {
            "manifest": str(manifest),
            "sources": source_inventory(),
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        },
    )
    (output / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM_FILE)
    parser.add_argument("--team-aliases", type=Path, default=DEFAULT_TEAM_ALIASES)
    parser.add_argument("--player-aliases", type=Path, default=DEFAULT_PLAYER_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--season", dest="seasons", action="append", type=int)
    parser.add_argument(
        "--allow-late-snapshots",
        action="store_true",
        help="permit after-cutoff snapshots for explicitly labeled 2026 beta research only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        manifest=args.manifest,
        team_file=args.team_file,
        team_aliases=args.team_aliases,
        player_aliases=args.player_aliases,
        output=args.output,
        seasons=args.seasons,
        allow_late_snapshots=args.allow_late_snapshots,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seasons": result["quality_report"]["seasons"],
                "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
