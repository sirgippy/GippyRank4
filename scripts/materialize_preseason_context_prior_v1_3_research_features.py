"""Materialize the retrospective transfer feature panel for Context 1.3.

This is a research-only adapter.  It is the one place in the 1.3 workflow
that may read the retrospective portal/usage oracle and the checked-in
defensive audit.  The Context 1.3 fit and annual inference paths consume only
the CSV written here (or the production CSV written by
``build_preseason_transfer_features.py``); they never parse those sources.

The output is intentionally labelled retrospective.  It is not an archived
preseason snapshot and must not be used to reinterpret the published 2026
Context 1.2 artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_preseason_context_prior_v1_2 as c12
import build_preseason_prior as v1
import investigate_defensive_transfer_position_groups as position_groups
import investigate_transfer_roster_continuity as transfer_research

from gippyrank.context_prior_v1_3 import CONTEXT_1_3_FEATURES
from gippyrank.transfer_oracle import (
    TransferIdentityKey,
    TransferRecord,
    normalize_player_name,
    normalize_team_name,
    transfer_identity_key,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET_SEASONS = (2022, 2023, 2024, 2025)
RESEARCH_TRANSFER_SEASONS = (2021, *TARGET_SEASONS)
TRANSFER_FEATURES = (
    "transfer_in_prior_usage_sum",
    "transfer_in_prior_defensive_impact_db_sum",
    "transfer_in_prior_defensive_impact_db_available",
)
OUTPUT_FIELDS = (
    "season",
    "subdivision",
    "team_id",
    "team_name",
    *TRANSFER_FEATURES,
    "provenance_class",
)
ZERO_EVIDENCE_RELATIVE_PATH = Path(
    "data/processed/transfer_data_repair/zero_contributors.csv"
)
ZERO_EVIDENCE_CATEGORY = "legitimate_zero_or_non_applicable_prior_offensive_usage"


def _verified_zero_usage_keys(
    records: list[TransferRecord],
    evidence_path: Path,
    team_rows: list[dict[str, object]],
) -> set[TransferIdentityKey]:
    """Resolve retained player-audit evidence to this exact portal input."""
    canonical_teams = {
        (int(row["season"]), normalize_team_name(str(row["team_name"]))): str(
            row["team_id"]
        )
        for row in team_rows
    }
    evidence_keys: set[TransferIdentityKey] = set()
    seen_indexes: set[int] = set()
    with evidence_path.open(newline="", encoding="utf-8") as handle:
        for evidence in csv.DictReader(handle):
            if (
                evidence.get("d5_resolution_category") != ZERO_EVIDENCE_CATEGORY
                or float(evidence.get("d5_feature_value", "nan")) != 0.0
            ):
                raise ValueError(
                    f"invalid explicit-zero evidence row in {evidence_path}"
                )
            portal_index = int(evidence["portal_index"])
            if portal_index in seen_indexes or portal_index >= len(records):
                raise ValueError(
                    f"duplicate or out-of-range portal_index {portal_index} in "
                    f"{evidence_path}"
                )
            seen_indexes.add(portal_index)
            record = records[portal_index]
            expected = (
                int(evidence["season"]),
                normalize_player_name(evidence["player"]),
                normalize_team_name(evidence["source_team"]),
                normalize_team_name(evidence["destination"]),
                evidence["transfer_date"],
            )
            actual = (
                record.season,
                normalize_player_name(record.player_name),
                normalize_team_name(record.origin),
                normalize_team_name(record.destination),
                record.transfer_date.isoformat() if record.transfer_date else "",
            )
            if expected != actual:
                raise ValueError(
                    "explicit-zero evidence does not match portal input at "
                    f"index {portal_index}: expected {expected!r}, found {actual!r}"
                )
            expected_player_id = evidence.get("portal_player_id", "")
            if expected_player_id and record.player_id != expected_player_id:
                raise ValueError(
                    f"portal player ID disagrees with zero evidence at index "
                    f"{portal_index}"
                )
            destination_id = canonical_teams.get(
                (record.season, normalize_team_name(record.destination))
            )
            if destination_id != evidence["destination_team_id"]:
                raise ValueError(
                    f"destination team disagrees with zero evidence at index "
                    f"{portal_index}: {destination_id!r} != "
                    f"{evidence['destination_team_id']!r}"
                )
            evidence_keys.add(transfer_identity_key(record))
    return evidence_keys


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Context 1.3 research panel")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    source_root: Path = ROOT,
    transfer_root: Path,
    output: Path,
    zero_evidence: Path | None = None,
) -> list[dict[str, object]]:
    """Write the frozen retrospective research representation."""
    source_root = source_root.resolve()
    transfer_root = transfer_root.resolve()
    output = output.resolve()
    evidence_path = (
        zero_evidence.resolve()
        if zero_evidence is not None
        else source_root / ZERO_EVIDENCE_RELATIVE_PATH
    )
    if not evidence_path.is_file():
        raise FileNotFoundError(
            f"required retained player-audit evidence not found: {evidence_path}"
        )
    transfer_research.configure_source_root(source_root)
    rows, _, _ = v1.load_rows(max_season=max(TARGET_SEASONS))
    fbs = [row for row in rows if row.subdivision == "fbs"]
    contextual, _ = c12.attach_context(
        fbs, c12.feature_index(), c12.cached_tenures()
    )
    records, usage, portal_seasons, usage_seasons = (
        transfer_research.load_raw_transfer_data(transfer_root)
    )
    required_portal = set(RESEARCH_TRANSFER_SEASONS)
    required_usage = set(range(2020, 2025))
    if not required_portal <= portal_seasons:
        raise FileNotFoundError(
            f"research panel requires portal seasons {sorted(required_portal)}; "
            f"found {sorted(portal_seasons)}"
        )
    if not required_usage <= usage_seasons:
        raise FileNotFoundError(
            f"research panel requires usage seasons {sorted(required_usage)}; "
            f"found {sorted(usage_seasons)}"
        )
    feature_rows = transfer_research.fbs_feature_rows(contextual)
    verified_zero_keys = _verified_zero_usage_keys(
        records, evidence_path, feature_rows
    )
    transfer_features = transfer_research.aggregate_team_features(
        records,
        usage,
        feature_rows,
        covered_seasons=portal_seasons,
        cutoff=date(2025, 8, 15),
        verified_zero_usage_keys=verified_zero_keys,
    )
    defensive = position_groups.load_position_features(
        source_root / "data/processed/defensive_transfer_audit/transfer_player_audit.csv",
        source_root / "data/processed/defensive_transfer_audit/coverage_by_position.csv",
    )
    rows_out: list[dict[str, object]] = []
    for row in contextual:
        key = (row.season, row.subdivision, row.team_id)
        offensive = transfer_features.get(key, {})
        defensive_values = defensive.values.get(
            key,
            {
                "transfer_in_prior_defensive_impact_db_sum": 0.0,
                "transfer_in_prior_defensive_impact_db_available": 0.0,
            },
        )
        rows_out.append(
            {
                "season": row.season,
                "subdivision": row.subdivision,
                "team_id": row.team_id,
                "team_name": row.team_name,
                "transfer_in_prior_usage_sum": offensive.get(
                    "transfer_in_prior_usage_sum"
                ),
                "transfer_in_prior_defensive_impact_db_sum": defensive_values[
                    "transfer_in_prior_defensive_impact_db_sum"
                ],
                "transfer_in_prior_defensive_impact_db_available": defensive_values[
                    "transfer_in_prior_defensive_impact_db_available"
                ],
                "provenance_class": "retrospective_research_reconstruction",
            }
        )
    rows_out.sort(key=lambda row: (int(row["season"]), str(row["team_id"])))
    _write_csv(output, rows_out)
    metadata = {
        "candidate_spec_version": "1.3",
        "context_features": list(CONTEXT_1_3_FEATURES),
        "transfer_features": list(TRANSFER_FEATURES),
        "provenance_class": "retrospective_research_reconstruction",
        "historical_cutoff_caveat": (
            "Values for 2021-2025 reproduce the frozen retrospective research "
            "representation; they are not archived August 15 snapshots."
        ),
        "source_root": "repository working tree",
        "transfer_root": "external retrospective research oracle (not committed)",
        "zero_evidence_path": (
            evidence_path.relative_to(source_root).as_posix()
            if evidence_path.is_relative_to(source_root)
            else evidence_path.name
        ),
        "zero_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "verified_zero_evidence_player_count": len(verified_zero_keys),
        "row_count": len(rows_out),
    }
    output.with_name(f"{output.stem}.provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transfer-root",
        type=Path,
        required=True,
        help="Retrospective portal/usage oracle directory; research-only input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zero-evidence",
        type=Path,
        help=(
            "retained player-level D5 legitimate-zero evidence CSV; defaults to "
            "data/processed/transfer_data_repair/zero_contributors.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(
        source_root=args.source_root,
        transfer_root=args.transfer_root,
        output=args.output,
        zero_evidence=args.zero_evidence,
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
