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
) -> list[dict[str, object]]:
    """Write the frozen retrospective research representation."""
    source_root = source_root.resolve()
    transfer_root = transfer_root.resolve()
    output = output.resolve()
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
    transfer_features = transfer_research.aggregate_team_features(
        records,
        usage,
        transfer_research.fbs_feature_rows(contextual),
        covered_seasons=portal_seasons,
        cutoff=date(2025, 8, 15),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(
        source_root=args.source_root,
        transfer_root=args.transfer_root,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
