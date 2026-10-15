#!/usr/bin/env python3
"""Run issue 194 Part A against the frozen issue 200/201/204 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.research.issue_194_part_a import run_issue194_part_a

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=ROOT / "data/raw/cfbd/games")
    parser.add_argument(
        "--issue200-root",
        type=Path,
        default=ROOT / "data/research/matched_history_context_issue_200",
    )
    parser.add_argument(
        "--issue201-root",
        type=Path,
        default=ROOT / "data/research/issue_201_prebowl_forecast",
    )
    parser.add_argument(
        "--issue204-root",
        type=Path,
        default=ROOT / "data/processed/massey_prebowl_issue_204_v1",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=ROOT / "data/processed/context_incremental_information",
    )
    parser.add_argument(
        "--target-snapshot-manifest",
        type=Path,
        default=ROOT
        / "data/research/matched_history_context_issue_200/target_snapshot_manifest.csv",
    )
    parser.add_argument(
        "--schedule-windows",
        type=Path,
        default=ROOT / "data/research/massey_prebowl_issue_198/schedule_windows.csv",
    )
    parser.add_argument(
        "--issue201-protocol",
        type=Path,
        default=ROOT / "docs/issue_201_prebowl_forecast_comparison.md",
    )
    parser.add_argument(
        "--part-a-protocol",
        type=Path,
        default=ROOT / "docs/issue_194_part_a_protocol.md",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/research/issue_194_part_a",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_issue194_part_a(
        game_root=args.game_root,
        issue200_root=args.issue200_root,
        issue201_root=args.issue201_root,
        issue204_root=args.issue204_root,
        frozen_root=args.frozen_root,
        target_snapshot_manifest=args.target_snapshot_manifest,
        schedule_windows=args.schedule_windows,
        issue201_protocol=args.issue201_protocol,
        part_a_protocol=args.part_a_protocol,
        output_root=args.output_root,
    )
    print(
        "Issue 194 Part A complete: "
        f"{manifest['crossed_message_design']['checkpoint_count']} checkpoints, "
        f"{manifest['crossed_message_design']['team_score_row_count']} rank rows, "
        f"historical co-movement {manifest['preseason_pair_design']['historical_co_movement_status']}"
    )
    print(f"Results: {args.output_root}")
    print(f"Manifest SHA-256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
