"""Build the offline CFBD primitive box-score data audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.research.box_score_audit import build_audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/box_score_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    summary = build_audit(root, output)
    print(
        "wrote box-score audit: "
        f"{summary['corpus']['stat_unique_team_games']:,} unique team-game rows, "
        f"{len(summary['category_inventory'])} raw categories"
    )


if __name__ == "__main__":
    main()
