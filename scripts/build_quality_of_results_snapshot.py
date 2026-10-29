"""Generate or independently validate a standalone Quality of Results snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.quality_of_results_snapshot import (
    build_quality_of_results_snapshot,
    validate_quality_of_results_against_context,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Frozen repository inputs"
    )
    parser.add_argument("--context", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output-root", type=Path)
    mode.add_argument(
        "--validate", type=Path, metavar="SNAPSHOT", help="Validate only; no inference"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / args.context
    if args.validate is not None:
        validate_quality_of_results_against_context(
            source, root / args.validate, root=root
        )
        print(f"Validated {root / args.validate}")
    else:
        snapshot = build_quality_of_results_snapshot(
            source,
            root=root,
            output_root=root / args.output_root
            if args.output_root is not None
            else None,
        )
        print(snapshot.directory)


if __name__ == "__main__":
    main()
