"""Build a standalone Quality of Record snapshot from a frozen Context snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.quality_of_record_snapshot import build_quality_of_record_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    source = args.context if args.context.is_absolute() else ROOT / args.context
    snapshot = build_quality_of_record_snapshot(
        source, root=ROOT, output_root=args.output_root
    )
    print(snapshot.directory)


if __name__ == "__main__":
    main()
