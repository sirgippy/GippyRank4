"""Build a durable production Performance snapshot from a Context snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.performance_snapshot import build_performance_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    source = args.context if args.context.is_absolute() else ROOT / args.context
    snapshot = build_performance_snapshot(
        source, root=ROOT, output_root=args.output_root
    )
    print(snapshot.directory.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
