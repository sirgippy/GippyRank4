"""Build a durable predictive ranking snapshot; publishing is intentionally separate."""

from __future__ import annotations

import argparse
from datetime import datetime

from gippyrank.posterior.snapshots import build_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--prior-family", choices=("context", "history"), required=True)
    parser.add_argument(
        "--snapshot-type", choices=("preseason", "weekly", "live"), required=True
    )
    parser.add_argument(
        "--cutoff", help="ISO-8601 UTC timestamp; required except preseason"
    )
    args = parser.parse_args()
    if args.snapshot_type != "preseason" and not args.cutoff:
        parser.error("--cutoff is required for weekly and live snapshots")
    cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else None
    snapshot = build_snapshot(
        season=args.season,
        cutoff=cutoff,
        prior_family=args.prior_family,
        snapshot_type=args.snapshot_type,
    )
    print(snapshot.directory)


if __name__ == "__main__":
    main()
