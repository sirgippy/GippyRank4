"""Build a future History 1.1 annual prior and its source attestation together.

Run with ``uv run python scripts/build_history_annual_v1_1.py YEAR`` after the
canonical historical rank and target team-feature artifacts are available.
The retained 2026 annual files are deliberately outside this build path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gippyrank.history_annual_v1_1 import build_canonical_history_annual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_season", type=int)
    args = parser.parse_args()
    annual = (
        Path(__file__).resolve().parents[1]
        / "data/processed/preseason/history/annual"
        / str(args.target_season)
    )
    source = build_canonical_history_annual(args.target_season, annual)
    print(source.source_identity_sha256)


if __name__ == "__main__":
    main()
