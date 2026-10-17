"""Run the frozen three-arm issue 213 Context likelihood experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gippyrank.research.issue_213_context_likelihood_forecast import (
    run_issue_213_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT)
    parser.add_argument("--game-root", type=Path, default=ROOT)
    parser.add_argument(
        "--issue195-root",
        type=Path,
        default=ROOT / "data/research/issue_195_likelihood_diagnostic",
    )
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
        "--protocol",
        type=Path,
        default=ROOT / "docs/issue_213_context_dependent_likelihood_forecast.md",
    )
    parser.add_argument(
        "--issue201-protocol",
        type=Path,
        default=ROOT / "docs/issue_201_prebowl_forecast_comparison.md",
    )
    parser.add_argument("--target-snapshot-manifest", type=Path)
    parser.add_argument("--schedule-windows", type=Path)
    parser.add_argument("--pre-bowl-targets", type=Path)
    parser.add_argument("--issue204-manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/research/issue_213_context_likelihood_forecast",
    )
    return parser.parse_args()


def main() -> None:
    result = run_issue_213_replay(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
