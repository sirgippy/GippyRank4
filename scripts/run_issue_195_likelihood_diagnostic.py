"""Run issue 195's ex-post rank-only likelihood diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gippyrank.research.issue_195_likelihood_diagnostic import parse_args, run


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "historical_scoring_performed": result.get(
                    "historical_scoring_performed", result["status"] == "complete"
                ),
                "output_root": str(args.output_root),
                "missing_required_inputs": result.get("missing_required_inputs", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
