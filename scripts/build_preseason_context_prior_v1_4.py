"""Build the corrected production Context 1.4 prior from repaired transfers."""

from __future__ import annotations

import json
from pathlib import Path

from rebuild_context_v1_4_db_repair import run

ROOT = Path(__file__).resolve().parents[1]


def build(root: Path = ROOT) -> dict[str, object]:
    return run(root)


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "2026_prior_pmfs_sha256": result["2026_prior_pmfs_sha256"],
                "2026_predictions_sha256": result["2026_predictions_sha256"],
            },
            indent=2,
        )
    )
