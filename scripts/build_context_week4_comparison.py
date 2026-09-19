"""Build the canonical matched 2026 official Week 4 Context pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from gippyrank.context_comparison import build_official_week4_pair

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="official Week 4 UTC cutoff")
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root"
    )
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    result = build_official_week4_pair(root=args.root, cutoff=cutoff)
    manifest = (
        args.root
        / "data/processed/preseason/context_v1_3_2026_reconstruction/comparison_quartet.json"
    )
    value = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    value.update(
        {
            "status": "frozen",
            "official_week_4": result,
            "artifacts": {
                **value.get("artifacts", {}),
                "context_1_2_official_week_4": result["context_1_2_path"],
                "context_1_3_official_week_4": result["context_1_3_path"],
            },
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
