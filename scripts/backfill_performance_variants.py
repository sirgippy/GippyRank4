"""Add both Performance variants to retained publications without acquiring data."""

import argparse
from pathlib import Path

from gippyrank.performance_backfill import backfill_performance_variants
from gippyrank.site_data import build_site_data

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Register valid publications and explicitly report blocked slots",
    )
    args = parser.parse_args()
    report = backfill_performance_variants(root=ROOT, allow_blocked=args.allow_blocked)
    build_site_data(
        root=ROOT,
        config_path=ROOT / "site/publish_config.json",
        output_directory=ROOT / "site/data",
    )
    print(
        f"Validated both Performance variants for {report['publication_count']} publications"
    )

    if report["blockers"]:
        print("Blocked publications:\n" + "\n".join(report["blockers"]))
