"""Reconstruct retained Quality of Results publications from frozen Context."""

import argparse
from pathlib import Path

from gippyrank.quality_of_results_backfill import backfill_quality_of_results
from gippyrank.site_data import build_site_data

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    report = backfill_quality_of_results(root=ROOT, allow_blocked=args.allow_blocked)
    build_site_data(
        root=ROOT,
        config_path=ROOT / "site/publish_config.json",
        output_directory=ROOT / "site/data",
    )
    print(
        f"Validated {report['publication_count']} publications; generated {report['run_generated_count']}, reused {report['run_reused_count']}."
    )
    if report["blockers"]:
        print("Blocked publications:\n" + "\n".join(report["blockers"]))
