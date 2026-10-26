"""Export explicitly selected posterior snapshots as static website data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gippyrank.site_data import build_site_data, plan_publication_retention

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show publications and generated artifacts that retention would remove",
    )
    args = parser.parse_args()
    arguments = {
        "root": ROOT,
        "config_path": ROOT / "site/publish_config.json",
        "output_directory": ROOT / "site/data",
    }
    if args.dry_run:
        print(
            json.dumps(
                plan_publication_retention(
                    **arguments,
                    static_api_directory=ROOT / "site/api/v1",
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        build_site_data(
            **arguments,
            static_api_directory=ROOT / "site/api/v1",
        )
