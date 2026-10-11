"""Import a dated FBS Massey export without changing its raw bytes."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from gippyrank.data.massey_prebowl import (
    MASSEY_EXPORT_URL,
    import_raw_export,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data/raw/massey_prebowl"


def import_export(
    source: Path,
    *,
    season: int,
    snapshot_date: date,
    output_root: Path,
    source_url: str,
) -> Path:
    destination = import_raw_export(
        source,
        season=season,
        snapshot_date=snapshot_date,
        output_root=output_root,
        source_url=source_url,
    )
    print(f"imported {destination}")
    print(f"provenance {destination}.provenance.json")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Downloaded Massey CSV")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--snapshot-date",
        type=date.fromisoformat,
        required=True,
        help="Accepted ranking date in YYYY-MM-DD form",
    )
    parser.add_argument(
        "--source-url",
        help="Source URL; defaults to the official dated Massey page",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    official_url = MASSEY_EXPORT_URL.format(
        yyyymmdd=arguments.snapshot_date.strftime("%Y%m%d")
    )
    import_export(
        arguments.source,
        season=arguments.season,
        snapshot_date=arguments.snapshot_date,
        output_root=arguments.output_root,
        source_url=arguments.source_url or official_url,
    )
