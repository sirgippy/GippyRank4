"""Export explicitly selected posterior snapshots as static website data."""

from __future__ import annotations

from pathlib import Path

from gippyrank.site_data import build_site_data

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    build_site_data(
        root=ROOT,
        config_path=ROOT / "site/publish_config.json",
        output_directory=ROOT / "site/data",
    )
