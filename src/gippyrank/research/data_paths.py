"""Shared paths for local research data that should not live in a checkout."""

from __future__ import annotations

import os
from pathlib import Path

CFBD_ROSTER_DATASET = "offensive_line_shared_roster_issue_183"


def research_data_dir() -> Path:
    """Return the configured research-data root outside the repository.

    ``GIPPYRANK_DATA_DIR`` takes precedence. Otherwise, use the standard user
    cache location ``$XDG_CACHE_HOME/gippyrank/research-data`` or
    ``~/.cache/gippyrank/research-data``.
    """
    configured = os.environ.get("GIPPYRANK_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return (cache_home / "gippyrank" / "research-data").expanduser().resolve()


def cfbd_roster_raw_dir() -> Path:
    """Return the shared raw CFBD roster corpus directory."""
    return research_data_dir() / "raw" / "cfbd" / CFBD_ROSTER_DATASET


def raw_root_relative_path(path: Path, raw_root: Path) -> str:
    """Serialize an artifact path relative to its corpus root."""
    return path.resolve().relative_to(raw_root.resolve()).as_posix()
