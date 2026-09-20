"""Capture immutable CFBD inputs for the production preseason transfer pipeline.

This command is the only network-facing step for issue 113.  It stores raw
response bytes unchanged, writes a per-response provenance sidecar, and
updates a manifest that names the canonical snapshot for each request.  A
refresh receives a new versioned filename; an existing raw file is never
overwritten.

Example::

    uv run python scripts/fetch_preseason_transfer_snapshots.py \
        --season 2027 --raw-root data/raw/cfbd/preseason/transfers

The command intentionally fails after storing responses if any response was
retrieved after that target season's August 15 cutoff.  Such bytes remain
available for diagnosis but cannot be used by the production derivation step.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from gippyrank.preseason_transfer import (
    CFBD_API,
    SnapshotRecord,
    SnapshotSpec,
    load_snapshot_manifest,
    required_snapshot_specs,
    write_immutable_snapshot,
    write_snapshot_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data/raw/cfbd/preseason/transfers"


def request_json(
    client: httpx.Client,
    spec: SnapshotSpec,
    *,
    api_base: str = CFBD_API,
    attempts: int = 5,
) -> httpx.Response:
    """Fetch one endpoint with bounded retry handling for transient CFBD errors."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        response = client.get(
            api_base.rstrip("/") + spec.endpoint,
            params=dict(spec.query_parameters),
            timeout=120,
        )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == attempts - 1:
                response.raise_for_status()
            delay = float(response.headers.get("Retry-After", 2**attempt))
            time.sleep(max(delay, 0.1))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise TypeError(
                f"CFBD {spec.endpoint} response was not a JSON array: {spec.query_parameters}"
            )
        return response
    raise RuntimeError("CFBD retry loop ended unexpectedly")


def _existing_canonical(
    manifest_path: Path,
    spec: SnapshotSpec,
) -> SnapshotRecord | None:
    if not manifest_path.exists():
        return None
    manifest = load_snapshot_manifest(manifest_path, verify_hashes=True)
    matches = [
        record
        for record in manifest.for_target(spec.target_season)
        if record.request_key == spec.request_key
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"manifest has multiple canonical snapshots for request {spec.request_key}"
        )
    return matches[0] if matches else None


def fetch_one(
    client: httpx.Client,
    *,
    raw_root: Path,
    manifest_path: Path,
    spec: SnapshotSpec,
    refresh: bool = False,
    version: str = "v1",
    sleep_seconds: float = 0.1,
) -> SnapshotRecord:
    """Fetch one request, preserving an existing canonical snapshot by default."""
    if not refresh:
        cached = _existing_canonical(manifest_path, spec)
        if cached is not None:
            return cached
    response = request_json(client, spec)
    retrieved_at = datetime.now(UTC).isoformat()
    record = write_immutable_snapshot(
        raw_root=raw_root,
        spec=spec,
        content=response.content,
        retrieval_timestamp=retrieved_at,
        version=version,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return record


def _season_arguments(args: argparse.Namespace) -> list[int]:
    seasons = sorted(set(args.seasons or []))
    if args.start_season is not None or args.end_season is not None:
        if args.start_season is None or args.end_season is None:
            raise ValueError(
                "--start-season and --end-season must be supplied together"
            )
        if args.start_season > args.end_season:
            raise ValueError("start season must not exceed end season")
        seasons.extend(range(args.start_season, args.end_season + 1))
        seasons = sorted(set(seasons))
    if not seasons:
        raise ValueError("supply at least one --season or a season range")
    return seasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", dest="seasons", action="append", type=int)
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--first-week", type=int, default=1)
    parser.add_argument("--last-week", type=int, default=16)
    parser.add_argument(
        "--classification",
        dest="classifications",
        action="append",
        choices=("fbs", "fcs"),
        help="repeat for each defensive classification; default: fbs and fcs",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="capture a new explicitly versioned snapshot instead of using the canonical cache",
    )
    parser.add_argument(
        "--retrospective-reconstruction",
        action="store_true",
        help="allow late canonical inputs only for the explicit 2026 reconstruction",
    )
    parser.add_argument(
        "--version",
        help="version suffix for a refresh (default: UTC retrieval timestamp)",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = _season_arguments(args)
    if args.retrospective_reconstruction and seasons != [2026]:
        raise ValueError(
            "--retrospective-reconstruction is restricted to target season 2026"
        )
    if args.first_week > args.last_week:
        raise ValueError("first week must not exceed last week")
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    raw_root = args.raw_root
    manifest_path = args.manifest or raw_root / "manifest.json"
    classifications = tuple(args.classifications or ("fbs", "fcs"))
    version = args.version
    if args.refresh and not version:
        version = "v" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not version:
        version = "v1"
    specs = [
        spec
        for season in seasons
        for spec in required_snapshot_specs(
            season,
            first_week=args.first_week,
            last_week=args.last_week,
            classifications=classifications,
        )
    ]
    records: list[SnapshotRecord] = []
    failures: list[dict[str, Any]] = []
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for spec in specs:
            try:
                records.append(
                    fetch_one(
                        client,
                        raw_root=raw_root,
                        manifest_path=manifest_path,
                        spec=spec,
                        refresh=args.refresh,
                        version=version,
                        sleep_seconds=args.sleep_seconds,
                    )
                )
            except (httpx.HTTPError, OSError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "target_season": spec.target_season,
                        "source": spec.source,
                        "source_season": spec.source_season,
                        "parameters": dict(spec.query_parameters),
                        "error": str(error),
                    }
                )
    if failures:
        raise RuntimeError(
            "preseason snapshot acquisition failed: "
            + json.dumps(failures, sort_keys=True)
        )
    write_snapshot_manifest(
        manifest_path,
        records,
        raw_root=raw_root,
        required_specs=specs,
        allow_late_canonical=args.retrospective_reconstruction,
    )
    manifest = load_snapshot_manifest(
        manifest_path, required_seasons=seasons, verify_hashes=True
    )
    late = [
        record.path
        for record in records
        if record.target_season in seasons and not record.captured_on_or_before_cutoff
    ]
    result = {
        "manifest": str(manifest_path),
        "target_seasons": seasons,
        "canonical_snapshots": len(
            [record for record in manifest.snapshots if record.canonical]
        ),
        "late_snapshots": late,
        "raw_root": str(raw_root),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if late and not args.retrospective_reconstruction:
        raise RuntimeError(
            "snapshots were retrieved after their preseason cutoff; "
            "they are retained but cannot feed production derivation"
        )


if __name__ == "__main__":
    main()
