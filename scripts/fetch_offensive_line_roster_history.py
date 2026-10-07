"""Acquire immutable historical CFBD rosters for issue 183.

The FBS roster corpus defines target-season team coverage.  FCS responses are
also retained so a program's pre-FBS seasons can contribute to prior same-
program exposure when the school name maps unambiguously to an FBS team ID.
Raw response bytes are never normalized or overwritten by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from gippyrank.research.data_paths import (
    CFBD_ROSTER_DATASET,
    cfbd_roster_raw_dir,
    raw_root_relative_path,
)

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.collegefootballdata.com"
DEFAULT_RAW_ROOT = cfbd_roster_raw_dir()
DEFAULT_START_SEASON = 2004


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _request(
    client: httpx.Client, endpoint: str, params: dict[str, Any]
) -> httpx.Response:
    for attempt in range(5):
        response = client.get(f"{API}{endpoint}", params=params, timeout=120)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(max(delay, 0.1))
            continue
        response.raise_for_status()
        if not isinstance(response.json(), list):
            raise TypeError(f"CFBD {endpoint} response must be a JSON array")
        return response
    raise RuntimeError(f"CFBD request retries exhausted for {endpoint} {params}")


def _cached_payload(
    destination: Path,
    *,
    endpoint: str,
    params: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any] | None:
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    if not destination.exists() and not sidecar.exists():
        return None
    if not destination.exists() or not sidecar.exists():
        raise RuntimeError(
            f"Incomplete raw artifact pair; refusing to overwrite: {destination}"
        )
    content = destination.read_bytes()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance.get("content_sha256") != _sha256(content):
        raise RuntimeError(
            f"Raw artifact hash mismatch; refusing to overwrite: {destination}"
        )
    if provenance.get("endpoint") != endpoint or provenance.get("parameters") != params:
        raise RuntimeError(f"Raw artifact request metadata mismatch: {destination}")
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise TypeError(f"Cached CFBD payload is not a JSON array: {destination}")
    return {
        "endpoint": endpoint,
        "parameters": params,
        "path": raw_root_relative_path(destination, raw_root),
        "path_base": "raw_root",
        "record_count": len(payload),
        "sha256": _sha256(content),
        "status": "cached",
    }


def _acquire_one(
    client: httpx.Client | None,
    *,
    destination: Path,
    endpoint: str,
    params: dict[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    cached = _cached_payload(
        destination, endpoint=endpoint, params=params, raw_root=raw_root
    )
    if cached is not None:
        return cached
    if client is None:
        raise RuntimeError("CFBD_API_KEY is required to fetch uncached CFBD responses")
    response = _request(client, endpoint, params)
    content = response.content
    payload = response.json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    retrieved_at = datetime.now(UTC).isoformat()
    digest = _sha256(content)
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "content_sha256": digest,
                "endpoint": endpoint,
                "parameters": params,
                "record_count": len(payload),
                "retrieved_at": retrieved_at,
                "source_kind": "cfbd_api_raw_response",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "endpoint": endpoint,
        "parameters": params,
        "path": raw_root_relative_path(destination, raw_root),
        "path_base": "raw_root",
        "record_count": len(payload),
        "sha256": digest,
        "status": "fetched",
    }


def acquire_roster_history(
    *,
    raw_root: Path,
    start_season: int,
    end_season: int,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch FBS team lists and FBS/FCS season rosters without rewriting raw data."""
    api_key = os.environ.get("CFBD_API_KEY")
    raw_root = raw_root.expanduser().resolve()
    owns_client = client is None and bool(api_key)
    if owns_client:
        client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
    requests: list[dict[str, Any]] = []
    try:
        for season in range(start_season, end_season + 1):
            requests.append(
                _acquire_one(
                    client,
                    destination=raw_root / "teams_fbs" / f"{season}.json",
                    endpoint="/teams/fbs",
                    params={"year": season},
                    raw_root=raw_root,
                )
            )
            for classification in ("fbs", "fcs"):
                requests.append(
                    _acquire_one(
                        client,
                        destination=raw_root
                        / "roster"
                        / classification
                        / f"{season}.json",
                        endpoint="/roster",
                        params={"year": season, "classification": classification},
                        raw_root=raw_root,
                    )
                )
            print(
                f"Acquired season {season}: "
                + ", ".join(
                    f"{row['parameters'].get('classification', 'teams')}={row['record_count']}"
                    for row in requests[-3:]
                )
            )
    finally:
        if owns_client:
            client.close()

    manifest = {
        "dataset": CFBD_ROSTER_DATASET,
        "source": "College Football Data API",
        "raw_root": raw_root.as_posix(),
        "request_path_base": "raw_root",
        "season_start": start_season,
        "season_end": end_season,
        "target_classification": "fbs",
        "history_classifications": ["fbs", "fcs"],
        "requests": requests,
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=datetime.now(UTC).year)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_season < DEFAULT_START_SEASON:
        raise SystemExit(f"CFBD roster availability starts in {DEFAULT_START_SEASON}")
    if args.end_season < args.start_season:
        raise SystemExit("--end-season must be on or after --start-season")
    manifest = acquire_roster_history(
        raw_root=args.raw_root,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    print(
        f"Saved {len(manifest['requests'])} immutable CFBD responses to "
        f"{args.raw_root.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
