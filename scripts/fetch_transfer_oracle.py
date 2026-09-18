"""Acquire raw CFBD transfer and player-usage payloads for issue #91.

The response bytes are stored unchanged under ``data/raw``.  Normalization and
feature construction happen in ``investigate_transfer_roster_continuity.py``;
this script never writes processed model inputs.  Raw files are ignored by
Git, and each payload receives a separate provenance sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.collegefootballdata.com"
ENDPOINTS = {
    "portal": "/player/portal",
    "usage": "/player/usage",
}


def request_json(
    client: httpx.Client, endpoint: str, params: dict[str, int]
) -> httpx.Response:
    for attempt in range(5):
        response = client.get(API + endpoint, params=params, timeout=90)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            delay = float(response.headers.get("Retry-After", 2**attempt))
            time.sleep(max(delay, 0.1))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("CFBD request retry loop ended unexpectedly")


def payload_path(raw_root: Path, kind: str, season: int) -> Path:
    return raw_root / kind / f"{season}.json"


def fetch_one(
    client: httpx.Client,
    *,
    raw_root: Path,
    kind: str,
    season: int,
    refresh: bool,
) -> dict[str, object]:
    destination = payload_path(raw_root, kind, season)
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    if destination.exists() and not refresh:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        return {
            "kind": kind,
            "season": season,
            "status": "cached",
            "path": str(destination.relative_to(raw_root.parent.parent.parent)),
            "record_count": len(payload) if isinstance(payload, list) else None,
        }
    response = request_json(client, ENDPOINTS[kind], {"year": season})
    content = response.content
    payload = response.json()
    if not isinstance(payload, list):
        raise TypeError(f"CFBD {kind} response for {season} was not an array")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    sidecar.write_text(
        json.dumps(
            {
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "endpoint": ENDPOINTS[kind],
                "parameters": {"year": season},
                "record_count": len(payload),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "source_kind": "cfbd_api_raw_response",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "kind": kind,
        "season": season,
        "status": "fetched",
        "path": str(destination.relative_to(raw_root.parent.parent.parent)),
        "record_count": len(payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data/raw/cfbd/preseason/transfers"
    )
    parser.add_argument("--portal-start", type=int, default=2021)
    parser.add_argument("--portal-end", type=int, default=2025)
    parser.add_argument("--usage-start", type=int, default=2020)
    parser.add_argument("--usage-end", type=int, default=2024)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    if args.portal_start > args.portal_end or args.usage_start > args.usage_end:
        raise ValueError("start seasons must not exceed end seasons")
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for kind, start, end in (
            ("portal", args.portal_start, args.portal_end),
            ("usage", args.usage_start, args.usage_end),
        ):
            for season in range(start, end + 1):
                try:
                    results.append(
                        fetch_one(
                            client,
                            raw_root=args.raw_root,
                            kind=kind,
                            season=season,
                            refresh=args.refresh,
                        )
                    )
                except (httpx.HTTPError, TypeError, ValueError) as error:
                    failures.append(
                        {"kind": kind, "season": season, "error": str(error)}
                    )
                time.sleep(0.1)
    manifest = {
        "source": API,
        "portal_endpoint": ENDPOINTS["portal"],
        "usage_endpoint": ENDPOINTS["usage"],
        "portal_seasons": [args.portal_start, args.portal_end],
        "usage_seasons": [args.usage_start, args.usage_end],
        "raw_payloads_unchanged": True,
        "results": results,
        "failures": failures,
    }
    args.raw_root.mkdir(parents=True, exist_ok=True)
    (args.raw_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"results": results, "failures": failures}, indent=2, sort_keys=True)
    )
    if failures:
        raise RuntimeError(f"transfer acquisition failed for {len(failures)} payloads")


if __name__ == "__main__":
    main()
