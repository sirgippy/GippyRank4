"""Fetch immutable CFBD inputs for the defensive-transfer research audit.

Portal responses are acquired by ``fetch_transfer_oracle.py`` and remain in
the existing transfer raw root.  This command fetches the complementary prior
season roster and weekly player-box-score responses.  It never normalizes or
rewrites a response: every raw byte is stored next to a provenance sidecar.
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

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.collegefootballdata.com"
ENDPOINTS = {
    "roster": "/roster",
    "games_players": "/games/players",
}


def _request_json(
    client: httpx.Client, endpoint: str, params: dict[str, Any]
) -> httpx.Response:
    for attempt in range(5):
        response = client.get(API + endpoint, params=params, timeout=120)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            delay = float(response.headers.get("Retry-After", 2**attempt))
            time.sleep(max(delay, 0.1))
            continue
        response.raise_for_status()
        if not isinstance(response.json(), list):
            raise TypeError(f"CFBD response for {endpoint} was not an array")
        return response
    raise RuntimeError("CFBD request retry loop ended unexpectedly")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_response(
    response: httpx.Response,
    *,
    destination: Path,
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    content = response.content
    payload = response.json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "content_sha256": _sha256(content),
                "endpoint": endpoint,
                "parameters": params,
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
        "endpoint": endpoint,
        "parameters": params,
        "path": str(destination.resolve().relative_to(ROOT.resolve())),
        "record_count": len(payload),
        "sha256": _sha256(content),
        "status": "fetched",
    }


def _fetch_one(
    client: httpx.Client,
    *,
    raw_root: Path,
    kind: str,
    season: int,
    week: int | None,
    classification: str,
    refresh: bool,
) -> dict[str, Any]:
    if kind == "roster":
        destination = raw_root / "roster" / classification / f"{season}.json"
        params: dict[str, Any] = {"year": season, "classification": classification}
    else:
        if week is None:
            raise ValueError("games_players requires a week")
        destination = (
            raw_root
            / "games_players"
            / classification
            / f"{season}-week-{week:02d}.json"
        )
        params = {
            "year": season,
            "week": week,
            "classification": classification,
            "seasonType": "both",
        }
    if destination.exists() and not refresh:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        return {
            "endpoint": ENDPOINTS[kind],
            "parameters": params,
            "path": str(destination.resolve().relative_to(ROOT.resolve())),
            "record_count": len(payload) if isinstance(payload, list) else None,
            "status": "cached",
        }
    response = _request_json(client, ENDPOINTS[kind], params)
    return _write_response(
        response,
        destination=destination,
        endpoint=ENDPOINTS[kind],
        params=params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/cfbd/preseason/defensive_transfers_complete",
    )
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--first-week", type=int, default=1)
    parser.add_argument("--last-week", type=int, default=16)
    parser.add_argument(
        "--classification",
        dest="classifications",
        action="append",
        choices=("fbs", "fcs"),
        help="division to fetch; repeat the option, default: fbs and fcs",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not configured")
    if args.start_season > args.end_season:
        raise ValueError("start season must not exceed end season")
    if args.first_week > args.last_week:
        raise ValueError("first week must not exceed last week")
    classifications = args.classifications or ["fbs", "fcs"]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for season in range(args.start_season, args.end_season + 1):
            for classification in classifications:
                try:
                    results.append(
                        _fetch_one(
                            client,
                            raw_root=args.raw_root,
                            kind="roster",
                            season=season,
                            week=None,
                            classification=classification,
                            refresh=args.refresh,
                        )
                    )
                except (httpx.HTTPError, TypeError, ValueError) as error:
                    failures.append(
                        {
                            "kind": "roster",
                            "season": season,
                            "classification": classification,
                            "error": str(error),
                        }
                    )
                time.sleep(max(args.sleep_seconds, 0.0))
                for week in range(args.first_week, args.last_week + 1):
                    try:
                        results.append(
                            _fetch_one(
                                client,
                                raw_root=args.raw_root,
                                kind="games_players",
                                season=season,
                                week=week,
                                classification=classification,
                                refresh=args.refresh,
                            )
                        )
                    except (httpx.HTTPError, TypeError, ValueError) as error:
                        failures.append(
                            {
                                "kind": "games_players",
                                "season": season,
                                "week": week,
                                "classification": classification,
                                "error": str(error),
                            }
                        )
                    time.sleep(max(args.sleep_seconds, 0.0))

    manifest = {
        "source": API,
        "endpoints": ENDPOINTS,
        "seasons": [args.start_season, args.end_season],
        "weeks": [args.first_week, args.last_week],
        "classifications": classifications,
        "season_type": "both",
        "raw_payloads_unchanged": True,
        "results": results,
        "failures": failures,
    }
    args.raw_root.mkdir(parents=True, exist_ok=True)
    (args.raw_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fetched_or_cached": len(results),
                "failures": failures,
                "raw_root": str(args.raw_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise RuntimeError(
            f"defensive data acquisition failed for {len(failures)} payloads"
        )


if __name__ == "__main__":
    main()
