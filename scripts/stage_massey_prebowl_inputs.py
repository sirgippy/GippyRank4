"""Stage existing Massey and CFBD raw inputs in repo-local data directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from gippyrank.data.massey_prebowl import SEASONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = (
    ROOT / "data/research/massey_prebowl_issue_198/collection_checklist.csv"
)
KAGGLE_URL = "https://www.kaggle.com/datasets/masseyratings/rankings"
CFBD_TEAMS_URL = "https://api.collegefootballdata.com/teams?year={season}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_file(source: Path, destination: Path, metadata: dict[str, object]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = {
        "schema_version": 1,
        **metadata,
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_name(f"{destination.name}.provenance.json")
    if destination.exists():
        if sha256(destination) != expected["sha256"]:
            raise FileExistsError(
                f"Refusing to replace different raw bytes: {destination}"
            )
        if sidecar.exists():
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
            mismatches = {
                key: (existing.get(key), value)
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Conflicting provenance at {sidecar}: {mismatches}")
            return
    else:
        shutil.copyfile(source, destination)
    sidecar.write_text(
        json.dumps(
            {
                **expected,
                "imported_at": datetime.now(UTC).isoformat(),
                "copied_byte_for_byte": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def stage(source_root: Path, raw_root: Path, checklist_path: Path) -> None:
    with checklist_path.open(newline="", encoding="utf-8-sig") as handle:
        checklist = {int(row["season"]): row for row in csv.DictReader(handle)}
    if set(checklist) != set(SEASONS):
        raise ValueError("Checklist must cover every season from 2003 through 2025")

    historical_seasons = [*range(2003, 2019), 2020]
    for season in historical_seasons:
        source = source_root / "massey_rankings" / f"cf{season}.csv"
        row = checklist[season]
        stage_file(
            source,
            raw_root / "massey_prebowl" / "historical" / source.name,
            {
                "source": "massey_kaggle_historical_rankings",
                "source_url": KAGGLE_URL,
                "season": season,
                "snapshot_date": row["snapshot_date"] or None,
            },
        )

    for season in SEASONS:
        source = source_root / "cfbd" / "teams" / f"{season}.json"
        stage_file(
            source,
            raw_root / "cfbd" / "teams" / source.name,
            {
                "source": "collegefootball_data_teams",
                "source_url": CFBD_TEAMS_URL.format(season=season),
                "season": season,
                "snapshot_date": None,
            },
        )

    print(
        f"staged {len(historical_seasons)} historical ranking files and "
        f"{len(SEASONS)} CFBD team mappings under {raw_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Existing data/raw directory containing massey_rankings/ and cfbd/.",
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    stage(arguments.source_root, arguments.raw_root, arguments.checklist)
