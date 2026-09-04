"""Minimal normalization adapters for the Massey ranking sources."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

NORMALIZED_FIELDS = (
    "season",
    "subdivision",
    "team_id",
    "team_source_id",
    "team_name",
    "system_code",
    "system_name",
    "ranking_date",
    "ordinal_rank",
    "is_composite",
    "source",
    "source_file",
)


@dataclass(frozen=True)
class MasseyObservation:
    season: int
    subdivision: str | None
    team_id: str | None
    team_source_id: str | None
    team_name: str
    system_code: str
    system_name: str | None
    ranking_date: str | None
    ordinal_rank: int
    is_composite: bool
    source: str
    source_file: str


@dataclass(frozen=True)
class CFBDTeam:
    id: int
    school: str
    classification: str
    alternate_names: tuple[str, ...] = ()


# These are spelling/abbreviation expansions, not subdivision decisions.
MASSEY_TEAM_ALIASES = {
    "Abilene Chr": "Abilene Christian",
    "Appalachian St": "App State",
    "Ark Pine Bluff": "Arkansas-Pine Bluff",
    "Ball St": "Ball State",
    "Cent Arkansas": "Central Arkansas",
    "Central Conn": "Central Connecticut",
    "Citadel": "The Citadel",
    "Coastal Car": "Coastal Carolina",
    "CS Sacramento": "Sacramento State",
    "Dixie St": "Utah Tech",
    "ETSU": "East Tennessee State",
    "FL Atlantic": "Florida Atlantic",
    "Gardner Webb": "Gardner-Webb",
    "Hawaii": "Hawai'i",
    "Houston Bap": "Houston Christian",
    "Houston Chr": "Houston Christian",
    "Iowa St": "Iowa State",
    "Jacksonville St": "Jacksonville State",
    "LIU Post": "Long Island University",
    "McNeese St": "McNeese",
    "Mississippi": "Ole Miss",
    "MS Valley St": "Mississippi Valley State",
    "Missouri St": "Missouri State",
    "Nicholls St": "Nicholls",
    "Northwestern LA": "Northwestern State",
    "Ohio St": "Ohio State",
    "Penn St": "Penn State",
    "S Carolina St": "South Carolina State",
    "SE Missouri St": "Southeast Missouri State",
    "Sam Houston St": "Sam Houston",
    "Southern Univ": "Southern",
    "TN Martin": "UT Martin",
    "TX Southern": "Texas Southern",
    "UT San Antonio": "UTSA",
    "UTRGV": "UT Rio Grande Valley",
    "Utah St": "Utah State",
    "W Salem St": "Winston-Salem State",
}


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(rb"[^a-z0-9]", b"", normalized.lower()).decode()


def _source_file(path: Path) -> str:
    """Use repository-relative source paths when the input is in this checkout."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def fetch_cfbd_teams(year: int, api_key: str, client: Any = httpx) -> list[CFBDTeam]:
    """Fetch season-specific CFBD teams; classification is supplied by CFBD."""
    response = client.get(
        "https://api.collegefootballdata.com/teams",
        params={"year": year},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return parse_cfbd_teams(response.json())


def parse_cfbd_teams(items: Iterable[dict[str, Any]]) -> list[CFBDTeam]:
    by_id: dict[int, CFBDTeam] = {}
    for item in items:
        by_id[int(item["id"])] = CFBDTeam(
            id=int(item["id"]),
            school=item["school"],
            classification=item["classification"],
            alternate_names=tuple(item.get("alternateNames") or ()),
        )
    return list(by_id.values())


def build_team_matches(
    massey_names: Iterable[str], cfbd_teams: Iterable[CFBDTeam]
) -> tuple[dict[str, CFBDTeam], dict[str, list[CFBDTeam]], list[str]]:
    """Match names using exact/curated aliases and return unresolved names."""
    index: dict[str, dict[int, CFBDTeam]] = defaultdict(dict)
    for team in cfbd_teams:
        for name in (team.school, *team.alternate_names):
            index[_key(name)][team.id] = team

    matched: dict[str, CFBDTeam] = {}
    ambiguous: dict[str, list[CFBDTeam]] = {}
    unmapped: list[str] = []
    for original in sorted(set(massey_names)):
        lookup_name = MASSEY_TEAM_ALIASES.get(original, original)
        candidates = list(index.get(_key(lookup_name), {}).values())
        if len(candidates) == 1:
            matched[original] = candidates[0]
        elif candidates:
            ambiguous[original] = candidates
        else:
            unmapped.append(original)
    return matched, ambiguous, unmapped


def read_kaggle_observations(
    path: Path, matches: dict[str, CFBDTeam]
) -> tuple[list[MasseyObservation], list[str]]:
    observations = list(iter_kaggle_observations(path, matches))
    unmapped = sorted({row.team_name for row in observations if row.team_id is None})
    return observations, unmapped


def iter_kaggle_observations(
    path: Path, matches: dict[str, CFBDTeam]
) -> Iterable[MasseyObservation]:
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            season_text, team_id, team_name, code, name, date, rank = fields
            season = int(season_text.strip().removeprefix("cf"))
            team_name = team_name.strip()
            team = matches.get(team_name)
            yield MasseyObservation(
                season=season,
                subdivision=(team.classification if team else None),
                team_id=str(team.id) if team else None,
                team_source_id=team_id.strip(),
                team_name=team_name,
                system_code=code.strip(),
                system_name=name.strip() or None,
                ranking_date=date.strip() or None,
                ordinal_rank=int(rank.strip()),
                is_composite=code.strip() == "CMP",
                source="massey_kaggle",
                source_file=_source_file(path),
            )


def read_export_observations(
    path: Path,
    matches: dict[str, CFBDTeam] | None = None,
) -> tuple[list[MasseyObservation], list[str]]:
    """Convert one wide final export to long observations without inventing metadata."""
    season_match = re.search(r"(\d{4})", path.name)
    if season_match is None:
        raise ValueError(f"Cannot determine season from {path}")
    subdivision = path.stem[:3]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:6] != ["Team", "Conf", "W-L", "&Delta;", "CMP", "Sort"]:
            raise ValueError(f"Unexpected Massey export header in {path}")
        observations = []
        unmapped: set[str] = set()
        for row in reader:
            if not row:
                continue
            team_name = row[0].strip()
            team = matches.get(team_name) if matches is not None else None
            if matches is not None and team is None:
                unmapped.add(team_name)
            for code, value in zip(header[4:], row[4:]):
                if code == "Sort" or value.strip() in {"", "--"}:
                    continue
                observations.append(
                    MasseyObservation(
                        season=int(season_match.group(1)),
                        subdivision=subdivision,
                        team_id=str(team.id) if team else None,
                        team_source_id=None,
                        team_name=team_name,
                        system_code=code,
                        system_name=None,
                        ranking_date=None,
                        ordinal_rank=int(value.strip()),
                        is_composite=code == "CMP",
                        source="massey_composite_export",
                        source_file=_source_file(path),
                    )
                )
    return observations, sorted(unmapped)


def write_observations(path: Path, observations: Iterable[MasseyObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(item) for item in observations)


def write_mapping_report(
    path: Path,
    matched: dict[str, CFBDTeam],
    ambiguous: dict[str, list[CFBDTeam]],
    unmapped: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matched_team_count": len(matched),
        "ambiguous": {
            name: [asdict(team) for team in candidates]
            for name, candidates in ambiguous.items()
        },
        "unmapped": unmapped,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
