"""Build the issue 183 OL shared-roster history panel and coverage audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gippyrank.research.offensive_line_roster_continuity import (
    ContinuityBuild,
    build_continuity_artifacts,
)
from gippyrank.transfer_oracle import normalize_team_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data/raw/cfbd/offensive_line_shared_roster_issue_183"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/offensive_line_shared_roster_issue_183"
DEFAULT_START_SEASON = 2004
ISSUE_181_SAMPLE_PATH = (
    ROOT / "data/research/offensive_line_shared_roster_issue_183/"
    "co_start_pilot_issue_181_sample.csv"
)
ISSUE_181_SAMPLE_COMMIT = "806836e5769d4dd80208546c1c0f60cc07038435"
MANUAL_VALIDATION_PATH = (
    ROOT / "data/research/offensive_line_shared_roster_issue_183/manual_validation.csv"
)
ISSUE_181_TEAM_ALIASES = {
    normalize_team_name("Appalachian State"): normalize_team_name("App State")
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_response(
    path: Path,
    *,
    endpoint: str,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    sidecar_path = path.with_name(f"{path.name}.provenance.json")
    if not path.exists() and not sidecar_path.exists():
        return None, {"path": str(path), "status": "not_acquired"}
    if not path.exists() or not sidecar_path.exists():
        raise RuntimeError(
            f"Raw source and provenance sidecar must appear together: {path}"
        )
    content = path.read_bytes()
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    digest = _sha256(content)
    if provenance.get("content_sha256") != digest:
        raise RuntimeError(f"CFBD source hash mismatch: {path}")
    if provenance.get("endpoint") != endpoint:
        raise RuntimeError(f"CFBD source endpoint mismatch: {path}")
    if provenance.get("parameters") != parameters:
        raise RuntimeError(f"CFBD source query mismatch: {path}")
    payload = json.loads(content)
    if not isinstance(payload, list):
        raise TypeError(f"CFBD response is not a JSON array: {path}")
    if provenance.get("record_count") != len(payload):
        raise RuntimeError(f"CFBD source record-count mismatch: {path}")
    return payload, {
        "endpoint": endpoint,
        "parameters": parameters,
        "path": str(path.resolve().relative_to(ROOT.resolve())),
        "record_count": len(payload),
        "sha256": digest,
        "retrieved_at": provenance.get("retrieved_at"),
        "status": "verified",
    }


def load_source_corpus(
    *, raw_root: Path, start_season: int, end_season: int
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    set[tuple[str, int]],
    list[dict[str, Any]],
]:
    roster_payloads: dict[tuple[str, int], list[dict[str, Any]]] = {}
    teams_by_season: dict[int, list[dict[str, Any]]] = {}
    roster_response_seasons: set[tuple[str, int]] = set()
    inventory: list[dict[str, Any]] = []
    for season in range(start_season, end_season + 1):
        teams, audit = _read_response(
            raw_root / "teams_fbs" / f"{season}.json",
            endpoint="/teams/fbs",
            parameters={"year": season},
        )
        inventory.append({"season": season, "source": "teams_fbs", **audit})
        if teams is not None:
            teams_by_season[season] = teams
        for classification in ("fbs", "fcs"):
            payload, audit = _read_response(
                raw_root / "roster" / classification / f"{season}.json",
                endpoint="/roster",
                parameters={"year": season, "classification": classification},
            )
            inventory.append(
                {"season": season, "source": f"roster_{classification}", **audit}
            )
            if payload is not None:
                roster_payloads[(classification, season)] = payload
                roster_response_seasons.add((classification, season))
    return roster_payloads, teams_by_season, roster_response_seasons, inventory


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fields = ["status"]
    with path.open("wb") as raw_handle:
        output_handle: Any
        if path.suffix == ".gz":
            compressed = gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            )
            output_handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        else:
            output_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", newline="")
        writer = csv.DictWriter(
            output_handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )
        output_handle.flush()
        output_handle.close()


def _read_manual_validation(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_issue181_overlap_coverage(
    artifacts: ContinuityBuild, sample_path: Path
) -> list[dict[str, Any]]:
    """Join the frozen issue 181 sample to exact target team-season outputs."""
    coverage_by_key = {
        (int(row["season"]), normalize_team_name(str(row["team_name"]))): row
        for row in artifacts.team_season_coverage
    }
    summaries_by_key = {
        (int(row["season"]), normalize_team_name(str(row["team_name"]))): row
        for row in artifacts.team_season_summaries
    }
    joined: list[dict[str, Any]] = []
    with sample_path.open(newline="", encoding="utf-8") as handle:
        for sample in csv.DictReader(handle):
            season = int(sample["target_season"])
            team = sample["team"].strip()
            normalized_sample_team = normalize_team_name(team)
            normalized_sample_team = ISSUE_181_TEAM_ALIASES.get(
                normalized_sample_team, normalized_sample_team
            )
            key = (season, normalized_sample_team)
            coverage = coverage_by_key.get(key)
            summary = summaries_by_key.get(key)
            if coverage is None or summary is None:
                raise ValueError(
                    f"Issue 181 sample team-season did not match the panel: {season} {team}"
                )
            ol_count = int(coverage["identifiable_ol_player_count"])
            joined.append(
                {
                    **sample,
                    "sample_source_commit": ISSUE_181_SAMPLE_COMMIT,
                    "team_id": coverage["team_id"],
                    "canonical_team_name": coverage["team_name"],
                    "roster_status": coverage["roster_status"],
                    "fbs_roster_player_rows": coverage["roster_player_rows"],
                    "identifiable_ol_player_count": ol_count,
                    "ol_player_ids_available": coverage["ol_player_ids_available"],
                    "ol_player_id_link_rate": (
                        coverage["ol_player_ids_available"] / ol_count
                        if ol_count
                        else ""
                    ),
                    "ol_with_prior_same_program_roster_link": coverage[
                        "ol_with_prior_same_program_roster_link"
                    ],
                    "ol_with_no_prior_same_program_roster_link": coverage[
                        "ol_with_no_prior_same_program_roster_link"
                    ],
                    "pair_count_evaluable": summary["pair_count_evaluable"],
                    "pairs_with_at_least_1_shared_season": summary[
                        "pairs_with_at_least_1_shared_season"
                    ],
                    "pairs_with_at_least_2_shared_seasons": summary[
                        "pairs_with_at_least_2_shared_seasons"
                    ],
                    "largest_target_ol_group_on_one_prior_roster": summary[
                        "largest_target_ol_group_on_one_prior_roster"
                    ],
                    "ol_share_with_no_prior_roster_season_at_current_program": summary[
                        "ol_share_with_no_prior_roster_season_at_current_program"
                    ],
                    "history_left_censored": summary["history_left_censored"],
                }
            )
    return joined


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        )
    return "\n".join(lines)


def _format_rate(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{100 * float(value):.1f}%"


def _build_report(
    artifacts: ContinuityBuild,
    *,
    source_inventory: list[dict[str, Any]],
    manual_validation: list[dict[str, str]],
    issue181_overlap: list[dict[str, Any]],
    start_season: int,
    end_season: int,
) -> str:
    coverage = artifacts.coverage_by_season
    expected_team_seasons = sum(int(row["expected_fbs_teams"]) for row in coverage)
    available_team_seasons = sum(
        int(row["roster_available_team_count"]) for row in coverage
    )
    missing_team_seasons = sum(
        int(row["roster_missing_team_count"]) for row in coverage
    )
    no_ol_team_seasons = sum(
        int(row["roster_present_no_identifiable_ol_team_count"]) for row in coverage
    )
    response_missing_seasons = [
        int(row["season"]) for row in coverage if not row["roster_response_available"]
    ]
    event_counts = artifacts.source_audit["identity_event_counts"]
    same_year_multi_team = int(
        event_counts.get("same_id_listed_at_multiple_programs_in_season", 0)
    )
    possible_id_changes = int(
        event_counts.get("possible_player_id_change_unresolved", 0)
    )
    duplicate_names = int(
        event_counts.get("duplicate_normalized_name_same_program_season", 0)
    )
    ambiguous_name_links = int(
        event_counts.get("ambiguous_same_name_across_adjacent_seasons", 0)
    )
    name_variants = int(event_counts.get("source_name_variant_for_same_id", 0))
    position_changes = int(
        event_counts.get("position_change_or_label_change_for_same_id", 0)
    )
    multi_program_ids = int(
        event_counts.get("same_id_observed_at_multiple_programs_across_seasons", 0)
    )
    fbs_unique_ids = int(artifacts.source_audit["unique_fbs_source_player_ids"])
    fbs_multi_season_ids = int(
        artifacts.source_audit["fbs_source_player_ids_observed_in_multiple_seasons"]
    )
    fbs_same_program_returns = int(
        artifacts.source_audit[
            "fbs_source_player_ids_with_adjacent_same_program_return"
        ]
    )
    unmatched_team_rows = int(artifacts.source_audit["unmatched_fbs_roster_team_rows"])
    all_fbs_roster_rows = int(artifacts.source_audit["roster_player_rows_fbs"])
    fbs_id_rows = sum(
        bool(row["source_player_id"])
        for row in artifacts.player_seasons
        if row["source_classification"] == "fbs"
    )
    fbs_id_rate = fbs_id_rows / all_fbs_roster_rows if all_fbs_roster_rows else 0.0
    fbs_duplicate_id_rows = sum(
        int(row["duplicate_id_team_season_rows"])
        for row in artifacts.team_season_coverage
    )

    position_rows = [
        [
            row["position_original"] or "(blank)",
            row["normalized_ol_status"],
            row["roster_row_count"],
        ]
        for row in artifacts.position_vocabulary
    ]
    season_rows = [
        [
            row["season"],
            row["expected_fbs_teams"],
            row["roster_available_team_count"],
            row["roster_present_no_identifiable_ol_team_count"],
            row["fbs_roster_player_rows"],
            _format_rate(row["fbs_roster_player_id_rate"]),
            _format_rate(row["fbs_roster_player_name_rate"]),
            _format_rate(row["fbs_roster_position_rate"]),
            _format_rate(row["fbs_roster_jersey_number_rate"]),
            row["ol_players_min"] or "—",
            row["ol_players_median"] or "—",
            row["ol_players_max"] or "—",
            _format_rate(row["ol_player_id_link_rate"]),
        ]
        for row in coverage
    ]
    validation_rows = [
        [
            row.get("case_id", ""),
            row.get("validation_type", ""),
            row.get("school", ""),
            row.get("season_pair", ""),
            row.get("cfbd_result", ""),
            row.get("official_result", ""),
            row.get("discrepancy_class", ""),
            " · ".join(
                f"[official]({url.strip()})"
                for url in row.get("official_source_url", "").split(";")
                if url.strip()
            ),
        ]
        for row in manual_validation
    ]
    not_yet_validated = not manual_validation
    validation_status = (
        "No manually verified school-roster cases have been added yet."
        if not_yet_validated
        else f"{len(manual_validation)} bounded manual validation cases are recorded below."
    )
    inventory_hashes = sum(bool(row.get("sha256")) for row in source_inventory)
    sample_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in issue181_overlap:
        sample_by_window.setdefault(str(row["window_start"]), []).append(row)
    issue181_rows = [
        [
            window,
            len(window_rows),
            sum(bool(row["identifiable_ol_player_count"]) for row in window_rows),
            sum(not bool(row["identifiable_ol_player_count"]) for row in window_rows),
            sum(
                int(row["pairs_with_at_least_1_shared_season"] or 0)
                for row in window_rows
            ),
        ]
        for window, window_rows in sorted(sample_by_window.items())
    ]
    latest_fbs_retrieval = max(
        (
            str(row["retrieved_at"])
            for row in source_inventory
            if row.get("source") == "roster_fbs" and row.get("retrieved_at")
        ),
        default="unknown",
    )

    lines = [
        "# Issue 183: historical offensive-line shared-roster continuity",
        "",
        f"**Panel:** CFBD seasons {start_season}–{end_season}; target teams are season-specific FBS members.",
        "",
        "## Scope and decision",
        "",
        "This research panel measures whether target-season offensive linemen shared earlier roster seasons at the same school. It does not estimate a production coefficient, infer team strength, or use game starts, games, outcomes, or the eventual starting five.",
        "",
        "**Decision: Stop.** Do not treat this CFBD-only history as a historical OL target pool or advance it into model development. The official Vanderbilt bio for Drew Birchmeier says he moved from defensive line to offensive line in 2020, while CFBD labels his 2017–2020 roster rows as OL. CFBD also identifies no OL at 119 FBS programs in 2004 and only 18 of 120 in 2008; the official Iowa 2008 roster includes offensive linemen despite CFBD returning 33 Iowa rows and zero OL labels. These checks show that the source position field can misstate the historical target pool. Keep the ID-linked membership outputs for bounded retrospective research only; an as-of-season OL source or an independently validated target list is needed before any feature use.",
        "",
        "## Source and acquisition",
        "",
        "The acquisition script requests CFBD `/teams/fbs?year=T` and `/roster?year=T&classification=fbs|fcs` for each season. FBS lists define the expected team-season denominator and target roster pool. FCS roster rows are retained for same-program history when the school name maps exactly to one CFBD FBS team ID, and for transfer-ID auditing when the source player ID also appears in an FBS roster. No fuzzy team matching is used; other FCS rows remain preserved in the raw corpus but are omitted from this FBS-focused normalized panel.",
        "",
        f"Verified raw requests in this build: {inventory_hashes}/{len(source_inventory)}. Each response is stored byte-for-byte with query parameters, retrieval time, row count, and SHA-256 sidecar. The acquisition manifest contains no API credential.",
        "",
        "CFBD documents historical rosters from 2004 onward and notes that player and biographical field completeness varies by season and team ([data availability](https://apinext.collegefootballdata.com/data-availability), [roster endpoint schema](https://apinext.collegefootballdata.com/api/teams)).",
        "",
        "## FBS coverage",
        "",
        f"Across {expected_team_seasons:,} expected FBS team-seasons, {available_team_seasons:,} have at least one roster row mapped to the CFBD team ID; {missing_team_seasons:,} have a successful season response but no team roster rows; {no_ol_team_seasons:,} have roster rows but no explicitly identifiable OL position. FBS roster-response seasons absent from the local acquisition are: {response_missing_seasons or 'none'}.",
        "",
        f"Across {all_fbs_roster_rows:,} FBS roster rows, ID coverage is {100 * fbs_id_rate:.1f}%, name coverage is {100 * float(artifacts.source_audit['fbs_roster_player_name_rate']):.1f}%, nonblank position coverage is {100 * float(artifacts.source_audit['fbs_roster_position_rate']):.1f}%, and jersey-number coverage is {100 * float(artifacts.source_audit['fbs_roster_jersey_number_rate']):.1f}%. There are {fbs_duplicate_id_rows:,} repeated ID/team-season rows and {unmatched_team_rows:,} FBS roster rows that did not map to one historical FBS team ID. Name, position, and jersey coverage are reported for every season below.",
        "",
        _markdown_table(
            [
                "Season",
                "Expected FBS",
                "Roster mapped",
                "No identifiable OL",
                "FBS rows",
                "ID rate",
                "Name rate",
                "Position rate",
                "Jersey rate",
                "OL min",
                "OL median",
                "OL max",
                "OL ID rate",
            ],
            season_rows,
        ),
        "",
        "Rates use FBS roster rows as the denominator; position rate measures whether CFBD supplied a nonblank label, not whether that label is season-accurate. The complete counts and rates are in `coverage_by_season.csv`; each expected team-season and its missingness category are in `team_season_coverage.csv`.",
        "",
        "## Position normalization",
        "",
        "The source position is preserved in `position_original`. Explicit labels `OL`, `OT`/`T`, `OG`/`G`, `C`/`OC`, `LT`/`RT`, `LG`/`RG`, and `IOL` map to `offensive_line`. Known non-OL labels map to `non_offensive_line`. Compound labels containing separators map to `ambiguous`; blank or unrecognized labels map to `unknown`. Neither ambiguous nor unknown positions are silently counted as OL.",
        "",
        _markdown_table(["Source position", "Classification", "Rows"], position_rows),
        "",
        "The table includes FBS target rows and relevant FCS historical OL rows; filter `source_classification=fbs` for target-season cohorts.",
        "",
        f"Across retained FBS/FCS rows, {artifacts.source_audit['ambiguous_position_rows_all_classifications']:,} rows have compound ambiguous labels and {artifacts.source_audit['unknown_position_rows_all_classifications']:,} have blank or unrecognized labels; neither category is counted as OL.",
        "",
        "## Player identity audit",
        "",
        f"The normalized identity key is the CFBD source player ID (`cfbd:<id>`). The retained panel has {int(artifacts.source_audit['unique_source_player_ids']):,} unique IDs, including {fbs_unique_ids:,} in FBS target rosters; {fbs_multi_season_ids:,} FBS IDs occur in multiple seasons and {fbs_same_program_returns:,} have at least one adjacent-season same-program return. Across retained program rows, {multi_program_ids:,} IDs appear at multiple programs across seasons. Same-ID source name variants occur for {name_variants:,} IDs; position or position-label variants occur for {position_changes:,}; duplicate normalized names within a program-season occur in {duplicate_names:,} groups; exact name/team matches with changing adjacent-season IDs produce {possible_id_changes:,} unresolved candidate links and {ambiguous_name_links:,} ambiguous name-only links. None of the latter are merged automatically.",
        "",
        f"Same IDs appear on multiple schools in the same season in {same_year_multi_team:,} cases. These are visible retrospective transfer/multi-school membership signals, not evidence of preseason availability. See `player_identity_audit.csv.gz` and `player_identity_events.csv.gz` for row-level evidence.",
        "",
        "Returning players are linked by exact source ID within the same CFBD team ID. Position changes and source-name variants do not break an ID link, but are retained in the identity audit. Duplicate names with different IDs remain distinct. Missing IDs and possible ID changes remain unresolved.",
        "An identity position-variant count of zero means the provider label stayed the same for each observed ID; it does not show that the label was correct for each season. The Vanderbilt biography check below finds a historical backfill case without any within-ID label variation.",
        "",
        "## Shared-roster continuity construction",
        "",
        "For each target FBS season T, the current roster identifies the target OL pool. A pair's prior shared seasons are the intersection of its members' earlier same-program roster seasons, restricted to years `< T`. The pair table reports the number, earliest, most recent, and consecutive shared seasons immediately before T. Seasons at a different school never contribute. Target-season roster membership itself never contributes to a pair score.",
        "",
        "The team-season summaries report total/mean/maximum pairwise shared seasons; pair counts at 1/2/3 shared seasons; the largest set of target OL simultaneously present on one prior same-program roster; the share of ID-linked target OL with no prior same-program roster row observed in this panel; and aggregate prior roster seasons. These are descriptive candidates, not selected production features.",
        "",
        f"Season {start_season} is left-censored: the source panel begins in that year, so its prior continuity values are blank rather than zero. For later seasons, ‘no prior’ means no earlier same-program roster row was observed in the acquired panel; it does not prove that the player had never attended the school before {start_season}.",
        "",
        "## Retrospective roster boundary",
        "",
        f"CFBD returns a season roster, not an archived Week 1 snapshot with an as-of date. The latest FBS roster response in this corpus was retrieved at {latest_fbs_retrieval}; the 2026 response is in-season. Responses may reflect transfers or roster changes that happened during the season. The target-season roster is used only to define which players enter the pair pool; it is not treated as evidence that target-season players had already spent time together. Same-ID listings at multiple schools in one season are counted above as obvious temporal conflicts. Less visible midseason additions/departures cannot be detected from this endpoint alone. No games, starts, outcomes, or later starting-lineup knowledge enter the continuity measures.",
        "",
        "## Frozen issue 181 sample comparison",
        "",
        f"The builder joins all {len(issue181_overlap)} target team-seasons from the frozen issue 181 sample, copied from commit `{ISSUE_181_SAMPLE_COMMIT}`. The sample rows and strata are unchanged; joins use target season and normalized canonical team name, with the explicit `Appalachian State` → `App State` alias. The table counts selected team-seasons with at least one identified OL, with none, and the number of pairs with any shared prior same-school roster season. Row-level joins are in `issue_181_overlap_coverage.csv`.",
        "",
        _markdown_table(
            [
                "Window start",
                "Sample rows",
                "With identified OL",
                "No identified OL",
                "Pairs with shared prior season",
            ],
            issue181_rows,
        ),
        "",
        "## Bounded official-roster validation",
        "",
        validation_status,
        "",
        _markdown_table(
            [
                "Case",
                "Coverage",
                "School",
                "Seasons",
                "CFBD",
                "Official roster",
                "Discrepancy",
                "Evidence",
            ],
            validation_rows,
        ),
        "",
        "The validation register is a bounded manual sample, not a repair queue. Discrepancies are classified and counted; the panel is not manually patched to force agreement.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python scripts/fetch_offensive_line_roster_history.py --start-season 2004 --end-season 2026",
        "uv run python scripts/build_offensive_line_shared_roster_continuity.py --start-season 2004 --end-season 2026",
        "```",
        "",
        "The fetch command requires `CFBD_API_KEY`; reruns validate and reuse the byte-preserved responses rather than overwriting them. The builder is offline and checks every raw payload against its provenance hash before producing compressed CSVs. Gzip output timestamps are fixed so identical inputs produce identical artifacts.",
    ]
    return "\n".join(lines) + "\n"


def build_dataset(
    *,
    raw_root: Path,
    output_dir: Path,
    start_season: int,
    end_season: int,
) -> ContinuityBuild:
    roster_payloads, teams_by_season, roster_responses, source_inventory = (
        load_source_corpus(
            raw_root=raw_root,
            start_season=start_season,
            end_season=end_season,
        )
    )
    artifacts = build_continuity_artifacts(
        roster_payloads,
        teams_by_season,
        roster_response_seasons=roster_responses,
        start_season=start_season,
        end_season=end_season,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "normalized_roster_player_seasons.csv.gz": artifacts.player_seasons,
        "normalized_ol_player_seasons.csv.gz": artifacts.ol_player_seasons,
        "position_vocabulary.csv": artifacts.position_vocabulary,
        "player_identity_audit.csv.gz": artifacts.identity_audit,
        "player_identity_events.csv.gz": artifacts.identity_events,
        "team_season_coverage.csv": artifacts.team_season_coverage,
        "coverage_by_season.csv": artifacts.coverage_by_season,
        "pairwise_shared_roster_continuity.csv.gz": artifacts.pairwise_continuity,
        "team_season_summaries.csv": artifacts.team_season_summaries,
        "source_inventory.csv": source_inventory,
    }
    issue181_overlap = _build_issue181_overlap_coverage(
        artifacts, ISSUE_181_SAMPLE_PATH
    )
    outputs["issue_181_overlap_coverage.csv"] = issue181_overlap
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)
    (output_dir / "source_audit.json").write_text(
        json.dumps(artifacts.source_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manual_validation = _read_manual_validation(MANUAL_VALIDATION_PATH)
    report = _build_report(
        artifacts,
        source_inventory=source_inventory,
        manual_validation=manual_validation,
        issue181_overlap=issue181_overlap,
        start_season=start_season,
        end_season=end_season,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=datetime.now(UTC).year)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_dataset(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    print(
        f"Wrote {len(artifacts.player_seasons):,} roster rows, "
        f"{len(artifacts.pairwise_continuity):,} pair rows, and "
        f"{len(artifacts.team_season_summaries):,} team-season summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
