"""Audit whether transfer-production inputs are production-feasible.

The command reads immutable raw portal and player-usage responses, the
canonical FBS team-season table, and an optional explicit alias file.  It
writes only derived audit artifacts.  No production model input is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from gippyrank.transfer_audit import (
    audit_transfer_records,
    cutoff_safety_assessment,
    read_team_aliases,
    snapshot_strategy,
    source_inventory,
)
from gippyrank.transfer_oracle import (
    TransferRecord,
    UsageRecord,
    parse_transfer_payload,
    parse_usage_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data/raw/cfbd/preseason/transfers"
DEFAULT_TEAM_FILE = ROOT / "data/processed/preseason/team_season_features.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/transfer_production_audit"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_team_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_raw(
    raw_root: Path,
) -> tuple[
    list[TransferRecord], list[UsageRecord], set[int], set[int], list[dict[str, Any]]
]:
    records: list[TransferRecord] = []
    usage: list[UsageRecord] = []
    portal_seasons: set[int] = set()
    usage_seasons: set[int] = set()
    source_files: list[dict[str, Any]] = []
    for kind, parser, seasons in (
        ("portal", parse_transfer_payload, portal_season_paths(raw_root)),
        ("usage", parse_usage_payload, usage_season_paths(raw_root)),
    ):
        for path in seasons:
            season = int(path.stem)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"{path} is not a JSON array")
            parsed = parser(payload, season=season)
            if kind == "portal":
                records.extend(parsed)
                portal_seasons.add(season)
            else:
                usage.extend(parsed)
                usage_seasons.add(season)
            sidecar = path.with_name(f"{path.name}.provenance.json")
            source_files.append(
                {
                    "kind": kind,
                    "season": season,
                    "path": path.relative_to(raw_root.parent.parent.parent).as_posix(),
                    "sha256": _sha256(path),
                    "record_count": len(payload),
                    "provenance": (
                        json.loads(sidecar.read_text(encoding="utf-8"))
                        if sidecar.exists()
                        else None
                    ),
                }
            )
    if not records:
        raise FileNotFoundError(
            f"no portal JSON files found under {raw_root / 'portal'}"
        )
    if not usage:
        raise FileNotFoundError(f"no usage JSON files found under {raw_root / 'usage'}")
    return records, usage, portal_seasons, usage_seasons, source_files


def portal_season_paths(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (raw_root / "portal").glob("*.json")
        if path.stem.isdigit() and not path.name.endswith(".provenance.json")
    )


def usage_season_paths(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (raw_root / "usage").glob("*.json")
        if path.stem.isdigit() and not path.name.endswith(".provenance.json")
    )


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(
    rows: list[Mapping[str, Any]], columns: list[tuple[str, str]]
) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(key)).replace("|", "\\|") for key, _ in columns)
            + " |"
        )
    return lines


def _aggregate_team_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for season in sorted({int(row["season"]) for row in rows}):
        scoped = [row for row in rows if int(row["season"]) == season]
        result.append(
            {
                "season": season,
                "teams": len(scoped),
                "complete": sum(
                    row["feature_coverage_status"] == "complete" for row in scoped
                ),
                "partial": sum(
                    row["feature_coverage_status"] == "partial" for row in scoped
                ),
                "no_usable": sum(
                    row["feature_coverage_status"] == "no_usable_transfer_production"
                    for row in scoped
                ),
                "no_incoming": sum(
                    row["feature_coverage_status"] == "no_incoming_transfer"
                    for row in scoped
                ),
                "mean_transfers": (
                    sum(float(row["incoming_transfer_count"]) for row in scoped)
                    / len(scoped)
                    if scoped
                    else None
                ),
                "mean_observed_usage": (
                    sum(float(row["observed_incoming_prior_usage"]) for row in scoped)
                    / len(scoped)
                    if scoped
                    else None
                ),
                "missingness_rate": (
                    sum(int(row["missingness_flag"]) for row in scoped) / len(scoped)
                    if scoped
                    else None
                ),
            }
        )
    return result


def render_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    season_rows: list[Mapping[str, Any]],
    team_summary: list[Mapping[str, Any]],
    unmatched: list[Mapping[str, Any]],
    mapping_rows: list[Mapping[str, Any]],
    cutoff_rows: list[Mapping[str, Any]],
    inventory_rows: list[Mapping[str, Any]],
    strategy: Mapping[str, Any],
) -> None:
    requirements = [
        {
            "field": "transfer season",
            "model": "yes",
            "audit": "query and snapshot scope",
        },
        {
            "field": "player identity",
            "model": "yes",
            "audit": "name normalization, collisions, stable-ID test",
        },
        {
            "field": "source school",
            "model": "yes",
            "audit": "canonical team mapping and usage join key",
        },
        {
            "field": "destination school",
            "model": "yes",
            "audit": "canonical FBS destination and cutoff availability",
        },
        {
            "field": "transfer date / cutoff",
            "model": "yes",
            "audit": "season-relative inclusion rule",
        },
        {
            "field": "prior-season usage",
            "model": "yes",
            "audit": "exact, alias, missing, ambiguous, and usage-weighted coverage",
        },
        {
            "field": "position",
            "model": "only if representation needs it",
            "audit": "position-group coverage",
        },
        {
            "field": "rating / stars",
            "model": "no for C10; audit proxy only",
            "audit": "unmatched prioritization",
        },
        {
            "field": "team identity mapping",
            "model": "yes",
            "audit": "explicit aliases; no fuzzy matching",
        },
        {
            "field": "preseason cutoff semantics",
            "model": "yes",
            "audit": "source classification and snapshot policy",
        },
    ]
    lines = [
        "# Transfer-production data-quality and production-feasibility audit (issue 98)",
        "",
        "## Recommendation",
        "",
        (
            "**Feasible only with a new snapshot pipeline and explicit identity controls.** "
            "The audited CFBD responses contain useful prior-production signal, but they are retrospective endpoint responses. Final destinations, publication timing, and rating revisions are not proven as-of the historical preseason cutoff. The current portal payload also has no player identifier shared with `/player/usage`, so the production fallback must be deterministic name + source-team matching that fails closed on ambiguity."
        ),
        "",
        "The selected issue-91 representation is `total RP + incoming prior transfer production`; this audit therefore focuses on incoming destination-resolved transfers and prior-season `usage.overall`. It does not redesign or promote the Context model.",
        "",
        "## Exact data requirements",
        "",
        *_markdown_table(
            requirements,
            [
                ("field", "Field"),
                ("model", "Required for model"),
                ("audit", "Audit use"),
            ],
        ),
        "",
        "Fields such as rating, stars, and position are useful for diagnosing important unmatched cases but are not required by C10. Scholarship status is not available from the selected endpoints.",
        "",
        "## Player identity and join coverage",
        "",
        *_markdown_table(
            season_rows,
            [
                ("season", "Season"),
                ("total_portal_records", "Portal"),
                ("records_with_destination", "Destination"),
                ("records_on_or_before_cutoff", "On/before cutoff"),
                ("incoming_fbs_transfers", "Incoming FBS"),
                ("incoming_from_fbs_source", "FBS source"),
                (
                    "incoming_from_non_fbs_or_unrecognized_source",
                    "Non-FBS/unrecognized source",
                ),
                ("incoming_fbs_with_exact_successful_join", "Exact joins"),
                ("incoming_fbs_with_alias_successful_join", "Alias joins"),
                ("failed_prior_usage_joins", "Failed joins"),
                ("ambiguous_joins", "Ambiguous"),
                ("normalization_changed_matches", "Normalization rescues"),
                ("portal_normalized_name_collision_keys", "Portal collisions"),
                ("usage_weighted_join_coverage_proxy", "Usage-weighted proxy"),
            ],
        ),
        "",
        "A raw record-count join rate is not enough. The usage-weighted proxy is `recoverable_unique_usage_mass / any_name_usage_mass`: the denominator is the unique prior-usage mass whose normalized player name appears in at least one in-scope transfer, while the numerator additionally requires a unique source-team/player join. It is the strongest reproducible identity-resolution proxy available from these endpoints, not a full-population denominator, because usage for a completely unmatched player is unobserved.",
        "",
        "## High-value unmatched transfers",
        "",
        "The full machine-readable list is `unmatched_high_value_transfers.csv`. Ranking uses rating, then stars, then QB status; it is a diagnostic ordering, not a model feature.",
        "",
        *_markdown_table(
            unmatched[:25],
            [
                ("priority_rank", "Rank"),
                ("season", "Season"),
                ("player_name", "Player"),
                ("origin", "Origin"),
                ("destination", "Destination"),
                ("position", "Pos"),
                ("rating", "Rating"),
                ("stars", "Stars"),
                ("usage_join_status", "Failure"),
            ],
        ),
        "",
        "Failure classes distinguish missing usage rows, source-team mismatches, usage rows without a numeric value, and ambiguous normalized joins. No fuzzy player match is applied.",
        "",
        "## Team identity audit",
        "",
        f"The complete mapping inventory is `team_mapping_by_season.csv`. {sum(row['match_method'] == 'explicit_alias' for row in mapping_rows)} encountered names were resolved through the supplied explicit alias table; names not in the FBS model population remain visible with a failure method rather than being silently coerced.",
        "",
        *_markdown_table(
            [
                row
                for row in mapping_rows
                if row["match_method"] != "exact_normalized_name"
            ][:30],
            [
                ("season", "Season"),
                ("field", "Field"),
                ("raw_name", "Name"),
                ("match_method", "Result"),
                ("record_count", "Records"),
            ],
        ),
        "",
        "## Cutoff safety and historical destination limitation",
        "",
        *_markdown_table(
            cutoff_rows,
            [
                ("field", "Field"),
                ("required_for_model", "Model"),
                ("classification", "Classification"),
                ("evidence", "Evidence"),
            ],
        ),
        "",
        "A transfer date on or before August 15 proves only that the current response carries an early event date. It does not prove that the destination stored today was known, published, or stable by August 15 in the historical year. The audit therefore keeps destination filtering deterministic while classifying destination as retrospective-oracle-only.",
        "",
        "## Source inventory",
        "",
        *_markdown_table(
            inventory_rows,
            [
                ("source", "Source"),
                ("player_identifier", "Player ID"),
                ("timestamp_semantics", "Timestamp semantics"),
                ("production_assessment", "Assessment"),
            ],
        ),
        "",
        "## Current/future snapshot strategy",
        "",
        *[f"{index}. {step}" for index, step in enumerate(strategy["steps"], start=1)],
        "",
        str(strategy["integration"]),
        "",
        "## Team-level feature coverage and unresolved-join sensitivity",
        "",
        *_markdown_table(
            team_summary,
            [
                ("season", "Season"),
                ("teams", "Teams"),
                ("complete", "Complete"),
                ("partial", "Partial"),
                ("no_usable", "No usable"),
                ("no_incoming", "No incoming"),
                ("mean_transfers", "Mean transfers"),
                ("mean_observed_usage", "Mean joined usage"),
                ("missingness_rate", "Missingness"),
            ],
        ),
        "",
        "For each team-season, `unmatched_prior_usage_upper_bound` is the observed usage plus the unmatched incoming count multiplied by the maximum numeric overall usage in the prior usage payload. This is intentionally conservative and not imputed into the model. The complete team-season table is `team_feature_coverage.csv`.",
        "",
        "Conference-stratified missingness is not reported as a numeric result because the canonical team-season feature table has no season-specific conference column. Competition level is FBS destination only; source level remains visible through mapping and join failure classes. Transfer-volume and roster-strength correlations are descriptive in `missingness_summary.json`.",
        "",
        "## Stable-player-ID assessment",
        "",
        f"{summary['stable_player_id']['limitation']}. The usage endpoint supplies IDs for {summary['stable_player_id']['usage_records_with_stable_id']} of {summary['stable_player_id']['usage_records']} usage records, but the portal endpoint supplies none. The recommended fallback is therefore explicit normalized name + source team, with ambiguity surfaced and excluded.",
        "",
        "## Reproducibility and artifacts",
        "",
        "Raw responses remain unchanged and ignored. `source_manifest.json` records the input paths, response hashes, record counts, and fetch sidecars. Derived artifacts are deterministic given those inputs, the canonical team table, the explicit alias file, and the configured cutoff.",
        "",
        "- `audit_summary.json` — machine-readable conclusion, inputs, coverage, and stable-ID assessment.",
        "- `player_join_coverage_by_season.csv` — record-count and usage-weighted join coverage.",
        "- `unmatched_high_value_transfers.csv` — all unmatched in-scope transfers in diagnostic priority order.",
        "- `team_mapping_by_season.csv` — every encountered origin/destination name and deterministic resolution.",
        "- `team_feature_coverage.csv` — team-season completeness and unresolved-production bounds.",
        "- `player_join_records.csv` — row-level explanation for every portal record.",
        "- `cutoff_safety.json`, `source_inventory.json`, `snapshot_strategy.json`, `missingness_summary.json` — supporting audit decisions.",
        "",
        "## Acceptance conclusion",
        "",
        "The signal is not production-safe from retrospective CFBD portal responses alone. It is a credible future production candidate only after (1) capturing immutable preseason snapshots before the cutoff, (2) retaining explicit endpoint provenance, and (3) adding stable cross-endpoint player identity or keeping the fail-closed name + source-team audit with manual resolution of high-value unmatched cases.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    team_file: Path = DEFAULT_TEAM_FILE,
    aliases_file: Path | None = None,
    output: Path = DEFAULT_OUTPUT,
    cutoff_month: int = 8,
    cutoff_day: int = 15,
) -> dict[str, Any]:
    records, usage, portal_seasons, usage_seasons, source_files = _load_raw(raw_root)
    team_rows = _load_team_rows(team_file)
    scoped_team_rows = [
        row for row in team_rows if int(row["season"]) in portal_seasons
    ]
    aliases = read_team_aliases(aliases_file)
    cutoff = date(2025, cutoff_month, cutoff_day)
    audit = audit_transfer_records(
        records,
        usage,
        scoped_team_rows,
        cutoff=cutoff,
        aliases=aliases,
    )
    cutoff_rows = cutoff_safety_assessment()
    inventory_rows = source_inventory()
    strategy = snapshot_strategy()
    team_summary = _aggregate_team_rows(audit["team_feature_rows"])
    summary: dict[str, Any] = {
        "audit": "issue_98_transfer_production_data_quality",
        "production_models_modified": False,
        "raw_inputs_unchanged": True,
        "raw_root": _display_path(raw_root),
        "team_file": _display_path(team_file),
        "aliases_file": _display_path(aliases_file) if aliases_file else None,
        "cutoff": f"season-relative {cutoff_month:02d}-{cutoff_day:02d}",
        "portal_seasons": sorted(portal_seasons),
        "usage_seasons": sorted(usage_seasons),
        "portal_record_count": len(records),
        "usage_record_count": len(usage),
        "explicit_alias_count": len(aliases),
        "field_requirements": cutoff_rows,
        "source_inventory": inventory_rows,
        "snapshot_strategy": strategy,
        "stable_player_id": audit["stable_player_id"],
        "recommendation": "feasible only with a new snapshot pipeline and additional identity controls",
        "season_rows": audit["season_rows"],
        "team_summary": team_summary,
        "missingness": audit["missingness"],
        "collision_summary": {
            "portal": audit["portal_normalized_name_collisions"],
            "usage": audit["usage_normalized_name_collisions"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "audit_summary.json", summary)
    _write_json(output / "cutoff_safety.json", cutoff_rows)
    _write_json(output / "source_inventory.json", inventory_rows)
    _write_json(output / "snapshot_strategy.json", strategy)
    _write_json(output / "missingness_summary.json", audit["missingness"])
    _write_json(output / "source_manifest.json", {"files": source_files})
    _write_csv(output / "player_join_coverage_by_season.csv", audit["season_rows"])
    _write_csv(output / "unmatched_high_value_transfers.csv", audit["unmatched_rows"])
    _write_csv(output / "team_mapping_by_season.csv", audit["team_mapping_rows"])
    _write_csv(output / "team_feature_coverage.csv", audit["team_feature_rows"])
    _write_csv(output / "player_join_records.csv", audit["join_rows"])
    render_report(
        output / "report.md",
        summary=summary,
        season_rows=audit["season_rows"],
        team_summary=team_summary,
        unmatched=audit["unmatched_rows"],
        mapping_rows=audit["team_mapping_rows"],
        cutoff_rows=cutoff_rows,
        inventory_rows=inventory_rows,
        strategy=strategy,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--team-file", type=Path, default=DEFAULT_TEAM_FILE)
    parser.add_argument("--aliases", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff-month", type=int, default=8)
    parser.add_argument("--cutoff-day", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        raw_root=args.raw_root,
        team_file=args.team_file,
        aliases_file=args.aliases,
        output=args.output,
        cutoff_month=args.cutoff_month,
        cutoff_day=args.cutoff_day,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "portal_seasons": summary["portal_seasons"],
                "recommendation": summary["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
