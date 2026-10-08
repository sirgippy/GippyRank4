"""Build issue 185's offline CFBD-to-official OL roster comparison.

The 40 official OL pools are transcribed independently from the linked official
sources. The builder compares those pools with the cached CFBD candidates; it
does not construct the official pool from CFBD names. It reads local inputs
only and does not download or mutate raw source data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data/research/offensive_line_target_pool_validation_issue_185"
PANEL = ROOT / "data/processed/offensive_line_shared_roster_issue_183"
OUTPUT = RESEARCH / "results"
SAMPLE_PATH = RESEARCH / "frozen_sample.csv"
OFFICIAL_SOURCES_PATH = RESEARCH / "official_source_inventory.csv"
OFFICIAL_ROSTERS_PATH = RESEARCH / "official_ol_rosters.csv"
REVIEWS_PATH = RESEARCH / "cfbd_candidate_reviews.csv"
CROSSWALK_PATH = RESEARCH / "identity_crosswalk.csv"
CFBD_PATH = PANEL / "normalized_roster_player_seasons.csv.gz"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_count(size: int) -> int:
    return math.comb(size, 2) if size >= 2 else 0


def load_cfbd() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, list[dict[str, str]]]]]:
    ol_by_key: dict[str, dict[str, str]] = defaultdict(dict)
    roster_by_key: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with gzip.open(CFBD_PATH, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_classification"] != "fbs":
                continue
            key = f"{row['season']}|{row['team_id']}"
            name_key = normalize(row["player_name"])
            if name_key:
                roster_by_key[key][name_key].append(row)
                if row["normalized_ol_status"] == "offensive_line":
                    ol_by_key[key].setdefault(name_key, row["player_name"])
    return ol_by_key, roster_by_key


def build(raw_root: Path, *, check: bool = False) -> None:
    sample = read_csv(SAMPLE_PATH)
    sources = {row["sample_id"]: row for row in read_csv(OFFICIAL_SOURCES_PATH)}
    official_rosters = read_csv(OFFICIAL_ROSTERS_PATH)
    reviews = read_csv(REVIEWS_PATH)
    aliases = read_csv(CROSSWALK_PATH)
    cfbd, cfbd_roster = load_cfbd()
    sample_by_id = {row["sample_id"]: row for row in sample}
    if len(sample) != 40 or set(sources) != set(sample_by_id):
        raise SystemExit("Expected 40 frozen sample rows and one official source per sample")

    official_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in official_rosters:
        sample_id = row["sample_id"]
        if sample_id not in sample_by_id:
            raise SystemExit(f"Official roster row references an unknown sample: {sample_id}")
        if row["official_position_class"] != "offensive_line":
            raise SystemExit(f"Non-OL row in independent official pool: {sample_id}/{row['official_player_name']}")
        if row["source_url"] != sources[sample_id]["source_url"]:
            raise SystemExit(f"Official roster source URL mismatch for {sample_id}")
        name_key = normalize(row["official_player_name"])
        if not name_key or name_key in official_by_id[sample_id]:
            raise SystemExit(f"Blank or duplicate official player for {sample_id}: {row['official_player_name']}")
        official_by_id[sample_id][name_key] = row

    reviews_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    aliases_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviews:
        name_key = normalize(row["cfbd_player_name"])
        if row["sample_id"] not in sample_by_id or name_key in reviews_by_id[row["sample_id"]]:
            raise SystemExit(f"Unknown or duplicate CFBD candidate review: {row['sample_id']}/{row['cfbd_player_name']}")
        if row["decision"] not in {"cfbd_only", "unresolved"}:
            raise SystemExit(f"Unknown CFBD candidate review decision: {row['decision']}")
        reviews_by_id[row["sample_id"]][name_key] = row
    for row in aliases:
        aliases_by_id[row["sample_id"]].append(row)

    if set(official_by_id) != set(sample_by_id):
        missing = sorted(set(sample_by_id) - set(official_by_id))
        raise SystemExit(f"Independent official OL pool missing sample rows: {missing}")

    player_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    mismatch_counter: Counter[tuple[str, str, str]] = Counter()
    era_acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample_row in sample:
        sample_id = sample_row["sample_id"]
        cfbd_key = f"{sample_row['season']}|{sample_row['team_id']}"
        cfbd_names = cfbd.get(cfbd_key, {})
        frozen_count = int(sample_row["cfbd_ol_count"])
        if len(cfbd_names) != frozen_count:
            raise SystemExit(
                f"CFBD pool count changed for {sample_id}: cache={len(cfbd_names)} frozen={frozen_count}"
            )
        candidate_name_by_key = dict(cfbd_names)
        official_records = official_by_id[sample_id]
        review_records = reviews_by_id[sample_id]

        for alias in aliases_by_id[sample_id]:
            cfbd_name = alias["cfbd_player_name"]
            official_name = alias["official_player_name"]
            ckey, okey = normalize(cfbd_name), normalize(official_name)
            if ckey not in candidate_name_by_key:
                raise SystemExit(f"Crosswalk candidate missing for {sample_id}: {cfbd_name}")
            if okey in candidate_name_by_key and okey != ckey:
                raise SystemExit(f"Crosswalk target collides with a candidate for {sample_id}: {official_name}")
            candidate_name_by_key[okey] = candidate_name_by_key.pop(ckey)

        for review_key, review in review_records.items():
            if review_key not in candidate_name_by_key:
                raise SystemExit(f"Reviewed CFBD candidate missing for {sample_id}: {review['cfbd_player_name']}")
            if review_key in official_records:
                raise SystemExit(f"Reviewed non-match is in official OL pool: {sample_id}/{review['cfbd_player_name']}")

        cfbd_keys = set(candidate_name_by_key)
        official_keys = set(official_records)
        overlap = cfbd_keys & official_keys
        only_cfbd = cfbd_keys - official_keys
        only_official = official_keys - cfbd_keys
        unresolved_keys = {key for key, review in review_records.items() if review["decision"] == "unresolved"}
        if only_cfbd != set(review_records):
            unreviewed = sorted(only_cfbd - set(review_records))
            unexpected = sorted(set(review_records) - only_cfbd)
            raise SystemExit(f"CFBD-only roster decisions incomplete for {sample_id}: unreviewed={unreviewed}, unexpected={unexpected}")
        if not overlap and only_cfbd and only_official and sample_id != "S03":
            raise SystemExit(f"Unexpectedly empty player overlap for {sample_id}")

        for name_key in sorted(cfbd_keys | official_keys):
            in_cfbd, in_official = name_key in cfbd_keys, name_key in official_keys
            cfbd_name = candidate_name_by_key.get(name_key, "")
            official_record = official_records.get(name_key, {})
            official_name = official_record.get("official_player_name", "")
            roster_hits = cfbd_roster.get(cfbd_key, {}).get(name_key, [])
            roster_position = roster_hits[0]["position_original"] if roster_hits else ""
            if in_cfbd:
                roster_presence = "present_as_cfbd_ol"
            elif not roster_hits:
                roster_presence = "absent_from_roster"
            elif all(
                not hit["position_original"].strip() or hit["normalized_ol_status"] == "unknown"
                for hit in roster_hits
            ):
                roster_presence = "present_with_unknown_or_blank_position"
            else:
                roster_presence = "present_at_another_position"
            if in_cfbd and in_official:
                status = "matched"
                category = "exact_or_crosswalk_name_match"
                note = ""
            elif in_cfbd and review_records[name_key]["decision"] == "unresolved":
                status = "unresolved"
                category = review_records[name_key]["mismatch_category"]
                note = review_records[name_key]["review_note"]
            elif in_cfbd:
                status = "cfbd_only"
                category = review_records[name_key]["mismatch_category"]
                note = review_records[name_key]["review_note"]
            else:
                status = "official_only"
                category = "official_ol_missing_from_cfbd_ol_pool"
                note = f"Independent official OL transcription; CFBD roster: {roster_presence}"
            player_rows.append(
                {
                    "sample_id": sample_id,
                    "season": sample_row["season"],
                    "team_name": sample_row["team_name"],
                    "cfbd_player_name": cfbd_name,
                    "official_player_name": official_name,
                    "cfbd_ol": str(in_cfbd).lower(),
                    "official_ol": str(in_official).lower(),
                    "comparison_status": status,
                    "mismatch_category": category,
                    "cfbd_roster_presence": roster_presence,
                    "cfbd_roster_player_name": roster_hits[0]["player_name"] if roster_hits else "",
                    "cfbd_roster_position_original": roster_position,
                    "review_note": note,
                    "source_url": sources[sample_id]["source_url"],
                }
            )
            if status in {"cfbd_only", "official_only", "unresolved"}:
                mismatch_counter[(status, category, roster_presence)] += 1

        tp = len(overlap)
        fp = len(only_cfbd - unresolved_keys)
        fn = len(only_official)
        resolved_predicted = tp + fp
        precision = tp / resolved_predicted if resolved_predicted else None
        recall = tp / len(official_keys) if official_keys else None
        exact_player_set = not fp and not fn and not unresolved_keys
        cfbd_pair_pool = pair_count(resolved_predicted)
        official_pair_pool = pair_count(len(official_keys))
        shared_pairs = pair_count(tp)
        row = {
            "sample_id": sample_id,
            "era": sample_row["era"],
            "season": sample_row["season"],
            "team_name": sample_row["team_name"],
            "conference": sample_row["conference"],
            "cfbd_ol_count": len(cfbd_keys),
            "official_ol_count": len(official_keys),
            "true_positive_names": tp,
            "cfbd_only_false_positive_names": fp,
            "official_only_false_negative_names": fn,
            "unresolved_cfbd_names": len(unresolved_keys),
            "player_mismatch_count": fp + fn + len(unresolved_keys),
            "precision_resolved": f"{precision:.6f}" if precision is not None else "",
            "recall_resolved": f"{recall:.6f}" if recall is not None else "",
            "absolute_count_difference": abs(len(cfbd_keys) - len(official_keys)),
            "exact_count_match": str(len(cfbd_keys) == len(official_keys)).lower(),
            "exact_player_set_match": str(exact_player_set).lower(),
            "candidate_pair_pool_resolved": cfbd_pair_pool,
            "official_pair_pool": official_pair_pool,
            "shared_pairs_if_any_two_pool_players_cooccur": shared_pairs,
            "cfbd_only_pairs_if_any_two_pool_players_cooccur": cfbd_pair_pool - shared_pairs,
            "missed_official_pairs_if_any_two_pool_players_cooccur": official_pair_pool - shared_pairs,
            "source_url": sources[sample_id]["source_url"],
        }
        team_rows.append(row)
        source_rows.append(
            {
                **sources[sample_id],
                "sample_season": sample_row["season"],
                "team_name": sample_row["team_name"],
                "cfbd_ol_count": len(cfbd_keys),
                "official_ol_count": len(official_keys),
                "official_count_basis": "Independent official roster transcription; see official_ol_rosters.csv",
            }
        )
        era = sample_row["era"]
        for key, value in {
            "samples": 1,
            "cfbd": len(cfbd_keys),
            "official": len(official_keys),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "unresolved": len(unresolved_keys),
            "exact_count": int(len(cfbd_keys) == len(official_keys)),
            "exact_set": int(exact_player_set),
            "mismatched_team_season": int(fp + fn + len(unresolved_keys) > 0),
            "one_player_mismatch_team_season": int(fp + fn + len(unresolved_keys) == 1),
            "multiple_player_mismatch_team_season": int(fp + fn + len(unresolved_keys) > 1),
            "absolute_count_difference": abs(len(cfbd_keys) - len(official_keys)),
            "candidate_pairs": cfbd_pair_pool,
            "official_pairs": official_pair_pool,
            "shared_pairs": shared_pairs,
        }.items():
            era_acc[era][key] += value

    era_rows: list[dict[str, object]] = []
    for era, values in sorted(era_acc.items()):
        denominator_predicted = values["tp"] + values["fp"]
        precision = values["tp"] / denominator_predicted if denominator_predicted else None
        recall = values["tp"] / values["official"] if values["official"] else None
        era_rows.append(
            {
                "era": era,
                "sampled_team_seasons": values["samples"],
                "cfbd_ol_players": values["cfbd"],
                "official_ol_players": values["official"],
                "matched_players": values["tp"],
                "cfbd_only_false_positives": values["fp"],
                "official_only_false_negatives": values["fn"],
                "unresolved_cfbd_players": values["unresolved"],
                "team_seasons_with_mismatch": values["mismatched_team_season"],
                "team_season_mismatch_rate": f"{values['mismatched_team_season'] / values['samples']:.6f}",
                "team_seasons_with_one_player_mismatch": values["one_player_mismatch_team_season"],
                "team_seasons_with_multiple_player_mismatches": values["multiple_player_mismatch_team_season"],
                "micro_precision_resolved": f"{precision:.6f}" if precision is not None else "",
                "micro_recall_resolved": f"{recall:.6f}" if recall is not None else "",
                "exact_count_match_team_seasons": values["exact_count"],
                "exact_player_set_match_team_seasons": values["exact_set"],
                "absolute_count_difference_total": values["absolute_count_difference"],
                "resolved_cfbd_pair_pool": values["candidate_pairs"],
                "official_pair_pool": values["official_pairs"],
                "shared_pair_pool": values["shared_pairs"],
                "cfbd_only_pair_pool": values["candidate_pairs"] - values["shared_pairs"],
                "missed_official_pair_pool": values["official_pairs"] - values["shared_pairs"],
            }
        )

    taxonomy_rows = [
        {
            "comparison_status": status,
            "mismatch_category": category,
            "cfbd_roster_presence": roster_presence,
            "player_rows": count,
        }
        for (status, category, roster_presence), count in sorted(mismatch_counter.items())
    ]
    input_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            SAMPLE_PATH,
            OFFICIAL_SOURCES_PATH,
            OFFICIAL_ROSTERS_PATH,
            REVIEWS_PATH,
            CROSSWALK_PATH,
            CFBD_PATH,
        )
    }
    report_path = OUTPUT / "report.md"
    report = render_report(team_rows, era_rows, taxonomy_rows, input_hashes)
    if check:
        expected_files = {
            OUTPUT / "cfbd_ol_candidates.csv": serialize_candidate_rows(sample, cfbd),
            OUTPUT / "player_comparison.csv": serialize_table(player_rows),
            OUTPUT / "team_season_summary.csv": serialize_table(team_rows),
            OUTPUT / "era_metrics.csv": serialize_table(era_rows),
            OUTPUT / "mismatch_taxonomy.csv": serialize_table(taxonomy_rows),
            OUTPUT / "source_inventory.csv": serialize_table(source_rows),
            OUTPUT / "build_manifest.json": json.dumps({"input_sha256": input_hashes}, indent=2, sort_keys=True) + "\n",
            report_path: report,
        }
        for path, expected in expected_files.items():
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Generated output is missing or stale: {path.relative_to(ROOT)}")
        print(f"Verified {len(sample)} frozen team-seasons and generated comparison outputs")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "cfbd_ol_candidates.csv", candidate_rows(sample, cfbd), [
        "sample_id", "season", "team_name", "player_name", "position_original", "position_normalized",
    ])
    write_csv(OUTPUT / "player_comparison.csv", player_rows, [
        "sample_id", "season", "team_name", "cfbd_player_name", "official_player_name", "cfbd_ol", "official_ol", "comparison_status", "mismatch_category", "cfbd_roster_presence", "cfbd_roster_player_name", "cfbd_roster_position_original", "review_note", "source_url",
    ])
    write_csv(OUTPUT / "team_season_summary.csv", team_rows, list(team_rows[0]))
    write_csv(OUTPUT / "era_metrics.csv", era_rows, list(era_rows[0]))
    write_csv(OUTPUT / "mismatch_taxonomy.csv", taxonomy_rows, ["comparison_status", "mismatch_category", "cfbd_roster_presence", "player_rows"])
    write_csv(OUTPUT / "source_inventory.csv", source_rows, list(source_rows[0]))
    (OUTPUT / "build_manifest.json").write_text(
        json.dumps({"input_sha256": input_hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Built comparison outputs for {len(sample)} frozen team-seasons in {OUTPUT.relative_to(ROOT)}")


def candidate_rows(sample: list[dict[str, str]], cfbd: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    position_by_key: dict[str, tuple[str, str]] = {}
    with gzip.open(CFBD_PATH, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_classification"] == "fbs" and row["normalized_ol_status"] == "offensive_line":
                position_by_key[f"{row['season']}|{row['team_id']}|{normalize(row['player_name'])}"] = (
                    row["position_original"], row["position_normalized"]
                )
    out = []
    for team in sample:
        key = f"{team['season']}|{team['team_id']}"
        for player_name in sorted(cfbd.get(key, {}).values(), key=normalize):
            pos = position_by_key.get(f"{key}|{normalize(player_name)}", ("", ""))
            out.append({"sample_id": team["sample_id"], "season": team["season"], "team_name": team["team_name"], "player_name": player_name, "position_original": pos[0], "position_normalized": pos[1]})
    return out


def serialize_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def serialize_candidate_rows(sample: list[dict[str, str]], cfbd: dict[str, dict[str, str]]) -> str:
    return serialize_table(candidate_rows(sample, cfbd))


def render_report(
    team_rows: list[dict[str, object]],
    era_rows: list[dict[str, object]],
    taxonomy_rows: list[dict[str, object]],
    hashes: dict[str, str],
) -> str:
    matched = sum(int(row["true_positive_names"]) for row in team_rows)
    false_positive = sum(int(row["cfbd_only_false_positive_names"]) for row in team_rows)
    false_negative = sum(int(row["official_only_false_negative_names"]) for row in team_rows)
    unresolved = sum(int(row["unresolved_cfbd_names"]) for row in team_rows)
    cfbd_total = sum(int(row["cfbd_ol_count"]) for row in team_rows)
    official_total = sum(int(row["official_ol_count"]) for row in team_rows)
    precision = matched / (matched + false_positive) if matched + false_positive else 0
    recall = matched / official_total if official_total else 0
    exact_count = sum(row["exact_count_match"] == "true" for row in team_rows)
    exact_sets = sum(row["exact_player_set_match"] == "true" for row in team_rows)
    one_mismatch = sum(int(row["player_mismatch_count"]) == 1 for row in team_rows)
    multiple_mismatches = sum(int(row["player_mismatch_count"]) > 1 for row in team_rows)
    mismatched_teams = one_mismatch + multiple_mismatches
    absolute_differences = Counter(int(row["absolute_count_difference"]) for row in team_rows)
    candidate_pairs = sum(int(row["candidate_pair_pool_resolved"]) for row in team_rows)
    official_pairs = sum(int(row["official_pair_pool"]) for row in team_rows)
    shared_pairs = sum(int(row["shared_pairs_if_any_two_pool_players_cooccur"]) for row in team_rows)
    unsupported_pairs = candidate_pairs - shared_pairs
    missed_pairs = official_pairs - shared_pairs
    official_only_presence = Counter()
    for row in taxonomy_rows:
        if row["comparison_status"] == "official_only":
            official_only_presence[row["cfbd_roster_presence"]] += int(row["player_rows"])
    difference_summary = ", ".join(
        f"{difference}: {count}" for difference, count in sorted(absolute_differences.items())
    )
    presence_summary = ", ".join(
        f"{category}: {count}"
        for category, count in sorted(official_only_presence.items())
    )
    report = [
        "# Issue 185: CFBD offensive-line target-pool validation",
        "",
        "## Scope and frozen sample",
        "",
        "This bounded study compares the target-season offensive-line (OL) pools for the same 40 frozen FBS team-seasons. The sample remains ten team-seasons in each of four windows: 2009–2012, 2013–2016, 2017–2020, and 2021–2026. The frozen draw includes 2011 Idaho with zero CFBD OL labels and 2025 Army with 36 candidates to expose both observed tails. No new team-seasons were added.",
        "",
        "The sample and protocol were frozen on 2026-10-07 before official roster inspection. The draw uses 3 lower-quartile, 4 middle-half, and 3 upper-quartile CFBD-count rows per window, balances roster-size quartiles, and includes at least eight conference labels per window. The sample and freeze metadata remain unchanged.",
        "",
        "## Independent source comparison",
        "",
        "`official_ol_rosters.csv` records the OL names transcribed from the single official season roster or roster guide linked for each sample row. That set is read independently by the builder; CFBD names do not initialize it. The two pools are reconciled only after both are loaded. `cfbd_candidate_reviews.csv` records CFBD candidates explicitly excluded by the official source or retained as unresolved, and `identity_crosswalk.csv` documents the Miami Feliciano name variation. `player_comparison.csv` includes each player's official membership, CFBD OL membership, source URL, and underlying CFBD roster position/status.",
        "",
        "Most official sources are retrospective season archive pages; the source inventory records timing and limitations. Their current contents do not always establish the exact date of the roster snapshot. Central Michigan 2009 has a roster-guide PDF; Northwestern 2016 and Florida State 2019 use postseason or media-guide material. This remains a single-review manual study, not a generalized roster scraper or a double-coded audit. No source corpus, CFBD raw response, production model, or Context data was changed.",
        "",
        "## Player and team-season results",
        "",
        f"The 40 official pools contain {official_total} players; CFBD supplies {cfbd_total} OL candidates. The reconciliation finds {matched} matched players, {false_positive} resolved CFBD-only players, {false_negative} official-only players, and {unresolved} unresolved CFBD candidates. Resolved micro-precision is {precision:.1%}; micro-recall against the independent official pools is {recall:.1%}. Exact roster counts match in {exact_count}/40 team-seasons, and exact player sets match in {exact_sets}/40. {mismatched_teams}/40 team-seasons have at least one player discrepancy: {one_mismatch} with one and {multiple_mismatches} with multiple. The four unresolved cases are excluded from precision and counted as mismatches for exact-set and team-season reporting.",
        "",
        f"Absolute roster-count differences (difference: team-seasons) are {difference_summary}.",
        "",
        "### Era metrics",
        "",
        "| Era | Teams | CFBD OL | Official OL | Matched | CFBD-only | Official-only | Unresolved | Precision | Recall | Mismatched teams | Mismatch rate | Exact counts | Exact sets |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in era_rows:
        report.append(
            f"| {row['era']} | {row['sampled_team_seasons']} | {row['cfbd_ol_players']} | {row['official_ol_players']} | {row['matched_players']} | {row['cfbd_only_false_positives']} | {row['official_only_false_negatives']} | {row['unresolved_cfbd_players']} | {float(row['micro_precision_resolved']):.1%} | {float(row['micro_recall_resolved']):.1%} | {row['team_seasons_with_mismatch']} | {float(row['team_season_mismatch_rate']):.1%} | {row['exact_count_match_team_seasons']} | {row['exact_player_set_match_team_seasons']} |"
        )
    report.extend(
        [
            "",
            "## Official-only omission mechanisms",
            "",
            f"The {false_negative} official-only names classify against the underlying same-team, same-season FBS CFBD roster as follows: {presence_summary}. A player present at another position is distinct from a player with a blank/unknown position, and both differ from a name absent from the CFBD roster entirely. These counts are in the player table and mismatch taxonomy.",
            "",
            "The clearest coverage failure is Idaho 2011: the independent official pool has 18 OL players while the CFBD OL pool is empty. Seventeen official players, including Spencer Beale, Dallas Sandberg, A.J. Jones, and Sam Tupua, are absent from the CFBD roster entirely; Matt Cleveland is present with a blank/unknown position. The official roster table and player details list all four additional players at OL. The 153 possible within-pool pairs (18 choose 2) are arithmetic combinations only and do not establish actual prior shared-roster relationships.",
            "",
            "South Carolina 2019 has 19 official OL players and 17 CFBD OL candidates. Will Rogers and M.J. Webb are both official OL but appear in the CFBD roster at DL, so they are official-only with `present_at_another_position`; the former 17/17 set match was incorrect. The official pool has 171 possible pairs, of which the 17 matched players cover 136, leaving 35 missed pairs. Iowa State 2009 also changes despite an unchanged count: Carter Bykowski is listed at TE, while official OL Mike Knapp is absent from the CFBD roster.",
            "",
            "Other confirmed CFBD candidates at non-OL positions include UConn's Andreas Knappe (DL) and Minnesota's Ernie Heifort (TE); CFBD candidates Rennick Bryan (UConn) and Chris Freeman (Missouri) are not listed in their linked official roster sources. These cases show why a candidate-seeded gold set cannot detect omissions or false inclusions by itself.",
            "",
            "## Descriptive pair-pool implications",
            "",
            f"Summed over the 40 team-seasons, the resolved CFBD candidate pools imply {candidate_pairs} possible within-pool pairs, the official pools imply {official_pairs}, and the matched names imply {shared_pairs} shared pairs. This leaves {unsupported_pairs} candidate-only pairs and {missed_pairs} missed official pairs. These sums are descriptive n-choose-2 roster-pool combinations; they do not identify actual prior shared-roster relationships, actual co-occurrence or starts, or the predictive value of a continuity feature. Team-level pair-pool differences are available in `team_season_summary.csv`.",
            "",
            "## Recommendation: Proceed with bounded repair",
            "",
            "### Exact roster reconstruction",
            "",
            f"CFBD alone cannot guarantee a complete, exact official OL cohort: {false_negative} official OL are missing from its OL candidate sets, including {official_only_presence['absent_from_roster']} names absent from the underlying CFBD roster, and {mismatched_teams}/40 sampled team-seasons have at least one discrepancy. Idaho has an empty CFBD OL pool against 18 official OL. The frozen sample is stratified rather than prevalence-weighted, so these rates describe this sample and should not be read as population estimates.",
            "",
            "### Exploratory continuity modeling",
            "",
            "Proceed with the existing CFBD-derived shared-roster continuity data for exploratory feature development and empirical evaluation, retaining explicit coverage and uncertainty limitations. The bounded handling is to mark unavailable or empty team-season pools as uncovered and exclude clearly unusable team-seasons from the feature calculation, while preserving unknown-position and roster-missingness information and reporting coverage. These guards will not detect every partial omission; the study does not show that such omissions erase useful continuity information. Do not manually reconstruct historical rosters to address this study's discrepancies.",
            "",
            "### Production adoption",
            "",
            "Production adoption remains undecided until the continuity feature is evaluated empirically. The pair-pool arithmetic above is not evidence of missing prior shared-roster relationships or predictive impact. The next research step is to exercise the full CFBD-derived feature and determine whether the observed imperfections matter.",
            "",
            "The four pre-existing ambiguous cases—Miami's Shane McDermott, Toledo's Jordan Fair, Old Dominion's Cameron Hill, and San José State's Reggie Jones Jr.—remain unresolved and are not prerequisites for exploratory research. This study makes no production model, Context, or data-panel changes.",
            "",
            "## Reproducibility",
            "",
            "Run from the repository root with the issue 183 raw corpus available:",
            "",
            "```bash",
            "uv run python scripts/freeze_issue_185_target_pool_sample.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check",
            "uv run python scripts/build_issue_185_ol_roster_validation.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check",
            "```",
            "",
            "The build manifest hashes the frozen sample, official source inventory, independent official OL transcriptions, candidate review decisions, identity crosswalk, and normalized CFBD cache. The raw cache remains outside the repository.",
            "",
            "## Build input hashes",
            "",
        ]
    )
    report.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(hashes.items()))
    report.append("")
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True, help="Existing external issue 183 CFBD cache root")
    parser.add_argument("--check", action="store_true", help="Verify generated artifacts without rewriting them")
    args = parser.parse_args()
    if not args.raw_root.expanduser().resolve().is_dir():
        raise SystemExit(f"CFBD raw cache not found: {args.raw_root}")
    build(args.raw_root, check=args.check)


if __name__ == "__main__":
    main()
