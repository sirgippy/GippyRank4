"""Build issue 185's offline CFBD-to-official OL roster comparison.

The sample was frozen before official roster inspection. This builder reads the
frozen rows, cached CFBD roster data, the official-source inventory, and a small
manual delta/crosswalk table. It does not download or mutate raw source data.
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
DELTAS_PATH = RESEARCH / "manual_roster_deltas.csv"
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


def load_cfbd() -> dict[str, dict[str, str]]:
    by_key: dict[str, dict[str, str]] = defaultdict(dict)
    with gzip.open(CFBD_PATH, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_classification"] != "fbs" or row["normalized_ol_status"] != "offensive_line":
                continue
            key = f"{row['season']}|{row['team_id']}"
            name_key = normalize(row["player_name"])
            if name_key:
                by_key[key].setdefault(name_key, row["player_name"])
    return by_key


def build(raw_root: Path, *, check: bool = False) -> None:
    sample = read_csv(SAMPLE_PATH)
    sources = {row["sample_id"]: row for row in read_csv(OFFICIAL_SOURCES_PATH)}
    deltas = read_csv(DELTAS_PATH)
    aliases = read_csv(CROSSWALK_PATH)
    cfbd = load_cfbd()
    sample_by_id = {row["sample_id"]: row for row in sample}
    if len(sample) != 40 or set(sources) != set(sample_by_id):
        raise SystemExit("Expected 40 frozen sample rows and one official source per sample")

    deltas_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    aliases_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deltas:
        deltas_by_id[row["sample_id"]].append(row)
    for row in aliases:
        aliases_by_id[row["sample_id"]].append(row)

    player_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    mismatch_counter: Counter[tuple[str, str]] = Counter()
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
        unresolved: set[str] = set()
        candidate_category: dict[str, str] = {}
        candidate_note: dict[str, str] = {}
        official_name_by_key: dict[str, str] = dict(cfbd_names)

        for alias in aliases_by_id[sample_id]:
            cfbd_name = alias["cfbd_player_name"]
            official_name = alias["official_player_name"]
            ckey, okey = normalize(cfbd_name), normalize(official_name)
            if ckey not in candidate_name_by_key:
                raise SystemExit(f"Crosswalk candidate missing for {sample_id}: {cfbd_name}")
            if okey in candidate_name_by_key and okey != ckey:
                raise SystemExit(f"Crosswalk target collides with a candidate for {sample_id}: {official_name}")
            candidate_name_by_key[okey] = candidate_name_by_key.pop(ckey)
            official_name_by_key.pop(ckey, None)
            official_name_by_key[okey] = official_name

        for delta in deltas_by_id[sample_id]:
            name_key = normalize(delta["player_name"])
            if delta["decision"] == "add":
                official_name_by_key[name_key] = delta["player_name"]
            elif delta["decision"] == "remove":
                if name_key not in candidate_name_by_key:
                    raise SystemExit(f"Removed candidate missing for {sample_id}: {delta['player_name']}")
                official_name_by_key.pop(name_key, None)
                candidate_category[name_key] = delta["category"]
                candidate_note[name_key] = delta["review_note"]
                if delta["category"] == "official_roster_membership_unresolved":
                    unresolved.add(name_key)
            else:
                raise SystemExit(f"Unknown manual delta decision: {delta['decision']}")

        cfbd_keys = set(candidate_name_by_key)
        official_keys = set(official_name_by_key)
        overlap = cfbd_keys & official_keys
        only_cfbd = cfbd_keys - official_keys
        only_official = official_keys - cfbd_keys
        if not overlap and only_cfbd and only_official and sample_id != "S03":
            raise SystemExit(f"Unexpectedly empty player overlap for {sample_id}")

        for name_key in sorted(cfbd_keys | official_keys):
            in_cfbd, in_official = name_key in cfbd_keys, name_key in official_keys
            cfbd_name = candidate_name_by_key.get(name_key, "")
            official_name = official_name_by_key.get(name_key, "")
            if in_cfbd and in_official:
                status = "matched"
                category = "exact_or_crosswalk_name_match"
                note = ""
            elif in_cfbd and name_key in unresolved:
                status = "unresolved"
                category = candidate_category[name_key]
                note = candidate_note[name_key]
            elif in_cfbd:
                status = "cfbd_only"
                category = candidate_category.get(name_key, "cfbd_only_requires_taxonomy")
                note = candidate_note.get(name_key, "")
            else:
                status = "official_only"
                matching_delta = next(
                    (d for d in deltas_by_id[sample_id] if d["decision"] == "add" and normalize(d["player_name"]) == name_key),
                    None,
                )
                category = matching_delta["category"] if matching_delta else "official_ol_omitted_by_cfbd"
                note = matching_delta["review_note"] if matching_delta else "Added from official source roster"
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
                    "review_note": note,
                    "source_url": sources[sample_id]["source_url"],
                }
            )
            if status in {"cfbd_only", "official_only", "unresolved"}:
                mismatch_counter[(status, category)] += 1

        tp = len(overlap)
        fp = len(only_cfbd - unresolved)
        fn = len(only_official)
        resolved_predicted = tp + fp
        precision = tp / resolved_predicted if resolved_predicted else None
        recall = tp / len(official_keys) if official_keys else None
        exact_player_set = not fp and not fn and not unresolved
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
            "unresolved_cfbd_names": len(unresolved),
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
                "official_count_basis": "Manual roster reconciliation; see manual_roster_deltas.csv and identity_crosswalk.csv",
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
            "unresolved": len(unresolved),
            "exact_count": int(len(cfbd_keys) == len(official_keys)),
            "exact_set": int(exact_player_set),
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
        {"comparison_status": status, "mismatch_category": category, "player_rows": count}
        for (status, category), count in sorted(mismatch_counter.items())
    ]
    input_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (SAMPLE_PATH, OFFICIAL_SOURCES_PATH, DELTAS_PATH, CROSSWALK_PATH, CFBD_PATH)
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
        "sample_id", "season", "team_name", "cfbd_player_name", "official_player_name", "cfbd_ol", "official_ol", "comparison_status", "mismatch_category", "review_note", "source_url",
    ])
    write_csv(OUTPUT / "team_season_summary.csv", team_rows, list(team_rows[0]))
    write_csv(OUTPUT / "era_metrics.csv", era_rows, list(era_rows[0]))
    write_csv(OUTPUT / "mismatch_taxonomy.csv", taxonomy_rows, ["comparison_status", "mismatch_category", "player_rows"])
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


def render_report(team_rows: list[dict[str, object]], era_rows: list[dict[str, object]], taxonomy_rows: list[dict[str, object]], hashes: dict[str, str]) -> str:
    resolved_tp = sum(int(row["true_positive_names"]) for row in team_rows)
    resolved_fp = sum(int(row["cfbd_only_false_positive_names"]) for row in team_rows)
    resolved_fn = sum(int(row["official_only_false_negative_names"]) for row in team_rows)
    unresolved = sum(int(row["unresolved_cfbd_names"]) for row in team_rows)
    predicted = resolved_tp + resolved_fp
    gold = sum(int(row["official_ol_count"]) for row in team_rows)
    precision = resolved_tp / predicted if predicted else 0
    recall = resolved_tp / gold if gold else 0
    exact_count = sum(row["exact_count_match"] == "true" for row in team_rows)
    exact_sets = sum(row["exact_player_set_match"] == "true" for row in team_rows)
    report = [
        "# Issue 185: CFBD offensive-line target-pool validation",
        "",
        "## Scope and frozen sample",
        "",
        "The study compares CFBD's target-season offensive-line (OL) labels with season-specific official athletics roster positions for 40 preselected FBS team-seasons. The frozen sample covers 2009–2012, 2013–2016, 2017–2020, and 2021–2026, with ten team-seasons in each window. It includes a 2011 Idaho team-season with zero CFBD OL labels and 2025 Army with 36, to expose both tails of the observed pool size.",
        "",
        "The draw was frozen on 2026-10-07 from the issue 183 CFBD season summary and team-conference metadata before official roster pages were inspected. Its sample, protocol, and input hashes are in `data/research/offensive_line_target_pool_validation_issue_185/`. The draw uses 3 lower-quartile, 4 middle-half, and 3 upper-quartile OL-count rows per window, balances roster-size quartiles, and includes at least eight conference labels per window. No official availability informed selection.",
        "",
        "## Source protocol and limitations",
        "",
        "The source inventory records one official athletics roster or season-specific roster guide per sampled season. Most sources are retrospective season archive pages viewed on 2026-10-07; several pages are hosted later than the season. A current archive page does not prove the date its content was first captured, so roster backfills and seasonal cutoffs cannot always be separated. The Central Michigan 2009 roster PDF provides a stronger contemporaneous roster-guide view; Northwestern 2016 and Florida State 2019 use postseason/media-guide material. South Carolina 2019 is supported by its season-specific official archive page with OL labels.",
        "",
        "No raw CFBD responses or official roster downloads are copied into the repository. The builder reads the existing shared issue 183 CFBD cache offline and leaves that source corpus unchanged. The official comparison set is represented as the CFBD candidate pool plus reviewer-recorded official additions/removals; the crosswalk resolves the one documented name variation. Each unlisted CFBD candidate was reviewed as an exact official OL match against the linked source. This is a single-review reconciliation, not an independently double-coded audit, and source pages are not archived in the repository. The delta/crosswalk inputs and row-level result files make the decision trail inspectable.",
        "",
        "## Aggregate comparison",
        "",
        f"Across the 40 rows, the current reconciliation contains {gold} official OL names, {resolved_tp} matched names, {resolved_fp} resolved CFBD-only names, {resolved_fn} official-only names, and {unresolved} unresolved CFBD names. Resolved micro-precision is {precision:.1%}; resolved micro-recall is {recall:.1%}. Counts match exactly in {exact_count}/40 team-seasons; complete player sets match in {exact_sets}/40. Unresolved names are excluded from precision and exact-set success, but not hidden.",
        "",
        "### Era metrics",
        "",
        "| Era | Teams | CFBD OL | Official OL | Matched | CFBD-only | Official-only | Unresolved | Precision | Recall | Exact counts | Exact sets |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in era_rows:
        report.append(
            f"| {row['era']} | {row['sampled_team_seasons']} | {row['cfbd_ol_players']} | {row['official_ol_players']} | {row['matched_players']} | {row['cfbd_only_false_positives']} | {row['official_only_false_negatives']} | {row['unresolved_cfbd_players']} | {float(row['micro_precision_resolved']):.1%} | {float(row['micro_recall_resolved']):.1%} | {row['exact_count_match_team_seasons']} | {row['exact_player_set_match_team_seasons']} |"
        )
    report.extend(
        [
            "",
            "## Player-level mismatch findings",
            "",
            "The mismatch taxonomy is in `mismatch_taxonomy.csv`; `player_comparison.csv` preserves the name-level comparison, source link, and review note. The directly documented cases include:",
            "",
            "- **Idaho, 2011:** CFBD has no identifiable OL labels; the official roster identifies 15 offensive linemen. This is a severe target-pool omission and changes 105 potential within-pool player pairs (15 choose 2).",
            "- **Central Michigan, 2009:** CFBD labels three players as OL whom the official guide labels defensive line (Aaron Kaczmarski, Aaron McCord, Cody Pettit), while it omits three listed OL (Allen Ollenburger, Anthony Quinn, Rocky Weaver). The total headcount remains 17, but six player identities differ; the CFBD pool adds 45 unsupported possible pairs and misses 45 official possible pairs.",
            "- **Miami, 2010:** The official roster's Jon Feliciano is the same player as CFBD's Jonathan Feliciano (manual name crosswalk). Jeremy Lewis is listed as defensive line. Shane McDermott is not on the retrieved 2010 roster page and remains an unresolved season-membership/backfill case. The official list also includes Cory White, omitted by CFBD.",
            "- **East Carolina, 2009:** Robert Jones is labeled defensive line on the official roster but appears in CFBD's OL pool. The official archive also makes clear why position-vocabulary handling must include OL/TE compound roles rather than exact-match one label.",
            "- **Other omissions:** the review records official OL omitted by CFBD at UAB 2009, Texas Tech 2014, Toledo 2015, Wake Forest 2014, Georgia 2016, Illinois 2017, Texas 2018, Tulane 2018, and Oklahoma State 2020. Details and official position labels are attached to each player row.",
            "- **Old Dominion, 2025:** CFBD includes Cameron Hill, who is absent from the linked official season roster. Because the source does not establish his official position or season membership, the case remains unresolved.",
            "- **San José State, 2025:** CFBD includes 24 OL candidates while the official roster lists 19 OL. Four CFBD candidates (Gafa Faga, Mata Hola, Quincy Likio, Tangata Tuitupou) are explicitly labeled defensive line by the official source; Reggie Jones Jr. is absent and remains unresolved. The four resolved position-label errors alone yield 82 unsupported possible pairs in the resolved pool.",
            "",
            "The main mismatch mechanisms observed are position-label disagreement, official OL missing from the CFBD pool, roster-season membership ambiguity, and name formatting. The sample is not a census and should not be interpreted as season-level prevalence without weighting; its stratification intentionally oversamples both extremes.",
            "",
            "## Descriptive pair-pool implications",
            "",
            "The pair columns in the team and era summaries apply combinations n choose 2 to the resolved roster pools. They describe how many possible pairs would be admitted or omitted if any two players in a target-season pool could be considered together. They do not estimate actual shared snaps, starts, continuity, or a model effect. The biggest observed case is Idaho 2011: all 105 official pairs are absent from CFBD's zero-player OL pool. Central Michigan 2009 keeps the same pool size but its three false labels and three omissions replace six of the 17 official player identities, adding 45 unsupported and missing 45 official possible pairs.",
            "",
            "## Recommendation",
            "",
            "Treat the CFBD OL target-season roster as a useful candidate pool, not an authoritative roster. Preserve its raw response; normalize explicit OL position labels separately; add source-season and position-label coverage checks; and keep official roster evidence in a reviewable validation layer. Flag empty or unusually large team-season pools, and do not interpret player-pair counts as playing continuity without separate participation evidence. Resolve the four unresolved player-season cases (Miami's Shane McDermott, Toledo's Jordan Fair, Old Dominion's Cameron Hill, and San José State's Reggie Jones Jr.) before using this study to justify any production change. This research does not alter inference, resume projection, model code, or production data.",
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
            "The build manifest stores SHA-256 hashes for the frozen sample/protocol inputs, manual roster review inputs, and normalized CFBD source file. The raw cache is outside the repository.",
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
