"""Build the bounded, manually sourced Issue 181 OL co-start pilot tables."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/offensive_line_co_start_pilot_issue_181"
POSITIONS = ("LT", "LG", "C", "RG", "RT")
SOURCE_PRIORITY = {"complete": 3, "partial": 2, "not_located": 1}
MATRIX_ROLES = {"matrix_source", "alternate_matrix_source", "search_candidate"}
EXPECTED_MATRICES = {
    ("Ohio State", 2014): 15,
    ("Toledo", 2014): 13,
    ("Clemson", 2019): 15,
    ("Liberty", 2023): 14,
    ("Air Force", 2023): 13,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def best_matrix_sources() -> dict[tuple[str, str], dict[str, str]]:
    sources = [
        row
        for row in read_csv(DATA_DIR / "source_inventory.csv")
        if row["source_role"] in MATRIX_ROLES
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for source in sources:
        grouped[(source["team"], source["season"])].append(source)

    return {
        key: max(
            candidates,
            key=lambda row: (
                SOURCE_PRIORITY[row["coverage_status"]],
                row["source_role"] == "matrix_source",
            ),
        )
        for key, candidates in grouped.items()
    }


def build_coverage() -> list[dict[str, object]]:
    sample = read_csv(DATA_DIR / "sample.csv")
    sources = best_matrix_sources()
    rows: list[dict[str, object]] = []
    for item in sample:
        source = sources.get((item["team"], item["target_season"]))
        note = (
            source["coverage_note"]
            if source
            else (
                "No complete OL starter matrix located in the targeted official-source "
                "search; this is not evidence that none exists."
            )
        )
        rows.append(
            {
                **item,
                "coverage_status": (
                    source["coverage_status"] if source else "not_located"
                ),
                "source_id": source["source_id"] if source else "",
                "source_type": source["source_type"] if source else "",
                "source_title": source["source_title"] if source else "",
                "source_url": source["source_url"] if source else "",
                "games_found": source["games_found"] if source else "",
                "games_expected": source["games_expected"] if source else "",
                "coverage_note": note,
            }
        )
    return rows


def build_coverage_by_window(
    coverage_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in coverage_rows:
        counts[str(row["window_start"])][str(row["coverage_status"])] += 1
    return [
        {
            "window_start": window,
            "window_label": f"{window}–{int(window) + 1}",
            "complete": counts[window]["complete"],
            "partial": counts[window]["partial"],
            "not_located": counts[window]["not_located"],
            "team_seasons": sum(counts[window].values()),
        }
        for window in sorted(counts)
    ]


def build_source_type_distribution() -> list[dict[str, object]]:
    sources = best_matrix_sources()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources.values():
        status = source["coverage_status"]
        if status in {"complete", "partial"}:
            counts[source["source_type"]][status] += 1
    return [
        {
            "source_type": source_type,
            "complete_matrices": counts[source_type]["complete"],
            "partial_matrices": counts[source_type]["partial"],
            "team_seasons_with_table": sum(counts[source_type].values()),
        }
        for source_type in sorted(counts)
    ]


def build_identity_summary() -> list[dict[str, object]]:
    rows = read_csv(DATA_DIR / "player_identity_resolution.csv")
    by_player: dict[str, str] = {}
    label_counts: Counter[str] = Counter()
    for row in rows:
        match_class = row["identity_match_class"]
        label_counts[match_class] += 1
        player_id = row["canonical_player_id"]
        previous = by_player.get(player_id, "normalized")
        by_player[player_id] = (
            "manual"
            if match_class == "manual" or previous == "manual"
            else match_class
        )
    player_counts = Counter(by_player.values())
    classes = ("exact", "normalized", "manual", "ambiguous", "unresolved")
    return [
        {
            "identity_match_class": match_class,
            "unique_players": player_counts[match_class],
            "source_label_rows": label_counts[match_class],
            "unique_players_total": len(by_player),
            "source_label_rows_total": len(rows),
        }
        for match_class in classes
    ]


def build_target_roster_summary() -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = Counter()
    for row in read_csv(DATA_DIR / "target_roster_resolution.csv"):
        counts[(row["target_roster_boundary"], row["target_roster_status"])] += 1
    return [
        {
            "target_roster_boundary": boundary,
            "target_roster_status": status,
            "players": count,
        }
        for (boundary, status), count in sorted(counts.items())
    ]


def read_manual_matrices() -> list[dict[str, str]]:
    manual_rows = read_csv(DATA_DIR / "manual_ol_game_starters.csv")
    liberty_rows = read_csv(DATA_DIR / "liberty_2023_ol_game_starters.csv")
    for row in liberty_rows:
        row.update(
            {
                "team": "Liberty",
                "season": "2023",
                "source_id": "LIBERTY_2023_GAME1_2024",
            }
        )
    return manual_rows + liberty_rows


def build_starter_rows() -> list[dict[str, object]]:
    identities = {
        (row["team"], row["season"], row["source_label"]): row
        for row in read_csv(DATA_DIR / "player_identity_resolution.csv")
    }
    sources = {
        row["source_id"]: row for row in read_csv(DATA_DIR / "source_inventory.csv")
    }
    matrices = read_manual_matrices()
    games_by_case: dict[tuple[str, int], set[int]] = defaultdict(set)
    rows: list[dict[str, object]] = []

    for game in matrices:
        team = game["team"]
        season = int(game["season"])
        game_no = int(game["game_no"])
        source_id = game["source_id"]
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"Unknown matrix source id {source_id!r}")
        if not game["opponent_source_label"]:
            raise ValueError(f"Missing opponent source label for {team} {season} game {game_no}")
        if any(not game[position] for position in POSITIONS):
            raise ValueError(f"Incomplete five-player row for {team} {season} game {game_no}")
        if len({game[position] for position in POSITIONS}) != len(POSITIONS):
            raise ValueError(f"Duplicate player within {team} {season} game {game_no}")

        games_by_case[(team, season)].add(game_no)
        for position in POSITIONS:
            label = game[position]
            identity = identities.get((team, str(season), label))
            if identity is None:
                raise ValueError(
                    f"Unresolved source label {label!r} for {team} {season}"
                )
            rows.append(
                {
                    "team": team,
                    "season": season,
                    "game_no": game_no,
                    "opponent_source_label": game["opponent_source_label"],
                    "position": position,
                    "source_label": label,
                    "canonical_player_id": identity["canonical_player_id"],
                    "full_name": identity["full_name"],
                    "identity_match_class": identity["identity_match_class"],
                    "source_id": source_id,
                    "source_url": source["source_url"],
                    "table_locator": source["table_locator"],
                }
            )

    for case, expected_games in EXPECTED_MATRICES.items():
        games = games_by_case.get(case, set())
        if games != set(range(1, expected_games + 1)):
            raise ValueError(
                f"{case} expected games 1–{expected_games}, found {sorted(games)}"
            )
    if set(games_by_case) != set(EXPECTED_MATRICES):
        raise ValueError(f"Unexpected normalized cases: {sorted(games_by_case)}")

    return sorted(
        rows,
        key=lambda row: (
            str(row["team"]),
            int(row["season"]),
            int(row["game_no"]),
            POSITIONS.index(str(row["position"])),
        ),
    )


def build_co_starts(
    starter_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    identities = read_csv(DATA_DIR / "player_identity_resolution.csv")
    identity_by_id = {row["canonical_player_id"]: row for row in identities}
    identities_by_case: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    identity_classes: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in identities:
        case = (row["team"], int(row["season"]))
        player_id = row["canonical_player_id"]
        identities_by_case[case].append(row)
        previous = identity_classes[case].get(player_id, "normalized")
        if row["identity_match_class"] == "manual" or previous == "manual":
            identity_classes[case][player_id] = "manual"
        else:
            identity_classes[case][player_id] = row["identity_match_class"]

    roster_rows = read_csv(DATA_DIR / "target_roster_resolution.csv")
    roster_by_case: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in roster_rows:
        case = (row["team"], int(row["prior_season"]))
        player_id = row["canonical_player_id"]
        if player_id in roster_by_case[case]:
            raise ValueError(f"Duplicate target-roster row for {case} player {player_id}")
        roster_by_case[case][player_id] = row

    starters_by_game: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in starter_rows:
        key = (str(row["team"]), int(row["season"]), int(row["game_no"]))
        starters_by_game[key].append(row)

    shared_games: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    pair_positions: dict[tuple[str, int, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for (team, season, game_no), game_rows in starters_by_game.items():
        case = (team, season)
        roster = roster_by_case[case]
        eligible = [
            row
            for row in game_rows
            if roster[str(row["canonical_player_id"])]["target_roster_status"]
            == "present"
        ]
        for first, second in combinations(eligible, 2):
            first_id = str(first["canonical_player_id"])
            second_id = str(second["canonical_player_id"])
            low_id, high_id = sorted((first_id, second_id))
            key = (team, season, low_id, high_id)
            shared_games[key].append(game_no)
            pair_positions[key][first_id].add(str(first["position"]))
            pair_positions[key][second_id].add(str(second["position"]))

    case_games: dict[tuple[str, int], set[int]] = defaultdict(set)
    for team, season, game_no in starters_by_game:
        case_games[(team, season)].add(game_no)

    pair_rows: list[dict[str, object]] = []
    for (team, season, first_id, second_id), game_numbers in sorted(shared_games.items()):
        first = identity_by_id[first_id]
        second = identity_by_id[second_id]
        first_positions = pair_positions[(team, season, first_id, second_id)][first_id]
        second_positions = pair_positions[(team, season, first_id, second_id)][second_id]
        source_id = next(
            row["source_id"]
            for row in starter_rows
            if row["team"] == team and row["season"] == season
        )
        source = next(
            row
            for row in read_csv(DATA_DIR / "source_inventory.csv")
            if row["source_id"] == source_id
        )
        first_roster = roster_by_case[(team, season)][first_id]
        second_roster = roster_by_case[(team, season)][second_id]
        pair_rows.append(
            {
                "team": team,
                "prior_season": season,
                "target_season": season + 1,
                "player_1_id": first_id,
                "player_1_name": first["full_name"],
                "player_1_positions": ";".join(
                    position for position in POSITIONS if position in first_positions
                ),
                "player_2_id": second_id,
                "player_2_name": second["full_name"],
                "player_2_positions": ";".join(
                    position for position in POSITIONS if position in second_positions
                ),
                "shared_starts": len(game_numbers),
                "prior_matrix_games": len(case_games[(team, season)]),
                "shared_start_rate": f"{len(game_numbers) / len(case_games[(team, season)]):.6f}",
                "shared_game_numbers": ";".join(
                    str(number) for number in sorted(game_numbers)
                ),
                "prior_matrix_source_id": source_id,
                "prior_matrix_source_url": source["source_url"],
                "target_roster_boundary": (
                    f"{first_roster['target_roster_boundary']};"
                    f"{second_roster['target_roster_boundary']}"
                ),
            }
        )

    pair_counts: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        pair_counts[(str(row["team"]), int(row["prior_season"]))].append(row)

    case_rows: list[dict[str, object]] = []
    for case, game_numbers in sorted(case_games.items()):
        team, season = case
        players = identity_classes[case]
        class_counts = Counter(players.values())
        roster = roster_by_case[case]
        status_counts = Counter(row["target_roster_status"] for row in roster.values())
        pairs = pair_counts[case]
        max_pair = max((int(row["shared_starts"]) for row in pairs), default=0)
        case_rows.append(
            {
                "team": team,
                "prior_season": season,
                "target_season": season + 1,
                "prior_matrix_games": len(game_numbers),
                "unique_prior_ol_starters": len(players),
                "source_label_rows": len(identities_by_case[case]),
                "identity_exact": class_counts["exact"],
                "identity_normalized": class_counts["normalized"],
                "identity_manual": class_counts["manual"],
                "identity_ambiguous": class_counts["ambiguous"],
                "identity_unresolved": class_counts["unresolved"],
                "target_roster_present": status_counts["present"],
                "target_roster_absent": status_counts["absent"],
                "target_roster_unresolved": status_counts["unresolved"],
                "eligible_returner_pairs": (
                    status_counts["present"] * (status_counts["present"] - 1) // 2
                ),
                "observed_co_start_pairs": len(pairs),
                "maximum_shared_starts": max_pair,
                "maximum_shared_start_rate": (
                    f"{max_pair / len(game_numbers):.6f}" if max_pair else "0.000000"
                ),
            }
        )

    return pair_rows, case_rows


def main() -> None:
    coverage_rows = build_coverage()
    coverage_fields = [
        "sample_id",
        "window_start",
        "target_season",
        "team",
        "program_stratum",
        "season_pair_id",
        "coverage_status",
        "source_id",
        "source_type",
        "source_title",
        "source_url",
        "games_found",
        "games_expected",
        "coverage_note",
    ]
    write_csv(DATA_DIR / "sample_coverage.csv", coverage_fields, coverage_rows)

    coverage_by_window = build_coverage_by_window(coverage_rows)
    write_csv(
        DATA_DIR / "coverage_by_window.csv",
        [
            "window_start",
            "window_label",
            "complete",
            "partial",
            "not_located",
            "team_seasons",
        ],
        coverage_by_window,
    )
    source_type_rows = build_source_type_distribution()
    write_csv(
        DATA_DIR / "source_type_distribution.csv",
        [
            "source_type",
            "complete_matrices",
            "partial_matrices",
            "team_seasons_with_table",
        ],
        source_type_rows,
    )
    write_csv(
        DATA_DIR / "identity_resolution_summary.csv",
        [
            "identity_match_class",
            "unique_players",
            "source_label_rows",
            "unique_players_total",
            "source_label_rows_total",
        ],
        build_identity_summary(),
    )
    write_csv(
        DATA_DIR / "target_roster_summary.csv",
        ["target_roster_boundary", "target_roster_status", "players"],
        build_target_roster_summary(),
    )

    starter_rows = build_starter_rows()
    write_csv(
        DATA_DIR / "normalized_game_starters.csv",
        [
            "team",
            "season",
            "game_no",
            "opponent_source_label",
            "position",
            "source_label",
            "canonical_player_id",
            "full_name",
            "identity_match_class",
            "source_id",
            "source_url",
            "table_locator",
        ],
        starter_rows,
    )

    pair_rows, case_rows = build_co_starts(starter_rows)
    write_csv(
        DATA_DIR / "eligible_co_start_pairs.csv",
        [
            "team",
            "prior_season",
            "target_season",
            "player_1_id",
            "player_1_name",
            "player_1_positions",
            "player_2_id",
            "player_2_name",
            "player_2_positions",
            "shared_starts",
            "prior_matrix_games",
            "shared_start_rate",
            "shared_game_numbers",
            "prior_matrix_source_id",
            "prior_matrix_source_url",
            "target_roster_boundary",
        ],
        pair_rows,
    )
    write_csv(
        DATA_DIR / "case_summary.csv",
        [
            "team",
            "prior_season",
            "target_season",
            "prior_matrix_games",
            "unique_prior_ol_starters",
            "source_label_rows",
            "identity_exact",
            "identity_normalized",
            "identity_manual",
            "identity_ambiguous",
            "identity_unresolved",
            "target_roster_present",
            "target_roster_absent",
            "target_roster_unresolved",
            "eligible_returner_pairs",
            "observed_co_start_pairs",
            "maximum_shared_starts",
            "maximum_shared_start_rate",
        ],
        case_rows,
    )

    statuses = Counter(str(row["coverage_status"]) for row in coverage_rows)
    expected_statuses = {"complete": 8, "partial": 7, "not_located": 25}
    if statuses != expected_statuses:
        raise ValueError(f"Unexpected coverage totals: {dict(statuses)!r}")
    if len(starter_rows) != 350 or len(pair_rows) != 8:
        raise ValueError(
            f"Unexpected normalized output sizes: starts={len(starter_rows)}, "
            f"pairs={len(pair_rows)}"
        )
    print(
        f"Built coverage for {len(coverage_rows)} frozen team-seasons "
        f"(complete={statuses['complete']}, partial={statuses['partial']}, "
        f"not_located={statuses['not_located']}); normalized "
        f"{len(starter_rows)} OL starts across {len(case_rows)} complete matrices "
        f"into {len(pair_rows)} observed eligible co-start pairs."
    )


if __name__ == "__main__":
    main()
