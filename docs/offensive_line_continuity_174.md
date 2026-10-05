# Offensive-line continuity reconnaissance (issue 174)

**Decision: keep data work first; reject raw `/games/players` rows as an OL participation proxy.** A small authenticated CFBD sample finds very sparse OL rows, no game with multiple rostered OL rows, and a direct gamebook check where the raw response has no row for two OL the team says started. The endpoint can retain placeholders, but none of the sampled OL rows were placeholders. It cannot support OL appearance or pairwise co-play. The lack of a timestamped target-season preseason roster remains a separate blocker.

This is a data-discovery result only. It does not add a Context feature or change ranking behavior.

## Scope and reproducibility

The audit examined the checked-in source tree and the local WSL CFBD cache available on 2026-10-05. Some historical raw files in that cache are ignored by Git, so the checked-in `source_inventory.json` records the cache inventory but does not include the raw payloads. A follow-up search across the main checkout, other Linux and Windows-mounted worktrees, and `/tmp` found no reusable `/games/players` or full roster snapshots; matching pytest files were synthetic fixtures. I then used the configured CFBD API key for one-off requests on 2026-10-05. The normalized per-game counts are in [raw_game_player_ol_coverage.csv](../data/processed/offensive_line_continuity_issue_174/raw_game_player_ol_coverage.csv); response bodies were kept under `/tmp` and are not committed. No acquisition code or model experiment was added.

The sample requested `/roster?year=...&team=...` and `/games/players?year=...&team=...` for Alabama, Iowa, and North Dakota State in 2004, 2010, 2015, 2020, and 2024. The usable FBS join is Alabama and Iowa in 2010, 2015, 2020, and 2024: eight team-seasons and 102 team-games. Alabama/Iowa 2004 and North Dakota State were endpoint-coverage probes, not part of the OL-row denominator because their retrieved rosters lacked usable OL positions. Roster and game-player records were joined by athlete ID and filtered to the queried team's side of each game. The tested binary is one when any raw athlete `gamePlayerStat` row appears for that player in any category/type, regardless of the stat value. Positions counted as OL were `OL`, `OT`, `OG`, `C`, `G`, `T`, `LT`, `RT`, `LG`, `RG`, and `OC` (the sampled records used `OL`, `OT`, `G`, or `C`). These are current retrospective roster responses, not archived August rosters.

Rebuild the offline source inventory against any cache with:

```text
uv run python scripts/audit_offensive_line_continuity_sources.py \
  --data-root /path/to/GippyRank4/data \
  --output data/processed/offensive_line_continuity_issue_174/source_inventory.json
```

The script is offline and reads filenames, provenance sidecars, and the existing transfer-snapshot manifest only. It does not inspect credentials, call CFBD, or change raw files. The candidate definitions and their evidence requirements are in [candidate_feature_inventory.csv](../data/processed/offensive_line_continuity_issue_174/candidate_feature_inventory.csv).

## Data findings

### Repository artifacts

| Source in the local cache | What it establishes | What it does not establish |
| --- | --- | --- |
| CFBD `/games` schedule payloads | Team-season games and game IDs | Player membership or participation |
| CFBD `/games/teams` payloads under `data/raw/cfbd/game_stats/` | Team-level game box-score categories; the local cache has 764 payloads and 27 provenance sidecars | Player-level appearances, starts, snaps, positions, or pairs |
| Preseason returning-production payloads | Aggregate team-season returning production; 13 non-empty files covering 2014–2026 in the local cache | OL-specific production or which players worked together |
| Preseason talent and team recruiting payloads | Team-level roster-talent and recruiting summaries | Player-game involvement or line continuity |
| Preseason team and coaching data | Team metadata and coach tenure | Player roster or player participation |
| Roster, player-game, player-usage, player-season-stat snapshots | **None found in the local raw cache** | Historical OL roster membership or participation coverage |

The local returning-production, talent, recruiting, and season-team JSON caches have no matching provenance sidecars in this scan. Their season fields describe the data year, not the retrieval date, so they cannot establish what was known at a historical August cutoff.

The raw player-snapshot manifest used by the existing transfer pipeline was also absent. Existing transfer-processing code can preserve timestamped `/roster`, `/games/players`, `/player/usage`, and `/stats/player/season` responses, but the saved roster in that workflow is a *prior-season identity/position reference* for transfer joins. It is not an archive of the full target-season roster as it stood before Week 1. See [the transfer pipeline contract](preseason_transfer_pipeline.md) and [the offline source inventory](../data/processed/offensive_line_continuity_issue_174/source_inventory.json).

The checked-in Context model currently has team/program history, recruiting, roster talent, returning production, coach tenure, and incoming-transfer inputs. Context 1.3 names the current aggregate features, including `returning_pct_ppa`, `talent_composite`, recruiting summaries, `coach_tenure_seasons`, and `transfer_in_prior_usage_sum` ([feature definitions](../src/gippyrank/context_prior_v1_3.py)). None is a player-pair or OL-lineup measurement. A continuity feature could overlap with returning production and transfer inputs, while still measuring a different concept; that overlap cannot be quantified without a valid panel.

### CFBD roster and identity

CFBD documents historical season rosters from 2004 onward. The current `/roster` request accepts team, season year, and classification; its roster records include a player ID and a string-valued position. The documented request has no `as_of`, retrieval-history, or roster-version parameter ([availability](https://apinext.collegefootballdata.com/data-availability), [roster endpoint](https://apinext.collegefootballdata.com/api/teams)).

That supports asking for a season roster today, but does not reproduce who was on that roster by an August cutoff. No historical response snapshots or retrieval timestamps exist in the local cache, and the API documentation does not say that a later season roster is identical to the pre-season roster. The stability of retrospective rosters therefore remains unknown; they are not safe evidence for this study.

The API schemas expose IDs in roster, player-usage, and game-player records, which may permit direct joins across those sources. The portal record schema instead exposes names, position, origin, destination, and transfer date without a shared player ID. The repository's existing transfer pipeline consequently uses verified IDs when available and otherwise exact normalized player/team matches or explicit aliases; ambiguous matches remain unresolved. The live sample could join FBS OL IDs for 2010–2024, but that is not an archived or preseason panel; see the row-coverage results below.

### Prior participation and pairwise co-play

CFBD documents player season statistics from 2004 onward, player usage from 2013 onward, returning production from 2014 onward, and portal data from 2021 onward ([historical coverage](https://apinext.collegefootballdata.com/data-availability)). These API availability ranges are not the same as local archived coverage.

The documented evidence has these limits:
- CFBD’s `/plays/stats` endpoint returns player-to-play statistical associations, capped at 2,000 records per request. Its documented rows include athlete IDs and stats but no full lineup record, so it can connect recorded action to a play without establishing which OL were on the field ([Plays API](https://apinext.collegefootballdata.com/api/plays)).

- `/games/players` returns player **box-score statistics** grouped by game, team, category, type, and athlete. It contains no start flag, complete appearance list, snaps, or line-combination identifier ([Games API](https://apinext.collegefootballdata.com/api/games)). A player with no recorded box-score statistic must not be treated as a DNP, and two OL players with stats in one game are not thereby shown to have played simultaneously. The empirical row check below also shows that raw OL rows are scarce and mostly defensive-category entries.
- `/player/season/overview` includes a season game count and box-score, usage, and PPA summaries. These fields do not identify starts or position-specific snaps ([Players API](https://apinext.collegefootballdata.com/api/players)).
- `/player/usage` provides rates for relevant plays, not snap totals or same-play co-occurrence. CFBD defines usage rate as the share of relevant team plays involving a player, and notes that usage requires qualifying attributed plays ([Players API](https://apinext.collegefootballdata.com/api/players), [metric definition](https://apinext.collegefootballdata.com/metrics-and-definitions)). It does not establish complete OL participation.
- The local `/games/teams` archive contains team box scores only. The existing repository audit explicitly treats it as a team-stat corpus, not player data ([box-score audit implementation](../src/gippyrank/research/box_score_audit.py)).

### Raw `/games/players` OL-row check

| Season | Alabama: rostered OL / games with at least one row | Iowa: rostered OL / games with at least one row |
| --- | ---: | ---: |
| 2010 | 21 / 0 of 13 | 16 / 0 of 13 |
| 2015 | 18 / 0 of 15 | 17 / 0 of 14 |
| 2020 | 19 / 2 of 13 | 23 / 3 of 8 |
| 2024 | 23 / 3 of 13 | 22 / 2 of 13 |

Across those 102 games, 10 had a raw row for any rostered OL, and every such game had exactly one. The 159 team-season OL roster entries produce 2,005 possible rostered-OL/game cells; only 10 cells had any row. The 92 other games had no OL row at all. This is not a participation recall estimate because roster membership includes nonparticipants, but the independent gamebook check below confirms that known starters can have no raw row.

The 10 player-game pairs generated 68 athlete/type rows:

| Category/type | OL player-games | Raw stat values |
| --- | ---: | --- |
| `defensive/TOT` | 9 | `1` in all 9 |
| `defensive/SOLO` | 9 | `1` in 7; `0` in 2 |
| `defensive/SACKS`, `TFL`, `PD`, `QB HUR`, `TD` | 9 each | `0` in every row |
| `receiving/REC`, `YDS`, `LONG`, `AVG`, `TD` | 1 each | `0`, `8`, `8`, `0.0`, `0` respectively |

So the nine defensive-category pairs are tackle lines (each with six additional zero-valued defensive fields), while the one remaining pair is an unusual receiving line. No rostered OL had a `--` or `--/--` row. A scan of all returned team-side payloads found eight `passing/QBR` rows with `--` and no `--/--` value; those placeholders were not attached to a rostered OL. Placeholder retention therefore did not improve OL coverage in this sample.

The check against independent team notes was particularly clear for Iowa at Maryland on Nov. 23, 2024 (CFBD game ID `401628556`). CFBD's queried Iowa side has zero rows for its 22 rostered OL. Iowa's [official game notes](https://storage.googleapis.com/hawkeyesports-prod/2025/01/29/sWdcEcDrRJ2pZAEI8xZSoF9NvT4RvmRIiuHrNeMl.pdf) identify Mason Richman as making his 50th career start and Nick DeJong as making his first start of the season in place of Gennings Dunker. Iowa's [official final notes](https://hawkeyesports.com/news/2025/01/30/2024-iowa-football-final-notes) say Dunker missed the final two regular-season games and DeJong started those games in his place. This is a concrete false negative for treating any raw row as OL participation.

North Dakota State also showed that broad historical roster coverage should not be inferred from the endpoint's season availability: its roster query returned 14 records in 2010 (no OL labels), 3 in 2015, none in 2020, and 53 in 2024 (again no OL labels); its game-player queries ranged from zero team-games to 16. Alabama and Iowa's 2004 roster queries returned 28 and 30 records, respectively, all without position values, despite 12 game-player team-games each. These older/FCS responses cannot be joined to OL and are excluded from the table above.

**Answer: no.** “Any raw game-player row in game G” is neither a defensible OL appearance indicator nor a co-appearance indicator. It misses known starters, rows that do appear are almost entirely defensive tackle rows (not offensive line evidence), and no game in the usable sample has two rostered OL rows from which a pair could even be nominated. The placeholders seen in `passing/QBR` did not add an OL record. Pairwise same-game co-appearance does not merit a follow-up using this field.

For a historical next-best proxy, seek official gamebook or team game-note starting-lineup records with player IDs (or carefully verified aliases), and call a pair measure **shared starts/co-starts**. That still would not measure every appearance or shared snaps. If such lineups cannot be acquired consistently, keep individual prior starts/experience separate and use shared roster seasons only as a coarse roster-overlap descriptor; do not label either as co-play.

## Candidate feature inventory

The inventory deliberately preserves multiple possible resolutions instead of naming a winning formula. Detailed definitions, required inputs, missing-data behavior, leakage risks, and Context overlap are in the CSV artifact.

1. **Shared prior starts:** count returning OL pairs that started the same prior-season game. This is a coarse, interpretable unit-experience measure if complete player-game starter data can be obtained. No start field is documented in the reviewed endpoint schemas; no local player-level archive is available to inspect.
2. **Shared prior games:** count pairs with verified prior-game offensive participation. This needs complete appearances; box-score stat rows alone are an incomplete proxy.
3. **Shared snaps or workload retained:** these need same-play lineups or snap-level participation. Individual player snap totals alone cannot establish co-play.
4. **Shared roster seasons:** this is the only simple roster-history proxy suggested by a season-keyed roster endpoint, but it means “listed at the same school,” not “functioned together.” It should not be promoted as playing experience and cannot fix missing target-season August membership.
5. **Individual experience control:** retain prior starts/participation separately from pair continuity. An experienced transfer can then have substantial individual experience and zero shared history with the current room. Zero shared history is not a negative transfer penalty, and transfer count/usage should remain a separate Context input.

If a future source only supports starts, pairwise co-starts should be called co-starts, not shared snaps. If it only supports individual snap totals, same-snap overlap remains unknown.

## Coverage, examples, and predictive reconnaissance

- **Reliable target-season preseason roster coverage:** 0 archived team-seasons in the local cache.
- **Live retrospective OL join coverage:** usable positions in eight FBS team-seasons (2010–2024), covering 102 Alabama/Iowa team-games. The 2004 sample and North Dakota State probes lacked usable OL labels.
- **Player participation/start/snap coverage:** 0 archived player-game datasets in the local cache. The live sample returned raw stat rows, not appearance, start, or snap records.
- **Pairwise co-play coverage:** 0 measurable team-seasons. No sampled team-game contained more than one rostered OL row, and a raw stat row does not establish simultaneous play.
- **Context redundancy correlations:** not calculated; there is no valid continuity column to compare with returning production, roster talent, recruiting, transfers, or program history.
- **Requested examples:** no highly continuous, rebuilt, transfer-heavy, or young internally continuous room can be labeled under the issue's point-in-time rules. Choosing a known target-season starting five would be hindsight.
- **Predictive reconnaissance:** not run. The issue says to stop rather than substitute retrospective target-season membership when a defensible preseason state cannot be recovered.

## Smallest useful follow-up

Create a data-acquisition ticket before a modeling ticket:

1. Prospectively capture each target team's full `/roster` response by the repository's existing August 15 cutoff (starting with the 2027 preseason). Save the raw response unchanged, request parameters, retrieval time, cutoff flag, hash, and canonical team mapping. Make this explicitly a **target-season preseason roster**, distinct from the prior-season roster reference used for transfer joins.
2. Prioritize official historical starting-lineup/gamebook sources with stable IDs and game IDs; evaluate shared starts and individual prior starts as separate measures. For appearance-level continuity, require a complete offensive participation list. For snap-level continuity, require actual shared-play/lineup evidence; do not infer it from individual totals. The CFBD raw any-row signal failed the OL spot check and should not be substituted.
3. Begin with one descriptive panel and coverage report. If only starters are reliable, evaluate shared co-start pairs alongside individual prior starts and existing Context inputs. If no complete participation source is found, stop at a roster-overlap description and do not call it on-field continuity.

Only after this panel passes identity and coverage checks should a separate ticket compare candidate measures against existing Context inputs and decide whether a controlled modeling experiment is warranted.
