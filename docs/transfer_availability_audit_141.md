# Context transfer-data availability audit (issue #141)

## Scope and interpretation

This audit covers all 138 FBS teams in the 2026 Context 1.3 reconstruction and 655 FBS team-seasons in the 2021–2025 retrospective Context transfer panel. The season population comes from the Context transfer feature artifacts, not every school returned by CFBD. The machine-readable inventory is [`team_seasons.csv`](../data/processed/transfer_availability_audit/team_seasons.csv); the complete 2026 affected list is also [`affected_2026.csv`](../data/processed/transfer_availability_audit/affected_2026.csv). [`summary.json`](../data/processed/transfer_availability_audit/summary.json) and [`reason_taxonomy.json`](../data/processed/transfer_availability_audit/reason_taxonomy.json) define counts and reason codes. Run `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_transfer_availability.py` to reproduce them from the committed derived audits.

An input is **complete** only when every known incoming transfer has either a resolved applicable offensive usage value or a defensible legitimate zero, offensive applicability is established, and every known incoming DB transfer has a resolved prior impact (including a verified zero recorded defensive box-score state). A team with no incoming transfers in a covered portal season is complete. **Partial** means some evidence exists but at least one of those conditions fails. **Entirely unavailable** means incoming transfers exist, the retrospective offensive aggregate is null, at least one incoming DB transfer is unresolved, and no incoming DB impact resolves. A team with no incoming DB transfers has a covered natural zero for that feature and therefore remains partial if its offensive aggregate is null. This strict input-completeness definition is wider than the model-facing DB availability flag. In particular, an observed numeric offensive sum can coexist with unresolved players and should not be read as a complete transfer input.

The counts below describe **present-day retrospective evidence**, not archived preseason knowledge. The 2026 manifest has 37 source responses captured on September 19, 2026, all after the August 15 cutoff; the project documents that no on-time immutable transfer snapshot existed for this reconstruction. The 2021–2025 source responses are also retrospective, with no archived August 15 transfer snapshot proving historical availability. The separate `checkpoint_status` column records this for every team-season. It does **not** claim the underlying transfers first became known after the cutoff: publication and revision times for individual portal records cannot be recovered from the committed artifacts.

## Counts

| Season | Team-seasons | Complete | Partial | Entirely unavailable |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 127 | 17 | 106 | 4 |
| 2022 | 130 | 19 | 106 | 5 |
| 2023 | 131 | 15 | 116 | 0 |
| 2024 | 133 | 11 | 120 | 2 |
| 2025 | 134 | 5 | 129 | 0 |
| 2026 | 138 | 4 | 134 | 0 |
| **Total** | **793** | **71 (9.0%)** | **711 (89.7%)** | **11 (1.4%)** |

The pattern recurs across all six seasons; it is not unique to 2026. In the 2026 reconstruction, 3,069 incoming FBS-destination portal records are classified: 705 applicable offensive transfers resolve, 1,342 are legitimate zero/non-applicable, 10 applicable transfers fail usage resolution, and 1,012 cannot be assigned offensive applicability from the available evidence. Those last two groups affect 8 and 134 teams respectively (with overlap). Of 604 incoming DB transfers, 518 resolve and 86 do not, making the model-facing DB feature unavailable for 58 of 138 teams (42.0%). The 4 complete 2026 teams are Air Force, Army, Navy, and SMU.

The 86 unresolved 2026 DB records split into 40 player joins without a covered roster match, 34 origin teams without a matching prior roster team, 11 position-group conflicts, and 1 ambiguous join. These are player counts; a team can have several failure classes. The 34 origin-team cases occur despite the acquisition workflow requesting both FBS and FCS roster/game data. The committed source responses are represented by manifests and derived audits, but their raw payload bytes are absent from this checkout, so the audit cannot prove whether each origin was unsupported by CFBD, returned under another name, or omitted during acquisition.

The 2026 manifest comprises 1 portal, 1 usage, 1 player-stat, 2 roster, and 32 weekly game-player responses. Of 4,471 portal records, 3,069 pass the FBS-destination and date scope rule. Among the 1,402 excluded records, 982 have no destination name, 408 have a destination outside the canonical FBS population, and 12 have a matched FBS destination but an after-cutoff or missing date. These counts show the intended population filter; they do not prove that every provider team name is mapped correctly. The transfer derivation joins portal, usage, stats, roster, and game-player inputs. It does not join recruiting data, so a recruiting/transfer join cannot explain these observed gaps.

The 2026 team-level reason counts are: 134 `offense_applicability_unproven`, 35 `db_player_join_unresolved`, 25 `db_source_team_uncovered`, 9 `db_position_conflict`, 8 `offense_usage_join_unresolved`, and 1 `db_ambiguous_player_join`. These overlap. Conference is not populated in the canonical transfer audit table, so the useful population split here is by source-team coverage and failure class; all target teams are FBS. Among the 134 affected teams, 36 have at least one plausible player-join repair candidate and 25 have a source-team coverage gap. The much larger applicability-unknown group prevents calling the remaining inputs complete even if an individual DB join is repaired.

## Root causes, evidence, and repair scope

The taxonomy distinguishes a **candidate repair** from a **proven repair**. A candidate means a safer identity or aggregation rule could plausibly recover evidence, subject to checking the raw provider response and the original player. An **unresolved** classification means the committed evidence cannot establish whether a pipeline fix or a new source is required. No case is claimed proven irreducible from the provider: absence from a derived roster join is not proof of upstream nonexistence. Within the currently captured FBS/FCS roster inputs, the 34 source-team cases are unavailable, but some may be names or aliases that can be repaired.

| Failure class | Evidence example | What is established | Follow-up repair work |
| --- | --- | --- | --- |
| Prior source-team roster uncovered | 2026 Chris Payne, Lake Erie → UL Monroe: `source_data_unavailable` | No matching prior source-team roster row in the frozen derivation | Check provider roster population and raw team spelling; add an explicit alias only if source identity is verified; assess non-FBS/FCS origins separately. |
| Covered roster, player join unresolved | 2026 Michael Patterson, Stephen F. Austin → Texas State: `identity_resolution_failure` | Source-team roster exists, but normalized player/source key did not match | Inspect source player records, suffixes, and identity aliases; retain ambiguous cases as unresolved. |
| Position-group conflict | 2026 Caden VerMaas, Nebraska → Rice: portal DB, prior roster DE | A player joined, but prior and portal position groups disagree | Review position chronology and mapping policy without making a team-specific correction in this audit. |
| Ambiguous DB join | 2026 Damill Bostic Jr., Villanova → Coastal Carolina | Multiple prior records match | Resolve with verified stable player identity or explicit alias; do not fuzzy-pick one. |
| Applicable offensive usage fails | 2026 Duke has one such transfer; the committed 2026 artifact retains only team-level counts | At least one player had evidence of applicable offensive usage but no resolved value | Persist the 2026 offensive player audit in a separate repair ticket, then diagnose its specific join cause. |
| Historical offensive source-team mismatch | 2021 DJ Turner, Maryland → Pittsburgh | Positive prior offensive stat exists, but the usage join has `source_team_mismatch` | Compare prior team and portal chronology; only accept a verified identity mapping. |
| Historical ambiguous offensive join | 2021 Rico Arnold, Charlotte → Massachusetts | Multiple usage matches meet the normalized key | Resolve identity using a stable source key or explicit verified player alias. |
| Historical applicable usage row absent | 2021 Davontavean Martin, Washington State → Oklahoma State | Positive prior offensive stat exists; no matching usage row | Confirm endpoint completeness and source participation semantics before choosing a replacement source. |
| Applicability cannot be determined | 2024 Kadyn Proctor, Iowa → Alabama (offensive line); 2026 Duke has five undetermined incoming transfers | Portal position and absent box-score stats cannot prove zero offensive usage | Obtain a player participation/roster source with suitable coverage, particularly for offensive line and non-FBS origins. |
| Historical aggregate null | 2021 Ohio State has a null retrospective offensive aggregate while the player-level audit classifies its sole incoming transfer as complete | The research aggregate emits null when no joined prior usage value exists, even where a legitimate zero is established | Reconcile historical research aggregation with the production feature contract in a separate ticket; preserve the published panel here. |

The historical null is a representation defect in the retrospective research panel for at least 17 team-seasons whose player-level offensive audit shows no unresolved or undetermined incoming player. Other nulls coincide with real evidence gaps, so filling every null with zero would be unsafe. The 2026 production reconstruction emits a numeric offensive sum for every team, but its team-level audit still records 10 unresolved applicable players and 1,012 with undetermined applicability. The numeric sum therefore represents resolved contributions, not proof of full coverage.

Across all 793 team-seasons, 321 have at least one candidate repair reason, while 87 have a source-team coverage gap. The mutually exclusive overall `repair_class` is 32 candidate-only gaps, 690 with at least one unresolved source or applicability question, and 71 complete. No apparently irreducible count is asserted without raw source or provider population evidence. The `reason_codes` field retains all contributing classes. `primary_reason` follows a diagnostic priority: DB source coverage and player-join blockers first, then offensive joins and applicability, then historical aggregation.

## 2026 affected teams

Every row below is `partial`; there are no entirely unavailable 2026 teams under the definition above. `Unknown O` counts incoming transfers whose offensive applicability is undetermined, `Failed O` counts applicable usage-resolution failures, and `Unresolved DB` counts incoming DB transfers with unusable prior impact. The full machine-readable row includes every reason code and the separate cutoff status.

| Team | Incoming | Unknown O | Failed O | Unresolved DB | Primary reason |
| --- | ---: | ---: | ---: | ---: | --- |
| Akron | 15 | 6 | 1 | 0 | `offense_usage_join_unresolved` |
| Alabama | 17 | 7 | 0 | 0 | `offense_applicability_unproven` |
| App State | 38 | 13 | 0 | 1 | `db_player_join_unresolved` |
| Arizona | 22 | 7 | 0 | 1 | `db_player_join_unresolved` |
| Arizona State | 24 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Arkansas | 42 | 10 | 0 | 1 | `db_player_join_unresolved` |
| Arkansas State | 25 | 10 | 0 | 0 | `offense_applicability_unproven` |
| Auburn | 39 | 13 | 2 | 1 | `db_player_join_unresolved` |
| BYU | 9 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Ball State | 24 | 8 | 0 | 1 | `db_player_join_unresolved` |
| Baylor | 31 | 7 | 1 | 1 | `db_position_conflict` |
| Boise State | 11 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Boston College | 26 | 8 | 0 | 0 | `offense_applicability_unproven` |
| Bowling Green | 17 | 6 | 0 | 1 | `db_player_join_unresolved` |
| Buffalo | 16 | 12 | 0 | 0 | `offense_applicability_unproven` |
| California | 32 | 7 | 0 | 1 | `db_player_join_unresolved` |
| Central Michigan | 12 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Charlotte | 19 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Cincinnati | 22 | 4 | 0 | 1 | `db_player_join_unresolved` |
| Clemson | 11 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Coastal Carolina | 32 | 15 | 0 | 1 | `db_ambiguous_player_join` |
| Colorado | 42 | 11 | 0 | 0 | `offense_applicability_unproven` |
| Colorado State | 32 | 10 | 0 | 0 | `offense_applicability_unproven` |
| Delaware | 13 | 5 | 0 | 1 | `db_source_team_uncovered` |
| Duke | 19 | 5 | 1 | 0 | `offense_usage_join_unresolved` |
| East Carolina | 24 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Eastern Michigan | 12 | 9 | 1 | 0 | `offense_usage_join_unresolved` |
| Florida | 27 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Florida Atlantic | 22 | 8 | 0 | 0 | `offense_applicability_unproven` |
| Florida International | 17 | 4 | 0 | 1 | `db_player_join_unresolved` |
| Florida State | 22 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Fresno State | 13 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Georgia | 9 | 1 | 0 | 0 | `offense_applicability_unproven` |
| Georgia Southern | 17 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Georgia State | 33 | 22 | 0 | 4 | `db_source_team_uncovered` |
| Georgia Tech | 19 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Hawai'i | 16 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Houston | 18 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Illinois | 20 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Indiana | 17 | 4 | 0 | 1 | `db_player_join_unresolved` |
| Iowa | 15 | 8 | 0 | 0 | `offense_applicability_unproven` |
| Iowa State | 48 | 17 | 0 | 1 | `db_position_conflict` |
| Jacksonville State | 21 | 14 | 0 | 2 | `db_source_team_uncovered` |
| James Madison | 37 | 23 | 0 | 3 | `db_source_team_uncovered` |
| Kansas | 31 | 7 | 0 | 1 | `db_player_join_unresolved` |
| Kansas State | 26 | 8 | 0 | 0 | `offense_applicability_unproven` |
| Kennesaw State | 29 | 10 | 0 | 1 | `db_player_join_unresolved` |
| Kent State | 11 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Kentucky | 31 | 9 | 0 | 1 | `db_player_join_unresolved` |
| LSU | 41 | 12 | 2 | 0 | `offense_usage_join_unresolved` |
| Liberty | 26 | 11 | 0 | 2 | `db_source_team_uncovered` |
| Louisiana | 5 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Louisiana Tech | 11 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Louisville | 33 | 9 | 0 | 0 | `offense_applicability_unproven` |
| Marshall | 28 | 10 | 0 | 0 | `offense_applicability_unproven` |
| Maryland | 14 | 2 | 0 | 0 | `offense_applicability_unproven` |
| Massachusetts | 21 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Memphis | 51 | 14 | 0 | 3 | `db_source_team_uncovered` |
| Miami | 12 | 2 | 0 | 0 | `offense_applicability_unproven` |
| Miami (OH) | 16 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Michigan | 17 | 3 | 0 | 1 | `db_player_join_unresolved` |
| Michigan State | 28 | 11 | 0 | 1 | `db_source_team_uncovered` |
| Middle Tennessee | 19 | 13 | 0 | 2 | `db_source_team_uncovered` |
| Minnesota | 19 | 3 | 0 | 2 | `db_source_team_uncovered` |
| Mississippi State | 27 | 10 | 0 | 1 | `db_player_join_unresolved` |
| Missouri | 30 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Missouri State | 19 | 7 | 0 | 1 | `db_source_team_uncovered` |
| NC State | 20 | 3 | 0 | 0 | `offense_applicability_unproven` |
| Nebraska | 17 | 4 | 0 | 1 | `db_player_join_unresolved` |
| Nevada | 14 | 6 | 0 | 0 | `offense_applicability_unproven` |
| New Mexico | 14 | 4 | 0 | 0 | `offense_applicability_unproven` |
| New Mexico State | 25 | 13 | 0 | 1 | `db_source_team_uncovered` |
| North Carolina | 19 | 6 | 0 | 0 | `offense_applicability_unproven` |
| North Dakota State | 3 | 2 | 0 | 0 | `offense_applicability_unproven` |
| North Texas | 49 | 15 | 0 | 2 | `db_source_team_uncovered` |
| Northern Illinois | 7 | 6 | 0 | 1 | `db_source_team_uncovered` |
| Northwestern | 17 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Notre Dame | 7 | 1 | 0 | 0 | `offense_applicability_unproven` |
| Ohio | 19 | 10 | 0 | 0 | `offense_applicability_unproven` |
| Ohio State | 17 | 1 | 0 | 0 | `offense_applicability_unproven` |
| Oklahoma | 16 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Oklahoma State | 55 | 14 | 0 | 1 | `db_player_join_unresolved` |
| Old Dominion | 15 | 9 | 0 | 0 | `offense_applicability_unproven` |
| Ole Miss | 28 | 4 | 0 | 2 | `db_player_join_unresolved` |
| Oregon | 13 | 2 | 0 | 0 | `offense_applicability_unproven` |
| Oregon State | 20 | 11 | 0 | 0 | `offense_applicability_unproven` |
| Penn State | 38 | 8 | 1 | 1 | `db_position_conflict` |
| Pittsburgh | 16 | 5 | 0 | 1 | `db_source_team_uncovered` |
| Purdue | 29 | 8 | 0 | 1 | `db_player_join_unresolved` |
| Rice | 20 | 3 | 0 | 2 | `db_position_conflict` |
| Rutgers | 15 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Sacramento State | 26 | 8 | 0 | 3 | `db_source_team_uncovered` |
| Sam Houston | 20 | 8 | 0 | 1 | `db_player_join_unresolved` |
| San Diego State | 26 | 14 | 0 | 0 | `offense_applicability_unproven` |
| San José State | 17 | 8 | 0 | 2 | `db_source_team_uncovered` |
| South Alabama | 12 | 6 | 0 | 0 | `offense_applicability_unproven` |
| South Carolina | 25 | 9 | 0 | 0 | `offense_applicability_unproven` |
| South Florida | 44 | 11 | 0 | 0 | `offense_applicability_unproven` |
| Southern Miss | 38 | 17 | 0 | 1 | `db_source_team_uncovered` |
| Stanford | 6 | 2 | 0 | 0 | `offense_applicability_unproven` |
| Syracuse | 19 | 2 | 0 | 0 | `offense_applicability_unproven` |
| TCU | 12 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Temple | 23 | 7 | 0 | 0 | `offense_applicability_unproven` |
| Tennessee | 21 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Texas | 22 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Texas A&M | 19 | 5 | 0 | 0 | `offense_applicability_unproven` |
| Texas State | 18 | 7 | 0 | 1 | `db_player_join_unresolved` |
| Texas Tech | 22 | 4 | 0 | 1 | `db_source_team_uncovered` |
| Toledo | 33 | 21 | 0 | 1 | `db_source_team_uncovered` |
| Troy | 14 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Tulane | 20 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Tulsa | 21 | 5 | 0 | 0 | `offense_applicability_unproven` |
| UAB | 38 | 12 | 0 | 2 | `db_player_join_unresolved` |
| UCF | 32 | 10 | 0 | 2 | `db_source_team_uncovered` |
| UCLA | 42 | 12 | 0 | 1 | `db_player_join_unresolved` |
| UConn | 53 | 12 | 0 | 0 | `offense_applicability_unproven` |
| UL Monroe | 20 | 8 | 0 | 3 | `db_source_team_uncovered` |
| UNLV | 16 | 5 | 0 | 0 | `offense_applicability_unproven` |
| USC | 9 | 1 | 0 | 0 | `offense_applicability_unproven` |
| UTEP | 21 | 14 | 0 | 1 | `db_source_team_uncovered` |
| UTSA | 22 | 8 | 0 | 1 | `db_source_team_uncovered` |
| Utah | 17 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Utah State | 29 | 10 | 0 | 1 | `db_player_join_unresolved` |
| Vanderbilt | 18 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Virginia | 31 | 6 | 0 | 0 | `offense_applicability_unproven` |
| Virginia Tech | 28 | 6 | 0 | 2 | `db_player_join_unresolved` |
| Wake Forest | 24 | 8 | 0 | 4 | `db_player_join_unresolved` |
| Washington | 14 | 2 | 0 | 0 | `offense_applicability_unproven` |
| Washington State | 28 | 9 | 0 | 0 | `offense_applicability_unproven` |
| West Virginia | 34 | 9 | 1 | 2 | `db_player_join_unresolved` |
| Western Kentucky | 19 | 11 | 0 | 1 | `db_source_team_uncovered` |
| Western Michigan | 14 | 4 | 0 | 0 | `offense_applicability_unproven` |
| Wisconsin | 33 | 9 | 0 | 1 | `db_player_join_unresolved` |
| Wyoming | 19 | 14 | 0 | 3 | `db_source_team_uncovered` |
