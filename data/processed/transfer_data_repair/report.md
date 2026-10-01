# Integrated transfer repair audit (#150 + #151)

This canonical audit combines #150's replayed historical repair with #151's 2026 reacquired evidence. Historical feature values and availability come from the corrected #150 materializer replay; 2026 candidate and coverage evidence comes from the #151 reacquisition. Reacquired sources were captured after the historical cutoff and are not represented as historical availability evidence.

## Historical transfer audit (2021–2025)

The historical panel covers **655** team-seasons: 83 complete, 570 partial, and 2 entirely unavailable. Across the combined 2021–2026 availability inventory, the checkpoint is 87 complete, 704 partial, and 2 entirely unavailable across 793 team-seasons.
Historical aggregate-null reasons fall from 73 to 51.

The legacy replay reproduces the frozen panel: **True**. The current replay matches the materializer: **True**. The change inventory reconciles with no unexplained cells: **True**, 24 feature values across 24 team-seasons.

| Historical feature change class | Changed team-seasons / values |
| --- | ---: |
| `ambiguous_usage_join_removed` | 1 |
| `legitimate_zero_restoration` | 21 |
| `name_normalization_join_added` | 2 |

The player audit contains 14422 records. It retains 1 genuinely ambiguous usage join; 24 duplicate-equivalent usage joins collapse under #150's identity/value rules. Changed players and source evidence are listed in the historical repair artifacts.

Integration status reconciliation: the #151 pre-materializer snapshot had 88 complete, 703 partial, and 2 unavailable team-seasons; the final combined rows have 87 / 704 / 2. Comparing the pre-materializer and final historical rows found 1 status change: 2022 Texas changed from `complete` to `partial`; row evidence changed from primary reason `complete`, reason codes `none` to primary reason `offense_applicability_unproven`, reason codes `offense_applicability_unproven`. Player-audit evidence: Diamonte Tucker-Dorsey (LB): usage join `no_usage_record`, 0 usage candidate(s), applicability `cannot_determine_applicability` (defensive_or_special_portal_position; source_team_outside_verified_stats_coverage).
Comparing row-level historical aggregate-null reasons found 52 before and 51 after; removed cases: 2021 Troy (final usage `0.163`); newly added cases: none.

## 2026 reacquired evidence

Incoming DB transfers: **604**; observed impacts: **550**; unresolved impacts: **54**.

| Unresolved DB impact reason | Players |
| --- | ---: |
| `ambiguous` | 0 |
| `identity_resolution_failure` | 17 |
| `position_mismatch` | 14 |
| `source_data_unavailable` | 23 |

Among 138 teams, DB impact coverage is 94 complete, 38 partial, and 6 with no incoming DB transfers. Applicable offensive usage failures remaining: 2; offensive applicability remains unknown for 1012 transfers.
Compared with the pre-repair audit, 35 DB player impact records changed: 32 became resolved and 3 exposed position conflicts. 0 changes came solely from normalization-reference drift. Changed impact classes: `{"newly_resolved_identity_or_source_coverage": 32, "roster_position_evidence_changed": 3}`.
Defensive-impact reference comparison: 34 matching requests; 0 reference hashes changed. Supplemental DII/III and team-filtered records are candidate evidence only. Context 1.3 model-facing inputs and published rankings remain unchanged: **False** changed and **False** regenerated.
The reacquired lineage uses 134 canonical responses; 0 are missing from the selected raw root. All were retrieved after the 2026-08-15 cutoff. Manifest hashes and request details are in `before_source_manifest.json` and `after_source_manifest.json`.

## Canonical and supporting artifacts

`historical_transfer_features.csv` is the canonical historical feature panel from #150's replay/materializer parity check. `historical_feature_changes.csv`, `historical_player_repair_audit.csv`, `changes.csv`, and `zero_contributors.csv` retain the detailed #150 evidence. `db_coverage_2026.csv`, `player_before_after_2026.csv`, `team_before_after_2026.csv`, `unresolved_db_players_2026.csv`, and the before/after source manifests retain #151's 2026 evidence. `historical_zero_candidate_details.csv` preserves #151's separate zero-candidate view; it is supporting evidence and does not define the canonical post-replay panel. `db_coverage_2026_pre_reacquisition.csv`, when present, is the older coverage snapshot and is not the current 2026 result.

`team_seasons.csv` combines #150's 2021–2025 corrected availability rows with #151's 2026 reacquired rows. The #151 pre-materializer historical panel is not retained as a competing canonical panel.
