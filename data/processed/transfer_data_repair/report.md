# Transfer data repair audit (#148)

This is a new audit location built from frozen #141/#142 artifacts and retained player-level derivations. The baseline files and published Context outputs are unchanged.

## Repairs and audit counts

- Historical legitimate-zero aggregates repaired: **21** team-seasons.
- Team-seasons whose availability status improved: **17**.
- 2026 covered-roster DB identity repairs: **0**; unresolved cases retained: **40**.
- 2026 source-team DB mappings added: **0**; uncovered source-team cases retained: **34**.
- 2026 position conflicts changed: **0**; ambiguous DB joins changed: **0**.
- Applicable offensive usage failures remaining: **10** across 8 teams. The committed 2026 aggregate does not retain those player identities or join causes.
- Offensive applicability remains unproven for **1012** incoming transfers; no absent evidence was converted to zero.

Historical availability is rebuilt from the corrected player-level D5 audit, the retained defensive player audit, and the corrected materializer panel. Each historical team's reason codes and status are reclassified with the same rules as the #141 audit; no individual reason is removed from a frozen row. The `post_repair_observed_usage_sum` field records known resolved contributions even where failures or unknown applicability remain.

## Historical materializer feature reconciliation

The exact retained source bytes passed SHA-256 verification against `data/processed/transfer_production_audit/source_manifest.json` (15 inputs; manifest SHA-256 `b1c30ed98c8b38b8f4eb5ea57e8090858471f2c11d2879388d96bd7ace60c517`). The pinned previous oracle is commit `fc8c3924164c19fbb8b28334f613c5937c744673` with `transfer_oracle.py` SHA-256 `e2d66036102f7ac8bd11fbf95d913b688a0b283224b218db513a43b47d819c49`. Replaying that resolver exactly reproduces the frozen panel before applying the current materializer.

Comparing 8232 transfer feature cells across 2744 historical panel rows found **24 changed team-seasons** and **24 changed feature values**. The generated inventory reconciles exactly to the full before/after panel diff; no changes are unexplained.

| Change class | Team-seasons | Changed feature values |
| --- | ---: | ---: |
| `ambiguous_usage_join_removed` | 1 | 1 |
| `legitimate_zero_restoration` | 21 | 21 |
| `name_normalization_join_added` | 2 | 2 |

The 21 legitimate-zero restorations are backed by 46 D5_ZERO contributors. The materializer rejected 1 genuine ambiguous join contribution. Other measured changes are listed explicitly in the class table.

The replay resolved 24 formerly multi-row joins as duplicate-equivalent provider observations. Rows collapse only when stable player ID, season, normalized player/source identity, and numeric overall usage agree; without stable IDs, only exact parsed observations are deduplicated. Different IDs and conflicting usage values remain distinct candidates. 1 genuinely ambiguous historical usage join remains; 0 have conflicting values under the same stable ID.

One genuine ambiguous-join example is North Texas 2023: its aggregate changes from `0.7639999999999999` to `0.07400000000000001` after Chandler Rogers's previously selected contribution is rejected because distinct logical usage candidates conflict. When conflicting candidates leave no unique contribution, the materializer keeps the aggregate unknown rather than replacing it with numeric zero. `historical_feature_changes.csv` records old and new feature values, responsible portal records, candidate details, and source hashes for every changed cell.

The corrected replay changes 2021 Troy's usage aggregate from `missing` to `0.163`; its rebuilt availability row uses this numeric value and does not retain `historical_aggregate_null`.

## Baseline versus post-repair availability

| Inventory | Team-seasons | Complete | Partial | Entirely unavailable |
| --- | ---: | ---: | ---: | ---: |
| #141 baseline | 793 | 71 | 720 | 2 |
| #148 post-repair | 793 | 87 | 704 | 2 |

Historical `historical_aggregate_null` reasons fall from 73 to 51. The original #141 artifact remains intact.

## 2026 DB observed coverage

The post-repair player-level inventory contains **518/604** observed DB impacts across **138** teams: 74 teams have complete observed coverage, 57 remain partial, 1 has no observed impacts, and 6 have no incoming DB transfers.
All 132 teams with incoming DB transfers match the #142 empirical coverage panel; the other 6 Context teams are added as natural-zero `0/0` rows. Transitions: `{"unchanged": 138}`. The full per-team `n/k`, observed sum, missing count, and unchanged Context 1.3 model values are in `coverage_2026.csv`.

The new DB sum is the sum of resolved players only. `incoming_db_count`, `observed_db_impact_count`, and `observed_db_impact_sum` distinguish a partial observed contribution, no observed impacts, and a natural zero with no incoming transfers. These fields are audit inputs; Context 1.3 still reads its original three-column contract.

## Systematic investigation boundary

The 2026 source manifest describes 37 portal, usage, player-stat, roster, and game-player snapshots. The raw manifest is present: **False**; raw snapshot payloads retained: **0**; missing payloads: **37**.

The checked-in verified alias tables contain 0 team aliases and 0 player aliases, so they provide no pre-verified repair for these rows.

The processed DB audit preserves player names, source/destination teams, positions, impact statuses, and resolved prior player IDs. It does not preserve the unmatched roster candidates, portal stable IDs for unresolved records, or raw provider spellings. Therefore it cannot establish that a particular punctuation variant, team alias, or player ID repairs an unresolved row. No player-specific aliases or source-team mappings were added. The 34 source-team gaps affect 25 destination teams and remain unresolved in `unresolved_db_players_2026.csv`.

The 11 DB position conflicts have a uniquely joined player and retained portal/prior positions, but the audit does not establish whether the discrepancy is a chronology change or provider taxonomy change. No general compatibility rule is supported by the retained evidence, so all remain unresolved. The remaining ambiguous DB player identity lacks retained stable IDs or candidate records and remains ambiguous.

The 10 applicable 2026 offensive usage failures and their player-level cause are not present in the committed #141 current-team artifacts. The updated derivation now emits an `offensive_player_audit.csv` with portal/usage IDs, candidate counts, join statuses, and D5 reasons when the immutable snapshots are available. This checkout lacks those source payloads, so the 2026 failures cannot be re-derived here. Existing historical player audits show source-team mismatches, ambiguous usage matches, and absent usage rows; those unsupported rows remain unresolved.

The remaining applicability-unknown cases need additional trustworthy player participation evidence where the retained box-score and usage records are inconclusive, especially for offensive-line and non-FBS-origin players. If CFBD does not cover those populations, a new provider must be evaluated in a separate follow-up; this repair does not infer participation from position or acquire a new source.

Current source responses were captured after the 2026 August 15 cutoff. They diagnose present-day provider behavior and do not prove historical availability. Historical checkpoint classifications remain `historical_timing_unverified`.

## Safety and reproducibility

- The player identity key now normalizes Unicode compatibility forms and punctuation variants while retaining suffixes and diacritics; multiple candidates still fail closed.
- Stable portal IDs are preferred. A conflicting stable ID blocks name fallback for both offensive and defensive joins.
- `zero_contributors.csv` contains only D5 legitimate-zero players; resolved positive usage contributions continue through the normal usage join. The Context 1.3 research materializer checks each evidence index against season, normalized player and team names, transfer date, and destination team ID before passing it to the aggregator. Stable portal IDs take precedence, duplicate fallback keys and ambiguous usage matches fail closed, and explicit zero evidence cannot override positive usage.
- No Context 1.3 coefficients, feature-selection behavior, or published ranking files were regenerated.
- #141 and #142 artifacts are read-only inputs; all new artifacts are under `data/processed/transfer_data_repair/`.

Input hashes and machine-readable before/after records are in `summary.json`, `changes.csv`, and `historical_feature_changes.csv` (24 historical feature changes).
