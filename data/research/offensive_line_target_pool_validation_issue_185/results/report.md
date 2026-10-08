# Issue 185: CFBD offensive-line target-pool validation

## Scope and frozen sample

This bounded study compares the target-season offensive-line (OL) pools for the same 40 frozen FBS team-seasons. The sample remains ten team-seasons in each of four windows: 2009–2012, 2013–2016, 2017–2020, and 2021–2026. The frozen draw includes 2011 Idaho with zero CFBD OL labels and 2025 Army with 36 candidates to expose both observed tails. No new team-seasons were added.

The sample and protocol were frozen on 2026-10-07 before official roster inspection. The draw uses 3 lower-quartile, 4 middle-half, and 3 upper-quartile CFBD-count rows per window, balances roster-size quartiles, and includes at least eight conference labels per window. The sample and freeze metadata remain unchanged.

## Independent source comparison

`official_ol_rosters.csv` records the OL names transcribed from the single official season roster or roster guide linked for each sample row. That set is read independently by the builder; CFBD names do not initialize it. The two pools are reconciled only after both are loaded. `cfbd_candidate_reviews.csv` records CFBD candidates explicitly excluded by the official source or retained as unresolved, and `identity_crosswalk.csv` documents the Miami Feliciano name variation. `player_comparison.csv` includes each player's official membership, CFBD OL membership, source URL, and underlying CFBD roster position/status.

Most official sources are retrospective season archive pages; the source inventory records timing and limitations. Their current contents do not always establish the exact date of the roster snapshot. Central Michigan 2009 has a roster-guide PDF; Northwestern 2016 and Florida State 2019 use postseason or media-guide material. This remains a single-review manual study, not a generalized roster scraper or a double-coded audit. No source corpus, CFBD raw response, production model, or Context data was changed.

## Player and team-season results

The 40 official pools contain 755 players; CFBD supplies 735 OL candidates. The reconciliation finds 717 matched players, 14 resolved CFBD-only players, 38 official-only players, and 4 unresolved CFBD candidates. Resolved micro-precision is 98.1%; micro-recall against the independent official pools is 95.0%. Exact roster counts match in 22/40 team-seasons, and exact player sets match in 20/40. 20/40 team-seasons have at least one player discrepancy: 8 with one and 12 with multiple. The four unresolved cases are excluded from precision and counted as mismatches for exact-set and team-season reporting.

Absolute roster-count differences (difference: team-seasons) are 0: 22, 1: 9, 2: 6, 4: 1, 5: 1, 14: 1.

### Era metrics

| Era | Teams | CFBD OL | Official OL | Matched | CFBD-only | Official-only | Unresolved | Precision | Recall | Mismatched teams | Mismatch rate | Exact counts | Exact sets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009-2012 | 10 | 163 | 173 | 152 | 10 | 21 | 1 | 93.8% | 87.9% | 9 | 90.0% | 3 | 1 |
| 2013-2016 | 10 | 182 | 189 | 181 | 0 | 8 | 1 | 100.0% | 95.8% | 4 | 40.0% | 6 | 6 |
| 2017-2020 | 10 | 185 | 194 | 185 | 0 | 9 | 0 | 100.0% | 95.4% | 5 | 50.0% | 5 | 5 |
| 2021-2026 | 10 | 205 | 199 | 199 | 4 | 0 | 2 | 98.0% | 100.0% | 2 | 20.0% | 8 | 8 |

## Official-only omission mechanisms

The 38 official-only names classify against the underlying same-team, same-season FBS CFBD roster as follows: absent_from_roster: 31, present_at_another_position: 6, present_with_unknown_or_blank_position: 1. A player present at another position is distinct from a player with a blank/unknown position, and both differ from a name absent from the CFBD roster entirely. These counts are in the player table and mismatch taxonomy.

The clearest coverage failure is Idaho 2011: the independent official pool has 14 OL players while the CFBD OL pool is empty. Thirteen official players are absent from the CFBD roster entirely and Matt Cleveland is present with a blank/unknown position. The corrected official pair pool is 91 (14 choose 2), all missed by the empty CFBD pool. The source lists Spencer Beale at TE; he is not counted as an official OL.

South Carolina 2019 has 19 official OL players and 17 CFBD OL candidates. Will Rogers and M.J. Webb are both official OL but appear in the CFBD roster at DL, so they are official-only with `present_at_another_position`; the former 17/17 set match was incorrect. The official pool has 171 possible pairs, of which the 17 matched players cover 136, leaving 35 missed pairs. Iowa State 2009 also changes despite an unchanged count: Carter Bykowski is listed at TE, while official OL Mike Knapp is absent from the CFBD roster.

Other confirmed CFBD candidates at non-OL positions include UConn's Andreas Knappe (DL) and Minnesota's Ernie Heifort (TE); CFBD candidates Rennick Bryan (UConn) and Chris Freeman (Missouri) are not listed in their linked official roster sources. These cases show why a candidate-seeded gold set cannot detect omissions or false inclusions by itself.

## Descriptive pair-pool implications

Summed over the 40 team-seasons, the resolved CFBD candidate pools imply 6789 possible within-pool pairs, the official pools imply 7031, and the matched names imply 6541 shared pairs. This leaves 248 candidate-only pairs and 490 missed official pairs. These sums are descriptive n-choose-2 pool counts; they do not measure actual co-occurrence, starts, or playing continuity. Team-level pair changes are available in `team_season_summary.csv`.

## Decision: Stop

Stop using the existing CFBD roster corpus alone to define the target OL cohort for the full shared-roster continuity experiment. Overall player recall is 95.0%, but 38 official OL are absent from the CFBD OL candidate pools, including 31 whose names are absent from the underlying CFBD roster, and 20 of 40 sampled team-seasons have at least one discrepancy. Count guards can catch an empty pool such as Idaho, but they cannot identify partial player omissions in otherwise ordinary-sized pools. The frozen sample is stratified rather than prevalence-weighted, so these rates are descriptive; the observed failure modes still establish that a CFBD-only cohort is not complete enough for the full experiment.

No scalable, narrowly defined CFBD-only repair is supported for the absent-player cases. A position normalizer cannot restore player rows that are absent from the corpus, and promoting DL/TE rows based on a manual list would not generalize. For exploratory work, coverage guards should mark empty or unavailable team-season pools as uncovered and keep unknown-position rows distinct from absent players. Those checks are useful but insufficient for a full experiment because they will not detect partial omissions in otherwise ordinary-sized pools. A future full experiment needs an independently sourced roster-coverage layer or a demonstrated source-specific import repair before cohort construction.

The four pre-existing ambiguous cases—Miami's Shane McDermott, Toledo's Jordan Fair, Old Dominion's Cameron Hill, and San José State's Reggie Jones Jr.—remain unresolved and are not prerequisites for continuing exploratory research. They are excluded from resolved precision/recall and are not used to justify the Stop recommendation. This study makes no production model, Context, or data-panel changes.

## Reproducibility

Run from the repository root with the issue 183 raw corpus available:

```bash
uv run python scripts/freeze_issue_185_target_pool_sample.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check
uv run python scripts/build_issue_185_ol_roster_validation.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check
```

The build manifest hashes the frozen sample, official source inventory, independent official OL transcriptions, candidate review decisions, identity crosswalk, and normalized CFBD cache. The raw cache remains outside the repository.

## Build input hashes

- `data/processed/offensive_line_shared_roster_issue_183/normalized_roster_player_seasons.csv.gz`: `5d3a09ba9ee979db095086ce1eaca852807efe6d6289f6fe66cdb32f50772a6a`
- `data/research/offensive_line_target_pool_validation_issue_185/cfbd_candidate_reviews.csv`: `4f2c32a142ac4287b52e0ed61fab50f8981f816258a2e2c67d73f2e929c3ff64`
- `data/research/offensive_line_target_pool_validation_issue_185/frozen_sample.csv`: `d0d998e40694de7a28cd3b009a56cf1f1de7ed1ef8fb58423aac5db68e680190`
- `data/research/offensive_line_target_pool_validation_issue_185/identity_crosswalk.csv`: `d43705e1fc5d87706872da8eeb63c79772d8938f0866de849f09dc6cda1c14a2`
- `data/research/offensive_line_target_pool_validation_issue_185/official_ol_rosters.csv`: `c63142f1f51ba1dc25fbb7ec70a05cbc27d36d8a92c6b4c0a960b4885013d500`
- `data/research/offensive_line_target_pool_validation_issue_185/official_source_inventory.csv`: `a45505156ece119a5908f75afca8dfe2921be97423c6dab3dd60a7942c545fef`
