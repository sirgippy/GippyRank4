# Issue 185: CFBD offensive-line target-pool validation

## Scope and frozen sample

This bounded study compares the target-season offensive-line (OL) pools for the same 40 frozen FBS team-seasons. The sample remains ten team-seasons in each of four windows: 2009–2012, 2013–2016, 2017–2020, and 2021–2026. The frozen draw includes 2011 Idaho with zero CFBD OL labels and 2025 Army with 36 candidates to expose both observed tails. No new team-seasons were added.

The sample and protocol were frozen on 2026-10-07 before official roster inspection. The draw uses 3 lower-quartile, 4 middle-half, and 3 upper-quartile CFBD-count rows per window, balances roster-size quartiles, and includes at least eight conference labels per window. The sample and freeze metadata remain unchanged.

## Independent source comparison

`official_ol_rosters.csv` records the OL names transcribed from the single official season roster or roster guide linked for each sample row. That set is read independently by the builder; CFBD names do not initialize it. The two pools are reconciled only after both are loaded. `cfbd_candidate_reviews.csv` records CFBD candidates explicitly excluded by the official source or retained as unresolved, and `identity_crosswalk.csv` documents the Miami Feliciano name variation. `player_comparison.csv` includes each player's official membership, CFBD OL membership, source URL, and underlying CFBD roster position/status.

Most official sources are retrospective season archive pages; the source inventory records timing and limitations. Their current contents do not always establish the exact date of the roster snapshot. Central Michigan 2009 has a roster-guide PDF; Northwestern 2016 and Florida State 2019 use postseason or media-guide material. This remains a single-review manual study, not a generalized roster scraper or a double-coded audit. No source corpus, CFBD raw response, production model, or Context data was changed.

## Player and team-season results

The 40 official pools contain 759 players; CFBD supplies 735 OL candidates. The reconciliation finds 717 matched players, 14 resolved CFBD-only players, 42 official-only players, and 4 unresolved CFBD candidates. Resolved micro-precision is 98.1%; micro-recall against the independent official pools is 94.5%. Exact roster counts match in 22/40 team-seasons, and exact player sets match in 20/40. 20/40 team-seasons have at least one player discrepancy: 8 with one and 12 with multiple. The four unresolved cases are excluded from precision and counted as mismatches for exact-set and team-season reporting.

Absolute roster-count differences (difference: team-seasons) are 0: 22, 1: 9, 2: 6, 4: 1, 5: 1, 18: 1.

### Era metrics

| Era | Teams | CFBD OL | Official OL | Matched | CFBD-only | Official-only | Unresolved | Precision | Recall | Mismatched teams | Mismatch rate | Exact counts | Exact sets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009-2012 | 10 | 163 | 177 | 152 | 10 | 25 | 1 | 93.8% | 85.9% | 9 | 90.0% | 3 | 1 |
| 2013-2016 | 10 | 182 | 189 | 181 | 0 | 8 | 1 | 100.0% | 95.8% | 4 | 40.0% | 6 | 6 |
| 2017-2020 | 10 | 185 | 194 | 185 | 0 | 9 | 0 | 100.0% | 95.4% | 5 | 50.0% | 5 | 5 |
| 2021-2026 | 10 | 205 | 199 | 199 | 4 | 0 | 2 | 98.0% | 100.0% | 2 | 20.0% | 8 | 8 |

## Official-only omission mechanisms

The 42 official-only names classify against the underlying same-team, same-season FBS CFBD roster as follows: absent_from_roster: 35, present_at_another_position: 6, present_with_unknown_or_blank_position: 1. A player present at another position is distinct from a player with a blank/unknown position, and both differ from a name absent from the CFBD roster entirely. These counts are in the player table and mismatch taxonomy.

The clearest coverage failure is Idaho 2011: the independent official pool has 18 OL players while the CFBD OL pool is empty. Seventeen official players, including Spencer Beale, Dallas Sandberg, A.J. Jones, and Sam Tupua, are absent from the CFBD roster entirely; Matt Cleveland is present with a blank/unknown position. The official roster table and player details list all four additional players at OL. The 153 possible within-pool pairs (18 choose 2) are arithmetic combinations only and do not establish actual prior shared-roster relationships.

South Carolina 2019 has 19 official OL players and 17 CFBD OL candidates. Will Rogers and M.J. Webb are both official OL but appear in the CFBD roster at DL, so they are official-only with `present_at_another_position`; the former 17/17 set match was incorrect. The official pool has 171 possible pairs, of which the 17 matched players cover 136, leaving 35 missed pairs. Iowa State 2009 also changes despite an unchanged count: Carter Bykowski is listed at TE, while official OL Mike Knapp is absent from the CFBD roster.

Other confirmed CFBD candidates at non-OL positions include UConn's Andreas Knappe (DL) and Minnesota's Ernie Heifort (TE); CFBD candidates Rennick Bryan (UConn) and Chris Freeman (Missouri) are not listed in their linked official roster sources. These cases show why a candidate-seeded gold set cannot detect omissions or false inclusions by itself.

## Descriptive pair-pool implications

Summed over the 40 team-seasons, the resolved CFBD candidate pools imply 6789 possible within-pool pairs, the official pools imply 7093, and the matched names imply 6541 shared pairs. This leaves 248 candidate-only pairs and 552 missed official pairs. These sums are descriptive n-choose-2 roster-pool combinations; they do not identify actual prior shared-roster relationships, actual co-occurrence or starts, or the predictive value of a continuity feature. Team-level pair-pool differences are available in `team_season_summary.csv`.

## Recommendation: Proceed with bounded repair

### Exact roster reconstruction

CFBD alone cannot guarantee a complete, exact official OL cohort: 42 official OL are missing from its OL candidate sets, including 35 names absent from the underlying CFBD roster, and 20/40 sampled team-seasons have at least one discrepancy. Idaho has an empty CFBD OL pool against 18 official OL. The frozen sample is stratified rather than prevalence-weighted, so these rates describe this sample and should not be read as population estimates.

### Exploratory continuity modeling

Proceed with the existing CFBD-derived shared-roster continuity data for exploratory feature development and empirical evaluation, retaining explicit coverage and uncertainty limitations. The bounded handling is to mark unavailable or empty team-season pools as uncovered and exclude clearly unusable team-seasons from the feature calculation, while preserving unknown-position and roster-missingness information and reporting coverage. These guards will not detect every partial omission; the study does not show that such omissions erase useful continuity information. Do not manually reconstruct historical rosters to address this study's discrepancies.

### Production adoption

Production adoption remains undecided until the continuity feature is evaluated empirically. The pair-pool arithmetic above is not evidence of missing prior shared-roster relationships or predictive impact. The next research step is to exercise the full CFBD-derived feature and determine whether the observed imperfections matter.

The four pre-existing ambiguous cases—Miami's Shane McDermott, Toledo's Jordan Fair, Old Dominion's Cameron Hill, and San José State's Reggie Jones Jr.—remain unresolved and are not prerequisites for exploratory research. This study makes no production model, Context, or data-panel changes.

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
- `data/research/offensive_line_target_pool_validation_issue_185/official_ol_rosters.csv`: `72058bd4add30bac8122016af047056f968b7b089b554a6536e3181388322d25`
- `data/research/offensive_line_target_pool_validation_issue_185/official_source_inventory.csv`: `a45505156ece119a5908f75afca8dfe2921be97423c6dab3dd60a7942c545fef`
