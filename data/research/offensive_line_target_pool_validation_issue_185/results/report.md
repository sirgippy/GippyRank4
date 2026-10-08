# Issue 185: CFBD offensive-line target-pool validation

## Scope and frozen sample

The study compares CFBD's target-season offensive-line (OL) labels with season-specific official athletics roster positions for 40 preselected FBS team-seasons. The frozen sample covers 2009–2012, 2013–2016, 2017–2020, and 2021–2026, with ten team-seasons in each window. It includes a 2011 Idaho team-season with zero CFBD OL labels and 2025 Army with 36, to expose both tails of the observed pool size.

The draw was frozen on 2026-10-07 from the issue 183 CFBD season summary and team-conference metadata before official roster pages were inspected. Its sample, protocol, and input hashes are in `data/research/offensive_line_target_pool_validation_issue_185/`. The draw uses 3 lower-quartile, 4 middle-half, and 3 upper-quartile OL-count rows per window, balances roster-size quartiles, and includes at least eight conference labels per window. No official availability informed selection.

## Source protocol and limitations

The source inventory records one official athletics roster or season-specific roster guide per sampled season. Most sources are retrospective season archive pages viewed on 2026-10-07; several pages are hosted later than the season. A current archive page does not prove the date its content was first captured, so roster backfills and seasonal cutoffs cannot always be separated. The Central Michigan 2009 roster PDF provides a stronger contemporaneous roster-guide view; Northwestern 2016 and Florida State 2019 use postseason/media-guide material. South Carolina 2019 is supported by its season-specific official archive page with OL labels.

No raw CFBD responses or official roster downloads are copied into the repository. The builder reads the existing shared issue 183 CFBD cache offline and leaves that source corpus unchanged. The official comparison set is represented as the CFBD candidate pool plus reviewer-recorded official additions/removals; the crosswalk resolves the one documented name variation. Each unlisted CFBD candidate was reviewed as an exact official OL match against the linked source. This is a single-review reconciliation, not an independently double-coded audit, and source pages are not archived in the repository. The delta/crosswalk inputs and row-level result files make the decision trail inspectable.

## Aggregate comparison

Across the 40 rows, the current reconciliation contains 758 official OL names, 722 matched names, 9 resolved CFBD-only names, 36 official-only names, and 4 unresolved CFBD names. Resolved micro-precision is 98.8%; resolved micro-recall is 95.3%. Counts match exactly in 26/40 team-seasons; complete player sets match in 25/40. Unresolved names are excluded from precision and exact-set success, but not hidden.

### Era metrics

| Era | Teams | CFBD OL | Official OL | Matched | CFBD-only | Official-only | Unresolved | Precision | Recall | Exact counts | Exact sets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009-2012 | 10 | 163 | 178 | 157 | 5 | 21 | 1 | 96.9% | 88.2% | 6 | 5 |
| 2013-2016 | 10 | 182 | 189 | 181 | 0 | 8 | 1 | 100.0% | 95.8% | 6 | 6 |
| 2017-2020 | 10 | 185 | 192 | 185 | 0 | 7 | 0 | 100.0% | 96.4% | 6 | 6 |
| 2021-2026 | 10 | 205 | 199 | 199 | 4 | 0 | 2 | 98.0% | 100.0% | 8 | 8 |

## Player-level mismatch findings

The mismatch taxonomy is in `mismatch_taxonomy.csv`; `player_comparison.csv` preserves the name-level comparison, source link, and review note. The directly documented cases include:

- **Idaho, 2011:** CFBD has no identifiable OL labels; the official roster identifies 15 offensive linemen. This is a severe target-pool omission and changes 105 potential within-pool player pairs (15 choose 2).
- **Central Michigan, 2009:** CFBD labels three players as OL whom the official guide labels defensive line (Aaron Kaczmarski, Aaron McCord, Cody Pettit), while it omits three listed OL (Allen Ollenburger, Anthony Quinn, Rocky Weaver). The total headcount remains 17, but six player identities differ; the CFBD pool adds 45 unsupported possible pairs and misses 45 official possible pairs.
- **Miami, 2010:** The official roster's Jon Feliciano is the same player as CFBD's Jonathan Feliciano (manual name crosswalk). Jeremy Lewis is listed as defensive line. Shane McDermott is not on the retrieved 2010 roster page and remains an unresolved season-membership/backfill case. The official list also includes Cory White, omitted by CFBD.
- **East Carolina, 2009:** Robert Jones is labeled defensive line on the official roster but appears in CFBD's OL pool. The official archive also makes clear why position-vocabulary handling must include OL/TE compound roles rather than exact-match one label.
- **Other omissions:** the review records official OL omitted by CFBD at UAB 2009, Texas Tech 2014, Toledo 2015, Wake Forest 2014, Georgia 2016, Illinois 2017, Texas 2018, Tulane 2018, and Oklahoma State 2020. Details and official position labels are attached to each player row.
- **Old Dominion, 2025:** CFBD includes Cameron Hill, who is absent from the linked official season roster. Because the source does not establish his official position or season membership, the case remains unresolved.
- **San José State, 2025:** CFBD includes 24 OL candidates while the official roster lists 19 OL. Four CFBD candidates (Gafa Faga, Mata Hola, Quincy Likio, Tangata Tuitupou) are explicitly labeled defensive line by the official source; Reggie Jones Jr. is absent and remains unresolved. The four resolved position-label errors alone yield 82 unsupported possible pairs in the resolved pool.

The main mismatch mechanisms observed are position-label disagreement, official OL missing from the CFBD pool, roster-season membership ambiguity, and name formatting. The sample is not a census and should not be interpreted as season-level prevalence without weighting; its stratification intentionally oversamples both extremes.

## Descriptive pair-pool implications

The pair columns in the team and era summaries apply combinations n choose 2 to the resolved roster pools. They describe how many possible pairs would be admitted or omitted if any two players in a target-season pool could be considered together. They do not estimate actual shared snaps, starts, continuity, or a model effect. The biggest observed case is Idaho 2011: all 105 official pairs are absent from CFBD's zero-player OL pool. Central Michigan 2009 keeps the same pool size but its three false labels and three omissions replace six of the 17 official player identities, adding 45 unsupported and missing 45 official possible pairs.

## Recommendation

Treat the CFBD OL target-season roster as a useful candidate pool, not an authoritative roster. Preserve its raw response; normalize explicit OL position labels separately; add source-season and position-label coverage checks; and keep official roster evidence in a reviewable validation layer. Flag empty or unusually large team-season pools, and do not interpret player-pair counts as playing continuity without separate participation evidence. Resolve the four unresolved player-season cases (Miami's Shane McDermott, Toledo's Jordan Fair, Old Dominion's Cameron Hill, and San José State's Reggie Jones Jr.) before using this study to justify any production change. This research does not alter inference, resume projection, model code, or production data.

## Reproducibility

Run from the repository root with the issue 183 raw corpus available:

```bash
uv run python scripts/freeze_issue_185_target_pool_sample.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check
uv run python scripts/build_issue_185_ol_roster_validation.py --raw-root ~/.cache/gippyrank/research-data/raw/cfbd/offensive_line_shared_roster_issue_183 --check
```

The build manifest stores SHA-256 hashes for the frozen sample/protocol inputs, manual roster review inputs, and normalized CFBD source file. The raw cache is outside the repository.

## Build input hashes

- `data/processed/offensive_line_shared_roster_issue_183/normalized_roster_player_seasons.csv.gz`: `5d3a09ba9ee979db095086ce1eaca852807efe6d6289f6fe66cdb32f50772a6a`
- `data/research/offensive_line_target_pool_validation_issue_185/frozen_sample.csv`: `d0d998e40694de7a28cd3b009a56cf1f1de7ed1ef8fb58423aac5db68e680190`
- `data/research/offensive_line_target_pool_validation_issue_185/identity_crosswalk.csv`: `d43705e1fc5d87706872da8eeb63c79772d8938f0866de849f09dc6cda1c14a2`
- `data/research/offensive_line_target_pool_validation_issue_185/manual_roster_deltas.csv`: `efa4262996ec61fd3b4eded59fd06d1b61f81458a33755c3c3ca81d2052b3f94`
- `data/research/offensive_line_target_pool_validation_issue_185/official_source_inventory.csv`: `a45505156ece119a5908f75afca8dfe2921be97423c6dab3dd60a7942c545fef`
