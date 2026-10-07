# Issue #181: offensive-line co-start acquisition pilot

## Decision

**Continue cautiously.** Official season tables were workable across five complete end-to-end cases spanning large and smaller programs and the 2014, 2019, and 2023 seasons. The fixed 40-team-season sample has only eight verified complete matrices, though, and two of 29 target-roster links remain unresolved. That is enough to justify a curated modeling panel, but not a national archive build.

This ticket now contains the end-to-end evidence needed for that decision. A second feasibility pilot is not needed to perform the work already requested here.

## Frozen-sample coverage

The 40-row sample and its membership remain frozen. Auditing the source documents changed two earlier “complete” labels:

- The 2010 BYU guide contains a table-of-contents reference to “Game-by-Game Starters,” but the OL matrix itself was not verified. BYU 2009 is counted as not located.
- The TCU 2014 final notes end with the regular-season game against Iowa State and omit the Alamo Bowl. They contain 12 of 13 games and are partial.
- The December 2023 Air Force bowl notes also omit the Armed Forces Bowl (12/13). The 2024 Air Force media guide supplies the complete 13-game matrix, including James Madison.

After those corrections, the sample contains eight complete matrices (20%), seven partial tables (17.5%), and 25 team-seasons where a complete matrix was not located (62.5%). “Not located” is not evidence that no source exists.

| Window | Complete | Partial | Not located | Team-seasons |
| --- | ---: | ---: | ---: | ---: |
| 2008–09 | 0 | 0 | 10 | 10 |
| 2014–15 | 2 | 3 | 5 | 10 |
| 2019–20 | 2 | 1 | 7 | 10 |
| 2023–24 | 4 | 3 | 3 | 10 |
| **Total** | **8** | **7** | **25** | **40** |

The complete-table discovery rate is a result for this stratified frozen sample, not an estimate for all FBS seasons. The search targeted official media guides, game notes, final releases, and roster pages; it was not exhaustive.

## Source types

The best source found for each of the 15 team-seasons with at least a partial table breaks down as follows. The Air Force 2023 bowl notes are an additional partial source for the same season; the 2024 guide is the complete source selected for that case.

| Source type | Complete | Partial | Team-seasons with a table |
| --- | ---: | ---: | ---: |
| Media guide | 4 | 0 | 4 |
| Official game notes | 2 | 6 | 8 |
| Final game notes | 0 | 1 | 1 |
| Spring prospectus | 1 | 0 | 1 |
| Final release | 1 | 0 | 1 |
| **Total** | **8** | **7** | **15** |

Complete examples include the [2015 Ohio State Football Guide](https://dxbhsrqyrr690.cloudfront.net/sidearm.nextgen.sites/ohiostatebuckeyes.com/images/2018/07/2015_guide.pdf), [2015 Toledo Spring Football Prospectus](https://utrockets.com/documents/download/2015/3/18/3309309_2015toledospringfootballprospectus.pdf), [2020 Clemson Football Media Guide](https://data.clemsontigers.com/pdf/football/2020-21/MediaGuide.pdf), [2024 Liberty Game 1 notes](https://libertyflames.com/documents/download/2024/8/26/2024_Football_Notes_-_Game_1_-_Campbell.pdf), and [2024 Air Force Football Media Guide](https://goairforcefalcons.com/documents/download/2024/7/8/2024_Air_Force_Football_Media_Guide.pdf).

## End-to-end normalized sample

Five complete matrices were normalized and identity-linked: Ohio State 2014, Toledo 2014, Clemson 2019, Liberty 2023, and Air Force 2023. The four newly transcribed matrices add 56 games and 280 OL position-start rows to the 14-game, 70-row Liberty slice, for **70 games and 350 position starts** total. The sample covers two large programs (Ohio State, Clemson) and three smaller programs (Toledo, Liberty, Air Force).

| Prior → target season | Prior games | Unique prior OL starters | Target roster present / absent / unresolved | Observed eligible pairs | Largest shared-start count |
| --- | ---: | ---: | ---: | ---: | --- |
| Ohio State 2014 → 2015 | 15 | 5 | 4 / 1 / 0 | 6 | 15/15 (100%) |
| Toledo 2014 → 2015 | 13 | 6 | 1 / 5 / 0 | 0 | — |
| Clemson 2019 → 2020 | 15 | 6 | 2 / 4 / 0 | 1 | 1/15 (6.7%) |
| Liberty 2023 → 2024 | 14 | 5 | 2 / 3 / 0 | 1 | 14/14 (100%) |
| Air Force 2023 → 2024 | 13 | 7 | 0 / 5 / 2 | 0 | — |

The source labels resolve to **29 unique player identities**. By unique player, 26/29 (89.7%) were normalized from a unique surname and three (10.3%) needed a manual correction. There were no exact full-name labels (0/29, 0%) because the game tables use surnames. After the manual checks, ambiguity and unresolved identity were both 0/29 (0%). Across the 31 distinct source-label rows, 28 were normalized and three were manually corrected; none remained ambiguous or unresolved.

The three manual aliases were:

- Toledo source label **Liskowski** → Robert Lisowski.
- Clemson source label **Anchrun** → Tremayne Anchrum.
- Air Force source label **Heistand** → Mark Hiestand.

Target-roster evidence resolved 27/29 players (93.1%): nine present and 18 absent. Two Air Force alternate starters (Mason Carlan and Mark Hiestand) remain unresolved because the selected July 2024 preseason notes list primary starters lost, not the full target roster. Those two are excluded from eligible-pair counts.

## Roster boundaries

Every target roster source used in the pair analysis has an explicit boundary:

- **Ohio State 2015:** the 2015 football guide is a true preseason source; it shows Taylor Decker, Pat Elflein, Jacoby Boren, and Billy Price returning. Darryl Baldwin is the one departure.
- **Toledo 2015:** the official 2015 roster page is a retrospective season roster, not a preseason snapshot. It lists Storm Norton and omits the other five prior starters. Treat the target membership result as lower-confidence than the preseason-guide cases.
- **Clemson 2020:** the 2020 football media guide is a true preseason guide. Jackson Carman and Matt Bockhorst return; four other prior starters do not appear on the 2020 roster.
- **Liberty 2024:** the official 2024 roster web page is retrospective. The dated Aug. 26, 2024 Game 1 notes are preferred: they include a 2024 depth chart and alphabetical roster before the Aug. 31 opener. Those notes list Jordan White and Xavior Gray and do not list Chase Mitchell, Jonathan Graham, or X’Zauvea Gadlin on the 2024 roster. The retrospective page corroborates the two positive listings.
- **Air Force 2024:** the July 8, 2024 media-day notes are a true preseason source. They name five prior offensive linemen on the lost-starters list: Adam Karas, Wesley Ndago, Thor Paglialong, Ethan Jackman, and Kaleb Holcomb. The notes do not establish the status of the two alternate starters, so those remain unresolved.

## Manually inspected continuity examples

- **High continuity — Ohio State:** four returning players form six observed pairs, each sharing all 15 starts. Billy Price and Pat Elflein both start throughout the year and switch between left and right guard after the early games, illustrating that a player’s position can change while the pair remains stable.
- **High continuity — Liberty:** center Jordan White and right tackle Xavior Gray both return and share all 14 of their 2023 starts (14/14).
- **Low continuity — Clemson:** Jackson Carman and Matt Bockhorst are the only returning players in this prior starter set. They share the Wofford start and no other game (1/15).
- **Low continuity — Toledo:** Storm Norton is the only 2014 starter found on the retrospective 2015 roster; no eligible pair remains.
- **Low continuity — Air Force:** the preseason notes identify all five primary 2023 OL starters as lost. No pair can be formed from confirmed returners; two alternate starters are unresolved.

These cases include both a position change (Ohio State guard rotation) and departures at the next-season boundary (Toledo and Air Force).

## Failure modes and effort

Common failure modes were stale or incomplete game notes that stop before a bowl, contents-page matches mistaken for an actual table, source spelling variants, and target rosters whose timing or completeness is weaker than the prior-season matrix. The Air Force and Liberty examples also show why a “starter lost” list or a retrospective web roster should not be treated as a complete preseason roster without checking its scope.

The measured intervals in manual_effort.csv total **18.4 minutes**. They are timestamped active-work intervals rounded to 0.1 minute, not a complete person-hours estimate.

| Major step | Measured minutes | Measurement limit |
| --- | ---: | --- |
| Source acquisition | 2.4 | Only the timed discovery/download batch and Air Force URL correction; the original search pass and some later lookups were untimed |
| PDF extraction and page review | 8.1 | Includes focused extraction and table-completeness checks |
| Manual matrix transcription | 6.2 | Four newly transcribed matrices; 56 games |
| Identity resolution | 0.3 | Final alias audit only; initial identity lookups were not separately timed |
| Target-roster resolution | 1.4 | Boundary/status verification only; initial roster lookups were not separately timed |

The initial Liberty search pass was not timed. No precision has been reconstructed for that work or for other untimed lookups. The extraction and transcription figures are measured subsets, not the full labor required to reproduce the search.

## Reproduction and data files

Run `uv run python scripts/build_issue_181_ol_co_start_pilot.py` to rebuild the 40-row coverage table, window and source-type summaries, 350 normalized position starts, identity and target-roster summaries, and eligible co-start pairs.

Reproducible inputs and outputs are under `data/processed/offensive_line_co_start_pilot_issue_181/`. Raw source PDFs are reacquirable cache files kept outside the repository. Their source URLs, cache-relative paths, byte counts, and SHA-256 hashes remain in `data/raw/offensive_line_co_start_pilot_issue_181/manifest.csv`. Resolve `cache_relative_path` under `GIPPYRANK_RESEARCH_DATA_ROOT` when the external files are present; the builder and normal CI use the checked-in manual transcriptions and derived inputs and do not require the PDFs. This remains a bounded acquisition assessment; it adds no scraper framework, national archive, or production model.
