# Issue 183: historical offensive-line shared-roster continuity

**Panel:** CFBD seasons 2004–2026; target teams are season-specific FBS members.

## Scope and decision

This research panel measures whether target-season offensive linemen shared earlier roster seasons at the same school. It does not estimate a production coefficient, infer team strength, or use game starts, games, outcomes, or the eventual starting five.

**Decision: Stop.** Do not treat this CFBD-only history as a historical OL target pool or advance it into model development. The official Vanderbilt bio for Drew Birchmeier says he moved from defensive line to offensive line in 2020, while CFBD labels his 2017–2020 roster rows as OL. CFBD also identifies no OL at 119 FBS programs in 2004 and only 18 of 120 in 2008; the official Iowa 2008 roster includes offensive linemen despite CFBD returning 33 Iowa rows and zero OL labels. These checks show that the source position field can misstate the historical target pool. Keep the ID-linked membership outputs for bounded retrospective research only; an as-of-season OL source or an independently validated target list is needed before any feature use.

## Source and acquisition

The acquisition script requests CFBD `/teams/fbs?year=T` and `/roster?year=T&classification=fbs|fcs` for each season. FBS lists define the expected team-season denominator and target roster pool. FCS roster rows are retained for same-program history when the school name maps exactly to one CFBD FBS team ID, and for transfer-ID auditing when the source player ID also appears in an FBS roster. No fuzzy team matching is used; other FCS rows remain preserved in the raw corpus but are omitted from this FBS-focused normalized panel.

Verified raw requests in this build: 69/69. Each response is stored byte-for-byte with query parameters, retrieval time, row count, and SHA-256 sidecar. The acquisition manifest contains no API credential. Source paths are relative to the selected raw-data root, so external cache locations do not depend on the checkout path.

CFBD documents historical rosters from 2004 onward and notes that player and biographical field completeness varies by season and team ([data availability](https://apinext.collegefootballdata.com/data-availability), [roster endpoint schema](https://apinext.collegefootballdata.com/api/teams)).

## FBS coverage

Across 2,909 expected FBS team-seasons, 2,909 have at least one roster row mapped to the CFBD team ID; 0 have a successful season response but no team roster rows; 564 have roster rows but no explicitly identifiable OL position. FBS roster-response seasons absent from the local acquisition are: none.

Across 284,278 FBS roster rows, ID coverage is 100.0%, name coverage is 100.0%, nonblank position coverage is 94.9%, and jersey-number coverage is 94.0%. There are 0 repeated ID/team-season rows and 0 FBS roster rows that did not map to one historical FBS team ID. Name, position, and jersey coverage are reported for every season below.

| Season | Expected FBS | Roster mapped | No identifiable OL | FBS rows | ID rate | Name rate | Position rate | Jersey rate | OL min | OL median | OL max | OL ID rate | Continuity history |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2004 | 119 | 119 | 119 | 3327 | 100.0% | 100.0% | 0.2% | 0.2% | — | — | — | — | Censored: incomplete pre-2009 history |
| 2005 | 119 | 119 | 118 | 3530 | 100.0% | 100.0% | 2.0% | 2.0% | 1 | 1.0 | 1 | 100.0% | Censored: incomplete pre-2009 history |
| 2006 | 119 | 119 | 114 | 3549 | 100.0% | 100.0% | 16.2% | 16.2% | 1 | 1.0 | 1 | 100.0% | Censored: incomplete pre-2009 history |
| 2007 | 119 | 119 | 103 | 3653 | 100.0% | 100.0% | 37.3% | 37.3% | 1 | 1.0 | 1 | 100.0% | Censored: incomplete pre-2009 history |
| 2008 | 120 | 120 | 102 | 3742 | 100.0% | 100.0% | 66.5% | 66.5% | 1 | 1.0 | 1 | 100.0% | Censored: incomplete pre-2009 history |
| 2009 | 120 | 120 | 1 | 12611 | 100.0% | 100.0% | 99.2% | 98.5% | 13 | 18.0 | 35 | 100.0% | Censored: incomplete pre-2009 history |
| 2010 | 120 | 120 | 1 | 13072 | 100.0% | 100.0% | 99.3% | 98.3% | 12 | 18.0 | 35 | 100.0% | Evaluable |
| 2011 | 120 | 120 | 1 | 13305 | 100.0% | 100.0% | 99.2% | 98.4% | 13 | 18.0 | 32 | 100.0% | Evaluable |
| 2012 | 124 | 124 | 1 | 13736 | 100.0% | 100.0% | 99.2% | 98.7% | 12 | 18.0 | 33 | 100.0% | Evaluable |
| 2013 | 125 | 125 | 1 | 14180 | 100.0% | 100.0% | 99.4% | 98.6% | 13 | 18.0 | 33 | 100.0% | Evaluable |
| 2014 | 128 | 128 | 1 | 14810 | 100.0% | 100.0% | 99.5% | 98.8% | 13 | 19.0 | 31 | 100.0% | Evaluable |
| 2015 | 128 | 128 | 1 | 15010 | 100.0% | 100.0% | 99.6% | 98.9% | 12 | 18.0 | 37 | 100.0% | Evaluable |
| 2016 | 128 | 128 | 1 | 14714 | 100.0% | 100.0% | 99.4% | 98.6% | 12 | 18.0 | 37 | 100.0% | Evaluable |
| 2017 | 130 | 130 | 0 | 14757 | 100.0% | 100.0% | 99.9% | 98.9% | 1 | 17.0 | 36 | 100.0% | Evaluable |
| 2018 | 130 | 130 | 0 | 14511 | 100.0% | 100.0% | 99.4% | 98.2% | 11 | 16.0 | 37 | 100.0% | Evaluable |
| 2019 | 130 | 130 | 0 | 15558 | 100.0% | 100.0% | 99.9% | 98.4% | 13 | 19.0 | 32 | 100.0% | Evaluable |
| 2020 | 128 | 128 | 0 | 15989 | 100.0% | 100.0% | 99.8% | 98.0% | 14 | 20.0 | 31 | 100.0% | Evaluable |
| 2021 | 130 | 130 | 0 | 15129 | 100.0% | 100.0% | 99.2% | 97.1% | 12 | 18.0 | 25 | 100.0% | Evaluable |
| 2022 | 131 | 131 | 0 | 15548 | 100.0% | 100.0% | 99.9% | 98.5% | 15 | 19.0 | 39 | 100.0% | Evaluable |
| 2023 | 133 | 133 | 0 | 15918 | 100.0% | 100.0% | 99.5% | 98.5% | 14 | 19.0 | 35 | 100.0% | Evaluable |
| 2024 | 134 | 134 | 0 | 16221 | 100.0% | 100.0% | 99.5% | 98.7% | 14 | 20.0 | 33 | 100.0% | Evaluable |
| 2025 | 136 | 136 | 0 | 15599 | 100.0% | 100.0% | 99.9% | 99.6% | 15 | 19.0 | 36 | 100.0% | Evaluable |
| 2026 | 138 | 138 | 0 | 15809 | 100.0% | 100.0% | 99.9% | 99.4% | 15 | 19.0 | 34 | 100.0% | Evaluable |

Rates use FBS roster rows as the denominator; position rate measures whether CFBD supplied a nonblank label, not whether that label is season-accurate. The complete counts, censor status, and rates are in `coverage_by_season.csv`; each expected team-season and its missingness category are in `team_season_coverage.csv`.

## Position normalization

The source position is preserved in `position_original`. Explicit labels `OL`, `OT`/`T`, `OG`/`G`, `C`/`OC`, `LT`/`RT`, `LG`/`RG`, and `IOL` map to `offensive_line`. Known non-OL labels map to `non_offensive_line`. Compound labels containing separators map to `ambiguous`; blank or unrecognized labels map to `unknown`. Neither ambiguous nor unknown positions are silently counted as OL.

| Source position | Classification | Rows |
| --- | --- | --- |
| (blank) | unknown | 16374 |
| ? | unknown | 3603 |
| ATH | non_offensive_line | 20 |
| C | offensive_line | 1303 |
| CB | non_offensive_line | 11564 |
| DB | non_offensive_line | 24900 |
| DE | non_offensive_line | 10172 |
| DL | non_offensive_line | 23042 |
| DT | non_offensive_line | 7163 |
| EDGE | non_offensive_line | 550 |
| FB | non_offensive_line | 2779 |
| G | offensive_line | 2401 |
| ILB | non_offensive_line | 768 |
| LB | non_offensive_line | 33501 |
| LS | non_offensive_line | 5260 |
| NT | non_offensive_line | 430 |
| OL | offensive_line | 38036 |
| OLB | non_offensive_line | 1014 |
| OT | offensive_line | 2387 |
| P | non_offensive_line | 5062 |
| PK | non_offensive_line | 8185 |
| PR | unknown | 6 |
| QB | non_offensive_line | 13502 |
| RB | non_offensive_line | 19539 |
| S | non_offensive_line | 12907 |
| TE | non_offensive_line | 16994 |
| WR | non_offensive_line | 39412 |

The table includes FBS target rows and relevant FCS historical OL rows; filter `source_classification=fbs` for target-season cohorts.

Across retained FBS/FCS rows, 0 rows have compound ambiguous labels and 19,983 have blank or unrecognized labels; neither category is counted as OL.

## Player identity audit

The normalized identity key is the CFBD source player ID (`cfbd:<id>`). The retained panel has 107,837 unique IDs, including 105,401 in FBS target rosters; 73,565 FBS IDs occur in multiple seasons and 72,157 have at least one adjacent-season same-program return. Across retained program rows, 16,748 IDs appear at multiple programs across seasons. Same-ID source name variants occur for 0 IDs; position or position-label variants occur for 0; duplicate normalized names within a program-season occur in 1,665 groups; exact name/team matches with changing adjacent-season IDs produce 811 unresolved candidate links and 8 ambiguous name-only links. None of the latter are merged automatically.

Same IDs appear on multiple schools in the same season in 283 cases. These are visible retrospective transfer/multi-school membership signals, not evidence of preseason availability. See `player_identity_audit.csv.gz` and `player_identity_events.csv.gz` for row-level evidence.

Returning players are linked by exact source ID within the same CFBD team ID. Position changes and source-name variants do not break an ID link, but are retained in the identity audit. Duplicate names with different IDs remain distinct. Missing IDs and possible ID changes remain unresolved.
An identity position-variant count of zero means the provider label stayed the same for each observed ID; it does not show that the label was correct for each season. The Vanderbilt biography check below finds a historical backfill case without any within-ID label variation.

## Shared-roster continuity construction

For each target FBS season T, the current roster identifies the target OL pool. A pair's prior shared seasons are the intersection of its members' earlier same-program roster seasons, restricted to years `< T` and on or after the complete-history window start. Seasons at a different school never contribute. Target-season roster membership itself never contributes to a pair score.

The team-season summaries report total/mean/maximum pairwise shared seasons; pair counts at 1/2/3 shared seasons; the largest set of target OL simultaneously present on one prior same-program roster; the share of ID-linked target OL with no prior same-program roster row observed in this panel; and aggregate prior roster seasons. These are descriptive candidates, not selected production features.

Roster coverage changes sharply before 2009: CFBD has 3,742 FBS rows in 2008 versus 12,611 in 2009. In 2009, only 4 of 2,142 identified OL have a prior same-program link. Those apparent zero links primarily reflect incomplete historical roster coverage. We treat 2009 as the earliest plausible full-roster history season and 2010 as the first target season whose continuity can be evaluated.
Targets before 2010 are censored when their history depends on incomplete pre-2009 roster coverage or on a missing requested-panel prior season. Their pair measures and continuity summaries are blank and carry `continuity_history_status`; these blanks are not zeros. An evaluated pair with no shared roster season has count `0` and status `observed_zero_shared_prior_roster_seasons_in_history_window`. The history window starts in 2009 for this build.

## Retrospective roster boundary

CFBD returns a season roster, not an archived Week 1 snapshot with an as-of date. The latest FBS roster response in this corpus was retrieved at 2026-10-07T15:02:46.374422+00:00; the 2026 response is in-season. Responses may reflect transfers or roster changes that happened during the season. The target-season roster is used only to define which players enter the pair pool; it is not treated as evidence that target-season players had already spent time together. Same-ID listings at multiple schools in one season are counted above as obvious temporal conflicts. Less visible midseason additions/departures cannot be detected from this endpoint alone. No games, starts, outcomes, or later starting-lineup knowledge enter the continuity measures.

## Frozen issue 181 sample comparison

The builder joins all 40 target team-seasons from the frozen issue 181 sample, copied from commit `806836e5769d4dd80208546c1c0f60cc07038435`. The sample rows and strata are unchanged; joins use target season and normalized canonical team name, with the explicit `Appalachian State` → `App State` alias. Pair totals include only evaluable rows. Censored rows are reported separately and their blank pair measures are not counted as zero. Row-level joins are in `issue_181_overlap_coverage.csv`.

| Window start | Sample rows | Censored rows | Evaluable rows | Evaluable with identified OL | Evaluable with no identified OL | Pairs with shared prior season |
| --- | --- | --- | --- | --- | --- | --- |
| 2008 | 10 | 10 | 0 | 0 | 0 | — |
| 2014 | 10 | 0 | 10 | 10 | 0 | 1127 |
| 2019 | 10 | 0 | 10 | 10 | 0 | 679 |
| 2023 | 10 | 0 | 10 | 10 | 0 | 671 |

## Bounded official-roster validation

5 bounded manual validation cases are recorded below.

| Case | Coverage | School | Seasons | CFBD | Official roster | Discrepancy | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | stable roster membership spot check | Clemson | 2023-2024 | 17 of 23 target-season OL IDs appear on the prior-year Clemson roster; Walker Parks, Marcus Tate, and Blake Miller retain the same CFBD IDs in both seasons. | Clemson's 2023 and 2024 official rosters list Parks, Tate, and Miller as offensive linemen. | spot_checked_membership_agreement | [official](https://data.clemsontigers.com/pdf/football/2023-24/ClemsonRoster.pdf) · [official](https://clemsontigers.com/sports/football/roster/season/2024) |
| V2 | high turnover roster sample | Colorado | 2023-2024 | Four of 15 target-season OL IDs appear on the 2023 Colorado roster; 11 have no earlier Colorado roster row in the panel, and six of 105 target-player pairs share any prior Colorado roster season. | Colorado's official 2023 and 2024 rosters corroborate the source rosters used for the comparison; the 2024 roster includes multiple new additions. | turnover_detected_not_a_position_audit | [official](https://cubuffs.com/sports/football/roster/2023) · [official](https://cubuffs.com/sports/football/roster/2024?path=football) |
| V3 | transfer identity check | Colorado | 2020-2024 | CFBD ID 4566190 lists Kahlil Benson at Indiana in 2020-2023 and Colorado in 2024; the Colorado target-season row has no prior Colorado membership. | Colorado's 2024 official roster identifies Benson as an offensive tackle who transferred from Indiana. | cross_program_history_excluded_from_same_program_overlap | [official](https://static.cubuffs.com/custompages/football/2024/roster.pdf) · [official](https://cubuffs.com/sports/football/roster/2024?path=football) |
| V4 | historical position-label check | Vanderbilt | 2017-2020 | CFBD ID 4035269 is labeled OL at Vanderbilt in 2017, 2018, 2019, and 2020. | Vanderbilt's official Drew Birchmeier biography says he moved from defensive line to offensive line in 2020. | historical_position_label_not_season_accurate | [official](https://vucommodores.com/sports/football/roster/player/drew-birchmeier) |
| V5 | older roster position coverage | Iowa | 2008 | CFBD returns 33 Iowa roster rows but zero are classified as OL; position values contain no offensive-line label. | Iowa's official 2008-09 roster includes offensive linemen. | historical_ol_pool_missing_from_provider_roster | [official](https://hawkeyesports.com/sports/football/roster/season/2008-09) |

The validation register is a bounded manual sample, not a repair queue. Discrepancies are classified and counted; the panel is not manually patched to force agreement.

## Reproduction

```bash
uv run python scripts/fetch_offensive_line_roster_history.py --start-season 2004 --end-season 2026
uv run python scripts/build_offensive_line_shared_roster_continuity.py --start-season 2004 --end-season 2026
```

Set `GIPPYRANK_DATA_DIR` to select the shared data root. When unset, it defaults to `$XDG_CACHE_HOME/gippyrank/research-data` or `~/.cache/gippyrank/research-data`; the CFBD corpus lives at `<data-root>/raw/cfbd/offensive_line_shared_roster_issue_183`. Pass `--raw-root` to select another corpus for a run. The fetch command requires `CFBD_API_KEY` only when a response is not already cached; reruns validate and reuse byte-preserved responses rather than overwriting them. The builder is offline and checks every raw payload against its provenance hash before producing compressed CSVs. Normal CI does not acquire or rebuild this external corpus. Gzip output timestamps are fixed so identical inputs produce identical artifacts.
