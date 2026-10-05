# Historical OL starter-data feasibility (issue 176)

**Recommendation: Proceed with co-starts.** Official season-review guides yielded complete, position-labeled offensive-line starter tables in this small cross-era sample. That is enough to justify a bounded co-start pilot. The evidence does not establish an FBS-wide coverage rate, and it is not yet strong enough to treat StatCrew games-played fields as complete OL participation.

## Scope and sample

This reconnaissance follows the [issue 174 review](offensive_line_continuity_174.md), which ruled out CFBD `/games/players` rows as an OL participation proxy. Source checks were made on 2026-10-05 using official school-hosted media guides, season reviews, and StatCrew HTML only. No production ingestion code, broad scrape, or OCR was built.

The seven team-seasons below are a purposive availability sample spanning the 2000s, early/mid 2010s, and a recent season; multiple conferences; and larger and smaller FBS programs. This is a source-feasibility sample, not a random set of FBS teams. “Complete” means a retrospective source lists the starting lineup for every game in that season.

| Team-season | Primary source and coverage | Full five-position starters? | StatCrew HTML/XML and participation fields | Caveat |
|---|---|---|---|---|
| Utah, 2004 (MWC) | [2005 media guide, 2004 review](https://utahutes.com/documents/download/2016/6/14/05_mg_sec10.pdf) | Yes, 12/12; explicit LT/LG/C/RG/RT | Unknown; not checked | Eleven regular-season games are in one grid; the Fiesta Bowl lineup is a separate table in the same guide. |
| Arizona, 2011 (Pac-12) | [2012 football media guide, pp. 47–48](https://static.arizonawildcats.com/old_site/pdf/m-footbl/2012-13/misc_non_event/12footballmediaguide.pdf) | Yes, 12/12; explicit LT/LG/C/RG/RT | Unknown; not checked | Retrospective guide table; names are generally abbreviated. |
| LSU, 2012 (SEC) | [2013 football media guide](https://static.lsusports.net/assets/docs/fb/pdf/13guide.pdf) | Yes, 13/13; explicit LT/LG/C/RG/RT | Unknown; not checked | One row per game, including the bowl game. |
| Utah State, 2012 (WAC) | [2013 spring prospectus, 2012 starters](https://s3.us-east-2.amazonaws.com/sidearm.nextgen.sites/utahstateaggies.com/documents/2018/6/14/12811__m_footbl_2012_13_prospectus__prospectus.pdf) | Yes, 13/13; LT/LG/**OC**/RG/RT | **Yes:** [school-hosted StatCrew HTML](https://utahstateaggies.com/sports/2018/8/1/_m_footbl_stats_2012_2013_indgbg_html.aspx) has `gp/gs` and game cells marked `START` or `XXX`. No XML located. | `OC` is the center; normalize it to C. A [UTSA-hosted gamebook](https://static.goutsa.com/stats/football/2012/game08.htm) independently lists the five USU starters for that game. |
| BYU, 2014 (independent) | [2015 football media guide](https://byucougars.com/sites/default/files/files/webalmanac.pdf) | Yes, 13/13; explicit LT/LG/C/RG/RT | Unknown; not checked | Some players appear by surname or first initial. |
| Kansas State, 2016 (Big 12) | [2017 football media guide](https://www.kstatesports.com/documents/download/2017/9/4/2017_K_State_Football_Media_Guide.pdf) | Yes, 13/13; explicit LT/LG/C/RG/RT | Unknown; not checked | Guide table has one row per game. |
| Utah, 2024 (Big 12) | [2025 football media guide, p. 54](https://utahutes.com/documents/download/2025/7/23/COMP_2025_Football_Media_Guide.pdf) | Yes, 12/12; explicit LT/LG/C/RG/RT | Unknown; not checked | End-of-year guide covers all 12 games; in-season notes are only current through games already played. |

## Findings

### Starter-table coverage and extraction

All seven retrospective samples have a complete five-position starter record: **7/7 in this selected sample**. Every source labels the five line positions directly or, for Utah State, uses `OC` for center. The tables are compact and appear amenable to deterministic transcription or table extraction with a small amount of source-specific cleanup. Utah's 2004 bowl table and Utah State's `OC` label are the notable exceptions to a direct one-table, standard-header shape.

Usable coverage is observed at least as far back as 2004. Utah's [media-guide archive](https://utahutes.com/sports/2016/6/10/sports-m-footbl-archive-utah-m-footbl-archive-html) advertises guides back to 1948, but I did not verify starter tables in those earlier guides. The 7/7 rate must not be read as an FBS-wide availability estimate: the sample is small, purposive, and centered on sources found for this review.

### StatCrew coverage and participation semantics

The only positive school-hosted StatCrew season sample was Utah State 2012. Its HTML reports all 13 opponents and gives per-player `gp/gs` totals plus per-game `START` and `XXX` values. The `gp/gs` totals reconcile with the game cells in the examples reviewed. The separate UTSA gamebook lists all five Utah State offensive line starters at their positions, matching the season report's `START` entries for that game.

This establishes that school-hosted StatCrew HTML can preserve a richer participation record than starters alone. It does not establish how often such archives fill seasons missing guide tables: none of the seven sampled seasons lacked a complete guide table, so there is no gap-season denominator. The USU HTML is overlapping evidence for a season whose starter table already exists. No XML file was located in that StatCrew sample.

The official [StatCrew football manual](https://qa.statcrew.com/photos/schools/statc/FBUG22.pdf) says players with game statistics are automatically credited with a game played, while players without statistics need a participation indicator entered manually, either at game wrap-up or later. That means `gp` can be meaningful but its completeness depends on operator entry, especially for offensive linemen who may not record a box-score statistic. A single game's starter match validates the `gs`/`START` side for that game; it does not independently validate nonstarter `XXX` entries or every game's `gp`. The fields record games, not snaps or same-play co-appearance. I would not use them as a complete shared-game measure without a larger manual comparison to gamebooks.

### Identity, panel size, and effort

Within a season, names are usually adequate to identify the starters, and gamebooks can add jersey numbers. Cross-season joins are less automatic: guide tables use surnames, initials, and abbreviations, while there is no common player ID across these documents. Utah State's two Whimpeys illustrate the issue: the prospectus distinguishes `Kev.` and `Ky.`, while its gamebook adds jersey numbers. Official rosters and player biographies should support a documented alias/jersey crosswalk, but ambiguous cases will need manual resolution. No multi-season identity-linkage rate was measured here.

A reasonable next scope is a **30–50 team-season starter panel**, with 50–100 curated team-seasons a plausible later target if source screening continues to hold. These are effort estimates, not observed coverage rates. Finding an official guide and transcribing one short table is a manageable research task; a national multi-decade census and identity resolution across every school would be a much larger, unmeasured burden. I recommend testing the co-start definition on a bounded panel before expanding collection.

## Recommendation

**Proceed with co-starts.** The observed guide tables are broad enough and structured enough to support a starter-only pilot. Keep the measure explicitly about co-starting, preserve source and missingness for each team-season, and do not describe it as shared snaps or complete game participation. Retain StatCrew `gp/gs` as a promising supplementary source; validate its nonstarter participation cells against several complete gamebooks before using it to claim shared games played.
