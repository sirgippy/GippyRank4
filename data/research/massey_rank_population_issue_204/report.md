# Massey rank population correction (issue 204)

**Corpus version:** `massey-rank-population-v1`

**Scope:** Final and pre-bowl rank observations for seasons 2003–2025

**Model fitting:** Not performed

## Finding and population rule

The model population is the number of distinct teams in that season and subdivision's Massey composite (`CMP`) ranking universe. Constituent ranks are source observations: preserve their original ordinal values and compare them with that population. Do not infer population from the largest observed rank, truncate ranks, or clip and renormalize probability mass.

The original corpora had source ranks beyond the inferred population in three cohorts. The out-of-support ranks flowed into targets and same-system game pairs. Season-specific identity decisions reconcile those observations with the cohort before any target or pair is built:

| Season/cohort | Stable CFBD ID | Decision | Population before → after |
| --- | --- | --- | ---: |
| 2004 FCS | `50` Florida A&M | Include in FCS | 119 → 120 |
| 2007 FBS | `98` WKU | Include in FBS, per user decision | 119 → 120 |
| 2020 FCS | `2627` Tarleton State | Include in FCS | 125 → 127 |
| 2020 FCS | `3101` Dixie State (now Utah Tech) | Include in FCS | 125 → 127 |

The 2004 pre-bowl FBS snapshot had Florida A&M in its FBS cohort. Applying the same identity-and-season rule moves it to FCS there: the pre-bowl FBS population changes from 118 to 117, while FCS changes from 119 to 120. The final 2004 corpus already places this team in FCS. WKU was in a transition period in 2007; the cohort choice here follows the user's explicit instruction to include WKU in the 2007 FBS model population. The override is only for season 2007.

The identity decisions are implemented in `src/gippyrank/data/massey.py` and applied to the dated pre-bowl exports and cached game participants too. Official context: [WKU's 2007 transition announcement](https://wkusports.com/news/2007/3/20/2007_wku_football_schedule_highlighted_by_match_up_with_defending_national_champ), [Florida A&M's 2004 team statistics](https://static.famuathletics.com/custompages/stats/football/2004/teamstat.htm), [Tarleton's 2020 Division I/FCS announcement](https://tarletonsports.com/news/2020/7/1/general-tarleton-officially-joins-ncaa-division-i-western-athletic-conference.aspx), and [Dixie State's transition announcement](https://utahtechtrailblazers.com/news/2020/7/1/general-its-official-dixie-state-joins-the-wac-and-begins-ncaa-d-i-transition.aspx). These institutional records provide historical context; the selected model cohort is identified by source team identity and the decisions above.

## Source universe, observations, and population

The normalized source derivative contains 4,739,306 historical observations across the available source seasons. The 2003–2025 final corpus selects 393,820 observations; the pre-bowl corpus contains 413,373 observations. Each has 5,762 final/pre-bowl team-season targets. Raw ordinal ranks and source files are unchanged; the output reports record source hashes and document 2,785 absent or unranked cells in the composite exports as non-observations. The normalization report records no unmapped/ambiguous composite teams or classification conflicts.

Population is not the number of source rows or systems. It is the distinct `CMP` team count. Every one of the 46 season/subdivision populations in each corrected corpus has at least one selected constituent rank for every CMP team. The selected-system threshold remains 0.90 coverage.

## Before and after

| Corpus | Original unsupported target team-seasons | Corrected unsupported target team-seasons | Original out-of-population source rows in selected systems | Corrected |
| --- | ---: | ---: | ---: | ---: |
| Final | 19 | 0 | 135 | 0 |
| Pre-bowl | 20 | 0 | 141 | 0 |

The original source-row counts above are reported separately from PMF team-seasons. Rank-support exceptions are confined to 2004 FCS (32 rows in each corpus), 2007 FBS (92 final and 98 pre-bowl rows), and 2020 FCS (11 rows in each corpus). All 276 exception rows across both artifact versions are retained in [`source_rank_support_dispositions.csv`](source_rank_support_dispositions.csv); corrected versions have no out-of-population rows. No other cohort population memberships changed. No ranks were removed, renumbered, clipped, or probability-renormalized.

The CMP populations after correction are 120 in 2004 FCS, 120 in 2007 FBS, and 127 in 2020 FCS. The full cohort-by-cohort counts, membership changes, coverage, and target deltas are in [`cohort_before_after_accounting.csv`](cohort_before_after_accounting.csv), [`population_membership_changes.csv`](population_membership_changes.csv), and [`target_team_changes.csv`](target_team_changes.csv). The corrected support disposition is summarized in [`target_support_anomalies.csv`](target_support_anomalies.csv); it is empty apart from its header.

All corrected PMF rows pass checks for canonical identity, positive population, integer ranks within `1..N`, count/probability agreement with retained rank observations, finite nonnegative probabilities, and total mass of one. For all 363 common final targets and 480 common pre-bowl targets in affected cohorts, retained rank observations and ordinal-rank PMFs are identical before and after; the corrected population field changes the rank-percentile mean while preserving those source ranks. Added/removed team-season targets are listed in `target_team_changes.csv`.

## Same-system game inputs

The matched inputs were rebuilt from cached CFBD historical schedules after applying the same cohort decisions to home and away classifications. Each output row joins both teams' ranks only where they share the same constituent system, while retaining both ranks and population sizes.

| Corpus | Pair rows before → after | Added / removed | Original pairs with a rank outside population | Corrected |
| --- | ---: | ---: | ---: | ---: |
| Final | 2,217,875 → 2,219,003 | 1,128 / 0 | 1,254 | 0 |
| Pre-bowl | 2,335,469 → 2,336,248 | 779 / 0 | 1,325 | 0 |

The pair accounting is in [`rank_pair_accounting.csv`](rank_pair_accounting.csv). The correction audit rebuilds the corrected rows from observations and games, compares every serialized pair row with the versioned pair artifact, and records artifact hashes in its manifest.

## Handoff to issues 200 and 201

- Issue 200 should use the corrected final and pre-bowl observation/target corpora from `data/processed/massey_issue_204_v1/` and `data/processed/massey_prebowl_issue_204_v1/`, with their corresponding `final_constituent_rank_pairs.csv` and `pre_bowl_constituent_rank_pairs.csv` as matched inputs.
- The generated PMFs and rank pairs are based on the same corrected season-specific populations. Do not mix one corrected corpus with the old issue-203 pre-bowl inputs.
- Issue 201's frozen compatibility baseline remains identified by the original input hashes below. Treat results using the corrected corpora as a separately labeled comparison; this work does not rewrite or replay frozen posterior artifacts.
- No fit, posterior replay, scoring, or performance search was run here.

## Provenance and reproduction

The original inputs were read without modification. Their SHA-256 values are:

| Input | SHA-256 |
| --- | --- |
| Original final observations | `dfd9f7ebdb9c22f597d814a2d1feadfb699af3b88b74488d6655ef5e718a34d3` |
| Original final target PMFs | `03e4e372017aedb45392ff7d790cea25f461db9b6dc846514c762ff46e2cab00` |
| Original pre-bowl observations | `b0e349851de709e054ac38f31294534859aeabba0ccdb8cd0f3eea561261483a` |
| Original pre-bowl target PMFs | `701e063420fc53cd33a3e923250d60ebcc8f14aabc3376d60e634bacdad6c918` |

Corrected observation and target outputs are versioned and hashed by the manifests in the two processed directories. Key hashes:

| Corrected artifact | SHA-256 |
| --- | --- |
| Normalized all-season observations | `ddacb87068a84626e0fc8f2e23426ad6e5d631910a9fe5cc3c1a649202f785c3` |
| Final 2003–2025 observations | `cd3ee420a830ca6ca1f1a9dd4accb0e844ffd5ee0b00398b4c3536dd57218d9a` |
| Final target PMFs | `cab882da17be2f79dd379fc296107e332e2768f9827d96752ea33d97dc026f28` |
| Pre-bowl observations | `c82c1cd4250f02d06e56aa6e16d00421c59a21e23bdd58d59378f937896af58e` |
| Pre-bowl target PMFs | `6c610ea91ea60ce5da45e20baa7d636f91de52af974420f28a74eb0ec5d66e3f` |
| Final same-system rank pairs | `4dd2376a488ede05c889372089fa0e0b16d62faaf879626c50cb359d5fe4e3af` |
| Pre-bowl same-system rank pairs | `3193da2bb427a2022a6b7a293d39093e7b26e3cde36eaa0a3c912e2db4eaa917` |

The normalized dataset, PMFs, and multi-million-row pair files are generated under `data/processed/` and are gitignored by the repository. Their deterministic builders and hash manifests are included; the compact audits and source inventory are retained beside this report. To rebuild, provide the preserved source files under `data/raw/`, then run:

```bash
uv run python scripts/normalize_massey.py --raw-root data/raw --output-root data/processed/massey_issue_204_v1
uv run python scripts/build_massey_datasets.py --source data/processed/massey_issue_204_v1/observations.csv --output-root data/processed/massey_issue_204_v1
uv run python scripts/build_massey_prebowl_targets.py --raw-root data/raw --final-observations data/processed/massey_issue_204_v1/final_observations.csv --checklist data/research/massey_prebowl_issue_198/collection_checklist.csv --output-root data/processed/massey_prebowl_issue_204_v1 --report-root data/research/massey_rank_population_issue_204 --coverage-threshold 0.90 --games data/raw/cfbd/games
```

Run `uv run python scripts/audit_massey_rank_population_issue_204.py --help` for the before/after audit's required original-artifact and corrected-artifact paths. It writes the compact comparison CSVs and a hash manifest to the `data/research/massey_rank_population_issue_204/` directory.
