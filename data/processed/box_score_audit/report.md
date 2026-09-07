# CFBD primitive box-score data audit

## Scope and conclusion

This is a data-validation audit only. It reads the cached CFBD `/games/teams` corpus through the shared sidecar-safe payload helper. It does not fit a model, evaluate predictive value, select features, or modify production ranking behavior.

The cache contains 42,404 unique team-game rows from 764 payload files. It contains 35 distinct raw stat categories. Exact duplicate rows are counted and retained as a provenance flag; conflicting duplicates are not silently repaired.

The cache has 4,280 exact duplicate team-game observations from overlapping requests and 0 conflicting duplicates. Candidate malformed values: `{}`; impossible values: `{'penalties': 1, 'penalty_yards': 1, 'total_yards': 3}`; extreme values under declared audit thresholds: `{'fumbles_recovered': 14, 'offensive_plays_derived': 1, 'penalties': 6, 'penalty_yards': 1, 'rushing_attempts': 1, 'rushing_yards': 79, 'sacks': 10, 'total_fumbles': 8, 'total_yards': 3}`.

The future-safe primitive representation is `total_yards`, `rushing_yards`, `rushing_attempts`, `passing_yards` (CFBD net passing yards), `pass_attempts`, `completions`, `sacks`, `interceptions_thrown`, `total_fumbles`, `fumbles_lost`, `first_downs`, `penalties`, and `penalty_yards`, with `offensive_plays_derived = rushing_attempts + pass_attempts`. Direct play count and sack-yard categories were not observed.

## Raw category inventory

The complete observed category inventory is in `category_inventory.csv`; the table below is intentionally compact.

| CFBD category | first season | training | development | modern | 2026 | formats |
|:---|---:|---:|---:|---:|---:|:---|
| `completionAttempts` | 2004 | 25291 | 7208 | 13916 | 256 | compound_pair=46671 |
| `defensiveTDs` | 2016 | 3900 | 7145 | 8214 | 252 | integer=19511 |
| `firstDowns` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `fourthDownEff` | 2004 | 25291 | 7208 | 13916 | 256 | compound_pair=46671 |
| `fumblesLost` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `fumblesRecovered` | 2004 | 25291 | 7207 | 13133 | 189 | integer=45820 |
| `interceptionTDs` | 2004 | 14948 | 3907 | 7556 | 119 | integer=26530 |
| `interceptionYards` | 2004 | 14948 | 3907 | 7556 | 119 | integer=26530 |
| `interceptions` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `kickReturnTDs` | 2005 | 16743 | 6100 | 11650 | 215 | integer=34708 |
| `kickReturnYards` | 2005 | 16743 | 6100 | 11650 | 215 | integer=34708 |
| `kickReturns` | 2005 | 16743 | 6100 | 11650 | 215 | integer=34708 |
| `kickingPoints` | 2004 | 24586 | 7081 | 13575 | 242 | integer=45484 |
| `netPassingYards` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `passesDeflected` | 2016 | 3900 | 7145 | 8214 | 252 | integer=19511 |
| `passesIntercepted` | 2004 | 14948 | 3907 | 7556 | 119 | integer=26530 |
| `passingTDs` | 2004 | 25210 | 7206 | 13914 | 254 | integer=46584 |
| `possessionTime` | 2004 | 25229 | 7191 | 13916 | 256 | time_mm_ss=46592 |
| `puntReturnTDs` | 2004 | 20746 | 5059 | 9394 | 175 | integer=35374 |
| `puntReturnYards` | 2004 | 20746 | 5059 | 9394 | 175 | integer=35374 |
| `puntReturns` | 2004 | 20746 | 5059 | 9394 | 175 | integer=35374 |
| `qbHurries` | 2016 | 3900 | 7145 | 8214 | 252 | integer=19511 |
| `rushingAttempts` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `rushingTDs` | 2004 | 25287 | 7207 | 13914 | 254 | integer=46662 |
| `rushingYards` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `sacks` | 2016 | 3900 | 7145 | 8214 | 252 | decimal=39; integer=19472 |
| `tackles` | 2016 | 3900 | 7145 | 8214 | 252 | integer=19511 |
| `tacklesForLoss` | 2016 | 3900 | 7145 | 8214 | 252 | decimal=116; integer=19395 |
| `thirdDownEff` | 2004 | 25291 | 7208 | 13916 | 256 | compound_pair=46671 |
| `totalFumbles` | 2016 | 3311 | 4362 | 11031 | 189 | integer=18893 |
| `totalPenaltiesYards` | 2004 | 25291 | 7208 | 13916 | 256 | compound_pair=46671 |
| `totalYards` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `turnovers` | 2004 | 25291 | 7208 | 13916 | 256 | integer=46671 |
| `yardsPerPass` | 2004 | 25291 | 7208 | 13916 | 256 | decimal=46665; other=6 |
| `yardsPerRushAttempt` | 2004 | 25291 | 7208 | 13916 | 256 | decimal=46665; other=6 |

Raw categories include the requested offense/turnover/first-down/penalty primitives plus basic return, possession, down-conversion, touchdown, tackle, and kicking categories. The audit does not scrape or adopt PPA/EPA, success rate, explosiveness, havoc, line yards, SP+, opponent-adjusted metrics, or other second-order analytics.

## Coverage by era and pairing

`coverage_by_season_pairing.csv` contains the complete season-by-pairing table. The compact table below reports game-level both-team support for the fields most relevant to this audit; percentages are weighted by expected games within each era/pairing.

| period | pairing | games | total yards | derived plays | sacks | INT thrown | total fumbles | fumbles lost |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| training_2004_2017 | FBS-FBS | 10147 | 99.7% | 99.7% | 15.1% | 99.7% | 11.6% | 99.7% |
| training_2004_2017 | FBS-FCS | 1285 | 98.0% | 98.0% | 16.3% | 98.0% | 12.8% | 98.0% |
| training_2004_2017 | FCS-FCS | 8096 | 0.1% | 0.1% | 0.0% | 0.1% | 0.0% | 0.1% |
| development_2018_2021 | FBS-FBS | 2850 | 100.0% | 100.0% | 98.4% | 100.0% | 51.2% | 100.0% |
| development_2018_2021 | FBS-FCS | 379 | 99.5% | 99.5% | 97.6% | 99.5% | 64.1% | 99.5% |
| development_2018_2021 | FCS-FCS | 2171 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| modern_2022_2025 | FBS-FBS | 3174 | 100.0% | 100.0% | 98.9% | 100.0% | 72.0% | 100.0% |
| modern_2022_2025 | FBS-FCS | 485 | 100.0% | 100.0% | 98.8% | 100.0% | 71.1% | 100.0% |
| modern_2022_2025 | FCS-FCS | 2663 | 98.3% | 98.3% | 0.3% | 98.3% | 69.9% | 98.3% |
| current_2026 | FBS-FBS | 50 | 50.0% | 50.0% | 50.0% | 50.0% | 30.0% | 50.0% |
| current_2026 | FBS-FCS | 48 | 31.2% | 31.2% | 31.2% | 31.2% | 20.8% | 31.2% |
| current_2026 | FCS-FCS | 73 | 67.1% | 67.1% | 65.8% | 67.1% | 38.4% | 67.1% |
| pre_training_2003 | FBS-FBS | 699 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| pre_training_2003 | FBS-FCS | 71 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| pre_training_2003 | FCS-FCS | 498 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

The table makes the pairing-specific support gap explicit: FBS–FBS and FBS–FCS are broadly supported for yards/plays in the training and development eras, while FCS–FCS is materially thinner. Sacks and especially total fumbles are much less complete; no missing value is treated as zero.

## Data dictionary

The complete machine-readable dictionary is `data_dictionary.csv`. `unsupported` means the category was not observed in the cached payloads; `usable_with_caveat` means raw values exist but coverage or semantics require explicit handling.

| proposed name | source | status | training coverage | development coverage | modern coverage | FBS-FBS | FBS-FCS | FCS-FCS |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| `completions/pass_attempts` | `completionAttempts` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `first_downs` | `firstDowns` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `fumbles_lost` | `fumblesLost` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `fumbles_recovered` | `fumblesRecovered` | `usable_with_caveat` | 58.3% | 60.8% | 93.8% | 95.4% | 93.3% | 17.5% |
| `interceptions_thrown` | `interceptions` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `offensive_plays_derived` | `rushingAttempts + completionAttempts` | `trusted_derived` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `offensive_plays_direct` | `plays/offensivePlays/totalPlays (not observed)` | `unsupported` | n/a | n/a | n/a | n/a | n/a | n/a |
| `passing_yards` | `netPassingYards` | `unresolved` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `penalties/penalty_yards` | `totalPenaltiesYards` | `usable_with_caveat` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `punts/punt_yards` | `punts/puntYards (not observed)` | `unsupported` | n/a | n/a | n/a | n/a | n/a | n/a |
| `rushing_attempts` | `rushingAttempts` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `rushing_yards` | `rushingYards` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `sack_yards` | `sackYards (not observed)` | `unsupported` | n/a | n/a | n/a | n/a | n/a | n/a |
| `sacks` | `sacks` | `usable_with_caveat` | 8.3% | 60.3% | 57.4% | 45.2% | 44.0% | 2.9% |
| `total_fumbles` | `totalFumbles` | `usable_with_caveat` | 7.0% | 31.6% | 78.6% | 28.8% | 29.3% | 13.8% |
| `total_yards` | `totalYards` | `trusted_primitive` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |
| `turnovers_reported` | `turnovers` | `usable_with_caveat` | 58.3% | 60.8% | 99.3% | 97.6% | 95.2% | 20.1% |

## Semantic change in passing yards

The `netPassingYards` label is not semantically stable across the full cache. In 2004–2011, `totalYards` frequently does not equal `rushingYards + netPassingYards`; from 2012 onward the reconciliation is essentially complete apart from a handful of anomalies. The 2004 Ohio State–Indiana official book shows the mechanism: official net passing is 189 for Indiana and 161 for Ohio State, while cached CFBD reports 238 and 164, differences of 49 and 3 that equal the official sack yards. Thus the early CFBD field behaves like pre-sack-loss passing yards despite its `netPassingYards` name. Keep the raw field, but mark its historical semantic as unresolved and do not treat all seasons as one comparable passing-yard measure without a separate era rule.

- `training_2004_2017`: 10,005/22,769 yard reconciliations mismatched (43.9%); common differences `{-8: 493, -9: 467, -7: 451, -10: 426, -6: 410, -11: 403, -12: 393, -5: 381}`.
- `development_2018_2021`: 0/6,454 yard reconciliations mismatched (0.0%); common differences `{}`.
- `modern_2022_2025`: 6/12,946 yard reconciliations mismatched (0.0%); common differences `{-1: 1, -23: 1, 18: 1, 1: 1, -2: 1, -11: 1}`.
- `current_2026`: 0/222 yard reconciliations mismatched (0.0%); common differences `{}`.
- `pre_training_2003`: 0/0 yard reconciliations mismatched (n/a); common differences `{}`.

## Yards and plays

The observed category names have no case-only spelling variants (`category_spelling_variants` is empty in `summary.json`). No direct `plays`, `offensivePlays`, or `totalPlays` category was observed. Derived plays are `rushingAttempts + pass attempts parsed from completionAttempts`; 42,386 team-game rows have this derivation. The distribution is min 0, p01 45.0, median 68.0, p99 95.0, max 151. These are data-quality summaries, not evidence for a model choice.

Derived plays equal official total plays in 8 checked team rows across 2004, 2018, 2021, and 2024. This is a semantic sanity check, not source-wide proof. The cache contains 889 games with overtime line-score periods (1,778 team rows); the maximum is 9 overtime periods. The 2021 Illinois–Penn State 9OT game is retained as an overtime case, and no overtime normalization is applied.

Total-yard reconciliation against `rushing_yards + passing_yards` is complete for 42,391 rows; 10,011 mismatch. The passing field is CFBD `netPassingYards`, not a separately available gross-passing field.

## Components, sacks, and turnovers

Rushing yards/attempts, net passing yards, completion counts, pass attempts, and sacks are retained separately in `team_game_audit.csv`. Sacks appear in the payloads but sack yards do not. Official books report sack yards in the selected sample, so future work must not invent a CFBD sack-yard field or infer it as zero.

The `interceptions` category matches the offensive INT component in the official sample and is named `interceptions_thrown` in the audit. `fumblesLost` is distinct from `totalFumbles`; total fumbles is historically intermittent and remains missing when absent. The reported `turnovers` field reconciles to interceptions thrown plus fumbles lost in 42,391 rows, with 0 mismatches, but no turnover collapse is performed. `fumblesRecovered` is also retained separately.

First downs are exposed as one total `firstDowns` category. Rushing/passing/penalty first-down components were not observed. Penalties are exposed as compound `totalPenaltiesYards` values and are parsed into count and yards without imputation.

## Official cross-checks

The fixed sample covers a 2004 cached-era low-volume FBS–FBS game in the source selection, a 2018 FBS–FCS game, the 2021 nine-overtime FBS–FBS game, and the 2024 FBS–FBS game. Official comparisons are recorded in `official_crosschecks.csv` and are intentionally not fetched at build time.

| season/game | source | verified fields | mismatches | unavailable | interpretation |
|:---|:---|---:|---:|---:|:---|
| 2004 / `242970194` | [https://ohiostatebuckeyes.com/documents/download/2023/6/30/2004-7-Indiana.pdf](https://ohiostatebuckeyes.com/documents/download/2023/6/30/2004-7-Indiana.pdf) | 16 | 8 | 10 | verified except raw CFBD mismatch; preserve raw value |
| 2018 / `401022511` | [https://static.sjsuspartans.com/Football/2018/HTML/ucd-sj.htm](https://static.sjsuspartans.com/Football/2018/HTML/ucd-sj.htm) | 27 | 0 | 7 | verified in sample |
| 2021 / `401282717` | [https://fightingillini.com/sports/football/stats/2021/penn-state/boxscore/22918](https://fightingillini.com/sports/football/stats/2021/penn-state/boxscore/22918) | 27 | 1 | 6 | verified except raw CFBD mismatch; preserve raw value |
| 2024 / `401628379` | [https://utsports.com/sports/football/stats/2024/arkansas/boxscore/28878](https://utsports.com/sports/football/stats/2024/arkansas/boxscore/28878) | 26 | 2 | 6 | verified except raw CFBD mismatch; preserve raw value |

The 2004 official sample exposes the early passing-yard semantic mismatch above and also shows first-down and penalty differences; those values are not silently corrected. The 2018 and 2021 official books agree on the audited yardage, attempts, completions, plays, sacks, INTs, first downs, and supplied penalty values. In 2024, Arkansas official total offense is 434 yards (137 rushing + 297 net passing), while cached CFBD reports 431 total and 134 rushing; Tennessee agrees. These are concrete semantic/data-quality findings, not repair instructions.

CFBD documents `/games/teams` as team box-score statistics with generic category/stat pairs: [CFBD Games API](https://apinext.collegefootballdata.com/api/games). Official references: [2004 Indiana–Ohio State game book](https://ohiostatebuckeyes.com/documents/download/2023/6/30/2004-7-Indiana.pdf), [2018 San José State box score](https://static.sjsuspartans.com/Football/2018/HTML/ucd-sj.htm), [2021 Illinois–Penn State box score](https://fightingillini.com/sports/football/stats/2021/penn-state/boxscore/22918), and [2024 Tennessee–Arkansas box score](https://utsports.com/sports/football/stats/2024/arkansas/boxscore/28878).

## Reproducibility and production boundary

The builder is offline and deterministic. It reads raw payloads without modifying them, excludes `.provenance.json` sidecars through the shared helper, emits stable sorted CSV/JSON, and leaves production CFBD processed files, Historical Likelihood V1, Posterior V1, H/C, Performance V1, YPP conclusions, the weekly updater, and the website unchanged.

Missing values remain missing; no imputation or zero-filling is performed. See `duplicates_and_quality_flags.csv` for conflicting duplicate and plausibility flags.
