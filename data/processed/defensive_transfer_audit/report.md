# Defensive transfer experience and production audit (issue 103)

## Recommendation

Outcome B: carry `transfer_in_prior_defensive_experience_sum` as a conservative recorded-defensive-box-score-game-rate candidate and carry `transfer_in_prior_defensive_impact_sum` as a separate optional position-normalized production candidate. Direct defensive snaps are unavailable; both features require the strict fail-closed missing-data rule documented here.

This is a feature-construction and data-quality study only. It does not fit a Context variant, compare NLL/CRPS, select weights from outcomes, or combine the candidates with D5.

## Source inventory

| Source | Fields | Player identity | Participation | Production | Coverage | Timing / limitation |
|---|---|---|---|---|---|---|
| CFBD /player/portal | season, firstName, lastName, origin, destination, position, transferDate | no player ID in the portal response; name + origin fallback | none | none | 2021 onward in the repository acquisition workflow | retrospective response; final destination is not proven as-of cutoff |
| CFBD /roster?year=&classification=fbs\|fcs | id, firstName, lastName, team, position | CFBD athlete ID | position identity only | none | season-wide FBS and FCS roster snapshots for requested years | retrospective roster response; used for prior-season identity and position |
| CFBD /games/players?year=&week=&classification=fbs\|fcs&seasonType=both | game/team defensive rows: TOT, SOLO, SACKS, TFL, PD, QB HUR; interception INT; fumble REC | CFBD athlete ID | recorded defensive box-score games; no snaps or observed participation | box-score event counts, not play-level opportunity-adjusted impact | week-level FBS and FCS game box scores for requested seasons | completed prior-season outcomes; safe for a prior-season feature after frozen retrieval |
| CFBD /player/usage | usage.overall and offensive usage components | CFBD athlete ID, not shared with portal | not a defensive snap measure | not used | existing transfer-oracle acquisition seasons | offensive participation only; excluded from this feature |

CFBD provides no defensive snaps or defensive snap share in these endpoints. `/player/usage.overall` is an offensive participation measure and is not substituted for defense. The fallback therefore measures recorded defensive box-score games, not observed defensive participation.

## Frozen definitions

- `prior_defensive_box_score_game_rate`: recorded defensive box-score games divided by all source-team FBS/FCS games present in the frozen `/games/players` inputs. This is a conservative experience proxy, never snap share or observed participation.
- `transfer_in_prior_defensive_experience_sum`: sum of that recorded-box-score-game rate over incoming defensive transfers. A team feature is numeric only when every known defensive incoming transfer is resolved or has zero recorded defensive box-score games; otherwise it is missing.
- `prior_defensive_impact`: equal-weight mean of `z(log1p(component))`, standardized within prior season × defensive position group. Components are frozen as DL/EDGE = tackles, TFL, sacks, QB hurries; LB = tackles, TFL, sacks, passes defended; DB = tackles, passes defended, interceptions.
- `transfer_in_prior_defensive_impact_sum`: sum of prior-player impact values under the same strict missingness rule. Experience and impact are never collapsed.
- `experience_mass_coverage_proxy`: resolved prior defensive experience mass divided by the recoverable prior defensive experience mass among source-team/player records that could be joined. This is explicitly a recoverable-denominator proxy, not full-population coverage.

## Field availability audit

| Field | Availability | Measurement | Decision |
|---|---|---|---|
| defensive snaps | unavailable | no snap-count field in selected CFBD endpoints | do not label the fallback as snap share |
| defensive snap share | unavailable | no defensive denominator or participation percentage | use recorded defensive box-score game rate only |
| games played | proxy available | distinct team games containing the player's recorded defensive box-score row | frozen experience proxy numerator |
| starts | unavailable | not present in roster or games/players responses | not used |
| defensive play participation | unavailable | no player defensive-play count in selected endpoint | not used |
| tackles / solo tackles / tackles for loss / sacks | available | games/players defensive category TOT, SOLO, TFL, SACKS | position-specific impact components where declared |
| quarterback hurries | available | games/players defensive category QB HUR | DL/EDGE impact component |
| passes defended | available | games/players defensive category PD | LB/DB impact component |
| interceptions | available | games/players interceptions category INT | DB impact component |
| forced fumbles | unavailable | no player forced-fumble field in selected endpoint | not used |
| fumble recoveries | available | games/players fumbles category REC; retained as audit data | not in the frozen impact composite |
| position / player ID / source team / season | available | roster identity plus game-player and portal season fields | join and position taxonomy inputs |

## Position taxonomy

DL / EDGE includes DL, EDGE, DE, DT, and NT; LB includes OLB, ILB, MLB, and LB; DB includes CB, DB, S, FS, SS, and NB. Known offensive/special-teams labels are non-defensive. Unknown or hybrid labels remain visible and fail closed; they are never silently assigned.

## Identity join and missing-data behavior

Portal records have no shared athlete ID with the roster/game-player sources. The deterministic fallback is normalized player name + normalized source team. Explicit aliases are supported; no fuzzy matching is used. Ambiguous joins, position mismatches, missing source seasons, and unknown positions remain unresolved. A roster player with complete team-game coverage but zero recorded defensive box-score games is a verified zero-record state, not a failed identity join.

## Coverage by season

| Season | Incoming FBS | Defensive | Experience resolved | Experience-mass proxy | Impact resolved | Zero recorded box-score games | Zero-game impact resolved | Identity fail | Ambiguous | Source unavailable | Position mismatch | Unknown position | Complete teams | Partial teams |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 816 | 362 | 166 | 0.941 | 166 | 72 | 72 | 70 | 1 | 30 | 23 | 51 | 35 | 69 |
| 2022 | 1089 | 484 | 266 | 0.922 | 266 | 81 | 81 | 87 | 6 | 14 | 30 | 18 | 47 | 70 |
| 2023 | 1418 | 633 | 386 | 0.925 | 386 | 87 | 87 | 107 | 1 | 9 | 43 | 21 | 42 | 76 |
| 2024 | 2066 | 984 | 614 | 0.952 | 614 | 159 | 159 | 123 | 4 | 27 | 57 | 0 | 33 | 96 |
| 2025 | 2905 | 1380 | 827 | 0.946 | 827 | 229 | 229 | 179 | 5 | 69 | 71 | 2 | 25 | 108 |

## Coverage by defensive position group

| Season | Group | Transfers | Experience resolved | Experience-mass proxy | Impact resolved | Zero recorded box-score games | Zero-game impact resolved | Identity fail | Ambiguous | Source unavailable | Position mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | dl_edge | 110 | 56 | 0.976 | 56 | 20 | 20 | 18 | 0 | 5 | 11 |
| 2021 | lb | 86 | 33 | 0.856 | 33 | 17 | 17 | 23 | 1 | 5 | 7 |
| 2021 | db | 166 | 77 | 0.954 | 77 | 35 | 35 | 29 | 0 | 20 | 5 |
| 2022 | dl_edge | 168 | 85 | 0.880 | 85 | 36 | 36 | 28 | 1 | 5 | 13 |
| 2022 | lb | 118 | 61 | 0.892 | 61 | 19 | 19 | 20 | 4 | 4 | 10 |
| 2022 | db | 198 | 120 | 0.967 | 120 | 26 | 26 | 39 | 1 | 5 | 7 |
| 2023 | dl_edge | 213 | 124 | 0.914 | 124 | 23 | 23 | 46 | 0 | 2 | 18 |
| 2023 | lb | 136 | 87 | 0.884 | 87 | 21 | 21 | 11 | 1 | 3 | 13 |
| 2023 | db | 284 | 175 | 0.959 | 175 | 43 | 43 | 50 | 0 | 4 | 12 |
| 2024 | dl_edge | 369 | 207 | 0.918 | 207 | 57 | 57 | 61 | 0 | 13 | 31 |
| 2024 | lb | 177 | 120 | 0.968 | 120 | 25 | 25 | 18 | 1 | 3 | 10 |
| 2024 | db | 438 | 287 | 0.973 | 287 | 77 | 77 | 44 | 3 | 11 | 16 |
| 2025 | dl_edge | 559 | 313 | 0.912 | 313 | 94 | 94 | 81 | 1 | 26 | 44 |
| 2025 | lb | 219 | 136 | 0.930 | 136 | 32 | 32 | 30 | 0 | 10 | 11 |
| 2025 | db | 602 | 378 | 0.980 | 378 | 103 | 103 | 68 | 4 | 33 | 16 |

## Distribution diagnostics

The machine-readable `distribution_diagnostics.csv` contains count, mean, median, standard deviation, p05/p25/p50/p75/p95, maximum, zero fraction, and missing count by season and position group for player fields, plus team-season aggregates for both candidates.

| Season | Group | Feature | N | Mean | Median | SD | P95 | Max | Zero fraction | Missing |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | dl_edge | defensive_box_score_game_rate | 2316 | 0.323 | 0.143 | 0.365 | 1.000 | 1.000 | 0.435 | 0 |
| 2020 | dl_edge | defensive_impact | 2316 | -0.000 | -0.463 | 0.885 | 1.882 | 3.537 | 0.000 | 0 |
| 2020 | lb | defensive_box_score_game_rate | 2048 | 0.328 | 0.143 | 0.379 | 1.000 | 1.000 | 0.447 | 0 |
| 2020 | lb | defensive_impact | 2048 | -0.000 | -0.492 | 0.864 | 1.963 | 3.161 | 0.000 | 0 |
| 2020 | db | defensive_box_score_game_rate | 2759 | 0.340 | 0.167 | 0.379 | 1.000 | 1.000 | 0.416 | 0 |
| 2020 | db | defensive_impact | 2759 | 0.000 | -0.418 | 0.861 | 1.851 | 3.269 | 0.000 | 0 |
| 2021 | dl_edge | defensive_box_score_game_rate | 2795 | 0.320 | 0.154 | 0.341 | 0.929 | 1.000 | 0.272 | 0 |
| 2021 | dl_edge | defensive_impact | 2795 | 0.000 | -0.417 | 0.908 | 1.933 | 3.157 | 0.000 | 0 |
| 2021 | lb | defensive_box_score_game_rate | 2442 | 0.320 | 0.143 | 0.354 | 1.000 | 1.000 | 0.278 | 0 |
| 2021 | lb | defensive_impact | 2442 | -0.000 | -0.399 | 0.868 | 1.906 | 3.470 | 0.000 | 0 |
| 2021 | db | defensive_box_score_game_rate | 3454 | 0.321 | 0.154 | 0.342 | 1.000 | 1.000 | 0.250 | 0 |
| 2021 | db | defensive_impact | 3454 | 0.000 | -0.348 | 0.866 | 1.873 | 3.064 | 0.000 | 0 |
| 2022 | dl_edge | defensive_box_score_game_rate | 4437 | 0.208 | 0.077 | 0.306 | 0.917 | 1.000 | 0.485 | 0 |
| 2022 | dl_edge | defensive_impact | 4437 | 0.000 | -0.564 | 0.915 | 2.168 | 3.894 | 0.000 | 0 |
| 2022 | lb | defensive_box_score_game_rate | 3745 | 0.202 | 0.077 | 0.303 | 0.923 | 1.000 | 0.476 | 0 |
| 2022 | lb | defensive_impact | 3745 | -0.000 | -0.507 | 0.887 | 2.206 | 4.302 | 0.000 | 0 |
| 2022 | db | defensive_box_score_game_rate | 5383 | 0.216 | 0.083 | 0.300 | 0.917 | 1.000 | 0.434 | 0 |
| 2022 | db | defensive_impact | 5383 | -0.000 | -0.435 | 0.841 | 1.956 | 3.500 | 0.000 | 0 |
| 2023 | dl_edge | defensive_box_score_game_rate | 3087 | 0.310 | 0.143 | 0.338 | 0.923 | 1.000 | 0.268 | 0 |
| 2023 | dl_edge | defensive_impact | 3087 | 0.000 | -0.451 | 0.910 | 1.914 | 3.136 | 0.000 | 0 |
| 2023 | lb | defensive_box_score_game_rate | 2595 | 0.294 | 0.133 | 0.340 | 1.000 | 1.000 | 0.287 | 0 |
| 2023 | lb | defensive_impact | 2595 | -0.000 | -0.394 | 0.877 | 1.982 | 3.424 | 0.000 | 0 |
| 2023 | db | defensive_box_score_game_rate | 4020 | 0.297 | 0.154 | 0.329 | 0.933 | 1.000 | 0.267 | 0 |
| 2023 | db | defensive_impact | 4020 | 0.000 | -0.329 | 0.827 | 1.743 | 2.910 | 0.000 | 0 |
| 2024 | dl_edge | defensive_box_score_game_rate | 3199 | 0.307 | 0.143 | 0.340 | 0.929 | 1.000 | 0.267 | 0 |
| 2024 | dl_edge | defensive_impact | 3199 | 0.000 | -0.445 | 0.911 | 1.913 | 3.548 | 0.000 | 0 |
| 2024 | lb | defensive_box_score_game_rate | 2567 | 0.298 | 0.143 | 0.346 | 1.000 | 1.000 | 0.277 | 0 |
| 2024 | lb | defensive_impact | 2567 | 0.000 | -0.387 | 0.870 | 1.911 | 3.508 | 0.000 | 0 |
| 2024 | db | defensive_box_score_game_rate | 4136 | 0.297 | 0.154 | 0.334 | 1.000 | 1.000 | 0.263 | 0 |
| 2024 | db | defensive_impact | 4136 | -0.000 | -0.295 | 0.824 | 1.772 | 2.891 | 0.000 | 0 |

## Experience versus impact correlation

| Level | N | Pearson r |
|---|---|---|
| player | 48983 | 0.886 |
| team_season | 226 | 0.874 |

These correlations are descriptive redundancy diagnostics, not feature-selection results.

## Spot checks

| Check | Season | Player | Origin | Destination | Position | Experience status | Impact status | Recorded box-score games | Team games | Box-score game rate | Impact | Implausible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high_experience | 2021 | Caleb Biggers | Bowling Green | Boise State | CB | resolved | resolved | 5 | 5 | 1.000 | 0.439 | False |
| high_experience | 2021 | Karon Prunty | Kansas | South Carolina | CB | resolved | resolved | 9 | 9 | 1.000 | 1.938 | False |
| high_experience | 2021 | Weston Kramer | Northern Illinois | Indiana | DT | resolved | resolved | 6 | 6 | 1.000 | 0.703 | False |
| light_experience | 2025 | Joseph Mupoyi | Penn State | North Carolina | EDGE | resolved | resolved | 1 | 16 | 0.062 | -0.542 | False |
| light_experience | 2025 | Kody Huisman | North Dakota State | Virginia Tech | DL | resolved | resolved | 1 | 16 | 0.062 | -0.619 | False |
| light_experience | 2025 | Jaray Bledsoe | Texas | Mississippi State | DL | resolved | resolved | 1 | 16 | 0.062 | -0.542 | False |
| strong_impact_modest_experience | 2023 | Shamari Simmons | Austin Peay | Arizona State | CB | resolved | resolved | 5 | 11 | 0.455 | 2.077 | False |
| strong_impact_modest_experience | 2025 | Ben Bell | Texas State | Virginia Tech | EDGE | resolved | resolved | 4 | 13 | 0.308 | 1.762 | False |
| strong_impact_modest_experience | 2024 | Aamir Hall | UAlbany | Michigan | CB | resolved | resolved | 5 | 15 | 0.333 | 1.693 | False |
| high_experience_modest_impact | 2021 | Darel Middleton | Tennessee | West Virginia | DL | resolved | resolved | 9 | 10 | 0.900 | 0.514 | False |
| high_experience_modest_impact | 2021 | Damir Faison | East Carolina | Kent State | EDGE | resolved | resolved | 8 | 9 | 0.889 | 0.222 | False |
| high_experience_modest_impact | 2021 | Jared Reed | Utah State | Boise State | S | resolved | resolved | 3 | 6 | 0.500 | 0.360 | False |
| unresolved | 2021 | Colby Burton | McNeese | Hawai'i | CB | source_data_unavailable | source_data_unavailable | n/a | n/a | n/a | n/a | False |
| unresolved | 2021 | Devin Aupiu | Notre Dame | UCLA | LB | identity_resolution_failure | identity_resolution_failure | n/a | n/a | n/a | n/a | False |
| unresolved | 2021 | Alec Bryant | Virginia Tech | Illinois | EDGE | position_mismatch | position_mismatch | 0 | 11 | 0.000 | -0.604 | False |

## Limitations and decision

The experience proxy is not snap share; a recorded defensive row can undercount special packages and can differ by source box-score completeness. Production is opportunity- and scheme-dependent, normalized only within season and broad position group, and does not include forced fumbles because CFBD does not expose a player forced-fumble field in the selected source. Portal destinations and historical endpoint responses are retrospective rather than archived as-of snapshots.

The candidate definitions are frozen for a later held-out incremental-value experiment. No predictive claim is made here.

## Artifacts

- `prior_player_season_features.csv` — defensive player-season aggregates and derived values.
- `transfer_player_audit.csv` — every portal row with scope, position, join, status, and derived values.
- `team_season_features.csv` — strict incoming experience and impact candidates with coverage flags.
- `coverage_by_season.csv`, `coverage_by_position.csv`, `distribution_diagnostics.csv`, `correlations.json`, `spot_checks.csv` — audit diagnostics.
- `position_mapping.json`, `source_inventory.json`, `cutoff_safety.json`, `source_manifest.json`, `summary.json` — frozen configuration and provenance.
