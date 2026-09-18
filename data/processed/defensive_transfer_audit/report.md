# Defensive transfer experience and production audit (issue 103)

## Recommendation

Outcome B: carry `transfer_in_prior_defensive_experience_sum` as a conservative defensive-game-appearance-rate candidate and carry `transfer_in_prior_defensive_impact_sum` as a separate optional position-normalized production candidate. Direct defensive snaps are unavailable; both features require the strict fail-closed missing-data rule documented here.

This is a feature-construction and data-quality study only. It does not fit a Context variant, compare NLL/CRPS, select weights from outcomes, or combine the candidates with D5.

## Source inventory

| Source | Fields | Player identity | Participation | Production | Coverage | Timing / limitation |
|---|---|---|---|---|---|---|
| CFBD /player/portal | season, firstName, lastName, origin, destination, position, transferDate | no player ID in the portal response; name + origin fallback | none | none | 2021 onward in the repository acquisition workflow | retrospective response; final destination is not proven as-of cutoff |
| CFBD /roster?year=&classification=fbs\|fcs | id, firstName, lastName, team, position | CFBD athlete ID | position identity only | none | season-wide FBS and FCS roster snapshots for requested years | retrospective roster response; used for prior-season identity and position |
| CFBD /games/players?year=&week=&classification=fbs\|fcs&seasonType=both | game/team defensive rows: TOT, SOLO, SACKS, TFL, PD, QB HUR; interception INT; fumble REC | CFBD athlete ID | games with a defensive box-score row; no snaps or snap share | box-score event counts, not play-level opportunity-adjusted impact | week-level FBS and FCS game box scores for requested seasons | completed prior-season outcomes; safe for a prior-season feature after frozen retrieval |
| CFBD /player/usage | usage.overall and offensive usage components | CFBD athlete ID, not shared with portal | not a defensive snap measure | not used | existing transfer-oracle acquisition seasons | offensive participation only; excluded from this feature |

CFBD provides no defensive snaps or defensive snap share in these endpoints. `/player/usage.overall` is an offensive participation measure and is not substituted for defense. The fallback therefore measures recorded defensive box-score game appearances.

## Frozen definitions

- `prior_defensive_experience`: defensive box-score game appearances divided by all source-team FBS/FCS games present in the frozen `/games/players` inputs. This is a participation proxy, never snap share.
- `transfer_in_prior_defensive_experience_sum`: sum of that rate over incoming defensive transfers. A team feature is numeric only when every known defensive incoming transfer is resolved or a legitimate zero; otherwise it is missing.
- `prior_defensive_impact`: equal-weight mean of `z(log1p(component))`, standardized within prior season × defensive position group. Components are frozen as DL/EDGE = tackles, TFL, sacks, QB hurries; LB = tackles, TFL, sacks, passes defended; DB = tackles, passes defended, interceptions.
- `transfer_in_prior_defensive_impact_sum`: sum of prior-player impact values under the same strict missingness rule. Experience and impact are never collapsed.
- `experience_mass_coverage_proxy`: resolved prior defensive experience mass divided by the recoverable prior defensive experience mass among source-team/player records that could be joined. This is explicitly a recoverable-denominator proxy, not full-population coverage.

## Field availability audit

| Field | Availability | Measurement | Decision |
|---|---|---|---|
| defensive snaps | unavailable | no snap-count field in selected CFBD endpoints | do not label the fallback as snap share |
| defensive snap share | unavailable | no defensive denominator or participation percentage | use defensive box-score game appearance rate only |
| games played | proxy available | distinct team games containing the player's defensive box-score row | frozen experience proxy numerator |
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

Portal records have no shared athlete ID with the roster/game-player sources. The deterministic fallback is normalized player name + normalized source team. Explicit aliases are supported; no fuzzy matching is used. Ambiguous joins, position mismatches, missing source seasons, and unknown positions remain unresolved. A roster player with complete team-game coverage but no defensive row is a legitimate zero, not a failed identity join.

## Coverage by season

| Season | Incoming FBS | Defensive | Experience resolved | Experience-mass proxy | Impact resolved | Legitimate zero | Identity fail | Ambiguous | Source unavailable | Position mismatch | Unknown position | Complete teams | Partial teams |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 816 | 362 | 166 | 0.941 | 166 | 72 | 70 | 1 | 30 | 23 | 51 | 35 | 69 |
| 2022 | 1089 | 484 | 266 | 0.922 | 266 | 81 | 87 | 6 | 14 | 30 | 18 | 47 | 70 |
| 2023 | 1418 | 633 | 386 | 0.925 | 386 | 87 | 107 | 1 | 9 | 43 | 21 | 42 | 76 |
| 2024 | 2066 | 984 | 614 | 0.952 | 614 | 159 | 123 | 4 | 27 | 57 | 0 | 33 | 96 |
| 2025 | 2905 | 1380 | 827 | 0.946 | 827 | 229 | 179 | 5 | 69 | 71 | 2 | 25 | 108 |

## Coverage by defensive position group

| Season | Group | Transfers | Experience resolved | Experience-mass proxy | Impact resolved | Legitimate zero | Identity fail | Ambiguous | Source unavailable | Position mismatch |
|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | dl_edge | 110 | 56 | 0.976 | 56 | 20 | 18 | 0 | 5 | 11 |
| 2021 | lb | 86 | 33 | 0.856 | 33 | 17 | 23 | 1 | 5 | 7 |
| 2021 | db | 166 | 77 | 0.954 | 77 | 35 | 29 | 0 | 20 | 5 |
| 2022 | dl_edge | 168 | 85 | 0.880 | 85 | 36 | 28 | 1 | 5 | 13 |
| 2022 | lb | 118 | 61 | 0.892 | 61 | 19 | 20 | 4 | 4 | 10 |
| 2022 | db | 198 | 120 | 0.967 | 120 | 26 | 39 | 1 | 5 | 7 |
| 2023 | dl_edge | 213 | 124 | 0.914 | 124 | 23 | 46 | 0 | 2 | 18 |
| 2023 | lb | 136 | 87 | 0.884 | 87 | 21 | 11 | 1 | 3 | 13 |
| 2023 | db | 284 | 175 | 0.959 | 175 | 43 | 50 | 0 | 4 | 12 |
| 2024 | dl_edge | 369 | 207 | 0.918 | 207 | 57 | 61 | 0 | 13 | 31 |
| 2024 | lb | 177 | 120 | 0.968 | 120 | 25 | 18 | 1 | 3 | 10 |
| 2024 | db | 438 | 287 | 0.973 | 287 | 77 | 44 | 3 | 11 | 16 |
| 2025 | dl_edge | 559 | 313 | 0.912 | 313 | 94 | 81 | 1 | 26 | 44 |
| 2025 | lb | 219 | 136 | 0.930 | 136 | 32 | 30 | 0 | 10 | 11 |
| 2025 | db | 602 | 378 | 0.980 | 378 | 103 | 68 | 4 | 33 | 16 |

## Distribution diagnostics

The machine-readable `distribution_diagnostics.csv` contains count, mean, median, standard deviation, p05/p25/p50/p75/p95, maximum, zero fraction, and missing count by season and position group for player fields, plus team-season aggregates for both candidates.

| Season | Group | Feature | N | Mean | Median | SD | P95 | Max | Zero fraction | Missing |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | dl_edge | defensive_game_appearance_rate | 1308 | 0.573 | 0.600 | 0.306 | 1.000 | 1.000 | 0.000 | 0 |
| 2020 | dl_edge | defensive_impact | 1308 | -0.000 | -0.141 | 0.851 | 1.566 | 2.797 | 0.000 | 0 |
| 2020 | lb | defensive_game_appearance_rate | 1133 | 0.593 | 0.636 | 0.320 | 1.000 | 1.000 | 0.000 | 0 |
| 2020 | lb | defensive_impact | 1133 | -0.000 | -0.217 | 0.838 | 1.571 | 2.312 | 0.000 | 0 |
| 2020 | db | defensive_game_appearance_rate | 1610 | 0.583 | 0.600 | 0.323 | 1.000 | 1.000 | 0.000 | 0 |
| 2020 | db | defensive_impact | 1610 | 0.000 | -0.107 | 0.830 | 1.527 | 2.511 | 0.000 | 0 |
| 2021 | dl_edge | defensive_game_appearance_rate | 2035 | 0.439 | 0.333 | 0.328 | 1.000 | 1.000 | 0.000 | 0 |
| 2021 | dl_edge | defensive_impact | 2035 | -0.000 | -0.316 | 0.897 | 1.751 | 2.767 | 0.000 | 0 |
| 2021 | lb | defensive_game_appearance_rate | 1763 | 0.443 | 0.308 | 0.346 | 1.000 | 1.000 | 0.000 | 0 |
| 2021 | lb | defensive_impact | 1763 | -0.000 | -0.330 | 0.858 | 1.718 | 3.005 | 0.000 | 0 |
| 2021 | db | defensive_game_appearance_rate | 2590 | 0.428 | 0.300 | 0.332 | 1.000 | 1.000 | 0.000 | 0 |
| 2021 | db | defensive_impact | 2590 | 0.000 | -0.350 | 0.856 | 1.698 | 2.682 | 0.000 | 0 |
| 2022 | dl_edge | defensive_game_appearance_rate | 2286 | 0.403 | 0.273 | 0.321 | 0.923 | 1.000 | 0.000 | 0 |
| 2022 | dl_edge | defensive_impact | 2286 | -0.000 | -0.304 | 0.896 | 1.816 | 2.875 | 0.000 | 0 |
| 2022 | lb | defensive_game_appearance_rate | 1962 | 0.386 | 0.231 | 0.324 | 1.000 | 1.000 | 0.000 | 0 |
| 2022 | lb | defensive_impact | 1962 | -0.000 | -0.384 | 0.874 | 1.824 | 3.146 | 0.000 | 0 |
| 2022 | db | defensive_game_appearance_rate | 3049 | 0.381 | 0.250 | 0.310 | 0.923 | 1.000 | 0.000 | 0 |
| 2022 | db | defensive_impact | 3049 | 0.000 | -0.260 | 0.788 | 1.571 | 2.662 | 0.000 | 0 |
| 2023 | dl_edge | defensive_game_appearance_rate | 2261 | 0.424 | 0.308 | 0.328 | 0.933 | 1.000 | 0.000 | 0 |
| 2023 | dl_edge | defensive_impact | 2261 | -0.000 | -0.301 | 0.898 | 1.749 | 2.735 | 0.000 | 0 |
| 2023 | lb | defensive_game_appearance_rate | 1850 | 0.413 | 0.250 | 0.337 | 1.000 | 1.000 | 0.000 | 0 |
| 2023 | lb | defensive_impact | 1850 | -0.000 | -0.384 | 0.869 | 1.825 | 2.933 | 0.000 | 0 |
| 2023 | db | defensive_game_appearance_rate | 2946 | 0.406 | 0.273 | 0.323 | 1.000 | 1.000 | 0.000 | 0 |
| 2023 | db | defensive_impact | 2946 | 0.000 | -0.209 | 0.791 | 1.565 | 2.516 | 0.000 | 0 |
| 2024 | dl_edge | defensive_game_appearance_rate | 2344 | 0.419 | 0.304 | 0.334 | 1.000 | 1.000 | 0.000 | 0 |
| 2024 | dl_edge | defensive_impact | 2344 | -0.000 | -0.294 | 0.900 | 1.733 | 3.132 | 0.000 | 0 |
| 2024 | lb | defensive_game_appearance_rate | 1856 | 0.413 | 0.250 | 0.344 | 1.000 | 1.000 | 0.000 | 0 |
| 2024 | lb | defensive_impact | 1856 | 0.000 | -0.362 | 0.860 | 1.759 | 3.024 | 0.000 | 0 |
| 2024 | db | defensive_game_appearance_rate | 3050 | 0.402 | 0.250 | 0.329 | 1.000 | 1.000 | 0.000 | 0 |
| 2024 | db | defensive_impact | 3050 | -0.000 | -0.263 | 0.788 | 1.583 | 2.514 | 0.000 | 0 |

## Experience versus impact correlation

| Level | N | Pearson r |
|---|---|---|
| player | 32043 | 0.851 |
| team_season | 226 | 0.800 |

These correlations are descriptive redundancy diagnostics, not feature-selection results.

## Spot checks

| Check | Season | Player | Origin | Destination | Position | Experience status | Impact status | Def games | Team games | Experience | Impact | Implausible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high_experience | 2021 | Caleb Biggers | Bowling Green | Boise State | CB | resolved | resolved | 5 | 5 | 1.000 | 0.069 | False |
| high_experience | 2021 | Karon Prunty | Kansas | South Carolina | CB | resolved | resolved | 9 | 9 | 1.000 | 1.358 | False |
| high_experience | 2021 | Weston Kramer | Northern Illinois | Indiana | DT | resolved | resolved | 6 | 6 | 1.000 | 0.246 | False |
| light_experience | 2025 | Joseph Mupoyi | Penn State | North Carolina | EDGE | resolved | resolved | 1 | 16 | 0.062 | -0.799 | False |
| light_experience | 2025 | Kody Huisman | North Dakota State | Virginia Tech | DL | resolved | resolved | 1 | 16 | 0.062 | -0.888 | False |
| light_experience | 2025 | Jaray Bledsoe | Texas | Mississippi State | DL | resolved | resolved | 1 | 16 | 0.062 | -0.799 | False |
| strong_impact_modest_experience | 2025 | Ben Bell | Texas State | Virginia Tech | EDGE | resolved | resolved | 4 | 13 | 0.308 | 1.405 | False |
| strong_impact_modest_experience | 2023 | Shamari Simmons | Austin Peay | Arizona State | CB | resolved | resolved | 5 | 11 | 0.455 | 1.394 | False |
| strong_impact_modest_experience | 2024 | Aamir Hall | UAlbany | Michigan | CB | resolved | resolved | 5 | 15 | 0.333 | 1.295 | False |
| high_experience_modest_impact | 2021 | Darel Middleton | Tennessee | West Virginia | DL | resolved | resolved | 9 | 10 | 0.900 | 0.059 | False |
| high_experience_modest_impact | 2021 | Damir Faison | East Carolina | Kent State | EDGE | resolved | resolved | 8 | 9 | 0.889 | -0.169 | False |
| high_experience_modest_impact | 2021 | Jared Reed | Utah State | Boise State | S | resolved | resolved | 3 | 6 | 0.500 | -0.036 | False |
| unresolved | 2021 | Colby Burton | McNeese | Hawai'i | CB | source_data_unavailable | source_data_unavailable | n/a | n/a | n/a | n/a | False |
| unresolved | 2021 | Devin Aupiu | Notre Dame | UCLA | LB | identity_resolution_failure | identity_resolution_failure | n/a | n/a | n/a | n/a | False |
| unresolved | 2021 | Alec Bryant | Virginia Tech | Illinois | EDGE | position_mismatch | position_mismatch | n/a | n/a | n/a | n/a | False |

## Limitations and decision

The experience proxy is not snap share; a recorded defensive row can undercount special packages and can differ by source box-score completeness. Production is opportunity- and scheme-dependent, normalized only within season and broad position group, and does not include forced fumbles because CFBD does not expose a player forced-fumble field in the selected source. Portal destinations and historical endpoint responses are retrospective rather than archived as-of snapshots.

The candidate definitions are frozen for a later held-out incremental-value experiment. No predictive claim is made here.

## Artifacts

- `prior_player_season_features.csv` — defensive player-season aggregates and derived values.
- `transfer_player_audit.csv` — every portal row with scope, position, join, status, and derived values.
- `team_season_features.csv` — strict incoming experience and impact candidates with coverage flags.
- `coverage_by_season.csv`, `coverage_by_position.csv`, `distribution_diagnostics.csv`, `correlations.json`, `spot_checks.csv` — audit diagnostics.
- `position_mapping.json`, `source_inventory.json`, `cutoff_safety.json`, `source_manifest.json`, `summary.json` — frozen configuration and provenance.
