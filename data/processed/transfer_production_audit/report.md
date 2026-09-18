# Transfer-production data-quality and production-feasibility audit (issue 98)

## Recommendation

**Feasible only with a new snapshot pipeline and explicit identity controls.** The audited CFBD responses contain useful prior-production signal, but they are retrospective endpoint responses. Final destinations, publication timing, and rating revisions are not proven as-of the historical preseason cutoff. The current portal payload also has no player identifier shared with `/player/usage`, so the production fallback must be deterministic name + source-team matching that fails closed on ambiguity.

The selected issue-91 representation is `total RP + incoming prior transfer production`; this audit therefore focuses on incoming destination-resolved transfers and prior-season `usage.overall`. It does not redesign or promote the Context model.

## Exact data requirements

| Field | Required for model | Audit use |
|---|---|---|
| transfer season | yes | query and snapshot scope |
| player identity | yes | name normalization, collisions, stable-ID test |
| source school | yes | canonical team mapping and usage join key |
| destination school | yes | canonical FBS destination and cutoff availability |
| transfer date / cutoff | yes | season-relative inclusion rule |
| prior-season usage | yes | exact, alias, missing, ambiguous, and usage-weighted coverage |
| position | only if representation needs it | position-group coverage |
| rating / stars | no for C10; audit proxy only | unmatched prioritization |
| team identity mapping | yes | explicit aliases; no fuzzy matching |
| preseason cutoff semantics | yes | source classification and snapshot policy |

Fields such as rating, stars, and position are useful for diagnosing important unmatched cases but are not required by C10. Scholarship status is not available from the selected endpoints.

## Player identity and join coverage

| Season | Portal | Destination | On/before cutoff | Incoming FBS | FBS source | Non-FBS/unrecognized source | Exact joins | Alias joins | Failed joins | Ambiguous | Normalization rescues | Portal collisions | Usage-weighted proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 1770 | 1053 | 1770 | 816 | 737 | 79 | 173 | 0 | 643 | 2 | 4 | 0 | 0.944 |
| 2022 | 2273 | 1367 | 2268 | 1089 | 964 | 125 | 257 | 0 | 832 | 2 | 2 | 1 | 0.935 |
| 2023 | 2502 | 1607 | 2502 | 1418 | 1218 | 200 | 362 | 0 | 1056 | 5 | 8 | 0 | 0.918 |
| 2024 | 3378 | 2654 | 3378 | 2066 | 1712 | 354 | 473 | 0 | 1593 | 3 | 10 | 1 | 0.951 |
| 2025 | 4499 | 3770 | 4497 | 2905 | 2271 | 634 | 557 | 0 | 2348 | 13 | 6 | 0 | 0.856 |

A raw record-count join rate is not enough. The usage-weighted proxy is `recoverable_unique_usage_mass / any_name_usage_mass`: the denominator is the unique prior-usage mass whose normalized player name appears in at least one in-scope transfer, while the numerator additionally requires a unique source-team/player join. It is the strongest reproducible identity-resolution proxy available from these endpoints, not a full-population denominator, because usage for a completely unmatched player is unobserved.

## High-value unmatched transfers

The full machine-readable list is `unmatched_high_value_transfers.csv`. Ranking uses rating, then stars, then QB status; it is a diagnostic ordering, not a model feature.

| Rank | Season | Player | Origin | Destination | Pos | Rating | Stars | Failure |
|---|---|---|---|---|---|---|---|---|
| 1 | 2022 | Quinn Ewers | Ohio State | Texas | QB | 1.000 | 5 | no_usage_record |
| 2 | 2024 | Caleb Downs | Alabama | Ohio State | S | 0.990 | 5 | no_usage_record |
| 3 | 2024 | Kadyn Proctor | Iowa | Alabama | OT | 0.990 | 5 | no_usage_record |
| 4 | 2024 | Walter Nolen | Texas A&M | Ole Miss | DL | 0.990 | 5 | no_usage_record |
| 5 | 2024 | Julian Sayin | Alabama | Ohio State | QB | 0.980 | 5 | no_usage_record |
| 6 | 2025 | Damon Wilson II | Georgia | Missouri | EDGE | 0.980 | 5 | no_usage_record |
| 7 | 2023 | Denver Harris | Texas A&M | LSU | CB | 0.980 | 5 | no_usage_record |
| 8 | 2022 | Eli Ricks | LSU | Alabama | CB | 0.980 | 5 | no_usage_record |
| 9 | 2021 | Henry To'o To'o | Tennessee | Alabama | LB | 0.970 | 4 | no_usage_record |
| 10 | 2024 | Jason Zandamela | USC | Florida | IOL | 0.970 | 4 | no_usage_record |
| 11 | 2023 | Ernest Hausmann | Nebraska | Michigan | LB | 0.960 | 4 | no_usage_record |
| 12 | 2022 | Kingsley Suamataia | Oregon | BYU | OT | 0.960 | 4 | no_usage_record |
| 13 | 2024 | A.J. Harris | Georgia | Penn State | CB | 0.950 | 4 | no_usage_record |
| 14 | 2024 | Cayden Green | Oklahoma | Missouri | OT | 0.950 | 4 | no_usage_record |
| 15 | 2025 | David Bailey | Stanford | Texas Tech | EDGE | 0.950 | 4 | no_usage_record |
| 16 | 2023 | Fentrell Cypress | Virginia | Florida State | CB | 0.950 | 4 | no_usage_record |
| 17 | 2024 | Nyland Green | Georgia | Purdue | CB | 0.950 | 4 | no_usage_record |
| 18 | 2025 | Patrick Payton | Florida State | LSU | EDGE | 0.950 | 4 | no_usage_record |
| 19 | 2024 | Cam Ward | Washington State | Miami | QB | 0.940 | 4 | no_usage_record |
| 20 | 2023 | Bear Alexander | Georgia | USC | DL | 0.940 | 4 | no_usage_record |
| 21 | 2022 | Brandon Joseph | Northwestern | Notre Dame | S | 0.940 | 4 | no_usage_record |
| 22 | 2024 | Cormani McClain | Colorado | Florida | CB | 0.940 | 4 | no_usage_record |
| 23 | 2023 | Davison Igbinosun | Ole Miss | Ohio State | CB | 0.940 | 4 | no_usage_record |
| 24 | 2021 | Derion Kendrick | Clemson | Georgia | CB | 0.940 | 4 | no_usage_record |
| 25 | 2024 | Dezz Ricks | Alabama | Texas A&M | CB | 0.940 | 4 | no_usage_record |

Failure classes distinguish missing usage rows, source-team mismatches, usage rows without a numeric value, and ambiguous normalized joins. No fuzzy player match is applied.

## Team identity audit

The complete mapping inventory is `team_mapping_by_season.csv`. 0 encountered names were resolved through the supplied explicit alias table; names not in the FBS model population remain visible with a failure method rather than being silently coerced.

| Season | Field | Name | Result | Records |
|---|---|---|---|---|
| 2021 | destination | Abilene Christian | not_in_fbs_population_or_unrecognized | 4 |
| 2021 | destination | Alabama A&M | not_in_fbs_population_or_unrecognized | 2 |
| 2021 | destination | Alcorn State | not_in_fbs_population_or_unrecognized | 2 |
| 2021 | destination | Angelo State | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Arkansas Tech | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Arkansas-Pine Bluff | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Austin Peay | not_in_fbs_population_or_unrecognized | 3 |
| 2021 | destination | Bethune-Cookman | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Blinn College | not_in_fbs_population_or_unrecognized | 4 |
| 2021 | destination | Bowie State | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Bryant | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Butler C.C. | not_in_fbs_population_or_unrecognized | 3 |
| 2021 | destination | Central Connecticut | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Central Missouri | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Central Oklahoma | not_in_fbs_population_or_unrecognized | 2 |
| 2021 | destination | Central Washington | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Charleston Southern | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Chattanooga | not_in_fbs_population_or_unrecognized | 4 |
| 2021 | destination | City College of San Francisco | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | College of Charleston | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | College of the Canyons | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Colorado Mesa | not_in_fbs_population_or_unrecognized | 3 |
| 2021 | destination | Delaware | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Delta State | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Duquesne | not_in_fbs_population_or_unrecognized | 3 |
| 2021 | destination | East Tennessee State | not_in_fbs_population_or_unrecognized | 2 |
| 2021 | destination | Eastern Illinois | not_in_fbs_population_or_unrecognized | 1 |
| 2021 | destination | Eastern Kentucky | not_in_fbs_population_or_unrecognized | 8 |
| 2021 | destination | Eastern Washington | not_in_fbs_population_or_unrecognized | 2 |
| 2021 | destination | Florida A&M | not_in_fbs_population_or_unrecognized | 4 |

## Cutoff safety and historical destination limitation

| Field | Model | Classification | Evidence |
|---|---|---|---|
| transfer season | True | likely cutoff-safe but not provable | The request selects an explicit season, but the source response is not an archived preseason snapshot. |
| player identity | True | retrospective oracle only | Portal identity is a name assembled from the current endpoint; no shared stable portal/usage ID is available. |
| source school | True | retrospective oracle only | Origin is read from the current portal record and is not an archived roster state. |
| destination school | True | retrospective oracle only | A final destination may have been resolved or revised after the preseason cutoff. |
| transfer date | True | retrospective oracle only | The date supports deterministic filtering, but publication and revision timing are not archived. |
| prior-season usage | True | retrospective oracle only | Usage is a prior-season outcome, but the cross-endpoint name join is not an archived transfer roster join. |
| position | False | retrospective oracle only | Position is supplied by the current portal response and is not timestamped as-of the cutoff. |
| rating / stars | False | retrospective oracle only | The response does not provide an archived rating revision history. |
| scholarship status | False | unavailable | Eligibility is not a scholarship indicator in the selected source fields. |

A transfer date on or before August 15 proves only that the current response carries an early event date. It does not prove that the destination stored today was known, published, or stable by August 15 in the historical year. The audit therefore keeps destination filtering deterministic while classifying destination as retrospective-oracle-only.

## Source inventory

| Source | Player ID | Timestamp semantics | Assessment |
|---|---|---|---|
| CFBD /player/portal | none in audited portal payload | current retrospective response; event date is not an as-of publication timestamp | usable for future snapshots only if fetched and frozen before the cutoff |
| CFBD /player/usage | stable within usage endpoint, not shared by portal endpoint | season-wide prior usage; not a transfer-time roster snapshot | supporting prior-production source after deterministic identity resolution |
| repository-managed preseason snapshot process | portal ID remains unavailable; exact name + source-team fallback is required | retrieval timestamp proves when GippyRank captured the response, not when CFBD first knew a destination | recommended minimal path for future production feasibility |

## Current/future snapshot strategy

1. On or before the configured preseason cutoff, fetch each required portal and prior-usage season response.
2. Store response bytes unchanged under data/raw/cfbd/preseason/transfers/{portal,usage}/.
3. Write endpoint, query parameters, retrieval timestamp, record count, and SHA-256 in a sidecar.
4. Never overwrite a prior season snapshot; refresh only into a new explicitly named snapshot when source semantics require it.
5. Run this audit and derive transfer features only from the frozen snapshot, with explicit aliases and fail-closed ambiguous joins.

Add the snapshot acquisition as a prerequisite to the existing preseason pipeline; do not let live portal endpoints enter model fitting directly.

## Team-level feature coverage and unresolved-join sensitivity

| Season | Teams | Complete | Partial | No usable | No incoming | Mean transfers | Mean joined usage | Missingness |
|---|---|---|---|---|---|---|---|---|
| 2021 | 130 | 3 | 88 | 34 | 5 | 6.277 | 0.190 | 0.938 |
| 2022 | 131 | 1 | 107 | 20 | 3 | 8.313 | 0.269 | 0.969 |
| 2023 | 133 | 2 | 113 | 15 | 3 | 10.662 | 0.361 | 0.962 |
| 2024 | 134 | 0 | 125 | 5 | 4 | 15.418 | 0.479 | 0.970 |
| 2025 | 136 | 0 | 129 | 4 | 3 | 21.360 | 0.466 | 0.978 |

For each team-season, `unmatched_prior_usage_upper_bound` is the observed usage plus the unmatched incoming count multiplied by the maximum numeric overall usage in the prior usage payload. This is intentionally conservative and not imputed into the model. The complete team-season table is `team_feature_coverage.csv`.

Conference-stratified missingness is not reported as a numeric result because the canonical team-season feature table has no season-specific conference column. Competition level is FBS destination only; source level remains visible through mapping and join failure classes. Transfer-volume and roster-strength correlations are descriptive in `missingness_summary.json`.

## Stable-player-ID assessment

CFBD /player/portal payloads contain no player ID in the audited seasons; /player/usage IDs cannot be linked cross-endpoint. The usage endpoint supplies IDs for 19423 of 19423 usage records, but the portal endpoint supplies none. The recommended fallback is therefore explicit normalized name + source team, with ambiguity surfaced and excluded.

## Reproducibility and artifacts

Raw responses remain unchanged and ignored. `source_manifest.json` records the input paths, response hashes, record counts, and fetch sidecars. Derived artifacts are deterministic given those inputs, the canonical team table, the explicit alias file, and the configured cutoff.

- `audit_summary.json` — machine-readable conclusion, inputs, coverage, and stable-ID assessment.
- `player_join_coverage_by_season.csv` — record-count and usage-weighted join coverage.
- `unmatched_high_value_transfers.csv` — all unmatched in-scope transfers in diagnostic priority order.
- `team_mapping_by_season.csv` — every encountered origin/destination name and deterministic resolution.
- `team_feature_coverage.csv` — team-season completeness and unresolved-production bounds.
- `player_join_records.csv` — row-level explanation for every portal record.
- `cutoff_safety.json`, `source_inventory.json`, `snapshot_strategy.json`, `missingness_summary.json` — supporting audit decisions.

## Acceptance conclusion

The signal is not production-safe from retrospective CFBD portal responses alone. It is a credible future production candidate only after (1) capturing immutable preseason snapshots before the cutoff, (2) retaining explicit endpoint provenance, and (3) adding stable cross-endpoint player identity or keeping the fail-closed name + source-team audit with manual resolution of high-value unmatched cases.
