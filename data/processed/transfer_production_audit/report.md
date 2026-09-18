# Transfer-production data-quality and production-feasibility audit (issue 98)

## Recommendation

**Feasible only with a new snapshot pipeline and explicit identity controls.** The audited CFBD responses contain useful incoming prior offensive usage signal, but they are retrospective endpoint responses. Final destinations, publication timing, and rating revisions are not proven as-of the historical preseason cutoff. The current portal payload also has no player identifier shared with `/player/usage`, so the production fallback must be deterministic name + source-team matching that fails closed on ambiguity. D5 feasibility is driven by applicable-usage resolution failures and transfers whose applicability cannot be determined; defensive or special-team transfers without offensive usage are not counted as identity failures.

The selected issue-91 representation is `total RP + incoming prior transfer usage`; this audit operationalizes D5 as `usage.overall` for incoming transfers where offensive applicability is established. Each transfer is classified as successfully resolved applicable usage, legitimate zero/non-applicable usage, failed resolution of recoverable offensive usage, or undetermined applicability. It does not redesign or promote the Context model.

## Exact data requirements

| Field | Required for model | Audit use |
|---|---|---|
| transfer season | yes | query and snapshot scope |
| player identity | yes | name normalization, collisions, stable-ID test |
| source school | yes | canonical team mapping and usage join key |
| destination school | yes | canonical FBS destination and cutoff availability |
| transfer date / cutoff | yes | season-relative inclusion rule |
| incoming prior offensive usage | yes | D5 applicability categories, identity resolution, and usage-weighted proxy |
| position | only if representation needs it | position-group coverage |
| rating / stars | no for D5; audit proxy only | unmatched prioritization |
| team identity mapping | yes | explicit aliases; no fuzzy matching |
| preseason cutoff semantics | yes | source classification and snapshot policy |

Fields such as rating and stars are useful for diagnosing important unresolved D5 cases but are not required by D5. Position is used only to distinguish offensive applicability from defensive/special-team non-applicability. Scholarship status is not available from the selected endpoints.

## Player identity and join coverage

| Season | Portal | Destination | On/before cutoff | Incoming FBS | FBS source | Non-FBS/unrecognized source | D5 applicable | D5 resolved | D5 zero/non-applicable | D5 failures | D5 unknown | D5 valid rate | Applicable resolution rate | Exact joins | Alias joins | Failed joins | Ambiguous | Normalization rescues | Portal collisions | Usage-weighted proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 1770 | 1053 | 1770 | 816 | 737 | 79 | 384 | 173 | 380 | 211 | 52 | 0.724 | 0.451 | 173 | 0 | 643 | 2 | 4 | 0 | 0.978 |
| 2022 | 2273 | 1367 | 2268 | 1089 | 964 | 125 | 548 | 257 | 523 | 291 | 18 | 0.728 | 0.469 | 257 | 0 | 832 | 2 | 2 | 1 | 0.951 |
| 2023 | 2502 | 1607 | 2502 | 1418 | 1218 | 200 | 714 | 362 | 686 | 352 | 18 | 0.749 | 0.507 | 362 | 0 | 1056 | 5 | 8 | 0 | 0.929 |
| 2024 | 3378 | 2654 | 3378 | 2066 | 1712 | 354 | 1005 | 473 | 1060 | 532 | 1 | 0.742 | 0.471 | 473 | 0 | 1593 | 3 | 10 | 1 | 0.959 |
| 2025 | 4499 | 3770 | 4497 | 2905 | 2271 | 634 | 1393 | 557 | 1511 | 836 | 1 | 0.712 | 0.400 | 557 | 0 | 2348 | 13 | 6 | 0 | 0.879 |

A raw record-count join rate is not enough. The usage-weighted proxy is `recoverable_unique_usage_mass / any_name_usage_mass` over D5-applicable transfers: the denominator is the unique prior offensive-usage mass whose normalized player name appears in at least one D5-applicable transfer, while the numerator additionally requires a unique source-team/player join. It is an applicable-D5 identity-resolution proxy, not overall D5 feature coverage or a full-population denominator, because usage for a completely unmatched player is unobserved.

## High-value unmatched transfers

The full machine-readable list is `unmatched_high_value_transfers.csv`. It contains only D5 resolution failures or undetermined-applicability cases; legitimate zero/non-applicable defensive and special-team transfers are excluded. Ranking uses rating, then stars, then QB status; it is a diagnostic ordering, not a model feature.

| Rank | Season | Player | Origin | Destination | Pos | Rating | Stars | D5 category | Applicability | Failure |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2022 | Quinn Ewers | Ohio State | Texas | QB | 1.000 | 5 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 2 | 2024 | Kadyn Proctor | Iowa | Alabama | OT | 0.990 | 5 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 3 | 2024 | Julian Sayin | Alabama | Ohio State | QB | 0.980 | 5 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 4 | 2024 | Jason Zandamela | USC | Florida | IOL | 0.970 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 5 | 2022 | Kingsley Suamataia | Oregon | BYU | OT | 0.960 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 6 | 2024 | Cayden Green | Oklahoma | Missouri | OT | 0.950 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 7 | 2024 | Cam Ward | Washington State | Miami | QB | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 8 | 2023 | Javion Cohen | Alabama | Miami | IOL | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 9 | 2025 | Josh Thompson | Northwestern | LSU | IOL | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 10 | 2025 | KC Concepcion | NC State | Texas A&M | WR | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 11 | 2024 | Lance Heard | LSU | Tennessee | OT | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 12 | 2022 | Victor Oluwatimi | Virginia | Michigan | IOL | 0.940 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 13 | 2025 | Jaron-Keawe Sagapolutele | Oregon | California | QB | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 14 | 2023 | Avery Jones | East Carolina | Auburn | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; source_team_mismatch | source_team_mismatch |
| 15 | 2025 | Emmanuel Pregnon | USC | Oregon | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 16 | 2025 | Hunter Zambrano | Illinois State | Texas Tech | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 17 | 2023 | LaDarius Henderson | Arizona State | Michigan | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 18 | 2023 | Matt Lee | UCF | Miami | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 19 | 2025 | Nic Anderson | Oklahoma | LSU | WR | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 20 | 2024 | Noah Rogers | Ohio State | NC State | WR | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 21 | 2024 | Parker Brailsford | Washington | Alabama | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 22 | 2025 | Patrick Kutas | Arkansas | Ole Miss | IOL | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 23 | 2021 | Wanya Morris | Tennessee | Oklahoma | OT | 0.930 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 24 | 2024 | Austin Mack | Washington | Alabama | QB | 0.920 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |
| 25 | 2023 | Ajani Cornelius | Rhode Island | Oregon | OT | 0.920 | 4 | should_have_recoverable_offensive_usage_but_resolution_failed | offensive_portal_position; no_usage_record | no_usage_record |

Failure classes distinguish missing usage rows, source-team mismatches, usage rows without a numeric value, and ambiguous normalized joins. The D5 applicability category is retained alongside each identity result, and no fuzzy player match is applied.

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
| incoming prior offensive usage | True | retrospective oracle only | Usage is a prior-season outcome, but the cross-endpoint name join is not an archived transfer roster join. |
| position | False | retrospective oracle only | Position is supplied by the current portal response and is not timestamped as-of the cutoff. |
| rating / stars | False | retrospective oracle only | The response does not provide an archived rating revision history. |
| scholarship status | False | unavailable | Eligibility is not a scholarship indicator in the selected source fields. |

A transfer date on or before August 15 proves only that the current response carries an early event date. It does not prove that the destination stored today was known, published, or stable by August 15 in the historical year. The audit therefore keeps destination filtering deterministic while classifying destination as retrospective-oracle-only.

## Source inventory

| Source | Player ID | Timestamp semantics | Assessment |
|---|---|---|---|
| CFBD /player/portal | none in audited portal payload | current retrospective response; event date is not an as-of publication timestamp | usable for future snapshots only if fetched and frozen before the cutoff |
| CFBD /player/usage | stable within usage endpoint, not shared by portal endpoint | season-wide prior offensive usage; not a transfer-time roster snapshot | supporting incoming prior offensive usage source after deterministic identity resolution |
| repository-managed preseason snapshot process | portal ID remains unavailable; exact name + source-team fallback is required | retrieval timestamp proves when GippyRank captured the response, not when CFBD first knew a destination | recommended minimal path for future production feasibility |

## Current/future snapshot strategy

1. On or before the configured preseason cutoff, fetch each required portal and prior-usage season response.
2. Store response bytes unchanged under data/raw/cfbd/preseason/transfers/{portal,usage}/.
3. Write endpoint, query parameters, retrieval timestamp, record count, and SHA-256 in a sidecar.
4. Never overwrite a prior season snapshot; refresh only into a new explicitly named snapshot when source semantics require it.
5. Run this audit and derive transfer features only from the frozen snapshot, with explicit aliases and fail-closed ambiguous joins.

Add the snapshot acquisition as a prerequisite to the existing preseason pipeline; do not let live portal endpoints enter model fitting directly.

## Team-level feature coverage and unresolved-join sensitivity

| Season | Teams | Complete | Partial | No usable | No incoming | Applicable failures | Undetermined | Both | Mean applicable transfers | Mean observed offensive usage | Missingness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021 | 130 | 23 | 99 | 3 | 5 | 3 | 0 | 0 | 2.954 | 0.190 | 0.785 |
| 2022 | 131 | 21 | 106 | 1 | 3 | 1 | 0 | 0 | 4.183 | 0.269 | 0.817 |
| 2023 | 133 | 16 | 113 | 1 | 3 | 1 | 0 | 0 | 5.368 | 0.361 | 0.857 |
| 2024 | 134 | 5 | 125 | 0 | 4 | 0 | 0 | 0 | 7.500 | 0.479 | 0.933 |
| 2025 | 136 | 1 | 132 | 0 | 3 | 0 | 0 | 0 | 10.243 | 0.466 | 0.971 |

For each team-season, `unresolved_prior_offensive_usage_upper_bound` equals observed incoming prior offensive usage plus the D5 resolution-failure count multiplied by the maximum numeric overall usage in the prior usage payload. `unmatched_prior_usage_upper_bound` is retained as the missing portion only; `undetermined_incoming_count` is reported separately because its applicability cannot be established. These bounds are intentionally conservative and are not imputed into the model. The complete team-season table is `team_feature_coverage.csv`.

Conference-stratified missingness is not reported as a numeric result because the canonical team-season feature table has no season-specific conference column. Competition level is FBS destination only; source level remains visible through mapping and join failure classes. Transfer-volume and roster-strength correlations are descriptive in `missingness_summary.json`, and missingness is based on applicable D5 failures or undetermined applicability rather than defensive/non-applicable transfers.

## Stable-player-ID assessment

CFBD /player/portal payloads contain no player ID in the audited seasons; /player/usage IDs cannot be linked cross-endpoint. The usage endpoint supplies IDs for 19423 of 19423 usage records, but the portal endpoint supplies none. The recommended fallback is therefore explicit normalized name + source team, with ambiguity surfaced and excluded.

## Reproducibility and artifacts

Raw responses remain unchanged and ignored. `source_manifest.json` records the input paths, response hashes, record counts, and fetch sidecars. Derived artifacts are deterministic given those inputs, the canonical team table, the explicit alias file, and the configured cutoff.

- `audit_summary.json` — machine-readable conclusion, inputs, coverage, and stable-ID assessment.
- `player_join_coverage_by_season.csv` — record-count and usage-weighted join coverage.
- `unmatched_high_value_transfers.csv` — all unmatched in-scope transfers in diagnostic priority order.
- `team_mapping_by_season.csv` — every encountered origin/destination name and deterministic resolution.
- `team_feature_coverage.csv` — team-season completeness and unresolved incoming offensive-usage bounds.
- `player_join_records.csv` — row-level explanation for every portal record.
- `cutoff_safety.json`, `source_inventory.json`, `snapshot_strategy.json`, `missingness_summary.json` — supporting audit decisions.

## Acceptance conclusion

The signal is not production-safe from retrospective CFBD portal responses alone. It is a credible future production candidate only after (1) capturing immutable preseason snapshots before the cutoff, (2) retaining explicit endpoint provenance, and (3) adding stable cross-endpoint player identity or keeping the fail-closed name + source-team audit with manual resolution of high-value unmatched cases.
