# Issue 238: Campbell historical publication repair

The September 6, September 7, and official Week 2 publications were
rebuilt from retained cutoff-local inputs after removing the malformed
completed 0–0 fixture `401866625`. The real 28–19 result `401917058`
remains absent from September 6 and remains in September 7 and Week 2.

Each repaired Context and History snapshot keeps its original prior
version, likelihood, requested/effective cutoff, and inference settings.
Quality of Play and Quality of Record were rebuilt from the corrected
Context evidence. The superseded bundles remain in `data/processed` and
are identified by source ID and hashes in the lineage audit.

The full repository scan found 12 contaminated bundles in the active publication config. It also found 5 contaminated candidate bundles that are absent from the active publication manifest; those unselected lineages remain unchanged.

## Evidence counts

| Publication | Previous games | Repaired games | `401917058` retained |
| --- | ---: | ---: | --- |
| 2026-09-06 | 165 | 164 | no |
| 2026-09-07 | 171 | 170 | yes |
| 2026-09-08 | 172 | 171 | yes |

## Ranking changes

The table reports changed display ranks and the largest moves for each
official Week 2 ranking. Positive display-rank changes mean the team
moved down the table. The full row-level comparison is in
`data/processed/issue_238_historical_repair/ranking_changes.csv`.

| Ranking | Teams moved | Largest display-rank moves |
| --- | ---: | --- |
| context | 0 | No display-rank changes |
| history | 0 | No display-rank changes |
| quality_of_play | 2 | Jacksonville State 106→107 (+1); Middle Tennessee 107→106 (-1) |
| quality_of_record | 0 | No display-rank changes |

## Validation

- Performance backfill: `complete`; 21 eligible publications, 0 blocked.
- Context, History, Quality of Play, and Quality of Record use the same
  corrected game rows within each publication slot.
- The active site and static API exports contain no temporary Quality
  of Record exception declaration.

Machine-readable details and artifact hashes are in
`data/processed/issue_238_historical_repair/lineage_audit.json`.
