# Context 1.4 production promotion audit

This document records the original 2026 promotion. Issue #172 corrected its DB transfer input and rebuilt the active Context 1.4 lineage; see [the correction audit](context_db_repair_172.md) for the current production hashes and results.

The frozen Issue #168 / PR #169 verdict was **promote**. Context 1.4 is the
active production prior. Its prior construction reuses the frozen candidate
implementation with alpha **0.75** on positive `context_only_subtotal` only.
History 1.1 and Historical Likelihood V1 are unchanged.

| Identity | SHA-256 |
| --- | --- |
| Frozen candidate semantics | `db84045d2f7d80b1648360da52ef1e71f76093696cad4e4e7bbdbd57a086573f` |
| Production semantics | `34754d53d5700365cdec6d0e4e1f7d66c25a6e830b8df790487887544154025d` |
| 2026 annual prior predictions | `b03850ff79f319d35b582cd733d035c1c7e4e48368c5354fb696958ae965ef8a` |

The annual prior covers 138 FBS teams. All 53 positive rows match the frozen
0.75 moderation; 83 nonpositive rows and two cold-start teams retain their
Context 1.3 prior PMFs. The historical snapshots use frozen origin-local
evidence. All checks in
[`migration_audit.json`](../data/processed/context_v1_4_production/migration_audit.json)
passed before the publication config switched.

| Origin | Included games | Candidate posterior SHA-256 | Production posterior SHA-256 | Result |
| --- | ---: | --- | --- | --- |
| Preseason | 0 | `23bfd81e9811470439aa90671c665ecb97f44a69e2c2f21de2677c8a66da31b5` | `23bfd81e9811470439aa90671c665ecb97f44a69e2c2f21de2677c8a66da31b5` | PASS |
| Sep 8 | 172 | `5e0edb83404628c4da6f4e9665b82e4b59b6b87d5e8ca269d1934e0ef8450552` | `5e0edb83404628c4da6f4e9665b82e4b59b6b87d5e8ca269d1934e0ef8450552` | PASS |
| Sep 13 | 291 | `df94b9604c98b9cfc0b452981313eb892cf6e706efa9538ae4d6b4416321045a` | `df94b9604c98b9cfc0b452981313eb892cf6e706efa9538ae4d6b4416321045a` | PASS |
| Sep 20 | 410 | `154e86822c6f858227a37add68b53a829b32c7ebc64659a79286993fb0df1425` | `154e86822c6f858227a37add68b53a829b32c7ebc64659a79286993fb0df1425` | PASS |
| Sep 27 | 530 | `dae4b183f14a3c2fefbfe797726f629b90c64720348bf9b19705a4b91b8dc38c` | `dae4b183f14a3c2fefbfe797726f629b90c64720348bf9b19705a4b91b8dc38c` | PASS |

## Ordinary Week 6 run

- Effective cutoff: `2026-10-04T14:51:53.251364+00:00`.
- Context snapshot: `2026-weekly-2026-10-04T14-51-53.251364Z-context-v1.4`.
- History snapshot: `2026-weekly-2026-10-04T14-51-53.251364Z-history`.
- Performance snapshot: `2026-weekly-2026-10-04T14-51-53.251364Z-context-v1.4-performance`.
- Included completed games: **644**, including **114** new Week 5 games since
  Sep 27. IDs are unique, every included game is final, and all kickoffs
  precede the cutoff. The latest included kickoff is `2026-10-04T03:59:00Z`.
- Context posterior SHA-256: `301367c19c06e51a645f8eba2a6c38f470a1eb2fe340deca7f13d71c8e6262df`.
- History posterior SHA-256: `26d21c9bde118fb16d06f01e62fa10f6d252f8e28f8ab8cbf0d036bac6b763d5`.
- Context and History have identical included IDs, evidence-row hash,
  effective cutoff, FCS fallback population (128), and likelihood identity.
  Both converged; Performance references this Context snapshot.

Sanity review of the ordinary weekly report found Ohio State, Georgia, and
Alabama at the top of Context's ranking. The largest Context movements were
Florida State (+22), Missouri (+19), and Virginia (-19). James Madison and
Navy had the largest Context/History rank gaps (13), while Massachusetts had
the largest Performance/Context expected-rank gap (39.8). There were no newly
rated FBS teams. The site generated 498 future predictions; the Week 6
Context team trajectory has preseason plus Weeks 2–6, with no operational
model-version break. These checks did not change model parameters.

The site retains old Context 1.2/1.3 files for audit. The Sep 27 Context 1.4
team-page projection carries frozen kickoff certainty from the retained
Context 1.3 site artifact; its source path and hash are recorded on the
projected artifact. This does not affect the posterior or cutoff evidence.
