# Issue 172: corrected Context 1.4 DB transfer input

## Decision and model contract

The corrected fit replaces the active Context 1.4 prior. The public version stays 1.4. The model uses the repaired observed incoming DB impact sum and observed fraction, alongside the existing offensive usage and other Context features. Missing incoming DB player impacts remain unknown: no count scaling, mean fill, or zero fill is applied to them. A team with no incoming DB transfers has impact zero and coverage one; a team with incoming transfers but no observed impacts has impact zero and coverage zero. Training seasons before 2021 have both DB fields missing because the repaired DB source does not cover them.

The DirectRank model is refit through 2025. Its positive Context-only location subtotal is still moderated with alpha 0.75; nonpositive subtotals and rank-history cold starts retain their established behavior. The standardized location coefficients for observed DB impact and coverage are -0.05145 and -0.00396. At fixed impact, more coverage moves the location slightly toward a better rank, but the coverage coefficient is small. History 1.1 and Historical Likelihood V1 were not refit.

The 2026 transfer data was reacquired after the August 15 cutoff. This is a retrospective reconstruction of preseason-semantic transfer facts; the artifact does not claim the evidence was available on August 15.

## DB repair inventory

The full 138-team before/after table is in [2026_before_after.csv](../data/processed/context_db_repair_172/2026_before_after.csv). Its old fields come from the shipped Context input audit; the new fields come from the repaired 2026 coverage artifact. The count and player-sum joins fail closed if their source audits disagree.

| Outcome | Teams |
| --- | ---: |
| Became complete | 20 |
| Already complete, impact corrected | 14 |
| Still partial, gained observed contributors | 11 |
| Still partial, known sum exposed instead of zero | 27 |
| Natural zero, no incoming DB transfers | 6 |
| Otherwise unchanged | 60 |

Excluding natural-zero representation changes, 72 teams gained or corrected model-facing DB evidence. Complete coverage, including natural zeros, rose from 80 to 100 teams; partial coverage fell from 57 to 38. The repaired 2026 source has 94 nonzero complete teams, 38 partial teams, and six natural zeros.

Auburn changed from five of six observed, impact 0.0, and unavailable to six of six observed, impact 3.3319573489815544, and complete. Its preseason expected rank moved from 30.128 to 27.530. This is a correctness example, not a tuning target.

The largest increases in model-facing observed DB impact were Cincinnati +8.765, Memphis +8.714, UCLA +6.917, Arizona +6.757, and Ole Miss +5.901. For partial teams, part of this change is previously discarded known impact becoming visible. The largest absolute prior expected-rank changes were Cincinnati 11.18 places, Memphis 9.56, UCLA 8.88, Arizona 7.62, and Michigan State 7.55. Mean absolute rank movement was 1.24 for complete teams, 2.27 for partial teams, and 0.15 for natural zeros.

## Paired model comparison

Historical preseason scores use rolling fits trained only through the preceding season, 2022–25, with the same 528 fitted team-seasons in each arm. NLL is the observed rank log score; MAE is expected-rank error. The historical predictive comparison uses one fixed season-local midseason cutoff in each year and scores 2,700 games after those cutoffs. Every arm receives the same earlier completed-game evidence and the same future-game IDs.

| Historical preseason arm | Rank NLL | Expected-rank MAE |
| --- | ---: | ---: |
| Shipped Context 1.4 | 4.50214 | 20.769 |
| Corrected Context 1.4 | 4.50218 | 20.605 |
| History 1.1 | 4.54560 | 22.385 |

| Historical predictive arm | Margin NLL | Margin MAE | Win Brier | 50% coverage | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped Context 1.4 | 4.20872 | 12.867 | 0.19048 | 0.485 | 0.786 | 0.946 |
| Corrected Context 1.4 | 4.20860 | 12.872 | 0.19026 | 0.486 | 0.787 | 0.946 |
| History 1.1 | 4.21121 | 12.909 | 0.19096 | 0.484 | 0.788 | 0.945 |

The 2026 comparison scores the same 644 completed games through Week 5 against the retained shipped Context and History forecasts. Each forecast origin excludes its target game's result. Scores are paired game by game.

| 2026 predictive arm | Margin NLL | Margin MAE | Win Brier | 50% coverage | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped Context 1.4 | 4.28013 | 13.692 | 0.15244 | 0.537 | 0.820 | 0.949 |
| Corrected Context 1.4 | 4.27741 | 13.661 | 0.15180 | 0.537 | 0.820 | 0.952 |
| History 1.1 | 4.29083 | 13.815 | 0.15351 | 0.533 | 0.812 | 0.950 |

Against shipped Context, corrected 2026 margin NLL improves by 0.00534 across 170 games involving a partial-coverage FBS team and by 0.00178 across 474 games with only complete or natural-zero FBS teams. The historical comparison shows no material predictive regression. This supports selecting the correction without optimizing for Auburn or the 2026 outcomes.

Full season and origin scores, interval coverage, source hashes, and coverage strata are in [model_comparison.json](../data/processed/context_db_repair_172/model_comparison.json), [historical_predictive.json](../data/processed/context_db_repair_172/historical_predictive.json), and [current_predictive.json](../data/processed/context_db_repair_172/current_predictive.json).

## Identities and canonical lineage

The production semantics SHA-256 is 4d8fa4011279b20c3c1d30f99093cedd8db5514ace8113c350caa036f1fe3180. The 2026 fitted model SHA-256 is d98dd9ade01cf6164d0b60da4e33adb5d1e0d35140279705a3bcad5679b04aed. The annual prediction file SHA-256 is 70a8e6c49f159eb91d5108acee0829f9194a51865198ee663c6af40037027a9a, and the 2026 FBS prior-PMF semantic hash is cb592e467ee9192ebb85c57527ab1b18560ccf630bc1c3a6a9df1b545d7985c9.

Selected input SHA-256 values:

| Input | SHA-256 |
| --- | --- |
| Repaired historical transfer features | d17d174b8b49e22cc118618afb9046c399e5889cb56f43141c2e5cbfcc6e028c |
| Repaired historical DB team coverage | 7b96e7be87f484712f2d88c1beffeb5508688a4aa6e996e309852c44fa174a34 |
| Historical player DB impact audit | cf7b25313093c9ae0f4d21ee5865336fbb81705062e8ea92c9dc5f535aa1135d |
| Repaired 2026 DB coverage | bdfe71f663817cac711408e9d5b66fea0c3b1a785091616fa1cb8b5ed740aeb0 |
| Generated historical rank distributions | 03e4e372017aedb45392ff7d790cea25f461db9b6dc846514c762ff46e2cab00 |
| 2026 retained outcome-free inference rows | 50be291962af424b337b848555080a53fe5b8dfde5030f00e502fdaf6251bfd2 |

The complete input-hash map is in model_comparison.json. The historical rank-distribution CSV is an ignored generated input produced by scripts/build_historical_modeling.py; its hash is pinned in the comparison report. The retained shipped 2026 annual predictions are copied unchanged into baseline_context14_predictions.csv for the paired comparison.

| Publication slot | Corrected Context snapshot ID | Included games |
| --- | --- | ---: |
| Preseason | 2026-preseason-context-v1.4-db-repair | 0 |
| Week 2 | 2026-weekly-2026-09-08T11-43-00.275833Z-context-v1.4-db-repair | 172 |
| Week 3 | 2026-weekly-2026-09-13T12-02-55.255941Z-context-v1.4-db-repair | 291 |
| Week 4 | 2026-weekly-2026-09-20T11-00-14.294077Z-context-v1.4-db-repair | 410 |
| Week 5 | 2026-weekly-2026-09-27T12-27-35.698895Z-context-v1.4-db-repair | 530 |
| Week 6 | 2026-weekly-2026-10-04T14-51-53.251364Z-context-v1.4-db-repair | 644 |

Each corrected Context snapshot uses the retained cutoff-local game rows and the matching History snapshot's effective cutoff, included-game file, and likelihood. All six parity checks passed. History posterior PMF hashes are recorded without change in [lineage_audit.json](../data/processed/context_db_repair_172/lineage_audit.json). Each weekly Performance snapshot was rebuilt from its corrected Context partner. The site export rebuilt rankings, team pages, distributions, predictions, season projections, methodology, and ignored static API resources. The published preseason evidence displays Auburn as 3.332 and 6/6 complete; Memphis preserves its observed 8.714 at 7/10 partial with the missing contribution identified as unknown.

## Rebuild sequence

Run these from the repository root using the pinned uv environment. The historical modeling input must be present or generated first. The model rebuild is deterministic against its recorded input hashes; the replay uses retained evidence and verifies every origin.

~~~sh
uv run python scripts/rebuild_context_v1_4_db_repair.py
uv run python scripts/evaluate_context_db_repair.py --historical
uv run python scripts/rebuild_context_v1_4_lineage.py
uv run python scripts/evaluate_context_db_repair_2026.py
uv run python scripts/build_site_data.py
~~~
