# GippyRank4 Preseason Context Prior V1.2

## Architecture

H (`history_prior` 1.1) remains the rank-history-only forecast. C (`context_prior` 1.2) is a sibling forecast that starts from H's historical features and may add only context fields admitted by the provenance gate. Neither family overwrites the other's PMFs or metadata.

H is frozen as full t-1 constituent-rank uncertainty, t-2/t-3 transformed-rank summaries, long-run program history, a heteroscedastic Normal predictive distribution, analytically integrated discrete PMFs, and the established FBS cold-start fallback. C uses the same historical foundation; it never consumes H's expected rank as a synthetic feature.

## Provenance decision

Coach tenure is reconstructable-safe when a cached CFBD continuous tenure proves the target coach by the August 15 cutoff. Coach change remains timing-uncertain because many tenure ends are undated. Recruiting, Team Talent Composite, and returning production remain timing-uncertain because their cached annual API payloads lack archival as-of timestamps. Transfers are rejected: no dated target-roster reconstruction is cached.

| Family | Status | Coverage | Production decision |
|---|---|---|---|
| Rank history | production-safe | 2003-2025 final outcomes | Retained |
| Coach tenure | reconstructable-safe | depends on cached team responses; CFBD exposes continuous historical tenures | A target-season start without a date by cutoff is missing. |
| Coach change | exploratory-timing-uncertain | No complete cutoff-safe historical comparison in the cached tenure payloads | Not used in C V1.2; unknown is never converted to no change. |
| Recruiting | exploratory-timing-uncertain | 2003-2026, incomplete across teams and years | Not admitted merely because a class ordinarily signs before kickoff. |
| Team Talent Composite | exploratory-timing-uncertain | 2015-2026; coverage varies materially in recent seasons | Excluded from V1.2 production. |
| Returning production / QB proxy | exploratory-timing-uncertain | 2014-2026 FBS-oriented payloads | Overall and passing/QB proxies remain exploratory. |
| Transfers | rejected | not acquired | Cleanly omitted rather than inferred from current roster state. |

## Context coverage and missingness

Missing or ambiguous context is never imputed as zero. The selected C PMF falls back exactly to the frozen H PMF. This applies to every FBS cold start as well as to an unavailable tenure row.

| Season | Dated tenure available | Missing / ambiguous |
|---|---:|---:|
| 2018 | 102 | 27 |
| 2019 | 106 | 24 |
| 2020 | 110 | 17 |
| 2021 | 98 | 29 |
| 2022 | 105 | 25 |
| 2023 | 103 | 28 |
| 2024 | 101 | 32 |
| 2025 | 101 | 33 |

## Development ablations

| Candidate | Population | NLL | ΔNLL vs H | CRPS | ΔCRPS vs H |
|---|---:|---:|---:|---:|---:|
| C0 = H | 416 | 4.5152 | 0.0000 | 0.1149 | 0.0000 |
| C1 = H + coach tenure (location and scale) | 416 | 4.5081 | -0.0071 | 0.1139 | -0.0010 |

## Frozen C selection

Selected candidate: **C0_history_only**. Development selection used only 2004–2017 training and 2018–2021 validation. The rule required ΔNLL ≤ -0.01 and at least 80% favorable descriptive season resamples; otherwise C falls back to the frozen H PMF.

## Untouched 2022–2025 comparison

| Population | H NLL | C NLL | ΔNLL (C − H) | H CRPS | C CRPS |
|---|---:|---:|---:|---:|---:|
| All FBS | 4.5442 | 4.5442 | 0.0000 | 0.1137 | 0.1137 |
| Coach-observed subset | 4.5370 | 4.5370 | 0.0000 | 0.1139 | 0.1139 |

Because C0 was selected, the final frozen C PMF is exactly H for all 534 untouched FBS team-seasons: ΔNLL, ΔCRPS, rank-error, interval, and Top-5/10/25 Brier differences are all zero. C1 was not evaluated on the untouched test because it did not clear the pre-2022 selection rule.

The model-report JSON retains full reliability bins, Brier scores, interval coverage and width, the development C1 location/scale optimizer diagnostics, and same-population metrics. The C1 development gain did not meet the pre-specified materiality threshold, so added complexity is not justified in the production specification.

## Cold starts

All FBS targets receive one H and one C PMF. C retains H's learned FCS-to-FBS transition PMF or broad generic FBS fallback for the six historical test cold starts; it does not manufacture coach context for them.

## 2026 readiness

H is reconstructable from completed 2025 ranks plus a pre-kickoff 2026 FBS universe. C is only partially reconstructable: dated coach tenures can be reconstructed where cached, but no production-safe snapshot exists for recruiting, talent, or returning production. No 2026 game outcome is read.

## Annual refit

For a new target season, preserve the specification version and record `trained_through_season`, `target_season`, and—only for C—an explicit `context_as_of` date. Refresh raw coaching tenures separately before fitting; the model build must run exclusively from that cached source.
