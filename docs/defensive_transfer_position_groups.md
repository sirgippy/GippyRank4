# Defensive-transfer production by position group

Issue 111 is the final defensive-transfer decomposition experiment. It tests
whether the frozen PR #105 defensive-impact sum contains signal within one of
the existing broad position groups even though the aggregate defensive feature
did not.

Run the study with the immutable transfer cache and the repository's frozen
audit artifacts:

```text
MPLCONFIGDIR=/tmp/gippyrank4-mplconfig \
UV_CACHE_DIR=/tmp/gippyrank4-uv-cache uv run python \
  scripts/investigate_defensive_transfer_position_groups.py \
  --transfer-root /path/to/frozen-transfer-cache
```

The runner reads `transfer_player_audit.csv` and
`coverage_by_position.csv` from `data/processed/defensive_transfer_audit/`.
The audit is the source of truth for the existing deterministic identity join,
portal position mapping, prior-season position group, impact status, and
frozen player-level impact. The runner does not rebuild or alter that metric.

## Frozen groups and features

Only these groups are used:

- `dl_edge`: DL, EDGE, DE, DT, NT; tackles, tackles for loss, sacks, and quarterback hurries.
- `lb`: OLB, ILB, MLB, LB; tackles, tackles for loss, sacks, and passes defended.
- `db`: CB, DB, S, FS, SS, NB; tackles, passes defended, and interceptions.

Each player impact is the existing equal-weight mean of season-by-group
standardized `log1p` components. The team feature is the sum of incoming
players' frozen impacts in that group:

- `transfer_in_prior_defensive_impact_dl_edge_sum`
- `transfer_in_prior_defensive_impact_lb_sum`
- `transfer_in_prior_defensive_impact_db_sum`

Each value has a matching `_available` indicator. A group with no incoming
transfers is an observed zero. A group with any unresolved incoming impact is
neutral-imputed to zero with availability 0. Team-seasons are never dropped.

## Candidate protocol

The predeclared candidates are P0 D5, P1 D5 plus DL/EDGE, P2 D5 plus LB, P3
D5 plus DB, and P4 D5 plus all three groups. D5 is the C-minus-RP baseline
plus `returning_pct_ppa` and `transfer_in_prior_usage_sum`.

Each candidate is fit through 2021 and evaluated unchanged on 2022–2025. The
same production Context preprocessing, model family, penalty, optimizer retry,
H fallback, FBS population, August 15 cutoff, response, and scoring are used.
The rolling diagnostic fits through 2021, 2022, 2023, and 2024 for targets
2022, 2023, 2024, and 2025.

Before interpretation, the runner requires exact D5 parity with the checked-in
transfer decomposition artifact and verifies that the three position sums
reconstruct the existing aggregate wherever all three groups are observed.

Availability-only and within-season permutation controls are generated only for
a candidate that improves frozen D5 NLL. The permutation preserves season,
missingness, availability, and the observed value distribution while removing
the team-to-impact assignment.

## Result for the checked-in frozen inputs

The stored run recommends:

> retain D5 and stop defensive research

P3 (DB) improves frozen D5 NLL by 0.001935 and beats both its availability-only
and within-season permutation controls. It also improves NLL in all four rolling
origins. However, its availability coefficient is larger in absolute value than
the defensive-value coefficient in 2 of 4 fits, so it fails the predeclared
stability screen. P1 and P2 do not improve frozen D5; P4 is worse than D5.

The full report and machine-readable outputs are written to
`data/processed/defensive_transfer_position_groups/`.

## Artifacts

- `report.md` — complete research report and recommendation.
- `summary.json`, `candidate_definitions.json` — configuration, parity, source hashes, controls, and screen decisions.
- `position_coverage.csv`, `training_coverage.csv` — season/group and rolling training coverage.
- `aggregate_reconstruction.csv` — position-sum versus existing aggregate parity.
- `candidate_summary.csv`, `candidate_annual_metrics.csv` — frozen metrics.
- `rolling_metrics.csv`, `rolling_aggregate.csv` — rolling metrics and weighted aggregate.
- `control_metrics.csv` — availability-only and permutation metrics when controls are required.
- `coefficients.csv` — standardized coefficient paths, signs, changes, and source coverage.
- `paired_nll_summary.csv`, `paired_nll_by_team.csv` — paired team-season diagnostics.
- `plots/` — frozen deltas, rolling NLL, coefficient paths, and paired losses.
