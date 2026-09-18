# Defensive-transfer predictive-value study

Issue 106 evaluates whether the frozen PR #105 defensive-transfer measurements
add predictive value beyond D5 (`total returning production + incoming prior
offensive usage`). The research-only runner is:

```text
uv run python scripts/investigate_defensive_transfer_predictive_value.py
```

It fits through 2021 and evaluates the unchanged 2022–2025 production FBS
population. If the large generated rank-distribution input is stored outside a
checkout, pass `--source-root` for that input root and pass `--output` for the
repository artifact directory.

The runner reuses the existing Context fit, H fallback, optimizer retry,
regularization, and scoring implementation. It attaches the existing
offensive transfer oracle before evaluating the E0–E6 variants, and it reads
the frozen defensive audit from
`data/processed/defensive_transfer_audit/team_season_features.csv`.

Defensive audit rows marked `complete` or
`no_incoming_defensive_transfer` are observed; the latter has an aggregate
value of zero. `partial` and `no_usable_defensive_transfer` rows are unresolved
and receive neutral zero imputation plus explicit availability indicators.
No team-season is removed. Availability-only and within-season permutation
controls are generated for E4, E5, and E6.

The generated report and machine-readable outputs are under
`data/processed/defensive_transfer_predictive_value/`. The study does not
change production model artifacts, redesign either defensive feature, or run
the subsequent stability study.
