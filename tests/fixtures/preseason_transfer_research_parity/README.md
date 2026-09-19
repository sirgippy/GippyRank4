# Frozen D5 / DB parity fixture

This is a compact 2021-to-2022 historical fixture for the production
derivation regression test. It keeps the raw endpoint shapes, stable player
identifiers, FBS and FCS source cases, and a checked-in expected feature table
small enough to review. The expected table is the frozen research output for
these inputs; the test must not regenerate it from the production output.

The cases cover resolved prior offensive usage, a natural zero with no
incoming offensive transfer, resolved DB impact, unresolved DB impact with
availability `0`, and an FCS source. The numeric DB values use the frozen
equal-weight mean of within-season DB-group standardized `log1p` components.
