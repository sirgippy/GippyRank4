# Defensive-transfer experience stability study

Issue 109 tests whether the frozen imported defensive-experience proxy remains
useful beyond D5 as portal-era seasons enter the training panel. Run it with
the same immutable transfer cache used by the earlier studies:

```text
UV_CACHE_DIR=/tmp/gippyrank4-uv-cache uv run python \
  scripts/investigate_defensive_transfer_stability.py \
  --source-root /path/to/model-input-root \
  --transfer-root /path/to/frozen-transfer-cache
```

The runner evaluates exactly three candidates: production C0, D5, and D5 plus
`transfer_in_prior_defensive_experience_sum` with
`transfer_in_prior_defensive_experience_available`. It fits 2022 through 2025
using only seasons strictly before each target and preserves the existing
Context C 1.2 optimizer, penalty, preprocessing, H fallback, FBS population,
and August 15 cutoff.

Before rolling results are interpreted, the runner reproduces production C0,
the checked-in transfer-decomposition D5 result, and the corrected PR #107
defensive-experience result. The reference metrics are read from the checked-in
artifacts rather than rounded issue prose; a parity failure stops the study.

Defensive audit rows marked `complete` or
`no_incoming_defensive_transfer` are observed. The latter has natural numeric
zero. `partial` and `no_usable_defensive_transfer` rows are unresolved and
retain neutral zero plus the explicit availability indicator. No row is
dropped.

Generated outputs are written under
`data/processed/defensive_transfer_stability/`: the report, summary, per-target
and aggregate metrics, coverage counts, coefficient paths, availability
diagnostics, paired losses, and plots. The report ends with exactly one of the
two issue decisions: advance `D5 + defensive experience`, or retain `D5` and
stop defensive feature research.
