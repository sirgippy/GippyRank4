# Production-safe preseason transfer pipeline

Issue 113 adds a preprocessing boundary for the transfer-aware Context
features. The network-facing command is
`scripts/fetch_preseason_transfer_snapshots.py`; the offline derivation
command is `scripts/build_preseason_transfer_features.py`.

## Operational flow

For a target season, run the acquisition command on or before August 15:

```text
CFBD_API_KEY=... uv run python scripts/fetch_preseason_transfer_snapshots.py \
  --season 2027
```

The command captures:

- `/player/portal` for the target season;
- `/player/usage` for the prior season;
- `/stats/player/season` for conservative offensive applicability evidence;
- `/roster` for prior-season player identity and position;
- `/games/players` for prior-season defensive box-score components.

Raw response bytes are stored below
`data/raw/cfbd/preseason/transfers/` and are never overwritten. A refresh
uses a new versioned filename, keeps the previous response, and updates the
manifest's canonical flag. Each manifest record includes the endpoint,
parameters, retrieval timestamp, target cutoff, record count, source filename,
and SHA-256.

The derivation step is offline and consumes only canonical paths listed in the
manifest:

```text
uv run python scripts/build_preseason_transfer_features.py \
  --manifest data/raw/cfbd/preseason/transfers/manifest.json
```

It writes `preseason_transfer_features.csv`, audit CSVs, a data-quality JSON
report, `feature_provenance.json`, and a report under
`data/processed/preseason/transfer_features/`. The three model-facing columns
are exactly:

- `transfer_in_prior_usage_sum`;
- `transfer_in_prior_defensive_impact_db_sum`;
- `transfer_in_prior_defensive_impact_db_available`.

The attach-only function
`gippyrank.preseason_transfer.merge_preseason_transfer_features` is the
integration point for a later Context 1.3 preprocessing step. Model fitting
does not fetch CFBD or discover raw files.

## Identity and cutoff policy

Player joins use a verified shared source ID when one exists. Otherwise the
fallback is normalized player name plus normalized source team. Exact
normalized matches and explicit aliases are allowed; ambiguous or conflicting
matches are retained as unresolved. Fuzzy matching is never used.

Team joins use the canonical season-specific team table and
`data/reference/preseason_team_aliases.csv`. Destinations must resolve to the
canonical FBS population before they contribute to a feature. Player-name
corrections belong in
`data/reference/preseason_player_aliases.csv`, not in procedural code. Alias
rows may be scoped by season and source team.

The derivation refuses a canonical snapshot whose retrieval timestamp is after
the target season's August 15 cutoff. A retrospective research response can
still be kept for research parity, but it cannot be promoted into the
production artifact.

## Feature semantics

`transfer_in_prior_usage_sum` is the sum of prior-season `usage.overall` for
incoming transfers with successfully resolved applicable offensive usage. The
audit retains resolved, legitimate-zero/non-applicable, unresolved applicable,
and applicability-undetermined classifications. Missing usage is not
automatically a failure: defensive and special-teams players without
offensive participation can be legitimate zero, while offensive-line absence,
source gaps, and ambiguous identity remain unresolved.

DB is exactly `CB`, `DB`, `S`, `FS`, `SS`, or `NB`. Prior DB impact is the
equal-weight mean of within-season × DB-group standardized `log1p` tackles,
passes defended, and interceptions. No position inference is performed for an
unknown label.

For a destination team with no incoming DB transfers, the DB sum is `0` and
availability is `1`. When every known incoming DB transfer resolves, the sum
is the sum of those player impacts and availability is `1`. If any known
incoming DB transfer is unresolved, the numeric value is neutral-imputed to
`0` and availability is `0`. The unresolved player records and reasons remain
in the audit and provenance artifacts.

## Parity and provenance

`validate_research_parity` compares the three production fields to a frozen
research fixture with an explicit tolerance. The production implementation
uses the same D5 applicability rules and DB impact construction as the
research modules; no approximate semantic substitute is accepted.

`feature_provenance.json` maps
`season|destination_team_id|feature` to source snapshot hashes and the
contributing player/portal records. Teams with natural zeros still have an
entry with an empty contributor list; unavailable features retain unresolved
records and reasons.

Historical research results use retrospective oracle data and are not evidence
that historical feature values were available as-of those preseason cutoffs.

Production safety begins only for seasons captured by the immutable preseason
snapshot process.
