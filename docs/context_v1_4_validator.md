# Context 1.4 confirmation validator

`scripts/validate_context_v1_4.py` is the execution path registered in the
merged issue #164 protocol. Issue #166 builds and tests it without evaluating
the 2026 candidate. The script loads
`config/context_v1_4_validation.json` through the registered loader, checks the
frozen candidate and baseline identities, audits a supplied game corpus, then
constructs forecast states and scores only if the source passes the gate.

## Week terminology

The September 27 official state is the **Week 5 snapshot**. It uses evidence
through Week 4 and forecasts the **prospective Week 5 games**. The normal
production snapshot after those games will be Week 6. The merged #164 text and
some JSON field names call the prospective game slice “Week 6.” This is an
inherited naming error. The validator preserves the registered September 27
origin, 500 game full-sample floor, 40 game prospective floor, bootstrap, and
asymmetric decision rule, and names the output slice `prospective_week_5`.
It does not read or score game Week 6 in this confirmation run.

## Source and CLI contract

Use a canonical CFBD games CSV with the same fields as
`data/processed/cfbd/games.csv`. The validator accepts exact duplicate rows by
canonical game ID and aborts on conflicting duplicates. It checks every game
in the frozen Weeks 1–5 FBS/FCS schedule, plus any newly supplied in-scope
game. Missing, unfinished, ambiguously classified, or invalid games cannot
silently reduce the sample. An audited terminal exception or out-of-scope
change may be supplied in `--exceptions` as a JSON list of objects with
`game_id`, `disposition` (`terminal_exception` or `out_of_scope`), `reason`, and
`evidence`.

Run an outcome-blind source audit:

```bash
uv run python scripts/validate_context_v1_4.py \
  --games data/processed/cfbd/games.csv \
  --output /tmp/context-v1-4-source-audit \
  --check-source-only
```

The default execution path also refuses an incomplete source before any
candidate state or score is constructed. Its `failure.json` reports
`SOURCE_INCOMPLETE`, unresolved game IDs, source and protocol hashes, and
`candidate_scores_opened: false`.

Every invocation clears prior result products before validation begins. Any
later abort writes a deterministic `VALIDATION_ABORTED` failure artifact with
the failed stage and a `candidate_scores_opened` value that becomes true when
predictive scoring starts. This also removes partial outputs if artifact
writing itself fails.

Once Week 5 is complete, preserve the raw FBS and FCS CFBD responses and
write a source manifest. The manifest is a JSON object of this shape:

```json
{
  "source_kind": "cfbd_api_schedule",
  "retrieved_at_utc": "2026-10-05T12:00:00+00:00",
  "games_sha256": "<SHA-256 of canonical games CSV>",
  "source_files": [
    {"coverage": "fbs", "path": "raw-fbs.json", "sha256": "<SHA-256>"},
    {"coverage": "fcs", "path": "raw-fcs.json", "sha256": "<SHA-256>"}
  ]
}
```

Raw paths are resolved relative to the manifest. The validator checks their
hashes, matching game scope and outcomes in the canonical CSV, and each
completed game's actual kickoff. It aborts on a raw `startTimeTBD` flag unless
that game has an explicit audited exception. A
complete scoring run uses:

```bash
uv run python scripts/validate_context_v1_4.py \
  --games path/to/completed-games.csv \
  --source-manifest path/to/source-manifest.json \
  --output data/processed/context_v1_4_validation
```

The default output directory writes the registered machine artifacts there
and the human report to `docs/context_v1_4_validation_result.md`. A custom
output directory keeps every artifact, including the report, inside that
directory for fixture runs. No CLI flag changes alpha, origins, populations,
inference, metrics, bootstrap, or decision settings.

## Reproducible forecast inputs

`data/processed/context_v1_4_validation/validator_inputs.jsonl` freezes only
outcome-free 2026 inference rows and the expected schedule. Its content hash
is pinned by the validator. The one-time construction script records hashes of
the retained historical rank distributions, coaching tenure cache, Context
features, validated transfer reconstruction, Context 1.3 predictions, and
registered schedule responses. Those historical rank and coaching caches are
ignored local research inputs; the later execution needs only this committed
bundle and the registered artifacts.

The published Context 1.3 PMFs have twelve decimal places. The validator
verifies each against the exact frozen fitted-model output within its
serialization bound, then uses that exact PMF as the common Context 1.3 and
candidate source. It calls only `construct_candidate_prior`, verifies the
frozen positive-only moderation and identities, and checks reconstructed
baseline posteriors against the retained snapshot PMFs. Candidate, Context
1.3, and History inference receive matched included-game evidence and FCS
fallback semantics. Completed target-game outcomes enter only the final
predictive scoring stage.
