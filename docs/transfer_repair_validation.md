# Transfer repair validation modes

The retained test suite is designed to run from a clean clone. Availability
classification and repair-core tests use the tracked fixtures under
`tests/fixtures/`; committed-artifact integrity tests read only checked-in
outputs under `data/processed/transfer_data_repair/`. These checks do not
discover local data directories or invoke a provider.

The complete historical and 2026 source replay is a separate research
validation. It requires the original historical payloads, the reacquired 2026
payloads and audit roots, and the frozen materializer inputs. Those payloads
remain gitignored and must be supplied from their retained local research
store. Do not replace missing manifest inputs with a newly fetched response.

With those original inputs present, run the integrated replay explicitly and
pass each source root:

```sh
uv run python scripts/build_integrated_transfer_repair_audit.py \
  --output data/processed/transfer_data_repair \
  --current-root data/processed/preseason/context_v1_3_2026_reacquired_20260930 \
  --raw-root data/raw/cfbd/preseason/transfers_reacquired_20260930 \
  --historical-raw-root data/raw \
  --before-current-root data/processed/preseason/context_v1_3_2026_reconstruction \
  --materializer-source-root .
```

The script verifies historical source hashes before replay and records the
selected input hashes in its generated summary. The command is research
validation, not part of ordinary CI.
