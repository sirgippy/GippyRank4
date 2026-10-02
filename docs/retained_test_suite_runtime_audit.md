# Retained test-suite runtime audit (issue 162)

## Measurement

The before/after Python measurements use the same worktree, Python environment,
pytest options, and one-off timing hook. The hook summed pytest setup, call, and
teardown durations per test and module and recorded the wall time around the
pytest run. Both profiled runs used this WSL-compatible command because the
repository's WSL-worker guidance excludes the Starlette `TestClient` tests from
sandboxed runs:

```bash
pytest -q --ignore=tests/test_api.py -p no:cacheprovider --durations=30
```

The full API-inclusive suite
was run separately outside the sandbox after the changes.

| Local Python suite | Collected | Passed | Skipped | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Before | 691 | 691 | 0 | 157.4 s (2:37) |
| After | 691 | 691 | 0 | 87.2 s (1:27) |

The comparable profiled run is **70.2 seconds (44.6%) faster**. The issue's
latest CI report cited approximately 2:58 for the Python stage; that is useful
as context but is not the controlled before measurement above. After the
changes, the full local suite including API tests passed **696 tests in 113.6
seconds (1:54)**. A CI-to-local comparison is not treated as a controlled
runtime comparison.

The timings come from this environment and will vary with CPU and filesystem
load. The baseline was collected immediately before editing, and the after run
used the same profiling method. No tests were removed or skipped to produce the
after result.

## Principal runtime contributors

### Slowest tests

| Test | Before | After | Change |
| --- | ---: | ---: | --- |
| `test_week_5_kickoff_certainty_survives_replacing_mutable_cfbd_cache` | 77.6 s | 35.4 s | One production site-data rebuild removed |
| `test_machine_outputs_are_byte_identical_on_two_runs` / `test_machine_outputs_reproduce_committed_artifact_bytes` | 26.4 s | 12.5 s | One research-artifact generation removed |
| `test_exported_team_logos_are_canonical_and_audited` setup | 13.5 s | 12.6 s | Shared session fixture; retained |
| `test_loopy_diagnostic_separates_tree_feedback_from_cycle_feedback` | 2.5 s | 2.4 s | No change |

The logo test's setup time is the single session-scoped production site-data
fixture used by the module. Keeping that fixture shared avoids repeating the
production build; the remaining setup is necessary for the module's production
export assertions.

### Cumulative module runtime

| Module | Before | After |
| --- | ---: | ---: |
| `tests/test_site_data.py` | 98.4 s | 55.0 s |
| `tests/test_context_location_error_study.py` | 29.7 s | 15.6 s |
| `tests/test_team_schedule_validation.py` | 3.1 s | 3.1 s |
| `tests/test_performance_v1.py` | 2.7 s | 2.6 s |
| `tests/test_retrospective_publication.py` | 2.2 s | 2.0 s |

The first two modules contained the material avoidable work. Other retained
modules completed in under four seconds each in the profiled runs. The
site-data production fixture was already session-scoped, and synthetic site
builds in the team-artifact and posterior-snapshot tests were small and
scenario-specific, so sharing them would add mutable coupling without a
meaningful measured saving.

## Coverage accounting

| Previous test | Prior invariant | Change | Where it remains protected | Approximate saving |
| --- | --- | --- | --- | ---: |
| `tests/test_site_data.py::test_week_5_kickoff_certainty_survives_replacing_mutable_cfbd_cache` | Replacing newer mutable CFBD raw responses must not change the published kickoff certainty in the frozen snapshot's team and weekly schedule views. | Use the existing session-scoped production build as the pre-replacement reference, then build the isolated publication once after replacing the raw cache. Compare the same selected team and weekly game views exactly. | The edited regression test still checks initial response provenance, Chicago-local display for known kickoffs, UTC-calendar display for a TBD game, and exact equality of the selected Context/History team and weekly records. `test_week_5_kickoff_certainty_uses_only_its_exact_cfbd_responses` continues to cover the source-response hash and schedule values. | About 42 s for this test (77.6 s to 35.4 s). |
| `tests/test_context_location_error_study.py::test_machine_outputs_are_byte_identical_on_two_runs` | The research builder deterministically reproduces its machine-readable artifacts and records source provenance. | Run the builder once and compare every generated file byte-for-byte with the six committed study artifacts, including provenance and its source-hash check. | `test_machine_outputs_reproduce_committed_artifact_bytes` now enforces exact committed-artifact parity for all six outputs. This keeps the research-only artifact in the always-run suite and protects exact reproducibility. | About 14 s for this test (26.4 s to 12.5 s). |

These changes remove duplicate full executions, not assertions about distinct
behavior. The suite still has 691 collected local non-API tests and 696 tests
in the full API-inclusive run.

## Browser audit

The browser suite was reviewed separately. Its retained checks exercise DOM
rendering, navigation and URL state, timezone-aware date presentation, keyboard
disclosures, and desktop/mobile layout. The JavaScript tests cover probability
formatting and Playwright endpoint configuration, while Python tests validate
generated data; those checks do not replace the browser behavior assertions.
No browser tests were removed or narrowed.

The current run completed **113 passed, 9 skipped, 122 total in 33.2 seconds**.
The skipped cases are the existing viewport-specific exclusions. The issue's
previous CI report listed approximately 1:17 for this stage; the browser runs
use different execution conditions and this ticket makes no browser-runtime
reduction claim.

## CI and validation

CI job structure was left unchanged. The retained coverage was reduced before
considering parallel jobs, and no browser tests were relocated or excluded.
The production static API build and publication-link validation remain in the
existing workflow.

Validation performed after the edits:

- Full Python suite including `tests/test_api.py`: 696 passed in 113.6 s.
- Sandbox-compatible profiled suite: 691 passed in 87.2 s.
- Ruff: passed.
- `npm run test:js`: 17 passed.
- `npm run test:browser`: 113 passed, 9 skipped.
- Production `scripts/build_site_data.py` and static API publication/link checks: passed for 41 publications.
- `git diff --check`: passed after generated-artifact validation.
