# Retained test-suite runtime audit (issue 162)

## Measurement method

The controlled comparison used the latest fetched `origin/main` commit
`9a4601f341f4c9b6d9fd15adf4eb9a092d4fd2c6` and the issue branch after the
runtime changes. Both checkouts used the repository's pinned Python version
(`.python-version`), `uv sync --locked`, and the same one-off pytest profiling
hook. The hook invoked pytest with:

```text
-q --ignore=tests/test_api.py -p no:cacheprovider --durations=30
```

It summed each test's setup, call, and teardown durations and grouped those
durations by module. The sandbox-compatible local run excludes API tests
because Starlette `TestClient` can stall in the WSL sandbox; CI still runs the
full suite, including API tests. There were 691 collected and passed tests, with
no skips, in every profile.

The profiles ran in repeated base-then-branch pairs:
base, branch, base, branch, base, branch. Times below are the pytest-reported
suite wall times; module and test times are sums of pytest phase durations.

| Run | Checkout | Pytest time |
| ---: | --- | ---: |
| 1 | `origin/main` | 161.75 s |
| 2 | issue branch | 53.62 s |
| 3 | `origin/main` | 133.45 s |
| 4 | issue branch | 51.23 s |
| 5 | `origin/main` | 134.09 s |
| 6 | issue branch | 56.42 s |
| **Median** | **`origin/main`** | **134.09 s** |
| **Median** | **issue branch** | **53.62 s** |

The observed median reduction is **80.47 seconds (60.0%)**. The base checkout
always ran first within each pair, so the sequence is not counterbalanced and
cannot rule out page-cache or order effects. Reporting three runs per checkout
shows the measured spread, but the local percentage should not be read as a
fully order-neutral causal estimate. The large Week 5 test reduction and the
independent CI improvement corroborate the direction of the result. These
measurements are local to this WSL environment; CI timing is tracked separately
below.

## Main runtime contributors

The following module totals are medians across the three profiles for each
checkout:

| Module | `origin/main` | Issue branch |
| --- | ---: | ---: |
| `tests/test_site_data.py` | 90.87 s | 20.99 s |
| `tests/test_context_location_error_study.py` | 28.30 s | 16.23 s |
| `tests/test_team_schedule_validation.py` | 3.07 s | 3.16 s |
| `tests/test_performance_v1.py` | 2.54 s | 2.74 s |
| `tests/test_retrospective_publication.py` | 2.06 s | 2.07 s |

Selected test totals are also medians across the three profiles. The team-logo
test duration includes setup of the shared production site-data fixture.

| Test | `origin/main` | Issue branch | Change |
| --- | ---: | ---: | ---: |
| `test_week_5_kickoff_certainty_survives_replacing_mutable_cfbd_cache` | 71.44 s | 0.97 s | −70.47 s |
| Research artifact parity test | 25.21 s | 13.10 s | −12.11 s |
| `test_exported_team_logos_are_canonical_and_audited` including fixture setup | 12.73 s | 12.72 s | shared fixture retained |
| `test_loopy_diagnostic_separates_tree_feedback_from_cycle_feedback` | 2.35 s | 2.52 s | no material change |

The shared production fixture still builds the complete 41-publication site
data once for the tests that need it. The reduced Week 5 regression no longer
repeats that build. The research-artifact test remains one of the slower
retained tests because it regenerates all research outputs and checks their
bytes.

## Coverage and reproducibility accounting

### Week 5 cache replacement

The regression still starts with the production build's frozen Week 5 Context
and History records. It checks the original CFBD response hashes and retrieval
times, the selected known kickoffs in America/Chicago, the TBD kickoff's UTC
calendar display, and the team and weekly schedule views. It then replaces the
mutable CFBD raw responses with a later acquisition and builds only the two
Week 5 snapshots to `test-output/week-5-site-data`. The resulting team and
weekly views must exactly match the original ones. The focused build does not
write to the conventional `site/data` destination and does not produce the
static API tree.

This preserves the behavior under test while using the shared 41-publication
build as the baseline instead of running another full build. The targeted
regression's median phase duration is now 0.97 s; the shared full-build fixture
remains covered by the suite and its cost appears in the module total.

### Research output parity

`test_machine_outputs_reproduce_committed_artifact_bytes` generates the four
CSV files, `summary.json`, and `provenance.json` once, then compares all six
files byte-for-byte with the committed issue-154 artifact. The provenance
records hashes for four source/reference files (`team_seasons.csv`,
`feature_inventory.csv`, the source `summary.json`, and source `provenance.json`),
the analysis script's SHA-256, and hashes for the four generated CSVs and
generated summary. The project's locked environment supplies repeatable
dependency versions in CI.

This is not literally the previous assertion that two generations in one test
match each other. It compares the current generation against a separately
committed canonical materialization. That pins the expected bytes across test
runs and checks that the recorded code and input provenance still identify the
artifact being reproduced. It catches output drift and nondeterministic output
that differs from the canonical bytes. Intentional analysis or source changes
must update the committed artifact and its provenance together. The provenance
does not hash itself; byte comparison includes the complete provenance file.

## Browser audit

The browser suite was reviewed and left unchanged. Its retained checks exercise
DOM rendering, navigation and URL state, timezone-aware date presentation,
keyboard disclosures, and desktop/mobile layout. JavaScript checks cover
probability formatting and Playwright endpoint configuration. The latest local
browser run passed **113 tests and skipped 9 existing viewport-specific cases**
in 32.4 seconds. No browser-runtime reduction is claimed.

## CI parallelism evaluation

The previous workflow ran the Python suite, static API build, and browser suite
sequentially in one job. Successful CI run
[37053660314](https://github.com/sirgippy/GippyRank4/actions/runs/37053660314)
took 4:59; pytest took about 119 seconds, static API validation 41 seconds,
Chromium setup 22 seconds, and browser tests 77 seconds.

The Python suite and Ruff do not read generated static API output, so they run
in the independent `python` job. The JavaScript checks are also independent of
the generated API, but took under a second, so a third validation job would add
setup without a meaningful wall-time gain. The static API build and browser
tests stay sequential in `site`: the browser harness exercises the built site.
The `python` job checks `git diff --exit-code` after pytest and Ruff; `site`
retains its own final side-effect check for site generation and browser steps.

The required `validate` status is now a small final gate with `needs: [python,
site]`. Its `always()` condition makes it run even when a dependency fails or is
skipped, and it exits successfully only when both dependency results are
`success`. Thus a failed Python or site check fails the existing required
status context rather than leaving it skipped.

The first split-workflow run, before adding the required final gate,
[37059307804](https://github.com/sirgippy/GippyRank4/actions/runs/37059307804)
passed. The workflow took **2:45** wall time; its then-named `validate` site
path ran for **2:42** and the `python` job for **1:47**. The Python step passed
all **696 tests in 78.39 seconds**. The static API check validated **41
publications**, and the browser suite passed **113 tests with 9 existing
skips**. The two jobs used **4:29** combined runner time. For the same code
with pytest appended to the site path, the observed 2:42 site duration plus
the 78.39-second test step estimates about **4:00** serial wall time. Running
the Python checks concurrently therefore saves about **75 seconds** on the
critical path and adds about **29 seconds** of duplicated setup and check
overhead to aggregate runner time. Against the older 4:59 PR run, that workflow
was about 2:14 shorter; the total also includes the further Week 5 test
reduction and is not attributed solely to parallelism.

## Validation

The alternating sandbox-compatible profiles each passed all 691 collected
tests. A final local compatible run passed **691 tests in 62.32 seconds** while
the browser suite was running concurrently. The initial split-workflow CI run
passed all **696 tests in 78.39 seconds**, including the API tests. Ruff, 17
JavaScript tests, production static API validation for 41 publications, and
113 browser tests with 9 existing skips also passed. `git diff --check` and
both validation jobs' tracked-file side-effect checks passed. The final
required-gate run is linked from the PR.
