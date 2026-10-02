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

To reduce cache and machine-load bias, the profiles ran in alternating order:
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

The median reduction is **80.47 seconds (60.0%)**. The first base run was
slower than its next two runs; alternating the checkout order and reporting
medians avoids presenting that first run as the baseline. These measurements
are local to this WSL environment. CI timing is tracked separately below.

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
browser run before the CI split passed **113 tests and skipped 9 existing
viewport-specific cases** in 33.2 seconds. No browser-runtime reduction is
claimed.

## CI parallelism evaluation

The previous workflow ran the Python suite, static API build, and browser suite
sequentially in one job. In successful CI run
[37053660314](https://github.com/sirgippy/GippyRank4/actions/runs/37053660314),
the whole job took 4:59 and the Python step took about 119 seconds. The
remaining validation path took about three minutes. The Python suite does not
consume the generated static API, so it can run independently while the
existing validation job builds and checks the site. The workflow now has a
separate Python job for pytest and Ruff, while the `validate` job retains the
static API build, JavaScript checks, browser setup and tests, and diff checks.

This duplicates checkout and Python environment setup. Based on that CI run,
the extra checkout, uv setup, and locked dependency installation add about 19
seconds of runner time. They run concurrently with the existing validation
path, so expected workflow wall time is about three minutes instead of 4:59,
while aggregate runner time increases by about those 19 seconds. The Python
job also checks `git diff --exit-code` after pytest and Ruff, preserving the
tracked-file side-effect check that was previously after the tests in the
single job. The `validate` job retains its own final side-effect check for the
static build and browser steps. The exact wall and summed job times will be
recorded from the first CI run with the split workflow.

## Validation

The alternating sandbox-compatible profiles each passed all 691 collected
tests. The full CI suite remains the validation path for the API tests. Other
checks on the existing PR before this follow-up passed: Ruff, 17 JavaScript
tests, production static API validation for 41 publications, and 113 browser
tests with 9 existing skips. Final post-change CI results will be added here
after the updated workflow completes.
