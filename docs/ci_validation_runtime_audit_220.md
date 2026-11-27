# CI validation runtime audit (issue 220)

## Measurement method and workload

The initial baseline is fetched `origin/main` at
`bfcbd7003701f9cc4a91fe03149bc00aa5e55257`. It ran four times on one
exact SHA via [Actions run 39051148611](https://github.com/sirgippy/GippyRank4/actions/runs/39051148611).
The fourth replay was interleaved with exploratory PR runs on October 18,
2026 to check day-to-day runner differences. A later rankings publication
advanced main to `d4a734be9e3f55eef7a866f8c0d703d8c03726b0` and added
three publications; the final matched comparison uses that new main SHA
and its PR merge checkout. The first measured optimization
changes only the `OPENBLAS_NUM_THREADS` environment of the full pytest step.
The test collection, application code, numeric settings, and site checks are
unchanged. Three same-revision optimized attempts are reported below.

Workflow wall time is Actions `updated_at - run_started_at`. Queue time is
the first job start minus `run_started_at`; execution is last job completion
minus first job start. Aggregate runner time sums the `python`, `site`,
and `validate` job durations. Step times use Actions timestamps rounded to
whole seconds. All compared CI runs used `ubuntu-24.04` image
`20261012.334.1`, runner version `2.337.0`, CPython `3.13.16` from
`.python-version`, uv `0.13.1` with `uv sync --locked`, Node 22, and locked
Playwright `1.63.0`. Baseline runs used the `push` event and PR runs used
`pull_request`; this event difference is a comparison limitation.

The initial revision has 2,099 tracked files. Locally, `data/` occupies about
1.1 GiB, committed `site/data/` about 333 MiB, and a generated
`site/api/v1/` tree about 630 MiB. The production build validates **49
publications**. Checkout and working-tree materialization remain material
costs even when dependency installation is cached.

## Initial exact-main baseline (49 publications)

| Attempt | Queue | Execution | Workflow wall | Python job | Site job | Aggregate runner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 s | 252 s | 255 s | 242 s | 192 s | 440 s |
| 2 | 4 s | 256 s | 261 s | 250 s | 147 s | 400 s |
| 3 | 3 s | 249 s | 253 s | 244 s | 163 s | 410 s |
| 4 | 3 s | 249 s | 252 s | 243 s | 173 s | 420 s |
| **Median; range** | **3 s; 2–4** | **250.5 s; 249–256** | **254 s; 252–261** | **243.5 s; 242–250** | **168 s; 147–192** | **415 s; 400–440** |

| Attempt | Python checkout | Pytest | Site checkout | Static API build | Chromium install | Browser tests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 62 s | 169 s | 25 s | 48 s | 24 s | 73 s |
| 2 | 44 s | 198 s | 21 s | 33 s | 21 s | 53 s |
| 3 | 44 s | 194 s | 25 s | 42 s | 22 s | 59 s |
| 4 | 44 s | 191 s | 23 s | 44 s | 23 s | 63 s |
| **Median; range** | **44 s; 44–62** | **192.5 s; 169–198** | **24 s; 21–25** | **43 s; 33–48** | **22.5 s; 21–24** | **61 s; 53–73** |

These four runs establish the original bottleneck and runner variation; they
are not used to calculate final savings after the 52-publication PR merge.
The Python job is the critical path in all four attempts. Locked
Python installation took 2 seconds in every attempt; `npm ci` took 1–4
seconds. They were small beside pytest, checkout, static API generation,
and Chromium validation.

## Profile of retained work

The local baseline and one-thread runs used the same WSL worktree, locked
dependencies, Python 3.13.15, and uv 0.12.9. Each ran pytest with
`-q --ignore=tests/test_api.py -p no:cacheprovider` and duration reporting.
This follows the documented WSL-compatible split; CI includes API tests.

| Local compatible suite | Default numeric threads | One OpenBLAS thread |
| --- | ---: | ---: |
| Pytest wall | 168.01 s | 127.30 s |
| Complete synthetic validator fixture setup | 94.84 s | 61.43 s |
| Full production site-data fixture setup | 15.35 s | 15.16 s |
| Research artifact byte-parity test call | 12.37 s | 12.16 s |
| Result | 1,056 passed; 0 skipped | 1,056 passed; 0 skipped |

The default run's phase report summed 110.24 seconds of reported setup and
49.52 seconds of reported calls. No individual teardown reached the 0.005
second display threshold. Its largest module totals were
`tests/test_context_v1_4_validator.py` at 96.33 seconds,
`tests/test_site_data.py` at 24.04 seconds, and
`tests/test_context_location_error_study.py` at 15.41 seconds.
Sub-threshold phases and pytest overhead explain why the displayed phase
sum is below suite wall time.

The unchanged validator fixture executes two complete independent CLI
validations to compare every machine artifact and report byte for byte. A
one-off timing probe around that retained test, with one OpenBLAS thread,
measured two forecast-state constructions at 25.76 seconds total; the 30
registered posterior inferences inside them took 22.23 seconds. Scoring the
game source twice took 39.21 seconds. Loading inputs and writing artifacts
together took about 0.02 seconds. The instrumented test passed and took
73.47 seconds, so those stage totals are a breakdown rather than a substitute
for the uninstrumented 61.43-second fixture measurement.

The full production site build separately took 42.31 seconds locally and
left tracked outputs unchanged. The CI-shaped local browser run against that
fresh build took 48.27 seconds: 113 passed, 9 existing viewport-specific
skips, and no retries. Its CI configuration already uses two workers and up
to two retries. JavaScript checks passed all 17 tests.

## Evaluated changes and cache behavior

The retained change sets `OPENBLAS_NUM_THREADS=1` only for pytest. At the
validator's small matrix dimensions, thread coordination outweighed
parallel work in the local paired run. The full suite passed with exact
artifact comparisons, posterior parity checks, deterministic generation,
and model precision intact. No fixture, test, assertion, tolerance, scenario,
or model path was removed or narrowed. There is consequently no
before/after coverage mapping for changed tests: collected/passed counts
remain 1,056 locally and 1,061 in CI, including all five API tests. The
independent two-run validator check and full production site build remain.

We tested a shallow site checkout that fetched missing diff endpoints before
the complete committed whitespace diff. It passed the PR checks; a local
multi-commit Git fixture also confirmed that a missing base endpoint is
fetched and trailing whitespace still fails. It did not improve checkout
time: two exploratory PR attempts took 44 and 45 seconds for site checkout,
versus 21–25 seconds for the four full-history baseline attempts, including
23 seconds on the interleaved October 18 baseline. The exploratory PR wall
times were 220 and 246 seconds, with site jobs of 212 and 184 seconds. We
therefore retained the full-history checkout and existing diff step. See
[exploratory Actions run 39053252936](https://github.com/sirgippy/GippyRank4/actions/runs/39053252936).

The baseline Actions logs show uv and npm cache hits; setup and restore took
1–6 seconds, and post-job steps did not save the already-hit caches. The
exploratory PR attempts also hit both caches. A local empty uv cache with a
forced locked reinstall succeeded in 39.14 seconds; repeating from the warm
cache took 0.40 seconds. `npm ci` succeeded in 2.16 seconds with an empty
npm cache and 0.61 seconds when repeated warm. These are local cold-path
checks, not a controlled Actions cold-cache benchmark. We added no cache and
do not reuse generated site artifacts or test results across revisions.

Separate-job pytest sharding was initially deferred because an extra checkout
would increase runner work. A subsequent experiment overlaps two disjoint
pytest processes within the existing Python job; its measurements and
isolation checks are recorded below.
Chromium remains after the freshly generated static API in the same job;
browser workers and retry policy are unchanged. Superseded-run cancellation
was also left unchanged because it would alter workflow behavior without a
measured benefit in these exact-revision replays.

The required `validate` job still uses `needs: [python, site]` and
`always()`, and succeeds only if both dependency results equal `success`.
A failure, cancellation, or unexpected skip of either dependency cannot pass
that gate. Ruff, JavaScript, desktop/mobile Chromium, full static API
generation and count/link validation, committed-diff whitespace, and both
tracked-file side-effect checks remain required. CI's full Python suite
includes `tests/test_api.py`; the WSL sandbox split is unchanged. All
sampled completed CI runs passed without observed browser retries or flaky
classifications. The browser failure step still uploads `test-results/` and
`playwright-report/`; no validation command has a caught or ignored exit
code. The local whitespace-failure fixture reported the offending file and
line. We did not inject an intentional failing Actions run.

## First optimization: one OpenBLAS thread

The comparable current main revision is
`d4a734be9e3f55eef7a866f8c0d703d8c03726b0`, after the October 18
rankings publication. The three optimized PR replays checked out the same
`5460370` merge of the BLAS-only branch into that revision. The current tree
has 2,136 tracked files, including 1,137,233,773 bytes under `data/` and
361,624,096 bytes under `site/data/`. The full production build validates
**52 publications** in both versions. The matched baseline is
[Actions run 39131201203](https://github.com/sirgippy/GippyRank4/actions/runs/39131201203);
the optimized revision is
[Actions run 39139937720](https://github.com/sirgippy/GippyRank4/actions/runs/39139937720)
at `d06ac0dd6c2ae28854ffa7b135c1afa5712ddccc`.
Run order was main 1, PR 1–2, main 2, PR 3, main 3. All measured jobs
ran from clean Actions checkouts with warm uv and npm caches.

| Revision, attempt | Queue | Execution | Workflow wall | Python job | Site job | Aggregate runner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main 1 | 3 s | 267 s | 271 s | 263 s | 186 s | 451 s |
| Main 2 | 3 s | 264 s | 268 s | 258 s | 192 s | 454 s |
| Main 3 | 3 s | 265 s | 268 s | 259 s | 128 s | 391 s |
| **Main median; range** | **3 s; 3–3** | **265 s; 264–267** | **268 s; 268–271** | **259 s; 258–263** | **186 s; 128–192** | **451 s; 391–454** |
| PR 1 | 2 s | 221 s | 225 s | 216 s | 135 s | 354 s |
| PR 2 | 3 s | 244 s | 248 s | 238 s | 143 s | 385 s |
| PR 3 | 3 s | 247 s | 251 s | 243 s | 189 s | 434 s |
| **PR median; range** | **3 s; 2–3** | **244 s; 221–247** | **248 s; 225–251** | **238 s; 216–243** | **143 s; 135–189** | **385 s; 354–434** |

| Revision, attempt | Python checkout | Pytest | Site checkout | Static API build | Chromium install | Browser tests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main 1 | 52 s | 199 s | 31 s | 49 s | 23 s | 68 s |
| Main 2 | 49 s | 198 s | 25 s | 57 s | 21 s | 76 s |
| Main 3 | 49 s | 200 s | 20 s | 33 s | 16 s | 47 s |
| **Main median; range** | **49 s; 49–52** | **199 s; 198–200** | **25 s; 20–31** | **49 s; 33–57** | **21 s; 16–23** | **68 s; 47–76** |
| PR 1 | 52 s | 154 s | 21 s | 33 s | 18 s | 49 s |
| PR 2 | 54 s | 172 s | 22 s | 36 s | 19 s | 52 s |
| PR 3 | 48 s | 184 s | 23 s | 56 s | 20 s | 78 s |
| **PR median; range** | **52 s; 48–54** | **172 s; 154–184** | **22 s; 21–23** | **36 s; 33–56** | **19 s; 18–20** | **52 s; 49–78** |

Median workflow wall time fell from **268 to 248 seconds**, a **20-second
(7.5%)** reduction. All three optimized wall times were below the fastest
matched baseline by at least 17 seconds. Queue medians stayed at 3 seconds,
so the reduction occurred in execution. The measured Python test step fell
from a 199-second median to 172 seconds, **27 seconds (13.6%)**; the complete
Python job fell from 259 to 238 seconds, **21 seconds (8.1%)**.

Aggregate runner time fell from a 451-second median to 385 seconds,
**66 seconds (14.6%)**. The site job and its unchanged build/browser steps
also varied widely (site ranges 128–192 seconds on main and 135–189 seconds
on the PR), so the entire runner-time difference cannot be attributed to
the BLAS limit. Checkout remains material: Python checkout took a
49-second main median and 52-second PR median. The site job stayed behind
Python on the critical path in every matched run. Browser tests ran after
the freshly built API throughout.

All three optimized CI attempts passed 1,061 Python tests, Ruff, 17
JavaScript tests, the 52-publication static API count/link check, and
113 desktop/mobile browser tests with the same 9 skips. The committed-diff
whitespace and both side-effect checks passed, followed by the unchanged
required `validate` gate. No browser retries or flaky classifications were
observed. The final documentation-only commit is checked separately below
because adding this audit changes the Git SHA without changing the measured
validation commands or inputs.

## Prior documentation revision

The required checks attached to [PR 221](https://github.com/sirgippy/GippyRank4/pull/221)
validated the documentation-only revision at `33dae617` from a fresh checkout
in [Actions run 39141831934](https://github.com/sirgippy/GippyRank4/actions/runs/39141831934).
Its first attempt is excluded from the earlier controlled timing medians
because its SHA differs from the three optimized replays; the validation
commands and publication inputs remained unchanged.

## Second optimization: overlap disjoint pytest processes

The complete Context 1.4 validator module uses two independent full CLI runs
and dominates the one-thread Python test step. The current workflow runs that
module in one pytest process and all remaining retained tests in another,
within the same checkout and Python job. Both processes keep
`OPENBLAS_NUM_THREADS=1`. The first process selects
`tests/test_context_v1_4_validator.py`; the second uses pytest's
`--ignore=tests/test_context_v1_4_validator.py`. Thus the split partitions
the original collection without narrowing assertions, fixtures, inference,
or scenarios. Both exit codes are checked after both finish, and both logs
are printed even if one fails. Ruff and the tracked-file side-effect check
still follow the combined step. The required `validate` gate still depends
on successful `python` and `site` jobs.

On the same WSL checkout, two concurrent processes passed 26 validator and
1,030 remaining tests (1,056 total, excluding the documented five API tests).
Their wall times were 72.87 and 75.72 seconds, respectively, versus 127.30
seconds for the prior one-thread combined suite. The CI split passed 26 and
1,035 tests (1,061 total, including the five API tests) in its first full
run. Simulated failures of either process produced a failing step and printed
both logs with the corresponding exit code. No separate checkout or extra
runner was added. Exact artifacts and the independent two-run validator
comparison remain intact.

### Matched CI measurement of the split

The one-thread baseline is three attempts of
[Actions run 39141831934](https://github.com/sirgippy/GippyRank4/actions/runs/39141831934)
at `33dae6176bee03b57dcd8af20d585e2201e02aaf`, checked out as PR merge
`c6c552b` into `d4a734b`. The split is three attempts of
[Actions run 39146750789](https://github.com/sirgippy/GippyRank4/actions/runs/39146750789)
at `8b9800363821130ad2b3fd9f865b6682e63a6894`, checked out as PR merge
`7e36934` into the same main revision. Run order was baseline 1, split 1,
baseline 2, split 2, baseline 3, split 3, all on October 18, 2026. These are
the same `pull_request` event, Ubuntu image `20261012.334.1`, runner version
`2.337.0`, locked Python/uv/Node/Playwright versions, 52 publications, and
warm uv/npm cache keys. The Python and site jobs still start in parallel.
The first baseline ran earlier in the day; the later attempts are closely
interleaved. Unchanged site-job variation remains a measurement limitation.

| Revision, attempt | Queue | Execution | Workflow wall | Python job | Site job | Aggregate runner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One thread 1 | 2 s | 234 s | 237 s | 230 s | 193 s | 425 s |
| One thread 2 | 3 s | 242 s | 245 s | 236 s | 167 s | 406 s |
| One thread 3 | 3 s | 241 s | 244 s | 235 s | 192 s | 430 s |
| **Median; range** | **3 s; 2–3** | **241 s; 234–242** | **244 s; 237–245** | **235 s; 230–236** | **192 s; 167–193** | **425 s; 406–430** |
| Split 1 | 2 s | 197 s | 200 s | 167 s | 193 s | 362 s |
| Split 2 | 3 s | 201 s | 205 s | 171 s | 195 s | 369 s |
| Split 3 | 3 s | 217 s | 221 s | 162 s | 211 s | 377 s |
| **Median; range** | **3 s; 2–3** | **201 s; 197–217** | **205 s; 200–221** | **167 s; 162–171** | **195 s; 193–211** | **369 s; 362–377** |

| Revision, attempt | Python checkout | Pytest step | Site checkout | Static API build | Chromium install | Browser tests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One thread 1 | 47 s | 173 s | 23 s | 57 s | 20 s | 79 s |
| One thread 2 | 48 s | 177 s | 25 s | 45 s | 21 s | 62 s |
| One thread 3 | 47 s | 179 s | 23 s | 56 s | 21 s | 79 s |
| **Median; range** | **47 s; 47–48** | **177 s; 173–179** | **23 s; 23–25** | **56 s; 45–57** | **21 s; 20–21** | **79 s; 62–79** |
| Split 1 | 55 s | 101 s | 24 s | 57 s | 19 s | 78 s |
| Split 2 | 48 s | 112 s | 31 s | 50 s | 24 s | 73 s |
| Split 3 | 46 s | 106 s | 36 s | 58 s | 20 s | 78 s |
| **Median; range** | **48 s; 46–55** | **106 s; 101–112** | **31 s; 24–36** | **57 s; 50–58** | **20 s; 19–24** | **78 s; 73–78** |

Median workflow wall time fell from **244 to 205 seconds**, saving **39
seconds (16.0%)**. Every split attempt finished at least 16 seconds below
the fastest one-thread attempt. Queue medians were 3 seconds in both groups.
The Python test step fell from 177 to 106 seconds, saving **71 seconds
(40.1%)**, and its full job fell from 235 to 167 seconds, saving **68
seconds (28.9%)**. The split put the unchanged site job on the critical path
in all three attempts; its 193–211 second range limits further wall-time
savings from Python alone. Aggregate runner time fell from a 425-second
median to 369 seconds, **56 seconds (13.2%)**, without an added runner.
The site job varied in both groups, so its timing difference should not be
attributed to the pytest split.

For context, the exact-main three-run baseline on the same 52-publication
revision above had a 268-second median, versus 205 seconds for the final
split: **63 seconds (23.5%)** less workflow wall time. Its aggregate runner
median was 451 seconds, versus 369 seconds here: **82 seconds (18.2%)** less.
Those runs occurred earlier and used `push` rather than `pull_request`, so
the two staged comparisons provide stronger attribution for each change.

Each split attempt passed 26 validator and 1,035 remaining Python tests,
Ruff, 17 JavaScript tests, the 52-publication production API checks, 113
desktop/mobile browser tests with the same 9 skips, whitespace and both
side-effect checks, and the required `validate` gate. No browser retries or
flaky classifications were observed. No test explicitly reads pytest cache
metadata, and the processes use separate pytest sessions and temporary test
roots. The workflow waits for both processes and reports both logs before
failing if either exits nonzero. The unchanged gate rejects a failed,
cancelled, or unexpectedly skipped Python or site job. As before, the
Actions timing samples used warm dependency caches; the earlier local
cold/warm installation checks apply to the same locked dependency setup,
which completes before either pytest process starts. No controlled cold-cache
Actions replay or intentional failing Actions run was performed.
