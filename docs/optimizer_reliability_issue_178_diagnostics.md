# Issue 178: History optimizer diagnostics

## Scope

This branch starts at `origin/main` commit
`191e774c101d989589d9908636dd3f3ffc2f97ea`. It adds a diagnostic script and a
GitHub Actions experiment. The original fit still raises on an unsuccessful
SciPy result. Separate, explicitly labeled diagnostic passes restart from
captured finite `ABNORMAL` iterates with `maxls=100`. If a later distinct fit
in the reproducer also fails, it gets its own captured-iterate restart in
another pass. The script does not retry the same failed fit repeatedly, alter
model code, or turn a failed fit into a successful baseline result.

The script invokes
`test_future_history_builder_matches_the_history_1_1_annual_procedure` with
the test's own `_write_canonical_history_2027_inputs(include_generic=True)`
fixture. The cited CI run failed in the first fit in that test and in nine
other tests that reconstruct the same small History fit. The diagnostic also
captures the canonical History build later in the test when the initial fit
succeeds.

The workflow launches five independent `ubuntu-24.04` jobs. Each job runs the
reproducer in two separate processes on the same VM: once with its default
environment and once with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. Each process writes one JSON artifact containing runtime
configuration, full `OptimizeResult` fields, reevaluated objective and
gradient, iterate/evaluation finiteness, bound diagnostics, coefficients,
gradient checks, prediction hash, and PMF-map hash. On an abnormal finite
result, the artifact also records the separate manual restart.

## Local repeated runs

The exact test path was run 100 times locally: 50 default-thread runs and 50
single-thread runs. The WSL host reported Python 3.13.15, NumPy 2.5.2, SciPy
1.18.1, 16 logical CPUs, and an Intel Core i9-9900K. Thread variables were
unset in the default mode and set to one in the single-thread mode.

All 100 test runs passed. Every main History fit returned status 0 after 67
iterations and 140 function evaluations with objective
`0.839382770347525`. Every transition fit returned status 0 after 7 iterations
and 8 function evaluations with objective `1.857601155500131`. Fitted model
metadata, emitted predictions, and PMF maps were byte-identical between every
run and both threading modes:

- Model metadata SHA-256: `b46ace5528f4c38357ac2ce59937452586fb4183e21164a3c55f2b49b8c486c4`
- Prediction CSV SHA-256: `4f1e5143471a39fb244c02aa87ac4383f2c330e4f14db9cd8046e0616a1a5a60`
- PMF-map SHA-256: `185f8b08aa09ead64a483493388bd1a719dbc82ab03ca312e8e13ed037d0309c`

This local sample did not reproduce an abnormal optimizer result, so it cannot
estimate the hosted failure rate or test the manual restart. It did reveal a
gradient inconsistency on the main History fit: central differences at the
successful result differed from the analytic gamma gradient by as much as
`0.0267`; beta-gradient error stayed below `4e-11`. Four of the 17 training
rows had scale predictors below the `-5` clipping bound. The transition fit,
which had no clipped scale predictors, had a maximum gradient error below
`1e-10`.

The best-supported local lead is that the objective clips the log-scale
predictor before exponentiation, while its analytic gamma derivative does not
include the derivative of that clipping operation. The main fit therefore
reports relative-objective convergence with a maximum analytic-gradient
magnitude of about `0.043`, rather than projected-gradient convergence. This
is consistent with line-search sensitivity, but the local run alone does not
prove it caused the runner-specific `ABNORMAL` result. The hosted experiment
below checks the finite iterate and gradient signature, single-thread
behavior, and whether a manual larger-`maxls` restart converges to the same
coefficients and PMFs as an ordinary successful run.

## Hosted results

The five-runner experiment is recorded by the pull request's
`Issue 178 optimizer diagnostics` workflow. Its per-process JSON files are
uploaded as artifacts named `issue-178-runner-1` through
`issue-178-runner-5`. The final revision's [workflow run
37495473551](https://github.com/sirgippy/GippyRank4/actions/runs/37495473551)
used five independent `ubuntu-24.04` jobs, each running default and
single-thread subprocesses on the same VM.

Four VMs passed the exact test in both modes. One VM, reporting an Intel Xeon
Platinum 8573C, failed in both modes with status 2 (`ABNORMAL: `) after 79
iterations and 261 objective evaluations. Its final iterate, objective, and
gradient were finite. The failure's result-vector SHA-256 was
`0477d31276131f4926c0e6c91294b00e155c8cbdf68def47b8253c05f952ff20`; the
objective reevaluated to `0.8391699592385299`. Default and single-thread
results were byte-identical on that VM. The four passing VMs also produced
identical results across both modes: status 0, 67 iterations, 140 evaluations,
and objective `0.839382770347525`.

The failure and pass outcomes varied across two five-VM matrix runs: the first
run had three failing VMs, while this final-revision run had one. The Intel
Xeon 8573C failed in both runs, the EPYC 7763 passed whenever scheduled, and
the EPYC 9V74 had mixed outcomes. This small sample does not establish a CPU
cause. Thread limiting made no difference within any VM. Across all failures,
the raw iterate was finite and identical, with the same gradient signature:
central differences disagreed with the analytic gamma gradient by at most
`0.0267188`, while beta-gradient error stayed below `5e-11`; four of 17 rows
were below the log-scale clipping bound.

The full-suite [CI run
37495473455](https://github.com/sirgippy/GippyRank4/actions/runs/37495473455)
also reproduced the failure on its standard test runner: 10 tests failed with
`preseason optimizer failed: ABNORMAL`, including the exact reproducer and its
nine dependent tests; 838 tests passed. The site check passed. The diagnostic
matrix itself completed successfully and uploaded all five artifacts.

The manual `maxls=100` restart made the exact test pass after separately
restarting the failing legacy and canonical History fits from their own
captured iterates. Each restarted fit returned status 0 after one iteration
and 28 evaluations at objective `0.8391699592385298`. Its maximum analytic
gradient was still `0.00884`, and the finite-difference gamma-gradient
discrepancy remained about `0.02672`. The restarted output did **not** match
the ordinary successful fit:

- Normal-fit PMF-map SHA-256: `185f8b08aa09ead64a483493388bd1a719dbc82ab03ca312e8e13ed037d0309c`
- Restarted PMF-map SHA-256: `2e22655ff4b57dc8cdd703a18889763f08df93dfb8aada28428ce0171cd91d66`
- Normal prediction CSV SHA-256: `4f1e5143471a39fb244c02aa87ac4383f2c330e4f14db9cd8046e0616a1a5a60`
- Restarted prediction CSV SHA-256: `17a8a4bdd073478e46c68564b6c3776e355b4d242508ee94e6933ad7360ea345`
- Normal model metadata SHA-256: `b46ace5528f4c38357ac2ce59937452586fb4183e21164a3c55f2b49b8c486c4`
- Restarted model metadata SHA-256: `125c32a6805b83ff05fd4191e1636bc3547a8a70718ab5f03b2323ad192525a7`

The restarted fit reached a lower objective than the ordinary successful fit,
but followed a different coefficient solution and produced a different PMF
map. `maxls=100` therefore resolves the line-search failure in this diagnostic
path, but it does not reproduce the same outcome distribution. No production
retry or optimizer change was made.
