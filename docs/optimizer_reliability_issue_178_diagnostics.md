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
`issue-178-runner-5`. The final hosted results will be added here after the
workflow completes on this revision.
