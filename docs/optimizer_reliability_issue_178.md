# Issue 178: preseason optimizer reliability

## Reproduction

The retained 2026 History fit was reconstructed from 2,744 FBS team-seasons
covering 2004–2025. The target panel contains 136 same-subdivision fitted rows.
The raw rank-distribution input is a local ignored cache; its SHA-256 is recorded
below, along with the tracked feature input and retained 2026 artifacts.

The available CI-like runtime was this WSL/Linux worker: Python 3.13.15, NumPy
2.5.2, SciPy 1.18.1, and SciPy OpenBLAS 0.3.34.0.0. A separate GitHub runner
was not available for this local comparison. Before the retry change, 100 fits
were run: 60 with threading
environment variables unset and 40 with OpenBLAS, OpenMP, and MKL set to one
thread. After the change, another 100 fits were run: 50 in each mode. Every fit
in both batches succeeded; none invoked the retry.

For all 200 fits, the result was status 0 with message
`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`, 41 iterations, 44
function evaluations, and objective `1.685613543083427`. Beta and gamma were
bit-identical in every run. The SHA-256 of the sorted map of 136 team IDs to
12-decimal PMF strings was
`e4b6824e5aaa453a4fd043cd08ce5a402f7dadc67487ca7cf39034860cf74172` in every
run. The per-run records, including beta, gamma, optimizer diagnostics, and PMF
hash, are in [repeated_fit_runs.jsonl](../data/processed/optimizer_reliability_issue_178/repeated_fit_runs.jsonl).

A central finite-difference check at the fitted solution, using a step of
`1e-5 * (1 + abs(parameter))`, found a maximum absolute analytic-gradient error
of `6.11e-11`. This does not reproduce the reported CI failure or prove its
cause. It does not indicate a gradient inconsistency at the fitted solution,
and the default-thread and single-thread runs were identical on this Linux
runtime.

The retained 2026 fitted instance has beta and gamma within `3.53e-11` and
`9.92e-12`, respectively, of this reconstruction. Its stored prediction CSV
has SHA-256 `0b3454a09288019e17739869c42aed3123fdda2163694f52f63bca65baf37f90`.
The 136-row PMF-map hash from that CSV is
`c19a509d48877f576f3f5a7be402b43d4cce92093f763308b2325727575b3193`; several
12-decimal PMF strings differ from the reconstructed fit. The retained fitted
instance does not record a training-row or raw-input fingerprint, so this
comparison cannot distinguish tiny cross-build numerical drift from a change
in the original fit inputs. The retry change does not affect any successful
first-pass fit, and the before/after repeated-run hashes are identical.

## Best-supported diagnosis

The CI message identifies an abnormal L-BFGS-B line-search termination. The
fit uses a tight relative objective tolerance (`ftol=1e-10`); near convergence,
small floating-point differences can make a line search unable to find an
acceptable step even when the last iterate is finite. This is the best-supported
explanation, not a reproduced root cause. SciPy documents L-BFGS-B's `maxls` as
the maximum line-search steps per iteration (default 20), and defines `ftol` as
a relative decrease stopping criterion ([SciPy L-BFGS-B options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)).

## Fix

History enables one retry only when L-BFGS-B returns status 2 with an
`ABNORMAL` message and a finite iterate of the expected shape. The retry starts
from that iterate, keeps the objective, bounds, gradient, and tolerances, and
raises `maxls` to at least 100. If the retry fails, the exception includes the
status, message, iterations, function evaluations, and objective from both
attempts. Other optimizer failures still raise immediately. Because this only
recovers a fit that was previously rejected and leaves successful first-pass
fits unchanged, the recovery policy stays outside the frozen statistical model
identity. It is recorded here and in the run manifest.

The Context 1.4 candidate provenance ledger now reflects the hashes of the two
shared source files changed here. Candidate PMFs and research conclusions were
not regenerated.

Normal successful fits follow the original path and emit the same coefficients
and PMFs. Regression tests cover repeated identical fits, recovery after an
abnormal result, and raising when the retry also fails.

## Run inputs

- Base commit: `191e774c101d989589d9908636dd3f3ffc2f97ea`
- Cached `team_season_rank_distributions.csv` SHA-256:
  `03e4e372017aedb45392ff7d790cea25f461db9b6dc846514c762ff46e2cab00`
- Tracked `team_season_features.csv` SHA-256:
  `fd6f171131e38b852b94619338c5580e5f4e8c19f45c74e8135b11b20b1a2019`
- Retained `fitted_instance.json` SHA-256:
  `583b2583ae30e9056266bc07a4cb034cd9f2a4c36ac4d217a35603c1aa1ece91`
- Recorded runs: 100 before the change and 100 after; 0 failures in both batches.
- [Run manifest](../data/processed/optimizer_reliability_issue_178/run_manifest.json)
