# Issue 269: λ=0.40 candidate validation

This research extension solves the existing blended likelihood at λ=0.40. The
[assessment](../data/research/issue_269_lambda_040/report.md) compares it with
authenticated λ=0.25 and λ=0.50 results. No production methodology is selected,
published or changed.

## Evidence and scope

The runner preserves the original inputs, priors, opponent networks, FCS treatment,
Historical Likelihood V1, Performance prior stripping, cold starts, maximum 500
iterations, 1e-9 message tolerance and .35 damping. It adds no schedule adjustment.
The sole extension to the issue 252 helpers is an optional candidate-weight tuple;
the original five-weight defaults remain intact. Terminal message deltas are now
available in new synthetic summaries.

The issue 261 manifest supplies exactly 500 unconditional, 200 primary-conditioned,
and 200 independently control-conditioned seeds for each of elite, strong, middle
and weak profiles. There is no new seed search. Regenerated trial evidence hashes
and win counts must equal the saved observations before inference. Primary-seed
controls remain distinct from controls selected on their own records. The retained
full focal distributions are packed into four deterministic gzip archives; resumable
per-trial cache files are disposable and excluded from Git.

Five issue 265 snapshot networks receive new baseline solves. The exact issue 267
inventory supplies eight score replacements and five removals in the Week 9
network. The score-replacement rule, focal orientation and every other observation
are preserved by the existing perturbation implementation. The two issue 252
approximately-even/weaker-opponent probes use their saved representative seed and
game selections, including the original continuous-margin construction. Their
regenerated observations must match every saved field exactly. Eight new states
cover loss by 3, wins by 3, 10 and 35.

The original design and result directories are read-only inputs. Their manifests,
output hashes and existing baseline/diagnostic handoffs are verified. New solve
fingerprints cover the scoped execution sources, locked dependencies, Python pin,
original input and manifest hashes, configuration, runtime, candidate, and exact
scenario identity. The execution revision is recorded before inference. Analysis
has a separate source hash, allowing analysis repair without recomputing posteriors.

## Reproduce and verify

Use `uv` and the repository's `.python-version`, currently 3.13. In a restricted
worker, use a writable cache:

```bash
export UV_CACHE_DIR=/tmp/gippyrank-uv-cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
uv sync --frozen
uv run python -m gippyrank.research.issue_269_analysis --verify-only
```

Verification authenticates originals, all 3,600 new synthetic trials, 18 new
network checkpoints, eight issue 252 states, coverage, analysis source and output
hashes. It performs no inference or writes. Runtime differences are allowed only
for portable reading; the recorded original environment is preserved. Resume for
new solves requires an exact execution fingerprint, including the runtime.

To resume or reproduce compatible checkpoints in place:

```bash
uv run python scripts/run_issue_269_candidate.py
uv run python -m gippyrank.research.issue_269_analysis --repeat-check
uv run python -m gippyrank.research.issue_269_analysis
```

For a fresh run, provide `--output-dir /absolute/path/to/new-run` to the execution
script. Its stage-specific commands support `--stage networks`, `--stage issue252`,
or `--stage synthetic --profile elite` (also strong, middle, weak). Initialize
provenance with one invocation before launching independent stages concurrently.
Separate profiles can run concurrently; each process limits BLAS/OMP to one thread.
Do not run two writers for the same stage/profile. The analysis module targets the
committed extension directory; a fresh execution directory can instead be verified
with the execution script's `--verify-only --output-dir ...` option. That option
checks the solve inventory; the analysis verifier additionally authenticates
derived output hashes.

The separate `--repeat-check` adds one Week 9 solve with reversed input order and
one repeated elite trial. Their new-weight numerical gates, full checkpoint data
and resource measurements are retained under `validation/` and in
`repeat_validation.json`. These are validation solves in addition to the 18
required network cells and 3,600 required synthetic trials. No historical
neighboring candidate is recomputed.

Network and schedule-trial solves retain the existing 600-second/4-GiB resource
guard. The eight small issue 252 states retain their original numerical settings
and record solve times. Each profile or
network-stage invocation has a three-hour budget. Completed checkpoints survive an
interruption and are validated before reuse. Invalid files fail closed rather than
being overwritten. Scientific CSV/gzip outputs are deterministic; wall times and
process high-water RSS vary. Timings include contention when stages run together;
RSS is a process maximum, not a per-game memory allocation.

## Output contract

All outputs are under `data/research/issue_269_lambda_040/`:

| Artifact | Meaning |
| --- | --- |
| `provenance.json`, `manifest.json` | Original identities, source/input hashes, runtime, complete output inventory |
| `synthetic/{profile}.json.gz` and sidecars | Every new trial identity, full focal PMFs, convergence, paired/team summaries and solve time |
| `synthetic/{profile}/execution.json` | Profile coverage, timing, reuse and peak RSS |
| `synthetic_pairs.csv.gz`, `synthetic_teams.csv.gz` | Matched .25/.40/.50 trial distributions and uncertainty summaries |
| `schedule_summary.csv`, `schedule_candidate_contrasts.csv`, `schedule_arms.csv` | Separate populations, pointwise mean/Wilson intervals, matched candidate differences, individual spread and uncertainty |
| `networks/*.json.gz` and sidecars | Full Context and prior-stripped FBS PMFs for all five baselines and 13 diagnostics |
| `networks/execution.json` | All 18 numerical/resource diagnostics |
| `issue252/*.json.gz` and sidecars | Eight frozen synthetic counterfactual states |
| `rankings.csv`, `team_contrasts.csv.gz`, `population_comparisons.json` | All FBS ranks, uncertainty, Top 10/12/25 differences, neighbor-selected contender unions and largest changes |
| `contenders.md` | Ordinals, means, latent Top 12 masses and SDs for every neighboring Top 18 team |
| `nonmonotonic.csv` | Candidate .40 values outside the .25/.50 endpoint range, by metric |
| `cases.csv.gz`, `contender_games.csv.gz` | All 70 frozen snapshot cases at three weights and actual games for the neighbors' Top 25 union |
| `network_responses.csv.gz`, `game_diagnostics.csv` | Every FBS response and focal/removal summaries |
| `counterfactual_states.csv`, `outcome_sensitivity.csv` | Four-state uncertainty and separate outcome/margin benefits, including their difference |
| `report.md`, `schedule_comparison.png` | Decision assessment and pointwise synthetic comparison figure |

Schedule differences are easy minus difficult (control B minus A); positive values
favor the difficult slate. Candidate differences are higher weight minus lower
weight. Diagnostic responses are altered evidence minus actual baseline. Positive
counterfactual benefits mean improvement. Inclusive equal-tail intervals, entropy
in bits and canonical-ID tie breaking follow the earlier studies. Latent top-set
mass is neither joint displayed-order probability nor CFP selection probability.

The expensive experiment is an explicit research command and is never added to
ordinary CI. Focused tests exercise candidate selection and checkpoint rejection;
normal CI retains the project's full test suite and Ruff checks.
