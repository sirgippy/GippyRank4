# Issue 265: frozen baseline execution and population analysis

The findings are in
[`report.md`](../data/research/issue_265_real_seasons/report.md). The experiment
executes only the baseline, independent historical endpoint reference, and
input-order validation portions of the [issue 263 protocol](issue_263_real_season_protocol.md).
It does not execute case studies, removals, or four-state score perturbations,
select a weight, or change production artifacts.

## Reproduce

Use the repository's `.python-version` through `uv` (currently 3.13). The input
verification command is read-only:

```bash
uv run python scripts/design_issue_263_real_seasons.py
uv run python scripts/run_issue_265_real_seasons.py --verify-only
```

On a restricted worker with a read-only default uv cache, prefix commands with
`UV_CACHE_DIR=/tmp/gippyrank-uv-cache`. No API key, acquisition, external checkout,
new publication discovery, or production snapshot builder is required.

The completed outputs live under `data/research/issue_265_real_seasons` and are
immutable historical evidence. The default command also verifies this saved run
without inference or writes. Use `--output-dir /absolute/path` to identify another
run; an empty directory still requests new inference. Newly generated checkpoints
require the exact scoped source/dependency/runtime fingerprint. Every cell
fingerprint covers the frozen input inventory, the explicit `BASELINE_SOURCES`
list, complete snapshot specification, weight, job kind, configuration and input order.
The list covers the runner, frozen loaders and validators, factors/BP, prior
stripping, shared likelihood/hash primitives, dependency lock and Python pin.
Adding or modifying unrelated research modules does not change it. A mismatch
is rejected before an existing run's provenance can be overwritten.
Analysis has a separate source hash, so refining a descriptive table or figure
does not require repeating inference. Runtime timings and measured peak memory
vary between executions; scientific summaries and gzip outputs are deterministic
in one compatible environment.

An existing checkpoint must match its content hash, complete fingerprint,
convergence diagnostics, network population, full rank supports and evidence
counts. A mismatched checkpoint fails the snapshot rather than silently being
reused or overwritten. Successful earlier cells remain intact if a later cell
fails. The saved execution state cannot claim completion without all 32 cells
and passing gates in all five snapshots. Every analysis replay independently
validates all saved cells and recomputes the parity gates.

The original 32 cells remain identified by execution revision
`aba5f265a7bbca3bb3f6014d7c5500d1e9fdcd5a` and fingerprint
`dc7ab5703b9a7d464c7782b4d0ab75d8167f9cedd7a03d533e6d21f50addc0fa`.
Neither their fingerprints nor their provenance have been migrated or rewritten.
The verifier authenticates the exact original manifest, all its output hashes,
the self-consistent original provenance, frozen inputs/configuration, relevant
dependency hashes, every checkpoint's identity/PMFs/convergence, and all endpoint
and input-order gates. Because this repair changes the runner's own file hash,
one explicit compatibility check pins its original inference/loader definitions
and project imports using their AST hash. Provenance/CLI/verification additions
do not change that inference path. Other relevant files must still match the
original source hashes. This exception accepts only this specific historical
run, never newly generated checkpoints.

Historical runtime versions remain audit data. Read-only verification works on
another platform without implying that the original solves ran there. Numerical
gates are recomputed against the frozen tolerances; insignificant differences in
floating-point error summaries across platforms need not be bit-identical.

The diagnostic study can consume the verified original cells directly:

```python
from pathlib import Path
from gippyrank.research.issue_265_real_seasons import verify_completed_execution

root = Path.cwd()
run = verify_completed_execution(root, root / "data/research/issue_265_real_seasons")
baseline = run.cells["2025-final"]["baseline", 0.25]
pmfs = baseline["pmfs"]  # Full FBS PMFs, with original execution identity.
```

For newly generated compatible runs, analysis can be regenerated without solving:

```bash
uv run python scripts/run_issue_265_real_seasons.py --analyze-only --output-dir /path/to/new-run
```

In-place analysis regeneration of the historical run is rejected to preserve its
anchored manifest and scientific outputs. Follow-up diagnostic outputs belong in
their own directory and use the read-only handoff above.

The serial runner polls Linux process RSS and wall time every 0.1 seconds with
`setitimer`; a solve stops at 600 seconds or 4 GiB RSS, and the invocation has a
three-hour ceiling including analysis. Signal delivery occurs when the current
native numerical call returns to Python. The saved peak RSS is the process
high-water mark, which includes earlier cells; it is not an independently
measured allocation attributable to a single solve. Factor surfaces are shared
within a snapshot, BP always cold starts, and the engine cache is cleared between
cells to bound memory. These implementation choices do not change inference.

## Artifact contract

All differences in candidate comparisons are **higher λ minus lower λ**.
Positive expected/ordinal rank changes mean worse ranks. Longitudinal differences
are later minus earlier. No cross-season rank contrasts are produced.

| Artifact | Contents |
| --- | --- |
| `provenance.json` | Execution revision, complete source/input hashes, dependency lock, environment and solve fingerprint |
| `checkpoints/<snapshot>/<kind>-<weight>.json.gz` | Full-network Context marginals, FBS prior-stripped marginals, evidence counts and convergence diagnostics; sidecar content hash |
| `execution.json`, `convergence.csv` | Coverage, failures, per-cell runtime, iterations, maximum message change, PMF validation and process peak RSS |
| `endpoint_parity.json` | Ten endpoint gates, three order gates, all numerical ties and non-tied inversions |
| `posterior_pmfs/candidates.csv.gz`, `rankings.csv` | Complete FBS PMFs and all required rank, interval, uncertainty, W–L and evidence summaries |
| `population_contrasts.csv`, `team_contrasts.csv.gz` | All 50 candidate-pair summaries and all team-level changes, TV and Wasserstein-1 distances |
| `boundary_movement.csv` | Union of ±3 reference-order bands around 10/12/25, crossing flags, adjacent mean gaps and latent rank-mass changes |
| `accomplishments.csv`, `actual_game_descriptors.csv.gz` | Frozen primary schedule/record strata and full probability profiles; actual margins, opponent PMFs, expected margins, close/dominant results, FBS/FCS mix; separate retrospective posterior descriptions |
| `accomplishment_panels.csv`, `uncertainty_panels.csv` | Every weight/pair across frozen strength/schedule/record/near-expectation groups, including empty cells and explicit membership |
| `matched_records.csv.gz` | All exact-W–L within-band pairs, with harder/easier orientation fixed by the primary frozen schedule descriptor |
| `descriptive_associations.csv` | Spearman descriptions, denominators and undefined constant-column cases; no significance or optimization scores |
| `weight_sensitivity.csv` | Adjacent changes and slopes, adjacent TV, nonmonotonic mean flags above numerical tolerance |
| `longitudinal_summary.csv`, `longitudinal_team_changes.csv` | All 15 candidate/transition summaries and rated-intersection team changes; entrants/exits, new game IDs, phase/subdivision counts |
| `candidate_progression.png`, `schedule_associations.png` | Separate-season research figures generated by the analyzer |
| `manifest.json` | All output hashes, execution provenance, analysis source hash and coverage |
| `report.md` | Population findings, limitations and questions for the frozen diagnostic follow-up |

Central intervals are equal-tail integer CDF quantiles; widths are inclusive.
Entropy is in bits. The mode uses the smallest rank attaining the maximum.
Expected-rank display sorting uses ascending canonical string ID for exact ties.
Zero-game teams retain uniform Performance distributions and no ordinal display
rank; contrasts exclude them and longitudinal rows use the rated intersection.
All selected populations in this execution happen to be rated.

Full team contrast rows plus explicit panel member IDs retain the distributions
under each panel, not only its mean. Top-set entries/exits and union movement are
JSON-valued CSV cells. The boundary mass fields describe latent rank coordinates,
not probabilities of occupying top display slots. Ordinal inversions use the
frozen display convention; inspect the explicit endpoint numerical ties and
adjacent expected-rank gaps before assigning substantive meaning to an inversion.

## Frozen diagnostic handoff

Reuse the saved baseline PMFs, counts, converged full-network Context marginals,
source fingerprints and loader functions. The independent 2025 margin reference
also supplies the separate posterior-based schedule descriptions. Keep 2025
preseason Context 1.2 primary strata and 2026 QoP primary strata fixed.
`cases.json`, `historical_cases.json` and `game_diagnostics` in issue 263 remain
the sole diagnostic inventories. This ticket does not select replacement cases.

The next ticket must implement the already frozen full-network score replacement
and single-observation removal jobs as separate checkpoint kinds with their
complete perturbation fingerprints. Do not reuse a baseline merely by matching
a snapshot or weight label, reuse warm BP messages, or treat the full-network
Context marginals as replacement priors. The authoritative priors and evidence
continue to come from the frozen loaders. Retain all original grouping/rematch
behavior and make new evidence counts for each perturbation.
