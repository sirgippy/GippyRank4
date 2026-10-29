# Quality of Results production snapshots

Quality of Results V1 evaluates opponent-adjusted season accomplishments with
full win/loss evidence and tempered conditional scoring-margin evidence:

`L(m | θ) = P(sign(m) | θ) × p(m | sign(m), θ)^0.40`.

The production choice is **λ = 0.40**. This is a normative decision informed by
issues #242–#269, not an estimated optimum. Winning receives special significance
while margins remain meaningful evidence. Middle-strength schedule ambiguity,
weak-team uncertainty behavior, and sensitivity to extreme margins remain known
limitations. This implementation does not change the likelihood, priors, schedule
adjustments, or ranking statistic.

## Supported interface

`gippyrank.quality_of_results` centrally defines the fixed margin weight,
`quality_of_results_v1` model, `1.0` model version, `quality_of_results` variant,
and `outcome_and_tempered_margin` evidence mode. Its inference function accepts
no weight override and calls the validated implementation in
`gippyrank.research.quality_of_results`. That implementation and every frozen
research source/artifact remain unchanged. Production reads no research results.

The snapshot interface is:

```python
from pathlib import Path
from gippyrank.quality_of_results_snapshot import (
    build_quality_of_results_snapshot,
    validate_quality_of_results_against_context,
)

root = Path("/path/to/GippyRank4").resolve()
context = root / "data/processed/snapshots/2026/<context-id>/predictive/context"
snapshot = build_quality_of_results_snapshot(context, root=root)
validate_quality_of_results_against_context(context, snapshot.directory, root=root)
```

Both functions accept a `Snapshot` or directory for their snapshot arguments.
The validator requires the frozen repository root and performs no inference.
The builder accepts `output_root` and a timezone-aware `generation_timestamp`;
the default timestamp is the actual UTC build time. It refuses to overwrite an
existing output directory. Use validation to consume a retained output or choose
a separate output root for a repeat build.

## Required frozen inputs

Provide a valid, converged Predictive Context snapshot with its metadata,
`included_games.csv`, `posterior_pmfs.csv`, `rankings.csv`, and `diagnostics.json`.
The corresponding versioned Context prior CSV must remain present under the
repository root. Historical Likelihood V1 must match the source hash; sources
with explicitly supplied parameters must retain their authenticated parameters.
The exact source BP configuration is reused. No games, priors, or external data
are fetched or refreshed.

Validation covers source identity and prior artifact hashes, full network
population and PMFs, included-game identity/hash and cutoff, FCS fallback handling,
likelihood provenance, supported inference settings, and source convergence.
Modern Context snapshots must retain all four semantic identity fields for game
rows, game IDs, prior PMFs and inference configuration. For legacy snapshots that
predate all four fields, these are derived from the retained artifacts and BP
diagnostics in memory, following the existing Quality of Record convention.
This compatibility path requires a declared `legacy_first_response` or
`legacy_unverified_availability` source contract. Legacy sources lack the newer
original row-level checksum; this derivation cannot
authenticate historical score changes against a checksum they never recorded.
Partially missing modern identities fail closed. Source artifacts are never
rewritten. Previously supported Quality of Play/Record artifacts are unaffected.

## Standalone commands

Use `uv` with the repository's `.python-version` pin. For example, from the root:

```bash
uv run python scripts/build_quality_of_results_snapshot.py \
  --context data/processed/snapshots/2026/2026-weekly-2026-10-25T12-59-22.246522Z-context-v1.4/predictive/context \
  --output-root /tmp/quality-of-results
```

The output directory is:

`<output-root>/<season>/<logical-context-id>/performance/quality_of_results/`

The default output root is `data/processed/snapshots`. The snapshot ID appends
`-performance-quality-of-results` to the same logical Context base used by the
other Performance variants. Source paths and an explicit `--output-root` are
resolved relative to `--root` (the repository by default); absolute paths work.

Independently validate a saved result:

```bash
uv run python scripts/build_quality_of_results_snapshot.py \
  --context data/processed/snapshots/2026/2026-weekly-2026-10-25T12-59-22.246522Z-context-v1.4/predictive/context \
  --validate /tmp/quality-of-results/2026/2026-weekly-2026-10-25T12-59-22.246522Z-context-v1.4/performance/quality_of_results
```

## Artifact contract

| Artifact | Contents |
| --- | --- |
| `metadata.json` | Model/schema IDs, fixed weight, evidence mode, cutoffs, generation time, frozen source identities/hashes, prior/likelihood provenance, BP configuration, population counts and artifact hashes |
| `included_games.csv` | Unmodified copy of the source game evidence |
| `posterior_pmfs.csv` | Complete prior-stripped FBS Performance distributions |
| `blended_context_posterior_pmfs.csv` | Complete FBS/FCS network posterior before stripping |
| `rankings.csv`, `rankings.json` | Expected-rank ordering and established Performance summaries, identities, records when present, rated status and evidence counts |
| `diagnostics.json` | Convergence, iteration count, terminal message delta, raw/grouped factor counts, every team's eligible-game count and runtime |

Rated teams sort by expected rank, then team name and canonical ID, following
existing production Performance conventions. Teams without eligible evidence
remain unrated with uniform PMFs. The source opponent network, venues, rematch
grouping and FBS/FCS supports are preserved. The output validator checks both
PMF sets, recomputes prior stripping and rankings, verifies the CSV/JSON
representations and evidence counts, and enforces convergence within the declared
configuration. Artifacts are written to a temporary sibling and moved to the
final directory only after validation succeeds.

## Reproduce research-to-production parity

After generating the Week 9 output above:

```bash
uv run python scripts/check_quality_of_results_parity.py \
  --snapshot /tmp/quality-of-results/2026/2026-weekly-2026-10-25T12-59-22.246522Z-context-v1.4/performance/quality_of_results \
  --reference data/research/issue_269_lambda_040/networks/baseline-2026-10-25T12-59-22.246522Z.json.gz
```

Optional `--report /path/to/report.json` retains the comparison. The check
authenticates the research checkpoint and its frozen source files, validates the
production snapshot independently, and compares all FBS PMFs, expected ranks,
ordering, uncertainty summaries, full Context PMFs and eligible counts. It reuses
the frozen research parity gates: `1e-8` per PMF cell and `rank_count × 1e-8` for
expected ranks, with explicit reporting of numerical ties. No reference inference
is rerun and no tolerance is fitted to the observed result.

The retained [Week 9 parity report](../data/validation/quality_of_results_v1_parity.json)
records **zero differences** across 138 FBS teams, the full 266-team network, all
uncertainty summaries and 975 included games. Production converged in 57
iterations with terminal delta `8.412847501970688e-10`. The new focused tests
exercise small-network research parity, rematches/FCS evidence, unrated handling,
source/output corruption, nonconvergence and preservation of existing variants.
The full-season parity command is explicit and does not add the research inventory
to ordinary CI.

Weekly publication generates this variant alongside Quality of Play and Quality of Record.
The site and static API expose it where published; older slots do not require backfills.
