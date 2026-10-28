# Issue 267: frozen Phase 4 diagnostics and case studies

The [research report](../data/research/issue_267_real_season_diagnostics/report.md)
synthesizes the completed population experiment, all frozen cases, 40 score
replacements and 25 single-observation removals. No production weight is selected.

## Execute and reproduce

Use the repository's `.python-version` with `uv`. The retained experiment used
Python 3.13.15, following that pin. No acquisition or API key is needed.

```bash
uv run python scripts/design_issue_263_real_seasons.py
uv run python scripts/run_issue_265_real_seasons.py --verify-only
uv run python scripts/run_issue_267_diagnostics.py --verify-only
```

These commands validate the frozen design, immutable completed baseline, all 65
diagnostic checkpoints and the diagnostic output hashes. The baseline is consumed
through `verify_completed_execution()`; none of its 32 solves is rerun, and its
original revision, fingerprints, PMFs, rankings, analysis and manifest are retained.

```bash
uv run python scripts/run_issue_267_diagnostics.py --analyze-only
uv run python scripts/run_issue_267_diagnostics.py --output-dir /tmp/issue-267-replay
```

The first command regenerates only this issue's descriptions/report from verified
checkpoints. The second performs a fresh 65-solve execution in a separate directory.
Default invocation resumes matching verified checkpoints; a completed execution
requires no new inference. Output paths overlapping the frozen design, baseline,
processed data or production site are rejected. On restricted workers, set
`UV_CACHE_DIR=/tmp/gippyrank-uv-cache`; for plotting, also set
`MPLCONFIGDIR=/tmp/gippyrank-matplotlib-cache` and
`XDG_CACHE_HOME=/tmp/gippyrank-xdg-cache` when the default caches are read-only.

The serial execution uses the original V1 likelihood and snapshot priors, all
five weights, complete FBS/FCS evidence, deterministic cold-start BP and existing
prior stripping. The configuration remains max iterations 500, tolerance 1e-9,
damping .35. Factor surfaces may be shared when their complete cache keys match;
BP messages/posteriors are never reused as starting values. The engine factor
cache is cleared between solves. Removing one observation retains any other
meeting and lets the engine rebuild grouped factors.

The resource guard polls every .1 seconds: three-hour wall budget, 600 seconds
per solve and 4 GiB process RSS. As with the original runner, signals interrupt
native numerical calls when control returns to Python. Saved execution elapsed
time counts against resumed execution budgets. Failed attempts are retained in
numbered files, completed checkpoints survive a failure, and a failure stops the
remaining affected-network experiments without changing settings. Analysis has
the same invocation wall/RSS guard. Timings/high-water RSS are measurements and
can vary between fresh runs; scientific CSV/gzip/report replay is deterministic
within the same compatible numerical/plotting environment.

## Checkpoint identity

Each checkpoint includes the frozen snapshot specification, complete original
dated observation, perturbation type, exact replacement scores or removal,
candidate weight, source/runtime/dependency fingerprint, baseline verification
identity, full configuration, ordered team/support and game populations, exact
game-object and input-row hashes, and prior PMF hash. Score replacement validates
the frozen opponent-point floor and original orientation; only the two score
fields change. A matching game ID/weight alone is insufficient.

Resume checks the content hash, full identity, execution revision, convergence,
PMF supports/populations/normalization, eligible-game counts and raw/grouped factor
counts. Source or runtime drift fails closed; the runner does not overwrite a
mismatched checkpoint. The inference source fingerprint excludes the separate
analysis/report modules, so descriptive repairs do not require repeating solves.
The manifest separately records analysis source hashes and every output hash.

Strict runner resume/verification also requires the original runtime fingerprint.
Read-only consumption on a different CI/OS runtime uses
`verify_completed_diagnostics(root, output, verified_baseline)` from
`gippyrank.research.issue_267_analysis`. It authenticates the self-consistent
original execution fingerprint, all current scientific source/input/configuration
hashes, every checkpoint, execution coverage and numerical gates; the original
runtime remains audit evidence. It never relabels provenance or infers on the
reader's runtime. `--analyze-only` uses this portable reader, and independently
guards analysis resources after subtracting the saved inference/checkpoint wall
time from the three-hour budget. The regression suite validates this handoff on
CI without rerunning the 65 inference solves.

## Artifact and interpretation contract

| Artifact | Contents |
|---|---|
| `checkpoints/*.json.gz`, hash sidecars | All 65 full-network Context and FBS prior-stripped PMF sets, exact evidence changes, counts, numerical diagnostics and provenance; gzip mtime=0 |
| `execution.json`, `convergence.csv`, `endpoint_parity.json` | Coverage, failures, timings/RSS, convergence and the four full-PMF λ=0 winning-margin equality gates |
| `provenance.json`, `manifest.json` | Immutable baseline verification identity, source/input hashes, implementation/runtime fingerprint and output hashes |
| `network_changes.csv.gz` | All 8,970 FBS team/probe rows, before/after uncertainty/intervals, ordinal and expected ranks, counts, TV/W1, top10/12/25 status and adjacent mean gaps |
| `game_diagnostics.csv` | 130 focal/opponent rows for the same probes; explicit posterior family and subdivision |
| `diagnostic_summary.csv` | All 65 exact evidence definitions, network spread, numerical diagnostics and top-set entries/exits/union movements |
| `outcome_sensitivity.csv`, figure | Ten focal game/weight comparisons of loss-by-3 to win-by-3 versus win-by-10 to win-by-35, separately retained with their difference and all four state summaries |
| `case_results.csv`, `case_interpretations.json` | All 350 case/candidate rows; original frozen selection/categories/questions, full actual games, uncertainty, endpoint differences, boundaries and answers after observing results |
| `case_games.csv`, `case_evolution.csv` | Complete selected actual-game evidence and all 215 case/candidate snapshot transitions, including own added observations and network additions |
| `case_studies.md`, `report.md` | Every frozen case with all five candidates, original questions and evolution; final Phase 4 synthesis |

Rank changes are perturbed minus original, or later minus earlier; positive is
worse. Counterfactual benefits are oriented in the opposite direction so positive
means improvement. Expected ranks and interval masses describe latent marginal
coordinates; display ordinal positions sort means with canonical string-ID ties.
Intervals use integer equal-tail quantiles with inclusive widths, entropy is in
bits, TV is half the L1 difference, W1 is summed absolute CDF difference.

FBS responses use existing prior-stripped Performance. Oregon's frozen dominant
win is over **Portland State (FCS)**. Its opponent response uses its full Context
posterior on FCS rank support; no FCS Performance distribution or FBS display rank
is fabricated. Its ordinal/FBS boundary fields are explicitly inapplicable. All
FBS effects remain in the population table, including all 137 other FBS teams in
this probe; FBS–FBS probes have 136 nonparticipant FBS teams.

Selection facts, actual observations, candidate-dependent measurements and
interpretations after observation remain separate. The 2025 primary expectations
come from preseason Context priors; the 2026 expectations reuse cutoff Context
posteriors and are retrospective. Removals measure total network responses and
must not be summed as additive contributions. No causal scheduling claim or
methodology selection follows from a matched-record pair or a boundary crossing.
