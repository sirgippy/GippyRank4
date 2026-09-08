# Test-suite inventory

This inventory records the Issue #24 audit of the pytest suite. The distinction
is about edit-loop cost, not about whether research tests are valuable: every
research test remains runnable through the complete-suite command.

## Commands

Default production/shared suite:

```bash
uv run pytest -q
```

Complete suite, including frozen research investigations:

```bash
uv run pytest -q -o addopts=""
```

The `research` marker is declared in `pyproject.toml`; the configured default
exclusion is `-m 'not research'`. `-o addopts=""` intentionally clears that
default for reproducibility runs.

## Baseline

Baseline was collected from the 2026-09-08 `origin/main` snapshot at
`71db6ea` before the suite was changed.

- 270 tests collected
- 267 passed and 3 failed
- 87.38 seconds wall time for the complete suite
- The three failures were test-fixture failures, not production failures: two
  tests referenced ignored/generated files absent from a clean checkout, and
  one referenced an obsolete generated path.

The slowest baseline test cases were:

| Test | Runtime |
| --- | ---: |
| `test_context_ablation.py::test_stored_ablation_metrics_recompute_from_paired_team_losses` | 9.28 s |
| `test_performance_v1.py::test_loopy_diagnostic_separates_tree_feedback_from_cycle_feedback` | 8.66 s |
| `test_site_data.py::test_site_data_is_byte_deterministic` | 6.33 s |
| `test_site_data.py::test_exported_80_percent_interval_remains_the_ranking_interval` | 3.40 s |
| `test_site_data.py::test_logical_publication_slots_pair_context_and_history[2026-09-06-context-history]` | 3.40 s |

The full baseline module inventory is below. Counts are declared test
functions; parametrized cases are included in the collected totals above and
in the before/after timing commands reported with the change.

## Classification

| Test module | Declared tests | Classification | Invariant or rationale |
| --- | ---: | --- | --- |
| `test_box_score_audit.py` | 5 | `KEEP_RESEARCH` | Offline primitive box-score audit parsing, coverage, duplicate handling, and reproducibility. |
| `test_cfbd_corpus.py` | 6 | `KEEP_DEFAULT` | CFBD raw ingestion, provenance sidecars, deterministic discovery, and protection against fake rows. |
| `test_context_ablation.py` | 14 | `KEEP_RESEARCH` | Frozen context-ablation populations, paired losses, interaction claims, and generated research artifacts. |
| `test_context_prior.py` | 16 | Mixed | Production H/C prior construction, leakage guards, cold starts, and artifact contracts stay default; three exact frozen report/backtest assertions are `KEEP_RESEARCH`. |
| `test_current_season_cfbd.py` | 3 | `KEEP_DEFAULT` | Current-season FBS/FCS acquisition, deduplication, conflict handling, and processed corpus behavior. |
| `test_fcs_fallback.py` | 7 | `KEEP_DEFAULT` | Full-universe FCS fallback semantics, cutoff independence, and fail-closed population handling. |
| `test_historical_likelihood_artifact.py` | 1 | `KEEP_DEFAULT` | Frozen Historical Likelihood V1 production artifact schema and provenance. |
| `test_lean_context_validation.py` | 9 | `KEEP_RESEARCH` | Frozen Lean Context candidate selection and validation investigation. |
| `test_margin_likelihood.py` | 12 | `KEEP_RESEARCH` | Issue #26 margin-likelihood candidate investigation and mathematical diagnostics. |
| `test_massey.py` | 2 | `KEEP_DEFAULT` | Source normalization, alias matching, and long-form export semantics. |
| `test_modeling.py` | 12 | `KEEP_DEFAULT` | Shared historical modeling primitives, rank-coordinate orientation, coverage, and marginalized scoring invariants. |
| `test_performance_snapshot.py` | 6 | `KEEP_DEFAULT` | Production performance snapshot construction, validation, and fail-closed evidence contracts. |
| `test_performance_v1.py` | 16 | `KEEP_DEFAULT` | Production Performance V1 inference, neutralization, diagnostics, and deterministic artifact inventory. |
| `test_posterior_engine.py` | 7 | `KEEP_DEFAULT` | Core factor graph/message-passing correctness and cross-subdivision evidence semantics. |
| `test_posterior_snapshots.py` | 8 | `KEEP_DEFAULT` | Snapshot schema, cutoff/provenance rules, FCS support, determinism, and publication-facing metadata. |
| `test_posterior_validation_artifact.py` | 1 | `KEEP_DEFAULT` | Exact belief-propagation validation acceptance contract for the posterior engine. |
| `test_preseason.py` | 23 | `KEEP_DEFAULT` | Production prior distributions, leakage-safe fitting, cold starts, and coverage invariants. |
| `test_primitive_box_score_likelihood.py` | 23 | `KEEP_RESEARCH` | Frozen primitive box-score likelihood investigation and explicit V1 fallback comparisons. |
| `test_regime_stability.py` | 7 | `KEEP_RESEARCH` | Frozen regime-stability and nested-selection investigation. |
| `test_site_data.py` | 33 | `KEEP_DEFAULT` | Static site export schema, publication pairing, deterministic output, and fail-closed validation. |
| `test_weekly_update.py` | 10 | `KEEP_DEFAULT` | Reviewed weekly update/publication workflow, idempotence, staging, and workflow safety. |
| `test_ypp_likelihood.py` | 26 | `KEEP_RESEARCH` | Frozen conditional YPP likelihood investigation and explicit production-engine isolation checks. |

There are no `DUPLICATE_REMOVE` or `OBSOLETE_REMOVE` tests in this audit. The
three baseline fixture failures were repaired in place: the FCS population
test now supplies a minimal durable-source fixture, the Massey parser test
uses a minimal export fixture, and the context-prior test follows the current
annual fitted-instance artifact path. No production or published artifact was
changed.

## Timing report

The measured before/after results are:

| Suite | Baseline on `main` | After change |
| --- | --- | --- |
| Default command (`uv run pytest -q`) | 270 collected; 267 passed, 3 failed; 87.38 s | 161 passed, 109 deselected; 66.52 s |
| Complete command (`uv run pytest -q -o addopts=""`) | 270 collected; 267 passed, 3 failed; 87.38 s | 270 passed; 71.40 s |

The baseline command necessarily ran the complete population because the
default/research distinction did not yet exist. Wall time includes normal
`uv`/filesystem startup and is therefore an operational comparison rather
than a benchmark; repeated runs varied materially with filesystem and CPU
cache state.

The slowest remaining default cases were:

| Test | Runtime |
| --- | ---: |
| `test_performance_v1.py::test_loopy_diagnostic_separates_tree_feedback_from_cycle_feedback` | 8.58 s |
| `test_site_data.py::test_site_data_publishes_all_initial_h_c_preseason_and_current_snapshots` | 4.74 s |
| `test_site_data.py::test_logical_publication_slots_pair_context_and_history[2026-preseason-context-history]` | 3.89 s |
| `test_site_data.py::test_all_ranking_families_receive_consistent_season_conferences` | 3.27 s |
| `test_site_data.py::test_logical_publication_slots_pair_context_and_history[2026-09-06-history-context]` | 3.23 s |

The final validation used these exact commands:

```bash
time uv run pytest -q
time uv run pytest -q -o addopts=""
```

The default suite excludes only tests marked `research`; the complete suite
still runs all collected tests and is the reproducibility path for the frozen
investigations.

The measured complete-suite call-time breakdown after the change was:

| Module | Call time |
| --- | ---: |
| `test_box_score_audit.py` | 0.02 s |
| `test_cfbd_corpus.py` | 0.00 s |
| `test_context_ablation.py` | 5.10 s |
| `test_context_prior.py` | 0.55 s |
| `test_current_season_cfbd.py` | 0.00 s |
| `test_fcs_fallback.py` | 0.01 s |
| `test_historical_likelihood_artifact.py` | 0.00 s |
| `test_lean_context_validation.py` | 0.01 s |
| `test_margin_likelihood.py` | 0.00 s |
| `test_massey.py` | 0.00 s |
| `test_modeling.py` | 0.02 s |
| `test_performance_snapshot.py` | 0.73 s |
| `test_performance_v1.py` | 4.62 s |
| `test_posterior_engine.py` | 0.02 s |
| `test_posterior_snapshots.py` | 0.06 s |
| `test_posterior_validation_artifact.py` | 0.00 s |
| `test_preseason.py` | 0.30 s |
| `test_primitive_box_score_likelihood.py` | 0.05 s |
| `test_regime_stability.py` | 0.01 s |
| `test_site_data.py` | 25.71 s |
| `test_weekly_update.py` | 0.27 s |
| `test_ypp_likelihood.py` | 0.21 s |
