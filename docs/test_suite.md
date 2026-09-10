# Test-suite inventory

This is the Issue #24 audit of the pytest suite. The final policy is simple:

```bash
uv run pytest -q
```

That command runs every retained test. Research-origin tests remain in the
normal suite when they protect reusable code, leakage prevention, scientific
correctness, or a durable production contract. Completed investigations no
longer get a separate opt-in test class merely because they were once useful.

## Classification

### `KEEP_DEFAULT`

All tests remaining in these modules run under `uv run pytest -q`:

| Test module | Retained scope |
| --- | --- |
| `test_box_score_audit.py` | CFBD stat parsing, provenance-sidecar discovery, duplicate/conflict handling, coverage, and deterministic audit output. |
| `test_cfbd_corpus.py` | Raw CFBD ingestion, provenance, deterministic discovery, and protection against fabricated rows. |
| `test_context_ablation.py` | Observed-population construction, leakage-safe rolling training, same-population pairing, standardization, imputation, and interaction semantics. |
| `test_context_prior.py` | H/C prior metadata, feature gating, leakage-safe construction, cold starts, annual inference, and canonical frozen-prior artifact integrity. |
| `test_current_season_cfbd.py` | FBS/FCS acquisition, deduplication, conflict handling, and processed-corpus behavior. |
| `test_fcs_fallback.py` | Full-universe FCS fallback semantics, cutoff independence, and fail-closed population handling. |
| `test_game_evidence.py` | Focal-game evidence factors, prior separation, rematch cavities, and cross-subdivision orientation. |
| `test_historical_likelihood_artifact.py` | Historical Likelihood V1 artifact schema and provenance. |
| `test_lean_context_validation.py` | Leakage-safe observed/training populations, nested selection, calibration decisions, and synthetic validation workflow. |
| `test_margin_likelihood.py` | Reusable margin-coordinate, scale, Jacobian, comparison, and leakage-safe selection mathematics. |
| `test_massey.py` | Source normalization, alias matching, and long-form export semantics. |
| `test_methodology.py` | Public methodology page coverage, generated metadata, shared version identifiers, and site navigation. |
| `test_modeling.py` | Shared historical modeling primitives, rank-coordinate orientation, coverage, and marginalized scoring. |
| `test_offense_defense.py` | Scoreboard-only offense/defense residual orientation, persistence diagnostics, identifiability, cutoff safety, and frozen candidate selection. |
| `test_performance_snapshot.py` | Production performance snapshot construction and fail-closed evidence validation. |
| `test_performance_v1.py` | Performance inference, neutralization, diagnostics, and deterministic artifact inventory. |
| `test_posterior_engine.py` | Factor-graph/message-passing correctness and cross-subdivision evidence semantics. |
| `test_predictive.py` | Exact posterior-predictive margin mixtures, tie handling, and site orientation. |
| `test_posterior_snapshots.py` | Snapshot schema, cutoff/provenance rules, FCS support, determinism, and publication metadata. |
| `test_posterior_validation_artifact.py` | Exact belief-propagation validation acceptance contract. |
| `test_preseason.py` | Production rank distributions, leakage-safe fitting, cold starts, and coverage invariants. |
| `test_primitive_box_score_likelihood.py` | Primitive evidence orientation, rank-neutrality, fallback behavior, target-driven cold starts, strict comparisons, and deterministic fitting. |
| `test_regime_stability.py` | Target-excluded training windows, family-specific nested selection, key matching, and score decomposition. |
| `test_site_data.py` | Static site export schema, publication pairing, deterministic output, and fail-closed validation. |
| `test_season_simulation.py` | Hierarchical latent-state sampling, shared-game dependence, Poisson-binomial forecasts, decomposition, determinism, and fail-closed schedule handling. |
| `test_site_navigation.py` | Canonical site branding, shared navigation, Schedule route compatibility, state-preserving links, and accessibility markup. |
| `test_team_season_artifact.py` | Team-season evidence redaction, provenance validation, and retained historical artifacts. |
| `test_weekly_update.py` | Reviewed weekly publication workflow, idempotence, staging, and workflow safety. |
| `test_ypp_likelihood.py` | YPP orientation, missing-evidence/rank-neutrality fallbacks, supported populations, strict comparisons, and deterministic fitting. |

`KEEP_RESEARCH` has zero retained tests as a separate category. Research-origin
tests that still matter are included in `KEEP_DEFAULT` above.

### `DUPLICATE_REMOVE`

Removed repeated protection for the same frozen 2026 H/C PMF hashes from the
context-ablation, Lean-context, and regime-stability investigations. The
canonical hash check remains in `test_context_prior.py`. Also removed the
primitive/YPP tests that rechecked unchanged production artifacts already
covered by production artifact and snapshot contracts.

### `OBSOLETE_REMOVE`

Removed tests whose only value was preserving completed-investigation output:

- Context-ablation report wording, stored CSV recomputation, 2025 decomposition,
  missingness-report inventory, and generated interaction-report contents.
- Frozen context-prior development/model-selection/C6 report assertions.
- Lean-context validation's duplicate frozen-artifact/source-literal check.
- Primitive-likelihood rolling-report wording and frozen candidate-definition
  thresholds.
- YPP's frozen diagnostic candidate-name assertion.

No tests were classified `DUPLICATE_REMOVE` or `OBSOLETE_REMOVE` merely to
reduce the headline count. The remaining tests exercise code paths or
invariants that can regress in future changes.

## Baseline and timing

The initial audit baseline was the 2026-09-08 `origin/main` snapshot at
`71db6ea`:

- 270 tests collected
- 267 passed and 3 failed
- 87.38 seconds wall time
- Slowest cases: context-ablation stored-metric recomputation (9.28 s),
  Performance V1 loopy diagnostics (8.66 s), and site-data deterministic or
  publication-slot tests (6.33 s and about 3.4 s each).

The three failures were clean-checkout fixture/path problems. The FCS
population and Massey parser tests now use minimal isolated fixtures, and the
context-prior test follows the current annual fitted-instance artifact path.

After rebasing onto current `main` and pruning, the final complete retained
suite was measured with:

```bash
uv run pytest -q --durations=25
```

The result was 265 passed in 67.85 seconds. The slowest retained cases were
Performance V1 loopy diagnostics (10.72 s), site-data byte determinism (6.30
s), site-data publication (4.14 s), and site-data publication-slot/conference
contracts (about 3.5-3.9 s each). Filesystem and CPU cache state made repeated
site-export runs vary materially. `ruff check .` and `git diff --check` are
also required validation commands. No production or published artifact is
changed by this test-suite work.
