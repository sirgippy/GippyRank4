# Offense/Defense Posterior V2 investigation

Issue #40 is a research diagnostic. It does not replace the scalar Posterior
V1, change production rankings, or alter publication artifacts.

## Frozen experiment

The historical split is 2008–2017 training, 2018–2021 development, and
2022–2025 evaluation. The 2022–2025 results are leakage-safe historical
evaluation, not untouched independent confirmation. 2026 outcomes are never
used.

The V1 expected margin is evaluated at the historical rank-pair observations.
Because V1 exposes a margin rather than separate expected scores, the runner
uses a training-only pairing/site total-scoring environment. If `m` is the
V1 expected margin and `T` is that environment's total, the two oriented
expected scores are `(T + m) / 2` and `(T - m) / 2`. Same-subdivision games
use home-minus-away orientation; cross-subdivision games use FBS-minus-FCS
orientation. This is symmetric and does not assign the whole margin to one
side.

For each team/game:

```text
offensive residual = points scored - expected points scored
defensive residual = expected opponent points - opponent points scored
```

Stage 0 reports same-component and cross-component lag correlations, raw and
team-season-demeaned variants, early/late relationships, elapsed-time bins,
season rows, shuffle-order nulls, shuffled unrelated-team pairings, and
deterministic summary effects.

Stage 1 has only two predeclared candidates:

- `OD0`: independent offense/defense prior components.
- `OD1`: the same model with a fixed `rho=0.25` offense/defense prior
  correlation.

The research fit uses scoreboard-only observations. Its score means are:

```text
home score = environment_home + O_home - D_away
away score = environment_away + O_away - D_home
```

Offense and defense each have a population sum-to-zero constraint. This fixed
centering removes the location transformation that would otherwise make
`O + c, D + c` indistinguishable. Context is the primary prior source; its
scalar quality is split symmetrically as a starting point, while the OD game
evidence estimates the component separation. The candidate's score
distribution is an independent Normal with a training-frozen scale, so score
NLL is reported only for this explicitly defined OD distribution.

Candidate selection uses only the development gate from the issue:

- next-game margin NLL improvement of at least `0.010`;
- margin MAE improvement of at least `0.10` points;
- win Brier worsening no greater than `0.001`;
- NLL improvement in at least three of four development seasons; and
- no development-season NLL worsening greater than `0.015`.

Evaluation runs only for the single development-selected candidate. If no
candidate passes, the investigation stops and no production OD posterior is
created.

## Reproduction

The checked-in runner accepts a sibling checkout containing the large
historical inputs:

```bash
uv run python scripts/investigate_offense_defense.py \
  --input-root /path/to/input/checkout \
  --output-root .
```

It writes deterministic artifacts under
`data/processed/offense_defense_latent/`:

```text
report.md
summary.json
model_spec.json
stage0_component_persistence.csv
stage0_null_metrics.csv
development_future_metrics.csv
development_season_metrics.csv
development_component_diagnostics.csv
```

Evaluation and profile artifacts are emitted only if the development gate
passes. The runner hashes the frozen V1 and prior artifacts before and after
the run and fails if those hashes change.
