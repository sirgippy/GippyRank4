# Context 1.3 production validation

Context 1.3 is the active production Context specification. The retained Context 1.2 artifacts remain historical and reproducible.

## Frozen contract

- Location: History plus the exact Context 1.3 feature list.
- Scale: History features only.
- Distribution: Normal; penalty 0.25; existing deterministic optimizer retry.
- Incoming transfer values: only the three model-facing columns from the #114 attach-only boundary.

## Aggregate 2022–2025 metrics

| Representation | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|
| context_1_2 | 4.542321404 | 0.111942700 | 20.163011 | 20.594569 | 0.824977048 | 71.919476 |
| D5 | 4.508080630 | 0.107986183 | 19.489575 | 19.675094 | 0.837702774 | 71.958801 |
| P3 | 4.506145604 | 0.107769293 | 19.408713 | 19.645131 | 0.836860128 | 71.621723 |

## Frozen parity

- D5 parity: passed.
- P3 parity: passed.
- P3 standardized coefficients: passed.
- Historical target population: unchanged.
- H fallback population: unchanged.
- Rolling-origin P3 parity: passed.

## Provenance and activation guardrails

Historical 2021–2025 transfer values are retrospective research reconstructions, not archived August 15 snapshots. Future annual inference requires a validated immutable manifest, complete canonical FBS rows, an on-time cutoff, and explicit provenance metadata; absent or late inputs fail closed. The 2026 activation is the documented retrospective exception and cannot masquerade as cutoff-safe.

The 2026 Context 1.2 annual artifact and its weekly publication lineage are not overwritten or reinterpreted by activation.
