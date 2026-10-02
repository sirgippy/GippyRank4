# Context 1.4 candidate freeze

Context 1.4 candidate is the frozen, research-only Context 1.3 prior with one
change: a positive net Context-only fitted location contribution is multiplied
by `0.75`. A zero or negative contribution is unchanged. The whole fitted
conditional-location mixture shifts by the same amount, preserving its centered
offsets and residual scale. Cold-start fallback PMFs are copied unchanged.

The machine-readable specification is
[`config/context_v1_4_candidate.json`](../config/context_v1_4_candidate.json).
The construction API is
`gippyrank.context_prior_v1_4_candidate.construct_candidate_prior`. It accepts
only a Context 1.3 `AnnualFittedInstance` trained through exactly `T - 1`, a
model hash, and the fitted prior inputs. It has no alpha parameter. The API
records the canonical candidate-spec SHA-256 with each serialized prior; any
specification change produces a different identity.

The candidate is not the active production model and has not been holdout
validated. This freeze does not change Context 1.3 fitting, publication,
rankings, snapshots, or API defaults.

Issue #158 and merged PR #159 are the development-selection lineage. The
2022–2025 development-panel prior PMF hashes in
[`data/processed/context_v1_4_candidate/development_panel_parity.json`](../data/processed/context_v1_4_candidate/development_panel_parity.json)
are parity fixtures only. They identify the `.75` variant previously produced
by PR #159; they are not fresh validation evidence.

Any change to alpha, selection logic, or the moderation semantics creates a
different candidate identity and requires a new validation cycle. Future
validation should retain the frozen candidate hash and compare it with Context
1.3 and History 1.1 on genuinely new evidence, without changing the candidate
after observing results. This issue does not validate or promote the candidate.
