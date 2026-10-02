# Context 1.4 candidate freeze

Context 1.4 candidate is the frozen, research-only Context 1.3 prior with one
change: a positive net Context-only fitted location contribution is multiplied
by `0.75`. A zero or negative contribution is unchanged. The whole fitted
conditional-location mixture shifts by the same amount, preserving its centered
offsets and residual scale. Cold-start fallback PMFs are copied unchanged.

The semantic contract is pinned in code and hashed independently of lifecycle
metadata. The checked-in
[`config/context_v1_4_candidate.json`](../config/context_v1_4_candidate.json)
is its audit mirror; candidate construction does not read from the repository
filesystem. The API
`gippyrank.context_prior_v1_4_candidate.construct_candidate_prior` requires the
actual Context 1.3 fitted model, its rolling-origin instance, an outcome-free
`InferenceRow`, the validated input provenance, and the source Context 1.3
PMF. It verifies the model specification and PMF, derives location terms from
the model and row, and computes the model and input hashes itself. It has no
alpha or caller-supplied model-hash parameter. Serialized candidates identify
the source PMF, frozen semantics, model, input row, team, target season, and
input provenance class.

The candidate is not the active production model and has not been holdout
validated. This freeze does not change Context 1.3 fitting, publication,
rankings, snapshots, or API defaults.

Issue #158 and PR #159 head at merge
(`ec0ef4c08b5aa10e534488bb7292cb2901626aee`) are recorded as the
development-selection lineage. The
2022–2025 development-panel prior PMF hashes in
[`data/processed/context_v1_4_candidate/development_panel_parity.json`](../data/processed/context_v1_4_candidate/development_panel_parity.json)
are parity fixtures only. The checked-in outcome-free model and inference
fixture exercises the canonical decomposition path and reproduces the `.75`
variant previously produced by PR #159; it is not fresh validation evidence.

Any change to alpha, selection logic, or the moderation semantics creates a
different candidate identity and requires a new validation cycle. Future
validation should retain the frozen candidate hash and compare it with Context
1.3 and History 1.1 on genuinely new evidence, without changing the candidate
after observing results. This issue does not validate or promote the candidate.
