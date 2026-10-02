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
`InferenceRow`, the authoritative transfer-input provenance returned by a
validated transfer loader, and the source Context 1.3 PMF. It verifies the
model specification and PMF, derives location terms from the model and row,
and computes the model and input hashes itself. It has no alpha or
caller-supplied model-hash parameter. The frozen model contract also requires
Context-only numeric and missingness scale coefficients to be zero.

`ContextTransferInputProvenance` identifies transfer lineage only; it does not
certify recruiting, talent, returning-production, coaching, or rank-history
inputs. Those values are independently bound by the complete inference-row
hash. Production and retrospective 2026 provenance can only be minted by their
class-specific loaders. Those loaders validate the canonical snapshot chain,
target FBS population, row schema and values, and derived feature artifact
contents. Identity hashes include the actual feature-file and manifest bytes,
snapshot IDs and raw hashes, cutoff state, stable logical artifact IDs, and
team-level transfer values. Absolute filesystem paths are diagnostic only and
do not affect identity. Retrospective 2026 provenance explicitly retains its
late retrieval timestamps, endpoints, raw hashes, missing archived August 15
snapshot declaration, and reconstructed-state statement. Development fixtures
use a constructor that can create only retrospective research provenance.

Cold starts cannot be constructed from a caller-supplied PMF. They are derived
through `GenericRankPrior` or loaded from the canonical annual History 1.1
prediction artifact, and serialize the fallback method, source parameters or
artifact hash, fallback PMF hash, reason, target team, season, and population.
Cold-start artifacts set `context_model_sha256` to null because the Context 1.3
fitted model did not produce those PMFs. PR #159 comparison fixtures use a
separately named research-only artifact-copy path.

The candidate is not the active production model and has not been holdout
validated. This freeze does not change Context 1.3 fitting, publication,
rankings, snapshots, or API defaults.

Issue #158 and PR #159 head at merge
(`ec0ef4c08b5aa10e534488bb7292cb2901626aee`) are recorded as the
development-selection lineage. The
2022–2025 hashes are named as retained PR #159 reference-output hashes in
[`data/processed/context_v1_4_candidate/development_panel_parity.json`](../data/processed/context_v1_4_candidate/development_panel_parity.json)
and are not candidate-output hashes. The canonical parity test computes the
candidate/reference PMF differences across all 534 2022–2025 team-seasons; the
record pins the observed maximum absolute difference and caps machine-level
variation at `5e-17`. The checked-in outcome-free model and inference fixture
exercises the canonical decomposition path; it is not fresh validation
evidence.

Any change to alpha, selection logic, or the moderation semantics creates a
different candidate identity and requires a new validation cycle. Future
validation should retain the frozen candidate hash and compare it with Context
1.3 and History 1.1 on genuinely new evidence, without changing the candidate
after observing results. This issue does not validate or promote the candidate.
