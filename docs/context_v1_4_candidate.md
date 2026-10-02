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
actual Context 1.3 fitted model, its rolling-origin instance, the typed fit
source returned by the canonical fitting or validated loading path, an
outcome-free `InferenceRow`, authoritative transfer-input provenance, and the
source Context 1.3 PMF. It verifies the model, fit source, inference row and
PMF, derives location terms from the model and row, and computes the model and
input hashes itself. It has no alpha or caller-supplied model-hash parameter.
The frozen model contract also requires Context-only numeric and missingness
scale coefficients to be zero.

The fitted-model source binds model family and spec version, target season,
rolling-origin cutoff, model metadata SHA-256, fitted-instance identity,
frozen specification identity, and a content-addressed training-corpus
identity. `fit_model_with_source` rejects non-FBS rows and any training row
after `target_season - 1`. When source rows are available, the loader refits
through that canonical path and verifies model and corpus identity. The
retained 2026 annual model is loaded through its checked-in, content-pinned fit
attestation, which binds the committed model and instance to the corpus
identity recorded by the canonical fit. A hand-built model without an
authoritative source, or a research-only source presented as canonical, is
rejected. Future Context 1.3 builds write the typed source metadata beside the
fitted model.

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
use a constructor that can create only retrospective research provenance. The
real committed 2026 reconstruction artifacts pass a content-pinned
compatibility loader with their original serialized class
`retrospective_2026_reconstruction`; no artifact regeneration or relabeling is
needed. The published Context 1.3 class strings remain
`production_preseason_immutable_snapshot` and
`retrospective_2026_reconstruction`.

Production-shaped cold starts cannot be constructed from a caller-supplied PMF
or arbitrary `GenericRankPrior` parameters. They select the requested row from
the validated canonical History 1.1 annual source. That source binds model
family/version, target season, `trained_through_season`, fitted-instance
identity, model metadata, prediction artifact bytes and semantics, and the
exact FBS team PMFs. The fallback identity records the selected team, prior
method, source artifact hash, History fitted-instance identity,
`trained_through_season`, reason, and PMF hash. The retained PR #159
cold-start fixture remains available only through its explicitly research-only
artifact-copy path. It does not contain its original reason, so it records
`unspecified_in_pr159_reference` rather than inferring one from the PMF.
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
candidate/reference PMF differences across all 534 2022–2025 team-seasons. The
record pins the largest observed absolute and relative differences across the
Python 3.13 local and GitHub CI runs (`8.326672684688674e-17` and
`1.03152444e-15`) and caps them at `1e-16` absolute and `1e-14` relative. The
checked-in outcome-free model and inference fixture exercises the canonical
decomposition path; it is not fresh validation evidence.

Any change to alpha, selection logic, or the moderation semantics creates a
different candidate identity and requires a new validation cycle. Future
validation should retain the frozen candidate hash and compare it with Context
1.3 and History 1.1 on genuinely new evidence, without changing the candidate
after observing results. This issue does not validate or promote the candidate.
