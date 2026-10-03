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
rolling-origin cutoff, model metadata SHA-256, fitted-instance identity, and
immutable semantic-specification identity. Canonical fits additionally bind a
verified corpus source, exact row payload digest, and row count. Legacy and
research classes have distinct lineage fields. Lifecycle fields such as active
version, candidate status, and promotion state are reported separately and do
not affect the Context 1.3 semantic hash. The semantic specification covers
the minimum scale, lag and distribution settings, equal row weights,
optimizer method, initialization, tolerances and retry, coefficient bounds,
regularization, scale link, and preprocessing scale floor and standard deviation
convention. Context 1.3 passes its frozen fit settings explicitly to the
generic fitter.

Only `load_context13_training_corpus` creates a typed
`Context13TrainingCorpusSource`. It binds the canonical historical rank and
Context inputs, feature-construction implementation, historical transfer
artifact and provenance, sorted FBS team-season keys, and exact resulting
`TeamSeason` rows. `fit_model_with_source` rejects non-FBS or post-cutoff rows;
without that typed source, caller-supplied rows can produce only a
`research_only` fit source. The source validator rejects subsets, extra rows,
identity changes, and changes to lag, target, or feature values.

The committed 2026 Context 1.3 model remains usable through its
`legacy_attested_context13_fit` source, which pins its model metadata, fitted
instance, published prediction artifact, and the retained 2,744-row coverage
claim. It is explicitly marked `retained_legacy_attestation`, with no verified
or claimed corpus digest. The checkout does not contain
`data/processed/modeling/team_season_rank_distributions.csv`, so that corpus
cannot currently be reconstructed. A hand-built model without an authoritative
source, or a research-only source presented as canonical, is rejected.

Fit and transfer provenance use an explicit compatibility matrix:
canonical fits pair with production or retrospective-2026 transfer lineage;
the retained legacy attestation pairs only with retrospective-2026 lineage;
research-only fits pair only with retrospective-research transfer fixtures.
Candidate artifacts serialize the fit provenance class, reproducibility level,
corpus identity and row count, semantic-spec hash, model metadata hash, and
fitted-instance hash.

## Artifact schema versions

New Context 1.3 model/report metadata uses artifact schema 2. Transfer
provenance uses schema 2, fitted-model provenance uses schema 3, and immutable
Context 1.3 semantic metadata uses schema 3. Context 1.4 candidate artifacts
use schema 4. Retained 2026 History provenance uses schema 1; future canonical
History sources and research fixtures use schema 2. The Context 1.3 prediction
CSV schema remains version 1. Its committed model, prediction, and publication
files remain unchanged. Provenance sidecars and model-spec metadata may advance
without changing PMFs or publication semantics.

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
the validated History 1.1 annual source. History owns its source type, builder,
loader, and cold-start row checks; Context 1.4 consumes that public API. The
2026 source is a pinned retained legacy artifact. A future canonical source is
emitted by the History annual builder from fixed processed inputs retained with
its output; loading it reproduces the fit and every PMF before accepting the
sidecar. See
[`history_annual_provenance.md`](history_annual_provenance.md) for the build and
verification path. The annual source binds model family/version, target and
trained-through seasons, History semantic and training-input identities,
fitted-instance and model identities, prediction bytes and semantics, actual
FBS population, prior methods, and exact team PMFs. The candidate fallback
records the source provenance class and lineage, selected team and method,
source artifact hash, fitted-instance identity, reason, and PMF hash. The retained PR #159
cold-start fixture remains available only through its explicitly research-only
artifact-copy path. It does not contain its original reason, so it records
`unspecified_in_pr159_reference` rather than inferring one from the PMF.
Cold-start artifacts set `context_model_sha256` to null because the Context 1.3
fitted model did not produce those PMFs. PR #159 comparison fixtures use a
separately named research-only artifact-copy path.

The candidate is not the active production model and has not been holdout
validated. Context 1.3's fitted coefficients, predictions, publications,
rankings, snapshots, and API defaults remain unchanged.

Issue #158 and PR #159 head at merge
(`ec0ef4c08b5aa10e534488bb7292cb2901626aee`) are recorded as the
development-selection lineage. The
2022–2025 hashes are named as retained PR #159 reference-output hashes in
[`data/processed/context_v1_4_candidate/development_panel_parity.json`](../data/processed/context_v1_4_candidate/development_panel_parity.json)
and are not candidate-output hashes. The canonical parity test computes the
candidate/reference PMF differences across all 534 2022–2025 team-seasons. The
record pins the largest observed absolute and relative differences across the
Python 3.13 local and GitHub CI full-panel runs (maximum absolute difference
`8.326672684688674e-17`, maximum relative difference
`2.7763615161996095e-14`) and caps them at `1e-16` absolute and `5e-14`
relative. The
checked-in outcome-free model and inference fixture exercises the canonical
decomposition path; it is not fresh validation evidence.

Any change to alpha, selection logic, or the moderation semantics creates a
different candidate identity and requires a new validation cycle. Future
validation should retain the frozen candidate hash and compare it with Context
1.3 and History 1.1 on genuinely new evidence, without changing the candidate
after observing results. This issue does not validate or promote the candidate.
