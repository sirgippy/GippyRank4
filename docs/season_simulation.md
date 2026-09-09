# Season Simulation V1

Season Simulation V1 is a snapshot-scoped posterior-predictive forecast of
the remaining regular season. Its canonical object is the
`season_simulation` field in the snapshot's lazy `team_seasons.json` artifact;
it is computed once per predictive snapshot and consumed by both team pages
and validation.

## Generative definition

For each outer universe, the engine samples one ordinal latent rank for every
team from that team's posterior PMF:

```text
r_i ~ P(r_i | snapshot)
```

The sampled rank matrix is created before any future game is visited. A
team's rank is then reused for every remaining game in that universe. Given
those fixed ranks, each future margin is sampled from the unchanged Historical
Likelihood V1 Student-t distribution. Simulated game results are shared by
both teams; an away-team win is the complement of the canonical home-team
result. Simulated results never update the posterior.

The current posterior API exposes marginal PMFs rather than joint posterior
draws. V1 therefore samples each team's marginal independently. Ranks are not
forced into a permutation: duplicate nominal ranks are an intentional and
documented approximation to the unavailable joint posterior.

## Conditional win totals

The production forecast uses outer Monte Carlo over latent quality and an
exact conditional Poisson-binomial PMF for each team's remaining wins. This is
equivalent to the nested generative story for individual win totals while
avoiding unnecessary inner simulation noise:

```text
outer latent-quality draws
    -> conditional game win probabilities
    -> exact conditional Poisson-binomial PMFs
    -> average across outer universes
```

The default configuration is:

| Field | Value |
| --- | ---: |
| `simulation_version` | `hierarchical_latent_state_v1` |
| `outer_draw_count` | 2,000 |
| `inner_rollout_count` | 0 |
| `seed` | 49,049 |
| `conditional_distribution_method` | `exact_poisson_binomial` |
| `likelihood_version` | `V1` |

The optional inner count is available for shared game-level marginal and
cross-game-dependence diagnostics. It is not used to update inference or to
replace the exact win-total distributions.

## Published fields

Each FBS team summary includes its fixed completed regular-season record,
remaining-game count, final-win distribution, final record probabilities,
expected and median final wins, central 50%/80%/95% final-win intervals,
threshold probabilities, conditional expected-win summaries, and a variance
decomposition:

```text
Var(final wins)
  = Var_r(E[final wins | r])
  + E_r(Var(final wins | r))
```

The first term is labeled `team_quality`; the second is
`game_randomness`. Event decompositions apply the same idea to selected
thresholds and `win_out`/`lose_out`.

Future games outside the regular-season scope, completed games, and games
whose Historical Likelihood representation is unsupported are not silently
converted to probabilities. A team whose final forecast depends on an
unsupported future game has `forecast_status: "unavailable"` and a structured
reason in `unsupported_games`.

Schedule accounting is explicit. Every regular-season row involving a known
FBS team is classified as durably completed, strictly future, unresolved, or
cancelled/postponed. Durably completed rows contribute to the fixed record;
future and unresolved rows remain in the forecast scope; cancelled/postponed
rows are recorded in `excluded_schedule_games`; and lower-division games are
kept in the fixed record when their score is known but cannot be forecast by
Historical Likelihood V1. Each FBS summary carries the invariant
`completed_regular_season_games + remaining_games = forecast_scope_games`.
An in-progress or otherwise non-durable row is therefore fail-closed rather
than silently omitted.

When an FCS opponent is scheduled but absent from the frozen prior, the
snapshot builder adds a uniform PMF over the authoritative full-season FCS
rank universe. For the current cached season that universe comes from the
full-season schedule, so future FBS/FCS games are represented before the FCS
team appears in completed evidence. If no authoritative universe is
available, the snapshot fails closed.

Retained snapshots keep the posterior's original `fcs_fallback_*` and
`fcs_population_*` metadata: those fields describe the team universe that
produced the stored posterior PMFs. The backfill command may add FCS teams only
for future simulation; that separate provenance is recorded under
`season_simulation.provenance` as `season_simulation_fcs_*` fields and does not
rewrite the posterior provenance.

## Determinism and validation

The PRNG is a local `numpy.random.default_rng` initialized from the published
configuration seed. Team IDs and future games are ordered canonically before
sampling. Runtime is not serialized into the forecast object, so replaying a
fixed snapshot, schedule, model version, configuration, and seed produces the
same JSON values.

`game_marginals` compares the outer conditional integration against the exact
existing single-game V1 mixture for every supported future game. The retained
tests cover deterministic replay, posterior point-mass limiting behavior,
positive dependence from shared latent quality, conditional PMF normalization,
record/threshold consistency, unsupported-game fail-closed behavior, and
fixed completed records. The existing exact future-game predictor and
Historical Likelihood V1 implementation are unchanged.

The raw helper `simulate_shared_game_outcomes` is available for development
validation of the shared-game contract. It returns one canonical home-result
matrix per game; the away result for the same outer/inner cell is its logical
complement.

Durable artifacts retain `game_marginals`, single-game equivalence checks,
event/dependence diagnostics, and Monte Carlo diagnostics for auditability.
The static browser export strips those diagnostic fields from the lazy
team-season payload while retaining team win distributions and summaries;
manifest entries report the durable size, browser increment, and diagnostic
component sizes. Historical validation results are recorded in
[`season_forecast_validation.md`](season_forecast_validation.md).
