# Posterior Snapshot V1

The posterior engine's latent variable is each team's **final ordinal rank in
its subdivision**. It does not introduce a scalar strength or Elo rating. A
team starts with its frozen H 1.1 or C 1.2 PMF and each completed game adds the
Historical Likelihood V1 pairwise margin factor:

`p(r_1, ..., r_n | games) ∝ ∏_i p_preseason(r_i) ∏_g p_t(margin_g | r_home(g), r_away(g), site_g)`.

The engine uses deterministic damped loopy sum-product belief propagation.
It is exact for trees, and the explicit convergence gate prevents an invalid
cyclic approximation from being marked valid. Marginal PMFs stay discrete and
are serialized directly; display order is expected rank within FBS only.

## Historical Likelihood V1 contract

The existing frozen implementation in `scripts/build_historical_modeling.py`
is used without a semantic change:

- Rank coordinate is `(ordinal_rank - 0.5) / subdivision_population`.
- Same-subdivision games use an odd smooth basis in home-minus-away rank
  coordinate: difference, two difference-by-mean terms, difference magnitude,
  four hinge terms, plus non-neutral home field.
- FBS/FCS games use stable FBS-first coordinates: intercept, FBS/FCS linear,
  quadratic, interaction, four FBS and four FCS hinges, and FBS-home/FCS-home
  indicators. Their target is FBS points minus FCS points regardless of the
  schedule listing order.
- Same-system constituent rank pairs are used; the historical builder's
  deterministic per-game sampling fallback only applies when none exist.
- Margin density is Student-t. The selected V1 degrees of freedom is 15.0;
  the selected production fit is the weighted-pseudo eight-iteration IRLS
  surface. Its scale is serialized with its coefficient vector.
- FBS/FBS, FBS/FCS, and FCS/FCS are all modeled. Neutral and home treatment is
  encoded in the factors. Postseason was not excluded by the frozen builder.
  No YPP field is in the fitted factor design; YPP is retained in the corpus
  only as a production-status/audit field.

The legacy `margin_model_results.json` omitted beta coefficients, so it cannot
on its own evaluate a posterior. `scripts/materialize_historical_likelihood_v1.py`
reconstructs the documented V1 weighted-pseudo fit strictly from the existing
immutable historical modeling corpus and saves a downstream coefficient
artifact. It neither edits nor refits an upstream preseason artifact.

## Snapshot contract

Schema `1.0` bundles live under
`data/processed/snapshots/<season>/<snapshot-id>/predictive/<prior-family>/`:

- `metadata.json`: provenance, input hashes, cutoff, included IDs, versions,
  validity, and lower-division audit count.
- `rankings.json` and `rankings.csv`: small static-client display rows.
- `posterior_pmfs.csv`: normalized per-team discrete PMFs.
- `included_games.csv`: exact game-source audit trail.
- `diagnostics.json`: convergence and runtime data.

`context` is the canonical public family; `history` is explicitly an alternate
shadow/control family. A preseason snapshot is the same contract with zero
games. Weekly and live snapshots differ only by explicit timestamp cutoff.
Snapshot construction has no publishing, website, polling, or ballot code.

Only final FBS/FCS-vs-FBS/FCS games at or before the cutoff are factors. Every
game involving another subdivision is excluded and counted. The current frozen
H/C artifacts have FBS rows only; when an FCS team appears before a cutoff the
engine obtains that season's full FCS population from the durable Massey
team-season distribution corpus. For a current season not yet represented
there, it instead enumerates the full cached CFBD season schedule—not games
through the cutoff—then adds a uniform PMF over ranks `1..N_FCS`. It fails
rather than inventing a support if neither full-season source is available.
The population, fallback IDs, kind, and PMF semantics are recorded in metadata.
The population source is also explicit (`massey_team_season_rank_distributions`
or `cfbd_full_season_schedule`).
This is a temporary upstream-data limitation, not a lower-division placeholder.

## Validation threshold

The toy suite predeclares a total-variation error limit of 0.03 for the
three-team cyclic approximation. The two-team one-game posterior is compared
to exact enumeration at numerical tolerance. It also tests a round robin,
cross-subdivision factor, multi-hop propagation, and disconnected-team prior
preservation.

The broader deterministic audit is [bp_exact_validation.json](../data/processed/posterior_validation/bp_exact_validation.json): 72 varied 2–5 team cases, including mixed FBS/FCS and repeated pairings. Before execution it required p95 marginal TV ≤ 0.05 and worst TV ≤ 0.15. The recorded result passes both thresholds. Rematches are consolidated by multiplying their likelihood matrices in canonical pair orientation; individual game IDs remain in the snapshot provenance.

Its observed convergence rate is 100%; median/p95/worst marginal TV are
0.000114/0.00607/0.02521, and median/worst expected-rank error are
0.000108/0.04993. This supports the approximation for this deliberately small
stress population, not as a proof for every full-season graph.

## Rolling historical evidence

The compact 2022–2025 panel uses seven actual-date regular-season cutoffs per
season; individual bundles are temporary during evaluation and only summaries
and plots are retained in `data/processed/posterior_backtest/`. Both H- and
C-started posteriors materially improve their own prior NLL by season end in
all four seasons. Interval widths fall from about 70 ranks early to 28–29 late.

Late C interval coverage is near nominal in 2022 (0.797) and 2024 (0.793), but
is below nominal in 2023 (0.740) and 2025 (0.726). H is better calibrated in
those latter seasons. This is evidence of a possible late-season C
overconfidence/calibration issue; no variance inflation, clipping, or recency
discount was added here. H-vs-C remains diagnostic: the posterior is judged by
improvement against its own prior and calibration, not by requiring C to beat H
in every season.
