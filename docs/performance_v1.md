# Performance V1

Performance V1 is a research-only descriptive estimate of how good a team
appears to be from the football it has actually played. It is not standings,
strength of record, postseason qualification, selection advice, a reward for
winning, or a human-poll emulator. Competition rules determine postseason
qualification separately.

The public website does not register, select, or publish this family. The
research build is `scripts/build_performance_v1.py`; its default output is
`data/processed/performance_v1/`.

## Mathematical definition

Let `r` be the final ordinal rank coordinate in the team's subdivision and let
`G_T` be the completed FBS/FCS games at cutoff `T`. The frozen Historical
Likelihood V1 supplies one pairwise factor for each game:

`p(margin_g | r_home(g), r_away(g), site_g, subdivision_g)`.

For the primary Context anchor, let `C_i(r)` be the focal team's Context
preseason PMF and let `Post_C_i(r | G_T)` be its marginal from the ordinary
Context-started BP run on the complete schedule network. The Performance C
PMF is:

`Performance_C_i(r | G_T) = normalize(Post_C_i(r | G_T) / C_i(r))`.

Performance H uses History priors for every network variable and removes the
focal team's History prior in the same way:

`Performance_H_i(r | G_T) = normalize(Post_H_i(r | G_T) / H_i(r))`.

Under exact inference with factorized team priors, the ratio is the game-only
likelihood profile for the focal team under a uniform prior. The focal
preseason prior is removed as a direct factor. Under loopy BP, a small indirect
feedback residue can remain because prior information may propagate through
opponents and return through schedule cycles. Explicit focal-prior neutralization
is the correctness baseline; empirical stripping error on historical cases is
quantified separately. Context or History still anchors non-focal teams because
opponent quality must be estimated in order to interpret the focal team's
margins. Context is the predeclared primary candidate; History is an
anchor-sensitivity/control analysis, not an ensemble or a selection rule.

The focal neutral prior is uniform over the applicable FBS support. With zero
eligible games, Performance is exactly uniform and `rated=false`; arbitrary
tie-breaking among identical zero-game PMFs is not meaningful.

## Inference and approximation audit

Posterior V1 uses deterministic damped loopy sum-product BP. Prior stripping is
the efficient implementation of the exact identity, but loopy BP means the
ratio can differ from rerunning BP with only the target prior neutralized. The
research build therefore runs a fixed representative baseline: ordinary
Context BP with the selected target replaced by a uniform PMF. It records TV,
expected-rank, median, and 80% interval differences in
`bp_prior_removal_validation.csv`.

The prior-removal panel uses a fixed prospective stratification across all four
evaluation seasons and fixed early/mid/late cutoff indices. It includes zero,
1–3, 4–6, and 7+ game strata; elite, middle, and weak ordinary posterior
locations; narrow, broad, and irregular PMFs; high Context/History disagreement;
and high graph degree and cycle-edge exposure. Selection uses only ordinary
posterior, anchor, and graph descriptors plus deterministic team-ID tie breaks;
it does not inspect stripping error, explicit-neutralization output, or future
outcomes. The resulting case count and strata are recorded in
`summary.json` and `bp_prior_removal_validation.csv`.

A small loopy-cycle audit is also recorded. Direct focal-prior removal is exact
on a tree, while loopy message feedback can leave a residual. The experiment
varies topology, cycle length, the number of returning paths, focal-prior
strength, and convergence settings. If the residual is material under the
predeclared rule, explicit target-neutralized inference is selected.

The acceptance rule is predeclared before inspecting the full study:

- p95 target PMF TV at most `0.05`;
- worst target PMF TV at most `0.15`.

If that rule fails in `--method auto`, the build selects explicit target
neutralization for the final Performance calculations. Explicit neutralization
is slower because it reruns ordinary BP for each focal team, but correctness
has priority over speed. The selected method and audit values are recorded in
`summary.json` and `report.md`.

The production interpretation is therefore not a claim of perfect mathematical
focal-prior independence under approximate loopy inference. It is a direct
factor-removal construction whose realistic approximation error is measured
against explicit focal neutralization.

## Evidence semantics

Performance uses the existing Historical Likelihood V1 unchanged:

- rank-percentile likelihood surface;
- Student-t degrees of freedom 15;
- score margin, site, and FBS/FCS semantics;
- completed FBS/FCS games only;
- established FCS support/fallback behavior.

There is no win indicator, record feature, win bonus, loss penalty,
quality-win bonus, undefeated bonus, capped margin, recency weighting, YPP,
poll input, betting input, or postseason logic. A one-point win and one-point
loss against the same opponent at the same site are neighboring continuous
margin observations. A close road loss to a strong opponent can imply high
quality, while a close home win over a weak opponent can imply ordinary
quality.

The opponent estimate is current through the cutoff, not frozen at game day.
If a team plays an opponent in Week 1 and that opponent later demonstrates
stronger football before cutoff, the target's Performance can improve while
the target is idle. This is a deliberate consequence of interpreting the
target's observed game with current schedule-network opponent knowledge.

Every game factor enters the joint inference exactly once. The implementation
does not estimate an opponent using the focal game, freeze that posterior, and
multiply the focal game again. `game_evidence.csv` preserves game identity,
site, scores, margin, anchor family, opponent anchor summaries, cutoff, and
the resulting focal evidence count for later schedule views.

## Output interpretation

Each rated FBS team has a discrete PMF over quality-equivalent rank and these
summaries:

- display rank, ordered by expected quality-equivalent rank;
- expected rank, median, and mode;
- central 50%, 80%, and 95% intervals;
- Top 5, Top 10, and Top 25 probabilities;
- eligible games/evidence count and `rated` flag.

The PMF is not the probability of an AP, Coaches, or CFP ranking, a standings
finish, a joint permutation of ranks 1 through N, or postseason qualification.
Display ties use deterministic team-name/ID ordering only. Broad early PMFs are
intended behavior; the build does not narrow them for visual appeal.

## Historical design

The temporal evaluation uses leakage-safe actual-date cutoffs in 2022–2025,
with the existing seven-cutoff early/mid/late panel. Ratings at `T` use only
completed games whose source `startDate` is at or before the effective cutoff;
completed games after `T` form the future pool and are never included in that
inference. Predictive anchors are the frozen preseason H/C artifacts, and no
target-game result is used to construct an anchor. Predictive C, Predictive H,
Performance C, and Performance H are required to share identical
`(season, cutoff, game_id, focal_team_id)` scoring keys; duplicate or mismatched
keys fail the build. `all_future` is the inclusive future population and
`next_game` is its subset, not a disjoint complement. Candidate, scored,
excluded, and common-key counts are recorded in `summary.json`. These are
descriptive external checks, not a promotion criterion for replacing
Predictive rankings. Frozen H/C prediction artifacts do not contain 2018–2021
rows, so those priors are not fabricated for this study.

The anchor sensitivity artifact reports Context-minus-History expected-rank
differences, median differences, and PMF TV by cutoff, season phase, and games
played, with a season/week breakdown. It also records unique opponents, graph
degree, component cycle rank, cycle-edge exposure, opponent posterior C/H TV
sum and maximum, and focal posterior width. The companion
`anchor_sensitivity_decomposition.csv` reports fixed grouped summaries and
correlations. This decomposition characterizes whether later disagreement is
associated with many small anchor differences or a few divergent anchors; it
does not remove or ensemble the sensitivity. The difference measures uncertainty
about opponent quality; it is not remaining focal-team preseason information.
The Performance-versus-Predictive artifact reports how observed games changed
the story relative to the Predictive C marginal.

## Production boundary and reproducibility

This work does not modify H/C priors, Historical Likelihood V1, Posterior V1
default behavior, Predictive outputs, website selectors, or GitHub Pages data.
It does not fetch CFBD or poll data. It reads the cached corpus, frozen prior
artifacts, and serialized likelihood. Run it with:

```bash
uv run python scripts/build_performance_v1.py
```

The generated research bundle contains `summary.json` (including source hashes,
season coverage, missingness counts, topology diagnostics, and strict scoring
population checks), `report.md`, `artifact_inventory.csv`, current C and H
rankings/PMFs, `future_game_validation.csv`,
`future_game_predictions.csv`, `bp_prior_removal_validation.csv`,
`anchor_sensitivity.csv`, `anchor_sensitivity_decomposition.csv`,
`performance_vs_predictive.csv`, `game_evidence.csv`, deterministic illustrative
case selections, idle-team examples, and plots. No runtime cache is committed.
The row-level future and evidence tables are retained because they support
leakage and audit review; the inventory records their bytes, row counts,
purpose, and retention role. Repeating the build with the same inputs produces
the same substantive PMFs, rankings, validation rows, and hashes; wall-clock
runtime is reported separately because it is hardware dependent.
