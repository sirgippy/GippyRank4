# Issue 194 Part B: frozen synthetic protocol

This protocol is fixed before generating or reviewing simulation scores. It is
an exploratory mechanism test, separate from Part A's historical diagnostics.

## Question and inference arms

The experiment asks whether (a) replacing a known dependent joint rank prior by
the product of its marginals changes posterior rank forecasts, and (b) the
existing loopy belief propagation (BP) adds approximation error beyond that
independence replacement.

For every simulated dataset, compare these three arms on identical ranks,
schedule, home/neutral assignments, and game margins:

1. Exact enumeration using the generating joint prior.
2. Exact enumeration using the product of the generating prior's team
   marginals.
3. Existing `infer_posterior` BP using those same product marginals.

The independent-prior case has the same joint prior in arms 1 and 2. In the
shared-prior case, arm 1 integrates the two joint scenarios with their correct
posterior evidence weights. It does not average scenario posteriors equally.

## State space and fixed graphs

There are four teams (`A`, `B`, `C`, `D`), all using actual ordinal rank states
1 through 5. There is no unique-rank or permutation constraint. The state space
has 5^4 = 625 assignments per inference, below the 5-team, 5-state ceiling of
3,125 assignments.

The complete fixed graph set is:

| Graph | Undirected schedule edges |
| --- | --- |
| `path_tree` | A–B, B–C, C–D |
| `cycle_4` | A–B, B–C, C–D, D–A |

Each edge contributes one game. For each draw and edge, a fair coin selects
which endpoint is home and an independent Bernoulli(0.25) draw selects a
neutral site. The same generated game objects are passed to all inference
arms. This is the only graph set; no graph or result dependent tuning is done.

## Fixed team marginal PMFs and shared scenarios

The same marginal PMFs are used in both generative cases. Rank columns are
ordered 1, 2, 3, 4, 5:

| Team | Marginal PMF |
| --- | --- |
| A | (0.35, 0.25, 0.18, 0.13, 0.09) |
| B | (0.10, 0.18, 0.24, 0.26, 0.22) |
| C | (0.17, 0.21, 0.24, 0.22, 0.16) |
| D | (0.26, 0.24, 0.22, 0.16, 0.12) |

The shared case has two equally weighted rank-distribution scenarios. Add
`delta = (0.04, 0.02, 0, -0.02, -0.04)` to each team's marginal in scenario A
and subtract it in scenario B. Conditional on the selected scenario, team
ranks are independent. The component is a shared choice between rank-PMF
profiles, not a scalar team-strength state. Each team marginal is exactly the
equally weighted average of its two conditional PMFs, so the only difference
from the independent case is cross-team dependence.

## Generative cases and frozen V1 observations

* `independent_null`: draw each team's rank independently from the fixed
  marginal PMF; that product is the true joint prior.
* `shared_context`: select scenario A/B with prior probability 0.5 each, then
  draw team ranks independently from that scenario's conditional PMFs; the
  true joint prior is the resulting two-component mixture.

For each game, use the pinned Historical Likelihood V1 parameters and current
V1 design to find the Student-t location at its generated home/away ranks and
site assignment. Draw one continuous Student-t margin with V1 degrees of
freedom and scale. The research-only synthetic `Game` stores this continuous
margin in `home_points` and zero in `away_points`; this preserves the exact
continuous V1 observation model instead of rounding it to an integer score.
The unchanged V1 game-factor builder then supplies the same likelihood to exact
enumeration and BP. These ephemeral values are not football score data and are
never written to production inputs or snapshots.

Generation is implemented separately from inference: the sampler receives the
generating prior, frozen likelihood, graph and seed; inference receives only
the generated games and the specified priors. Known simulated ranks are used
only to score posterior output.

## Frozen budget and seeds

Run 100 draws for each of the four case/graph combinations (400 total). The
complete seed list is the contiguous integer range 194000 through 194399,
assigned in blocks:

| Case | Graph | Seed range |
| --- | --- | --- |
| independent_null | path_tree | 194000–194099 |
| independent_null | cycle_4 | 194100–194199 |
| shared_context | path_tree | 194200–194299 |
| shared_context | cycle_4 | 194300–194399 |

Each draw uses a fresh NumPy `default_rng(seed)`. Draw order is: shared scenario
if applicable, team ranks in A–D order, then graph edges in table order; for
each edge draw home orientation, site, and Student-t margin. No reruns, tuning,
or early stopping based on observed scores are allowed.

## Reported measures and interpretation

For each arm, report mean posterior rank NLL at the known simulated rank,
multicategory Brier score, signed and absolute posterior expected-rank error,
rank-state calibration tables (mean forecast probability against realized
frequency), and BP convergence/iteration summaries where applicable.

For each case/graph, report mean team-level total variation and absolute
expected-rank error for:

* product-prior exact versus true-joint exact (dependence replacement effect);
* BP versus product-prior exact (small-graph BP approximation effect); and
* BP versus true-joint exact (combined difference).

The four-team tree must agree with exact product-prior inference within the
focused numerical tolerance. Simulation summaries are descriptive over this
fixed seed list; they are not estimates of a full-season causal share. A
synthetic positive control supports possibility only under the stated setup.

For the controlled shared-prior case, classify NLL degradation from replacing
the joint prior by its product as supported when the product-prior exact mean
NLL is higher in both graph panels, unsupported when it is not higher in both,
and unresolved when the two graph directions disagree. If one graph is exactly
zero and the other is positive, classify it as unresolved. The BP approximation
is present on the cycle only when its mean team-level total variation from
product-prior exact exceeds 1e-10; report its paired NLL change separately so
a marginal difference is not presented as a score effect. Tree agreement is
reported independently. A nonzero marginal approximation with opposite-signed
rank-NLL changes across the two generated cases is evidence of a posterior
approximation, while its rank-score effect remains unresolved.

The final report will state the Part A historical classification alongside the
Part B controlled result, distinguish rank scoring from game forecasting, and
will not infer that Context degrades real-game forecasts. No production prior,
likelihood, inference configuration, snapshot, website default, or raw download
is changed.
