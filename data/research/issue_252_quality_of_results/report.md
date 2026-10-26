# Issue 252: Schedule equivalence and outcome sensitivity

## Scope and frozen method

This is research only. It evaluates the issue 242 Quality of Results inference at λ = 0, 0.25, 0.50, 0.75, and 1; it does not change or publish a ranking. Synthetic margins are drawn from the byte-pinned Historical Likelihood V1 Student-t model with site effects. Focal and opponent beliefs are discrete 2026 Context 1.4 rank PMFs, and the issue 242 inference strips each focal Context prior after posterior inference. The shared focal prior is Colorado's preseason PMF (expected rank 63.14) from the complete 138-team FBS support.

The unconditional panel has 500 paired seasons. The record-conditioned panel has 200 exact 5–5 / 8–2 pairs, found in 4351 independent seed attempts. That panel conditions on the motivating records and must be read separately from the unconditional panel. The same-schedule control compares two teams facing cloned approximately-even prior schedules, with the same latent focal and opponent ranks but independent margin draws.

The schedule-control conclusion applies to the median-prior focal strength in the 2026 FBS population. Additional focal-strength strata were not run.

## Computed schedules

Each slate has 10 games with 4 home, 4 away, and 2 neutral sites. Opponent probabilities integrate the frozen V1 rank-pair margin distribution over the complete opponent and focal prior PMFs.

| role | target | computed | range |
| --- | --- | --- | --- |
| strong_opponents | 0.200 | 0.208 | 0.084–0.339 |
| approximately_even | 0.500 | 0.501 | 0.302–0.686 |
| weaker_opponents | 0.800 | 0.780 | 0.628–0.941 |

The strong-opponent slate is used for outcome sensitivity only; the primary schedule-equivalence comparison is approximately-even versus weaker opponents.

## Representative season comparisons

One comparison per panel is selected by distance to the median generated-model residual score. The record-conditioned panel therefore shows the motivating 5–5 / 8–2 pair. No inferred rank or candidate output was used to select either example.

| panel | λ | even record | weaker record | even E[rank] | weaker E[rank] | Δ rank | rank SD Δ | entropy Δ | PMF TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unconditional | 0.0 | 6-4 | 9-1 | 50.4 | 46.2 | -4.3 | +4.0 | +0.19 | 0.130 |
| unconditional | 0.25 | 6-4 | 9-1 | 52.0 | 51.5 | -0.5 | +3.2 | +0.20 | 0.086 |
| unconditional | 0.5 | 6-4 | 9-1 | 53.5 | 55.3 | +1.9 | +2.4 | +0.17 | 0.075 |
| unconditional | 0.75 | 6-4 | 9-1 | 54.9 | 58.1 | +3.2 | +1.7 | +0.13 | 0.078 |
| unconditional | 1.0 | 6-4 | 9-1 | 56.2 | 60.1 | +4.0 | +1.2 | +0.09 | 0.084 |
| record_conditioned | 0.0 | 5-5 | 8-2 | 63.8 | 64.2 | +0.4 | +2.7 | +0.15 | 0.069 |
| record_conditioned | 0.25 | 5-5 | 8-2 | 63.2 | 62.3 | -0.9 | +2.0 | +0.13 | 0.049 |
| record_conditioned | 0.5 | 5-5 | 8-2 | 62.5 | 60.5 | -2.0 | +1.5 | +0.10 | 0.047 |
| record_conditioned | 0.75 | 5-5 | 8-2 | 62.0 | 59.0 | -3.1 | +1.1 | +0.08 | 0.063 |
| record_conditioned | 1.0 | 5-5 | 8-2 | 61.6 | 57.5 | -4.1 | +0.7 | +0.05 | 0.087 |

## Unconditional schedule equivalence

The rank difference is weaker-schedule expected rank minus approximately-even expected rank; a negative value favors the weaker-schedule team. Intervals are 95% Monte Carlo intervals for means. The outrank probability uses a 95% Wilson interval. TV and Wasserstein measure paired posterior PMF differences.

| λ | mean Δ rank | median | 5–95% | P(weaker outranks) | mean SD Δ | mean entropy Δ | mean 80% width Δ | mean TV | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | -1.23 [-2.93, 0.46] | 0.28 | -37.0 to 28.9 | 0.484 [0.440, 0.528] | +2.68 | +0.16 | +7.52 | 0.290 | 500 |
| 0.25 | -1.03 [-2.09, 0.03] | -0.01 | -23.3 to 18.1 | 0.500 [0.456, 0.544] | +1.95 | +0.13 | +5.39 | 0.205 | 500 |
| 0.5 | -0.78 [-1.41, -0.15] | -0.15 | -12.9 to 10.2 | 0.508 [0.464, 0.552] | +1.34 | +0.10 | +3.63 | 0.139 | 500 |
| 0.75 | -0.61 [-0.98, -0.25] | -0.37 | -7.4 to 6.1 | 0.544 [0.500, 0.587] | +0.89 | +0.07 | +2.42 | 0.094 | 500 |
| 1.0 | -0.54 [-0.82, -0.26] | -0.63 | -6.0 to 4.7 | 0.576 [0.532, 0.619] | +0.56 | +0.05 | +1.48 | 0.077 | 500 |

A small mean by itself is not evidence of equivalence. The interval and the spread of paired rank differences show how often individual season realizations differ.

## Record-conditioned schedule comparison

| λ | mean Δ rank (95% MC interval) | median | 5–95% | P(weaker outranks; 95% Wilson interval) | mean rank SD Δ | mean entropy Δ | mean 80% width Δ |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: |
| 0 | 0.10 [-0.01, 0.21] | 0.03 | -1.1 to 1.5 | 0.495 [0.426, 0.564] | +2.92 | +0.16 | +8.47 |
| 0.25 | 0.30 [0.09, 0.52] | 0.30 | -2.0 to 3.2 | 0.450 [0.383, 0.519] | +2.07 | +0.13 | +5.87 |
| 0.5 | 0.32 [-0.03, 0.66] | 0.20 | -3.1 to 4.8 | 0.460 [0.392, 0.529] | +1.43 | +0.10 | +3.84 |
| 0.75 | 0.21 [-0.23, 0.65] | 0.02 | -4.1 to 5.9 | 0.495 [0.426, 0.564] | +0.96 | +0.07 | +2.55 |
| 1 | 0.02 [-0.49, 0.53] | -0.35 | -5.3 to 6.7 | 0.535 [0.466, 0.603] | +0.60 | +0.05 | +1.48 |

The same-schedule control estimates outcome-driven variation when opponent priors, venue slots, and latent ranks are shared. Its distribution is the reference for interpreting schedule-pair variation; it is not subtracted from the schedule comparison.

### Same-schedule control

| λ | mean Δ rank (95% MC interval) | 5–95% | P(right team outranks; 95% Wilson interval) | mean PMF TV |
| ---: | --- | --- | --- | ---: |
| 0 | 1.27 [-0.78, 3.32] | -39.6 to 40.7 | 0.466 [0.423, 0.510] | 0.314 |
| 0.25 | 1.20 [-0.62, 3.02] | -32.1 to 38.4 | 0.498 [0.454, 0.542] | 0.307 |
| 0.5 | 1.14 [-0.57, 2.84] | -30.2 to 36.9 | 0.478 [0.435, 0.522] | 0.306 |
| 0.75 | 1.08 [-0.59, 2.75] | -29.5 to 33.9 | 0.486 [0.442, 0.530] | 0.313 |
| 1 | 1.03 [-0.65, 2.71] | -29.7 to 33.6 | 0.482 [0.439, 0.526] | 0.327 |

## Outcome sensitivity

The contrast reports rank benefit of outcome B over A as expected rank A minus expected rank B, so positive values mean B receives a better expected rank. Each row reuses the same opponents, venue, and other nine game results. A zero effect at λ=0 for same-winner margin changes is expected from the binary endpoint and serves as a consistency check. All five weights are shown.

| schedule | case | λ | focal p(win) | rank benefit | rank SD Δ | entropy Δ | PMF TV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong_opponents | win_loss_boundary: narrow_loss_vs_win | 0.0 | 0.20 | 22.40 | -2.09 | -0.10 | 0.320 |
| strong_opponents | win_loss_boundary: narrow_loss_vs_win | 0.25 | 0.20 | 16.29 | +1.16 | +0.12 | 0.256 |
| strong_opponents | win_loss_boundary: narrow_loss_vs_win | 0.5 | 0.20 | 9.73 | +1.83 | +0.17 | 0.179 |
| strong_opponents | win_loss_boundary: narrow_loss_vs_win | 0.75 | 0.20 | 4.85 | +1.23 | +0.12 | 0.105 |
| strong_opponents | win_loss_boundary: narrow_loss_vs_win | 1.0 | 0.20 | 1.63 | +0.53 | +0.05 | 0.040 |
| strong_opponents | within_win_margin: ordinary_vs_decisive_win | 0.0 | 0.20 | 0.00 | +0.00 | +0.00 | 0.000 |
| strong_opponents | within_win_margin: ordinary_vs_decisive_win | 0.25 | 0.20 | 1.76 | +0.26 | +0.01 | 0.027 |
| strong_opponents | within_win_margin: ordinary_vs_decisive_win | 0.5 | 0.20 | 2.82 | +0.87 | +0.05 | 0.047 |
| strong_opponents | within_win_margin: ordinary_vs_decisive_win | 0.75 | 0.20 | 3.19 | +1.30 | +0.09 | 0.060 |
| strong_opponents | within_win_margin: ordinary_vs_decisive_win | 1.0 | 0.20 | 3.26 | +1.45 | +0.12 | 0.071 |
| strong_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.0 | 0.20 | 0.00 | +0.00 | +0.00 | 0.000 |
| strong_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.25 | 0.20 | -4.63 | -1.01 | -0.10 | 0.081 |
| strong_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.5 | 0.20 | -6.58 | -2.08 | -0.21 | 0.138 |
| strong_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.75 | 0.20 | -7.20 | -2.50 | -0.29 | 0.180 |
| strong_opponents | within_loss_margin: ordinary_vs_decisive_loss | 1.0 | 0.20 | -7.38 | -2.56 | -0.34 | 0.212 |
| approximately_even | win_loss_boundary: narrow_loss_vs_win | 0.0 | 0.50 | 12.51 | +3.81 | +0.49 | 0.291 |
| approximately_even | win_loss_boundary: narrow_loss_vs_win | 0.25 | 0.50 | 8.40 | +2.52 | +0.33 | 0.214 |
| approximately_even | win_loss_boundary: narrow_loss_vs_win | 0.5 | 0.50 | 5.36 | +1.55 | +0.21 | 0.148 |
| approximately_even | win_loss_boundary: narrow_loss_vs_win | 0.75 | 0.50 | 3.13 | +0.87 | +0.12 | 0.093 |
| approximately_even | win_loss_boundary: narrow_loss_vs_win | 1.0 | 0.50 | 1.50 | +0.42 | +0.06 | 0.047 |
| approximately_even | within_win_margin: ordinary_vs_decisive_win | 0.0 | 0.50 | 0.00 | +0.00 | +0.00 | 0.000 |
| approximately_even | within_win_margin: ordinary_vs_decisive_win | 0.25 | 0.50 | 1.69 | +0.55 | +0.05 | 0.038 |
| approximately_even | within_win_margin: ordinary_vs_decisive_win | 0.5 | 0.50 | 3.01 | +0.96 | +0.10 | 0.073 |
| approximately_even | within_win_margin: ordinary_vs_decisive_win | 0.75 | 0.50 | 4.06 | +1.23 | +0.13 | 0.106 |
| approximately_even | within_win_margin: ordinary_vs_decisive_win | 1.0 | 0.50 | 4.90 | +1.40 | +0.15 | 0.138 |
| approximately_even | within_loss_margin: ordinary_vs_decisive_loss | 0.0 | 0.50 | 0.00 | +0.00 | +0.00 | 0.000 |
| approximately_even | within_loss_margin: ordinary_vs_decisive_loss | 0.25 | 0.50 | -2.17 | -0.92 | -0.13 | 0.064 |
| approximately_even | within_loss_margin: ordinary_vs_decisive_loss | 0.5 | 0.50 | -3.57 | -1.45 | -0.22 | 0.113 |
| approximately_even | within_loss_margin: ordinary_vs_decisive_loss | 0.75 | 0.50 | -4.49 | -1.71 | -0.27 | 0.151 |
| approximately_even | within_loss_margin: ordinary_vs_decisive_loss | 1.0 | 0.50 | -5.08 | -1.79 | -0.29 | 0.180 |
| weaker_opponents | win_loss_boundary: narrow_loss_vs_win | 0.0 | 0.80 | 7.73 | +3.43 | +0.52 | 0.257 |
| weaker_opponents | win_loss_boundary: narrow_loss_vs_win | 0.25 | 0.80 | 5.44 | +2.30 | +0.37 | 0.196 |
| weaker_opponents | win_loss_boundary: narrow_loss_vs_win | 0.5 | 0.80 | 3.71 | +1.48 | +0.25 | 0.144 |
| weaker_opponents | win_loss_boundary: narrow_loss_vs_win | 0.75 | 0.80 | 2.37 | +0.88 | +0.15 | 0.097 |
| weaker_opponents | win_loss_boundary: narrow_loss_vs_win | 1.0 | 0.80 | 1.30 | +0.44 | +0.08 | 0.056 |
| weaker_opponents | within_win_margin: ordinary_vs_decisive_win | 0.0 | 0.80 | 0.00 | +0.00 | +0.00 | 0.000 |
| weaker_opponents | within_win_margin: ordinary_vs_decisive_win | 0.25 | 0.80 | 1.73 | +0.69 | +0.09 | 0.055 |
| weaker_opponents | within_win_margin: ordinary_vs_decisive_win | 0.5 | 0.80 | 3.17 | +1.16 | +0.16 | 0.106 |
| weaker_opponents | within_win_margin: ordinary_vs_decisive_win | 0.75 | 0.80 | 4.39 | +1.47 | +0.21 | 0.155 |
| weaker_opponents | within_win_margin: ordinary_vs_decisive_win | 1.0 | 0.80 | 5.45 | +1.66 | +0.25 | 0.200 |
| weaker_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.0 | 0.80 | 0.00 | +0.00 | +0.00 | 0.000 |
| weaker_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.25 | 0.80 | -1.08 | -0.50 | -0.10 | 0.047 |
| weaker_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.5 | 0.80 | -1.90 | -0.86 | -0.17 | 0.087 |
| weaker_opponents | within_loss_margin: ordinary_vs_decisive_loss | 0.75 | 0.80 | -2.54 | -1.10 | -0.23 | 0.121 |
| weaker_opponents | within_loss_margin: ordinary_vs_decisive_loss | 1.0 | 0.80 | -3.04 | -1.26 | -0.28 | 0.150 |
| weaker_opponents | weak_opponent_large_win: ordinary_vs_decisive_win | 0.0 | 0.80 | 0.00 | +0.00 | +0.00 | 0.000 |
| weaker_opponents | weak_opponent_large_win: ordinary_vs_decisive_win | 0.25 | 0.80 | 1.38 | +0.56 | +0.07 | 0.043 |
| weaker_opponents | weak_opponent_large_win: ordinary_vs_decisive_win | 0.5 | 0.80 | 2.55 | +0.94 | +0.13 | 0.085 |
| weaker_opponents | weak_opponent_large_win: ordinary_vs_decisive_win | 0.75 | 0.80 | 3.56 | +1.20 | +0.17 | 0.124 |
| weaker_opponents | weak_opponent_large_win: ordinary_vs_decisive_win | 1.0 | 0.80 | 4.45 | +1.35 | +0.20 | 0.162 |

These fixed counterfactuals characterize the candidates' relative response to a winner change and to margins within the same outcome. They do not establish a universal exchange rate between a win and additional points.

## Interpretation and next step

The unconditional panel estimates a weaker-schedule advantage of −1.03 expected-rank places at λ=0.25 (95% Monte Carlo interval −2.09 to +0.03; weaker team outranks probability 0.500). The interval includes zero, and the representative 5–5 / 8–2 panel slightly favors the approximately-even schedule at this weight (+0.30 places; weaker team outranks in 45% of draws). At λ=0.50 and above, unconditional mean shifts toward the weaker-schedule team are under one place, while the paired difference distribution remains much wider than that mean. The outrank probability rises to 0.544 at λ=0.75 and 0.576 at λ=1. Different posterior uncertainty is present: at λ=0.25 the weaker slate has mean rank SD +1.95, entropy +0.13 bits, and 80% interval width +5.39 places.

At λ=0.25, changing a narrow loss to a narrow win improves expected rank by 8.40 places against the approximately-even opponent and 5.44 against the weaker opponent; changing an ordinary +10 win to +35 improves rank by 1.69 and 1.73 places in those same cases. The weak-opponent +14 to +35 comparison improves rank by 1.38. By λ=0.50, decisive-margin changes approach the win/loss change for a weaker opponent. These counterfactuals make λ=0.25 the clearest existing candidate in this experiment for preserving the win/loss distinction while retaining margin evidence.

**Recommendation:** carry λ=0.25 into a preregistered follow-up with additional focal-strength strata, then review it against the intended season-accomplishment tradeoff before any separate model or publication proposal. The evidence supports further evaluation, not activation: the unconditional mean interval at λ=0.25 narrowly crosses zero, the conditional comparison favors the stronger schedule slightly, and only one focal-strength stratum was evaluated.

## Reproducibility

`manifest.json` records source hashes, candidate weights, prior population, seeds, sample counts, generator and inference settings, and artifact hashes. `schedule_definitions.csv`, `prior_profiles.csv.gz`, and `historical_likelihood_v1.json` retain the frozen model inputs. `simulated_games.csv.gz` preserves unconditional and accepted conditional observations. `representative_comparison.csv` and `representative_pmfs.csv.gz` retain selected season comparisons; `counterfactual_pmfs.csv.gz` retains the outcome-sensitivity distributions. Per-trial rank and uncertainty summaries allow the aggregate tables to be reconstructed.
