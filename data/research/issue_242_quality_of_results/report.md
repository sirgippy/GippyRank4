# Issue 242: Quality of Results research

## Scope and method

This is a research replay only. It does not activate or publish a third Performance ranking. For each completed game, the experimental factor is the binary winner likelihood raised to `1 − λ` times the existing scoring-margin likelihood raised to `λ`. Each λ is inferred as its own network posterior with the frozen Historical Likelihood V1, site effects, subdivision handling, selected Context prior, deterministic damped belief propagation, and the existing Performance prior-stripping method. The endpoints use the existing Quality of Record and Quality of Play factors directly; no completed rankings or posterior distributions are averaged.

The replay used 8 official weekly Context snapshots and 40 candidate inference runs, from 2026-09-08 through 2026-10-25T12-59-22.246522Z, with candidate weights 0, 0.25, 0.5, 0.75, 1. Each snapshot retained its own frozen eligible-game evidence and inference settings.

The September 8 snapshot is the repaired issue 238 lineage: malformed completed 0–0 fixture `401866625` is excluded and its actual replacement `401917058` is retained. The runner checks that the malformed ID is absent from every selected snapshot and records whether the replacement game is present. Quality of Play and Quality of Record baseline artifacts are checked against the exact Context evidence before endpoint parity is measured.

## Endpoint parity

| Endpoint | Snapshots | Largest absolute PMF difference | Result |
| --- | ---: | ---: | --- |
| Quality of Record (λ=0) | 8 | 6.94e-17 | pass at tolerance 1e-08 |
| Quality of Play (λ=1) | 8 | 1.67e-16 | pass at tolerance 1e-08 |

## Candidate sensitivity across official snapshots

Pairwise disagreement is the fraction of rated FBS team pairs whose order differs from the named endpoint, using expected-rank ordering. Top-10 and Top-25 changes are the average symmetric difference in membership. Rank changes use display rank. Uncertainty reports the average posterior rank standard deviation and entropy across rated teams.

| λ | Pairwise disagreement vs QOR | vs QOP | Mean display-rank change vs QOR / QOP | Mean expected-rank change vs QOR / QOP | Top-10 changes vs QOR / QOP | Top-25 changes vs QOR / QOP | Adjacent display / expected-rank change | Adjacent pair disagreement | Mean rank SD | Mean entropy (bits) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000% | 17.014% | 0.00 / 17.16 | 0.00 / 13.84 | 0.00 / 8.75 | 0.00 / 16.75 | 0.00 / 0.00 | 0.000% | 27.72 | 6.57 |
| 0.25 | 7.156% | 9.908% | 7.38 / 10.18 | 4.99 / 8.88 | 4.75 / 4.75 | 6.50 / 10.50 | 7.38 / 4.99 | 7.156% | 26.05 | 6.48 |
| 0.5 | 11.839% | 5.228% | 12.03 / 5.46 | 8.77 / 5.10 | 7.50 / 2.00 | 11.25 / 5.50 | 4.95 / 3.78 | 4.698% | 24.40 | 6.38 |
| 0.75 | 14.809% | 2.224% | 14.99 / 2.44 | 11.63 / 2.23 | 8.25 / 0.50 | 14.00 / 3.00 | 3.22 / 2.87 | 3.004% | 22.93 | 6.28 |
| 1 | 17.014% | 0.000% | 17.16 / 0.00 | 13.84 / 0.00 | 8.75 / 0.00 | 16.75 / 0.00 | 2.44 / 2.23 | 2.224% | 21.65 | 6.19 |

The largest team-level expected-rank movement between neighboring weights was 21.0 places for team `276` at 2026-09-27 (λ=0.0 to 0.25). The full per-team trajectories are in `rankings.csv`.
This is meaningful sensitivity, not numerical non-convergence: every candidate run converged. Marshall had a 3-1 record in this cutoff (3 wins and 1 loss). Its largest loss was by 45 points to Penn State. The margin-weighted factor distinguishes results by score margin, while the binary factor counts each win or loss once.

Rank changes across λ are fully reported in `rankings.csv`; per-cutoff convergence and pairwise metrics are in `snapshot_summary.csv` and `sensitivity_summary.csv`.

## Team-level case studies

The primary examples use the latest official cutoff (Week 9); the largest adjacent-λ sensitivity case uses its own source cutoff. Game-level evidence follows each case; FCS opponents are shown but do not receive an FBS expected rank.

- **winning_record_vs_margin — Buffalo (5-3) at Week 9:** expected rank is 78.3 under Quality of Record, 101.4 at λ=0.50, and 111.6 under Quality of Play. Average scoring margin is -2.8; the mean opponent Quality of Play rank is 79.6.
  Game evidence: W 25–18 vs Massachusetts (opponent Quality of Play rank 101.6); L 13–55 vs Penn State (opponent Quality of Play rank 32.0).
- **competitive_losses_vs_elite — Nebraska (5-2) at Week 9:** expected rank is 36.7 under Quality of Record, 18.1 at λ=0.50, and 10.8 under Quality of Play. Average scoring margin is 20.1; the mean opponent Quality of Play rank is 59.6.
  Game evidence: L 17–20 vs Indiana (opponent Quality of Play rank 6.6); L 14–17 vs Oregon (opponent Quality of Play rank 20.1).
- **dominant_win_vs_lower_ranked_opponent — South Carolina (4-3) at Week 9:** expected rank is 39.9 under Quality of Record, 32.4 at λ=0.50, and 27.7 under Quality of Play. Average scoring margin is 10.9; the mean opponent Quality of Play rank is 33.3.
  Game evidence: W 57–0 vs Kent State (opponent Quality of Play rank 119.9).
- **single_margin_outlier — Oregon (4-3) at Week 9:** expected rank is 38.1 under Quality of Record, 26.9 at λ=0.50, and 20.1 under Quality of Play. Average scoring margin is 12.4; the mean opponent Quality of Play rank is 21.7.
  Game evidence: W 84–0 vs Portland State (opponent Quality of Play rank not ranked in FBS).
- **same_record_schedule_quality — Nevada (3-4) at Week 9:** expected rank is 101.3 under Quality of Record, 94.0 at λ=0.50, and 88.8 under Quality of Play. Average scoring margin is 4.3; the mean opponent Quality of Play rank is 97.3.
  Game evidence: L 17–31 vs North Dakota State (opponent Quality of Play rank 48.2); W 30–13 vs UTEP (opponent Quality of Play rank 130.7).
- **same_record_schedule_quality — Purdue (3-4) at Week 9:** expected rank is 53.7 under Quality of Record, 50.2 at λ=0.50, and 46.8 under Quality of Play. Average scoring margin is -1.0; the mean opponent Quality of Play rank is 30.0.
  Game evidence: L 10–49 vs Notre Dame (opponent Quality of Play rank 4.9); L 26–38 vs Washington (opponent Quality of Play rank 54.4).
- **Largest adjacent-λ shift — Marshall (3-1) at Week 5:** from λ=0 to λ=0.25, expected rank moved from 58.2 to 79.2 (+21.0); display rank moved from 63 to 95 (+32).
  Game evidence: L 0–45 vs Penn State (opponent Quality of Play rank 25.2); W 36–35 vs Gardner-Webb (opponent Quality of Play rank not ranked in FBS); W 28–26 vs Middle Tennessee (opponent Quality of Play rank 99.9); W 30–24 vs Missouri State (opponent Quality of Play rank 103.6).

## Oregon blowout counterfactual

Using the frozen Week 9 inputs, Oregon's 84–0 Portland State win was changed only for this sensitivity run to a 35–7 win (28-point margin). The matchup, Oregon win, priors, likelihood, all other games, and inference settings were held fixed. This is not a proposed score correction or a production ranking change.

| λ | Original expected rank | Counterfactual expected rank | Change | Original display rank | Counterfactual display rank | Change | Expected-rank change as % of λ=1 | Display-rank change as % of λ=1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38.08 | 38.08 | +0.00 | 38 | 38 | +0 | 0.0% | 0.0% |
| 0.25 | 31.79 | 35.99 | +4.20 | 33 | 39 | +6 | 38.8% | 46.2% |
| 0.5 | 26.94 | 34.08 | +7.14 | 28 | 37 | +9 | 65.9% | 69.2% |
| 0.75 | 23.12 | 32.40 | +9.28 | 27 | 39 | +12 | 85.6% | 92.3% |
| 1 | 20.12 | 30.96 | +10.84 | 24 | 37 | +13 | 100.0% | 100.0% |

The λ=0 result is unchanged because Oregon still wins; margin evidence is absent at that endpoint. Relative to the λ=1 effect, the counterfactual movement at λ=0.25 is 38.8% for expected rank and 46.2% for display rank. At λ=0.50 those shares are 65.9% and 69.2%; at λ=0.75 they are 85.6% and 92.3%. The λ=1 expected-rank and display-rank changes are +10.84 and +13. Thus lower intermediate weights materially reduce this blowout's ranking effect, though λ=0.75 remains close to λ=1. These observed differences quantify this game's influence on these rankings; they do not decide how wins and margins ought to be valued.

## Marshall fine weight sweep

Using the same frozen Week 5 evidence, the λ interval from 0 to 0.25 was sampled in 0.05 steps. The added runs leave the full-grid results untouched; the endpoints reuse their Week 5 results from the main replay.

| λ | Expected rank | Expected-rank change from prior step | Display rank | Second difference | Converged |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.00 | 58.21 | — | 63 | — | True |
| 0.05 | 62.65 | +4.44 | 70 | — | True |
| 0.10 | 67.03 | +4.38 | 76 | -0.06 | True |
| 0.15 | 71.28 | +4.25 | 83 | -0.13 | True |
| 0.20 | 75.35 | +4.07 | 88 | -0.18 | True |
| 0.25 | 79.19 | +3.85 | 95 | -0.22 | True |

Expected rank is monotone across this sweep. The five 0.05-step expected-rank changes range from +3.85 to +4.44 places; the largest absolute second difference is 0.22 places. These adjacent changes and curvature values assess trajectory shape separately from convergence. Marshall was 3–1: a 45-point loss at Penn State and three close wins (36–35 over Gardner-Webb, 28–26 over Middle Tennessee, 30–24 over Missouri State). The sweep shows how increasing margin weight gives that blowout loss more relative influence than the binary record does. This is an empirical response of the model; it does not determine the intended normative tradeoff.

## Interpretation and recommendation

The candidate is a coherent one-pass inference model: λ=0.50 retains all winner evidence while tempering how strongly the observed margin differentiates rank pairs. This gives a direct normative knob between rewarding the win and crediting strong play in a loss. The model continues to use the opponent network, so schedule context is not replaced with a record bonus or a named-opponent adjustment. Large margins still contribute at intermediate λ, though their conditional evidence is tempered.

The historical replay describes rank behavior; it does not establish which balance of wins and margins best represents season accomplishment. The issue explicitly makes λ a judgment about that meaning, and there is no objective target here that can select it without changing the question into poll imitation or predictive-margin optimization.

The Oregon counterfactual shows that λ=0.25 and λ=0.50 substantially temper this extreme margin's effect; the Week 5 Marshall sweep is smooth at the tested 0.05 resolution. Neither sensitivity result selects a preferred λ: the choice still depends on how the ranking should value wins against margin evidence.

**Recommendation:** retain Quality of Results as a research candidate and continue with a separate implementation/publication proposal only after the product owner chooses the intended win-versus-margin tradeoff. Treat λ=0.50 as a transparent central candidate for that discussion, not an empirically selected winner. Do not promote a third production variant from this study alone. A follow-up should preregister the interpretation and compare the documented game-level cases with that interpretation before publication.

## Reproducibility

The machine-readable manifest pins each publication slot, Context metadata, included game rows, prior, likelihood, and published endpoint artifacts. `rankings.csv` preserves rank and uncertainty summaries for every FBS team and λ; `posterior_pmfs/` retains the full distributions with deterministic gzip timestamps (`mtime=0`). `case_studies.csv` and `case_study_games.csv` provide selected team cases and underlying game rows. The two targeted experiments are in `oregon_blowout_counterfactual.csv` and `marshall_fine_weight_sweep.csv`. All candidate runs converged; see `snapshot_summary.csv` and the targeted outputs for convergence details.
