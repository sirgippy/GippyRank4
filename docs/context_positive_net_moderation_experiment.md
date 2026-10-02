# Positive Context-only net location moderation (issue 158)

## Executive result

**Partial positive-net moderation improves mean posterior NLL in this rolling-origin replay, but the improvement is uneven across teams and does not select a production Context 1.4 model.** The fixed `alpha=0.75` and `alpha=0.50` variants have lower mean posterior NLL than unchanged Context 1.3 at every 2023–2025 shared-evidence checkpoint. Their pooled December gains are 0.0308 and 0.0420 NLL per team-season. All three primary seasons improve in mean December NLL, including 2024, and the 2022 control also improves by 0.0109 and 0.0091. History 1.1 remains lower in December NLL than either partial variant in every 2023–2025 season.

The gain is not broad. In 2024, `alpha=0.75` improves 55 of 134 teams and worsens 79; its median team-level NLL **worsens** by 0.0155. The top 10% of 2024 teams by gain supply 62.3% of all positive gain. Across 2023–2025, 200 improve and 203 worsen, and the top 10% supply 51.8% of positive gain. `alpha=0.50` makes the concentration and majority-worsening pattern stronger. Strong moderation (`alpha=0.25` or `0`) harms the 2024 final checkpoint and gives back much more of Context's preseason expected-rank advantage.

Context's pooled preseason expected-rank advantage over History is 1.434 ranks per team-season. `alpha=0.75` retains 1.419 ranks of it, with lower pooled preseason NLL and CRPS than unchanged Context. Its 2022 prior expected-rank error rises by 0.307 rank, while its 2022 final NLL improves. This is a favorable mean-score signal for **a separate, preregistered candidate-validation study**, not evidence to choose an alpha from the already-studied 2022–2025 panel. The 2024 team-level pattern, concentrated gains, and retrospective motivation from issue 154 make an untouched validation population essential before any model promotion.

## Design and rolling-origin safeguards

For each fitted team-season, let `x` be the signed `context_only_subtotal`: the sum of fitted Context-only location contributions after positive and negative terms cancel. The experiment uses exactly the predeclared grid `alpha ∈ {1.00, 0.75, 0.50, 0.25, 0.00}` and applies

```text
moderated_x = min(x, 0) + alpha * max(x, 0)
new conditional location point = fitted conditional location point + (moderated_x - x)
new center = intercept + history-derived subtotal + moderated_x
```

Every point of Context's fitted location mixture receives the same shift. Its centered offsets, residual scale, rank-bin mapping, uncertainty, likelihood, and belief-propagation settings are unchanged. Negative or zero net Context-only contributions, and six cold-start fallback priors, remain identical to Context 1.3 for every alpha. History 1.1 is the unchanged orientation baseline. No recruiting/talent threshold or transfer repair enters the intervention. There is no feature refit.

The diagnostic source contains later outcome columns, but prior construction projects **only** the fitted intercept, lag-1 contribution, frozen per-feature contributions, and conditional location points. It reconstructs `x` from the 15 Context 1.3 location features and checks it against the issue 154 contribution artifact. The evaluated target PMFs are loaded only after every candidate prior has been constructed. Each fitted row must name the matching rolling Context 1.3 and History 1.1 model hashes and have a training cutoff of `target season − 1`; any mismatch aborts. The only learned quantities are those frozen prior fits from the original rolling training populations. No moderation parameter is learned or selected, and no evaluated outcome sets a threshold or alpha.

The replay verifies the target corpus, game-source files, likelihood, checkpoint panel, and earlier diagnostic lineage by SHA-256. At each of the seven established checkpoints in 2022, 2023, 2024, and 2025, it checks cutoff-safe included game IDs and rows, FCS fallback IDs, and inference population against the retained issue 147 comparison. FBS team-seasons receive equal weight; the 2023–2025 pool concatenates 133, 134, and 136 teams rather than averaging the three season means. Checkpoint 1 is the zero-game preseason boundary. The model is evaluated after the same shared game evidence at checkpoints 2–7.

## Baseline parity

`alpha=1.00` was inferred independently from unchanged Context at all 28 checkpoints **before** any `alpha<1` posterior was evaluated. Its prior arrays were bit-identical to Context's retained prior arrays. The largest difference between independently computed `alpha=1.00` and Context posterior PMFs was **0**; the largest retained team-level and aggregate metric difference was **0**. Reconstructing the native Context prior PMF from its fitted conditional locations and scale differed by at most `2.78e-17`. The retained and diagnostic center values matched exactly, and the reconstructed net contribution differed from the rounded issue 154 artifact by at most `4.93e-12`. The retained baseline had already passed all 700 issue 140 reproduction checks.

## Preseason results

Lower NLL, CRPS, and expected-rank error are better. The final column is History's expected-rank error minus the model's, so positive values preserve some of Context's preseason accuracy advantage. `alpha=1.00` exactly duplicates the Context row.

| Season | Model | Prior NLL | Prior CRPS | Expected-rank error | Rank advantage over History |
| --- | --- | ---: | ---: | ---: | ---: |
| 2022 | History | 4.5276 | 0.0825 | 21.195 | 0.000 |
| 2022 | Context | 4.4562 | 0.0726 | 18.466 | 2.729 |
| 2022 | α=.75 | 4.4603 | 0.0732 | 18.773 | 2.422 |
| 2022 | α=.50 | 4.4720 | 0.0748 | 19.266 | 1.929 |
| 2022 | α=.25 | 4.4913 | 0.0773 | 19.989 | 1.206 |
| 2022 | α=.00 | 4.5182 | 0.0809 | 20.859 | 0.336 |
| 2023 | History | 4.4878 | 0.0735 | 20.103 | 0.000 |
| 2023 | Context | 4.4678 | 0.0719 | 19.216 | 0.887 |
| 2023 | α=.75 | 4.4597 | 0.0708 | 19.199 | 0.904 |
| 2023 | α=.50 | 4.4614 | 0.0706 | 19.356 | 0.747 |
| 2023 | α=.25 | 4.4728 | 0.0715 | 19.590 | 0.513 |
| 2023 | α=.00 | 4.4940 | 0.0736 | 19.984 | 0.119 |
| 2024 | History | 4.5879 | 0.0873 | 22.651 | 0.000 |
| 2024 | Context | 4.5512 | 0.0808 | 20.493 | 2.158 |
| 2024 | α=.75 | 4.5416 | 0.0801 | 20.617 | 2.034 |
| 2024 | α=.50 | 4.5432 | 0.0805 | 20.851 | 1.800 |
| 2024 | α=.25 | 4.5560 | 0.0821 | 21.294 | 1.357 |
| 2024 | α=.00 | 4.5799 | 0.0850 | 22.036 | 0.615 |
| 2025 | History | 4.5663 | 0.0784 | 21.111 | 0.000 |
| 2025 | Context | 4.5504 | 0.0739 | 19.856 | 1.255 |
| 2025 | α=.75 | 4.5412 | 0.0731 | 19.794 | 1.318 |
| 2025 | α=.50 | 4.5432 | 0.0736 | 19.904 | 1.207 |
| 2025 | α=.25 | 4.5565 | 0.0753 | 20.295 | 0.817 |
| 2025 | α=.00 | 4.5809 | 0.0785 | 21.081 | 0.030 |
| 2023–2025 | History | 4.5476 | 0.0798 | 21.290 | 0.000 |
| 2023–2025 | Context | 4.5234 | 0.0755 | 19.857 | 1.434 |
| 2023–2025 | α=.75 | 4.5145 | 0.0747 | 19.871 | 1.419 |
| 2023–2025 | α=.50 | 4.5162 | 0.0749 | 20.038 | 1.252 |
| 2023–2025 | α=.25 | 4.5287 | 0.0763 | 20.394 | 0.896 |
| 2023–2025 | α=.00 | 4.5519 | 0.0790 | 21.036 | 0.254 |

## Shared-evidence posterior results

Mean posterior NLL at each retained checkpoint follows. Each row is a separate season/checkpoint replay; the pooled rows use equal team-season weights. The machine-readable [checkpoint results](../data/processed/context_positive_net_moderation/checkpoint_results.csv) also give CRPS, expected-rank error, and explicit NLL differences versus both baselines for every model and checkpoint.

| Season | Checkpoint | Context | History | α=.75 | α=.50 | α=.25 | α=.00 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 1 (August) | 4.4562 | 4.5276 | 4.4603 | 4.4720 | 4.4913 | 4.5182 |
| 2022 | 2 (September) | 4.2873 | 4.3114 | 4.2845 | 4.2884 | 4.2988 | 4.3157 |
| 2022 | 3 (October) | 4.1603 | 4.1858 | 4.1626 | 4.1720 | 4.1885 | 4.2119 |
| 2022 | 4 (October) | 4.0650 | 4.0755 | 4.0584 | 4.0610 | 4.0725 | 4.0931 |
| 2022 | 5 (November) | 3.9807 | 3.9947 | 3.9735 | 3.9766 | 3.9900 | 4.0136 |
| 2022 | 6 (November) | 3.8517 | 3.8604 | 3.8433 | 3.8461 | 3.8603 | 3.8858 |
| 2022 | 7 (December) | 3.7964 | 3.8025 | 3.7855 | 3.7873 | 3.8016 | 3.8283 |
| 2023 | 1 (August) | 4.4678 | 4.4878 | 4.4597 | 4.4614 | 4.4728 | 4.4940 |
| 2023 | 2 (September) | 4.3876 | 4.3753 | 4.3750 | 4.3707 | 4.3746 | 4.3866 |
| 2023 | 3 (October) | 4.2769 | 4.2441 | 4.2609 | 4.2546 | 4.2578 | 4.2705 |
| 2023 | 4 (October) | 4.1286 | 4.1013 | 4.1084 | 4.0993 | 4.1013 | 4.1143 |
| 2023 | 5 (November) | 4.0294 | 3.9768 | 4.0015 | 3.9870 | 3.9859 | 3.9979 |
| 2023 | 6 (November) | 3.9858 | 3.9174 | 3.9493 | 3.9273 | 3.9197 | 3.9266 |
| 2023 | 7 (December) | 3.9137 | 3.8352 | 3.8694 | 3.8417 | 3.8305 | 3.8352 |
| 2024 | 1 (August) | 4.5512 | 4.5879 | 4.5416 | 4.5432 | 4.5560 | 4.5799 |
| 2024 | 2 (September) | 4.4071 | 4.3771 | 4.3932 | 4.3898 | 4.3968 | 4.4142 |
| 2024 | 3 (October) | 4.2547 | 4.2285 | 4.2440 | 4.2450 | 4.2575 | 4.2814 |
| 2024 | 4 (October) | 4.1451 | 4.0986 | 4.1300 | 4.1293 | 4.1427 | 4.1698 |
| 2024 | 5 (November) | 4.1238 | 4.0532 | 4.1023 | 4.0973 | 4.1087 | 4.1363 |
| 2024 | 6 (November) | 4.0132 | 3.9434 | 3.9917 | 3.9883 | 4.0030 | 4.0357 |
| 2024 | 7 (December) | 3.8719 | 3.8023 | 3.8580 | 3.8642 | 3.8898 | 3.9345 |
| 2025 | 1 (August) | 4.5504 | 4.5663 | 4.5412 | 4.5432 | 4.5565 | 4.5809 |
| 2025 | 2 (September) | 4.3609 | 4.3396 | 4.3467 | 4.3428 | 4.3494 | 4.3663 |
| 2025 | 3 (October) | 4.2368 | 4.1930 | 4.2159 | 4.2080 | 4.2133 | 4.2317 |
| 2025 | 4 (October) | 4.1137 | 4.0698 | 4.0934 | 4.0885 | 4.0991 | 4.1251 |
| 2025 | 5 (November) | 4.0635 | 4.0123 | 4.0398 | 4.0339 | 4.0459 | 4.0758 |
| 2025 | 6 (November) | 3.9997 | 3.9398 | 3.9711 | 3.9630 | 3.9754 | 4.0084 |
| 2025 | 7 (December) | 3.9252 | 3.8548 | 3.8909 | 3.8787 | 3.8889 | 3.9212 |
| 2023–2025 | 1 (August) | 4.5234 | 4.5476 | 4.5145 | 4.5162 | 4.5287 | 4.5519 |
| 2023–2025 | 2 (September) | 4.3851 | 4.3639 | 4.3715 | 4.3677 | 4.3735 | 4.3889 |
| 2023–2025 | 3 (October) | 4.2560 | 4.2217 | 4.2401 | 4.2357 | 4.2427 | 4.2610 |
| 2023–2025 | 4 (October) | 4.1290 | 4.0898 | 4.1105 | 4.1056 | 4.1143 | 4.1364 |
| 2023–2025 | 5 (November) | 4.0723 | 4.0142 | 4.0479 | 4.0395 | 4.0470 | 4.0702 |
| 2023–2025 | 6 (November) | 3.9996 | 3.9336 | 3.9708 | 3.9596 | 3.9662 | 3.9905 |
| 2023–2025 | 7 (December) | 3.9037 | 3.8309 | 3.8729 | 3.8617 | 3.8699 | 3.8972 |

The final shared checkpoint shows the full grid's absolute and paired results. Negative deltas improve on the named baseline. Improved/worsened counts use team-level NLL differences exceeding `1e-10` in magnitude; unchanged fallbacks count as ties.

| Season | Model | Final NLL | Δ vs Context | Δ vs History | Improved / worsened vs Context |
| --- | --- | ---: | ---: | ---: | ---: |
| 2022 | History | 3.8025 | +0.0061 | 0.0000 | 63 / 68 |
| 2022 | Context | 3.7964 | 0.0000 | −0.0061 | 0 / 0 |
| 2022 | α=.75 | 3.7855 | −0.0109 | −0.0169 | 76 / 55 |
| 2022 | α=.50 | 3.7873 | −0.0091 | −0.0152 | 68 / 63 |
| 2022 | α=.25 | 3.8016 | +0.0051 | −0.0009 | 64 / 67 |
| 2022 | α=.00 | 3.8283 | +0.0318 | +0.0258 | 61 / 70 |
| 2023 | History | 3.8352 | −0.0785 | 0.0000 | 73 / 60 |
| 2023 | Context | 3.9137 | 0.0000 | +0.0785 | 0 / 0 |
| 2023 | α=.75 | 3.8694 | −0.0443 | +0.0342 | 77 / 56 |
| 2023 | α=.50 | 3.8417 | −0.0719 | +0.0066 | 76 / 57 |
| 2023 | α=.25 | 3.8305 | −0.0832 | −0.0047 | 72 / 61 |
| 2023 | α=.00 | 3.8352 | −0.0784 | +0.0001 | 69 / 64 |
| 2024 | History | 3.8023 | −0.0695 | 0.0000 | 84 / 50 |
| 2024 | Context | 3.8719 | 0.0000 | +0.0695 | 0 / 0 |
| 2024 | α=.75 | 3.8580 | −0.0138 | +0.0557 | 55 / 79 |
| 2024 | α=.50 | 3.8642 | −0.0077 | +0.0618 | 51 / 83 |
| 2024 | α=.25 | 3.8898 | +0.0180 | +0.0875 | 46 / 88 |
| 2024 | α=.00 | 3.9345 | +0.0626 | +0.1321 | 41 / 93 |
| 2025 | History | 3.8548 | −0.0704 | 0.0000 | 76 / 60 |
| 2025 | Context | 3.9252 | 0.0000 | +0.0704 | 0 / 0 |
| 2025 | α=.75 | 3.8909 | −0.0343 | +0.0360 | 68 / 68 |
| 2025 | α=.50 | 3.8787 | −0.0465 | +0.0239 | 62 / 74 |
| 2025 | α=.25 | 3.8889 | −0.0363 | +0.0340 | 59 / 77 |
| 2025 | α=.00 | 3.9212 | −0.0040 | +0.0664 | 57 / 79 |
| 2023–2025 | History | 3.8309 | −0.0728 | 0.0000 | 233 / 170 |
| 2023–2025 | Context | 3.9037 | 0.0000 | +0.0728 | 0 / 0 |
| 2023–2025 | α=.75 | 3.8729 | −0.0308 | +0.0420 | 200 / 203 |
| 2023–2025 | α=.50 | 3.8617 | −0.0420 | +0.0308 | 189 / 214 |
| 2023–2025 | α=.25 | 3.8699 | −0.0337 | +0.0390 | 177 / 226 |
| 2023–2025 | α=.00 | 3.8972 | −0.0064 | +0.0664 | 167 / 236 |

## Team-level concentration and controls

The final [team-season results](../data/processed/context_positive_net_moderation/team_season_results.csv) retain every paired NLL difference and expected-rank diagnostic. At `alpha=0.75`, 2023 improves for 77/133 teams, 2024 for 55/134, 2025 for 68/136, and 2022 for 76/131. The pooled 2023–2025 median NLL gain is **−0.0009**, despite a positive mean gain of 0.0308. The top 10% of team-seasons by positive gain contribute 51.8% of gross gains. At `alpha=0.50`, 189/403 teams improve, the median gain is **−0.0117**, and the top 10% contribute 53.5% of gross gains. These are concentrated tail repairs with losses elsewhere, not a general shift toward better scores for most teams.

The 2024 wrinkle is especially clear: `alpha=0.75` produces 6.124 total positive NLL-gain units and 4.272 total worsening units. Buffalo, Navy, and Sam Houston are among the largest gains; Nevada and Florida International are among the largest losses. Its 0.0138 mean gain is real under equal weighting but coexists with a 0.0155 median loss. `alpha=0.25` and `alpha=0` worsen 2024 mean December NLL by 0.0180 and 0.0626. This rejects the idea that simply removing more of the positive Context-only term is reliably better.

The 2022 control also rejects full moderation. `alpha=0` raises December NLL by 0.0318 and nearly erases Context's preseason expected-rank advantage (2.729 to 0.336 ranks). Partial moderation at 0.75 or 0.50 improves 2022 December NLL but gives back 0.307 or 0.800 ranks of the preseason advantage. The 0.75 tradeoff is smaller, yet it remains a genuine cost in that control season.

## Conclusion and reproduction

The issue 154 association is actionable **as a constrained research signal**: partial positive-net moderation lowers mean posterior NLL consistently across the three primary seasons and their shared-evidence checkpoints. It does not establish a broadly improving correction, close the remaining gap to History, or identify a production alpha. A separate candidate-validation issue would need to preregister its rule and evaluate genuinely new seasons or select strength strictly within each training fold. This issue changes no production prior, transfer input, ranking artifact, or History baseline.

Run the dedicated [experiment script](../scripts/experiment_context_positive_net_moderation.py) from the repository root with the pinned `uv` environment and the immutable historical corpus:

```bash
uv run python scripts/experiment_context_positive_net_moderation.py \
  --targets /path/to/data/processed/modeling/team_season_rank_distributions.csv \
  --raw-games /path/to/data/raw/cfbd/games
```

The committed [summary](../data/processed/context_positive_net_moderation/summary.json) records baseline parity and concentration. [Provenance](../data/processed/context_positive_net_moderation/provenance.json) records source, model, script, and output hashes; target seasons and training origins; and the fixed grid with no learned moderation parameter. Two complete runs from the same inputs produced byte-identical copies of all four machine-readable files. Historical Context transfer features retain the prior studies' frozen retrospective reconstruction, whose archived preseason availability remains unverified. No repaired transfer evidence is used. The 2022–2025 panel motivated this experiment and is not untouched validation data.
