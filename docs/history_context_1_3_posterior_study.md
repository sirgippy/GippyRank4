# Issue 140: Does History 1.1 still add value after Context 1.3?

## Recommendation

**A — keep History 1.1 as a first-class production predictive lineage.** The primary operational comparison refits **both** Context 1.3 and History 1.1 through the preceding season. Context has lower preseason NLL in all four target seasons. At the first observed September checkpoint, History has lower posterior NLL in 2023–2025, and retains that lead at every later observed checkpoint in those seasons. Context remains slightly ahead throughout 2022. The four-season, team-weighted gap changes from a Context advantage of 0.036 nats per team in August to a History advantage of 0.010 in September, 0.023 in October, 0.044 in November, and 0.053 at the final December checkpoint. The frozen-through-2021 panel and the rolling-Context/frozen-History sensitivity panel show the same qualitative pattern. This is evidence of useful in-season value rather than a benefit confined to the postseason.

The annual-refit comparison and the conservative cutoff filter address the two interpretation problems in the previous version of this report. They do not alter any production model, likelihood, inference setting, or publication policy. Context 1.2 needs **retained-artifact reading and reconstruction support only** for its earlier publications; that compatibility need is separate from History's active status.

## Method and provenance

Seven retained rolling-backtest checkpoint dates per season are replayed for 2022–2025. The matched evaluation population has 131, 133, 134, and 136 FBS teams, respectively, with a final Massey constituent-rank target PMF. Each final target distribution is reused at every checkpoint. NLL is target-weighted cross entropy. The reported 80% coverage is target probability mass inside the forecast interval, rather than a binary hit against a presumed single true rank. CRPS and expected-rank absolute error are additional scores. The 28 checkpoints repeatedly evaluate 534 team-seasons; the 11,214 panel-specific team-checkpoint rows are not independent observations.

The three comparison panels are:

1. **Primary operational, rolling C versus rolling H:** validated Context 1.3 P3 specification and production History 1.1 specification are each fit through 2021, 2022, 2023, and 2024 for target seasons 2022, 2023, 2024, and 2025. Context uses the validated rolling fit path and passes prior-score parity against retained `rolling_metrics.csv` at 1e-8 tolerance. History uses the annual production builder's `build_history_prior`, target-season outcome-free `inference_rows`, annual cold-start fits, and `future_predictions`. Each fit excludes outcomes from its target season and later.
2. **Frozen architecture comparison:** retained Context 1.3 P3 research PMFs fitted through 2021 versus retained historical History 1.1 PMFs, also fitted through 2021, for all four seasons.
3. **Sensitivity:** rolling Context 1.3 versus frozen History 1.1. This preserves the preceding report's asymmetric comparison but does not drive the recommendation.

Every arm receives the same eligible game rows, FCS fallback population and rule, pinned Historical Likelihood V1 (SHA-256 `89eff21a304939a62beed6fd58dd642e0e4dbba698d7bdaa9aaca9f09f2de153`), and production posterior inference settings: **500 maximum iterations, 1e-9 tolerance, 0.35 damping**. All **112** inference runs converged, using at most 134 iterations. FBS team IDs match across priors and final targets. Model, feature, target, raw schedule, and coaching-source hashes are in [provenance.json](../data/processed/history_context_posterior_study/provenance.json).

### Cutoff evidence audit

Historical CFBD schedule caches show a final `completed` flag and score but **no completion timestamp**. Kickoff before a cutoff does not prove the final result was available then. The replay therefore treats a score as eligible only when scheduled kickoff preceded the UTC cutoff by **at least 48 hours**. Closer games remain indeterminate and are excluded from every arm. This conservative buffer prevents ordinary in-progress games from entering the replay; it is an availability proxy, not proof of an archival as-of score, because unusually delayed or suspended games cannot be certified from these caches. The exact cutoff-safe findings are conditional on this rule. An archived result feed with completion timestamps would be needed to certify exact historical information sets.

[historical_cutoff_audit.csv](../data/processed/history_context_posterior_study/historical_cutoff_audit.csv) records all **934** excluded game/checkpoint occurrences, with kickoff, teams, and age at cutoff. This includes games whose final cache row was previously treated as known after only minutes. At the September 19, 2024 cutoff, Appalachian State–South Alabama had kicked off **30 minutes** earlier and is excluded. At the September 19, 2025 cutoff, three games are excluded, including Oklahoma State–Tulsa at 30 minutes and Lafayette–Columbia at two hours. The September 17, 2023 cutoff excludes 114 games kicked off within the preceding 48 hours. [historical_evidence.csv](../data/processed/history_context_posterior_study/historical_evidence.csv) reports eligible and excluded counts for every checkpoint and family.

The processed historical game CSV is reconstructed from cached raw FBS/FCS schedule responses, rejecting conflicting duplicate IDs. The final-rank corpus and raw schedules are local inputs identified by hashes. Historical Context transfer inputs remain **retrospective research reconstructions**, not archived August 15 inputs; annual refitting does not remove that data-timing limitation.

## Calendar crossover and matched checkpoints

Positive Δ means History has lower posterior NLL; Δ = Context minus History, in nats per team. Dates are UTC. The table's rolling C and H NLL columns are the primary comparison. The first *observed* History-leading checkpoints remain September 17, 2023, September 19, 2024, and September 19, 2025 after both annual History refitting and cutoff filtering. The available checkpoints cannot locate a crossover between observation dates.

| Season | Checkpoint date | Eligible games | Excluded near cutoff | Frozen Δ | Rolling C/H Δ | Sensitivity Δ | Rolling C NLL | Rolling H NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | Aug 27 | 0 | 10 | -0.071 | -0.071 | -0.071 | 4.456 | 4.528 |
| 2022 | Sep 22 | 363 | 2 | -0.024 | -0.024 | -0.024 | 4.287 | 4.311 |
| 2022 | Oct 07 | 597 | 3 | -0.026 | -0.026 | -0.026 | 4.160 | 4.186 |
| 2022 | Oct 27 | 921 | 2 | -0.011 | -0.011 | -0.011 | 4.065 | 4.075 |
| 2022 | Nov 09 | 1,146 | 3 | -0.014 | -0.014 | -0.014 | 3.981 | 3.995 |
| 2022 | Nov 23 | 1,396 | 2 | -0.009 | -0.009 | -0.009 | 3.852 | 3.860 |
| 2022 | Dec 10 | 1,490 | 5 | -0.006 | -0.006 | -0.006 | 3.796 | 3.802 |
| 2023 | Aug 26 | 0 | 8 | -0.023 | -0.020 | -0.022 | 4.468 | 4.488 |
| 2023 | Sep 17 | 250 | 114 | +0.009 | +0.012 | +0.011 | 4.388 | 4.375 |
| 2023 | Oct 06 | 594 | 6 | +0.028 | +0.033 | +0.032 | 4.277 | 4.244 |
| 2023 | Oct 21 | 813 | 90 | +0.022 | +0.027 | +0.026 | 4.129 | 4.101 |
| 2023 | Nov 04 | 1,035 | 109 | +0.047 | +0.053 | +0.051 | 4.029 | 3.977 |
| 2023 | Nov 18 | 1,280 | 105 | +0.062 | +0.068 | +0.067 | 3.986 | 3.917 |
| 2023 | Dec 09 | 1,494 | 4 | +0.072 | +0.079 | +0.078 | 3.914 | 3.835 |
| 2024 | Aug 24 | 0 | 5 | -0.041 | -0.037 | -0.037 | 4.551 | 4.588 |
| 2024 | Sep 19 | 349 | 1 | +0.037 | +0.030 | +0.031 | 4.407 | 4.377 |
| 2024 | Oct 06 | 573 | 98 | +0.034 | +0.026 | +0.027 | 4.255 | 4.229 |
| 2024 | Oct 24 | 890 | 5 | +0.055 | +0.046 | +0.048 | 4.145 | 4.099 |
| 2024 | Nov 08 | 1,113 | 5 | +0.082 | +0.071 | +0.072 | 4.124 | 4.053 |
| 2024 | Nov 23 | 1,341 | 104 | +0.078 | +0.070 | +0.071 | 4.013 | 3.943 |
| 2024 | Dec 14 | 1,556 | 6 | +0.077 | +0.070 | +0.071 | 3.872 | 3.802 |
| 2025 | Aug 23 | 0 | 8 | -0.018 | -0.016 | -0.019 | 4.550 | 4.566 |
| 2025 | Sep 19 | 357 | 3 | +0.024 | +0.021 | +0.018 | 4.361 | 4.340 |
| 2025 | Oct 08 | 676 | 1 | +0.052 | +0.044 | +0.043 | 4.237 | 4.193 |
| 2025 | Oct 24 | 896 | 5 | +0.051 | +0.044 | +0.043 | 4.114 | 4.070 |
| 2025 | Nov 09 | 1,121 | 108 | +0.059 | +0.051 | +0.049 | 4.063 | 4.012 |
| 2025 | Nov 23 | 1,359 | 116 | +0.072 | +0.060 | +0.058 | 4.000 | 3.940 |
| 2025 | Dec 13 | 1,570 | 6 | +0.084 | +0.070 | +0.069 | 3.925 | 3.855 |

The complete per-checkpoint prior, posterior, CRPS, error, coverage, interval-width, rank-agreement, and transfer-group metrics are in [historical_checkpoints.csv](../data/processed/history_context_posterior_study/historical_checkpoints.csv). Paired team-level observations are in [historical_teams.csv](../data/processed/history_context_posterior_study/historical_teams.csv).

## September, October, November, and postseason behavior

These are team-weighted means across seasons. October and November each include two checkpoints per season. The December row is each season's final observed checkpoint. Coverage is target mass inside the forecast's 80% interval.

| Period | Frozen Δ NLL | Rolling C/H Δ NLL | Sensitivity Δ NLL | Rolling C / H coverage | Rolling C / H width |
|---|---:|---:|---:|---:|---:|
| August | -0.038 | -0.036 | -0.037 | .836 / .840 | 70.7 / 76.9 |
| September | +0.012 | +0.010 | +0.009 | .780 / .794 | 50.5 / 53.1 |
| October | +0.026 | +0.023 | +0.023 | .767 / .776 | 39.1 / 40.4 |
| November | +0.048 | +0.044 | +0.044 | .770 / .785 | 32.1 / 33.1 |
| December | +0.057 | +0.053 | +0.053 | .778 / .803 | 28.6 / 29.4 |

The annual-refit History prior changes the old asymmetric result by only about 0.001 NLL pooled at these depths. The September lead is smaller than the November and December leads, but it appears within a useful weekly publication window in three consecutive seasons. The 2022 exception and heterogeneous team effects prevent treating the aggregate as a universal winner.

## Prior improvement, calibration, uncertainty, and stability

At the final checkpoint, rolling Context's mean prior/posterior NLL is **4.507 / 3.877**, an improvement of 0.629. Rolling History's is **4.543 / 3.824**, an improvement of 0.719. Both gain substantially from identical game evidence. The frozen comparison ends at 3.881 / 3.824 for Context / History, and the asymmetric sensitivity panel ends at 3.877 / 3.824.

Rolling Context's 80% interval narrows from 70.7 ranks preseason to 28.6 in December; rolling History's narrows from 76.9 to 29.4. At the final checkpoint, History's slightly wider interval contains more target mass: **0.803 versus 0.778** pooled. The final Context / History coverage by season is 0.805 / 0.802 in 2022, 0.752 / 0.795 in 2023, 0.793 / 0.820 in 2024, and 0.762 / 0.795 in 2025. Final pooled CRPS is 0.0223 / 0.0194 and expected-rank absolute error is 8.23 / 7.67.

The final team-level rolling C-minus-H NLL gap has median **+0.016**, 10th/90th percentiles **-0.140 / +0.304**, and favors History for **296 of 534** team-seasons. The mean advantage includes large gains and substantial losses. Within-season expected-rank Spearman agreement is 0.930–0.947 at the first checkpoint and 0.996–0.998 at the final one. Mean absolute adjacent-checkpoint expected-rank movement falls from 12.8 to 3.3 for Context and from 14.7 to 3.5 for History between the first and last transitions. The lineages' ordering becomes similar while their probability quality and interval coverage still differ.

## Transfer availability and examples

The historical Context 1.3 coverage artifact flags whether both incoming prior-usage and defensive DB impact sums were available before imputation. This does not certify a complete real-world transfer census. Among 534 team-seasons, 488 are classified complete, 40 incomplete, and 6 have no Context coverage record and use a History fallback. The DB flag is true throughout this panel, so the incomplete split primarily reflects missing prior usage. The tiny incomplete and fallback groups limit subgroup claims.

| Group | Team-seasons | Rolling August Δ | Rolling September Δ | Rolling December Δ | December C−H coverage |
|---|---:|---:|---:|---:|---:|
| Complete | 488 | -0.035 | +0.012 | +0.056 | -0.026 |
| Incomplete | 40 | -0.066 | -0.031 | +0.019 | -0.016 |
| History fallback | 6 | +0.090 | +0.099 | +0.090 | -0.056 |

History's final advantage is larger among transfer-complete teams. Their final rolling Δ NLL by season is -0.007, +0.085, +0.068, and +0.071. Incomplete teams show approximately 0.000, +0.033, +0.138, and -0.086 on annual counts of 19, 14, 4, and 3. The last two incomplete estimates are especially unstable.

Large final rolling team-level C-minus-H NLL differences include 2023 James Madison **+1.402**, 2023 Wyoming **+1.043**, and 2025 Navy **+0.914** in History's favor. Context does better for 2022 Nevada **-0.776**, 2023 East Carolina **-0.673**, and 2025 Northern Illinois **-0.627**. The repeated aggregate History advantage is meaningful, but not uniform across teams.

## Current 2026 checkpoints: descriptive only

Six retained Context 1.3/History 1.1 weekly snapshot pairs from September 8–27 use matched effective cutoffs and identical included-game rows. They describe 138 FBS teams. There is no final 2026 rank-distribution target, so **no 2026 NLL or calibration claim** is made. Context 1.3's 2026 prior is a retrospective reconstruction using transfer inputs retrieved after the preseason cutoff. Full per-team values are in [current_2026_descriptive.csv](../data/processed/history_context_posterior_study/current_2026_descriptive.csv). Some older History snapshot metadata does not record its inference configuration; any configuration that is recorded is checked against 500 / 1e-9 / 0.35.

| Date | Included games | C/H mean 80% width | Mean absolute C/H expected-rank gap | C/H rank Spearman | Top-25 overlap |
|---|---:|---:|---:|---:|---:|
| Sep 08 | 172 | 62.3 / 66.7 | 7.25 | .971 | 24 |
| Sep 13 | 291 | 55.0 / 58.6 | 6.09 | .985 | 23 |
| Sep 19 | 295 | 54.8 / 58.2 | 6.06 | .985 | 23 |
| Sep 20 | 410 | 50.0 / 52.9 | 5.51 | .988 | 22 |
| Sep 26 | 417 | 49.6 / 52.4 | 5.42 | .989 | 22 |
| Sep 27 | 530 | 46.2 / 48.4 | 5.11 | .991 | 21 |

The two current rankings are highly correlated but still assign different uncertainty and top-25 memberships. These snapshots show descriptive divergence, not which lineage is better calibrated in 2026.

## Reproduction

With the local final-rank corpus and cached raw CFBD schedules available, run:

```bash
uv run python scripts/study_history_context_posterior.py \
  --targets /path/to/team_season_rank_distributions.csv \
  --raw-games /path/to/raw/cfbd/games
```

The script writes only `data/processed/history_context_posterior_study/`. The committed CSVs include the three paired panels, 28 checkpoint summaries, cutoff audit, evidence diagnostics, and 2026 descriptive sample. The retained historical backtest supplies checkpoint dates only. The prior parity check verifies the rolling P3 Context scores before posterior scoring.
