# Issue 140: Does History 1.1 still add value after Context 1.3?

## Recommendation

**A — keep History 1.1 as a first-class production predictive lineage.** In a
matched four-season replay, Context 1.3 is the better preseason and first
checkpoint forecast, but History 1.1 has better late posterior NLL, CRPS,
expected-rank error, and 80% interval coverage in three of four seasons. The
2023–2025 final-checkpoint NLL gaps (C minus H) are 0.072, 0.078, and 0.085
nats per team. History is therefore a useful independently maintained
comparison, even though the two rankings become very similar in order. This
study makes **no** publication or model change.

Context 1.2 needs **artifact-reading and reconstruction support only** for its
retained 2026 preseason and early weekly publications. This is a separate
compatibility question; the evidence here does not justify active Context 1.2
generation after Context 1.3 activation.

## Method and inputs

The study replays the seven actual-date checkpoints from the existing rolling
posterior backtest for each of 2022–2025. The population is all matched FBS
teams with a frozen final Massey constituent-rank PMF: 131, 133, 134, and 136
teams respectively. Each team receives the same target at every checkpoint.
The target is a distribution, so NLL is target-weighted cross entropy and 80%
coverage is target probability mass inside the forecast interval. It is not a
single assumed true rank. CRPS and expected-rank absolute error supplement
NLL. The seven checkpoints are season depths, not fixed calendar weeks.

- **C prior:** Context 1.3 P3 historical research PMFs from
  `data/processed/preseason/context_v1_3_candidate/predictions.csv`. Their
  model was frozen through 2021 and applied to 2022–2025, with H-based
  cold-start fallback rows. They are the validated historical C1.3
  specification, not a newly fitted model for this study.
- **H prior:** frozen History 1.1 PMFs from
  `data/processed/preseason/history/predictions.csv`.
- **Shared update:** pinned Historical Likelihood V1, SHA-256
  `89eff21a304939a62beed6fd58dd642e0e4dbba698d7bdaa9aaca9f09f2de153`,
  and the existing deterministic damped loopy sum-product inference with 100
  maximum iterations, 1e-6 tolerance, and 0.35 damping. All 56 runs converged.
  Each C/H checkpoint uses the same completed games, cutoff, FCS population,
  uniform FCS fallback rule, and FBS team IDs. Neither prior, likelihood, nor
  inference semantics were changed.

The processed historical CFBD game CSV is not checked into this worktree, so
the study reconstructs its game rows from the cached raw FBS/FCS schedule
responses, rejecting conflicting duplicate game IDs. The final-rank target
corpus is likewise supplied as an external processed input. Input and
reconstructed-corpus hashes are in
[`provenance.json`](../data/processed/history_context_posterior_study/provenance.json).
The independent History replay differs from the older retained backtest by at
most 0.00155 NLL (2024 final), consistent with a small processed-corpus or
implementation difference. Its difference is recorded at every checkpoint in
[`historical_evidence.csv`](../data/processed/history_context_posterior_study/historical_evidence.csv),
and every C/H result here is a fresh, matched replay rather than a comparison
against those retained History numbers.

Historical 2021–2025 Context transfer values are retrospective research
reconstructions, not archived August 15 snapshots. These results evaluate the
frozen C1.3 historical artifact's predictive behavior, subject to that data
timing limitation. They do not imply those transfer features were available
to a live operator in those years. The four season outcomes are also related;
28 checkpoints and 3,738 team-checkpoint rows are repeated observations, not
3,738 independent trials.

## Matched historical checkpoints

The table gives posterior NLL and target-mass coverage; lower NLL is better.
`Δ` is C minus H in nats per team. C/H 80% interval widths are in rank slots.
The full prior, posterior, CRPS, error, coverage, width, and ranking-agreement
metrics for every checkpoint and transfer group are in
[`historical_checkpoints.csv`](../data/processed/history_context_posterior_study/historical_checkpoints.csv).

| Season | Depth | Cutoff (UTC) | C NLL | H NLL | Δ | C cover | H cover | C/H width |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 2022 | 1 | Aug 27 | 4.448 | 4.519 | -0.071 | .834 | .846 | 69.9 / 74.7 |
| 2022 | 2 | Sep 22 | 4.282 | 4.311 | -0.029 | .800 | .797 | 48.5 / 50.3 |
| 2022 | 3 | Oct 07 | 4.158 | 4.183 | -0.026 | .792 | .783 | 42.4 / 43.6 |
| 2022 | 4 | Oct 27 | 4.068 | 4.078 | -0.010 | .776 | .770 | 36.1 / 36.9 |
| 2022 | 5 | Nov 09 | 3.980 | 3.994 | -0.014 | .769 | .767 | 33.0 / 33.6 |
| 2022 | 6 | Nov 23 | 3.852 | 3.861 | -0.009 | .805 | .799 | 30.2 / 30.8 |
| 2022 | 7 | Dec 10 | 3.797 | 3.801 | -0.004 | .805 | .801 | 29.0 / 29.5 |
| 2023 | 1 | Aug 26 | 4.475 | 4.493 | -0.018 | .864 | .865 | 70.2 / 76.0 |
| 2023 | 2 | Sep 17 | 4.337 | 4.306 | +0.032 | .780 | .807 | 49.3 / 51.4 |
| 2023 | 3 | Oct 06 | 4.274 | 4.244 | +0.030 | .737 | .756 | 42.0 / 43.4 |
| 2023 | 4 | Oct 21 | 4.077 | 4.049 | +0.028 | .771 | .787 | 36.6 / 37.5 |
| 2023 | 5 | Nov 04 | 4.005 | 3.945 | +0.060 | .775 | .798 | 33.2 / 34.0 |
| 2023 | 6 | Nov 18 | 3.929 | 3.873 | +0.057 | .779 | .804 | 30.5 / 31.3 |
| 2023 | 7 | Dec 09 | 3.908 | 3.836 | +0.072 | .753 | .794 | 28.6 / 29.3 |
| 2024 | 1 | Aug 24 | 4.548 | 4.585 | -0.036 | .810 | .803 | 68.9 / 76.6 |
| 2024 | 2 | Sep 19 | 4.410 | 4.375 | +0.035 | .753 | .765 | 50.2 / 53.4 |
| 2024 | 3 | Oct 06 | 4.232 | 4.193 | +0.039 | .764 | .763 | 40.2 / 42.2 |
| 2024 | 4 | Oct 24 | 4.171 | 4.113 | +0.058 | .769 | .776 | 35.9 / 37.4 |
| 2024 | 5 | Nov 08 | 4.116 | 4.037 | +0.080 | .756 | .775 | 32.9 / 34.3 |
| 2024 | 6 | Nov 23 | 3.981 | 3.897 | +0.083 | .773 | .798 | 29.9 / 31.1 |
| 2024 | 7 | Dec 14 | 3.876 | 3.798 | +0.078 | .792 | .821 | 28.5 / 29.6 |
| 2025 | 1 | Aug 23 | 4.547 | 4.564 | -0.017 | .829 | .841 | 71.2 / 78.0 |
| 2025 | 2 | Sep 19 | 4.364 | 4.340 | +0.024 | .774 | .805 | 48.7 / 51.6 |
| 2025 | 3 | Oct 08 | 4.247 | 4.195 | +0.052 | .750 | .768 | 39.4 / 41.1 |
| 2025 | 4 | Oct 24 | 4.117 | 4.067 | +0.050 | .766 | .786 | 35.7 / 37.0 |
| 2025 | 5 | Nov 09 | 4.037 | 3.967 | +0.070 | .757 | .786 | 32.1 / 33.2 |
| 2025 | 6 | Nov 23 | 4.022 | 3.947 | +0.075 | .746 | .776 | 29.4 / 30.3 |
| 2025 | 7 | Dec 13 | 3.945 | 3.860 | +0.085 | .762 | .793 | 28.1 / 29.1 |

## Prior improvement, calibration, and uncertainty

Both posteriors improve materially over their own prior. At the final depth,
the team-weighted aggregate C/H prior NLL is 4.506/4.544; posterior NLL is
3.882/3.824. Thus C improves 0.624 and H improves 0.720 nats per team.
Average 80% interval width contracts from 70.6 to 28.6 ranks for C and from
77.0 to 29.4 for H. The H posterior is still slightly wider, yet has better
final aggregate coverage (0.802 versus C's 0.778). Across the four final
season checkpoints, H's coverage is 0.801, 0.794, 0.821, 0.793; C's is
0.805, 0.753, 0.792, 0.762. This is a recurring C undercoverage pattern in
2023–2025, not only a ranking-order difference.

At final depth, H also has lower aggregate CRPS (0.0194 versus 0.0225) and
expected-rank absolute error (7.66 versus 8.29). The season exception is 2022:
C has slightly better final NLL, CRPS, and expected-rank error. Thus H does
not dominate every metric or year. Early C is materially sharper and better
in NLL: aggregate first-depth C minus H NLL is -0.035. At the middle fourth
checkpoint it is +0.032, and at final depth +0.058. The median team-level
final difference is only +0.018, with 10th/90th percentiles -0.136/+0.324;
H has lower final NLL for 302 of 534 teams. The mean reflects some sizable
individual H gains as well as a broad shift.

The two posterior expected-rank orders converge during each season: within-year
C/H Spearman correlations are 0.93–0.94 at depth 1 and 0.996–0.998 at depth
7. Across years, adjacent-checkpoint mean absolute expected-rank movement
falls from 13.2 to 2.8 for C and 15.1 to 2.9 for H. H moves more early, but
both become stable late. That high rank agreement coexists with meaningful
distributional differences in NLL and coverage.

## Transfer availability and notable disagreements

Transfer completeness is determined from the frozen historical C1.3 coverage
artifact **before imputation**: both incoming prior-usage and defensive DB
impact sums must be marked available. It does not assert that every actual
transfer was identified. Across 534 team-seasons there are 488 complete, 40
incomplete, and 6 History fallback rows with no contextual coverage record.
The DB-sum availability flag is true throughout this four-season target panel;
the incomplete group therefore primarily reflects missing prior usage. These
are 40 distinct team-season observations repeated across depths, not 280
independent teams.

| Group | Team-seasons | First Δ NLL | Middle Δ NLL | Final Δ NLL | Final C−H coverage |
|---|---:|---:|---:|---:|---:|
| Complete | 488 | -0.034 | +0.033 | +0.062 | -0.025 |
| Incomplete | 40 | -0.069 | +0.012 | +0.016 | -0.013 |
| History fallback | 6 | +0.037 | +0.049 | +0.052 | -0.031 |

The final H advantage is **larger among transfer-complete teams**, the opposite
of the proposed missing-transfer concentration. Year by year, final C−H NLL
among complete teams is -0.005, +0.079, +0.078, +0.088 in 2022–2025. The
incomplete estimates are +0.001, +0.025, +0.148, -0.098 on counts of 19,
14, 4, and 3, so their late variation is too noisy for a firm subgroup
conclusion. The fallback group is too small to support an inference.

Examples of large final team-level differences in target-weighted NLL (C−H)
from [`historical_teams.csv`](../data/processed/history_context_posterior_study/historical_teams.csv):

- H advantage: 2023 James Madison +1.368, 2025 Navy +1.235, 2024 Buffalo
  +0.838. All three are transfer-complete.
- C advantage: 2022 Nevada -0.778 and 2023 East Carolina -0.608 are
  transfer-complete; 2025 Northern Illinois -0.695 is incomplete.

These examples illustrate heterogeneity. The multi-season late aggregate and
coverage pattern is more persuasive than any one team's loss.

## Current 2026 checkpoints: descriptive only

Six retained C1.3/H1.1 weekly snapshot pairs from September 8–27 use matched
effective cutoff and identical included-game rows. The table describes the
138 FBS teams; there is no final 2026 rank-distribution target, so **no 2026
NLL or calibration claim** is made. Context 1.3's 2026 prior is explicitly a
retrospective reconstruction using transfer inputs retrieved after the
preseason cutoff. Full per-team values are in
[`current_2026_descriptive.csv`](../data/processed/history_context_posterior_study/current_2026_descriptive.csv).

| Date | Included games | C/H mean 80% width | Mean absolute C/H expected-rank gap | C/H rank Spearman | Top-25 overlap |
|---|---:|---:|---:|---:|---:|
| Sep 08 | 172 | 62.3 / 66.7 | 7.25 | .971 | 24 |
| Sep 13 | 291 | 55.0 / 58.6 | 6.09 | .985 | 23 |
| Sep 19 | 295 | 54.8 / 58.2 | 6.06 | .985 | 23 |
| Sep 20 | 410 | 50.0 / 52.9 | 5.51 | .988 | 22 |
| Sep 26 | 417 | 49.6 / 52.4 | 5.42 | .989 | 22 |
| Sep 27 | 530 | 46.2 / 48.4 | 5.11 | .991 | 21 |

The 2026 rankings are already highly correlated, while the two lineages still
assign different uncertainty and top-25 memberships. The snapshot comparison
supports keeping both available during the incomplete season; it cannot decide
which is better calibrated.

## Reproduction

With the historical final-rank corpus and cached raw CFBD game responses
available locally, run:

```bash
uv run python scripts/study_history_context_posterior.py \
  --targets /path/to/team_season_rank_distributions.csv \
  --raw-games /path/to/raw/cfbd/games
```

The script writes only `data/processed/history_context_posterior_study/`.
The checked-in CSVs retain the team-level observations, all checkpoint and
transfer-group summaries, evidence diagnostics, and the 2026 descriptive
sample. The retained historical backtest was used for dates and a replay
diagnostic, not as the H arm of the comparison.
