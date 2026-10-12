# Issue 201: pre-bowl forecast comparison results

This is an exploratory four-season comparison using the fixed issue 200 fits. No model was refit. The result does not promote a model or change production defaults.

## Primary rolling FBS–FBS game forecast

| Arm | Games | Equal-game CRPS | Equal-season CRPS | Margin NLL | Home-win Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| History existing | 2998 | 8.997410 | 8.998747 | 4.193157 | 0.182869 |
| Context existing | 2998 | 8.907925 | 8.909120 | 4.183742 | 0.180600 |
| History pre-bowl | 2998 | 8.994172 | 8.995502 | 4.192674 | 0.182881 |
| Context pre-bowl | 2998 | 8.909285 | 8.910540 | 4.183773 | 0.180752 |

Equal-game CRPS order, lower to higher: Context existing → Context pre-bowl → History pre-bowl → History existing.
The preregistered expected order was Context pre-bowl, History existing, History pre-bowl, Context existing; the observed order differs.
Classification: **inconclusive**. Recommendation: revise the comparison plan before follow-up work.

### Paired policy and model-family contrasts

| Contrast | Equal-game CRPS difference | Equal-season CRPS difference | Equal-game margin NLL difference |
| --- | ---: | ---: | ---: |
| Context − History, existing | -0.089486 | -0.089626 | -0.009415 |
| Context − History, pre-bowl | -0.084887 | -0.084962 | -0.008901 |
| Pre-bowl − existing, Context | 0.001361 | 0.001420 | 0.000031 |
| Pre-bowl − existing, History | -0.003238 | -0.003245 | -0.000483 |

Policy interaction, (Context pre-bowl − Context existing) − (History pre-bowl − History existing): 0.004599 equal-game and 0.004664 equal-season. Positive values mean the combined intervention shifts CRPS less favorably for Context.
Context pre-bowl lowered CRPS in 3/4 seasons, but did not lower it in either aggregate. Margin NLL was worse in 1 season; average 50/80/95% coverage deviation was worse in 1 season.

### Per-season rolling CRPS

| Season | History existing | Context existing | History pre-bowl | Context pre-bowl |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 9.108537 | 8.996967 | 9.103798 | 9.007685 |
| 2023 | 8.897336 | 8.844700 | 8.897339 | 8.842965 |
| 2024 | 9.213743 | 9.098935 | 9.208889 | 9.095786 |
| 2025 | 8.775371 | 8.695879 | 8.771984 | 8.695725 |

### Event-phase CRPS (rolling FBS–FBS)

| Phase | Games | History existing | Context existing | History pre-bowl | Context pre-bowl |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre thanksgiving | 2701 | 9.006583 | 8.906861 | 9.003410 | 8.908799 |
| rivalry window | 254 | 8.923135 | 8.919139 | 8.919403 | 8.917129 |
| conference championship period | 39 | 9.066836 | 9.118995 | 9.063219 | 9.103208 |
| army navy | 4 | 6.842767 | 6.855983 | 6.830823 | 6.849102 |

### Separate FBS–FCS game diagnostic

| Arm | Games | CRPS | Margin NLL | Home-win Brier |
| --- | ---: | ---: | ---: | ---: |
| History existing | 485 | 9.612918 | 4.253398 | 0.040689 |
| Context existing | 485 | 9.493454 | 4.241193 | 0.040024 |
| History pre bowl | 485 | 9.614477 | 4.253201 | 0.040688 |
| Context pre bowl | 485 | 9.494627 | 4.241092 | 0.040044 |

### Preseason-only FBS–FBS CRPS diagnostic

| Arm | Games | CRPS |
| --- | ---: | ---: |
| History existing | 2998 | 9.900009 |
| Context existing | 2998 | 9.634382 |
| History pre bowl | 2998 | 9.898014 |
| Context pre bowl | 2998 | 9.640744 |

Preseason-only forecasts are kept separate from rolling game counts. Margin NLL, median absolute error, win Brier, interval coverage, widths, and the full phase/FBS–FCS tables are in `forecast_summary.csv`.

Army–Navy games after the pre-bowl target snapshot: 4; all are separately labeled and listed in `calendar_mismatches.csv`.

## Rolling rank-target evaluation

Scores use the common FBS support against corrected #204 pre-bowl PMFs, only at cutoffs strictly before the target date with evidence strictly before it. Each arm has 8280 common team-checkpoint rows; 4 target-date exclusions were recorded.

| Arm | Ranking NLL | Rank CRPS | Expected-rank error | 80% target mass |
| --- | ---: | ---: | ---: | ---: |
| History existing | 4.172127 | 0.043839 | 13.174196 | 0.802175 |
| Context existing | 4.186379 | 0.043718 | 12.991238 | 0.789240 |
| History pre bowl | 4.174158 | 0.043914 | 13.184069 | 0.800399 |
| Context pre bowl | 4.188645 | 0.043840 | 13.008876 | 0.788023 |

These means weight each retained team-checkpoint equally. Ranking NLL/CRPS are secondary descriptions, not a second model-selection gate. `ranking_summary.csv` retains each season, cutoff, and event phase.

## Frozen #192 target sensitivity

The retained History 1.1 and repaired Context 1.4 PMFs were read-only. The common cohort keeps native cold starts and excludes target-unsafe checkpoints before comparing the original #192 final targets with corrected #204 pre-bowl targets.

| Target | Safe checkpoints | Context − History NLL | Context − History rank CRPS | Expected-rank error difference | 80% target-mass difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original final #192 | 24 | 0.012862 | -0.000251 | -0.243340 | -0.010037 |
| Corrected pre-bowl #204 | 24 | 0.013789 | -0.000141 | -0.190731 | -0.010619 |

2022 remains visible: across its six safe checkpoints, Context − History rank CRPS was -0.003239 against the original target and -0.002950 against the corrected pre-bowl target. Full season/checkpoint details are in `frozen_target_paired_summary.csv`.

Frozen PMF identity `3956b852b0bc2ab6bdc7a9cdf7555b9a3ec0f3934971d91b0464bc11c739a3ff` remained unchanged. Original-target parity compared 76896 retained metrics and passed with maximum absolute error 4.55e-13. Target-safety/support exclusions: 4; checkpoint safety details are in `frozen_checkpoint_safety.csv`.

## Exclusions and numerical diagnostics

Forecast schedule/outcome exclusions by reason:
- `no_completed_score_in_source`: 1
- `not_fbs_fbs_or_fbs_fcs`: 2626
- `not_regular_season`: 213

All 27864 forecast IDs are unique. Each stage/population uses an identical game set for all four arms; included evidence IDs and source-row fingerprints are in `replay_evidence.jsonl.gz`.
Maximum CRPS numerical error estimate: 1.99939212e-06 points (tolerance `2e-6`; quadrature orders [256, 512]; tail multiples [40.0]).
Protocol SHA-256: `ddea4e6dbf8d048c2a977d2a1f6d13f9e92c6948547afb115e0f81d54e677477`.

Four seasons and repeated team matchups are not independent confirmation. The ranking and game-score comparisons are descriptive and do not establish statistical significance. No follow-up work is dispatched by this issue.

See `manifest.json` for exact input/output hashes, forward-only fit origins, support counts, schedule plan, evidence hashes, and provenance.
