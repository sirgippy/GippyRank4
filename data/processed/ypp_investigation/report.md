# YPP conditional-likelihood investigation

## Question and conclusion

This research asks whether yards per play adds latent team-quality information beyond score margin, opponent quality, and site in Historical Likelihood V1. Applying the unchanged predeclared gate to the corrected supported-pairing candidate gives Recommendation **B**.

The production V1 margin factor is frozen. The corrected candidate is `Y2-supported`: `p(margin | ranks, site) × p(YPP | margin, ranks, site)` only for historically supported FBS–FBS and FBS–FCS pairings. FCS–FCS uses the exact V1 margin factor because its YPP support is absent before the final test. The independent product remains a separate double-counting diagnostic and is not a promotion candidate.

## Temporal split and prospective support policy

The temporal experiment is unchanged: training is 2004-2017 (2003 has no usable YPP); development is 2018-2021; the final test is 2022-2025 untouched until candidate fitting, df selection, and promotion selection were complete. 2022–2025 remained untouched for candidate fitting, degrees-of-freedom selection, and promotion selection.

| Pairing | Candidate fitting | Inference YPP factor |
|:---|:---|:---|
| FBS–FBS | enabled | conditional YPP when usable |
| FBS–FCS | enabled | conditional YPP when usable |
| FCS–FCS | unsupported; excluded from corrected fits | exactly 1; V1 margin only regardless of YPP |

This is a prospective support boundary, not a post-hoc performance adjustment. `Y2-all-pairings-original` is retained only as the explicitly labelled invalid diagnostic that reproduces the first PR formulation. It is not used for promotion. For supported pairings, missing or unusable YPP also gives exactly the V1 margin factor.

## Data audit

The research uses the frozen historical modeling corpus for 2003–2025; 2003 has no usable team-game YPP, so the defensible common-data fit starts in 2004. The primary final-rank evaluation is for FBS teams using strict common FBS keys.

CFBD raw `/games/teams` responses contain no direct `yardsPerPlay`, `plays`, `offensivePlays`, or `totalPlays` category in the cached corpus. The derivation remains `plays = rushingAttempts + pass attempts parsed from completionAttempts`, then `totalYards / plays`. On 42398 raw/processed common team-game rows, derivation mismatches were 0.

The raw audit found 4280 exact duplicate team-game rows from overlapping classification/week queries and 0 conflicting duplicates. The schedule audit found 2379 exact overlaps and 0 conflicts.

Usable YPP ranges from -0.868 to 13.943; negative values=1, zero values=48, values below 1=69, values above 12=26. Coverage changes materially over time: True.

Representative raw payloads contain the expected `rushingAttempts`, `completionAttempts`, `totalYards`, and `sacks` fields. No direct YPP/plays value was available to compare. In the stored 2024 Tennessee–Arkansas example, CFBD gives 36 rush attempts + 29 pass attempts = 65 plays for Tennessee and 44 + 30 = 74 for Arkansas; the official team notes report total offensive plays of 65 and 74 while listing sacks separately. This supports the current NCAA-style attempt semantics for this audit, without changing the corpus derivation. Because CFBD did not expose a source-wide official-play field in these payloads, this one official cross-check cannot establish a season-wide sacks mismatch rate; that remains a follow-up data-quality check if a direct play field becomes available.

The current corpus builder has an operational sidecar hazard: a broad `game_stats/*.json` glob will see `.provenance.json` objects if a refreshed corpus is present. The audit excluded sidecars and did not rewrite or silently correct the stored corpus. This does not create a YPP-value mismatch in the common rows above, but it should be fixed in a separate acquisition-maintenance change.

CFBD API schema reference: https://apinext.collegefootballdata.com/api/games. Official attempt/play cross-check: https://utsports.com/documents/download/2024/11/4/G9_UT_Notes_MSU.pdf.

### Pairing-specific coverage by era

The aggregate coverage series is misleading for this question because it is dominated by the much larger FCS–FCS schedule. The pairing-specific audit is the relevant support check:

| Period | Pairing | Games | Both YPP | Game coverage | Team-row coverage |
|:---|:---|---:|---:|---:|---:|
| development_2018_2021 | fbs-fbs | 2850 | 2849 | 100.0% | 100.0% |
| development_2018_2021 | fbs-fcs | 379 | 377 | 99.5% | 99.5% |
| development_2018_2021 | fcs-fcs | 2171 | 0 | 0.0% | 0.0% |
| final_2022_2025 | fbs-fbs | 3174 | 3173 | 100.0% | 100.0% |
| final_2022_2025 | fbs-fcs | 485 | 485 | 100.0% | 100.0% |
| final_2022_2025 | fcs-fcs | 2663 | 2619 | 98.3% | 98.3% |
| training_2004_2017 | fbs-fbs | 10147 | 10112 | 99.7% | 99.7% |
| training_2004_2017 | fbs-fcs | 1285 | 1257 | 97.8% | 98.1% |
| training_2004_2017 | fcs-fcs | 8096 | 7 | 0.1% | 0.1% |

FBS–FBS YPP is essentially complete throughout most of 2004–2021, and FBS–FCS is generally near-complete. FCS–FCS is essentially absent through 2021, then becomes approximately 98–99% covered in 2022–2025. The appropriate conclusion is that FCS–FCS YPP is unsupported by the training/development data for this experiment; the post-2022 FCS–FCS relationship is not treated as a validated negative or unstable YPP effect.


## Conditional signal

Y1 uses only the fixed margin/site/pairing basis; Y2 adds the predeclared rank-percentile contrast basis. Student-t degrees of freedom were selected on 2018–2021 equal-game marginalized YPP NLL only, separately for the all-pairings diagnostic and the supported-pairing panel. Selected values: `{"all_pairings_original": {"naive": {"development_marginalized_nll": 1.9718990352633612, "development_seasons": "2018-2021", "final_fit_game_count": 14583, "final_fit_pseudo_observation_count": 1425395, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs", "fcs-fcs"], "include_margin": false, "model_panel": "all_pairings_original", "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}, "y1": {"development_marginalized_nll": 1.6854220894622056, "development_seasons": "2018-2021", "final_fit_game_count": 14583, "final_fit_pseudo_observation_count": 1425395, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs", "fcs-fcs"], "include_margin": true, "model_panel": "all_pairings_original", "rank_signal": false, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}, "y2": {"development_marginalized_nll": 1.6824966624833098, "development_seasons": "2018-2021", "final_fit_game_count": 14583, "final_fit_pseudo_observation_count": 1425395, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs", "fcs-fcs"], "include_margin": true, "model_panel": "all_pairings_original", "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}}, "supported_pairings": {"naive": {"development_marginalized_nll": 1.9718912560301372, "development_seasons": "2018-2021", "final_fit_game_count": 14576, "final_fit_pseudo_observation_count": 1425080, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs"], "include_margin": false, "model_panel": "supported_pairings", "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}, "y1": {"development_marginalized_nll": 1.68540460923251, "development_seasons": "2018-2021", "final_fit_game_count": 14576, "final_fit_pseudo_observation_count": 1425080, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs"], "include_margin": true, "model_panel": "supported_pairings", "rank_signal": false, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}, "y2": {"development_marginalized_nll": 1.6824731451856763, "development_seasons": "2018-2021", "final_fit_game_count": 14576, "final_fit_pseudo_observation_count": 1425080, "final_fit_seasons": "2004-2021", "fit_pairings": ["fbs-fbs", "fbs-fcs"], "include_margin": true, "model_panel": "supported_pairings", "rank_signal": true, "selection_rule": "minimum development equal-game marginalized YPP NLL; ties choose lower df", "student_t_df": 15.0, "training_seasons": "2004-2017"}}}`.

A. Direct residual signal. The residual is observed YPP differential minus the supported-pairing Y1 conditional mean. Positive residual means the V1-oriented side produced more YPP than its margin/site/pairing relationship predicted. This table includes only FBS–FBS and FBS–FCS rows; FCS–FCS rows are not mixed into the direct-signal conclusion. `conditional_signal.csv` reports margin and residual bins; `temporal_stability.csv` reports season/pairing effects.

| Period | Season | Pairing | Site | N | Quality slope | Quality Spearman | Next-game slope |
|:---|---:|:---|:---|---:|---:|---:|---:|
| development_2018_2021 | 2018 | fbs-fbs | home_site | 710 | 0.026 | 0.103 | 0.182 |
| development_2018_2021 | 2018 | fbs-fbs | neutral | 61 | 0.027 | 0.200 | -6.209 |
| development_2018_2021 | 2018 | fbs-fcs | home_site | 111 | 0.036 | 0.107 | 0.919 |
| development_2018_2021 | 2018 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| development_2018_2021 | 2019 | fbs-fbs | home_site | 715 | 0.052 | 0.168 | 3.164 |
| development_2018_2021 | 2019 | fbs-fbs | neutral | 59 | 0.002 | 0.007 | -4.407 |
| development_2018_2021 | 2019 | fbs-fcs | home_site | 114 | 0.037 | 0.129 | 2.586 |
| development_2018_2021 | 2020 | fbs-fbs | home_site | 499 | 0.038 | 0.137 | 0.694 |
| development_2018_2021 | 2020 | fbs-fbs | neutral | 35 | 0.023 | 0.108 | 1.024 |
| development_2018_2021 | 2020 | fbs-fcs | home_site | 34 | 0.037 | 0.052 | 4.918 |
| development_2018_2021 | 2021 | fbs-fbs | home_site | 712 | 0.052 | 0.163 | 3.028 |
| development_2018_2021 | 2021 | fbs-fbs | neutral | 58 | 0.017 | 0.052 | -0.557 |
| development_2018_2021 | 2021 | fbs-fcs | home_site | 116 | 0.012 | 0.026 | 2.230 |
| development_2018_2021 | 2021 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2022 | fbs-fbs | home_site | 714 | 0.031 | 0.105 | 1.873 |
| final_2022_2025 | 2022 | fbs-fbs | neutral | 61 | 0.014 | 0.091 | 5.172 |
| final_2022_2025 | 2022 | fbs-fcs | home_site | 119 | 0.045 | 0.146 | 3.655 |
| final_2022_2025 | 2022 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2023 | fbs-fbs | home_site | 730 | 0.049 | 0.162 | 2.452 |
| final_2022_2025 | 2023 | fbs-fbs | neutral | 62 | 0.033 | 0.045 | 1.062 |
| final_2022_2025 | 2023 | fbs-fcs | home_site | 117 | 0.032 | 0.139 | -0.044 |
| final_2022_2025 | 2023 | fbs-fcs | neutral | 1 | n/a | n/a | n/a |
| final_2022_2025 | 2024 | fbs-fbs | home_site | 722 | 0.045 | 0.145 | 1.553 |
| final_2022_2025 | 2024 | fbs-fbs | neutral | 76 | 0.043 | 0.246 | 0.563 |
| final_2022_2025 | 2024 | fbs-fcs | home_site | 117 | 0.047 | 0.141 | 0.306 |
| final_2022_2025 | 2024 | fbs-fcs | neutral | 4 | -0.472 | -0.200 | -31.204 |
| final_2022_2025 | 2025 | fbs-fbs | home_site | 744 | 0.048 | 0.174 | 1.369 |
| final_2022_2025 | 2025 | fbs-fbs | neutral | 64 | 0.020 | 0.090 | 4.611 |
| final_2022_2025 | 2025 | fbs-fcs | home_site | 126 | 0.041 | 0.153 | 0.220 |
| training_2004_2017 | 2004 | fbs-fbs | home_site | 614 | 0.017 | 0.047 | -0.727 |
| training_2004_2017 | 2004 | fbs-fbs | neutral | 29 | 0.009 | 0.175 | 6.538 |
| training_2004_2017 | 2004 | fbs-fcs | home_site | 34 | 0.024 | 0.178 | 3.389 |
| training_2004_2017 | 2004 | fbs-fcs | neutral | 9 | 0.153 | 0.650 | 6.527 |
| training_2004_2017 | 2005 | fbs-fbs | home_site | 626 | 0.015 | 0.013 | 2.993 |
| training_2004_2017 | 2005 | fbs-fbs | neutral | 28 | 0.010 | -0.014 | 5.353 |
| training_2004_2017 | 2005 | fbs-fcs | home_site | 48 | 0.034 | 0.110 | 4.856 |
| training_2004_2017 | 2005 | fbs-fcs | neutral | 6 | -0.151 | -0.771 | -17.570 |
| training_2004_2017 | 2006 | fbs-fbs | home_site | 678 | 0.027 | 0.099 | 2.557 |
| training_2004_2017 | 2006 | fbs-fbs | neutral | 32 | 0.017 | 0.039 | 4.102 |
| training_2004_2017 | 2006 | fbs-fcs | home_site | 70 | -0.005 | -0.068 | 1.823 |
| training_2004_2017 | 2006 | fbs-fcs | neutral | 7 | -0.057 | -0.179 | 5.496 |
| training_2004_2017 | 2007 | fbs-fbs | home_site | 677 | 0.008 | 0.020 | 1.966 |
| training_2004_2017 | 2007 | fbs-fbs | neutral | 32 | 0.001 | -0.057 | -8.058 |
| training_2004_2017 | 2007 | fbs-fcs | home_site | 61 | -0.088 | -0.251 | -2.317 |
| training_2004_2017 | 2007 | fbs-fcs | neutral | 11 | -0.067 | -0.282 | -0.697 |
| training_2004_2017 | 2008 | fbs-fbs | home_site | 670 | 0.012 | 0.050 | 1.991 |
| training_2004_2017 | 2008 | fbs-fbs | neutral | 47 | 0.009 | 0.058 | -2.329 |
| training_2004_2017 | 2008 | fbs-fcs | home_site | 77 | 0.071 | 0.272 | 6.746 |
| training_2004_2017 | 2008 | fbs-fcs | neutral | 7 | 0.015 | 0.000 | 6.417 |
| training_2004_2017 | 2009 | fbs-fbs | home_site | 667 | 0.032 | 0.094 | 0.648 |
| training_2004_2017 | 2009 | fbs-fbs | neutral | 47 | -0.005 | -0.077 | 1.157 |
| training_2004_2017 | 2009 | fbs-fcs | home_site | 84 | 0.020 | 0.076 | 2.664 |
| training_2004_2017 | 2009 | fbs-fcs | neutral | 10 | -0.203 | -0.309 | -5.509 |
| training_2004_2017 | 2010 | fbs-fbs | home_site | 662 | 0.035 | 0.106 | 2.154 |
| training_2004_2017 | 2010 | fbs-fbs | neutral | 56 | 0.025 | 0.060 | 2.349 |
| training_2004_2017 | 2010 | fbs-fcs | home_site | 83 | 0.024 | 0.025 | 2.269 |
| training_2004_2017 | 2010 | fbs-fcs | neutral | 7 | 0.154 | 0.571 | 31.837 |
| training_2004_2017 | 2011 | fbs-fbs | home_site | 654 | 0.041 | 0.139 | 2.630 |
| training_2004_2017 | 2011 | fbs-fbs | neutral | 54 | -0.019 | -0.168 | 4.426 |
| training_2004_2017 | 2011 | fbs-fcs | home_site | 88 | 0.059 | 0.240 | 2.684 |
| training_2004_2017 | 2011 | fbs-fcs | neutral | 9 | -0.172 | -0.283 | -4.010 |
| training_2004_2017 | 2012 | fbs-fbs | home_site | 679 | 0.026 | 0.072 | 1.164 |
| training_2004_2017 | 2012 | fbs-fbs | neutral | 53 | 0.015 | 0.129 | -0.990 |
| training_2004_2017 | 2012 | fbs-fcs | home_site | 97 | 0.049 | 0.167 | 3.085 |
| training_2004_2017 | 2012 | fbs-fcs | neutral | 6 | -0.295 | -0.371 | -38.608 |
| training_2004_2017 | 2013 | fbs-fbs | home_site | 685 | 0.039 | 0.123 | 2.576 |
| training_2004_2017 | 2013 | fbs-fbs | neutral | 52 | 0.003 | 0.078 | 12.702 |
| training_2004_2017 | 2013 | fbs-fcs | home_site | 101 | 0.048 | 0.165 | 0.101 |
| training_2004_2017 | 2013 | fbs-fcs | neutral | 10 | 0.017 | -0.091 | 4.867 |
| training_2004_2017 | 2014 | fbs-fbs | home_site | 699 | 0.024 | 0.085 | 1.484 |
| training_2004_2017 | 2014 | fbs-fbs | neutral | 61 | -0.027 | -0.144 | 2.665 |
| training_2004_2017 | 2014 | fbs-fcs | home_site | 100 | 0.070 | 0.251 | 3.801 |
| training_2004_2017 | 2014 | fbs-fcs | neutral | 7 | 0.061 | -0.071 | 8.599 |
| training_2004_2017 | 2015 | fbs-fbs | home_site | 704 | 0.034 | 0.108 | 1.177 |
| training_2004_2017 | 2015 | fbs-fbs | neutral | 61 | 0.027 | 0.113 | -2.925 |
| training_2004_2017 | 2015 | fbs-fcs | home_site | 98 | 0.045 | 0.154 | 0.492 |
| training_2004_2017 | 2015 | fbs-fcs | neutral | 7 | -0.070 | -0.286 | -2.132 |
| training_2004_2017 | 2016 | fbs-fbs | home_site | 693 | 0.017 | 0.056 | 0.926 |
| training_2004_2017 | 2016 | fbs-fbs | neutral | 66 | -0.002 | -0.009 | 0.508 |
| training_2004_2017 | 2016 | fbs-fcs | home_site | 111 | 0.036 | 0.104 | 1.394 |
| training_2004_2017 | 2016 | fbs-fcs | neutral | 2 | n/a | n/a | n/a |
| training_2004_2017 | 2017 | fbs-fbs | home_site | 719 | 0.036 | 0.109 | 2.058 |
| training_2004_2017 | 2017 | fbs-fbs | neutral | 57 | 0.015 | 0.013 | 2.099 |
| training_2004_2017 | 2017 | fbs-fcs | home_site | 92 | 0.059 | 0.207 | 7.894 |
| training_2004_2017 | 2017 | fbs-fcs | neutral | 6 | 0.004 | -0.143 | -6.712 |

The descriptive relationship is the direct residual evidence: inspect FBS–FBS and FBS–FCS slopes and correlations rather than treating the extra-variable YPP NLL as comparable with margin-only NLL. A positive quality slope means higher conditional YPP residual was associated with better eventual oriented rank percentile. This direct question is separate from the end-to-end posterior question below; a direct signal can exist while the connected posterior does not improve.

### Candidate fitting and selection

The corrected `Y2-supported` and the retained `Y2-all-pairings-original` diagnostic use the already-declared Y1/Y2 formulations, feature basis, Student-t grid, and development selection rule. Only the corrected panel is eligible for promotion. The final test is never used for candidate or df selection.

| Panel | Candidate | Fit pairings | Selected df | Development marginalized NLL | Final fit seasons |
|:---|:---|:---|---:|---:|:---|
| all_pairings_original | y1 | fbs-fbs, fbs-fcs, fcs-fcs | 15.0 | 1.685 | 2004-2021 |
| all_pairings_original | y2 | fbs-fbs, fbs-fcs, fcs-fcs | 15.0 | 1.682 | 2004-2021 |
| supported_pairings | y1 | fbs-fbs, fbs-fcs | 15.0 | 1.685 | 2004-2021 |
| supported_pairings | y2 | fbs-fbs, fbs-fcs | 15.0 | 1.682 | 2004-2021 |
Y2 selected df was 15.0 for the original panel and 15.0 for the corrected panel; the selected formulation/df therefore did not change. The supported development NLL differs slightly because the corrected fit removes the unsupported FCS–FCS training observations.

## Candidate selection and evaluation

The primary evaluation is FBS final-rank quality on the untouched 2022–2025 test period. Context and History priors, cutoffs, rank supports, final-rank targets, and strict comparison keys are identical across candidates. `candidate_metrics.csv` retains every matched cutoff row.

| Prior | Candidate | View | NLL | CRPS | Expected-rank MAE | Median-rank MAE | 80% coverage | 80% width | Top-5 Brier | Top-10 Brier | Top-25 Brier |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context | V1 | full | 3.928 | 0.025 | 8.688 | 9.098 | 0.762 | 28.6 | 0.007 | 0.009 | 0.020 |
| history | V1 | full | 3.824 | 0.019 | 7.655 | 7.919 | 0.802 | 29.4 | 0.007 | 0.009 | 0.019 |
| context | Y1 | full | 3.928 | 0.025 | 8.688 | 9.098 | 0.762 | 28.6 | 0.007 | 0.009 | 0.020 |
| history | Y1 | full | 3.824 | 0.019 | 7.655 | 7.919 | 0.802 | 29.4 | 0.007 | 0.009 | 0.019 |
| context | Y2-all-pairings-original | full | 4.024 | 0.029 | 9.501 | 9.955 | 0.730 | 27.9 | 0.007 | 0.009 | 0.021 |
| history | Y2-all-pairings-original | full | 3.886 | 0.023 | 8.224 | 8.583 | 0.777 | 28.6 | 0.006 | 0.009 | 0.020 |
| context | Y2-supported | full | 3.948 | 0.025 | 8.778 | 9.197 | 0.752 | 28.1 | 0.007 | 0.009 | 0.019 |
| history | Y2-supported | full | 3.859 | 0.021 | 8.032 | 8.321 | 0.786 | 28.7 | 0.007 | 0.009 | 0.020 |
| context | naive-independence diagnostic | full | 4.196 | 0.029 | 9.311 | 9.610 | 0.661 | 23.0 | 0.007 | 0.010 | 0.022 |
| history | naive-independence diagnostic | full | 4.106 | 0.027 | 8.935 | 9.209 | 0.680 | 23.4 | 0.007 | 0.010 | 0.023 |
| context | V1 | ypp_observed | 3.929 | 0.025 | 8.689 | 9.101 | 0.762 | 28.6 | 0.007 | 0.009 | 0.020 |
| history | V1 | ypp_observed | 3.824 | 0.019 | 7.655 | 7.921 | 0.802 | 29.4 | 0.007 | 0.009 | 0.019 |
| context | Y1 | ypp_observed | 3.929 | 0.025 | 8.689 | 9.101 | 0.762 | 28.6 | 0.007 | 0.009 | 0.020 |
| history | Y1 | ypp_observed | 3.824 | 0.019 | 7.655 | 7.921 | 0.802 | 29.4 | 0.007 | 0.009 | 0.019 |
| context | Y2-all-pairings-original | ypp_observed | 4.024 | 0.029 | 9.502 | 9.954 | 0.729 | 27.9 | 0.007 | 0.009 | 0.021 |
| history | Y2-all-pairings-original | ypp_observed | 3.886 | 0.023 | 8.224 | 8.579 | 0.777 | 28.6 | 0.006 | 0.009 | 0.020 |
| context | Y2-supported | ypp_observed | 3.948 | 0.025 | 8.779 | 9.202 | 0.751 | 28.0 | 0.007 | 0.009 | 0.019 |
| history | Y2-supported | ypp_observed | 3.858 | 0.021 | 8.030 | 8.336 | 0.786 | 28.7 | 0.007 | 0.009 | 0.020 |

The primary end-to-end question is whether adding supported YPP to the connected ranking network improves FBS posterior quality. Full production-style keeps every eligible game: supported-pairing missing YPP gets exactly V1 margin evidence, and every FCS–FCS game gets V1 margin evidence regardless of YPP. The YPP-observed view restricts both V1 and YPP candidates to the same games where YPP could contribute; it is a matched diagnostic, while the full view is primary.

### Y2-supported minus V1 by final-test season

| Prior | Season | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-5 Brier | Δ Top-10 Brier | Δ Top-25 Brier |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 2022 | 0.026 | 0.001 | 0.286 | 0.290 | -0.016 | -0.5 | -0.000 | -0.000 | -0.001 |
| context | 2023 | -0.001 | -0.001 | -0.359 | -0.398 | -0.006 | -0.5 | 0.000 | -0.000 | -0.002 |
| context | 2024 | 0.025 | 0.001 | 0.229 | 0.231 | -0.007 | -0.3 | -0.001 | -0.000 | 0.002 |
| context | 2025 | 0.027 | 0.001 | 0.206 | 0.272 | -0.012 | -0.7 | 0.000 | 0.000 | -0.000 |
| history | 2022 | 0.032 | 0.002 | 0.470 | 0.458 | -0.019 | -0.8 | -0.000 | -0.000 | -0.000 |
| history | 2023 | 0.013 | 0.001 | 0.036 | 0.000 | -0.010 | -0.5 | 0.000 | -0.000 | -0.001 |
| history | 2024 | 0.043 | 0.002 | 0.606 | 0.672 | -0.012 | -0.5 | -0.001 | -0.000 | 0.003 |
| history | 2025 | 0.052 | 0.002 | 0.395 | 0.478 | -0.024 | -0.9 | 0.000 | 0.001 | 0.003 |

### Aggregate Y2-supported minus V1

| Prior | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-5 Brier | Δ Top-10 Brier | Δ Top-25 Brier |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 0.020 | 0.001 | 0.090 | 0.099 | -0.010 | -0.5 | -0.000 | -0.000 | -0.000 |
| history | 0.035 | 0.002 | 0.377 | 0.402 | -0.016 | -0.7 | -0.000 | 0.000 | 0.001 |

Y1 is a semantic null: its factor is rank-invariant, and the posterior rows match V1 up to the deterministic alias used by the research evaluator. The naïve independent diagnostic is shown in `candidate_metrics.csv` only at final full cutoffs; any sharper posterior without commensurate rank scores is double-counting warning evidence.

### Original all-pairings diagnostic versus corrected supported pairing

The following differences are `Y2-supported − Y2-all-pairings-original` on the same full-population final cutoffs. They quantify how much the unsupported FCS–FCS factor in the first result changed downstream FBS posterior evaluation; they are diagnostic, not a new model search.

| Prior | Season | Δ NLL | Δ CRPS | Δ expected-rank MAE | Δ median-rank MAE | Δ 80% coverage | Δ width | Δ Top-10 Brier |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 2022 | -0.075 | -0.004 | -0.743 | -0.679 | 0.024 | 0.2 | -0.001 |
| history | 2022 | -0.043 | -0.002 | -0.314 | -0.450 | 0.015 | 0.0 | -0.001 |
| context | 2023 | -0.096 | -0.005 | -0.945 | -0.970 | 0.025 | 0.3 | 0.001 |
| history | 2023 | -0.057 | -0.003 | -0.546 | -0.639 | 0.016 | 0.1 | 0.001 |
| context | 2024 | -0.041 | -0.002 | -0.353 | -0.463 | 0.011 | 0.1 | 0.000 |
| history | 2024 | 0.008 | 0.000 | 0.320 | 0.313 | -0.005 | 0.1 | 0.001 |
| context | 2025 | -0.093 | -0.004 | -0.851 | -0.919 | 0.030 | 0.1 | -0.000 |
| history | 2025 | -0.020 | -0.001 | -0.228 | -0.272 | 0.011 | 0.1 | 0.001 |

Across the full final-cutoff rows, the mean corrected-minus-original NLL change was -0.052 and the mean 80% coverage change was 0.016. The full row-level comparison is in `pairing_contamination.csv`.

The largest FBS expected-rank shifts are listed below. `fbs_fcs_game_count` and `fcs_opponents` are descriptive network exposure fields, not causal attribution.

| Season | Prior | Team | Δ expected rank | FBS–FCS games | FCS opponents |
|---:|:---|:---|---:|---:|:---|
| 2022 | history | App State (2026) | -6.688 | 2 | Robert Morris (2523); The Citadel (2643) |
| 2022 | history | Virginia (258) | -6.611 | 1 | Richmond (257) |
| 2022 | context | App State (2026) | -6.370 | 2 | Robert Morris (2523); The Citadel (2643) |
| 2022 | context | Virginia (258) | -6.287 | 1 | Richmond (257) |
| 2022 | history | Marshall (276) | -6.122 | 2 | Gardner-Webb (2241); Norfolk State (2450) |
| 2022 | context | Marshall (276) | -5.778 | 2 | Gardner-Webb (2241); Norfolk State (2450) |
| 2022 | history | Georgia Tech (59) | -5.414 | 1 | Western Carolina (2717) |
| 2022 | history | Army (349) | -5.266 | 2 | Colgate (2142); Villanova (222) |
| 2022 | history | Troy (2653) | -5.167 | 1 | Alabama A&M (2010) |
| 2022 | history | Duke (150) | -5.146 | 1 | North Carolina A&T (2448) |
| 2022 | context | Georgia Tech (59) | -5.102 | 1 | Western Carolina (2717) |
| 2022 | history | NC State (152) | -5.050 | 1 | Charleston Southern (2127) |

This is a propagation diagnostic through the connected schedule graph. It does not claim that a particular FBS–FCS opponent caused the change; it identifies where the unsupported FCS–FCS evidence reached the FBS posterior most strongly.

### Future-game check

Future games are scored with the same frozen V1 margin density from each cutoff posterior. The YPP factor is not used to score future margins; it only changes the state estimate. Future and next-game keys are identical across V1, Y1, Y2-all-pairings-original, and Y2-supported.

| Prior | Candidate | Future games | Next-game MAE | Future Win Brier | Next-game Brier | Future margin NLL | Next-game NLL |
|:---|:---|---:|---:|---:|---:|---:|---:|
| context | V1 | 17284 | 12.840 | 0.202 | 0.186 | 4.235 | 4.209 |
| context | Y1 | 17284 | 12.840 | 0.202 | 0.186 | 4.235 | 4.209 |
| context | Y2-all-pairings-original | 17284 | 12.921 | 0.204 | 0.188 | 4.237 | 4.213 |
| context | Y2-supported | 17284 | 12.826 | 0.203 | 0.186 | 4.234 | 4.207 |
| history | V1 | 17284 | 12.957 | 0.204 | 0.187 | 4.239 | 4.216 |
| history | Y1 | 17284 | 12.957 | 0.204 | 0.187 | 4.239 | 4.216 |
| history | Y2-all-pairings-original | 17284 | 13.033 | 0.205 | 0.189 | 4.242 | 4.220 |
| history | Y2-supported | 17284 | 12.941 | 0.204 | 0.187 | 4.238 | 4.214 |

## Margin/YPP disagreement games

The examples below are real corpus FBS–FBS or FBS–FCS games. `v1_margin_predictive_nll` is the pre-game V1 predictive evidence; quality shifts are oriented percentile shifts from a context-prior local update; negative YPP-augmented effect means the supported YPP factor moves the V1-oriented side toward a better latent rank relative to V1 alone.

| Type | Season | Game | Teams | Score | Margin | YPP diff | V1 NLL | V1 shift | YPP effect |
|:---|---:|---:|:---|:---|---:|---:|---:|---:|---:|
| large_win_with_nonpositive_ypp | 2025 | 401752818 | Rutgers–Miami (OH) | 45–17 | 28 | -2.710 | 4.138 | -0.173 | 0.067 |
| large_win_with_nonpositive_ypp | 2024 | 401729780 | Montana State–Idaho | 52–19 | 33 | -1.089 | 4.534 | -0.030 | -0.005 |
| large_win_with_nonpositive_ypp | 2025 | 401756942 | Utah–Cincinnati | 45–14 | 31 | -1.063 | 4.378 | -0.061 | 0.047 |
| close_loss_with_strong_positive_ypp | 2022 | 401424403 | Monmouth–Fordham | 49–52 | -3 | 3.742 | 3.981 | -0.027 | 0.000 |
| close_loss_with_strong_positive_ypp | 2023 | 401540237 | UC Davis–Eastern Washington | 24–27 | -3 | 3.613 | 3.916 | 0.080 | 0.003 |
| close_loss_with_strong_positive_ypp | 2024 | 401636379 | Florida A&M–Mississippi Valley State | 21–24 | -3 | 3.566 | 5.958 | 0.055 | 0.014 |
| close_win_with_strong_negative_ypp | 2025 | 401761648 | Louisiana–Texas State | 42–39 | 3 | -4.402 | 3.710 | -0.020 | 0.023 |
| close_win_with_strong_negative_ypp | 2025 | 401767343 | Sacred Heart–Delaware State | 35–31 | 4 | -3.490 | 3.910 | -0.081 | 0.007 |
| close_win_with_strong_negative_ypp | 2022 | 401420815 | North Dakota–Northern Iowa | 29–27 | 2 | -3.125 | 3.958 | 0.012 | 0.011 |

## Temporal and subdivision stability

The fixed pre-test model is evaluated by season and supported pairing; no adaptive era weighting, NIL break, or post-test refit is introduced. FCS–FCS is excluded from the direct residual-signal interpretation and receives V1-only evidence in the corrected posterior.

Before reading 2022–2025, the promotion gate was fixed at: aggregate final-rank NLL improvement of at least 0.02 nats/team; CRPS degradation no greater than 0.002; 80% coverage drop no greater than 0.03; improvement in at least 3 of 4 held-out seasons for both prior families; no single-season NLL degradation above 0.10 or CRPS degradation above 0.02; future margin MAE degradation no greater than 0.50 points and NLL degradation no greater than 0.02; and positive common-subset quality slope in at least 3 seasons.

The promotion assessment for Y2-supported is `{"aggregate_delta_crps": 0.0011175157679119184, "aggregate_delta_interval_80_coverage": -0.013332636538606793, "aggregate_delta_nll": 0.027303460101990196, "checks": {"aggregate_crps": true, "aggregate_nll": false, "calibration": true, "common_subset_signal": true, "future_direction": true, "majority_seasons": false, "no_catastrophic_season": true}, "common_subset_positive_quality_slope_seasons": 4, "mean_future_delta_margin_mae": -0.004630995552892037, "mean_future_delta_margin_nll": -9.259828480256971e-05, "recommendation": "B", "seasons_with_nll_improvement": {"context": 1, "history": 0}, "worst_single_season_delta_crps": 0.00241943610353864, "worst_single_season_delta_nll": 0.051875802527989734}`. These are the unchanged predeclared thresholds; the original all-pairings diagnostic is not used for this decision.

## Production feasibility (not productionized here)

The lightweight weekly updater currently fetches only two `/games` schedule responses for the season and deliberately does not fetch `/games/teams`. The historical builder has 764 cached team-stat response files across the corpus; the cached 2026 shape contains 27 regular-season `/games/teams` response artifacts ({"fbs": 14, "fcs": 13}). That is the current full-season request estimate if each classification/week response is fetched, with fewer requests for an in-season snapshot. Raw/provenance sidecars, a game-id/team-id join, and exact missing-YPP fallback semantics would be required. Current-season YPP acquisition and weekly snapshot runtime were not changed or claimed production-safe in this PR; a later PR should benchmark request latency, rate limits, raw artifact size, and local factor-construction cost before promotion.

## Frozen production boundary and reproducibility

No Historical Likelihood V1, Posterior V1, H 1.1, C 1.2, Performance V1, weekly publication, or website file is modified. The research script reads frozen artifacts and writes only this investigation directory. The input SHA-256 values and output inventory are in `summary.json`.

The branch is research-only. Recommendation A would mean a future Likelihood V2 productionization PR is justified; B means the residual signal exists but formulation/coverage needs more research; C means YPP does not earn the added game-likelihood complexity.
