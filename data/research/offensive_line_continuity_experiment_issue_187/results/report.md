# Issue 187: offensive-line shared-roster continuity experiment

**Decision:** Reject

**Recommendation:** Do not advance these four-season OL continuity features into production Context on the current evidence. All added-feature arms worsened held-out rank-distribution likelihood overall, and continuity was worse in each of the four evaluation seasons. Revisit only if point-in-time OL rosters or new independent evidence become available.

## Scope and cohort

The feature panel covers 2009–2025; 2013–2025 is the primary window. Every primary feature uses the same four prior roster seasons, T-4 through T-1. 2010–2012 remains a partial-history descriptive panel only. The target cohort is the retrospective CFBD offensive-line roster for target season T. Pair evidence comes only from the same team ID and seasons strictly before T; player position in prior seasons is ignored.

The panel contains 1,691 primary-window team-seasons. Feature status counts: observed=1686, target_pool_unusable_or_too_small=5. The rank-prediction comparison retained 1,668 identical team-seasons across all model arms. Exclusions: target_pool_unusable_or_too_small=5.

## Controlled evaluation

Current Context is Context 1.3's direct rank-distribution model and its existing rank-history, recruiting, talent, returning-production, coaching, and transfer inputs. The research adapter holds that feature set, normal likelihood, training-only imputation/standardization, missingness indicators, location-only Context placement, optimizer, and 0.25 regularization fixed. New coefficients are fitted only on 2013–2021; the identical 2022–2025 FBS team-seasons are scored in every arm. Feature scaling and imputation use training data only. The reported score is the repository's empirical rank-distribution negative log-likelihood (nats per team-season), not a game-level score.

## Held-out predictive results

Negative ΔNLL favors the added-feature model. The interval resamples held-out seasons as clusters and is descriptive with four evaluation seasons.

| Model | Team-seasons | NLL | ΔNLL vs Context | 95% season-bootstrap interval | Seasons better | Team-seasons improved | Top 10 share of gross gains |
|:--|--:|--:|--:|:--|--:|--:|--:|
| current_context_1_3 | 528 | 4.5170 | 0.0000 | [0.0000, 0.0000] | 0 / 4 | 0 (0.0%) | 0.0% |
| context_plus_ol_individual_experience | 528 | 4.5196 | 0.0026 | [-0.0002, 0.0059] | 1 / 4 | 252 (47.7%) | 19.4% |
| context_plus_ol_shared_roster_continuity | 528 | 4.5258 | 0.0088 | [0.0024, 0.0192] | 0 / 4 | 253 (47.9%) | 32.9% |
| context_plus_both_ol_experience_and_continuity | 528 | 4.5297 | 0.0127 | [0.0042, 0.0267] | 0 / 4 | 257 (48.7%) | 29.8% |

Continuity beyond individual experience: combined ΔNLL versus the individual-experience arm is 0.0101, with season-cluster interval 0.0022 to 0.0208.

### By season

| Season | Model | Team-seasons | NLL | ΔNLL vs Context | ΔCRPS vs Context |
|--:|:--|--:|--:|--:|--:|
| 2022 | current_context_1_3 | 130 | 4.4633 | 0.0000 | 0.0000 |
| 2022 | context_plus_ol_individual_experience | 130 | 4.4625 | -0.0008 | -0.0000 |
| 2022 | context_plus_ol_shared_roster_continuity | 130 | 4.4651 | 0.0017 | 0.0000 |
| 2022 | context_plus_both_ol_experience_and_continuity | 130 | 4.4716 | 0.0083 | 0.0006 |
| 2023 | current_context_1_3 | 131 | 4.4788 | 0.0000 | 0.0000 |
| 2023 | context_plus_ol_individual_experience | 131 | 4.4819 | 0.0030 | 0.0003 |
| 2023 | context_plus_ol_shared_roster_continuity | 131 | 4.4835 | 0.0046 | 0.0007 |
| 2023 | context_plus_both_ol_experience_and_continuity | 131 | 4.4818 | 0.0030 | 0.0004 |
| 2024 | current_context_1_3 | 133 | 4.5713 | 0.0000 | 0.0000 |
| 2024 | context_plus_ol_individual_experience | 133 | 4.5716 | 0.0004 | -0.0000 |
| 2024 | context_plus_ol_shared_roster_continuity | 133 | 4.5755 | 0.0042 | 0.0005 |
| 2024 | context_plus_both_ol_experience_and_continuity | 133 | 4.5767 | 0.0055 | 0.0008 |
| 2025 | current_context_1_3 | 134 | 4.5525 | 0.0000 | 0.0000 |
| 2025 | context_plus_ol_individual_experience | 134 | 4.5602 | 0.0077 | 0.0007 |
| 2025 | context_plus_ol_shared_roster_continuity | 134 | 4.5767 | 0.0242 | 0.0013 |
| 2025 | context_plus_both_ol_experience_and_continuity | 134 | 4.5863 | 0.0337 | 0.0025 |

## Feature behavior

Correlations use the mean observed final rank divided by team population, where lower is better. They are descriptive associations, not causal effects. Coefficients use training-standardized features; missingness terms are shown separately.

| Feature | Observed team-seasons | Mean | Median | P10 | P90 | Pearson r vs final-rank fraction |
|:--|--:|--:|--:|--:|--:|--:|
| ol_returning_player_share_4y | 1668 | 0.6519 | 0.6667 | 0.4737 | 0.8125 | -0.0968 |
| ol_mean_prior_same_program_seasons_4y | 1668 | 1.3252 | 1.3191 | 0.8725 | 1.7926 | -0.1170 |
| ol_mean_shared_roster_seasons_4y | 1668 | 0.6420 | 0.6099 | 0.3041 | 1.0233 | -0.1042 |
| ol_shared_pair_share_4y | 1668 | 0.4314 | 0.4277 | 0.2105 | 0.6500 | -0.0844 |
| ol_returning_group_share_4y | 1668 | 0.6512 | 0.6603 | 0.4737 | 0.8125 | -0.0954 |

| Model | Feature | Standardized location coefficient | Missingness coefficient | Mean held-out location contribution |
|:--|:--|--:|--:|--:|
| context_plus_ol_individual_experience | ol_returning_player_share_4y | -0.0124 | 0.0000 | 0.0106 |
| context_plus_ol_individual_experience | ol_mean_prior_same_program_seasons_4y | -0.0180 | 0.0000 | 0.0123 |
| context_plus_ol_shared_roster_continuity | ol_mean_shared_roster_seasons_4y | -0.0146 | 0.0000 | 0.0102 |
| context_plus_ol_shared_roster_continuity | ol_shared_pair_share_4y | 0.3445 | 0.0000 | -0.2651 |
| context_plus_ol_shared_roster_continuity | ol_returning_group_share_4y | -0.3509 | 0.0000 | 0.3047 |
| context_plus_both_ol_experience_and_continuity | ol_returning_player_share_4y | 0.6103 | 0.0000 | -0.5225 |
| context_plus_both_ol_experience_and_continuity | ol_mean_prior_same_program_seasons_4y | -0.0763 | 0.0000 | 0.0522 |
| context_plus_both_ol_experience_and_continuity | ol_mean_shared_roster_seasons_4y | 0.0892 | 0.0000 | -0.0626 |
| context_plus_both_ol_experience_and_continuity | ol_shared_pair_share_4y | 0.2257 | 0.0000 | -0.1737 |
| context_plus_both_ol_experience_and_continuity | ol_returning_group_share_4y | -0.8778 | 0.0000 | 0.7623 |

### Held-out room profiles

| Profile | Team-seasons | Seasons | Mean final-rank fraction | Mean OL pool size | ΔNLL vs Context (individual / continuity / both) |
|:--|--:|--:|--:|--:|:--|
| stable_veteran_room | 55 | 4 | 0.3956 | 19.1455 | 0.0007, -0.0001, 0.0004 |
| heavily_rebuilt_room | 6 | 3 | 0.6102 | 19.1667 | 0.0458, 0.2613, 0.2318 |
| transfer_heavy_by_prior_other_school_share | 4 | 2 | 0.4224 | 17.0000 | -0.0028, 0.0172, 0.0553 |
| small_target_ol_pool_p10 | 54 | 4 | 0.5833 | 15.4815 | 0.0042, 0.0341, 0.0438 |
| large_target_ol_pool_p90 | 81 | 4 | 0.4287 | 24.1235 | 0.0070, -0.0017, 0.0022 |
| incomplete_pair_or_identity_coverage | 0 | 0 | — | — | —, —, — |

## Roster-quality sensitivity

Issue 185's independent frozen sample contains 40 team-seasons (20 exact CFBD/official player sets and 20 mismatches). Of the sample rows in 2013–2025, 30 are in this study window; 23 have a complete mapping from every official target-pool player to a unique existing CFBD roster identity. No IDs were inferred for players absent from or ambiguous in the existing CFBD history.

For complete mappings, feature shifts compare recomputed official-pool values with CFBD-pool values on the same team-season. The frozen sample is stratified and small; these rates are not population estimates.

| Feature | Mapped sample rows | Mean absolute change | Median absolute change | P90 absolute change | Maximum absolute change |
|:--|--:|--:|--:|--:|--:|
| ol_returning_player_share_4y | 23 | 0.0041 | 0.0000 | 0.0206 | 0.0329 |
| ol_mean_prior_same_program_seasons_4y | 23 | 0.0081 | 0.0000 | 0.0289 | 0.0680 |
| ol_mean_shared_roster_seasons_4y | 23 | 0.0088 | 0.0000 | 0.0156 | 0.1222 |
| ol_shared_pair_share_4y | 23 | 0.0058 | 0.0000 | 0.0266 | 0.0591 |
| ol_returning_group_share_4y | 23 | 0.0041 | 0.0000 | 0.0206 | 0.0329 |

### Held-out scores by validation coverage

Rows not included in the issue 185 sample remain unclassified; they are not called high confidence. Any group without at least 20 team-seasons over three seasons is reported as descriptive only.

| Coverage group | Model | Team-seasons | Seasons | ΔNLL vs Context | 95% interval | Status |
|:--|:--|--:|--:|--:|:--|:--|
| all_primary_evaluation_rows | context_plus_ol_individual_experience | 528 | 4 | 0.0026 | [-0.0002, 0.0059] | sample_supports_cluster_interval |
| all_primary_evaluation_rows | context_plus_ol_shared_roster_continuity | 528 | 4 | 0.0088 | [0.0024, 0.0192] | sample_supports_cluster_interval |
| all_primary_evaluation_rows | context_plus_both_ol_experience_and_continuity | 528 | 4 | 0.0127 | [0.0042, 0.0267] | sample_supports_cluster_interval |
| not_in_issue185_validation_sample | context_plus_ol_individual_experience | 520 | 4 | 0.0025 | [-0.0002, 0.0054] | sample_supports_cluster_interval |
| not_in_issue185_validation_sample | context_plus_ol_shared_roster_continuity | 520 | 4 | 0.0085 | [0.0026, 0.0181] | sample_supports_cluster_interval |
| not_in_issue185_validation_sample | context_plus_both_ol_experience_and_continuity | 520 | 4 | 0.0126 | [0.0045, 0.0261] | sample_supports_cluster_interval |
| official_pool_exact_set_match | context_plus_ol_individual_experience | 6 | 4 | 0.0090 | — | descriptive_only_small_or_uneven_sample |
| official_pool_exact_set_match | context_plus_ol_shared_roster_continuity | 6 | 4 | 0.0342 | — | descriptive_only_small_or_uneven_sample |
| official_pool_exact_set_match | context_plus_both_ol_experience_and_continuity | 6 | 4 | 0.0311 | — | descriptive_only_small_or_uneven_sample |
| official_pool_target_mismatch | context_plus_ol_individual_experience | 2 | 1 | 0.0055 | — | descriptive_only_small_or_uneven_sample |
| official_pool_target_mismatch | context_plus_ol_shared_roster_continuity | 2 | 1 | -0.0012 | — | descriptive_only_small_or_uneven_sample |
| official_pool_target_mismatch | context_plus_both_ol_experience_and_continuity | 2 | 1 | -0.0032 | — | descriptive_only_small_or_uneven_sample |

### Official-pool score substitution

For target team-seasons in the held-out years with complete official-to-CFBD ID mapping, the already fitted models are rescored after substituting official-pool features. No coefficients are refit on validation data.

| Model | Team-seasons | Seasons | CFBD-pool ΔNLL | Official-pool ΔNLL | Official minus CFBD | Status |
|:--|--:|--:|--:|--:|--:|:--|
| context_plus_ol_individual_experience | 8 | 4 | 0.0081 | 0.0067 | -0.0014 | descriptive_only_small_or_uneven_sample |
| context_plus_ol_shared_roster_continuity | 8 | 4 | 0.0253 | 0.0233 | -0.0020 | descriptive_only_small_or_uneven_sample |
| context_plus_both_ol_experience_and_continuity | 8 | 4 | 0.0225 | 0.0212 | -0.0013 | descriptive_only_small_or_uneven_sample |

## Co-start comparison

The issue 181 pilot supplied 8 positive co-start pairs; 8 matched both players to the CFBD target OL pool across 3 team-seasons. Spearman correlation between shared-start rate and four-year shared-roster seasons among matched pairs: 0.1807. Positive co-start pairs only; a small descriptive compatibility check, not a population-level validation.

## Limits and interpretation

The target-season CFBD roster responses are retrospective and may include in-season arrivals/departures. Continuity evidence itself uses only earlier seasons, but the target cohort is not a preseason snapshot; this study is not proof of prospective deployability. CFBD's position errors and omissions remain visible through target-pool counts, unknown-position counts, identity/pair coverage flags, and the independent issue 185 sensitivity. The validation sample cannot correct historical rosters at scale. Rank distributions are measurements and the held-out score is an existing Context rank-likelihood score, not a direct game-outcome score.

No production Context feature, coefficient, or ranking output was modified. The added-feature fitting path is research only.

## Reproduction

The script reads the committed #183 normalized roster panel, #185 validation inputs, #181 co-start pairs, the historical Context feature tables, and the locally generated rank-distribution artifact. Large raw/reacquirable CFBD inputs remain outside Git. Local coach-tenure snapshots are also needed by the existing Context row builder.

```bash
uv run python scripts/evaluate_offensive_line_continuity_issue_187.py --decision Reject --recommendation "Do not advance these four-season OL continuity features into production Context on the current evidence. All added-feature arms worsened held-out rank-distribution likelihood overall, and continuity was worse in each of the four evaluation seasons. Revisit only if point-in-time OL rosters or new independent evidence become available."
uv run python scripts/evaluate_offensive_line_continuity_issue_187.py --decision Reject --recommendation "Do not advance these four-season OL continuity features into production Context on the current evidence. All added-feature arms worsened held-out rank-distribution likelihood overall, and continuity was worse in each of the four evaluation seasons. Revisit only if point-in-time OL rosters or new independent evidence become available." --check
```

Inputs and hashes are recorded in `manifest.json`; compact team-season features, paired losses, validation changes, and co-start joins are retained beside this report.

### Inputs

- `coach_tenure/air_force.json`: `data/raw/cfbd/preseason/coach_tenures/air_force.json` · SHA-256 `401bd757f32157c17c66e489eef68570d1bb1e66558ea601301a7edf9c81a9e6`
- `coach_tenure/akron.json`: `data/raw/cfbd/preseason/coach_tenures/akron.json` · SHA-256 `43de5ed377a43f8c660122296b032c95adfd6a42f23e6a042ad19b0176f1ab3d`
- `coach_tenure/alabama.json`: `data/raw/cfbd/preseason/coach_tenures/alabama.json` · SHA-256 `c3ccb2819bf3d2f0aad4723d02d1e81f8a7d1ae9dee6d26c9f9ccac4188e7306`
- `coach_tenure/app_state.json`: `data/raw/cfbd/preseason/coach_tenures/app_state.json` · SHA-256 `dc918911a4c130ffd52d5c21e0bf774d75e47a28b69ec4e9a5e3fc4f082a6cf1`
- `coach_tenure/arizona.json`: `data/raw/cfbd/preseason/coach_tenures/arizona.json` · SHA-256 `7c7d7f04bec3becaa41b8983f20ffd5f561cba802cb16acb27cd5c128045a24d`
- `coach_tenure/arizona_state.json`: `data/raw/cfbd/preseason/coach_tenures/arizona_state.json` · SHA-256 `37e8bd13c44c96d2a3df4de0b8bbb0c37a59adb00b1e6539c51d423b467a31c9`
- `coach_tenure/arkansas.json`: `data/raw/cfbd/preseason/coach_tenures/arkansas.json` · SHA-256 `d35d83cde958447bd2abcbf788eec1877fb3e66156b4cd4e19e2a4b3abfca8ff`
- `coach_tenure/arkansas_state.json`: `data/raw/cfbd/preseason/coach_tenures/arkansas_state.json` · SHA-256 `5584d88cce85e5ae2190faef0e84adb531df541ff453002eec89d34e895d2e83`
- `coach_tenure/army.json`: `data/raw/cfbd/preseason/coach_tenures/army.json` · SHA-256 `2b4a7b0115788bfb425d1500dc003d691cb740eae13ab47ac1fb1e68289c6533`
- `coach_tenure/auburn.json`: `data/raw/cfbd/preseason/coach_tenures/auburn.json` · SHA-256 `0c38cbb272530b99f67a75c52c530c0f2928bbbc69b37ef4f96982b9d2cce9b3`
- `coach_tenure/ball_state.json`: `data/raw/cfbd/preseason/coach_tenures/ball_state.json` · SHA-256 `5d0f06e552bb6bcf25bcb5e5d780f5db0400aa84d33a47215077f3d68e45f188`
- `coach_tenure/baylor.json`: `data/raw/cfbd/preseason/coach_tenures/baylor.json` · SHA-256 `151d1f8ac953476d27c12294935555d4b6b9f68537d1fefb17c9e91804fec9ba`
- `coach_tenure/boise_state.json`: `data/raw/cfbd/preseason/coach_tenures/boise_state.json` · SHA-256 `e2bcbea472b4c158ebfed1b4035294e8f4566370ce99a265a8859aefe496be74`
- `coach_tenure/boston_college.json`: `data/raw/cfbd/preseason/coach_tenures/boston_college.json` · SHA-256 `ac68a42dff87c67acb06cf2d29071e11aae5bdec7945df0a20202e2b7ca40ebd`
- `coach_tenure/bowling_green.json`: `data/raw/cfbd/preseason/coach_tenures/bowling_green.json` · SHA-256 `412bba69a3aa50fa8187f48472e9b7939fec71cbb1b0b9ceaa32281627f86df1`
- `coach_tenure/buffalo.json`: `data/raw/cfbd/preseason/coach_tenures/buffalo.json` · SHA-256 `f9aaf3c61c5fd58a8dba7298efd90a6e2ae2a6085dfa022368af08805d0432da`
- `coach_tenure/byu.json`: `data/raw/cfbd/preseason/coach_tenures/byu.json` · SHA-256 `a2b7ec84798fd8f93cf012bf985ff249064b512f71aaf10afb2fbc0dba908704`
- `coach_tenure/california.json`: `data/raw/cfbd/preseason/coach_tenures/california.json` · SHA-256 `6588de3a3a39ec75147dc878213781c76b1956ae282c9443d2c501440b94390a`
- `coach_tenure/central_michigan.json`: `data/raw/cfbd/preseason/coach_tenures/central_michigan.json` · SHA-256 `2dc05cb4a810ebcbf72accb6945790791213eeb693cbfb2464788a9ed2e18a61`
- `coach_tenure/charlotte.json`: `data/raw/cfbd/preseason/coach_tenures/charlotte.json` · SHA-256 `58fb9a8d16ab243f87b39c22ff8a24a6bd229282576240ce8f260b3d52cc8d62`
- `coach_tenure/cincinnati.json`: `data/raw/cfbd/preseason/coach_tenures/cincinnati.json` · SHA-256 `342e82877dc39328049c23ecfe81399762b9eb9a626bb630aded6ba55a192445`
- `coach_tenure/clemson.json`: `data/raw/cfbd/preseason/coach_tenures/clemson.json` · SHA-256 `7e3f582486c81f4fccabaaca6a8a8fb8f9a6ccfc2ef5b82b452ff232c40e0edf`
- `coach_tenure/coastal_carolina.json`: `data/raw/cfbd/preseason/coach_tenures/coastal_carolina.json` · SHA-256 `9cd3637dd880d6247054d85a9cb38e10c5137b87027fc195f78b0cb3c6ab62e3`
- `coach_tenure/colorado.json`: `data/raw/cfbd/preseason/coach_tenures/colorado.json` · SHA-256 `63a9b5e66ea3d1afd9cdd967b4fa7a5f437f9c92e295f67edaaac7f925c87ae9`
- `coach_tenure/colorado_state.json`: `data/raw/cfbd/preseason/coach_tenures/colorado_state.json` · SHA-256 `784cd972dfa00aaef3242a55d9d40d2c55cbcd0b4be74ab6c99082165da667ef`
- `coach_tenure/delaware.json`: `data/raw/cfbd/preseason/coach_tenures/delaware.json` · SHA-256 `b3ca70f772aa9c45972545f8ab310ca31e74e9263778db0553188c269b693902`
- `coach_tenure/duke.json`: `data/raw/cfbd/preseason/coach_tenures/duke.json` · SHA-256 `6d537bbaf719134a466634e6f2cde0b2f62c837fd33cb5284a107d7cc26ed757`
- `coach_tenure/east_carolina.json`: `data/raw/cfbd/preseason/coach_tenures/east_carolina.json` · SHA-256 `22574f22211dcb2ebc9df1451f50c4445ecf57b621f489fdc59691bae7edfee7`
- `coach_tenure/eastern_michigan.json`: `data/raw/cfbd/preseason/coach_tenures/eastern_michigan.json` · SHA-256 `25fa64f7c264b4c7d7c3fcf441df04f614eb4e4bc79d2dbc3915883dcc0a8c5c`
- `coach_tenure/florida.json`: `data/raw/cfbd/preseason/coach_tenures/florida.json` · SHA-256 `1483888cfa97795d2034ff7e288e4a50091f65912589c6bb38ecd6030de3ef19`
- `coach_tenure/florida_a&m.json`: `data/raw/cfbd/preseason/coach_tenures/florida_a&m.json` · SHA-256 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `coach_tenure/florida_atlantic.json`: `data/raw/cfbd/preseason/coach_tenures/florida_atlantic.json` · SHA-256 `d8c8615fd2f91022616ed42f50b3ac6797e960d33d1a7b4d8de62b605e986c3a`
- `coach_tenure/florida_international.json`: `data/raw/cfbd/preseason/coach_tenures/florida_international.json` · SHA-256 `e50cc98f8520991cef2357a1753aa2e10ffaab751dbbd60df64c5e400279de2f`
- `coach_tenure/florida_state.json`: `data/raw/cfbd/preseason/coach_tenures/florida_state.json` · SHA-256 `c668ee728ba60ae65f1b9edbcd218b290bfc796cca40552be0af06e89f008e8c`
- `coach_tenure/fresno_state.json`: `data/raw/cfbd/preseason/coach_tenures/fresno_state.json` · SHA-256 `a159de49e61de9c5dc06f1513b6b92ed16c6a1dadcf77674b54542abb9364e93`
- `coach_tenure/georgia.json`: `data/raw/cfbd/preseason/coach_tenures/georgia.json` · SHA-256 `53abf36df8c776323a8e98d1483f188a80956b69225c4f9989bce69f9b4f8208`
- `coach_tenure/georgia_southern.json`: `data/raw/cfbd/preseason/coach_tenures/georgia_southern.json` · SHA-256 `6da6c11ee4a9014b797f7665ac08660da4fb9e3a374048c791dc4af0d989b60f`
- `coach_tenure/georgia_state.json`: `data/raw/cfbd/preseason/coach_tenures/georgia_state.json` · SHA-256 `0c5e76c6d0e1ebeeadba99dd131f22a19e7d5700bb022375230c8c73ed72c175`
- `coach_tenure/georgia_tech.json`: `data/raw/cfbd/preseason/coach_tenures/georgia_tech.json` · SHA-256 `b9c672c87f013842e0a9e8632f4802ff0ed95fb2dc82533809d77efc812535f3`
- `coach_tenure/hawai'i.json`: `data/raw/cfbd/preseason/coach_tenures/hawai'i.json` · SHA-256 `cc1fc52a25cd27c4a73813142103ab6fe10119fe1e84a50a8a2d4e8da4288201`
- `coach_tenure/houston.json`: `data/raw/cfbd/preseason/coach_tenures/houston.json` · SHA-256 `4b86e5d8a7955630e2971870e8855b3b38a83159b3ab04468080ab5cab9cf856`
- `coach_tenure/idaho.json`: `data/raw/cfbd/preseason/coach_tenures/idaho.json` · SHA-256 `a11328cc516f7754a4d86e4b2e541d4ce0ca096cde8aaff4433e323192548ee0`
- `coach_tenure/illinois.json`: `data/raw/cfbd/preseason/coach_tenures/illinois.json` · SHA-256 `e572519223ab2ab4e52ef7f1cc5ee82b6da65486a45f8a6dabfe633cb6472f3b`
- `coach_tenure/indiana.json`: `data/raw/cfbd/preseason/coach_tenures/indiana.json` · SHA-256 `2865b3538550803e425cdffa1a5ebf6925d5486f5987183cc614b17ab84f8ffc`
- `coach_tenure/iowa.json`: `data/raw/cfbd/preseason/coach_tenures/iowa.json` · SHA-256 `23c3f46cf21787e3155c5b7a02b57f695fe4fd850cffbfd95930976da1e73011`
- `coach_tenure/iowa_state.json`: `data/raw/cfbd/preseason/coach_tenures/iowa_state.json` · SHA-256 `c1398bc18078c62b13d89b60eecf949f0d2e0249c155c7544f31fce21a63f7d5`
- `coach_tenure/jacksonville_state.json`: `data/raw/cfbd/preseason/coach_tenures/jacksonville_state.json` · SHA-256 `c8bf7c2bb74d356a78881f123f2dcd8d3630299d03cfa83eec4249a69bdd902b`
- `coach_tenure/james_madison.json`: `data/raw/cfbd/preseason/coach_tenures/james_madison.json` · SHA-256 `dce793895036d431dd79f3e90c0f366507ae87c5a5dd91e18cad00cb4f841eb5`
- `coach_tenure/kansas.json`: `data/raw/cfbd/preseason/coach_tenures/kansas.json` · SHA-256 `efe4b451b99e576d214393dbae1813d7e407c62087e91e8ab52047563ca45903`
- `coach_tenure/kansas_state.json`: `data/raw/cfbd/preseason/coach_tenures/kansas_state.json` · SHA-256 `af83e9c626c4f0c1f4c25e76d22319708b134de0db5733d0b86ff0575258fb35`
- `coach_tenure/kennesaw_state.json`: `data/raw/cfbd/preseason/coach_tenures/kennesaw_state.json` · SHA-256 `2997ef7d760443732c4f6b783e9a4a8681edbcf64c0232e0aeeb04b28908b7bc`
- `coach_tenure/kent_state.json`: `data/raw/cfbd/preseason/coach_tenures/kent_state.json` · SHA-256 `82482a8734e2a4adf73df26b862296e4d472d4e2ef8874d68228214a3bcb4174`
- `coach_tenure/kentucky.json`: `data/raw/cfbd/preseason/coach_tenures/kentucky.json` · SHA-256 `3630d86784f4766d782e7af815b40d514ae71aaa28ae8183e3a803605874d381`
- `coach_tenure/liberty.json`: `data/raw/cfbd/preseason/coach_tenures/liberty.json` · SHA-256 `f726236dff4e05af32785b640b1baa2d5b3256ffce72c9972f213ee55574551f`
- `coach_tenure/louisiana.json`: `data/raw/cfbd/preseason/coach_tenures/louisiana.json` · SHA-256 `aafd1cdd7b8e8830b70cd3f8c4c248ff5823b1a10e8eb27f51a9dc6382929fe6`
- `coach_tenure/louisiana_tech.json`: `data/raw/cfbd/preseason/coach_tenures/louisiana_tech.json` · SHA-256 `cc86246413209f9ec30b2343bbcffd6aab75956e074bc1790711220dba80aece`
- `coach_tenure/louisville.json`: `data/raw/cfbd/preseason/coach_tenures/louisville.json` · SHA-256 `9385e64461f07cde62927c1f47423d0ee161f35420ca3bee2c7f0d23749d1e41`
- `coach_tenure/lsu.json`: `data/raw/cfbd/preseason/coach_tenures/lsu.json` · SHA-256 `85a0b5f87e394559d4371228a370e869a0fdc20f8c3d392c93dd051df026c4f1`
- `coach_tenure/manifest.json`: `data/raw/cfbd/preseason/coach_tenures/manifest.json` · SHA-256 `1f44e644ed3cec4989cbd779cbb1de331d06ec03187a8bf730c1dd4213e2928c`
- `coach_tenure/marshall.json`: `data/raw/cfbd/preseason/coach_tenures/marshall.json` · SHA-256 `22ca9f7ae8eaa6687380bfc5eb84c846f0f4361c34265b15b5c67fa1340a1dd9`
- `coach_tenure/maryland.json`: `data/raw/cfbd/preseason/coach_tenures/maryland.json` · SHA-256 `cd895a489b8badfa4aad16b5a608fb173dabf0d48d4e4075bd15c050d73c19e2`
- `coach_tenure/massachusetts.json`: `data/raw/cfbd/preseason/coach_tenures/massachusetts.json` · SHA-256 `930aefef1c0a00ebd3a0fbef9a4de0c8e0bb811cb5a3152c54779e2ebc20e386`
- `coach_tenure/memphis.json`: `data/raw/cfbd/preseason/coach_tenures/memphis.json` · SHA-256 `40e8dfbe1b281b2e76a767ccc4560a366328b4dd51606fe4069e884a33a610d5`
- `coach_tenure/miami.json`: `data/raw/cfbd/preseason/coach_tenures/miami.json` · SHA-256 `80ad76a9f19a345f78a91268986d40425b1a040903c41a018070ec6d54611959`
- `coach_tenure/miami_(oh).json`: `data/raw/cfbd/preseason/coach_tenures/miami_(oh).json` · SHA-256 `838fd92d73bff9ac2845322cfddddbc5d883ca696bfc105ddfe29ddc06e5a29f`
- `coach_tenure/michigan.json`: `data/raw/cfbd/preseason/coach_tenures/michigan.json` · SHA-256 `ac9c004acf6dc1ed584148e66d97c076ea9f6fd4e3aa15b8b84e22ab3cb07797`
- `coach_tenure/michigan_state.json`: `data/raw/cfbd/preseason/coach_tenures/michigan_state.json` · SHA-256 `0749f2455f86f063d0a4cbb46c63755dc1e19c47620993e6acae8b4cc1098238`
- `coach_tenure/middle_tennessee.json`: `data/raw/cfbd/preseason/coach_tenures/middle_tennessee.json` · SHA-256 `0f7d8f620172486e38be73f11e89cbfa5b5e3d21da4ef226a18bf40341e99d57`
- `coach_tenure/minnesota.json`: `data/raw/cfbd/preseason/coach_tenures/minnesota.json` · SHA-256 `f31b9e576807fff70d1a65b5bf928c586e12e489b4c22e9419fac0a0e842be52`
- `coach_tenure/mississippi_state.json`: `data/raw/cfbd/preseason/coach_tenures/mississippi_state.json` · SHA-256 `073bfbb1dc823b8ea2207cec9e23cddc7dadcdd6d23063ebde828b7e73d95d11`
- `coach_tenure/missouri.json`: `data/raw/cfbd/preseason/coach_tenures/missouri.json` · SHA-256 `6d050f8c7c2c00de96a072b867d7f58c722d44ee9bc508da588a2ad73babded5`
- `coach_tenure/missouri_state.json`: `data/raw/cfbd/preseason/coach_tenures/missouri_state.json` · SHA-256 `96dc90915636e359a32133a579d3e8288d5a7292a6b9a52ef758e2eef2919978`
- `coach_tenure/navy.json`: `data/raw/cfbd/preseason/coach_tenures/navy.json` · SHA-256 `166af25ba314370c95ab0aff9b3b4933417ffc7f8bc1bde7aaab07f70e498b48`
- `coach_tenure/nc_state.json`: `data/raw/cfbd/preseason/coach_tenures/nc_state.json` · SHA-256 `84df2846bc3f226b3ea155506854579cb52cd1c55fe346105f422f3608f3bdb8`
- `coach_tenure/nebraska.json`: `data/raw/cfbd/preseason/coach_tenures/nebraska.json` · SHA-256 `7103d2ae6eb258ce85a0edc1a2992cf4dc184a5767d6056f48efa525c435fc81`
- `coach_tenure/nevada.json`: `data/raw/cfbd/preseason/coach_tenures/nevada.json` · SHA-256 `2dbb103a53c7abb64a6287362ea6582307631ff02ce54479d7e55bc823da3295`
- `coach_tenure/new_mexico.json`: `data/raw/cfbd/preseason/coach_tenures/new_mexico.json` · SHA-256 `bd8196720d6849e671e08dfddb4f15baebc179a4c7c42af1f7e25dbafcfcc7f6`
- `coach_tenure/new_mexico_state.json`: `data/raw/cfbd/preseason/coach_tenures/new_mexico_state.json` · SHA-256 `720934d9bb73d2c79ce32f742c970c017497543940c2cd21e5169fc78b8c50ef`
- `coach_tenure/north_carolina.json`: `data/raw/cfbd/preseason/coach_tenures/north_carolina.json` · SHA-256 `1275595b80f4f080cd80838d64d59085d58a975a33c14db03c426a9df93395f7`
- `coach_tenure/north_dakota_state.json`: `data/raw/cfbd/preseason/coach_tenures/north_dakota_state.json` · SHA-256 `9aee50a1482d989236533fa265491ec3587dd7161339249f159bd4acc2b39eb4`
- `coach_tenure/north_texas.json`: `data/raw/cfbd/preseason/coach_tenures/north_texas.json` · SHA-256 `ad5fb6d1a933900a54c033453fb6b1f57e4c9cdab31b791b2655bc839c1469d6`
- `coach_tenure/northern_illinois.json`: `data/raw/cfbd/preseason/coach_tenures/northern_illinois.json` · SHA-256 `6dca3b8c53d5695883410031719db0f43213b60ad1b0cb4927ed7a1e661b3b45`
- `coach_tenure/northwestern.json`: `data/raw/cfbd/preseason/coach_tenures/northwestern.json` · SHA-256 `b2df2c7b507216fcd0131116791e073a39d94c3092d373d01fea7f33fe532ae0`
- `coach_tenure/notre_dame.json`: `data/raw/cfbd/preseason/coach_tenures/notre_dame.json` · SHA-256 `11f2570a5f5b2903a3184fdd90e8316093485abad91d1da9b2d806a20d18bf10`
- `coach_tenure/ohio.json`: `data/raw/cfbd/preseason/coach_tenures/ohio.json` · SHA-256 `7b987ba69ed169f6135f331392a553eb79eb56d0cd92dd82cbd7078d4fe8afda`
- `coach_tenure/ohio_state.json`: `data/raw/cfbd/preseason/coach_tenures/ohio_state.json` · SHA-256 `53ebfa01d0fa28a9ac07ce77a6d3dfe1e1b62bdd37fbbe5f8adf026b1ef04a45`
- `coach_tenure/oklahoma.json`: `data/raw/cfbd/preseason/coach_tenures/oklahoma.json` · SHA-256 `426e694d9348e3019c3346f7fa36e3744fac9449a1ca34b73c1e0897a6c7a627`
- `coach_tenure/oklahoma_state.json`: `data/raw/cfbd/preseason/coach_tenures/oklahoma_state.json` · SHA-256 `0d57c5dfbc3fe57ac631032d46fca91b4cdbfc923ef9293e3e587caac0735b97`
- `coach_tenure/old_dominion.json`: `data/raw/cfbd/preseason/coach_tenures/old_dominion.json` · SHA-256 `d04d51f3f75e62bdb338c0fbabe792c8eb83f80b58fd66500170f4f628c875d3`
- `coach_tenure/ole_miss.json`: `data/raw/cfbd/preseason/coach_tenures/ole_miss.json` · SHA-256 `d4baffc58594d23c504b3e5ec798a1506ff7008998166f8230c48d4453860b0c`
- `coach_tenure/oregon.json`: `data/raw/cfbd/preseason/coach_tenures/oregon.json` · SHA-256 `ff8f46fc0f6027180f5c6f2598fb94e4b53d64da4ce479c120c15ba40d269094`
- `coach_tenure/oregon_state.json`: `data/raw/cfbd/preseason/coach_tenures/oregon_state.json` · SHA-256 `d3e15a8d30703b3fab0ceba4c7b3309edfbac29366d2d9cb813c5f5ea492c468`
- `coach_tenure/penn_state.json`: `data/raw/cfbd/preseason/coach_tenures/penn_state.json` · SHA-256 `3718e170a71a5cee783070a1ca1845f153204b1961154f85e186a355ae989b50`
- `coach_tenure/pittsburgh.json`: `data/raw/cfbd/preseason/coach_tenures/pittsburgh.json` · SHA-256 `361bfa70afd4a61b9722821edd73808b7da8d689251be32138916c03e93d9ac3`
- `coach_tenure/purdue.json`: `data/raw/cfbd/preseason/coach_tenures/purdue.json` · SHA-256 `66be698dc97520899871180cf45d0a28e4430ff50ca062eea2a933b5c0175c15`
- `coach_tenure/rice.json`: `data/raw/cfbd/preseason/coach_tenures/rice.json` · SHA-256 `4129276812c2325be2b4e80e255d9c47b39b25738e25d83747594e55de1ed66a`
- `coach_tenure/rutgers.json`: `data/raw/cfbd/preseason/coach_tenures/rutgers.json` · SHA-256 `6a3d681a6029fc5e9eb5196170acf18d6c63609620478492dc4321182bce1cd2`
- `coach_tenure/sacramento_state.json`: `data/raw/cfbd/preseason/coach_tenures/sacramento_state.json` · SHA-256 `0f831ebe1bac1a3e7a022972fd95a615de8f75453ce126b7d106bb3432a103b6`
- `coach_tenure/sam_houston.json`: `data/raw/cfbd/preseason/coach_tenures/sam_houston.json` · SHA-256 `db3feb4a94eb8c3c52c2daad4677182a3c67aa56bea0a146ea6a0139745e5f42`
- `coach_tenure/san_diego_state.json`: `data/raw/cfbd/preseason/coach_tenures/san_diego_state.json` · SHA-256 `755767b7c4fbdc67f0f2739c3863e7de00f0453d3884fea0c1183a9421d8822d`
- `coach_tenure/san_josé_state.json`: `data/raw/cfbd/preseason/coach_tenures/san_josé_state.json` · SHA-256 `34579acb5cd30df0217c92b0a9acebd432d6da6e8cbd5d04abd1fa4833e02df7`
- `coach_tenure/smu.json`: `data/raw/cfbd/preseason/coach_tenures/smu.json` · SHA-256 `bfd4a88b76cef5555edd7c08241c18bc11ce81147d3b3513310ff2dda318f90f`
- `coach_tenure/south_alabama.json`: `data/raw/cfbd/preseason/coach_tenures/south_alabama.json` · SHA-256 `eab82ed2e8896b046c520388fda303af8c0b5cc89f1425470897a4656b83c116`
- `coach_tenure/south_carolina.json`: `data/raw/cfbd/preseason/coach_tenures/south_carolina.json` · SHA-256 `45ea567a12436c3031adb9a927659a3c80b02ab3bd9a4a33ad2c07987c9e96ba`
- `coach_tenure/south_florida.json`: `data/raw/cfbd/preseason/coach_tenures/south_florida.json` · SHA-256 `2fed34600ed2f3ccc6e88ae140d719d03a7286f9467f9aa4f6a9472cc9c53c37`
- `coach_tenure/southern_miss.json`: `data/raw/cfbd/preseason/coach_tenures/southern_miss.json` · SHA-256 `2f772b11a143ed2a05d240db06b3936cf2c81c5236c4542dcef4311440151b56`
- `coach_tenure/stanford.json`: `data/raw/cfbd/preseason/coach_tenures/stanford.json` · SHA-256 `ba5f66359b3d4cbf9fde5490ddfa3d693b36aaf709800fec99ffa8bc1af78b1b`
- `coach_tenure/syracuse.json`: `data/raw/cfbd/preseason/coach_tenures/syracuse.json` · SHA-256 `1f38c1a8b3e7a1a279fab44b851f2ef657ab62ff4ae3efdbe5a79665d05fc53d`
- `coach_tenure/tcu.json`: `data/raw/cfbd/preseason/coach_tenures/tcu.json` · SHA-256 `9fb35ca047a7274e9d7bc3a562c26d879255cf62cefc2e9dd00b478ea7b564e3`
- `coach_tenure/temple.json`: `data/raw/cfbd/preseason/coach_tenures/temple.json` · SHA-256 `1834bb5bc5db1251e6a2523c9b32d40d5685d6a5b02f167d09e7b186d0dab458`
- `coach_tenure/tennessee.json`: `data/raw/cfbd/preseason/coach_tenures/tennessee.json` · SHA-256 `de6f0eb1f6996f2b9d107ee2579992f087dabb5ddc6127bc3b61d6818f1ec0c9`
- `coach_tenure/texas.json`: `data/raw/cfbd/preseason/coach_tenures/texas.json` · SHA-256 `e43d7cd90f3a9c50739e0ed4c292d537b34feccb663d9322530f1207456631ac`
- `coach_tenure/texas_a&m.json`: `data/raw/cfbd/preseason/coach_tenures/texas_a&m.json` · SHA-256 `71305c105d6d96677fab1543a1823ebb185c1b08bb3a10e83e8e2f30e0d51945`
- `coach_tenure/texas_state.json`: `data/raw/cfbd/preseason/coach_tenures/texas_state.json` · SHA-256 `f513b19787c386665008d9cb8ca31057f4ec6565779ae1d20191a2501b2ef4da`
- `coach_tenure/texas_tech.json`: `data/raw/cfbd/preseason/coach_tenures/texas_tech.json` · SHA-256 `eb5c0a42f5c55467ff07ffe033d88c9b35a467085534d72196b991db4f6f26d3`
- `coach_tenure/toledo.json`: `data/raw/cfbd/preseason/coach_tenures/toledo.json` · SHA-256 `ddba99df359f0fb28c3e78c768d4cece89ac48b01e6467a1432f0ad5d8306bb8`
- `coach_tenure/troy.json`: `data/raw/cfbd/preseason/coach_tenures/troy.json` · SHA-256 `0ae811873a9b03da9b497589ae3f5d6d97543274c4a0ae2f485d5a7f7943b606`
- `coach_tenure/tulane.json`: `data/raw/cfbd/preseason/coach_tenures/tulane.json` · SHA-256 `715064244aa99b42300a5a0e9f1f38e7ae93f81d1e462d09d625f348b7f21b7b`
- `coach_tenure/tulsa.json`: `data/raw/cfbd/preseason/coach_tenures/tulsa.json` · SHA-256 `3c63d23b6963b009a666414708b047dd0f056583ae40f3d42b7027f4193c5880`
- `coach_tenure/uab.json`: `data/raw/cfbd/preseason/coach_tenures/uab.json` · SHA-256 `ccbcd0d0da5108a891a94b6afc1d960027c9f389e5bdc9073a713716aeaf556c`
- `coach_tenure/ucf.json`: `data/raw/cfbd/preseason/coach_tenures/ucf.json` · SHA-256 `25ec65a4fa9e80897ce8ed368014bc136bd5671d677fd2effada4036b0c8c1e0`
- `coach_tenure/ucla.json`: `data/raw/cfbd/preseason/coach_tenures/ucla.json` · SHA-256 `e2cfd6ca1dd42b8606b413a858eaa42523fe8003fa855ee6911aa61f2f2184ae`
- `coach_tenure/uconn.json`: `data/raw/cfbd/preseason/coach_tenures/uconn.json` · SHA-256 `7f999d4518cfd358a1e0cd763a9f178ca429f0f8ea39e461905200cca122b468`
- `coach_tenure/ul_monroe.json`: `data/raw/cfbd/preseason/coach_tenures/ul_monroe.json` · SHA-256 `cfa538772744e07506d0d77ef48be4c075c27727dd14b981e8090acc37f1192b`
- `coach_tenure/unlv.json`: `data/raw/cfbd/preseason/coach_tenures/unlv.json` · SHA-256 `3cea0b194bc1eaacafaae8d0bfd43c72dce72dab09d2b73378aec13a42adf782`
- `coach_tenure/usc.json`: `data/raw/cfbd/preseason/coach_tenures/usc.json` · SHA-256 `6d1c56be8e72e747e363bba5fba78d001304a22b699517761af84b41575cd043`
- `coach_tenure/utah.json`: `data/raw/cfbd/preseason/coach_tenures/utah.json` · SHA-256 `882ce057c01441d2a2488aacaa4e26e2d8adcf633696a6502efdc5d4ba888509`
- `coach_tenure/utah_state.json`: `data/raw/cfbd/preseason/coach_tenures/utah_state.json` · SHA-256 `ae3e7bf1bc1dbf568bdf954a4c4872ffb36f86092fbfebdf9b7d0c52d54f03a9`
- `coach_tenure/utep.json`: `data/raw/cfbd/preseason/coach_tenures/utep.json` · SHA-256 `b6b2ce5b551136b3fc75d8666f6e82837e33011689a649d3f73c9291c6539b39`
- `coach_tenure/utsa.json`: `data/raw/cfbd/preseason/coach_tenures/utsa.json` · SHA-256 `92bda72e92f350dce7c34b105a10c190186b9adb55b5d1f3b2c2dfe2da700426`
- `coach_tenure/vanderbilt.json`: `data/raw/cfbd/preseason/coach_tenures/vanderbilt.json` · SHA-256 `8b5f2d6c8da2939beff0c93762c4400ccd5da435234f04a219caf81c5217fb26`
- `coach_tenure/virginia.json`: `data/raw/cfbd/preseason/coach_tenures/virginia.json` · SHA-256 `a4e473f5181e6942931d03f1fd6cbcb2961dd72209bfff005ff0fccaf8c36542`
- `coach_tenure/virginia_tech.json`: `data/raw/cfbd/preseason/coach_tenures/virginia_tech.json` · SHA-256 `763e947a23299e1b5474c1a62f3573a9893c5945ef57e7f7f041e2aeb3ad7ce3`
- `coach_tenure/wake_forest.json`: `data/raw/cfbd/preseason/coach_tenures/wake_forest.json` · SHA-256 `8b1c23e8486581eec6a2e17c44e79be5f0ca0238443c9ab3a2a5c36ea8e3aade`
- `coach_tenure/washington.json`: `data/raw/cfbd/preseason/coach_tenures/washington.json` · SHA-256 `46387b29f745695b1200a8ae0eeef6740c10d5b03f3034931219c36f11233da9`
- `coach_tenure/washington_state.json`: `data/raw/cfbd/preseason/coach_tenures/washington_state.json` · SHA-256 `a982c37b2b9722f2f7d9bf9298681ed2e1e2f79b8f85c7eb5074f6769ac874e2`
- `coach_tenure/west_virginia.json`: `data/raw/cfbd/preseason/coach_tenures/west_virginia.json` · SHA-256 `6ff8094e0a91cb5a157df84bca3e4aa74871a72fb04a81275d8def68d43d8d4f`
- `coach_tenure/western_kentucky.json`: `data/raw/cfbd/preseason/coach_tenures/western_kentucky.json` · SHA-256 `d281544272370aa861fc27f04315bef56ce11b339a9d29ec56de04d0bcebccff`
- `coach_tenure/western_michigan.json`: `data/raw/cfbd/preseason/coach_tenures/western_michigan.json` · SHA-256 `c6e09409cc8910a7a4c1519c2b2cb145506c86faa60592187c5966c002a5e725`
- `coach_tenure/wisconsin.json`: `data/raw/cfbd/preseason/coach_tenures/wisconsin.json` · SHA-256 `9ea04ce59f033e2e451732dc0a653c253e4d55ca83ddc699f90aae1ec0e94388`
- `coach_tenure/wyoming.json`: `data/raw/cfbd/preseason/coach_tenures/wyoming.json` · SHA-256 `f5695da5eb84cd662a5bcaca1382253fe9cd0f78660dac5309c15b50eba649bf`
- `context13_semantic_implementation`: `src/gippyrank/context_prior_v1_3.py` · SHA-256 `d6ae41c967c848a6d82e13893473b7bd57ac6be99a2465b06fc0b6a7a07f2438`
- `context13_transfer_features`: `data/processed/preseason/context_v1_3_candidate/historical_transfer_features.csv` · SHA-256 `3b3680cf63806033cf6d67813a1359ada455e8e7528bf310a27b37d5cda7a726`
- `context13_transfer_provenance`: `data/processed/preseason/context_v1_3_candidate/historical_transfer_features.provenance.json` · SHA-256 `9e930fabdd434154727c8f7d1d2a107fda3762d96284ca2aabc7594ecfd3927f`
- `context_features`: `data/processed/preseason/team_season_features.csv` · SHA-256 `fd6f171131e38b852b94619338c5580e5f4e8c19f45c74e8135b11b20b1a2019`
- `direct_rank_implementation`: `src/gippyrank/preseason.py` · SHA-256 `f47d4b5bf68b233f15f26204c130eaeba719c1355547d11b5d49ef00bb267831`
- `issue181_eligible_co_start_pairs`: `data/processed/offensive_line_co_start_pilot_issue_181/eligible_co_start_pairs.csv` · SHA-256 `6e1b62fdf00099fc3e0efd045279f3160b8412272be9bc31b7c30c08f8f9edd3`
- `issue185_identity_crosswalk`: `data/research/offensive_line_target_pool_validation_issue_185/identity_crosswalk.csv` · SHA-256 `d43705e1fc5d87706872da8eeb63c79772d8938f0866de849f09dc6cda1c14a2`
- `issue185_player_comparison`: `data/research/offensive_line_target_pool_validation_issue_185/results/player_comparison.csv` · SHA-256 `91ee98615abb646ea8fd01dc6f8973800e880e89855e7242e6f6b48544382272`
- `issue185_team_season_summary`: `data/research/offensive_line_target_pool_validation_issue_185/results/team_season_summary.csv` · SHA-256 `d51ac81a69f99ac385ece00da62c376bbefe3a74cfc4687d5a6e80b98d86ed72`
- `rank_distributions`: `data/processed/modeling/team_season_rank_distributions.csv` · SHA-256 `03e4e372017aedb45392ff7d790cea25f461db9b6dc846514c762ff46e2cab00`
- `roster_normalized_ol_player_seasons`: `data/processed/offensive_line_shared_roster_issue_183/normalized_ol_player_seasons.csv.gz` · SHA-256 `8577754a6a392378ec1cb63ba3cf13e521cf38c584b900ae44b4dab2de2cec4a`
- `roster_normalized_player_seasons`: `data/processed/offensive_line_shared_roster_issue_183/normalized_roster_player_seasons.csv.gz` · SHA-256 `5d3a09ba9ee979db095086ce1eaca852807efe6d6289f6fe66cdb32f50772a6a`
- `roster_pairwise_continuity`: `data/processed/offensive_line_shared_roster_issue_183/pairwise_shared_roster_continuity.csv.gz` · SHA-256 `d21252cba4d81ecf23e60f55afd487ddd22e780c10c4ae6f4b9d2b0615cde981`
- `roster_team_season_summaries`: `data/processed/offensive_line_shared_roster_issue_183/team_season_summaries.csv` · SHA-256 `5c6dd0d9b533e95900e81d626ae36a32fd05a7a2f9ab091fdcc9b0abd04d5ed3`
