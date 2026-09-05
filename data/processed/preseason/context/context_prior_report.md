# GippyRank4 Preseason Context Prior V1.2

H (`history_prior` 1.1) remains the unchanged rank-history-only sibling. C (`context_prior` 1.2) adds only approved preseason context. 2022--2025 is a backtest, not a single annual fit.

## Provenance

| Family | Classification | Evidence |
|---|---|---|
| coach_change | exploratory-unresolved | An undated target-year end can be in-season, so a safe historical comparison is not available. |
| coaching_continuity | production-safe-by-construction | A continuous tenure must prove the coach active by the August 15 cutoff. |
| rank_history | production-safe-by-construction | Each H target uses rank distributions from seasons strictly before the target. |
| recruiting_class | production-safe-by-semantic-definition | CFBD defines year as recruiting class year. Rank and points evaluate a signed class, not later college performance; that class is finalized before its football season. |
| returning_production | production-safe-with-retrospective-stability-caveat | It is previous-season player production filtered by return status into the named season. Target-season games cannot change its previous-production component. |
| team_talent_composite | production-safe-with-retrospective-stability-caveat | CFBD describes a season's 247Sports roster recruiting-talent composite, not a target-season performance statistic. |
| transfers | rejected | Portal entry, destination, enrollment, eligibility, and late moves are genuinely time-varying; no frozen cutoff roster reconstruction exists here. |

## Development ablations (2018--2021, exact same H/C keys)

| Candidate | Mode | N | H NLL | C NLL | ΔNLL | H CRPS | C CRPS |
|---|---|---:|---:|---:|---:|---:|---:|
| C0_history_only | exact H | 513 | 4.5197 | 4.5197 | 0.0000 | 0.1160 | 0.1160 |
| C1_coach_tenure_both | both | 513 | 4.5197 | 4.5067 | -0.0129 | 0.1160 | 0.1143 |
| C1_coach_tenure_location | location | 513 | 4.5197 | 4.5045 | -0.0151 | 0.1160 | 0.1142 |
| C1_coach_tenure_scale | scale | 513 | 4.5197 | 4.5209 | 0.0013 | 0.1160 | 0.1161 |
| C2_recruiting_both | both | 513 | 4.5197 | 4.5182 | -0.0015 | 0.1160 | 0.1164 |
| C2_recruiting_location | location | 513 | 4.5197 | 4.5193 | -0.0004 | 0.1160 | 0.1163 |
| C2_recruiting_scale | scale | 513 | 4.5197 | 4.5210 | 0.0013 | 0.1160 | 0.1162 |
| C3_team_talent_both | both | 513 | 4.5197 | 4.5173 | -0.0023 | 0.1160 | 0.1163 |
| C3_team_talent_location | location | 513 | 4.5197 | 4.5196 | -0.0001 | 0.1160 | 0.1163 |
| C3_team_talent_scale | scale | 513 | 4.5197 | 4.5187 | -0.0010 | 0.1160 | 0.1160 |
| C4_returning_components_both | both | 513 | 4.5197 | 4.5036 | -0.0161 | 0.1160 | 0.1138 |
| C4_returning_components_location | location | 513 | 4.5197 | 4.5047 | -0.0149 | 0.1160 | 0.1138 |
| C4_returning_components_scale | scale | 513 | 4.5197 | 4.5179 | -0.0018 | 0.1160 | 0.1159 |
| C4_returning_passing_both | both | 513 | 4.5197 | 4.5186 | -0.0011 | 0.1160 | 0.1160 |
| C4_returning_passing_location | location | 513 | 4.5197 | 4.5195 | -0.0002 | 0.1160 | 0.1160 |
| C4_returning_passing_scale | scale | 513 | 4.5197 | 4.5185 | -0.0012 | 0.1160 | 0.1159 |
| C4_returning_total_both | both | 513 | 4.5197 | 4.4934 | -0.0263 | 0.1160 | 0.1125 |
| C4_returning_total_location | location | 513 | 4.5197 | 4.4976 | -0.0221 | 0.1160 | 0.1128 |
| C4_returning_total_scale | scale | 513 | 4.5197 | 4.5154 | -0.0042 | 0.1160 | 0.1159 |
| C5_recruiting_returning_both | both | 513 | 4.5197 | 4.4912 | -0.0285 | 0.1160 | 0.1130 |
| C5_recruiting_returning_location | location | 513 | 4.5197 | 4.4935 | -0.0261 | 0.1160 | 0.1129 |
| C5_recruiting_returning_scale | scale | 513 | 4.5197 | 4.5147 | -0.0050 | 0.1160 | 0.1160 |
| C6_recruiting_talent_returning_both | both | 513 | 4.5197 | 4.4916 | -0.0281 | 0.1160 | 0.1131 |
| C6_recruiting_talent_returning_location | location | 513 | 4.5197 | 4.4938 | -0.0258 | 0.1160 | 0.1130 |
| C6_recruiting_talent_returning_scale | scale | 513 | 4.5197 | 4.5155 | -0.0041 | 0.1160 | 0.1160 |
| C6_roster_context_with_coaching_both | both | 513 | 4.5197 | 4.4864 | -0.0332 | 0.1160 | 0.1125 |
| C6_roster_context_with_coaching_location | location | 513 | 4.5197 | 4.4835 | -0.0361 | 0.1160 | 0.1121 |
| C6_roster_context_with_coaching_scale | scale | 513 | 4.5197 | 4.5159 | -0.0038 | 0.1160 | 0.1161 |
| C6_talent_returning_both | both | 513 | 4.5197 | 4.4918 | -0.0278 | 0.1160 | 0.1130 |
| C6_talent_returning_location | location | 513 | 4.5197 | 4.4957 | -0.0240 | 0.1160 | 0.1130 |
| C6_talent_returning_scale | scale | 513 | 4.5197 | 4.5166 | -0.0030 | 0.1160 | 0.1159 |

## Frozen selection

Selected C: **C6_roster_context_with_coaching_location**. Before test data, the rule required ΔNLL ≤ -0.005, non-worse CRPS, and ≥75% favorable season-bootstrap resamples.

## Untouched 2022--2025 H vs C

N=534; H NLL 4.5442, C NLL 4.5423, ΔNLL -0.0018; H CRPS 0.1137, C CRPS 0.1119.

| Season | H NLL | C NLL | ΔNLL |
|---|---:|---:|---:|
| 2022 | 4.5276 | 4.4669 | -0.0607 |
| 2023 | 4.4902 | 4.4722 | -0.0180 |
| 2024 | 4.5882 | 4.5840 | -0.0042 |
| 2025 | 4.5696 | 4.6425 | 0.0729 |

## Annual inference

Annual 2026 inputs are outcome-free rows: completed history through 2025, the frozen 2026 FBS universe, and approved preseason context only. Missing context is handled with training-only imputation and indicators; rank-history cold starts retain exact H fallback.
