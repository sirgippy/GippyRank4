# Coverage-restricted context-feature ablation

## Design

All comparisons use raw observed coverage chosen before any imputation. For each rolling target, both H and its candidate are trained on eligible rows strictly before the target and predicted on identical team-season keys. Location-only context is primary. H 1.1 and C 1.2 remain immutable references; 2026 is absent from this research build.

## Headline results

- Coach tenure on its natural raw-coverage population: mean ΔNLL -0.0006; 10 wins/9 losses.
- Current recruiting: mean ΔNLL -0.0105; 15 wins/4 losses; recruiting history: mean ΔNLL -0.0124; 15 wins/3 losses; all recruiting: mean ΔNLL -0.0110; 16 wins/2 losses.
- Talent: mean ΔNLL -0.0028; 4 wins/4 losses; total/passing/skill returning production: mean ΔNLL -0.0186; 6 wins/3 losses, mean ΔNLL -0.0001; 6 wins/3 losses, mean ΔNLL -0.0123; 7 wins/2 losses.
- On one exact all-context common population, recruiting+Talent+returning: mean ΔNLL -0.0222; 6 wins/2 losses; full frozen-spec C-equivalent: mean ΔNLL -0.0185; 6 wins/2 losses.
- Talent×total-returning interaction: mean ΔNLL -0.0181; 5 wins/3 losses; Talent×passing-returning: mean ΔNLL +0.0272; 3 wins/5 losses.
- Interpret annual wins/losses and descriptive season-bootstrap ranges conservatively; the coverage-era target count is intentionally limited.
- No production H/C specification, frozen 2026 PMF, or 2026 outcome was modified or accessed.

## Interpretation limits

The season-level bootstrap ranges in `summary.json` are descriptive because the common coverage era has few seasons. Raw annual ΔNLL points in `annual_ablation_metrics.csv` are the primary robustness evidence. No candidate is proposed as C 1.3 here.

## Artifacts

- `coverage_populations.csv` / `.json` — raw, pre-imputation coverage definitions and annual counts.
- `annual_ablation_metrics.csv`, `candidate_summary.csv`, and `same_population_comparisons.csv` — paired held-out scores with the `same_population_keys` invariant.
- `per_team_losses.csv`, `interaction_results.csv`, `missingness_results.csv`, `adjustment_magnitude.csv`, `decomposition_2025.csv`, and `team_examples_2025.csv` — auditable diagnostics.
- `plots/` — raw coverage, ablations, annual effects, interactions, adjustment-risk, and 2025 diagnostics.

## 2025 component disagreement

Component directions are calculated on the coach-free RTP observed population. Returning-only ΔNLL averaged 0.0002 for 36 agreement cases and 0.0772 for 95 conflict cases. Team examples retain rows with unavailable coach coverage rather than silently excluding them.

## 2025 component disagreement

Component directions are calculated on the coach-free RTP observed population. Returning-only ΔNLL averaged 0.0002 for 36 agreement cases and 0.0772 for 95 conflict cases. Team examples retain rows with unavailable coach coverage rather than silently excluding them.
