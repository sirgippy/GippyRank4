# Lean Context Candidate L validation

## Predeclared design and promotion criteria

L is fixed as H 1.1 historical features plus six recruiting features (current class rank/points, 2/3/4-year points means, trend) and four returning-production features (total, passing, receiving, rushing). Context enters location only; its scale uses the H features. Talent, coaching, transfers, interactions, continuity measures, and poll/media inputs are excluded.

Promotion requires a mean rolling NLL improvement of at least 0.005 versus H, non-worse CRPS, no worse than 0.05 80% coverage, no recurrent catastrophic regression, and either a clear C improvement (L − C NLL ≤ −0.005) or prediction within 0.002 NLL of C with lower complexity. These thresholds were written before results.

## Results

Observed-common targets: [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]. L − H mean annual NLL is -0.01665 (8 wins, 4 losses); L − C is +0.01756 (1 wins, 11 losses). Negative favors L.

Raw annual values, paired team losses, calibration, complexity, disagreement, and the nested prospective simulation are retained in the companion CSV/JSON artifacts. Bootstrap-style inference is deliberately omitted: the small number of seasonal clusters makes raw annual results the primary evidence.

## Independence and status

The 2018–2025 ablation generated this hypothesis, so these results are a prospective-style historical validation, not independent confirmation. The nested selector uses only earlier rolling targets. The 2025 row is descriptive only. No feature selection or retuning was performed. H 1.1, C 1.2, and frozen 2026 H/C artifacts were not modified; this study emits no 2026 forecast or outcome-based artifact.

## Recommendation

Recommendation C: L improves over H but not over C; keep C 1.2.
