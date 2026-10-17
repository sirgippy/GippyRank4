# Issue 213: Context dependent likelihood forecast experiment

**Conclusion: Retain unmodified Context / no useful replacement established**

This is a fixed exploratory replay of 2023–2025 using #195's saved forward fits. It is not an untouched confirmation and does not authorize production promotion.

Frozen protocol SHA-256: `139c220cddb8ea07a5af997aafbe152a77b3c1f9d3af8ed701d3e29d9b7c37fb`.

## Primary margin CRPS

CRPS is lower when forecasts improve. Deltas below are candidate minus reference; negative favors the candidate.

| Comparison | Pooled equal-game delta | Equal-season delta | Seasons lower | Practical rule |
|---|---:|---:|---:|---|
| Generic vs unmodified Context | 0.01551 | 0.01552 | 0/3 | fail |
| Context correction vs unmodified Context | -0.02645 | -0.02652 | 3/3 | fail |
| Context correction vs generic | -0.04196 | -0.04204 | 3/3 | fail |

The comparisons share the same retained games, Context priors, and cutoff evidence. The experiment uses the saved 2023 fit trained on 2022, 2024 fit trained on 2022–2023, and 2025 fit trained on 2022–2024.

## Replay and calibration coverage

Primary forecasts: 2264 rolling game rows per arm; 2264 separate preseason rows per arm.
Corrected pre-bowl team-checkpoint scores: 20154 rows across arms and stages.
#201 baseline parity: passed across 4528 per-game rows; max numeric difference 0 (limit 1e-08).

Season, phase, margin NLL, win Brier, median absolute error, 50/80/95% coverage and widths, secondary ranking scores, and team/game gain concentration are in the accompanying CSV and compressed JSONL artifacts.
The complete normalized rank PMFs are losslessly preserved in `posterior_pmfs.jsonl.gz`; per-season and per-cutoff entropy, effective rank count, maximum rank mass, and top-10 rank mass are in `ranking_pmf_concentration.csv`.
Each future-game Student-t rank mixture is losslessly reconstructible from `predictive_mixtures.jsonl.gz`, its referenced team PMFs, and `predictive_model_parameters.json`.

## Guardrails and interpretation

Early preseason and pre-Thanksgiving CRPS and coverage-error guardrails are reported in `decision.json`. They are descriptive safeguards, not significance tests. Repeated games, teams, and checkpoints are dependent.

The frozen decision rule yields: **Retain unmodified Context / no useful replacement established**. Any promising result needs a new protocol that freezes predictions before the next season's outcomes are available.

A score improvement would not prove that rankings are wrong, that Context belongs in production likelihoods, or that the ranking measurements recover latent true ranks.
