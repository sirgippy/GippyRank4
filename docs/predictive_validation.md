# Posterior-predictive game validation

Issue 41 uses the frozen Historical Likelihood V1 margin model without
refitting or calibrating it.  For a scheduled game, the published distribution
is the exact finite mixture

`P(M | snapshot) = ΣᵢΣⱼ P(M | rᵢ, rⱼ, site, pairing) P(rᵢ | snapshot) P(rⱼ | snapshot)`.

Each component is the existing Student-t V1 margin distribution.  Win
probabilities are derived from the same mixture CDF; a point mass at zero, if a
future discrete likelihood supplies one, receives half credit for each side.
The production Student-t mixture is continuous, so its stored tie probability
is zero.  Central intervals are deterministic numerical inversions of that
mixture CDF at the 0.25/0.75, 0.10/0.90, and 0.025/0.975 quantiles.

## Leakage-safe sanity check

The check reused the existing `build_performance_v1.py` cutoff preparation and
posterior inference for seasons 2022–2025.  It used the standard mid-season
cutoff (cutoff index 3) for each season, scored only completed games strictly
after each cutoff, and evaluated the Context and History posterior families
separately.  The four cutoffs produced 2,700 game instances per family.  No
coefficients, priors, probability calibration, or interval definitions were
changed for this check.

The check is reproducible with
`uv run python scripts/validate_predictive_game_v1.py --cutoff-index 3`.

| Predictive family | N | Win Brier | Margin MAE | Margin NLL | 50% coverage | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context | 2,700 | 0.1905 | 12.8673 | 4.2096 | 48.4% | 78.6% | 94.6% |
| History | 2,700 | 0.1910 | 12.9089 | 4.2112 | 48.4% | 78.8% | 94.5% |

The established full 2022–2025 validation panel in
`data/processed/performance_v1/report.md` reports Predictive C/H all-future
win Brier of 0.1983/0.2001, margin MAE of 13.1830/13.2820, and marginalized
margin NLL of 4.2384/4.2439 over 22,571 common team-game keys.  The interval
figures above are the compact exact-mixture coverage check requested for this
feature; no post-hoc calibration layer was added.

## Published payload impact

The generated manifest records prediction bytes per selected snapshot and
separates lazy team-page data from the initial rankings payload:

- additional lazy future-prediction payload: 7,027,363 bytes;
- total lazy team-season payload after export: 14,912,752 bytes, versus
  7,121,534 bytes before predictions;
- initial rankings-page payload: 1,785,971 bytes, unchanged.

Predictions are referenced by game ID in each team schedule, so the canonical
summary is not duplicated for the two teams' pages.
