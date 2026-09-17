# Talent-replacement Context experiment

This focused study tests whether returning production is more useful when
conditioned on year-over-year roster-talent movement. It leaves production C
1.2 and the frozen 2026 artifacts unchanged.

## Design

The frozen production Context predictions are C0. C1 adds
`talent_delta` to the Context location equation. The primary C2 adds a
training-standardized interaction between `returning_pct_ppa` and
`talent_delta`; passing, receiving, and rushing interactions are secondary
diagnostics.

`talent_delta` is defined as:

```text
z(current Team Talent within the target season)
- z(previous Team Talent within the previous season)
```

The model is fit using outcomes through 2021 and scored without refitting on
the unchanged 2022–2025 FBS population. It preserves C 1.2's preprocessing,
penalty (`0.25`), H-only scale equation, scoring, and exact H fallback for
rank-history cold starts.

The refit C0 control agrees with the stored production artifact within the
study tolerances: maximum regular-row PMF difference `1.34e-4` and maximum
metric difference `5.41e-6`.

## Held-out results

Negative deltas favor the candidate.

| Variant | NLL | ΔNLL | CRPS | ΔCRPS | Expected MAE | Median MAE | 80% coverage | 80% width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 4.5423 | — | 0.1119 | — | 20.16 | 20.59 | 0.825 | 71.9 |
| C1: talent delta | 4.5485 | +0.0062 | 0.1125 | +0.0006 | 20.23 | 20.65 | 0.825 | 71.9 |
| C2: total RP interaction | 4.5528 | +0.0105 | 0.1132 | +0.0012 | 20.39 | 20.87 | 0.819 | 71.8 |
| C2: passing interaction | 4.5480 | +0.0057 | 0.1125 | +0.0005 | 20.25 | 20.65 | 0.824 | 72.0 |
| C2: receiving interaction | 4.5474 | +0.0051 | 0.1125 | +0.0006 | 20.27 | 20.73 | 0.824 | 72.0 |
| C2: rushing interaction | 4.5501 | +0.0078 | 0.1126 | +0.0007 | 20.24 | 20.67 | 0.823 | 71.8 |

Year-by-year NLL deltas versus C0 were:

| Season | C1 | C2 total | C2 passing | C2 receiving | C2 rushing |
|---:|---:|---:|---:|---:|---:|
| 2022 | +0.0081 | +0.0153 | +0.0120 | +0.0091 | +0.0068 |
| 2023 | −0.0083 | −0.0009 | −0.0079 | +0.0001 | −0.0052 |
| 2024 | +0.0007 | +0.0159 | −0.0043 | +0.0034 | +0.0034 |
| 2025 | +0.0239 | +0.0116 | +0.0227 | +0.0078 | +0.0258 |

The complete generated report includes year-by-year CRPS, expected-rank MAE,
median-rank MAE, interval coverage, interval width, model metadata, and the
interaction diagnostic under `data/processed/talent_replacement/`.

## Interpretation

Talent change alone does not improve recent performance. The primary RP ×
talent-delta interaction does not improve beyond C1 or C0, and does not
substantially repair the 2024–2025 degradation. Its modeled RP q25→q75
location effect was `−0.215` at declining talent, `−0.307` at flat talent, and
`−0.428` at improving talent. Since lower rank-z is better, this is opposite
the hypothesized pattern that returning production should matter more when
talent declines.

This result does not show that returning production is useless or that
transfers are irrelevant. It says that this season-normalized Team Talent
change and the focused interaction do not explain the recent C degradation
under the established evaluation protocol. No production C 1.3 promotion is
recommended.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/gippyrank-uv-cache uv run python \
  scripts/investigate_talent_replacement.py
```

Use `--data-root` when processed inputs are cached in another checkout and
`--output-dir` to choose the research artifact directory.
