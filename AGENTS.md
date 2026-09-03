# GippyRank4

GippyRank4 is a research project for probabilistic college-football ranking.

## Core modeling principles

- The primary model should infer probability distributions over actual season outcomes/rank positions.
- Do not introduce an intermediate scalar team-strength or Elo-style rating without explicit discussion.
- Historical ranking observations are measurements, not unquestioned ground truth.
- Preserve uncertainty rather than collapsing distributions prematurely.
- Keep "team quality inference" separate from any later "resume" projection.
- Game order should not affect the base season-wide inference model unless chronology is explicitly modeled.

## Data principles

- Preserve raw downloaded source data unchanged.
- Normalize and transform data in separate processing steps.
- Keep data acquisition separate from statistical modeling.
- Record source, season, coverage, and missingness.
- Use stable canonical team identifiers and explicit aliases.
- Never commit API keys, tokens, credentials, or secrets.

## Engineering

- Use Python 3.12.
- Use uv for dependency and environment management.
- Reusable code belongs under src/gippyrank.
- Tests belong under tests.
- Notebooks are exploratory only; reusable logic must be moved into src.
- Prefer typed data models and deterministic scripts.
- Add tests for parsers, normalization, and inference invariants.
- Use ruff for linting/formatting.
