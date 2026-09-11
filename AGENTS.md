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

## Python environment

- This project uses `uv`.
- Use the Python version pinned by `.python-version`.
- Run Python commands through `uv run`, for example:
  - `uv run python ...`
  - `uv run pytest`
  - `uv run ruff check .`
- Add dependencies with `uv add`, not `pip install`.
- Do not rely on Debian's system `python3` except for environment diagnostics.

## Browser/UI validation

- The static site has a repo-local Playwright harness using Chromium, with desktop and narrow/mobile projects.
- For changes affecting user-facing files under `site/` or `site/assets/`, make a best effort to run `npm run test:ui` in addition to the normal Python and Ruff checks.
- The browser command starts the local `site/` server itself through the configured Python environment; do not claim browser validation is unavailable without first attempting the configured Playwright checks.
- For substantive layout changes, add or update browser-level regression coverage rather than relying solely on static tests.
- In a fresh environment, install the Node dependencies and Chromium/Linux browser dependencies before running the UI checks; see `docs/browser_validation.md`.

## Validation in Codex workers

- The Codex Linux/WSL sandbox has a known limitation with Python asyncio cross-thread wakeups used by Starlette/FastAPI `TestClient`; `tests/test_api.py` may stall indefinitely there.
- Do not change application code, API code, pytest fixtures, dependencies, or test behavior to work around this sandbox-specific hang.
- In a sandboxed Codex worker, run the sandbox-compatible local suite with:
  `uv run pytest -q --ignore=tests/test_api.py`
- Validate `tests/test_api.py` and the complete suite through CI or an explicitly approved unsandboxed/escalated command. If an API test run goes silent and hangs, terminate it rather than waiting indefinitely.
- Validation split: sandboxed Codex worker → run the sandbox-compatible suite locally and rely on CI or approved unsandboxed execution for API tests; normal WSL/CI → run the full suite.
