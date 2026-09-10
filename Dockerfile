FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    GIPPYRANK_PUBLICATION_DATA=/app/site/data

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
COPY site/data ./site/data

RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["sh", "-c", "exec uv run --no-dev uvicorn gippyrank.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
