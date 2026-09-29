# syntax=docker/dockerfile:1
# Cited MASH/MASLD briefing agent. Demo and learning exercise; not a clinical tool.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# uv: compile bytecode, copy (not hardlink) into the venv, and use the image's Python.
# Output locations are set by environment so the container needs no flags for them.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:${PATH}" \
    MASH_AGENT_OUT_DIR=/out \
    MASH_AGENT_EVAL_DIR=/out/eval_results \
    MASH_AGENT_CACHE_DIR=/cache

WORKDIR /app

# 1. Dependencies only: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. The project (README.md is required by the package metadata; evals/ holds the question set).
COPY README.md ./
COPY src ./src
COPY evals ./evals
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run as a non-root user; /out (results) and /cache (API response cache) are mount points.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /out /cache \
    && chown app:app /out /cache
USER app
VOLUME ["/out", "/cache"]

ENTRYPOINT ["mash-agent"]
CMD ["--help"]
