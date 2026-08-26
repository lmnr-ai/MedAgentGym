# CPU-only image: the harness talks to hosted LLM APIs (Azure AI Foundry / OpenAI),
# so there is no CUDA runtime, no torch and no local model serving here.
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/.venv \
    PATH="/home/.venv/bin:$PATH"

# build-essential + git are needed to build the few bioinformatics wheels that
# have no manylinux build, and because agents may `pip install` at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home

# Dependency layer: only invalidated when the lockfile changes.
COPY pyproject.toml uv.lock .python-version /home/
RUN uv sync --locked --extra tasks --no-install-project

COPY . /home/
RUN chmod +x /home/entrypoint.sh

ENTRYPOINT ["/home/entrypoint.sh"]
