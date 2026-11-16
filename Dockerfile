FROM python:3.14.10-slim-bookworm

# nodejs: pyright's PyPI wrapper uses a system node instead of downloading one
# into the noexec /tmp.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
	curl \
	jq \
	vim \
	git \
	nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .agents ./

# The harness venv lives in /opt/app. UV_PROJECT_ENVIRONMENT is set only for
# these build steps, so at runtime `uv run` in /workspace uses the clone's own
# /workspace/.venv (with dev tools) instead of the read-only harness venv.
RUN UV_PROJECT_ENVIRONMENT=/opt/app uv sync --locked --no-dev --no-install-project

COPY src/ src/

RUN UV_PROJECT_ENVIRONMENT=/opt/app uv sync --locked --no-dev --no-editable

# Run as an unprivileged user. Everything above stays root-owned, so the agent
# cannot modify its own harness. The root filesystem is mounted read-only at
# runtime, so HOME points at the writable /tmp.
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --no-create-home --home-dir /tmp --shell /bin/bash agent
ENV HOME=/tmp
# The host as seen from the container (see README "Run the agent").
ENV OLLAMA_HOST=http://host.docker.internal:11434
# Dev tools (uv, ruff, pyright, pytest, pre-commit) in the workspace:
# - use the image's Python and never download one (it would not fit in /tmp);
# - keep caches on /workspace, because /tmp is small and mounted noexec.
ENV UV_PYTHON=/usr/local/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/workspace/.uv-cache \
    UV_LINK_MODE=copy \
    PRE_COMMIT_HOME=/workspace/.pre-commit-cache
USER agent

# Model-written commands start in the host-provided clone, not in the
# read-only harness copy under /app. See README "Run the agent".
WORKDIR /workspace

CMD ["/bin/bash"]
