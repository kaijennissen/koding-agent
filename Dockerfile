FROM python:3.14.4-slim-bookworm

RUN apt-get update -y && \
apt-get install -y curl jq vim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/app
WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY src/ src/

RUN uv sync --locked --no-dev --no-editable

# Run as an unprivileged user. Everything above stays root-owned, so the agent
# cannot modify its own harness. The root filesystem is mounted read-only at
# runtime, so HOME (and with it uv's cache) points at the writable /tmp.
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --no-create-home --home-dir /tmp --shell /bin/bash agent
ENV HOME=/tmp
# The host as seen from the container (see README "Run the agent").
ENV OLLAMA_HOST=http://host.docker.internal:11434
USER agent

# Model-written commands start in the host-provided clone, not in the
# read-only harness copy under /app. See README "Run the agent".
WORKDIR /workspace

CMD ["/bin/bash"]
