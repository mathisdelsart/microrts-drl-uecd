# Minimal CPU-only image for MicroRTS Deep-RL Agent.
#
# Build:
#   docker build -t microrts-drl-uecd .
#   # Skip the ~225 MB RAISocketAI wheel (tournament against RAISocketAI bot
#   # won't work, every other feature still does):
#   docker build --build-arg SKIP_RAISOCKETAI=1 -t microrts-drl-uecd .
#
# Run out of the box (eval against the shipped agents):
#   docker run --rm microrts-drl-uecd                          # CLI help
#   docker run --rm microrts-drl-uecd evaluate \
#       --agent data/agents/UECD-SingleMap-Best \
#       --opponent CoacAI --nb_games 10 --max_steps 1000
#
# Run with extra data mounted from the host (the image only bakes in
# data/agents/; other data/ subtrees are dropped by .dockerignore to keep
# the image small, mount only what you need):
#   # BC training (needs the teacher dataset under data/BC/):
#   docker run --rm -v $(pwd)/data/BC:/app/data/BC microrts-drl-uecd \
#       bc train --data 'data/BC/training/bc_chunk_*.npz' ...
#   # Round-robin tournament (needs tournament_configs/ shipped + extra agents):
#   docker run --rm -v $(pwd)/data:/app/data microrts-drl-uecd \
#       tournament run single_map
#   # Training (writes checkpoints/logs to outputs/):
#   docker run --rm -v $(pwd)/outputs:/app/outputs microrts-drl-uecd \
#       train --exp-name docker-smoke --total-timesteps 100000
#
# GPU note: this image ships CPU-only torch (no nvidia-* wheels). For GPU
# training, use Dockerfile.gpu (CUDA 12.9 + cuDNN base).
#
# Security note: image runs as root. Standard for ML research images, and
# avoids UID/GID friction with bind-mounted host dirs. Switch to a non-root
# USER if deploying as a long-running service.
# Pinned to the bookworm variant: Debian trixie (the current `:slim` default
# as of late 2026) renamed `openjdk-17-jdk-headless` and the apt install
# below fails. Bookworm still ships it and is also bumped by Dependabot.
FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/mathisdelsart/microrts-drl-uecd"
LABEL org.opencontainers.image.description="Deep-RL agent for MicroRTS (UCLouvain master's thesis), CPU-only image."
LABEL org.opencontainers.image.licenses="MIT"

# Build-time only (does not persist to the runtime image).
ARG DEBIAN_FRONTEND=noninteractive
ARG SKIP_RAISOCKETAI=0
ARG RAI_WHEEL_URL="https://github.com/mathisdelsart/microrts-drl-uecd/releases/download/assets-rai-v0.2.1/rl_algo_impls-0.2.1-py3-none-any.whl"
ARG RAI_WHEEL_SHA256="1e0a60133f4b96fa95f4331e258fd20495d2209d88c319116ac1bd19431e71d1"

# Java 17 for the JNI bridge + microrts engine; curl for the RAISocketAI wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        ca-certificates \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer-cache trick: install torch first (rarely changes), then install our
# deps from pyproject.toml + a stub package, then copy the real source. This
# way pure-code changes only invalidate the final COPY layer, not pip install.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN mkdir -p microrts_agent \
    && touch microrts_agent/__init__.py \
    && pip install --no-cache-dir -e ".[dev,tournament]"

# RAISocketAI wheel (~225 MB). Hash-verified. Skip with --build-arg
# SKIP_RAISOCKETAI=1 if you don't need to face that bot in tournaments.
RUN if [ "$SKIP_RAISOCKETAI" != "1" ]; then \
        echo "Fetching RAISocketAI wheel..." \
        && curl -L --fail -o /tmp/rl_algo_impls.whl "$RAI_WHEEL_URL" \
        && echo "${RAI_WHEEL_SHA256}  /tmp/rl_algo_impls.whl" | sha256sum -c - \
        && pip install --no-deps --no-cache-dir /tmp/rl_algo_impls.whl \
        && rm /tmp/rl_algo_impls.whl; \
    else \
        echo "RAISocketAI skipped (SKIP_RAISOCKETAI=1)."; \
    fi

# Now copy the real source (only this layer is invalidated by code changes).
COPY . .

# Rebuild the Java<->Python bridge from src/ against the vendored microrts.jar.
RUN bash microrts_agent/microrts/build_bridge.sh

# Smoke-test the import path at build time so a broken image fails fast.
RUN python -c "import microrts_agent; print('microrts_agent ready')"

ENTRYPOINT ["microrts-agent"]
CMD ["--help"]
