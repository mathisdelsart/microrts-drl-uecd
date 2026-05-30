# Minimal CPU-only image for MicroRTS Deep-RL Agent.
#
# Build:  docker build -t microrts-drl-uecd .
# Run:    docker run --rm microrts-drl-uecd                  # prints CLI help
#         docker run --rm microrts-drl-uecd evaluate \
#             --agent data/agents/UECD-SingleMap-Best \
#             --opponent CoacAI --nb_games 10 --max_steps 1000
#
# GPU note: this image ships CPU-only torch (no nvidia-* wheels). For GPU
# training, swap the `torch --index-url ...` line for the default
# CUDA-enabled wheel and add a CUDA base image.
FROM python:3.10-slim

# Java 17 for the JNI bridge + microrts engine.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the repo. .dockerignore drops data/, outputs/, dissertation/,
# cog-2026-paper/, .git/ etc. so the build context stays small.
COPY . .

# CPU-only torch first to skip ~3 GB of nvidia-* wheels (same trick as CI).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dev]"

# Rebuild the Java<->Python bridge from src/ against the vendored microrts.jar.
RUN bash microrts_agent/microrts/build_bridge.sh

# Smoke-test the import path at build time so a broken image fails fast.
RUN python -c "import microrts_agent; print('microrts_agent ready')"

ENTRYPOINT ["python", "-m", "microrts_agent"]
CMD ["--help"]
