#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Setup DOCKER image(s) for MicroRTS Master Thesis project.
#
#  Builds the requested image(s) from the repo's Dockerfile(s) and runs a
#  one-shot smoke test (`microrts-agent --help`) inside.
#
#  Usage:
#    bash setup/docker.sh                  # CPU image (default)
#    bash setup/docker.sh gpu              # GPU image (requires nvidia-container-toolkit)
#    bash setup/docker.sh both             # build both
#    bash setup/docker.sh --help
#
#  Requirements:
#    - Docker daemon running. For the GPU image, also nvidia-container-toolkit
#      installed on the host and a working `docker run --gpus all` path.
# ──────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CPU_IMAGE="microrts-drl-uecd"
GPU_IMAGE="microrts-drl-uecd:gpu"

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

build_cpu() {
    echo "========================================"
    echo "  Building CPU image: $CPU_IMAGE"
    echo "========================================"
    docker build -t "$CPU_IMAGE" "$PROJECT_DIR"
    echo ""
    echo "=== Smoke test (CPU image) ==="
    docker run --rm "$CPU_IMAGE" --help | head -20
    echo ""
    echo "OK: CPU image built and smoke-tested."
    echo "  Evaluate:  docker run --rm $CPU_IMAGE evaluate \\"
    echo "                 --agent data/agents/UECD-SingleMap-Best \\"
    echo "                 --opponent CoacAI --nb_games 10 --max_steps 1000"
    echo ""
}

build_gpu() {
    echo "========================================"
    echo "  Building GPU image: $GPU_IMAGE"
    echo "========================================"
    docker build -f "$PROJECT_DIR/Dockerfile.gpu" -t "$GPU_IMAGE" "$PROJECT_DIR"
    echo ""
    echo "=== Smoke test (GPU image, CPU-only --help) ==="
    # Use --help without --gpus all: this only loads the CLI, no CUDA runtime needed.
    docker run --rm "$GPU_IMAGE" --help | head -20
    echo ""
    echo "OK: GPU image built. Run training with:"
    echo "  docker run --rm --gpus all $GPU_IMAGE train \\"
    echo "      --exp-name docker-smoke --total-timesteps 100000"
    echo ""
}

# ── Prereqs ──────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker Desktop / Engine first."
    exit 1
fi
if ! docker info &>/dev/null; then
    echo "ERROR: docker daemon not reachable. Start Docker first."
    exit 1
fi

# ── Dispatch ─────────────────────────────────────────────────────────────────
MODE="${1:-cpu}"
case "$MODE" in
    cpu)        build_cpu ;;
    gpu)        build_gpu ;;
    both)       build_cpu; build_gpu ;;
    -h|--help)  usage ;;
    *)
        echo "ERROR: unknown mode '$MODE'. Expected: cpu | gpu | both."
        echo "Run with --help for usage."
        exit 1
        ;;
esac
