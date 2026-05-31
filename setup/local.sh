#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Setup LOCAL development environment for MicroRTS Master Thesis project.
#
#  Creates a conda env "microrts_agent" with all dependencies for:
#    - Training (PPO + GridNet)
#    - Evaluation & benchmarking
#    - Tournament (bot-vs-bot, agent-vs-bot, agent-vs-agent)
#    - Tournament bots (RAISocketAI, UtsImass, etc.)
#    - Recording game videos
#    - Plotting & visualization
#
#  Requirements:
#    - conda or miniconda installed
#    - Java 17+ (JDK; the committed bridge.jar is Java 17 bytecode)
#
#  Usage:
#    bash setup/local.sh
#
#  Activate:
#    conda activate microrts_agent
# ──────────────────────────────────────────────────────────────────────────────
set -e

ENV_NAME="microrts_agent"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # repo root (script lives in setup/)
# shellcheck source=setup/_common.sh
source "$PROJECT_DIR/setup/_common.sh"

echo "========================================"
echo "  MicroRTS Local Environment Setup"
echo "========================================"
echo "  Project:  $PROJECT_DIR"
echo "  Env:      $ENV_NAME (Python $PYTHON_VERSION)"
echo "========================================"
echo ""

# ── Prerequisites ────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

check_java
check_microrts_jar

# Bridge rebuild from src/ (committed bridge.jar must never be the source of truth).
build_bridge

# ── Conda environment ────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Conda env '$ENV_NAME' already exists."
    read -p "Recreate from scratch? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Updating existing environment..."
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: $ENV_NAME (Python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate (eval is the documented way to use `conda activate` in scripts).
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Python: $(python --version) @ $(which python)"
echo ""

# ── Python deps + RAISocketAI wheel + verification ───────────────────────────
install_python_deps
echo ""
install_raisocketai_wheel
echo ""
verify_install

echo "Setup complete! Activate with:"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "Quick test:"
echo "  cd $PROJECT_DIR"
echo "  python -m microrts_agent train --help"
echo ""
