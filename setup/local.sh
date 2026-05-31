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

# Detect a stray active venv (e.g. cluster_venv left over from an earlier
# session). `conda activate` later does not cleanly override an active venv,
# so the resulting Python ends up being the venv's, not the conda env's,
# and RAISocketAI rejects Python 3.12+ with "requires <3.12".
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: a Python venv is active: $VIRTUAL_ENV"
    echo "  Running 'conda activate microrts_agent' on top of it will likely"
    echo "  leave the venv's python on PATH, which breaks the RAISocketAI install."
    echo "  Deactivate first, then re-run this script:"
    echo "      deactivate"
    echo "      bash setup/local.sh"
    exit 1
fi

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
install_pre_commit_hooks
echo ""
verify_install

echo ""
echo "###############################################################################"
echo "#                                                                             #"
echo "#  SETUP COMPLETE.                                                            #"
echo "#                                                                             #"
echo "#  THE CONDA ENV IS NOT ACTIVE IN YOUR SHELL YET. Run this:                  #"
echo "#                                                                             #"
echo "#      conda activate $ENV_NAME"
echo "#                                                                             #"
echo "#  (A bash script runs in a subshell; the activate inside it dies with the   #"
echo "#   script. You MUST activate the env yourself in your parent shell.)        #"
echo "#                                                                             #"
echo "#  If you have another venv already active (e.g. cluster_venv), 'deactivate' #"
echo "#  first; nested envs cause surprising Python-version conflicts.             #"
echo "#                                                                             #"
echo "#  Quick test once activated:                                                 #"
echo "#      microrts-agent --help                                                  #"
echo "#                                                                             #"
echo "###############################################################################"
echo ""
