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
#    - Java 11+ (for microrts.jar via JPype)
#
#  Usage:
#    bash setup_local_env.sh
#
#  Activate:
#    conda activate microrts_agent
# ──────────────────────────────────────────────────────────────────────────────

ENV_NAME="microrts_agent"
PYTHON_VERSION="3.9"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL_FILE="$PROJECT_DIR/microrts_agent/tournaments/competition_winners/RAISocketAI/rl_algo_impls-0.2.1-py3-none-any.whl"

echo "========================================"
echo "  MicroRTS Local Environment Setup"
echo "========================================"
echo "  Project:  $PROJECT_DIR"
echo "  Env:      $ENV_NAME (Python $PYTHON_VERSION)"
echo "========================================"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────────────

# Conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Java
if ! command -v java &>/dev/null; then
    echo "ERROR: Java not found. Install Java 11+ (JDK)."
    echo "  macOS: brew install openjdk@17"
    echo "  Linux: sudo apt install openjdk-17-jdk"
    exit 1
fi

JAVA_VER=$(java -version 2>&1 | head -1)
echo "Java:  $JAVA_VER"

# microrts.jar
MICRORTS_JAR="$PROJECT_DIR/microrts_agent/microrts/microrts.jar"
if [ ! -f "$MICRORTS_JAR" ]; then
    echo "WARNING: microrts.jar not found at $MICRORTS_JAR"
    echo "  The Java engine is vendored in microrts_agent/microrts/."
fi

# ── 2. Build the Java<->Python bridge (bridge.jar) ───────────────────────────
# Always rebuild from src/ so we never run a possibly-stale committed jar.
echo ""
echo "=== Building Java bridge (bridge.jar) ==="
if command -v javac &>/dev/null; then
    bash "$PROJECT_DIR/microrts_agent/microrts/build_bridge.sh"
else
    echo "WARNING: javac (JDK) not found — falling back to the committed bridge.jar."
    echo "  Install a JDK and run: bash microrts_agent/microrts/build_bridge.sh"
fi

# ── 3. Create conda environment ──────────────────────────────────────────────

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

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(python --version) @ $(which python)"
echo ""

# ── 3. Python dependencies (single source of truth: pyproject.toml) ──────────
# Core stack + the [tournament] extra (RAISocketAI's cherry-picked runtime deps).

echo "=== Installing Python dependencies (pip install -e .[tournament]) ==="
pip install --upgrade pip
pip install -e "${PROJECT_DIR}[tournament]"

echo ""

# ── 4. RAISocketAI tournament bot wheel ──────────────────────────────────────
# Installed separately with --no-deps: its pinned dependency set conflicts with
# the core stack (the [tournament] extra above provides the compatible subset).

echo "=== Installing RAISocketAI wheel ==="
if [ -f "$WHEEL_FILE" ]; then
    echo "Installing rl_algo_impls wheel (--no-deps to avoid conflicts)..."
    pip install --no-deps --force-reinstall "$WHEEL_FILE"
else
    echo "WARNING: RAISocketAI wheel not found at:"
    echo "  $WHEEL_FILE"
    echo "  Tournament bots RAISocketAI/RAISocketAIBestModels won't work."
    echo "  This is optional — all other features work fine."
fi

echo ""

# ── 5. Verification ──────────────────────────────────────────────────────────

echo "========================================"
echo "  Verification"
echo "========================================"

PASS=0
FAIL=0

check() {
    if python -c "$2" 2>/dev/null; then
        echo "  OK  $1"
        ((PASS++))
    else
        echo "  FAIL  $1"
        ((FAIL++))
    fi
}

check "PyTorch"       "import torch; print(f'    v{torch.__version__}, CUDA: {torch.cuda.is_available()}')"
check "NumPy"         "import numpy"
check "Gym 0.23"      "import gym; assert gym.__version__.startswith('0.23')"
check "SB3"           "import stable_baselines3"
check "TensorBoard"   "from torch.utils.tensorboard import SummaryWriter"
check "JPype1"        "import jpype"
check "Pillow"        "from PIL import Image"
check "Matplotlib"    "import matplotlib"
check "Seaborn"       "import seaborn"
check "Pandas"        "import pandas"
check "Rich"          "import rich"
check "PettingZoo"    "import pettingzoo"
check "rl_algo_impls" "import rl_algo_impls"
check "Gymnasium"     "import gymnasium"
check "wandb"         "import wandb"

# Java & microrts
echo ""
[ -f "$MICRORTS_JAR" ] && echo "  OK  microrts.jar" && ((PASS++)) || { echo "  FAIL  microrts.jar not found"; ((FAIL++)); }

echo ""
echo "========================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Some checks failed. Review the output above."
    exit 1
fi

echo "Setup complete! Activate with:"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "Quick test:"
echo "  cd $PROJECT_DIR"
echo "  python -m microrts_agent.train --help"
echo ""
