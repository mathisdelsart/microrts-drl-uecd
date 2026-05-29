#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Setup CLUSTER environment for MicroRTS Master Thesis project.
#
#  Creates a venv at ~/microrts-drl-uecd/cluster_venv with all dependencies for:
#    - Training (PPO + GridNet, GPU-accelerated)
#    - Evaluation & benchmarking
#    - Tournament (bot-vs-bot, agent-vs-bot, agent-vs-agent)
#    - Tournament bots (RAISocketAI, UtsImass, etc.)
#    - Plotting & visualization
#
#  Requirements:
#    - Cluster with module system (Python 3.9+, Java 11+, CUDA)
#    - For RAISocketAI: transfer wheel first (see prerequisites below)
#
#  Prerequisites (tournament bots):
#    Transfer the wheel (225MB, too large for git):
#      scp microrts_agent/tournaments/competition_winners/RAISocketAI/rl_algo_impls-0.2.1-py3-none-any.whl \
#          lyra:~/microrts-drl-uecd/microrts_agent/tournaments/competition_winners/RAISocketAI/
#
#  Usage:
#    ssh lyra
#    cd ~/microrts-drl-uecd && bash setup_cluster_env.sh
#
#  Activate:
#    source ~/microrts-drl-uecd/cluster_venv/bin/activate
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/cluster_venv"
WHEEL_FILE="$PROJECT_DIR/microrts_agent/tournaments/competition_winners/RAISocketAI/rl_algo_impls-0.2.1-py3-none-any.whl"
MICRORTS_JAR="$PROJECT_DIR/microrts_agent/microrts/microrts.jar"

echo "========================================"
echo "  MicroRTS Cluster Environment Setup"
echo "========================================"
echo "  Project:  $PROJECT_DIR"
echo "  Venv:     $VENV_DIR"
echo "========================================"
echo ""

# ── 1. Load cluster modules ──────────────────────────────────────────────────

echo "=== Loading modules ==="
echo "(Skipped — load modules manually before running this script)"
# module load Python/3.9.6-GCCcore-11.2.0
# module load Java/17.0.6
# module load CUDA/12.1.1

JAVA_VER=$(java -version 2>&1 | head -1)
echo "Java:  $JAVA_VER"

if [ ! -f "$MICRORTS_JAR" ]; then
    echo "WARNING: microrts.jar not found at $MICRORTS_JAR"
    echo "  The Java engine is vendored in microrts_agent/microrts/."
fi

# ── Build the Java<->Python bridge (bridge.jar) ──────────────────────────────
# Always rebuild from src/ so we never run a possibly-stale committed jar.
echo ""
echo "=== Building Java bridge (bridge.jar) ==="
if command -v javac &>/dev/null; then
    bash "$PROJECT_DIR/microrts_agent/microrts/build_bridge.sh"
else
    echo "WARNING: javac (JDK) not found — falling back to the committed bridge.jar."
    echo "  Install a JDK and run: bash microrts_agent/microrts/build_bridge.sh"
fi

# ── 2. Create venv ───────────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Venv already exists at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

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
    echo "  Transfer it with:"
    echo "    scp rl_algo_impls-0.2.1-py3-none-any.whl $(hostname):$WHEEL_FILE"
fi

echo ""

# ── 5. UTS_Imass Python 3.6 environment (optional) ───────────────────────────

echo "=== UTS_Imass Setup (requires Python 3.6) ==="

UTS_PY36="$PROJECT_DIR/uts_imass_env/bin/python"
MICROMAMBA="$HOME/.local/bin/micromamba"

if [ -x "$UTS_PY36" ]; then
    echo "UTS_Imass Python 3.6 already exists: $UTS_PY36"
else
    # 1. Try module load Python/3.6
    PY36_MOD=""
    for mod_name in $(module avail Python/3.6 2>&1 | grep -oE 'Python/3\.6[^ )]*' | head -1); do
        PY36_MOD="$mod_name"
    done

    if [ -n "$PY36_MOD" ]; then
        echo "Found Python 3.6 module: $PY36_MOD"
        module load "$PY36_MOD" 2>/dev/null
        python3 -m venv "$PROJECT_DIR/uts_imass_env"
        echo "Created UTS_Imass env via module"
        module load Python/3.11 2>/dev/null || module load Python/3.9 2>/dev/null || true
        source "$VENV_DIR/bin/activate"
    else
        # 2. Fallback: use micromamba to install Python 3.6
        echo "No Python/3.6 module found. Using micromamba to install Python 3.6..."

        if [ ! -x "$MICROMAMBA" ]; then
            echo "Downloading micromamba..."
            mkdir -p "$HOME/.local/bin"
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME/.local/bin/" --strip-components=1 bin/micromamba
            chmod +x "$MICROMAMBA"
        fi

        echo "Creating Python 3.6 environment..."
        "$MICROMAMBA" create -p "$PROJECT_DIR/uts_imass_env" python=3.6 -c conda-forge -y

        if [ -x "$UTS_PY36" ]; then
            echo "UTS_Imass Python 3.6 installed: $("$UTS_PY36" --version)"
        else
            echo "WARNING: Failed to install Python 3.6 via micromamba."
            echo "  UTS_Imass will not work. This is optional."
        fi

        # Re-activate main venv
        source "$VENV_DIR/bin/activate"
    fi
fi

echo ""

# ── 6. Verification ──────────────────────────────────────────────────────────

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

JAR="$PROJECT_DIR/microrts_agent/microrts/lib/bots/RAISocketAI.jar"
[ -f "$JAR" ] && echo "  OK  RAISocketAI.jar" && ((PASS++)) || { echo "  FAIL  RAISocketAI.jar not found"; ((FAIL++)); }

# UTS_Imass
[ -x "$UTS_PY36" ] && echo "  OK  UTS_Imass Python 3.6" && ((PASS++)) || { echo "  FAIL  UTS_Imass (Python 3.6 not found)"; ((FAIL++)); }

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
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Quick test:"
echo "  cd $PROJECT_DIR"
echo "  python microrts_agent/train.py --help"
echo ""
