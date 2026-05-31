#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  Setup CLUSTER environment for MicroRTS Master Thesis project.
#
#  Creates a venv at $PROJECT_DIR/cluster_venv with all dependencies for:
#    - Training (PPO + GridNet, GPU-accelerated)
#    - Evaluation & benchmarking
#    - Tournament (bot-vs-bot, agent-vs-bot, agent-vs-agent)
#    - Tournament bots (RAISocketAI, UtsImass, etc.)
#    - Plotting & visualization
#
#  Requirements:
#    - Cluster with module system (Python 3.10+, Java 17+, CUDA)
#    - Outbound HTTPS to github.com (to fetch the RAISocketAI wheel on first run;
#      skip with SKIP_RAISOCKETAI=1 if compute nodes have no outbound network)
#
#  Usage:
#    ssh lyra
#    cd ~/microrts-drl-uecd && bash setup/cluster.sh
#
#  Activate:
#    source $PROJECT_DIR/cluster_venv/bin/activate
# ──────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # repo root (script lives in setup/)
VENV_DIR="$PROJECT_DIR/cluster_venv"
# shellcheck source=setup/_common.sh
source "$PROJECT_DIR/setup/_common.sh"

echo "========================================"
echo "  MicroRTS Cluster Environment Setup"
echo "========================================"
echo "  Project:  $PROJECT_DIR"
echo "  Venv:     $VENV_DIR"
echo "========================================"
echo ""

# ── Cluster modules (operator-driven, not loaded by this script) ─────────────
echo "=== Cluster modules (load BEFORE running this script) ==="
echo ""
echo "Discover what's available on your CECI site (names drift across sites):"
echo "  module avail Java"
echo "  module avail Python"
echo "  module avail CUDA"
echo ""
echo "Then load. Working example on Lyra (CECI HPC) as of 2026:"
echo "  module load Java/17.0.6                          # JDK 17 for the JNI bridge"
echo "  module load Python/3.11.3-GCCcore-12.3.0         # 3.11 (Lyra has no 3.10; RAISocketAI wheel pins <3.12)"
echo "  module load CUDA/12.1.1                          # only if training on GPU"
echo ""
echo "On other CECI sites the toolchain suffix may differ ('Python/3.10'"
echo "instead of 'Python/3.11.3-GCCcore-12.3.0'). Always cross-check with"
echo "'module avail <X>' above and pick what's there."
echo ""
echo "Sanity check after loading:"
echo "  java -version          # should print 17.x"
echo "  python3 --version      # should print 3.10.x or 3.11.x"
echo "  nvidia-smi             # if on a GPU node, should print the card"
echo "========================================"
echo ""

check_java
check_microrts_jar
build_bridge

# ── Python version check ─────────────────────────────────────────────────────
# The venv below inherits whatever python3 is on PATH. RAISocketAI's
# rl_algo_impls wheel pins `<3.12,>=3.8`, and the project requires `>=3.10`,
# so the practical window is 3.10 or 3.11. Fail fast with a clear pointer
# if we're outside that range.
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
case "$PY_VERSION" in
    3.10|3.11)
        echo "Python $PY_VERSION: OK"
        ;;
    *)
        echo "ERROR: python3 is $PY_VERSION; this script needs 3.10 or 3.11."
        echo "  On CECI: module load Python/3.10 (or Python/3.11) before this script."
        echo "  Locally: use bash setup/local.sh instead (conda manages Python 3.10)."
        echo "  Or skip RAISocketAI: SKIP_RAISOCKETAI=1 bash setup/cluster.sh"
        echo "  (works on 3.12+ but the RAISocketAI tournament bot won't be available)."
        exit 1
        ;;
esac

# ── Venv ─────────────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Venv already exists at $VENV_DIR"
else
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version) @ $(which python)"
echo ""

# ── Python deps + RAISocketAI wheel ──────────────────────────────────────────
install_python_deps
echo ""
install_raisocketai_wheel
echo ""

# ── UTS_Imass Python 3.6 environment (optional, cluster-only) ────────────────
# UTS_Imass uses BL_JPS path-finding compiled against Python 3.6; we keep a
# dedicated venv for it so the main env stays on 3.10+. Two discovery paths:
# the cluster's `module load Python/3.6`, or a micromamba install we bootstrap.
echo "=== UTS_Imass Setup (requires Python 3.6) ==="
UTS_PY36="$PROJECT_DIR/uts_imass_env/bin/python"
MICROMAMBA="$HOME/.local/bin/micromamba"

if [ -x "$UTS_PY36" ]; then
    echo "UTS_Imass Python 3.6 already exists: $UTS_PY36"
else
    PY36_MOD=$(module avail Python/3.6 2>&1 | grep -oE 'Python/3\.6[^ )]*' | head -1)

    if [ -n "$PY36_MOD" ]; then
        echo "Found Python 3.6 module: $PY36_MOD"
        module load "$PY36_MOD" 2>/dev/null
        python3 -m venv "$PROJECT_DIR/uts_imass_env"
        echo "Created UTS_Imass env via module"
        module load Python/3.11 2>/dev/null || module load Python/3.10 2>/dev/null || true
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    else
        echo "No Python/3.6 module found. Using micromamba to install Python 3.6..."
        if [ ! -x "$MICROMAMBA" ]; then
            echo "Downloading micromamba..."
            mkdir -p "$HOME/.local/bin"
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
                | tar -xvj -C "$HOME/.local/bin/" --strip-components=1 bin/micromamba
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
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    fi
fi
echo ""

# ── Verification ─────────────────────────────────────────────────────────────
verify_install

# Cluster-only: also check the UTS_Imass python is callable.
if [ -x "$UTS_PY36" ]; then
    echo "  OK    UTS_Imass Python 3.6"
else
    echo "  FAIL  UTS_Imass (Python 3.6 not found)"
fi
echo ""

echo "Setup complete! Activate with:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Quick test:"
echo "  cd $PROJECT_DIR"
echo "  microrts-agent train --help"
echo ""
