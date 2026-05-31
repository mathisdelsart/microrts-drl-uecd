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
echo "=== Loading modules ==="
echo "(Skipped, load modules manually before running this script)"
# module load Python/3.10
# module load Java/17.0.6
# module load CUDA/12.1.1

check_java
check_microrts_jar
build_bridge

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
