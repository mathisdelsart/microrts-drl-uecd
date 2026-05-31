#!/bin/bash
# Shared helpers for setup/local.sh, setup/cluster.sh, setup/docker.sh.
#
# Sourced: never executed directly. The sourcing script is expected to
# already have set PROJECT_DIR. Helpers print status with the standard
# "OK / FAIL / WARNING" prefixes so the parent script's output stays uniform.

# ── Constants ────────────────────────────────────────────────────────────────
WHEEL_FILE="$PROJECT_DIR/microrts_agent/bots/RAISocketAI/rl_algo_impls-0.2.1-py3-none-any.whl"
WHEEL_URL="https://github.com/mathisdelsart/microrts-drl-uecd/releases/download/assets-rai-v0.2.1/rl_algo_impls-0.2.1-py3-none-any.whl"
WHEEL_SHA256="1e0a60133f4b96fa95f4331e258fd20495d2209d88c319116ac1bd19431e71d1"
MICRORTS_JAR="$PROJECT_DIR/microrts_agent/microrts/microrts.jar"
RAI_BOT_JAR="$PROJECT_DIR/microrts_agent/microrts/lib/bots/RAISocketAI.jar"


# ── Prerequisite checks ──────────────────────────────────────────────────────

# Print Java version (treats missing javac as a warning, not an error: the
# committed bridge.jar still lets people run agents without a JDK).
check_java() {
    if ! command -v java &>/dev/null; then
        echo "ERROR: Java not found. Install Java 17+ (JDK)."
        echo "  macOS: brew install openjdk@17"
        echo "  Linux: sudo apt install openjdk-17-jdk"
        return 1
    fi
    echo "Java:  $(java -version 2>&1 | head -1)"
}

# microrts.jar (vendored). Missing it is a hard fail at runtime, but setup can
# still proceed so the user sees the full picture from the verification block.
check_microrts_jar() {
    if [ ! -f "$MICRORTS_JAR" ]; then
        echo "WARNING: microrts.jar not found at $MICRORTS_JAR"
        echo "  The Java engine should be vendored in microrts_agent/microrts/."
    fi
}


# ── Java <-> Python bridge ───────────────────────────────────────────────────

# Always rebuild bridge.jar from src/ so the committed jar can never drift.
# If javac is missing we fall back to the committed jar with a warning.
build_bridge() {
    echo ""
    echo "=== Building Java bridge (bridge.jar) ==="
    if command -v javac &>/dev/null; then
        bash "$PROJECT_DIR/microrts_agent/microrts/build_bridge.sh"
    else
        echo "WARNING: javac (JDK) not found, falling back to the committed bridge.jar."
        echo "  Install a JDK and run: bash microrts_agent/microrts/build_bridge.sh"
    fi
}


# ── Python dependencies ──────────────────────────────────────────────────────

# Install the core stack + the [tournament] extra in editable mode. Bumps pip
# first so resolver-related issues with old pip versions don't appear.
install_python_deps() {
    echo "=== Installing Python dependencies (pip install -e .[tournament]) ==="
    pip install --upgrade pip
    pip install -e "${PROJECT_DIR}[tournament]"
}


# ── RAISocketAI tournament bot wheel ─────────────────────────────────────────

# Download the rl_algo_impls wheel from this repo's release assets, verify the
# pinned SHA-256, install with --no-deps (its pinned deps conflict with ours,
# the [tournament] extra above provides the compatible subset).
# Caller can skip with SKIP_RAISOCKETAI=1.
install_raisocketai_wheel() {
    echo "=== Installing RAISocketAI wheel ==="
    if [ "${SKIP_RAISOCKETAI:-0}" = "1" ]; then
        echo "Skipping RAISocketAI install (SKIP_RAISOCKETAI=1)."
        echo "  Tournament bots RAISocketAI/RAISocketAIBestModels won't work,"
        echo "  but all other features still do."
        return
    fi

    if [ ! -f "$WHEEL_FILE" ]; then
        echo "Downloading RAISocketAI wheel (~225 MB) from GitHub Releases..."
        if ! curl -L --fail --progress-bar "$WHEEL_URL" -o "$WHEEL_FILE"; then
            echo "ERROR: download failed from $WHEEL_URL"
            echo "  If the machine has no outbound network, re-run with"
            echo "  SKIP_RAISOCKETAI=1 or transfer the wheel manually:"
            echo "    scp rl_algo_impls-0.2.1-py3-none-any.whl <host>:$WHEEL_FILE"
            rm -f "$WHEEL_FILE"
            return 1
        fi
        if command -v sha256sum &>/dev/null; then
            actual=$(sha256sum "$WHEEL_FILE" | awk '{print $1}')
        else
            actual=$(shasum -a 256 "$WHEEL_FILE" | awk '{print $1}')
        fi
        if [ "$actual" != "$WHEEL_SHA256" ]; then
            echo "ERROR: SHA-256 mismatch for $WHEEL_FILE"
            echo "  expected: $WHEEL_SHA256"
            echo "  got:      $actual"
            rm -f "$WHEEL_FILE"
            return 1
        fi
        echo "SHA-256 verified."
    fi
    echo "Installing rl_algo_impls wheel (--no-deps to avoid conflicts)..."
    pip install --no-deps --force-reinstall "$WHEEL_FILE"
}


# ── Post-install verification ────────────────────────────────────────────────

# Counters used by check(); reset on every verify_install() call.
PASS=0
FAIL=0

check() {
    if python -c "$2" 2>/dev/null; then
        echo "  OK    $1"
        ((PASS++))
    else
        echo "  FAIL  $1"
        ((FAIL++))
    fi
}

# Run the full verification block. Imports the libraries we actually depend on
# (matches pyproject.toml's core stack + [tournament] extra). The list is the
# source of truth for "did the install succeed"; keep it aligned with
# pyproject.toml whenever deps move.
verify_install() {
    echo ""
    echo "========================================"
    echo "  Verification"
    echo "========================================"
    PASS=0
    FAIL=0

    # Core stack (always installed).
    check "PyTorch"       "import torch; print(f'    v{torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    check "NumPy"         "import numpy"
    check "SB3"           "import stable_baselines3"
    check "TensorBoard"   "from torch.utils.tensorboard import SummaryWriter"
    check "JPype1"        "import jpype"
    check "Pillow"        "from PIL import Image"
    check "Matplotlib"    "import matplotlib"
    check "Scipy"         "import scipy"
    check "Seaborn"       "import seaborn"
    check "Pandas"        "import pandas"
    check "Rich"          "import rich"
    check "Moviepy"       "import moviepy"

    # [tournament] extra (also installed by install_python_deps).
    check "Gymnasium 0.29" "import gymnasium; assert gymnasium.__version__.startswith('0.29')"
    check "PyYAML"        "import yaml"
    check "tqdm"          "import tqdm"
    check "einops"        "import einops"
    check "torchvision"   "import torchvision"
    check "accelerate"    "import accelerate"
    check "wandb"         "import wandb"
    check "GPUtil"        "import GPUtil"
    check "pyvirtualdisplay" "import pyvirtualdisplay"

    # RAISocketAI wheel (skipped if SKIP_RAISOCKETAI=1).
    if [ "${SKIP_RAISOCKETAI:-0}" != "1" ]; then
        check "rl_algo_impls" "import rl_algo_impls"
    fi

    # Vendored Java assets.
    echo ""
    if [ -f "$MICRORTS_JAR" ]; then
        echo "  OK    microrts.jar"; ((PASS++))
    else
        echo "  FAIL  microrts.jar not found"; ((FAIL++))
    fi
    if [ -f "$RAI_BOT_JAR" ]; then
        echo "  OK    RAISocketAI.jar"; ((PASS++))
    else
        echo "  FAIL  RAISocketAI.jar not found"; ((FAIL++))
    fi

    echo ""
    echo "========================================"
    echo "  Results: $PASS passed, $FAIL failed"
    echo "========================================"
    echo ""

    if [ "$FAIL" -gt 0 ]; then
        echo "Some checks failed. Review the output above."
        return 1
    fi
}
