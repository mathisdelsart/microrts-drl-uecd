#!/bin/bash
# Start the UTS_Imass Python inference server.
# Requires Python 3.6 (BL_JPS native module is compiled for 3.6 only).
#
# Local (macOS): Uses pyenv with Python 3.6.15
# Cluster (HPC): Uses uts_imass_env (created by setup_cluster_env.sh via micromamba)
#
# Usage:
#   bash microrts_agent/tournaments/competition_winners/UTS_Imass/start_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/UTS_Imass_2019_Server"
# Project root is 4 levels up: UTS_Imass -> competition_winners -> tournaments -> microrts_agent -> root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=========================================="
echo "UTS_Imass Python Server"
echo "=========================================="

# Find Python 3.6
PYTHON=""

# 1. Try dedicated env (cluster — created by setup_cluster_env.sh)
UTS_ENV="$PROJECT_ROOT/uts_imass_env"
if [ -z "$PYTHON" ] && [ -f "$UTS_ENV/bin/python" ]; then
    PY_VER=$("$UTS_ENV/bin/python" --version 2>&1 | awk '{print $2}')
    if [[ "$PY_VER" =~ ^3\.6 ]]; then
        PYTHON="$UTS_ENV/bin/python"
        echo "Using env: $UTS_ENV (Python $PY_VER)"
    fi
fi

# 2. Try pyenv (local dev) — look for 3.6.* directly in pyenv versions dir
if [ -z "$PYTHON" ]; then
    PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    for pydir in "$PYENV_ROOT"/versions/3.6.*/bin/python; do
        if [ -x "$pydir" ]; then
            PY_VER=$("$pydir" --version 2>&1 | awk '{print $2}')
            PYTHON="$pydir"
            echo "Using pyenv: $pydir (Python $PY_VER)"
            break
        fi
    done
fi

# 3. Try system python3.6
if [ -z "$PYTHON" ]; then
    for candidate in python3.6 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            PY_VER=$("$candidate" --version 2>&1 | awk '{print $2}')
            if [[ "$PY_VER" =~ ^3\.6 ]]; then
                PYTHON="$candidate"
                echo "Using system: $candidate (Python $PY_VER)"
                break
            fi
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.6 not found."
    echo "  Local:   pyenv install 3.6.15 && pyenv local 3.6.15"
    echo "  Cluster: Run setup_cluster_env.sh (installs Python 3.6 via micromamba)"
    exit 1
fi

# Verify server script
if [ ! -f "$SERVER_DIR/UTS_Imass_Server.py" ]; then
    echo "ERROR: UTS_Imass_Server.py not found in $SERVER_DIR"
    exit 1
fi

cd "$SERVER_DIR"

echo "Starting server on port 9823..."
echo "Press Ctrl+C to stop."
echo "(Per-map data directories passed via preGameAnalysis at runtime)"
echo ""

# No --dir: per-map folders are passed by the tournament runner via
# preGameAnalysis(gs, budget, folder). "$@" allows manual --dir override.
$PYTHON UTS_Imass_Server.py --port 9823 "$@"
