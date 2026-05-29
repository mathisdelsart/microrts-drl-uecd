#!/bin/bash
# Build all competition winner bots.
# Each bot's build.sh compiles against microrts.jar.
# Built JARs are collected into jars/ (copy to microrts/lib/bots/ manually if needed).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi

echo ""
echo "  Build all tournament bots"
echo "  Java: $(java -version 2>&1 | head -1)"
echo "  ----------------------------------------"
echo ""

# Clean and prepare output directory
rm -rf jars
mkdir -p jars

# All bots to build (each has its own build.sh)
ALL_BOTS=(
    "CoacAI"
    "Mayari"
    "StrategyTactics"
    "ObiBotKenobi"
    "MixedBot"
    "Tiamat"
    "TMA"
    "UTS_Imass"
    "RAISocketAI"
)

SUCCESS_COUNT=0
FAILED_BOTS=()
TOTAL=${#ALL_BOTS[@]}

for bot in "${ALL_BOTS[@]}"; do
    botname=$(basename "$bot")
    IDX=$((SUCCESS_COUNT + ${#FAILED_BOTS[@]} + 1))

    cd "$bot"
    if ./build.sh > build.log 2>&1; then
        printf "  [%2d/%d]  %-20s  OK\n" "$IDX" "$TOTAL" "$botname"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        printf "  [%2d/%d]  %-20s  FAILED\n" "$IDX" "$TOTAL" "$botname"
        FAILED_BOTS+=("$botname")
    fi
    cd "$SCRIPT_DIR"
done

echo ""
echo "  ----------------------------------------"

if [ ${#FAILED_BOTS[@]} -gt 0 ]; then
    echo "  Result: $SUCCESS_COUNT / $TOTAL succeeded"
    echo ""
    echo "  Failed:"
    for bot in "${FAILED_BOTS[@]}"; do
        echo "    - $bot (see build.log)"
    done
    echo ""
    exit 1
fi

echo "  Result: $TOTAL / $TOTAL succeeded"
echo ""
echo "  Output: jars/"
ls jars/ | sed 's/^/    /'
echo ""
