#!/bin/bash
# Build Mayari.jar — IEEE-CoG 2019 MicroRTS competition winner
# Source: Mayari.java | Main-Class: mayariBot.Mayari
# Dependencies: bundles ai/abstraction/ and rts/ from microrts.jar (self-contained JAR)
set -e

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi
MICRORTS_JAR="$(cd "$(dirname "$0")/../../.." && pwd)/microrts/microrts.jar"

rm -rf build
mkdir -p build/classes build/jar

# Compile (single file)
javac -cp "$MICRORTS_JAR" -d build/classes Mayari.java

# Extract framework classes needed at runtime
cd build/classes
jar xf "$MICRORTS_JAR" ai/abstraction/ 2>/dev/null || true
jar xf "$MICRORTS_JAR" rts/

# Package self-contained JAR with all dependencies
mkdir -p ../META-INF
cat > ../META-INF/MANIFEST.MF << 'EOF'
Manifest-Version: 1.0
Main-Class: mayariBot.Mayari

EOF

jar cfm ../jar/Mayari.jar ../META-INF/MANIFEST.MF mayariBot/ ai/ rts/
cd ../..

# Deploy
mkdir -p ../../jars
cp build/jar/*.jar ../../jars/
