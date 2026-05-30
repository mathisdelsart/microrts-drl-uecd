#!/bin/bash
# Build UTS_Imass.jar — IEEE-CoG 2019 MicroRTS competition bot (socket-based)
# Source: MicroRTS_Modifications/src/ai/socket/ | Main-Class: ai.socket.UTS_Imass
# Dependencies: bundles rts/ from microrts.jar (self-contained JAR)
#
# NOTE: This bot requires a separate Python server running on port 9823.
#   Start it before the tournament:
#     cd UTS_Imass/UTS_Imass_2019_Server && python3 UTS_Imass_Server.py --port 9823
set -e

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi
MICRORTS_JAR="$(cd "$(dirname "$0")/../.." && pwd)/microrts/microrts.jar"

rm -rf build
mkdir -p build/classes build/jar

# Compile Java socket wrapper
javac -cp "$MICRORTS_JAR" -d build/classes \
    MicroRTS_Modifications/src/ai/socket/UTS_Imass_SocketAI.java \
    MicroRTS_Modifications/src/ai/socket/UTS_Imass.java

# Extract framework classes needed at runtime
cd build/classes
jar xf "$MICRORTS_JAR" rts/

# Package self-contained JAR with all dependencies
mkdir -p ../META-INF
cat > ../META-INF/MANIFEST.MF << 'EOF'
Manifest-Version: 1.0
Main-Class: ai.socket.UTS_Imass

EOF

jar cfm ../jar/UTS_Imass.jar ../META-INF/MANIFEST.MF ai/ rts/
cd ../..

# Deploy
mkdir -p ../jars
cp build/jar/*.jar ../jars/
