#!/bin/bash
# Build ObiBotKenobi.jar — IEEE-CoG MicroRTS competition bot
# Source: ObiBotKenobi.java (default package) | Main-Class: ObiBotKenobi
# Dependencies: requires microrts.jar in classpath (not bundled)
set -e

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi
MICRORTS_JAR="$(cd "$(dirname "$0")/../../.." && pwd)/microrts/microrts.jar"

rm -rf build
mkdir -p build/classes build/jar

# Compile (single file, default package)
javac -cp "$MICRORTS_JAR" -d build/classes ObiBotKenobi.java

# Package JAR (only the bot class, framework deps come from microrts.jar)
mkdir -p build/META-INF
cat > build/META-INF/MANIFEST.MF << 'EOF'
Manifest-Version: 1.0
Main-Class: ObiBotKenobi

EOF

cd build/classes
jar cfm ../jar/ObiBotKenobi.jar ../META-INF/MANIFEST.MF *.class
cd ../..

# Deploy
mkdir -p ../../jars
cp build/jar/*.jar ../../jars/
