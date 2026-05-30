#!/bin/bash
# Build StrategyTactics.jar — IEEE-CoG MicroRTS competition bot
# Source: src/standard/ | Main-Class: standard.StrategyTactics
# Dependencies: requires microrts.jar in classpath (not bundled)
set -e

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi
MICRORTS_JAR="$(cd "$(dirname "$0")/../.." && pwd)/microrts/microrts.jar"

rm -rf build
mkdir -p build/classes build/jar

# Compile all source files
find src -name "*.java" > sources.txt
javac -cp "$MICRORTS_JAR" -d build/classes @sources.txt
rm -f sources.txt

# Package JAR (only bot classes, framework deps come from microrts.jar)
mkdir -p build/META-INF
cat > build/META-INF/MANIFEST.MF << 'EOF'
Manifest-Version: 1.0
Main-Class: standard.StrategyTactics

EOF

cd build/classes
jar cfm ../jar/StrategyTactics.jar ../META-INF/MANIFEST.MF standard/
cd ../..

# Deploy
mkdir -p ../jars
cp build/jar/*.jar ../jars/
