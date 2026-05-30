#!/bin/bash
# Build Tiamat.jar — IEEE-CoG 2018 MicroRTS competition winner
# Source: ai/ | Main-Class: ai.competition.tiamat.Tiamat
# Dependencies: all ai/ classes are bundled (self-contained JAR)
set -e

if [ -x /usr/libexec/java_home ]; then
    export JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export PATH=$JAVA_HOME/bin:$PATH
fi
MICRORTS_JAR="$(cd "$(dirname "$0")/../.." && pwd)/microrts/microrts.jar"

rm -rf build
mkdir -p build/classes build/jar

# Compile all source files
find ai -name "*.java" > sources.txt
javac -cp "$MICRORTS_JAR" -d build/classes @sources.txt
rm -f sources.txt

# Package JAR (includes all compiled ai/ classes)
mkdir -p build/META-INF
cat > build/META-INF/MANIFEST.MF << 'EOF'
Manifest-Version: 1.0
Main-Class: ai.competition.tiamat.Tiamat

EOF

cd build/classes
jar cfm ../jar/Tiamat.jar ../META-INF/MANIFEST.MF ai/
cd ../..

# Deploy
mkdir -p ../jars
cp build/jar/*.jar ../jars/
