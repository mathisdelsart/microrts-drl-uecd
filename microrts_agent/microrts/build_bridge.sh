#!/bin/bash
# Compile src/ into bridge.jar
# Classes in bridge.jar override those in microrts.jar (classpath order matters)
#
# Usage: cd microrts/ && ./build_bridge.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
JAR_PATH="$SCRIPT_DIR/lib/bridge.jar"
ENGINE_JAR="$SCRIPT_DIR/microrts.jar"

echo "Compiling src/ against microrts.jar..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Find all .java files in src/
find "$SRC_DIR" -name "*.java" > "$BUILD_DIR/sources.txt"

# Compile against the engine JAR (needed for rts.*, gui.*, etc.)
# --release 17: target Java 17 bytecode (cluster compatibility)
javac --release 17 \
      -cp "$ENGINE_JAR" \
      -d "$BUILD_DIR" \
      @"$BUILD_DIR/sources.txt"

# Package into bridge.jar
echo "Packaging into lib/bridge.jar..."
jar cf "$JAR_PATH" -C "$BUILD_DIR" .

# Clean up
rm -rf "$BUILD_DIR"

echo "Done: $JAR_PATH"
echo "Classes overriding microrts.jar:"
jar tf "$JAR_PATH" | grep '\.class$' | sed 's/\.class$//' | sed 's/\//./g' | sort
