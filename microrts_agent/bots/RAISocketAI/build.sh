#!/bin/bash
# Build RAISocketAI — IEEE-CoG 2023/2024 winner (pre-compiled JARs)
# No compilation needed: just copy the shipped JARs to jars/
set -e

mkdir -p ../../jars
cp RAISocketAI.jar ../../jars/
