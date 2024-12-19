#!/usr/bin/env bash
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

BASE_DIR=$PWD
VIMA_BENCH_REPO_URL="https://github.com/vimalabs/VIMABench.git"
BRANCH="main"
TEXTURES_DIR="vima_bench/tasks/assets/textures/"
DESTINATION=$BASE_DIR/clevr_skills/assets/vima_textures
VIMA_REPO_PATH="/tmp/vima_bench"

echo "Cloning the Vima Bench repo..."

# Clone the repository
git clone --depth 1 --branch $BRANCH $VIMA_BENCH_REPO_URL $VIMA_REPO_PATH
cd $VIMA_REPO_PATH

echo "Creating a tarball for vima_textures"
tar -cvf ../vima_textures.tar.gz --directory=$TEXTURES_DIR .

echo "Moving to the right destination"
cd ..
mkdir -p $DESTINATION
tar -xvf vima_textures.tar.gz --directory=$DESTINATION

echo "Cleaning up"
# Clean up
rm -rf $VIMA_REPO_PATH vima_textures.tar.gz
cd $BASE_DIR
echo "Done."