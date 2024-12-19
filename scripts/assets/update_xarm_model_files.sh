#!/usr/bin/env bash
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

BASE_DIR=$PWD
XARM_REPO_URL="https://github.com/xArm-Developer/xarm_ros.git"
COMMIT_SHA="740b003d5b330ea70f474a358ffc29c4aad3444c"
XARM_REPO_PATH="/tmp/xarm_ros"
PATCH_PATH=$BASE_DIR/clevr_skills/assets/descriptions/
OUTPUT_PATH=$BASE_DIR/clevr_skills/assets/descriptions/
URDF_PATH=$BASE_DIR/clevr_skills/assets/descriptions/

mkdir $PATCH_PATH

echo "Cloning the UFACTORY xArm developer repo..."

git clone $XARM_REPO_URL $XARM_REPO_PATH
cd $XARM_REPO_PATH
git checkout $COMMIT_SHA
cd $BASE_DIR

echo "Applying the patch..."

python3 $BASE_DIR/scripts/assets/robot_models/apply_xarm_patch.py \
   --xarm-repo-path $XARM_REPO_PATH \
   --patch-path $PATCH_PATH \
   --output-path $OUTPUT_PATH

echo "Cleaning up..."
rm -rf $XARM_REPO_PATH

echo "Done."