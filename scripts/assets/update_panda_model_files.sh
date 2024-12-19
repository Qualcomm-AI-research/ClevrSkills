#!/usr/bin/env bash
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

BASE_DIR=$PWD
MANISKILL2_URDF_PATH="/ManiSkill2/mani_skill2/assets/descriptions/panda_v2.urdf"
PATCH_PATH="/tmp/patch_files"
OUTPUT_PATH="/tmp/patched"
URDF_PATH=$BASE_DIR/clevr_skills/assets/descriptions/

echo "Applying the patch..."

python3 $BASE_DIR/scripts/assets/robot_models/apply_panda_patch.py \
   --maniskill2-urdf-path $MANISKILL2_URDF_PATH \
   --patch-path $BASE_DIR/clevr_skills/assets/descriptions \
   --output-path $BASE_DIR/clevr_skills/assets/descriptions

echo "Done."
