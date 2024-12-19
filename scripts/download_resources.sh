# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

#!/usr/bin/env bash

# Call this script from the root directory of the repo, i.e.,:
# cd $REPO_ROOT
# ./scripts/download_resources.sh

echo "Downloading resources, please wait..."

# Download Vima textures
source scripts/assets/get_vima_textures.sh

# Download other ManiSkill2 artefacts
python3 scripts/assets/get_maniskill_assets.py

# Copy UFACTORY xArm6 robot model from UFACTORY repo and patch the model description files
source scripts/assets/update_xarm_model_files.sh

# Copy Panda robot model from ManiSkill2 repo and patch model description files
source scripts/assets/update_panda_model_files.sh
