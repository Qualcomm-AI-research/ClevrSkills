# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# This is a convoluted scheme to get temporary directories that get automatically deleted, mostly.
# The reason this is implemented this way is that some classes in sapien will retain the filename
# of models, URDFs, textures internally and try to load them later. And causing a segfault if the files
# don't exist anymore. You also can't quite be sure when the env (sapien scene) has been deleted.
# I.e., it is hard to know the scope of your temporary directories for sure.

# So instead we create a new temp dir on every environment reset.
# And delete temp files from more than 3 env resets ago.
# The assumption is that this will suffice to keep the required files alive.

import shutil
from tempfile import mkdtemp

temp_dir = None
temp_dirs = {}
PRESERVE_FOR_NUM_RESETS = 3


def get_temp_dir(scene) -> str:
    """
    Returns a temporary directory.
    The directory will be deleted after PRESERVE_FOR_NUM_RESETS resets,
    or when clear_temp_dir() is called.
    :return:
    """
    global temp_dir, temp_dirs
    scene_id = (
        id(scene) if scene else 0
    )  # Use scene ID to avoid keeping a reference to the scene itself
    if temp_dir is None:
        temp_dir = mkdtemp()
    if not scene_id in temp_dirs:
        temp_dirs[scene_id] = [[]]
    td = mkdtemp(dir=temp_dir)
    temp_dirs[scene_id][0].append(td)
    return td


def reset_temp_dir(scene) -> None:
    """
    Should be called on every environment reset.
    Deletes older temp dirs.
    :return: None
    """
    global temp_dir, temp_dirs, PRESERVE_FOR_NUM_RESETS
    scene_id = (
        id(scene) if scene else 0
    )  # Use scene ID to avoid keeping a reference to the scene itself
    if scene_id in temp_dirs:
        temp_dirs[scene_id].insert(0, [])
        while len(temp_dirs[scene_id]) > PRESERVE_FOR_NUM_RESETS:
            for td in temp_dirs[scene_id][-1]:
                shutil.rmtree(td)
            temp_dirs[scene_id].pop()


def clear_temp_dir() -> None:
    """
    Should be called at program exit to delete any remaining temp files
    :return: None
    """
    global temp_dir, temp_dirs
    if temp_dir:
        shutil.rmtree(temp_dir)
    temp_dir = None
    temp_dirs = {}
