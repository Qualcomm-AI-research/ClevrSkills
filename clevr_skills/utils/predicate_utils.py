# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from copy import deepcopy

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult


def rotate_pose_z(pose, angle_in_degrees):
    """
    Rotate a pose about the vertical (Z) axis.
    :param pose: The pose.
    :param angle_in_degrees: The angle (in degrees).
    :return: Pose, rotated by angle_in_degrees
    """
    tpose = deepcopy(pose)
    temp_q = tpose.q
    rot = (angle_in_degrees - 1) * np.pi / 180  # convert to radians; sub 1 to ensure the rotation
    rot_q = euler2quat(0, 0, rot)
    temp_q = qmult(temp_q, rot_q)
    tpose.set_q(temp_q)
    return tpose
