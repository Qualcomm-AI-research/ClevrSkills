# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat

from clevr_skills.utils.controller import align_quat, quat2smallest_axangle


def test_quat2smallest_axangle():
    q = euler2quat(1.0, 2.0, 3.0)
    axis, angle = quat2smallest_axangle(q)
    q2 = axangle2quat(axis, angle)
    assert np.linalg.norm(q - q2) < 1e-5


def test_align_quat():
    q = euler2quat(1.0, 2.0, 3.0)
    assert np.linalg.norm(align_quat(-q, q) - q) < 1e-5
