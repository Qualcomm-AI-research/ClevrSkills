# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Tuple

import numpy as np
import sapien.core as sapien
from mani_skill2.agents.controllers.pd_ee_pose import PDEEPosController
from mani_skill2.utils.common import inv_scale_action
from transforms3d.quaternions import quat2axangle

from clevr_skills.utils.logger import log


def quat2smallest_axangle(quat, identity_thresh=None):
    """
    Converts quaternion to axis-angle but makes sure that the abs(angle) is the smallest possible.
    :param quat: Quaternion
    :param identity_thresh: Threshold for "identity"; passed on to quat2axangle
    :return: tuple(axis, angle)
    """
    axis, angle = quat2axangle(quat, identity_thresh)
    angle = ((angle + np.pi) % (2 * np.pi)) - np.pi  # get the smallest angle
    return axis, angle


def align_quat_pose(pose, reference_pose):
    """
    Returns the same pose, but with orientation aligned.
    :param pose:
    :param reference_pose:
    :return: pose, with orientation aligned with reference_pose.q
    """
    return sapien.Pose(pose.p, -pose.q) if np.linalg.norm(pose.q + reference_pose.q) < 0.5 else pose


def align_quat(q, reference_q):
    """
    :param q: Quaternion
    :param reference_q: Reference quaternion.
    :return: q aligned with reference_q.
    """
    # The norm of the sum of two quaternions which are close should result in ~2.
    # So if the norm is below 0.5, then one of them is definitely inverted.
    return -q if np.linalg.norm(q + reference_q) < 0.5 else q


def delta_pose_to_action(
    pose: sapien.Pose,
    gripper: float,
    pos_thresh=0.1,
    rot_thresh=np.deg2rad(2),
    pos_multiplier=0.5,
    rot_multiplier=1.0,
) -> np.ndarray:
    """
    Converts a delta pose into an action for the Sapien env with pd_ee_delta_pose controller.
    The pd_ee_delta_pose uses normalized position and orientation commands.
    This function will normalize the positional and rotational commands to unit, unless
    the norm is smaller than the provided thresholds.
    :param pose: Delta-pose.
    :param gripper: Gripper action.
    :param pos_thresh: Used for normalization.
    And if the norm of the delta-position is larger than this threshold, the delta-position gets clipped.
    :param rot_thresh: Used for normalization.
    If the norm of the delta-rotation is larger than this threshold, the delta-rotation gets clipped.
    :param pos_multiplier: Use this to scale the delta-pos action, after conversion.
    :param rot_multiplier: Use this to scale the delta-rot action, after conversion.
    :return:
    """
    if abs(gripper) != 1.0:
        log(f"delta_pose_to_action: Warning gripper value is {gripper}")
    axis, angle = quat2smallest_axangle(pose.q)

    p = pose.p
    pos_norm = np.linalg.norm(pose.p)
    if pos_norm <= pos_thresh:
        p = p / pos_thresh
    else:
        p = p / pos_norm

    if abs(angle) < rot_thresh:
        angle /= rot_thresh
    else:
        angle = np.sign(angle)

    return np.concatenate((pos_multiplier * p, axis * (rot_multiplier * angle), [gripper]))


def ee_pose_to_ee_delta_action(
    agent,
    target_pose: sapien.Pose,
    gripper: float,
    scale_action_to_avoid_clipping: bool = True,
    verify_action: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Converts an absolute end-effector target_pose to a delta action,
    such that the target pose of the controller will be the target_pose
    Assumes that the PDEEPosController is in "delta" mode.
    :param agent: Required to retrieve the pose of the robot and the controller.
    :param target_pose: The absolute target pose of the end-effector
    :param gripper: a single float (gripper/vacuum pose)
    :param scale_action_to_avoid_clipping: If true, the action will be scaled down to
    avoid clipping.
    :param verify_action: If true, a check is made that the conversion was correct and assertion is made if not.
    :return: Tuple(delta action, was_action_scaled)
    """
    controller: PDEEPosController = agent.controller.controllers["arm"]

    ee_pose = agent.ee_link.pose
    target_pose = align_quat_pose(target_pose, ee_pose)
    delta_ee_pose = ee_pose.inv().transform(target_pose)

    axis, angle = quat2smallest_axangle(delta_ee_pose.q)
    delta_ee_pose_vector = np.concatenate([delta_ee_pose.p, axis * angle])
    delta_ee_pose_vector_inv = inv_scale_action(
        delta_ee_pose_vector, controller._action_space.low, controller._action_space.high
    )
    m = np.max(np.abs(delta_ee_pose_vector_inv))
    scaled = m > 1.0 and scale_action_to_avoid_clipping
    if scaled:
        delta_ee_pose_vector_inv /= m

    action = np.concatenate([delta_ee_pose_vector_inv, [gripper]])
    if verify_action:
        verify_ee_pose, _gripper = ee_delta_action_to_ee_pose(agent, action)
        diff = target_pose.inv().transform(verify_ee_pose)
        diff_p = np.linalg.norm(diff.p)
        diff_q = np.linalg.norm(np.abs(diff.q[0]) - 1)
        if m <= 0 and (diff_p > 1e-5 or diff_q > 1e-5):  # if not clipping and difference is large
            log("Error:")
            log(
                f"         action: pos: {delta_ee_pose_vector_inv[0:3]} "
                f"rot: {delta_ee_pose_vector_inv[3:6]}  "
                f"norm: {np.linalg.norm(delta_ee_pose_vector_inv[3:6])} "
            )
            log(f"verified target pose: {verify_ee_pose}")
            log(f"  actual target pose: {agent.robot.pose.inv().transform(target_pose)}")
            log(f"                diff: {diff}")
            assert (
                False
            ), f"diff was {diff} in ee_pose_to_ee_delta_action verification. Max coord of action: {m}"

    return action, scaled


def ee_delta_action_to_ee_pose(agent, action: np.ndarray) -> Tuple[sapien.Pose, float]:
    """
    Converts a delta into an absolute pose. Assumes that the PDEEPosController is in "delta" mode.
    In principle, you should be able to go back and forth between delta and abs pose
    using ee_pose_to_ee_delta_action and ee_delta_action_to_ee_pose, provided that no
    scaling or clipping occurs.
    :param agent: Required to retrieve the pose of the robot and the controller.
    :param action: A delta-action (6DOF + 1 gripper)
    :return: tuple (absolute pose, gripper)
    """
    controller: PDEEPosController = agent.controller.controllers["arm"]

    prev_ee_pose_at_base = agent.ee_link.pose
    preprocessed_action = controller._preprocess_action(action[0:6])
    target_pose = controller.compute_target_pose(prev_ee_pose_at_base, preprocessed_action)

    return (target_pose, action[-1])
