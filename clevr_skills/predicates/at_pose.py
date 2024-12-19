# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import qinverse, qmult

from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.predicate_utils import rotate_pose_z

from .at_position import AtPosition


class AtPose(AtPosition):
    """
    Predicate similar to at_position but also cares about the orientation of the object
    """

    def __init__(self, env, actor: sapien.Actor, pose: Pose, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: Actor to be placed at specified pose.
        :param pose: The target pose.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"{actor.name} at {pose}"
        super().__init__(env, actor, pose, name=name, match_ori=True)

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        eval = super().evaluate()

        # Compute whether actor is at pose
        _, angle = quat2smallest_axangle(qmult(self.actor.pose.q, qinverse(self.pose.q)))
        angle_diff = abs(angle) / np.pi

        actor_at_pose = (angle_diff < 0.1) and eval["actor_at_pos"]

        success = eval["success"] and actor_at_pose
        eval["success"] = success
        eval["actor_at_pose"] = actor_at_pose
        return eval


class AtPoseRotated(AtPose):
    """
    Predicate that specifies a final position and rotation relative to the initial orientation
    """

    def __init__(
        self, env, actor: sapien.Actor, pose: Pose, degrees: float, clockwise: bool, name=None
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: Actor to be placed at specified pose.
        :param pose: Target pose of actor
        :param degrees: Number of degrees to rotate about z (vertical) axis)
        :param clockwise: Whether to rotate clockwise or not.
        :param name: name of the predicate.
        """
        self.degree = -degrees if clockwise else degrees

        target_pose = Pose(pose.p, actor.pose.q)
        target_pose = rotate_pose_z(target_pose, self.degree)
        super().__init__(env, actor, target_pose, name=name)
