# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
from sapien.core import Pose

from .predicate import EnvPredicate


class EEAtPose(EnvPredicate):
    """
    Predicate that succeeds if the end-effector is at a target pose (position + orientation)
    """

    def __init__(self, env, target_pose: Pose, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param target_pose: Target pose for the end effector.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"EE at {target_pose}"
        super().__init__(env, name)
        self.pose = target_pose
        self._pos_diffs = []
        self._rot_diffs = []

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        ee_pose = self._env.agent.ee_link.pose
        target_pos = np.copy(self.pose.p)
        target_rot = np.copy(self.pose.q)

        pos_diff = np.linalg.norm(ee_pose.p - target_pos)
        rot_diff = min(
            np.linalg.norm(ee_pose.q - target_rot), np.linalg.norm(-ee_pose.q - target_rot)
        )

        self._pos_diffs.append(pos_diff)
        self._rot_diffs.append(rot_diff)

        ee_at_pose = (pos_diff + rot_diff) < 0.15

        return {
            "pos_diff": pos_diff,
            "rot_diff": rot_diff,
            "min_pos_diff": min(self._pos_diffs),
            "min_rot_diff": min(self._rot_diffs),
            "actor_at_pose": ee_at_pose,
            "success": ee_at_pose,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 2]).
        """
        eval = self.evaluate()

        if eval["success"]:
            return 2.0

        ee_pose = self._env.agent.ee_link.pose
        target_pos = np.copy(self.pose.p)
        target_rot = np.copy(self.pose.q)

        return 2.0 - (
            np.linalg.norm(ee_pose.p - target_pos) + np.linalg.norm(ee_pose.q - target_rot)
        )
