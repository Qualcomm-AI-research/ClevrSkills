# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
from sapien.core import Pose

from .predicate import EnvPredicate


class EEAtPosition(EnvPredicate):
    """
    Predicate that succeeds if the end-effector is at a target position
    """

    def __init__(self, env, target_pose: Pose, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param target_pose: Target position for end-effector.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"EE at {target_pose}"
        super().__init__(env, name)
        self.pose = target_pose

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        ee_pose = self._env.agent.ee_link.pose
        target_pos = np.copy(self.pose.p)

        ee_at_pose = np.linalg.norm((ee_pose.p - target_pos)) < 0.05

        return {
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

        return 2.0 - np.linalg.norm(ee_pose.p - target_pos)
