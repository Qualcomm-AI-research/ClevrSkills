# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, Tuple

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from clevr_skills.utils.logger import log

from .predicate import EnvPredicate


class TracePredicate(EnvPredicate):
    """
    Predicate for trace task.
    A future goal is initialized as a red sphere which turns green when it is next goal position,
    once the goal is achieved, the sphere turns yellow.
    """

    color_next_goal = (0, 1, 0)
    color_visited_goal = (1, 1, 0)
    color_future_goal = (1, 0, 0)

    def __init__(
        self,
        env,
        goal_sites: Dict[Tuple, sapien.Actor],
        pose: Pose,
        name=None,
        first_predicate=False,
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param goal_sites: positions of the goals.
        :param pose: Target position.
        :param name: Descriptive name of the predicate.
        :param first_predicate: Used to initialize the markers in the correct color.
        """
        name = name if name else f"EE at {pose}"
        super().__init__(env, name)
        self.pose = pose
        self.goal_sites = goal_sites
        self._gravity = self._env._scene.get_config().gravity
        self.solving = None

        if not first_predicate:
            self.goal_sites[self.color_next_goal].set_pose(Pose([0.0, 0.0, 1000.0]))
            self.goal_sites[self.color_visited_goal].set_pose(Pose([0.0, 0.0, 1000.0]))
            self.goal_sites[self.color_future_goal].set_pose(self.pose)
        else:
            self.goal_sites[self.color_next_goal].set_pose(self.pose)
            self.goal_sites[self.color_visited_goal].set_pose(Pose([0.0, 0.0, 1000.0]))
            self.goal_sites[self.color_future_goal].set_pose(Pose([0.0, 0.0, 1000.0]))

    def start_solving(self):
        """
        Replace the current goal position with a green sphere and
        any past achieved goal with a yellow sphere
        """
        self.goal_sites[self.color_future_goal].set_pose(Pose([0.0, 0.0, 1000.0]))
        self.goal_sites[self.color_next_goal].set_pose(self.pose)
        self.solving = True
        log(f"solving {self.name} {self.pose}")

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        Successful if end-effector is at pose.
        """
        ee_pose = self._env.agent.ee_link.pose
        target_pos = np.copy(self.pose.p)

        ee_at_pose = np.linalg.norm((ee_pose.p - target_pos)) < 0.05

        if ee_at_pose:
            self.goal_sites[self.color_next_goal].set_pose(Pose([0.0, 0.0, 1000.0]))
            self.goal_sites[self.color_visited_goal].set_pose(self.pose)

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
            return 6.0

        ee_pose = self._env.agent.ee_link.pose
        target_pos = np.copy(self.pose.p)

        return 2.0 - np.linalg.norm(ee_pose.p - target_pos)
