# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
from typing import Dict, Tuple

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import qinverse, qmult, quat2mat

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.controller import ee_pose_to_ee_delta_action
from clevr_skills.utils.geometry import unit
from clevr_skills.utils.visualize_prompt import get_actor_description

from .move3d_solver import Move3dSolver
from .push_solver import get_feasible_gripper_push_poses


class AbstractTouchSolver(AbstractSolver):

    def __init__(self, env: ClevrSkillsEnv, actor: sapien.Actor, push: bool = False, topple=False):
        """
        :param env: ClevrSkillsEnv
        :param actor: The actor to be touched, pushed, toppled.
        :param push: Whether to push.
        :param topple: Whether to topple. Toppling takes priority over pushing.
        """
        super().__init__(env)

        self.actor = actor
        self.push = push
        self.topple = topple
        self._done = False
        self._agent = self._env.agent
        self._neutral_gripper_quat = self._agent.neutral_ee_quat
        self._actor_distance = self._env._actor_distance
        self.space = 0.03
        self.initial_pose = copy.deepcopy(actor.pose)

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["push"] = push
        self.action_state["topple"] = topple

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done

    def get_verb(self, cap: bool = False):
        """
        :param cap: Whether the first letter of the verb should be a capital or not.
        :return: verb for converting the solving actions into natural language. I.e., "touch", "topple" or "push".
        """
        v = "touch"
        if self.topple:
            v = "topple"
        elif self.push:
            v = "push"
        return v[0:1].upper() + v[1:] if cap else v

    def get_goal_info(self) -> Tuple[bool, bool]:
        """
        :return: Tuple (actor moved, actor toppled).
        """

        distance_moved = np.linalg.norm(self.actor.pose.p - self.initial_pose.p)
        moved = distance_moved > 0.15
        mat = quat2mat(qmult(self.actor.pose.q, qinverse(self.initial_pose.q)))
        topple_angle = np.rad2deg(np.arccos(abs(np.dot(mat[:, 2], [0, 0, 1]))))
        toppled = topple_angle >= 50
        return moved, toppled


class TouchSolver(AbstractTouchSolver):
    def __init__(self, env: ClevrSkillsEnv, actor: sapien.Actor, push: bool = False, topple=False):
        """
        :param env: ClevrSkillsEnv
        :param actor: The actor to be touched, pushed, toppled.
        :param push: Whether to push.
        :param topple: Whether to topple. Toppling takes priority over pushing.
        """
        super().__init__(env, actor, push=push, topple=topple)

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"{self.get_verb(cap=True)} {get_actor_description(self.actor)}"
        )
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        # determine at what height to push
        bounds = self._actor_distance.get_bounds(self.actor)
        min_z, max_z = bounds[:, 2]
        target_ee_z = min_z + 0.03
        if self.topple:
            target_ee_z = max_z - 0.015

        distance_to_actor = self._actor_distance.distance(self.actor, self._agent.ee_links)
        ee_pose = self._agent.ee_link.pose

        p = unit((self.actor.pose.p - ee_pose.p)[0:2])  # push direction
        d = min(0.02, distance_to_actor + 0.001)

        moved, toppled = self.get_goal_info()

        if self.topple:
            if toppled:
                self._done = True
            d = 0.1
        elif self.push:
            if moved:
                self._done = True
            d = 0.05
        else:  # touch
            contacts = self._env.get_actor_contacts(self.actor)
            touching = len(set(contacts).intersection(set(self._env.agent.ee_links))) > 0
            if touching:
                self._done = True
                d = 0.0
        target_pos = np.array(
            [ee_pose.p[0] + d * p[0], ee_pose.p[1] + d * p[1], target_ee_z],
            dtype=np.float32,
        )
        target_pose = sapien.Pose(target_pos, ee_pose.q)
        action, _ = ee_pose_to_ee_delta_action(self._agent, target_pose, gripper=self.gripper_off)
        return self.return_action(action)


class Move3dTouchSolver(AbstractTouchSolver):
    def __init__(self, env: ClevrSkillsEnv, actor: sapien.Actor, push: bool = False, topple=False):
        """
        :param env: ClevrSkillsEnv
        :param actor: The actor to be touched, pushed, toppled.
        :param push: Whether to push.
        :param topple: Whether to topple. Toppling takes priority over pushing.
        """
        super().__init__(env, actor, push=push, topple=topple)

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"{self.get_verb(cap=True)} {get_actor_description(self.actor)}"
        )
        if not self._sub_solver is None:
            low_level_act = (
                f"Moving to a good place to {self.get_verb(cap=False)} "
                f"{get_actor_description(self.actor)}"
            )

        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        # determine at what height to push
        bounds = self._actor_distance.get_bounds(self.actor)
        min_z, max_z = bounds[:, 2]
        target_ee_z = min_z + 0.03
        if self.topple:
            target_ee_z = max_z - 0.015

        if self._sub_solver and self._sub_solver.is_done():
            if isinstance(self._sub_solver, TouchSolver):
                self._done = True
            self._sub_solver = None

        if self._sub_solver:
            return self._sub_solver.step(obs)

        distance_to_actor = self._actor_distance.distance(self.actor, self._agent.ee_links)
        ee_pose = self._agent.ee_link.pose
        if self._done:
            # Task is done. If the caller persists to call this solver:
            # Move to a neutral pose, 5cm up, 5cm back
            target_ee_z += 0.05
            d = 0.05 * unit((self.actor.pose.p - self._agent.robot.pose.p)[0:2])
            self._sub_solver = Move3dSolver(
                self._env,
                sapien.Pose([d[0], d[1], target_ee_z]).transform(ee_pose),
                match_ori=True,
                vacuum=False,
                target_pose_name=f"neutral pose",
            )
        elif distance_to_actor <= 2 * self.space and abs(ee_pose.p[2] - target_ee_z) < 0.01:
            self._sub_solver = TouchSolver(self._env, self.actor, self.push, self.topple)
        else:
            poses, scores = get_feasible_gripper_push_poses(
                self._env,
                self.actor,
                self.actor.pose.p - self._agent.robot.pose.p,
                self._neutral_gripper_quat,
                target_ee_z=target_ee_z,
                space=self.space,
            )
            pose = np.random.choice(poses, 1, p=scores)[0]
            self._sub_solver = Move3dSolver(
                self._env,
                pose,
                match_ori=True,
                vacuum=False,
                extend_bounds=0.0,
                target_pose_name=(
                    f"a good place to {self.get_verb()} " f"{get_actor_description(self.actor)}"
                ),
            )

        return self._sub_solver.step(obs)
