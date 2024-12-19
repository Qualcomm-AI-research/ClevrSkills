# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat, qinverse, qmult

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pose
from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.visualize_prompt import get_actor_description


class PlaceSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        actor: sapien.Actor,
        target_pose: Pose,
        match_ori_2d: bool = False,
        drop_distance: float = 0.02,
        target_pose_name: str = None,
    ):
        """
        :param env: The ClevrSkillsEnv
        :param actor: Actor to be placed.
        :param target_pose: Absolute pose to place the actor.
        :param match_ori_2d: Match z-axis rotation?
        :param drop_distance: The actor will be dropped from this height relative to target
        """
        super().__init__(env)
        self.actor = actor
        self._target_pose = target_pose
        self._drop_distance = drop_distance
        self._match_ori_2d = match_ori_2d
        self._actor_distance = self._env._actor_distance
        self._lift_z = None  # set when the actor is first lifted
        self._lift_delta = 0.03
        self._done = False

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["target_pose"] = at_get_pose(target_pose)
        self.action_state["match_ori_2d"] = match_ori_2d
        self.action_state["drop_distance"] = drop_distance
        self.action_state["target_pose_name"] = target_pose_name

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"Place {get_actor_description(self.actor)} on {{pos:{self._target_pose.p}}}"
        )
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        actor_bounds = self._actor_distance.get_bounds(self.actor)
        actor_min_z = actor_bounds[0, 2]
        ee_pose = self._env.agent.ee_link.pose
        actor_pose = self.actor.pose

        delta_p = [0, 0, 0]
        gripper = self.gripper_on  # vacuum gripper is on by default

        target_pos = np.copy(self._target_pose.p)
        at_target_xy = np.linalg.norm((actor_pose.p - target_pos)[0:2]) < 0.01

        # check if there must be any rotation about the z-axis
        at_target_rot_z = True
        if self._match_ori_2d:
            axis, angle = quat2smallest_axangle(qmult(qinverse(actor_pose.q), self._target_pose.q))
            if abs(axis[2]) >= 0.95:
                at_target_rot_z = abs(np.rad2deg(angle)) < 1  # must match orientation to 1 degree

        # By default, make the gripper "orbit" the robot base
        # But if self._match_ori_2d is True, the code below will overwrite delta_rot_z
        relative_pose = self._env.agent.robot.pose.inv().transform(ee_pose)
        target_ori = qmult(
            axangle2quat([0, 0, 1], np.arctan2(relative_pose.p[1], relative_pose.p[0])),
            [0, -1, 0, 0],
        )

        if self._env.agent.check_grasp(self.actor):
            drop_target_z = self._target_pose.p[2] + self._drop_distance

            if self._match_ori_2d:
                target_ori = qmult(qmult(self._target_pose.q, qinverse(actor_pose.q)), ee_pose.q)

            # actor is grasped, move it to target, then release
            if at_target_xy and at_target_rot_z:
                # actor is above target
                if actor_min_z - drop_target_z < 0:
                    # actor is above target and close to target Z: release
                    gripper = self.gripper_off
                else:
                    # actor is above target
                    delta_p = target_pos - actor_pose.p
                    delta_p[2] = -(actor_min_z - drop_target_z) - 0.002
            else:
                # actor is grasped, move it to target
                target_pos[2] += self._drop_distance
                delta_p = target_pos - actor_pose.p
        else:
            self._done = True

        target_pose = Pose(ee_pose.p + delta_p, target_ori)
        return self.return_global_pose_action(target_pose, gripper=gripper)

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done


class PlaceOnActorSolver(PlaceSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        actor: sapien.Actor,
        dest_actor: sapien.Actor,
        target_pose: Pose,
        match_ori_2d: bool = False,
        drop_distance: float = 0.02,
    ):
        """
        :param env: ClevrSkillsEnv
        :param actor: The actor to be placed
        :param dest_actor: The actor to-be-placed-upon
        :param target_pose: pose (relative to dest_actor)
        :param match_ori_2d: Match z-axis rotation?
        :param drop_distance: From what distance to drop the actor?
        """

        self._env: ClevrSkillsEnv = env
        self.actor = actor
        self.dest_actor = dest_actor

        self._relative_target_pose = dest_actor.pose.inv().transform(target_pose)

        super().__init__(
            env,
            actor,
            target_pose,
            target_pose_name=get_actor_description(dest_actor),
            match_ori_2d=match_ori_2d,
            drop_distance=drop_distance,
        )

        self.action_state["target_actor"] = at_get_actor(dest_actor)

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"Place {get_actor_description(self.actor)} on {get_actor_description(self.dest_actor)}"
        )
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        self._target_pose = self.dest_actor.pose.transform(self._relative_target_pose)
        self.action_state["target_pose"] = at_get_pose(self._target_pose)
        return super().step(obs)
