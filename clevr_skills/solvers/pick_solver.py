# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat, qmult

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.visualize_prompt import get_actor_description


class PickSolver(AbstractSolver):
    def __init__(self, env: ClevrSkillsEnv, actor: sapien.Actor, lift: float = 0.1):
        """
        :param env: The ClevrSkillsEnv
        :param actor: The actor to be picked.
        :param lift: How much to lift the actor above the initial pose at pickup.
        Without lifting, actors below the picked-up-actor could be pushed aside
        during horizontal transport.
        """
        super().__init__(env)
        self.actor = actor
        self._lift_z = lift
        self._done = False

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["lift"] = lift

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = f"Pick {get_actor_description(self.actor)}"
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        actor_bounds = self._env._actor_distance.get_bounds(self.actor)
        actor_min_z = actor_bounds[0, 2]
        actor_max_z = actor_bounds[1, 2]
        ee_pose = self._env.agent.ee_link.pose
        actor_pose = self.actor.pose

        gripper = self.gripper_on  # vacuum gripper is on by default

        # By default, make the gripper "orbit" the robot base
        # But if self._match_ori_2d is True, the code below will overwrite delta_rot_z
        relative_pose = self._env.agent.robot.pose.inv().transform(ee_pose)
        target_ori = qmult(
            axangle2quat([0, 0, 1], np.arctan2(relative_pose.p[1], relative_pose.p[0])),
            [0, -1, 0, 0],
        )

        self._done = False
        if self._env.agent.check_grasp(self.actor):
            if actor_min_z < self._lift_z:
                # lift actor until above target z
                delta_p = [0.0, 0.0, 0.025]
            else:
                delta_p = [0.0, 0.0, 0.01]
                self._done = True
        else:
            # actor is not grasped, move gripper to target
            gripper = self.gripper_off
            target_pos = np.copy(actor_pose.p)
            distance_to_target = np.linalg.norm((ee_pose.p - target_pos)[0:2])
            at_target_xy = distance_to_target < 0.02

            # Before doing anything, make sure that the wrist is in a neutral pose
            wrist_rot = self._env.agent.get_wrist_qpos()
            if wrist_rot < -0.5 * np.pi:
                delta_p = [0, 0, 0]
                target_ori = qmult(ee_pose.q, axangle2quat([0, 0, 1], 0.5))
            elif wrist_rot > 0.5 * np.pi:
                delta_p = [0, 0, 0]
                target_ori = qmult(ee_pose.q, axangle2quat([0, 0, 1], -0.5))
            elif at_target_xy:  # and at_target_ori:
                # gripper is above actor, move down
                gripper = self.gripper_on

                delta_p = list((actor_pose.p - ee_pose.p)[0:2]) + [
                    min(actor_max_z - ee_pose.p[2], -0.001)
                ]
            else:
                # move gripper toward actor
                lift_z = max(
                    0.2, actor_max_z + 0.03
                )  # this assumes that moving the gripper above 0.2m is always safe
                if ee_pose.p[2] < lift_z - 0.03:
                    # lift gripper to hard-coded safe height
                    if distance_to_target < 0.05:
                        delta_p = (
                            np.array([target_pos[0], target_pos[1], lift_z - 0.025]) - ee_pose.p
                        )
                    else:
                        delta_p = [0.0, 0.0, 0.05]
                else:
                    # move to actor
                    target_pos[2] = lift_z + 0.03
                    delta_p = target_pos - ee_pose.p

        target_pose = Pose(ee_pose.p + delta_p, target_ori)
        return self.return_global_pose_action(target_pose, gripper=gripper)

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
