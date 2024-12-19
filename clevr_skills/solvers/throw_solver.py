# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat, qmult

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.logger import log
from clevr_skills.utils.visualize_prompt import get_actor_description


class ThrowSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        throw_actor: sapien.Actor,
        target_actor: sapien.Actor,
        target_pos=None,
        target_2d: bool = False,
        topple_target: bool = False,
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param throw_actor: the actor that is to-be-thrown.
        :param target_actor: the actor that must be hit.
        :param target_pos: Optional target position to throw at. By default, the pose of the target_actor is used.
        :param target_2d: Whether the target is 2D (embedded in the floor).
        :param topple_target: Whether the target must be toppled. Not actually used currently.
        """
        super().__init__(env)

        self.throw_actor = throw_actor
        self.target_actor = target_actor
        self.target_pos = self.target_actor.pose.p if self.target_actor is not None else target_pos
        self.has_released_actor = False  # set to True once the throw_actor is released from gripper
        self.has_released_actor_num_frames = 0
        self._actor_distance: ActorDistance = env._actor_distance
        self.target_2d = target_2d
        self.topple_target = topple_target
        self._done = False

        self.action_state["actor"] = at_get_actor(self.throw_actor)

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = f"Throw {get_actor_description(self.throw_actor)}"
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        ee_pose: sapien.Pose = self._env.agent.ee_link.pose

        self.has_released_actor_num_frames += int(self.has_released_actor)
        if self.has_released_actor and self.has_released_actor_num_frames >= 5:
            self._done = True

        if self.should_release():  # compute trajectory of throw actor, determine whether to release
            self.has_released_actor = True
            return self.return_hold_action(self.gripper_off)

        # get 2D direction towards target
        direction = self.target_pos - ee_pose.p
        direction[2] = 0.0
        direction = direction / np.linalg.norm(direction)
        # get rotation axis towards target
        axis = np.cross(direction, np.array([0, 0, 1]))
        rot = axangle2quat(axis, np.deg2rad(10), is_normalized=True)

        new_ee_pose = sapien.Pose(
            ee_pose.p + 0.1 * direction + np.array([0, 0, 0.025]), qmult(rot, ee_pose.q)
        )
        delta_new_ee_pose = ee_pose.inv().transform(new_ee_pose)
        return self.return_delta_pose_action(
            delta_new_ee_pose,
            gripper=self.gripper_on,
            pos_thresh=0.0,
            rot_thresh=0,
            pos_multiplier=1.0,
            rot_multiplier=3.0,
        )

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

    def should_release(self, interval=3.0):
        """
        Computes the trajectory of the throw_actor, assuming it will follow a ballistic trajectory
        Then determine if the throw_actor will fly over / hit the target.
        Uses a trivial method for integration
        :param interval: how many second to probe into the future
        :return:
        """
        trajectory = []
        throw_actor_pose = self.throw_actor.pose
        pos = throw_actor_pose.p
        vel = self.throw_actor.get_velocity()
        gravity = self._env._scene.get_config().gravity
        for t in np.linspace(0, interval, round(interval / 0.05), endpoint=True):
            trajectory.append(pos + t * vel + 0.5 * t * t * gravity)

        # Target to hit the center-top of the target
        if self.target_actor is not None:
            target_bounds = self._actor_distance.get_bounds(self.target_actor)
            target_pos = np.array(
                [target_bounds[0, 0], np.mean(target_bounds[:, 1]), target_bounds[1, 2]]
            )
        else:
            target_pos = self.target_pos

        best_pos = None
        best_distance = 1000
        for pos in trajectory:
            distance = np.linalg.norm(target_pos - pos)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        if self.target_actor is not None:
            flat_dim = 2 if self.target_2d else -1
            computed_distance = self._actor_distance.distance(
                self.throw_actor,
                self.target_actor,
                actor_pose=sapien.Pose(best_pos, throw_actor_pose.q),
                flat_dim=flat_dim,
            )
        else:
            vertices = get_actor_mesh(self.throw_actor, to_world_frame=False).vertices
            rds = np.max(np.linalg.norm(vertices, axis=1))
            computed_distance = np.linalg.norm(best_pos - self.target_pos) - rds - 0.1
        log(f"Computed distance: {computed_distance}")
        return computed_distance < 0.02

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
