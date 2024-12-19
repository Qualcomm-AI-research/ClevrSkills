# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import axangle2quat, qmult, quat2axangle

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pose
from clevr_skills.utils.controller import (
    align_quat_pose,
    ee_pose_to_ee_delta_action,
    quat2smallest_axangle,
)
from clevr_skills.utils.geometry import unit
from clevr_skills.utils.logger import log
from clevr_skills.utils.path_planner import (
    get_collisions,
    get_mplib_planner_with_grasped_actor,
    plan_path,
)
from clevr_skills.utils.visualize_prompt import get_actor_description


class Move3dSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        ee_target_pose: sapien.Pose,
        match_ori: bool = True,
        vacuum: bool = False,
        extend_bounds: float = 0.01,
        target_pose_name: str = None,
        check_done: bool = True,
    ):
        """
        :param env: The ClevrSkillsEnv
        :param ee_target_pose: the target pose of the end-effector.
        Note that path-planning will take into account anything that the end-effector is
        carrying.
        :param match_ori: Whether the orientation of the ee_target_pose must be matched.
        :param vacuum: Whether to turn vacuum on or off during moving.
        :param extend_bounds: How much to extend bounds of the grasped actor during path planning (in meters).
        :param target_pose_name: Used for action label.
        :param check_done: whether the solver should check and self-report that it is done.
        In most cases you want to set this to True.
        """
        super().__init__(env)

        self.ee_target_pose = ee_target_pose
        self.vacuum = vacuum
        self.check_done = check_done
        self._done = False
        self._agent = self._env.agent
        self._robot = self._agent.robot
        self._actor_distance = self._env._actor_distance
        self._previous_qpos = None
        self._match_ori = match_ori
        self._planning_timeout_last_attempt = False
        self._num_replanning = 0  # how many times has a new path been planned
        # This is used as an indicated for the planner getting stuck
        self._NUM_REPLANNING_STUCK = 4

        joint_limits = self._env.agent.get_mplib_joint_limits()
        self.planner = get_mplib_planner_with_grasped_actor(
            self._env, joint_limits=joint_limits, extend_bounds=extend_bounds
        )

        self.plan_path()

        self.action_state["target_pose"] = at_get_pose(ee_target_pose)
        self.action_state["match_ori"] = match_ori
        self.action_state["vacuum"] = vacuum
        self.action_state["extend_bounds"] = extend_bounds
        self.action_state["target_pose_name"] = target_pose_name
        self.action_state["grasped"] = []
        self.action_state["check_done"] = check_done
        self._previous_target_idx = 0

    def plan_path(self) -> bool:
        """
        Plan a path to reach goal position.
        :return:  True if path planning succeeded.
        """
        self._planned_qpos_path = []
        self._planned_ee_path = []
        self._planned_path_collisions = {}
        self._collision_radii = [0.05, 0.025, 0.01]
        self._previous_target_idx = 0
        self._path_planning_succeeded = True
        self._num_replanning += 1

        if not self._match_ori:
            # adjust the EE target orientation to be neutral relative to the robot base
            relative_pose = self._robot.pose.inv().transform(self.ee_target_pose)
            target_ori = qmult(
                axangle2quat([0, 0, 1], np.arctan2(relative_pose.p[1], relative_pose.p[0])),
                [0, -1, 0, 0],
            )
            self.ee_target_pose = sapien.Pose(self.ee_target_pose.p, target_ori)

        # Come up with an (intermediate) target pose if planning failed on the
        # previous step due to a time-out
        ee_pose = self._env.agent.ee_link.pose
        ee_target_pose = self.ee_target_pose
        if self._planning_timeout_last_attempt:
            s = np.random.uniform(low=0.25, high=0.6)
            p = s * ee_target_pose.p + (1.0 - s) * self._env.agent.ee_link.pose.p
            p[2] += ee_target_pose.p[2] + 0.1  # move up, just to be safe
            p[0] += np.random.uniform(low=-0.1, high=0.1)
            p[1] += np.random.uniform(low=-0.1, high=0.1)
            ee_target_pose = sapien.Pose(p, ee_target_pose.q)
        self._planning_timeout_last_attempt = False

        # check here if initial qpos is collision free
        ee_links = self._agent.ee_links
        collision_threshold = self._collision_radii[0]
        collision_free_moves = []
        for actor in self._env._scene.get_all_actors():
            if actor in self._agent.get_grasped_actors():
                continue
            d = self._actor_distance.distance(actor, ee_links)
            if d < collision_threshold:
                direction = -unit(self._actor_distance.gradient(actor, ee_links, known_distance=d))
                collision_free_moves.append(1.1 * max(0.01, (collision_threshold - d)) * direction)
        if collision_free_moves:
            collision_free_move = np.sum(collision_free_moves, axis=0)
            log(
                f"Laying out a plan to move to collision free pose: delta pos = "
                f"{collision_free_move}"
            )
            target_pose = sapien.Pose(ee_pose.p + collision_free_move, ee_pose.q)
            self._planned_qpos_path = [None]  # fake entry
            self._planned_ee_path = [target_pose]
            for radius in self._collision_radii:
                self._planned_path_collisions[radius] = [False]
            return True

        for point_cloud_radius, octree_resolution in [(0.02, 0.02), (0.01, 0.005), (0.0, 0.0025)]:
            log(
                f"Planning a path with point cloud at radius "
                f"{point_cloud_radius}, octree_resolution {octree_resolution}"
            )
            planned_qpos_path, planned_ee_path, collisions = plan_path(
                self.planner,
                self._env,
                ee_target_pose,
                octree_resolution=octree_resolution,
                point_cloud_radius=point_cloud_radius,
            )
            if (not planned_qpos_path is None) or (
                isinstance(collisions, str) and "Timeout" in collisions
            ):
                break

        if planned_qpos_path is None:
            reason = collisions
            log(f"Path planning to pose {ee_target_pose} failed: {reason}")
            self._path_planning_succeeded = False
            if "Timeout" in reason:
                self._planning_timeout_last_attempt = True
            return False

        self._planned_qpos_path = planned_qpos_path
        self._planned_ee_path = planned_ee_path
        self._planned_ee_path[-1] = ee_target_pose

        if np.any(collisions):
            log(f"{np.sum(collisions)} collisions detected in the planned path")

        for radius in self._collision_radii:
            self._planned_path_collisions[radius] = get_collisions(
                self.planner, self._planned_qpos_path, detailed=False, point_cloud_radius=radius
            )

        return True

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        p = self.ee_target_pose.p
        low_level_act = f"Move to {{pos: [{p[0]:0.2f} {p[1]:0.2f} {p[2]:0.2f}]}}"
        return {"mid_act_label": "INDETERMINABLE", "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        self.action_state["grasped_actors"] = [
            at_get_actor(actor) for actor in self._env.agent.get_grasped_actors()
        ]

        ee_pose = self._env.agent.ee_link.pose

        # Determine if (close to) done:
        # But when self.check_done is False, never actually declare that the solver is done.
        # This use-case is for when the agent must hold an object at a certain location.
        target_pose = align_quat_pose(self.ee_target_pose, ee_pose)
        delta_ee_pose = ee_pose.inv().transform(target_pose)
        self._done = (
            self.check_done
            and np.linalg.norm(delta_ee_pose.p) < 0.01
            and np.rad2deg(quat2axangle(delta_ee_pose.q)[1]) < 1.0
        )

        # If the robot is barely moving and the EE is not at the target pose,
        # it is stuck somehow -> replan
        must_replan = not self._path_planning_succeeded
        current_qpos = self._robot.get_qpos()
        if (not self._previous_qpos is None) and (not self._done):
            if np.linalg.norm(current_qpos - self._previous_qpos) < np.deg2rad(0.1):
                must_replan = True
        self._previous_qpos = current_qpos

        # Robot might be making a maneuver to get away from collision condition;
        # re-plan when goal reached
        if len(self._planned_ee_path) == 1:
            delta_ee_pose = ee_pose.inv().transform(self._planned_ee_path[-1])
            must_replan = (
                np.linalg.norm(delta_ee_pose.p) < 0.005
                and np.rad2deg(quat2axangle(delta_ee_pose.q)[1]) < 0.5
            )

        if must_replan:
            self.plan_path()

        if self._previous_target_idx >= len(self._planned_ee_path) - 1:
            target = (
                self.ee_target_pose
                if len(self._planned_ee_path) == 0
                else self._planned_ee_path[-1]
            )
            return self.return_global_pose_action(
                target, gripper=self.gripper_on if self.vacuum else self.gripper_off
            )

        # Find the longest step that is collision free according to the pre-computed
        # collision path. While also taking into account the largest step (action)
        # that can be taken
        collision_at_radius = {radius: False for radius in self._planned_path_collisions}
        for target_idx in range(self._previous_target_idx, len(self._planned_ee_path)):
            largest_collision_free_radius = 0
            for radius in collision_at_radius.keys():
                collision_at_radius[radius] = (
                    collision_at_radius[radius] or self._planned_path_collisions[radius][target_idx]
                )
                if not collision_at_radius[radius] and radius > largest_collision_free_radius:
                    largest_collision_free_radius = radius

            target_pose = self._planned_ee_path[target_idx]
            delta = ee_pose.inv().transform(target_pose)
            delta_p = np.linalg.norm(delta.p)
            delta_q = abs(np.rad2deg(quat2smallest_axangle(delta.q)[1]))
            if delta_p > largest_collision_free_radius:
                target_idx = max(self._previous_target_idx + 1, target_idx - 1)
                break

            _action, scaled = ee_pose_to_ee_delta_action(
                self._env.agent,
                target_pose,
                gripper=self.gripper_on if self.vacuum else self.gripper_off,
            )
            if scaled:
                break

        # Force move to next target pose if previous target was reached up to 1cm,
        # 1 degree, despite collision threat
        if (
            target_idx == self._previous_target_idx
            and largest_collision_free_radius == 0
            and (delta_p < 0.01)
            and (delta_q < 1.0)
        ):
            target_idx += 1

        self._previous_target_idx = target_idx

        target_pose = self._planned_ee_path[target_idx]

        return self.return_global_pose_action(
            target_pose, gripper=self.gripper_on if self.vacuum else self.gripper_off
        )

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done

    def is_stuck(self) -> bool:
        """
        :return: True if path planning failed too many times.
        """
        return self._num_replanning >= self._NUM_REPLANNING_STUCK
