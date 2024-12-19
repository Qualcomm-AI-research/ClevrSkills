# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, List, Tuple

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.solvers.move3d_solver import Move3dSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pos, at_get_pose
from clevr_skills.utils.geometry import unit
from clevr_skills.utils.path_planner import (
    get_cartesian_robot_mplib_planner_with_attached_actors,
    plan_path_qpos,
)
from clevr_skills.utils.visualize_prompt import get_actor_description


def get_feasible_gripper_push_poses(
    env: ClevrSkillsEnv,
    actor: sapien.Actor,
    pushing_direction,
    neutral_gripper_quat,
    num_attempts: int = 32,
    space: float = 0.05,
    target_ee_z: float = None,
) -> Tuple[List[sapien.Pose], np.ndarray]:
    """
    :param env: The ClevrSkillsEnv
    :param actor: The actor to be pushed.
    :param pushing_direction: The direction the actor shoyld be pushed. Only X & Y are used.
    :param neutral_gripper_quat: the neutral orientation of the gripper (upside down for most robots).
    :param num_attempts: How many samples to generate.
    :param space: How far the gripper should stay away from the actor.
    :param target_ee_z: Optional; explicit end-effector Z.
    :return: a list of proposals for poses where the gripper can push the actor.
    Each entry is the pose + the score (how suitable is this pose for pushing towards the target).
    """
    actor_distance = env._actor_distance
    # Determine sizes of actor and gripper
    actor_bounds = actor_distance.get_bounds(actor)
    gripper_bounds = actor_distance.get_bounds(
        env.agent.ee_links, actor_pose=sapien.Pose([0, 0, 0], neutral_gripper_quat)
    )

    # Get all actors (besides self.actor) that need to be checked for collision
    all_other_actors = [
        a for a in env._scene.get_all_actors() if ((not a is actor) and (not a.name == "ground"))
    ]

    # Sample num_attempts orientations around the actor
    angles = np.linspace(
        start=-np.pi, stop=np.pi, num=num_attempts, endpoint=False
    ) + env._episode_rng.uniform(
        low=-np.pi / num_attempts, high=np.pi / num_attempts, size=num_attempts
    )

    # get crude estimate on the required 2D distance between gripper and actor
    max_distance = (
        np.sum((actor_bounds[1] - actor_bounds[0])[0:2])
        + np.sum((gripper_bounds[1] - gripper_bounds[0])[0:2])
    ) / 2

    # for each angle, find out the closest distance that the gripper can come
    feasible_gripper_poses = []
    scores = []
    for angle in angles:
        # get quaternion and direction corresponding to the angle
        orientation = euler2quat(0, 0, angle)
        direction = -quat2mat(orientation)[:, 0]

        r = max_distance
        best_gripper_pose_for_this_angle = None
        while True:
            gripper_position = actor.pose.p + r * direction
            gripper_position[2] = (
                -gripper_bounds[1, 2] + space + 0.005 if target_ee_z is None else target_ee_z
            )
            gripper_pose = sapien.Pose(gripper_position, qmult(orientation, neutral_gripper_quat))
            d = actor_distance.distance(env.agent.ee_links, actor, actor_pose=gripper_pose)
            if d > space and not actor_distance.intersects(
                env.agent.ee_links, all_other_actors, actor_pose=gripper_pose
            ):
                best_gripper_pose_for_this_angle = gripper_pose
                r -= max(0.005, d / 2)
            else:
                break
        if best_gripper_pose_for_this_angle:
            target_direction = unit(pushing_direction[0:2])
            push_direction = unit((actor.pose.p - best_gripper_pose_for_this_angle.p)[0:2])
            score = max(0, np.dot(push_direction, target_direction))
            score = pow(score, 3.0)
            feasible_gripper_poses.append(best_gripper_pose_for_this_angle)
            scores.append(score)

    scores = np.array(scores, dtype=np.float32) / np.sum(scores)

    return feasible_gripper_poses, scores


class PushAlongPathSolver(AbstractSolver):
    """
    Solver (used internally by PushSolver) to push an actor along a pre-determined
    path, assuming that the gripper of the agent is already in the right pose,
    "behind" the actor
    """

    def __init__(
        self,
        env: ClevrSkillsEnv,
        actor: sapien.Actor,
        path: np.ndarray,
        neutral_gripper_quat,
        distance_threshold: float = 0.02,
    ):
        """
        :param env: The ClevrSkillsEnv
        :param actor: The actor to be pushed.
        :param path: The path (numpy N x 3 array)
        :param neutral_gripper_quat: the neutral orientation of the gripper (upside down for most robots).
        :param distance_threshold: How close the gripper must stay to the path and the actor.
        """
        super().__init__(env)
        self.actor = actor
        self.path = path
        self._done = False
        self.distance_threshold = distance_threshold
        self._actor_distance = self._env._actor_distance
        self._prev_distance_to_target = (
            1000  # used to determine if the solver if still making progress
        )
        self._neutral_gripper_quat = neutral_gripper_quat
        self._prev_ee_to_actor_distance = 1000

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["current_target_pos"] = at_get_pos(path[0])
        self.action_state["path"] = path.tolist()
        self.action_state["neutral_gripper_quat"] = list(neutral_gripper_quat)
        self.action_state["distance_threshold"] = distance_threshold

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"Pushing {get_actor_description(self.actor)} along a path to {{pos:{self.path[-1]}}}"
        )
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        Get closest position of actor to planned path
        If more than Xcm away, abort declare done, return nop
        get the delta to the next position on the path
        If the next position is final position, abort, return nop, done
        If robot gripper is further than 0.5 cm from the actor, move it to
        that location, with correct orientation of the gripper
        Else move along that direction

        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        # Check if contact between gripper and object was lost:
        ee_to_actor_distance = self._actor_distance.distance(self.actor, self._env.agent.ee_links)
        if (
            ee_to_actor_distance > self._prev_ee_to_actor_distance
            and ee_to_actor_distance > self.distance_threshold
        ):
            self._done = True
            return self.return_hold_action(gripper=self.gripper_off)

        self._prev_ee_to_actor_distance = ee_to_actor_distance

        # Determine distance to each waypoint on the path
        distance_to_path = np.linalg.norm(self.path - self.actor.pose.p[:2], axis=1)

        # Check if done
        distance_to_target = distance_to_path[-1]
        progress = self._prev_distance_to_target - distance_to_target
        if (
            progress < (self.distance_threshold / 10)
            and distance_to_target < self.distance_threshold
        ):
            self._done = True
            return self.return_hold_action(gripper=self.gripper_off)
        self._prev_distance_to_target = distance_to_target

        # Get index of point on the path that is closest,
        closest_on_path_idx = np.argmin(distance_to_path)

        # Check if veered off-path
        if distance_to_path[closest_on_path_idx] > 0.05:
            self._done = True
            return self.return_hold_action(gripper=self.gripper_off)

        # determine next target
        target_path_idx = min(len(self.path) - 1, closest_on_path_idx + 1)
        distance_to_target = distance_to_path[target_path_idx]
        target = self.path[target_path_idx]
        self.action_state["current_target_pos"] = at_get_pos(target)

        # push!
        push_direction = target - self.actor.pose.p[:2]
        push_direction = push_direction / np.linalg.norm(push_direction)

        ee_pose = self._env.agent.ee_link.pose

        angle = np.arctan2(push_direction[1], push_direction[0])
        orientation = euler2quat(0, 0, angle)
        push_direction = min(distance_to_path[-1], 0.05) * push_direction
        ee_target_pose = sapien.Pose(
            ee_pose.p + [push_direction[0], push_direction[1], 0.0],
            qmult(orientation, self._neutral_gripper_quat),
        )

        return self.return_global_pose_action(ee_target_pose, gripper=self.gripper_off)

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done


class PushSolver(AbstractSolver):

    def __init__(self, env: ClevrSkillsEnv, actor: sapien.Actor, target_pose: Pose = None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be pushed to a certain location
        :param target_pose: Optional. The pose to place the actor.
        """
        super().__init__(env)

        self.actor = actor
        self.target_pose = target_pose
        self._done = False
        self._actor_distance = self._env._actor_distance
        self._scene = self._env._scene
        self._planning_space = 0.02  # how much distance to keep from target actor
        self._distance_threshold = 0.02  # when to declare the tasks as "done"
        self._ee_actor_distance_threshold = 0.04
        self._neutral_gripper_quat = self._env.agent.neutral_ee_quat

        self._prev_sub_solver_type = None

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["target_pose"] = at_get_pose(target_pose)
        self._prev_distance_to_target = None

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = low_level_act = (
            f"Pushing {get_actor_description(self.actor)} to {{pos:{self.target_pose.p}}}"
        )
        if self._sub_solver:
            _, low_level_act = self._sub_solver.get_current_action()
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        if self._sub_solver and self._sub_solver.is_done():
            self._prev_sub_solver_type = type(self._sub_solver)
            self._sub_solver = None

        if self._sub_solver is None:  # init new sub solver
            ee_pose = self._env.agent.ee_link.pose
            # Check if done:
            # we could have a check on the direction of the residual error,
            # compared to the last push direction,
            # and then keep pushing in orthogonal directions for "perfection".
            distance_to_target = np.linalg.norm(self.actor.pose.p[0:2] - self.target_pose.p[0:2])
            if distance_to_target < self._distance_threshold:
                ee_links = self._env.agent.ee_links
                actor_ee_distance = self._actor_distance.distance(self.actor, ee_links)
                if actor_ee_distance < self._ee_actor_distance_threshold:
                    # Move EE away from the actor
                    from clevr_skills.utils.geometry import unit

                    xy = self._ee_actor_distance_threshold * unit(
                        ee_pose.p[0:2] - self.actor.pose.p[0:2]
                    )
                    ee_target_pose = sapien.Pose(
                        ee_pose.p + [xy[0], xy[1], self._ee_actor_distance_threshold],
                        self._neutral_gripper_quat,
                    )
                    self._sub_solver = Move3dSolver(
                        self._env,
                        ee_target_pose,
                        match_ori=True,
                        vacuum=False,
                        target_pose_name=f"neutral pose away from actor",
                    )
                    return self._sub_solver.step(obs)
                else:
                    self._done = True
                    return self.return_hold_action(gripper=self.gripper_off)
            self._prev_distance_to_target = distance_to_target

            if self._prev_sub_solver_type is None or (
                self._prev_sub_solver_type == PushAlongPathSolver or ee_pose.p[2] > 0.05
            ):
                pushing_direction = (self.target_pose.p - self.actor.pose.p)[0:2]
                poses, scores = get_feasible_gripper_push_poses(
                    self._env,
                    self.actor,
                    pushing_direction,
                    self._neutral_gripper_quat,
                    space=self._planning_space,
                )

                pose = np.random.choice(poses, 1, p=scores)[0]
                self._sub_solver = Move3dSolver(
                    self._env,
                    pose,
                    match_ori=True,
                    vacuum=False,
                    extend_bounds=0.0,
                    target_pose_name=f"a good place to push {get_actor_description(self.actor)}",
                )
            else:
                path, _collisions = self.plan_path()
                if path is None:
                    # path planning failed; try from another angle on the next step()
                    self._prev_sub_solver_type = None
                else:
                    self._sub_solver = PushAlongPathSolver(
                        self._env,
                        self.actor,
                        path,
                        self._neutral_gripper_quat,
                        distance_threshold=self._distance_threshold,
                    )

        if self._sub_solver:
            return self._sub_solver.step(obs)
        return self.return_hold_action(gripper=self.gripper_off)

    def plan_path(self):
        """
        :return: A 2D path for pushing the actor.
        """
        # create a 2D planner
        planner_2d = get_cartesian_robot_mplib_planner_with_attached_actors(
            translation_axes=[0, 1],
            neutral_ee_pos=[0, 0, self._planning_space],
            attached_actors=[self.actor],
            mesh=True,
            actor_distance=self._actor_distance,
        )
        initial_qpos = np.copy(self.actor.pose.p[0:2])
        target_qpos = np.copy(self.target_pose.p[0:2])

        time_step = 1.0 / self._env.control_freq

        # set the point cloud
        point_cloud = self._env.get_point_cloud_for_planning(
            excluded_grasped_actors=False, exclude_actors=[self.actor], num_floor_points=0
        )
        point_cloud = np.ascontiguousarray(point_cloud, dtype=np.float32)
        planner_2d.update_point_cloud(point_cloud, resolution=0.01)

        # plan
        planning_result = plan_path_qpos(
            planner_2d,
            time_step,
            initial_qpos,
            target_qpos,
            check_collisison=True,
            replan_collisions=True,
            simplify_path=True,
        )
        return planning_result

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
