# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, Optional

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.solvers.move3d_solver import Move3dSolver
from clevr_skills.solvers.pick_solver import PickSolver
from clevr_skills.solvers.place_solver import PlaceOnActorSolver, PlaceSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pose
from clevr_skills.utils.actor_placement import get_random_non_intersecting_pose_v2
from clevr_skills.utils.controller import align_quat
from clevr_skills.utils.logger import log
from clevr_skills.utils.visualize_prompt import get_actor_description


class PickMove3dPlaceSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        actor: sapien.Actor,
        target_pose: Pose = None,
        target_pose_name: str = None,
        target_actor: sapien.Actor = None,
        match_ori: bool = False,
        lift: float = 0.03,
        action_state_extra: Optional[Dict] = None,
    ):
        """
        :param env: The ClevrSkillsEnv
        :param actor: The actor to be picked-and-placed
        :param target_pose: Optional. The pose to place the actor.
        :param target_pose_name: a human-readable description of the target pose.
        Please specify when target_actor is None such that the action label can be informative
        :param target_actor: Optional. The actor to-be-placed upon. May be None.
        If both target_pose and target_actor are specified, the pose is relative to the actor pose.
        If target_actor is specified, but no pose is given, a pose is selected using
        get_random_non_intersecting_pose_v2(). If neither target_pose nor target_actor are
        given, then the solver will declare itself "done" after lifting the object.
        :param match_ori: Match the orientation of target_pose. If False, a neutral orientation
        will be used.
        :param lift: How much to lift the actor above the initial pose at pickup.
        Without lifting, actors below the picked-up-actor could be pushed aside during
        horizontal transport.
        :param action_state_extra: any additional information to add to the action state.
        E.g., how this target pose was computed.
        """
        super().__init__(env)
        if not action_state_extra:
            action_state_extra = {}

        self.actor = actor
        self.target_actor = target_actor
        self.target_pose = target_pose
        self._target_pose_name = (
            target_pose_name
            if target_pose_name
            else (get_actor_description(target_actor) if target_actor else "UNSPECIFIED POSE")
        )
        self._target_pose_is_sampled = (
            False  # used to know if self.target_pose  is the result of random sampling
        )
        self._lift_z = lift
        self._match_ori = match_ori
        self._done = False
        self._actor_distance = self._env._actor_distance

        # If no target pose is given, the target is simply to lift the object
        if self.target_pose is None and self.target_actor is None:
            self._lift_only = True
            self.target_pose = sapien.Pose([0, 0, self._lift_z]).transform(self.actor.pose)
        else:
            self._lift_only = False

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["target_actor"] = at_get_actor(target_actor)
        self.action_state["target_pose"] = at_get_pose(target_pose)
        self.action_state["target_pose_name"] = target_pose_name
        self.action_state["match_ori"] = match_ori
        self.action_state["lift"] = lift
        self.action_state.update(action_state_extra)

    def _compute_relative_target_pose(self):
        """
        :return: a pose, relative to self.target_actor, where the self.actor can be placed.
        """
        actor_bounds = self._actor_distance.get_bounds(self.actor, actor_pose=sapien.Pose())
        target_actor_bounds = self._actor_distance.get_bounds(self.target_actor)

        offset_z = (
            target_actor_bounds[1, 2] - actor_bounds[0, 2] + 0.01
        )  # top of dest actor, minus the bottom of the actor, plus a small offset

        target_pose, is_intersecting = get_random_non_intersecting_pose_v2(
            self._env,
            self.actor,
            self.target_actor,
            offset_z=offset_z,
            exclude_actors=[self._env.agent.robot, self.target_actor, self._env._ground],
            angle_z=[0, 2 * np.pi],
            max_attempts=100,
            grow_actor_bounds=0.02,
        )
        self._target_pose_is_sampled = True

        if is_intersecting:
            log("Warning: target pose is intersecting!")

        return self.target_actor.pose.inv().transform(target_pose)

    def get_target_pose(self) -> sapien.Pose:
        """
        :return: The absolute target pose.
        """
        if self.target_pose is None:
            self.target_pose = self._compute_relative_target_pose()

        if self.target_actor is None:
            return self.target_pose
        return self.target_actor.pose.transform(self.target_pose)

    def actor_is_at_target_pose(
        self, target_pose: sapien.Pose, distance_threshold=0.03, angle_threshold=5.0
    ) -> bool:
        """
        :param target_pose: The absolute target pose.
        :param distance_threshold: in meters.
        :param angle_threshold: in degrees.
        :return: True, is self.actor is at the target_pose, within the thresholds.
        """
        # Check position
        actor_at_pos = np.linalg.norm(self.actor.pose.p - target_pose.p) <= distance_threshold

        if self._match_ori:
            _, angle = quat2axangle(qmult(self.actor.pose.q, qinverse(target_pose.q)))
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            actor_at_ori = np.rad2deg(angle) <= angle_threshold
        else:
            actor_at_ori = True

        return actor_at_pos and actor_at_ori

    def actor_is_above_target(self, target_pose: sapien.Pose, distance_threshold=0.03) -> bool:
        """
        :param target_pose: The absolute target pose.
        :param distance_threshold: Distance in meters, how close self.actor must be to the target pose.
        X, Y axes only, height Z is ignored.
        :return: True if actor is above threshold within the threshold.
        """
        return np.linalg.norm(self.actor.pose.p[:2] - target_pose.p[:2]) <= distance_threshold

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: The language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        if not self.target_actor is None:
            mid_level_act = (
                f"Place {get_actor_description(self.actor)} on "
                f"{get_actor_description(self.target_actor)} without hitting anything else"
            )
        elif not self.target_pose is None:
            mid_level_act = (
                f"Place {get_actor_description(self.actor)} on "
                f"{{pos:{self.target_pose.p}}} without hitting anything else"
            )
        else:
            mid_level_act = f"Pick up {get_actor_description(self.actor)}"

        if self._sub_solver is None:
            low_level_act = "Do nothing"
        else:
            low_level_act = self._sub_solver.get_current_action()["low_act_label"]

        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        if self._sub_solver and self._sub_solver.is_done():
            self._sub_solver = None

        # Check if Move3dSolver is stuck; if so, get rid of it and a new
        if (
            self._sub_solver
            and self._target_pose_is_sampled
            and isinstance(self._sub_solver, Move3dSolver)
        ):
            m3ds: Move3dSolver = self._sub_solver
            if m3ds.is_stuck():
                log("Move3D solver is stuck! Will sample a new relative target pose")
                self._sub_solver = None
                self.target_pose = None  # new target pose will be sampled by self.get_target_pose()

        # compute current target pose
        target_pose = self.get_target_pose()
        ee_pose = self._env.agent.ee_link.pose

        if self._sub_solver is None:
            if self._env.agent.check_grasp(self.actor):
                if self._lift_only:
                    # Simply lift a bit more and then hold still
                    ee_target_pose = sapien.Pose([0, 0, 0.05]).transform(ee_pose)
                    self._sub_solver = Move3dSolver(
                        self._env,
                        ee_target_pose,
                        match_ori=False,
                        vacuum=True,
                        target_pose_name=self._target_pose_name,
                        check_done=False,
                    )
                elif self.actor_is_above_target(target_pose):
                    # Create PlaceSolver
                    if self.target_actor:
                        self._sub_solver = PlaceOnActorSolver(
                            self._env,
                            self.actor,
                            self.target_actor,
                            target_pose,
                            self._match_ori,
                            drop_distance=self._lift_z,
                        )
                    else:
                        self._sub_solver = PlaceSolver(
                            self._env,
                            self.actor,
                            target_pose,
                            self._match_ori,
                            drop_distance=self._lift_z,
                            target_pose_name=self._target_pose_name,
                        )
                else:
                    # create Move3dSolver
                    # If not feasible, try another pose, if possible
                    target_pose = sapien.Pose([0.0, 0.0, self._lift_z]).transform(
                        target_pose
                    )  # the target for the move3d is a bit above true target
                    delta_p = target_pose.p - self.actor.pose.p

                    delta_q = qmult(
                        qinverse(align_quat(self.actor.pose.q, target_pose.q)), target_pose.q
                    )

                    ee_target_pose = sapien.Pose(ee_pose.p + delta_p, qmult(delta_q, ee_pose.q))

                    self._sub_solver = Move3dSolver(
                        self._env,
                        ee_target_pose,
                        match_ori=self._match_ori,
                        vacuum=True,
                        target_pose_name=self._target_pose_name,
                    )
            elif (
                self._done
                or (not self._env.any_contact(self.actor))
                or self.actor_is_at_target_pose(target_pose)
            ):
                if self.actor_is_at_target_pose(target_pose):
                    self._done = True
                    return self.return_hold_action(self.gripper_off)
                # Determine distance of gripper to actor. If very close, keep gripper on
                d = self._actor_distance.distance(self.actor, self._env.agent.ee_links)
                return self.return_hold_action(self.gripper_on if d < 0.005 else self.gripper_off)
            else:
                if np.linalg.norm(self.actor.pose.p[:2] - ee_pose.p[:2]) <= 0.03:
                    self._sub_solver = PickSolver(self._env, self.actor, self._lift_z)
                else:
                    bounds = self._actor_distance.get_bounds(self.actor)
                    ee_target_pose_for_picking = sapien.Pose(
                        [self.actor.pose.p[0], self.actor.pose.p[1], bounds[1, 2] + 0.03], ee_pose.q
                    )
                    self._sub_solver = Move3dSolver(
                        self._env,
                        ee_target_pose_for_picking,
                        match_ori=self._match_ori,
                        vacuum=False,
                        target_pose_name=get_actor_description(self.actor),
                    )

        if self._sub_solver:
            return self._sub_solver.step(obs)
        else:
            return self.return_hold_action(self.gripper_off)

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
