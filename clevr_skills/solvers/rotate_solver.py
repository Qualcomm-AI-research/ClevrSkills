# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Dict, Optional

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import qinverse, qmult

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pose
from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.visualize_prompt import get_actor_description

POSITION_TOLERANCE = 0.01
Z_AXIS_ALIGNMENT_THRESHOLD = 0.95
ANGLE_TOLERANCE_DEG = 1


class RotateSolverError(RuntimeError):
    """Raised when the rotate solver cannot compute a valid action."""


class RotateSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        actor: sapien.Actor,
        target_pose: Pose,
        drop_distance: float = 0.02,
        target_pose_name: Optional[str] = None,
        action_state_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param env: ClevrSkills environment with privileged state access.
        :param actor: Actor that is already grasped and should be rotated.
        :param target_pose: Desired actor pose.
        :param drop_distance: Height relative to target used by action traces.
        :param target_pose_name: Optional semantic name for the target pose.
        :param action_state_extra: Extra action-trace metadata to record.
        """
        super().__init__(env)
        if not action_state_extra:
            action_state_extra = {}

        self.actor = actor
        self._target_pose = target_pose
        self._drop_distance = drop_distance
        self._actor_distance = self._env._actor_distance
        self._lift_z = None  # set when the actor is first lifted
        self._lift_delta = 0.03
        self._done = False

        self.action_state["actor"] = at_get_actor(actor)
        self.action_state["target_pose"] = at_get_pose(target_pose)
        self.action_state["match_ori_2d"] = True
        self.action_state["drop_distance"] = drop_distance
        self.action_state["target_pose_name"] = target_pose_name
        self.action_state.update(action_state_extra)

    def get_current_action(self) -> Dict[str, str]:
        """Return a natural-language label for the current rotate action."""
        angle = self.action_state.get("angle", None)
        clockwise = self.action_state.get("clockwise", None)
        if angle is not None and clockwise is not None:
            mid_level_act = low_level_act = (
                f"Rotate {get_actor_description(self.actor)} by {angle} degrees "
                f"{'clockwise' if clockwise else 'anti-clockwise'}"
            )
        else:
            mid_level_act = low_level_act = f"Rotate {get_actor_description(self.actor)}"
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs: Any) -> np.ndarray:
        """Compute the next end-effector action for rotating the grasped actor."""
        ee_pose = self._env.agent.ee_link.pose
        actor_pose = self.actor.pose

        delta_p = [0, 0, 0]
        gripper = self.gripper_on  # vacuum gripper is on by default

        target_pos = np.copy(self._target_pose.p)
        at_target_xy = np.linalg.norm((actor_pose.p - target_pos)[0:2]) < POSITION_TOLERANCE

        # check if there must be any rotation about the z-axis
        at_target_rot_z = False
        axis, angle = quat2smallest_axangle(qmult(qinverse(actor_pose.q), self._target_pose.q))
        if abs(axis[2]) >= Z_AXIS_ALIGNMENT_THRESHOLD:
            at_target_rot_z = abs(np.rad2deg(angle)) < ANGLE_TOLERANCE_DEG

        if self._env.agent.check_grasp(self.actor):

            target_ori = qmult(qmult(self._target_pose.q, qinverse(actor_pose.q)), ee_pose.q)

            if at_target_rot_z:
                self._done = True

            if not at_target_xy:
                delta_p = target_pos - actor_pose.p
        else:
            raise RotateSolverError("Rotate solver called on ungrasped actor.")

        target_pose = Pose(ee_pose.p + delta_p, target_ori)
        return self.return_global_pose_action(target_pose, gripper=gripper)

    def is_done(self) -> bool:
        """Return whether the actor has reached the target orientation."""
        return self._done
