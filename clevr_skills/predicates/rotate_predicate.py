# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Dict, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import qinverse, qmult

from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.predicate_utils import rotate_pose_z
from clevr_skills.utils.sapien_utils import check_actor_static

from .at_pose import AtPose

ANGLE_TOLERANCE_DEG = 1


class RotatePredicate(AtPose):
    """
    Predicate for rotating the object at current position.
    """

    def __init__(
        self,
        env: Any,
        actor: sapien.Actor,
        degree: float,
        clockwise: bool,
        restore: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be rotated.
        :param degree: how many degrees should the actor be rotated relative to the initial orientation
        :param clockwise: if true, rotate clockwise else rotate anti-clockwise.
        :param restore: if true, restore the original orientation.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"Rotate {actor.name} by {degree}"

        self.degree = -degree if clockwise else degree
        self.clockwise = clockwise
        self.restore = restore
        pose = rotate_pose_z(actor.pose, self.degree)
        if restore:
            pose = rotate_pose_z(pose, -self.degree)
        super().__init__(env, actor, pose, name=name)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate whether the actor has reached the requested rotation."""

        actor_static = check_actor_static(self.actor)
        actor_grasped = self._env.agent.check_grasp(self.actor)

        # Compute whether actor is at pose
        _, angle = quat2smallest_axangle(qmult(self.actor.pose.q, qinverse(self.pose.q)))
        angle_diff = abs(np.rad2deg(angle))
        actor_at_pose = angle_diff < ANGLE_TOLERANCE_DEG

        success = actor_at_pose
        return {
            "actor_static": actor_static,
            "actor_grasped": actor_grasped,
            "actor_at_pose": actor_at_pose,
            "success": success,
        }
