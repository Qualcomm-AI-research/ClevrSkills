# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import sapien.core as sapien

from clevr_skills.utils.predicate_utils import rotate_pose_z

from .at_pose import AtPose


class RotatePredicate(AtPose):
    """
    Predicate for rotating the object at current position.
    """

    def __init__(
        self, env, actor: sapien.Actor, degree: float, clockwise: bool, restore=False, name=None
    ):
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
        self.restore = restore
        pose = rotate_pose_z(actor.pose, self.degree)
        if restore:
            pose = rotate_pose_z(pose, -self.degree)
        super().__init__(env, actor, pose, name=name)
