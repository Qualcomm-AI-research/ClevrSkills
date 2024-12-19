# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse
from sapien.core import Pose
from transforms3d.quaternions import qinverse, qmult

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.sapien_utils import check_actor_static

from .predicate import EnvPredicate


class AtPosition(EnvPredicate):
    """
    Predicate that specifies the position where the actor should be placed.
    Agnostic to the final orientation of the object.
    """

    def __init__(self, env, actor: sapien.Actor, pose: Pose, match_ori=False, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be placed at the specified position/pose.
        :param pose: The pose. By default, the predicate only cares about the position.
        :param match_ori: Should the 2D orientation be matched, too?
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"{actor.name} at {pose}"
        super().__init__(env, name)
        self.pose = pose
        self.actor = actor
        self._match_ori = match_ori
        self._actor_distance: ActorDistance = env._actor_distance

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        actor_static = check_actor_static(self.actor)
        actor_grasped = self._env.agent.check_grasp(self.actor)

        # Compute impulse between ground and actors
        contacts = self._env._scene.get_contacts()
        ground_impulse = get_pairwise_contact_impulse(contacts, self._env._ground, self.actor)

        actor_touching_ground = np.linalg.norm(ground_impulse) > 1e-5

        actor_at_pos = np.linalg.norm(self.actor.pose.p[:2] - self.pose.p[:2]) < 0.05

        success = actor_static and (not actor_grasped) and (actor_touching_ground) and actor_at_pos

        return {
            "actor_static": actor_static,
            "actor_grasped": actor_grasped,
            "actor_touching_ground": actor_touching_ground,
            "actor_at_pos": actor_at_pos,
            "success": success,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6])
        """
        eval = self.evaluate()

        if eval["success"]:
            return 6.0

        if (self._match_ori and eval["actor_at_pose"]) or (
            not self._match_ori and eval["actor_at_pos"]
        ):
            return 5.0 + (0.0 if eval["actor_grasped"] else 1.0)
        if eval["actor_grasped"]:  # max 5 reward
            CLOSE = 0.03
            dist_xy = np.linalg.norm(self.actor.pose.p[0:2] - self.pose.p[0:2])
            dist_z = self.actor.pose.p[2] - self.pose.p[2]
            # need to hoover

            is_close = dist_xy < CLOSE

            if self._match_ori:
                _, angle = quat2smallest_axangle(qmult(self.actor.pose.q, qinverse(self.pose.q)))
                angle_diff = abs(angle) / np.pi
                is_close = is_close and angle_diff < 0.05

            if is_close:  # max 5 reward
                # reward getting close to the object in the Z direction
                if dist_z < 0.0:
                    return 4.0
                return 4.0 + np.exp(-6.0 * dist_z)

            # max 4 reward
            # reward getting close in the XY direction, and being above a
            # certain height above the target
            dist_xy_reward = np.exp(-max(0.0, dist_xy - CLOSE))
            dist_z_reward = np.exp(-max(0.0, 6.0 * (-dist_z + 0.15)))
            angle_reward = 1.0 - angle_diff if self._match_ori else 0.0

            return (
                2
                + dist_z_reward
                + ((dist_xy_reward + angle_reward) / 2 if self._match_ori else dist_xy_reward)
            )

        # reach for actor; max 2 reward
        # reward EE getting close to top of object
        CLOSE = 0.03
        REALLY_CLOSE = 0.005
        actor_bounds = self._actor_distance.get_bounds(self.actor)
        tcp_pos = self._env.agent.ee_link.pose.p
        dist_xy = np.linalg.norm(self.actor.pose.p[0:2] - tcp_pos[0:2])
        dist_z = tcp_pos[2] - actor_bounds[1, 2]

        is_close_xy = dist_xy < CLOSE
        if is_close_xy:  # max 2 reward
            # reward getting close to the object in the Z direction
            if dist_z < REALLY_CLOSE:
                return 2.0
            return 1.0 + np.exp(-dist_z)
        # max 1 reward
        # reward getting close in the XY direction
        return np.exp(-max(0.0, dist_xy - CLOSE))
