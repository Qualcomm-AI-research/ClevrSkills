# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.sapien_utils import check_actor_static

from .predicate import EnvPredicate


class AtPositionPushed(EnvPredicate):
    """
    Predicate similar to AtPosition, but doesn't allow for grasping.
    """

    def __init__(self, env, actor: sapien.Actor, pose: Pose, match_ori=False, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be pushed to a certain position.
        :param pose: The target position (orientation not supported yet)
        :param match_ori: Not supported yet.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"{actor.name} pushed to {pose}"
        super().__init__(env, name)
        self.pose = pose
        self.actor = actor
        self._match_ori = match_ori
        self._actor_has_been_grasped = False
        self._actor_distance: ActorDistance = env._actor_distance
        self._distance_threshold = 0.001

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        actor_static = check_actor_static(self.actor)
        actor_grasped = self._env.agent.check_grasp(self.actor)

        actor_at_pos = np.linalg.norm(self.actor.pose.p[:2] - self.pose.p[:2]) < 0.03

        self._actor_has_been_grasped = actor_grasped or self._actor_has_been_grasped

        # Determine distance and contact
        ee_links = self._env.agent.ee_links
        actor_ee_distance = self._actor_distance.distance(self.actor, ee_links)
        contact_actors = self._env.get_actor_contacts(self.actor)
        actor_ee_contact = len(set(ee_links).intersection(set(contact_actors))) > 0

        success = (
            actor_static
            and (not self._actor_has_been_grasped)
            and actor_at_pos
            and not actor_ee_contact
            and actor_ee_distance >= self._distance_threshold
        )

        return {
            "actor_static": actor_static,
            "actor_grasped": actor_grasped,
            "actor_has_been_grasped": self._actor_has_been_grasped,
            "actor_at_pos": actor_at_pos,
            "actor_ee_distance": actor_ee_distance,
            "actor_ee_contact": actor_ee_contact,
            "success": success,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6]).
        """
        eval = self.evaluate()

        if eval["actor_has_been_grasped"]:  # no grasping!
            return 0.0
        if eval["actor_at_pos"]:  # max 5 reward
            # if the actor is at the target position, reward "no contact" and
            # getting away from the actor
            return (
                3.0
                + (0 if eval["actor_ee_contact"] else 1)
                + np.exp(
                    min(0, 100 * (max(0, eval["actor_ee_distance"]) - self._distance_threshold))
                )
            )
        # max 3 reward
        # Reward:
        # + low distance between EE and actor
        # + getting the actor close to the target
        # + EE being "behind" the actor, relative the target pose
        reward = np.exp(-5 * (max(0, eval["actor_ee_distance"]))) + np.exp(
            -5 * np.linalg.norm(self.actor.pose.p[0:2] - self.pose.p[0:2])
        )

        # compute whether the EE is "behind" the actor, able to push it towards the target
        ee_to_actor_vec = self._env.agent.ee_link.pose.p[0:2] - self.actor.pose.p[0:2]
        actor_to_target_vec = self.actor.pose.p[0:2] - self.pose.p[0:2]
        ee_to_actor_vec = ee_to_actor_vec / np.linalg.norm(ee_to_actor_vec)
        actor_to_target_vec = actor_to_target_vec / np.linalg.norm(actor_to_target_vec)
        d = np.dot(
            ee_to_actor_vec, actor_to_target_vec
        )  # 1: EE is at the correct side; -1: EE is the opposite side
        reward += np.exp(d - 1)

        return reward
