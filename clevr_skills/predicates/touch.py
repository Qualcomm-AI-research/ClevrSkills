# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import qinverse, qmult, quat2mat

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.controller import quat2smallest_axangle
from clevr_skills.utils.sapien_utils import check_actor_static

from .predicate import EnvPredicate


class TouchPredicate(EnvPredicate):
    """
    Predicate that requires agent to touch the actor.
    Optionally, push it from it's initially position.
    Optionally, push it over (>45 degree orientation change w.r.t. vertical axis).
    If neither push nor topple is True, the actor must stay at original pose.
    """

    def __init__(self, env, actor: sapien.Actor, push: bool = False, topple=False, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be touched.
        :param push: if true, push the actor after touching it.
        :param topple: if true, push enough so that the actor topples over.
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"Touch {actor.name}"
        if topple:
            name = name + " and push over"
        elif push:
            name = name + " and push"
        super().__init__(env, name)

        self.actor = actor
        self.initial_pose = None
        self.push = push
        self.topple = topple
        self._actor_distance: ActorDistance = env._actor_distance
        self._touched = False
        self._grasped = False  # once the actor is grasped, can never have success

        self._displacement_threshold = 0.1
        self._topple_threshold = 45  # in degrees

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        if self.initial_pose is None:
            self.initial_pose = self.actor.pose

        actor_static = check_actor_static(self.actor)

        # Determine if grasped
        actor_grasped = self._env.agent.check_grasp(self.actor)
        if actor_grasped:
            self._grasped = actor_grasped

        # Determine if touched:
        if not self._touched:
            contacts = self._env.get_actor_contacts(self.actor)
            if len(set(contacts).intersection(set(self._env.agent.ee_links))) > 0:
                self._touched = True

        # Determine if a original pose:
        distance_moved = np.linalg.norm(self.actor.pose.p - self.initial_pose.p)
        actor_at_pos = distance_moved < self._displacement_threshold
        _, angle = quat2smallest_axangle(qmult(self.actor.pose.q, qinverse(self.initial_pose.q)))
        angle_diff = abs(angle) / np.pi
        actor_at_pose = (angle_diff < 0.1) and actor_at_pos

        # Determine if toppled
        mat = quat2mat(qmult(self.actor.pose.q, qinverse(self.initial_pose.q)))
        topple_angle = np.rad2deg(np.arccos(abs(np.dot(mat[:, 2], [0, 0, 1]))))
        toppled = topple_angle >= self._topple_threshold

        success = self._touched and actor_static

        if self.topple:
            success = success and not actor_at_pos and toppled
        elif self.push:
            success = success and not actor_at_pos and not toppled
        else:
            success = not self._grasped and success and actor_at_pose

        return {
            "actor_static": actor_static,
            "actor_grasped": actor_grasped,
            "grasped_sometime": self._grasped,
            "actor_at_pos": actor_at_pos,
            "actor_at_pose": actor_at_pose,
            "touched": self._touched,
            "toppled": toppled,
            "topple_angle": topple_angle,
            "distance_moved": distance_moved,
            "success": success,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 3]).
        """
        eval = self.evaluate()

        if eval["success"]:
            return 3.0

        distance_to_actor = self._actor_distance.distance(self.actor, self._env.agent.ee_links)
        reward = np.exp(-5.0 * distance_to_actor)

        if self.topple:
            reward += min(1.0, eval["topple_angle"] / self._topple_threshold)
        if self.push:
            reward += 1.0 - min(1.0, eval["topple_angle"] / self._topple_threshold)
        if self.topple or self.push:
            reward += min(1.0, eval["distance_moved"] / self._displacement_threshold)
        else:
            reward += 0.5 if eval["actor_at_pos"] else 0.0
            reward += 0.5 if eval["actor_at_pose"] else 0.0

        return reward
