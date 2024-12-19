# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien

from clevr_skills.utils.actor_distance import ActorDistance

from .predicate import EnvPredicate


def is_actor_in_contact(
    scene: sapien.Scene, actor: sapien.Actor, robot: sapien.Articulation = None
):
    """
    :param scene: Used to retrieve contacts and gravity.
    :param actor: The actor.
    :param robot: the robot (if specified) is ignored in the contact check.
    :return: True if the actor is not touching any other actor (except for the robot
    that might be grasping it).
    """
    contacts = scene.get_contacts()
    for contact in contacts:
        other_actor = None
        if contact.actor0 == actor:
            other_actor = contact.actor1
        elif contact.actor1 == actor:
            other_actor = contact.actor0

        if other_actor is None:
            continue  # this contact does not involve "actor"
        if (
            not robot is None
            and isinstance(other_actor, sapien.Link)
            and other_actor.get_articulation() == robot
        ):
            continue  # this contact is with the robot itself (which is OK)
        return True  # actor is in contact with another actor.
    return False


class PickPredicate(EnvPredicate):
    """
    Predicate for picking an actor.
    """

    def __init__(self, env, actor: sapien.Actor, name=None, clear_distance=0.02):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be picked.
        :param name: Descriptive name of the predicate.
        :param clear_distance: How far away the actor must be from other actors before success.
        """
        name = name if name else f"Pick {actor.name}"
        super().__init__(env, name)
        self.actor = actor
        self._actor_distance: ActorDistance = env._actor_distance
        self._clear_distance = clear_distance

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        Successful if an actor is picked by the agent.
        """
        actor_grasped = self._env.agent.check_grasp(self.actor)

        if actor_grasped:
            # It is expensive to calculate the distance
            other_actors = [
                actor for actor in self._env._scene.get_all_actors() if not actor is self.actor
            ] + [
                articulation
                for articulation in self._env._scene.get_all_articulations()
                if not articulation is self._env.agent.robot
            ]
            clearance = self._actor_distance.distance(self.actor, other_actors)
        else:
            clearance = 0.0

        success = actor_grasped and clearance >= self._clear_distance

        return {
            "actor_grasped": actor_grasped,
            "clearance": clearance,  # distance from all other actors (except the robot)
            "success": success,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 4]).
        """

        eval = self.evaluate()

        if eval["success"]:
            return 4.0

        if eval["actor_grasped"]:
            # reward getting away from other actors
            return 2.0 + np.clip(eval["clearance"] / self._clear_distance, 0, 1)

        # reach for actor; max 2 reward
        # reward EE for getting close to the actor
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
