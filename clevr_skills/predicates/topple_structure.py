# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.touch import TouchPredicate
from clevr_skills.utils.actor_distance import ActorDistance

from .predicate import EnvPredicate


class ToppleStructurePredicate(EnvPredicate):
    """
    This predicate will return success if all actors are on the ground, indicating that
    the structure that the actors were part of has been toppled.
    """

    ON_GROUND_IMPULSE = 1e-4

    def __init__(self, env, actors: List[sapien.Actor], name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actors: list of actors that make up the structure that must be toppled.
        :param name: Descriptive name of the predicate.
        :param clear_distance: How far away the actor must be from other actors before success.
        """
        name = name if name else f"Topple structure of {len(actors)} actors"
        super().__init__(env, name)
        self.actors = actors
        self._actor_distance: ActorDistance = env._actor_distance

        self._topple_predicates = [TouchPredicate(env, actor, topple=True) for actor in self.actors]

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        result = {}

        num_on_ground = 0

        env: ClevrSkillsEnv = self._env
        for actor in self.actors:
            impulse = env.get_contact(actor, env._ground)
            on_ground = impulse > ToppleStructurePredicate.ON_GROUND_IMPULSE
            result[f"actor {actor.name} on ground"] = on_ground
            if on_ground:
                num_on_ground += 1

        result["success"] = num_on_ground == len(self.actors)

        return result

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 3*num_actors].
        """
        reward = 0.0
        max_topple_predicate_reward = 0.0
        env: ClevrSkillsEnv = self._env
        for actor_idx, actor in enumerate(self.actors):
            impulse = env.get_contact(actor, env._ground)
            on_ground = impulse > ToppleStructurePredicate.ON_GROUND_IMPULSE
            if on_ground:
                reward += 3.0
            else:
                r = self._topple_predicates[actor_idx].compute_dense_reward()
                max_topple_predicate_reward = max(max_topple_predicate_reward, r)

        return reward + max_topple_predicate_reward
