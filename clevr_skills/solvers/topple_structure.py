# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, List

import numpy as np
import sapien.core as sapien

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.topple_structure import ToppleStructurePredicate
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.visualize_prompt import get_actor_description

from .touch_solver import Move3dTouchSolver


class ToppleStructure(AbstractSolver):
    def __init__(self, env: ClevrSkillsEnv, actors: List[sapien.ActorBase]):
        """
        :param env: The ClevrSkillsEnv.
        :param actors: List of actors that make up the structure that should be toppled.
        """
        super().__init__(env)

        self.actors = actors
        self._done = False
        self._action_label = "Topple the structure that consists of "
        for actor_idx, actor in enumerate(actors):
            if len(actors) > 0 and (actor_idx == (len(actors) - 1)):
                self._action_label += " and "
            elif actor_idx > 0:
                self._action_label += ", "
            self._action_label += get_actor_description(actor)

        self.action_state["actors"] = [at_get_actor(actor) for actor in actors]

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        low_level_act = (
            self._sub_solver.get_current_action()["low_act_label"]
            if self._sub_solver
            else "Picking next object to topple"
        )
        return {"mid_act_label": self._action_label, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        if self._sub_solver:
            if self._sub_solver.is_done():
                self._sub_solver = None
            else:
                # check if actor is on ground. If so, get rid of solver
                impulse = self._env.get_contact(self._sub_solver.actor, self._env._ground)
                on_ground = impulse > ToppleStructurePredicate.ON_GROUND_IMPULSE
                if on_ground:
                    self._sub_solver = None

        if self._sub_solver is None:
            actor_ratings = []  # how likely to pick this actor for toppling next.
            for actor in self.actors:
                impulse = self._env.get_contact(actor, self._env._ground)
                on_ground = impulse > ToppleStructurePredicate.ON_GROUND_IMPULSE
                if on_ground:
                    actor_ratings.append(0)
                else:
                    pose = actor.pose
                    rating = np.exp(-5.0 * pose.p[2])
                    actor_ratings.append(rating)
            if np.sum(actor_ratings) > 0:
                actor_ratings = np.array(actor_ratings) / np.sum(actor_ratings)
                actor = np.random.choice(self.actors, 1, p=actor_ratings)[0]
                self._sub_solver = Move3dTouchSolver(self._env, actor, topple=True)

        if self._sub_solver:
            return self._sub_solver.step(obs)
        return self.return_hold_action(self.gripper_off)

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
