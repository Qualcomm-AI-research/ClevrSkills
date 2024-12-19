# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from itertools import permutations
from typing import List

import sapien.core as sapien

from .on_top import OnTop
from .order import Order


class StackPredicate(Order):
    """
    This class implements a predicate for stacking.
    """

    def __init__(self, env, actors: List[sapien.Actor]):
        """
        :param env: The ClevrSkillsEnv.
        :param actors: list of actors to be stacked in order. Actor at position 0 goes at the bottom,
        actor at position 1 goes on top of actor 0, and so on.
        """
        self._predicates = []
        self._num_preds = len(actors) - 1
        track_set = set()
        for idx in range(len(actors) - 1):
            self._predicates.append(OnTop(env, actors[idx + 1], actors[idx]))
            track_set.add((idx + 1, idx))

        # will work for small len(actors) -- larger lists will become too expensive
        for i, j in permutations(range(len(actors)), 2):
            if (i, j) not in track_set:
                self._predicates.append(OnTop(env, actors[i], actors[j]))

        super().__init__("stack", self._predicates)

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        counts = 0
        for pred in self._predicates:
            if pred.evaluate()["success"]:
                counts += 1

        if counts == self._num_preds:
            return {"success": True}

        result = {"success": False}
        while (
            self._current_idx < len(self._predicates)
            and self._predicates[self._current_idx].evaluate()["success"]
        ):
            self._current_idx += 1

        for predicate in self._predicates:
            predicate_eval_dict = predicate.evaluate()
            name = predicate.name
            for key, value in predicate_eval_dict.items():
                result[f"{name}_{key}"] = value

        return result

    def get_current_predicate(self):
        """
        :return: current predicate being solved.
        """
        if self._current_idx < len(self._predicates):
            return self._predicates[self._current_idx]
        return None

    def compute_dense_reward(self):
        """
        :return: dense reward.
        """
        reward = 0.0
        for idx, predicate in enumerate(
            self._predicates
        ):  # no reward for predicates that are beyong the current predicate
            pred_reward = predicate.compute_dense_reward()
            self._max_predicate_reward[idx] = max(self._max_predicate_reward[idx], pred_reward)
            if idx < self._current_idx:
                reward += self._max_predicate_reward[idx]
            else:
                reward += pred_reward
        return reward
