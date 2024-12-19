# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np

from .predicate import Predicate


class Order(Predicate):
    """
    This class implements a list of predicates that must be satisfied in-order.
    However, not all the predicates need to be satisfied at the same time.
    """

    def __init__(self, name: str, predicates: List[Predicate]):
        """
        :param name: Descriptive name of the predicate.
        :param predicates: list of predicates to solve in order.
        """
        super().__init__(name)
        self._predicates = predicates
        self._current_idx = 0

        self._past_reward = 0
        self._max_predicate_reward = [-np.inf] * len(self._predicates)

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        For order, the evaluation info of sub-predicates is included.
        """
        if self._current_idx == len(self._predicates):
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
            self._predicates[: self._current_idx + 1]
        ):  # no reward for predicates that are beyong the current predicate
            pred_reward = predicate.compute_dense_reward()
            self._max_predicate_reward[idx] = max(self._max_predicate_reward[idx], pred_reward)
            if idx < self._current_idx:
                reward += self._max_predicate_reward[idx]
            else:
                reward += pred_reward
        return reward
