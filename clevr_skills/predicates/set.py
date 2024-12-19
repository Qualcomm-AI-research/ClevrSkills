# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

from .predicate import Predicate


class Set(Predicate):
    """
    This class implements a set of predicates that must be satisfied in arbitrary order.
    """

    def __init__(self, name: str, predicates: List[Predicate]):
        """
        :param name: Descriptive name of the predicate.
        :param predicates: The predicates that should be satisfied in arbitrary order.
        """
        super().__init__(name)
        self._predicates = predicates

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        For set, the evaluation info of sub-predicates is included.
        """
        success = True
        result = {}
        for predicate in self._predicates:
            predicate_eval_dict = predicate.evaluate()
            name = predicate.name
            success = success and predicate_eval_dict["success"]
            for key, value in predicate_eval_dict.items():
                result[f"{name}_{key}"] = value
        result["success"] = success
        return result

    def compute_dense_reward(self):
        """
        :return: dense reward.
        """
        reward = 0.0
        max_non_success_reward = 0.0
        for predicate in self._predicates:
            predicate_eval_dict = predicate.evaluate()
            r = predicate.compute_dense_reward()
            if predicate_eval_dict["success"]:
                reward += r  # get full reward for complete predicates
            else:
                # otherwise figure out the max reward for any non-completed
                max_non_success_reward = max(max_non_success_reward, r)
        return reward + max_non_success_reward
