# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

from .predicate import Predicate


class Sequence(Predicate):
    """
    This class implements a sequence of predicates that must be satisfied in-order.
    Predicates must stay satisfied for the sequence to progress.
    """

    def __init__(self, name: str, predicates: List[Predicate]):
        """
        :param name:  Descriptive name of the predicate.
        :param predicates: List of predicates to solve in order.
        """
        super().__init__(name)
        self._predicates = predicates

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        For sequence, the evaluation info of sub-predicates is included.
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
        for predicate in self._predicates:
            predicate_eval_dict = predicate.evaluate()
            reward += predicate.compute_dense_reward()
            if not predicate_eval_dict["success"]:
                break
        return reward
