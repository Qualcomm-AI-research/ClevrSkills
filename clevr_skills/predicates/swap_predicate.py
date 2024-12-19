# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

from .order import Order
from .predicate import Predicate


class SwapPredicate(Order):
    """
    This class implements a swap predicate.
    """

    def __init__(self, name: str, predicates: List[Predicate]):
        """
        :param name: Descriptive name of the predicate.
        :param predicates: List of 3 predicate that define the swap (move actor 1 to free space, move actor 2 to actor 1, move actor 1 to actor 2).
        """
        super().__init__(name, predicates)
        assert len(predicates) == 3

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        if self._predicates[1].evaluate()["success"] and self._predicates[2].evaluate()["success"]:
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
