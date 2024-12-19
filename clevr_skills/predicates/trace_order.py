# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

from clevr_skills.predicates.trace_predicate import TracePredicate

from .order import Order


class TraceOrder(Order):
    """
    Predicate for tracing goals.
    """

    def __init__(self, name: str, predicates: List[TracePredicate]):
        """
        :param name: Descriptive name of the predicate.
        :param predicates: List of TracePredicate.
        """
        for pred in predicates:
            assert isinstance(
                pred, TracePredicate
            ), f"{type(pred)} not allowed in TraceOrder. Only use with TracePredicate"

        super().__init__(name, predicates)

        self._predicates[self._current_idx].start_solving()

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        if self._current_idx == len(self._predicates):
            return {"success": True}

        result = {"success": False}
        while (
            self._current_idx < len(self._predicates)
            and self._predicates[self._current_idx].evaluate()["success"]
        ):
            self._current_idx += 1
            if self._current_idx < len(self._predicates):
                self._predicates[self._current_idx].start_solving()

        for predicate in self._predicates:
            predicate_eval_dict = predicate.evaluate()
            name = predicate.name
            for key, value in predicate_eval_dict.items():
                result[f"{name}_{key}"] = value

        return result
