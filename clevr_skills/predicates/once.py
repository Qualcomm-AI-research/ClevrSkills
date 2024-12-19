# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


from .predicate import Predicate


class Once(Predicate):
    """
    This predicates wraps another predicate. If the inside predicate reports success once,
    then Once will continue to report success forever.

    This can be used e.g. to build a structure and then topple it
    Sequence(Once(build structure), topple structure).

    Dense reward and eval will remain at the state observed when success was first reported.
    """

    def __init__(self, name: str, predicate: Predicate):
        """
        :param name: Descriptive name of the predicate.
        :param predicate: The predicate that must go to success, once.
        """
        super().__init__(predicate.name)
        self._predicate = predicate
        self._dense_reward_at_first_success = 0.0
        self._eval_at_first_success = None
        self._success = None

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        if self._eval_at_first_success is None:
            eval = self._predicate.evaluate()
            self._success = eval["success"]
            if self._success:
                self._dense_reward_at_first_success = self.compute_dense_reward()
                self._eval_at_first_success = eval
            return eval
        return self._eval_at_first_success

    def compute_dense_reward(self):
        """
        :return: dense reward.
        """
        return (
            self._predicate.compute_dense_reward()
            if (self._eval_at_first_success is None)
            else self._dense_reward_at_first_success
        )
