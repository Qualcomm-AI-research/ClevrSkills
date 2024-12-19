# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.order import Order
from clevr_skills.predicates.predicate import Predicate
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.logger import log


class OrderedSolver(AbstractSolver):
    def __init__(self, env: ClevrSkillsEnv):
        """
        :param env: The ClevrSkillsEnv
        """
        super().__init__(env)
        self._current_solver = None

    def step(self, obs):
        """
        This solver checks if the current predicate that it is working on is solved.
        If so, it asks the current solver to return an action.
        Otherwise it initializes a new solver for the next predicate.
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv.
        """
        # check if current predicate is complete
        if not self._current_predicate or self._current_predicate.evaluate()["success"]:
            self._current_predicate = self.select_next_predicate_to_solve(
                self._env.task.get_predicate()
            )
            self._current_solver = (
                self.init_solver(self._current_predicate) if self._current_predicate else None
            )

        if self._current_solver:
            return self._current_solver.step(obs)
        log("Warning: Solver has completed all predicates . . .")
        return self.return_hold_action(self.gripper_off)

    def select_next_predicate_to_solve(self, predicate: Predicate):
        """
        :param predicate: The order predicate that is in the process of solving.
        :return: The next predicate to solve.
        """
        if isinstance(predicate, Order):
            return predicate.get_current_predicate()
        raise RuntimeError("predicate is not an Order!")
