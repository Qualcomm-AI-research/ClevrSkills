# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import ABC, abstractmethod

import sapien.core as sapien
from mani_skill2.envs.sapien_env import BaseEnv


class Predicate(ABC):
    """
    This class defines a "predicate" that must be true for the task to succeed.
    Examples of predicates can be "OnTop(X, Y)" or "Visible(X)".
    Most Predicates will require access to the environment for evaluation.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def evaluate(self):
        """
        To be implemented by sub-classes.
        :return: A dictionary with information about success.
        The most important entry is dict["success"] which is used to indicate that the
        Predicate is met.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_dense_reward(self):
        """
        To be implemented by sub-classes.
        :return: dense reward
        """
        raise NotImplementedError


class EnvPredicate(Predicate):
    """
    Many predicates will require access to the environment for evaluation.
    """

    def __init__(self, env: BaseEnv, name: str):
        """
        :param env: The ClevrSkillsEnv.
        :param name: Descriptive name of the predicate.
        """
        super().__init__(name)
        self._env = env

    def compute_dense_reward(self):
        """
        To be implemented by sub-classes.
        :return: dense reward.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        To be implemented by sub-classes.
        :return: A dictionary with information about success.
        The most important entry is dict["success"] which is used to indicate that the
        Predicate is met.
        """
        raise NotImplementedError
