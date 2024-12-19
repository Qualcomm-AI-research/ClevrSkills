# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from .on_top import OnTop


class Inside2d(OnTop):
    def __init__(self, env, top_actor, bottom_actor):
        """
        :param env: The ClevrSkillsEnv.
        :param top_actor: The actor that should be inside the other actor.
        :param bottom_actor: The actor that should contain the other actor.
        """
        super().__init__(
            env, top_actor, bottom_actor, name=f"{top_actor.name} inside {bottom_actor.name}"
        )

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        eval_dict = super().evaluate()
        return eval_dict
