# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.set import Set
from clevr_skills.predicates.touch import TouchPredicate
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .task import Task, register_task


@register_task("Touch")
class TouchTask(Task):
    """
    Touch actors and optionally push them or topple them over.
    """

    def __init__(
        self,
        env,
        num_actors: int = 2,
        push: bool = False,
        topple: bool = False,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param push: Whether to push (move) the actor.
        :param topple: Whether to topple the actor. Toppling takes priority over pushing.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._actor_size = [0.05, 0.05, 0.2] if topple else [0.1, 0.1, 0.05]
        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.0, 0.3],
        ]
        assert num_actors >= 1

        self._num_actors = self._sample_int(min=1, max=num_actors, sample=sample_num_actors)
        self._actors = []
        self.push = push
        self.topple = topple
        self.actor_textures = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """
        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]

        self._actors = []
        for idx in range(self._num_actors):
            actor = self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_materials[idx][1],
                tex_name=render_materials[idx][0],
                size=self._actor_size,
            )
            self._place_actor_at_rand_pose_v2(actor, grow_actor_bounds=0.02)
            self._actors.append(actor)

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        predicates = []
        for i in range(self._num_actors):
            predicates.append(TouchPredicate(self._env, self._actors[i], self.push, self.topple))

        self.predicate = Set("Touch", predicates)

        self.prompts = [
            self._get_prompt(),
        ]

    def _get_prompt(self) -> str:
        """
        :return: Prompt for this task.
        """
        additional_action = ""
        if self.push:
            additional_action = " and push"
        if self.topple:
            additional_action = " and topple"
        desc = f"Touch{additional_action}" + ",".join(
            [f" {get_actor_description(self._actors[idx])}" for idx in range(self._num_actors)]
        )
        return desc

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures
