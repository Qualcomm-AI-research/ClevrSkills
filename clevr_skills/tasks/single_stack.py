# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.on_top import OnTop
from clevr_skills.predicates.once import Once
from clevr_skills.predicates.sequence import Sequence
from clevr_skills.predicates.topple_structure import ToppleStructurePredicate
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("SingleStack")
class SingleStackTask(Task):
    """
    Stack objects into a stack, in a specific order.
    Optionally topple the stack afterwards.
    """

    def __init__(
        self,
        env,
        num_actors=2,
        sample_num_actors: bool = True,
        reverse=False,
        topple: bool = False,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param num_actors: The number of actors (objects) in the scene.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param reverse: Reverse order of stacking.
        :param topple: Topple the stack after completion.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, variant=variant, record_dir=record_dir, split=split)
        self._actor_size = [0.1, 0.1, 0.05]
        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.1, 0.3],
        ]
        assert num_actors >= 2

        self.topple = topple

        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self.reverse = reverse
        self._actors = []
        self.actor_textures = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]

        self._actors = []
        for idx in range(self._num_actors):
            actor = self.model_factory.get_random_area_object(
                self._scene,
                self._renderer,
                render_materials[idx][1],
                tex_name=render_materials[idx][0],
                static=False,
            )
            self._place_actor_at_rand_pose_v2(actor, grow_actor_bounds=0.02)
            self._actors.append(actor)

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        predicates = []
        if not self.reverse:
            for i in range(self._num_actors - 1):
                predicates.append(OnTop(self._env, self._actors[i + 1], self._actors[i]))
        else:
            for i in range(self._num_actors - 1, 0, -1):
                predicates.append(OnTop(self._env, self._actors[i - 1], self._actors[i]))

        self.predicate = Sequence("Stack the objects", predicates)

        if self.topple:
            push_predicate = ToppleStructurePredicate(self._env, self._actors, name="topple stack")
            self.predicate = Sequence(
                "First build the stack, and then topple it",
                [Once("Build the stack", self.predicate), push_predicate],
            )

        if not self.reverse:
            self.prompts = [
                self._get_prompt(),
                "Stack "
                + ",".join(
                    [
                        f" {get_actor_description(self._actors[idx+1])} on "
                        f"{get_actor_description(self._actors[idx])}"
                        for idx in range(self._num_actors - 1)
                    ]
                ),
                "Stack "
                + ",".join(
                    [
                        f" object with {get_texture_description(self._actors[idx+1])} "
                        f"texture on object with {get_texture_description(self._actors[idx])}  "
                        f"texture"
                        for idx in range(self._num_actors - 1)
                    ]
                ),
                f"Stack objects as in {{ks:keystep_{self._num_actors-1}}}",
                "Stack objects in this order "
                + " ".join([f"{{ks:keystep_{idx}}}" for idx in range(self._num_actors)]),
            ]
        else:
            self.prompts = [
                "Stack "
                + ", ".join(
                    [
                        f" {get_actor_description(self._actors[idx])}"
                        for idx in range(self._num_actors)
                    ]
                )
                + " in the reversed order",
                "Stack "
                + ", ".join(
                    [
                        f" object with {get_texture_description(self._actors[idx])} texture"
                        for idx in range(self._num_actors)
                    ]
                )
                + " in the reversed order",
            ]

        if self.topple:
            for idx, prompt in enumerate(self.prompts):
                self.prompts[idx] = prompt + " and then topple the stack"

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task
        """
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task
        """
        return self.actor_textures

    def _get_prompt(self) -> str:
        """
        :return: Prompt for this task.
        """
        desc = "Stack "
        N = self._num_actors - 1
        for i in range(self._num_actors - 1):
            if i > 0:
                desc = desc + ", "
            final = i == self._num_actors - 2
            if (not self.reverse and final and i > 0) or (self.reverse and N - i == 1):
                desc = desc + "and "
            if not self.reverse:
                bottom_actor = self._actors[i]
                top_actor = self._actors[i + 1]
            else:
                bottom_actor = self._actors[N - i]
                top_actor = self._actors[N - i - 1]
            desc = (
                desc
                + f"{get_actor_description(top_actor)} on {get_actor_description(bottom_actor)}"
            )
        if self.topple:
            desc = desc + " and then topple the stack"

        return desc
