# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections import defaultdict
from typing import List

import sapien.core as sapien

from clevr_skills.predicates.set import Set
from clevr_skills.predicates.stack_predicate import StackPredicate
from clevr_skills.utils.render import get_render_material

from .sort_2d import Sort2dTask
from .task import register_task


@register_task("SortStack")
class SortStack(Sort2dTask):
    """
    Sort and stack objects by texture.
    """

    def __init__(
        self,
        env,
        num_actors=4,
        num_areas=2,
        variant: int = 0,
        sample_num_actors: bool = True,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param num_actors: The number of actors (objects) in the scene.
        :param num_areas: Number of areas where objects can be placed.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        assert num_actors >= num_areas
        super().__init__(
            env, num_actors, num_areas, variant=variant, record_dir=record_dir, split=split
        )
        self._num_actors = self._sample_int(
            min=self._num_areas + 1, max=num_actors, sample=sample_num_actors
        )
        self.area_textures = None
        self.actor_textures = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.area_textures = self.model_factory.get_random_base_textures(self._num_areas)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.area_textures
        ]

        actor_match_area = [
            self._episode_rng.choice(range(self._num_areas)) for _ in range(self._num_actors)
        ]
        self._actors = []
        self.actor_textures = []
        for idx in range(self._num_actors):
            area_idx = actor_match_area[idx]
            self.actor_textures.append(self.area_textures[area_idx])
            actor = self.model_factory.get_random_area_object(
                self._scene,
                self._renderer,
                render_materials[area_idx][1],
                tex_name=render_materials[area_idx][0],
                static=False,
            )
            self._place_actor_at_rand_pose_v2(actor, grow_actor_bounds=0.02)
            self._actors.append(actor)

        # Putting the predicates in a different order will cause different order of solver actions.
        self._actors, self.actor_textures, actor_match_area = self._get_variant(
            self._actors, self.actor_textures, actor_match_area
        )

        actor_sets = defaultdict(list)
        for idx in range(self._num_actors):
            actor_sets[actor_match_area[idx]].append(self._actors[idx])

        predicates = []
        for _, actors in actor_sets.items():
            predicates.append(StackPredicate(self._env, actors))

        self.predicate = Set("sort_stack", predicates)

        self.prompts = [
            "Stack identically textured objects",
            "Place identically textured objects on top of each other",
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._actors + self._area_actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures + self.area_textures
