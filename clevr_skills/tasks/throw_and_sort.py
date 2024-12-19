# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.hit_predicate import HitPredicate
from clevr_skills.predicates.set import Set
from clevr_skills.utils.models import SIZE
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .task import Task, register_task


@register_task("ThrowAndSort")
class ThrowAndSortTask(Task):
    """
    The task creates one or more actors and one or more areas.
    The goal is to sort objects by texture by throwing the actors in correspondingly textured areas.
    """

    def __init__(
        self,
        env,
        num_actors: int = 4,
        num_areas: int = 2,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
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
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._actor_size = np.array([0.1, 0.1, 0.05])
        self._area_size = np.array([0.35, 0.35, 0.01])
        self._area_positions = [
            [-0.2, -0.2],
            [-0.2, 0.2],
            [0.2, -0.2],
            [0.2, 0.2],
        ]
        assert num_areas <= len(self._area_positions)

        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )

        self._num_actors = self._sample_int(min=1, max=num_actors, sample=sample_num_actors)
        self._num_areas = self._sample_int(min=1, max=num_areas, sample=sample_num_actors)
        self._area_actors = []
        self._actors = []
        self.area_textures = None
        self.predicate = None
        self.prompts = None
        self.actor_textures = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.area_textures = self.model_factory.get_random_base_textures(self._num_areas)

        render_material = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.area_textures
        ]
        self._area_actors = [
            self.model_factory.get_random_area_object(
                self._scene,
                self._renderer,
                render_material[idx][1],
                tex_name=render_material[idx][0],
                size=SIZE(*self._area_size),
            )
            for idx in range(self._num_areas)
        ]

        OUT_OF_REACH = 0.75  # at least 75 cm away from robot
        for area_actor in self._area_actors:
            self._place_actor_out_of_reach(area_actor, robo_dist=OUT_OF_REACH)

        actor_match_area = [
            self._episode_rng.choice(range(self._num_areas)) for _ in range(self._num_actors)
        ]
        self._actors = []
        self.actor_textures = []
        for idx in range(self._num_actors):
            area_idx = actor_match_area[idx]
            self.actor_textures.append(self.area_textures[area_idx])
            actor = self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_material[area_idx][1],
                tex_name=render_material[area_idx][0],
            )
            self._place_actor_at_rand_pose_v2(actor)
            self._actors.append(actor)

        # Putting the predicates in a different order will cause different order of solver actions.
        self._actors, self.actor_textures, actor_match_area = self._get_variant(
            self._actors, self.actor_textures, actor_match_area
        )

        predicates = []
        for idx in range(self._num_actors):
            predicates.append(
                HitPredicate(
                    self._env,
                    self._actors[idx],
                    self._area_actors[actor_match_area[idx]],
                    False,
                    False,
                )
            )

        self.predicate = Set("throw_and_sort", predicates)

        self.prompts = [
            "Place the objects in the identically textured areas by throwing",
            "Throw "
            + " throw ".join(
                [
                    f"{actor.name} in {texture[0]} area"
                    for (actor, texture) in zip(self._actors, self.actor_textures)
                ]
            ),
            "Throw "
            + " throw ".join(
                [
                    f"{get_actor_description(self._actors[idx])} in "
                    f"{get_actor_description(self._area_actors[actor_match_area[idx]])}"
                    for idx in range(self._num_actors)
                ]
            ),
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
