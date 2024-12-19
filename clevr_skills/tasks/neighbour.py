# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.inside_2d import Inside2d
from clevr_skills.predicates.set import Set
from clevr_skills.utils.actor_placement import place_actor_and_neighbour_randomly_v2
from clevr_skills.utils.models import SIZE, get_actor_size
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .sort_2d import Sort2dTask
from .task import register_task


@register_task("Neighbour")
class NeighbourTask(Sort2dTask):
    """
    This task creates two or more actors and one or more areas.
    A specific actor, along with its neighbour - specified by a direction,
    must be picked-and-placed on a specific target area.
    """

    def __init__(
        self,
        env,
        num_areas=2,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_areas: Number of areas for placing actors.
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

        self._num_actors = 2
        self._num_areas = self._sample_int(min=1, max=num_areas, sample=sample_num_actors)
        self._area_actors = []
        self._actors = []
        self.area_textures = None
        self._main_area = None
        self.actor_textures = None
        self._main_actor = None
        self._neighbour = None
        self._actors_left = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicatess()
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

        area_poses = [sapien.Pose([p[0], p[1], 0]) for p in self._area_positions]
        for area_actor, apose in zip(self._area_actors, area_poses):
            ac_size = get_actor_size(area_actor)
            # adding the area height/2 as offset to avoid intersecting ground
            p = apose.p
            p[2] += ac_size[2] / 2
            apose.set_p(p)
            area_actor.set_pose(apose)

        self._area_actors, self.area_textures = self._get_variant(
            self._area_actors, self.area_textures
        )

        self._main_area = self._episode_rng.choice(
            self._area_actors
        )  # which area to place the objects in

        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)

        render_material = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]
        self._actors = [
            self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_material[idx][1],
                tex_name=render_material[idx][0],
            )
            for idx in range(self._num_actors)
        ]

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        self._main_actor, self._neighbour = self._actors[:2]
        self._actors_left = self._actors[2:]

        direction = self._episode_rng.choice(["north", "south", "east", "west"])
        place_actor_and_neighbour_randomly_v2(
            self._main_actor, self._neighbour, direction, self._env, self._workspace
        )

        for actor in self._actors_left:
            self._place_actor_at_rand_pose_v2(actor)

        predicates = []
        for actor in [self._main_actor, self._neighbour]:
            predicates.append(Inside2d(self._env, actor, self._main_area))

        self.predicate = Set("neighbour", predicates)

        self.prompts = [
            f"First put {get_actor_description(self._main_actor)} in "
            f"{get_actor_description(self._main_area)} and then put the object "
            f"that was at its {direction} in the same "
            f"{get_actor_description(self._main_area)}",
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
