# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.inside_2d import Inside2d
from clevr_skills.predicates.order import Order
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_placement import place_actor_on_area
from clevr_skills.utils.models import SIZE
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .task import Task, register_task


@register_task("FollowOrder")
class FollowOrderTask(Task):
    """
    This task creates an actor and 2 or more areas.
    The actor must be picked-and-placed on areas following the order as specified by "sub-goal" images.
    If restore is True, the actor must be restored to area it was placed at initialization.
    """

    def __init__(
        self,
        env,
        num_areas=2,
        num_predicates=2,
        restore=False,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param num_actors: The number of actors (objects) in the scene.
        :param num_areas: Number of areas where objects can be placed.
        :param num_predicates: Number of predicates (i.e., number of actors to be placed).
        :param restore: Restore actors to original pose after placing them?
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
        self._area_size = np.array([0.18, 0.18, 0.01])
        self._area_positions = [
            [-0.2, -0.2],
            [-0.2, 0.0],
            [-0.2, 0.2],
            [0.0, -0.2],
            [0.0, 0.0],
            [0.0, 0.2],
            [0.2, -0.2],
            [0.2, 0.0],
            [0.2, 0.2],
        ]
        assert 2 <= num_areas <= len(self._area_positions)
        assert num_predicates <= num_areas

        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )

        self._num_actors = 1
        self._num_areas = self._sample_int(min=2, max=num_areas, sample=sample_num_actors)
        self._num_predicates = min(
            self._sample_int(min=1, max=num_predicates, sample=sample_num_actors),
            self._num_areas - 1,
        )  # how many areas we want to interact with. Must be <= num_areas

        assert self._num_predicates <= self._num_areas - 1
        self._area_actors = []
        self._actors = []
        self.restore = restore
        self.area_textures = None
        self.actor_textures = None
        self.main_actor = None
        self.predicate = None
        self.prompts = None

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

        for area_actor in self._area_actors:
            self._place_actor_at_rand_pose_v2(area_actor, grow_actor_bounds=0.01)

        self._episode_rng.shuffle(self._area_actors)

        self._area_actors, self.area_textures = self._get_variant(
            self._area_actors, self.area_textures
        )

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

        self.main_actor = self._actors[0]

        place_actor_on_area(self.main_actor, self._area_actors[0])

        predicates = []
        for actor in self._actors:
            for idx in range(1, 1 + self._num_predicates):
                area_actor = self._area_actors[idx]
                p = Inside2d(self._env, actor, area_actor)
                p.action_state_extra = {
                    "target_pose_function": "on_top",
                    "target_actor": at_get_actor(area_actor),
                    "target_image": f"{{ks:keystep_{idx}}}",
                }
                predicates.append(p)

        if self.restore:
            p = Inside2d(self._env, self._actors[0], self._area_actors[0])
            p.action_state_extra = {
                "target_pose_function": "on_top",
                "target_actor": at_get_actor(self._area_actors[0]),
            }
            predicates.append(p)

        self.predicate = Order("follow", predicates)

        if not self.restore:
            self.prompts = [
                f"Follow the motion for {get_actor_description(self.main_actor)}: "
                + " ".join([f"{{ks:keystep_{idx}}}" for idx in range(1, self._num_predicates + 1)]),
                f"Put {get_actor_description(self.main_actor)} on "
                + " then on ".join(
                    [
                        f"{get_actor_description(x)}"
                        for x in self._area_actors[1 : 1 + self._num_predicates]
                    ]
                ),
            ]
        else:
            self.prompts = [
                f"Follow the motion for {get_actor_description(self.main_actor)}: "
                + " ".join([f"{{ks:keystep_{idx}}}" for idx in range(1, self._num_predicates + 1)])
                + " and then restore",
                f"Put {get_actor_description(self.main_actor)} on "
                + " then on ".join(
                    [
                        f"{get_actor_description(x)}"
                        for x in self._area_actors[1 : 1 + self._num_predicates]
                    ]
                )
                + " and then restore",
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
