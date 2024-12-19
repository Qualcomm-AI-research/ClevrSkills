# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.order import Order
from clevr_skills.predicates.rotate_predicate import RotatePredicate
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("Rotate")
class Rotate(Task):
    """
    The task creates one or more actors. The goal is to rotate a specified actor by a
    specified degrees in a specific direction.
    If restore is True, the actor must be restored to the pose at initialization.
    """

    def __init__(
        self,
        env,
        num_actors=1,
        sample_num_actors: bool = True,
        restore=False,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param restore: Should actors be restore to their original pose after being moved to their target pose?
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._actor_size = [0.1, 0.1, 0.05]
        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.1, 0.3],
        ]
        self._num_actors = self._sample_int(min=1, max=num_actors, sample=sample_num_actors)
        self.restore = restore
        self._actors = []
        self.clockwise = None
        self.rot_degrees = None
        self.actor_textures = None
        self.main_ac_idx = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.clockwise = self._episode_rng.choice([0, 1])
        self.rot_degrees = self._episode_rng.choice([30, 60, 90, 120, 150])

        self._actors = []
        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]
        self._actors = [
            self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_materials[idx][1],
                tex_name=render_materials[idx][0],
            )
            for idx in range(self._num_actors)
        ]

        for actor in self._actors:
            self._place_actor_at_rand_pose_v2(actor)

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        self.main_ac_idx = self._episode_rng.choice(self._num_actors)
        predicates = [
            RotatePredicate(
                self._env, self._actors[self.main_ac_idx], self.rot_degrees, self.clockwise
            )
        ]

        if self.restore:
            predicates.append(
                RotatePredicate(
                    self._env,
                    self._actors[self.main_ac_idx],
                    self.rot_degrees,
                    self.clockwise,
                    restore=True,
                )
            )

        self.predicate = Order("rotate", predicates)

        if not self.restore:
            self.prompts = [
                f"Rotate {get_actor_description(self._actors[self.main_ac_idx])} {abs(self.rot_degrees)} "
                f"degrees {['anti-clockwise', 'clockwise'][self.clockwise]}",
                f"Rotate object with {get_texture_description(self._actors[self.main_ac_idx])} {abs(self.rot_degrees)} "
                f"degrees {['anti-clockwise', 'clockwise'][self.clockwise]}",
            ]
        else:
            self.prompts = [
                f"Rotate {get_actor_description(self._actors[self.main_ac_idx])} {abs(self.rot_degrees)} "
                f"degrees {['anti-clockwise', 'clockwise'][self.clockwise]} and "
                f"then restore",
                f"Rotate object with {get_texture_description(self._actors[self.main_ac_idx])} {abs(self.rot_degrees)} "
                f"degrees {['anti-clockwise', 'clockwise'][self.clockwise]} and "
                f"then restore",
            ]

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
