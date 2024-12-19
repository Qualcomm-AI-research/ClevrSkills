# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.order import Order
from clevr_skills.predicates.rotate_predicate import RotatePredicate
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_texture_description

from .rotate import Rotate
from .task import register_task


@register_task("RotateSymmetry")
class RotateSymmetry(Rotate):
    """
    The task creates two or more actors.
    The goal is to rotate actors with a specified texture by a
    specified degrees in a specific direction.
    """

    def __init__(
        self,
        env,
        num_actors=2,
        variant: int = 0,
        sample_num_actors: bool = True,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(
            env, num_actors=num_actors, variant=variant, record_dir=record_dir, split=split
        )
        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self.actor_a = None
        self.actor_b = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.clockwise = self._episode_rng.choice([False, True])
        self.rot_degrees = self._episode_rng.choice([30, 60, 90, 120, 150])

        self._actors = []
        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors - 1)
        self.actor_textures = self.actor_textures + [self.actor_textures[-1]]

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

        self.actor_a, self.actor_b = self._actors[-2:]

        predicates = [
            RotatePredicate(self._env, self.actor_a, self.rot_degrees, self.clockwise),
            RotatePredicate(self._env, self.actor_b, self.rot_degrees, self.clockwise),
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

        self.prompts = [
            f"Rotate objects with {get_texture_description(self.actor_a)} texture "
            f"{abs(self.rot_degrees)} degrees {['anti-clockwise', 'clockwise'][self.clockwise]}",
            f"Rotate identically textured objects {abs(self.rot_degrees)} degrees "
            f"{['anti-clockwise', 'clockwise'][self.clockwise]}",
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
