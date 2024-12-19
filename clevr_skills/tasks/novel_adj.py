# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import cv2
import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.inside_2d import Inside2d
from clevr_skills.utils.render import (
    get_camera_by_name,
    get_render_material,
    take_picture_with_camera,
)
from clevr_skills.utils.visualize_prompt import get_actor_description

from .place_on_top import PlaceOnTop
from .task import register_task


@register_task("NovelAdjective")
class NovelAdjective(PlaceOnTop):
    """
    This task creates one or more actors and one or more areas.
    A specific actor must be picked-and-placed on a specific target area.
    The actor and the area are specified by combination of adjectives instead of images,
    where the meaning of adjective must be inferred from example images.
    """

    unsupported_task_args = ["sample_num_actors", "variant"]

    def __init__(
        self, env, sample_num_actors: bool = True, variant: int = 0, record_dir=None, split="train"
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param sample_num_actors: Ignored for this task
        :param variant: Ignored for this task.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(
            env,
            num_actors=3,
            num_areas=1,
            spawn_at_gripper=False,
            record_dir=record_dir,
            split=split,
            variant=variant,
        )

        self._num_actors += 1  # 1-3 examples and then 1 actual actor
        self.adjectives = ["daxer", "blicker", "modier", "kobar"]
        self.area_textures = None
        self._area_no_tex = None
        self.actor_textures = None
        self._ex_actors_a = None
        self._ex_actors_b = None
        self.main_actor = None
        self.main_actor_no_tex = None
        self.main_area = None
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

        area_names = self._episode_rng.choice(
            self.model_factory.task_pool["base_objects"]
            + self.model_factory.task_pool["primitive_objects"],
            replace=False,
            size=self._num_areas,
        )
        self._area_no_tex = [
            self.model_factory.get_object(
                name,
                None,
                self._scene,
                self._renderer,
                render_material=None,
                size=None,
                density=1000.0,
                static=False,
            )
            for name in area_names
        ]
        self._area_actors = [
            self.model_factory.get_object(
                name,
                render_material[idx][0],
                self._scene,
                self._renderer,
                render_material=render_material[idx][1],
                size=None,
                density=1000.0,
                static=False,
            )
            for idx, name in enumerate(area_names)
        ]

        for area_actor in self._area_actors:
            self._place_actor_at_rand_pose_v2(area_actor)

        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)
        render_material = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]
        actor_names = self._episode_rng.choice(
            self.model_factory.task_pool["top_objects"]
            + self.model_factory.task_pool["primitive_objects"],
            replace=False,
            size=self._num_actors,
        )

        size = self._episode_rng.choice([0.3, 0.5, 0.75, 1.5])
        actor_char = [(rm, size) for rm in render_material]

        self._ex_actors_a = [
            self.model_factory.get_object(
                name,
                actor_char[idx][0][0],
                self._scene,
                self._renderer,
                render_material=actor_char[idx][0][1],
                size=actor_char[idx][1],
                density=1000.0,
                static=False,
            )
            for idx, name in enumerate(actor_names)
        ]
        self._ex_actors_b = [
            self.model_factory.get_object(
                name,
                render_material[idx][0],
                self._scene,
                self._renderer,
                render_material=render_material[idx][1],
                size=None,
                density=1000.0,
                static=False,
            )
            for idx, name in enumerate(actor_names)
        ]

        # separate examples and target actors
        self._actors = [self._ex_actors_a[-1], self._ex_actors_b[-1]]
        self._ex_actors_a, self._ex_actors_b = self._ex_actors_a[:-1], self._ex_actors_b[:-1]

        # get no texture target actor
        self.main_actor = self._actors[0]
        self.main_actor_no_tex = self.model_factory.get_object(
            actor_names[-1],
            None,
            self._scene,
            self._renderer,
            render_material=None,
            size=None,
            density=1000.0,
            static=False,
        )
        self.main_area = self._area_actors[0]

        for actor in self._actors:
            self._place_actor_at_rand_pose_v2(actor)

        # move them out of the scene
        for actor in self._ex_actors_a + self._ex_actors_b + [self.main_actor_no_tex]:
            actor.set_pose(sapien.Pose([0, 0, 1000], [0, 0, 0, 1.0]))

        self.predicate = Inside2d(self._env, self.main_actor, self.main_area)

        adj = self._episode_rng.choice(self.adjectives, replace=False, size=1)[0]

        examples = ", ".join(
            [
                f"{get_actor_description(actor_a)} is {adj} than "
                f"{get_actor_description(actor_b)}"
                for actor_a, actor_b in zip(self._ex_actors_a, self._ex_actors_b)
            ]
        )
        self.prompts = [
            f"{examples}. Put the {adj} {get_actor_description(self.main_actor_no_tex)} "
            f"on {get_actor_description(self.main_area)}"
        ]

    def initialize_task(self):
        """
        :return: None.
        """
        super().initialize_task()

        # bring examples and no_tex_objects in the scene to take pictures
        for actor in (
            self._area_no_tex + self._ex_actors_a + self._ex_actors_b + [self.main_actor_no_tex]
        ):
            actor.set_pose(sapien.Pose([0, 0, 0], [0, 0, 0, 1.0]))

        _actors = (
            self._area_no_tex + self._ex_actors_a + self._ex_actors_b + [self.main_actor_no_tex]
        )
        self._save_actor_images(_actors)
        render_cam = get_camera_by_name(self._scene, name="object_camera")

        # scene image without the no-tex objects
        image = take_picture_with_camera(
            self._scene, render_cam, self._actors + self._area_actors, hide_all_non_framed=True
        )
        cv2.imwrite(f"{self.record_dir}/scene.jpg", np.flip(image[:, :, 0:3], axis=2))

        # move them out of the scene again
        for actor in (
            self._area_no_tex + self._ex_actors_a + self._ex_actors_b + [self.main_actor_no_tex]
        ):
            actor.set_pose(sapien.Pose([0, 0, 1000], [0, 0, 0, 1.0]))

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._actors + self._area_actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures[-1:] + self.actor_textures[-1:] + self.area_textures
