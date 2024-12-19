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


@register_task("NovelNoun")
class NovelNoun(PlaceOnTop):
    """
    This task creates one or more actors and one or more areas.
    A specific actor must be picked-and-placed on a specific target area.
    The actor and the area are specified by nouns instead of images.
    """

    unsupported_task_args = ["sample_num_actors", "variant"]

    def __init__(
        self,
        env,
        num_actors=3,
        num_areas=4,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param num_actors: The number of actors (objects) in the scene.
        :param num_areas: Number of areas where objects can be placed.
        :param sample_num_actors: Ignored for this task.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(
            env,
            num_actors=num_actors,
            num_areas=num_areas,
            spawn_at_gripper=False,
            record_dir=record_dir,
            split=split,
            variant=variant,
        )

        self.nouns = ["dax", "blicket", "wug", "zup"]
        self.area_textures = None
        self._area_no_tex = None
        self.actor_textures = None
        self._actors_no_tex = None
        self.main_actor = None
        self.main_actor_no_tex = None
        self.main_actor_texture = None
        self.main_area = None
        self.main_area_no_tex = None
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
            + self.model_factory.task_pool["primitive_objects"]
            + self.model_factory.task_pool["random_objects"],
            replace=False,
            size=self._num_actors,
        )
        self._actors_no_tex = [
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
            for idx, name in enumerate(actor_names)
        ]
        self._actors = [
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

        for actor in self._actors:
            self._place_actor_at_rand_pose_v2(actor)

        # move them out of the scene
        for actor in self._area_no_tex + self._actors_no_tex:
            actor.set_pose(sapien.Pose([0, 0, 1000], [0, 0, 0, 1.0]))

        ac_idx = self._episode_rng.choice(len(self._actors), size=1)[0]
        self.main_actor: sapien.Actor = self._actors[ac_idx]
        self.main_actor_no_tex: sapien.Actor = self._actors_no_tex[ac_idx]
        self.main_actor_texture = self.actor_textures[ac_idx]

        area_idx = self._episode_rng.choice(len(self._area_actors), size=1)[0]
        self.main_area = self._area_actors[area_idx]
        self.main_area_no_tex = self._area_no_tex[area_idx]
        self.main_area_texture = self.area_textures[area_idx]

        self.predicate = Inside2d(self._env, self.main_actor, self.main_area)

        nouns = self._episode_rng.choice(self.nouns, replace=False, size=2)
        actor_noun, area_noun = nouns

        self.prompts = [
            f"{get_actor_description(self.main_actor_no_tex)} is "
            f"{actor_noun} and {get_actor_description(self.main_area_no_tex)}"
            f" is {area_noun}. Put {actor_noun} on {area_noun}"
        ]

    def initialize_task(self):
        super().initialize_task()

        if self.record_dir is None:
            return

        # bring no tex objects in the scene to take pictures
        for actor in self._area_no_tex + self._actors_no_tex:
            actor.set_pose(sapien.Pose([0, 0, 0], [0, 0, 0, 1.0]))

        _no_tex_actors = self._area_no_tex + self._actors_no_tex
        self._save_actor_images(_no_tex_actors)

        # scene image without the no-tex objects
        render_cam = get_camera_by_name(self._scene, name="render_camera")
        image = take_picture_with_camera(
            self._scene,
            render_cam,
            self._obj_cam_position,
            self._actors + self._area_actors,
            hide_all_non_framed=True,
        )
        cv2.imwrite(f"{self.record_dir}/scene.jpg", np.flip(image[:, :, 0:3], axis=2))

        # move them out of the scene again
        for actor in self._area_no_tex + self._actors_no_tex:
            actor.set_pose(sapien.Pose([0, 0, 1000], [0, 0, 0, 1.0]))

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task
        """
        return self._actors + self._area_actors + self._actors_no_tex + self._area_no_tex

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task
        """
        return self.actor_textures + self.area_textures + [(None, None), (None, None)]
