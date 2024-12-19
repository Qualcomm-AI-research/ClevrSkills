# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.inside_2d import Inside2d
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("PlaceOnTop")
class PlaceOnTop(Task):
    """
    This task creates one or more actors and one or more areas.
    A specific actor must be picked-and-placed on a specific target area.
    If spawn_at_gripper is True, the specific actor will spawn right underneath
    the robot gripper such that only the place part of the action needs to be performed.
    """

    def __init__(
        self,
        env,
        num_actors=3,
        num_areas=3,
        spawn_at_gripper: bool = False,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param spawn_at_gripper: When True, the main actor will be created directly underneath the
        gripper with a small upwards velocity. If the agent turns on the gripper, the actor
        will stick to it.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._variant = variant
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
        self._spawn_at_gripper = spawn_at_gripper
        self.set_initial_grasping_action(spawn_at_gripper)
        self.area_textures = None
        self.actor_textures = None
        self.main_actor = None
        self.main_actor_texture = None
        self.main_area = None
        self.main_area_texture = None
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
            self.model_factory.get_random_base_object(
                self._scene,
                self._renderer,
                render_material[idx][1],
                tex_name=render_material[idx][0],
            )
            for idx in range(self._num_areas)
        ]

        for area_actor in self._area_actors:
            self._place_actor_at_rand_pose_v2(area_actor)

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

        for actor in self._actors:
            self._place_actor_at_rand_pose_v2(actor)

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        ac_idx = self._episode_rng.choice(len(self._actors), size=1)[0]
        self.main_actor: sapien.Actor = self._actors[ac_idx]
        self.main_actor_texture = self.actor_textures[ac_idx]

        area_idx = self._episode_rng.choice(len(self._area_actors), size=1)[0]
        self.main_area = self._area_actors[area_idx]
        self.main_area_texture = self.area_textures[area_idx]

        if self._spawn_at_gripper:
            ee_pose = self._env.agent.ee_link.pose
            bounds = self._env._actor_distance.get_bounds(self.main_actor, sapien.Pose())
            self.main_actor.set_pose(
                sapien.Pose(ee_pose.p - np.array([0, 0.0, bounds[1, 2] + 0.0005]))
            )
            self.main_actor.set_velocity([0.0, 0.0, 0.1])  # slight upward velocity

        self.predicate = Inside2d(self._env, self.main_actor, self.main_area)

        verb = "Put"

        self.prompts = [
            f"{verb} {get_actor_description(self.main_actor)} on "
            f"{get_actor_description(self.main_area)}",
            f"{verb} {self.main_actor.name} on {self.main_area.name}",
            f"{verb} object with {get_texture_description(self.main_actor)} texture on "
            f"object with {get_texture_description(self.main_area)} texture",
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
