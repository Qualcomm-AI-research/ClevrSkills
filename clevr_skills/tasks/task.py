# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import sapien.core as sapien
from mani_skill2.utils.trimesh_utils import get_actor_mesh

from clevr_skills.predicates.predicate import Predicate
from clevr_skills.utils.actor_placement import (  # place_actor_randomly,
    place_actor_out_of_reach_v2,
    place_actor_randomly_v2,
)
from clevr_skills.utils.logger import log
from clevr_skills.utils.models import ModelAndTextureFactory
from clevr_skills.utils.permute import get_permutation_first_indices, get_permutation_indices
from clevr_skills.utils.render import get_camera_by_name, take_picture, take_picture_with_camera
from clevr_skills.utils.visualize_prompt import load_texture


class Task(ABC):
    """
    Base class for Tasks.
    """

    # Sub-classes can list unsupported task_args here so the user will get a warning.
    # See Task.check_task_args()
    unsupported_task_args = []

    def __init__(
        self, env, record_dir: Optional[str] = None, split: str = "train", variant: int = 0
    ):
        """
        A new Task instance is created for every episode.
        :param env: The ClevrSkillsEnv
        :param record_dir: Path where episode is to be recorded.
        :param split: What object split to use (e.g., "train", "test").
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" scene looks identical but the roles of actors are changed.
        """
        self._env = env
        self._episode_rng = env._episode_rng
        self._scene: sapien.Scene = env._scene
        self._renderer = env._renderer

        self._workspace = None
        self.model_factory = None

        self.record_dir = record_dir
        self.split = split

        self._obj_cam_position = [
            1.0,
            3.0,
            5.0,
        ]  # camera can be anywhere, as long as it not "inside" the framed_actors point cloud

        self.initial_grasp_action = 0.0  # grasping action during reset

        self._variant = variant

        classname: str = self.__class__.__name__
        for remove_suffix in ["Task", "Env"]:
            if classname.endswith(remove_suffix):
                classname = classname[: -len(remove_suffix)]
        self.model_factory = ModelAndTextureFactory(
            task=classname, episode_rng=self._episode_rng, split=split
        )

        self.prompts = []
        self.prompt_assets = {"objects": {}, "textures": {}}

    def _get_variant(
        self,
        list1: List,
        list2: Optional[List] = None,
        list3: Optional[List] = None,
        first_actor_only: bool = False,
    ) -> Union[List, Tuple]:
        """
        Unility function to generate a variant of a task.
        self._variant is used and updated in the process.
        :param list1: The first list of items (usually actors) that must be permuted.
        :param list2: The optional second list of items (usually actors) that must be permuted.
        :param list3: The optional third list of items (usually actors) that must be permuted.
        :param first_actor_only: If true, only the first item (actor) in the list is swapped.
        :return: a permutation of list1, or a tuple with variants of (list1, list2, ...)
        """
        if first_actor_only:
            indices, self._variant = get_permutation_first_indices(len(list1), self._variant)
        else:
            indices, self._variant = get_permutation_indices(len(list1), self._variant)
        list1 = [list1[idx] for idx in indices]
        if list2:
            assert len(list1) == len(list2)
            list2 = [list2[idx] for idx in indices]
            if list3:
                assert len(list1) == len(list3)
                list3 = [list3[idx] for idx in indices]
                return list1, list2, list3
            return list1, list2
        return list1

    def set_initial_grasping_action(self, spawn_at_gripper: bool):
        """
        Sets self.initial_grasp_action to "gripper on" if the object is spawned at the gripper.
        :param spawn_at_gripper: Whether the to-be-grasped actor spawned at the gripper.
        :return: None
        """
        self.initial_grasp_action = 1.0 if spawn_at_gripper else 0.0

    def _place_actor_at_rand_pose_v2(
        self, actor, offset=0.0, allow_top=False, grow_actor_bounds: float = 0.0
    ):
        """
        :param actor: The actor to be placed.
        :param offset: Vertical offset.
        :param allow_top: Allow actor to be placed on top of another actor?
        :param grow_actor_bounds: How much to grow the bounds of the actor while determining empty space.
        :return: None. Actor pose is set.
        """
        place_actor_randomly_v2(
            actor,
            self._env,
            self._workspace,
            offset=offset,
            allow_top=allow_top,
            grow_actor_bounds=grow_actor_bounds,
        )

    def _place_actor_out_of_reach(
        self, actor, robo_dist, offset=0.0, grow_actor_bounds: float = 0.0
    ):
        """
        Places actor out of reach for the robot.
        :param actor: Actor to be placed.
        :param robo_dist: minimum distance from the robot.
        :param offset: Vertical offset.
        :param grow_actor_bounds: How much to grow the bounds of the actor while determining empty space.
        :return: None. Actor pose is set.
        """
        place_actor_out_of_reach_v2(
            actor,
            self._env,
            robo_dist=robo_dist,
            offset=offset,
            grow_actor_bounds=grow_actor_bounds,
        )

    def _take_actor_picture(self, actors):
        """
        Takes images of actors in the environment.
        :param actors: list of actors
        :return: One image per actor.
        """
        images = []
        for i in range(len(actors)):
            image_rgba = take_picture(
                self._scene,
                self._renderer,
                self._obj_cam_position,
                framed_actors=actors[i : i + 1],
                hide_all_non_framed=True,
                resolution=(256, 256),
            )
            images.append(image_rgba)
        return images

    def _take_actor_picture_with_camera(self, actors, camera):
        """
        Takes images of actors in the environment using the given camera.
        :param actors: list of actors.
        :param camera: camera instance to take images with.
        :return: One image per actor.
        """
        images = []
        for i, actor in enumerate(actors):
            image_rgba = take_picture_with_camera(
                self._scene,
                camera,
                framed_actors=actors[i : i + 1],
                hide_all_non_framed=True,
                actor_poses={
                    f"{actor.name}_{actor.id}": sapien.Pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
                },
            )
            images.append(image_rgba)
        return images

    def _sample_int(self, min: int = 1, max: int = 1, sample: bool = True) -> int:
        """
        :param min: Minimum number of actors.
        :param max: Maximum number of actors, inclusive.
        If max < min, then max is set to min, inside the function.
        :param sample: Randomly sample the number of actors.
        :return: If sample is true, an integer in the range [min, max] is returned.
        If sample is false, max is returned
        """
        max = min if min > max else max
        if sample:
            return self._episode_rng.randint(min, max + 1)
        else:
            return max

    @abstractmethod
    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task.
        Actors can immediately be placed in the desired pose.
        :return:
        """
        raise NotImplementedError

    def _save_actor_images(self, actors):
        """
        Save a picture on disk and in `prompt_assets` of each actor for use in multi-modal prompt.
        :param actors: List of actors.
        :return: None
        """
        object_cam = get_camera_by_name(self._scene, name="object_camera")
        actor_imgs = self._take_actor_picture_with_camera(actors, object_cam)
        for i, image_rgba in enumerate(actor_imgs):
            # save to disk
            image_rgba[:, :, 0:3] = np.flip(image_rgba[:, :, 0:3], axis=2)
            image_rgba = cv2.resize(image_rgba, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            path = f"{self.record_dir}/objects/{actors[i].name}_{actors[i].id}.png"
            cv2.imwrite(path, image_rgba)

            # save in prompt_assets
            self.prompt_assets["objects"][f"{actors[i].name}_{actors[i].id}"] = np.flip(
                image_rgba, axis=2
            )

    def _save_texture_images(self):
        """
        Save textures in `prompt_assets`
        """
        textures = self.get_textures()
        for obj_name, (tex_name, tex_path) in textures.items():
            if tex_name is None:
                continue
            tex_img = load_texture(tex_name)
            self.prompt_assets["textures"][obj_name] = tex_img

    def initialize_task(self):
        """
        Called by the env after actors, lighting, etc has been setup.
        Initializes the task.
        """
        if self.record_dir is None:
            return

        os.makedirs(f"{self.record_dir}/objects", exist_ok=True)

        _task_actors = self.get_task_actors()
        self._save_actor_images(_task_actors)

        self._save_texture_images()

        render_cam = get_camera_by_name(self._scene, name="render_camera")
        image = take_picture_with_camera(
            self._scene, render_cam, _task_actors, hide_all_non_framed=True, hide_articulations=True
        )
        cv2.imwrite(f"{self.record_dir}/scene.jpg", np.flip(image[:, :, 0:3], axis=2))

        base_cam = get_camera_by_name(self._scene, name="base_camera")
        image = take_picture_with_camera(
            self._scene, base_cam, _task_actors, hide_articulations=True
        )
        cv2.imwrite(f"{self.record_dir}/scene_base.jpg", np.flip(image[:, :, 0:3], axis=2))

    @abstractmethod
    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: all actors that are relevant to solve the task
        This is used by the environment to return the state observation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_task_textures(self) -> List[tuple]:
        """
        :return: all the textures that are used with corresponding task actors
        """
        raise NotImplementedError

    def get_textual_description(self) -> str:
        """
        Returns the goal of the task in natural language.
        :return:
        """
        return self.get_prompts()[0]

    def get_textures(self) -> Dict[str, str]:
        """
        Returns a mapping of object name and texture used for that object.
        :return:
        """
        task_actors = self.get_task_actors()
        task_textures = self.get_task_textures()
        return {
            f"{actor.name}_{actor.id}": texture
            for (actor, texture) in zip(task_actors, task_textures)
        }

    def get_prompts(self) -> List[str]:
        """
        Returns a list of all the possible prompts for the current episode.
        """
        return self.prompts

    def get_prompt_assets(self) -> Dict[str, np.array]:
        """
        Returns the prompt assets for current episode
        """
        return self.prompt_assets

    def get_predicate(self) -> Predicate:
        """
        :return: top predicate (e.g. a set or a sequence).
        """
        return self.predicate

    def compute_dense_reward(self):
        """
        :return: reward of the current predicate.
        """
        return self.get_predicate().compute_dense_reward()

    def evaluate(self):
        """
        Evaluates the top predicate.
        :return: True if the top predicate is solved successfully and False otherwise
        """
        return self.get_predicate().evaluate()

    @staticmethod
    def check_task_args(task_class, task_args: Dict) -> None:
        """
        This method will emit a warning when an unsupported task_arg is passed to a Task
        :param task_class:
        :param task_args:
        :return:
        """
        for key in task_args:
            if key in task_class.unsupported_task_args:
                log(
                    f"Warning: task {task_class.__name__} does not support task_arg '{key}'.",
                    info=True,
                )


REGISTERED_TASKS = {}


def register_task(name: str, **kwargs):
    """
    Register a new task in ClevrSkills task suite
    :param name: name of the task to register
    :param kwargs:
    :return:
    """

    def _register_task(cls):
        REGISTERED_TASKS[name] = cls
        return cls

    return _register_task


def get_task_class(task_name: str):
    """
    This method returns the class of a registered task
    :param task_name: name of the task
    :return: Class of a registered task.
    Raises a runtime error if the task_name is not found in the registry
    """
    if task_name in REGISTERED_TASKS:
        return REGISTERED_TASKS[task_name]
    task_name = task_name.lower()
    for reg_task_name, task_class in REGISTERED_TASKS.items():
        if reg_task_name.lower() == task_name:
            return task_class
    raise RuntimeError(f"Unknown task {task_name}")
