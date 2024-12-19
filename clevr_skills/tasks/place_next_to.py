# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.next_to import NextTo
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .task import Task, register_task


def filter_actor_positions(workspace, pos):
    """
    Filters feasible positions to place an actor according to the workspace
    :param workspace: 2x2 numpy array [[min_x, min_y], [max_x, max_y]]
    :param pos: Nx3 array of positions
    :return: pos, filtered for poinst inside the workspace
    """
    if len(pos.shape) != 2:
        return pos
    mask = np.logical_and(
        np.logical_and(pos[:, 0] >= workspace[0, 0], pos[:, 1] >= workspace[0, 1]),
        np.logical_and(pos[:, 0] <= workspace[1, 0], pos[:, 1] <= workspace[1, 1]),
    )
    return pos[mask]


@register_task("PlaceNextTo")
class PlaceNextTo(Task):
    """
    This task creates one or more actors and one or more areas.
    A specific actor must be picked-and-placed next to a target actor.
    If spawn_at_gripper is True, the specific actor will spawn right underneath
    the robot gripper such that only the place part of the action needs to be performed.
    """

    def __init__(
        self,
        env,
        num_actors=3,
        sample_num_actors: bool = True,
        spawn_at_gripper: bool = False,
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
        self._actor_size = np.array([0.1, 0.1, 0.05])
        self._area_size = np.array([0.18, 0.18, 0.01])

        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )

        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self._area_actors = []
        self._actors = []
        self._spawn_at_gripper = spawn_at_gripper
        self.set_initial_grasping_action(spawn_at_gripper)
        self.actor_textures = None
        self.main_actor = None
        self.main_actor_texture = None
        self.relative_to_actor = None
        self.relative_to_texture = None
        self.relative_description = None
        self.relative_direction = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

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

        ac_idx, rel_ac_idx = self._episode_rng.choice(len(self._actors), size=2, replace=False)
        self.main_actor: sapien.Actor = self._actors[ac_idx]
        self.main_actor_texture = self.actor_textures[ac_idx]

        self.relative_to_actor: sapien.Actor = self._actors[rel_ac_idx]
        self.relative_to_texture = self.actor_textures[rel_ac_idx]

        # Options: left, right, in front, behind
        options = [
            ("left of", [0.0, 1.0, 0.0]),
            ("right of", [0.0, -1.0, 0.0]),
            ("in front of", [1.0, 0.0, 0.0]),
            ("behind", [-1.0, 0.0, 0.0]),
        ]

        # Filter out options that are not feasible in the environment.
        # E.g., can't put something "behind" an actor that is close to the robot base
        feasible_options = []
        backup_options = []
        for desc, direction in options:
            predicate = NextTo(
                self._env, self.main_actor, self.relative_to_actor, np.array(direction), desc
            )
            # Determine if predicate is successful at initial actor placement
            predicate_solved_at_start = predicate.evaluate()["actor_at_pos"]

            # Limit feasible positions to those inside the workspace
            fp = filter_actor_positions(self._workspace, predicate.feasible_positions())

            o = (desc, direction, predicate)
            if fp.shape[0] > 0 and not predicate_solved_at_start:
                feasible_options.append(o)
            else:
                backup_options.append(o)
        feasible_options = feasible_options if len(feasible_options) > 0 else backup_options

        feasible_options = self._get_variant(feasible_options)

        self.relative_description, self.relative_direction, self.predicate = feasible_options[
            self._episode_rng.randint(0, len(feasible_options))
        ]
        self.relative_direction = np.array(self.relative_direction)

        if self._spawn_at_gripper:
            ee_pose = self._env.agent.ee_link.pose
            bounds = self._env._actor_distance.get_bounds(self.main_actor, sapien.Pose())
            self.main_actor.set_pose(
                sapien.Pose(ee_pose.p - np.array([0, 0.0, bounds[1, 2] + 0.0005]))
            )
            self.main_actor.set_velocity([0.0, 0.0, 0.1])  # slight upward velocity

        verb = "Place" if self._spawn_at_gripper else "Pick and place"

        self.prompts = [
            f"{verb} {get_actor_description(self.main_actor)} "
            f"{self.relative_description} "
            f"{get_actor_description(self.relative_to_actor)}",
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures
