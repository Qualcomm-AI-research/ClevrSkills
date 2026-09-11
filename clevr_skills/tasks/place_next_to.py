# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.at_position import AtPosition
from clevr_skills.predicates.next_to import NextTo
from clevr_skills.predicates.order import Order
from clevr_skills.utils.action_trace import at_get_pose
from clevr_skills.utils.geometry import euclidean_distance_matrix
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


def filter_area_edges(area: np.ndarray, threshold: float = 0.001):
    """
    Assuming that "area" represents a (part of a) regular grid of points,
    discards the points on the edges.
    A heuristic is used to determine the grid spacing. If the heuristic fails, the full area array is returned.
    :param area: N x 3 array of points. Assumed to be sampled from a regular grid.
    :param threshold: threshold for distances that are considered to be 0. Default is 1mm.
    :return: The points from area which are not on the edge, or (if the heuristic fails) the full area array.
    """
    # Compute distance between each grid point
    dm = euclidean_distance_matrix(area, area)

    # Figure out diagonal distance in the grid (using heuristic)
    d = np.sort(dm.flatten())
    d = d[d > threshold]
    if len(d) == 0:  # heuristic failure?
        return area
    min_d = (
        np.sqrt(3) * d[0] + threshold
    )  # sqrt 3 is used to make distance threshold slightly larger

    # Compute the number of neighbors
    num_neighbors = np.sum(dm < min_d, axis=0)
    for c, t in [(9, 2), (7, 1), (5, 1)]:
        valid_indices = np.where(num_neighbors >= c)[0]
        if len(valid_indices) >= t:
            return area[valid_indices]
    return area


@register_task("PlaceNextTo")
class PlaceNextTo(Task):
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
        sample_num_actors: bool = True,
        spawn_at_gripper: bool = False,
        variant: int = 0,
        record_dir=None,
        split="train",
        restore=False,
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
                    -0.35,
                    -0.27,
                ],
                [0.0, 0.30],
            ]
        )

        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self._area_actors = []
        self._actors = []
        self._spawn_at_gripper = spawn_at_gripper
        self.set_initial_grasping_action(spawn_at_gripper)

        self.restore = restore

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        :return:
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

        for _attempt in range(
            10
        ):  # attempt the actor placement 10 times, to avoid starting in solved state
            for actor in self._actors:
                self._place_actor_at_rand_pose_v2(actor)

            feasible_options = []
            backup_options = []
            for desc, direction in options:
                if not self.restore:
                    predicate = NextTo(
                        self._env,
                        self.main_actor,
                        self.relative_to_actor,
                        np.array(direction),
                        desc,
                    )
                else:
                    init_pos = self.main_actor.get_pose()
                    predicate = Order(
                        "PlaceNextToRestore",
                        [
                            NextTo(
                                self._env,
                                self.main_actor,
                                self.relative_to_actor,
                                np.array(direction),
                                desc,
                            ),
                            AtPosition(self._env, self.main_actor, init_pos),
                        ],
                    )
                next_to_pred = (
                    predicate if isinstance(predicate, NextTo) else predicate._predicates[0]
                )
                # Determine if predicate is successful at initial actor placement
                predicate_solved_at_start = next_to_pred.evaluate()["actor_at_pos"]

                # Limit feasible positions to those inside the workspace
                fp = filter_actor_positions(self._workspace, next_to_pred.feasible_positions())

                o = (desc, direction, predicate)
                if fp.shape[0] > 0 and not predicate_solved_at_start:
                    feasible_options.append(o)
                else:
                    backup_options.append(o)

            if len(feasible_options) > 0:
                break

        feasible_options = feasible_options if len(feasible_options) > 0 else backup_options

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

        self.prompts = [
            f"Place {get_actor_description(self.main_actor)} {self.relative_description} {get_actor_description(self.relative_to_actor)}",
        ]
        if self.restore:
            for idx in range(len(self.prompts)):
                self.prompts[idx] += " and restore to initial position"

        self.init_state = {
            "objects": {
                get_actor_description(actor): str(at_get_pose(actor.pose)) for actor in self._actors
            },
            "init_pose": str(at_get_pose(self.main_actor.pose)),
        }

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

    def get_textual_description(self) -> str:
        return self.prompts[0]
