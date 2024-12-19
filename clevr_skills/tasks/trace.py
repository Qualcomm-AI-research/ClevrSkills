# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

from typing import List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from clevr_skills.predicates.trace_order import TraceOrder
from clevr_skills.predicates.trace_predicate import TracePredicate
from clevr_skills.utils.logger import log

from .task import Task, register_task


@register_task("Trace")
class Trace(Task):
    """
    Trace out a path indicated by colored spheres.
    """

    goal_thresh = 0.05
    perfect_goal_thresh = 0.005

    color_next_goal = (0, 1, 0)
    color_visited_goal = (1, 1, 0)
    color_future_goal = (1, 0, 0)
    all_colors = [color_next_goal, color_visited_goal, color_future_goal]
    color_names = ["next", "visited", "future"]

    unsupported_task_args = ["sample_num_actors"]

    def __init__(
        self,
        env,
        ngoals: int = 3,
        variant: int = 0,
        sample_num_actors: bool = True,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param ngoals: The number of goal poses.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param sample_num_actors: Ignored for this task.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """

        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self.ngoals = ngoals

        self.max_trials = 100
        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )
        self._colored_goal_sites = None
        self.goal_sites = None
        self.goal_pos = None
        self._current_goal_idx = None
        self._accumulated_reward = None
        self._previously_released_reward = None
        self._goal_tcp_min_distance = None
        self._initial_tcp_pos = None
        self._last_qvel_thresh = None
        self.predicate = None
        self.prompts = None

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Borrowed from pick and place environment. Build a sphere site"
        (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        sphere.hide_visual()
        return sphere

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """
        # adding a distractor
        actor = self.model_factory.get_random_top_object(
            self._scene, self._renderer, None, tex_name=None
        )
        self._place_actor_at_rand_pose_v2(actor)

        # Create a sphere for each goal, in each color (color can't be
        # changed after object has been created)
        self._colored_goal_sites = []
        for g in range(self.ngoals):
            self._colored_goal_sites.append({})
            for color, name in zip(self.all_colors, self.color_names):
                radius = self.goal_thresh * 2.0 if name == "next" else self.goal_thresh
                self._colored_goal_sites[g][color] = self._build_sphere_site(
                    radius, color=color, name=f"goal_site_{g}_{name}"
                )
                self._colored_goal_sites[g][color].unhide_visual()

        self.goal_sites = [None] * self.ngoals

        self.goal_pos = []
        for g in range(self.ngoals):
            for _ in range(self.max_trials):
                goal_x = self._episode_rng.uniform(-0.25, 0.05)
                goal_y = self._episode_rng.uniform(-0.3, 0.3)
                goal_z = self._episode_rng.uniform(0.1, 0.4)
                goal_pos = np.hstack([goal_x, goal_y, goal_z])

                # Ask IK to solve for the goal_pos. The orientation is with the gripper down
                goal_pose = sapien.Pose(goal_pos, [0.0, 1.0, 0.0, 0.0])
                if self._env.agent.controller.controllers["arm"].compute_ik(goal_pose) is None:
                    log(f"_initialize_task: Goal {goal_pos} was out of reach")
                    continue
                for (
                    other_goal_pos
                ) in self.goal_pos:  # goal must be at least a certain distance from other goals
                    if np.linalg.norm(goal_pos - other_goal_pos) < 3 * self.goal_thresh:
                        continue
                break

            self.goal_pos.append(goal_pos)

        self.goal_sites, self.goal_pos = self._get_variant(self.goal_sites, self.goal_pos)

        # These variables are used to compute reward and do visualizations
        self._current_goal_idx = 0
        self._accumulated_reward = 0.0
        self._previously_released_reward = 0.0
        self._goal_tcp_min_distance = 1e10 * np.ones((self.ngoals,))
        self._initial_tcp_pos = None
        self._last_qvel_thresh = 1.0

        predicates = [
            TracePredicate(
                self._env,
                self._colored_goal_sites[idx],
                Pose(goal, [0, 0, 0, 1]),
                first_predicate=idx == 0,
            )
            for idx, goal in enumerate(self.goal_pos)
        ]
        self.predicate = TraceOrder("Trace", predicates)
        self.prompts = ["Trace the sequence of goals by moving to the next green goal."]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return []

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return []
