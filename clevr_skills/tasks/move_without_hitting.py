# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from clevr_skills.predicates.ee_at_pos import EEAtPosition
from clevr_skills.predicates.order import Order
from clevr_skills.utils.logger import log
from clevr_skills.utils.render import get_render_material

from .task import register_task
from .trace import Trace


@register_task("MoveWithoutHitting")
class MoveWithoutHitting(Trace):
    """
    This task creates a random pose for the robot arm.
    The goal is to match the pose of the robot given in the goal image without hitting
    any objects suspended in the air.
    """

    unsupported_task_args = ["variant"]

    def __init__(
        self,
        env,
        num_distractions=5,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_distractions: Number of distractor objects.
        :param sample_num_actors: Ignored for this task.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, ngoals=1, record_dir=record_dir, split=split, variant=variant)
        self.num_distractions = num_distractions

        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )
        self.goal_thresh = 1
        self.goal_pos = None
        self.actor_textures = None
        self._actors = None
        self._current_goal_idx = None
        self._accumulated_reward = None
        self._previously_released_reward = None
        self._goal_tcp_min_distance = None
        self._initial_tcp_pos = None
        self._last_qvel_thresh = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """
        self.goal_pos = []
        for _ in range(self.num_distractions + 1):
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

        self.actor_textures = self.model_factory.get_random_top_textures(self.num_distractions)
        render_material = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]
        self._actors = [
            self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_material[idx][1],
                tex_name=render_material[idx][0],
                size=0.5,
                static=True,
            )
            for idx in range(self.num_distractions)
        ]

        log("---" * 10)
        for actor, pos in zip(self._actors, self.goal_pos[:-1]):
            log(f"actor {actor.name}")
            actor.set_pose(Pose(pos, [0, 0, 0, 1]))

        # These variables are used to compute reward and do visualizations
        self._current_goal_idx = 0
        self._accumulated_reward = 0.0
        self._previously_released_reward = 0.0
        self._goal_tcp_min_distance = 1e10 * np.ones((self.ngoals,))
        self._initial_tcp_pos = None
        self._last_qvel_thresh = 1.0

        predicates = [EEAtPosition(self._env, Pose(self.goal_pos[-1], [0, 0, 0, 1]))]
        self.predicate = Order("MoveWithoutHitting", predicates)
        self.prompts = [
            f"Match the pose of the end effector in {{ks:keystep_{1}}} without hitting any objects"
            + " followed by ".join([f"{{ks:keystep_{idx+1}}}" for idx in range(1, self.ngoals)])
        ]
