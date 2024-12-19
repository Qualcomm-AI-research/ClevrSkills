# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat

from clevr_skills.predicates.ee_at_pose import EEAtPose
from clevr_skills.predicates.order import Order
from clevr_skills.utils.logger import log

from .task import Task, register_task


@register_task("MatchPose")
class MatchPose(Task):
    """
    This task creates a random pose for the robot arm.
    The goal is to match the pose given an image of the goal pose.
    """

    goal_thresh = 0.05
    perfect_goal_thresh = 0.005

    unsupported_task_args = ["sample_num_actors"]

    def __init__(
        self,
        env,
        ngoals: int = 1,
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
        self.goal_pose = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """
        actor = self.model_factory.get_random_top_object(
            self._scene, self._renderer, None, tex_name=None
        )
        self._place_actor_at_rand_pose_v2(actor)

        self.goal_pose = []
        for _ in range(self.ngoals):
            for _ in range(self.max_trials):
                goal_x = self._episode_rng.uniform(-0.25, 0.05)
                goal_y = self._episode_rng.uniform(-0.3, 0.3)
                goal_z = self._episode_rng.uniform(0.1, 0.4)
                goal_pos = np.hstack([goal_x, goal_y, goal_z])

                goal_ax = [1, 0, 0]
                goal_angle = (1 + self._episode_rng.rand() * 2) * (np.pi / 2)
                goal_rot = axangle2quat(goal_ax, goal_angle)

                # Ask IK to solve for the goal_pos. The orientation is with the gripper down
                goal_pose = Pose(goal_pos, goal_rot)
                if self._env.agent.controller.controllers["arm"].compute_ik(goal_pose) is None:
                    log(f"_initialize_task: Goal {goal_pos} was out of reach")
                    continue
                for (
                    other_goal_pose
                ) in self.goal_pose:  # goal must be at least a certain distance from other goals
                    if np.linalg.norm(goal_pos - other_goal_pose.p) < 3 * self.goal_thresh:
                        continue
                break

            self.goal_pose.append(goal_pose)

        self.goal_pose = self._get_variant(self.goal_pose)

        predicates = [EEAtPose(self._env, goal) for goal in self.goal_pose]
        self.predicate = Order("MatchPoses", predicates)
        self.prompts = [
            f"Match the pose of the end effector in {{ks:keystep_{1}}}"
            + " followed by ".join([f"{{ks:keystep_{idx+1}}}" for idx in range(1, self.ngoals)])
        ]

    def get_task_actors(self):
        """
        :return: list of actors used in the task.
        """
        return []

    def get_task_textures(self):
        """
        :return: list of textures used in the task.
        """
        return []
