# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien

from clevr_skills.predicates.on_top import OnTop
from clevr_skills.utils import mesh_primitives
from clevr_skills.utils.actor_placement import place_actor_randomly_v2
from clevr_skills.utils.logger import log
from clevr_skills.utils.render import get_render_material

from .task import Task, register_task


@register_task("Move2d")
class Move2d(Task):
    """
    Pick-and-place a specified object to a specified area without hitting anything else.
    """

    unsupported_task_args = ["variant"]

    def __init__(
        self,
        env,
        spawn_at_gripper: bool = False,
        num_distractors: int = 0,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param spawn_at_gripper: Spawn the to-be-moved actor at the gripper.
        :param num_distractors: Maximum number of additional distractor objects (to be avoided)
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._workspace = [
            [
                -0.5,
                -0.4,
            ],
            [0.2, 0.4],
        ]

        self._spawn_at_gripper = spawn_at_gripper
        self.set_initial_grasping_action(spawn_at_gripper)
        self._num_distractors = (
            self._sample_int(min=1, max=num_distractors, sample=sample_num_actors)
            if num_distractors
            else 0
        )
        self._distractors = []
        self.actor_textures = None
        self._move_actor = None
        self._target_actor = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        # generate actors
        # Add a cylinder to prevent objects from spawning too close to the robot
        temp_cylinder = mesh_primitives.add_cylinder_actor(
            self._scene,
            self._renderer,
            width=0.4,
            depth=0.4,
            height=0.1,
            density=1000,
            name="temp_cylinder",
            render_material=get_render_material(self._renderer, (0.5, 0.5, 0.5)),
            static=True,
        )
        temp_cylinder.set_pose(sapien.Pose(self._env._robot_pos))

        num_actors = 1 + 1 + self._num_distractors

        self.actor_textures = self.model_factory.get_random_top_textures(num_actors)

        render_materials = [
            get_render_material(self._renderer, at[1]) for at in self.actor_textures
        ]

        # Create to-be-moved actor
        self._move_actor = self.model_factory.get_random_top_object(
            self._scene, self._renderer, render_materials[0], size=[0.1, 0.1, 0.05]
        )
        if self._spawn_at_gripper:
            ee_pose = self._env.agent.ee_link.pose
            bounds = self._env._actor_distance.get_bounds(self._move_actor, sapien.Pose())
            self._move_actor.set_pose(
                sapien.Pose(ee_pose.p - np.array([0, 0.0, bounds[1, 2] + 0.0005]))
            )
            self._move_actor.set_velocity([0.0, 0.0, 0.1])  # slight upward velocity
        else:
            self._move_actor.set_pose(
                sapien.Pose(
                    self._episode_rng.uniform(
                        low=np.array([-0.3, -0.2, 0.03]), high=np.array([0.0, 0.2, 0.031])
                    )
                )
            )
        log(f"Move actor name: {self._move_actor.name}")

        # Create target actor. It can be 2D target (embedded in the floor) or a
        # 3D target that might be
        ta_size = [0.2, 0.2, 0.01]
        self._target_actor = self.model_factory.get_random_area_object(
            self._scene, self._renderer, render_materials[1], size=ta_size
        )
        place_actor_randomly_v2(
            self._target_actor, self._env, self._workspace, offset=0.01, allow_top=False
        )

        # add the distractors
        for render_mat in render_materials[2:]:
            size = self._episode_rng.uniform([0.05, 0.05, 0.05], [0.1, 0.1, 0.15])
            distractor = self.model_factory.get_random_top_object(
                self._scene, self._renderer, render_mat, size=size
            )
            place_actor_randomly_v2(
                distractor, self._env, self._workspace, offset=0.01, allow_top=False
            )
            self._distractors.append(distractor)

        # remove cylinder from scene
        temp_cylinder.set_pose(sapien.Pose([1000.0, 0.0, -1000.0]))

        self.predicate = OnTop(self._env, self._move_actor, self._target_actor)
        self.prompts = [
            f"Take up the {self._move_actor.name} and move it to {self._target_actor.name} "
            f"without hitting anything"
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._distractors + [self._move_actor, self._target_actor]

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures
