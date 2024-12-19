# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.trimesh_utils import get_actor_mesh

from clevr_skills.predicates.hit_pose_predicate import HitPosPredicate
from clevr_skills.predicates.hit_predicate import HitPredicate
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .task import Task, register_task


@register_task("Throw")
class Throw(Task):
    """
    Throw one object against (or over) another object, optionally toppling the object.
    If spawn_at_gripper is True, the specific actor will spawn right underneath
    the robot gripper such that only the throwing part of the action needs to be performed.
    """

    unsupported_task_args = ["variant"]

    def __init__(
        self,
        env,
        spawn_at_gripper: bool = False,
        target_2d: bool = True,
        topple_target: bool = False,
        direction: str = None,
        num_distractors: int = 0,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param spawn_at_gripper: Spawn the to-be-thrown actor at the gripper.
        :param target_2d: If True, the target will be a flat, static shape embedded
        the ground and merely throwing the actor overhead will complete the task.
        If False, the target will be a dynamic actor, and you have to hit it.
        :param topple_target: If true, the task does not succeed until the target is toppled
        (>45 degree change in orientation)
        :param direction: Where the actor should be thrown. One of: "north", "south", "east", "west".
        :param num_distractors: Maximum number of additional distractor objects.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param spawn_at_gripper: When True, the main actor will be created directly underneath the
        gripper with a small upwards velocity. If the agent turns on the gripper, the actor
        will stick to it.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._workspace = np.array(
            [
                [
                    -0.4,
                    -0.30,
                ],
                [0.05, 0.30],
            ]
        )

        self._num_distractors = (
            self._sample_int(min=1, max=num_distractors, sample=sample_num_actors)
            if num_distractors
            else 0
        )

        self._target_2d = target_2d
        self._topple_target = topple_target
        self._distractors = []
        self._spawn_at_gripper = spawn_at_gripper
        self.set_initial_grasping_action(spawn_at_gripper)
        self.direction = direction
        if self.direction is not None:
            assert self.direction in ["north", "south", "east", "west"]
        self.actor_textures = None
        self._target_actor = None
        self._target_pos = None
        self.predicate = None
        self.prompts = None
        self._throw_actor = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        # generate actors
        num_actors = 1 + 1 + self._num_distractors

        self.actor_textures = self.model_factory.get_random_top_textures(num_actors)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]

        # Create to-be-thrown actor
        self._throw_actor = self.model_factory.get_random_top_object(
            self._scene,
            self._renderer,
            render_materials[0][1],
            tex_name=render_materials[0][0],
            size=[0.125, 0.125, 0.025],
        )
        if self._spawn_at_gripper:
            ee_pose = self._env.agent.ee_link.pose
            bounds = self._env._actor_distance.get_bounds(self._throw_actor, sapien.Pose())
            self._throw_actor.set_pose(
                sapien.Pose(ee_pose.p - np.array([0, 0.0, bounds[1, 2] + 0.0005]))
            )
            self._throw_actor.set_velocity([0.0, 0.0, 0.1])  # slight upward velocity
        else:
            self._throw_actor.set_pose(
                sapien.Pose(
                    self._episode_rng.uniform(
                        low=np.array([-0.3, -0.2, 0.03]), high=np.array([0.0, 0.2, 0.031])
                    )
                )
            )

        # Create target actor. It can be 2D target (embedded in the floor) or
        # a 3D target that can be toppled
        if self._target_2d or (not self._topple_target and self._episode_rng.random() < 0.5):
            ta_size = [0.2, 0.2, 0.01]
            self._target_actor = self.model_factory.get_random_area_object(
                self._scene,
                self._renderer,
                render_materials[1][1],
                tex_name=render_materials[1][0],
                size=ta_size,
            )
        else:
            ta_height = 0.25 if self._topple_target else 0.15
            ta_size = [0.05, 0.05, ta_height]
            self._target_actor = self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_materials[1][1],
                tex_name=render_materials[1][0],
                size=ta_size,
                density=100,
            )

        # Place the target actor. It must be out of reach for the robot
        OUT_OF_REACH = 0.75  # at least 80 cm away from robot
        target_distance = self._episode_rng.uniform(low=OUT_OF_REACH, high=OUT_OF_REACH + 0.1)
        robot_pos = self._env._robot_pos
        target_angle = self._episode_rng.uniform(low=np.deg2rad(45), high=np.deg2rad(45 + 90))
        target_pos = np.array(
            [
                robot_pos[0] + np.sin(target_angle) * target_distance,
                robot_pos[1] + np.cos(target_angle) * target_distance,
                ta_size[2] / 2 + 0.001,
            ]
        )
        self._target_actor.set_pose(sapien.Pose(target_pos))

        if self.direction is not None:
            actor_size = get_actor_mesh(self._target_actor).vertices
            actor_size = actor_size.max(0) - actor_size.min(0)
            dir_axis = {"west": 1, "east": 1, "north": 0, "south": 0}
            dir_mult = {"west": 1, "east": -1, "north": 1, "south": -1}
            dir_offset = 0.1 + actor_size[dir_axis[self.direction]]

            self._target_pos = self._target_actor.pose.p

            self._target_pos[dir_axis[self.direction]] += dir_mult[self.direction] * dir_offset

        self._distractors = [
            self.model_factory.get_random_object(
                self._scene, self._renderer, render_material=rm[1], tex_name=rm[0]
            )
            for rm in render_materials[2:]
        ]
        for da in self._distractors:
            self._place_actor_at_rand_pose_v2(da, grow_actor_bounds=0.01)

        if self.direction is None:
            self.predicate = HitPredicate(
                self._env,
                self._throw_actor,
                self._target_actor,
                self._target_2d,
                self._topple_target,
            )
        else:
            self.predicate = HitPosPredicate(self._env, self._throw_actor, self._target_pos)

        if self.direction is None:
            if self._target_2d:
                self.prompts = [
                    f"Throw the {get_actor_description(self._throw_actor)} to the "
                    f"{get_actor_description(self._target_actor)}"
                ]
            else:
                if self._topple_target:
                    self.prompts = [
                        f"Throw the {get_actor_description(self._throw_actor)} to the "
                        f"{get_actor_description(self._target_actor)} such that it falls over",
                        f"Hit the {get_actor_description(self._target_actor)} with the "
                        f"{get_actor_description(self._throw_actor)} such that it falls over",
                    ]
                else:
                    self.prompts = [
                        f"Throw the {get_actor_description(self._throw_actor)} to the "
                        f"{get_actor_description(self._target_actor)}",
                        f"Hit the {get_actor_description(self._target_actor)} with the "
                        f"{get_actor_description(self._throw_actor)}",
                    ]
        else:
            self.prompts = [
                f"Throw the {get_actor_description(self._throw_actor)} to the "
                f"{self.direction} of {get_actor_description(self._target_actor)}"
            ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return [self._throw_actor, self._target_actor] + self._distractors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures
