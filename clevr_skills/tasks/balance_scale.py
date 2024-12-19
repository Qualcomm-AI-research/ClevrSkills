# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qinverse

from clevr_skills.predicates.balance_scale import BalanceScale
from clevr_skills.predicates.predicate import Predicate
from clevr_skills.utils import mesh_primitives
from clevr_skills.utils.actor_placement import place_actor_randomly_v2
from clevr_skills.utils.logger import log
from clevr_skills.utils.render import get_render_material

from .task import Task, register_task


@register_task("BalanceScale")
class BalanceScaleTask(Task):
    """
    This task creates a scale with two plates.
    There are objects on the ground that must be placed on the scale such that it is in balance.
    There might be "istractor objects that are not related to the task.
    """

    unsupported_task_args = ["variant"]

    def __init__(
        self,
        env,
        num_actors=2,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of objects to be placed on the scale.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
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
                -0.3,
            ],
            [-0.25, 0.3],
        ]
        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self._plate_actors = []
        self.actor_textures = None
        self._scale = None
        self._plate1 = None
        self._plate2 = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """
        # Add a cylinder to prevent objects from spawning too close to the robot
        temp_cylinder = mesh_primitives.add_cylinder_actor(
            self._scene,
            self._renderer,
            width=0.475,
            depth=0.475,
            height=0.1,
            density=1000,
            name="temp_cylinder",
            render_material=get_render_material(self._renderer, (0.5, 0.5, 0.5)),
            static=True,
        )
        temp_cylinder.set_pose(sapien.Pose(self._env._robot_pos))

        # sample the primitive type for the stacking objects
        primitive_name = self._episode_rng.choice(["cube", "cylinder"])

        # sample the number of blocks destined for each plate
        while True:  # loop until a good config is found
            plate1_num_blocks = 1 + (
                self._episode_rng.randint(self._num_actors - 2) if self._num_actors > 2 else 0
            )
            plate2_num_blocks = self._num_actors - plate1_num_blocks
            if plate1_num_blocks > plate2_num_blocks:  # ensure plate 1 has less blocks that plate 2
                plate1_num_blocks, plate2_num_blocks = plate2_num_blocks, plate1_num_blocks

            # sample the height of each "slice" (block height will be a discrete multiple
            # of the slice height)
            slice_height = self._episode_rng.uniform(low=0.005, high=0.015)

            # sample height of the blocks.
            MIN_BLOCK_HEIGHT = 0.021
            MAX_BLOCK_HEIGHT = 0.08
            MAX_TOWER_HEIGHT = 0.15
            block_min_num_slices = int(np.ceil(MIN_BLOCK_HEIGHT / slice_height))
            block_max_num_slices = int(np.trunc(MAX_BLOCK_HEIGHT / slice_height))
            tower_max_slices = min(
                int(np.trunc(MAX_TOWER_HEIGHT / slice_height)),
                plate1_num_blocks * block_max_num_slices,
            )
            plate1_num_slices = np.array([block_min_num_slices] * plate1_num_blocks)
            while (
                np.sum(plate1_num_slices) < tower_max_slices
            ):  # sample the number of slices for the plate1 blocks
                plate1_num_slices[
                    self._episode_rng.choice(np.where(plate1_num_slices < block_max_num_slices)[0])
                ] += 1
            plate2_num_slices = np.array([block_min_num_slices] * plate2_num_blocks)
            while (
                np.sum(plate2_num_slices) < tower_max_slices
            ):  # sample the number of slices for the plate2 blocks
                plate2_num_slices[
                    self._episode_rng.choice(np.where(plate2_num_slices < block_max_num_slices)[0])
                ] += 1

            if np.sum(plate1_num_slices) == np.sum(plate2_num_slices):
                break

        block_heights = np.concatenate(
            (slice_height * plate1_num_slices, slice_height * plate2_num_slices)
        )
        log(f"block_heights: {block_heights}")

        # sample the color / texture
        self.actor_textures = self.model_factory.get_random_top_textures(1)
        render_materials = [
            get_render_material(self._renderer, at[1]) for at in self.actor_textures
        ]

        # Create the actors that are to-be-placed on the scale
        for _, h in enumerate(block_heights):
            render_material = render_materials[0]
            actor_size = (0.1, 0.1, h)
            width = actor_size[0]
            depth = actor_size[1]
            height = actor_size[2]
            actor_name = f"{self.actor_textures[0][0]} {primitive_name}"
            fn = self.model_factory.primitives[primitive_name]
            actor = fn(
                self._scene,
                self._renderer,
                render_material=render_material,
                width=width,
                depth=depth,
                height=height,
                name=actor_name,
                density=300,
            )
            self._plate_actors.append(actor)

            place_actor_randomly_v2(actor, self._env, self._workspace, offset=0.01, allow_top=False)

        temp_cylinder.set_pose(sapien.Pose([1000.0, 0.0, -1000.0]))

        self._scale, self._plate1, self._plate2 = self._add_scale(
            sapien.Pose([-0.05, 0.0, 0.7]),
            arm1_length=0.25,
            arm2_length=0.25,
            limit1=np.deg2rad(20),
            limit2=np.deg2rad(20),
            damping=0.01,
            friction=0.0001,
            plate1_offset=0.6,
            plate2_offset=0.6,
        )

        self.predicate = BalanceScale(
            self._env, self._scale, self._plate1, self._plate2, self._plate_actors
        )

        self.prompts = [
            f"Place all the {self.actor_textures[0][0]} {primitive_name}s on the "
            f"scale while keeping it in balance",
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task
        """
        return self._plate_actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: all the textures that are used with corresponding task actors
        """
        return []

    def get_predicate(self) -> Predicate:
        return self.predicate

    def _add_scale(
        self,
        pose: sapien.Pose = None,
        arm1_length: float = 0.3,
        arm2_length: float = 0.3,
        limit1=np.deg2rad(30),
        limit2=np.deg2rad(30),
        damping: float = 0.1,
        friction: float = 0.1,
        plate1_offset: float = 0.4,
        plate2_offset: float = 0.4,
    ):
        """
        Adds a scale model to the scene
        :param pose: position of central rotation point (i.e. top of the scale).
        The main points is mounted as a fixed body (we could make it mobile as well so the robot
        can throw it over)
        :param arm1_length: how long the arm 1 is (measured from the main central point)
        :param arm2_length: how long the arm 1 is (measured the main central point)
        :param limit1: rotation limit arm 1 downwards (radians)
        :param limit1: rotation limit arm 2 downwards ((radians)
        :param damping: damping in the central rotation point. More damping makes the task easier.
        :param friction: friction in the central plate 1 is (measured from the arm)
        :param plate2_offset: how far down the plate 2 is (measured from the arm)
        :return:
        """
        # Build the scale articulation (root, horizontal arm, both plates)
        builder: sapien.ArticulationBuilder = self._scene.create_articulation_builder()
        root_builder: sapien.LinkBuilder = (
            builder.create_link_builder()
        )  # LinkBuilder is similar to ActorBuilder
        root_builder.set_name("scales_root")
        # Note: capsule geometry (as used before in this spot) has a bug in
        # ManiSkill2 trimesh_utils.py -> get_actor_meshes
        root_builder.add_box_visual(half_size=[0.05, 0.015, 0.015], color=(1.0, 0.0, 0.0))
        root_builder.add_box_collision(half_size=[0.05, 0.015, 0.015])

        arm_builder: sapien.LinkBuilder = builder.create_link_builder(root_builder)
        arm_builder.set_name("scales_arm")
        arm_builder.set_joint_name("scales_arm")
        arm_builder.set_joint_properties(
            "revolute",
            limits=[[-limit1, limit2]],  # joint limits (for each DoF)
            pose_in_parent=sapien.Pose(p=[0, 0, 0.0], q=[1.0, 0.0, 0.0, 0.0]),
            pose_in_child=sapien.Pose(p=[0, 0, 0], q=qinverse([1.0, 0.0, 0.0, 0.0])),
            friction=friction,
            damping=damping,
        )
        arm_capsule_pose = sapien.Pose([0, 0, 0], euler2quat(0, 0, np.pi / 2))
        hl = (arm1_length + arm2_length) / 2
        # Note: capsule geometry (as used before in this spot) has a bug in ManiSkill2
        # trimesh_utils.py -> get_actor_meshes
        arm_builder.add_box_visual(
            pose=arm_capsule_pose, half_size=[hl, 0.02, 0.02], color=(0.0, 1.0, 0.0)
        )
        arm_builder.add_box_collision(pose=arm_capsule_pose, half_size=[hl, 0.02, 0.02])

        plate1_name = "plate1"
        plate2_name = "plate2"

        plate1_builder: sapien.LinkBuilder = builder.create_link_builder(arm_builder)
        plate1_builder.set_name(plate1_name)
        plate1_builder.set_joint_name(plate1_name)
        plate1_builder.set_joint_properties(
            "revolute",
            limits=[[-np.deg2rad(89), np.deg2rad(89)]],  # joint limits (for each DoF)
            pose_in_parent=sapien.Pose(p=[0, -arm1_length, 0], q=[1.0, 0.0, 0.0, 0.0]),
            pose_in_child=sapien.Pose(p=[0, 0, plate1_offset], q=qinverse([1.0, 0.0, 0.0, 0.0])),
            friction=friction,
            damping=damping,
        )
        plate1_builder.add_box_collision(sapien.Pose([0.0, 0.0, 0.0]), half_size=[0.05, 0.05, 0.01])
        plate1_builder.add_box_visual(
            sapien.Pose([0.0, 0.0, 0.0]), half_size=[0.05, 0.05, 0.01], color=(0.0, 1.0, 0.0)
        )

        plate2_builder = builder.create_link_builder(arm_builder)
        plate2_builder.set_name(plate2_name)
        plate2_builder.set_joint_name(plate2_name)
        plate2_builder.set_joint_properties(
            "revolute",
            limits=[[-np.deg2rad(89), np.deg2rad(89)]],  # joint limits (for each DoF)
            pose_in_parent=sapien.Pose(p=[0, arm2_length, 0], q=[1.0, 0.0, 0.0, 0.0]),
            pose_in_child=sapien.Pose(p=[0, 0, plate2_offset], q=qinverse([1.0, 0.0, 0.0, 0.0])),
            friction=friction,
            damping=damping,
        )
        plate2_builder.add_box_collision(sapien.Pose([0.0, 0.0, 0.0]), half_size=[0.05, 0.05, 0.01])
        plate2_builder.add_box_visual(
            sapien.Pose([0.0, 0.0, 0.0]), half_size=[0.05, 0.05, 0.01], color=(0.0, 1.0, 0.0)
        )

        # Build the articulation
        scale = builder.build(fix_root_link=True)
        scale.set_name("scale")

        # Give the scale a slight preference for going to "neutral" pose
        scale.get_active_joints()[0].set_drive_property(stiffness=0.1, damping=0.1)
        scale.get_active_joints()[0].set_drive_target(0.0)

        # Retrieve the links
        plate1 = plate2 = None
        for link in scale.get_links():
            if link.name == plate1_name:
                plate1 = link
            elif link.name == plate2_name:
                plate2 = link

        # Optionally set pose
        if pose:
            scale.set_pose(pose)

        del builder  # in the past, had segfault due to not cleaning up builder explicitly

        return scale, plate1, plate2
