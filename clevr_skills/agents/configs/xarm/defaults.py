# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import numpy as np
from mani_skill2.agents.controllers import (
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill2.sensors.camera import CameraConfig

from clevr_skills.agents.controllers.vacuum_controller import VacuumControllerConfig
from clevr_skills.utils.paths import PACKAGE_ASSET_DIR


class XarmDefaultConfig:
    """
    Base class for Xarm Configs.
    These control configurations for simulation of the UFACTORY xArm robot.
    """

    def __init__(self) -> None:
        self.ee_link_name = "link_tcp"


class XarmWithGripperDefaultConfig(XarmDefaultConfig):
    """
    Config for xArm with two-fingered gripper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gripper_joint_names = [
            "left_outer_knuckle_joint",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        ]

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

    @property
    def cameras(self):
        """
        :return: the default camera configuration for cameras that are mounted on the robot.
        """
        return CameraConfig(
            uid="hand_camera",
            p=[
                0.070,
                0.01,
                -0.0360011,
            ],
            q=[0, 0.70710678, 0, 0.70710678],
            width=256,
            height=256,
            fov=np.deg2rad(90),
            near=0.01,
            far=10,
            actor_uid="xarm_gripper_base_link",
            hide_link=True,
            texture_names=("Color", "Position", "Segmentation"),
        )


class XarmWithVacuumDefaultConfig(XarmDefaultConfig):
    """
    Base class config for xArm with a vacuum gripper.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def cameras(self):
        """
        :return: CameraConfig for camera mounted on the wrist of the robot.
        """
        return CameraConfig(
            uid="hand_camera",
            p=[0.0464982, -0.0200011, 0.0360011],
            q=[0, 0.70710678, 0, 0.70710678],
            width=256,
            height=256,
            fov=np.deg2rad(90),
            # we used to have focal length (fx) = 128, which at 256 image size translates
            # to 90 degree FOV
            near=0.01,
            far=10,
            actor_uid="vacuum_base_link",  # This used to be called mount_link
            hide_link=True,
            texture_names=("Color", "Position", "Segmentation"),
        )


class Xarm6VacuumDefaultConfig(XarmWithVacuumDefaultConfig):
    """
    Config for xArm6 with a vacuum gripper.
    """

    def __init__(self) -> None:
        super().__init__()  # this will take care of the end-effector TCP

        self.urdf_path = f"{PACKAGE_ASSET_DIR}/descriptions/xarm6_with_vacuum.urdf"
        self.urdf_config = {
            "_materials": {},
            "link": {},
        }

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

    @property
    def controllers(self):
        # Arm
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.05,
            0.05,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose_target = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_target=True,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            frame="base",
            use_delta=False,
            normalize_action=False,
        )

        # Vacuum
        vacuum_controller = VacuumControllerConfig(
            # joint_names: the suction cups are on prismatic joints to simulate the
            # compliance of the gripper
            [
                "suction_cup_1_joint",
                "suction_cup_2_joint",
                "suction_cup_3_joint",
                "suction_cup_4_joint",
                "suction_cup_5_joint",
            ],
        )

        controller_configs = {
            "pd_joint_pos": {"arm": arm_pd_joint_pos, "gripper": vacuum_controller},
            "pd_joint_delta_pos": {"arm": arm_pd_joint_delta_pos, "gripper": vacuum_controller},
            "pd_ee_delta_pos": {"arm": arm_pd_ee_delta_pos, "gripper": vacuum_controller},
            "pd_ee_delta_pose": {"arm": arm_pd_ee_delta_pose, "gripper": vacuum_controller},
            "pd_ee_delta_pose_target": {
                "arm": arm_pd_ee_delta_pose_target,
                "gripper": vacuum_controller,
            },
            "pd_ee_pose": {"arm": arm_pd_ee_pose, "gripper": vacuum_controller},
        }

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
