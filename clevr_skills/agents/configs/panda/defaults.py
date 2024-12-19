# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

from mani_skill2.agents.configs.panda.defaults import PandaDefaultConfig
from mani_skill2.agents.controllers import deepcopy_dict
from mani_skill2.sensors.camera import CameraConfig

from clevr_skills.agents.controllers.vacuum_controller import VacuumControllerConfig
from clevr_skills.utils.paths import PACKAGE_ASSET_DIR


class PandaWithVacuumDefaultConfig(PandaDefaultConfig):
    def __init__(self) -> None:
        """
        Creates a Panda robot with a simple vacuum gripper with 5 suction cups.
        """
        super().__init__()  # this will take care of the end-effector TCP

        self.urdf_path = f"{PACKAGE_ASSET_DIR}/descriptions/panda_v2_with_vacuum.urdf"
        self.ee_link_name = "link_tcp"

    @property
    def controllers(self):
        """
        :return: the ManiSkill2 controllers for the arm and the gripper.
        """
        controller_configs = deepcopy_dict(PandaDefaultConfig.controllers.__get__(self))

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

        for _, cc in controller_configs.items():
            if "gripper" in cc:
                cc["gripper"] = vacuum_controller
        return controller_configs

    @property
    def cameras(self):
        """
        :return: CameraConfig for camera mounted on the wrist of the robot.
        """
        # Copy the default panda camera, but change the actor_uid
        c = PandaDefaultConfig.cameras.__get__(self)
        return CameraConfig(
            uid=c.uid,
            p=c.p,
            q=c.q,
            width=c.width,
            height=c.height,
            fov=c.fov,
            near=c.near,
            far=c.far,
            actor_uid="vacuum_base_link",
        )
