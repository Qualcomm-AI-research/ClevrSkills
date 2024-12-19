# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os
from xml.dom import minidom

import numpy as np
import sapien.core as sapien
from mani_skill2 import format_path
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.utils.sapien_utils import get_entity_by_name

from clevr_skills.agents.configs.panda import defaults


class ClevrSkillsPanda(Panda):
    _config: defaults.PandaWithVacuumDefaultConfig

    @classmethod
    def get_default_config(cls):
        """
        Called by ManiSkill2 to retrieve the default config of robot.
        :return: default config.
        """
        return defaults.PandaWithVacuumDefaultConfig()

    def _after_init(self):
        """
        Called by ManiSkill2; performs initialization on the robot.
        :return: None
        """
        super()._after_init()

        self.ee_link = get_entity_by_name(self.robot.get_links(), "link_tcp")

        # Get list of all links that are part of the gripper:
        self.ee_links = []
        srdf_path = os.path.splitext(self.urdf_path)[0] + ".srdf"
        srdf_xml = minidom.parse(srdf_path)
        for group_xml in srdf_xml.getElementsByTagName("group"):
            if group_xml.attributes["name"].value == "gripper_without_camera":
                for link_xml in group_xml.getElementsByTagName("link"):
                    link_name = link_xml.attributes["name"].value
                    link = get_entity_by_name(self.robot.get_links(), link_name)
                    if link:
                        self.ee_links.append(link)

        self.neutral_ee_quat = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)

        # Points inside this cylinder are ignored during path planning:
        self.base_link_radius = 0.09
        self.base_link_height = [-0.025, 0.2]

        self.urdf_path = format_path(self.urdf_path)

    def reset(self, init_qpos=None):
        """
        Resets the robot, optionally to init_qpos
        :param init_qpos: pose of each joint.
        :return: None
        """
        super().reset(init_qpos=init_qpos)

    def before_simulation_step(self):
        """
        Stores contact info from scene in the gripper controller, then calls superclass.
        :return: None
        """
        self.controller.controllers["gripper"]._contact_during_last_sim_step = (
            self.env._contact_during_last_sim_step
        )
        super().before_simulation_step()

    def check_grasp(self, actor: sapien.ActorBase):
        """
        :param actor: any actor in the scene.
        :return: True if actor is grasped
        """
        return actor in self.controller.controllers["gripper"].grasping

    def get_grasped_actors(self):
        """
        Warning: this method is not supported by other ManiSkill2 robots
        :return: A list of actors that are currently grasped by the robot.
        """
        return self.controller.controllers["gripper"].grasping

    def get_wrist_qpos(self):
        """
        :return: rotation of wrist of robot, in radians.
        """
        return self.robot.get_qpos()[6]

    def get_mplib_joint_limits(self):
        """
        :return: Joint limits for MPLib motion planning. For the Panda robot,
        this is an empty Dict because the version of Panda in ManiSkill2 already
        has reasonable joint limits by default.
        """
        return {}
