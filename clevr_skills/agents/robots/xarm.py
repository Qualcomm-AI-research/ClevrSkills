# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os
from xml.dom import minidom

import numpy as np
import sapien.core as sapien
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import get_entity_by_name, get_pairwise_contact_impulse

from clevr_skills.agents.configs.xarm import defaults as xarm_defaults


class XArm(BaseAgent):

    def _after_init(self):
        """
        Called by ManiSkill2 after initialization of the robot.
        Allows for initialization of the class.
        :return:
        """
        self.finger1_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "left_finger"
        )
        self.finger2_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "right_finger"
        )
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

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        """
        :param actor: any actor from the scene.
        :param min_impulse: Threshold on impulse between fingers and actors.
        :param max_angle: Threshold on the angle between the actor and the fingers.
        :return: True if actor is currently grasped by the gripper
        """
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        rflag = np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle

        return all([lflag, rflag])

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        """
        :param actor: any actor from the scene.
        :param min_impulse: Threshold on impulse between fingers and actors.
        :return: True if either finger is in contact with the actor.
        """
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
        Build a grasp pose (panda_hand_tcp).
        :param approaching: approaching vector.
        :param closing: vector in direction between fingers.
        :param center: Position where the end-effector center should go.
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        tf_matrix[:3, 3] = center
        return sapien.Pose.from_transformation_matrix(tf_matrix)

    def get_fingers_info(self):
        """
        :return: Dictionary with both position and velocity of end-effector fingers
        """
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        """
        :return: Coordinates of end-effector fingers
        """
        finger_tips = [
            self.finger1_link.pose.transform(sapien.Pose([0, -0.0275, 0.055])).p,
            self.finger2_link.pose.transform(sapien.Pose([0, 0.0275, 0.055])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        """
        :return: Velocity of end-effector fingers
        """
        finger_vels = [
            self.finger1_link.get_velocity(),
            self.finger2_link.get_velocity(),
        ]
        return np.array(finger_vels)

    def get_wrist_qpos(self):
        """
        :return: rotation of wrist of robot, in radians.
        """
        return self.robot.get_qpos()[5]

    def get_mplib_joint_limits(self):
        """
        :return: Joint limits for MPLib motion planning.
        This restricts the ability of MPLib to find "twisted" solutions using the full
        720 degree rotational range on some joints.
        """
        return {
            "joint1": (-np.pi, np.pi),
            "joint4": (-np.pi, np.pi),
            "joint6": (-1.5 * np.pi, 1.5 * np.pi),
        }


class XArm6Vacuum(XArm):
    _config: xarm_defaults.Xarm6VacuumDefaultConfig

    @classmethod
    def get_default_config(cls):
        """
        Called by ManiSkill2 to retrieve the default config of robot.
        :return: default config.
        """
        return xarm_defaults.Xarm6VacuumDefaultConfig()

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
        Warning: this method is not part of other ManiSkill2 robots
        :return: A list of actors that are currently grasped by the robot.
        """
        return self.controller.controllers["gripper"].grasping
