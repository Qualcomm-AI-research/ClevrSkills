# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from mani_skill2.agents.base_controller import BaseController, ControllerConfig
from transforms3d.quaternions import quat2mat


class VacuumController(BaseController):
    config: "VacuumControllerConfig"

    def _initialize_action_space(self):
        """
        Called by superclass. Sets the action space of this controller.
        :return: None
        """
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

    def set_drive_property(self):
        """
        Called by superclass. Sets the stiffness, damping, friction, force limit of the joints.
        Eachjoint corresponds to one suction cup. Each cup can move forward backward a little for some compliance.
        :return: None
        """
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(stiffness[i], damping[i], force_limit=force_limit[i])
            joint.set_friction(friction[i])

    def reset(self):
        """
        Resets vacuum controller to original state.
        :return:
        """
        super().reset()
        self.vacuum_on = False
        num_suction_cups = len(self.config.joint_names)
        self.suction_history = np.zeros(num_suction_cups)
        self.grasping = []

    def _preprocess_action(self, action: np.ndarray):
        """
        Part of interface, called by superclass. This particular implementation does not modify the action.
        :param action: Whether the suction is on or off. This is actually a numpy array of length 1.
        :return: action
        """
        return action

    def set_action(self, action: np.ndarray):
        """
        Stores the action, sets joint drive targets.
        :param action: a single float. If the value is above 0, the vacuum is considered to be "on".
        :return:
        """
        self.vacuum_on = action[0] > 0.0
        if not self.vacuum_on:
            self.suction_history = self.suction_history * 0.0
        for _, joint in enumerate(self.joints):
            joint.set_drive_target(0)

    def after_simulation_step_ext(self, scene):
        """
        Not used
        """
        raise NotImplementedError

    def before_simulation_step_ext(self, scene):
        """
        Not used
        """
        raise NotImplementedError

    def before_simulation_step(self):
        """Called before each simulation step in one control step."""

        self.grasping = []

        if not self.vacuum_on:
            return

        suction_cups_in_contact, suction_cup_names, suction_contacts, object_contacts = (
            self._get_suction_cup_contact()
        )

        # Add moving average filter when contact is lost, but reset to all '1' immediately when
        # contact is made on all cups
        w = pow(0.5, self._sim_steps)
        self.suction_history = (
            suction_cups_in_contact
            if np.all(suction_cups_in_contact)
            else w * self.suction_history + (1.0 - w) * suction_cups_in_contact
        )
        num_suction_cups_in_contact = np.sum(self.suction_history > 0.5)

        object_contacts = {actor_name: len(set(v)) for actor_name, v in object_contacts.items()}

        min_num_suction_cups_in_contact = (
            1  # allows for picking up tiny objects or objects with irregular geometry
        )
        if num_suction_cups_in_contact >= min_num_suction_cups_in_contact:
            force_per_cup = self.config.force * len(suction_cup_names) / num_suction_cups_in_contact
            for _link_name, contacts in suction_contacts.items():
                for suction_cup_actor, other_actor, position in contacts:
                    if object_contacts[other_actor.name] == num_suction_cups_in_contact:
                        self.grasping.append(other_actor)
                        force = quat2mat(suction_cup_actor.pose.q)
                        force = force_per_cup * force[:, 2]
                        if (
                            other_actor.type == "dynamic"
                        ):  # static objects can not have forces applied to them
                            other_actor.add_force_at_point(-force, position)
                        suction_cup_actor.add_force_at_point(force, position)

        if len(self.grasping) > 1:
            self.grasping = list(set(self.grasping))

    def _get_suction_cup_contact(self):
        """
        Internal function to get info about suction cup contacts.
        :return: suction_cups_in_contact, suction_cup_names, suction_contacts, object_contacts
        """
        suction_cup_prefix = self.config.suction_cup_prefix

        # determine what objects the suction cups are contact with
        contacts = (
            self._contact_during_last_sim_step
            if hasattr(self, "_contact_during_last_sim_step")
            else []
        )

        suction_cup_names = [j.split("_joint")[0] for j in self.config.joint_names]
        suction_contacts = {s: [] for s in suction_cup_names}
        object_contacts = defaultdict(list)
        for contact in contacts:
            impulse = contact.total_impulse
            if np.linalg.norm(impulse) > 1e-05:
                position = contact.position
                if (
                    contact.actor0.name.startswith(suction_cup_prefix)
                    and not contact.actor1.name == self.config.vacuum_link_name
                    and not contact.actor1.name.startswith(suction_cup_prefix)
                ):
                    suction_contacts[contact.actor0.name].append(
                        (contact.actor0, contact.actor1, position)
                    )
                    object_contacts[contact.actor1.name].append(contact.actor0.name)
                if (
                    contact.actor1.name.startswith(suction_cup_prefix)
                    and not contact.actor0.name == self.config.vacuum_link_name
                    and not contact.actor0.name.startswith(suction_cup_prefix)
                ):
                    suction_contacts[contact.actor1.name].append(
                        (contact.actor1, contact.actor0, position)
                    )
                    object_contacts[contact.actor0.name].append(contact.actor1.name)

        # get which suction cups are in contact, and apply
        suction_cups_in_contact = np.array(
            [len(contacts) > 0 for link_name, contacts in suction_contacts.items()], dtype=bool
        )
        return suction_cups_in_contact, suction_cup_names, suction_contacts, object_contacts

    def get_suction_cup_contact(self):
        """
        Returns whether suctions cups are in contact with any object.
        :return:
        """
        suction_cups_in_contact, _, _, _ = self._get_suction_cup_contact()
        return suction_cups_in_contact

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        raise NotImplementedError


@dataclass
class VacuumControllerConfig(ControllerConfig):
    controller_cls = VacuumController
    stiffness: float = 200.0
    damping: float = 5.00
    force_limit: float = 200.0
    friction: float = 0.00
    force: float = 20.0
    vacuum_link_name: str = "vacuum_base_link"
    suction_cup_prefix: str = "suction_cup"
