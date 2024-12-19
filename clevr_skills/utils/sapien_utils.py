# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List, Union

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.sapien_utils import check_actor_static as ms2_check_actor_static
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse


def get_actor_by_name(actors, names: Union[List[str], str]):
    """
    Returns actors by name.
    :param actors: List of potential actors.
    :param names: Names of actors.
    :return: The selected actors. Length will be len(names).
    Each entry will be one of the actors, or None if not found.
    """
    assert isinstance(actors, (list, tuple))
    # Actors can be joint and link
    if isinstance(names, str):
        names = [names]
        single = True
    else:
        single = False
    ret = [None for _ in names]
    for actor in actors:
        if actor.get_name() in names:
            ret[names.index(actor.get_name())] = actor
    return ret[0] if single else ret


def check_actor_static(actor: sapien.Actor, lin_thresh=1e-3, ang_thresh=1e-2):
    """
    :param actor: The actor.
    :param lin_thresh: Linear velocity threshold (meters / second).
    :param ang_thresh: Angular velocity threshold (radians / second).
    :return: True if actor is static relative to the threshold.
    """
    return isinstance(actor, sapien.ActorStatic) or ms2_check_actor_static(
        actor, lin_thresh=lin_thresh, ang_thresh=ang_thresh
    )


def is_actor_resting_on_actor(
    scene: sapien.Scene,
    top_actor: sapien.Actor,
    bottom_actor: sapien.Actor,
    ground_actor: sapien.Actor = None,
):
    """
    :param scene: used to retrieve contacts and gravity
    :param top_actor: The actor that should be on top (relative to gravity).
    :param bottom_actor: The actor that should be at the bottom (relative to gravity).
    :param ground_actor: if not None, it is used to determine if the top_actor
    is touching the ground.
    :return: True if the top_actor is resting on top of bottom_actor.
    """
    contacts = scene.get_contacts()
    gravity = scene.get_config().gravity
    timestep = scene.get_timestep()

    # Compute impulse between ground and actors
    actors_impulse = get_pairwise_contact_impulse(contacts, bottom_actor, top_actor)
    # Compute the expected impulse that the top actor should receive
    expected_impulse = gravity * top_actor.get_mass() * timestep

    if not ground_actor is None:
        ground_impulse = get_pairwise_contact_impulse(contacts, ground_actor, top_actor)
        top_actor_touching_ground = np.linalg.norm(ground_impulse) > 1e-5
    else:
        top_actor_touching_ground = False

    # Compute whether top actor is resting on the bottom actor, based on impulse
    eis = np.linalg.norm(expected_impulse)
    actors_impulse /= eis
    expected_impulse /= eis
    top_actor_resting_on_bottom_actor = np.dot(expected_impulse, actors_impulse) > 0.5

    return top_actor_resting_on_bottom_actor and not top_actor_touching_ground


class ContactPoint:
    def __init__(self, p: sapien.ContactPoint):
        """
        :param p: Sapien contact point.
        """
        self.position = np.copy(p.position)
        self.impulse = np.copy(p.impulse)
        self.normal = np.copy(p.normal)
        self.separation = p.separation


class Contact:
    def __init__(self, c: sapien.Contact, filter: bool):
        """
        :param c: Sapien contact.
        :param filter: if True, a zero-sum impulse will result in self.total_impulse and
        self.position being None
        These Contact instances will then be filtered out by copy_contacts.
        """
        self.actor0 = c.actor0
        self.actor1 = c.actor1
        self.starts = c.starts
        self.persists = c.persists
        self.ends = c.ends
        self.total_impulse = np.sum([p.impulse for p in c.points], axis=0)
        self.position = None
        use_mean = True
        if np.any(self.total_impulse):
            imp = np.sum([np.linalg.norm(p.impulse) for p in c.points])
            if imp > 0:
                self.position = np.sum(
                    [p.position * np.linalg.norm(p.impulse) for p in c.points] / imp, axis=0
                )
                use_mean = False
            elif filter:
                self.total_impulse = None
                return
        elif filter:
            self.total_impulse = None
            return
        if use_mean:
            self.position = np.mean([p.position for p in c.points], axis=0)


def copy_contacts(contacts: List[sapien.Contact], filter: bool = True):
    """
    Returns a simplified Python copy of the contact.
    We are not sure if keeping sapien C++ objects around longer than one simulation step is valid.
    This might lead to segfault. But this function (copy_contacts) is quite expensive and could
    benefit from an implementation in C++.
    :param contacts: List of sapien contacts.
    :param filter: discard contacts with zero impulse.
    :return: Python copy of contact.
    """
    copied_contacts = [Contact(c, filter) for c in contacts]
    if filter:
        copied_contacts = [c for c in copied_contacts if not c.total_impulse is None]
    return copied_contacts
