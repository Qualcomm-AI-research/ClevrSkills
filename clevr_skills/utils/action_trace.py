# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""
Some utilities for storing the action trace of a solver
"""

import sapien.core as sapien


def at_get_actor(actor: sapien.Actor):
    """
    Used to store an actor in the actor trace in a uniquely identifiable manner.
    :param actor: Any Sapien actor in the scene
    :return: A tuple (actor.name, actor.id).
    """
    if isinstance(actor, sapien.Articulation):
        articulation: sapien.Articulation = actor
        return (articulation.name, articulation.get_links()[0].id)
    return None if actor is None else (actor.name, actor.id)


def at_get_pose(pose: sapien.Pose):
    """
    :param pose:
    :return: pose, in a form that can be stored in the action trace.
    """
    return None if pose is None else (list(pose.p), list(pose.q))


def at_get_pos(pos):
    """
    :param pos: Position (np.ndarray or list)
    :return: pos, in a form that can be stored in the action trace.
    """
    return None if pos is None else list(pos)
