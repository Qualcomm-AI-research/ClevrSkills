# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from copy import deepcopy
from typing import List

import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill2.utils.trimesh_utils import get_actor_mesh, get_articulation_meshes
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.logger import log


def get_random_non_overlapping_poses_2d_v2(env, actors: List[sapien.Actor], workspace, z):
    """
    :param env: ClevrSkillsEnv
    :param actors: List of actors that need to be placed.
    :param workspace: Available workspace for placement.
    :param z: How high above the surface the actor needs to be placed.
    :return: List of poses.
    """
    # Remember original pose of actors, and place them far outside the scene
    far_away = np.array([1000.0, 1000.0, 1000.0])
    original_poses = []
    for actor in actors:
        original_poses.append(actor.pose)
        actor.set_pose(Pose(far_away))

    random_poses = []
    for actor in actors:
        place_actor_randomly_v2(actor, env, workspace, offset=z, allow_top=False)
        random_poses.append(actor.pose)

    # Restore original poses
    for actor, original_pose in zip(actors, original_poses):
        actor.set_pose(original_pose)

    return random_poses


def get_random_non_overlapping_poses_2d_with_actor_sizes(env, num_poses, workspace, actor_sizes):
    """
    :param env: The ClevrSkillsEnv
    :param num_poses: How many poses to sample.
    :param workspace: available workspace for placement.
    :param actor_sizes: Size of each actor (depth X, width Y, height Z).
    :return: List of poses.
    """

    poses = []
    positions = []

    for p in range(num_poses):
        min_distance = np.linalg.norm(actor_sizes[p][0:2])

        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            position = env._episode_rng.uniform(low=workspace[0], high=workspace[1])
            if len(positions) == 0 or np.all(
                np.linalg.norm(np.array(positions) - position, axis=1) >= min_distance
            ):
                break

        if attempt == MAX_ATTEMPTS:
            log("Warning: object could spawn at location intersecting with other object")
        positions.append(position)

        z = actor_sizes[p][2] / 2  # half the actor height

        position = np.concatenate((position, [z]))
        angle = env._episode_rng.uniform(0.0, 2 * np.pi)
        poses.append(Pose(position, axangle2quat([0.0, 0.0, 1.0], angle)))
    return poses


def get_all_meshes(env, exclude=None, subdivide_to_size=-1.0):
    """
    :param env:The ClevrSkillsEnv
    :param exclude: Actors to exclude, by name. E.g., the "ground".
    :param subdivide_to_size: if subdivide_to_size is > 0, then trimesh.subdivide_to_size()
    is called to add more vertices until the maximum edge length is subdivide_to_size
    :return: meshes of all actors and articulations in the scene, excluding (by name)
    the ones listed in exclude
    """
    if not exclude:
        exclude = ["ground"]

    meshes = []
    for actor in env._scene.get_all_actors():
        if actor.name in exclude:
            continue
        meshes.append(get_actor_mesh(actor, to_world_frame=True))
    for articulation in env._scene.get_all_articulations():
        meshes = meshes + get_articulation_meshes(articulation)

    if subdivide_to_size > 0:
        meshes = [mesh.subdivide_to_size(subdivide_to_size, max_iter=5) for mesh in meshes]

    return meshes


def _check_mesh_intersects_bounds(bounds, meshes: List[trimesh.Trimesh]):
    """
    :param bounds: 3D bounding box
    :param meshes: List of meshes.
    :return: True if any of the vertices of the meshes is inside the bounds.
    """
    for mesh in meshes:
        if np.any(
            np.all(np.logical_and(mesh.vertices >= bounds[0], mesh.vertices <= bounds[1]), axis=1)
        ):
            return True
    return False


def _check_mesh_on_top_of_other(actor: trimesh.Trimesh, meshes: List[trimesh.Trimesh]):
    """
    :param actor: Actor to be tested.
    :param meshes: Meshes to be checked against.
    :return: True if any of the actor vertices lies within the xy-bounds of any of the meshes.
    """
    for mesh in meshes:
        bounds = mesh.bounds
        if np.any(
            np.all(
                np.logical_and(
                    actor.vertices[:, :2] >= bounds[0][:2], actor.vertices[:, :2] <= bounds[1][:2]
                ),
                axis=1,
            )
        ):
            return True
    return False


def get_random_non_intersecting_pose_v2(
    env,
    actor,
    workspace_2d,
    offset_z,
    other_actors=None,
    exclude_actors=None,
    angle_z=None,
    max_attempts=100,
    set_pose=False,
    grow_actor_bounds=0.0,
    allow_top=True,
):
    """
    :param env:The ClevrSkillsEnv
    :param actor: an existing actor that must be placed
    :param workspace_2d: 2D bounding box of available workspace.
    :param other_actors: the other actors to consider during placement. If other_actors is None,
    then all actors in the scene are assumed to be other_actors
    :param exclude_actors: actors to exclude from consideration during placement.
    This will override other_actors
    :param offset_z: Z coordinate of object
    :param angle_z: Rotation about z axis. If this is a list of 2 scalars, an angle will be sampled.
    Otherwise, the value will be used directly
    :param max_attempts: How many times to try to find intersection-free pose
    :param set_pose: when True, the pose of the actor is set to the returned pose
    :param grow_actor_bounds: how much to grow the bounding box of actor
    :param allow_top: allow actors to be initialized on top of other actors
    :return: tuple (random pose, intersects). And the actor will be placed at the pose
    if set_pose is True
    """
    if not exclude_actors:
        exclude_actors = []
    if not angle_z:
        angle_z = [0, 2 * np.pi]

    workspace_actor = None
    if isinstance(workspace_2d, sapien.ActorBase):
        # sample positions from the mesh.
        workspace_actor = workspace_2d
        workspace_samples = get_actor_mesh(workspace_actor).sample(max_attempts)
        workspace_samples[-1, :] = (
            actor.pose.p
        )  # always include the center of the mesh -> best place to put actors that are
        # too large to fit
        workspace_samples[:, 2] = offset_z
    else:
        workspace_2d = np.array(workspace_2d, dtype=np.float32)
        workspace_samples = env._episode_rng.uniform(
            low=workspace_2d[0], high=workspace_2d[1], size=(max_attempts, 2)
        )
        workspace_samples[-1, :] = 0.5 * (
            workspace_2d[0] + workspace_2d[1]
        )  # always include the exact center of the workspace -> best place to put actors
        # that are too large to fit
        workspace_samples = np.concatenate(
            (workspace_samples, offset_z * np.ones((max_attempts, 1))), axis=1
        )
    workspace_samples = workspace_samples.astype(np.float32)

    if other_actors is None:
        other_actors = env._scene.get_all_actors() + env._scene.get_all_articulations()
        if not allow_top:  # remove "ground"
            other_actors = [oa for oa in other_actors if oa.name != "ground"]

    other_actors = [oa for oa in other_actors if not oa in exclude_actors]

    ad: ActorDistance = env._actor_distance
    best_score = -1.0  # 2 points for not intersecting, and up to 1.0 points for points being inside
    # the 2D convex hull SDF
    best_pose = None
    for _attempt_idx in range(max_attempts):
        random_position = workspace_samples[_attempt_idx]
        angle_z = (
            env._episode_rng.uniform(low=angle_z[0], high=angle_z[1])
            if isinstance(angle_z, list)
            else angle_z
        )
        pose = Pose(random_position, axangle2quat([0, 0, 1.0], angle_z))

        inside_fraction = (
            1.0
            if workspace_actor is None
            else env._actor_distance.inside(
                actor, workspace_2d, actor_pose=pose, flat_dim=2, return_fraction=True
            )
        )

        intersects_other_actor = ad.intersects(
            actor,
            other_actors,
            actor_pose=pose,
            min_distance=grow_actor_bounds,
            flat_dim=-1 if allow_top else 2,
        )

        score = (0 if intersects_other_actor else 2) + inside_fraction
        if score > best_score:
            best_score = score
            best_pose = pose

        if not intersects_other_actor and inside_fraction == 1.0:
            break

    if set_pose:  # set pose of actor
        actor.set_pose(best_pose)
    return best_pose, best_score < 2.0


def get_out_of_reach_non_intersecting_pose_v2(
    env,
    actor,
    offset_z,
    robo_dist=0.75,
    other_actors=None,
    exclude_actors=None,
    angle_z=None,
    max_attempts=100,
    set_pose=False,
    grow_actor_bounds=0.0,
    allow_top=True,
):
    """
    :param env: The ClevrSkillsEnv
    :param actor: an already-existing actor
    :param robo_dist: minimum distance from the robot. default 75cm
    :param other_actors: the other actors to consider during placement. If other_actors is None,
    then all actors in the scene are assumed to be other_actors
    :param exclude_actors: actors to exclude from consideration during placement.
    This will override other_actors
    :param offset_z: Z coordinate of object
    :param angle_z: Rotation about z axis. If this is a list of 2 scalars, an angle will be sampled.
    Otherwise, the value will be used directly
    :param max_attempts: How many times to try to find intersection-free pose
    :param set_pose: when True, the pose of the actor is set to the returned pose
    :param grow_actor_bounds: how much to grow the bounding box of actor
    :param allow_top: allow actors to be initialized on top of other actors
    :return: tuple (random pose, intersects). And the actor will be placed at the pose
    if set_pose is True
    """
    if not exclude_actors:
        exclude_actors = []
    if not angle_z:
        angle_z = [np.pi / 2, 3 * np.pi / 2]

    if other_actors is None:
        other_actors = env._scene.get_all_actors() + env._scene.get_all_articulations()
        if not allow_top:  # remove "ground"
            other_actors = [oa for oa in other_actors if oa.name != "ground"]

    other_actors = [oa for oa in other_actors if not oa in exclude_actors]

    ad: ActorDistance = env._actor_distance
    best_score = -1.0  # 2 points for not intersecting, and up to 1.0 points for points being inside
    # the 2D convex hull SDF
    best_pose = None
    for _attempt_idx in range(max_attempts):
        target_distance = env._episode_rng.uniform(low=robo_dist, high=robo_dist + 0.1)
        robot_pos = env._robot_pos
        target_angle = (
            env._episode_rng.uniform(low=angle_z[0], high=angle_z[1])
            if isinstance(angle_z, list)
            else angle_z
        )
        target_pos = np.array(
            [
                robot_pos[0] + np.sin(target_angle) * target_distance,
                robot_pos[1] + np.cos(target_angle) * target_distance,
                offset_z / 2 + 0.001,
            ]
        )

        pose = Pose(target_pos)

        intersects_other_actor = ad.intersects(
            actor,
            other_actors,
            actor_pose=pose,
            min_distance=grow_actor_bounds,
            flat_dim=-1 if allow_top else 2,
        )

        score = 0 if intersects_other_actor else 2
        if score > best_score:
            best_score = score
            best_pose = pose

        if not intersects_other_actor:
            break

    if set_pose:  # set pose of actor
        actor.set_pose(best_pose)
    return best_pose, best_score < 2.0


def get_random_non_intersecting_pose_with_neighbour_v2(
    env,
    actor,
    neighbour,
    direction,
    workspace_2d,
    off_z_actor,
    off_z_neighbour,
    other_actors=None,
    angle_z=None,
    max_attempts=100,
    set_pose=False,
    grow_actor_bounds=0.0,
    allow_top=True,
):
    """
    :param env: ClevrSkillsEnv
    :param actor: The actor to be placed next to neighbor
    :param neighbour: The to-be-neighbor actor.
    :param direction: Direction relation to neighbor ("west", "east", "north", "south")
    :param workspace_2d: 2D bounding box of available workspace.
    :param off_z_actor: Z offset for actor.
    :param off_z_neighbour: Z offset for neighor.
    :param other_actors: Any other actor to consider during placement.
    :param angle_z: Rotation about z axis. If this is a list of 2 scalars, an angle
    will be sampled. Otherwise, the value will be used directly
    :param max_attempts: How many times to try to find intersection-free pose
    :param set_pose: when True, the pose of the actor is set to the returned pose
    :param grow_actor_bounds: how much to grow the bounding box of actor
    :param allow_top: allow actors to be initialized on top of other actors
    :return: tuple (random pose, intersects). And the actor will be placed at the pose
    if set_pose is True
    """
    if not angle_z:
        angle_z = [0, 2 * np.pi]
    if other_actors is None:
        other_actors = (
            env._scene.get_all_actors() + env._scene.get_all_articulations()
            if other_actors is None
            else other_actors
        )
        if not allow_top:  # remove "ground"
            other_actors = [oa for oa in other_actors if oa.name != "ground"]

    neighbour_verts = get_actor_mesh(neighbour).vertices
    neighbour_size = neighbour_verts.max(0) - neighbour_verts.min(0)
    dir_axis = {"west": 1, "east": 1, "north": 0, "south": 0}
    dir_mult = {"west": 1, "east": -1, "north": 1, "south": -1}
    dir_offset = 0.1 + neighbour_size[dir_axis[direction]]

    intersects_other_actor = True
    pose = None
    off_z_actor = np.array([off_z_actor])
    off_z_neighbour = np.array([off_z_neighbour])
    ad: ActorDistance = env._actor_distance
    for _attempt_idx in range(max_attempts):
        random_position = np.concatenate(
            (env._episode_rng.uniform(low=workspace_2d[0], high=workspace_2d[1]), off_z_actor)
        )
        angle_z = (
            env._episode_rng.uniform(low=angle_z[0], high=angle_z[1])
            if isinstance(angle_z, list)
            else angle_z
        )
        pose = Pose(random_position, axangle2quat([0, 0, 1.0], angle_z))

        # neighbour pose
        random_position[2] = off_z_neighbour
        random_position[dir_axis[direction]] += dir_mult[direction] * dir_offset
        if np.any(random_position[0:2] < workspace_2d[0]) or np.any(
            random_position[0:2] > workspace_2d[1]
        ):
            continue
        neighbour_pose = Pose(random_position, axangle2quat([0, 0, 1.0], angle_z))

        intersects_other_actor = ad.intersects(
            actor,
            other_actors,
            actor_pose=pose,
            min_distance=grow_actor_bounds,
            flat_dim=-1 if allow_top else 2,
        )
        intersects_other_neighbour = ad.intersects(
            neighbour,
            other_actors,
            actor_pose=neighbour_pose,
            min_distance=grow_actor_bounds,
            flat_dim=-1 if allow_top else 2,
        )

        if not intersects_other_actor and not intersects_other_neighbour:
            break

    if set_pose:  # set pose of actor
        actor.set_pose(pose)
        neighbour.set_pose(neighbour_pose)
    return pose, neighbour_pose, intersects_other_actor, intersects_other_neighbour


def get_random_non_intersecting_pose(
    env,
    actor,
    workspace_2d,
    offset_z,
    other_actor_meshes,
    angle_z=None,
    max_attempts=100,
    set_pose=False,
    grow_actor_bounds=0.0,
    allow_top=True,
):
    """
    :param env: ClevrSkillsEnv
    :param actor: an already-existing actor
    :param workspace_2d: 2D bounding box of available workspace.
    :param offset_z: Z offset for actor.
    :param other_actor_meshes: meshes of all actors that should not be intersected
    :param angle_z: Rotation about z axis. If this is a list of 2 scalars,
    an angle will be sampled. Otherwise, the value will be used directly
    :param max_attempts: How many times to try to find intersection-free pose
    :param set_pose: when True, the pose of the actor is set to the returned pose
    :param grow_actor_bounds: how much to grow the bounding box of actor
    :param allow_top: allow actors to be initialized on top of other actors
    :return: tuple (random pose, intersects). And the actor will be placed at the pose
    if set_pose is True
    """
    if not angle_z:
        angle_z = [0, 2 * np.pi]

    intersects_other_actor = True
    pose = None
    offset_z = np.array([offset_z])
    for _attempt_idx in range(max_attempts):
        random_position = np.concatenate(
            (env._episode_rng.uniform(low=workspace_2d[0], high=workspace_2d[1]), offset_z)
        )
        angle_z = (
            env._episode_rng.uniform(low=angle_z[0], high=angle_z[1])
            if isinstance(angle_z, list)
            else angle_z
        )
        pose = Pose(random_position, axangle2quat([0, 0, 1.0], angle_z))

        actor_mesh: trimesh.Trimesh = get_actor_mesh(actor, to_world_frame=False)
        actor_mesh.apply_transform(pose.to_transformation_matrix())
        actor_bounds = np.copy(actor_mesh.bounds)
        if grow_actor_bounds:
            actor_bounds[0, 0:2] -= grow_actor_bounds
            actor_bounds[1, 0:2] += grow_actor_bounds

        if not allow_top:
            if _check_mesh_on_top_of_other(actor_mesh, other_actor_meshes):
                continue

        intersects_other_actor = _check_mesh_intersects_bounds(actor_bounds, other_actor_meshes)
        if not intersects_other_actor:
            break

    if set_pose:  # set pose of actor
        actor.set_pose(pose)
    return pose, intersects_other_actor


def place_actor_randomly_v2(
    actor, env, workspace, offset=0.0, allow_top=False, grow_actor_bounds: float = 0.0
):
    """
    :param actor:
    :param env:
    :param workspace:
    :param offset:
    :param allow_top:
    :param grow_actor_bounds:
    :return:
    """
    bounds = env._actor_distance.get_bounds(actor)
    actor_size = bounds[1, :] - bounds[0, :]
    offset_z = actor_size[2] / 2 + offset
    pose, intersect = get_random_non_intersecting_pose_v2(
        env,
        actor,
        workspace,
        offset_z,
        angle_z=[0, 2 * np.pi],
        grow_actor_bounds=grow_actor_bounds,
        max_attempts=100,
        allow_top=allow_top,
    )

    if intersect:
        log(
            f"Warning _load_random_actors_v2: could not find intersection-free pose "
            f"for actor {actor.name}"
        )

    actor.set_pose(pose)


def place_actor_out_of_reach_v2(actor, env, robo_dist, offset=0.0, grow_actor_bounds: float = 0.0):
    """
    :param actor: actor to be placed out of reach for the robot.
    :param env: ClevrSkillsEnv
    :param robo_dist: Distance to robot (meters)
    :param offset: Vertical offset above the ground.
    :param grow_actor_bounds: How much to grow actor bound for intersection tests.
    :return: None; Actor is placed at the new pose.
    """
    bounds = env._actor_distance.get_bounds(actor)
    actor_size = bounds[1, :] - bounds[0, :]
    offset_z = actor_size[2] / 2 + offset
    pose, intersect = get_out_of_reach_non_intersecting_pose_v2(
        env,
        actor,
        offset_z,
        robo_dist=robo_dist,
        angle_z=[np.pi / 2, 3 * np.pi / 2],
        grow_actor_bounds=grow_actor_bounds,
        max_attempts=100,
        allow_top=False,
    )

    if intersect:
        log(
            f"Warning _load_random_actors_v2: could not find intersection-free pose "
            f"for actor {actor.name}"
        )

    actor.set_pose(pose)


def place_actor_and_neighbour_randomly_v2(
    actor, neighbour, direction, env, workspace, offset=0.0, allow_top=False
):
    """
    :param env: ClevrSkillsEnv
    :param actor: The actor to be placed next to neighbor
    :param neighbour: The to-be-neighbor actor.
    :param direction: Direction relation to neighbor ("west", "east", "north", "south")
    :param workspace: 2D bounding box of available workspace.
    :param offset: Z offset relative to ground.
    :param allow_top: allow actors to be initialized on top of other actors
    :return: None; actor and neighbor are placed.
    """
    act_bounds = env._actor_distance.get_bounds(actor)
    actor_size = act_bounds[1, :] - act_bounds[0, :]
    off_z_actor = actor_size[2] / 2 + offset

    nbr_bounds = env._actor_distance.get_bounds(neighbour)
    nbr_size = nbr_bounds[1, :] - nbr_bounds[0, :]
    off_z_nbr = nbr_size[2] / 2 + offset

    act_pose, nbr_pose, intersect, nbr_intersect = (
        get_random_non_intersecting_pose_with_neighbour_v2(
            env,
            actor,
            neighbour,
            direction,
            workspace,
            off_z_actor,
            off_z_nbr,
            angle_z=[0, 2 * np.pi],
            max_attempts=100,
            allow_top=allow_top,
        )
    )

    if intersect or nbr_intersect:
        log(
            f"Warning place_actor_and_neighbour_randomly_v2: could not find intersection-free pose "
            f"for actor {actor.name} or {neighbour.name}"
        )

    actor.set_pose(act_pose)
    neighbour.set_pose(nbr_pose)


def place_actor_on_area(actor, area):
    """
    :param actor: An actor
    :param area: Another actor that represents the area that the actor should be placed on.
    :return: None
    """
    area_verts = get_actor_mesh(area).vertices
    area_size = area_verts.max(0) - area_verts.min(0)

    ac_verts = get_actor_mesh(actor).vertices
    actor_size = ac_verts.max(0) - ac_verts.min(0)

    offset_z = actor_size[2] / 2 + area_size[2]

    pose = deepcopy(area.pose)
    p = pose.p
    p[2] += offset_z
    pose.set_p(p)

    actor.set_pose(pose)
