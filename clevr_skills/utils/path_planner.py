# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
Some functions to make it easy to use mplib planner

Had to work around some issues for things that we did not get to work properly:
- Encoding (reduced) joint limits into the URDF
- Encoding the attached box or mesh into the URDF
"""

import os
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

import mplib
import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill2.utils.sapien_utils import vectorize_pose
from mplib.pymp.collision_detection import fcl
from transforms3d.euler import quat2euler

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.controller import align_quat_pose
from clevr_skills.utils.logger import log
from clevr_skills.utils.temp_dir import get_temp_dir
from clevr_skills.utils.xml import write_xml_clean_and_pretty


def resolve_relative_path_references(
    xml: minidom.Node, xml_path: str, resource_path: str, attributes: List[str] = None
):
    """
    Resolves all relative paths found in the xml such that it can be written in a different path
    that the resources.
    This is used to re-write the agent (robot) XML with modifications, outside of the source tree.
    :param xml: The XML datastructure.
    :param xml_path: Where the XML will be written. Resource references will be made relative to
    the parent of this path.
    :param resource_path: where resources are located.
    :param attributes: What attributes to inspect. By default, only "filename" is inspected.
    """
    if not attributes:
        attributes = ["filename"]

    for node in xml.childNodes:
        if isinstance(node, minidom.Element):
            for attribute in attributes:
                if node.hasAttribute(attribute):
                    path = node.attributes[attribute].value
                    abs_path = os.path.join(resource_path, path)
                    if os.path.exists(abs_path):
                        resolved_rel_path = os.path.relpath(abs_path, os.path.split(xml_path)[0])
                        node.setAttribute(attribute, resolved_rel_path)
            resolve_relative_path_references(node, xml_path, resource_path, attributes=attributes)


def dilate(points: np.ndarray, amount: float, center: np.ndarray = None) -> np.ndarray:
    """
    Grows a point cloud by an amount, relative to the center.
    :param points: The point cloud
    :param amount: Scaling factor.
    :param center: If not specified, the median of the cloud is used
    :return: Dilated point cloud
    """
    if center is None:
        center = np.median(points, axis=0)
    vec = points - center
    norm = np.expand_dims(np.linalg.norm(vec, axis=1), axis=1)
    points = vec * (norm + amount) / norm + center
    return points


def get_mplib_planner_with_grasped_actor(
    env: ClevrSkillsEnv,
    joint_limits: Dict[str, Tuple[float, float]] = None,
    mesh: bool = True,
    extend_bounds=0.01,
) -> mplib.Planner:
    """
    :param env: The ClevrSkillsEnv
    :param joint_limits: dict from joint_name to tuple [lower_limit, upper_limit]. User this to
    narrow the limits of the robot. E.g., to avoid 720 degree turns of the base joint.
    It will not allow you to enlarge the joint limits!
    :param mesh: whether to represent the grasped actor(s) as convex mesh or as box
    :param extend_bounds: How much to extend the bounds of the grasped object during path planning (in meters).
    :return: MPLib planner.
    """
    if not joint_limits:
        joint_limits = {}

    agent = env.agent
    actor_distance = env._actor_distance
    grasped_actors = agent.get_grasped_actors()

    attach_link = agent.ee_link.name if len(grasped_actors) else None
    if attach_link is None:
        return get_mplib_planner(agent, joint_limits=joint_limits)
    if mesh:
        grasped_actor_point_cloud = actor_distance.get_point_cloud(
            grasped_actors, reference_pose=agent.ee_link.pose
        )
        if extend_bounds > 0:
            grasped_actor_point_cloud = dilate(grasped_actor_point_cloud, extend_bounds)
        grasped_actor_mesh = trimesh.convex.convex_hull(grasped_actor_point_cloud)
        log(f"grasped_actor_mesh water tight: {grasped_actor_mesh.is_watertight}")
        grasped_actor_pose = sapien.Pose()
        return get_mplib_planner(
            agent,
            joint_limits=joint_limits,
            attach_mesh_link=attach_link,
            attach_mesh_pose=grasped_actor_pose,
            attach_mesh=grasped_actor_mesh,
        )

    bounds = actor_distance.get_bounds(
        agent.get_grasped_actors(), reference_pose=agent.ee_link.pose
    )
    bounds[0, :] -= extend_bounds
    bounds[1, :] += extend_bounds
    box_size = bounds[1] - bounds[0]
    box_pose = sapien.Pose(np.mean(bounds, axis=0))
    attach_link = attach_link if np.product(box_size) > 0 else None
    return get_mplib_planner(
        agent,
        joint_limits=joint_limits,
        attach_box_link=attach_link,
        attach_box_pose=box_pose,
        attach_box_size=box_size,
    )


def modify_urdf_joint_limits(robot_xml, joint_limits: Dict[str, Tuple[float, float]]) -> None:
    """
    Modifies the joints limits of robot_xml in-place.
    :param robot_xml: loaded from URDF
    :param joint_limits: dict from joint_name to tuple [lower_limit, upper_limit]. User this to
    narrow the limits of the robot. E.g., to avoid 720 degree turns of the base joint.
    It will not allow you to enlarge the joint limits!
    :return: None
    """
    joints_xml = robot_xml.getElementsByTagName("joint")
    for joint_xml in joints_xml:
        if joint_xml.attributes["name"].value in joint_limits:
            lower, upper = joint_limits[joint_xml.attributes["name"].value]
            limit_xml = joint_xml.getElementsByTagName("limit")[0]
            if lower > float(limit_xml.attributes["lower"].value):
                limit_xml.attributes["lower"] = str(lower)
            if upper < float(limit_xml.attributes["upper"].value):
                limit_xml.attributes["upper"] = str(upper)


def attach_urdf(
    urdf_path,
    robot_xml,
    attach_mesh_link: str = None,
    attach_mesh_pose=None,
    attach_mesh=None,
    attach_box_link: str = None,
    attach_box_pose=None,
    attach_box_size=None,
) -> str:
    """
    Modifies the robot_xml in-place, attached a mesh and/or box.
    :param urdf_path: Used to determine a place to store the mesh
    :param robot_xml: loaded from URDF
    :param attach_mesh_link: Optional: name of the link to attach an additional mesh to.
    :param attach_mesh_pose: Optional: pose of mesh relative to link
    :param attach_mesh: Optional:path of mesh
    :param attach_box_link: Optional: name of the link to attach an additional box to.
    :param attach_box_pose: Optional: pose of box relative to link
    :param attach_box_size: Optional: size of box
    :return: path to temp location of mesh
    """
    attach_mesh_actual_path = None
    if attach_mesh_link:
        attach_mesh_path = os.path.splitext(urdf_path)[0] + "_attached_mesh.stl"
        # Note: mplib assumes that Sapien has already loaded the robot and made the parts convex!
        trimesh.exchange.export.export_mesh(attach_mesh, attach_mesh_path)
        attach_mesh_actual_path = attach_mesh_path + ".convex.stl"
        trimesh.exchange.export.export_mesh(attach_mesh, attach_mesh_actual_path)

    links_xml = robot_xml.getElementsByTagName("link")
    for link_xml in links_xml:
        if link_xml.attributes["name"].value == attach_box_link:
            for keyword in ["visual", "collision"]:
                origin_xml = robot_xml.createElement("origin")
                origin_xml.setAttribute(
                    "xyz", f"{attach_box_pose.p[0]} {attach_box_pose.p[1]} {attach_box_pose.p[2]}"
                )

                euler = quat2euler(
                    attach_box_pose.q
                )  # default Euler angle axis order matches ROS rpy convention.
                origin_xml.setAttribute("rpy", f"{euler[0]} {euler[1]} {euler[2]}")
                box_xml = robot_xml.createElement("box")
                box_xml.setAttribute(
                    "size", f"{attach_box_size[0]} {attach_box_size[1]} {attach_box_size[2]}"
                )
                geometry_xml = robot_xml.createElement("geometry")
                geometry_xml.appendChild(box_xml)
                collision_xml = robot_xml.createElement(keyword)
                collision_xml.appendChild(origin_xml)
                collision_xml.appendChild(geometry_xml)
                link_xml.appendChild(collision_xml)
        if link_xml.attributes["name"].value == attach_mesh_link:
            for keyword in ["visual", "collision"]:
                origin_xml = robot_xml.createElement("origin")

                rpy = quat2euler(
                    attach_mesh_pose.q
                )  # default Euler angle axis order matches ROS rpy convention.
                origin_xml.setAttribute(
                    "xyz",
                    f"{attach_mesh_pose.p[0]} {attach_mesh_pose.p[1]} {attach_mesh_pose.p[2]}",
                )
                origin_xml.setAttribute("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
                mesh_xml = robot_xml.createElement("mesh")
                mesh_xml.setAttribute("filename", os.path.split(attach_mesh_path)[1])
                geometry_xml = robot_xml.createElement("geometry")
                geometry_xml.appendChild(mesh_xml)
                collision_xml = robot_xml.createElement(keyword)
                collision_xml.appendChild(origin_xml)
                collision_xml.appendChild(geometry_xml)
                link_xml.appendChild(collision_xml)
    return attach_mesh_actual_path


def get_mplib_planner(
    agent,
    joint_limits: Dict[str, Tuple[float, float]] = None,
    attach_mesh_link: str = None,
    attach_mesh_pose=None,
    attach_mesh=None,
    attach_box_link: str = None,
    attach_box_pose=None,
    attach_box_size=None,
) -> mplib.Planner:
    """
    :param agent: ManiSkill2 agent
    :param joint_limits: dict from joint_name to tuple [lower_limit, upper_limit]. User this to
    narrow the limits of the robot. E.g., to avoid 720 degree turns of the base joint.
    It will not allow you to enlarge the joint limits!

    :param attach_mesh_link: Optional: name of the link to attach an additional mesh to.
    :param attach_mesh_pose: Optional: pose of mesh relative to link
    :param attach_mesh: Optional: path of mesh
    :param attach_box_link: Optional: name of the link to attach an additional box to.
    :param attach_box_pose: Optional: pose of box relative to link
    :param attach_box_size: Optional: size of box
    :return: mplib planner for the current agent.
    This assumes that an srdf is available.
    :return: MPLib planner.
    """
    if not joint_limits:
        joint_limits = {}

    urdf_path = agent.urdf_path
    srdf_path = os.path.splitext(urdf_path)[0] + ".srdf"

    temp_path = get_temp_dir(agent.scene)
    if (
        len(joint_limits) or attach_box_link or attach_mesh_link
    ):  # modify the URDF (xml) to override the joint limits
        output_urdf_path = os.path.join(
            temp_path,
            os.path.splitext(os.path.split(urdf_path)[1])[0] + "_joint_limits_modified.urdf",
        )
        robot_xml = minidom.parse(urdf_path)
        if len(joint_limits):
            modify_urdf_joint_limits(robot_xml, joint_limits)
        if attach_box_link or attach_mesh_link:
            _ = attach_urdf(
                output_urdf_path,
                robot_xml,
                attach_mesh_link=attach_mesh_link,
                attach_mesh_pose=attach_mesh_pose,
                attach_mesh=attach_mesh,
                attach_box_link=attach_box_link,
                attach_box_pose=attach_box_pose,
                attach_box_size=attach_box_size,
            )

        # write XML
        resolve_relative_path_references(robot_xml, output_urdf_path, os.path.split(urdf_path)[0])
        write_xml_clean_and_pretty(robot_xml, output_urdf_path)
        urdf_path = output_urdf_path

    link_names = [link.get_name() for link in agent.robot.get_links()]
    joint_names = [joint.get_name() for joint in agent.robot.get_active_joints()]

    controller = agent.controller.controllers["arm"]
    num_joints = len(controller.joint_indices)
    tcp_link_name = controller.ee_link.name

    fcl_floor_object = fcl.FCLObject(
        "floor",
        mplib.Pose([0, 0, -0.0251], [1, 0, 0, 0]),
        [fcl.CollisionObject(fcl.Box([2.0, 2.0, 0.05]))],
        [mplib.Pose()],
    )
    planner = mplib.Planner(
        urdf_path,
        srdf=srdf_path,
        move_group=tcp_link_name,
        user_link_names=link_names,
        user_joint_names=joint_names,
        joint_vel_limits=2 * np.ones(num_joints),
        joint_acc_limits=2 * np.ones(num_joints),
        objects=[fcl_floor_object],
    )

    planner.set_base_pose(mplib.Pose(agent.robot.pose.p, agent.robot.pose.q))

    return planner


def get_collisions(planner: mplib.Planner, qpos_path, detailed=False, point_cloud_radius=None):
    """
    Used to check for collisions along a path generated by the planner.
    :param planner: The MPLib planner.
    :param qpos_path: np.ndarray path_length x qpos
    :param detailed: If False, an array of booleans is returned (True means a collision
    occurs for the qpos).
    If True, a list of lists of collision details is returned.
    :param point_cloud_radius:
    :return: returns array of bool, or list of collisions, for each qpos
    """
    if not point_cloud_radius is None:
        set_planner_point_cloud(planner, point_cloud=None, point_radius=point_cloud_radius)

    result = [planner.check_for_env_collision(state=qpos) for qpos in qpos_path]
    if detailed:
        return [None if len(c) == 0 else c for c in result]
    return np.array([len(c) for c in result], dtype=bool)


def get_collision_segments(collisions: np.ndarray, gap: int = 5) -> List[Tuple[int, int]]:
    """
    Internal function.
    :param collisions: 1D boolean array
    :param gap: how many frames can a gap of "no collision" last before getting fused into
    the collisions around it.
    :return: List of (start, end_exclusive) segments where collisions occur
    """
    collision_segments = []
    i = 0
    while i < len(collisions):
        if collisions[i]:
            for j in range(i + 1, len(collisions) + 1):
                if not np.any(collisions[j : j + gap + 1]):
                    collision_segments.append((i, j))
                    i = j
                    break
        else:
            i += 1
    return collision_segments


def set_planner_point_cloud(
    planner: mplib.Planner,
    point_cloud: np.ndarray = None,
    octree_resolution=None,
    point_radius=None,
    augment_pcd_diagonal=False,
):
    """
    Sets the point cloud in the planner, sets the octree resolution,
    remembers the point cloud for subsequent calls.
    If point_cloud, octree_resolution, point_radius are None, the value from the previous call
    will be re-used.
    :param planner: MPLib planner
    :param point_cloud: The point cloud (N x 3 numpy array).
    :param octree_resolution: Inside the mplib documentation, this would be called radius,
    but if you dig into the internals of mplib, it controls the resolution of the octree.
    :param point_radius: If point_radius > 0, duplicates of the point cloud at certain offsets
    will be added. The offsets are 6 axis aligned (+- XYZ) offsets.
    If augment_pcd_diagonal is True, 8 more diagonal offsets (one for each quadrant) will be added.
    :param augment_pcd_diagonal: whether to augment the point with points in the
    diagonal directions.
    """
    point_cloud = (
        planner.retained_point_cloud
        if point_cloud is None
        else np.ascontiguousarray(point_cloud, dtype=np.float32)
    )

    octree_resolution = (
        planner.retained_octree_resolution if octree_resolution is None else octree_resolution
    )
    point_radius = planner.retained_point_radius if point_radius is None else point_radius
    planner.retained_point_cloud = point_cloud
    planner.retained_octree_resolution = octree_resolution
    planner.retained_point_radius = point_radius

    if point_radius > 0:
        point_cloud_augmented = []
        # axis-aligned:
        for axis in range(3):
            for direction in [-1, 1]:
                offset = np.zeros(3, dtype=np.float32)
                offset[axis] = direction * point_radius
                offset = np.zeros(3, dtype=np.float32)
                point_cloud_augmented.append(point_cloud + offset)
        if augment_pcd_diagonal:  # diagonal
            for a in range(8):
                offset = (
                    np.array([-1 if a & bit else 1 for bit in [1, 2, 4]], dtype=np.float32)
                    * point_radius
                )
                point_cloud_augmented.append(point_cloud + offset)
        point_cloud = np.ascontiguousarray(
            np.concatenate(point_cloud_augmented, axis=0), dtype=np.float32
        )

    planner.update_point_cloud(point_cloud, resolution=octree_resolution)


def solve_ik(
    planner: mplib.Planner, controller, robot: sapien.Articulation, target_pose: sapien.Pose
) -> np.ndarray:
    """
    Employs various libraries to solve IK (inverse kinematics).
    :param planner: MPLib planner. Used for IK.
    :param controller: ManiSkill controller. Use for IK, joint indices.
    :param robot: The Sapien robot articulation.
    :param target_pose: The target pose of the end effector.
    :return: Pose of robot that gets end-effector into designed pose.
    """
    joint_mask = np.ones(len(robot.get_active_joints()), dtype=bool)
    joint_mask[controller.joint_indices] = False
    qpos_padding = np.zeros(int(np.sum(joint_mask)), dtype=np.float32)
    target_pose = robot.pose.inv().transform(target_pose)
    current_robot_qpos = robot.get_qpos()

    # Preferred: use the Maniskill2 controller to solve for IK
    # Generally results in solutions that are close to the current qpos.
    ik_qpos_solutions = None
    try:
        ik_result = controller.compute_ik(target_pose)  # Use the IK of the controller
        ik_qpos_solutions = np.array([np.concatenate([ik_result, qpos_padding])])
        collisions = planner.check_for_env_collision(state=ik_qpos_solutions[0])
        if len(collisions):  # the controller IK solved for the pose, but it causes an env collision
            ik_qpos_solutions = None
    except Exception as ex:
        # This might happen if controller is None, or controller does not have compute_ik()
        # Then below an alternative solution will be tried
        pass

    if ik_qpos_solutions is None:  # ManiSkill2 controller IK failed, use mplib IK
        # Preferred: solve IK, without trying random starting positions
        ik_result = planner.IK(
            mplib.Pose(target_pose.p, target_pose.q),
            current_robot_qpos,
            mask=joint_mask,
            n_init_qpos=1,
        )
        if ik_result[0] != "Success":
            # Last resort: Solve IK with many random starting positions
            # When n_init_qpos > 1, random initial positions will be sampled
            ik_result = planner.IK(
                vectorize_pose(target_pose), current_robot_qpos, mask=joint_mask, n_init_qpos=100
            )
            if ik_result[0] != "Success":
                raise RuntimeError(
                    f"Path planning IK failed relative target pose {target_pose}. "
                    f"Distance of target to base {np.linalg.norm(target_pose.p[0:2])} "
                    f"(might also be collision related)"
                )
        # Sort the IK solutions for the first joint being closest to the current position
        # (try to avoid "twisted" solutions)
        ik_qpos_solutions = np.array(ik_result[1])  # each row is a solution
        ik_qpos_solutions = ik_qpos_solutions[
            np.linalg.norm(ik_qpos_solutions - current_robot_qpos, axis=1).argsort()
        ]

    # Sometimes the planner IK comes up with "twisted" solutions.
    # Simplify the solution for joints that allow > 2pi rotation.
    for joint_idx in controller.joint_indices:
        if robot.get_active_joints()[joint_idx].type == "revolute":
            limits = robot.get_qlimits()[joint_idx]
            if limits[1] - limits[0] > 2 * np.pi:
                for mod in [-2 * np.pi, 2 * np.pi]:
                    alternate_qpos = ik_qpos_solutions[:, joint_idx] + mod
                    mask = np.logical_and(
                        np.abs(alternate_qpos) < np.abs(ik_qpos_solutions[:, joint_idx]),
                        np.logical_and(alternate_qpos >= limits[0], alternate_qpos <= limits[1]),
                    )

                    if np.any(mask):
                        log(
                            f"simplifying IK solution for joint {joint_idx} from "
                            f"{ik_qpos_solutions[mask, joint_idx]} "
                            f"to {ik_qpos_solutions[mask, joint_idx]+mod}"
                        )
                        ik_qpos_solutions[mask, joint_idx] += mod

    return np.ascontiguousarray(ik_qpos_solutions, dtype=np.float32)


def plan_path(
    planner: mplib.Planner,
    env: ClevrSkillsEnv,
    target_pose: sapien.Pose,
    check_collisison: bool = True,
    replan_collisions: bool = True,
    simplify_path: bool = True,
    octree_resolution: float = 0.02,
    point_cloud_radius: float = 0.02,
    return_poses=True,
):
    """
    :param planner: MPLib planner.
    :param env: ClevrSkillsEnv
    :param target_pose: Target pose for the end effector
    :param check_collisison: Whether to explicitly check that the planned path is collision free.
    mplib can introduce collisions due to interpolation, simplification, etc.
    :param replan_collisions: if True, if there are any collisions along the planned path,
    the planner will be called again for those segments. Mostly makes sense when
    no_simplification=True
    :param simplify_path: if True, a heuristic will try to simplify the path returned by
    the planner. Paths are often overly complex, so simplification is nice.
    However, often the simplification causes collisions. So only use this flag when
    replan_collisions is set to True.
    :param octree_resolution:
    :param point_cloud_radius: resolution of the OcTree used to represent the point cloud.
    :param return_poses: also return 6DOF EE poses.
    :return: Tuple ([qpos], [sapien.Pose], collisions). I.e., the trajectory in both qpos and pose.
    collisions will be None if
    When the robot has redundant degrees of freedom, using qpos is the only way to accurately
    reproduce the path.
    Will throw exception when path planning fails!
    :param return_poses: If True, end-effector poses are returned instead of joint angles.
    """
    # Set point cloud in the planner
    point_cloud = env.get_point_cloud_for_planning(num_floor_points=0)
    set_planner_point_cloud(
        planner, point_cloud, octree_resolution=octree_resolution, point_radius=point_cloud_radius
    )

    controller = env.agent.controller.controllers["arm"]
    robot = env.agent.robot
    pmodel = controller.pmodel

    joint_mask = np.ones(len(robot.get_active_joints()), dtype=bool)
    joint_mask[controller.joint_indices] = False
    qpos_padding = np.zeros(int(np.sum(joint_mask)), dtype=np.float32)

    # Solve inverse kinematics for the target pose
    try:
        ik_qpos_solutions = solve_ik(planner, controller, robot, target_pose)
    except Exception as ex:
        return None, None, str(ex)

    time_step = 1.0 / env.control_freq
    qpos_path, collisions = plan_path_qpos(
        planner,
        time_step,
        robot.get_qpos(),
        ik_qpos_solutions,
        check_collisison=check_collisison,
        replan_collisions=replan_collisions,
        simplify_path=simplify_path,
    )
    if qpos_path is None:  # error detected
        return None, None, collisions

    # convert qpos to poses, if requested by caller
    ee_poses = None
    if return_poses:
        ee_poses = []
        prev_pose = env.agent.ee_link.pose
        for target_qpos in qpos_path:
            pmodel.compute_forward_kinematics(np.concatenate([target_qpos, qpos_padding]))
            target_ee_pose = pmodel.get_link_pose(controller.ee_link_idx)
            target_ee_pose = robot.pose.transform(target_ee_pose)

            target_ee_pose = align_quat_pose(target_ee_pose, prev_pose)
            prev_pose = target_ee_pose

            ee_poses.append(target_ee_pose)
    return qpos_path, ee_poses, collisions


def plan_path_qpos(
    planner: mplib.Planner,
    time_step: float,
    initial_qpos: np.ndarray,
    target_qpos: np.ndarray,
    check_collisison: bool = True,
    replan_collisions: bool = True,
    simplify_path: bool = True,
):
    """
    :param planner: the mplib planner. This assumes that the point cloud has already been set!
    :param time_step: execution time between two steps on the planned path
    :param initial_qpos:
    :param target_qpos:
    :param check_collisison:
    :param replan_collisions:
    :param simplify_path:
    :return: qpos_path, collisions
    """
    SUCCESS = "Success"
    ik_qpos_solution = None

    # sanitize initial_qpos and target_qpos
    if not isinstance(initial_qpos, np.ndarray):
        initial_qpos = np.ndarray(initial_qpos, dtype=np.float32)
    if not isinstance(target_qpos, np.ndarray):
        target_qpos = np.ndarray(target_qpos, dtype=np.float32)
    ik_qpos_solutions = target_qpos if len(target_qpos.shape) == 2 else [target_qpos]

    # Plan a path in qpos space
    planning_time = 2.0
    for ik_qpos_solution in ik_qpos_solutions:
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):  # prefer not to accept > pi rotations
            result = planner.plan_qpos(
                [ik_qpos_solution.astype(np.float32)],
                initial_qpos.astype(np.float32),
                time_step=time_step,
                rrt_range=0.04,
                planning_time=planning_time,
                simplify=simplify_path,
            )
            if result["status"] == SUCCESS:
                qpos_path = result["position"]
                if (
                    qpos_path.shape[0] == 0
                    or attempt == MAX_ATTEMPTS - 1
                    or not np.any(
                        np.abs(ik_qpos_solution[: qpos_path.shape[1]] - qpos_path[-1, :]) > np.pi
                    )
                ):
                    break
                log("Replanning because solution had a rotation larger than 180 degrees")

    if result["status"] != SUCCESS:
        return None, f"Path planner failed: {result['status']}"

    # Get the path in terms of qpos, and check for collisions
    qpos_path = result["position"]

    # Check for zero-length plan (this happens when robot is already in the target qpos)
    if qpos_path.shape[0] == 0:
        # Return a plan of at least length 1
        qpos_path = ik_qpos_solutions[0:1, 0 : len(planner.joint_vel_limits)]
        replan_collisions = False

    # Check for collisions
    collisions = (
        get_collisions(planner, qpos_path, detailed=False)
        if check_collisison or replan_collisions
        else None
    )

    # Try to replan segments with collisions, if requested by caller
    # This is not guaranteed
    qpos_padding = ik_qpos_solution[
        qpos_path.shape[1] :
    ]  # extra qpos for the joints that are not involved in the arm controller
    if replan_collisions and np.any(collisions):
        log(f"Collisions: {collisions}")
        col_segs = get_collision_segments(
            collisions, gap=5
        )  # list of list of [start_idx, end_idx_exclusive] of segments where collisions occur
        qpos_path_segments = []  # the good and fixed segments of the qpos_path go here
        collisions_segments = []  # the good and fixed segments of the collisions go here
        previous_end_exclusive = 0
        for start, end_exclusive in col_segs:
            log(f"Re-plan from {start} to {end_exclusive}")
            start_exclusive = max(start - 1, 0)
            qpos_path_segments.append(qpos_path[previous_end_exclusive:start_exclusive])
            collisions_segments.append(collisions[previous_end_exclusive:start_exclusive])
            cs_result = planner.plan_qpos(
                [np.concatenate([qpos_path[end_exclusive, :], qpos_padding])],
                np.concatenate([qpos_path[start_exclusive, :], qpos_padding]),
                time_step=time_step,
                rrt_range=0.04,
                planning_time=planning_time,
                simplify=False,  # Setting this to True may lead to collision
            )

            if cs_result["status"] == SUCCESS:
                replacement_qpos_path = cs_result["position"]
                replacement_collisions = get_collisions(
                    planner, replacement_qpos_path, detailed=False
                )
                log(f"Replanning succeeded: replacement_collisions: {replacement_collisions}")

                if np.any(replacement_collisions):
                    log(
                        f"Replanned path from step {start_exclusive} to {end_exclusive}, "
                        f"but still got {np.sum(replacement_collisions)} collisions in the new plan"
                    )
                qpos_path_segments.append(cs_result["position"])
                collisions_segments.append(replacement_collisions)
            else:  # re-planning failed, keep the collision segment
                qpos_path_segments.append(qpos_path[start_exclusive:end_exclusive])
                collisions_segments.append(collisions_segments[start_exclusive:end_exclusive])
            previous_end_exclusive = end_exclusive
        qpos_path_segments.append(qpos_path[previous_end_exclusive:])
        collisions_segments.append(collisions[previous_end_exclusive:])
        qpos_path = np.concatenate([s for s in qpos_path_segments if len(s) > 0], axis=0)
        collisions = np.concatenate([c for c in collisions_segments if len(c) > 0])

    return qpos_path, collisions


def get_link_xml_by_name(links_xml: List[minidom.Element], name: str) -> minidom.Element:
    for link_xml in links_xml:
        if link_xml.attributes["name"].value == name:
            return link_xml
    return None


def joints_xml_to_dict(joints_xml: List[minidom.Element]):
    result_children = {}
    result_parent = {}
    for joint_xml in joints_xml:
        parent_name = joint_xml.getElementsByTagName("parent")[0].attributes["link"].value
        child_name = joint_xml.getElementsByTagName("child")[0].attributes["link"].value
        if not parent_name in result_children:
            result_children[parent_name] = []
        result_children[parent_name].append(joint_xml)
        result_parent[child_name] = joint_xml
    return result_children, result_parent


def get_link_and_joint_hierarchy(
    links_xml: List[minidom.Element],
    joint_children_dict: Dict[str, List[minidom.Element]],
    parent_link_name: str,
):
    result_links = [get_link_xml_by_name(links_xml, parent_link_name)]
    result_joints = []
    if parent_link_name in joint_children_dict:
        for joint_xml in joint_children_dict[parent_link_name]:
            result_joints.append(joint_xml)
            child_name = joint_xml.getElementsByTagName("child")[0].attributes["link"].value
            l, j = get_link_and_joint_hierarchy(links_xml, joint_children_dict, child_name)
            result_links += l
            result_joints += j
    return result_links, result_joints


def get_cartesian_robot_mplib_planner_with_attached_actors(
    agent=None,
    translation_axes=None,
    neutral_ee_pos=None,
    attached_actors: Optional[List[sapien.Actor]] = None,
    mesh: bool = True,
    extend_bounds=0.01,
    actor_distance: ActorDistance = None,
) -> mplib.Planner:
    """
    Creates a Cartesian flying gripper from the agent robot and/or attached actors, and returns the
    corresponding motion planner. This is essentially a way of constraining the motion of the EE.
    (could not get mplib constraints to work properly)

    :param agent: Used to obtain the agent and the actor distance
    :param translation_axes: Along what axes the robot should be able to translate.
    :param neutral_ee_pos: Where the EE should be when the translation axes and rotation axis
    are all neutral (0)
    :param attached_actors: the mesh or bounding box of these actors will be attached
    to the EE link. The relative pose will be the current relative pose!
    :param mesh: whether to use a mesh or a bounding box for the attached_actors.
    :param extend_bounds: How much to grow the bound of the attached actors
    :param actor_distance:
    """
    if not translation_axes:
        translation_axes = [0, 1, 2]
    if not neutral_ee_pos:
        neutral_ee_pos = [0, 0, 0]
    if not attached_actors:
        attached_actors = []

    temp_path = get_temp_dir(agent.scene if agent else None)
    attach_mesh_link = attach_mesh_pose = attach_mesh = None
    attach_box_link = attach_box_pose = attach_box_size = None
    attach_link = agent.ee_link if agent else attached_actors[0]

    if len(attached_actors) > 0:
        reference_pose = (
            attach_link.pose
            if agent
            else sapien.Pose(
                [attach_link.pose.p[0], attach_link.pose.p[1], 0.0], [1.0, 0.0, 0.0, 0.0]
            )
        )

        assert (
            actor_distance
        ), "Instance of ActorDistance must be supplied when attached_actors is specified"

        if mesh:
            actor_point_cloud = actor_distance.get_point_cloud(
                attached_actors, reference_pose=reference_pose
            )
            if extend_bounds > 0:
                actor_point_cloud = dilate(actor_point_cloud, extend_bounds)
            attach_mesh = trimesh.convex.convex_hull(actor_point_cloud)
            log(f"grasped_actor_mesh water tight: {attach_mesh.is_watertight}")
            attach_mesh_link = attach_link.name if agent else "actor"
            attach_mesh_pose = sapien.Pose()
        else:
            bounds = actor_distance.get_bounds(attached_actors, reference_pose=reference_pose)
            bounds[0, :] -= extend_bounds
            bounds[1, :] += extend_bounds
            attach_box_size = bounds[1] - bounds[0]
            attach_box_pose = sapien.Pose(np.mean(bounds, axis=0))
            attach_box_link = attach_link.name if agent else "actor"
            attach_box_link = attach_box_link if np.product(attach_box_size) > 0 else None

    if agent:
        output_urdf_path = os.path.join(
            temp_path,
            os.path.splitext(os.path.split(agent.urdf_path)[1])[0] + "_cartesian.urdf",
        )
        urdf_path, srdf_path, _attached_mesh_path = generate_cartesian_robot_urdf(
            agent,
            translation_axes=translation_axes,
            neutral_ee_pos=neutral_ee_pos,
            attach_mesh_link=attach_mesh_link,
            attach_mesh_pose=attach_mesh_pose,
            attach_mesh=attach_mesh,
            attach_box_link=attach_box_link,
            attach_box_pose=attach_box_pose,
            attach_box_size=attach_box_size,
            output_urdf_path=output_urdf_path,
        )
        move_group_name = agent.ee_link.name
    else:
        output_urdf_path = os.path.join(temp_path, "actor.urdf")
        urdf_path, srdf_path, _attached_mesh_path = generate_cartesian_actor_urdf(
            output_urdf_path,
            name=attach_link.name,
            translation_axes=translation_axes,
            attach_mesh_link=attach_mesh_link,
            attach_mesh_pose=attach_mesh_pose,
            attach_mesh=attach_mesh,
            attach_box_link=attach_box_link,
            attach_box_pose=attach_box_pose,
            attach_box_size=attach_box_size,
        )
        move_group_name = "actor"

    num_joints = len(translation_axes)
    if agent:
        num_joints += 1  # rotation of the EE
    planner = mplib.Planner(
        urdf_path,
        srdf=srdf_path,
        move_group=move_group_name,
        user_link_names=[],
        user_joint_names=[],
        joint_vel_limits=2 * np.ones(num_joints),
        joint_acc_limits=2 * np.ones(num_joints),
    )

    if agent:
        # planner.set_base_pose(agent.robot.pose)
        planner.set_base_pose(mplib.Pose(agent.robot.pose.p, agent.robot.pose.q))

    # Be careful: when the planner is used for pushing,
    # Having a floor will cause collisions for objects that are sliding over it
    # And since the robot is 2D anyway,
    if 2 in translation_axes:  # Can we translate along Z?
        # Then add a floor
        fcl_floor = fcl.CollisionObject(fcl.Box([2.0, 2.0, 0.05]), [0, 0, -0.0251], [1, 0, 0, 0])
        planner.set_normal_object("floor", fcl_floor)

    return planner


def generate_cartesian_robot_urdf(
    agent,
    translation_axes=None,
    neutral_ee_pos=None,
    attach_mesh_link: str = None,
    attach_mesh_pose=None,
    attach_mesh=None,
    attach_box_link: str = None,
    attach_box_pose=None,
    attach_box_size=None,
    output_urdf_path: str = None,
) -> Tuple[str, str, str]:
    """
    Takes the agent.robot from the env and generates a flying "Cartesian robot" gripper URDF/SRDF.
    Optionally attached a mesh or a box.
    :param agent:
    :param translation_axes:
    :param neutral_ee_pos:
    :param attach_mesh_link:
    :param attach_mesh_pose:
    :param attach_mesh:
    :param attach_box_link:
    :param attach_box_pose:
    :param attach_box_size:
    :param output_urdf_path:
    :return: tuple (path to URDF, path to SRDF, path to mesh)
    """
    if not translation_axes:
        translation_axes = [0, 1, 2]
    if not neutral_ee_pos:
        neutral_ee_pos = [0, 0, 0]

    robot = agent.robot

    robot_xml = minidom.parse(agent.urdf_path)

    root_link_name = robot.get_links()[0].name
    ee_link_name = agent.ee_link.name

    joints_xml = robot_xml.getElementsByTagName("joint")
    links_xml = robot_xml.getElementsByTagName("link")

    root_link_xml = get_link_xml_by_name(links_xml, root_link_name)
    joint_children_dict, joint_parent_dict = joints_xml_to_dict(joints_xml)

    # Walk up the chain until the first revolute joint is found
    # We will remove the robot from the root link until that parent link of that joint.
    top_link_name = ee_link_name
    top_joint_name = None
    while True:
        joint_xml = joint_parent_dict[top_link_name]
        joint_type = joint_xml.attributes["type"].value
        parent_link_name = joint_xml.getElementsByTagName("parent")[0].attributes["link"].value
        if joint_type == "fixed":
            top_link_name = parent_link_name
        elif joint_type == "revolute":
            top_link_name = parent_link_name
            top_joint_name = joint_xml.attributes["name"].value
            break
        else:
            log("Robot not suitable for vertical planner")
            return None

    # Get the neutral pose of the top_link
    pmodel = robot.create_pinocchio_model()
    neutral_qpos = np.zeros(len(robot.get_active_joints()), dtype=np.float32)
    pmodel.compute_forward_kinematics(neutral_qpos)
    top_link_idx = [l.name for l in robot.get_links()].index(top_link_name)
    ee_link_idx = [l.name for l in robot.get_links()].index(ee_link_name)
    top_link_neutral_pose = pmodel.get_link_pose(top_link_idx)
    ee_link_neutral_pose = pmodel.get_link_pose(ee_link_idx)

    top_link_neutral_rpy = quat2euler(top_link_neutral_pose.q)

    # determine which links and joints to keep, and remove the unneeded elements
    keep_links, keep_joints = get_link_and_joint_hierarchy(
        links_xml, joint_children_dict, top_link_name
    )
    keep_links.append(root_link_xml)

    all_elements_xml = joints_xml + links_xml
    keep_xml = keep_links + keep_joints
    main_xml = keep_xml[0].parentNode
    deleted_joints = []
    deleted_links = []
    for element_xml in all_elements_xml:
        if not element_xml in keep_xml:
            name = element_xml.attributes["name"].value
            node_type = element_xml.nodeName
            if node_type == "joint":
                deleted_joints.append(name)
            elif node_type == "link":
                deleted_links.append(name)
            else:
                # unknown node type . . .
                continue
            main_xml.removeChild(element_xml)
    log(f"Deleted joints: {deleted_joints}")
    log(f"Deleted links: {deleted_links}")

    # create links for translation (without collision geometry)
    translation_links = []
    for axis_idx in translation_axes:
        axis_name = ["0", "1", "2"][axis_idx]
        box_size = [0.01] * 3
        box_size[axis_idx] = 2.0
        origin_xml = robot_xml.createElement("origin")
        origin_xml.setAttribute("xyz", "0 0 0")
        origin_xml.setAttribute("rpy", "0 0 0")
        box_xml = robot_xml.createElement("box")
        box_xml.setAttribute("size", f"{box_size[0]} {box_size[1]} {box_size[2]}")
        geometry_xml = robot_xml.createElement("geometry")
        geometry_xml.appendChild(box_xml)
        visual_xml = robot_xml.createElement("visual")
        visual_xml.appendChild(origin_xml)
        visual_xml.appendChild(geometry_xml)
        link_xml = robot_xml.createElement("link")
        link_xml.setAttribute("name", f"rail_{axis_name}")
        link_xml.appendChild(visual_xml)
        translation_links.append(link_xml)
        main_xml.appendChild(link_xml)

    connector_link = robot_xml.createElement("link")
    connector_link.setAttribute("name", "connector_link")
    main_xml.appendChild(connector_link)

    joint_info = [
        (None, "world_joint", "fixed", root_link_xml, translation_links[0], [0, 0, 0], [0, 0, 0])
    ]
    for idx, axis_idx in enumerate(translation_axes):
        axis_name = ["x", "y", "z"][axis_idx]
        joint_name = f"prismatic_joint_{axis_name}"
        next_link = (
            translation_links[idx + 1] if (idx + 1) < len(translation_links) else connector_link
        )
        joint_info.append(
            (
                axis_idx,
                joint_name,
                "prismatic",
                translation_links[idx],
                next_link,
                [0, 0, 0],
                [0, 0, 0],
            )
        )
    joint_info.append(
        (
            None,
            "connector_joint",
            "fixed",
            connector_link,
            get_link_xml_by_name(links_xml, top_link_name),
            np.array(neutral_ee_pos, dtype=np.float32)
            + top_link_neutral_pose.p
            - ee_link_neutral_pose.p,
            top_link_neutral_rpy,
        )
    )

    # create joints for translation
    new_joint_names = []
    for axis_idx, joint_name, joint_type, parent_link, child_link, xyz, rpy in joint_info:
        origin_xml = robot_xml.createElement("origin")
        origin_xml.setAttribute("xyz", f"{xyz[0]} {xyz[1]} {xyz[2]}")
        origin_xml.setAttribute("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        parent_xml = robot_xml.createElement("parent")
        parent_xml.setAttribute("link", parent_link.attributes["name"].value)
        child_xml = robot_xml.createElement("child")
        child_xml.setAttribute("link", child_link.attributes["name"].value)
        joint_xml = robot_xml.createElement("joint")
        joint_xml.setAttribute("name", joint_name)
        joint_xml.setAttribute("type", joint_type)

        for element in [origin_xml, parent_xml, child_xml]:
            joint_xml.appendChild(element)

        if not axis_idx is None:
            axis_xml = robot_xml.createElement("axis")
            axis_xml.setAttribute(
                "xyz", f"{int(axis_idx==0)} {int(axis_idx==1)} {int(axis_idx==2)}"
            )
            limit_xml = robot_xml.createElement("limit")
            limit_xml.setAttribute("lower", "-1.0")
            limit_xml.setAttribute("upper", "1.0")
            limit_xml.setAttribute("effort", "1000")
            limit_xml.setAttribute("velocity", "10")
            for element in [axis_xml, limit_xml]:
                joint_xml.appendChild(element)

        new_joint_names.append(joint_name)
        main_xml.appendChild(joint_xml)

    attach_mesh_actual_path = None
    if attach_box_link or attach_mesh_link:
        attach_mesh_actual_path = attach_urdf(
            agent.urdf_path,
            robot_xml,
            attach_mesh_link=attach_mesh_link,
            attach_mesh_pose=attach_mesh_pose,
            attach_mesh=attach_mesh,
            attach_box_link=attach_box_link,
            attach_box_pose=attach_box_pose,
            attach_box_size=attach_box_size,
        )

    if output_urdf_path is None:
        output_urdf_path = os.path.splitext(agent.urdf_path)[0] + "_cartesian_flying_gripper.urdf"
    resolve_relative_path_references(robot_xml, output_urdf_path, os.path.split(agent.urdf_path)[0])
    write_xml_clean_and_pretty(robot_xml, output_urdf_path)

    # Adjust the SRDF to match the URDF
    srdf_path = os.path.splitext(agent.urdf_path)[0] + ".srdf"
    srdf_xml = minidom.parse(srdf_path)

    # Find groups with the joint top_joint_name, and adjust them
    for tag in ["group", "group_state"]:
        for group_xml in srdf_xml.getElementsByTagName(tag):
            for joint_xml in group_xml.getElementsByTagName("joint"):
                if joint_xml.attributes["name"].value == top_joint_name:
                    arm_group_xml = group_xml

                    # Remove the deleted joints
                    for joint_xml in arm_group_xml.getElementsByTagName("joint"):
                        if joint_xml.attributes["name"].value in deleted_joints:
                            arm_group_xml.removeChild(joint_xml)
                    # Add the new joints
                    for new_joint_name in new_joint_names:
                        joint_xml = srdf_xml.createElement("joint")
                        joint_xml.setAttribute("name", new_joint_name)
                        if tag == "group_state":
                            joint_xml.setAttribute("value", "0")
                        arm_group_xml.appendChild(joint_xml)
                    break

    # SRDF disable_collisions does not have to be adjusted because the prismatic
    # joints do not have collision bodies

    output_srdf_path = os.path.splitext(output_urdf_path)[0] + ".srdf"
    write_xml_clean_and_pretty(srdf_xml, output_srdf_path)

    return output_urdf_path, output_srdf_path, attach_mesh_actual_path


def generate_cartesian_actor_urdf(
    output_urdf_path: str,
    name="actor",
    translation_axes: Optional[List[int]] = None,
    attach_mesh_link: str = None,
    attach_mesh_pose=None,
    attach_mesh=None,
    attach_box_link: str = None,
    attach_box_pose=None,
    attach_box_size=None,
) -> (str, str, str):
    """
    Generates a Cartesian robot for path planning from a mesh or box.
    :param output_urdf_path:
    :param name:
    :param translation_axes:
    :param attach_mesh_link:
    :param attach_mesh_pose:
    :param attach_mesh:
    :param attach_box_link:
    :param attach_box_pose:
    :param attach_box_size:
    :return: URDF path, SRDF path, path of temp mesh
    """
    if not translation_axes:
        translation_axes = [0, 1, 2]

    xml_text = ['<?xml version="1.0" ?>']
    xml_text.append(f'<robot name="{name}">')
    xml_text.append('  <link name="world"/>')

    # create links
    links = []
    for axis in translation_axes:
        axis_name = ["x", "y", "z"][axis]
        link_name = f"rail_{axis_name}"
        box = [0.01] * 3
        box[axis] = 2.0

        xml_text.append(f'  <link name="{link_name}">')
        xml_text.append("    <visual>")
        xml_text.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml_text.append("      <geometry>")
        xml_text.append(f'        <box size="{box[0]} {box[1]} {box[2]}"/>')
        xml_text.append("      </geometry>")
        xml_text.append("    </visual>")
        xml_text.append("  </link>")
        xml_text.append("")
        links.append((axis, axis_name, link_name))

    xml_text.append('  <link name="actor"/>')
    links.append((None, None, "actor"))

    # Create joints
    joint_names = ["world_joint"]
    xml_text.append('  <joint name="world_joint" type="fixed">')
    xml_text.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
    xml_text.append('    <parent link="world"/>')
    xml_text.append(f'    <child link="{links[0][2]}"/>')
    xml_text.append("  </joint>")

    for idx in range(len(translation_axes)):
        axis, axis_name, link_name = links[idx]
        _, _, next_link_name = links[idx + 1]
        a = [0.0, 0.0, 0.0]
        a[axis] = 1.0
        joint_name = f"prismatic_joint_{axis_name}"
        joint_names.append(joint_name)
        xml_text.append(f'  <joint name="{joint_name}" type="prismatic">')
        xml_text.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml_text.append(f'    <parent link="{link_name}"/>')
        xml_text.append(f'    <child link="{next_link_name}"/>')
        xml_text.append(f'    <axis xyz="{a[0]} {a[1]} {a[2]}"/>')
        xml_text.append('    <limit lower="-1.0" upper="1.0" effort="1000" velocity="10"/>')
        xml_text.append("  </joint>")

    xml_text.append("</robot>")

    xml_str = "\n".join(xml_text)
    robot_xml = minidom.parseString(xml_str)

    # Append the mesh / box
    attach_mesh_actual_path = None
    if attach_box_link or attach_mesh_link:
        attach_mesh_actual_path = attach_urdf(
            output_urdf_path,
            robot_xml,
            attach_mesh_link=attach_mesh_link,
            attach_mesh_pose=attach_mesh_pose,
            attach_mesh=attach_mesh,
            attach_box_link=attach_box_link,
            attach_box_pose=attach_box_pose,
            attach_box_size=attach_box_size,
        )

    # Write URDF
    write_xml_clean_and_pretty(robot_xml, output_urdf_path)

    # Create SRDF
    xml_text = ['<?xml version="1.0" ?>']
    xml_text.append(f'<robot name="{name}">')
    xml_text.append('  <group name="actor_arm">')
    for joint_name in joint_names:
        xml_text.append(f'    <joint name="{joint_name}"/>')
    xml_text.append("  </group>")
    xml_text.append('  <group name="actor_ee">')
    xml_text.append('    <link name="actor"/>')
    xml_text.append("  </group>")
    xml_text.append('  <group_state name="zero" group="actor_arm">')
    for joint_name in joint_names[1:]:
        xml_text.append(f'    <joint name="{joint_name}" value="0"/>')
    xml_text.append("  </group_state>")
    xml_text.append('  <end_effector name="actor_ee" parent_link="actor" group="actor_ee"/>')
    for idx in range(len(translation_axes)):
        _, _, link_name = links[idx]
        _, _, next_link_name = links[idx + 1]
        xml_text.append(
            f'  <disable_collisions link1="{link_name}" link2="{next_link_name}" '
            f'reason="Adjacent"/>'
        )
    xml_text.append("</robot>")
    xml_str = "\n".join(xml_text)
    srdf_xml = minidom.parseString(xml_str)
    output_srdf_path = os.path.splitext(output_urdf_path)[0] + ".srdf"
    write_xml_clean_and_pretty(srdf_xml, output_srdf_path)

    return (output_urdf_path, output_srdf_path, attach_mesh_actual_path)
