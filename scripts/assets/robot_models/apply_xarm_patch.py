# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# This script applies a patch to a checkout of the official UFACTORY xArm
# GitHub repo in order to obtain the URDF that is used in ClevrSkills.
#
# The repo is at https://github.com/xArm-Developer/xarm_ros.git,
# commit 740b003d5b330ea70f474a358ffc29c4aad3444c

import argparse
import os
import shutil
import tempfile
from typing import List
from xml.dom import minidom

import numpy as np
import trimesh
import trimesh.convex

from clevr_skills.utils.xml import write_xml_clean_and_pretty


def copy_stl_files(xarm_repo_path: str, patch_path: str, output_path: str) -> None:
    """
    Copies STL files, verbatim. Most from the xArm repo, and some simple primitives
    from ClevrSkills repo
    """
    visual_meshes_base = "xarm_description/meshes/xarm6/visual/"  # visual meshes base
    filenames = [
        "base.stl",
        "link1.stl",
        "link2.stl",
        "link3.stl",
        "link4.stl",
        "link5.stl",
        "link6.stl",
    ]

    for filename in filenames:
        src_path = os.path.join(xarm_repo_path, visual_meshes_base + filename)
        dst_path = os.path.join(output_path, "xarm6_description", filename)
        shutil.copy(src_path, dst_path)

    src_path = os.path.join(xarm_repo_path, "xarm_description/meshes/collision/end_tool.STL")
    dst_path = os.path.join(output_path, "xarm_gripper_description/end_tool.STL")
    shutil.copy(src_path, dst_path)

    src_path = os.path.join(
        xarm_repo_path, "xarm_description/meshes/vacuum_gripper/visual/vacuum_gripper.STL"
    )
    dst_path = os.path.join(output_path, "xarm_vacuum_description/vacuum_gripper.STL")
    shutil.copy(src_path, dst_path)

    # Also copy the cylinders for the suction cup collision models.
    # These are simple cylinder shapes.
    # filenames = [
    #     "xarm_vacuum_description/vacuum_suction_cup_collision_1.stl",
    #     "xarm_vacuum_description/vacuum_suction_cup_collision_2.stl",
    #     "xarm_vacuum_description/vacuum_suction_cup_collision_3.stl",
    #     "xarm_vacuum_description/vacuum_suction_cup_collision_4.stl",
    #     "xarm_vacuum_description/vacuum_suction_cup_collision_5.stl",
    # ]
    # for filename in filenames:
    #     src_path = os.path.join(patch_path, filename)
    #     dst_path = os.path.join(output_path, filename)
    #     shutil.copy(src_path, dst_path)


def patch_stl_files(xarm_repo_path: str, patch_path: str, output_path: str) -> None:
    """
    Applies patches to STL file.

    A "patch" is a numpy array consisting of the indices of the vertices from the original STL
    that should be used.
    Plus another numpy array with newly introduced vertices (due to the convex-ify-cation)
    The mesh is created by creating the convex hull of these vertices.
    """
    files = [
        (
            "xarm_description/meshes/xarm6/visual/base.stl",
            "xarm6_description/base_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/base.stl",
            "xarm6_description/base_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/base.stl",
            "xarm6_description/base_collision_part3.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link1.stl",
            "xarm6_description/link1_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link1.stl",
            "xarm6_description/link1_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link2.stl",
            "xarm6_description/link2_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link2.stl",
            "xarm6_description/link2_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link2.stl",
            "xarm6_description/link2_collision_part3.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link2.stl",
            "xarm6_description/link2_collision_part4.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link3.stl",
            "xarm6_description/link3_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link3.stl",
            "xarm6_description/link3_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link3.stl",
            "xarm6_description/link3_collision_part3.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link4.stl",
            "xarm6_description/link4_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link4.stl",
            "xarm6_description/link4_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link4.stl",
            "xarm6_description/link4_collision_part3.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link4.stl",
            "xarm6_description/link4_collision_part4.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link4.stl",
            "xarm6_description/link4_collision_part5.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link5.stl",
            "xarm6_description/link5_collision_part1.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link5.stl",
            "xarm6_description/link5_collision_part2.stl",
        ),
        (
            "xarm_description/meshes/xarm6/visual/link5.stl",
            "xarm6_description/link5_collision_part3.stl",
        ),
        (
            "xarm_description/meshes/vacuum_gripper/visual/vacuum_gripper.STL",
            "xarm_vacuum_description/vacuum_head_collision.stl",
        ),
        (
            "xarm_description/meshes/vacuum_gripper/visual/vacuum_gripper.STL",
            "xarm_vacuum_description/vacuum_mount_ring_collision.stl",
        ),
        (
            "xarm_description/meshes/vacuum_gripper/visual/vacuum_gripper.STL",
            "xarm_vacuum_description/vacuum_trunk_collision.stl",
        ),
    ]

    for src, dst in files:
        src_path = os.path.join(xarm_repo_path, src)
        base_patch_path = os.path.join(patch_path, dst)
        dst_path = os.path.join(output_path, dst)

        src_mesh: trimesh.Trimesh = trimesh.load_mesh(src_path)
        with open(base_patch_path + "_vertex_indices.npy", "rb") as file:
            original_vtx_idx = np.load(file)
        with open(base_patch_path + "_clip_vertex.npy", "rb") as file:
            vertices = np.load(file)
        if len(original_vtx_idx):
            original_vtx = src_mesh.vertices[original_vtx_idx]
            vertices = np.concatenate([vertices, original_vtx], axis=0)

        mesh = trimesh.convex.convex_hull(vertices)
        trimesh.exchange.export.export_mesh(mesh, dst_path)


def prepare_files_for_diff(paths: List[str]) -> List[str]:
    lines = []
    for path in paths:
        with open(path, "r", encoding="utf8") as file:
            for line in file:
                line = line.strip()
                line = line.replace('" />', '"/>')
                line = line.replace("axis   xyz", "axis xyz")
                line = line.replace('rpy="0 0 0" xyz="0 0 0"', 'xyz="0 0 0" rpy="0 0 0"')
                lines.append(line + "\n")

    return lines


def prepare_xarm_xacro(xarm_repo_path: str, output_path: str) -> None:
    """
    Fuses the xArm6 and vacuum gripper xacro, removes whitespace.
    Prepares for creating the patch
    :param xarm_repo_path:
    :param output_path: path to file where prepared xacro will be written
    """
    paths = [os.path.join(xarm_repo_path, "xarm_description/urdf/xarm6.urdf.xacro")]
    lines = prepare_files_for_diff(paths)
    with open(output_path, "w+", encoding="utf8") as file:
        for line in lines:
            file.write(line)


def apply_urdf_patch(xarm_repo_path: str, patch_path: str, output_path: str) -> str:
    with tempfile.TemporaryDirectory() as temp_path:
        xacro_path = os.path.join(temp_path, "temp.xacro")
        urdf_patch_filename = "xarm6_with_vacuum.urdf.patch"
        urdf_patch_path = os.path.join(patch_path, urdf_patch_filename)
        urdf_path = os.path.join(output_path, "xarm6_with_vacuum.urdf")
        prepare_xarm_xacro(xarm_repo_path, xacro_path)
        os.system(f"patch {xacro_path} -i {urdf_patch_path} -o {urdf_path}")
    return urdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xarm-repo-path",
        type=str,
        help=(
            "Path to a clone of https://github.com/xArm-Developer/xarm_ros.git at commit "
            "740b003d5b330ea70f474a358ffc29c4aad3444c (i.e., the source of the patch)"
        ),
        required=True,
    )
    parser.add_argument(
        "--patch-path", type=str, help="Path to directory containing the patch files", required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to where the patched files should be written",
        required=True,
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm6_description"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm_gripper_description"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm_vacuum_description"), exist_ok=True)

    copy_stl_files(args.xarm_repo_path, args.patch_path, args.output_path)
    patch_stl_files(args.xarm_repo_path, args.patch_path, args.output_path)
    urdf_path = apply_urdf_patch(args.xarm_repo_path, args.patch_path, args.output_path)

    robot_xml = minidom.parse(urdf_path)
    write_xml_clean_and_pretty(robot_xml, urdf_path)


if __name__ == "__main__":
    main()
