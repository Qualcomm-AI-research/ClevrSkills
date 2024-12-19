# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# This script computes a patch relative to the official UFACTORY xArm
# GitHub repo in order to obtain the URDF that is used in ClevrSkills.
#
# The patch can then be applied using apply_xarm_patch.py to recreate the URDF
# on the end-user side, without requiring us to re-distribute the UFACTORY source code or meshes.
#
# The repo is at https://github.com/xArm-Developer/xarm_ros.git,
# commit 740b003d5b330ea70f474a358ffc29c4aad3444c

import argparse
import os
import shutil
import tempfile

import numpy as np
import trimesh

from scripts.assets.robot_models.apply_xarm_patch import prepare_files_for_diff, prepare_xarm_xacro


def copy_stl_files(xarm_repo_path: str, clevrskills_urdf_path: str, output_path: str) -> None:
    """
    Copies STL files, verbatim.
    Also, copies over STL files that we created (cylinders that represent the collision model of the vacuum suction cups)
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

    # Also copy the cylinders for the suction cup collision files
    # These come from our own repo, but were created us as pure cylinder shapes.
    filenames = [
        "xarm_vacuum_description/vacuum_suction_cup_collision_1.stl",
        "xarm_vacuum_description/vacuum_suction_cup_collision_2.stl",
        "xarm_vacuum_description/vacuum_suction_cup_collision_3.stl",
        "xarm_vacuum_description/vacuum_suction_cup_collision_4.stl",
        "xarm_vacuum_description/vacuum_suction_cup_collision_5.stl",
    ]
    target_path = os.path.split(clevrskills_urdf_path)[0]
    for filename in filenames:
        src_path = os.path.join(target_path, filename)
        dst_path = os.path.join(output_path, filename)
        shutil.copy(src_path, dst_path)


def create_stl_patches(xarm_repo_path: str, clevrskills_urdf_path: str, output_path: str) -> None:
    """
    Creates patches for STL files.
    This is only required for collision meshes. Collision meshes are convex.

    A patch is a numpy array consisting of the indices of the vertices from the original STL that should be used.
    Plus another numpy array with newly introduced vertices (due to the convex-ify-cation)
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

    # The "target" is what the patch aims to reproduce
    target_path = os.path.split(clevrskills_urdf_path)[0]

    for src, tgt in files:
        src_path = os.path.join(xarm_repo_path, src)
        tgt_path = os.path.join(target_path, tgt)
        src_mesh: trimesh.Trimesh = trimesh.load_mesh(src_path)
        tgt_mesh: trimesh.Trimesh = trimesh.load_mesh(tgt_path)

        # Try to find vertices in the original model.
        # Otherwise, this vertex was created as the result of clipping or convex-i-fi-cation.
        original_vtx_idx = []
        new_vtx = []
        for tgt_vtx in tgt_mesh.vertices:
            diff = np.linalg.norm(src_mesh.vertices - tgt_vtx, axis=1)
            if np.min(diff) == 0:
                original_vtx_idx.append(int(np.argmin(diff)))
            else:
                new_vtx.append(tgt_vtx)

        original_vtx_idx = np.array(original_vtx_idx)
        new_vtx = np.array(new_vtx, dtype=np.float32)

        # Save the vertex indices and new vertices
        with open(os.path.join(output_path, tgt + "_vertex_indices.npy"), "wb+") as file:
            np.save(file, original_vtx_idx)
        with open(os.path.join(output_path, tgt + "_clip_vertex.npy"), "wb+") as file:
            np.save(file, new_vtx)

        print(f"{tgt}: found {original_vtx_idx.shape[0]}/{tgt_mesh.vertices.shape[0]} vertices.")


def prepare_target_urdf(clevrskills_urdf_path: str, output_path: str) -> None:
    paths = [clevrskills_urdf_path]
    lines = prepare_files_for_diff(paths)
    with open(output_path, "w+", encoding="utf8") as file:
        for line in lines:
            file.write(line)


def create_urdf_patch(xarm_repo_path: str, clevrskills_urdf_path: str, output_path: str) -> str:
    with tempfile.TemporaryDirectory() as temp_path:
        temp_path = tempfile.mkdtemp()
        xacro_path = os.path.join(temp_path, "temp.xacro")
        urdf_path = os.path.join(temp_path, "temp.urdf")
        prepare_xarm_xacro(xarm_repo_path, xacro_path)
        prepare_target_urdf(clevrskills_urdf_path, urdf_path)

        urdf_patch_filename = os.path.split(clevrskills_urdf_path)[1] + ".patch"
        urdf_patch_path = os.path.join(output_path, urdf_patch_filename)
        os.system(f"diff {xacro_path} {urdf_path} > {urdf_patch_path}")
    return urdf_patch_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xarm-repo-path",
        type=str,
        help=(
            "Path to a clone of https://github.com/xArm-Developer/xarm_ros.git at "
            "commit 740b003d5b330ea70f474a358ffc29c4aad3444c (i.e., the source of the patch)"
        ),
        required=True,
    )
    parser.add_argument(
        "--clevrskills-urdf-path",
        type=str,
        help="Path to xarm_with_vacuum.urdf that was already modified (i.e., the target of the patch)",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to directory where the patch files should be written",
        required=True,
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm6_description"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm_gripper_description"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "xarm_vacuum_description"), exist_ok=True)

    copy_stl_files(args.xarm_repo_path, args.clevrskills_urdf_path, args.output_path)
    create_stl_patches(args.xarm_repo_path, args.clevrskills_urdf_path, args.output_path)
    create_urdf_patch(args.xarm_repo_path, args.clevrskills_urdf_path, args.output_path)


if __name__ == "__main__":
    main()
