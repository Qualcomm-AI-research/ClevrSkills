# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# This script copies over the Panda robot from ManiSkill2
# This makes it more straightforward to load from ClevrSkills

import argparse
import os
import shutil
import tempfile
from xml.dom import minidom

from clevr_skills.utils.xml import write_xml_clean_and_pretty
from scripts.assets.robot_models.apply_xarm_patch import prepare_files_for_diff


def prepare_panda_v2_urdf(maniskill2_urdf_path: str, output_path: str) -> None:
    """
    Removes all whitespace.
    Deletes the "hand" from the maniskill2_urdf_path for a cleaner diff.
    :param maniskill2_urdf_path:
    :param output_path: path to file where prepared xacro will be written
    """
    paths = [maniskill2_urdf_path]
    lines = prepare_files_for_diff(paths)

    clean_lines = []
    for line_idx, line in enumerate(lines):
        if 'name="panda_hand_joint"' in line:
            for line in lines[line_idx:]:
                if "</robot>" in line:
                    clean_lines.append(line)
                    break
            break
        clean_lines.append(line)

    with open(output_path, "w+", encoding="utf8") as file:
        for line in clean_lines:
            file.write(line)


def prepare_generic_urdf(clevrskills_urdf_path: str, output_path: str) -> None:
    """
    Removes all whitespace.
    :param clevrskills_urdf_path:
    :param output_path: path to file where prepared xacro will be written
    """
    # Remove all whitespace from maniskill2_urdf_path (source) and clevrskills_urdf_path (target)
    lines = prepare_files_for_diff([clevrskills_urdf_path])
    with open(output_path, "w+", encoding="utf8") as file:
        for line in lines:
            file.write(line)


def copy_mesh_files(maniskill2_urdf_path: str, output_path: str) -> None:
    """
    Copies over the meshes for the Panda
    :param maniskill2_urdf_path:
    :param output_path:
    """
    # Make this into a function
    maniskill2_descriptions_path = os.path.split(maniskill2_urdf_path)[0]
    src_collision_meshes_path = os.path.join(
        maniskill2_descriptions_path, "franka_description", "meshes", "collision"
    )
    src_visual_meshes_path = os.path.join(
        maniskill2_descriptions_path, "franka_description", "meshes", "visual"
    )

    tgt_collision_meshes_path = os.path.join(
        output_path, "franka_description", "meshes", "collision"
    )
    tgt_visual_meshes_path = os.path.join(output_path, "franka_description", "meshes", "visual")
    os.makedirs(tgt_collision_meshes_path, exist_ok=True)
    os.makedirs(tgt_visual_meshes_path, exist_ok=True)

    part_filenames = [
        "finger",
        "hand",
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
    ]
    for filename in part_filenames:
        shutil.copy(
            os.path.join(src_collision_meshes_path, filename + ".stl"),
            os.path.join(tgt_collision_meshes_path, filename + ".stl"),
        )

        # Remove stale convex file; it will be re-generated
        shutil.rmtree(
            os.path.join(tgt_collision_meshes_path, filename + ".stl.convex.stl"),
            ignore_errors=True,
        )

        shutil.copy(
            os.path.join(src_visual_meshes_path, filename + ".dae"),
            os.path.join(tgt_visual_meshes_path, filename + ".dae"),
        )


def apply_urdf_patch(maniskill2_urdf_path: str, patch_path: str, output_path: str) -> str:
    with tempfile.TemporaryDirectory() as temp_path:
        src_path = os.path.join(temp_path, "panda_v2.urdf")
        urdf_patch_filename = "panda_v2_with_vacuum.urdf.patch"
        urdf_patch_path = os.path.join(patch_path, urdf_patch_filename)
        urdf_path = os.path.join(output_path, "panda_v2_with_vacuum.urdf")
        prepare_panda_v2_urdf(maniskill2_urdf_path, src_path)
        os.system(f"patch {src_path} -i {urdf_patch_path} -o {urdf_path}")

    return urdf_path


def apply_srdf_patch(maniskill2_urdf_path: str, patch_path: str, output_path: str) -> str:
    maniskill2_srdf_path = os.path.splitext(maniskill2_urdf_path)[0] + ".srdf"

    with tempfile.TemporaryDirectory() as temp_path:
        src_path = os.path.join(temp_path, "panda_v2.srdf")
        urdf_patch_filename = "panda_v2_with_vacuum.srdf.patch"
        urdf_patch_path = os.path.join(patch_path, urdf_patch_filename)
        urdf_path = os.path.join(output_path, "panda_v2_with_vacuum.srdf")
        prepare_generic_urdf(maniskill2_srdf_path, src_path)
        os.system(f"patch {src_path} -i {urdf_patch_path} -o {urdf_path}")

    return urdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maniskill2-urdf-path",
        type=str,
        help="Path to panda_v2.urdf inside ManiSkill2 repo, at tag v0.5.3",
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

    copy_mesh_files(args.maniskill2_urdf_path, args.output_path)

    urdf_path = apply_urdf_patch(args.maniskill2_urdf_path, args.patch_path, args.output_path)
    robot_xml = minidom.parse(urdf_path)
    write_xml_clean_and_pretty(robot_xml, urdf_path)

    srdf_path = apply_srdf_patch(args.maniskill2_urdf_path, args.patch_path, args.output_path)
    robot_xml = minidom.parse(srdf_path)
    write_xml_clean_and_pretty(robot_xml, srdf_path)


if __name__ == "__main__":
    main()
