# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import os
import tempfile

from clevr_skills import visualize_all_tasks


def test_visualize_all_tasks_sequence():
    with tempfile.TemporaryDirectory() as temp_path:
        args = argparse.Namespace(
            output_dir=temp_path,
            robot="xarm6_vacuum",
            pretty_video=True,
            compose_grid=False,
            grid_width=6,
            grid_height=4,
            grid_num_frames=100,
            compose_sequence=True,
            output_width=-1,
            max_seeds=-1,
            shader_dir="ibl",
            rt_samples_per_pixel=64,
            enable_sapien_viewer=False,
            levels=None,
            tasks=["Touch"],
            prompt_visualization="multi_modal",
            action_label_visualization="natural_language",
        )
        visualize_all_tasks.main(args)
        assert os.path.isdir(os.path.join(temp_path, "level_0_simple"))
        assert os.path.isfile(os.path.join(temp_path, "sequence.mp4"))


def test_visualize_all_tasks_grid():
    with tempfile.TemporaryDirectory() as temp_path:
        args = argparse.Namespace(
            output_dir=temp_path,
            robot="xarm6_vacuum",
            pretty_video=False,
            compose_grid=True,
            grid_width=1,
            grid_height=1,
            grid_num_frames=100,
            compose_sequence=True,
            output_width=-1,
            max_seeds=-1,
            shader_dir="ibl",
            rt_samples_per_pixel=64,
            enable_sapien_viewer=False,
            levels=["1"],
            tasks=["Throw_to_topple"],
            prompt_visualization="multi_modal",
            action_label_visualization="natural_language",
        )
        visualize_all_tasks.main(args)
        assert os.path.isdir(os.path.join(temp_path, "level_1_intermediate"))
        assert os.path.isdir(os.path.join(temp_path, "level_1_intermediate", "Throw_to_topple"))
        assert os.path.isfile(os.path.join(temp_path, "grid.mp4"))
