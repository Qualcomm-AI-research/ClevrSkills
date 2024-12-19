# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import copy
import os
import shlex
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from clevr_skills import clevr_skills_oracle
from clevr_skills.assets.video.video_template import (
    crop_text_image,
    get_text_image,
    single_color_image,
)
from clevr_skills.utils.logger import log
from clevr_skills.utils.temp_dir import clear_temp_dir
from clevr_skills.utils.visualize_prompt import PromptVisualizer

# level: ("nice_name", "level_name")
# task: ("nice_name", "task_name", {task_args}, seed)
task_specification = {
    ("Level 0: Simple tasks", "level_0_simple"): [
        ("Match_pose", "MatchPose", {"ngoals": 1}, 0),
        ("Move_without_hitting", "MoveWithoutHitting", {"num_distractions": 5}, 0),
        ("Pick", None, {"num_actors": 1}, 1),
        (
            "Place_on_top",
            "PlaceOnTop",
            {"num_actors": 1, "num_areas": 1, "spawn_at_gripper": True},
            7,
        ),
        (
            "Place_next_to",
            "PlaceNextTo",
            {"num_actors": 2, "spawn_at_gripper": True},
            11,
        ),
        ("Rotate", "Rotate", {"num_actors": 1}, 1),
        (
            "Throw_at",
            "Throw",
            {"spawn_at_gripper": True, "target_2d": False, "topple_target": False},
            0,
        ),
        (
            "Throw_to_topple",
            "Throw",
            {"spawn_at_gripper": True, "target_2d": False, "topple_target": True},
            1,
        ),
        ("Touch", "Touch", {"num_actors": 1, "push": False, "topple": False}, 0),
        ("Push", "Touch", {"num_actors": 1, "push": True, "topple": False}, 0),
        (
            "Topple",
            "Touch",
            {"num_actors": 1, "push": False, "topple": True},
            [0, 1, 2, 3],
        ),
        ("Move_to_goal", "Trace", {"ngoals": 4}, 0),
    ],
    ("Level 1: Intermediate tasks", "level_1_intermediate"): [
        ("Pick_and_place_on_top", "PlaceOnTop", {"num_areas": 3, "num_actors": 3}, 5),
        ("Pick_and_place_next_to", "PlaceNextTo", {"num_actors": 2}, 11),
        (
            "Follow_order",
            "FollowOrder",
            {"num_areas": 3, "num_predicates": 3},
            5,
        ),
        (
            "Follow_order_and_restore",
            "FollowOrder",
            {"num_areas": 3, "num_predicates": 3, "restore": True},
            6,
        ),
        ("Neighbour", "Neighbour", {"num_areas": 2}, 3),
        ("NovelAdjective", None, {}, 3),
        ("NovelNoun", None, {"num_actors": 3, "num_areas": 3}, 3),
        ("NovelNounAdjective", None, {}, 3),
        (
            "Rearrange",
            None,
            {"num_actors": 3},
            0,
        ),
        (
            "Rearrange_and_restore",
            "Rearrange",
            {"num_actors": 3, "restore": True},
            0,
        ),
        ("Rotate_and_restore", "Rotate", {"num_actors": 3, "restore": True}, 10),
        ("RotateSymmetry", None, {}, 10),
        ("Stack", "SingleStack", {"num_actors": 4}, 3),
        (
            "Stack_in_reversed_order",
            "SingleStack",
            {"num_actors": 4, "reverse": True},
            3,
        ),
        ("Sort_by_texture", "Sort2d", {"num_actors": 4, "num_areas": 2}, 10),
        (
            "Swap",
            None,
            {"num_actors": 2, "push": False},
            0,
        ),
        (
            "Throw_onto",
            "Throw",
            {"spawn_at_gripper": True, "target_2d": True, "topple_target": False},
            0,
        ),
        (
            "Throw_onto",
            "Throw",
            {"spawn_at_gripper": False, "target_2d": True, "topple_target": False},
            2,
        ),
        (
            "Throw_at",
            "Throw",
            {"spawn_at_gripper": False, "target_2d": False, "topple_target": False},
            3,
        ),
        (
            "Throw_to_topple",
            "Throw",
            {"spawn_at_gripper": False, "target_2d": False, "topple_target": True},
            4,
        ),
    ],
    ("Level 2: complex", "level_2_complex"): [  # sequence
        ("Balance_scale", "BalanceScale", {"num_actors": 4}, [1, 2, 3]),
        ("Stack_sorted_by_texture", "SortStack", {"num_actors": 4, "num_areas": 2}, 6),
        (
            "Stack_and_topple",
            "SingleStack",
            {"num_actors": 4, "topple": True},
            [61, 60],
        ),
        ("Swap_by_pushing", "Swap", {"num_actors": 2, "push": True}, 0),
        ("Swap_and_rotate", "SwapRotate", {"num_actors": 2}, 0),
        ("Throw_and_sort", "ThrowAndSort", {"num_actors": 2, "num_areas": 2}, 3),
    ],
}


def get_default_oracle_args(record_dir: str, args: argparse.Namespace) -> argparse.Namespace:
    """
    Generates a default set of command-line args for clevr_skills_oracle, and then
    overrides with some of the command-line given to this script.
    :param record_dir: Path to where recording should go.
    :param args: command-line args that were given to this script.
    :return: Command-line arguments for clevr_skills_oracle
    """
    oracle_args = clevr_skills_oracle.get_args_parser().parse_args(
        shlex.split(f"--record-dir {record_dir}")
    )
    oracle_args.num_steps_done = 20
    oracle_args.num_episodes = 1
    oracle_args.save_video = True
    oracle_args.pretty_video = args.pretty_video
    oracle_args.shader_dir = args.shader_dir
    oracle_args.rt_samples_per_pixel = args.rt_samples_per_pixel
    oracle_args.enable_sapien_viewer = args.enable_sapien_viewer
    oracle_args.prompt_visualization = args.prompt_visualization
    oracle_args.action_label_visualization = args.action_label_visualization
    oracle_args.robot = args.robot

    return oracle_args


def get_fps_and_resolution(sequence: List[Dict]) -> Tuple[float, Tuple[int, int]]:
    """
    Determines framerate (fps) and resolution by traversing the sequence and examining the first video found.
    :param sequence: Sequence of videos.
    :return: Framerate and resolution: (fps, (width, height))
    """
    resolution = None
    fps = 20  # will be overwritten with fps from video
    for entry in sequence:
        if "video" in entry:
            src_mp4_path = entry["video"]
            video = cv2.VideoCapture(src_mp4_path)
            assert video.isOpened(), f"Could not open {src_mp4_path}"
            ret, frame = video.read()
            assert ret, f"Error reading the first frame of {src_mp4_path}"
            fps = video.get(cv2.CAP_PROP_FPS)
            resolution = (frame.shape[1], frame.shape[0])
            video.release()
            break

    assert resolution, "No video in sequence; can't determine resolution"
    return fps, resolution


def scale_image(image, size):
    """
    Scale image to resolution.
    :param image: OpenCV-style image (numpy array [height x width x 3])
    :param size: (width, height)
    :return: Scaled image.
    """
    if image.shape[0] != size[1] or image.shape[1] != size[0]:
        interpolation = cv2.INTER_CUBIC if image.shape[0] > size[1] else cv2.INTER_LINEAR
        return cv2.resize(image, size, interpolation=interpolation)
    return image


def compose_sequence(mp4_path: str, sequence: List[Dict], output_width: int = -1):
    """
    Sequences all the videos (for each task) and seque titles into one video.
    :param mp4_path: Where to write the .mp4
    :param sequence: The sequence, as generated by main().
    The sequences have {"title":"text"} and {"video":"path"} entries.
    :param output_width: Used to control the output resolution. Video is scaled to match output_width.
    Use -1 to use default.
    :return: None
    """

    fps, resolution = get_fps_and_resolution(sequence)

    if output_width > 0:
        output_height = int(round(resolution[1] * output_width / resolution[0]))
        output_resolution = (output_width, output_height)
    else:
        output_resolution = resolution

    log(f"Writing {mp4_path}")
    video = cv2.VideoWriter(
        filename=mp4_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=output_resolution,
        isColor=True,
    )

    for entry in tqdm(sequence):
        if "video" in entry:
            src_mp4_path = entry["video"]
            src_video = cv2.VideoCapture(src_mp4_path)
            while src_video.isOpened():
                ret, frame = src_video.read()
                if ret:
                    frame = scale_image(frame, output_resolution)
                    video.write(frame)
                else:
                    break
            src_video.release()
        else:
            background_color = (11, 58, 182)
            text = entry["title"]
            text_image, _width, _remaining_text = get_text_image(
                text,
                int(0.9 * resolution[0]),
                int(resolution[1] / 10),
                background_color=background_color,
                text_color=(255, 255, 255),
            )
            text_image = crop_text_image(text_image)
            title_frame = single_color_image((resolution[1], resolution[0], 3), background_color)
            h, w, _ = text_image.shape
            x = (title_frame.shape[1] - w) // 2
            y = (title_frame.shape[0] - h) // 2
            title_frame[y : y + h, x : x + w, :] = text_image
            title_frame = scale_image(title_frame, output_resolution)
            for _ in range(int(round(2 * fps))):
                video.write(np.flip(title_frame, axis=2))

    video.release()
    log(f"Written {mp4_path}", info=True)


def get_next_frame(grid_x, grid_y, tile_width, tile_height, active_video_grid, videos_path_grid):
    """
    Gets the next frame for the grid-of-videos mode.
    :param grid_x: X-location in video grid
    :param grid_y: Y-location in video grid
    :param tile_width: Width of a single tile
    :param tile_height: Height of a single tile.
    :param active_video_grid: List of list of cv2.VideoCapture video objects (or None, if no video is active)
    :param videos_path_grid: Path (string) of each video in the gride.
    :return: The next frame in the video for location (grid_x, grid_y)
    """
    src_video_idx, src_video = active_video_grid[grid_y][grid_x]

    if src_video is None:
        paths = videos_path_grid[grid_y][grid_x]
        if len(paths) == 0:
            return None
        src_video_idx = (src_video_idx + 1) % len(paths)
        src_mp4_path = paths[src_video_idx]
        src_video = cv2.VideoCapture(src_mp4_path)
        active_video_grid[grid_y][grid_x] = (src_video_idx, src_video)

    valid, frame = src_video.read()
    if valid:
        if frame.shape[0:2] != (tile_height, tile_width):
            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
        return np.flip(frame, axis=2)
    if src_video:
        src_video.release()
    active_video_grid[grid_y][grid_x] = (src_video_idx, None)
    return get_next_frame(
        grid_x, grid_y, tile_width, tile_height, active_video_grid, videos_path_grid
    )


def compose_grid(
    mp4_path: str,
    sequence: List[Dict],
    grid_width: int,
    grid_height: int,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    num_frames: int = 100,
    output_width: int = -1,
):
    """
    Composes all videos (or as many that will fit) into a single video which displays a grid
    :param mp4_path: Where to write the .mp4
    :param sequence: The sequence, as generated by main().
    The sequences have {"title":"text"} and {"video":"path"} entries.
    :param grid_width: Number of videos in grid, horizontally.
    :param grid_height: Number of videos in grid, vertically.
    :param background_color: Color of background.
    :param text_color: Color of text.
    :param num_frames: Number of frames to output in video.
    :param output_width: Used to control the output resolution. Video is scaled to match output_width.
    Use -1 to use default.
    """

    # create a grids
    videos_path_grid = [[[] for _ in range(grid_width)] for _ in range(grid_height)]
    videos_title_grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    active_video_grid = [[(-1, None) for _ in range(grid_width)] for _ in range(grid_height)]

    # put the videos into the grid
    grid_x = grid_y = 0
    for entry in sequence:
        if "video" in entry:
            src_mp4_path = entry["video"]
            videos_path_grid[grid_y][grid_x].append(src_mp4_path)
        else:
            if len(videos_path_grid[grid_y][grid_x]) > 0:
                grid_x += 1
                if grid_x >= grid_width:
                    grid_x = 0
                    grid_y += 1
                    if grid_y >= grid_height:
                        log("Warning: too many videos to fit into the grid")
                        break
            if "title" in entry:
                title = entry["title"]
                if title.startswith("Task: "):
                    title = title[len("Task: ") :]
                title = title.replace("_", " ")
                videos_title_grid[grid_y][grid_x] = title

    # determine the size of the output video. For now hard-code to reduce the size
    # by int(round(grid_height + grid_width)/2))
    fps, vresolution = get_fps_and_resolution(sequence)
    scale_factor = 1  # 2.0 / (grid_height + grid_width)
    frame_width = tile_width = int(round(vresolution[0] * scale_factor))
    frame_height = int(round(vresolution[1] * scale_factor))
    text_height = int(round(frame_height / 10))
    tile_height = text_height + frame_height
    space_width = space_height = min(int(round(tile_width / 10)), int(round(tile_height / 10)))
    resolution = output_resolution = [
        (tile_width + space_width) * grid_width,
        (tile_height + space_height) * grid_height,
    ]

    if output_width > 0:
        output_height = int(round(output_resolution[1] * output_width / output_resolution[0]))
        output_resolution = (output_width, output_height)

    log(f"Writing {mp4_path}. resolution: {output_resolution} fps:{fps}")
    video = cv2.VideoWriter(
        filename=mp4_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=output_resolution,
        isColor=True,
    )

    # write frames to video
    for _frame_idx in tqdm(range(num_frames)):
        frame = single_color_image((resolution[1], resolution[0], 3), background_color)

        for grid_y in range(grid_height):
            for grid_x in range(grid_width):
                # Write video to each individual tile
                tile_x = grid_x * tile_width + int((grid_x + 0.5) * space_width)
                tile_y = grid_y * tile_height + int((grid_y + 0.5) * space_height)

                title = videos_title_grid[grid_y][grid_x]
                if not title is None:
                    title_image, _width, _remaining_words = get_text_image(
                        title,
                        tile_width,
                        text_height,
                        background_color=background_color,
                        text_color=text_color,
                    )
                    title_image = crop_text_image(title_image)
                    offset_x = (tile_width - title_image.shape[1]) // 2
                    frame[
                        tile_y : tile_y + title_image.shape[0],
                        tile_x + offset_x : tile_x + offset_x + title_image.shape[1],
                        :,
                    ] = title_image

                vframe = get_next_frame(
                    grid_x, grid_y, frame_width, frame_height, active_video_grid, videos_path_grid
                )
                if not vframe is None:
                    frame[
                        tile_y + text_height : (tile_y + vframe.shape[0] + text_height),
                        tile_x : (tile_x + vframe.shape[1]),
                        :,
                    ] = vframe

        frame = scale_image(frame, output_resolution)
        video.write(np.flip(frame, axis=2))

    video.release()
    log(f"Written {mp4_path}")


def filter_tasks_from_clargs(tasks, args):
    """
    Uses the command-line --tasks and --levels, filters the tasks and levels to
    what the user requested.
    :param tasks: all the (pre-defined) tasks. I.e., the global variable task_specification.
    :param args: Command-line arguments.
    :return: Tasks, but filtered.
    """

    # filter the levels:
    if args.levels:
        filtered_tasks = {}
        for (level_description, level_name), level_content in tasks.items():
            for args_level in args.levels:
                if args_level in level_description:
                    filtered_tasks[(level_description, level_name)] = copy.deepcopy(level_content)
    else:
        filtered_tasks = copy.deepcopy(tasks)

    # filter the tasks
    if args.tasks:
        tasks = filtered_tasks
        filtered_tasks = {}
        for (level_description, level_name), level in tasks.items():
            filtered = []
            for name, task, task_args, seed in level:
                for args_task in args.tasks:
                    if args_task == name:
                        filtered.append((name, task, task_args, seed))

            if len(filtered) > 0:
                filtered_tasks[(level_description, level_name)] = filtered

    return filtered_tasks


def parse_args():
    """
    :return: argparser for this script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        default=None,
        help="Whether to the videos (existing results will be overwritten)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="xarm6_vacuum",
        help="Robot ('xarm6_vacuum' or 'panda_vacuum')",
    )
    parser.add_argument("--pretty-video", action="store_true", help="Whether to save pretty videos")
    parser.add_argument(
        "--compose-grid",
        action="store_true",
        help="Compose the resulting videos into a strip (one per level)",
    )
    parser.add_argument(
        "--grid-width", type=int, default=6, help="How many videos in grid, horizontally?"
    )
    parser.add_argument(
        "--grid-height", type=int, default=4, help="How many videos in grid, vertically?"
    )
    parser.add_argument(
        "--grid-num-frames", type=int, default=100, help="How many frames in the grid video?"
    )
    parser.add_argument(
        "--compose-sequence",
        action="store_true",
        help="Compose the resulting videos into a sequence",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=-1,
        help="Width of composed sequence and/or grid video",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=-1,
        help="Maximum number of seeds of a single task to run",
    )
    parser.add_argument(
        "--shader-dir",
        type=str,
        default="ibl",
        help="Shader directory. 'ibl' image based or 'rt' for ray-traced",
    )
    parser.add_argument(
        "--rt-samples-per-pixel",
        type=int,
        default=64,
        help="How many ray-tracing samples per pixel (if --shader-dir is 'rt')",
    )
    parser.add_argument(
        "--enable-sapien-viewer",
        action="store_true",
        help="Display a real-time visualization of the environment",
    )
    parser.add_argument(
        "--levels",
        action="append",
        type=str,
        nargs="+",
        default=None,
        help="To explicitly override what levels to visualize. Levels can be identified by a digit like '0', '1', '2'.",
    )
    parser.add_argument(
        "--tasks",
        action="append",
        type=str,
        nargs="+",
        default=None,
        help="To explicitly override what tasks to visualize. The name must match the nice_name of a task in the task_specification list",
    )

    parser.add_argument(
        "--prompt-visualization",
        type=str,
        default=PromptVisualizer.MULTI_MODAL,
        help=f"Prompt visualization mode: {PromptVisualizer.NATURAL_LANGUAGE}, "
        f"{PromptVisualizer.MULTI_MODAL} or {PromptVisualizer.RAW}",
    )
    parser.add_argument(
        "--action-label-visualization",
        type=str,
        default=PromptVisualizer.NATURAL_LANGUAGE,
        help=f"Action label visualization mode: '{PromptVisualizer.NATURAL_LANGUAGE}', "
        f"'{PromptVisualizer.MULTI_MODAL}', '{PromptVisualizer.RAW}', 'llava_python'",
    )

    args = parser.parse_args()

    if not args.tasks is None:
        args.tasks = [task for tasks in args.tasks for task in tasks]
    if not args.levels is None:
        args.levels = [level for levels in args.levels for level in levels]

    return args


def main(args: argparse.Namespace):
    """
    Generates data and visualizes all tasks, or a selection thereof
    :param args: Command-line arguments.
    :return: None
    """
    output_path = args.output_dir
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    global task_specification
    tasks = filter_tasks_from_clargs(task_specification, args)

    # create a list of stuff to be shown in sequnce
    sequence = [{"title": "ClevrSkills task suite"}]

    for (level_description, level_name), level in tasks.items():
        sequence.append({"title": level_description})
        for name, task, task_args, seeds in level:
            sequence.append({"title": f"Task: {name.replace('_', ' ')}"})
            if task is None:  # task name might be different from name, but is the same by default
                task = name
            task_output_path = os.path.join(output_path, level_name, name)
            if os.path.isdir(task_output_path):
                shutil.rmtree(output_path)
            os.makedirs(task_output_path)

            # Create the args:
            oracle_args = get_default_oracle_args(task_output_path, args)
            oracle_args.task = task
            oracle_args.task_args = task_args
            oracle_args.max_time_per_step = 0.5

            if isinstance(seeds, int):
                seeds = [seeds]

            if args.max_seeds > 0:
                seeds = seeds[: args.max_seeds]

            for video_idx, seed in enumerate(seeds):
                oracle_args.seed = seed
                recordings = clevr_skills_oracle.main(oracle_args, proc_id=0, num_procs=1)
                src_path = os.path.join(recordings[0], "video.mp4")
                dst_path = os.path.join(task_output_path, f"video_{video_idx}_seed_{seed}.mp4")
                os.rename(src_path, dst_path)
                sequence.append({"video": dst_path})

    if args.compose_sequence:
        mp4_path = os.path.join(output_path, "sequence.mp4")
        compose_sequence(mp4_path, sequence, output_width=args.output_width)
    if args.compose_grid:
        mp4_path = os.path.join(output_path, "grid.mp4")
        compose_grid(
            mp4_path,
            sequence,
            args.grid_width,
            args.grid_height,
            num_frames=args.grid_num_frames,
            output_width=args.output_width,
        )

    clear_temp_dir()


if __name__ == "__main__":
    args = parse_args()
    main(args)
