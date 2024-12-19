# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import bz2
import glob
import json
import multiprocessing as mp
import os
import pickle
import time
import traceback
from copy import deepcopy
from typing import Dict, List, Optional

import gymnasium as gym
import imageio
import numpy as np
from tqdm.auto import tqdm

from clevr_skills.solvers.solver import Solver
from clevr_skills.utils.logger import log, setup_logger
from clevr_skills.utils.record_env import RecordEnv
from clevr_skills.utils.record_env_pretty import RecordPrettyEnv
from clevr_skills.utils.temp_dir import clear_temp_dir
from clevr_skills.utils.visualize_prompt import PromptVisualizer

MAX_STEPS = 1000


def discretize_actions(actions, num_bins=256):
    """
    When discretize_actions is set to true from command-line, discretizes the actions.
    Assumes that action values range from -1 to +1
    :param actions: continuous action array of dimension Nx7
    :param num_bins: number of bins to discretize the actions.
    :return: actions, discretized.
    """
    actions = (actions + 1) / 2.0
    actions = actions * num_bins
    actions = np.round(actions)

    actions = actions / num_bins

    return actions


def save_keystep(eps_path, info, keysteps_idx):
    """
    Save keysteps to file.
    :param eps_path: path to the episode directory
    :param info: episode info which holds the current keystep images
    :param keysteps_idx: current keystep index
    :return:
    """
    rimg = info["render_img"]
    bimg = info["base_img"]

    if not eps_path is None:
        os.makedirs(os.path.join(eps_path, "keysteps"), exist_ok=True)
        imageio.imwrite(os.path.join(eps_path, f"keysteps/{keysteps_idx}_rimg.jpg"), rimg)
        imageio.imwrite(os.path.join(eps_path, f"keysteps/{keysteps_idx}_bimg.jpg"), bimg)


def get_action_trace(solver, check: bool = True):
    """
    :param solver: The top-level solver
    :param check: Whether to perform a check that only a single solver in the hierarchy emitted the action action.
    :return: The actions trace for the most recent step the solver took.
    """
    trace = (
        []
        if solver is None
        else [deepcopy(solver.action_state)] + get_action_trace(solver._sub_solver)
    )
    if check:
        assert len(trace) == 0 or np.sum(list(t["action_emitted"] for t in trace)) == 1, (
            f"A single solver in the hierarchy must emit an action; "
            f"instead found that {np.sum(t['action_emitted'] for t in trace)} solvers "
            "reported emitting an action"
        )
    return trace


def save_action_trace(episode_action_trace, eps_path):
    """
    Save action trace to file.
    :param episode_action_trace: action trace for entire episode.
    :param eps_path: path to episode directory.
    :return: NNone
    """
    with bz2.BZ2File(os.path.join(eps_path, "action_trace.pickle.bz2"), "w") as file:
        pickle.dump(episode_action_trace, file)


def get_render_config(args: argparse.Namespace) -> Dict:
    """
    Converts command-line arguments to a render configuration that can be passed to ManiSkill2
    :param args: Command-line arguments.
    :return: Render configuration (dictionary).
    """
    if args.shader_dir == "rt":
        return {
            "rt_samples_per_pixel": args.rt_samples_per_pixel,
            "rt_max_path_depth": 4,
            "rt_use_denoiser": True,  # note: this setting requires particular class of GPU & driver
        }
    render_config = {}
    return render_config


def wrap_record_env(env, args: argparse.Namespace, proc_id: int, num_procs: int = 0):
    """
    Wraps ens in RecordEnv (or RecordPrettyEnv)
    :param env: The environment to be wrapped.
    :param args: Command-line arguments.
    :param proc_id: Process-id (when using multiple processes to generate trajectories in parallel).
    :param num_procs: Number of processes (when using multiple processes to generate trajectories in parallel).
    :return: Wrapped environment.
    """
    suffix = f"{env.obs_mode}.{env.control_mode}"
    if num_procs > 1:
        traj_name = f"{args.task}_{suffix}.{proc_id}"
    else:
        traj_name = f"{args.task}_{suffix}"
    if args.enable_sapien_viewer and num_procs == 1:
        env.render_human()  # calling this _after_ RecordEpisode has been
        # called causes a segfault on my machine
    kwargs = (
        {
            "prompt_visualization_mode": args.prompt_visualization,
            "action_label_visualization_mode": args.action_label_visualization,
        }
        if args.pretty_video
        else {}
    )
    record_env_class = RecordPrettyEnv if args.pretty_video else RecordEnv
    env = record_env_class(
        env,
        args.record_dir,
        save_video=args.save_video,
        info_on_video=args.info_on_video,
        trajectory_name=traj_name,
        save_depth=args.save_depth,
        save_segmentation=args.save_segmentation,
        save_on_reset=False,
        **kwargs,
    )
    return env


def generate_episode(
    args: argparse.Namespace, env, ue, episode_idx, num_procs, enable_viewer
) -> bool:
    """
    Generates the data for a single episode.
    :param args: Command-line arguments
    :param env: The environment (already wrapped by wrap_record_env)
    :param ue: The unwrapped environment.
    :param episode_idx: Index of episode (used for random seed)
    :param num_procs: Not used
    :param enable_viewer: Whether to enable sapien viewer.
    :return: boolean, success or not
    """
    eps_path = None
    ep_seed = args.seed + episode_idx

    # If variant is not 0, append it to the traj name
    variant = env.task_args["variant"] if "variant" in env.task_args else 0
    traj_name = f"traj_{ep_seed}" if variant == 0 else f"traj_{ep_seed}_variant_{variant}"

    if not args.record_dir is None:
        eps_path = os.path.join(args.record_dir, traj_name)
        if os.path.exists(eps_path):
            return

    log(f"RESETTING Task {args.task} with SEED: {ep_seed}", info=True)
    obs, _ = env.reset(seed=ep_seed, options={"record_dir": eps_path, "reconfigure": True})
    solver = Solver(ue, ks_wrobot=args.ks_wrobot)

    eps_info = ue._get_eps_info()

    log(f"Textual Description: {eps_info['textual_description']}", info=True)
    log(f"Textures: {eps_info['textures']}")

    if eps_path:
        os.makedirs(eps_path, exist_ok=True)
        with open(os.path.join(eps_path, "info.json"), "w", encoding="utf8") as fp:
            for k in eps_info["textures"]:
                if not isinstance(eps_info["textures"][k][1], tuple):
                    eps_info["textures"][k] = (
                        eps_info["textures"][k][0],
                        str(eps_info["textures"][k][1]),
                    )
            json.dump(eps_info, fp, indent=2)

    keysteps_idx = 0
    num_steps_done = 0

    episode_start_time = time.time()
    episode_action_trace = []
    last_saved_keystep_idx = -1
    for step_idx in range(MAX_STEPS):
        action, extra_info = solver.step(obs)

        episode_action_trace.append(get_action_trace(solver))

        if extra_info["keystep"]:
            log(f"Save keystep at {step_idx}")
            save_keystep(eps_path, info=extra_info, keysteps_idx=keysteps_idx)
            last_saved_keystep_idx = step_idx
            keysteps_idx += 1

        # discretize actions
        if args.discretize_actions:
            action = discretize_actions(action) * 2 - 1

        if args.record_dir:
            obs, _reward, done, _truncated, info = env.step(action, extra_info)
        else:
            obs, _reward, done, _truncated, info = env.step(action)
        if enable_viewer:
            env.render_human()
        if done:
            num_steps_done += 1
        if num_steps_done > args.num_steps_done:
            log("Done!")
            break

        if step_idx >= 20 and args.max_time_per_step > 0:
            time_per_step = (time.time() - episode_start_time) / (step_idx + 1)
            if time_per_step > args.max_time_per_step:
                log(
                    f"At step {step_idx}, the average time per step is "
                    f"{time_per_step:.2f}, which is more than the limit {args.max_time_per_step}."
                )
                log("Terminating episode")
                break

    if info["success"] and last_saved_keystep_idx not in [step_idx, step_idx - 1]:
        # saving the final keystep
        action, extra_info = solver.step(obs)
        save_keystep(eps_path, info=extra_info, keysteps_idx=keysteps_idx)

    if args.record_dir:
        save_action_trace(episode_action_trace, eps_path)
        env.flush_trajectory()
        env.flush_video()

    env.unwrapped.close_viewer()  # otherwise viewer won't close under certain conditions

    return info["success"]


def main(args, proc_id: int = 0, num_procs: int = 1):
    """
    This main() provides an oracle policy ClevrSkills tasks.
    :param args: Command-line args
    :param proc_id: Process-id (when using multiple processes to generate trajectories in parallel).
    :param num_procs: Number of processes (when using multiple processes to generate trajectories in parallel).
    :return: List of paths with recordings.
    """
    assert num_procs >= 1
    setup_logger(args.log_path)
    task_args = eval(args.task_args) if isinstance(args.task_args, str) else args.task_args

    render_config = get_render_config(args)

    kwargs = {}
    if args.tabletop_texture:
        kwargs["tabletop_texture"] = args.tabletop_texture
    if args.floor_texture:
        kwargs["floor_texture"] = args.floor_texture

    env = gym.make(
        "ClevrSkills-v0",
        obs_mode="rgbd",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        robot=args.robot,
        ego_centric_camera=args.ego_centric_camera,
        task=args.task,
        strip_eval=False,
        shader_dir=args.shader_dir,
        render_config=render_config,
        enable_shadow=not args.disable_shadow,
        task_args=task_args,
        render_mode="rgb_array",
        **kwargs,
    )

    enable_viewer = args.enable_sapien_viewer

    if args.record_dir:
        env = wrap_record_env(env, args, proc_id, num_procs)

    ue: gym.Env = env.unwrapped

    unsuccessful_eps = []
    episode_idx = proc_id
    valid_eps = 0
    num_episodes = int(np.ceil(args.num_episodes / num_procs))
    recordings = []
    while valid_eps < num_episodes:

        success = False
        try:
            success = generate_episode(
                args, env, ue, episode_idx, num_procs, enable_viewer=enable_viewer
            )
            if success and args.record_dir:
                recordings.append(env._episode_info["reset_kwargs"]["options"]["record_dir"])
        except Exception as e:
            log(
                f"Task={args.task}, Seed={args.seed + episode_idx} failed with "
                f"{traceback.format_exception(e)}"
            )
            traceback.print_exception(e)

        episode_idx += num_procs
        if success:
            valid_eps += 1
        else:
            unsuccessful_eps.append(args.seed + episode_idx)

    if args.record_dir:
        np.save(f"{args.record_dir}/unsuccessful_eps_{proc_id}.npy", unsuccessful_eps)

    clear_temp_dir()

    return recordings


def get_args_parser():
    """
    :return: Argument parser for oracle.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot", type=str, default="xarm6_vacuum", help="Robot ('xarm6_vacuum' or 'panda_vacuum')"
    )
    parser.add_argument(
        "--ego-centric-camera",
        action="store_true",
        help="When set, the base camera is ego-centric (mounted on the robot)",
    )
    parser.add_argument("--num-episodes", type=int, default=100, help="How many episodes to run.")
    parser.add_argument(
        "--num-steps-done",
        type=int,
        default=0,
        help=(
            "How many steps the environment should be `done` before resetting to "
            "the next episode."
        ),
    )
    parser.add_argument(
        "--max-time-per-step",
        type=float,
        default=-1.0,
        help="Episode is terminated if it take too much time per step.",
    )
    parser.add_argument(
        "--enable-sapien-viewer",
        action="store_true",
        help="Display a real-time visualization of the environment",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="When specified, recordings of the trajectories will be stored in this directory",
    )
    parser.add_argument("--task", type=str, default="SingleStack", help="Name of task")
    parser.add_argument(
        "--task-args",
        type=str,
        default="{}",
        help="kwargs of the task. Use a string like \"@{'num_actors':4}\"",
    )
    parser.add_argument(
        "--render-mode", type=str, default="cameras", help="Render mode for recording"
    )
    parser.add_argument(
        "--shader-dir",
        type=str,
        default="ibl",
        help="Shader directory. 'ibl' image based or 'rt' for ray-traced",
    )
    parser.add_argument(
        "--disable-shadow", action="store_true", help="Disable shadows in IBL rendering mode"
    )
    parser.add_argument(
        "--rt-samples-per-pixel",
        type=int,
        default=32,
        help="How many ray-tracing samples per pixel (if --shader-dir is 'rt')",
    )
    parser.add_argument(
        "--save-imgs-as-jpegs",
        action="store_true",
        help="Where to save images as jpegs or inside h5",
    )
    parser.add_argument("--save-video", action="store_true", help="Whether to save videos")
    parser.add_argument("--pretty-video", action="store_true", help="Whether to save pretty videos")
    parser.add_argument(
        "--info-on-video", action="store_true", help="Write env info on top of video"
    )
    parser.add_argument(
        "--discretize-actions", action="store_true", help="Discretize the actions for LLMs"
    )
    parser.add_argument(
        "--save-depth", action="store_true", help="Save the depth images as a npy file"
    )
    parser.add_argument(
        "--save-segmentation", action="store_true", help="Save the segmentations as a npy file"
    )
    parser.add_argument("--ks-wrobot", action="store_true", help="Save the keystep with the robot")

    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to run")
    parser.add_argument("--log-path", type=str, default="logs/debug.log")

    parser.add_argument(
        "--floor-texture", type=str, default=None, help="Path to texture for the floor."
    )
    parser.add_argument(
        "--floor-color",
        action="append",
        type=float,
        nargs="+",
        default=None,
        help="Color of the floor (if --floor-texture is not specified). Default color is dark green.",
    )
    parser.add_argument(
        "--tabletop-texture",
        type=str,
        default=None,
        help="Path to texture for the tabletop. Default is a wood texture.",
    )
    parser.add_argument(
        "--tabletop-color",
        action="append",
        type=float,
        nargs="+",
        default=None,
        help="Color of the tabletop (if --tabletop-texture is not specified).",
    )

    # help=argparse.SUPPRESS means: these options will not be visible on the command line help
    parser.add_argument(
        "--prompt-visualization",
        type=str,
        default=PromptVisualizer.MULTI_MODAL,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--action-label-visualization",
        type=str,
        default=PromptVisualizer.NATURAL_LANGUAGE,
        help=argparse.SUPPRESS,
    )

    return parser


if __name__ == "__main__":

    def parse_args() -> argparse.Namespace:
        """
        :return: Command-line arguments, parsed
        """

        def sanitize_color(color: Optional[List[List[float]]]) -> Optional[List[float]]:
            """
            Input sanitation: makes sure that the color is a list of 4 floats (or None)
            """
            if color is None:
                return color
            sc = [0.0, 0.0, 0.0, 1.0]
            color = (np.clip(color[0], 0.0, 1.0)[:4]).astype(np.float32)
            for idx, value in enumerate(color):
                sc[idx] = value
            return sc

        parser = get_args_parser()
        args, _opts = parser.parse_known_args()

        args.tabletop_color = sanitize_color(args.tabletop_color)
        args.floor_color = sanitize_color(args.floor_color)
        if args.tabletop_texture is None:
            args.tabletop_texture = args.tabletop_color
        if args.floor_texture is None:
            args.floor_texture = args.floor_color
        return args

    args = parse_args()

    if args.num_procs > 1:
        with mp.Pool(args.num_procs) as pool:
            proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
            res = pool.starmap(main, proc_args)

        if args.record_dir:
            # removing empty h5 files
            res = glob.glob(os.path.join(args.record_dir, "*.h5"))
            for h5_path in res:
                tqdm.write(f"Remove {h5_path}")
                os.remove(h5_path)
            # merging unsuccessful eps
            us_eps_paths = sorted(glob.glob(os.path.join(args.record_dir, "unsuccessful_eps*.npy")))
            unsuccessful_eps = []
            for us_path in us_eps_paths:
                unsuccessful_eps.extend(np.load(us_path).tolist())
            np.save(os.path.join(args.record_dir, "unsuccessful_eps.npy"), unsuccessful_eps)
            for us_path in us_eps_paths:
                tqdm.write(f"Remove {us_path}")
                os.remove(us_path)
    else:
        main(args, proc_id=0, num_procs=1)
