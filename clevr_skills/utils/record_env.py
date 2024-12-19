# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os

import cv2
import imageio
import numpy as np
from gymnasium import spaces
from mani_skill2.utils.common import flatten_dict_keys
from mani_skill2.utils.io_utils import dump_json
from mani_skill2.utils.visualization.misc import images_to_video
from mani_skill2.utils.wrappers import RecordEpisode

from clevr_skills.utils.logger import log


def make_video(images, path, fps=20):
    """
    Uses imageio to save an video.
    :param images: Numpy array of images (sequence x h x w x 3)
    :param path: Output path
    :param fps: Frame rate.
    :return: None
    """
    writer = imageio.get_writer(path, fps=fps, quality=10, pixelformat="yuv444p")
    for img in images:
        writer.append_data(img)
    writer.close()


class RecordEnv(RecordEpisode):
    """Record trajectories or videos for episodes.
    The trajectories are stored in output_dir with each obs forming an independent
    npy file.
    And the images are saved as videos.

    """

    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        clean_on_close=True,
        save_depth=False,
        save_segmentation=False,
    ):
        """
        :param env: The env-to-be-wrapped. Also called inner environment in the comments below.
        :param output_dir: Where output should be written.
        :param save_trajectory:  whether to save trajectory
        :param trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        :param save_video: whether to save video
        :param info_on_video: Overlay information on the video?
        :param save_on_reset: whether to save the previous trajectory automatically when resetting.
            If True, the trajectory with empty transition will be ignored automatically.
        :param clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
        :param save_depth: Save depth images?
        :param save_segmentation: Save segmentation images?
        """
        super().__init__(
            env,
            output_dir,
            save_trajectory,
            trajectory_name,
            save_video,
            info_on_video,
            save_on_reset,
            clean_on_close,
        )

        self.save_depth = save_depth
        self.save_segmentation = save_segmentation
        self.extra_info = {}

    def step(self, action, extra_info={}):
        """
        Takes an environment step.
        :param action: Passed on to inner environment.
        :param extra_info: Extra info to be stored with the episode (not passed to inner environment).
        :return: The observation (from inner environment).
        """
        self.extra_info[len(self.extra_info) + 1] = extra_info
        return super().step(action)

    def reset(self, **kwargs):
        """
        Reset the environment; call is passed on to inner environment.
        :param kwargs: Passed on to inner environment.
        :return: The observation (from inner environment).
        """
        self.extra_info = {}
        return super().reset(**kwargs)

    def _write_action_labels(self, images):
        """
        Internal function; writes action labels on images.
        :param images:
        :return:
        """
        for idx, img in enumerate(images):
            if idx in self.extra_info and "act_label" in self.extra_info[idx]:
                cv2.putText(
                    img,
                    "Mid: " + self.extra_info[idx]["act_label"]["mid_act_label"],
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                )
                cv2.putText(
                    img,
                    "Low: " + self.extra_info[idx]["act_label"]["low_act_label"],
                    (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                )
        return images

    def _save_action_labels(self, path):
        """
        Internal functions; saves the action labels.
        :param path: Output path.
        :return: None.
        """
        action_labels = {}
        for idx, value in self.extra_info.items():
            if "act_label" in value:
                action_labels[idx] = value["act_label"]
        np.save(path, action_labels)

    def decorate_images(self):
        """
        Overlays information on video images.
        :return: None
        """
        if len(self._render_images) > 0 and self.info_on_video:
            return self._write_action_labels(self._render_images)
        return self._render_images

    def flush_video(self, *args, **kwargs):
        """
        Calls this function to save the video explicitly.
        :param args: Ignored
        :param kwargs: Ignored
        :return: None
        """
        # The following base on record.py from ManiSkill2
        # It had to be duplicated because record.py does not support custom output path
        ignore_empty_transition = False
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return

        if len(self._render_images) > 0:
            self._render_images = self.decorate_images()

        traj_path = self._episode_info["reset_kwargs"]["options"]["record_dir"]
        video_name = "video"

        images_to_video(
            self._render_images,
            traj_path,
            video_name=video_name,
            fps=20,
            verbose=False,
        )

    def flush_trajectory(self, verbose=False, ignore_empty_transition=False):
        """
        Call this function to trajectory explicitly.
        :param verbose:
        :param ignore_empty_transition:
        :return:
        """
        if not self.save_trajectory or len(self._episode_data) == 0:
            return
        if ignore_empty_transition and len(self._episode_data) == 1:
            return

        traj_path = self._episode_info["reset_kwargs"]["options"]["record_dir"]

        # Observations need special processing
        obs = [x["o"] for x in self._episode_data]
        if isinstance(obs[0], dict):
            obs = [flatten_dict_keys(x) for x in obs]
            obs = {k: [x[k] for x in obs] for k in obs[0].keys()}
            obs = {k: np.stack(v) for k, v in obs.items()}
            for k, v in obs.items():

                if "rgb" in k and v.ndim == 4:
                    out_path = f"{traj_path}/{k}.mp4"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    make_video(v, out_path)
                elif "depth" in k and v.ndim in (3, 4):
                    if not self.save_depth:
                        continue
                    # skipping depth videos as don't need it for now.
                    if not np.all(np.logical_and(v >= 0, v < 2**6)):
                        raise RuntimeError(
                            f"The depth map({k}) is invalid with min({v.min()}) and max({v.max()})."
                        )
                    v = (v * (2**10)).astype(np.uint16)
                    out_path = f"{traj_path}/{k}.npy"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, v)
                elif ("seg" in k or "Seg" in k) and v.ndim in (3, 4):
                    if not self.save_segmentation:
                        continue
                    assert np.issubdtype(v.dtype, np.integer) or v.dtype == np.bool_, v.dtype
                    out_path = f"{traj_path}/{k}.npy"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, v)
                else:
                    out_path = f"{traj_path}/{k}.npy"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, v)

        elif isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
            np.save(f"{traj_path}/obs.npy", obs)
        else:
            log(obs[0])
            raise NotImplementedError(type(obs[0]))

        if len(self._episode_data) == 1:
            action_space = self.env.action_space
            assert isinstance(action_space, spaces.Box), action_space
            actions = np.empty(
                shape=(0,) + action_space.shape,
                dtype=action_space.dtype,
            )
            dones = np.empty(shape=(0,), dtype=bool)
        else:
            # Record transitions (ignore the first padded values during reset)
            actions = np.stack([x["a"] for x in self._episode_data[1:]])
            dones = np.stack([x["info"]["success"] for x in self._episode_data[1:]])
            rewards = np.stack([x["r"] for x in self._episode_data[1:]])

        # Only support array like states now
        env_states = np.stack([x["s"] for x in self._episode_data])

        # Dump
        np.save(f"{traj_path}/actions.npy", actions)
        np.save(f"{traj_path}/success.npy", dones)
        np.save(f"{traj_path}/rewards.npy", rewards)

        # save action labels
        self._save_action_labels(f"{traj_path}/action_labels.npy")

        if self.init_state_only:
            np.save(f"{traj_path}/env_init_state.npy", env_states[0])
        else:
            np.save(f"{traj_path}/env_states.npy", env_states)

        # Handle JSON
        dump_json(f"{traj_path}/ep_info.json", self._episode_info, indent=2)

        if verbose:
            log(f"Record the {self._episode_id}-th episode")
