# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
import tempfile

import numpy as np

from clevr_skills import clevr_skills_oracle
from clevr_skills.utils.load_traj import find_trajectories

default_args = argparse.Namespace(
    robot="panda_vacuum",
    ego_centric_camera=False,
    num_episodes=1,
    num_steps_done=10,
    max_time_per_step=-1.0,
    enable_sapien_viewer=False,
    seed=2,
    record_dir=None,
    task="Pick",
    task_args="{'num_actors': 1, 'sample_num_actors':False}",
    render_mode="cameras",
    shader_dir="ibl",
    disable_shadow=False,
    rt_samples_per_pixel=32,
    save_imgs_as_jpegs=False,
    save_video=False,
    pretty_video=False,
    info_on_video=False,
    discretize_actions=False,
    save_depth=False,
    save_segmentation=False,
    ks_wrobot=False,
    floor_texture=None,
    floor_color=None,
    tabletop_texture=None,
    tabletop_color=None,
    log_path="logs/debug.log",
    prompt_visualization="multi_modal",
    action_label_visualization="natural_language",
)


def test_clevr_skill_oracle_pick():
    global default_args
    for robot in "panda_vacuum", "xarm6_vacuum":
        with tempfile.TemporaryDirectory() as temp_path:
            args = copy.deepcopy(default_args)
            args.robot = robot
            args.record_dir = temp_path

            clevr_skills_oracle.main(args)
            traj = find_trajectories(temp_path, require_success=True)
            assert len(traj) >= 1


def test_clevr_skill_oracle_swap():
    global default_args
    with tempfile.TemporaryDirectory() as temp_path:
        args = copy.deepcopy(default_args)
        args.record_dir = temp_path
        args.robot = "xarm6_vacuum"
        args.task = "Swap"
        args.task_args = "{'num_actors': 2, 'sample_num_actors':False, 'push': True}"
        args.seed = 2

        clevr_skills_oracle.main(args)
        traj = find_trajectories(temp_path, require_success=True)
        assert len(traj) >= 1


def test_clevr_skill_oracle_balance_scale():
    global default_args
    with tempfile.TemporaryDirectory() as temp_path:
        args = copy.deepcopy(default_args)
        args.ego_centrc_camera = True
        args.record_dir = temp_path
        args.task = "BalanceScale"
        args.task_args = "{'num_actors': 2, 'sample_num_actors':False}"

        clevr_skills_oracle.main(args)
        traj = find_trajectories(temp_path, require_success=True)
        assert len(traj) >= 1


class DummyOracleEnv:
    task_args = {}

    def __init__(self):
        self.unwrapped = self
        self._episode_info = {"reset_kwargs": {"options": {"record_dir": ""}}}


def test_unsuccessful_eps_records_failed_seed(monkeypatch):
    args = copy.deepcopy(default_args)
    args.record_dir = tempfile.mkdtemp()
    args.num_episodes = 2
    args.seed = 42

    env = DummyOracleEnv()
    outcomes = iter([False, True])

    def fake_generate_episode(call_args, call_env, _ue, episode_idx, _num_procs, enable_viewer):
        del _ue, _num_procs, enable_viewer
        success = next(outcomes)
        call_env._episode_info["reset_kwargs"]["options"][
            "record_dir"
        ] = f"{call_args.record_dir}/traj_{call_args.seed + episode_idx}"
        return success

    monkeypatch.setattr(clevr_skills_oracle.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(clevr_skills_oracle, "wrap_record_env", lambda env, *args: env)
    monkeypatch.setattr(clevr_skills_oracle, "generate_episode", fake_generate_episode)
    monkeypatch.setattr(clevr_skills_oracle, "clear_temp_dir", lambda: None)

    clevr_skills_oracle.main(args, proc_id=0, num_procs=2)

    unsuccessful = np.load(f"{args.record_dir}/unsuccessful_eps_0.npy").tolist()
    assert unsuccessful == [42]


def test_merge_unsuccessful_eps_ignores_existing_merged_file(tmp_path):
    np.save(tmp_path / "unsuccessful_eps.npy", [999])
    np.save(tmp_path / "unsuccessful_eps_0.npy", [46])
    np.save(tmp_path / "unsuccessful_eps_1.npy", [44])

    merged = clevr_skills_oracle.merge_unsuccessful_eps(str(tmp_path))

    assert merged == [44, 46]
    assert np.load(tmp_path / "unsuccessful_eps.npy").tolist() == [44, 46]
    assert not (tmp_path / "unsuccessful_eps_0.npy").exists()
    assert not (tmp_path / "unsuccessful_eps_1.npy").exists()
