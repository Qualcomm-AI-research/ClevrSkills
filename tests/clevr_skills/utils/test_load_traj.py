# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import copy
import tempfile

from clevr_skills import clevr_skills_oracle
from clevr_skills.utils.load_traj import Traj, find_trajectories


def test_traj():
    with tempfile.TemporaryDirectory() as temp_path:
        args = argparse.Namespace(
            robot="panda_vacuum",
            ego_centric_camera=False,
            num_episodes=1,
            num_steps_done=10,
            max_time_per_step=-1.0,
            enable_sapien_viewer=False,
            seed=2,
            record_dir=temp_path,
            task="Rearrange",
            task_args="{'num_actors': 3}",
            render_mode="cameras",
            shader_dir="ibl",
            disable_shadow=False,
            rt_samples_per_pixel=32,
            save_imgs_as_jpegs=False,
            save_video=True,
            pretty_video=False,
            info_on_video=True,
            discretize_actions=False,
            save_depth=False,
            save_segmentation=False,
            ks_wrobot=False,
            floor_texture=None,
            floor_color=None,
            tabletop_texture=None,
            tabletop_color=None,
            num_procs=1,
            log_path="logs/debug.log",
            prompt_visualization="multi_modal",
            action_label_visualization="natural_language",
        )
        clevr_skills_oracle.main(args)
        trajs = find_trajectories(temp_path, require_success=True, return_traj=True)
        assert len(trajs) >= 1
        traj = trajs[0]

        actions = traj.get_actions()
        assert len(actions.shape) == 2
        assert actions.shape[0] > 0
        assert actions.shape[1] == 7

        rewards = traj.get_rewards()
        assert len(rewards.shape) == 1
        assert rewards.shape[0] == actions.shape[0]

        success = traj.get_success()
        assert success.shape[0] == rewards.shape[0]

        actions_labels = traj.get_actions_labels()
        assert len(actions_labels) == success.shape[0]
        for key in ["mid_act_label", "low_act_label"]:
            assert key in actions_labels[0]

        info = traj.get_info()
        assert not info is None

        ep_info = traj.get_ep_info()
        assert not ep_info is None

        action_trace = traj.get_action_trace()
        assert len(action_trace) == success.shape[0]

        action_trace_labels_python = traj.get_action_trace_labels(traj.ActionTraceLabel.PYTHON)
        assert not action_trace_labels_python is None

        action_trace_labels_nat_lang = traj.get_action_trace_labels(
            traj.ActionTraceLabel.NATURAL_LANGUAGE
        )
        assert not action_trace_labels_nat_lang is None

        action_trace_solver_level = traj.get_action_trace_solver_level()
        assert len(action_trace_solver_level) == success.shape[0]

        actor_name_id = traj.get_actor_name_id()
        assert len(actor_name_id) > 0

        bounding_boxes = traj.get_bounding_boxes()
        for actor in actor_name_id:
            assert bounding_boxes[actor].shape[0] == success.shape[0] + 1
            assert bounding_boxes[actor].shape[1] == 2
            assert bounding_boxes[actor].shape[2] == 3

        visibility = traj.get_visibility()
        for actor in actor_name_id:
            assert visibility[actor].shape[0] == success.shape[0] + 1

        actor_state_raw = traj.get_actor_state()
        assert actor_state_raw.shape[0] == success.shape[0] + 1

        actor_state_dict = traj.get_actor_state(return_dict=True)
        for actor in actor_name_id:
            for key in ["pos", "ori", "vel", "ang_vel"]:
                assert actor_state_dict[actor][key].shape[0] == success.shape[0] + 1

        camera_params = {
            c.value: traj.get_camera_param(c)
            for c in [Traj.Camera.OBJECT, Traj.Camera.BASE, Traj.Camera.WRIST]
        }
        for c in [Traj.Camera.OBJECT, Traj.Camera.BASE, Traj.Camera.WRIST]:
            for key in ["cam2world_gl", "extrinsic_cv", "intrinsic_cv"]:
                assert key in camera_params[c.value]

        base_pose = traj.get_agent_base_pose()
        ee_pose = traj.get_agent_ee_pose()
        ee_velocity = traj.get_agent_ee_vel()
        qpos = traj.get_agent_qpos()
        qvel = traj.get_agent_qvel()
        vacuum_on = traj.get_agent_vacuum_on()
        vacuum_grasping = traj.get_agent_vacuum_grasping()
        vacuum_ready = traj.get_agent_vacuum_ready()

        assert base_pose.shape[0] == success.shape[0] + 1
        assert base_pose.shape[1] == 7
        assert ee_pose.shape[0] == success.shape[0] + 1
        assert ee_pose.shape[1] == 7
        assert ee_velocity.shape[0] == success.shape[0] + 1
        assert ee_velocity.shape[1] == 6
        assert qpos.shape[0] == success.shape[0] + 1
        assert qpos.shape[1] >= 6
        assert qvel.shape == qpos.shape
        assert vacuum_on.shape[0] == success.shape[0] + 1
        assert vacuum_grasping.shape[0] == success.shape[0] + 1
        assert vacuum_ready.shape[0] == success.shape[0] + 1
        assert vacuum_ready.shape[1] > 0

        keystep_base_camera_images = traj.get_keystep_images(Traj.Camera.BASE)
        assert len(keystep_base_camera_images) > 0
        assert len(keystep_base_camera_images[0].shape) == 3

        keystep_render_camera_images = traj.get_keystep_images(Traj.Camera.RENDER)
        assert len(keystep_render_camera_images) == len(keystep_base_camera_images)
        assert len(keystep_render_camera_images[0].shape) == 3

        object_images = traj.get_object_images()
        for actor in actor_name_id:
            assert actor in object_images
            assert len(object_images[actor].shape) == 3

        prompts = traj.get_multi_modal_prompts()
        assert len(prompts) > 0

        videos = {
            c: traj.get_video(c) for c in [Traj.Camera.RENDER, Traj.Camera.BASE, Traj.Camera.WRIST]
        }
        for frame_idx in range(len(actions) + 1):
            for cam_name, video in videos.items():
                ret, _image = video.read()
                assert ret, f"Could not read frame {frame_idx} of {cam_name}"
