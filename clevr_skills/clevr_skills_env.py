# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/haosulab/ManiSkill
# Copyright (c) 2024, ManiSkill Contributors, licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os
import types
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Type, Union

import mplib
import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)
from mani_skill2.utils.trimesh_utils import get_actor_meshes, get_articulation_meshes, merge_meshes
from sapien.core import Pose

from clevr_skills.agents.controllers.vacuum_controller import VacuumController
from clevr_skills.agents.robots.panda import ClevrSkillsPanda
from clevr_skills.agents.robots.xarm import XArm6Vacuum
from clevr_skills.tasks.task import get_task_class
from clevr_skills.utils import mesh_primitives
from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.controller import ee_pose_to_ee_delta_action
from clevr_skills.utils.logger import log
from clevr_skills.utils.render import get_actor_bbox_in_img_space
from clevr_skills.utils.sapien_utils import copy_contacts
from clevr_skills.utils.temp_dir import reset_temp_dir
from clevr_skills.utils.textures import VT_DIR


def check_resources_downloaded():
    """
    Checks if resources are present and logs a warning if something seems missing.
    """
    paths = [
        ("VIMA textures", "assets/vima_textures", ".jpg", 9, None),
        ("ManiSkill2 YCB models", "assets/mani_skill2_ycb/models", "0", 78, None),
        (
            "ManiSkill2 Assembly Kit models",
            "assets/assembling_kits/models/collision",
            ".obj",
            20,
            None,
        ),
        (
            "UFACTORY xArm6 robot meshes",
            "assets/descriptions/xarm6_description",
            ".stl",
            6,
            [
                "base.stl",
                "link1.stl",
                "link2.stl",
                "link3.stl",
                "link4.stl",
                "link5.stl",
                "link6.stl",
            ],
        ),
        (
            "Panda robot meshes",
            "assets/descriptions/franka_description/meshes/visual",
            ".dae",
            10,
            None,
        ),
    ]
    for resources, sub_path, filter, expected_number, expected_filenames in paths:
        path = os.path.join(os.path.split(__file__)[0], sub_path)
        files = [filename for filename in os.listdir(path) if filter in filename]
        filtered_files = (
            [filename for filename in files if filename in expected_filenames]
            if expected_filenames
            else files
        )
        if len(filtered_files) < expected_number:
            log(
                f"Warning: it looks like {resources} are missing; "
                "please double check and run download_resources.sh if needed",
                info=True,
            )
            assert False, "Please run download_resources.sh (see README.md)"


@register_env("ClevrSkills-v0", max_episode_steps=1000)
class ClevrSkillsEnv(BaseEnv):
    SUPPORTED_ROBOTS = {
        "panda_vacuum": ClevrSkillsPanda,
        "xarm6_vacuum": XArm6Vacuum,
    }
    agent: XArm6Vacuum

    def __init__(
        self,
        *args,
        robot="xarm6_vacuum",
        robot_init_qpos_noise=0.02,
        ego_centric_camera=False,
        task="SingleStackTask",
        strip_eval=True,
        arm_controller_rot_bound=None,
        arm_controller_lower=None,
        arm_controller_upper=None,
        arm_controller_pos_lower=None,
        arm_controller_pos_upper=None,
        tabletop_texture: Union[str, List[float]] = os.path.join(VT_DIR, "wood_light.png"),
        vis_table: bool = False,
        floor_texture: Optional[Union[str, List[float]]] = None,
        extra_obs: bool = False,
        task_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Creates ClevrSkillsEnv.

        :param args: Forwarded to ManiSkill2 sapien BaseEnv (superclass)
        :param robot: Which robot to instantiate: "panda_vacuum" or "xarm6_vacuum"
        :param robot_init_qpos_noise: How much to vary the joint positions, in radians.
        :param ego_centric_camera: If true, the base camera mounted close to the robot base.
        Otherwise, it is mounted looking towards the robot.
        :param task: Which task to instantiate. See visualize_all_tasks.py for a list.
        :param strip_eval: If true, task evaluate will be reduced to just a boolean success value.
        :param arm_controller_rot_bound: Use this to override the max rotational velocity in the ManiSkill2 controller.
        :param arm_controller_lower: Use this to override the lower bound on the joint angle in the ManiSkill2 controller.
        :param arm_controller_upper: Use this to override the upper bound on the joint angle in the ManiSkill2 controller.
        :param arm_controller_pos_lower: Use this to override the lower bound on the joint angle in the ManiSkill2 controller.
        :param arm_controller_pos_upper: Use this to override the upper bound on the joint angle in the ManiSkill2 controller.
        :param tabletop_texture: Controls the texture (i.e., string, path to texture) or color (list of 4 floats) of the table top.
        :param floor_texture: Controls the texture (i.e., string, path to texture) or color (list of 4 floats) of the floor.
        :param extra_obs: If true, additional observations will be returned (render image, bounding boxes, visiblity)
        :param task_args: Dictionary that is passed as arguments to the task.
        :param kwargs: Forwarded on to ManiSkill2 sapien BaseEnv (superclass)
        """
        check_resources_downloaded()

        if not floor_texture:
            floor_texture = [0.15, 0.25, 0.25, 1.0]
        if not task_args:
            task_args = {}

        robot_mapping = {"xarm": "xarm6_vacuum", "xarm6": "xarm6_vacuum"}
        robot = robot_mapping.get(robot, robot)

        self.robot_uuid = robot
        self._robot_pos = np.array([-0.562, 0, 0]) if "xarm" in robot else np.array([-0.65, 0, 0])
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.ego_centric_camera = ego_centric_camera

        self.task_args = task_args

        self.task_class = get_task_class(task)

        self._strip_eval = strip_eval
        self._ground_altitude = 0.0
        self._actor_distance: ActorDistance = None

        # Controller customizations (applied by _get_customized_agent_config)
        self._controller_customizations = {
            "arm": {
                "rot_bound": arm_controller_rot_bound,
                "pos_lower": arm_controller_pos_lower,
                "pos_upper": arm_controller_pos_upper,
                "lower": arm_controller_lower,
                "upper": arm_controller_upper,
            }
        }

        self._tabletop_texture = tabletop_texture
        self._floor_texture = floor_texture
        self._vis_table = vis_table

        self._contact_during_control_step = []
        self._contact_during_last_sim_step = []

        self.extra_obs = extra_obs
        self._return_render_image = extra_obs
        self._return_bounding_box = extra_obs
        self._return_visiblity = extra_obs
        self.task = None
        self._ground = None
        self._tabletop = None
        self._floor = None
        self.tcp = None

        super().__init__(*args, **kwargs)

    def close_viewer(self):
        """
        Closes the Sapien viewer
        :return:
        """
        super()._close_viewer()

    def _get_default_scene_config(self):
        """
        :return: Default Sapien scene config. We reduce the contact offset to 5mm (from 20mm).
        Otherwise, objects poke through thin containers (such as bowl) too easily.
        """
        scene_config = super()._get_default_scene_config()
        # default contact_offset is 0.02, allowing objects to poke through each other quite easily.
        # This will make predicates that check for (lack of) contact fail
        scene_config.contact_offset = 0.005
        return scene_config

    def reset(self, seed=None, options=None):
        """
        :param seed: If set, the seed to new episode.
        :param options: Forwarded to ManiSkill2 sapien BaseEnv.reset().
        :return: First observation of the new episode.
        """
        self.set_episode_rng(seed)

        mplib.set_global_seed(seed)
        self._actor_distance = ActorDistance(
            self._episode_rng, ground_name="ground", ground_altitude=self._ground_altitude
        )

        # disable time-consuming observations if they are not being recorded for the dataset
        # Unless the user asks for this by setting extra_obs = True
        self._return_render_image = self.extra_obs
        self._return_bounding_box = self.extra_obs
        self._return_visiblity = self.extra_obs
        if options and "record_dir" in options:
            self.task_args["record_dir"] = options["record_dir"]
            if not self.task_args["record_dir"] is None:
                self._return_render_image = True
                self._return_bounding_box = True
                self._return_visiblity = True

        result = super().reset(seed, options)

        # doing some dummy steps in the simulation to make sure objects have settled at the start
        initial_gripper_action = self.task.initial_grasp_action
        self.agent.env = self
        initial_ee_pose = self.agent.ee_link.pose
        for _ in range(25):
            # We need to actively hold the EE at the initial location, otherwise it might drift
            # when it is already holding an object
            hold_action = ee_pose_to_ee_delta_action(
                self.agent, initial_ee_pose, gripper=initial_gripper_action
            )[0]
            self.step_action(hold_action)
        result = self.get_obs(), {}

        reset_temp_dir(self._scene)

        return result

    def _clear(self):
        """
        Internal function, called by ManiSkill2 sapien BaseEnv.
        :return: None
        """
        super()._clear()
        self.task = None
        self._contact_during_control_step = []
        self._contact_during_last_sim_step = []

    def _setup_lighting(self):
        """
        Sets up lighting for scene.
        :return:
        """
        if self.bg_name is not None:
            return

        super()._setup_lighting()

    def _add_ground_pretty(self):
        """
        Creates a prettier version of the ground that consists of a central "table top"
        and an extended "floor". The color/texture of these elements is determined by constructor
        arguments table_top_texture and floor_texture
        :return: None
        """
        # Add the default Sapien ground, but don't render it
        self._ground = self._add_ground(altitude=self._ground_altitude, render=not self._vis_table)

        # Create table-top
        if self._vis_table:
            render_material = self._renderer.create_material()
            if isinstance(self._tabletop_texture, str):
                render_material.set_base_color([0.0, 0.0, 0.0, 1.0])
                render_material.set_diffuse_texture_from_file(self._tabletop_texture)
            else:
                render_material.set_base_color(list(self._tabletop_texture))
            render_material.set_metallic(0.1)
            render_material.set_specular(0.5)
            self._tabletop = mesh_primitives.add_cube_actor(
                self._scene,
                self._renderer,
                width=1.5,
                depth=1.5,
                height=0.01,
                name="ground",  # must be called "ground" (will be excluded from world point cloud)
                render_material=render_material,
                static=True,
                collision_geometry=False,
            )  # This is visual only. The actual ground will be used as collision geometry
            self._tabletop.set_pose(sapien.Pose([-0.5, 0.0, -0.005]))

            # Create extended ground
            render_material = self._renderer.create_material()
            if isinstance(self._floor_texture, str):
                render_material.set_base_color([0.0, 0.0, 0.0, 1.0])
                render_material.set_diffuse_texture_from_file(self._floor_texture)
            else:
                render_material.set_base_color(list(self._floor_texture))
            render_material.set_metallic(0.1)
            render_material.set_specular(0.5)
            self._floor = mesh_primitives.add_cube_actor(
                self._scene,
                self._renderer,
                width=100,
                depth=100,
                height=0.005,
                name="ground",  # must be called "ground" (will be excluded from world point cloud)
                render_material=render_material,
                static=True,
                collision_geometry=False,
            )  # This is visual only. The actual ground will be used as collision geometry
            self._floor.set_pose(sapien.Pose([-0.5, 0.0, -0.005]))

    def _load_actors(self):
        """
        Loads actors, sets up task
        :return: None
        """
        self._add_ground_pretty()

        self.task_class.check_task_args(self.task_class, self.task_args)
        self.task = self.task_class(self, **self.task_args)
        self.task.setup_task()

    def _configure_agent(self):
        """
        Called by ManiSkill2 sapien BaseEnv. Sets the agent configuration (the type of robot + customizations)
        :return: None
        """
        agent_cls: Type[XArm6Vacuum] = self.SUPPORTED_ROBOTS[self.robot_uuid]
        self._agent_cfg = self._get_customized_agent_config(agent_cls.get_default_config())

    def _load_agent(self):
        """
        Creates the robot (i.e., self.agent)
        :return: None
        """
        agent_cls: Type[XArm6Vacuum] = self.SUPPORTED_ROBOTS[self.robot_uuid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
        self._initialize_agent()  # call this here, otherwise the agent will be in the
        # wrong place during actor placement.

    def _initialize_agent(self):
        """
        Places the robot in the initial pose.
        ManiSkill2 defines several variants of this function
        :return: None
        """
        if self.robot_uuid == "xarm6_vacuum":
            qpos = np.array([0, 0, -np.pi / 2, 0, np.pi / 2, 0, 0, 0, 0, 0, 0])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose(self._robot_pos))

        elif self.robot_uuid == "panda_vacuum":
            qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0, 0, 0, 0, 0])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose(self._robot_pos))

        else:
            raise NotImplementedError(self.robot_uuid)

    def _initialize_agent_v1(self):
        """
        Places the robot in the initial pose.
        ManiSkill2 defines several variants of this function
        :return: None
        """
        if self.robot_uuid == "xarm7":
            qpos = np.array([0, 0, 0, np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        elif self.robot_uuid == "xarm6":
            qpos = np.array([0, 0, -np.pi / 2, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        elif self.robot_uuid == "xarm6_linear_motor":
            qpos = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uuid)

    def _register_cameras(self):
        """
        :return: A list of CameraConfigs the define the parameters of the cameras.
        """
        base_camera_pose = (
            look_at(self._robot_pos + [0.022, -0.20, 0.3], [0.1, 0.1, 0.15])
            if self.ego_centric_camera
            else look_at([0.3, -0.1, 0.3], [-0.1, 0, 0.1])
        )
        object_camera_pose = look_at([0.20, 0.20, 0.20], [0.0, 0.0, 0.0])
        base_camera_fov = (  # slight difference in fov between regular and ego, for historical reasons
            np.deg2rad(100) if self.ego_centric_camera else np.deg2rad(90)
        )
        texture_names = ("Color", "Position", "Segmentation")
        return [
            CameraConfig(
                "base_camera",
                base_camera_pose.p,
                base_camera_pose.q,
                256,
                256,
                base_camera_fov,
                0.01,
                10,
                texture_names=texture_names,
            ),
            CameraConfig(
                "object_camera", object_camera_pose.p, object_camera_pose.q, 256, 256, 1.0, 0.01, 10
            ),
        ]

    def _setup_cameras(self):
        """
        Called by ManiSkill2 sapien BaseEnv. This function applies a hack to make ensure all robot links are visible
        by hiding / unhiding them.
        :return:
        """
        super()._setup_cameras()

        # This unhides the vacuum gripper mesh (which is, sometimes, hidden for unclear reasons)
        # Must be called after the cameras and the agent have been created.
        for link in self.agent.robot.get_links():
            link.hide_visual()
            link.unhide_visual()

    def step_action(self, action):
        """
        This was function was copied literally from ManiSkill2 sapien_env.py in
        order to add the call to self._before_simulation_step()
        :param action: The action to be performed (forward to robot controller)
        """
        if action is not None:
            if isinstance(action, np.ndarray):
                self.agent.set_action(action)
            elif isinstance(action, dict):
                if action["control_mode"] != self.agent.control_mode:
                    self.agent.set_control_mode(action["control_mode"])
                self.agent.set_action(action["action"])
            else:
                raise TypeError(type(action))

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.agent.before_simulation_step()
            self._scene.step()
            self._after_simulation_step()

    def _before_simulation_step(self):
        """
        Allows custom code to run before each simulation step.
        """
        pass

    def _after_simulation_step(self):
        """
        Keeps track of contact events during simulation.
        Contacts are used to evaluate predicates.
        """
        self._contact_during_last_sim_step = copy_contacts(self._scene.get_contacts())
        self._contact_during_control_step += self._contact_during_last_sim_step

    def _before_control_step(self):
        """
        Called before every control step, used to keep track of contact between objects.
        :return:
        """
        super()._before_control_step()
        self.reset_contact()

    def _register_render_cameras(self):
        """
        Called by ManiSkill2 sapien BaseEnv to get the render camera config.
        :return:
        """
        pose = look_at([1.0, 1.0, 0.8], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        """
        Sets up the Sapien Viewer.
        """
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        """
        :return: Agent observations, which are merged into the environment step observations.
        """
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        obs["ee_pose"] = vectorize_pose(self.agent.ee_link.pose)
        obs["ee_velocity"] = np.concatenate(
            [self.agent.ee_link.velocity, self.agent.ee_link.angular_velocity]
        )

        if "gripper" in self.agent.controller.controllers and isinstance(
            self.agent.controller.controllers["gripper"], VacuumController
        ):
            vacuum_controller: VacuumController = self.agent.controller.controllers["gripper"]

            obs["vacuum_on"] = vacuum_controller.vacuum_on
            obs["vacuum_grasping"] = len(self.agent.get_grasped_actors()) > 0
            obs["vacuum_ready"] = vacuum_controller.get_suction_cup_contact()

        return obs

    def _initialize_task(self):
        """
        Calls the task to initialize itself.
        :return: None
        """
        self.task.initialize_task()

    def check_robot_static(self, thresh=0.2):
        """
        :param thresh: joint velocity threshold (radians/second) for determining if the robot is static.
        :return: True if the agent is not moving
        """
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        """
        :param kwargs: ignored
        :return: A dictionary with detailed info on the task.
        """
        task_eval = self.task.evaluate()
        if self._strip_eval:
            task_eval = {"success": task_eval["success"]}
        task_eval["task"] = self.task.get_textual_description()
        return task_eval

    def compute_dense_reward(self, info, **kwargs):
        """
        :param info: ignored
        :param kwargs:ignored
        :return: A single float; the dense reward of the task.
        """
        return self.task.compute_dense_reward()

    def _get_obs_extra(self) -> OrderedDict:
        """
        :return: a dictionary with additional observations, returned with the environment step.
        The call is basically forwarded to self._get_obs_priviledged()
        """
        obs = OrderedDict()
        if self._obs_mode in ["state", "state_dict", "rgbd", "image"]:
            obs.update(self._get_obs_priviledged())
        return obs

    def _get_obs_priviledged(self) -> OrderedDict:
        """
        Get priviledged (not meant for regular agent) observations such as actor visibility, bounding boxes, etc.
        :return: a dictionary with additional observations, returned with the environment step.
        """
        obs = OrderedDict()
        actors = self._get_task_actors()
        if len(actors) > 0:
            obs["actors"] = np.hstack([get_actor_state(actor) for actor in actors])
            obs["actor_is_visible"] = {}
            obs["actor_mesh_bbox"] = {}
            cameras = [self._cameras["base_camera"], self._cameras["hand_camera"]]
            if self._return_render_image:
                cameras = cameras + [self._render_cameras["render_camera"]]
            for camera in cameras:
                camera_name = camera.camera.name
                # The call to self._get_camera_images() assumes that
                # self._cameras["base_camera"].take_picture() has just been called.
                # This will be true at this point because _get_obs_images() is called before).
                try:
                    images = camera.get_images(take_picture=False)
                except IndexError:
                    images = camera.get_images(take_picture=True)

                if self._return_visiblity and "Segmentation" in images:
                    actor_segmentation = images["Segmentation"][
                        :, :, 1
                    ]  # 0 = mesh-level, 1 = actor-level

                    actor_is_visible = {}
                    for actor in actors:
                        if np.sum(actor_segmentation == actor.id) > 0:
                            is_visible = 1.0
                        else:
                            is_visible = 0.0
                        actor_is_visible[f"{actor.name}_{actor.id}"] = is_visible
                    obs["actor_is_visible"][camera_name] = actor_is_visible
                if self._return_bounding_box:
                    actor_mesh_bbox = {
                        f"{actor.name}_{actor.id}": get_actor_bbox_in_img_space(
                            self, actor, camera, use_3d_bounding_box=False
                        )
                        for actor in actors
                    }
                    obs["actor_mesh_bbox"][camera_name] = actor_mesh_bbox
        return obs

    def _get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: The list of actors that are involved in the current task.
        """
        return self.task.get_task_actors()

    def _get_eps_info(self):
        """
        :return: Dictionary with information about the current episode (prompt, textures).
        """
        prompts = self.task.get_prompts()
        text_desc = self.task.get_textual_description()
        textures = self.task.get_textures()

        return {"prompts": prompts, "textual_description": text_desc, "textures": textures}

    def _get_prompt_assets(self):
        return self.task.get_prompt_assets()

    def _get_obs_without_robot(self):
        """
        :return: visual observation of the scene, without the robot visible.
        This is used to generate key step images.
        """
        cur_pose = self.agent.robot.get_pose()
        temp_pose = deepcopy(cur_pose)
        temp_pose.set_p([0.0, 0.0, 1000.0])  # hide robot by placing it 1km away
        self.agent.robot.set_pose(temp_pose)

        self.update_render()

        render_camera = self._render_cameras["render_camera"]
        base_camera = self._cameras["base_camera"]
        base_camera_images = base_camera.get_images(take_picture=True)
        render_camera_images = render_camera.get_images(take_picture=True)
        render_img = np.clip(render_camera_images["Color"][:, :, 0:3] * 255, 0, 255).astype(
            np.uint8
        )
        base_img = np.clip(base_camera_images["Color"][:, :, 0:3] * 255, 0, 255).astype(np.uint8)

        self.agent.robot.set_pose(cur_pose)  # bring robot back

        return render_img, base_img

    def _get_obs_with_robot(self):
        """
        :return: visual observation of the scene, with the robot visible.
        This is used to generate key step images.
        """
        self.update_render()

        render_camera = self._render_cameras["render_camera"]
        base_camera = self._cameras["base_camera"]
        base_camera_images = base_camera.get_images(take_picture=True)
        render_camera_images = render_camera.get_images(take_picture=True)
        render_img = np.clip(render_camera_images["Color"][:, :, 0:3] * 255, 0, 255).astype(
            np.uint8
        )
        base_img = np.clip(base_camera_images["Color"][:, :, 0:3] * 255, 0, 255).astype(np.uint8)

        return render_img, base_img

    def _get_customized_agent_config(self, default_cfg):
        """
        The ManiSkill2 default_cfg object has some properties that intentionally resist modification . . .
        To customize it anyway, create  copy of the config, because the idea of configurations is that
        they are CONFIGURABLE.
        :param default_cfg:
        :return: A modified version of agent configuration.
        """
        customized_cfg = types.SimpleNamespace()
        for attr in dir(default_cfg):
            if not (attr.startswith("__") and attr.endswith("__")):
                setattr(customized_cfg, attr, getattr(default_cfg, attr))

        anything_changed = False
        for (
            c_part_name,
            cc,
        ) in (
            self._controller_customizations.items()
        ):  # each part (arm, gripper) can have its own customizations
            for attr, value in cc.items():  # loop over each custom value
                if value is None:
                    continue
                # Find matches by iterating over the controller config
                for controller_name, controller_dict in customized_cfg.controllers.items():
                    for part_name, part_config in controller_dict.items():
                        if part_name == c_part_name and hasattr(part_config, attr):
                            try:
                                value = type(getattr(part_config, attr))(value)
                            except Exception:
                                # just guess that floating point will be the correct type
                                value = float(value)

                            if getattr(part_config, attr) != value:
                                old_value = getattr(part_config, attr)
                                setattr(part_config, attr, value)
                                anything_changed = True
                                log(
                                    f"Overriding robot controller config "
                                    f"{controller_name}.{part_name}.{attr} = "
                                    f"{getattr(part_config, attr)} (original value: {old_value})"
                                )

        # If anything changed, return the original object
        return customized_cfg if anything_changed else default_cfg

    def get_contact(self, actor0: sapien.Actor, actor1: sapien.Actor) -> float:
        """
        :param actor0: Any actor
        :param actor1: Any actor
        :return: Total impulse between the (registered) actor0 and any actor1 since
        the last call to reset().
        """
        total_impulse = np.zeros(3, dtype=np.float32)
        for contact in self._contact_during_control_step:
            if contact.actor0 == actor0 and contact.actor1 == actor1:
                total_impulse += contact.total_impulse
            if contact.actor0 == actor1 and contact.actor1 == actor0:
                total_impulse += -contact.total_impulse

        impulse = np.linalg.norm(np.sum(total_impulse, axis=0))
        return impulse

    def any_contact(self, actor0: sapien.Actor) -> bool:
        """
        :param actor0: The actor to be queried for contact.
        :return: True if any contact for actor0
        """
        for contact in self._contact_during_control_step:
            if actor0 in (contact.actor0, contact.actor1):
                impulse = np.linalg.norm(contact.total_impulse)
                if impulse > 1e-5:
                    return True
        return False

    def get_actor_contacts(self, actor: sapien.Actor) -> List[sapien.Actor]:
        """
        :param actor0: The actor to be queried for contact.
        :return: List of actors that actor came into contact with.
        """
        result = set()
        for contact in self._contact_during_control_step:
            if actor in (contact.actor0, contact.actor1):
                impulse = np.linalg.norm(contact.total_impulse)
                if impulse > 1e-5:
                    result.add(contact.actor0)
                    result.add(contact.actor1)
        result = list(result)
        if actor in result:
            result.remove(actor)
        return result

    def reset_contact(self):
        """
        Must be called on every control step to reset the contact list.
        :return: None
        """
        self._contact_during_control_step = []

    def gen_scene_pcd(
        self, num_points: int = int(1e5), exclude_actors: Optional[List] = None
    ) -> np.ndarray:
        """
        Based on ManiSkill2 code.
        Generate scene point cloud for motion planning,
        excluding the robot and actors listed in exclude_actors
        :param num_points: How many points to sample, total.
        :param exclude_actors: the list of actors will not be present in the point cloud.
        :return: numpy array, representing the scene point cloud
        """
        if not exclude_actors:
            exclude_actors = []

        meshes = []
        articulations = self._scene.get_all_articulations()
        if self.agent is not None:
            articulations.pop(articulations.index(self.agent.robot))
        for articulation in articulations:
            if articulation in exclude_actors:
                continue
            articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
            if articulation_mesh:
                meshes.append(articulation_mesh)

        for actor in self._scene.get_all_actors():
            if actor in exclude_actors:
                continue
            actor_mesh = merge_meshes(get_actor_meshes(actor))
            if actor_mesh:
                meshes.append(
                    actor_mesh.apply_transform(actor.get_pose().to_transformation_matrix())
                )

        scene_mesh = merge_meshes(meshes)
        # Note: it might happen that the scene is empty (after excluding the
        # articulation, specified actors, etc). In this case one fake point 10m
        # below the ground is returned
        scene_pcd = (
            np.array([[0, 0, -10]], dtype=np.float32)
            if scene_mesh is None
            else trimesh.sample.sample_surface(mesh=scene_mesh, count=num_points, seed=0)[0].astype(
                np.float32
            )
        )
        return scene_pcd

    def get_point_cloud_for_planning(
        self,
        excluded_grasped_actors=True,
        exclude_actors=None,
        num_actor_points=10000,
        num_floor_points=10000,
        floor_size=2.0,
    ):
        """
        :param excluded_grasped_actors: If True, the actors that are being grasped by the
        robot are excluded from the point cloud
        :param exclude_actors: actors that must be explicitly excluded
        :param num_actor_points: Number of points sampled to represent the actors.
        :param num_floor_points: Number of points sampled to represent the floor.
        :param floor_size: The size of the floor which is assumed to be a square with the
        robot at the center
        :return: numpy array, representing the scene point cloud
        """
        if not exclude_actors:
            exclude_actors = []

        ex_ac = exclude_actors + (
            self.agent.get_grasped_actors() if excluded_grasped_actors else []
        )
        point_cloud = self.gen_scene_pcd(num_points=num_actor_points, exclude_actors=ex_ac)
        point_cloud = point_cloud[np.linalg.norm(point_cloud, axis=1) < 10.0]  # keep it local

        if num_floor_points > 0:
            fn = int(np.ceil(np.sqrt(num_floor_points) / 2))
            fs = floor_size / (2 * fn)
            floor_pcd = (
                np.reshape(np.transpose(np.mgrid[-fn:fn, -fn:fn, 0:0.1], (1, 2, 3, 0)), (-1, 3))
                * np.array([fs, fs, 1.0])
                + self.agent.robot.pose.p
            )
            floor_pcd = floor_pcd[
                np.linalg.norm(floor_pcd - self.agent.robot.pose.p, axis=1) > 0.08
            ]
            point_cloud = np.vstack([floor_pcd, point_cloud])

        # Mask out the points around the base link of agent
        # Otherwise the path planner might report collision for the (non-mobile) base link
        distance_to_base_link = np.linalg.norm(
            (point_cloud - self.agent.robot.pose.p)[:, 0:2], axis=1
        )
        mask = np.logical_and(
            point_cloud[:, 2] >= self.agent.base_link_height[0],
            point_cloud[:, 2] <= self.agent.base_link_height[1],
        )
        mask = np.logical_and(distance_to_base_link <= self.agent.base_link_radius, mask)
        point_cloud = point_cloud[np.logical_not(mask)]

        return point_cloud

    def compute_normalized_dense_reward(self, *args, **kwargs):
        raise NotImplementedError
