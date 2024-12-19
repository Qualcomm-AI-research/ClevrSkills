# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import bz2
import json
import os
import pickle
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import tqdm

from clevr_skills.utils.logger import log


class Traj:
    """
    A utility class that makes it easy to load data from a ClevrSkills trajectory.

    Light-weight data that is loaded one (using .get_DATA()) is kept cache
    inside the class instance.
    This excludes video and images
    """

    class Camera(Enum):
        RENDER = "render"
        BASE = "base"
        WRIST = "hand"
        OBJECT = "object"

    class Proprioception(Enum):
        BASE = "base_pose"
        QPOS = "qpos"
        QVEL = "qvel"
        EE = "ee_pose"
        EE_VEL = "ee_velocity"
        VACUUM_ON = "vacuum_on"
        VACUUM_GRASPING = "vacuum_grasping"
        VACUUM_READY = "vacuum_ready"

    class ActionTraceLabel(Enum):
        PYTHON = "python"
        NATURAL_LANGUAGE = "nat_lang_eng"

    def __init__(self, path: str):
        self.path = path

    def get_actions(self) -> np.ndarray:
        """
        :return: numpy array of shape (num_steps x action_space), data type = np.float64;
        The action for each step.
        """
        if not hasattr(self, "actions"):
            self.actions = np.load(os.path.join(self.path, "actions.npy"))
        return self.actions

    def get_rewards(self) -> np.ndarray:
        """
        :return: numpy array of shape (num_steps), data type = np.float64;
        The rewards for each step.
        """
        if not hasattr(self, "rewards"):
            self.rewards = np.load(os.path.join(self.path, "rewards.npy"))
        return self.rewards

    def get_success(self) -> np.ndarray:
        """
        :return: numpy array of shape (num_steps), data type = bool
        The success for each step.
        """
        if not hasattr(self, "success"):
            self.success = np.load(os.path.join(self.path, "success.npy"))
        return self.success

    def get_actions_labels(self) -> List[Dict[str, str]]:  #
        """
        :return: The actions labels for each step.
        For each step there is a dictionary with
        """
        if not hasattr(self, "action_labels"):
            d = np.load(os.path.join(self.path, "action_labels.npy"), allow_pickle=True).item()
            # Note: for some reason, the action labels are offset by 1
            # This is caused by record_env.py,
            # the line that says "self.extra_info[len(self.extra_info)+1] = extra_info"
            self.action_labels = [d[k] for k in d.keys()]
        return self.action_labels

    def get_info(self) -> Dict:
        """
        :return: A dictionary with prompts, textures.
        """
        if not hasattr(self, "info"):
            with open(os.path.join(self.path, "info.json"), "r", encoding="utf8") as file:
                self.info = json.load(file)
        return self.info

    def get_ep_info(self) -> Dict:
        """
        :return: A dictionary with information about the episode, such as
        seed, environment kwargs, number of steps, etc
        """
        if not hasattr(self, "ep_info"):
            with open(os.path.join(self.path, "ep_info.json"), "r", encoding="utf8") as file:
                self.ep_info = json.load(file)
        return self.ep_info

    def get_action_trace(self) -> List[Dict]:
        """
        :return: For each step, information on the active solvers.
        The idea is that the trace contains enough sematic information to re-execute the roll-out,
        to convert it to python calls, to convert it to natural language, etc.
        """
        if not hasattr(self, "action_trace"):
            with bz2.BZ2File(os.path.join(self.path, "action_trace.pickle.bz2"), "rb") as file:
                self.action_trace = pickle.load(file)
        return self.action_trace

    def get_action_trace_labels(
        self, mode: ActionTraceLabel, prompt: str = None
    ) -> List[List[Tuple[int, int, str]]]:
        """
        :param mode: Traj.ActionTraceLabel.PYTHON or Traj.ActionTraceLabel.NATURAL_LANGUAGE
        :param prompt: If the prompt is not specified by the caller, the first prompt is
        used.
        :return: Returns segments of labels, based on the action trace.
        List of List of Tuple[start_idx, end_idx, text].
        """
        if prompt is None:
            prompt_mode = (
                "natural_language"
                if mode == Traj.ActionTraceLabel.NATURAL_LANGUAGE
                else "placeholder"
            )
            prompt = self.get_multi_modal_prompts(mode=prompt_mode)[0]  # pick the first prompt
            prompt = prompt[0] if isinstance(prompt, list) else prompt
        at = self.get_action_trace()
        success = self.get_success()

        from clevr_skills.dataset_converters.llava import LlavaActionTraceConverter
        from clevr_skills.dataset_converters.nat_lang_english import NatLangEngActionTraceConverter

        conv_class = (
            NatLangEngActionTraceConverter
            if mode == Traj.ActionTraceLabel.NATURAL_LANGUAGE
            else LlavaActionTraceConverter
        )

        converter = conv_class(self.path)
        return converter.extract_labels(prompt, action_trace=at, success=success)

    def get_action_trace_solver_level(self) -> List[int]:
        """
        Ideally, the deepest level of solver generates the action, but some
        intermediate level solvers insert "hold still" actions and such.
        This function returns the index of the solver that actually generated the action
        for each step.
        If this cannot be determined (i.e., an older data format), then -1 is returned
        for each step.
        :return: index of the solver that generated the action, for each step.
        """
        action_trace = self.get_action_trace()
        action_emitted = [
            [level["action_emitted"] if "action_emitted" in level else 0 for level in at]
            for at in action_trace
        ]
        solver_level = [
            np.argmax(level) if level[np.argmax(level)] else -1 for level in action_emitted
        ]
        return solver_level

    def get_bounding_boxes(self, camera_name=Camera.BASE) -> Dict[Tuple[str, int], np.ndarray]:
        """
        :param camera_name: can be Traj.Camera.BASE or Traj.Camera.WRIST, Traj.Camera.RENDER
        :return: Dictionary from (object_name, object_id) -> 2D bounding box for each frame
        The bounding box is based on the projection of the 3D mesh.
        """
        if not hasattr(self, "bounding_boxes"):
            self.bounding_boxes = {}
            bbox_path = os.path.join(
                self.path, "extra", "actor_mesh_bbox", camera_name.value + "_camera"
            )
            if os.path.isdir(bbox_path):
                for filename in os.listdir(bbox_path):
                    if ".npy" in filename:
                        object_name_id = os.path.splitext(filename)[0]
                        s = object_name_id.split("_")
                        obj = "_".join(s[:-1])
                        object_id = int(s[-1])
                        with open(os.path.join(bbox_path, filename), "rb") as file:
                            bb = np.load(file)
                        self.bounding_boxes[(obj, object_id)] = bb
        return self.bounding_boxes

    def get_visibility(self, camera_name=Camera.BASE) -> Dict[Tuple[str, int], np.ndarray]:
        """
        :param camera_name: can be Traj.Camera.BASE or Traj.Camera.WRIST, Traj.Camera.RENDER
        :return: Dictionary from (object_name, object_id) -> visibility for each frame.
        A numpy array of type bool
        """
        if not hasattr(self, "visibility"):
            self.visibility = {}
            vis_path = os.path.join(
                self.path, "extra", "actor_is_visible", camera_name.value + "_camera"
            )
            if os.path.isdir(vis_path):
                for filename in os.listdir(vis_path):
                    if ".npy" in filename:
                        object_name_id = os.path.splitext(filename)[0]
                        s = object_name_id.split("_")
                        obj = "_".join(s[:-1])
                        object_id = int(s[-1])
                        with open(os.path.join(vis_path, filename), "rb") as file:
                            vis = np.load(file)
                        self.visibility[(obj, object_id)] = vis
        return self.visibility

    def get_video(
        self, camera_name: Camera, return_path: bool = False
    ) -> Union[cv2.VideoCapture, Tuple[cv2.VideoCapture, bool]]:
        """
        :param camera_name: can be Traj.Camera.RENDER, Traj.Camera.BASE or Traj.Camera.WRIST,
        :param return_path: also return the path.
        :return: cv2 camera stream
        """
        assert camera_name in [Traj.Camera.RENDER, Traj.Camera.BASE, Traj.Camera.WRIST]
        mp4_path = (
            os.path.join(self.path, "video.mp4")
            if camera_name == Traj.Camera.RENDER
            else os.path.join(self.path, "image", f"{camera_name.value}_camera", "rgb.mp4")
        )

        assert os.path.isfile(mp4_path)

        video = cv2.VideoCapture(mp4_path)
        assert video.isOpened(), f"Could not open {mp4_path}"
        return (video, str) if return_path else video

    def get_camera_param(self, camera_name: Camera, step_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        :param camera_name: can be Traj.Camera.BASE or Traj.Camera.WRIST, Traj.Camera.OBJECT
        Note that the object camera's pose changes depending on the object that it takes a
        picture off, so the extrinsics are not useful.
        :param step_idx: The parameters are stored for each step. But in practice do not change.
        By default only a single copy is returned; use step_idx < 0 to get data for each step.
        :return: Dictionary from string ("cam2world_gl", "extrinsic_cv", "intrinsic_cv")
        to numpy array.
        """
        cams = [Traj.Camera.OBJECT, Traj.Camera.BASE, Traj.Camera.WRIST]
        assert camera_name in cams
        if not hasattr(self, "camera_param"):
            param_names = ["cam2world_gl", "extrinsic_cv", "intrinsic_cv"]
            self.camera_param = {}
            for cam in cams:
                self.camera_param[cam.value] = {
                    param_name: np.load(
                        os.path.join(
                            self.path, "camera_param", f"{cam.value}_camera", f"{param_name}.npy"
                        )
                    )
                    for param_name in param_names
                }

        result = self.camera_param[camera_name.value]
        if step_idx >= 0:
            result = {key: value[step_idx] for key, value in result.items()}
        return result

    def get_agent_state(self, name: Proprioception) -> np.ndarray:
        """
        :param name: Proprioception.BASE, Traj.Proprioception.QPOS or Traj.Proprioception.QVEL]
        :return: numpy array of shape [num_steps, length of state]
        """
        assert isinstance(name, Traj.Proprioception)
        if not hasattr(self, "proprioception"):
            self.proprioception = {}
            for p in Traj.Proprioception:
                path = os.path.join(self.path, "agent", f"{p.value}.npy")
                self.proprioception[p] = np.load(path) if os.path.isfile(path) else None
        return self.proprioception[name]

    def get_agent_base_pose(self) -> np.ndarray:
        """
        :return: Base pose of agent. numpy array of shape [num_steps, 7]
        (7 = 3 floats for position, 4 floats for quaternion).
        The current agents are fixed so the base pose does not vary over time.
        """
        return self.get_agent_state(Traj.Proprioception.BASE)

    def get_agent_qpos(self) -> np.ndarray:
        """
        :return: q-pos of robot.
        numpy array of shape [num_steps, num_qs]. num_qs depends on the robot. It includes
        the end-effector q.
        """
        return self.get_agent_state(Traj.Proprioception.QPOS)

    def get_agent_qvel(self) -> np.ndarray:
        """
        :return: q-vel of robot.
        numpy array of shape [num_steps, num_qs]. num_qs depends on the robot. It includes
        the end-effector q.
        """
        return self.get_agent_state(Traj.Proprioception.QVEL)

    def get_agent_ee_pose(self) -> np.ndarray:
        """
        :return: End effector pose of agent. numpy array of shape [num_steps, 7]
        (7 = 3 floats for position, 4 floats for quaternion).
        """
        return self.get_agent_state(Traj.Proprioception.EE)

    def get_agent_ee_vel(self) -> np.ndarray:
        """
        :return: End effector pose of agent. numpy array of shape [num_steps, 6]
        (6 = 3 floats for linear velocity, 3 floats for angular velocity).
        """
        return self.get_agent_state(Traj.Proprioception.EE_VEL)

    def get_agent_vacuum_on(self) -> np.ndarray:
        """
        :return: If available, returns vacuum "proprioception":
        vacuum_on is True if the vacuum was enabled by the last action.
        """
        return self.get_agent_state(Traj.Proprioception.VACUUM_ON)

    def get_agent_vacuum_grasping(self) -> np.ndarray:
        """
        :return: If available, returns vacuum "proprioception":
        vacuum_grasping is True if any object is grasped by the vacuum
        """
        return self.get_agent_state(Traj.Proprioception.VACUUM_GRASPING)

    def get_agent_vacuum_ready(self) -> np.ndarray:
        """
        :return: If available, returns vacuum "proprioception":
        vacuum_ready is a boolean array, with one value for each suction cup.
        The value will be True if anything is touching the suction cup.
        """
        return self.get_agent_state(Traj.Proprioception.VACUUM_READY)

    def get_actor_name_id(self):
        # Because the order of actors is not explicitly recorded, we need to assume
        # that they are ordered by ID
        id_name_sorted = sorted(
            [(actor_id, actor_name) for actor_name, actor_id in self.get_visibility()]
        )
        return [(actor_name, actor_id) for actor_id, actor_name in id_name_sorted]

    def get_actor_state(
        self, return_dict: bool = False
    ) -> Union[np.ndarray, Dict[Tuple[str, int], Dict[str, np.ndarray]]]:
        """
        :param return_dict: by default, a np array of shape [num_steps, num_actors * 13] is returned
        (See mani_skill2.utils.sapien_utils.get_actor_state for how the state is encoded)
        When return_dict is True, returns a Dict[(actor_name, actor_id), Dict[state_name,
        np.ndarray], where state_name can be "pos", "ori", "vel", "ang_vel"
        :return: The position, orientation, velocity and angular velocity of each actor.
        """
        if not hasattr(self, "actor_state"):
            self.actor_state = np.load(os.path.join(self.path, "extra/actors.npy"))
        if return_dict:
            result = {}
            for idx, (actor_name, actor_id) in enumerate(self.get_actor_name_id()):
                s = self.actor_state[:, idx * 13 : (idx + 1) * 13]
                result[(actor_name, actor_id)] = {
                    "pos": s[:, 0:3],
                    "ori": s[:, 3:7],
                    "vel": s[:, 7:10],
                    "ang_vel": s[:, 10:13],
                }
            return result
        return self.actor_state

    def get_scene_image(self, camera_name: Camera) -> np.ndarray:
        """
        Returns scene observation at the start of the task.
        Images are read using OpenCV so returned in BGR
        :param camera_name: Must be Traj.Camera.RENDER or Traj.Camera.BASE
        :return:
        """
        assert camera_name in [Traj.Camera.RENDER, Traj.Camera.BASE]
        filename = {Traj.Camera.RENDER: "scene.jpg", Traj.Camera.BASE: "scene_base.jpg"}[
            camera_name
        ]
        return cv2.imread(os.path.join(self.path, filename))

    def get_keystep_images(self, camera_name: Camera) -> List[np.ndarray]:
        """
        Key step images are used in some of the prompts.
        They visually represent important steps in the completion of the task.

        :param camera_name: Must be Traj.Camera.RENDER or Traj.Camera.BASE
        :return: List of key step image. Images are read using OpenCV, so they are
        returned in GBR format.
        """
        assert camera_name in [Traj.Camera.RENDER, Traj.Camera.BASE]
        cam_short = {Traj.Camera.RENDER: "r", Traj.Camera.BASE: "b"}[camera_name]

        images = []
        for index in range(10000):
            path = os.path.join(self.path, "keysteps", f"{index}_{cam_short}img.jpg")
            if not os.path.isfile(path):
                break
            images.append(cv2.imread(path))

        return images

    def get_object_images(self) -> Dict[Tuple[str, int], np.ndarray]:
        """
        :return: dictionary from (object_name, object_id) -> image of that object
        """
        # import here such that class can function without explicit dependency on
        # ClevrSkills as much as possible
        from clevr_skills.utils.visualize_prompt import split_actor_name_id

        result = {}
        obj_img_path = os.path.join(self.path, "objects")
        for filename in os.listdir(obj_img_path):
            image = cv2.imread(os.path.join(obj_img_path, filename))
            if not image is None:
                actor_name, actor_id = split_actor_name_id(os.path.splitext(filename)[0])
                result[(actor_name, actor_id)] = image
        return result

    def get_multi_modal_prompts(
        self,
        img_height: int = 128,
        mode: str = "multi_modal",
        background_color: Tuple[int, int, int] = None,
        compose_image: bool = False,
    ) -> List[List[Union[str, np.ndarray]]]:
        """
        Returns the multi-modal prompt for the task.
        :param img_height: Height (in pixels) used to render text.
        :param mode: "natural_language", "multi_modal" or "placeholder". See corresponding
        constants in clevrskills.utils.PromptVisualizer
        :param background_color: Background color used for object images
        :param compose_image: Compose results into a single image.
        :return:
        """
        # import here such that class can function without explicit dependency on
        # ClevrSkills as much as possible
        from clevr_skills.utils.visualize_prompt import PromptVisualizer

        """
        NATURAL_LANGUAGE = "natural_language"
        MULTI_MODAL = "multi_modal"
        RAW = "placeholder"
        """
        info = self.get_info()
        pv = PromptVisualizer(img_height=img_height, background_color=background_color)
        result = [
            pv.visualize_prompt(prompt, self.path, mode, compose_image)
            for prompt in info["prompts"]
        ]
        return result


def is_trajectory(path: str, require_success: bool = False):
    """
    :param path:
    :param require_success: Only return True when any
    :return: Returns True if path looks like a ClevrSkills trajectory.
    This test is based on the files that are present inside the path
    """
    if os.path.isdir(path):
        # This list might need to be updated as dataset format evolves
        expected_files = [
            "action_labels.npy",
            "actions.npy",
            "action_trace.pickle.bz2",
            "agent",
            "camera_param",
            "env_states.npy",
            "image",
            "info.json",
            "keysteps",
            "objects",
            "rewards.npy",
            "scene_base.jpg",
            "scene.jpg",
            "success.npy",
        ]

        for filename in expected_files:
            if not os.path.exists(os.path.join(path, filename)):
                return False

        if require_success:
            with open(os.path.join(path, "success.npy"), "rb") as file:
                return np.any(np.load(file))
        else:
            return True
    else:
        return False


def find_trajectories(
    parent_path: str, max_depth: int = -1, require_success: bool = False, return_traj: bool = False
) -> List[Union[str, Traj]]:
    """
    Recursively scans the parent_path for ClevrSkills trajectories.
    :param parent_path:
    :param max_depth: How deep to scan the tree.
    When negative, scan as deep as the tree goes
    When max_depth==0 only the parent_path is checked with is_trajectory()
    :param require_success: when True, only successful trajectories are returned
    :param return_traj: When True, instances of Traj are returned instead of strings
    :return: a list of paths that look like ClevrSkills directories.
    """
    result = []
    if is_trajectory(parent_path, require_success=require_success):
        result.append(Traj(parent_path) if return_traj else parent_path)
    elif max_depth != 0:
        for filename in os.listdir(parent_path):
            path = os.path.join(parent_path, filename)
            if os.path.isdir(path):
                result += find_trajectories(
                    path,
                    max_depth=max_depth - 1,
                    require_success=require_success,
                    return_traj=return_traj,
                )
    return result


def main():
    """
    A small test function that also shows how to find trajectories in a directory tree and
    how to read all details of a trajectory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-path", type=str, required=True, help="Path containing trajectories."
    )

    args = parser.parse_args()

    log(f"Scanning {args.trajectory_path} for trajectories")
    traj_paths = find_trajectories(
        args.trajectory_path, max_depth=-1, require_success=True, return_traj=False
    )
    log(f"{len(traj_paths)} trajectories found")

    for traj_path in tqdm.tqdm(sorted(traj_paths)):
        try:
            traj = Traj(traj_path)

            # Below is example code of how to read each part of a trajectory
            actions = traj.get_actions()
            rewards = traj.get_rewards()
            success = traj.get_success()
            actions_labels = traj.get_actions_labels()
            info = traj.get_info()
            ep_info = traj.get_ep_info()
            action_trace = traj.get_action_trace()
            action_trace_labels_python = traj.get_action_trace_labels(traj.ActionTraceLabel.PYTHON)
            action_trace_labels_nat_lang = traj.get_action_trace_labels(
                traj.ActionTraceLabel.NATURAL_LANGUAGE
            )
            print(action_trace_labels_nat_lang)
            action_trace_solver_level = traj.get_action_trace_solver_level()
            actor_name_id = traj.get_actor_name_id()
            bounding_boxes = traj.get_bounding_boxes()
            visibility = traj.get_visibility()
            actor_state_raw = traj.get_actor_state()
            actor_state_dict = traj.get_actor_state(return_dict=True)

            camera_params = {
                c.value: traj.get_camera_param(c)
                for c in [Traj.Camera.OBJECT, Traj.Camera.BASE, Traj.Camera.WRIST]
            }

            base_pose = traj.get_agent_base_pose()
            ee_pose = traj.get_agent_ee_pose()
            ee_velocity = traj.get_agent_ee_vel()
            qpos = traj.get_agent_qpos()
            qvel = traj.get_agent_qvel()
            vacuum_on = traj.get_agent_vacuum_on()
            vacuum_grasping = traj.get_agent_vacuum_grasping()
            vacuum_ready = traj.get_agent_vacuum_ready()

            keystep_base_camera_images = traj.get_keystep_images(Traj.Camera.BASE)
            keystep_render_camera_images = traj.get_keystep_images(Traj.Camera.RENDER)

            object_images = traj.get_object_images()

            prompts = traj.get_multi_modal_prompts()

            videos = {
                c: traj.get_video(c)
                for c in [Traj.Camera.RENDER, Traj.Camera.BASE, Traj.Camera.WRIST]
            }
            for frame_idx in range(len(actions) + 1):
                for cam_name, video in videos.items():
                    ret, _image = video.read()
                    assert ret, f"Could not read frame {frame_idx} of {cam_name}"
        except Exception as ex:
            log(f"Exception loading {traj_path}")
            traceback.print_exception(ex)


if __name__ == "__main__":
    main()
