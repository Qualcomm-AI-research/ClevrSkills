# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
Script to convert ClevrSkills dataset to "LLaVa format" using 
a Python API to invoke the oracle skills.
"""

import argparse
import copy
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from clevr_skills.assets.video.video_template import get_text_image
from clevr_skills.dataset_converters.action_trace_converter import ActionTraceConverter
from clevr_skills.utils.load_traj import Traj, find_trajectories
from clevr_skills.utils.logger import log
from clevr_skills.utils.visualize_prompt import PromptVisualizer


def project_actor_bounding_box(
    bounding_box: np.ndarray,
    camera_resolution: Tuple[int, int, int],
    projection_method: str = "GRID",
    projection_resolution: int = 10,
):
    """
    Projects a bounding box to a coarse grid.
    :param bounding_box: The 2D bounding box
    :param camera_resolution: Camera width, height
    :param projection_method: Not used currently.
    :param projection_resolution: Integer specifying the desired grid resolution.
    :return: (X, Y) coordinates on the grid.
    """
    proj = (bounding_box[:, 0:2] / camera_resolution[0:2]) * projection_resolution
    return tuple(np.round(np.mean(proj, axis=0)).astype(int))


def write_json(path, json_list: List[Dict]) -> None:
    """
    Writes json_list to .json or .jsonl file, depending on the extension of path
    :param path: Destination path. If the extension is .jsonl, the file will be written as a json list.
    :param json_list: assumed to be a list of dicts.
    """
    with open(path, "w+", encoding="utf8") as json_file:
        if os.path.splitext(path)[1] == ".jsonl":
            # JSON-line format.
            for entry in json_list:
                json.dump(entry, json_file)
                json_file.write("\n")
        else:
            json.dump(json_list, json_file, indent=2)
    log(f"Written: {path}", info=True)


def read_json(path):
    """
    Reads both json and jsonl formats.
    :param path: Path of file to be read. If the extension is .jsonl, the file will read as if it is a json list.
    :return: The loaded json file (a Dict or a List)
    """
    jsonl = os.path.splitext(path)[1] == ".jsonl"
    with open(path, "r", encoding="utf8") as file:
        if jsonl:
            result = []
            while line := file.readline():
                result.append(json.loads(line))
            return result
        return json.load(file)


class LlavaActionTraceConverter(ActionTraceConverter):
    """
    This class converts a single trajectory to LLaVA format.

    At a high-level, this class scans through the action trace to see when a new solver starts.
    Then it generates the corresponding Python code to replay the start of that solver.
    """

    def __init__(
        self, traj_path: Optional[str] = None, emit_actor_pos: bool = False, cheat: bool = False
    ):
        """
        :param traj_path: Trajectory will be read from this path.
        :param emit_actor_pos: Whether to emit the 2D actor position with each actor.
        :param cheat: allow cheating: until some issues with some tasks are fixed.
        Specifically: The "Swap" predicate currently requires the solver to use a
        _specific_ free space locations as a temporary location. Otherwise, it won't
        progress. When cheat is True, this pose is given to the free_space() function.
        """
        super().__init__(traj_path)
        self._placeholder_images = {
            "base": np.zeros((256, 256, 3), dtype=np.uint8)
        }  # only the image shape matters
        self.emit_actor_pos = emit_actor_pos
        self.cheat = cheat

        self.projection_method = "GRID"
        self.projection_resolution = 10

    def extract_labels(
        self,
        prompt: str,
        action_trace: Optional[List] = None,
        success: Optional[np.ndarray] = None,
        camera_videos: Optional[Dict] = None,
        camera_paths: Optional[Dict] = None,
    ) -> List[List[Tuple[int, int, str]]]:
        """
        :param prompt: The prompt relative to which the labels should be extracted.
        :param action_trace: Action trace for the entire sequence.
        :param success: Success (boolean) for the entire sequence.
        :param camera_videos: Optional cv2 video objects for each camera.
        :param camera_paths: Optional paths to camera videos.
        :return: Returns segments of labels, based on the action trace.
        List of List of Tuple[start_idx, end_idx, text].
        The first index is depth of the action trace, the second index is segments.
        """
        if not camera_videos:
            camera_videos = {}
        if not camera_paths:
            camera_paths = {}

        if action_trace is None or success is None:  # load from traj_path, if not given by caller
            action_trace = self.traj.get_action_trace()
            success = self.traj.get_success()

        # Detect when new solver is activated
        conversation_points = self.extract_conversation_points(
            action_trace, success, camera_videos=camera_videos, camera_paths=camera_paths
        )
        conversation_points.append(
            {"step_idx": len(action_trace)}
        )  # add a final conversion point to simplify getting the final step_idx

        # Convert to tuples (start_idx, end_idx, text)
        result = []
        for cp_idx in range(len(conversation_points) - 1):
            cp = conversation_points[cp_idx]
            cp_next = conversation_points[cp_idx + 1]
            cp["images"] = self._placeholder_images
            action = self.conversation_point_to_action(cp, prompt)
            start_idx = cp["step_idx"]
            end_idx = cp_next["step_idx"]
            result.append((start_idx, end_idx, action))

        return [result]  # the caller expects a multi-depth

    def extract_conversation_points(
        self,
        action_trace: List[Dict],
        success: np.ndarray,
        camera_videos: Optional[Dict] = None,
        camera_paths: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Extract all moments when the solver changes (i.e., a point where LLaVA can have a conversation),
        and keep the corresponding image(s) and action trace.
        :param action_trace: Action trace for the entire sequence.
        :param success: Success (boolean) for the entire sequence.
        :param camera_videos: Optional cv2 video objects for each camera.
        :param camera_paths: Optional paths to camera videos.
        :return:
        """
        if not camera_videos:
            camera_videos = {}
        if not camera_paths:
            camera_paths = {}

        first_success = False
        cs_class = cs_python_id = cs_random_id = None  # "cs" stands for "current solver"
        conversation_points = []  # these will be used to generate the conversations
        for frame_idx, at in enumerate(action_trace):
            images = {}
            for camera_name, camera_video in camera_videos.items():
                ret, image = camera_video.read()
                assert ret, f"Could not read {camera_paths[camera_name]}"
                images[camera_name] = image

            if len(at) >= 2:
                # we extract the level 1 solvers. Alternatively could extract the "deepest-level-solvers"
                at = at[1]
                if (
                    cs_class != at["solver_class"]
                    or cs_python_id != at["solver_python_id"]
                    or cs_random_id != at["solver_random_id"]
                ):
                    conversation_points.append({"at": at, "images": images, "step_idx": frame_idx})
                    cs_class = at["solver_class"]
                    cs_python_id = at["solver_python_id"]
                    cs_random_id = at["solver_random_id"]
            if success[frame_idx] and not first_success:
                conversation_points.append(
                    {"success": True, "images": images, "step_idx": frame_idx}
                )
                break
        return conversation_points

    def _project_actor_bounding_box(self, at: Dict, actor_id: Tuple[str, int]) -> Tuple[int, int]:
        """
        :param at: Action trace item.
        :param actor_id: The actor (name, ID).
        :return: 2D coordinate of actor on the coarse projection grid.
        """
        if actor_id in at["bounding_box"]:
            bounding_box = at["bounding_box"][actor_id]
            camera_resolution = at["image_shapes"]["base"]
            return project_actor_bounding_box(
                bounding_box, camera_resolution, self.projection_method, self.projection_resolution
            )

        # Missing bounding box
        return (-1, -1)

    def _actor_to_func_arg(
        self, at: Dict, actor: Tuple[str, int], actor_arg_name: str = None, pos_arg_name: str = None
    ):
        """
        Converts a single actor to a function argument. Supports keyword arguments (actor=..., pos=...).
        :param at: Action trace item.
        :param actor: The actor (name, ID).
        :param actor_arg_name: optional keyword for the actor name
        :param pos_arg_name: optional keyword for the actor 2D position
        :return: a string that can be used as an argument in a function.
        """
        result = ""
        if actor_arg_name:
            result += actor_arg_name + "="
        result += f"get_actor('{actor[0]}'"
        if self.emit_actor_pos:
            result += ", "
            actor_pos_2d = self._project_actor_bounding_box(at, actor)
            if pos_arg_name:
                result += pos_arg_name + "="
            result += f"{actor_pos_2d}"
        result += ")"
        return result

    def BalanceScaleSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a BalanceScale solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        result = "balance_scale(["
        for actor_idx, actor in enumerate(at["objects"]):
            if actor_idx > 0:
                result += ", "
            actor_arg_str = self._actor_to_func_arg(at, actor)
            result += f"({actor_arg_str})"
        result += "])"
        return result

    def PickMove3dThrowSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a PickMove3dThrow solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        actor_arg_str = self._actor_to_func_arg(at, at["throw_actor"])
        result = f"hit({actor_arg_str}"

        target_actor_arg_str = self._actor_to_func_arg(at, at["target_actor"])
        result += f", {target_actor_arg_str}"

        if at["target_2d"]:
            result += ", target_2d=True"
        if at["topple_target"]:
            result += ", topple_target=True"

        result += ")"

        return result

    def PickSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a Pick solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        actor_arg_str = self._actor_to_func_arg(
            at, at["actor"], actor_arg_name="actor", pos_arg_name="actor_pos"
        )
        result = f"pick({actor_arg_str})"
        return result

    def Move3dSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a Move solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        return at["solver_class"] + "(to do: not implemented yet)"

    def PlaceSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a Place solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        return at["solver_class"] + "(to do: not implemented yet)"

    def PickMove3dPlaceSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a PickMove3dPlace solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        actor_arg_str = self._actor_to_func_arg(
            at, at["actor"], actor_arg_name="actor", pos_arg_name="actor_pos"
        )
        func_calls = []
        pnp = f"pick_move3d_place({actor_arg_str}"

        # Remember pose of actors, if needed
        if "store_pose" in at:
            for actor in at["store_pose"]:
                aas = self._actor_to_func_arg(
                    at, actor, actor_arg_name="actor", pos_arg_name="actor_pos"
                )
                func_calls.append(f"pose_dict['{actor[0]}'] = get_pose({aas})")

        if "target_pose_function" in at:
            target_pose_function = at["target_pose_function"]
            if target_pose_function == "rotate":
                if not "restore" in at:
                    log("Warning: adding 'restore' key for compatibility with old data")
                    at["restore"] = False
                angle = -at["angle"] if at["restore"] else at["angle"]
                tp = f"target_pose = rotate_z(get_pose({actor_arg_str}), {angle})"
                func_calls.append(tp)
                pnp += ", target_pose=target_pose"

            elif target_pose_function == "rearrange":
                pnp += (
                    f", target_pose=get_actor_pose(image='{at['target_pose_image']}', "
                    f"{actor_arg_str})"
                )
            elif target_pose_function == "rearrange_replace":
                # This assumes that there is a dict "pose_dict" available for storing poses
                pnp += f", target_pose=pose_dict['{at['actor'][0]}']"

            elif target_pose_function == "free_space":  # move actor to free space
                pnp += f", target_pose=free_space({actor_arg_str}"
                if self.cheat:
                    # Until "swap" predicate is properly formalized using a "free_space" predicate,
                    # we will need to cheat here by specifying the target pose.
                    pnp += f", pose=Pose({at['target_pose'][0]}, {at['target_pose'][1]})"
                pnp += ")"

            elif target_pose_function == "swap":
                func_calls.append(f"pose_dict['{at['actor'][0]}'] = get_pose({actor_arg_str})")
                pnp += f", target_pose=pose_dict['{at['target_actor'][0]}']"

            elif target_pose_function == "swap_rotate":
                # The target pose is:
                # - the position of the target actor, combined with:
                # - the orientation of the actor, rotated by angle degrees:
                tp = (
                    f"target_pose = rotate_z(Pose(pose_dict['{at['target_actor'][0]}'].p, "
                    f"pose_dict['{at['actor'][0]}'].q), {at['angle']})"
                )
                func_calls.append(tp)
                pnp += ", target_pose=target_pose"
                # remember pose of actor after this call

            elif target_pose_function == "on_top":
                # need to know the prompt
                if "target_image" in at and at["target_image"] in prompt:
                    # Analyze what actor the at['actor'] is resting on in the keystep image
                    func_calls.append(
                        f"target_actor = get_supporting_actor(\"{at['target_image']}\", "
                        f"\"{at['actor'][0]}\")"
                    )
                    pnp += ", target_actor=target_actor"
                else:
                    # The name of the target actor is given in the prompt:
                    target_actor_arg_str = self._actor_to_func_arg(
                        at,
                        at["target_actor"],
                        actor_arg_name="target_actor",
                        pos_arg_name="actor_pos",
                    )
                    pnp += f", {target_actor_arg_str}"

            elif target_pose_function == "in_area":
                area_actor_arg_str = self._actor_to_func_arg(
                    at, at["area_actor"], actor_arg_name="next_to_actor"
                )
                d = at["area_direction"]
                func_calls.append(
                    f"target_pose = free_space_next_to({actor_arg_str}, {area_actor_arg_str}, "
                    f"direction=[{d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}], "
                    f"description=\"{at['area_description']}\")"
                )
                pnp += ", target_pose=target_pose"

        elif at["target_pose"] is None:
            if at["target_actor"] is not None:
                target_actor_arg_str = self._actor_to_func_arg(
                    at, at["target_actor"], actor_arg_name="target_actor", pos_arg_name="actor_pos"
                )
                pnp += f", {target_actor_arg_str}"
        else:
            return at["solver_class"] + " # to do: this case is not implemented yet"

        if at["match_ori"]:
            pnp += ", match_ori=True"
        pnp += ")"

        func_calls.append(pnp)
        return "\n".join(func_calls)

    def PushSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a Push solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        actor_arg_str = self._actor_to_func_arg(
            at, at["actor"], actor_arg_name="actor", pos_arg_name="actor_pos"
        )
        func_calls = []
        pnp = f"push({actor_arg_str}"

        # Remember pose of actors, if needed
        if "store_pose" in at:
            for actor in at["store_pose"]:
                aas = self._actor_to_func_arg(
                    at, actor, actor_arg_name="actor", pos_arg_name="actor_pos"
                )
                func_calls.append(f"pose_dict['{actor[0]}'] = get_pose({aas})")

        if "target_pose_function" in at:
            target_pose_function = at["target_pose_function"]

            if target_pose_function == "free_space":  # move actor to free space
                pnp += f", target_pose=free_space({actor_arg_str}"
                if self.cheat:
                    # Until "swap" predicate is properly formalized using a "free_space"
                    # predicate, we will need to cheat here by specifying the target pose.
                    pnp += f", pose=Pose({at['target_pose'][0]}, {at['target_pose'][1]})"
                pnp += ")"

            elif target_pose_function == "swap":
                func_calls.append(f"pose_dict['{at['actor'][0]}'] = get_pose({actor_arg_str})")
                pnp += f", target_pose=pose_dict['{at['target_actor'][0]}']"

        else:
            return at["solver_class"] + " # to do: this case is not implemented yet"

        pnp += ")"

        func_calls.append(pnp)
        return "\n".join(func_calls)

    def ToppleStructure_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a ToppleStructure solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        result = "topple_structure(["
        for actor_idx, actor in enumerate(at["actors"]):
            if actor_idx > 0:
                result += ", "
            actor_arg_str = self._actor_to_func_arg(at, actor)
            result += f"({actor_arg_str})"
        result += "])"
        return result

    def Move3dTouchSolver_to_python(self, at: Dict, prompt: str):
        """
        :param at: Action trace item corresponding to the start of a Move3DTouch solver.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """
        actor_arg_str = self._actor_to_func_arg(
            at, at["actor"], actor_arg_name="actor", pos_arg_name="actor_pos"
        )
        result = f"touch({actor_arg_str}"
        if at["push"]:
            result += ", push=True"
        if at["topple"]:
            result += ", topple=True"
        result += ")"
        return result

    def action_trace_to_python(self, action_trace: Dict, prompt: str) -> str:
        """
        :param action_trace: Action trace that should be converted to code.
        :param prompt: The prompt relative to which the code should be extracted.
        :return: Python code to replay the action trace.
        """

        if not hasattr(self, "_python_table"):
            # construct table for converting ActionTrace to "python code"
            self._python_table = {
                "BalanceScaleSolver": self.BalanceScaleSolver_to_python,
                "PickMove3dThrowSolver": self.PickMove3dThrowSolver_to_python,
                "PickMove3dPlaceSolver": self.PickMove3dPlaceSolver_to_python,
                "PushSolver": self.PushSolver_to_python,
                "PickSolver": self.PickSolver_to_python,
                "Move3dSolver": self.Move3dSolver_to_python,
                "PlaceSolver": self.PlaceSolver_to_python,
                "ToppleStructure": self.ToppleStructure_to_python,
                "Move3dTouchSolver": self.Move3dTouchSolver_to_python,
            }
        solver_class = action_trace["solver_class"]
        if solver_class in self._python_table:
            return self._python_table[solver_class](action_trace, prompt)

        log(
            "Warning: conversion of action trace to Python not "
            f"implemented yet for {solver_class}"
        )
        return action_trace["solver_class"] + "# to do: not implemented yet"

    def conversation_point_to_action(self, cp: Dict, prompt: str) -> str:
        """
        :param cp: The conversation point (a Dict with info about a specific point in the action
        trace where a LLaVA conversation should be generated).
        :param prompt: The prompt is required because the actions can depend on it.
        :return: Python code.
        """
        if "success" in cp:
            return "done()"

        return self.action_trace_to_python(cp["at"], prompt)


class LlavaConverter:
    """
    This class handles the conversion of an entire dataset to LLaVA Visual Question Answering format.

    The generated dataset could be used to train a LLaVA VQA to be the high-level policy for solving ClevrSkills tasks.
    """

    def __init__(
        self,
        input_paths: List[str],
        output_path: str,
        emit_actor_pos: bool = False,
        eval: bool = False,
        cheat: bool = False,
        save_verification_images: bool = False,
        split_object_json: bool = False,
    ):
        """
        :param self:
        :param input_paths: Where to search for trajectories. Can be a list of multiple paths
        :param output_path: path to json file for output.
        :param emit_actor_pos: Whether to emit the 2D actor position with each actor
        :param eval: In evaluation mode, the output format is different. This is for
        running the test-set.
        :param cheat: allow cheating, until some issues with some tasks are fixed.
        Specifically: The "Swap" predicate currently requires the solver to use a
        _specific_ free space locations as a temporary location. Otherwise, it won't progress.
        When cheat is True, this pose is handed to the free_space() function
        :param save_verification_images: Save verification images.
        :param split_object_json: Split the output into separate json for robot control conversations and object-related conversations?
        """
        self.emit_actor_pos = emit_actor_pos
        self.eval = eval
        self.cheat = cheat
        self.save_verification_images = save_verification_images
        self.is_split_object_json = split_object_json
        self._python_table = None

        # Constants for the "from" field in the json:
        self.human_name = "human"
        # The agent_name is "gpt" only for compatibility with existing LLaVA code
        self.agent_name = "gpt"

        # collect all trajectories
        if isinstance(input_paths, str):
            input_paths = [input_paths]
        self.trajectory_paths = sum(
            [sorted(find_trajectories(path, require_success=True)) for path in input_paths], []
        )
        log(f"Number of trajectories found: {len(self.trajectory_paths)}")

        # Prepare output paths
        self.json_output_path = output_path
        self.base_output_path, self.json_filename = os.path.split(self.json_output_path)
        self.image_output_path = os.path.join(
            "clevrskills", os.path.splitext(self.json_filename)[0] + "_images"
        )
        iop = os.path.join(self.base_output_path, self.image_output_path)
        if os.path.isdir(iop):
            shutil.rmtree(iop)
        os.makedirs(iop, exist_ok=True)

        if self.save_verification_images:
            self.verification_output_path = os.path.join(
                self.base_output_path, os.path.splitext(self.json_filename)[0] + "_verification"
            )
            self.object_verification_output_path = self.verification_output_path + "_objects"
            for path in [self.verification_output_path, self.object_verification_output_path]:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

        self.num_llava_images_written = 0
        self.num_llava_conversations = 0

        # Get all possible object names in the dataset
        all_object_names = []
        for traj_path in self.trajectory_paths:
            traj = Traj(traj_path)
            visibility = traj.get_visibility()
            all_object_names += [object_name for object_name, _object_id in visibility]
        self.all_object_names = sorted(list(set(all_object_names)))

    def convert(self):
        """
        Converts each trajectory
        :return: None
        """
        log(f"Writing LLaVA dataset to {self.base_output_path}")
        json_list = []  # list of individual "conversations"
        for traj_idx in tqdm(range(len(self.trajectory_paths))):
            traj_path = self.trajectory_paths[traj_idx]
            json_list += self.convert_to_llava(
                traj_path, self.base_output_path, self.image_output_path
            )
        log(f"Number of extracted conversations: {len(json_list)}")
        log(f"Number of extracted images: {self.num_llava_images_written}")

        # Write verification
        if self.save_verification_images:
            regular, obj = self.split_object_json(json_list)
            self._write_verification_images(regular, self.verification_output_path)
            self._write_verification_images(obj, self.object_verification_output_path)

        # Write json with object questions
        if self.is_split_object_json:
            json_list, json_object_dict = self.split_object_json(json_list)
            p = os.path.splitext(self.json_output_path)
            json_object_output_path = p[0] + "_objects" + p[1]
            write_json(json_object_output_path, json_object_dict)

        # write json or jsonl
        write_json(self.json_output_path, json_list)

    def split_object_json(self, json_list) -> Tuple[List[Dict], List[Dict]]:
        """
        Splits json list into one list for robot control related conversations,
        and one list for object related questions.
        :param json_list: List with conversions.
        :return: Tuple (robot_control_conversation, object_conversation)
        """
        regular = [j for j in json_list if not j["object_conversation"]]
        obj = [j for j in json_list if j["object_conversation"]]
        return regular, obj

    def generate_object_conversations(self, traj_path: str, conversation_points: List[Dict]):
        """
        Generates additional conversations around objects to enhance the object detector
        These questions are targeted at making the object detector better.
        :param traj_path: Path to trajectory.
        :param conversation_points: List of conversation points.
        :return: List of conversations about objects (to be saved to json file).
        """

        def _project_actor_bounding_boxes(at: Dict, num_fake=0):
            """
            Internal function
            :param at: Action trace
            :param num_fake: How many fake projections  (where there are no actors) to add.
            :return: List of [name, pos_2d, real]
            """
            latc = LlavaActionTraceConverter(None, self.emit_actor_pos, self.cheat)
            proj = [
                (actor_id[0], latc._project_actor_bounding_box(at, actor_id), True)
                for actor_id in at["bounding_box"].keys()
            ]
            loc = [p[1] for p in proj]  # all (X, Y) locations
            for _f in range(num_fake):
                samples = np.random.randint(low=0, high=latc.projection_resolution, size=(10, 2))
                mask = np.ones(len(samples), dtype=bool)
                for pos2d in loc:
                    mask = np.logical_and(
                        mask,
                        np.linalg.norm(samples - pos2d, axis=1) > latc.projection_resolution // 3,
                    )
                if np.any(mask):
                    sample_idx = np.random.choice(np.where(mask)[0])
                    pos2d = tuple(samples[sample_idx].tolist())
                    while True:
                        fake_name = np.random.choice(self.all_object_names)
                        if not fake_name in [name for (name, _, _) in proj]:
                            break
                    proj.append((fake_name, pos2d, False))
            return proj

        result = []  # json goes here
        for cp in conversation_points:
            if "success" in cp:
                continue

            at = cp["at"]
            rel_image_path = cp["image_paths"]["base"]

            conversation = [
                {"from": self.human_name, "value": "Put question here"},
                {"from": self.agent_name, "value": "Put answer here"},
            ]

            # Basically, we need a list of object names and locations
            # Plus one imaginary object and one imaginary location

            # Question 1. What is the location of [obj]?
            object_projections = _project_actor_bounding_boxes(at, num_fake=1)
            object_names = []
            for object_name, _, real in object_projections:
                if object_name in object_names:
                    continue  # don't ask the same question twice
                object_names.append(object_name)
                conversation[0]["value"] = f"<image>\nWhat is the location of the {object_name}?"
                actor_pos_2d: List[Tuple[int, int]] = [
                    p for n, p, real in object_projections if real and n == object_name
                ]
                if len(actor_pos_2d) == 0:
                    conversation[1]["value"] = f"There is no {object_name}"
                elif len(actor_pos_2d) == 1:
                    conversation[1]["value"] = f"The {object_name} is at {actor_pos_2d[0]}"
                else:
                    value = f"There are multiple {object_name}s visible, at "
                    for idx, pos in enumerate(actor_pos_2d):
                        if idx == len(actor_pos_2d) - 1:
                            value += " and "
                        elif idx > 0:
                            value += ", "
                        value += f"{pos}"
                    conversation[1]["value"] = value
                json_obj = self.conversation_to_json(
                    traj_path, conversation, rel_image_path, object_convo=True
                )
                result.append(json_obj)

            # Question 2. What object is at x, y?
            object_projections = _project_actor_bounding_boxes(at, num_fake=1)
            for object_name, pos2d, real in object_projections:
                conversation[0]["value"] = f"<image>\nWhat object is at {pos2d}?"
                if real:
                    conversation[1]["value"] = f"There is a {object_name} at {pos2d}"
                else:
                    conversation[1]["value"] = f"There is no object at {pos2d}"

                json_obj = self.conversation_to_json(
                    traj_path, conversation, rel_image_path, object_convo=True
                )
                result.append(json_obj)

            # Question 3. Is there a [obj] at [x,y]   (answer short, long)
            object_projections = _project_actor_bounding_boxes(at, num_fake=1)
            for object_name_1, _, _ in object_projections:
                for object_name_2, pos2d_2, real_2 in object_projections:
                    conversation[0]["value"] = f"<image>\nIs there a {object_name_1} at {pos2d_2}?"
                    if (object_name_1, pos2d_2, True) in object_projections:
                        conversation[1]["value"] = f"Yes, there is {object_name_1} at {pos2d_2}"
                    else:
                        if real_2:
                            conversation[1]["value"] = f"No, there is {object_name_2} at {pos2d_2}"
                        else:
                            conversation[1][
                                "value"
                            ] = f"No, there is no {object_name_1} at {pos2d_2}"

                    json_obj = self.conversation_to_json(
                        traj_path, conversation, rel_image_path, object_convo=True
                    )
                    result.append(json_obj)

            # Question 4. Is there a [obj] in the scene (answer short, long)
            object_projections = _project_actor_bounding_boxes(at, num_fake=1)
            for object_name, pos2d, real in object_projections:
                conversation[0]["value"] = f"<image>\nIs there a {object_name} visible?"
                if real:
                    conversation[1]["value"] = f"Yes, there is a {object_name} visible"
                else:
                    conversation[1]["value"] = f"No, there is no {object_name} visible"
                json_obj = self.conversation_to_json(
                    traj_path, conversation, rel_image_path, object_convo=True
                )
                result.append(json_obj)

            # Question 5. How many [obj] are in the visible (answer short, long)
            object_projections = _project_actor_bounding_boxes(at, num_fake=1)
            for object_name, pos2d, real in object_projections:
                num_visible = np.sum(
                    [1 for n, p, r in object_projections if r and n == object_name]
                )
                conversation[0]["value"] = f"<image>\nHow many {object_name} are visible?"
                if num_visible == 0:
                    conversation[1]["value"] = f"There are no {object_name}s visible"
                else:
                    conversation[1]["value"] = (
                        f"There are "
                        f"{num_visible} {object_name}{'s' if num_visible > 1 else ''} "
                        f"visible"
                    )
                json_obj = self.conversation_to_json(
                    traj_path, conversation, rel_image_path, object_convo=True
                )
                result.append(json_obj)

        return result

    def conversation_to_json(
        self,
        traj_path: str,
        conversation: List[Dict],
        rel_image_path: str,
        object_convo: bool = False,
    ):
        """
        :param traj_path: Path to trajectory
        :param conversation: The conversation
        :param rel_image_path: Path to the image that the conversation is about.
        :param object_convo: Is this conversion about object location & visibility?
        This flag is used to split the conversations into regular and object, later on.
        :return: json dictionary.
        """
        if self.eval:
            text = conversation[0]["value"].replace("<image>\n", "")
            json_obj = {
                "source": traj_path,
                "question_id": f"{self.num_llava_conversations:012d}",
                "image": rel_image_path,
                "text": text,
                "answer": conversation[1]["value"],
                "object_conversation": object_convo,
            }
        else:
            json_obj = {
                "source": traj_path,
                "id": f"{self.num_llava_conversations:012d}",
                "image": rel_image_path,
                "conversations": copy.deepcopy(conversation),
                "object_conversation": object_convo,
            }
        self.num_llava_conversations += 1
        return json_obj

    def convert_to_llava(
        self, traj_path: str, base_output_path: str, image_output_path: str
    ) -> List[Dict]:
        """
        :param traj_path: directory with a single ClevrSkills trajectory
        :param base_output_path: The image paths in the returned json will be relative to this.
        :param image_output_path: where to store images, relative to base_output_path
        I.e., images will be stored in base_output_path/image_output_path/image_{:08d}.jpg
        :return: List of conversations (to be saved to json file)
        """
        prompt_vis = PromptVisualizer()

        traj = Traj(traj_path)

        # Load the action trace
        action_trace = traj.get_action_trace()

        # Load bounding boxes, fuse them into action trace (for simplicity)
        bounding_boxes = traj.get_bounding_boxes()
        visibility = traj.get_visibility()

        # Fuse bounding boxes and visibility into the action trace (for simplicity)
        for (obj, object_id), bb in bounding_boxes.items():
            vis = visibility[(obj, object_id)]
            for frame_idx, at_frame in enumerate(action_trace):
                for at in at_frame:
                    for key in ["bounding_box", "visibility"]:
                        if not key in at:
                            at[key] = {}
                    at["bounding_box"][(obj, object_id)] = bb[frame_idx]
                    at["visibility"][(obj, object_id)] = vis[frame_idx]

        # Load success
        success = traj.get_success()

        # Open camera videos
        camera_videos = {}
        camera_paths = {}
        for camera_name in [Traj.Camera.BASE, Traj.Camera.WRIST]:
            video, camera_mp4_path = traj.get_video(camera_name, return_path=True)
            assert video.isOpened(), f"Could not open {camera_mp4_path}"
            camera_videos[camera_name.value] = video
            camera_paths[camera_name.value] = camera_mp4_path

        latc = LlavaActionTraceConverter(traj_path, self.emit_actor_pos, self.cheat)
        conversation_points = latc.extract_conversation_points(
            action_trace, success, camera_videos, camera_paths
        )

        for camera_name, video in camera_videos.items():
            video.release()

        # generate conversations
        prompts = action_trace[0][0]["prompts"]
        prompts = [
            "".join(
                prompt_vis.visualize_prompt(
                    prompt,
                    data_path=traj_path,
                    mode=PromptVisualizer.NATURAL_LANGUAGE,
                    compose_image=False,
                )
            )
            for prompt in prompts
        ]
        # remove keystep prompts for now
        prompts = [prompt for prompt in prompts if not "keystep" in prompt]
        # Get rid of double spaces:
        for _ in range(3):
            prompts = [prompt.replace("  ", " ") for prompt in prompts]

        # Save the images for each conversion point
        for cp in conversation_points:
            rel_image_path = os.path.join(
                image_output_path, f"image_{self.num_llava_images_written:012d}.jpg"
            )
            self.num_llava_images_written += 1
            abs_image_path = os.path.join(base_output_path, rel_image_path)
            cv2.imwrite(abs_image_path, cp["images"]["base"], [cv2.IMWRITE_JPEG_QUALITY, 100])
            cp["image_paths"] = {"base": rel_image_path}
            # fuse image shapes into action trace (for simplicity)
            if "at" in cp:
                cp["at"]["image_shapes"] = {
                    image_name: image.shape for image_name, image in cp["images"].items()
                }

        # Generate conversations based on the prompts and action trace
        result = []
        for prompt in prompts:
            actions_so_far = []
            for cp in conversation_points:
                action = latc.conversation_point_to_action(cp, prompt)
                agent_name = "gpt"
                conversation = [
                    {
                        "from": "human",
                        "value": f"<image>\n{prompt}\nPast actions:\n{'; '.join(actions_so_far)}",
                    },
                    {"from": agent_name, "value": action},
                ]
                rel_image_path = cp["image_paths"]["base"]

                json_obj = self.conversation_to_json(traj_path, conversation, rel_image_path)

                result.append(json_obj)

                if action:
                    actions_so_far.append(action)

        # Generate additional questions about locations of objects
        result += self.generate_object_conversations(traj_path, conversation_points)

        return result

    def count(self):
        """
        This is a function that counts the number of solvers used for each task.
        It was used to produce a table for the paper.
        """
        counts = {}
        MAX_DEPTH = 10

        prev_at = []
        for d in range(MAX_DEPTH):
            prev_at.append(
                {"solver_class": None, "solver_python_id": None, "solver_random_id": None}
            )

        for traj_idx in tqdm(range(len(self.trajectory_paths))):
            traj_path = self.trajectory_paths[traj_idx]
            traj = Traj(traj_path)
            # Load the action trace
            action_trace = traj.get_action_trace()
            # Load success
            success = traj.get_success()
            if not np.any(success):
                continue

            # Get name of task, dataset (hard-coded, expecting a specific directory
            # structure for now)
            parent_path = os.path.split(traj_path)[0]
            task_name = os.path.split(parent_path)[1]
            _dataset_split = os.path.split(os.path.split(parent_path)[0])[1]

            if not task_name in counts:
                counts[task_name] = np.zeros(MAX_DEPTH, dtype=int)

            for act_trace in action_trace:
                for d, at in enumerate(act_trace):
                    if (
                        prev_at[d]["solver_class"] != at["solver_class"]
                        or prev_at[d]["solver_python_id"] != at["solver_python_id"]
                        or prev_at[d]["solver_random_id"] != at["solver_random_id"]
                    ):
                        prev_at[d]["solver_class"] = at["solver_class"]
                        prev_at[d]["solver_python_id"] = at["solver_python_id"]
                        prev_at[d]["solver_random_id"] = at["solver_random_id"]
                        counts[task_name][d] += 1

        json_dict = {}
        for task_name, value in counts.items():
            num_episodes = int(value[0])
            c = value[1:]
            num_solvers = int(np.sum(c))
            s = [int(x) for x in c]
            json_dict[task_name] = {
                "number_of_episodes": num_episodes,
                "number_of_solvers": num_solvers,
                "number_of_solvers_at_depth": s,
            }

        with open(self.json_output_path, "w+", encoding="utf8") as json_file:
            json.dump(json_dict, json_file, indent=2)
        log(f"Written {self.json_output_path}")

    def _write_verification_images(self, json_list: List[Dict], output_path: str):
        """
        Writes verification images to output_path.
        Verification images allow humans to easily inspect if the conversations make sense.
        :param json_list: all conversation, in json output format
        :param output_path: Where to write the verification images.
        """
        log(f"Writing verification images to {output_path}")

        question_color = (191, 174, 98)  # in BGR order because using OpenCV
        answer_color = (169, 135, 116)

        for entry in tqdm(json_list):
            # Load image, upsample so the text can be reasonable sharp
            image_path = os.path.join(self.base_output_path, entry["image"])
            image = cv2.imread(image_path)
            image = cv2.resize(
                image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR
            )

            # Retrieve question, answer
            if "conversations" in entry:
                c_question = entry["conversations"][0]
                question = c_question["from"] + ": " + c_question["value"]
                c_answer = entry["conversations"][1]
                answer = c_answer["from"] + ": " + c_answer["value"]
            else:
                question = self.human_name + ": " + entry["text"]
                answer = self.agent_name + ": " + entry["answer"]

            # Draw object X, Y "boxes" into the images
            latc = LlavaActionTraceConverter(None, self.emit_actor_pos, self.cheat)
            r = latc.projection_resolution
            h, w = [s / r for s in image.shape[0:2]]
            thickness = 3
            true_question = question.split("Past actions")[
                0
            ]  # exclude the coordinates from past actions from the question
            for text, color, offset in (
                (true_question, question_color, -1),
                (answer, answer_color, 2),
            ):
                # Split the text; X, Y coordinates will appear as s[i], s[i+1]
                s = re.split("\(|,|\)| ", text.replace(" ", ""))
                for i in range(len(s) - 1):
                    if s[i].isnumeric() and s[i + 1].isnumeric():
                        x, y = int(s[i]), int(s[i + 1])
                        upper_left = (int((x - 0.5) * w + offset), int((y - 0.5) * h + offset))
                        lower_right = (int((x + 0.5) * w - offset), int((y + 0.5) * h - offset))
                        cv2.rectangle(image, upper_left, lower_right, color, thickness)

            # Render the question & answer
            images = [image]
            for text, background_color in ((question, question_color), (answer, answer_color)):

                image_width = image.shape[1]
                font_size = image_width // 24
                text_image, _, _ = get_text_image(
                    text,
                    image_width=image_width,
                    font_size=font_size,
                    background_color=background_color,
                )
                images.append(text_image)

            # merge image, question, answer
            image = np.concatenate(images, axis=0)

            # Write image to file
            question_id = entry["id"] if "id" in entry else entry["question_id"]
            verification_image_output_path = os.path.join(output_path, f"{question_id}.jpg")
            cv2.imwrite(
                verification_image_output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )  # perfect quality not required because for visual verification only


def get_args_parser():
    """
    :return: Argument parser for the LLaVA converter.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        nargs="+",
        required=True,
        help="Path that will be scanned for trajectories",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for dataset.json."
    )
    parser.add_argument(
        "--actor-pos2d",
        action="store_true",
        help="Should actors be identified by their position, too?",
    )
    parser.add_argument(
        "--split-object-json",
        action="store_true",
        help=(
            "Should the questions about objects locatiion and visiblity be stored in a "
            "separate json? (output_path_objects.json)"
        ),
    )
    parser.add_argument(
        "--save-verification-images",
        action="store_true",
        help=(
            "When set, save images that visualize each question and answer, "
            "for visual verification by human"
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "When true, data is output in 'eval mode', which is a different json "
            "format that 'train mode'"
        ),
    )

    parser.add_argument(
        "--count",
        action="store_true",
        help=(
            "When true, the script simply reports the number of solvers used for "
            "each task (+ their depth)"
        ),
    )

    return parser


if __name__ == "__main__":

    def parse_args():
        parser = get_args_parser()
        args, _opts = parser.parse_known_args()
        return args

    args = parse_args()

    if args.eval and os.path.splitext(args.output_path)[1] == ".json":
        log("Warning: --eval was specified, but the output is .json.")
        args.output_path = args.output_path + "l"
        log(f"Changed output path to: {args.output_path}")

    converter = LlavaConverter(
        args.input_path,
        args.output_path,
        emit_actor_pos=args.actor_pos2d,
        eval=args.eval,
        save_verification_images=args.save_verification_images,
        split_object_json=args.split_object_json,
    )
    if args.count:
        converter.count()
    else:
        converter.convert()
