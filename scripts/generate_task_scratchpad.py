# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import argparse
import ast
import bz2
import glob
import json
import os
import pickle
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import natsort
import numpy as np
import tqdm

from clevr_skills.utils.load_traj import Traj

ScratchPad = Dict[str, Any]
REQUIRED_SCRATCHPAD_TRAJ_FILES = ("action_labels.npy", "action_trace.pickle.bz2", "info.json")


def _parse_literal(value: Any) -> Any:
    """Safely parse values stored as Python literals in trajectory metadata."""
    return ast.literal_eval(value) if isinstance(value, str) else value


class BaseScratchPadConverter:
    """Shared utilities for turning a recorded trajectory into a scratchpad dict.

    Subclasses customise:
      - ``_sanitize_act_label`` (how mid-level labels get rewritten)
      - ``create_scratchpad`` (the per-task plan / pos-map shape)
    """

    def __init__(self, traj_path: str) -> None:
        """Load trajectory metadata required to build a scratchpad."""
        self._traj_path = traj_path
        self._traj = Traj(path=traj_path)

        self._act_labels = np.load(f"{self._traj_path}/action_labels.npy", allow_pickle=True).item()
        with bz2.open(f"{self._traj_path}/action_trace.pickle.bz2", "r") as fp:
            self._act_trace = pickle.load(fp)
        with open(f"{self._traj_path}/info.json", "r") as fp:
            self._info = json.load(fp)

    # ---------- low-level utilities ----------
    @staticmethod
    def _pos_to_str(pos: Sequence[float]) -> str:
        """Format a numeric position using two decimal places."""
        return f"[{', '.join([f'{x:.2f}' for x in pos])}]"

    @staticmethod
    def _round_pos(pos: Sequence[float], digits: int = 5) -> List[float]:
        """Round each coordinate in a position vector."""
        return [round(x, digits) for x in pos]

    @staticmethod
    def _project_actor_bounding_box(
        bounding_box: np.ndarray,
        camera_resolution: Tuple[int, int],
        projection_method: str = "GRID",
        projection_resolution: int = 10,
    ) -> Tuple[int, int]:
        """Project an actor bounding box to a coarse image-grid coordinate."""
        proj = (bounding_box[:, 0:2] / camera_resolution[0:2]) * projection_resolution
        return tuple(np.round(np.mean(proj, axis=0)).astype(int))

    def _get_init_pos_in_pixels(
        self,
        obj_name: str,
        camera: str = "base_camera",
        cam_resolution: Tuple[int, int] = (256, 256),
    ) -> Tuple[int, int]:
        """Return the initial image-grid position of an actor from saved boxes."""
        obj_bboxes = np.load(
            os.path.join(self._traj_path, f"extra/actor_mesh_bbox/{camera}/", f"{obj_name}.npy")
        )
        return self._project_actor_bounding_box(obj_bboxes[0], cam_resolution)

    # ---------- act-label helpers ----------
    @staticmethod
    def _get_act_idxes(act_labels: List[str]) -> List[int]:
        """Boundaries at first occurrence of each label (default mode)."""
        labels = act_labels + ["end"]
        seen: Dict[str, int] = {}
        boundaries: List[int] = []
        for i, item in enumerate(labels):
            if item not in seen:
                seen[item] = i
                boundaries.append(i)
        return boundaries

    @staticmethod
    def _get_act_idxes_on_change(act_labels: List[str]) -> List[int]:
        """Boundaries every time the label changes (used by rotate-restore)."""
        labels = act_labels + ["end"]
        curr = ""
        boundaries: List[int] = []
        for i, item in enumerate(labels):
            if item != curr:
                curr = item
                boundaries.append(i)
        return boundaries

    def _act_label_sequence(self) -> List[str]:
        """Return mid-level action labels ordered by recorded timestep."""
        return [
            self._act_labels[k]["mid_act_label"] for k in natsort.natsorted(self._act_labels.keys())
        ]

    @staticmethod
    def _fill_empty_labels(act_labels: List[str]) -> List[str]:
        for x in range(1, len(act_labels)):
            if act_labels[x] == "":
                act_labels[x] = act_labels[x - 1]
        return act_labels

    # ---------- prompt / object helpers ----------
    @staticmethod
    def _extract_objects(prompt: str, dedupe: bool = False) -> Dict[str, str]:
        stubs = re.findall(r"\{(.*?)\}", prompt)
        if dedupe:
            stubs = list(dict.fromkeys(stubs))
        objects: Dict[str, str] = {}
        for st in stubs:
            if "obj:" in st:
                objects[f"obj{len(objects)}"] = st.replace("obj:", "")
        return objects

    def _actor_pos_map(self, objects: Dict[str, str]) -> Dict[str, str]:
        """Map scratchpad position names to actor initial world positions."""
        actor_states = self._traj.get_actor_state(return_dict=True)
        return {
            x.replace("obj", "pos"): self._pos_to_str(actor_states[y]["pos"][0])
            for x, y in objects.items()
        }

    @staticmethod
    def _strip_obj_ids(objects: Dict[str, str]) -> Dict[str, str]:
        """Remove trailing simulator IDs from object names."""
        return {x: "_".join(y.split("_")[:-1]) for x, y in objects.items()}

    # ---------- sanitize building blocks ----------
    @staticmethod
    def _sub_obj(act_label: str, stub: str, obj_name_map: Dict[str, str]) -> str:
        """Replace one object stub with its scratchpad object token."""
        obj = stub.split("obj:")[1]
        return act_label.replace(f"{{{stub}}}", f"<{obj_name_map[obj]}>")

    def _sub_pos(
        self,
        act_label: str,
        stub: str,
        obj_pos_map: Dict[str, str],
        pos_map: Dict[str, str],
    ) -> str:
        """Replace one position stub with its scratchpad position token."""
        pos = stub.split("pos:")[1]
        pos = [float(x) for x in pos.strip("[] ").split()]
        pos = self._round_pos(pos)
        pos_name = pos_map.get(str(pos), "<pos-temp>")
        pos_name = str(obj_pos_map.get(pos_name, "<pos-temp>"))
        return act_label.replace(f"{{{stub}}}", pos_name)

    def _sanitize_objs_only(self, act_label: str, obj_name_map: Dict[str, str]) -> str:
        """Replace every obj stub in the label."""
        for st in re.findall(r"\{(.*?)\}", act_label):
            act_label = self._sub_obj(act_label, st, obj_name_map)
        return act_label

    def _sanitize_obj_then_pos(
        self,
        act_label: str,
        obj_name_map: Dict[str, str],
        obj_pos_map: Dict[str, str],
        pos_map: Dict[str, str],
    ) -> str:
        """Replace 0th stub as obj and 1th (optional) stub as pos."""
        stubs = re.findall(r"\{(.*?)\}", act_label)
        if not stubs:
            return act_label
        act_label = self._sub_obj(act_label, stubs[0], obj_name_map)
        if len(stubs) > 1:
            act_label = self._sub_pos(act_label, stubs[1], obj_pos_map, pos_map)
        return act_label

    # ---------- subtask/act loop ----------
    @staticmethod
    def _build_sub_tasks_and_acts(
        act_labels: List[str],
        st_boundaries: List[int],
        sanitize: Callable[[str], str],
        post_process: Optional[Callable[[int, str], str]] = None,
    ) -> Tuple[List[Tuple[int, str]], List[Any]]:
        """Build subtask boundaries and low-level action indices."""
        sub_task: List[Tuple[int, str]] = []
        act: List[Any] = []
        for start, end in zip(st_boundaries[:-1], st_boundaries[1:]):
            act_lab = sanitize(act_labels[start])
            if post_process is not None:
                act_lab = post_process(start, act_lab)
            sub_task.append((start, act_lab))
            act.extend(range(start, end))
            act.append("<done>")
        return sub_task, act

    # ---------- subclass entry point ----------
    def _sanitize_act_label(self, act_label: str, **ctx) -> str:
        raise NotImplementedError

    def create_scratchpad(self) -> ScratchPad:
        raise NotImplementedError


class TouchPickScratchPadConverter(BaseScratchPadConverter):
    """Touch one object then reset, then pick another object."""

    def _sanitize_act_label(self, act_label: str, obj_name_map: Dict[str, str]) -> str:
        stubs = re.findall(r"\{(.*?)\}", act_label)
        if len(stubs) > 0:
            return self._sub_obj(act_label, stubs[0], obj_name_map)
        if act_label == "INDETERMINABLE":
            return "Reset to <init_ee_pos>"
        return act_label

    def create_scratchpad(self) -> ScratchPad:
        """Create the scratchpad for a touch-pick trajectory."""
        prompt = self._info["prompts"][0]
        lang_prompt = self._traj.get_multi_modal_prompts(mode="natural_language")[0][0]

        objects = self._extract_objects(prompt)
        obj_name_map = {y: x for x, y in objects.items()}

        pos_map = self._actor_pos_map(objects)
        init_ee_pos = self._traj.get_agent_ee_pose()[0][:3]
        pos_map["init_ee_pos"] = self._pos_to_str(init_ee_pos)

        # Note: TouchPick keeps the trailing label and rewrites any 'Place' label
        # so the sequence always ends in a final pick subtask.
        act_labels = self._act_label_sequence()
        for x in range(1, len(act_labels)):
            if act_labels[x] == "":
                act_labels[x] = act_labels[x - 1]
            elif "Place" in act_labels[x]:
                act_labels[x] = f"Pick {{obj:{objects['obj1']}}} without hitting anything else"
        st_boundaries = self._get_act_idxes(act_labels)

        plan = [
            "Init: ",
            f"End effector is at {self._pos_to_str(init_ee_pos)}",
            "<obj0> is located at <pos0>",
            "<obj1> is located at <pos1>",
            "We first need to touch <obj0>, then reset to initial pose and then pick <obj1>",
        ]

        sub_task, act = self._build_sub_tasks_and_acts(
            act_labels,
            st_boundaries,
            sanitize=lambda lab: self._sanitize_act_label(lab, obj_name_map),
        )

        return {
            "lang_prompt": lang_prompt,
            "plan": plan,
            "sub_tasks": sub_task,
            "act": act,
            "obj_name_map": self._strip_obj_ids(objects),
            "pos_map": pos_map,
        }


class SwapScratchPadConverter(BaseScratchPadConverter):
    """Swap the positions of two objects via a buffer location."""

    def _sanitize_act_label(
        self,
        act_label: str,
        obj_name_map: Dict[str, str],
        obj_pos_map: Dict[str, str],
        pos_map: Dict[str, str],
    ) -> str:
        return self._sanitize_obj_then_pos(act_label, obj_name_map, obj_pos_map, pos_map)

    def create_scratchpad(self) -> ScratchPad:
        """Create the scratchpad for a swap trajectory."""
        prompt = self._info["prompts"][0]
        lang_prompt = self._traj.get_multi_modal_prompts(mode="natural_language")[0][0]

        objects = self._extract_objects(prompt)
        obj_name_map = {y: x for x, y in objects.items()}
        object_pos = {x: f"<{x.replace('obj', 'pos')}>" for x in objects}

        pos_map = self._actor_pos_map(objects)
        pos_map["pos-temp"] = self._pos_to_str(
            _parse_literal(self._info["init_state"]["buffer_pose"])[0]
        )
        pos_to_name = {
            str(
                self._round_pos(
                    _parse_literal(self._info["init_state"]["objects"][f"{{obj:{x}}}"])[0]
                )
            ): obj_name_map[x]
            for x in obj_name_map
        }

        act_labels = self._act_label_sequence()[:-1]
        act_labels = self._fill_empty_labels(act_labels)
        st_boundaries = self._get_act_idxes(act_labels)

        plan = [
            "Init: ",
            "<obj0> is located at <pos0>",
            "<obj1> is located at <pos1>",
            "To swap positions, need to move <obj0> to <pos-temp>, then move <obj1> to <pos0> and move <obj0> to <pos1>",
        ]

        sub_task, act = self._build_sub_tasks_and_acts(
            act_labels,
            st_boundaries,
            sanitize=lambda lab: self._sanitize_act_label(
                lab, obj_name_map, object_pos, pos_to_name
            ),
        )

        return {
            "lang_prompt": lang_prompt,
            "plan": plan,
            "sub_tasks": sub_task,
            "act": act,
            "obj_name_map": self._strip_obj_ids(objects),
            "pos_map": pos_map,
        }


class StackToppleScratchPadConverter(BaseScratchPadConverter):
    """Stack objects, then topple the structure."""

    def _sanitize_act_label(self, act_label: str, obj_name_map: Dict[str, str], *_) -> str:
        return self._sanitize_objs_only(act_label, obj_name_map)

    def create_scratchpad(self) -> ScratchPad:
        """Create the scratchpad for a stack-topple trajectory."""
        prompt = self._info["prompts"][0]
        lang_prompt = self._traj.get_multi_modal_prompts(mode="natural_language")[0][0]

        objects = self._extract_objects(prompt, dedupe=True)
        obj_name_map = {y: x for x, y in objects.items()}
        object_pos = {x: self._get_init_pos_in_pixels(objects[x]) for x in objects}

        pos_map = self._actor_pos_map(objects)

        # StackTopple does NOT carry empty labels forward; it just trims the trailing one.
        act_labels = self._act_label_sequence()[:-1]
        st_boundaries = self._get_act_idxes(act_labels)

        init_pos = [f"<{x}> is located at <{x.replace('obj', 'pos')}>" for x in objects]
        plan = (
            ["Init: "]
            + init_pos
            + [
                f"To stack, first {self._sanitize_act_label(prompt, obj_name_map, object_pos, pos_map)}"
            ]
        )

        def _post(start: int, lab: str) -> str:
            return "Topple the structure" if "Topple the structure" in lab else lab

        sub_task, act = self._build_sub_tasks_and_acts(
            act_labels,
            st_boundaries,
            sanitize=lambda lab: self._sanitize_act_label(lab, obj_name_map, object_pos, pos_map),
            post_process=_post,
        )

        return {
            "lang_prompt": lang_prompt,
            "plan": plan,
            "sub_tasks": sub_task,
            "act": act,
            "obj_name_map": self._strip_obj_ids(objects),
            "pos_map": pos_map,
        }


class PlaceNextToRestoreScratchPadConverter(BaseScratchPadConverter):
    """Place an object next to another, then restore it."""

    def _sanitize_act_label(
        self,
        act_label: str,
        obj_name_map: Dict[str, str],
        obj_pos_map: Dict[str, str],
        pos_map: Dict[str, str],
        prompt: bool = False,
    ) -> str:
        if prompt:
            return self._sanitize_objs_only(act_label, obj_name_map)
        return self._sanitize_obj_then_pos(act_label, obj_name_map, obj_pos_map, pos_map)

    def create_scratchpad(self) -> ScratchPad:
        """Create the scratchpad for a place-next-to-restore trajectory."""
        prompt = self._info["prompts"][0]
        lang_prompt = self._traj.get_multi_modal_prompts(mode="natural_language")[0][0]

        objects = self._extract_objects(prompt, dedupe=True)
        obj_name_map = {y: x for x, y in objects.items()}
        object_pos = {x: f"<{x.replace('obj', 'pos')}>" for x in objects}

        pos_map = self._actor_pos_map(objects)
        pos_to_name = {
            str(
                self._round_pos(
                    _parse_literal(self._info["init_state"]["objects"][f"{{obj:{x}}}"])[0]
                )
            ): obj_name_map[x]
            for x in obj_name_map
        }

        act_labels = self._act_label_sequence()[:-1]
        act_labels = self._fill_empty_labels(act_labels)
        st_boundaries = self._get_act_idxes(act_labels)

        init_pos = [f"<{x}> is located at <{x.replace('obj', 'pos')}>" for x in objects]
        first_step = self._sanitize_act_label(
            prompt, obj_name_map, object_pos, pos_map, prompt=True
        )
        plan = ["Init: "] + init_pos + [f"First {first_step}"]

        # The first subtask label gets replaced with the plan's first step minus
        # the "restore" suffix (matches the original behaviour exactly).
        first_subtask_label = (
            plan[-1].replace(" and restore to initial position", "").replace("First ", "")
        )

        def _post(start: int, lab: str) -> str:
            return first_subtask_label if start == 0 else lab

        sub_task, act = self._build_sub_tasks_and_acts(
            act_labels,
            st_boundaries,
            sanitize=lambda lab: self._sanitize_act_label(
                lab, obj_name_map, object_pos, pos_to_name
            ),
            post_process=_post,
        )

        return {
            "lang_prompt": lang_prompt,
            "plan": plan,
            "sub_tasks": sub_task,
            "act": act,
            "obj_name_map": self._strip_obj_ids(objects),
            "pos_map": pos_map,
        }


class RotateRestoreScratchPadConverter(BaseScratchPadConverter):
    """Rotate an object then rotate it back to restore."""

    # Rotate uses change-detection boundaries, not first-occurrence.
    _get_act_idxes = staticmethod(BaseScratchPadConverter._get_act_idxes_on_change)

    def _sanitize_act_label(
        self,
        act_label: str,
        obj_name_map: Dict[str, str],
        obj_pos_map: Dict[str, str],
        pos_map: Dict[str, str],
        prompt: bool = False,
    ) -> str:
        if prompt:
            return self._sanitize_objs_only(act_label, obj_name_map)
        return self._sanitize_obj_then_pos(act_label, obj_name_map, obj_pos_map, pos_map)

    @staticmethod
    def _fix_rotate_labels(act_labels: List[str], boundaries: List[int]) -> List[str]:
        """Flip clockwise<->anti-clockwise for the second rotate, and make all angles positive."""
        rot_boundaries = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if act_labels[start].lower().split()[0] == "rotate":
                rot_boundaries.append((start, end))

        assert len(rot_boundaries) == 2
        st2, end2 = rot_boundaries[1]
        for idx in range(st2, end2):
            lab = act_labels[idx]
            if "anti" in lab.split()[-1].lower():
                act_labels[idx] = lab.replace("anti-clockwise", "clockwise")
            else:
                act_labels[idx] = lab.replace("clockwise", "anti-clockwise")

        for st, end in rot_boundaries:
            for idx in range(st, end):
                lab = act_labels[idx]
                angle = lab.split()[-3]
                act_labels[idx] = lab.replace(angle, str(abs(int(angle))))

        return act_labels

    def create_scratchpad(self) -> ScratchPad:
        """Create the scratchpad for a rotate-restore trajectory."""
        prompt = self._info["prompts"][0]
        lang_prompt = self._traj.get_multi_modal_prompts(mode="natural_language")[0][0]

        objects = self._extract_objects(prompt, dedupe=True)
        obj_name_map = {y: x for x, y in objects.items()}
        object_pos = {x: f"<{x.replace('obj', 'pos')}>" for x in objects}

        pos_map = self._actor_pos_map(objects)
        pos_to_name = {
            str(
                self._round_pos(
                    _parse_literal(self._info["init_state"]["objects"][f"{{obj:{x}}}"])[0]
                )
            ): obj_name_map[x]
            for x in obj_name_map
        }

        act_labels = self._act_label_sequence()[:-1]
        act_labels = self._fill_empty_labels(act_labels)
        st_boundaries = self._get_act_idxes(act_labels)
        act_labels = self._fix_rotate_labels(act_labels, st_boundaries)

        init_pos = [f"<{x}> is located at <{x.replace('obj', 'pos')}>" for x in objects]
        first_step = self._sanitize_act_label(
            prompt, obj_name_map, object_pos, pos_map, prompt=True
        )
        plan = ["Init: "] + init_pos + [f"First {first_step}"]

        sub_task, act = self._build_sub_tasks_and_acts(
            act_labels,
            st_boundaries,
            sanitize=lambda lab: self._sanitize_act_label(
                lab, obj_name_map, object_pos, pos_to_name
            ),
        )

        return {
            "lang_prompt": lang_prompt,
            "plan": plan,
            "sub_tasks": sub_task,
            "act": act,
            "obj_name_map": self._strip_obj_ids(objects),
            "pos_map": pos_map,
        }


# ---------- task name -> converter ----------
TASK_CONVERTERS = {
    "touchpick": TouchPickScratchPadConverter,
    "swap": SwapScratchPadConverter,
    "stacktopple": StackToppleScratchPadConverter,
    "placenexttorestore": PlaceNextToRestoreScratchPadConverter,
    "rotaterestore": RotateRestoreScratchPadConverter,
}


# ---------- post-processing helpers (unchanged in behaviour) ----------
def simplify_scratchpad(pad: ScratchPad) -> ScratchPad:
    """Replace detailed plan and subtask labels with generic labels."""
    pad["plan"] = [f"Complete the next {len(pad['sub_tasks'])} subtasks"]
    for idx, (start, _) in enumerate(pad["sub_tasks"]):
        pad["sub_tasks"][idx] = (start, f"Subtask {idx}")
    return pad


def project_3d_point_to_2d(
    pos: Sequence[float],
    scratch: BaseScratchPadConverter,
    resolution: Tuple[int, int] = (256, 256),
    bins: int = -1,
) -> Tuple[int, int]:
    """Project a 3D world point to image coordinates for the base camera."""
    camera_params = scratch._traj.get_camera_param(scratch._traj.Camera.BASE)
    in_cam = camera_params["intrinsic_cv"]
    ex_cam = camera_params["extrinsic_cv"]
    pos_homogeneous = np.array([pos[0], pos[1], pos[2], 1.0])

    transformed_pos = pos_homogeneous.dot(ex_cam[:3].T)
    if transformed_pos[2] <= 0:
        raise ValueError("The point is behind the camera, cannot project to 2D.")
    pos_img = transformed_pos.dot(in_cam.T)
    projected_x = pos_img[0] / pos_img[2]
    projected_y = pos_img[1] / pos_img[2]

    projected_x = np.clip(projected_x, 0, resolution[0] - 1)
    projected_y = np.clip(projected_y, 0, resolution[1] - 1)

    if bins > 0:
        return int(projected_x / resolution[0] * bins), int(projected_y / resolution[1] * bins)
    return int(projected_x), int(projected_y)


def shift_bbox(
    bbox: Sequence[float], pos_img: Sequence[float], resolution: Tuple[int, int] = (256, 256)
) -> List[float]:
    """Shift a bounding box to a new center point and clip it to the image."""
    bbox = np.array(bbox)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    shift_x = pos_img[0] - center_x
    shift_y = pos_img[1] - center_y
    bbox[0] += shift_x
    bbox[1] += shift_y
    bbox[2] += shift_x
    bbox[3] += shift_y
    bbox[0] = max(0, min(bbox[0], resolution[0] - 1))
    bbox[1] = max(0, min(bbox[1], resolution[1] - 1))
    bbox[2] = max(0, min(bbox[2], resolution[0] - 1))
    bbox[3] = max(0, min(bbox[3], resolution[1] - 1))
    return bbox.tolist()


def update_positions_with_bbox(pad: ScratchPad, scratch: BaseScratchPadConverter) -> ScratchPad:
    """Replace position tokens with object bounding boxes from trajectory metadata."""
    pos_map = pad["pos_map"]
    obj_name_map = pad["obj_name_map"]
    obj_name_map_rev = {v: k for k, v in obj_name_map.items()}

    obj_bboxes = scratch._traj.get_bounding_boxes()
    for (obj, _obj_id), bboxes in obj_bboxes.items():
        obj_key = obj_name_map_rev.get(obj, None)
        if obj_key is not None:
            pos_key = obj_key.replace("obj", "pos")
            pos_map[pos_key] = bboxes[0][:, :2].flatten()
            pad["pos_map"] = pos_map

    if "pos-temp" in pos_map:
        pos_temp_img = project_3d_point_to_2d(_parse_literal(pos_map["pos-temp"]), scratch)
        pos_map["pos-temp"] = shift_bbox(pos_map["pos0"], pos_temp_img)

    for key in pos_map:
        if isinstance(pos_map[key], list):
            pos_map[key] = str(pos_map[key])
        elif isinstance(pos_map[key], np.ndarray):
            pos_map[key] = str(pos_map[key].tolist())
    pad["pos_map"] = pos_map
    return pad


def update_positions_with_2dpoints(pad: ScratchPad, scratch: BaseScratchPadConverter) -> ScratchPad:
    """Replace position tokens with coarse 2D image-bin coordinates."""
    pos_map = pad["pos_map"]
    for k in pos_map:
        if "pos" in k:
            bbox = np.array(_parse_literal(pos_map[k]))
            point = project_3d_point_to_2d(bbox, scratch, bins=10)
            pos_map[k] = str(point)
    pad["pos_map"] = pos_map
    return pad


def _missing_required_files(traj_path: str) -> List[str]:
    """Return trajectory files required before a scratchpad converter can be built."""
    return [
        filename
        for filename in REQUIRED_SCRATCHPAD_TRAJ_FILES
        if not os.path.exists(os.path.join(traj_path, filename))
    ]


def main(args: argparse.Namespace) -> None:
    """Generate scratchpad JSON files for all trajectories under a task path."""
    trajectories = [x for x in glob.glob(os.path.join(args.path, "traj_*")) if os.path.isdir(x)]

    if args.task not in TASK_CONVERTERS:
        raise ValueError(f"Invalid task name '{args.task}'. Choose from {list(TASK_CONVERTERS)}")
    Converter = TASK_CONVERTERS[args.task]

    for traj in tqdm.tqdm(trajectories):
        missing_files = _missing_required_files(traj)
        if missing_files:
            tqdm.tqdm.write(
                "Warning: skipping incomplete trajectory "
                f"{traj}; missing {', '.join(missing_files)}"
            )
            continue

        scratch = Converter(traj)

        try:
            pad = scratch.create_scratchpad()
        except Exception as e:
            print(f"Error in creating scratchpad for {traj}: {e}")
            continue

        if args.simplify:
            pad = simplify_scratchpad(pad)
        if args.use_bbox:
            pad = update_positions_with_bbox(pad, scratch)
        if args.use_2dpoints:
            pad = update_positions_with_2dpoints(pad, scratch)

        with open(os.path.join(traj, f"scratchpad_{args.scratchpad_version}.json"), "w") as fp:
            json.dump(pad, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to trajectories")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_CONVERTERS.keys()))
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument(
        "--use_bbox", action="store_true", help="Use bounding box for object positions"
    )
    parser.add_argument(
        "--use_2dpoints", action="store_true", help="Use 2d image points for object positions"
    )
    parser.add_argument("--scratchpad_version", type=str, default="v1")
    args = parser.parse_args()

    main(args)
