# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import cv2
import numpy as np
from scipy.cluster.vq import kmeans

from clevr_skills.assets.video.video_template import (
    crop_image,
    crop_image_area,
    draw_circle,
    draw_line_in_horizontal_bar,
    get_area_size,
    get_multi_modal_text_image,
    get_template_image,
    get_text_image,
    paste_image,
    single_color_image,
    template,
)
from clevr_skills.dataset_converters.llava import LlavaActionTraceConverter
from clevr_skills.dataset_converters.nat_lang_english import NatLangEngActionTraceConverter
from clevr_skills.utils.visualize_prompt import PromptVisualizer, split_actor_name_id

from .record_env import RecordEnv


class RecordPrettyEnv(RecordEnv):
    """
    Environment wrapper that renders 'pretty' videos based on a template image.

    It shows:
    - prompt (natural language or multi-modal)
    - main (human) camera
    - wrist camera
    - EE control (action)
    - Action labels (various formats)
    - reward

    """

    def __init__(
        self,
        *args,
        prompt_visualization_mode=PromptVisualizer.MULTI_MODAL,
        action_label_visualization_mode=PromptVisualizer.NATURAL_LANGUAGE,
        render_bounding_boxes: bool = False,
        **kwargs,
    ):
        """
        :param args: passed on to parent class (RecordEnv)
        :param prompt_visualization_mode: Can be PromptVisualizer.NATURAL_LANGUAGE,
        PromptVisualizer.MULTI_MODAL
        :param action_label_visualization_mode: Can be PromptVisualizer.NATURAL_LANGUAGE,
        PromptVisualizer.MULTI_MODAL
        :pararm render_bounding_boxes: Whether to render the bounding boxes on top of the camera images.
        :param kwargs: passed on to parent class (RecordEnv)
        """
        super().__init__(*args, **kwargs)
        self.template = template
        self.template_image = np.flip(get_template_image(), axis=2)

        self.prompt_visualization_mode = prompt_visualization_mode
        self.action_label_visualization_mode = action_label_visualization_mode
        self.prompt_visualizers: Dict[int, PromptVisualizer] = {}
        self.render_bounding_boxes = render_bounding_boxes

    def _get_prompt_visualizer(self, multi_modal_image_height: int):
        """
        Internal function; create prompt visualizer and caches it.
        :param multi_modal_image_height: Target height of visualizer.
        :return: Prompt visualizer
        """
        if not multi_modal_image_height in self.prompt_visualizers:
            self.prompt_visualizers[multi_modal_image_height] = PromptVisualizer(
                img_height=multi_modal_image_height
            )
        return self.prompt_visualizers[multi_modal_image_height]

    def _get_action_labels(self, prompt: str):
        """
        Returns action labels.
        :param prompt: The prompt is required because it might affect the labels.
        :return: Action labels.
        """
        action_labels = [["action label not available"]] * len(self._render_images)
        traj_path = self._episode_info["reset_kwargs"]["options"]["record_dir"]
        if self.action_label_visualization_mode == PromptVisualizer.MULTI_MODAL:
            for idx, _ in enumerate(self._render_images):
                labels = [self.unwrapped._get_eps_info()["prompts"][0]]
                if idx in self.extra_info and "act_label" in self.extra_info[idx]:
                    labels.append(self.extra_info[idx]["act_label"]["mid_act_label"])
                    low_level_label = self.extra_info[idx]["act_label"]["low_act_label"]
                    if low_level_label != "Do nothing":
                        labels.append(low_level_label)
                action_labels[idx] = labels
        elif self.action_label_visualization_mode == PromptVisualizer.NATURAL_LANGUAGE:
            latc = NatLangEngActionTraceConverter(traj_path)
            labels = latc.extract_labels(prompt)
            action_labels = latc.expand_labels(labels, len(self._render_images))
        elif self.action_label_visualization_mode == "llava_python":
            latc = LlavaActionTraceConverter(traj_path)
            labels = latc.extract_labels(prompt)
            action_labels = latc.expand_labels(labels, len(self._render_images))

        return action_labels

    def _get_actor_colors(self):
        """
        Uses K-Means clustering on the texture to retrieve a representative color for each actor.
        :return: (r, g, b)
        """
        texture_info = self.unwrapped._get_eps_info()["textures"]
        actor_color = {}
        mean_color = 128  # for the "whitening"
        for actor_name, (_texture_name, texture_path) in texture_info.items():
            if texture_path is None:
                color = [224, 224, 224]
            elif isinstance(texture_path, tuple):
                color = np.array(texture_path)[0:3] * 255
            else:
                texture = cv2.imread(str(texture_path))
                texture = (
                    cv2.resize(texture, (32, 32))[:, :, 0:3].astype(float).reshape((-1, 3))
                    - mean_color
                ) / mean_color

                colors = kmeans(texture, 4)[0]
                color = (
                    colors[np.argmax(np.var(colors, axis=1))] * mean_color + mean_color
                )  # take the color with the most variance
                color = np.round(color)
            actor_color[actor_name] = (int(color[2]), int(color[1]), int(color[0]))
        return actor_color

    def _render_bounding_boxes(self, image, bboxes, transform, actor_colors, area):
        """
        :param image: image (typically the full template image).
        :param bboxes: the bounding boxes.
        :param transform: 3x3 homogeneous transformation.
        :param actor_colors: dict from actor name -> color.
        :param area: the area inside the image that is available for rendering.
        :return: None; bounding boxes are rendered into the image.
        """

        def draw_label(image, label_image, area, area_mask, candidate_positions):
            """
            Internal function.

            Draws label_image into image at one of the candidate_positions,
            taking into account the area. Sets the area_mask to 0 where the label_image is rendered
            """
            # find best candidate_positions (i.e., has the most pixels available for rendering)
            best_pos = None
            best_count = -1
            for pos in candidate_positions:
                num_pixels_available = np.sum(
                    area_mask[
                        pos[1] : pos[1] + label_image.shape[0],
                        pos[0] : pos[0] + label_image.shape[1],
                    ]
                )
                if num_pixels_available > best_count:
                    best_pos = pos
                    best_count = num_pixels_available

            if best_count > 0:
                # clip label_image such that it will fit inside the area
                pos = list(best_pos)
                for i in range(2):
                    if pos[i] < area[0][i]:
                        d = area[0][i] - pos[i]
                        pos[i] += d
                        label_image = label_image[d:, :, :] if i else label_image[:, d:, :]
                for i in range(2):
                    if pos[i] > area[1][i]:
                        d = pos[i] - area[0][i]
                        label_image = label_image[:-d, :, :] if i else label_image[:, :-d, :]

                # draw label_image into image, update the area mask
                image[
                    pos[1] : pos[1] + label_image.shape[0],
                    pos[0] : pos[0] + label_image.shape[1],
                    :,
                ] = label_image
                area_mask[
                    pos[1] : pos[1] + label_image.shape[0], pos[0] : pos[0] + label_image.shape[1]
                ] = 0

        max_text_width = area[1][0] - area[0][0]

        white = (255, 255, 255)
        black = (0, 0, 0)

        line_thickness = 3

        # Get mask of available area to draw text into
        area_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        area_mask[area[0][1] : area[1][1], area[0][0] : area[1][0]] = 1

        for pass_idx in range(2):
            for actor_name_id, box_coords in bboxes.items():
                color = actor_colors[actor_name_id]
                text_color = white if np.mean(color) < 128 else black

                box_coords = np.matmul(transform, box_coords.T).T
                box_coords = np.round(box_coords).astype(np.int)
                box_coords[0, 0] = max(box_coords[0, 0], area[0][0])
                box_coords[0, 1] = max(box_coords[0, 1], area[0][1])
                box_coords[1, 0] = min(box_coords[1, 0], area[1][0])
                box_coords[1, 1] = min(box_coords[1, 1], area[1][1])
                if (box_coords[1, 0] - box_coords[0, 0] <= 0) or (
                    box_coords[1, 1] - box_coords[0, 1] <= 0
                ):
                    continue  # box is fully clipped

                if pass_idx == 0:
                    cv2.rectangle(
                        image,
                        tuple(box_coords[0, 0:2]),
                        tuple(box_coords[1, 0:2]),
                        color=color,
                        thickness=line_thickness,
                    )
                    cv2.rectangle(
                        area_mask,
                        tuple(box_coords[0, 0:2]),
                        tuple(box_coords[1, 0:2]),
                        color=0,
                        thickness=line_thickness,
                    )
                elif pass_idx == 1:
                    actor_name, _ = split_actor_name_id(actor_name_id)
                    text_image, width, _ = get_text_image(
                        actor_name,
                        max_text_width,
                        21,
                        background_color=color,
                        text_color=text_color,
                    )
                    width = int(width)
                    text_image = text_image[:, 0:width, :]
                    hl = line_thickness // 2
                    c1 = (box_coords[0, 0] - hl, box_coords[0, 1] - text_image.shape[0] - 1 - hl)
                    c2 = (
                        box_coords[1, 0] - width + hl,
                        box_coords[0, 1] - text_image.shape[0] - 1 - hl,
                    )
                    c3 = (c1[0], box_coords[1, 1] + 1 + hl)
                    c4 = (c2[0], box_coords[1, 1] + 1 + hl)

                    draw_label(image, text_image, area, area_mask, [c1, c2, c3, c4])

    def decorate_images(self):
        """
        Performs the actual rendering; combines camera, observation, action, prompt, action labels, etc.
        :return:
        """
        images = []

        clean_prompt = self.unwrapped._get_eps_info()["prompts"][0]

        # get all reward values
        rewards = [
            0.0 if self._episode_data[idx]["r"] is None else self._episode_data[idx]["r"]
            for idx in range(len(self._render_images))
        ]
        _, max_reward = min(rewards), max(rewards)

        # Prepare the action log image
        area = self.template["action_log"]
        area_size = get_area_size(area)
        action_log_background_color = tuple(
            self.template_image[area[0][1] + 20, area[0][0] + 20, :]
        )
        action_log_image = np.zeros((5, area_size[0], 3))
        action_log_image[:, :, 0:3] = action_log_background_color
        last_action_labels = [None] * 10

        action_labels = self._get_action_labels(clean_prompt)

        if self.render_bounding_boxes:
            actor_colors = self._get_actor_colors()
            main_bboxes = [
                x["o"]["extra"]["actor_mesh_bbox"]["render_camera"] for x in self._episode_data
            ]
            wrist_bboxes = [
                x["o"]["extra"]["actor_mesh_bbox"]["hand_camera"] for x in self._episode_data
            ]

        # render each image
        for idx, main_camera_image in enumerate(self._render_images):
            prompt = "Prompt: " + clean_prompt
            image = np.copy(self.template_image)

            area = self.template["main_camera"]
            main_camera_image, crop_bounds = crop_image(
                main_camera_image, self.template["main_camera_crop"], return_crop_bounds=True
            )

            transform = paste_image(image, main_camera_image, area, crop=True)
            transform = np.matmul(
                transform,
                np.array([[1, 0, -crop_bounds[0, 0]], [0, 1, -crop_bounds[0, 1]], [0, 0, 1]]),
            )

            if self.render_bounding_boxes:
                self._render_bounding_boxes(image, main_bboxes[idx], transform, actor_colors, area)

            area = self.template["wrist_camera"]
            wrist_camera_image = self._episode_data[idx]["o"]["image"]["hand_camera"]["rgb"]
            wrist_camera_image, crop_bounds = crop_image(
                wrist_camera_image, self.template["wrist_camera_crop"], return_crop_bounds=True
            )
            transform = paste_image(image, wrist_camera_image, area)
            transform = np.matmul(
                transform,
                np.array([[1, 0, -crop_bounds[0, 0]], [0, 1, -crop_bounds[0, 1]], [0, 0, 1]]),
            )

            if self.render_bounding_boxes:
                self._render_bounding_boxes(image, wrist_bboxes[idx], transform, actor_colors, area)

            # Render the action:
            action_color = self.template["end_effector_color"]
            if self._episode_data[idx]["a"] is not None:
                ee_x, ee_y, ee_z, ee_rx, ee_ry, ee_rz, ee_grasp = self._episode_data[idx]["a"]
                draw_circle(
                    image,
                    self.template["end_effector_xy"],
                    [0.5 + 0.5 * ee_x, 0.5 + 0.5 * ee_y],
                    color=action_color,
                )
                draw_line_in_horizontal_bar(
                    image, self.template["end_effector_z"], 0.5 + 0.5 * ee_z, color=action_color
                )
                draw_line_in_horizontal_bar(
                    image, self.template["end_effector_yaw"], 0.5 + 0.5 * ee_rz, color=action_color
                )
                draw_circle(
                    image,
                    self.template["end_effector_roll_pitch"],
                    [0.5 + 0.5 * ee_rx, 0.5 + 0.5 * ee_ry],
                    color=action_color,
                )
                draw_line_in_horizontal_bar(
                    image,
                    self.template["end_effector_grasp"],
                    1.0 - (0.5 + 0.5 * ee_grasp),
                    color=action_color,
                )

            # Render the prompt
            area = self.template["prompt"]
            area_size = get_area_size(area)
            font_size = self.template["prompt_font_size"]
            multi_modal_image_size = int(round(font_size * 1.5))
            if self.prompt_visualization_mode in [
                PromptVisualizer.MULTI_MODAL,
                PromptVisualizer.NATURAL_LANGUAGE,
            ]:
                pv = self._get_prompt_visualizer(multi_modal_image_size)
                traj_path = self._episode_info["reset_kwargs"]["options"]["record_dir"]
                prompt = pv.visualize_prompt(
                    prompt, traj_path, compose_image=False, mode=self.prompt_visualization_mode
                )
            background_color = tuple(image[area[0][1] + 20, area[0][0] + 20, :])
            text_image = get_multi_modal_text_image(
                prompt,
                area_size[0] - 50,
                font_size=font_size,
                background_color=background_color,
                text_color=(255, 255, 255),
                alignment="center",
            )

            paste_image(
                image, text_image, area, crop=False, scale=False, pad_color=background_color
            )

            # render the reward
            area = self.template["reward"]
            area_size = get_area_size(area)
            reward_color = self.template["reward_color"]
            reward_prev_xy = None
            reward_image = np.copy(crop_image_area(image, area))
            for ridx, reward in enumerate(rewards[0 : idx + 1]):
                x = int(round(ridx / len(rewards) * area_size[0]))
                y = area_size[1] - 1 - int(round(reward / max_reward * area_size[1]))
                if reward_prev_xy is None:
                    reward_image[y : y + 1, x : x + 1, :] = reward_color
                else:
                    cv2.line(
                        reward_image,
                        (x, y),
                        reward_prev_xy,
                        color=reward_color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                reward_prev_xy = (x, y)
            paste_image(
                image,
                reward_image,
                area,
                crop=False,
                rounded_corners=self.template["reward_rounded_corners"],
            )

            # Render the actions labels
            area = self.template["action_log"]
            area_size = get_area_size(area)
            text_colors = self.template["action_log_text_color"]
            font_sizes = self.template["action_log_font_size"]
            if idx in self.extra_info and "act_label" in self.extra_info[idx]:
                labels = action_labels[idx]
                for depth, label in enumerate(labels):
                    if last_action_labels[depth] != label:
                        last_action_labels[depth] = label
                        for d in range(depth + 1, len(last_action_labels)):
                            last_action_labels[d] = None

                        action_label = label
                        font_size = font_sizes[depth]
                        if self.action_label_visualization_mode in [
                            PromptVisualizer.MULTI_MODAL,
                            PromptVisualizer.NATURAL_LANGUAGE,
                        ]:
                            multi_modal_image_size = int(round(font_size * 1.5))
                            pv = self._get_prompt_visualizer(multi_modal_image_size)
                            traj_path = self._episode_info["reset_kwargs"]["options"]["record_dir"]
                            action_label = pv.visualize_prompt(
                                action_label,
                                traj_path,
                                compose_image=False,
                                mode=self.action_label_visualization_mode,
                            )
                        indentation = font_sizes[0] * depth
                        text_image = get_multi_modal_text_image(
                            action_label,
                            action_log_image.shape[1] - indentation,
                            font_size=font_size,
                            background_color=action_log_background_color,
                            text_color=text_colors[depth],
                            alignment="left",
                        )
                        if indentation > 0:
                            indentation_image = single_color_image(
                                [text_image.shape[0], indentation, text_image.shape[2]],
                                action_log_background_color,
                            )
                            text_image = np.concatenate([indentation_image, text_image], axis=1)

                        # add the image to the log
                        action_log_image = np.concatenate([action_log_image, text_image], axis=0)
            if action_log_image.shape[0] > area_size[1]:
                diff = action_log_image.shape[0] - area_size[1]
                scroll = max(int(round(diff / 10)), 1)
                action_log_image = action_log_image[scroll:, :, :]
            padded_action_log_image = action_log_image
            if action_log_image.shape[0] < area_size[1]:
                padding_image = np.zeros(
                    (area_size[1] - action_log_image.shape[0], area_size[0], 3)
                )
                padding_image[:, :, 0:3] = action_log_background_color
                padded_action_log_image = np.concatenate([action_log_image, padding_image], axis=0)

            paste_image(
                image,
                padded_action_log_image[0 : area_size[1], :, :],
                area,
                crop=False,
                scale=False,
                pad_color=action_log_background_color,
            )

            images.append(image)
        return images
