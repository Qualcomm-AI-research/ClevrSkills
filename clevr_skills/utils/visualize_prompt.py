# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import json
import os
import re
from typing import Dict, Tuple

import cv2
import numpy as np
import sapien.core as sapien

from clevr_skills.utils.textures import _COLORS, _TEXTURES


def get_actor_description(actor: sapien.ActorBase):
    """
    :param actor:
    :return: Placeholder description of actor.
    """
    return f"{{obj:{actor.name}_{actor.id}}}"


def get_texture_description(actor: sapien.ActorBase):
    """
    :param actor:
    :return: Placeholder description of texture.
    """
    return f"{{tex:{actor.name}_{actor.id}}}"


def split_actor_name_id(actor_name_id: str) -> Tuple[str, int]:
    """
    :param actor_name_id: A string such as "red cube_51"
    :return: Tuple (name, id), e.g., ["red cube", 51].
    """
    s = actor_name_id.split("_")
    actor_name = "_".join(s[:-1])
    actor_id = int(s[-1])
    return (actor_name, actor_id)


def load_texture(name: str, width: int = 256, height: int = 256) -> np.array:
    """
    Loads a known texture; if texture can't be read, a white texture of specified size if returned.
    :param name: Name of texture.
    :param width: Width of texture (in case of loading failure).
    :param height: Height of texture (in case of loading failure).
    :return:
    """
    matching_textures = [
        (name, str(texture_path))
        for texture_name, texture_path in _TEXTURES.items()
        if name.startswith(texture_name)
    ]
    if len(matching_textures) == 0:
        matching_textures = [
            (name, color) for color_name, color in _COLORS.items() if name.startswith(color_name)
        ]
    texture = matching_textures[0]

    if isinstance(texture[1], str):
        asset_img = cv2.imread(texture[1])[..., ::-1]
    else:
        asset_img = np.ones((height, width, 3)) * texture[1] * 255
        asset_img = asset_img.astype(np.uint8)

    return asset_img


class PromptVisualizer:
    """
    Converts a prompt into an image.
    E.g., for visualization in a video.
    """

    # How to return object and texture references:
    NATURAL_LANGUAGE = "natural_language"  # references are replaced with natural language
    MULTI_MODAL = "multi_modal"  # references are replaced with images
    RAW = "placeholder"  # the raw references are returned

    def __init__(
        self, img_height: int = 128, background_color: Tuple[int, int, int] = None
    ) -> None:
        """
        Initializes
        :param img_height: Height of generated image. If there is too much content,
        it will be clipped.
        :param background_color: Background color of generated image.
        """
        self.img_height = img_height
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 1.0
        self.FONT_THICKNESS = 2
        self.background_color = background_color

    def text_to_rgb(self, text, margin=5):
        """
        Converts text into image.
        :param text: The text.
        :param margin: margin (on the right side).
        :return: Image with rendered text.
        """

        (text_width, text_height), _ = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
        )

        temp_img = np.ones((self.img_height, text_width + margin, 3), dtype=np.uint8) * 255
        temp_img = cv2.putText(
            temp_img,
            text,
            (0, (text_height + self.img_height) // 2),
            self.FONT,
            self.FONT_SCALE,
            (0, 0, 0),
            self.FONT_THICKNESS,
        )

        return temp_img

    def _get_asset_nat_lang(self, asset_raw, info_json: Dict, data_path: str):
        """
        Converts an asset to natural language.
        :param asset_raw: Asset string, such as "obj" or "tex".
        :param info_json: Info (for textures)
        :param data_path: Not used.
        :return: Natural language description of asset.
        """
        atype, aname = asset_raw.split(":")
        if atype in ["pos", "ks"]:
            return asset_raw
        if atype == "obj":
            return " the " + "_".join(aname.split("_")[:1])
        if atype == "tex":
            return info_json["textures"][aname][0]
        raise RuntimeError(f"Asset type '{atype}' not known")

    def _get_asset_img(self, aname, data_path: str):
        """
        :param aname: symbolic name of an asset.
        :param data_path: Path to trajectory; used to load asset (such as key step or object image).
        :return: Image of asset.
        """
        atype, aname = aname.split(":")
        if atype == "ks":
            ks_idx = aname.split("_")[1]
            asset_img = cv2.imread(f"{data_path}/keysteps/{ks_idx}_bimg.jpg")
        elif atype == "obj":
            img_paths = [f"{data_path}/objects/{aname}.png", f"{data_path}/objects/{aname}.jpg"]
            for img_path in img_paths:  # support jpeg for compatibility with older datasets
                if os.path.isfile(img_path):
                    asset_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    break
            if asset_img.shape[2] == 4:
                if self.background_color is None:
                    asset_img = asset_img[:, :, 0:3]
                else:  # alpha channel -> background color
                    asset_img = asset_img.astype(np.int)
                    alpha = asset_img[:, :, 3:4]
                    asset_img = asset_img[:, :, 0:3]
                    background_image = np.zeros(asset_img.shape, dtype=np.int)
                    background_image[:, :, 0:3] = self.background_color
                    asset_img = asset_img[:, :, 0:3] * alpha + background_image * (255 - alpha)
                    asset_img = np.round(asset_img / 255).astype(np.uint8)
        elif atype == "tex":
            asset_img = load_texture(aname)
        else:
            raise RuntimeError(f"Asset type '{atype}' not known")

        asset_img = self._resize_asset_img(asset_img)
        return asset_img

    def _resize_asset_img(self, img):
        """
        Resizes an asset such that it will fit.
        :param img: Asset image.
        :return: Resized asset image.
        """
        H, W = img.shape[:2]

        img_width = int(W / H * self.img_height)
        return cv2.resize(img, (img_width, self.img_height))

    def visualize_prompt(
        self, prompt: str, data_path: str, mode: str = MULTI_MODAL, compose_image: bool = True
    ):
        """
        :param prompt: The (multi-modal) prompt that must be visualized.
        :param data_path: Path to trajectory.
        :param mode: Visualization mode. Can be PromptVisualizer.MULTI_MODAL or PromptVisualizer.NATURAL_LANGUAGE.
        :param compose_image: if True, text is converted to images using OpenCV and a single
        image is returned. If false, a list of strings and images is returned (it is up to the
        caller to compose the final image)
        """

        with open(os.path.join(data_path, "info.json"), "r", encoding="utf8") as fp:
            info_json = json.load(fp)

        text_splits = re.split(r"\{(.*?)\}", prompt)[0::2]
        asset_names = re.findall(r"\{(.*?)\}", prompt)

        if compose_image:
            img_splits = [self.text_to_rgb(text_splits[0])]
        elif len(text_splits) > 0:
            img_splits = [text_splits[0]]
        else:
            img_splits = [""]

        for ts, asset in zip(text_splits[1:], asset_names):

            if mode == self.MULTI_MODAL:
                visualization = self._get_asset_img(asset, data_path=data_path)
            elif mode == self.NATURAL_LANGUAGE:
                visualization = self._get_asset_nat_lang(asset, info_json, data_path=data_path)
            else:
                visualization = asset

            img_splits.append(visualization)

            if compose_image:
                text_img = self.text_to_rgb(ts)
                img_splits.append(text_img)
            else:
                if len(ts) > 0:
                    img_splits.append(ts)

        if compose_image:
            return np.concatenate(img_splits, axis=1)
        if len(img_splits) > 0:
            result = [img_splits[0]]
            for text_or_image in img_splits[1:]:
                if isinstance(text_or_image, str) and isinstance(result[-1], str):
                    result[-1] = re.sub(
                        " +", " ", result[-1] + text_or_image
                    )  # fuse + remove double spaces
                else:
                    result.append(text_or_image)
            return result
        return img_splits


def main():
    """
    Simple main() function for testing.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-path", type=str, required=True, help="Trajectory path.")

    args = parser.parse_args()
    vis = PromptVisualizer()
    img = vis.visualize_prompt(
        "Follow the motion for {obj:cylinder_20}: {ks:keystep_1}",
        args.trajectory_path,
    )
    cv2.imwrite("check_prompt.jpg", img)


if __name__ == "__main__":
    main()
