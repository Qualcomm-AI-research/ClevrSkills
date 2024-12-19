# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import copy
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clevr_skills.utils.render import scale_and_crop_image

# Areas in the video template
# Sizes: [width, height]
# Area: [[upper_X, left_Y], [lower_X, right_Y]]
# Crop: {dict}
template = {
    "image_size": [2280, 1504],  # expected image size
    "prompt": [[50, 14], [2224, 223]],
    "prompt_font_size": 50,
    "main_camera": [[16, 240], [1505, 1146]],
    "main_camera_crop": {"left": 0.0, "right": 0.0, "top": 0.2, "bottom": 0.0},
    "wrist_camera": [[1534, 240], [2257, 660]],
    "wrist_camera_crop": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.15},
    "end_effector_color": (255, 0, 0),  # color used for rendering the lines & circles
    "end_effector_xy": [[1590, 759], [1851, 935]],
    "end_effector_z": [[1879, 759], [1967, 935]],
    "end_effector_grasp": [[2145, 759], [2235, 935]],
    "end_effector_roll_pitch": [[1708, 978], [1975, 1155]],
    "end_effector_yaw": [[2000, 978], [2090, 1155]],
    "action_log": [[56, 1174], [1464, 1480]],
    "action_log_text_color": [
        (255, 255, 255),
        (224, 224, 224),
        (192, 192, 192),
        (160, 160, 160),
        (128, 128, 128),
    ],
    "action_log_font_size": [40, 32, 24, 20, 16, 14],
    "reward": [[1598, 1235], [2230, 1478]],
    "reward_color": (64, 64, 64),  # color used for rendering the lines & circles
    "reward_rounded_corners": 32,  # radius of rounded corners of reward area
}


def get_template_image() -> np.ndarray:
    """
    :return: Template image, read from hard-coded file that is part of the repo.
    """
    image_path = os.path.join(os.path.split(__file__)[0], "video_template.png")
    image = cv2.imread(image_path)[:, :, 0:3]
    assert image.shape[0] == template["image_size"][1]
    assert image.shape[1] == template["image_size"][0]
    return image.astype(np.uint8)


def get_area_size(area: List[List[int]]) -> List[int]:
    """
    :param area: area definition from template
    :return: width and height of the area
    """
    return [area[1][0] - area[0][0], area[1][1] - area[0][1]]


def crop_image(image: np.ndarray, crop: Dict[str, float], return_crop_bounds: bool = False):
    """
    (Pre-) crops an image according to the crop specification from the template.
    :param image: The image.
    :param crop: dictionary with keys "left", "right", "top",  "bottom", specifying the
    fraction that should be cropped from each side.
    :param return_crop_bounds: return the crop bounds, in addition to the cropped image.
    Bounds are returned as numpy array ((ulx, uly), (lrx, lry)).
    :return: If return_crop_bounds is True, return (cropped_image, bounds), otherwise just the cropped_image is returned.
    """
    ulx = int(round(image.shape[1] * crop["left"]))
    lrx = int(round(image.shape[1] - image.shape[1] * crop["right"]))
    uly = int(round(image.shape[0] * crop["top"]))
    lry = int(round(image.shape[0] - image.shape[0] * crop["bottom"]))
    image = image[uly:lry, ulx:lrx, :]
    bounds = np.array(((ulx, uly), (lrx, lry)), dtype=np.int)
    return (image, bounds) if return_crop_bounds else image


def crop_image_area(image: np.ndarray, area: List[List[int]]):
    """
    :param image: The image
    :param area: The definition of the area bounds [[min_y, max_y], [min_x, max_x]]
    :return: The cropped image
    """
    return image[area[0][1] : area[1][1], area[0][0] : area[1][0], :]


def paste_image(
    dest: np.ndarray,
    src: np.ndarray,
    area: List[List[int]],
    crop: bool = True,
    scale: bool = False,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    rounded_corners: int = 0,
):
    """
    Pastes src in dest with scaling and cropping, following the area definition
    :param dest: Destination image.
    :param src: Source image.
    :param area: The area where to paste the source image.
    :param crop: Whether to crop to source image to fit the destination area.
    :param scale: Whether to crop the input image to fit the destination area.
    :param pad_color: The color used for padding.
    :param rounded_corners: If not 0, corners are rounded with a corresponding radius.
    :return: the transform that allows one to map text and other graphcs into the destination area.
    """
    area_size = get_area_size(area)
    src, transform = scale_and_crop_image(
        src, area_size, crop=crop, scale=scale, pad_color=pad_color, return_transform=True
    )

    if rounded_corners:
        mask = np.zeros((area_size[1], area_size[0], src.shape[2]), dtype=np.uint8)
        cv2.rectangle(
            mask,
            (rounded_corners, 0),
            (area_size[0] - rounded_corners, area_size[1]),
            color=(255, 255, 255),
            thickness=-1,
        )
        cv2.rectangle(
            mask,
            (0, rounded_corners),
            (area_size[0], area_size[1] - rounded_corners),
            color=(255, 255, 255),
            thickness=-1,
        )
        for x in [rounded_corners, area_size[0] - rounded_corners]:
            for y in [rounded_corners, area_size[1] - rounded_corners]:
                cv2.circle(
                    mask,
                    (x, y),
                    rounded_corners,
                    (255, 255, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        mask = mask.astype(int)
        dst = dest[area[0][1] : area[1][1], area[0][0] : area[1][0], :]
        src = np.round((mask * src + (255 - mask) * dst) / 255).astype(np.uint8)

    dest[area[0][1] : area[1][1], area[0][0] : area[1][0], :] = src

    transform = np.matmul(np.array([[1, 0, area[0][0]], [0, 1, area[0][1]], [0, 0, 1]]), transform)

    return transform


def draw_circle(
    image: np.ndarray,
    area: List[List[int]],
    value: List[float],
    radius: int = 7,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """
    Draw a circle in the image area.
    Used specifically to render the end-effector action.
    :param image: The image to render the circle into.
    :param area: The area to render the circle into.
    :param value: normalized X, Y coordinate, relatively to the area.
    :param radius: Radius of circle.
    :param color: Color of the circle.
    :return: None
    """
    area_size = get_area_size(area)
    pc = [int(round(area[0][c] + value[c] * area_size[c])) for c in [0, 1]]
    cv2.circle(image, pc, radius, color, thickness=2, lineType=cv2.LINE_AA)


def draw_line_in_horizontal_bar(
    image: np.ndarray,
    area: List[List[int]],
    value: float,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """
    Draws vertical line in a horizontal bar.
    Used specifically to render the end-effector action.
    :param image: The image to render the circle into.
    :param area: The area to render the circle into.
    :param value: normalized value, i.e., relative to the area.
    :param color: Color of the line.
    :return: None
    """
    area = copy.deepcopy(area)
    area[0][1] += 10
    area[1][1] -= 10
    area_size = get_area_size(area)
    py = int(round(area[0][1] + value * (area_size[1])))
    image[py, area[0][0] : area[1][0], 0:3] = color


def draw_line_in_vertical_bar(
    image: np.ndarray,
    area: List[List[int]],
    value: float,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """
    Draws horizontal line in a vertical bar.
    Used specifically to render the end-effector action.
    :param image: The image to render the circle into.
    :param area: The area to render the circle into.
    :param value: normalized value, i.e., relative to the area.
    :param color: Color of the line.
    :return: None
    """
    area = copy.deepcopy(area)
    area[0][0] += 10
    area[1][0] -= 10
    area_size = get_area_size(area)
    px = int(round(area[0][0] + value * area_size[0]))
    image[area[0][1] : area[1][1], px, 0:3] = color


def crop_text_image(text_image: np.ndarray):
    """
    Crops text image to minimal width (discarding white space on the right side).
    :param text_image: The image.
    :return: Cropped text image.
    """
    text_width = 10
    for text_width in reversed(range(text_width, text_image.shape[1])):
        mask = text_image[:, text_width] == text_image[-1, text_width]
        if not np.all(mask):
            break
    return text_image[:, 0 : text_width + 1, :]


font = {}


def get_text_image(
    text: str,
    image_width: int,
    font_size,
    background_color=(0, 0, 0),
    text_color=(255, 255, 255),
    single_line: bool = False,
) -> Tuple[np.ndarray, int, str]:
    """
    Renders text using QualcommNext-Regular.ttf font; returned as an image.
    :param text: The text to be rendered..
    :param image_width: width of image.
    :param font_size: height, in pixels
    :param background_color: Color used for filling the background.
    :param text_color: Color used for rendering text.
    :param single_line: if true, generate only a single line of text and returns the remaining text.
    :return: (text_image, actual_width, remaining_words)
    """
    global font
    if not font_size in font:
        font_path = os.path.join(os.path.split(__file__)[0], "QualcommNext-Regular.ttf")
        font[font_size] = ImageFont.truetype(font_path, size=font_size)

    # Turn '\n' into ' \n ' such that carriage returns become individual words
    text = " \n ".join(text.split("\n"))
    words = [w for w in text.split(" ") if len(w)]

    images = []
    max_width = 0
    while len(words):
        # skip \n as the start of a new sentence
        while len(words) > 0 and words[0] == "\n":
            words = words[1:]
        if len(words) == 0:
            break

        image_height = int(round(font_size * 1.25))
        img = Image.new(mode="RGBA", size=(image_width, image_height), color=background_color)
        draw = ImageDraw.Draw(img)

        num_words = 1
        carriage_return = False
        for nw in range(1, len(words) + 1):
            if words[nw - 1] == "\n":  # start a new line when \n is encountered
                carriage_return = True
                num_words = nw - 1
                break
            width = draw.textlength(" ".join(words[0:nw]), font=font[font_size])
            if width < image_width - font_size:
                num_words = nw
            else:
                break

        width = draw.textlength(" ".join(words[0:num_words]), font=font[font_size])
        max_width = max(max_width, width)
        draw.text((0, 0), " ".join(words[0:num_words]), font=font[font_size], fill=text_color)
        if carriage_return or (single_line and num_words < len(words)):
            # Put one pixel with a slightly-off-background color at the end of the line
            bg = background_color
            img.putpixel((image_width - 1, 0), (bg[0], bg[1], bg[2] ^ 1))
        words = words[num_words:]

        images.append(np.array(img))
        if single_line:
            break

    if len(images) == 0:
        # Make sure to return at least a single pixel:
        images = [np.array(background_color, dtype=np.uint8).reshape((1, 1, 3))]

    remaining_text = " ".join(words).replace(" \n ", "\n")
    if len(remaining_text) > 0:
        remaining_text = "" if np.all(np.array(list(remaining_text)) == " ") else remaining_text
    return np.concatenate(images, axis=0)[:, :, 0:3], max_width, remaining_text


def single_color_image(shape, color):
    """
    Returns an image of given shape, filled with color.
    :param shape: (H, W, C)
    :param color: Color used to fill the image.
    :return: Image.
    """
    image = 255 * np.ones(shape, dtype=np.uint8)
    image[:, :, 0 : len(color)] = color
    return image


def _concat_row(row, image_width, background_color, alignment: str):
    """
    Concatenates a row of images.
    :param row: list of images.
    :param image_width: Target width of concatenated row. Padding is used if necessary.
    :param background_color: Background color used for padding.
    :param alignment: How to align the row of images inside the final image: "justify", "left", "right" or "center".
    :return: a single image.
    """
    assert len(row) > 0
    height = np.max([image.shape[0] for image in row])

    # make all images the same height
    hp_row = []  # height-padded row
    for image in row:
        if image.shape[0] != height:
            top_pad = (height - image.shape[0]) // 2
            bottom_pad = height - image.shape[0] - top_pad
            top_pad = single_color_image(
                (top_pad, image.shape[1], image.shape[2]), background_color
            )
            bottom_pad = single_color_image(
                (bottom_pad, image.shape[1], image.shape[2]), background_color
            )
            image = np.concatenate([top_pad, image, bottom_pad], axis=0)
        hp_row.append(image)
    row = hp_row

    row_width = np.sum([image.shape[1] for image in row])

    if alignment == "justify" and row_width < image_width:
        j_row = [row[0]]
        for idx, image in enumerate(row[1:]):
            j_pad = int(round((image_width - row_width) / (len(row) - idx)))
            if j_pad > 0:
                row_width += j_pad
                j_pad = single_color_image((height, j_pad, image.shape[2]), background_color)
                j_row.append(j_pad)
            j_row.append(image)
        alignment = "left"

    pad_left = pad_right = 0
    if alignment == "left":
        pad_right = image_width - row_width
    elif alignment == "right":
        pad_left = image_width - row_width
    elif alignment == "center":
        pad_left = (image_width - row_width) // 2
        pad_right = image_width - row_width - pad_left
    if pad_left > 0:
        left_pad = single_color_image((height, pad_left, image.shape[2]), background_color)
        row = [left_pad] + row
    if pad_right > 0:
        right_pad = single_color_image((height, pad_right, image.shape[2]), background_color)
        row.append(right_pad)

    return np.concatenate(row, axis=1)


def get_multi_modal_text_image(
    mm_text: str,
    image_width,
    font_size,
    background_color=(0, 0, 0),
    text_color=(255, 255, 255),
    alignment="left",
):
    """
    Renders multi-modal content into an image of width image_width.
    :param mm_text: a single string, or a list of strings and images
    :param image_width: Target width image. Padding is used if necessary.
    :param font_size: Font size used for rendering.
    :param background_color: Background color of image.
    :param text_color: Image for rendering text.
    :param alignment: "justify", "left", "right" or "center".
    :return: image
    """
    if isinstance(mm_text, str):
        mm_text = [mm_text]

    rows = []

    pre_image_space = font_size // 3  # the width of the " " that must be inserted before an image

    while len(mm_text):  # keep rendering rows until the mm_text is all done
        row = []
        remaining_width = image_width

        while (
            len(mm_text) and remaining_width > 0
        ):  # keep rendering words/text until the row is full
            if isinstance(mm_text[0], str):  # text
                text: str = mm_text[0]
                text_image, text_width, remaining_text = get_text_image(
                    text,
                    remaining_width,
                    font_size,
                    background_color=background_color,
                    text_color=text_color,
                    single_line=True,
                )
                text_image = crop_text_image(text_image)  # crop text image to have just the text
                if len(row) == 0 or text_width < remaining_width:
                    remaining_width -= text_image.shape[1]
                    text_image[0, -1, :] = text_image[
                        -1, -1, :
                    ]  # get rid of the off-color pixel inserted by get_text_image
                    text_image = crop_text_image(
                        text_image
                    )  # crop text image to have just the text
                    row.append(text_image)
                    if len(remaining_text) > 0:
                        mm_text[0] = remaining_text
                    else:
                        mm_text = mm_text[1:]
                else:
                    break  # end of row; next word doesn't fit
            else:  # image
                image: np.ndarray = np.flip(mm_text[0], axis=2)
                if len(row) == 0 or (image.shape[1] + pre_image_space) < remaining_width:
                    if len(row) > 0:  # insert a small "space" image
                        pre_image = single_color_image(
                            (image.shape[0], pre_image_space, image.shape[2]), background_color
                        )
                        row.append(pre_image)
                        remaining_width -= pre_image.shape[1]
                    row.append(image)
                    remaining_width -= image.shape[1]
                    mm_text = mm_text[1:]
                else:
                    break  # end of row; image doesn't fit
        rows.append(_concat_row(row, image_width, background_color, alignment))

    return np.concatenate(rows, axis=0)


def render_test_image(output_path) -> None:
    """
    Writes the template (with all defined areas filled with green color) to output_path.
    :param output_path: Where to write the test image.
    :return: None
    """
    image = get_template_image()

    for _key, area in template.items():
        if isinstance(area, list) and len(area) == 2 and isinstance(area[0], list):
            ulx, uly = area[0]
            lrx, lry = area[1]
            image[uly:lry, ulx:lrx] = (image[uly:lry, ulx:lrx] // 2) + np.array(
                [0, 127, 0], dtype=np.uint8
            )

    cv2.imwrite(output_path, image)


def main():
    """
    A small script to test the template.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, required=True, help="Export path for the video template file"
    )

    args = parser.parse_args()
    render_test_image(args.output_path)


if __name__ == "__main__":
    main()
