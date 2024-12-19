# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import sapien.core as sapien
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat


def get_camera_by_name(scene: sapien.Scene, name: str) -> Union[sapien.CameraEntity, None]:
    """
    :param scene: The sapien scene.
    :param name: The name of the camera.
    :return: camera, if there is a camera called 'name'; None otherwise
    """
    for c in scene.get_cameras():
        if c.name == name:
            return c
    return None


def get_render_material(renderer: sapien.SapienRenderer, color) -> sapien.RenderMaterial:
    """
    Creates a sapien RenderMaterial with base color.
    :param renderer: Sapien renderer.
    :param color: The color
    :return: RenderMaterial.
    """
    render_material = renderer.create_material()
    if isinstance(color, (list, tuple)):
        render_material.set_base_color(np.hstack([color, 1.0]))
    else:
        if isinstance(color, Path):
            color = str(color)
        render_material.set_base_color([0.0, 0.0, 0.0, 1.0])
        render_material.set_metallic(0.5)
        render_material.set_diffuse_texture_from_file(color)
    return render_material


def get_actor_bbox_in_img_space(env, actor, camera, use_3d_bounding_box=False):
    """
    :param env: ClevrSkillsEnv
    :param actor: The actor
    :param camera: The camera
    :param use_3d_bounding_box: if False, the vertices of the actor are used (which takes more, but is more precise).
    If True, the 3D bounding box is used.
    :return: 2D bounding box in camera image space.
    """
    cc = camera.camera
    in_mat = cc.get_intrinsic_matrix()
    ex_mat = cc.get_extrinsic_matrix()

    if use_3d_bounding_box:
        ab = env._actor_distance.get_bounds(
            actor
        )  # this returns the min and max XYZ coordinates (2x3 matrix)
        vertices = np.array(
            [
                [ab[1 if idx & 1 else 0, 0], ab[1 if idx & 2 else 0, 1], ab[1 if idx & 4 else 0, 2]]
                for idx in range(8)
            ]
        )  # form the full 3D bounding box (8x3 matrix)
    else:
        vertices = get_actor_mesh(actor, to_world_frame=True).vertices
    vertices = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1))], axis=-1
    )  # make homogeneous (Nx4 matrix)
    vertices = vertices.dot(ex_mat[:3].T)
    vertices = vertices[vertices[:, 2] > 0.0]  # Remove vertices on the wrong side of the camera
    if len(vertices) > 0:
        bb_img = vertices.dot(in_mat.T)
        bb_img = bb_img / bb_img[:, 2:3]

        # Clip to image bounds:
        bb_img[:, 0] = np.clip(0, bb_img[:, 0], cc.width - 1)
        bb_img[:, 1] = np.clip(0, bb_img[:, 1], cc.height - 1)

        bb_img = np.stack(
            [np.floor(np.min(bb_img, axis=0)), np.ceil(np.max(bb_img, axis=0))]
        ).astype(
            int
        )  # Get the min and max image coordinate (2x3 matrix, Z=1)
    else:
        bb_img = np.array([[-1, -1, 1], [-1, -1, 1]])
    return bb_img


def _get_actors(
    scene: sapien.Scene, actors: List[Union[sapien.ActorBase, str]]
) -> List[sapien.ActorBase]:
    """
    :param scene: The sapien scene.
    :param actors: a list of either strings and/or Actors.
    :return: the same list, but all strings have been resolved to the Actor with the same name.
    """
    if actors is None:
        return []
    all_actors = {a.name: a for a in scene.get_all_actors()}
    result = []
    for a in actors:
        if isinstance(a, str):
            if a in all_actors:
                result.append(all_actors[a])
        else:
            result.append(a)
    return result


def _compute_rotation_and_fov(point_cloud: np.ndarray, dim0: int, dim1: int, multiplier: float):
    """
    Compute the rotation and field-of-view (fov) required to frame the point cloud in a
    certain direction.
    :param point_cloud: N x 3 numpy array
    :param dim0: The first dimension to frame.
    :param dim1: The second dimension to frame.
    :param multiplier: sign of rotation (+1 or -1).
    :return: (rotation axis-angle(radians), fov)
    """
    pcl_angles = np.arctan2(point_cloud[:, dim0], point_cloud[:, dim1])
    histogram, bin_edges = np.histogram(
        pcl_angles, 8, range=(-np.pi, np.pi)
    )  # compute coarse histogram to get estimate of "center"
    max_bin = np.argmax(histogram)
    coarse_middle_angle = np.mean(bin_edges[max_bin : max_bin + 2])
    pcl_angles = pcl_angles - coarse_middle_angle
    pcl_angles += (pcl_angles < -np.pi) * (2 * np.pi)
    min_angle, max_angle = np.min(pcl_angles), np.max(pcl_angles)
    fov = max_angle - min_angle
    rot = 0.5 * (min_angle + max_angle) + coarse_middle_angle
    return rot * multiplier, fov


def take_picture(
    scene: sapien.Scene,
    renderer,
    cam_position: np.array,
    framed_actors: List[Union[sapien.ActorBase, str]],
    hidden_actors: List[Union[sapien.ActorBase, str]] = None,
    hide_all_non_framed: bool = False,
    actor_poses: Dict[str, Pose] = None,
    fovy: float = None,
    resolution: Tuple[int, int] = (640, 480),
):
    """
    :param scene: The Sapien scene.
    :param renderer: Not used.
    :param cam_position: X Y Z location of camera. The camera will point itself at the framed_actors
    automatically.
    :param framed_actors: List of at least on Actor (or name[str] of Actor) that must be framed
    in the picture.
    :param hidden_actors: Optional list of Actors (or names[str] of Actor) that will be hidden
    in the picture.
    :param hide_all_non_framed: If True, all actors that are not in framed_actors will be hidden.
     If True, this will override hidden_actors.
    :param actor_poses: Optional dictionary str->Pose that says where to place actors.
    :param fovy: If None, the field-of-view (in y direction of image) is automatically computed.
    Otherwise,the supplied number is used.
    :param resolution: Resolution of picture
    :return: Numpy array/picture of shape [y, x, 3], dtype=np.uint8.
    The picture is made using a temporary camera.
    """
    if not actor_poses:
        actor_poses = {}
    cam_position = np.array(cam_position)  # make sure it is a numpy array

    # Resolve actors by name, if needed
    framed_actors: List[sapien.ActorBase] = _get_actors(scene, framed_actors)
    hidden_actors: List[sapien.ActorBase] = _get_actors(scene, hidden_actors)
    all_actors: Dict[str, sapien.ActorBase] = {
        f"{a.name}_{a.id}": a for a in scene.get_all_actors()
    }
    for art in scene.get_all_articulations():
        all_actors.update({(art.name + "_" + link.name): link for link in art.get_links()})

    # Position actors according to (optional) actor_poses
    original_actor_poses: Dict[str, Pose] = {}  # preserve original poses here
    for name, pose in actor_poses.items():
        if name in all_actors:
            actor = all_actors[name]
            original_actor_poses[name] = actor.pose
            actor.set_pose(pose)

    # hide actors
    if hide_all_non_framed:
        # if hide_all_non_framed is True, all actors that are not framed_actors will be hidden
        hidden_actors = [a for a in all_actors.values() if not a in framed_actors]

    for actor in hidden_actors:
        actor.hide_visual()

    # get point cloud of all actors (for computing the framing) relative to the camera position
    meshes = [get_actor_mesh(actor) for actor in framed_actors]
    point_cloud = (
        np.concatenate([mesh.vertices for mesh in meshes if not mesh is None]) - cam_position
    )
    assert point_cloud.shape[0], "The framed actors have no mesh"

    # Compute yaw (z-axis rotation)
    camera_z_rot, required_fovx = _compute_rotation_and_fov(
        point_cloud, dim0=1, dim1=0, multiplier=1.0
    )

    # Rotate point cloud to match camera z rotation and compute pitch (y-axis rotation)
    point_cloud = np.matmul(point_cloud, quat2mat(euler2quat(0, 0, camera_z_rot)))
    camera_y_rot, required_fovy = _compute_rotation_and_fov(
        point_cloud, dim0=2, dim1=0, multiplier=-1.0
    )

    # compute focal length, if not specified by caller
    if fovy is None:
        # Convert the required focal length in the X direction to the required focal
        # length in the Y direction
        focal_length_x = (0.5 * resolution[0]) / np.tan(0.5 * required_fovx)
        required_fovx2y = 2.0 * np.arctan((0.5 * resolution[1]) / focal_length_x)
        fovy = max(required_fovy, required_fovx2y)
        fovy = fovy * 1.2  # add a little empty space around the framed objects

    # Create camera
    near = 0.001
    far = 10.0
    camera = scene.add_camera("temp snapshot camera", resolution[0], resolution[1], fovy, near, far)
    camera.set_pose(
        Pose(cam_position, qmult(euler2quat(0, 0, camera_z_rot), euler2quat(0, camera_y_rot, 0)))
    )

    # Take picture
    scene.update_render()
    camera.take_picture()
    rgba_img = (camera.get_float_texture("Color") * 255).clip(0, 255).astype("uint8")

    # Remove camera
    scene.remove_camera(camera)
    camera = None  # ensure that this object is not used after removal from scene
    # (will cause segfault otherwise)

    # un-hide actors
    for actor in hidden_actors:
        actor.unhide_visual()

    # Re-position actors
    for name, pose in original_actor_poses.items():
        all_actors[name].set_pose(pose)

    return rgba_img


def take_picture_with_camera(
    scene: sapien.Scene,
    camera,
    framed_actors: List[Union[sapien.ActorBase, str]],
    hidden_actors: List[Union[sapien.ActorBase, str]] = None,
    hide_all_non_framed: bool = False,
    hide_articulations: bool = True,
    actor_poses: Dict[str, Pose] = None,
):
    """
     :param scene: The Sapien scene.
     :param camera: The camera to be used.
     :param framed_actors: List of at least on Actor (or name[str] of Actor) that must be framed
     in the picture.
     :param hidden_actors: Optional list of Actors (or names[str] of Actor) that will be hidden
     in the picture.
     :param hide_all_non_framed: If True, all actors that are not in framed_actors will be hidden.
      If True, this will override hidden_actors.
     :param hide_articulations: If True, all articulations are hidden.
    :param actor_poses: Optional dictionary str->Pose that says where to place actors.
     :return: Numpy array/picture of shape [y, x, 3], dtype=np.uint8
    """

    if not actor_poses:
        actor_poses = {}

    # Resolve actors by name, if needed
    framed_actors: List[sapien.ActorBase] = _get_actors(scene, framed_actors)
    hidden_actors: List[sapien.ActorBase] = _get_actors(scene, hidden_actors)
    all_actors: Dict[str, sapien.ActorBase] = {
        f"{a.name}_{a.id}": a for a in scene.get_all_actors()
    }

    # articulations
    for art in scene.get_all_articulations():
        articulations = {(art.name + "_" + link.name): link for link in art.get_links()}

    # Position actors according to (optional) actor_poses
    original_actor_poses: Dict[str, Pose] = {}  # preserve original poses here
    for name, pose in actor_poses.items():
        if name in all_actors:
            actor = all_actors[name]
            original_actor_poses[name] = actor.pose
            actor.set_pose(pose)

    # hide actors
    if hide_all_non_framed:
        # if hide_all_non_framed is True, all actors that are not framed_actors will be hidden
        hidden_actors = [a for a in all_actors.values() if not a in framed_actors]

    # hide articulations
    if hide_articulations or hide_all_non_framed:
        for a in articulations.values():
            if not a in framed_actors:
                hidden_actors.append(a)

    for actor in hidden_actors:
        actor.hide_visual()

    # Take picture
    scene.update_render()
    camera.take_picture()
    rgba_img = (camera.get_float_texture("Color") * 255).clip(0, 255).astype("uint8")

    # Add a alpha channel
    pos_img = camera.get_float_texture("Position")
    mask = pos_img[:, :, 2] != 0.0
    rgba_img[:, :, 3] = mask.astype(np.uint8) * 255

    # un-hide actors
    for actor in hidden_actors:
        actor.unhide_visual()

    # Re-position actors
    for name, pose in original_actor_poses.items():
        all_actors[name].set_pose(pose)

    return rgba_img


def scale_and_crop_image(
    image: np.ndarray,
    size: Tuple[int, int],
    crop: bool = True,
    scale: bool = True,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    return_transform: bool = False,
):
    """
    Scales image to size.
    If the aspect ratio does not match, either crops or pads.
    :param image: input image.
    :param size: [width, height].
    :param crop: Whether to crop the input image to get the desired aspect ratio;
    the alternative is to pad the image.
    :param scale: Whether to crop the input image to fit.
    :param pad_color: the color used for padding.
    :param return_transform: whether to return the transform that is applied to
    each 2D point in the image.
    The transform is a 3x3 homogeneous transformation matrix.
    Useful if the caller wants to draw onto the image after scaling.
    :return: image, or (image, transform).
    """
    image_width = image.shape[1]
    image_height = image.shape[0]
    if image_height == size[1] and image_width == size[0]:
        transform = np.eye(3)
    elif crop:
        sw = int(round(size[0] * image_height / size[1]))
        sh = int(round(size[1] * image_width / size[0]))
        useable_shape = (image_width, sh) if sh < image_height else (sw, image_height)

        ulx = (image_width - useable_shape[0]) // 2
        lrx = ulx + useable_shape[0]
        uly = (image_height - useable_shape[1]) // 2
        lry = uly + useable_shape[1]
        transform_crop = np.array([[1, 0, -ulx], [0, 1, -uly], [0, 0, 1]], dtype=np.float)
        image = image[uly:lry, ulx:lrx, :]
        transform_scale = np.array(
            [[size[0] / image.shape[1], 0, 0], [0, size[1] / image.shape[0], 0], [0, 0, 1]],
            dtype=np.float,
        )
        transform = np.matmul(transform_scale, transform_crop)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    else:
        if scale:
            if image_width / image_height > size[0] / size[1]:  # scale to fit width
                size = [size[0], int(round(image_height * size[0] / image_width))]
            else:  # scale to fit height
                size = [int(round(image_width * size[1] / image_height)), size[1]]
            transform_scale = np.array(
                [[size[0] / image.shape[1], 0, 0], [0, size[1] / image.shape[0], 0], [0, 0, 1]],
                dtype=np.float,
            )
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        else:
            transform_scale = np.eye(3)
        padded_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        padded_image[:, :, 0:3] = np.array(pad_color, dtype=np.uint8)
        ulx = (size[0] - image.shape[1]) // 2
        lrx = ulx + image.shape[1]
        uly = (size[1] - image.shape[0]) // 2
        lry = uly + image.shape[0]
        padded_image[uly:lry, ulx:lrx, :] = image
        transform_paste = np.array([[1, 0, ulx], [0, 1, uly], [0, 0, 1]], dtype=np.float)
        transform = np.matmul(transform_paste, transform_scale)
        image = padded_image
    return (image, transform) if return_transform else image
