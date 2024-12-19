# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import sapien.core as sapien
import trimesh
from mani_skill2.utils.sapien_utils import parse_urdf_config
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from omegaconf import OmegaConf
from transforms3d.euler import euler2quat

from clevr_skills.utils import mesh_primitives
from clevr_skills.utils.temp_dir import get_temp_dir

from .paths import MANISKILL2_ASSET_DIR, PACKAGE_ASSET_DIR
from .textures import _COLORS_AND_TEXTURES


def merge_lists(*args):
    res = []
    for x in args:
        res.extend(x)
    return res


OmegaConf.register_new_resolver("merge", merge_lists)
SIZE = namedtuple("Size", ["width", "depth", "height"])


def generate_uv_mesh(
    mesh: trimesh.Trimesh, dim_u: int = 0, dim_v: int = 1, scale=0.1
) -> trimesh.Trimesh:
    """
    Adds UV (texture) coordinates to a trimesh
    :param mesh: Trimesh that needs UV coordinates.
    :param dim_u: dimension (0, 1, 2) to use for U
    :param dim_v: dimension (0, 1, 2) to use for V
    :param scale: How much to scale UV coordinates.
    :return: the mesh
    """
    vertices = mesh.vertices
    uv = np.zeros((vertices.shape[0], 2), dtype=vertices.dtype)
    uv[:, 0] = (vertices[:, dim_u] - np.min(vertices[:, dim_u])) * scale
    uv[:, 1] = (vertices[:, dim_v] - np.min(vertices[:, dim_v])) * scale

    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)

    return mesh


def generate_uv_obj(path: Path, temp_path: str, dim_u: int = 0, dim_v: int = 1, scale=0.1) -> Path:
    """
    Loads mesh from path, generates UV coordinates, writes mesh back to temp file.
    :param path: Path of mesh to load.
    :param temp_path: temporary path for writing the mesh
    :param dim_u: dimension (0, 1, 2) to use for U
    :param dim_v: dimension (0, 1, 2) to use for V
    :param scale: How much to scale UV coordinates.
    :return: path to temporary file.
    """
    mesh = trimesh.load(path)
    mesh = generate_uv_mesh(mesh, dim_u, dim_v, scale)
    export_path = Path(temp_path) / path.name
    trimesh.exchange.export.export_mesh(mesh, export_path)
    return export_path


def get_actor_size(actor):
    """
    :param actor: An actor
    :return: XYZ size of actor.
    """
    ac_verts = get_actor_mesh(actor).vertices
    return ac_verts.max(0) - ac_verts.min(0)


YCB_DIR = PACKAGE_ASSET_DIR / "mani_skill2_ycb" / "models"


def load_ycb_object(
    model_id,
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    name=None,
    static=False,
    render_material: sapien.RenderMaterial = None,
    orientation: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    Loads a YCB object
    :param model_id: A string like "024_bowl"
    :param scene: Sapien scene
    :param renderer: Sapien renderer
    :param width: used to scale the object (X)
    :param depth: used to scale the object (Y)
    :param height: used to scale the object (Z)
    :param density: weight in kg/m3
    :param name: Name of actor.
    :param static: When True, the object can't move
    :param render_material: if provided, this material will override the default texture
    of the object
    :param kwargs: ignored
    :return: actor
    """
    builder = scene.create_actor_builder()
    model_dir = YCB_DIR / model_id

    physical_material = None
    scale = [width, depth, height]

    pose = sapien.Pose() if orientation is None else sapien.Pose([0, 0, 0], orientation)

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        pose=pose,
        scale=scale,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(
        filename=visual_file, pose=pose, scale=scale, material=render_material
    )

    if static:
        actor = builder.build_static()
    else:
        actor = builder.build()

    actor.set_name(name)

    return actor


ASSEMBLING_KIT_DIR = PACKAGE_ASSET_DIR / "assembling_kits" / "models"
"""
load_assembling_kit_object model IDs:
00 = R
01 = A
02 = triangle shape
03 = square shape
04 = plus shape
05 = T
06 = diamond shape
07 = pentagon shape
08 = rectangle shape
09 = 4 bubbles
10 = 5 pointed star 
11 = disc (circle)
12 = G
13 = V
14 = E
15 = L
16 = O
17 = hexagon shape
18 = heart shape
19 = M
"""


def load_assembling_kit_object(
    model_id,
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    name=None,
    static=False,
    render_material: sapien.RenderMaterial = None,
    orientation: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    Loads assembling kit models (which contain letters, amongst others
    :param model_id: An integer in range [0 ... 19]
    :param scene: Sapien scene
    :param renderer: Sapien renderer
    :param width: used to scale the object (X)
    :param depth: used to scale the object (Y)
    :param height: used to scale the object (Z)
    :param density: weight in kg/m3
    :param name: Name of actor.
    :param static: When True, the object can't move
    :param render_material: if provided, this material will override the default texture
    of the object
    :param orientation: optional orientation (quaternion) of mesh when object is created.
    :param kwargs: ignored
    :return: actor
    """

    # the assembling kit models are insanely large, so scale them to be about unit
    scale = np.array([width, depth, height]) / 20.0
    pose = sapien.Pose() if orientation is None else sapien.Pose([0, 0, 0], orientation)

    collision_path = ASSEMBLING_KIT_DIR / "collision" / f"{model_id:02d}.obj"
    visual_path = ASSEMBLING_KIT_DIR / "visual" / f"{model_id:02d}.obj"

    builder = scene.create_actor_builder()
    physical_material = None

    builder.add_multiple_collisions_from_file(
        str(collision_path), scale=scale, pose=pose, material=physical_material, density=density
    )
    temp_path = get_temp_dir(scene)
    # The assembly kit objects don't have texture coordinates, add them:
    visual_path = generate_uv_obj(visual_path, temp_path=temp_path, dim_u=0, dim_v=1, scale=0.05)

    # Add a dummy camera in order to decide shader
    builder.add_visual_from_file(str(visual_path), pose=pose, scale=scale, material=render_material)

    if static:
        actor = builder.build_static(f"obj_{model_id:02d}")
    else:
        actor = builder.build(f"obj_{model_id:02d}")

    actor.set_name(name)

    return actor


PARTNET_DIR = MANISKILL2_ASSET_DIR / "partnet_mobility" / "dataset"


def load_partnet_mobility(
    model_id,
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    name=None,
    static=False,
    urdf_config: dict = None,
    orientation: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    Loads a PartNet articulated model.
    Current use-case is loading buckets.

    The function arguments are identical to

    :param model_id: An integer or string
    :param scene: sapien scene.
    :param renderer: not used
    :param width: The mean of width, depth, height is used to scale the articulation
    :param depth: The mean of width, depth, height is used to scale the articulation
    :param height: The mean of width, depth, height is used to scale the articulation
    :param name: name of actor.
    :param static: When True, the root link is can't move
    :param urdf_config: Optional dict with options
    :param orientation: ignored
    :return: Articulation.
    """
    loader = scene.create_urdf_loader()
    loader.fix_root_link = static
    loader.scale = np.mean([width, depth, height])
    loader.load_multiple_collisions_from_file = True

    urdf_config = parse_urdf_config(urdf_config or {}, scene)

    for filename in ["mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf"]:
        urdf_path = PARTNET_DIR / str(model_id) / filename
        if urdf_path.is_file():
            break
    articulation = loader.load(str(urdf_path), config=urdf_config)

    articulation.set_name(name)
    return articulation


class ModelAndTextureFactory:

    def __init__(self, task: str, episode_rng, split="train"):
        """
        :param task: Classname of the task.
        Used to set up factory for specific task (each tasks can have its own settings).
        :param episode_rng: Random number generator.
        :param split: "train" or "test".
        """

        self.task = task
        self._episode_rng = episode_rng
        self.split = split

        ycb_info_path = PACKAGE_ASSET_DIR / "mani_skill2_ycb" / "info_raw.json"
        with open(ycb_info_path, "r", encoding="utf8") as fp:
            ycb_info = json.load(fp)

        self.ycb_objects = set(list(ycb_info.keys()))
        self.assembling_objects = {
            "0": "R",
            "1": "A",
            "2": "triangle shape",
            "3": "square shape",
            "4": "plus shape",
            "5": "T",
            "6": "diamond shape",
            "7": "pentagon shape",
            "8": "rectangle shape",
            "9": "4 bubbles",
            "10": "5 pointed star",
            "11": "disc (circle)",
            "12": "G",
            "13": "V",
            "14": "E",
            "15": "L",
            "16": "O",
            "17": "hexagon shape",
            "18": "heart shape",
            "19": "M",
        }

        self.primitives = {
            "cube": mesh_primitives.add_cube_actor,
            "cylinder": mesh_primitives.add_cylinder_actor,
            "triangle": mesh_primitives.add_triangle_actor,
            "hexagon": mesh_primitives.add_hexagon_actor,
            "star": mesh_primitives.add_star_actor,
        }

        self.load_fns = {
            "ycb": lambda name, scene, renderer, render_material, size, density, static, orientation: load_ycb_object(
                name,
                scene,
                renderer,
                width=size.width,
                depth=size.depth,
                height=size.height,
                name=" ".join(name.split("_")[1:]),
                render_material=render_material,
                density=density,
                static=static,
                texture_scale=4.0,
                orientation=orientation,
            ),
            "assembling_kits": lambda name, scene, renderer, render_material, size, density, static, orientation: load_assembling_kit_object(
                int(name),
                scene,
                renderer,
                width=size.width,
                depth=size.depth,
                height=size.height,
                name=self.assembling_objects[name],
                render_material=render_material,
                density=density,
                static=static,
                texture_scale=4.0,
                orientation=orientation,
            ),
            "primitives": lambda name, scene, renderer, render_material, size, density, static, orientation: self.primitives[
                name
            ](
                scene,
                renderer,
                width=size.width,
                depth=size.depth,
                height=size.height,
                name=name,
                render_material=render_material,
                density=density,
                static=static,
                texture_scale=4.0,
                orientation=orientation,
            ),
        }

        asset_split_path = PACKAGE_ASSET_DIR / "asset_splits" / f"{self.split}.yaml"
        pool = OmegaConf.to_container(OmegaConf.load(asset_split_path), resolve=True)
        self.task_pool = pool[self.task]

        self.obj_scale = {"070-b_colored_wood_blocks": 2.0, "062_dice": 2.0, "036_wood_block": 0.5}
        self.obj_density_mul = {"036_wood_block": 0.25}
        self.obj_orientation = {"006_mustard_bottle": euler2quat(np.pi / 2, 0, 0)}

    def _get_size(self, obj_name):
        """
        Used to customize the size of some objects.
        :param obj_name: canonical name of object. I.e., the YCB name, the assembling object name, etc.
        :return: XYZ scaling factor (list of 3 floats).
        """
        size_multiplier = self.obj_scale[obj_name] if obj_name in self.obj_scale else 1.0
        if str(obj_name) in self.ycb_objects:
            size = [1.0, 1.0, 1.0]
        elif str(obj_name) in self.assembling_objects:
            size = [0.1, 0.1, 0.025]
        elif str(obj_name) in self.primitives:
            size = [0.1, 0.1, 0.05]
        else:
            raise RuntimeError(f"{obj_name} not recognized.")

        size = [x * size_multiplier for x in size]
        size = SIZE(*size)

        return size

    def _get_density(self, obj_name, density_default):
        """
        Some objects are very large and thus very heavy.
        This is solved by reducing the density of the material.
        :param obj_name: Canonical name of object.
        :param density_default: The user-supplied density, typically 1000.0
        :return: Density of object, scaled by hard-coded value.
        """
        density_multiplier = (
            self.obj_density_mul[obj_name] if obj_name in self.obj_density_mul else 1.0
        )
        return density_default * density_multiplier

    def get_object(
        self, name, tex_name, scene, renderer, render_material, size, density, static
    ) -> sapien.Actor:
        """
        Produces an object instance (Actor)
        :param name: Canonical name of object.
        :param tex_name: Name of texture to be applied.
        :param scene: Sapien scene.
        :param renderer: Sapien renderer.
        :param render_material: Sapien render material.
        :param size: Size scaling of object.
        :param density: Density of object (kg/m3).
        :param static: If True, the object can't move.
        :return: Sapien Actor.
        """
        if str(name) in self.ycb_objects:
            obj_set = "ycb"
        elif str(name) in self.assembling_objects:
            obj_set = "assembling_kits"
        elif str(name) in self.primitives:
            obj_set = "primitives"
        else:
            raise RuntimeError(f"{name} object does not exist")

        if size is None:
            size = self._get_size(name)
        elif isinstance(size, float):
            size_mul = size
            obj_size = self._get_size(name)
            size = SIZE(
                width=obj_size.width * size_mul,
                depth=obj_size.depth * size_mul,
                height=obj_size.height * size_mul,
            )
        elif hasattr(size, "__len__") and len(size) == 3:
            size = SIZE(width=size[0], depth=size[1], height=size[2])

        density = self._get_density(name, density)

        orientation = (
            self.obj_orientation[str(name)]
            if str(name) in self.obj_orientation
            else euler2quat(0.0, 0.0, 0.0)
        )

        actor = self.load_fns[obj_set](
            name,
            scene,
            renderer,
            render_material,
            size=size,
            density=density,
            static=static,
            orientation=orientation,
        )
        if tex_name is not None:  # improve the name if tex_name is provided
            actor.set_name(f"{tex_name} {actor.name}")
        return actor

    def get_random_object(
        self,
        scene,
        renderer,
        render_material,
        tex_name=None,
        size: SIZE = None,
        density: float = 1000.0,
        static: bool = False,
    ) -> sapien.Actor:
        """
        Returns a random object (from the pool of available objects for the task).
        :param scene: Sapien scene.
        :param renderer: Sapien renderer.
        :param render_material: Sapien material.
        :param tex_name: Name of texture to be applied.
        :param size: Size of object.
        :param density: Density of object (kg/m3).
        :param static: If True, the object can't move.
        :return: Sapien Actor.
        """
        name = self._episode_rng.choice(self.task_pool["random_objects"])
        return self.get_object(
            name, tex_name, scene, renderer, render_material, size, density, static
        )

    def get_random_base_object(
        self,
        scene,
        renderer,
        render_material,
        tex_name=None,
        size: SIZE = None,
        density: float = 1000.0,
        static: bool = False,
    ) -> sapien.Actor:
        """
        Returns a random base object (from the pool of available objects for the task).
        A base object is an object that you can place other objects on / in.
        :param scene: Sapien scene.
        :param renderer: Sapien renderer.
        :param render_material: Sapien material.
        :param tex_name: Name of texture to be applied.
        :param size: Size of object.
        :param density: Density of object (kg/m3).
        :param static: If True, the object can't move.
        :return: Sapien Actor.
        """
        name = self._episode_rng.choice(self.task_pool["base_objects"])
        return self.get_object(
            name, tex_name, scene, renderer, render_material, size, density, static
        )

    def get_random_top_object(
        self,
        scene,
        renderer,
        render_material,
        tex_name=None,
        size: SIZE = None,
        density: float = 1000.0,
        static: bool = False,
    ) -> sapien.Actor:
        """
        Returns a random top object (from the pool of available objects for the task).
        A top object is an object that can be placed easily on top of another object.
        :param scene: Sapien scene.
        :param renderer: Sapien renderer.
        :param render_material: Sapien material.
        :param tex_name: Name of texture to be applied.
        :param size: Size of object.
        :param density: Density of object (kg/m3).
        :param static: If True, the object can't move.
        :return: Sapien Actor.
        """
        name = self._episode_rng.choice(self.task_pool["top_objects"])
        return self.get_object(
            name, tex_name, scene, renderer, render_material, size, density, static
        )

    def get_random_area_object(
        self,
        scene,
        renderer,
        render_material,
        tex_name=None,
        size: SIZE = None,
        density: float = 1000.0,
        static: bool = True,
    ) -> sapien.Actor:
        """
        Returns a random area object (from the pool of available objects for the task).
        A area object is an object that you can place other objects on.
        :param scene: Sapien scene.
        :param renderer: Sapien renderer.
        :param render_material: Sapien material.
        :param tex_name: Name of texture to be applied.
        :param size: Size of object.
        :param density: Density of object (kg/m3).
        :param static: If True, the object can't move.
        :return: Sapien Actor.
        """
        name = self._episode_rng.choice(self.task_pool["primitive_objects"])
        return self.get_object(
            name, tex_name, scene, renderer, render_material, size, density, static
        )

    def get_random_textures(self, num):
        """
        :param num: Number of textures requested.
        :return: List of random textures.
        """
        tex_names = self._episode_rng.choice(
            self.task_pool["top_textures"] + self.task_pool["base_textures"],
            replace=False,
            size=num,
        )
        return [(tname, _COLORS_AND_TEXTURES[tname]) for tname in tex_names]

    def get_random_base_textures(self, num):
        """
        :param num: Number of textures requested.
        :return: List of random base textures.
        """
        tex_names = self._episode_rng.choice(
            self.task_pool["base_textures"], replace=False, size=num
        )
        return [(tname, _COLORS_AND_TEXTURES[tname]) for tname in tex_names]

    def get_random_top_textures(self, num):
        """
        :param num: Number of textures requested.
        :return: List of top textures.
        """
        tex_names = self._episode_rng.choice(
            self.task_pool["top_textures"], replace=False, size=num
        )
        return [(tname, _COLORS_AND_TEXTURES[tname]) for tname in tex_names]
