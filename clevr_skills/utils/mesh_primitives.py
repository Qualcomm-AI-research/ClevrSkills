# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from typing import List, Optional, Tuple

import numpy as np
import sapien.core as sapien
import trimesh

from clevr_skills.utils.temp_dir import get_temp_dir


def generate_uv(
    uv: np.ndarray,
    vertices: np.ndarray,
    target_idx: List[int],
    distance_reference=None,
    u_axis=0,
    v_axis=1,
):
    """
    Generate uv coordinates for the vertices.
    :param uv: Target for uv coordinates.
    :param vertices: 3D vertex locations.
    :param target_idx: the indices of the vertices that should be set.
    :param distance_reference: reference vertices (for distance computation).
    :param u_axis: 0, 1, 2, or  (-1) for distance.
    :param v_axis:0, 1, 2, or "distance".
    :return: None; the value of "uv" is set.
    """

    def compute_u(vertex, axis, distance, distance_reference):
        if 0 <= axis < 3:
            return vertex[axis]
        return distance[np.argmin(np.linalg.norm(distance_reference - vertex, axis=1))]

    if distance_reference is None:
        distances = None
    else:
        distances = np.zeros(len(distance_reference))
        for i in range(1, len(distance_reference)):
            distances[i] = distances[i - 1] + np.linalg.norm(
                distance_reference[i] - distance_reference[i - 1]
            )

    for t in target_idx:
        uv[t, 0] = compute_u(vertices[t], u_axis, distances, distance_reference)
        uv[t, 1] = compute_u(vertices[t], v_axis, distances, distance_reference)
    uv[target_idx, 0] -= np.min(uv[target_idx, 0])
    uv[target_idx, 1] -= np.min(uv[target_idx, 1])


def create_cylinder_mesh(
    num_vertices=16,
    width=0.1,
    depth=0.1,
    height=0.1,
    rotation=0.0,
    odd_scale=1.0,
    visualization=False,
    smooth_side=True,
    texture_scale=1.0,
):
    """
    Creates a RenderMesh in the shape of a cylinder
    :param num_vertices: number of vertices on each rim. Can be used to create triangle,
    cube, hexagon, etc.
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param rotation: in radians. How much to rotate the upper and lower rings. Used to make
    a cube axis-aligned
    :param odd_scale: used to create stars. This will scale the vertex coordinates of the odd
    vertices along the ring in the width and depth dimensions.
    :param visualization: if the mesh is for visualization, the vertices along the rims are
    duplicated to ensure proper face normals
    :param smooth_side: for a real cylinder, the sides should be rendered in a smooth way
    (i.e., the vertices are shared between faces).
    For primitives like triangle, cube, hexagon, you'd want the sides to be rendered
    as individual faces.
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :return: a Trimesh instance.
    """

    # create vertices for top (center), top ring, bottom (center) and bottom ring
    ramp = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + rotation
    s = np.sin(ramp)
    c = np.cos(ramp)
    o = np.ones(num_vertices)
    top = np.array([[0, 0, 0.5]])
    top_ring = np.array([0.5 * s, 0.5 * c, 0.5 * o]).T
    bottom = np.array([[0, 0, -0.5]])
    bottom_ring = np.array([0.5 * s, 0.5 * c, -0.5 * o]).T

    if odd_scale != 1.0:
        mask = np.linspace(
            start=0, stop=num_vertices, num=num_vertices, endpoint=False, dtype=np.int
        )
        mask = mask % 2 == 1
        top_ring[mask, 0:2] *= odd_scale
        bottom_ring[mask, 0:2] *= odd_scale

    vertices = [top, top_ring, bottom, bottom_ring]

    # create faces for top
    top_center_idx = np.zeros(num_vertices, dtype=int)
    top_vtx1_idx = np.linspace(0, num_vertices, num_vertices, endpoint=False, dtype=int) + 1
    top_vtx2_idx = np.roll(top_vtx1_idx, 1)
    top_idx = np.column_stack((top_center_idx, top_vtx1_idx, top_vtx2_idx))

    # create faces for bottom
    bottom_center_idx = top_center_idx + 1 + num_vertices
    bottom_vtx1_idx = 1 + num_vertices + top_vtx1_idx
    bottom_vtx2_idx = np.roll(bottom_vtx1_idx, 1)
    bottom_idx = np.column_stack((bottom_center_idx, bottom_vtx2_idx, bottom_vtx1_idx))

    if visualization:  # add separate vertices for the side
        vertices.append(top_ring)
        vertices.append(bottom_ring)
        top_vtx1_idx += 2 * num_vertices + 1
        top_vtx2_idx += 2 * num_vertices + 1
        bottom_vtx1_idx += 2 * num_vertices
        bottom_vtx2_idx += 2 * num_vertices

    side_idx_1 = np.column_stack((top_vtx1_idx, bottom_vtx1_idx, top_vtx2_idx))
    side_idx_2 = np.column_stack((bottom_vtx2_idx, top_vtx2_idx, bottom_vtx1_idx))

    if visualization and not smooth_side:
        vertices.append(top_ring)
        vertices.append(bottom_ring)
        for i in range(0, len(side_idx_1), 2):
            side_idx_1[i] += 2 * num_vertices
            side_idx_2[i] += 2 * num_vertices

    # create vertex and face arrays
    vertices = np.concatenate(vertices)

    # Scale vertices according to depth, width, height (also top ring and bottom ring,
    # they are used below for UV coords)
    _scaling = np.array([depth, width, height])
    top_ring *= _scaling
    bottom_ring *= _scaling
    vertices *= _scaling

    faces = np.concatenate((top_idx, bottom_idx, side_idx_1, side_idx_2))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # check if normals were correct
    trimesh.repair.fix_normals(mesh)

    assert np.all(faces == mesh.faces)

    if visualization:
        uv = np.zeros((vertices.shape[0], 2))
        generate_uv(uv, vertices, np.unique(top_idx.flatten()), u_axis=0, v_axis=1)
        generate_uv(uv, vertices, np.unique(bottom_idx.flatten()), u_axis=0, v_axis=1)
        side_vtx_idx = np.unique(np.concatenate((side_idx_1, side_idx_2)).flatten())
        top_vtx_idx = side_vtx_idx[vertices[side_vtx_idx][:, 2] > 0]
        bottom_vtx_idx = side_vtx_idx[vertices[side_vtx_idx][:, 2] < 0]
        generate_uv(uv, vertices, top_vtx_idx, distance_reference=top_ring, u_axis=-1, v_axis=2)
        generate_uv(
            uv, vertices, bottom_vtx_idx, distance_reference=bottom_ring, u_axis=-1, v_axis=2
        )
        uv = texture_scale * uv
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)

    return mesh


def get_cylinder_mesh_by_file(
    num_vertices=16,
    width=0.1,
    depth=0.1,
    height=0.1,
    rotation=0.0,
    odd_scale=1.0,
    smooth_side=True,
    texture_scale=1.0,
    temp_path: str = None,
) -> Tuple[str, str]:
    """
    Creates a .glb file
    :param num_vertices: number of vertices on each rim. Can be used to create triangle,
    cube, hexagon, etc.
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param rotation: in radians. How much to rotate the upper and lower rings. Used to make
    a cube axis-aligned
    :param odd_scale: used to create stars. This will scale the vertex coordinates of the odd
    vertices along the ring in the width and depth dimensions.
    :param smooth_side: for a real cylinder, the sides should be rendered in a smooth way
    (i.e., the vertices are shared between faces).
    For primitives like triangle, cube, hexagon, you'd want the sides to be rendered
    as individual faces.
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :return: a tuple (collision_path, visual_path)
    """
    result = []
    for visualization in [False, True]:
        vis_name = "visual" if visualization else "collision"
        filename = (
            f"cylinder_vertices_{num_vertices}_width_{width}_depth_{depth}_"
            f"height_{height}_smooth_side_{smooth_side}_{vis_name}.obj"
        )
        output_path = os.path.join(temp_path, filename)
        mesh = create_cylinder_mesh(
            num_vertices=num_vertices,
            width=width,
            depth=depth,
            height=height,
            rotation=rotation,
            odd_scale=odd_scale,
            visualization=visualization,
            smooth_side=smooth_side,
            texture_scale=texture_scale,
        )
        trimesh.exchange.export.export_mesh(mesh, output_path)
        result.append(output_path)
    return result


def add_cylinder_actor(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    num_vertices=16,
    width=0.1,
    depth=0.1,
    height=0.1,
    rotation=0.0,
    density=1000,
    odd_scale=1.0,
    name="cylinder",
    static=False,
    render_material: sapien.RenderMaterial = None,
    smooth_side=True,
    texture_scale=1.0,
    collision_geometry: bool = True,
    orientation: Optional[np.ndarray] = None,
):
    """
    Adds cylinder actor to scene.
    :param scene:  Sapien scene.
    :param renderer: Sapien renderer,
    :param num_vertices: number of vertices on each rim. Can be used to create triangle,
    cube, hexagon, etc.
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param rotation: in radians. How much to rotate the upper and lower rings. Used to make
    a cube axis-aligned
    :param odd_scale: used to create stars. This will scale the vertex coordinates of the odd
    vertices along the ring in the width and depth dimensions.
    :param smooth_side: for a real cylinder, the sides should be rendered in a smooth way
    (i.e., the vertices are shared between faces).
    For primitives like triangle, cube, hexagon, you'd want the sides to be rendered
    as individual faces.
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :param density: Density of actor in kg/m3
    :param name: name of actor.
    :param static: Whether the actor is static (can't move) or dynamic.
    :param render_material:  Render material to apply.
    :param collision_geometry: Add collision geometry? If not, the actor will be a "ghost".
    :param orientation: quaternion; optional rotation of actor.
    :return: sapien.Actor
    """

    # The actor builder does not allow to add a collision mesh directly from an object;
    # it has to come from file.
    # So we store the meshes (visual and collision) in file and read them back
    temp_path = get_temp_dir(scene)
    collision_path, visual_path = get_cylinder_mesh_by_file(
        num_vertices=num_vertices,
        width=width,
        depth=depth,
        height=height,
        rotation=rotation,
        odd_scale=odd_scale,
        smooth_side=smooth_side,
        texture_scale=texture_scale,
        temp_path=temp_path,
    )

    pose = sapien.Pose() if orientation is None else sapien.Pose([0, 0, 0], orientation)

    builder = scene.create_actor_builder()
    if collision_geometry:
        builder.add_collision_from_file(collision_path, density=density, pose=pose)
    builder.add_visual_from_file(visual_path, material=render_material, pose=pose)
    if static:
        return builder.build_static(name)
    return builder.build(name)


def add_cube_actor(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    name="cube",
    static=False,
    render_material: sapien.RenderMaterial = None,
    texture_scale=1.0,
    collision_geometry: bool = True,
    orientation: Optional[np.ndarray] = None,
):
    """
    Adds cube actor to scene.
    :param scene:  Sapien scene.
    :param renderer: Sapien renderer,
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :param density: Density of actor in kg/m3
    :param name: name of actor.
    :param static: Whether the actor is static (can't move) or dynamic.
    :param render_material:  Render material to apply.
    :param collision_geometry: Add collision geometry? If not, the actor will be a "ghost".
    :param orientation: quaternion; optional rotation of actor.
    :return: sapien.Actor
    """
    width = width * np.sqrt(2)
    depth = depth * np.sqrt(2)
    return add_cylinder_actor(
        scene,
        renderer,
        num_vertices=4,
        width=width,
        depth=depth,
        height=height,
        rotation=np.pi / 4,
        density=density,
        name=name,
        static=static,
        render_material=render_material,
        smooth_side=False,
        texture_scale=texture_scale,
        collision_geometry=collision_geometry,
        orientation=orientation,
    )


def add_triangle_actor(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    name="triangle",
    static=False,
    render_material: sapien.RenderMaterial = None,
    texture_scale=1.0,
    orientation: Optional[np.ndarray] = None,
):
    """
    Adds triangle actor to scene.
    :param scene:  Sapien scene.
    :param renderer: Sapien renderer,
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :param density: Density of actor in kg/m3
    :param name: name of actor.
    :param static: Whether the actor is static (can't move) or dynamic.
    :param render_material:  Render material to apply.
    :param orientation: quaternion; optional rotation of actor.
    :return: sapien.Actor
    """
    width = width / np.sin(2 * np.pi / 3)
    depth = depth / 0.75
    return add_cylinder_actor(
        scene,
        renderer,
        num_vertices=3,
        width=width,
        depth=depth,
        height=height,
        density=density,
        name=name,
        static=static,
        render_material=render_material,
        smooth_side=False,
        texture_scale=texture_scale,
        orientation=orientation,
    )


def add_hexagon_actor(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    name="hexagon",
    static=False,
    render_material: sapien.RenderMaterial = None,
    texture_scale=1.0,
    orientation: Optional[np.ndarray] = None,
):
    """
    Adds hexagon actor to scene.
    :param scene:  Sapien scene.
    :param renderer: Sapien renderer,
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :param density: Density of actor in kg/m3
    :param name: name of actor.
    :param static: Whether the actor is static (can't move) or dynamic.
    :param render_material:  Render material to apply.
    :param orientation: quaternion; optional rotation of actor.
    :return: sapien.Actor
    """
    width = width / np.sin(2 * np.pi / 3)
    return add_cylinder_actor(
        scene,
        renderer,
        num_vertices=6,
        width=width,
        depth=depth,
        height=height,
        density=density,
        name=name,
        static=static,
        render_material=render_material,
        smooth_side=False,
        texture_scale=texture_scale,
        orientation=orientation,
    )


def add_star_actor(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    width=0.1,
    depth=0.1,
    height=0.1,
    density=1000,
    num_points=6,
    inner_scale=0.5,
    name="star",
    static=False,
    render_material: sapien.RenderMaterial = None,
    texture_scale=1.0,
    orientation: Optional[np.ndarray] = None,
):
    """
    Adds star actor to scene.
    :param scene:  Sapien scene.
    :param renderer: Sapien renderer,
    :param width: diameter of cylinder in Y direction
    :param depth: diameter of cylinder in X direction
    :param height: height (Z) of cylinder
    :param num_points: number of points of the star.
    :param inner_scale: How much smaller is the "inside" of the star.
    :param texture_scale: how much to scale the texture coordinates. With unit scaling,
    every 1 meter the texture will be repeated.
    :param density: Density of actor in kg/m3
    :param name: name of actor.
    :param static: Whether the actor is static (can't move) or dynamic.
    :param render_material:  Render material to apply.
    :param orientation: quaternion; optional rotation of actor.
    :return: sapien.Actor
    """

    return add_cylinder_actor(
        scene,
        renderer,
        num_vertices=num_points * 2,
        width=width,
        depth=depth,
        height=height,
        density=density,
        name=name,
        static=static,
        render_material=render_material,
        smooth_side=False,
        odd_scale=inner_scale,
        texture_scale=texture_scale,
        orientation=orientation,
    )
