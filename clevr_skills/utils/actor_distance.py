# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, List

import numpy as np
import sapien.core as sapien
import trimesh
import trimesh.convex
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from pysdf import SDF  # pylint: disable=no-name-in-module


class ActorDistance:
    """
    This utility class computes distance between actors.
    It caches Signed Distance Functions and vertices of the meshes.
    Still, it is relatively inefficient.
    It would be far better if Sapien exposed a distance calculation API.
    """

    def __init__(
        self,
        rng,
        ground_name=None,
        ground_altitude=0.0,
        bounding_sphere_approximation_threshold: float = 0.1,
    ):
        """
        :param bounding_sphere_approximation_threshold: If the distance between the bounding
        spheres of the objects is larger than this, that distance is reported as
        the object distance.
        """
        self._rng = rng
        self._sdf_cache: Dict[sapien.Actor, SDF] = (
            {}
        )  # Signed Distance Function for each actor (local coord frame)
        self._vtx_cache: Dict[sapien.Actor, np.ndarray] = (
            {}
        )  # Vertices for each actor (local coord frame)
        self._rds_cache: Dict[sapien.Actor, float] = (
            {}
        )  # max distance (bounding sphere radius) of vertices from center of actor
        # local coordinate system
        self._cvx_sdf_cache: Dict[(sapien.Actor, int), SDF] = (
            {}
        )  #  Convex hull Signed Distance Function. The hull get be in 3D, or flattened (2D).
        # The integer in the key is the "flat dimension". It can be -1, 0, 2, 3
        self._bounding_sphere_approximation_threshold = bounding_sphere_approximation_threshold

        self._ground_name = ground_name
        self._ground_altitude = ground_altitude

    def __del__(self):
        self.clear()

    def clear(self) -> None:
        """
        Must be called when actors are deleted (e.g., scene is deleted) to avoid crashes.
        """
        self._sdf_cache.clear()
        self._vtx_cache.clear()

    def distance(
        self,
        actors: List[sapien.Actor],
        other_actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        flat_dim: int = -1,
    ) -> float:
        """
        :param actors: List of actors.
        :param other_actors: List of other actors (the distance to which should be measured).
        :param actor_pose: If not None, this will be used as the pose of actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0].
        :param flat_dim: If in range [0, 1, 2] the scene will be "flattened" along that dimension.
        :return: Minimum distance between actors and other actors.
        """
        arg_actor_pose = actor_pose

        actors = self._prepare_actors(self._flatten_actors(actors))
        other_actors = self._prepare_actors(self._flatten_actors(other_actors))
        other_actors = [oa for oa in other_actors if oa not in actors]

        flat_dim_range = self.get_bounds(other_actors)[:, flat_dim] if 0 <= flat_dim <= 2 else None

        min_distance = np.finfo(np.float32).max
        for actor in actors:
            actor_pose = (
                actor.pose
                if arg_actor_pose is None
                else arg_actor_pose.transform(actors[0].pose.inv().transform(actor.pose))
            )

            actor_pose_inv = actor_pose.inv()
            sdf = self._sdf_cache[actor]
            for other_actor in other_actors:
                m = actor_pose_inv.transform(other_actor.pose).to_transformation_matrix()

                translation = m[0:3, 3]
                if 0 <= flat_dim <= 2:
                    translation = np.copy(translation)
                    translation[flat_dim] = 0
                actor_bounding_sphere_distance = (
                    np.linalg.norm(translation)
                    - self._rds_cache[actor]
                    - self._rds_cache[other_actor]
                )
                if actor_bounding_sphere_distance > self._bounding_sphere_approximation_threshold:
                    min_distance = min(min_distance, actor_bounding_sphere_distance)
                    continue  # use actor_bounding_sphere_distance as approximation and skip
                    # expensive calculation below

                vertices = np.matmul(self._vtx_cache[other_actor], m.T)[:, 0:3]
                if 0 <= flat_dim <= 2:
                    # flatten vertices along the flat_dim, sample distance at N_SAMPLES positions
                    # along the flat_dim
                    N_SAMPLES = 16
                    for d in np.linspace(
                        flat_dim_range[0], flat_dim_range[1], N_SAMPLES, endpoint=True
                    ):
                        vertices[:, flat_dim] = d
                        distances = -sdf(vertices)
                        min_distance = min(min_distance, np.min(distances))
                else:
                    distances = -sdf(vertices)
                    min_distance = min(min_distance, np.min(distances))
        return min_distance

    def intersects(
        self,
        actors: List[sapien.Actor],
        other_actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        flat_dim: int = -1,
        min_distance: float = 0.0,
    ) -> float:
        """
        :param actors: List of actors.
        :param other_actors: List of other actors (for intersection check).
        :param actor_pose: If not None, this will be used as the pose of actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0].
        :param flat_dim: If in range [0, 1, 2] the scene will be "flattened" along that dimension.
        :param min_distance: Can be used to "grow" the actors, so they intersect when geometry
        is still at a small distance.
        :return: True if actors and other actors intersect.
        """
        d = self.distance(actors, other_actors, actor_pose=actor_pose, flat_dim=flat_dim)
        return d < min_distance

    def gradient(
        self,
        actors: List[sapien.Actor],
        other_actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        flat_dim: int = -1,
        eps: float = 0.0001,
        known_distance=None,
    ) -> np.ndarray:
        """
        :param actors: List of actors.
        :param other_actors: List of other actors (the distance to which should be measured).
        :param actor_pose: If not None, this will be used as the pose of actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0]
        :param flat_dim: If in range [0, 1, 2] the scene will be "flattened" along that dimension.
        :param eps: epsilon used for finite difference
        :param known_distance: if you already know the distance between actors and other_actors,
        supply it so avoid re-computation
        :return: Gradient of distance of actor w.r.t. other actors.
        """
        if actor_pose is None:
            singleton = isinstance(actors, (sapien.ActorBase, sapien.Articulation))
            actor_pose = actors.pose if singleton else actors[0].pose
        if known_distance is None:
            known_distance = self.distance(
                self, actors, other_actors, actor_pose=actor_pose, flat_dim=flat_dim
            )
        gradient = [0, 0, 0]
        for axis in [0, 1, 2]:
            pos = np.copy(actor_pose.p)
            pos[axis] += eps
            d = self.distance(
                actors, other_actors, actor_pose=sapien.Pose(pos, actor_pose.q), flat_dim=flat_dim
            )
            gradient[axis] = (d - known_distance) / eps
        return np.array(gradient, dtype=np.float32)

    def get_point_cloud(
        self,
        actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        reference_pose: sapien.Pose = None,
        homogeneous: bool = False,
    ) -> np.ndarray:
        """
        :param actors: List of actors for which point cloud is requested.
        :param actor_pose: Can be used to override the pose of the actors[0]
        To get the bounds in the local coordinate frame, use actor_pose=sapien.Pose()
        :param reference_pose: The pose relative to which the point cloud is computed
        :param homogeneous: Should the returned numpy array be homogeneous? (4th columns filled with 1)
        :return: Point cloud of actors.
        """
        arg_actor_pose = actor_pose
        actors = self._prepare_actors(self._flatten_actors(actors))
        vertices = []
        reference_pose_inv = reference_pose.inv() if reference_pose else None
        N = 4 if homogeneous else 3
        for actor in actors:
            actor_pose = (
                actor.pose
                if arg_actor_pose is None
                else arg_actor_pose.transform(actors[0].pose.inv().transform(actor.pose))
            )
            if reference_pose_inv:
                actor_pose = reference_pose_inv.transform(actor_pose)
            m = actor_pose.to_transformation_matrix()
            vertices.append(np.matmul(self._vtx_cache[actor], m.T)[:, 0:N])
        return np.concatenate(vertices, axis=0)

    def get_volume_point_cloud(
        self,
        actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        reference_pose: sapien.Pose = None,
        homogeneous: bool = False,
    ) -> np.ndarray:
        """
        :param actors: List of actors for which point cloud is requested.
        :param actor_pose: Can be used to override the pose of the actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0].
        To get the bounds in the local coordinate frame, use actor_pose=sapien.Pose()
        :param reference_pose: The pose relative to which the point cloud is computed
        :param homogeneous: Should the returned numpy array be homogeneous? (4th columns filled with 1)
        :return: A volume point cloud; 1000 samples per actor.
        """
        arg_actor_pose = actor_pose
        actors = self._prepare_actors(self._flatten_actors(actors))
        N = 4 if homogeneous else 3
        point_cloud = []
        reference_pose_inv = reference_pose.inv() if reference_pose else None
        for actor in actors:
            bounds = self.get_bounds(actor, reference_pose=actor.pose)
            points = self._rng.uniform(low=bounds[0], high=bounds[1], size=(1000, 3))
            points = points[self._sdf_cache[actor](points) >= 0]
            homogeneous_points = np.concatenate(
                [points, np.ones((points.shape[0], 1))], axis=1
            ).astype(np.float32)

            actor_pose = (
                actor.pose
                if arg_actor_pose is None
                else arg_actor_pose.transform(actors[0].pose.inv().transform(actor.pose))
            )
            if reference_pose_inv:
                actor_pose = reference_pose_inv.transform(actor_pose)
            m = actor_pose.to_transformation_matrix()
            point_cloud.append(np.matmul(homogeneous_points, m.T)[:, 0:N])
        point_cloud = np.concatenate(point_cloud, axis=0)

        return point_cloud

    def get_bounds(
        self,
        actors: List[sapien.Actor],
        actor_pose: sapien.Pose = None,
        reference_pose: sapien.Pose = None,
    ) -> np.ndarray:
        """
        :param actors: List of actors for which the collective bounds is requested.
        :param actor_pose: Can be used to override the pose of the actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0].
        :param reference_pose: The pose relative to which the bounding box is computed
        :return: The axis-aligned (relative to reference_pose) bounding box (np array of size 2x3)
        of the actor(s)
        """
        vertices = self.get_point_cloud(
            actors, actor_pose=actor_pose, reference_pose=reference_pose
        )
        return np.stack((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)

    def inside(
        self,
        actors: List[sapien.Actor],
        containing_actor: sapien.Actor,
        actor_pose: sapien.Pose = None,
        flat_dim: int = -1,
        return_fraction: bool = False,
    ):
        """
        Returns True if actor is fully inside containing_actor
        :param actors: the actor or actors that must be inside the containing_actor
        :param containing_actor: The actor that the actor must be inside of.
        :param actor_pose: Can be used to override the pose of the actors[0].
        actors[1:] will be transformed as if rigidly attached to actors[0].
        :param flat_dim: can be -1 (no flattening) or 0, 1, 2 for flattening along the
        corresponding dimension.
        :param return_fraction: if True, the fraction of vertices inside the containing_actor
        is returned.
        :return: bool (all inside) or float (fraction)
        """
        # Get point cloud of actors, relative to the pose of containing_actor
        if return_fraction:
            point_cloud = self.get_volume_point_cloud(actors, actor_pose, homogeneous=True)
        else:
            point_cloud = self.get_point_cloud(actors, actor_pose=actor_pose, homogeneous=True)

        m = containing_actor.pose.inv().to_transformation_matrix()
        point_cloud = np.matmul(point_cloud, m.T)[:, 0:3]

        convex_hull_sdf = self._get_convex_hull(containing_actor, flat_dim=flat_dim)

        distances = convex_hull_sdf(point_cloud)
        mask = distances >= 0
        return np.sum(mask) / len(mask) if return_fraction else np.all(mask)

    def _prepare_actor(self, actor: sapien.Actor) -> sapien.Actor:
        """
        Ensures that SDF and vertices have been cached for this actor.
        _d_sdf : signed distance function
        _d_vertices : N x 4 matrix of homogenous vertices
        :param actor: The actor
        :return The actor, or None on failure.
        """
        if not actor in self._sdf_cache:
            # Add SDF to actor
            mesh = get_actor_mesh(actor, to_world_frame=False)
            if (
                mesh is None
            ):  # this happens for the ground (Plane Geometry) and links that do not have geometry
                if actor.name == self._ground_name:  # does the user
                    ground_thickness = 0.02
                    mesh = trimesh.creation.box((2.0, 2.0, ground_thickness))
                    mesh.apply_translation((0, 0, -ground_thickness / 2 + self._ground_altitude))
                else:
                    return None
            self._sdf_cache[actor] = SDF(mesh.vertices, mesh.faces)

            # Add vertices to actor, sampled at decent density, and in homogeneous form
            vertices = mesh.vertices
            self._rds_cache[actor] = np.max(np.linalg.norm(vertices, axis=1))
            vertex_per_area = 0.025 * 0.025  # one vertex per every square inch
            expected_num_vertices = min(10000, int(round(mesh.area / vertex_per_area)))
            if vertices.shape[0] < expected_num_vertices:
                vertices = np.concatenate(
                    (vertices, mesh.sample(expected_num_vertices - vertices.shape[0])), axis=0
                )
            homogeneous_vertices = np.concatenate(
                [vertices, np.ones((vertices.shape[0], 1))], axis=1
            ).astype(np.float32)
            self._vtx_cache[actor] = homogeneous_vertices

        return actor

    def _prepare_actors(self, actors: List[sapien.Actor]) -> List[sapien.Actor]:
        """
        Calls self._prepare_actor() for each actor in actors.
        :param actors: The actors.
        :return: The actors that have been successfully prepared for distance calculations.
        """
        actors = [self._prepare_actor(actor) for actor in actors]
        return [actor for actor in actors if not actor is None]

    def _flatten_actors(self, actors: List[sapien.Actor]) -> List[sapien.Actor]:
        """
        Flattens articulations to a list of actors.
        :param actors: A list of actors (can be articulations, too)
        :return: List of Actors and Links.
        """
        if isinstance(actors, sapien.ActorBase):
            actors = [actors]
        if isinstance(actors, sapien.Articulation):
            actors = [actors]

        result = []
        for actor in actors:
            if isinstance(actor, sapien.Articulation):
                # Articulations are flattened to just the "root" parts
                result += self._get_static_articulation_links(actor)
            else:
                result.append(actor)
        return result

    def _get_static_articulation_links(
        self, articulation: sapien.Articulation
    ) -> List[sapien.Link]:
        """
        :param articulation: A Sapien articulation.
        :return: The Links of the articulation that can't move (heuristic)
        """
        active_joints = articulation.get_active_joints()
        links = []
        for j in articulation.get_joints():
            if j in active_joints:
                break
            if not j.get_parent_link() is None:
                links.append(j.get_parent_link())
            links.append(j.get_child_link())
        return list(set(links))

    def _get_convex_hull(self, actor: sapien.Actor, flat_dim: int = -1):
        """
        :param actor: The actor for which convex hull must be computed.
        :param flat_dim: can be -1 (no flattening) or 0, 1, 2 for flattening along the
        corresponding dimension.
        :return: Convex hull of actor.
        """
        key = (actor, flat_dim)
        if key not in self._cvx_sdf_cache:
            # Get vertices (in local frame of actor)
            self._prepare_actor(actor)
            vertices = self._vtx_cache[actor][:, 0:3]

            if flat_dim >= 0:  # flatten acros the requested dimension
                # Actually, the 2D convex hull is "stretched" from -10 to +10 to 3D SDF can be used
                vtx_top = np.copy(vertices)
                vtx_bottom = np.copy(vertices)
                vtx_top[:, flat_dim] = 10.0
                vtx_bottom[:, flat_dim] = -10.0
                vertices = np.concatenate((vtx_top, vtx_bottom), axis=0)

            mesh = trimesh.convex.convex_hull(vertices)
            self._cvx_sdf_cache[key] = SDF(mesh.vertices, mesh.faces)

        return self._cvx_sdf_cache[key]
