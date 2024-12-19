# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien

from clevr_skills.utils.actor_distance import ActorDistance

from .in_area import InArea


class NextTo(InArea):
    """
    Succeeds when actor is placed next to other_actor.
    The geometrical direction of "next to" is determined by direction.

    A point-cloud of acceptable locations is generated and InArea handles the actual
    success & reward computation
    """

    def __init__(
        self,
        env,
        actor: sapien.Actor,
        other_actor: sapien.Actor,
        direction: np.ndarray,
        description: str,
        close: float = 0.5,
        far: float = 2.5,
        name: str = None,
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor to be placed next to other_actor
        :param other_actor: The used as the relative target for placement.
        :param direction: The XY direction
        :param description: The description of the direction. A string like "left of",
        or "behind"
        :param close: This constant scales the mean XY of actor and mean_actor to get
        the distance that is considered to be "too close" in the direction of placement.
        It also determines the width of the acceptable area.
        :param far: This constant scales the mean XY of actor and mean_actor to get the
        distance that is considered to be "too far".
        :param name: Descriptive name of the predicate.
        """
        self._actor_distance: ActorDistance = env._actor_distance
        self.actor = actor
        self.other_actor = other_actor
        self._step_size = 0.025  # approximate step size of point cloud grid
        self.close = close
        self.far = far
        self.direction = direction
        self.ortho_direction = np.cross(self.direction, [0, 0, 1])
        self.description = description
        self._other_actor_pose = other_actor.pose
        self._env = env

        self.update_area(force_update=True)

        super().__init__(env, actor, self.area_point_cloud, name=name)

    def update_area(self, force_update: bool = False) -> None:
        """
        Updates the area point cloud of the super class if the self.other_actor has moved,
        or if the update is forced.
        :param force_update: If force_update is True, the area is always updated.
        """
        if (
            force_update
            or np.linalg.norm(self._other_actor_pose.p - self.other_actor.pose.p) > 1e-5
        ):
            self._other_actor_pose = self.other_actor.pose

            # Get mean XY size of involved actor
            other_actor_bounds = self._actor_distance.get_bounds(self.other_actor)
            actor_bounds = self._actor_distance.get_bounds(self.actor)
            size = max(
                0.05,
                np.mean(
                    [
                        np.mean(other_actor_bounds[1, 0:2] - other_actor_bounds[0, 0:2]),
                        np.mean(actor_bounds[1, 0:2] - actor_bounds[0, 0:2]),
                    ]
                ),
            )

            # Create a grid to represent the area where the actor can be placed
            # The area scales with the size of the involved actors
            # The grid is longer in the placement direction (self.direction) than in the
            # orthogonal direction (self.ortho_direction)
            start, end = self.close * size, self.far * size
            s1 = np.linspace(start, end, num=int(round((end - start) / self._step_size)))
            start, end = -self.close * size, self.close * size
            s2 = np.linspace(start, end, num=int(round((end - start) / self._step_size)))
            p = np.copy(self.other_actor.pose.p)
            p[2] = -actor_bounds[0, 2] + self.actor.pose.p[2] + 0.01
            pcd = np.array(
                [
                    p + d1 * self.direction + d2 * self.ortho_direction
                    for d1 in s1
                    for d2 in s2
                    if (2 * abs(d2) <= abs(d1))
                ],
                dtype=np.float32,
            )

            self.set_area_point_cloud(pcd)

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        self.update_area()  # update the point cloud area, if the other actor has moved
        return super().evaluate()

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6]).
        """
        self.update_area()  # update the point cloud area, if the other actor has moved
        return super().compute_dense_reward()
