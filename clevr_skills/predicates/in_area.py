# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien

from clevr_skills.utils.actor_distance import ActorDistance

from .at_position import AtPosition


class InArea(AtPosition):
    """
    Like AtPosition, but the target is updated before evaluate() compute_dense_reward()
    """

    def __init__(self, env, actor: sapien.Actor, area_point_cloud: np.ndarray, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param actor: The actor that must be placed in the specified area.
        :param area_point_cloud: The area, specified by point cloud.
        :param name: Descriptive name of the predicate.
        """
        assert len(area_point_cloud.shape) == 2
        assert area_point_cloud.shape[1] == 3
        self.set_area_point_cloud(area_point_cloud)
        name = name if name else f"{actor.name} at in area"
        self._actor_distance: ActorDistance = env._actor_distance
        self.actor = actor
        pose = self.get_closest_pose()
        super().__init__(env, actor, pose, match_ori=False, name=name)
        self.actor = actor

    def get_closest_pose(self):
        """
        :return: The pose in the area closest to the actor.
        """
        actor_bounds = self._actor_distance.get_bounds(self.actor, actor_pose=sapien.Pose())
        closest_point_idx = np.argmin(
            np.linalg.norm((self.area_point_cloud - self.actor.pose.p)[:, 0:2], axis=1)
        )
        return sapien.Pose(
            self.area_point_cloud[closest_point_idx] - [0, 0, actor_bounds[0, 2] + 0.005]
        )

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        self.pose = self.get_closest_pose()
        return super().evaluate()

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6]).
        """
        self.pose = self.get_closest_pose()
        return super().compute_dense_reward()

    def set_area_point_cloud(self, area_point_cloud: np.ndarray):
        """
        :param area_point_cloud: Numpy array containing the point cloud.
        :return: None
        """
        self.area_point_cloud = area_point_cloud

    def feasible_positions(self, min_distance=0.05):
        """
        :return: Array of feasible positions that are available in the area.
        Unfeasible positions are too close to other actors, too close to the robot,
        or too far away from the robot.
        """
        other_actors = [
            a
            for a in self._env._scene.get_all_actors()
            if ((not a is self.actor) and (not a.name == "ground"))
        ] + [
            self._env.agent.robot
        ]  # for robot, only the static root will be used

        # Discard positions with distances below min_distance and those that are too far away
        distances = [
            (
                self._actor_distance.distance(
                    self.actor, other_actors, actor_pose=sapien.Pose(pos, self.actor.pose.q)
                ),
                pos,
            )
            for pos in self.area_point_cloud
        ]
        distances = [(d, pos) for d, pos in distances if d > min_distance]
        distances = [
            (d, pos)
            for d, pos in distances
            if np.linalg.norm(self._env.agent.robot.pose.p - pos) < 0.6
        ]

        result = np.array([pos for _, pos in distances], dtype=np.float32)
        result = np.reshape(result, (-1, 3))
        return result
