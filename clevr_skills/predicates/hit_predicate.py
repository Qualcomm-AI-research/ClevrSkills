# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien
from transforms3d.quaternions import quat2mat

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.logger import log

from .pick_predicate import PickPredicate
from .predicate import EnvPredicate


class HitPredicate(EnvPredicate):
    """
    Predicate for hitting the target_actor with the throw_actor
    The goal can be 2D (throw_actor only has to "fly over" the target_actor).
    Or it can be in full 3D, requiring physical contact, and potentially requiring the
    toppling of the target. Toppling is described as changing the orientation by more
    than 45 degrees, measured along the (originally) vertical axis.

    This predicate has a state. I.e., it will remember whether a hit was achieved once,
    and continue to return perfect reward after that.
    """

    def __init__(
        self,
        env,
        throw_actor: sapien.Actor,
        target_actor: sapien.Actor,
        target_2d,
        topple_target,
        fly_distance=0.05,
        name=None,
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param throw_actor: The actor that should be thrown.
        :param target_actor: The actor that should be hit with the throw_actor.
        :param target_2d: If True, the world is treated as "2D" (the vertical Z dimension
        is ignored). "Hitting" the target can also be done by flying over the target.
        :param topple_target: Should the target be thrown over (change in orientation
        of vertical axis > 45 degrees).
        :param fly_distance: How far the throw_actor should fly, measured in XY (horizontal)
        place for the hit to "count".
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"Hit {target_actor.name} with {throw_actor.name}"
        super().__init__(env, name)
        self.throw_actor = throw_actor
        self.target_actor = target_actor
        self.target_2d = target_2d
        self.topple_target = topple_target
        self.target_hit = False
        self.required_flight_distance = fly_distance

        self.throw_actor_pose_at_takeoff = None

        self.target_initial_vertical_axis = self._get_target_vertical_axis()

        self._actor_distance: ActorDistance = env._actor_distance

        self._pick_predicate = PickPredicate(env, self.throw_actor)

    def _get_target_vertical_axis(self):
        """
        :return: The vertical axis of the target actor.
        """
        return quat2mat(self.target_actor.pose.q)[:, 2]

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        Successful if target_actor was hit and the throw_actor has flown far enough.
        """
        throw_actor_grasped = self._env.agent.check_grasp(self.throw_actor)
        throw_actor_in_contact = self._env.any_contact(self.throw_actor)

        # Remember the pose when the throw actor started "flying"
        if (
            not throw_actor_in_contact
            and self.throw_actor_pose_at_takeoff is None
            and not self.target_hit
        ):
            self.throw_actor_pose_at_takeoff = self.throw_actor.pose

        distance_flown = (
            np.linalg.norm((self.throw_actor_pose_at_takeoff.p - self.throw_actor.pose.p)[0:2])
            if self.throw_actor_pose_at_takeoff
            else 0.0
        )
        if not self.target_hit and not throw_actor_grasped:
            # Check if target was hit during this sim step
            if (
                self.target_2d
            ):  # throw_actor is only required to fly over the target, so flatten the Z axis
                distance_2d = self._actor_distance.distance(
                    self.throw_actor, self.target_actor, flat_dim=2
                )
                log(f"Distance to target: {distance_2d}   distance_flown: {distance_flown}")
                self.target_hit = distance_2d < 0.0
            else:
                self.target_hit = self._env.get_contact(self.target_actor, self.throw_actor) > 0

        if throw_actor_grasped or throw_actor_in_contact:
            # If the throw actor comes into contact, reset the takeoff pose
            self.throw_actor_pose_at_takeoff = None

        target_vertical_axis = quat2mat(self.target_actor.pose.q)[:, 2]
        cos_angle = np.dot(target_vertical_axis, self.target_initial_vertical_axis)
        target_toppled = np.rad2deg(np.arccos(cos_angle)) > 45

        min_distance = self.predict_trajectory_min_distance()

        success = self.target_hit and (not self.topple_target or target_toppled)

        result = {
            "throw_actor_grasped": throw_actor_grasped,
            "throw_actor_in_flight": not throw_actor_in_contact,
            "target_hit": self.target_hit,
            "target_toppled": target_toppled,
            "flight_distance": distance_flown,
            "trajectory_min_distance": min_distance,
            "success": success,
        }
        return result

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 12]).
        """
        eval = self.evaluate()

        if eval["success"]:
            return 12.0
        if eval["throw_actor_grasped"]:
            return 4.0 + 4.0 * np.exp(-eval["trajectory_min_distance"] * 10)
        return self._pick_predicate.compute_dense_reward()

    def predict_trajectory_min_distance(self, interval=3.0):
        """
        Computes the trajectory of the throw_actor, assuming it will follow a ballistic trajectory
        Then determine if the throw_actor will fly over / hit the target.
        Uses a trivial method for integration
        :param interval: how many second to probe into the future
        :return: Closest distance the actor will come to the target actor.
        """
        trajectory = []
        throw_actor_pose = self.throw_actor.pose
        pos = throw_actor_pose.p
        vel = self.throw_actor.get_velocity()
        gravity = self._env._scene.get_config().gravity
        for t in np.linspace(0, interval, round(interval / 0.05), endpoint=True):
            trajectory.append(pos + t * vel + 0.5 * t * t * gravity)

        # Target to hit the center-top of the target
        target_bounds = self._actor_distance.get_bounds(self.target_actor)
        target_pos = np.array(
            [target_bounds[0, 0], np.mean(target_bounds[:, 1]), target_bounds[1, 2]]
        )

        best_pos = None
        best_distance = 1000
        for pos in trajectory:
            distance = np.linalg.norm(target_pos - pos)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        flat_dim = 2 if self.target_2d else -1
        computed_distance = self._actor_distance.distance(
            self.throw_actor,
            self.target_actor,
            actor_pose=sapien.Pose(best_pos, throw_actor_pose.q),
            flat_dim=flat_dim,
        )

        return computed_distance
