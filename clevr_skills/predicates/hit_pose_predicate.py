# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
import sapien.core as sapien
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse
from mani_skill2.utils.trimesh_utils import get_actor_mesh

from clevr_skills.utils.actor_distance import ActorDistance

from .pick_predicate import PickPredicate
from .predicate import EnvPredicate


class HitPosPredicate(EnvPredicate):
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
        self, env, throw_actor: sapien.Actor, target_pos: np.ndarray, fly_distance=0.05, name=None
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param throw_actor: The actor that should be thrown.
        :param target_pos: The target pos where we want the actor to land.
        :param fly_distance: How far the throw_actor should fly, measured in XY (horizontal)
        place for the hit to "count".
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"Hit {{pos:{target_pos}}} with {throw_actor.name}"
        super().__init__(env, name)
        self.throw_actor = throw_actor
        self.target_pos = target_pos
        self.target_hit = False
        self.required_flight_distance = fly_distance

        self.throw_actor_pose_at_takeoff = None

        self._actor_distance: ActorDistance = env._actor_distance

        self._pick_predicate = PickPredicate(env, self.throw_actor)

        vertices = get_actor_mesh(self.throw_actor, to_world_frame=False).vertices
        self.throw_actor_rds = np.max(np.linalg.norm(vertices, axis=1))

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
            ground_impulse = get_pairwise_contact_impulse(
                self._env._scene.get_contacts(), self._env._ground, self.throw_actor
            )
            actor_touching_ground = np.linalg.norm(ground_impulse) > 1e-5
            if actor_touching_ground:
                distance = (
                    np.linalg.norm(self.throw_actor.pose.p - self.target_pos) - self.throw_actor_rds
                )
                self.target_hit = distance < 0.1

        if throw_actor_grasped or throw_actor_in_contact:
            # If the throw actor comes into contact, reset the takeoff pose
            self.throw_actor_pose_at_takeoff = None

        min_distance = self.predict_trajectory_min_distance()

        success = self.target_hit

        result = {
            "throw_actor_grasped": throw_actor_grasped,
            "throw_actor_in_flight": not throw_actor_in_contact,
            "target_hit": self.target_hit,
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
        :return: Distance to the target, at any point during the ballistic trajectory.
        """
        trajectory = []
        throw_actor_pose = self.throw_actor.pose
        pos = throw_actor_pose.p
        vel = self.throw_actor.get_velocity()
        gravity = self._env._scene.get_config().gravity
        for t in np.linspace(0, interval, round(interval / 0.05), endpoint=True):
            trajectory.append(pos + t * vel + 0.5 * t * t * gravity)

        best_pos = None
        best_distance = 1000
        for pos in trajectory:
            distance = np.linalg.norm(self.target_pos - pos)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        computed_distance = np.linalg.norm(self.throw_actor.pose.p - best_pos)

        return computed_distance
