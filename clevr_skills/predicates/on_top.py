# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import sapien.core as sapien

from clevr_skills.utils.actor_distance import ActorDistance
from clevr_skills.utils.sapien_utils import check_actor_static, is_actor_resting_on_actor

from .predicate import EnvPredicate


class OnTop(EnvPredicate):
    """
    Predicate for placing an actor on top of another actor
    """

    def __init__(self, env, top_actor: sapien.Actor, bottom_actor: sapien.Actor, name=None):
        """
        :param env: The ClevrSkillsEnv.
        :param top_actor:
        :param bottom_actor:
        :param name: Descriptive name of the predicate.
        """
        name = name if name else f"{top_actor.name} on top of {bottom_actor.name}"
        super().__init__(env, name)
        self.bottom_actor = bottom_actor
        self.top_actor = top_actor
        self._actor_distance: ActorDistance = env._actor_distance
        self._gravity = self._env._scene.get_config().gravity
        self._top_actor_grasped_at_env_step = (
            -100
        )  # used to give correct reward while top actor is falling
        self._last_grasped_reward = -1  # used to give correct reward while top actor is falling

    def evaluate(self):
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        Successful if top_actor is on top of bottom actor and objects are not moving
        """
        top_actor_static = check_actor_static(self.top_actor)
        bottom_actor_static = check_actor_static(self.bottom_actor)
        top_actor_grasped = self._env.agent.check_grasp(self.top_actor)
        bottom_actor_grasped = self._env.agent.check_grasp(self.bottom_actor)

        top_actor_resting_on_bottom_actor = is_actor_resting_on_actor(
            self._env._scene, self.top_actor, self.bottom_actor, self._env._ground
        )

        success = (
            top_actor_static
            and bottom_actor_static
            and (not top_actor_grasped)
            and (not bottom_actor_grasped)
            and top_actor_resting_on_bottom_actor
        )

        return {
            "top_actor_static": top_actor_static,
            "bottom_actor_static": bottom_actor_static,
            "top_actor_grasped": top_actor_grasped,
            "bottom_actor_grasped": bottom_actor_grasped,
            "top_actor_resting_on_bottom_actor": top_actor_resting_on_bottom_actor,
            "success": success,
        }

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6]).
        """
        eval = self.evaluate()

        if eval["success"]:
            return 6.0
        if eval["top_actor_resting_on_bottom_actor"]:
            return 5.0 + (0.0 if eval["top_actor_grasped"] else 1.0)
        if eval["top_actor_grasped"]:  # max 5 reward
            CLOSE = 0.03
            dist_xy = np.linalg.norm(self.top_actor.pose.p[0:2] - self.bottom_actor.pose.p[0:2])
            dist_z = self.top_actor.pose.p[2] - self.bottom_actor.pose.p[2]
            # need to hoover

            self._top_actor_grasped_at_env_step = self._env.elapsed_steps

            is_close = dist_xy < CLOSE
            if is_close:  # max 5 reward
                # reward getting close to the object in the Z direction
                if dist_z < 0.0:
                    reward = 4.0
                else:
                    reward = 4.0 + np.exp(-6.0 * dist_z)
            else:  # max 4 reward
                # reward getting close in the XY direction, and being above a certain height
                # above the target
                reward = (
                    2
                    + np.exp(-max(0.0, dist_xy - CLOSE))
                    + np.exp(-max(0.0, 6.0 * (-dist_z + 0.15)))
                )
            self._last_grasped_reward = reward
            return reward

        # reach for top actor; max 2 reward (or 5 reward, if top_actor is falling)
        contact = np.any(
            [
                (not isinstance(a, sapien.Link) or a == self.bottom_actor)
                for a in self._env.get_actor_contacts(self.top_actor)
            ]
        )
        if (self._env.elapsed_steps - self._top_actor_grasped_at_env_step < 10) and not contact:
            return self._last_grasped_reward

        # reward EE getting close to top of object
        CLOSE = 0.03
        REALLY_CLOSE = 0.005
        actor_bounds = self._actor_distance.get_bounds(self.top_actor)
        tcp_pos = self._env.agent.ee_link.pose.p
        dist_xy = np.linalg.norm(self.top_actor.pose.p[0:2] - tcp_pos[0:2])
        dist_z = tcp_pos[2] - actor_bounds[1, 2]

        is_close_xy = dist_xy < CLOSE
        if is_close_xy:  # max 2 reward
            # reward getting close to the object in the Z direction
            if dist_z < REALLY_CLOSE:
                return 2.0
            return 1.0 + np.exp(-dist_z)

        # max 1 reward
        # reward getting close in the XY direction
        return np.exp(-max(0.0, dist_xy - CLOSE))
