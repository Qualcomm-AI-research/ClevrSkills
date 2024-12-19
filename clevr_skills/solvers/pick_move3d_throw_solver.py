# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.sapien_utils import check_actor_static

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.solvers.move3d_solver import Move3dSolver
from clevr_skills.solvers.pick_solver import PickSolver
from clevr_skills.solvers.throw_solver import ThrowSolver
from clevr_skills.utils.action_trace import at_get_actor, at_get_pos
from clevr_skills.utils.geometry import unit
from clevr_skills.utils.visualize_prompt import get_actor_description


class PickMove3dThrowSolver(AbstractSolver):
    def __init__(
        self,
        env: ClevrSkillsEnv,
        throw_actor,
        target_actor,
        target_pos=None,
        target_2d: bool = False,
        topple_target: bool = False,
    ):
        """
        :param env:The ClevrSkillsEnv
        :param throw_actor: the actor that is to-be-thrown
        :param target_actor: the actor that must be hit
        :param target_pos: Optional target position to throw at. By default, the pose of the target_actor is used.
        :param target_2d: Whether the target is 2D (embedded in the floor)
        :param topple_target: Whether the target must be toppled. Not actually used currently
        """
        super().__init__(env)
        self.throw_actor = throw_actor
        self.target_actor = target_actor
        self.target_pos = self.target_actor.pose.p if self.target_actor is not None else target_pos
        self.grasping_history = [False]
        self.move_to_prethrow_pose = False
        self.target_2d = target_2d
        self.topple_target = topple_target
        self._done = False

        # Determine from what point to start the throw
        # Use two candidates that are perpendicular to the throwing direction,
        # plus some small random factor. Pick the one closest to the current
        # location of the to-be-thrown actor
        self.prethrow_pose = None

        direction = unit(np.cross(self.target_pos - self._env._robot_pos, [0, 0, 1]))
        candidates = [
            self._env._robot_pos + 0.25 * direction + [0, 0, 0.25],
            self._env._robot_pos - 0.25 * direction + [0, 0, 0.25],
        ]
        candidates = [c + np.random.uniform(low=[-0.05] * 3, high=[0.05] * 3) for c in candidates]
        best_distance = 100.0
        for p in candidates:
            d = np.linalg.norm(self.throw_actor.pose.p - p)
            if d < best_distance:
                best_distance = d
                self.prethrow_pose = sapien.Pose(p, self._env.agent.neutral_ee_quat)

        self.action_state["throw_actor"] = at_get_actor(throw_actor)
        self.action_state["target_actor"] = at_get_actor(target_actor)
        self.action_state["target_pos"] = at_get_pos(target_pos)
        self.action_state["target_2d"] = target_2d
        self.action_state["topple_target"] = topple_target

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        if self.target_actor is not None:
            mid_level_act = (
                f"Hit {get_actor_description(self.target_actor)} "
                f"with {get_actor_description(self.throw_actor)} "
                f"by throwing"
            )
        else:
            mid_level_act = (
                f"Throw {get_actor_description(self.throw_actor)} at " f"{{pos:{self.target_pos}}}"
            )
        grasping = self._env.agent.check_grasp(self.throw_actor)
        if not grasping:
            low_level_act = (
                self._sub_solver.get_current_action()["low_act_label"]
                if self._sub_solver is not None
                else f"Picking {get_actor_description(self.throw_actor)}"
            )
        elif self.move_to_prethrow_pose or (
            len(self.grasping_history) >= 2 and not self.grasping_history[-2]
        ):
            low_level_act = f"Wind before throwing {get_actor_description(self.throw_actor)}"
        else:
            if self.target_actor is not None:
                low_level_act = (
                    f"Throw {get_actor_description(self.throw_actor)} "
                    f"at {get_actor_description(self.target_actor)}"
                )
            else:
                low_level_act = (
                    f"Throw {get_actor_description(self.throw_actor)} at "
                    f"{{pos:{self.target_pos}}}"
                )

        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        ee_pose: sapien.Pose = self._env.agent.ee_link.pose

        if self._sub_solver and self._sub_solver.is_done():
            self._sub_solver = None
            if isinstance(self._sub_solver, ThrowSolver):
                self._done = True

        close_to_prethrow_pose = np.linalg.norm(ee_pose.p - self.prethrow_pose.p) < 0.05
        if isinstance(self._sub_solver, Move3dSolver) and close_to_prethrow_pose:
            self._sub_solver = None

        grasping = self._env.agent.check_grasp(self.throw_actor)
        self.grasping_history.append(grasping)

        if self._sub_solver is None:
            if grasping:
                # Can't we test here for distance to pre-throw pose?
                # And if close enough, throw
                # else wind?
                if not close_to_prethrow_pose:
                    self.action_state["action"] = (
                        "winding"  # this should not be needed; should use sub-solvers instead
                    )
                    self._sub_solver = Move3dSolver(
                        self._env,
                        self.prethrow_pose,
                        match_ori=True,
                        vacuum=True,
                        target_pose_name="pre-throw pose",
                    )

                else:
                    self.action_state["action"] = "throwing"
                    self._sub_solver = ThrowSolver(
                        self._env,
                        self.throw_actor,
                        self.target_actor,
                        self.target_pos,
                        self.target_2d,
                        self.topple_target,
                    )

            elif check_actor_static(self.throw_actor):
                self.action_state["action"] = "picking"
                self._sub_solver = PickSolver(self._env, self.throw_actor)
        elif self._sub_solver is None:
            self.action_state["action"] = "waiting"
        else:
            self.action_state["action"] = "picking"

        if self._sub_solver is None:
            return self.return_hold_action(self.gripper_on)
        else:
            action = self._sub_solver.step(obs)
            return action

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return self._done
