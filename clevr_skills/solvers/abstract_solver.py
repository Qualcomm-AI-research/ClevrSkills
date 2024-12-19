# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import sapien.core as sapien

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.utils.controller import delta_pose_to_action, ee_pose_to_ee_delta_action


class AbstractSolver(ABC):
    def __init__(self, env: ClevrSkillsEnv):
        self._env: ClevrSkillsEnv = env
        self._sub_solver = None  # solvers can use other solvers internally
        self.gripper_on = 1.0
        self.gripper_off = -1.0
        self.nop_action = self._env.action_space.sample() * 0

        # A dict that describes the current action
        # This info is extracted during the run of the oracle and stored with the recording
        # After the run, information for imitation learning can extracted
        self.action_state = {
            "solver_class": type(self).__name__,
            "solver_python_id": id(self),
            "solver_random_id": np.random.randint(1000000000),
            "action_emitted": False,
        }

    def get_current_action(self) -> Dict[str, str]:
        """
        Returns the language label of the current action being taken.
        """
        return {"mid_act_label": "", "low_act_label": ""}

    @abstractmethod
    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """
        Solvers that are meant to be "chained" should have the ability to report when they are done.
        :return: True if solver has completed its task.
        """
        return False

    def return_action(self, action: np.ndarray) -> np.ndarray:
        """
        This function sets self.action_state["generated_action"] to True and returns
        the action.
        This is a mechanism that makes to possible to see which specific solver returned the action.
        :param action: The action (step) to be returned.
        :return: the action
        """
        self.action_state["action_emitted"] = True

        return action

    def return_hold_action(self, gripper: float) -> np.ndarray:
        """
        Returns a "zero" action that holds the end-effector still.
        The gripper action can have arbitrary value.
        self.return_action() is called such that it is logged that the solver emitted an action.
        :param gripper: The gripper action.
        :return: The action.
        """
        action = np.copy(self.nop_action)
        action[-1] = gripper
        return self.return_action(action)

    def return_delta_pose_action(
        self,
        delta_pose: sapien.Pose,
        gripper: float,
        pos_thresh=0.1,
        rot_thresh=np.deg2rad(2),
        pos_multiplier=0.5,
        rot_multiplier=1.0,
    ) -> np.ndarray:
        """
        Returns an action such that the end-effector will move according to the delta_pose.
        delta_pose_to_action() is called to convert the delta_pose to the agent action.
        The gripper action can have arbitrary value.
        self.return_action() is called such that it is logged that the solver emitted
        an action.
        :param delta_pose: The delta pose.
        :param gripper: The gripper action.
        :param pos_thresh: If the norm of the delta-position is larger than this threshold, the delta-position gets scaled.
        :param rot_thresh: If the norm of the delta-rotation is larger than this threshold, the delta-rotation gets scaled.
        :param pos_multiplier: Use this to scale the delta-pos action, after conversion.
        :param rot_multiplier: Use this to scale the delta-rot action, after conversion.
        :return: The action
        """

        action = delta_pose_to_action(
            delta_pose,
            gripper=gripper,
            pos_thresh=pos_thresh,
            rot_thresh=rot_thresh,
            pos_multiplier=pos_multiplier,
            rot_multiplier=rot_multiplier,
        )
        return self.return_action(action)

    def return_global_pose_action(
        self,
        target_pose: sapien.Pose,
        gripper: float,
        scale_action_to_avoid_clipping: bool = True,
        verify_action: bool = False,
    ):
        """
        Returns an action such that the end-effector will move towards the target_pose
        in global coordinates.
        ee_pose_to_ee_delta_action() is called to convert the global target_pose to a
        delta action. The gripper action can have arbitrary value.
        self.return_action() is called such that it is logged that the solver emitted
        an action.
        :param target_pose: Target pose in global coordinate system.
        :param gripper: the gripper action.
        :param scale_action_to_avoid_clipping: This value is passed on to
        ee_pose_to_ee_delta_action().
        :param verify_action: this value is passed on to ee_pose_to_ee_delta_action().
        If true, the converted action is verified.
        :return: Action.
        """
        action, _ = ee_pose_to_ee_delta_action(
            self._env.agent,
            target_pose,
            gripper=gripper,
            scale_action_to_avoid_clipping=scale_action_to_avoid_clipping,
            verify_action=verify_action,
        )
        return self.return_action(action)

    def _reset_action_state(self) -> None:
        """
        Sets self.action_state["action_emitted"] to False.
        Should be called at the start of each step.
        """
        self.action_state["action_emitted"] = False
        if self._sub_solver:
            self._sub_solver._reset_action_state()
