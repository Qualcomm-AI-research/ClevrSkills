# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from sapien.core import Pose

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.trace_predicate import TracePredicate
from clevr_skills.solvers.move3d_solver import Move3dSolver


class TraceSolver(Move3dSolver):
    def __init__(self, env: ClevrSkillsEnv, predicate: TracePredicate):
        """
        :param env: The ClevrSkillsEnv
        :param predicate: The predicate to solve.
        """
        p = predicate.pose.p
        super().__init__(
            env,
            ee_target_pose=predicate.pose,
            match_ori=False,
            target_pose_name=f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
        )
        self.predicate: TracePredicate = predicate
        self.pose: Pose = predicate.pose

    def get_current_action(self) -> str:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        mid_level_act = "INDETERMINABLE"
        low_level_act = f"Moving end-effector to {{pos:{self.pose.p}}}"
        return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}
