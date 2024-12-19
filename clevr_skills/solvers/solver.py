# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import sapien.core as sapien

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.at_pose import AtPose, AtPoseRotated
from clevr_skills.predicates.at_position import AtPosition
from clevr_skills.predicates.at_position_pushed import AtPositionPushed
from clevr_skills.predicates.balance_scale import BalanceScale
from clevr_skills.predicates.ee_at_pos import EEAtPosition
from clevr_skills.predicates.ee_at_pose import EEAtPose
from clevr_skills.predicates.hit_pose_predicate import HitPosPredicate
from clevr_skills.predicates.hit_predicate import HitPredicate
from clevr_skills.predicates.in_area import InArea
from clevr_skills.predicates.inside_2d import Inside2d
from clevr_skills.predicates.next_to import NextTo
from clevr_skills.predicates.on_top import OnTop
from clevr_skills.predicates.once import Once
from clevr_skills.predicates.order import Order
from clevr_skills.predicates.pick_predicate import PickPredicate
from clevr_skills.predicates.predicate import Predicate
from clevr_skills.predicates.rotate_predicate import RotatePredicate
from clevr_skills.predicates.sequence import Sequence
from clevr_skills.predicates.set import Set
from clevr_skills.predicates.topple_structure import ToppleStructurePredicate
from clevr_skills.predicates.touch import TouchPredicate
from clevr_skills.predicates.trace_predicate import TracePredicate
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.solvers.balance_scale_solver import BalanceScaleSolver
from clevr_skills.solvers.move3d_solver import Move3dSolver
from clevr_skills.solvers.pick_move3d_place_solver import PickMove3dPlaceSolver
from clevr_skills.solvers.pick_move3d_throw_solver import PickMove3dThrowSolver
from clevr_skills.solvers.push_solver import PushSolver
from clevr_skills.solvers.topple_structure import ToppleStructure
from clevr_skills.solvers.touch_solver import Move3dTouchSolver
from clevr_skills.solvers.trace_solver import TraceSolver
from clevr_skills.tasks.place_next_to import filter_actor_positions
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.logger import log


class Solver(AbstractSolver):
    """
    Top-level solver. Manages selection of next predicate to solve.
    """

    def __init__(self, env: ClevrSkillsEnv, ks_wrobot=False):
        """
        :param env: the ClevrSkillsEnv.
        :param ks_wrobot: Include the robot in keystep images?
        """
        super().__init__(env)
        self._current_predicate: Predicate = None
        self._last_step = None
        self.action_state["prompts"] = env.task.get_prompts()
        self._prev_action = None
        self.ks_wrobot = ks_wrobot
        self._warned_has_completed = (
            False  # this flag is used to warn only once when solver has nothing more to do
        )

    def get_eps_info(self):
        """
        :return: episode info from environment.
        """
        return self._env._get_eps_info()

    def step(self, obs):
        """
        Should return an action to take given the obs
        :param obs: The environment observation.
        :return: Tuple (action (numpy array), extra info)
        """
        self._reset_action_state()

        # check if current predicate is complete
        extra_info = {"keystep": False}

        if not self._current_predicate or self._current_predicate.evaluate()["success"]:
            if self.ks_wrobot:
                rimg, bimg = self._env._get_obs_with_robot()
            else:
                rimg, bimg = self._env._get_obs_without_robot()
            extra_info = {"keystep": True, "render_img": rimg, "base_img": bimg}
            self._current_predicate = self.select_next_predicate_to_solve(
                self._env.task.get_predicate()
            )
            self._sub_solver = (
                self.init_solver(self._current_predicate) if self._current_predicate else None
            )

        if not self._sub_solver is None:
            extra_info["act_label"] = self._sub_solver.get_current_action()
            self._prev_action = action = self._sub_solver.step(obs)
            self._warned_has_completed = False
            return action, extra_info

        extra_info["act_label"] = {"mid_act_label": "", "low_act_label": ""}
        log(
            "Warning: Solver has completed all predicates . . .",
            info=not self._warned_has_completed,
        )
        self._warned_has_completed = True
        gripper = self.gripper_off if self._prev_action is None else self._prev_action[-1]
        return self.return_hold_action(gripper=gripper), extra_info

    def select_next_predicate_to_solve(self, predicate: Predicate):
        """
        Recursively selects the next predicate to solve.
        :param predicate: Current predicate under consideration.
        :return: Next predicate to solve.
        """
        if predicate is None:
            return None
        if isinstance(predicate, Once):
            return self.select_next_predicate_to_solve(predicate._predicate)
        if isinstance(predicate, Order):
            return self.select_next_predicate_to_solve(predicate.get_current_predicate())
        if isinstance(predicate, (Sequence, Set)):
            for r in predicate._predicates:
                if not r.evaluate()["success"]:
                    return self.select_next_predicate_to_solve(r)
        elif not predicate.evaluate()["success"]:
            return predicate

    def init_solver(self, predicate):
        """
        Returns a solver for the predicate.
        :param predicate: The predicate
        :return: solver for predicate.
        """
        if isinstance(predicate, InArea):
            r: InArea = predicate
            pos = r.feasible_positions()

            pos = filter_actor_positions(self._env.task._workspace, pos)

            action_state_extra = {
                "target_pose_function": "in_area",
                "feasible_positions": pos,
                "area_description": r.description,
            }
            target_pose_name = "area"
            if isinstance(predicate, NextTo):
                n: NextTo = predicate
                area_actor = at_get_actor(n.other_actor)
                action_state_extra["area_actor"] = area_actor
                action_state_extra["area_direction"] = n.direction
                target_pose_name = f"area {n.description} {area_actor[0]}"
            p = pos[
                self._env._episode_rng.randint(len(pos))
            ]  # randomly samples on of the feasible positions
            solver = PickMove3dPlaceSolver(
                self._env,
                r.actor,
                target_pose=sapien.Pose(p, r.actor.pose.q),
                target_pose_name=target_pose_name,
                action_state_extra=action_state_extra,
            )
        elif isinstance(predicate, OnTop):
            r: OnTop = predicate
            solver = PickMove3dPlaceSolver(self._env, r.top_actor, target_actor=r.bottom_actor)
        elif isinstance(predicate, Inside2d):
            r: Inside2d = predicate
            solver = PickMove3dPlaceSolver(
                self._env, r.top_actor, target_actor=r.bottom_actor, match_ori=True
            )
        elif isinstance(predicate, RotatePredicate):
            r: RotatePredicate = predicate
            action_state_extra = {
                "target_pose_function": "rotate",
                "angle": r.degree,
                "restore": r.restore,
            }
            solver = PickMove3dPlaceSolver(
                self._env,
                r.actor,
                target_pose=r.pose,
                target_pose_name="designated pose",
                match_ori=True,
                action_state_extra=action_state_extra,
            )
        elif isinstance(predicate, AtPoseRotated):
            r: AtPoseRotated = predicate
            solver = PickMove3dPlaceSolver(
                self._env,
                r.actor,
                target_pose=r.pose,
                target_pose_name="designated pose",
                match_ori=True,
            )
        elif isinstance(predicate, AtPose):
            r: AtPose = predicate
            solver = PickMove3dPlaceSolver(
                self._env,
                r.actor,
                target_pose=r.pose,
                target_pose_name="designated pose",
                match_ori=True,
            )
        elif isinstance(predicate, AtPosition):
            r: AtPosition = predicate
            solver = PickMove3dPlaceSolver(
                self._env,
                r.actor,
                target_pose=r.pose,
                target_pose_name="designated pose",
                match_ori=False,
            )
        elif isinstance(predicate, AtPositionPushed):
            r: AtPositionPushed = predicate
            solver = PushSolver(self._env, r.actor, target_pose=r.pose)
        elif isinstance(predicate, BalanceScale):
            solver = BalanceScaleSolver(self._env, predicate.scale, predicate.objects)
        elif isinstance(predicate, PickPredicate):
            r: PickPredicate = predicate
            solver = PickMove3dPlaceSolver(
                self._env, r.actor, target_pose_name="lifted pose", match_ori=False, lift=0.1
            )
        elif isinstance(predicate, TracePredicate):
            r: TracePredicate = predicate
            solver = TraceSolver(self._env, predicate)
        elif isinstance(predicate, HitPredicate):
            r: HitPredicate = predicate
            solver = PickMove3dThrowSolver(
                self._env,
                predicate.throw_actor,
                predicate.target_actor,
                target_2d=predicate.target_2d,
                topple_target=predicate.topple_target,
            )
        elif isinstance(predicate, HitPosPredicate):
            r: HitPosPredicate = predicate
            solver = PickMove3dThrowSolver(
                self._env, predicate.throw_actor, target_actor=None, target_pos=r.target_pos
            )
        elif isinstance(predicate, TouchPredicate):
            r: TouchPredicate = predicate
            solver = Move3dTouchSolver(self._env, predicate.actor, predicate.push, predicate.topple)
        elif isinstance(predicate, EEAtPosition):
            r: EEAtPosition = predicate
            p = r.pose.p
            solver = Move3dSolver(
                self._env,
                predicate.pose,
                match_ori=False,
                target_pose_name=f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
            )
        elif isinstance(predicate, EEAtPose):
            r: EEAtPose = predicate
            p = r.pose.p
            solver = Move3dSolver(
                self._env,
                predicate.pose,
                match_ori=True,
                target_pose_name=f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
            )
        elif isinstance(predicate, ToppleStructurePredicate):
            r: ToppleStructurePredicate = predicate
            solver = ToppleStructure(self._env, r.actors)
        else:
            assert False, f"No solver for {predicate}"

        if hasattr(predicate, "action_state_extra"):
            solver.action_state.update(predicate.action_state_extra)

        return solver
