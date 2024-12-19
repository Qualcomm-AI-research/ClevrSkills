# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, List

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.sapien_utils import get_entity_by_name

from clevr_skills.clevr_skills_env import ClevrSkillsEnv
from clevr_skills.predicates.balance_scale import BalanceScale
from clevr_skills.solvers.abstract_solver import AbstractSolver
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_placement import get_all_meshes, get_random_non_intersecting_pose_v2
from clevr_skills.utils.logger import log

from .move3d_solver import Move3dSolver
from .pick_move3d_place_solver import PickMove3dPlaceSolver


class BalanceScaleSolver(AbstractSolver):
    def __init__(
        self, env: ClevrSkillsEnv, scale: sapien.Articulation, objects: List[sapien.Actor]
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param scale: The scale (sapien articulation).
        :param objects: The objects that must be placed on the scale.
        """
        super().__init__(env)
        self._scale = scale
        self._plate1 = get_entity_by_name(scale.get_links(), "plate1")
        self._plate2 = get_entity_by_name(scale.get_links(), "plate2")
        self.predicate = BalanceScale(
            env, scale, self._plate1, self._plate2, objects
        )  # used to eval the task, get placed/to-be-placed objects

        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.1, 0.3],
        ]
        self._sub_solver = None
        self._num_steps = 0

        self.action_state["scale"] = at_get_actor(scale)
        self.action_state["plate1"] = at_get_actor(self._plate1)
        self.action_state["plate2"] = at_get_actor(self._plate2)
        self.action_state["objects"] = [at_get_actor(obj) for obj in objects]

    def get_current_action(self) -> Dict[str, str]:
        """
        :return: the language label of the current action being taken.
        A dict with keys "mid_act_label" and "low_act_label".
        """
        if self._sub_solver is None:
            mid_level_act = low_level_act = "Place stuff on scale"
            return {"mid_act_label": mid_level_act, "low_act_label": low_level_act}
        return self._sub_solver.get_current_action()

    def step(self, obs):
        """
        :param obs: observation of environment.
        :return: The action, given the observation. Note that most solvers use privileged information
        from the ClevrSkillsEnv
        """
        eval = self.predicate.evaluate(return_by_name=False)
        self._num_steps += 1

        if eval["success"] or self._num_steps < 20:
            return self.return_hold_action(self.gripper_off)

        if self._sub_solver and self._sub_solver.is_done():
            self._sub_solver = None

        if self._sub_solver is None:
            incorrectly_placed_actors = eval["incorrect_plate1"] + eval["incorrect_plate2"]

            # figure out which actors can be picked up
            pickup_actors = []
            if len(incorrectly_placed_actors):
                pickup_actors += incorrectly_placed_actors
            else:
                pickup_actors += eval["unplaced"]

            # Select the actor closest to the gripper to pick up
            tcp_pos = self._env.agent.ee_link.pose.p
            _dist_axy = [
                np.linalg.norm(actor.pose.p[0:2] - tcp_pos[0:2]) for actor in pickup_actors
            ]
            if len(_dist_axy) == 0:
                # Nothing to do; waiting for success. Move to neutral pose
                ee_pose = self._env.agent.ee_link.pose
                target_p = [
                    min(self.predicate.plate1.pose.p[0], self.predicate.plate2.pose.p[0]) - 0.15,
                    0,
                    ee_pose.p[2],
                ]
                target_ee_pose = sapien.Pose(target_p, ee_pose.q)

                self._sub_solver = Move3dSolver(
                    self._env, target_ee_pose, match_ori=True, target_pose_name="neutral pose"
                )
            else:
                idx = np.argmin(_dist_axy)
                pickup_actor = pickup_actors[idx]

                # Figure out where to place the actor
                contacts = self._env._scene.get_contacts()
                stack1 = self.predicate._get_stack(
                    self.predicate.plate1, contacts, self.predicate._gravity
                )
                stack2 = self.predicate._get_stack(
                    self.predicate.plate2, contacts, self.predicate._gravity
                )
                target_actors = self.predicate._where_to_place([pickup_actor], stack1, stack2)

                # Get a target to move to and create the PickMove3dPlaceSolver
                if len(target_actors) == 0:
                    # Move this object to free space
                    # Have we already found a location self.target
                    _other_actor_meshes = get_all_meshes(
                        self._env, subdivide_to_size=0.02, exclude=["ground", pickup_actor.name]
                    )
                    target_pose, _ = get_random_non_intersecting_pose_v2(
                        self._env,
                        pickup_actor,
                        self._workspace,
                        offset_z=0.05,
                        angle_z=[0, 2 * np.pi],
                        max_attempts=100,
                        grow_actor_bounds=0.01,
                        allow_top=False,
                    )

                    log(f"Init new solver for {pickup_actor.name} to be placed on ground")
                    self._sub_solver = PickMove3dPlaceSolver(
                        self._env,
                        pickup_actor,
                        target_pose=target_pose,
                        match_ori=False,
                        target_pose_name="ground",
                    )

                else:
                    target_pos = [t.pose.p for t in target_actors]
                    _dist = np.linalg.norm(np.array(target_pos) - pickup_actor.pose.p, axis=1)
                    target_actor = target_actors[np.argmin(_dist)]

                    log(
                        f"Init new solver for {pickup_actor.name} to be placed on {target_actor.name}"
                    )
                    self._sub_solver = PickMove3dPlaceSolver(
                        self._env, pickup_actor, target_actor=target_actor, match_ori=False
                    )

        return self._sub_solver.step(obs)
