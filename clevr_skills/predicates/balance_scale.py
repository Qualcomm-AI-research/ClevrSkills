# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.trimesh_utils import get_actor_mesh

from .predicate import EnvPredicate


class BalanceScale(EnvPredicate):
    """
    Predicate that succeeds if the objects are placed balanced on the scale.
    The scale is expected to have two plates where object can be placed.
    """

    CLOSE = 0.03

    def __init__(
        self,
        env,
        scale: sapien.Articulation,
        plate1: sapien.Link,
        plate2: sapien.Link,
        objects: List[sapien.Actor],
        name=None,
    ):
        """
        :param env: The ClevrSkillsEnv.
        :param scale: The scale articulation.
        :param plate1: Plate 1 of the scale (a link of the scale articulation).
        :param plate2: Plate 2 of the scale (a link of the scale articulation).
        :param objects: The objects to be placed on the scale.
        :param name: Optional, descriptive name of the predicate.
        """
        name = name if name else f"Balance {len(objects)} objects on scale {scale.name}"
        super().__init__(env, name)
        self.scale = scale
        self.plate1 = plate1
        self.plate2 = plate2
        self.objects = objects
        self._gravity = self._env._scene.get_config().gravity

        # Figure out all combinations that lead to equal weight
        # self._possible_solutions is a List of 2-tuples. Each 2-tuple is two lists of actors.
        # I.e., which actors should go onto which plate
        self._possible_solutions = self._find_balanced_solutions(self.objects)
        assert len(self._possible_solutions) > 0, "Could not find solution for BalanceScale"

    def evaluate(self, return_by_name=True):  # add argument to return by name
        """
        Evaluate if the predicate is successful or not.
        :return: A dictionary containing at least "success", but also additional information.
        """
        # Get the objects stacked on the plates
        contacts = self._env._scene.get_contacts()
        stack1 = self._get_stack(self.plate1, contacts, self._gravity)[1:]
        stack2 = self._get_stack(self.plate2, contacts, self._gravity)[1:]

        # Match the stacks to the best possible solution
        solution, stack_success = self._match_best_solution(
            self._possible_solutions, stack1, stack2
        )
        solution_stack1, solution_stack2 = ([], []) if solution is None else solution

        # Check if scale is not hitting the limits and not moving
        scale_balanced = (
            (self.scale.get_qpos()[0] - self.scale.get_qlimits()[0, 0] > np.deg2rad(1.0))
            and (self.scale.get_qlimits()[0, 1] - self.scale.get_qpos()[0] > np.deg2rad(1.0))
            and (abs(self.scale.get_qvel()[0]) < np.deg2rad(1.0))
        )

        # extract data for eval
        correct_plate1 = [o for o in stack1 if o in solution_stack1]
        correct_plate2 = [o for o in stack2 if o in solution_stack2]
        incorrect_plate1 = [o for o in stack1 if not o in solution_stack1]
        incorrect_plate2 = [o for o in stack2 if not o in solution_stack2]
        stack_12 = stack1 + stack2
        unplaced = [o for o in self.objects if not o in stack_12]

        result = {
            "stack1": stack1,
            "stack2": stack2,
            "correct_plate1": correct_plate1,
            "correct_plate2": correct_plate2,
            "incorrect_plate1": incorrect_plate1,
            "incorrect_plate2": incorrect_plate2,
            "unplaced": unplaced,
            "scale_balanced": scale_balanced,
            "stacked_success": stack_success,
            "success": stack_success and scale_balanced,
        }
        if return_by_name:
            for key, value in result.items():
                if isinstance(value, list):
                    result[key] = [o.name for o in value]

        return result

    def compute_dense_reward(self):
        """
        :return: dense reward (float in range [0, 6 * num_actors]).
        """
        REWARD_PER_CORRECTLY_PLACED = 6.0  # or whatever reward per object
        eval = self.evaluate(return_by_name=False)
        if eval["success"]:
            return len(self.objects) * REWARD_PER_CORRECTLY_PLACED

        incorrectly_placed_actors = eval["incorrect_plate1"] + eval["incorrect_plate2"]
        correctly_placed_actor = eval["correct_plate1"] + eval["correct_plate2"]

        # negative reward for incorrectly placed objects
        reward = (
            len(correctly_placed_actor) - len(incorrectly_placed_actors)
        ) * REWARD_PER_CORRECTLY_PLACED

        grasped_actors: List[sapien.Actor] = self._env.agent.get_grasped_actors()

        if len(grasped_actors) > 0:

            contacts = self._env._scene.get_contacts()
            stack1 = self._get_stack(self.plate1, contacts, self._gravity)
            stack2 = self._get_stack(self.plate2, contacts, self._gravity)

            target_actors = self._where_to_place(grasped_actors, stack1, stack2)

            if len(target_actors) == 0:
                # This object previously gave a reward of -6 because it was incorrectly placed.
                # Picking it up lowers that negative reward to -3
                reward -= 3

                distance = self._computed_nearest_actor_2d(grasped_actors)
                reward += 3 * np.exp(10.0 * min(distance, 0.0))
            else:
                reward += 3  # 3 reward for simply picking up an object that can be placed

                # An object was picked that should be placed on a stack
                grasped_actors_min_z, _ = self._get_z_range(grasped_actors)
                _, target_actors_max_z = self._get_z_range(target_actors)
                _dist_txy = [
                    np.linalg.norm(actor.pose.p[0:2] - grasped_actors[0].pose.p[0:2])
                    for actor in target_actors
                ]
                idx = np.argmin(_dist_txy)
                dist_xy = _dist_txy[idx]
                dist_z = grasped_actors_min_z - target_actors_max_z

                if dist_xy < self.CLOSE:
                    # at target, need to lower
                    dist_z = max(0.0, dist_z - self.CLOSE)
                    reward += 2.0 + np.exp(-5.0 * dist_z)
                else:
                    if dist_z < self.CLOSE:
                        reward += np.exp(-5.0 * max(0, self.CLOSE - dist_z))
                    else:
                        reward += 1.0 + np.exp(-5.0 * max(0.0, dist_xy - self.CLOSE))
        else:
            reward_distance_actors = []
            if len(incorrectly_placed_actors):
                reward_distance_actors += incorrectly_placed_actors
            else:
                reward_distance_actors += eval["unplaced"]

            # get nearest reward_distance_actors
            if len(reward_distance_actors) > 0:
                tcp_pos = self._env.agent.ee_link.pose.p
                _dist_axy = [
                    np.linalg.norm(actor.pose.p[0:2] - tcp_pos[0:2])
                    for actor in reward_distance_actors
                ]
                idx = np.argmin(_dist_axy)
                target_actor = reward_distance_actors[idx]
                _, target_actor_max_z = self._get_z_range(target_actor)
                dist_z = tcp_pos[2] - target_actor_max_z
                dist_xy = _dist_axy[idx]

                if dist_xy < self.CLOSE:  # max 2 reward + reward getting to grasp
                    _, target_actor_max_z = self._get_z_range(target_actor)
                    reward += 2.0 + np.exp(-5.0 * dist_z)
                else:
                    if dist_z < self.CLOSE:  # reward moving up enough above the object
                        reward += np.exp(-5.0 * max(0, self.CLOSE - dist_z))
                    else:  # reward getting close in XY direction
                        reward += 1.0 + np.exp(-5.0 * max(0.0, dist_xy - self.CLOSE))
        return reward

    def _find_balanced_solutions(
        self, objects: List[sapien.Actor], tolerance=0.001
    ) -> List[Tuple[List[sapien.Actor], List[sapien.Actor]]]:
        """
        Computes all combinations of objects that leads to two groups with the same mass.
        :param objects: ojbects to be placed on the scale.
        :param tolerance: fraction of total mass that the weight between the two groups can differ
        :return: a list of balanced solutions. Each solution is a tuple consisting of a list of objects that should
        be placed on place 1, and a list of objects that should be placed on scale 2.
        """
        masses = [o.mass for o in objects]
        mass_tolerance = tolerance * np.sum(masses)
        solutions = []
        for i in range(1 << len(objects)):
            m0 = m1 = 0.0
            o0 = []
            o1 = []
            for object_idx, obj in enumerate(objects):
                if i & (1 << object_idx):
                    o1.append(obj)
                    m1 += masses[object_idx]
                else:
                    o0.append(obj)
                    m0 += masses[object_idx]
            if abs(m0 - m1) <= mass_tolerance:
                solutions.append((o0, o1))
        return solutions

    def _match_best_solution(
        self,
        possible_solutions,
        stack1: List[sapien.Actor],
        stack2: List[sapien.Actor],
        allow_misplaced=True,
    ) -> Tuple[Tuple[List[sapien.Actor], List[sapien.Actor]], bool]:
        """
        :param possible_solutions: List of tuples of list of actors.
        :param stack1: Actors already placed on plate 1.
        :param stack2: Actors already placed on plate 1.
        :param allow_misplaced: can still match even though some object might have been misplaced
        :return: Tuple[best_matching_solution, completed:bool]. completed is True when the solution
        is complete.
        """
        best_matching_solution = None
        best_num_matches = 0
        for ps_stack1, ps_stack2 in possible_solutions:
            num_matches = 0
            misplaced = False
            for obj in stack1:
                if obj in ps_stack1:
                    num_matches += 1
                else:
                    misplaced = True
                    break
            for obj in stack2:
                if obj in ps_stack2:
                    num_matches += 1
                else:
                    misplaced = True
                    break
            if num_matches > best_num_matches and (allow_misplaced or not misplaced):
                best_matching_solution = (ps_stack1, ps_stack2)
                best_num_matches = num_matches

        if best_matching_solution is None:
            return None, 0

        expected_num_matches = len(best_matching_solution[0]) + len(best_matching_solution[1])
        return best_matching_solution, best_num_matches == expected_num_matches

    def _get_stack(self, bottom_actor, contacts, gravity) -> List[sapien.Actor]:
        """
        Returns objects that are stacked on top of the bottom_actor, from bottom to top.
        This function a simple stacking topology (one object fully on top of another).
        If multiple objects are on top of the same object, the object which generates most
        impulse wins
        :param bottom_actor: The bottom actor in the stack.
        :param contacts: Contact info from previous control step.
        :param gravity: Gravity vector.
        :return: the stack, in the form of a List[Actor], including the bottom_actor.
        """
        stack = [bottom_actor]

        grasped_actors: List[sapien.Actor] = self._env.agent.get_grasped_actors()

        while True:
            top_of_stack_actor = stack[-1]
            on_top_actors = {}
            for contact in contacts:
                if contact.actor0 == top_of_stack_actor:
                    other_actor = contact.actor1
                elif contact.actor1 == top_of_stack_actor:
                    other_actor = contact.actor0
                else:
                    continue
                if (
                    other_actor in grasped_actors
                ):  # grasped actors are not part of the stack (must release first)
                    continue
                if (
                    np.dot((other_actor.pose.p - top_of_stack_actor.pose.p), gravity) < 0
                ):  # other_actor is on top of top_of_stack_actor
                    # sum impulse
                    if not other_actor in on_top_actors:
                        on_top_actors[other_actor] = 0.0
                    on_top_actors[other_actor] += np.dot(
                        np.sum([point.impulse for point in contact.points], axis=0), gravity
                    )
            max_impulse_actor = None
            max_impulse = 0
            for actor, impulse in on_top_actors.items():
                if isinstance(actor, sapien.Link):
                    continue
                if abs(impulse) > max_impulse:
                    max_impulse_actor, max_impulse = actor, impulse

            if max_impulse_actor is None:
                break

            stack.append(max_impulse_actor)

        return stack

    def _where_to_place(
        self, actors: List[sapien.Actor], stack1: List[sapien.Actor], stack2: List[sapien.Actor]
    ) -> List[sapien.Actor]:
        """
        :param actors: The actors currently carried by the agent; must be length 1.
        :param stack1: The stack (including the plate).
        :param stack2: The stack (including the plate).
        :return: What actor (from stack1 and/or stack2) the actors can be placed on to reach
        a balanced solution. The return value can also be an empty list
        """
        # Make sure that the agent is carrying only a single actor
        if actors is None or len(actors) != 1:
            return []
        actor = actors[0]

        # See if placing the actor on stack1 or stack2 still result in a viable solution
        place_on_stack1_solution, _ = self._match_best_solution(
            self._possible_solutions, stack1[1:] + [actor], stack2[1:], allow_misplaced=False
        )
        place_on_stack2_solution, _ = self._match_best_solution(
            self._possible_solutions, stack1[1:], stack2[1:] + [actor], allow_misplaced=False
        )

        result = []
        if not place_on_stack1_solution is None:
            result.append(stack1[-1])  # Then return the top of stack 1
        if not place_on_stack2_solution is None:
            result.append(stack2[-1])  # Then return the top of stack 2

        return result

    def _get_z_range(self, actors: List[sapien.Actor]) -> Tuple[float, float]:
        """
        :param actors: List of actors to query.
        :return: Tuple, min Z and max Z of any part of the actors.
        """
        if not isinstance(actors, list):
            actors = [actors]
        vertices = []
        for actor in actors:
            mesh = get_actor_mesh(actor, to_world_frame=True)
            if mesh is not None:
                vertices.append(mesh.vertices)
        if len(vertices) == 0:
            return 0.0, 0.0
        vertices = np.concatenate(vertices, axis=0)
        return np.min(vertices[:, 2]), np.max(vertices[:, 2])

    def _computed_nearest_actor_2d(self, actors, padding=0.01):
        """
        Computes the minimum XY distance of actors to another other actor in the scene, ignoring
        actors that are "above". This function can be used to determine where it is "save" to
        place an object, such that it does not lie on any other object.

        Note: This is quite a costly function because it retrieves ot vertices of all meshes
        and performes distance computations on them.
        :param actors: List of actors.
        :param padding: additional padding in the size of the actors.
        :return: the distance (in XY) of the actors to the nearest vertex of any other actor.
        """
        scene = self._env._scene

        # get meshes of all other actors
        other_actor_vertices = []
        temp_names = []
        other_actor = None

        for other_actor in scene.get_all_actors():
            if not other_actor in actors and not other_actor.name == "ground":
                mesh = get_actor_mesh(other_actor, to_world_frame=True)
                if not mesh is None:
                    other_actor_vertices.append(mesh.vertices)
                    temp_names.append(other_actor.name)
        temp_names.append(other_actor.name)
        for articulation in scene.get_all_articulations():
            if articulation != self._env.agent.robot:
                for link in articulation.get_links():
                    if not link in actors:
                        mesh = get_actor_mesh(link, to_world_frame=True)
                        if not mesh is None:
                            other_actor_vertices.append(mesh.vertices)
                            temp_names.append(link.name)

        min_distance = 10e10
        for actor in actors:
            mesh = get_actor_mesh(actor, to_world_frame=True)
            vertices = mesh.vertices
            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
            _, max_z = np.min(vertices[:, 2]), np.max(vertices[:, 2])
            position = np.array([min_x + max_x, min_y + max_y]) / 2
            radius = max(max_x - min_x, max_y - min_y) / 2 + padding

            for vertices, _ in zip(other_actor_vertices, temp_names):
                other_actor_min_x, other_actor_max_x = np.min(vertices[:, 0]), np.max(
                    vertices[:, 0]
                )
                other_actor_min_y, other_actor_max_y = np.min(vertices[:, 1]), np.max(
                    vertices[:, 1]
                )
                other_actor_min_z, _ = np.min(vertices[:, 2]), np.max(vertices[:, 2])

                if other_actor_min_z > max_z:  # if other actor is above the actor; ignore
                    continue

                other_actor_position = (
                    np.array(
                        [
                            other_actor_min_x + other_actor_max_x,
                            other_actor_min_y + other_actor_max_y,
                        ]
                    )
                    / 2
                )
                other_actor_radius = (
                    max(
                        other_actor_max_x - other_actor_min_x, other_actor_max_y - other_actor_min_y
                    )
                    / 2
                )

                d = np.linalg.norm(position - other_actor_position) - other_actor_radius - radius
                min_distance = min(min_distance, d)

        return min_distance
