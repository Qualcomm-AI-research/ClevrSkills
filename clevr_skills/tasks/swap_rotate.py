# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.at_pose import AtPoseRotated
from clevr_skills.predicates.at_position import AtPosition
from clevr_skills.predicates.swap_predicate import SwapPredicate
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_placement import get_random_non_overlapping_poses_2d_with_actor_sizes
from clevr_skills.utils.models import get_actor_size
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description

from .swap import Swap
from .task import register_task


@register_task("SwapRotate")
class SwapRotate(Swap):
    """
    The tasks create two or more actors.
    The goal is to swap the positions of two specified actors while also
    rotating both by specified angle in a specified direction.
    """

    def __init__(
        self,
        env,
        num_actors: int = 2,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(
            env,
            num_actors=num_actors,
            sample_num_actors=sample_num_actors,
            record_dir=record_dir,
            split=split,
            variant=variant,
        )
        self.clockwise = None
        self.rot_degrees = None
        self.actor_textures = None
        self._target_poses = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors, setup predicates and prompts.
        It is guaranteed to be called before get_textual_description() and get_predicates()
        such that it can be used to initialize the entire Task
        """

        self.clockwise = self._episode_rng.choice([0, 1])
        self.rot_degrees = self._episode_rng.choice([30, 60, 90, 120, 150])

        # generate actors
        self.actor_textures = self.model_factory.get_random_top_textures(self._num_actors)

        render_materials = [
            (at[0], get_render_material(self._renderer, at[1])) for at in self.actor_textures
        ]
        self._actors = [
            self.model_factory.get_random_top_object(
                self._scene,
                self._renderer,
                render_materials[idx][1],
                tex_name=render_materials[idx][0],
            )
            for idx in range(self._num_actors)
        ]

        actor_sizes = [get_actor_size(ac) for ac in self._actors]

        # generate poses
        self._target_poses = get_random_non_overlapping_poses_2d_with_actor_sizes(
            self._env,
            self._num_actors + 1,
            self._workspace,
            actor_sizes + [max(actor_sizes, key=lambda x: x.prod())],
        )

        # buffer pose
        buff_pose = self._target_poses[-1]

        self._target_poses = self._target_poses[:-1]

        # place actors on generated poses
        for idx, actor in enumerate(self._actors):
            actor.set_pose(self._target_poses[idx])

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        # choose actors to swap. actors are generated randomly so just take the first 2
        actor_a, actor_b = self._actors[:2]

        p1 = AtPosition(self._env, actor_a, buff_pose)
        temp_pose = actor_b.pose
        p2 = AtPoseRotated(self._env, actor_b, actor_a.pose, self.rot_degrees, self.clockwise)
        p3 = AtPoseRotated(self._env, actor_a, temp_pose, self.rot_degrees, self.clockwise)

        self.predicate = SwapPredicate("Swap and rotate the actors", [p1, p2, p3])

        # inject extra info for action trace:
        angle = self.rot_degrees * [1, -1][self.clockwise]
        p1.action_state_extra = {
            "target_pose_function": "free_space",
            "store_pose": [at_get_actor(actor_a), at_get_actor(actor_b)],
        }
        p2.action_state_extra = {
            "target_pose_function": "swap_rotate",
            "target_actor": at_get_actor(actor_a),
            "angle": angle,
        }
        p3.action_state_extra = {
            "target_pose_function": "swap_rotate",
            "target_actor": at_get_actor(actor_b),
            "angle": angle,
        }

        self.prompts = [
            f"Swap positions of {get_actor_description(actor_a)} and "
            f"{get_actor_description(actor_b)} while rotating them by "
            f"{abs(self.rot_degrees)} degrees "
            f"{['anti-clockwise', 'clockwise'][self.clockwise]}",
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task
        """
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task
        """
        return self.actor_textures
