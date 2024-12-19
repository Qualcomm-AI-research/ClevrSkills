# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.at_position import AtPosition
from clevr_skills.predicates.at_position_pushed import AtPositionPushed
from clevr_skills.predicates.order import Order
from clevr_skills.predicates.set import Set
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_placement import get_random_non_overlapping_poses_2d_with_actor_sizes
from clevr_skills.utils.models import get_actor_size
from clevr_skills.utils.render import get_render_material

from .task import Task, register_task


@register_task("Rearrange")
class Rearrange(Task):
    """
    This task creates two or more actors. The goal of the task is to rearrange
    the actors to match the positions as specified in the goal image.
    """

    unsupported_task_args = ["variant"]

    def __init__(
        self,
        env,
        num_actors: int = 2,
        restore: bool = False,
        push: bool = False,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param restore: Should actors be restore to their original pose after being moved to their target pose?
        :param push: Should pushing be used to manipulate the actors? If not, pick and place is used.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param variant: Set this to a non-zero value to generate variants of the same seed.
        A "variant" of a scene looks identical to the original, but the roles of actors are different.
        This is to avoid an agent learning behavior solely based on the looks of the scene, ignoring the prompt.
        :param record_dir: Path where episode is recorded.
        :param split: What object split to use. E.g., "train" or "test".
        """
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._push = push
        self._actor_size = [0.1, 0.1, 0.05]
        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.1, 0.3],
        ]
        assert num_actors >= 2

        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self._actors = []
        self.restore = restore
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
            self._env, 2 * self._num_actors, self._workspace, actor_sizes + actor_sizes
        )

        # place actors on generated poses
        for idx, actor in enumerate(self._actors):
            actor.set_pose(self._target_poses[idx])

        PredicateClass = AtPositionPushed if self._push else AtPosition

        keystep_image_name = f"{{ks:keystep_{self._num_actors}}}"

        order_predicates = []
        set_predicates = []
        for i in range(self._num_actors):
            p = PredicateClass(self._env, self._actors[i], self._target_poses[i + self._num_actors])
            p.action_state_extra = {
                "target_pose_function": "rearrange",
                "target_pose_image": keystep_image_name,
                "store_pose": [at_get_actor(self._actors[i])],
            }
            set_predicates.append(p)
        order_predicates.append(Set("Put objects at new position", set_predicates))

        if self.restore:
            set_predicates = []
            for i in range(self._num_actors):
                p = PredicateClass(self._env, self._actors[i], self._target_poses[i])
                p.action_state_extra = {"target_pose_function": "rearrange_restore"}
                set_predicates.append(p)
            order_predicates.append(Set("Put objects back to original position", set_predicates))

        self.predicate = Order("rearrange", order_predicates)

        self.prompts = [
            (
                f"Rearrange to {keystep_image_name} and then restore"
                if self.restore
                else f"Rearrange to {keystep_image_name}"
            )
        ]

    def get_task_actors(self) -> List[sapien.Actor]:
        """
        :return: list of actors used in the task.
        """
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        """
        :return: list of textures used in the task.
        """
        return self.actor_textures
