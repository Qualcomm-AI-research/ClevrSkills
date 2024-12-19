# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien
from sapien.core import Pose

from clevr_skills.predicates.direction_pushed import DirectionPushed
from clevr_skills.predicates.order import Order
from clevr_skills.utils.action_trace import at_get_actor
from clevr_skills.utils.actor_placement import get_random_non_overlapping_poses_2d_with_actor_sizes
from clevr_skills.utils.models import get_actor_size
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("Push")
class Push(Task):
    """
    This task creates two or more actors.
    A specified actor must be pushed towards another actor without being picked up.
    """

    def __init__(
        self,
        env,
        num_actors: int = 2,
        push: bool = False,
        sample_num_actors: bool = True,
        variant: int = 0,
        record_dir=None,
        split="train",
    ):
        """
        :param env: The ClevrSkillsEnv
        :param num_actors: The number of actors (objects) in the scene.
        :param push: Ignored.
        :param sample_num_actors: When true, the number of actors will be exactly num_actors.
        When false, the number of actor will be uniformly sampled between 2 and num_actors.
        :param spawn_at_gripper: When True, the main actor will be created directly underneath the
        gripper with a small upwards velocity. If the agent turns on the gripper, the actor
        will stick to it.
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
            [0.0, 0.3],
        ]
        assert num_actors >= 2

        self._num_actors = self._sample_int(min=2, max=num_actors, sample=sample_num_actors)
        self._actors = []
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

        self._actors, self.actor_textures = self._get_variant(self._actors, self.actor_textures)

        actor_sizes = [get_actor_size(ac) * 2 for ac in self._actors]

        # generate poses
        self._target_poses = get_random_non_overlapping_poses_2d_with_actor_sizes(
            self._env,
            self._num_actors,
            self._workspace,
            actor_sizes + [max(actor_sizes, key=lambda x: x.prod())],
        )

        # buffer pose
        alpha = 0.5
        buff_pos = (1 - alpha) * self._target_poses[0].p + alpha * self._target_poses[1].p
        buff_pose = Pose(buff_pos, self._target_poses[0].q)

        # place actors on generated poses
        for idx, actor in enumerate(self._actors):
            actor.set_pose(self._target_poses[idx])

        # choose actors to swap. actors are generated randomly so just take the first 2
        actor_a, actor_b = self._actors[:2]

        # Inject info about "why" here
        predicate = DirectionPushed(self._env, actor_a, buff_pose)

        predicate.action_state_extra = {
            "target_pose_function": "towards",
            "target_actor": at_get_actor(actor_b),
        }

        self.predicate = Order("push", [predicate])

        self.prompts = [
            f"Push {get_actor_description(actor_a)} towards {get_actor_description(actor_b)}",
            f"Push object with {get_texture_description(actor_a)} texture towards the object "
            f"with {get_texture_description(actor_b)} texture",
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
