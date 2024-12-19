# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.pick_predicate import PickPredicate
from clevr_skills.utils.actor_placement import get_random_non_overlapping_poses_2d_v2
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("Pick")
class Pick(Task):
    """
    This task requires the agent to pick up a specified object.
    """

    def __init__(
        self,
        env,
        num_actors=2,
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
        super().__init__(env, record_dir=record_dir, split=split, variant=variant)
        self._actor_size = [0.1, 0.1, 0.05]
        self._workspace = [
            [
                -0.35,
                -0.3,
            ],
            [0.1, 0.3],
        ]
        assert num_actors >= 1

        self._num_actors = self._sample_int(min=1, max=num_actors, sample=sample_num_actors)
        self._actors = []
        self.actor_textures = None
        self._pick_actor = None
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

        actor_poses = get_random_non_overlapping_poses_2d_v2(
            self._env, self._actors, self._workspace, z=0.01
        )

        # place actors at generated poses
        for idx, actor in enumerate(self._actors):
            actor.set_pose(actor_poses[idx])

        self._actors, self.actor_textures = self._get_variant(
            self._actors, self.actor_textures, first_actor_only=True
        )

        self._pick_actor = self._actors[0]

        self.predicate = PickPredicate(self._env, self._pick_actor, clear_distance=0.02)

        self.prompts = [
            f"Pick up the {self._pick_actor.name}",
            f"Grab the {self._pick_actor.name}",
            f"Lift the {self._pick_actor.name}",
            f"Pick up the {get_actor_description(self._pick_actor)}",
            f"Grab the {get_actor_description(self._pick_actor)}",
            f"Lift the {get_actor_description(self._pick_actor)}",
            f"Pick up the object with " f"{get_texture_description(self._pick_actor)} texture",
            f"Grab the object with {get_texture_description(self._pick_actor)} texture",
            f"Lift the object with {get_texture_description(self._pick_actor)} texture",
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
