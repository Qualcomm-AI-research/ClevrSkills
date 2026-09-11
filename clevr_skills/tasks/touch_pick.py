# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List

import sapien.core as sapien

from clevr_skills.predicates.ee_at_pose import EEAtPose
from clevr_skills.predicates.order import Order
from clevr_skills.predicates.pick_predicate import PickPredicate
from clevr_skills.predicates.touch import TouchPredicate
from clevr_skills.utils.action_trace import at_get_pose
from clevr_skills.utils.actor_placement import get_random_non_overlapping_poses_2d_v2
from clevr_skills.utils.render import get_render_material
from clevr_skills.utils.visualize_prompt import get_actor_description, get_texture_description

from .task import Task, register_task


@register_task("TouchPick")
class TouchPick(Task):
    """
    This task requires the agent to pick up a specific object.
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

        self._num_actors = self._sample_int(
            min=num_actors, max=num_actors, sample=sample_num_actors
        )
        self._actors = []
        self.actor_textures = None
        self._pick_actor = None
        self.predicate = None
        self.prompts = None

    def setup_task(self):
        """
        This function should load (create) the actors.
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
        self._touch_actor = self._actors[1]
        self._reset_pose = self._env.agent.controller.controllers["arm"].ee_pose

        self.predicate = Order(
            "touch_pick",
            [
                TouchPredicate(self._env, self._touch_actor, push=False, topple=False),
                EEAtPose(self._env, self._reset_pose, threshold=0.05),
                PickPredicate(self._env, self._pick_actor, clear_distance=0.03),
            ],
        )

        self.prompts = [
            f"Touch the {get_actor_description(self._touch_actor)}, reset and then Pick up the {get_actor_description(self._pick_actor)}",
        ]

        self.init_state = {
            "objects": {
                get_actor_description(act): str(at_get_pose(act.pose)) for act in self._actors
            },
            "pick_obj": {
                get_actor_description(self._pick_actor): str(at_get_pose(self._pick_actor.pose))
            },
            "touch_obj": {
                get_actor_description(self._touch_actor): str(at_get_pose(self._touch_actor.pose))
            },
            "reset_pose": str(at_get_pose(self._reset_pose)),
        }

    def get_task_actors(self) -> List[sapien.Actor]:
        return self._actors

    def get_task_textures(self) -> List[tuple]:
        return self.actor_textures
