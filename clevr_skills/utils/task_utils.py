from typing import Dict

import gymnasium as gym

TASK_PROMPT_IDX = {
    "match_pose": [0],
    "move_without_hitting": [0],
    "pick": [3],
    "place": [0],
    "push": [0],
    "rotate": [0],
    "throw": [0],
    "throw_topple": [0],
    "touch": [0],
    "touch_push": [0],
    "touch_topple": [0],
    "trace": [0],
    # L1 tasks
    "follow_order": [0],
    "follow_order_restore": [0],
    "rearrange": [0],
    "rearrange_restore": [0],
    "rotate_restore": [0],
    "simple_manipulation": [0],
    "single_stack": [0],
    "sort2d": [0],
    "swap": [0],
    "neighbour": [0],
    "rotate_symmetry": [0],
    "stack_reversed": [0],
    "novel_adj": [0],
    "novel_noun": [0],
    "novel_adj_noun": [0],
    # L2 tasks
    "balance": [0],
    "sort_stack": [0],
    "stack_topple": [0],
    "swap_push": [0],
    "swap_rotate": [0],
    "throw_sort": [0],
}

TASK_NAME_MAPPING = {
    "match_pose": "MatchPose",
    "move_without_hitting": "MoveWithoutHitting",
    "pick": "Pick",
    "place": "PlaceOnTop",
    "push": "Push",
    "rotate": "Rotate",
    "throw": "Throw",
    "throw_topple": "Throw",
    "touch": "Touch",
    "touch_push": "Touch",
    "touch_topple": "Touch",
    "trace": "Trace",
    # intermediate
    "follow_order": "FollowOrder",
    "follow_order_restore": "FollowOrder",
    "neighbour": "Neighbour",
    "novel_adj": "NovelAdjective",
    "novel_adj_noun": "NovelNounAdjective",
    "novel_noun": "NovelNoun",
    "rearrange": "Rearrange",
    "rearrange_restore": "Rearrange",
    "rotate_restore": "Rotate",
    "rotate_symmetry": "RotateSymmetry",
    "simple_manipulation": "PlaceOnTop",
    "single_stack": "SingleStack",
    "sort2d": "Sort2d",
    "stack_reversed": "SingleStack",
    "swap": "Swap",
    # complex
    "balance": "BalanceScale",
    "sort_stack": "SortStack",
    "stack_topple": "SingleStack",
    "swap_push": "Swap",
    "swap_rotate": "SwapRotate",
    "throw_sort": "ThrowAndSort",
}

L0_tasks = {
    "pick": {},
    "place": {"spawn_at_gripper": True},
    "match_pose": {"ngoals": 1},
    "move_without_hitting": {},
    "push": {},
    "rotate": {"num_actors": 3},
    "throw": {"spawn_at_gripper": True, "target_2d": False},
    "throw_topple": {"spawn_at_gripper": True, "target_2d": False, "topple_target": True},
    "touch": {},
    "touch_push": {"push": True},
    "touch_topple": {"topple": True},
    "trace": {"ngoals": 3},
}

L1_tasks = {
    "follow_order": {"num_areas": 4, "num_predicates": 4},
    "follow_order_restore": {"num_areas": 2, "num_predicates": 2, "replace": True},
    "rearrange": {"num_actors": 3},
    "rearrange_restore": {"num_actors": 3, "replace": True},
    "rotate_restore": {"num_actors": 3, "replace": True},
    "simple_manipulation": {"num_actors": 3, "num_areas": 3},
    "single_stack": {"num_actors": 4},
    "sort2d": {"num_actors": 4, "num_areas": 3},
    "swap": {},
    "neighbour": {"num_areas": 3},
    "rotate_symmetry": {
        "num_actors": 4,
    },
    "stack_reversed": {"num_actors": 4, "reverse": True},
    "novel_adj": {},
    "novel_noun": {"num_actors": 3, "num_areas": 3},
    "novel_adj_noun": {},
}

L2_tasks = {
    "balance": {"num_actors": 4},
    "sort_stack": {"num_actors": 6, "num_areas": 3},
    "stack_topple": {"num_actors": 4, "topple": True},
    "swap_push": {"push": True},
    "swap_rotate": {},
    "throw_sort": {"num_actors": 4, "num_areas": 3},
}


def get_clevrskills_env(tasks_args: Dict, split="train") -> Dict:
    """
    Utility function to get one instance of every env.
    :param tasks_args: Tasks. Use L0_tasks, L1_tasks or L2_tasks.
    :param split: "train" or "test".
    :return: Dict of environments.
    """
    envs = {}
    for task in tasks_args:

        task_name = TASK_NAME_MAPPING.get(task, task)

        task_params = tasks_args[task]
        task_params.update({"split": split})

        env = gym.make(
            "ClevrSkills-v0",
            obs_mode="rgbd",
            reward_mode="dense",
            control_mode="pd_ee_delta_pose",
            robot="xarm6_vacuum",
            task=task_name,
            strip_eval=False,
            shader_dir="ibl",
            render_config=dict(),
            enable_shadow=True,
            task_args=task_params,
            render_mode="rgb_array",
        )
        envs[task] = env

    return envs
