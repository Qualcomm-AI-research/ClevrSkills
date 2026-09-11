# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Task modules for ClevrSkills.
"""

from importlib import import_module

__all__ = [
    "balance_scale",
    "follow_order",
    "match_pose",
    "move_2d",
    "move_without_hitting",
    "neighbour",
    "novel_adj",
    "novel_noun",
    "novel_noun_and_adj",
    "pick",
    "place_next_to",
    "place_on_top",
    "push",
    "rearrange",
    "rotate",
    "rotate_symmetry",
    "single_stack",
    "sort_2d",
    "sort_stack",
    "swap",
    "swap_rotate",
    "task",
    "throw",
    "throw_and_sort",
    "touch",
    "touch_pick",
    "trace",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
