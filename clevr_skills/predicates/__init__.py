# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Predicate modules for ClevrSkills, imported lazily."""

from importlib import import_module

__all__ = [
    "at_pose",
    "at_position",
    "balance_scale",
    "direction_pushed",
    "ee_at_pos",
    "inside_2d",
    "on_top",
    "once",
    "order",
    "pick_predicate",
    "predicate",
    "sequence",
    "set",
    "stack_predicate",
    "touch",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
