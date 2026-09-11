# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Solver classes for ClevrSkills, imported lazily."""

from importlib import import_module

_EXPORTS = {
    "BalanceScaleSolver": "balance_scale_solver",
    "OrderedSolver": "ordered_solver",
    "PickMove3dPlaceSolver": "pick_move3d_place_solver",
    "PickSolver": "pick_solver",
    "Solver": "solver",
    "Move3dTouchSolver": "touch_solver",
    "RotateSolver": "rotate_solver",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(f"{__name__}.{_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
