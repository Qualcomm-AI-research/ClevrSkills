# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import mplib

__all__ = ["clevr_skills_env", "predicates", "tasks", "utils"]

if not mplib.__version__.startswith("0.2"):
    warnings.warn(
        f"mplib version is {mplib.__version__}; version 0.2.1 is recommended",
        RuntimeWarning,
        stacklevel=2,
    )
