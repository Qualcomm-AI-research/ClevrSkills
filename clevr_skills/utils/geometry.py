# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np


def unit(vec: np.ndarray, ordinal=None) -> np.ndarray:
    """
    :param vec: The vector to be normalized.
    :param ordinal: The ordinal of the norm.
    :return: vec, with unit norm.
    """
    return vec / max(1e-7, np.linalg.norm(vec, ord=ordinal))
