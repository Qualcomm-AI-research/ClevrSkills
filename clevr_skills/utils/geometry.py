# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np


def unit(vec: np.ndarray, ordinal=None) -> np.ndarray:
    """
    :param vec: The vector to be normalized.
    :param ordinal: The ordinal of the norm.
    :return: vec, with unit norm.
    """
    return vec / max(1e-7, np.linalg.norm(vec, ord=ordinal))


def euclidean_distance_matrix(arr1, arr2):
    """
    :param arr1: shape (N, K)
    :param arr2: shape (M, K)
    :return: Euclidean distance matrix, distance between each point. Shape (N * M).
    """
    s1 = np.sum(arr1**2, axis=1)[:, np.newaxis]
    s2 = np.sum(arr2**2, axis=1)
    s3 = -2.0 * np.dot(arr1, arr2.T)
    return np.sqrt(np.maximum(s1 + s2 + s3, 0.0))
