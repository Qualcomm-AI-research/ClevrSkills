# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math

from clevr_skills.utils.permute import get_permutation_first_indices, get_permutation_indices


def test_get_permutation_indices():
    assert get_permutation_indices(0, 0) == ([], 0)
    assert get_permutation_indices(0, 1) == ([], 1)
    assert get_permutation_indices(0, 2) == ([], 2)

    assert get_permutation_indices(2, 0) == ([0, 1], 0)
    assert get_permutation_indices(2, 1) == ([1, 0], 0)
    assert get_permutation_indices(2, 2) == ([0, 1], 1)

    assert get_permutation_indices(3, 0) == ([0, 1, 2], 0)
    assert get_permutation_indices(3, 1) == ([1, 0, 2], 0)
    assert get_permutation_indices(3, 2) == ([2, 1, 0], 0)
    assert get_permutation_indices(3, 3) == ([0, 2, 1], 0)
    assert get_permutation_indices(3, 4) == ([1, 2, 0], 0)
    assert get_permutation_indices(3, 5) == ([2, 0, 1], 0)
    assert get_permutation_indices(3, 6) == ([0, 1, 2], 1)

    for n in range(10):
        assert get_permutation_indices(n, 0)[0] == get_permutation_indices(n, math.factorial(n))[0]
        assert get_permutation_indices(n, math.factorial(n))[1] == 1


def test_get_permutation_first_indices():
    assert get_permutation_first_indices(0, 0) == ([], 0)
    assert get_permutation_first_indices(0, 1) == ([], 1)
    assert get_permutation_first_indices(0, 2) == ([], 2)

    assert get_permutation_first_indices(2, 0) == ([0, 1], 0)
    assert get_permutation_first_indices(2, 1) == ([1, 0], 0)
    assert get_permutation_first_indices(2, 2) == ([0, 1], 1)

    assert get_permutation_first_indices(3, 0) == ([0, 1, 2], 0)
    assert get_permutation_first_indices(3, 1) == ([1, 0, 2], 0)
    assert get_permutation_first_indices(3, 2) == ([2, 1, 0], 0)
    assert get_permutation_first_indices(3, 3) == ([0, 1, 2], 1)

    for n in range(1, 10):
        assert get_permutation_first_indices(n, 0)[0] == get_permutation_first_indices(n, n)[0]
        assert get_permutation_first_indices(n, n)[1] == 1
