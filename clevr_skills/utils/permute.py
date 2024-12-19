# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List, Tuple

# Two simple utility functions for permuting a full list, or permuting only the first entry in a list
# These functions are used for generating variants of tasks
# I.e., just permuting the _role_ of actors in a task.


def get_permutation_indices(n: int, p: int) -> Tuple[List[int], int]:
    """
    :param n: Length of list that must be permuted.
    :param p: index of the permutation variant.
    It is guaranteed that variant "0" always does not any permutation.
    The permutation will repeat itself at p = factorial(n), p = 2 * factorial(n) and so on.
    :return: A tuple. The first entry is the indices of the permuted list.
    The second entry is the remainder of p. This can be used as input to another call to get_permutation_indices
    """
    indices = list(range(n))
    for i in range(n - 1):
        remaining_n = n - i
        j = i + (p % remaining_n)
        p = p // remaining_n
        indices[i], indices[j] = indices[j], indices[i]
    return indices, p


def get_permutation_first_indices(n: int, p: int) -> Tuple[List[int], int]:
    """
    :param n: Length of list that must be permuted.
    :param p: index of the permutation variant.
    It is guaranteed that variant "0" always does not any permutation.
    The permutation will repeat itself at p = n, p = 2 * n and so on
    :return: A tuple. The first entry is the indices of the list where the onyl the _first_ entry is permuted.
    The second entry is the remainder of p. This can be used as input to another call to get_permutation_indices
    """
    indices = list(range(n))
    if n > 0:
        j = p % n
        p = p // n
        indices[0], indices[j] = indices[j], indices[0]
    return indices, p
