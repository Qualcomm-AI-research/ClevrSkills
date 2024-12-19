# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from clevr_skills.utils.load_traj import Traj


class ActionTraceConverter(ABC):
    """
    Abstract base class that is able to convert the action trace to another format.
    Actual implementations included are natural language and Python Code ("llava").
    """

    def __init__(self, traj_path: Optional[str] = None):
        """
        :param traj_path: Trajectory will be read from this path
        """
        self.traj_path = traj_path
        if traj_path is None:
            self.traj = None
            self.success = None
            self.action_trace = None
        else:
            self.traj = Traj(traj_path)
            self.success = self.traj.get_success()
            self.action_trace = self.traj.get_action_trace()

    @abstractmethod
    def extract_labels(
        self, prompt: str, action_trace: Optional[List] = None, success: Optional[np.ndarray] = None
    ) -> List[List[Tuple[int, int, str]]]:
        """
        :param prompt: The prompt relative to which the labels should be extracted.
        :param action_trace: Action trace for the entire sequence.
        :param success: Success (boolean) for the entire sequence.
        :return: Returns segments of labels, based on the action trace.
        List of List of Tuple[start_idx, end_idx, text].
        The first index is depth of the action trace, the second index is segments.
        """
        raise NotImplementedError

    def expand_labels(
        self, labels: List[List[Tuple[int, int, str]]], traj_length: int, no_action=""
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Utility function that expands the labels datastructure (as returned by extract_labels())
        into a simple List of Lists of strs that can be indexed as labels[step_idx][depth_idx]
        :param labels: List of List of [start_idx, end_idx, label].  Indexed as:
        [depth_idx][action_idx][start, end, label]
        :param traj_length: Length of trajectory. Required to pad the result to full
        length of the trajectory.
        :param no_action:
        :return:
        """
        expanded = []  # expand the labels into [depth_idx][step_idx]
        for level in labels:
            level_result = []
            for start_idx, end_idx, label in level:
                level_result += [no_action] * (start_idx - len(level_result))
                level_result += [label] * (end_idx - start_idx)
            level_result += [no_action] * (traj_length - len(level_result))
            expanded.append(level_result)

        # Return the transpose of expanded
        return [
            [expanded[depth][step_idx] for depth in range(len(expanded))]
            for step_idx in range(traj_length)
        ]
