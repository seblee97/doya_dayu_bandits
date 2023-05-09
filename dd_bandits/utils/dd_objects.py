import collections

import numpy as np


class LikelihoodMemory:

    BASELINE = 0.0001

    def __init__(self, memory_size: int, num_arms: int) -> None:
        self._memory_size = memory_size

        self._memory = collections.deque(
            self._memory_size * [self.BASELINE], self._memory_size
        )
        self._per_arm_memory = {
            arm: collections.deque(
                int(2 * self._memory_size) * [self.BASELINE],
                int(2 * self._memory_size),
            )
            for arm in range(num_arms)
        }
