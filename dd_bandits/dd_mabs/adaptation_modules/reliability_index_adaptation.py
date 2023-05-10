from typing import Optional

import numpy as np

from dd_bandits.dd_mabs.adaptation_modules import base_adaptation_module


class ReliabilityIndexAdaptation(base_adaptation_module.BaseAdaptation):
    def __init__(
        self, learning_rate: float, num_arms: int, multiple: Optional[float] = 1
    ) -> None:

        self._lr = learning_rate
        self._num_arms = num_arms
        self._multiple = multiple

        self._expected_delta_2 = 10 * np.random.random() ** 2
        self._expected_delta_2_per_arm = [
            10 * np.random.random() ** 2 for _ in range(self._num_arms)
        ]

        super().__init__()

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        self._expected_delta_2 += self._lr * (
            np.mean(delta**2) - self._expected_delta_2
        )
        self._expected_delta_2_per_arm[arm] += self._lr * (
            np.mean(delta**2) - self._expected_delta_2_per_arm[arm]
        )

    def __call__(self, arm: int):
        # return self._multiple * np.max(self._expected_delta_2_per_arm)
        return self._multiple * np.sqrt(self._expected_delta_2)
