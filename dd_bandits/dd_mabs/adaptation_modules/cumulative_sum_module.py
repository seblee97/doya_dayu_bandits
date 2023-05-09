from typing import Optional

import numpy as np

from dd_bandits.dd_mabs.adaptation_modules import base_adaptation_module


class CumulativeSumPlusAdaptation(base_adaptation_module.BaseAdaptation):
    def __init__(self, weight: float, multiple: Optional[float] = 1) -> None:

        self._weight = weight
        self._multiple = multiple
        self._max_sum = 0

        super().__init__()

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        mean = qvals[..., 0][arm].mean().item()
        var = qvals[..., 1][arm].mean().item()
        centred_reward = (reward - mean) / np.sqrt(var)
        self._max_sum = max(0, self._max_sum + centred_reward - self._weight)

    def __call__(self, arm: int):
        return self._multiple * self._max_sum


class CumulativeSumMinusAdaptation(base_adaptation_module.BaseAdaptation):
    def __init__(self, weight: float, multiple: Optional[float] = 1) -> None:

        self._weight = weight
        self._multiple = multiple
        self._min_sum = 0

        super().__init__()

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        mean = qvals[..., 0][arm].mean().item()
        var = qvals[..., 1][arm].mean().item()
        centred_reward = (reward - mean) / np.sqrt(var)
        self._min_sum = max(0, self._min_sum - centred_reward - self._weight)

    def __call__(self, arm: int):
        return self._multiple * self._min_sum
