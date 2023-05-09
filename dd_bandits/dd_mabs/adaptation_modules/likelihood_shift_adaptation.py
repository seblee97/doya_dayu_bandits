import collections

import numpy as np

from dd_bandits.dd_mabs.adaptation_modules import base_adaptation_module
from dd_bandits.utils import custom_functions


class LikelihoodShiftAdaptation(base_adaptation_module.BaseAdaptation):

    BASELINE = 0.001

    def __init__(self, memory: int, num_arms, use_direct: bool) -> None:
        self._memory = memory
        self._num_arms = num_arms
        self._use_direct = use_direct

        self._memory = memory
        self._likelihood_memory = collections.deque(
            int(2 * self._memory) * [self.BASELINE], int(2 * self._memory)
        )
        self._per_arm_likelihood_memory = {
            arm: collections.deque(
                int(2 * self._memory) * [self.BASELINE],
                int(2 * self._memory),
            )
            for arm in range(self._num_arms)
        }

    def _likelihood_shift(self):
        def _shift(likelihood_deque):
            list_l = list(likelihood_deque)

            past = np.mean(list_l[self._memory : 2 * self._memory])
            pres = np.mean(list_l[: self._memory])
            return past / pres + pres / past - 2

        likelihood_shifts = [
            _shift(self._per_arm_likelihood_memory[arm])
            for arm in range(self._num_arms)
        ]

        likelihood_shift = _shift(self._likelihood_memory)

        return likelihood_shift, likelihood_shifts

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        if self._use_direct:
            likelihood = np.mean(
                [
                    custom_functions.gaussian_likelihood(
                        mean, np.sqrt(np.exp(logvar)), reward
                    )
                    for (mean, logvar) in qvals[arm]
                ]
            )
        else:
            likelihood = np.mean(
                [
                    custom_functions.gaussian_likelihood(
                        mean,
                        np.sqrt(np.clip(sqr_rew - mean**2, self.BASELINE, 100.0)),
                        reward,
                    )
                    for (mean, sqr_rew) in qvals[arm]
                ]
            )
        self._likelihood_memory.appendleft(likelihood)
        self._per_arm_likelihood_memory[arm.item()].appendleft(likelihood)

    def __call__(self, arm: int):
        # return np.max(self._likelihood_shift()[1])
        return self._likelihood_shift()[0]
