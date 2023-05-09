from typing import Optional

import numpy as np

from dd_bandits.dd_mabs.adaptation_modules import base_adaptation_module


class DEUP(base_adaptation_module.BaseAdaptation):
    def __init__(self, multiple: Optional[float] = 1) -> None:

        self._multiple = multiple

        super().__init__()

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        # import pdb

        # pdb.set_trace()
        aleatoric_uncertainty_estimate = qvals[..., 1][arm]
        self._epistemic_uncertainty = np.mean(
            delta**2 - aleatoric_uncertainty_estimate
        ).item()

    def __call__(self, arm: int):
        return self._multiple * self._epistemic_uncertainty
