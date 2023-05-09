import numpy as np

from dd_bandits.dd_mabs.adaptation_modules import base_adaptation_module


class ConstantAdaptation(base_adaptation_module.BaseAdaptation):
    def __init__(self, value: float) -> None:
        super().__init__()

        self._value = value

    def update(self, qvals: np.ndarray, delta: float, arm: int, reward: float):
        pass

    def __call__(self, arm: int):
        return self._value
