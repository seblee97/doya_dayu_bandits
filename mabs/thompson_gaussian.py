import jax
import numpy as np

from mabs import mab


class ThompsonSamplingGaussian(mab.MAB):
    def __init__(self, n_arms, rng=None):

        super().__init__(n_arms=n_arms, rng=rng)

    def predict_bandits(self):
        return (self._rewards / self._step_arm), np.log(self._step) / (
            self._step_arm + 1.0
        )

    def policy(self):
        return np.eye(self._n_arms)[self.play()]

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        sampled = self._rng.normal(
            jax.device_get(self._rewards), np.sqrt(1.0 / (self._step_arm + 1.0))
        )
        return sampled.argmax()

    def _update(self, arm, reward):
        pass

    def temperature(self):
        return None
