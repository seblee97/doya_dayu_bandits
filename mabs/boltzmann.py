import jax
import numpy as np
import rlax

from mabs import mab


class Boltzmann(mab.MAB):
    def __init__(self, n_arms, temperature, rng=None):

        self._temperature = temperature

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

        return jax.device_get(
            rlax.softmax(self._temperature).sample(
                self._rng_key, self._rewards - self._rewards.max()
            )
        ).ravel()[0]

    def _update(self, arm, reward):
        pass

    def temperature(self):
        return self._temperature
