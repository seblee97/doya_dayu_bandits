import numpy as np

from mabs import mab


class EpsilonGreedy(mab.MAB):
    def __init__(self, n_arms, epsilon, rng=None):

        self._epsilon = epsilon

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

        if self._rng.random() <= self._epsilon:
            return self._rng.choice(self._n_arms)

        return self._rewards.argmax()

    def _update(self, arm, reward):
        pass

    def temperature(self):
        return None
