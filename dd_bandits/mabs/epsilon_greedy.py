import numpy as np

from dd_bandits.mabs import mab


class EpsilonGreedy(mab.MAB):
    def __init__(self, num_arms, epsilon, rng=None):

        self._epsilon = epsilon

        super().__init__(num_arms=num_arms, rng=rng)

    def predict_bandits(self):
        return (self._rewards / self._step_arm), np.log(self._step) / (
            self._step_arm + 1.0
        )

    def policy(self):
        return np.eye(self._num_arms)[self.play()]

    def play(self):
        if not self._arm_seen.all():
            return self._arm_seen.argmin()

        if self._rng.random() <= self._epsilon:
            return self._rng.choice(self._num_arms)

        return self._rewards.argmax()

    def _update(self, arm, reward):
        pass

    def temperature(self):
        return None
