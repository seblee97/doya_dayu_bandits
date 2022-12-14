import numpy as np

from mabs import mab


class DiscountedUCB(mab.MAB):
    def __init__(self, n_arms, rho, gamma, rng=None):

        self._gamma = gamma
        self._rho = rho

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

        ucb_values = np.zeros(self._n_arms)
        for arm in range(self._n_arms):
            if self._step_arm[arm] > 0:
                ucb_values[arm] = np.sqrt(
                    self._rho * np.log(self._step) / self._step_arm[arm]
                )

        action = ((self._rewards / self._step_arm) + ucb_values).argmax()
        return action

    def _update(self, arm, reward):
        self._step *= self._gamma
        self._step_arm[arm] *= self._gamma

    def temperature(self):
        return None
