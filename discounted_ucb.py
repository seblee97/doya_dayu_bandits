import numpy as np

from random_bandit import RandomMAB


class DiscountedUCB(RandomMAB):
    def __init__(self, n_arms, rho, gamma, rng=None):
        self._n_arms = n_arms
        self.rho = rho
        self.gamma = gamma

        self.arm_seen = np.zeros(n_arms, dtype=bool)
        self.rewards = np.zeros(n_arms)

        # Total and per-arm step count
        self.step = 0
        self.step_arm = np.zeros(self._n_arms)

    def predict_bandits(self):
        return (self.rewards / self.step_arm), np.log(self.step) / (self.step_arm + 1.0)

    def policy(self):
        return np.eye(self._n_arms)[self.play()]

    def play(self):
        if not self.arm_seen.all():
            return self.arm_seen.argmin()

        ucb_values = np.zeros(self._n_arms)
        for arm in range(self._n_arms):
            if self.step_arm[arm] > 0:
                ucb_values[arm] = np.sqrt(
                    self.rho * np.log(self.step) / self.step_arm[arm]
                )

        action = ((self.rewards / self.step_arm) + ucb_values).argmax()
        return action

    def update(self, arm, reward):
        self.arm_seen[arm] = True
        self.step *= self.gamma
        self.step_arm *= self.gamma
        self.step += 1
        self.step_arm[arm] += 1
        self.rewards[arm] += reward
